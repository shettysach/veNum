use crate::core::errors::*;
use anyhow::Result;
use std::{
    cmp::{max, Ordering},
    collections::HashSet,
    iter::repeat,
    ops::Mul,
};

#[derive(Clone)]
pub(crate) struct Shape {
    pub sizes: Vec<usize>,
    pub strides: Vec<Stride>,
    pub offset: usize,
}

#[derive(Copy, Clone)]
pub enum Stride {
    Positive(usize),
    Negative(usize),
}

impl Shape {
    pub fn new(sizes: &[usize]) -> Shape {
        let mut current = 1;
        let mut strides: Vec<Stride> = sizes
            .iter()
            .rev()
            .map(|size| {
                let stride_val = current;
                current *= size;
                Stride::new(stride_val, true)
            })
            .collect::<Vec<Stride>>();
        strides.reverse();

        Shape {
            sizes: sizes.to_vec(),
            strides,
            offset: 0,
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.sizes.len()
    }

    pub(crate) fn numel(&self) -> usize {
        self.sizes.iter().product()
    }

    // --- Shape operations ---

    pub(crate) fn view(&self, sizes: &[usize]) -> Result<Shape> {
        self.valid_contiguity()?;
        self.valid_reshape(sizes)?;

        let mut current = 1;
        let positive = match self.strides.first().ok_or(EmptyTensorError::View)? {
            Stride::Positive(_) => true,
            Stride::Negative(_) => false,
        };

        let mut strides = sizes
            .iter()
            .rev()
            .map(|size| {
                let stride_val = current;
                current *= size;
                Stride::new(stride_val, positive)
            })
            .collect::<Vec<Stride>>();
        strides.reverse();

        Ok(Shape {
            sizes: sizes.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn permute(&self, permutation: &[usize]) -> Result<Shape> {
        self.valid_rank(permutation.len())?;
        self.valid_dimensions(permutation)?;

        let (sizes, strides) = permutation
            .iter()
            .map(|i| (self.sizes[*i], self.strides[*i]))
            .collect();

        Ok(Shape {
            sizes,
            strides,
            offset: self.offset,
        })
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Shape> {
        let rank = self.rank();
        if rank < 2 {
            return Err(TransposeError.into());
        }

        let mut permutation = Vec::from_iter(0..rank);
        permutation.swap(dim_1, dim_2);

        self.permute(&permutation)
    }

    pub(crate) fn flip(&self, flips: &[usize]) -> Result<Shape, DimensionError> {
        self.valid_dimensions(flips)?;

        let strides = self
            .strides
            .iter()
            .enumerate()
            .map(|(i, &stride)| {
                if flips.contains(&i) {
                    match stride {
                        Stride::Positive(stride_val) => Stride::Negative(stride_val),
                        Stride::Negative(stride_val) => Stride::Positive(stride_val),
                    }
                } else {
                    stride
                }
            })
            .collect();

        Ok(Shape {
            sizes: self.sizes.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn expand(&self, expansions: &[usize]) -> Result<Shape> {
        if self.sizes == expansions {
            return Ok(self.clone());
        }

        self.valid_rank(expansions.len())?;

        let (sizes, strides) = self
            .sizes
            .iter()
            .zip(self.strides.iter())
            .zip(expansions)
            .map(|((&size, &stride), &expansion)| {
                if expansion == size {
                    Ok((size, stride))
                } else if size == 1 {
                    Ok((expansion, Stride::new(0, true)))
                } else {
                    Err(ExpansionError { size, expansion })
                }
            })
            .collect::<Result<(Vec<usize>, Vec<Stride>), ExpansionError>>()?;

        Ok(Shape {
            sizes,
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn squeeze(&self) -> Result<Shape, PhantomError> {
        if self.numel() == 1 {
            return Ok(Shape {
                sizes: vec![1],
                strides: vec![Stride::Positive(1)],
                offset: self.offset,
            });
        }

        let (sizes, strides) = self
            .sizes
            .iter()
            .zip(&self.strides)
            .filter_map(|(&size, &stride)| (size != 1).then_some((size, stride)))
            .collect();

        Ok(Shape {
            sizes,
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn unsqueeze(&self, unsqueezed: usize) -> Result<Shape, UnsqueezeError> {
        let current = self.rank();

        match unsqueezed.cmp(&current) {
            Ordering::Equal => Ok(self.clone()),
            Ordering::Less => Err(UnsqueezeError {
                current,
                unsqueezed,
            }),
            Ordering::Greater => {
                let ones_len = unsqueezed - current;
                let mut sizes = self.sizes.to_vec();
                sizes.splice(..0, repeat(1).take(ones_len));

                Ok(Shape::new(&sizes))
            }
        }
    }

    // --- Index, Slice and Pad ---

    pub(crate) fn idx(&self, indices: &[usize]) -> usize {
        self.sizes
            .iter()
            .zip(self.strides.iter())
            .zip(indices)
            .map(|((&size, stride), &index)| stride.offset(index, size))
            .sum::<usize>()
            + self.offset
    }

    pub(crate) fn index(&self, indices: &[usize]) -> Result<usize, IndexError> {
        let mut indices = indices.to_vec();
        indices.resize(self.rank(), 0);
        self.valid_indices(&indices, &Vec::from_iter(0..indices.len()))?;

        Ok(self.idx(&indices))
    }

    pub(crate) fn index_dims(
        &self,
        dimensions: &[usize],
        indices: &[usize],
    ) -> Result<usize, IndexError> {
        self.valid_indices(indices, dimensions)?;

        Ok((0..self.rank())
            .map(|dimension| {
                if let Some(position) = dimensions.iter().position(|&d| d == dimension) {
                    let index = indices[position];
                    let size = self.sizes[dimension];
                    let stride = self.strides[dimension];

                    stride.offset(index, size)
                } else {
                    let size = self.sizes[dimension];
                    let stride = self.strides[dimension];

                    stride.offset(0, size)
                }
            })
            .sum::<usize>()
            + self.offset)
    }

    pub(crate) fn slice(&self, indices: &[(usize, usize)]) -> Result<Shape> {
        self.valid_contiguity()?;

        if indices.is_empty() {
            return Ok(self.clone());
        }

        let mut indices = indices.to_vec();
        indices.resize(self.rank(), (0, 0));
        self.valid_ranges(&indices, &Vec::from_iter(0..indices.len()))?;

        let mut offset = match self.strides.first().ok_or(EmptyTensorError::Slice)? {
            Stride::Positive(_) => self.offset,
            Stride::Negative(_) => self.numel() - 1 - self.offset,
        };

        let sizes = self
            .sizes
            .iter()
            .zip(self.strides.iter())
            .zip(indices)
            .map(|((&size, stride), (start, end))| {
                let end = if end == 0 { size } else { end };

                match stride {
                    Stride::Positive(stride_val) => offset += start * stride_val,
                    Stride::Negative(stride_val) => offset -= (end - 1) * stride_val,
                };

                end - start
            })
            .collect();

        Ok(Shape {
            sizes,
            strides: self.strides.to_vec(),
            offset,
        })
    }

    pub(crate) fn slice_dims(
        &self,
        dimensions: &[usize],
        indices: &[(usize, usize)],
    ) -> Result<Shape> {
        self.valid_contiguity()?;
        self.valid_dimensions(dimensions)?;
        self.valid_ranges(indices, dimensions)?;

        if indices.is_empty() {
            return Ok(self.clone());
        }

        let mut offset = match self.strides.first().ok_or(EmptyTensorError::Slice)? {
            Stride::Positive(_) => self.offset,
            Stride::Negative(_) => self.numel() - 1 - self.offset,
        };

        let sizes = (0..self.rank())
            .map(|dimension| {
                if let Some(position) = dimensions.iter().position(|&d| d == dimension) {
                    let (start, end) = indices[position];
                    let end = if end == 0 { self.sizes[dimension] } else { end };

                    match self.strides[dimension] {
                        Stride::Positive(stride_val) => offset += start * stride_val,
                        Stride::Negative(stride_val) => offset -= (end - 1) * stride_val,
                    };

                    end - start
                } else {
                    let size = self.sizes[dimension];

                    match self.strides[dimension] {
                        Stride::Positive(_) => {}
                        Stride::Negative(stride_val) => offset -= size * stride_val,
                    };

                    size
                }
            })
            .collect();

        Ok(Shape {
            sizes,
            strides: self.strides.to_vec(),
            offset,
        })
    }

    pub(crate) fn pad(&self, padding: &[(usize, usize)]) -> Result<Shape, PhantomError> {
        let mut padding = padding.to_vec();
        padding.resize(self.rank(), (0, 0));

        let sizes = self
            .sizes
            .iter()
            .zip(padding)
            .map(|(&size, (start, end))| start + size + end)
            .collect::<Vec<usize>>();

        Ok(Shape::new(&sizes))
    }

    pub(crate) fn pad_dims(
        &self,
        padding: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<Shape, DimensionError> {
        self.valid_dimensions(dimensions)?;

        let sizes = (0..self.rank())
            .map(|dimension| {
                if let Some(position) = dimensions.iter().position(|&d| d == dimension) {
                    let (start, end) = padding[position];
                    start + self.sizes[dimension] + end
                } else {
                    self.sizes[dimension]
                }
            })
            .collect::<Vec<usize>>();

        Ok(Shape::new(&sizes))
    }

    pub(crate) fn slicer(&self, indices: &[Option<usize>]) -> Result<Shape> {
        self.valid_contiguity()?;

        let mut offset = match self.strides.first().ok_or(EmptyTensorError::Slice)? {
            Stride::Positive(_) => self.offset,
            Stride::Negative(_) => self.numel() - 1 - self.offset,
        };

        let sizes = self
            .sizes
            .iter()
            .zip(&self.strides)
            .zip(indices)
            .map(|((&size, &stride), i)| {
                if let Some(i) = i {
                    match stride {
                        Stride::Positive(stride_val) => offset += i * stride_val,
                        Stride::Negative(stride_val) => offset -= i * stride_val,
                    }

                    1
                } else {
                    size
                }
            })
            .collect();

        Ok(Shape {
            sizes,
            strides: self.strides.to_vec(),
            offset,
        })
    }

    // --- Broadcast ---

    pub(crate) fn broadcast(
        lhs_sizes: &[usize],
        rhs_sizes: &[usize],
    ) -> Result<Vec<usize>, BroadcastError> {
        let mut lhs_iter = lhs_sizes.iter();
        let mut rhs_iter = rhs_sizes.iter();

        let max_len = max(lhs_sizes.len(), rhs_sizes.len());
        let mut result = Vec::with_capacity(max_len);

        loop {
            match (lhs_iter.next_back(), rhs_iter.next_back()) {
                (Some(&l), Some(&r)) => {
                    if l == r {
                        result.push(l);
                    } else if l == 1 {
                        result.push(r);
                    } else if r == 1 {
                        result.push(l);
                    } else {
                        return Err(BroadcastError {
                            lhs_sizes: lhs_sizes.to_vec(),
                            rhs_sizes: rhs_sizes.to_vec(),
                        });
                    }
                }
                (Some(&l), None) => result.push(l),
                (None, Some(&r)) => result.push(r),
                (None, None) => break,
            }
        }

        result.reverse();
        Ok(result)
    }

    // --- Validation ---

    pub(crate) fn is_contiguous(&self) -> bool {
        for i in 0..self.rank() - 1 {
            if self.strides[i] != self.strides[i + 1] * self.sizes[i + 1] {
                return false;
            }
        }

        true
    }

    pub(crate) fn valid_contiguity(&self) -> Result<(), NonContiguousError> {
        if self.is_contiguous() {
            Ok(())
        } else {
            Err(NonContiguousError)
        }
    }

    pub(crate) fn valid_reshape(&self, sizes: &[usize]) -> Result<(), ReshapeError> {
        if self.numel() != sizes.iter().product::<usize>() {
            return Err(ReshapeError {
                current_shape: self.sizes.to_vec(),
                new_shape: sizes.to_vec(),
            });
        }

        Ok(())
    }

    fn valid_indices(&self, indices: &[usize], dimensions: &[usize]) -> Result<(), IndexError> {
        for (&dimension, &index) in dimensions.iter().zip(indices) {
            let size = self.sizes[dimension];

            if index >= size {
                return Err(IndexError::OutOfRange {
                    index,
                    dimension,
                    size,
                });
            }
        }

        Ok(())
    }

    fn valid_ranges(
        &self,
        ranges: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<(), RangeError> {
        for (&dimension, &range) in dimensions.iter().zip(ranges) {
            let size = self.sizes[dimension];

            if range.0 > range.1 && range.1 > 0 {
                return Err(RangeError::GreaterStartRange(range.0, range.1));
            } else if range.0 > size || range.1 > size {
                return Err(RangeError::OutOfRange {
                    range,
                    dimension,
                    size,
                });
            }
        }

        Ok(())
    }

    pub(crate) fn valid_dimensions(&self, dimensions: &[usize]) -> Result<(), DimensionError> {
        let dim_range = self.rank() - 1;
        let mut set = HashSet::with_capacity(dimensions.len());

        for &dimension in dimensions {
            if dim_range < dimension {
                return Err(DimensionError::OutOfRange {
                    dimension,
                    dim_range,
                });
            } else if set.contains(&dimension) {
                return Err(DimensionError::Repetition(dimension));
            } else {
                set.insert(dimension);
            }
        }

        Ok(())
    }

    fn valid_rank(&self, num_indices: usize) -> Result<(), IndexError> {
        let num_dimensions = self.rank();

        if num_indices != num_dimensions {
            Err(IndexError::IndicesLength {
                num_indices,
                num_dimensions,
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn valid_data_length(
        &self,
        data_length: usize,
    ) -> Result<(), InvalidDataLengthError> {
        let numel = self.numel();

        if data_length != numel {
            Err(InvalidDataLengthError {
                data_length,
                tensor_size: numel,
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn conv_larger_input(
        input_sizes: &[usize],
        kernel_sizes: &[usize],
    ) -> Result<bool, ValidConvShapeError> {
        if input_sizes.iter().zip(kernel_sizes).all(|(&i, &k)| i >= k) {
            Ok(true)
        } else if input_sizes.iter().zip(kernel_sizes).all(|(&i, &k)| k >= i) {
            Ok(false)
        } else {
            Err(ValidConvShapeError {
                input_sizes: input_sizes.to_vec(),
                kernel_sizes: kernel_sizes.to_vec(),
            })
        }
    }
}

impl PartialEq for Shape {
    fn eq(&self, rhs: &Shape) -> bool {
        self.sizes == rhs.sizes && self.strides == rhs.strides
    }
}

impl Stride {
    pub(crate) fn new(stride_val: usize, positive: bool) -> Stride {
        if positive {
            Stride::Positive(stride_val)
        } else {
            Stride::Negative(stride_val)
        }
    }

    pub(crate) fn offset(&self, index: usize, size: usize) -> usize {
        match self {
            Stride::Positive(stride_val) => index * stride_val,
            Stride::Negative(stride_val) => (size - 1 - index) * stride_val,
        }
    }
}

impl Mul<usize> for Stride {
    type Output = Stride;

    fn mul(self, rhs: usize) -> Self::Output {
        match self {
            Stride::Positive(stride_val) => Stride::Positive(stride_val * rhs),
            Stride::Negative(stride_val) => Stride::Negative(stride_val * rhs),
        }
    }
}

impl PartialEq for Stride {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Stride::Positive(lhs), Stride::Positive(rhs)) => lhs == rhs,
            (Stride::Negative(lhs), Stride::Negative(rhs)) => lhs == rhs,
            (Stride::Positive(0), Stride::Negative(0)) => true,
            (Stride::Negative(0), Stride::Positive(0)) => true,
            _ => false,
        }
    }
}
