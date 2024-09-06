use crate::{Res, Slide};
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

    pub(crate) fn ndims(&self) -> usize {
        self.sizes.len()
    }

    pub(crate) fn numel(&self) -> usize {
        self.sizes.iter().product()
    }

    // Shape operations

    pub(crate) fn view(&self, sizes: &[usize]) -> Res<Shape> {
        self.valid_contiguity()?;
        self.valid_reshape(sizes)?;

        let mut current = 1;
        let positive = match self
            .strides
            .first()
            .ok_or_else(|| String::from("Strides are empty. Unable to reshape/view."))?
        {
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

    pub(crate) fn squeeze(&self) -> Res<Shape> {
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

    pub(crate) fn unsqueeze(&self, expansion: usize) -> Res<Shape> {
        let ndims = self.ndims();

        match expansion.cmp(&ndims) {
            Ordering::Equal => Ok(self.clone()),
            Ordering::Less => Err(format!(
                "Current ndims ({}) is greater than expanded ndims ({}).",
                ndims, expansion
            )),
            Ordering::Greater => {
                let ones_len = expansion - ndims;
                let mut sizes = self.sizes.to_vec();
                sizes.splice(..0, repeat(1).take(ones_len));

                Ok(Shape::new(&sizes))
            }
        }
    }

    pub(crate) fn permute(&self, permutation: &[usize]) -> Res<Shape> {
        self.valid_ndims(permutation.len())?;
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

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Res<Shape> {
        let ndims = self.ndims();
        if ndims < 2 {
            return Err(String::from("Transpose requires at least two dimensions"));
        }

        let mut permutation = Vec::from_iter(0..ndims);
        permutation.swap(dim_1, dim_2);

        self.permute(&permutation)
    }

    pub(crate) fn flip(&self, flips: &[usize]) -> Res<Shape> {
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

    pub(crate) fn expand(&self, expansions: &[usize]) -> Res<Shape> {
        if self.sizes == expansions {
            return Ok(self.clone());
        }

        self.valid_ndims(expansions.len())?;

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
                    Err(format!(
                        "Size {size} cannot be expaned to size {expansion}."
                    ))
                }
            })
            .collect::<Res<(Vec<usize>, Vec<Stride>)>>()?;

        Ok(Shape {
            sizes,
            strides,
            offset: self.offset,
        })
    }

    // Index

    pub(crate) fn index(&self, indices: &[usize]) -> Res<usize> {
        self.valid_ndims(indices.len())?;
        self.valid_indices(indices, &Vec::from_iter(0..indices.len()))?;

        Ok(self
            .sizes
            .iter()
            .zip(self.strides.iter())
            .zip(indices)
            .map(|((&size, stride), &index)| stride.offset(index, size))
            .sum::<usize>()
            + self.offset)
    }

    pub(crate) fn index_dimension(&self, dimensions: &[usize], indices: &[usize]) -> Res<usize> {
        self.valid_ndims(indices.len())?;
        self.valid_indices(indices, dimensions)?;

        Ok((0..self.ndims())
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

    pub(crate) fn slice(&self, indices: &[(usize, usize)]) -> Res<Shape> {
        self.valid_contiguity()?;
        self.valid_ranges(indices, &Vec::from_iter(0..indices.len()))?;

        let mut indices = indices.to_vec();
        indices.resize(self.ndims(), (0, 0));

        let mut offset = match self
            .strides
            .first()
            .ok_or_else(|| String::from("Strides are empty. Unable to slice."))?
        {
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

    pub(crate) fn slice_dimensions(
        &self,
        dimensions: &[usize],
        indices: &[(usize, usize)],
    ) -> Res<Shape> {
        self.valid_contiguity()?;
        self.valid_dimensions(dimensions)?;
        self.valid_ranges(indices, dimensions)?;

        let mut offset = match self
            .strides
            .first()
            .ok_or_else(|| String::from("Strides are empty. Unable to slice."))?
        {
            Stride::Positive(_) => self.offset,
            Stride::Negative(_) => self.numel() - 1 - self.offset,
        };

        // NOTE: better way to find pos / iter ?
        let sizes = (0..self.ndims())
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

    pub(crate) fn pad(&self, padding: &[(usize, usize)]) -> Res<Shape> {
        let mut padding = padding.to_vec();
        padding.resize(self.ndims(), (0, 0));

        let sizes = self
            .sizes
            .iter()
            .zip(padding)
            .map(|(&size, (start, end))| start + size + end)
            .collect::<Vec<usize>>();

        Ok(Shape::new(&sizes))
    }

    pub(crate) fn pad_dimensions(
        &self,
        padding: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Res<Shape> {
        self.valid_dimensions(dimensions)?;

        let sizes = (0..self.ndims())
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

    pub(crate) fn slicer(&self, indices: &[Option<usize>]) -> Res<Shape> {
        self.valid_contiguity()?;

        let mut offset = match self
            .strides
            .first()
            .ok_or_else(|| String::from("Strides are empty. Unable to slice."))?
        {
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

    // Broadcast

    pub(crate) fn broadcast(lhs_sizes: &[usize], rhs_sizes: &[usize]) -> Res<Vec<usize>> {
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
                        return Err(format!(
                            "Shapes {:?} and {:?} cannot be broadcast together.",
                            lhs_sizes, rhs_sizes
                        ));
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

    // Validation

    pub(crate) fn is_contiguous(&self) -> bool {
        for i in 0..self.ndims() - 1 {
            if self.strides[i] != self.strides[i + 1] * self.sizes[i + 1] {
                return false;
            }
        }

        true
    }

    pub(crate) fn valid_contiguity(&self) -> Res<()> {
        if self.is_contiguous() {
            Ok(())
        } else {
            Err("Shape is not contiguous. Use `to_contiguous()`.".to_string())
        }
    }

    fn valid_ndims(&self, length: usize) -> Res<()> {
        let num_dimensions = self.ndims();

        if length != num_dimensions {
            Err(format!(
                "Number of indices ({}) does not match the number of dimensions ({}).",
                length, num_dimensions
            ))
        } else {
            Ok(())
        }
    }

    pub(crate) fn valid_reshape(&self, sizes: &[usize]) -> Res<()> {
        let data_len = self.numel();
        let new_size = sizes.iter().product::<usize>();

        if data_len != new_size {
            return Err(format!(
            "({:?}) cannot be reshaped to ({:?}). Data length ({}) does not match new length ({}).",
            self.sizes, sizes, data_len, new_size
        ));
        }

        Ok(())
    }

    fn valid_indices(&self, indices: &[usize], dimensions: &[usize]) -> Res<()> {
        for (&dimension, &index) in dimensions.iter().zip(indices) {
            let size = self.sizes[dimension];

            if index >= size {
                return Err(format!(
                    "Index {} is out of range for dimension {} (size: {}).",
                    index, dimension, size
                ));
            }
        }

        Ok(())
    }

    fn valid_ranges(&self, ranges: &[(usize, usize)], dimensions: &[usize]) -> Res<()> {
        for (&dimension, index) in dimensions.iter().zip(ranges) {
            let size = self.sizes[dimension];

            if index.0 > index.1 && index.1 > 0 {
                return Err(format!(
                    "Range start index {} is greater than range end index {}.",
                    index.0, index.1
                ));
            } else if index.0 > size || index.1 > size {
                return Err(format!(
                    "Range {:?} is out of range for dimension {} (size: {}).",
                    index, dimension, size
                ));
            }
        }

        Ok(())
    }

    pub(crate) fn valid_dimensions(&self, dimensions: &[usize]) -> Res<()> {
        let max_dimension = self.ndims() - 1;
        let mut set = HashSet::with_capacity(dimensions.len());

        for &dimension in dimensions {
            if max_dimension < dimension {
                return Err(format!(
                    "Dimension {} is greater than max range of dimensions ({}).",
                    dimension, max_dimension
                ));
            } else if set.contains(&dimension) {
                return Err(format!("Dimension {} repeats.", dimension));
            } else {
                set.insert(dimension);
            }
        }

        Ok(())
    }

    pub(crate) fn valid_data_length<T>(&self, data: &[T]) -> Res<()> {
        let data_length = data.len();
        let numel = self.numel();

        if data_length != numel {
            Err(format!(
                "Data length ({}) does not match size of slice ({}).",
                data_length, numel
            ))
        } else {
            Ok(())
        }
    }

    // TODO: Handle cases where kernel > input
    pub(crate) fn valid_convolution(
        input_shape: &[usize],
        kernel_shape: &[usize],
        mode: &Slide,
    ) -> Res<()> {
        match mode {
            Slide::Valid => {
                for (&i, &k) in input_shape.iter().zip(kernel_shape) {
                    if k > i {
                        return Err(format!(
                            "Kernel size ({}) is greater than input size ({}), for mode Valid.",
                            k, i
                        ));
                    }
                }
            }
            Slide::Full => {}
            Slide::Same => {
                for (&i, &k) in input_shape.iter().zip(kernel_shape) {
                    if k > i + 1 {
                        return Err(format!(
                            "Kernel size ({}) is greater than input size ({}) + 1, for mode Same.",
                            k, i
                        ));
                    }
                }
            }
        }

        Ok(())
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
