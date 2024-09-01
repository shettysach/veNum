use crate::Res;
use std::{cmp::max, cmp::Ordering, collections::HashSet, iter::repeat, ops::Mul};

#[derive(Clone)]
pub(crate) struct Shape {
    pub sizes: Vec<usize>,
    pub strides: Vec<Stride>,
    pub offset: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum Stride {
    Positive(usize),
    Negative(usize),
}

impl Shape {
    pub fn new(sizes: &[usize], offset: usize) -> Shape {
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
            offset,
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
            return Ok(Shape::new(&[1], self.offset));
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

                Ok(Shape::new(&sizes, 0))
            }
        }
    }

    pub(crate) fn permute(&self, permutation: &[usize]) -> Res<Shape> {
        self.matches_size(permutation.len())?;
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

        let mut permutation: Vec<usize> = (0..ndims).collect();
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
            sizes: self.sizes.clone(),
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn expand(&self, expansions: &[usize]) -> Res<Shape> {
        if self.sizes == expansions {
            return Ok(self.clone());
        }
        self.matches_size(expansions.len())?;

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

    pub(crate) fn element(&self, indices: &[usize]) -> Res<usize> {
        self.matches_size(indices.len())?;
        self.valid_indices(indices)?;

        Ok(self
            .sizes
            .iter()
            .zip(self.strides.iter())
            .zip(indices)
            .map(|((&size, stride), &index)| stride.offset(index, size))
            .sum::<usize>()
            + self.offset)
    }

    pub(crate) fn slice(&self, indices: &[(usize, usize)]) -> Res<Shape> {
        self.valid_contiguity()?;
        self.valid_ranges(indices)?;

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
            strides: self.strides.clone(),
            offset,
        })
    }

    pub(crate) fn single_slice(&self, indices: &[Option<usize>]) -> Res<Shape> {
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
            strides: self.strides.clone(),
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

    fn matches_size(&self, length: usize) -> Res<()> {
        let num_dimensions = self.ndims();

        if length != num_dimensions {
            return Err(format!(
                "Number of indices ({}) does not match the number of dimensions ({}).",
                length, num_dimensions
            ));
        }

        Ok(())
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

    fn valid_indices(&self, indices: &[usize]) -> Res<()> {
        for (dimension, (&size, &index)) in self.sizes.iter().zip(indices).enumerate() {
            if index >= size {
                return Err(format!(
                    "Index {} is out of range for dimension {} (size: {}).",
                    index, dimension, size
                ));
            }
        }

        Ok(())
    }

    fn valid_ranges(&self, indices: &[(usize, usize)]) -> Res<()> {
        for (dimension, index) in indices.iter().enumerate() {
            let size = self.sizes.get(dimension).ok_or_else(|| {
                format!(
                    "Index ({}) is greater than ndims ({}).",
                    dimension,
                    self.ndims()
                )
            })?;

            if index.0 > index.1 && index.1 > 0 {
                return Err(format!(
                    "Range start index {} is greater than range end index {}.",
                    index.0, index.1
                ));
            } else if &index.0 > size || &index.1 > size {
                return Err(format!(
                    "Index {:?} is out of range for dimension {} (size: {}).",
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
