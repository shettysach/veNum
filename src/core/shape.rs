use core::panic;
use std::{cmp::max, collections::HashSet, ops::Mul};

#[derive(Clone)]
pub(crate) struct Shape {
    pub sizes: Vec<usize>,
    pub strides: Vec<Stride>,
    pub offset: usize,
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum Stride {
    Positive(usize),
    Negative(usize),
}

impl Shape {
    pub fn new(sizes: &[usize], offset: usize) -> Shape {
        let mut current = 1;
        let strides = sizes
            .iter()
            .rev()
            .map(|size| {
                let magnitude = current;
                current *= size;
                Stride::new(magnitude, true)
            })
            .collect::<Vec<Stride>>()
            .into_iter()
            .rev()
            .collect();

        Shape {
            sizes: sizes.to_vec(),
            strides,
            offset,
        }
    }

    pub(crate) fn numdims(&self) -> usize {
        self.sizes.len()
    }

    pub(crate) fn numel(&self) -> usize {
        self.sizes.iter().product()
    }

    // Sizes

    // TODO: Modify reshape now that flip has been introduced
    // Needs validation for when Tensors cannot be reshaped without copying
    pub(crate) fn reshape(&self, sizes: &[usize], offset: usize) -> Shape {
        self.valid_reshape(sizes);

        let mut current = 1;
        let mut last_sign = true;

        let strides = sizes
            .iter()
            .rev()
            .enumerate()
            .map(|(i, size)| {
                let stride_val = current;

                last_sign = if let Some(stride) = self.strides.get(i) {
                    match stride {
                        Stride::Positive(_) => true,
                        Stride::Negative(_) => false,
                    }
                } else {
                    last_sign
                };

                current *= size;

                Stride::new(stride_val, last_sign)
            })
            .collect::<Vec<Stride>>()
            .into_iter()
            .rev()
            .collect();

        Shape {
            sizes: sizes.to_vec(),
            strides,
            offset,
        }
    }

    pub(crate) fn squeeze(&self) -> Shape {
        if self.sizes.iter().product::<usize>() == 1 {
            Shape::new(&[1], self.offset)
        } else {
            let (sizes, strides) = self
                .sizes
                .iter()
                .zip(self.strides.iter())
                .filter_map(|(&size, &stride)| {
                    if size != 1 {
                        Some((size, stride))
                    } else {
                        None
                    }
                })
                .unzip();

            Shape {
                sizes,
                strides,
                offset: self.offset,
            }
        }
    }

    pub(crate) fn permute(&self, permutation: &[usize]) -> Shape {
        self.matches_size(permutation.len());
        self.dimensions_occur_only_once(permutation);

        let (sizes, strides) = permutation
            .iter()
            .map(|i| (self.sizes[*i], self.strides[*i]))
            .collect();

        Shape {
            sizes,
            strides,
            offset: self.offset,
        }
    }

    pub(crate) fn flip(&self, flips: &[usize]) -> Shape {
        self.dimensions_occur_only_once(flips);

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

        Shape {
            sizes: self.sizes.clone(),
            strides,
            offset: self.offset,
        }
    }

    pub(crate) fn expand(&self, expansions: &[usize]) -> Shape {
        if self.sizes == expansions {
            self.clone()
        } else {
            self.matches_size(expansions.len());

            let (sizes, strides) = self
                .sizes
                .iter()
                .zip(self.strides.iter())
                .zip(expansions)
                .map(|((&size, &stride), &expansion)| {
                    if expansion == size {
                        (size, stride)
                    } else if expansion % size == 0 {
                        (expansion, Stride::new(0, true))
                    } else {
                        panic!("Size {size} cannot be expaned to size {expansion}.");
                    }
                })
                .collect();

            Shape {
                sizes,
                strides,
                offset: self.offset,
            }
        }
    }

    // Broadcast

    pub(crate) fn broadcast(&self, rhs: &Shape) -> Vec<usize> {
        let lhs_shape = &self.sizes;
        let rhs_shape = &rhs.sizes;

        let mut lhs_iter = lhs_shape.iter();
        let mut rhs_iter = rhs_shape.iter();

        let max_len = max(lhs_shape.len(), rhs_shape.len());
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
                        panic!(
                            "Shapes {:?} and {:?} cannot be broadcast together.",
                            lhs_shape, rhs_shape
                        );
                    }
                }
                (Some(&l), None) => result.push(l),
                (None, Some(&r)) => result.push(r),
                (None, None) => break,
            }
        }

        result.reverse();
        result
    }

    // Index

    pub(crate) fn element(&self, indices: &[usize]) -> usize {
        self.matches_size(indices.len());
        self.valid_indices(indices);

        self.sizes
            .iter()
            .zip(self.strides.iter())
            .zip(indices)
            .map(|((&size, stride), &index)| stride.offset(index, size))
            .sum::<usize>()
            + self.offset
    }

    pub(crate) fn slice(&self, indices: &[(usize, usize)]) -> Shape {
        self.valid_ranges(indices);

        let mut indices = Vec::from(indices);
        indices.resize(self.numdims(), (0, 0));
        let mut offset = self.offset;

        let sizes = self
            .sizes
            .iter()
            .zip(self.strides.iter())
            .zip(indices)
            .map(|((&size, stride), index)| {
                offset += stride.offset(index.0, size);

                if index.1 == 0 {
                    size - index.0
                } else {
                    index.1 - index.0
                }
            })
            .collect();

        let strides = self.strides.clone();

        Shape {
            sizes,
            strides,
            offset,
        }
    }

    // Validation

    fn matches_size(&self, length: usize) {
        let num_dimensions = self.numdims();

        if length != num_dimensions {
            panic!(
                "Number of indices ({}) does not match the number of dimensions ({}).",
                length, num_dimensions
            )
        }
    }

    fn valid_reshape(&self, sizes: &[usize]) {
        let data_len = self.numel();
        let new_size = sizes.iter().product::<usize>();

        if new_size == 1 {
            sizes.len()
        } else {
            new_size
        };

        if data_len != new_size {
            panic!(
                "({:?}) cannot be reshaped to ({:?}).\nData length ({data_len}) does not match new length ({new_size}).",
                self.sizes, sizes,
            )
        }
    }

    fn valid_indices(&self, indices: &[usize]) {
        for (dimension, (size, index)) in self.sizes.iter().zip(indices).enumerate() {
            if index >= size {
                panic!(
                    "Index {} is out of range for dimension {} (size: {}).",
                    index, dimension, size
                );
            };
        }
    }

    fn valid_ranges(&self, indices: &[(usize, usize)]) {
        for (dimension, index) in indices.iter().enumerate() {
            let size = self
                .sizes
                .get(dimension)
                .expect("Indices length is longer than number of dimensions");

            if index.0 > index.1 && index.1 > 0 {
                panic!(
                    "Range start index {} is greater than range end index {}.",
                    index.0, index.1
                );
            } else if &index.0 > size || &index.1 > size {
                panic!(
                    "Index {:?} is out of range for dimension {} (size: {}).",
                    index, dimension, size
                );
            };
        }
    }

    fn dimensions_occur_only_once(&self, dimensions: &[usize]) {
        let mut set = HashSet::with_capacity(dimensions.len());
        for dimension in dimensions {
            if set.contains(dimension) {
                panic!("Dimension {dimension} repeats");
            } else {
                set.insert(dimension)
            };
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

    fn mul(self, rhs: usize) -> Stride {
        match self {
            Stride::Positive(stride_val) => Stride::Positive(stride_val * rhs),
            Stride::Negative(stride_val) => Stride::Negative(stride_val * rhs),
        }
    }
}
