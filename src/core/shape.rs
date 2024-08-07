use core::panic;
use std::{cmp::max, collections::HashSet};

#[derive(Debug, Clone)]
pub(crate) struct Shape {
    pub sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Shape {
    pub fn new(sizes: &[usize], offset: usize) -> Shape {
        let mut current = 1;
        let mut strides = Vec::with_capacity(sizes.len());

        for dim in sizes.iter().rev() {
            strides.push(current);
            current *= dim;
        }
        strides.reverse();

        Shape {
            sizes: sizes.to_vec(),
            strides,
            offset,
        }
    }

    pub(crate) fn numdims(&self) -> usize {
        self.sizes.len()
    }

    pub(crate) fn index_to_position(&self, indices: &[usize]) -> usize {
        self.matches_size(indices.len());
        self.within_range(indices);

        self.strides
            .iter()
            .zip(indices)
            .map(|(stride, index)| stride * index)
            .sum::<usize>()
            + self.offset
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

    // Broadcast

    pub fn broadcast(&self, rhs: &Shape) -> Vec<usize> {
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

    pub fn expand(&self, expansions: &[usize]) -> Shape {
        if self.sizes == expansions {
            self.clone()
        } else {
            self.matches_size(expansions.len());
            let (expanded_sizes, expanded_strides) = self
                .sizes
                .iter()
                .zip(self.strides.iter())
                .zip(expansions)
                .map(|((&size, &stride), &expansion)| {
                    if expansion == size {
                        (size, stride)
                    } else if expansion % size == 0 {
                        (expansion, 0)
                    } else {
                        panic!("Size {size} cannot be expaned to size {expansion}");
                    }
                })
                .collect();

            Shape {
                sizes: expanded_sizes,
                strides: expanded_strides,
                offset: self.offset,
            }
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

    fn within_range(&self, indices: &[usize]) {
        for (dimension, (size, index)) in self.sizes.iter().zip(indices).enumerate() {
            if index >= size {
                panic!(
                    "Index {} is out of range for dimension {} (size: {}).",
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

    pub(crate) fn valid_indices(&self, indices: &[usize]) -> bool {
        !self.sizes.is_empty()
            && self.numdims() == indices.len()
            && indices.iter().zip(self.sizes.iter()).all(|(i, s)| i < s)
    }
}

impl PartialEq for Shape {
    fn eq(&self, rhs: &Shape) -> bool {
        self.sizes == rhs.sizes && self.strides == rhs.strides
    }
}

impl Eq for Shape {}
