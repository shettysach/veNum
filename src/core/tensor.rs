use crate::core::{
    errors::*,
    iters::{Indexer, Slicer},
    shape::{Shape, Stride},
    utils::cast_usize,
};
use anyhow::Result;
use num_traits::{FromPrimitive, NumOps, One, Zero};
use std::{cmp::Ordering, fmt::Debug, iter::successors, ops::Add, sync::Arc};

pub struct Tensor<T> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Shape,
}

impl<T: Copy> Tensor<T> {
    // --- Init ---

    pub(crate) fn init(data: Vec<T>, sizes: &[usize]) -> Tensor<T> {
        Tensor {
            data: Arc::new(data),
            shape: Shape::new(sizes),
        }
    }

    pub fn new(data: &[T], sizes: &[usize]) -> Result<Tensor<T>> {
        let data_length = data.len();
        let tensor_size = sizes.iter().product();

        if data_length != tensor_size {
            return Err(InvalidDataLengthError {
                data_length,
                tensor_size,
            }
            .into());
        }

        Ok(Tensor::init(data.to_vec(), sizes))
    }

    pub fn new_1d(data: &[T]) -> Result<Tensor<T>> {
        Ok(Tensor::init(data.to_vec(), &[data.len()]))
    }

    pub fn scalar(data: T) -> Result<Tensor<T>> {
        Ok(Tensor {
            data: Arc::new(vec![data]),
            shape: Shape::scalar(),
        })
    }

    pub fn same(element: T, size: usize) -> Result<Tensor<T>> {
        Ok(Tensor::init(vec![element; size], &[size]))
    }

    pub fn zeroes(size: usize) -> Result<Tensor<T>>
    where
        T: Zero,
    {
        Tensor::same(T::zero(), size)
    }

    pub fn ones(size: usize) -> Result<Tensor<T>>
    where
        T: One,
    {
        Tensor::same(T::one(), size)
    }

    pub fn arange(start: T, end: T, step: T) -> Result<Tensor<T>>
    where
        T: Add<Output = T> + PartialOrd + Zero,
    {
        let step_order = step
            .partial_cmp(&T::zero())
            .ok_or(ArangeError::Comparison)?;

        let comparison_fn: fn(T, T) -> bool = match step_order {
            Ordering::Greater => match end > start {
                true => |current, end| end > current,
                false => return Err(ArangeError::Positive.into()),
            },
            Ordering::Less => match start > end {
                true => |current, end| current > end,
                false => return Err(ArangeError::Negative.into()),
            },
            Ordering::Equal => return Err(ArangeError::Zero.into()),
        };

        let data = successors(Some(start), |&prev| {
            let current = prev + step;
            comparison_fn(current, end).then_some(current)
        })
        .collect::<Vec<T>>();
        let num = data.len();

        Ok(Tensor::init(data, &[num]))
    }

    pub fn linspace(start: T, end: T, num: usize) -> Result<Tensor<T>>
    where
        T: NumOps + FromPrimitive + Debug,
    {
        let num_casted = cast_usize::<T>(num - 1)?;
        let step = (end - start) / num_casted;

        let data = successors(Some(start), |&prev| Some(prev + step))
            .take(num)
            .collect::<Vec<T>>();

        Ok(Tensor::init(data, &[num]))
    }

    pub fn eye(size: usize) -> Result<Tensor<T>>
    where
        T: Zero + One,
    {
        let diagonal = size + 1;
        let data = (0..size * size)
            .map(|elem| {
                if elem % diagonal == 0 {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect::<Vec<T>>();

        Ok(Tensor::init(data, &[size, size]))
    }

    pub fn to_contiguous(&self) -> Result<Tensor<T>> {
        Ok(Tensor {
            data: Arc::new(self.data_non_contiguous()),
            shape: Shape::new(&self.shape.sizes),
        })
    }

    pub(crate) fn into_contiguous(self) -> Result<Tensor<T>> {
        if self.is_contiguous() {
            Ok(self)
        } else {
            self.to_contiguous()
        }
    }

    // --- Data ---

    pub fn data(&self) -> Vec<T> {
        if self.is_contiguous() {
            self.data_contiguous().to_vec()
        } else {
            self.data_non_contiguous()
        }
    }

    pub(crate) fn data_contiguous(&self) -> &[T] {
        let start = self.offset();
        let end = start + self.numel();
        &self.data[start..end]
    }

    pub(crate) fn data_non_contiguous(&self) -> Vec<T> {
        Indexer::new(&self.shape.sizes)
            .map(|index| self.idx(&index))
            .collect()
    }

    pub(crate) fn idx(&self, indices: &[usize]) -> T {
        self.data[self.shape.idx(indices)]
    }

    pub fn index(&self, indices: &[usize]) -> Result<T, IndexError> {
        Ok(self.data[self.shape.index(indices)?])
    }

    pub fn index_dims(&self, dimensions: &[usize], indices: &[usize]) -> Result<T, IndexError> {
        Ok(self.data[self.shape.index_dims(dimensions, indices)?])
    }

    // --- New Data, New Shape ---

    pub fn reshape(&self, sizes: &[usize]) -> Result<Tensor<T>> {
        self.shape.valid_reshape(sizes)?;

        Ok(Tensor::init(self.data_non_contiguous(), sizes))
    }

    pub fn flatten(&self) -> Result<Tensor<T>> {
        self.reshape(&[self.numel()])
    }

    pub fn view_else_reshape(&self, sizes: &[usize]) -> Result<Tensor<T>> {
        self.view(sizes).or_else(|_| self.reshape(sizes))
    }

    pub fn pad(&self, constant: T, padding: &[(usize, usize)]) -> Result<Tensor<T>> {
        let shape = self.shape.pad(padding)?;
        let data = Arc::new(vec![constant; shape.numel()]);
        let tensor = Tensor { data, shape };

        let ranges = padding
            .iter()
            .enumerate()
            .map(|(dimension, &(start, _))| (start, self.shape.sizes[dimension] + start))
            .collect::<Vec<(usize, usize)>>();

        tensor.slice_zip(&self.data(), |_, new| new, &ranges)
    }

    pub fn pad_dims(
        &self,
        constant: T,
        dimensions: &[usize],
        padding: &[(usize, usize)],
    ) -> Result<Tensor<T>> {
        let shape = self.shape.pad_dims(padding, dimensions)?;
        let data = Arc::new(vec![constant; shape.numel()]);
        let tensor = Tensor { data, shape };

        let ranges = dimensions
            .iter()
            .zip(padding)
            .map(|(&dimension, &(start, _))| (start, self.shape.sizes[dimension] + start))
            .collect::<Vec<(usize, usize)>>();

        tensor.slice_zip_dims(&self.data(), |_, new| new, dimensions, &ranges)
    }

    // --- Maps, Zips and Reduce ---

    pub fn unary_map<R>(&self, f: impl Fn(T) -> R) -> Result<Tensor<R>> {
        let contiguous = self.is_contiguous();

        let data = if contiguous {
            self.data_contiguous().iter().map(|&elem| f(elem)).collect()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| {
                    let elem = self.idx(&index);
                    f(elem)
                })
                .collect()
        };

        let shape = if contiguous {
            Shape {
                sizes: self.sizes().to_vec(),
                strides: self.strides().to_vec(),
                offset: 0,
            }
        } else {
            Shape::new(self.sizes())
        };

        Ok(Tensor {
            data: Arc::new(data),
            shape,
        })
    }

    pub fn binary_map<R>(&self, rhs: T, f: impl Fn(T, T) -> R) -> Result<Tensor<R>> {
        let contiguous = self.is_contiguous();

        let data = if contiguous {
            self.data_contiguous()
                .iter()
                .map(|&elem| f(elem, rhs))
                .collect()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| {
                    let lhs_elem = self.idx(&index);
                    f(lhs_elem, rhs)
                })
                .collect()
        };

        let shape = if contiguous {
            Shape {
                sizes: self.sizes().to_vec(),
                strides: self.strides().to_vec(),
                offset: 0,
            }
        } else {
            Shape::new(self.sizes())
        };

        Ok(Tensor {
            data: Arc::new(data),
            shape,
        })
    }

    pub fn zip<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Result<Tensor<R>> {
        if self.shape == rhs.shape {
            self.equal_zip(rhs, f)
        } else {
            self.broadcast_zip(rhs, f)
        }
    }

    fn equal_zip<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Result<Tensor<R>> {
        let contiguous = self.is_contiguous() && rhs.is_contiguous();

        let data = if contiguous {
            self.data_contiguous()
                .iter()
                .zip(rhs.data_contiguous())
                .map(|(&lhs_elem, &rhs_elem)| f(lhs_elem, rhs_elem))
                .collect()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| {
                    let lhs_elem = self.idx(&index);
                    let rhs_elem = rhs.idx(&index);

                    f(lhs_elem, rhs_elem)
                })
                .collect()
        };

        let shape = if contiguous {
            Shape {
                sizes: self.sizes().to_vec(),
                strides: self.strides().to_vec(),
                offset: 0,
            }
        } else {
            Shape::new(self.sizes())
        };

        Ok(Tensor {
            data: Arc::new(data),
            shape,
        })
    }

    fn broadcast_zip<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Result<Tensor<R>> {
        let sizes = Shape::broadcast(&self.shape.sizes, &rhs.shape.sizes)?;
        let shape = Shape::new(&sizes);
        let expansion = sizes.len();

        let lhs_broadcasted = self.unsqueeze(expansion)?.expand(&sizes)?;
        let rhs_broadcasted = rhs.unsqueeze(expansion)?.expand(&sizes)?;

        let data = Arc::new(
            Indexer::new(&shape.sizes)
                .map(|index| {
                    let lhs_elem = lhs_broadcasted.idx(&index);
                    let rhs_elem = rhs_broadcasted.idx(&index);

                    f(lhs_elem, rhs_elem)
                })
                .collect(),
        );

        Ok(Tensor { data, shape })
    }

    pub fn zip_array<R>(&self, rhs: &[T], f: impl Fn(T, T) -> R) -> Result<Tensor<R>> {
        self.shape.valid_data_length(rhs.len())?;

        let data = Indexer::new(&self.shape.sizes)
            .zip(rhs)
            .map(|(index, &rhs_elem)| {
                let offset = self.shape.index(&index)?;
                let lhs_elem = self.data[offset];

                Ok(f(lhs_elem, rhs_elem))
            })
            .collect::<Result<Vec<R>>>()?;

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn reduce<R>(
        &self,
        dimensions: &[usize],
        f: impl Fn(&Tensor<T>) -> Result<R>,
        keepdims: bool,
    ) -> Result<Tensor<R>>
    where
        R: Copy,
    {
        self.shape.valid_dimensions(dimensions)?;

        let data = Slicer::new(&self.shape.sizes, dimensions, keepdims)
            .map(|index| f(&self.slicer(&index)?))
            .collect::<Result<Vec<R>>>()?;

        let sizes: Vec<usize> = self
            .shape
            .sizes
            .iter()
            .enumerate()
            .map(|(d, &size)| {
                if keepdims == dimensions.contains(&d) {
                    1
                } else {
                    size
                }
            })
            .collect();

        Ok(Tensor::init(data, &sizes))
    }

    pub fn index_map(&self, f: impl Fn(T) -> T, index: &[usize]) -> Result<Tensor<T>> {
        let mut data = self.data();
        let offset = self.shape.index(index)?;
        data[offset] = f(data[offset]);

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn index_map_dims(
        &self,
        f: impl Fn(T) -> T,
        dimensions: &[usize],
        index: &[usize],
    ) -> Result<Tensor<T>> {
        let mut data = self.data();
        let offset = self.shape.index_dims(dimensions, index)?;
        data[offset] = f(data[offset]);

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_map(&self, f: impl Fn(T) -> T, ranges: &[(usize, usize)]) -> Result<Tensor<T>> {
        let slice_shape = self.shape.slice(ranges)?;

        let mut data = self.data();
        for index in Indexer::new(&slice_shape.sizes) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset]);
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_map_dims(
        &self,
        f: impl Fn(T) -> T,
        dimensions: &[usize],
        ranges: &[(usize, usize)],
    ) -> Result<Tensor<T>> {
        let mut data = self.data();
        let slice_shape = self.shape.slice_dims(dimensions, ranges)?;

        for index in Indexer::new(&slice_shape.sizes) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset]);
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_zip(
        &self,
        rhs: &[T],
        f: impl Fn(T, T) -> T,
        ranges: &[(usize, usize)],
    ) -> Result<Tensor<T>> {
        let slice_shape = self.shape.slice(ranges)?;
        slice_shape.valid_data_length(rhs.len())?;

        let mut data = self.data();

        for (index, &rhs_value) in Indexer::new(&slice_shape.sizes).zip(rhs) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset], rhs_value);
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_zip_dims(
        &self,
        rhs: &[T],
        f: impl Fn(T, T) -> T,
        dimensions: &[usize],
        ranges: &[(usize, usize)],
    ) -> Result<Tensor<T>> {
        let slice_shape = self.shape.slice_dims(dimensions, ranges)?;
        slice_shape.valid_data_length(rhs.len())?;

        let mut data = self.data();

        for (index, &rhs_value) in Indexer::new(&slice_shape.sizes).zip(rhs) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset], rhs_value);
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }
}

impl<T> Tensor<T> {
    // --- Same Data, Modified Shape ---

    pub(crate) fn with_shape(&self, shape: Shape) -> Tensor<T> {
        Tensor {
            data: Arc::clone(&self.data),
            shape,
        }
    }

    pub fn view(&self, sizes: &[usize]) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.view(sizes)?))
    }

    pub fn ravel(&self) -> Result<Tensor<T>> {
        self.view(&[self.numel()])
    }

    pub fn squeeze(&self) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.squeeze()?))
    }

    pub fn unsqueeze(&self, unsqueezed: usize) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.unsqueeze(unsqueezed)?))
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.permute(permutation)?))
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.transpose(dim_1, dim_2)?))
    }

    pub fn expand(&self, expansions: &[usize]) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.expand(expansions)?))
    }

    pub fn flip(&self, flips: &[usize]) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.flip(flips)?))
    }

    pub fn flip_all(&self) -> Result<Tensor<T>> {
        self.flip(&Vec::from_iter(0..self.rank()))
    }

    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.slice(ranges)?))
    }

    pub fn slice_dims(&self, dimensions: &[usize], ranges: &[(usize, usize)]) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.slice_dims(dimensions, ranges)?))
    }

    pub(crate) fn slicer(&self, indices: &[Option<usize>]) -> Result<Tensor<T>> {
        Ok(self.with_shape(self.shape.slicer(indices)?))
    }

    // --- Attributes ---

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn sizes(&self) -> &[usize] {
        &self.shape.sizes
    }

    pub fn strides(&self) -> &[Stride] {
        &self.shape.strides
    }

    pub fn offset(&self) -> usize {
        self.shape.offset
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous()
    }
}

impl<T: Copy + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, rhs: &Tensor<T>) -> bool {
        self.data == rhs.data && self.shape == rhs.shape
    }
}
