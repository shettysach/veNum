use crate::{
    core::{
        cpu::one::One,
        indexer::IndexIterator,
        shape::{Shape, Stride},
        slicer::SliceIterator,
    },
    Res,
};
use std::{
    iter::successors,
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

pub struct Tensor<T> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Shape,
}

impl<T> Tensor<T>
where
    T: Copy,
{
    // --- Init ---

    pub fn new(data: &[T], sizes: &[usize]) -> Res<Tensor<T>> {
        let data_length = data.len();
        let tensor_size = sizes.iter().product();

        if data_length != tensor_size {
            return Err(format!(
                "Data length ({}) does not match size of tensor ({}).",
                data_length, tensor_size
            ));
        }

        let arc_data = Arc::new(data.to_vec());

        Ok(Tensor {
            data: Arc::clone(&arc_data),
            shape: Shape::new(sizes),
        })
    }

    pub fn new_1d(data: &[T]) -> Res<Tensor<T>> {
        Tensor::new(data, &[data.len()])
    }

    pub fn scalar(data: T) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::new(vec![data]),
            shape: Shape {
                sizes: vec![1],
                strides: vec![Stride::Positive(1)],
                offset: 0,
            },
        })
    }

    pub fn same(element: T, sizes: &[usize]) -> Res<Tensor<T>> {
        let data = vec![element; sizes.iter().product()];
        Tensor::new(&data, sizes)
    }

    pub fn ones(sizes: &[usize]) -> Res<Tensor<T>>
    where
        T: One,
    {
        Tensor::same(T::one(), sizes)
    }

    pub fn zeroes(sizes: &[usize]) -> Res<Tensor<T>>
    where
        T: Default,
    {
        Tensor::same(T::default(), sizes)
    }

    pub fn arange(start: T, end: T, step: T) -> Res<Tensor<T>>
    where
        T: Add<Output = T> + PartialOrd,
    {
        let data: Vec<T> = successors(Some(start), |&prev| {
            let current = prev + step;
            (current < end).then_some(current)
        })
        .collect();

        Tensor::new(&data, &[data.len()])
    }

    // TODO: Better method ?
    pub fn linspace(start: T, end: T, num: u16) -> Res<Tensor<T>>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<u16>,
    {
        let step = (end - start) / T::from(num - 1);
        let data = (0..num)
            .map(|i| start + step * i.into())
            .collect::<Vec<T>>();

        Tensor::new(&data, &[num as usize])
    }

    pub fn eye(size: usize) -> Res<Tensor<T>>
    where
        T: Default + One,
    {
        let diagonal = size + 1;
        let data = (0..size * size)
            .map(|elem| {
                if elem % diagonal == 0 {
                    T::one()
                } else {
                    T::default()
                }
            })
            .collect::<Vec<T>>();

        Tensor::new(&data, &[size, size])
    }

    pub fn to_contiguous(&self) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::new(self.data_non_contiguous()),
            shape: Shape::new(&self.shape.sizes),
        })
    }

    pub(crate) fn into_contiguous(self) -> Res<Tensor<T>> {
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

    pub fn data_contiguous(&self) -> &[T] {
        let start = self.offset();
        let end = start + self.numel();
        &self.data[start..end]
    }

    pub(crate) fn data_non_contiguous(&self) -> Vec<T> {
        IndexIterator::new(&self.shape)
            .map(|index| self.index(&index).unwrap())
            .collect()
    }

    pub fn index(&self, indices: &[usize]) -> Res<T> {
        Ok(self.data[self.shape.index(indices)?])
    }

    pub fn index_dims(&self, dimensions: &[usize], indices: &[usize]) -> Res<T> {
        Ok(self.data[self.shape.index_dims(dimensions, indices)?])
    }

    pub fn slice(&self, ranges: &[(usize, usize)]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.slice(ranges)?,
        })
    }

    pub fn slice_dims(&self, dimensions: &[usize], ranges: &[(usize, usize)]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.slice_dims(dimensions, ranges)?,
        })
    }

    pub fn index_assign(&self, new_value: T, index: &[usize]) -> Res<Tensor<T>> {
        let mut data = self.data();
        let offset = self.shape.index(index)?;
        data[offset] = new_value;

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn index_assign_dims(
        &self,
        new_value: T,
        dimensions: &[usize],
        index: &[usize],
    ) -> Res<Tensor<T>> {
        let mut data = self.data();
        let offset = self.shape.index_dims(dimensions, index)?;
        data[offset] = new_value;

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_assign(&self, new_data: &[T], ranges: &[(usize, usize)]) -> Res<Tensor<T>> {
        let slice_shape = self.shape.slice(ranges)?;
        slice_shape.valid_data_length(new_data)?;

        let mut data = self.data();
        for (index, &new_value) in IndexIterator::new(&slice_shape).zip(new_data) {
            let offset = slice_shape.index(&index)?;
            data[offset] = new_value;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_assign_dims(
        &self,
        new_data: &[T],
        dimensions: &[usize],
        ranges: &[(usize, usize)],
    ) -> Res<Tensor<T>> {
        let slice_shape = self.shape.slice_dims(dimensions, ranges)?;
        slice_shape.valid_data_length(new_data)?;

        let mut data = self.data();
        for (index, &new_value) in IndexIterator::new(&slice_shape).zip(new_data) {
            let offset = slice_shape.index(&index)?;
            data[offset] = new_value;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn pad(&self, constant: T, padding: &[(usize, usize)]) -> Res<Tensor<T>> {
        let shape = self.shape.pad(padding)?;
        let data = Arc::new(vec![constant; shape.numel()]);
        let tensor = Tensor { data, shape };

        let ranges = padding
            .iter()
            .enumerate()
            .map(|(dimension, &(start, _))| (start, self.shape.sizes[dimension] + start))
            .collect::<Vec<(usize, usize)>>();

        tensor.slice_assign(&self.data(), &ranges)
    }

    pub fn pad_dims(
        &self,
        constant: T,
        dimensions: &[usize],
        padding: &[(usize, usize)],
    ) -> Res<Tensor<T>> {
        let shape = self.shape.pad_dims(padding, dimensions)?;
        let data = Arc::new(vec![constant; shape.numel()]);
        let tensor = Tensor { data, shape };

        let ranges = dimensions
            .iter()
            .zip(padding)
            .map(|(&dimension, &(start, _))| (start, self.shape.sizes[dimension] + start))
            .collect::<Vec<(usize, usize)>>();

        tensor.slice_assign_dims(&self.data(), dimensions, &ranges)
    }

    pub(crate) fn slicer(&self, indices: &[Option<usize>]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.slicer(indices)?,
        })
    }

    // --- Reshape ---

    pub fn view(&self, sizes: &[usize]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.view(sizes)?,
        })
    }

    pub fn ravel(&self) -> Res<Tensor<T>> {
        self.view(&[self.numel()])
    }

    pub fn reshape(&self, sizes: &[usize]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::new(self.data_non_contiguous()),
            shape: Shape::new(sizes),
        })
    }

    pub fn flatten(&self) -> Res<Tensor<T>> {
        self.reshape(&[self.numel()])
    }

    pub fn view_else_reshape(&self, sizes: &[usize]) -> Res<Tensor<T>> {
        self.view(sizes).or_else(|_| self.reshape(sizes))
    }

    // --- Maps, Zips and Reduce ---

    pub fn unary_map<R>(&self, f: impl Fn(T) -> R) -> Res<Tensor<R>> {
        let (data, shape) = if self.is_contiguous() {
            (
                Arc::new(self.data_contiguous().iter().map(|&elem| f(elem)).collect()),
                Shape {
                    sizes: self.sizes().to_vec(),
                    strides: self.strides().to_vec(),
                    offset: 0,
                },
            )
        } else {
            (
                Arc::new(
                    IndexIterator::new(&self.shape)
                        .map(|index| {
                            let elem = self.index(&index)?;
                            Ok(f(elem))
                        })
                        .collect::<Res<Vec<R>>>()?,
                ),
                Shape::new(self.sizes()),
            )
        };

        Ok(Tensor { data, shape })
    }

    pub fn binary_map<R>(&self, rhs: T, f: impl Fn(T, T) -> R) -> Res<Tensor<R>> {
        let (data, shape) = if self.is_contiguous() {
            (
                Arc::new(
                    self.data_contiguous()
                        .iter()
                        .map(|&elem| f(elem, rhs))
                        .collect(),
                ),
                Shape {
                    sizes: self.sizes().to_vec(),
                    strides: self.strides().to_vec(),
                    offset: 0,
                },
            )
        } else {
            (
                Arc::new(
                    IndexIterator::new(&self.shape)
                        .map(|index| {
                            let lhs_elem = self.index(&index)?;
                            Ok(f(lhs_elem, rhs))
                        })
                        .collect::<Res<Vec<R>>>()?,
                ),
                Shape::new(self.sizes()),
            )
        };

        Ok(Tensor { data, shape })
    }

    pub fn zip<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Res<Tensor<R>> {
        if self.shape == rhs.shape {
            self.equal_zip(rhs, f)
        } else {
            self.broadcast_zip(rhs, f)
        }
    }

    fn equal_zip<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Res<Tensor<R>> {
        let (data, shape) = if self.is_contiguous() && rhs.is_contiguous() {
            (
                Arc::new(
                    self.data_contiguous()
                        .iter()
                        .zip(rhs.data_contiguous())
                        .map(|(&lhs_elem, &rhs_elem)| f(lhs_elem, rhs_elem))
                        .collect(),
                ),
                Shape {
                    sizes: self.sizes().to_vec(),
                    strides: self.strides().to_vec(),
                    offset: 0,
                },
            )
        } else {
            (
                Arc::new(
                    IndexIterator::new(&self.shape)
                        .map(|index| {
                            let lhs_elem = self.index(&index)?;
                            let rhs_elem = rhs.index(&index)?;
                            Ok(f(lhs_elem, rhs_elem))
                        })
                        .collect::<Res<Vec<R>>>()?,
                ),
                Shape::new(self.sizes()),
            )
        };

        Ok(Tensor { data, shape })
    }

    fn broadcast_zip<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Res<Tensor<R>> {
        let sizes = Shape::broadcast(&self.shape.sizes, &rhs.shape.sizes)?;
        let shape = Shape::new(&sizes);
        let expansion = sizes.len();

        let lhs_broadcasted = self.unsqueeze(expansion)?.expand(&sizes)?;
        let rhs_broadcasted = rhs.unsqueeze(expansion)?.expand(&sizes)?;

        let data = Arc::new(
            IndexIterator::new(&shape)
                .map(|index| {
                    let lhs_elem = lhs_broadcasted.index(&index)?;
                    let rhs_elem = rhs_broadcasted.index(&index)?;
                    Ok(f(lhs_elem, rhs_elem))
                })
                .collect::<Res<Vec<R>>>()?,
        );

        Ok(Tensor { data, shape })
    }

    pub fn reduce<R>(
        &self,
        dimensions: &[usize],
        f: impl Fn(&Tensor<T>) -> Res<R>,
        keepdims: bool,
    ) -> Res<Tensor<R>>
    where
        R: Copy,
    {
        self.shape.valid_dimensions(dimensions)?;

        let data = SliceIterator::new(&self.shape, dimensions, keepdims)
            .map(|index| {
                let dimension_slice = self.slicer(&index)?;
                f(&dimension_slice)
            })
            .collect::<Res<Vec<R>>>()?;

        let sizes: Vec<usize> = self
            .shape
            .sizes
            .iter()
            .enumerate()
            .map(|(d, &size)| match (keepdims, dimensions.contains(&d)) {
                (true, true) => 1,
                (true, false) => size,
                (false, true) => size,
                (false, false) => 1,
            })
            .collect();

        Tensor::new(&data, &sizes)
    }
}

impl<T> Tensor<T> {
    // --- Attributes ---

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn ndims(&self) -> usize {
        self.shape.ndims()
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

    // --- Shape ---

    pub fn squeeze(&self) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.squeeze()?,
        })
    }

    pub fn unsqueeze(&self, expansion: usize) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.unsqueeze(expansion)?,
        })
    }

    pub fn permute(&self, permutation: &[usize]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.permute(permutation)?,
        })
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.transpose(dim_1, dim_2)?,
        })
    }

    pub fn expand(&self, expansions: &[usize]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.expand(expansions)?,
        })
    }

    pub fn flip(&self, flips: &[usize]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.flip(flips)?,
        })
    }

    pub fn flip_all(&self) -> Res<Tensor<T>> {
        self.flip(&Vec::from_iter(0..self.ndims()))
    }
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
        }
    }
}

impl<T> PartialEq for Tensor<T>
where
    T: Copy + PartialEq,
{
    fn eq(&self, rhs: &Tensor<T>) -> bool {
        self.data() == rhs.data() && self.shape == rhs.shape
    }
}
