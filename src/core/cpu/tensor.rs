use crate::{
    core::{
        indexer::IndexIterator,
        shape::{Shape, Stride},
        slicer::SliceIterator,
    },
    Res,
};
use std::{
    iter::{repeat, successors},
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
    // Init

    fn init(data: &Arc<Vec<T>>, sizes: &[usize], offset: usize) -> Res<Tensor<T>> {
        let data_len = data.len();
        let tensor_size = sizes.iter().product();

        if data_len == tensor_size {
            Ok(Tensor {
                data: Arc::clone(data),
                shape: Shape::new(sizes, offset),
            })
        } else {
            Err(format!(
                "Data length ({}) does not match size of tensor ({}).",
                data_len, tensor_size
            ))
        }
    }

    pub fn new(data: &[T], sizes: &[usize]) -> Res<Tensor<T>> {
        Tensor::init(&Arc::new(data.to_vec()), sizes, 0)
    }

    pub fn new_1d(data: &[T]) -> Res<Tensor<T>> {
        Tensor::new(data, &[data.len()])
    }

    pub fn scalar(data: T) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::new(vec![data]),
            shape: Shape::new(&[1], 0),
        })
    }

    pub fn same(element: T, sizes: &[usize]) -> Res<Tensor<T>> {
        let data = vec![element; sizes.iter().product()];
        Tensor::new(&data, sizes)
    }

    pub fn arange(start: T, end: T, step: T) -> Res<Tensor<T>>
    where
        T: std::ops::Add<Output = T> + PartialOrd,
    {
        let data: Vec<T> = successors(Some(start), |&prev| {
            let current = prev + step;
            if current < end {
                Some(current)
            } else {
                None
            }
        })
        .collect();

        Tensor::new(&data, &[data.len()])
    }

    pub fn linspace(start: T, end: T, num: u16) -> Res<Tensor<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::convert::From<u16>,
    {
        let step = (end - start) / T::from(num - 1);
        let data = (0..num)
            .map(|i| start + step * i.into())
            .collect::<Vec<T>>();

        Tensor::new(&data, &[num as usize])
    }

    pub fn to_contiguous(&self) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::new(self.data_non_contiguous()?),
            shape: Shape::new(&self.shape.sizes, 0),
        })
    }

    pub(crate) fn contiguous_if_not(self) -> Res<Tensor<T>> {
        if self.is_contiguous() {
            Ok(self)
        } else {
            self.to_contiguous()
        }
    }

    // Attributes

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous()
    }

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

    // Data

    pub fn data(&self) -> Res<Vec<T>> {
        if self.is_contiguous() {
            Ok(self.data_contiguous().to_vec())
        } else {
            self.data_non_contiguous()
        }
    }

    pub(crate) fn data_contiguous(&self) -> &[T] {
        let start = self.offset();
        let end = start + self.numel();
        &self.data[start..end]
    }

    pub(crate) fn data_non_contiguous(&self) -> Res<Vec<T>> {
        IndexIterator::new(&self.shape)
            .map(|index| self.element(&index))
            .collect()
    }

    pub fn element(&self, indices: &[usize]) -> Res<T> {
        self.data
            .get(self.shape.element(indices)?)
            .copied()
            .ok_or_else(|| format!("Index {:?} is out of bounds.", indices))
    }

    pub fn slice(&self, indices: &[(usize, usize)]) -> Res<Tensor<T>> {
        if self.is_contiguous() {
            Ok(Tensor {
                data: Arc::clone(&self.data),
                shape: self.shape.slice(indices)?,
            })
        } else {
            Err("Tensor is not contiguous. Use `to_contiguous()`.".to_string())
        }
    }

    pub(crate) fn single_slice(&self, indices: &[Option<usize>]) -> Res<Tensor<T>> {
        if self.is_contiguous() {
            Ok(Tensor {
                data: Arc::clone(&self.data),
                shape: self.shape.single_slice(indices)?,
            })
        } else {
            Err("Tensor is not contiguous. Use `to_contiguous()`.".to_string())
        }
    }

    // Shape

    pub fn reshape(&self, sizes: &[usize]) -> Res<Tensor<T>> {
        self.shape.valid_reshape(sizes)?;

        if self.is_contiguous() {
            self.view(sizes)
        } else {
            Ok(Tensor {
                data: Arc::new(self.data_non_contiguous()?),
                shape: Shape::new(sizes, 0),
            })
        }
    }

    pub fn view(&self, sizes: &[usize]) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.view(sizes)?,
        })
    }

    pub fn ravel(&self) -> Res<Tensor<T>> {
        self.reshape(&[self.numel()])
    }

    pub fn squeeze(&self) -> Res<Tensor<T>> {
        Ok(Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.squeeze()?,
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

    pub fn reshape_then_expand(&self, expansions: &[usize]) -> Res<Tensor<T>> {
        let resize_len = expansions.len();
        let numdims = self.ndims();

        if numdims == resize_len {
            self.expand(expansions)
        } else {
            let ones_len = expansions.len() - numdims;
            let mut sizes = self.sizes().to_vec();
            sizes.splice(..0, repeat(1).take(ones_len));

            self.reshape(&sizes)?.expand(expansions)
        }
    }

    // Maps

    pub fn unary_map<R>(&self, f: impl Fn(T) -> R) -> Tensor<R> {
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
                            let elem = self.element(&index).unwrap();
                            f(elem)
                        })
                        .collect(),
                ),
                Shape::new(self.sizes(), 0),
            )
        };

        Tensor { data, shape }
    }

    pub fn binary_scalar_map<R>(&self, rhs: T, f: impl Fn(T, T) -> R) -> Tensor<R> {
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
                            let lhs_elem = self.element(&index).unwrap();
                            f(lhs_elem, rhs)
                        })
                        .collect(),
                ),
                Shape::new(self.sizes(), 0),
            )
        };

        Tensor { data, shape }
    }

    pub fn binary_tensor_map<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Res<Tensor<R>> {
        if self.shape == rhs.shape {
            self.equal_shapes_map(rhs, f)
        } else {
            self.broadcasted_shapes_map(rhs, f)
        }
    }

    fn equal_shapes_map<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Res<Tensor<R>> {
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
                            let lhs_elem = self.element(&index)?;
                            let rhs_elem = rhs.element(&index)?;
                            Ok(f(lhs_elem, rhs_elem))
                        })
                        .collect::<Res<Vec<R>>>()?,
                ),
                Shape::new(self.sizes(), 0),
            )
        };

        Ok(Tensor { data, shape })
    }

    fn broadcasted_shapes_map<R>(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> R) -> Res<Tensor<R>> {
        let sizes = Shape::broadcast(&self.shape.sizes, &rhs.shape.sizes)?;
        let shape = Shape::new(&sizes, 0);

        let lhs_broadcasted = self.reshape_then_expand(&sizes)?;
        let rhs_broadcasted = rhs.reshape_then_expand(&sizes)?;

        let data = Arc::new(
            IndexIterator::new(&shape)
                .map(|index| {
                    let lhs_elem = lhs_broadcasted.element(&index)?;
                    let rhs_elem = rhs_broadcasted.element(&index)?;
                    Ok(f(lhs_elem, rhs_elem))
                })
                .collect::<Res<Vec<R>>>()?,
        );

        Ok(Tensor { data, shape })
    }

    pub fn reduce_map(&self, dimensions: &[usize], f: impl Fn(&Tensor<T>) -> T) -> Res<Tensor<T>> {
        self.shape.valid_dimensions(dimensions)?;

        let data = SliceIterator::new(&self.shape, dimensions)
            .map(|index| {
                let dimension_slice = self.single_slice(&index)?;
                Ok(f(&dimension_slice))
            })
            .collect::<Res<Vec<T>>>()?;

        let sizes: Vec<usize> = self
            .shape
            .sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| if dimensions.contains(&i) { size } else { 1 })
            .collect();

        Tensor::new(&data, &sizes)
    }
}
