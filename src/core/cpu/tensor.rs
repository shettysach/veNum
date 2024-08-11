use crate::core::{
    index::IndexIterator,
    shape::{Shape, Stride},
};
use std::{iter::successors, sync::Arc};

pub struct Tensor<T> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Shape,
}

impl<T> Tensor<T>
where
    T: Copy,
{
    // Init

    pub fn init(data: &Arc<Vec<T>>, sizes: &[usize], offset: usize) -> Tensor<T> {
        let data_len = data.len();
        let tensor_size = sizes.iter().product();

        if data_len != tensor_size {
            panic!("Data length ({data_len}) does not match size of tensor ({tensor_size}).")
        }

        Tensor {
            data: Arc::clone(data),
            shape: Shape::new(sizes, offset),
        }
    }

    pub fn new(data: &[T], sizes: &[usize]) -> Tensor<T> {
        Tensor::init(&Arc::new(data.to_vec()), sizes, 0)
    }

    pub fn new_1d(data: &[T]) -> Tensor<T> {
        Tensor::init(&Arc::new(data.to_vec()), &[data.len()], 0)
    }

    pub fn same(element: T, sizes: &[usize]) -> Tensor<T> {
        let data = vec![element; sizes.iter().product()];
        Tensor::new(&data, sizes)
    }

    pub fn arange(start: T, end: T, step: T) -> Tensor<T>
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

    pub fn linspace(start: T, end: T, num: u16) -> Tensor<T>
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

    // Attributes

    pub fn data(&self) -> Vec<T> {
        self.data.to_vec()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn numdims(&self) -> usize {
        self.shape.numdims()
    }

    pub fn sizes(&self) -> &Vec<usize> {
        &self.shape.sizes
    }

    pub(crate) fn strides(&self) -> &Vec<Stride> {
        &self.shape.strides
    }

    pub fn offset(&self) -> usize {
        self.shape.offset
    }

    // Elements

    pub fn element(&self, indices: &[usize]) -> T {
        self.data.get(self.shape.element(indices)).copied().unwrap()
    }

    pub fn slice(&self, indices: &[(usize, usize)]) -> Tensor<T> {
        let shape = self.shape.slice(indices);

        Tensor {
            data: Arc::clone(&self.data),
            shape,
        }
    }

    // Shape

    pub fn reshape(&self, sizes: &[usize]) -> Tensor<T> {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.reshape(sizes, self.offset()),
        }
    }

    pub fn ravel(&self) -> Tensor<T> {
        self.reshape(&[self.numel()])
    }

    pub fn squeeze(&self) -> Tensor<T> {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.squeeze(),
        }
    }

    pub fn permute(&self, permutation: &[usize]) -> Tensor<T> {
        Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.permute(permutation),
        }
    }

    pub fn expand(&self, expansions: &[usize]) -> Tensor<T> {
        let expanded_shape = self.shape.expand(expansions);

        Tensor {
            data: Arc::clone(&self.data),
            shape: expanded_shape,
        }
    }

    pub fn flip(&self, flips: &[usize]) -> Tensor<T> {
        let flipped_shape = self.shape.flip(flips);

        Tensor {
            data: Arc::clone(&self.data),
            shape: flipped_shape,
        }
    }

    // Maps

    pub fn unary_map<R>(&self, f: impl Fn(T) -> R) -> Tensor<R> {
        let data = self.data.iter().map(|l| f(*l)).collect::<Vec<R>>();

        Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        }
    }

    pub fn binary_scalar_map<R>(&self, rhs: T, f: impl Fn(T, T) -> R) -> Tensor<R> {
        let data = self.data.iter().map(|l| f(*l, rhs)).collect::<Vec<R>>();

        Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        }
    }

    pub fn binary_tensor_map(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> T) -> Tensor<T> {
        if self.shape == rhs.shape {
            self.equal_shapes_map(rhs, f)
        } else {
            self.broadcasted_shapes_map(rhs, f)
        }
    }

    fn equal_shapes_map(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> T) -> Tensor<T> {
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(l, r)| f(*l, *r))
            .collect();

        Tensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        }
    }

    fn broadcasted_shapes_map(&self, rhs: &Tensor<T>, f: impl Fn(T, T) -> T) -> Tensor<T> {
        let sizes = self.shape.broadcast(&rhs.shape);
        let shape = Shape::new(&sizes, 0);

        let lhs_broadcasted = self.reshape_then_expand(&sizes);
        let rhs_broadcasted = rhs.reshape_then_expand(&sizes);

        let result = IndexIterator::new(&shape)
            .map(|index| {
                let lhs_elem = lhs_broadcasted.element(&index);
                let rhs_elem = rhs_broadcasted.element(&index);
                f(lhs_elem, rhs_elem)
            })
            .collect();

        Tensor {
            data: Arc::new(result),
            shape,
        }
    }

    pub fn reshape_then_expand(&self, expansions: &[usize]) -> Tensor<T> {
        let resize_len = expansions.len();

        if self.numdims() == resize_len {
            self.expand(expansions)
        } else {
            let mut sizes = self.sizes().clone();
            sizes.reverse();
            sizes.resize(expansions.len(), 1);
            sizes.reverse();

            self.reshape(&sizes).expand(expansions)
        }
    }
}
