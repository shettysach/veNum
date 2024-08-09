use crate::core::index::IndexIterator;
use crate::core::shape::Shape;
use std::sync::Arc;

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

        if data_len == tensor_size {
            Tensor {
                data: Arc::clone(data),
                shape: Shape::new(sizes, offset),
            }
        } else {
            panic!("Data length ({data_len}) does not match size of tensor ({tensor_size}).")
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

    pub fn linspace(start: T, end: T, num: u16) -> Tensor<T>
    where
        T: std::ops::Add<Output = T>,
        T: std::ops::Sub<Output = T>,
        T: std::ops::Mul<Output = T>,
        T: std::ops::Div<Output = T>,
        T: std::convert::From<u16>,
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

    pub fn strides(&self) -> &Vec<usize> {
        &self.shape.strides
    }

    pub fn offset(&self) -> usize {
        self.shape.offset
    }

    // Elements

    pub fn index_element(&self, indices: &[usize]) -> T {
        self.data.get(self.shape.element(indices)).copied().unwrap()
    }

    pub fn index_slice(&self, indices: &[(usize, usize)]) -> Tensor<T> {
        let shape = self.shape.slice(indices);

        Tensor {
            data: Arc::clone(&self.data),
            shape,
        }
    }

    // Shape

    pub fn reshape(&self, sizes: &[usize]) -> Tensor<T> {
        let data_len = self.shape.numel();
        let tensor_size = sizes.iter().product();

        if data_len == tensor_size {
            Tensor {
                data: Arc::clone(&self.data),
                shape: Shape::new(sizes, self.offset()),
            }
        } else {
            panic!("Data length ({data_len}) does not match size of tensor ({tensor_size}).")
        }
    }

    pub fn ravel(&self) -> Tensor<T> {
        self.reshape(&[self.numel()])
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
                let lhs_elem = lhs_broadcasted.index_element(&index);
                let rhs_elem = rhs_broadcasted.index_element(&index);
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
