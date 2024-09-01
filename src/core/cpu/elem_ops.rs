use crate::{core::indexer::IndexIterator, Res, Tensor};
use std::{
    iter::{Product, Sum},
    ops::{Add, Div, Mul, Sub},
};

// Standard binary operations

macro_rules! binary_tensor_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.binary_tensor_map(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.binary_tensor_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<Tensor<T>> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.binary_tensor_map(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<&Tensor<T>> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.binary_tensor_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: T) -> Self::Output {
                self.binary_scalar_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: T) -> Self::Output {
                self.binary_scalar_map(rhs, |l, r| l $op r)
            }
        }
    };
}

binary_tensor_ops!(Add, add, +);
binary_tensor_ops!(Sub, sub, -);
binary_tensor_ops!(Mul, mul, *);
binary_tensor_ops!(Div, div, /);

// Reduction

impl<T> Tensor<T>
where
    T: Copy,
{
    pub fn sum(&self) -> Res<T>
    where
        T: Sum<T>,
    {
        let sum = if self.is_contiguous() {
            self.data_contiguous().iter().copied().sum()
        } else {
            IndexIterator::new(&self.shape)
                .map(|index| self.element(&index).unwrap())
                .sum()
        };

        Ok(sum)
    }

    pub fn product(&self) -> Res<T>
    where
        T: Product<T>,
    {
        let product = if self.is_contiguous() {
            self.data_contiguous().iter().copied().product()
        } else {
            IndexIterator::new(&self.shape)
                .map(|index| self.element(&index).unwrap())
                .product()
        };

        Ok(product)
    }

    pub fn max(&self) -> Res<T>
    where
        T: Ord,
    {
        let max = if self.is_contiguous() {
            self.data_contiguous().iter().copied().max()
        } else {
            IndexIterator::new(&self.shape)
                .map(|index| self.element(&index).unwrap())
                .max()
        };

        max.ok_or("Empty tensor. No max.".to_string())
    }

    pub fn min(&self) -> Res<T>
    where
        T: Ord,
    {
        let min = if self.is_contiguous() {
            self.data_contiguous().iter().copied().min()
        } else {
            IndexIterator::new(&self.shape)
                .map(|index| self.element(&index).unwrap())
                .min()
        };

        min.ok_or("Empty tensor. No min.".to_string())
    }

    pub fn sum_dimensions(&self, dimensions: &[usize], keepdims: bool) -> Res<Tensor<T>>
    where
        T: Sum<T>,
    {
        self.reduce_map(dimensions, Tensor::sum, keepdims)
    }

    pub fn product_dimensions(&self, dimensions: &[usize], keepdims: bool) -> Res<Tensor<T>>
    where
        T: Product<T>,
    {
        self.reduce_map(dimensions, Tensor::product, keepdims)
    }

    pub fn max_dimensions(&self, dimensions: &[usize], keepdims: bool) -> Res<Tensor<T>>
    where
        T: Ord,
    {
        self.reduce_map(dimensions, Tensor::max, keepdims)
    }

    pub fn min_dimensions(&self, dimensions: &[usize], keepdims: bool) -> Res<Tensor<T>>
    where
        T: Ord,
    {
        self.reduce_map(dimensions, Tensor::min, keepdims)
    }
}

// Operations for floats

impl Tensor<f32> {
    pub fn ln(&self) -> Res<Tensor<f32>> {
        self.unary_map(|elem| elem.ln())
    }

    pub fn exp(&self) -> Res<Tensor<f32>> {
        self.unary_map(|elem| elem.exp())
    }

    pub fn powi(&self, rhs: i32) -> Res<Tensor<f32>> {
        self.unary_map(|elem| elem.powi(rhs))
    }

    pub fn powf(&self, rhs: f32) -> Res<Tensor<f32>> {
        self.unary_map(|elem| elem.powf(rhs))
    }

    pub fn sqrt(&self) -> Res<Tensor<f32>> {
        self.unary_map(|elem| elem.sqrt())
    }

    pub fn softmax(&self) -> Res<Tensor<f32>> {
        let exp = &self.exp()?;
        exp / exp.sum()?
    }
}

impl Tensor<f64> {
    pub fn ln(&self) -> Res<Tensor<f64>> {
        self.unary_map(|elem| elem.ln())
    }

    pub fn exp(&self) -> Res<Tensor<f64>> {
        self.unary_map(|elem| elem.exp())
    }

    pub fn powi(&self, rhs: i32) -> Res<Tensor<f64>> {
        self.unary_map(|elem| elem.powi(rhs))
    }

    pub fn powf(&self, rhs: f64) -> Res<Tensor<f64>> {
        self.unary_map(|elem| elem.powf(rhs))
    }

    pub fn sqrt(&self) -> Res<Tensor<f64>> {
        self.unary_map(|elem| elem.sqrt())
    }

    pub fn softmax(&self) -> Res<Tensor<f64>> {
        let exp = &self.exp()?;
        exp / exp.sum()?
    }
}
