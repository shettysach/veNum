use crate::{core::index::IndexIterator, Tensor};
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
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.binary_tensor_map(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.binary_tensor_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<Tensor<T>> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.binary_tensor_map(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<&Tensor<T>> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.binary_tensor_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: T) -> Self::Output {
                self.binary_scalar_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
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

// Sum and product

impl<T> Tensor<T>
where
    T: Copy + Sum<T> + Product<T>,
{
    pub fn sum_all(&self) -> T {
        IndexIterator::new(&self.shape)
            .map(|index| self.element(&index))
            .sum()
    }

    pub fn product_all(&self) -> T {
        IndexIterator::new(&self.shape)
            .map(|index| self.element(&index))
            .product()
    }
}

// Unary operations for floats

impl Tensor<f32> {
    pub fn ln(&self) -> Tensor<f32> {
        self.unary_map(|elem| elem.ln())
    }

    pub fn exp(&self) -> Tensor<f32> {
        self.unary_map(|elem| elem.exp())
    }
}

impl Tensor<f64> {
    pub fn ln(&self) -> Tensor<f64> {
        self.unary_map(|elem| elem.ln())
    }

    pub fn exp(&self) -> Tensor<f64> {
        self.unary_map(|elem| elem.exp())
    }
}
