use anyhow::Result;
use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Sub},
};

use crate::Tensor;

// --- Standard binary operations ---

macro_rules! binary_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Result<Tensor<T>>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.zip(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Result<Tensor<T>>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.zip(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<Tensor<T>> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Result<Tensor<T>>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.zip(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<&Tensor<T>> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Result<Tensor<T>>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.zip(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Result<Tensor<T>>;
            fn $method(self, rhs: T) -> Self::Output {
                self.binary_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Result<Tensor<T>>;
            fn $method(self, rhs: T) -> Self::Output {
                self.binary_map(rhs, |l, r| l $op r)
            }
        }
    };
}

binary_ops!(Add, add, +);
binary_ops!(Sub, sub, -);
binary_ops!(Mul, mul, *);
binary_ops!(Div, div, /);

// --- Operations for floats ---

impl<F> Tensor<F>
where
    F: num_traits::Float,
{
    pub fn ln(&self) -> Result<Tensor<F>> {
        self.unary_map(|elem| elem.ln())
    }

    pub fn exp(&self) -> Result<Tensor<F>> {
        self.unary_map(|elem| elem.exp())
    }

    pub fn powi(&self, rhs: i32) -> Result<Tensor<F>> {
        self.unary_map(|elem| elem.powi(rhs))
    }

    pub fn powf(&self, rhs: F) -> Result<Tensor<F>> {
        self.unary_map(|elem| elem.powf(rhs))
    }

    pub fn sqrt(&self) -> Result<Tensor<F>> {
        self.unary_map(|elem| elem.sqrt())
    }

    pub fn softmax(&self) -> Result<Tensor<F>>
    where
        F: Sum,
    {
        let exp = &self.exp()?;
        exp / exp.sum()?
    }
}
