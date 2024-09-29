use crate::{core::utils::Res, Tensor};
use std::ops::{Add, Div, Mul, Sub};

// --- Standard binary operations ---

macro_rules! binary_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.zip(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.zip(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<Tensor<T>> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.zip(&rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<&Tensor<T>> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.zip(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
            fn $method(self, rhs: T) -> Self::Output {
                self.binary_map(rhs, |l, r| l $op r)
            }
        }

        impl<T> $trait<T> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Res<Tensor<T>>;
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
        let exp = self.exp()?;
        &exp / exp.sum()?
    }
}
