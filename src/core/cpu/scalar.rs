use crate::Tensor;
use std::ops::{Add, Div, Mul, Sub};

// -- Scalar struct to bypass orphan rule on T --

#[derive(Copy, Clone)]
pub struct Scalar<T>(pub(crate) T);

impl<T> Scalar<T>
where
    T: Copy,
{
    pub fn new(value: T) -> Scalar<T> {
        Scalar(value)
    }

    pub fn value(&self) -> T {
        self.0
    }
}

macro_rules! scalar_tensor_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait<Scalar<T>> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Scalar<T>) -> Self::Output {
                self.binary_scalar_map(rhs.0, |l, r| l $op r)
            }
        }

        impl<T> $trait<Scalar<T>> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Scalar<T>) -> Self::Output {
                self.binary_scalar_map(rhs.0, |l, r| l $op r)
            }
        }

        impl<T> $trait<Tensor<T>> for Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                rhs.binary_scalar_map(self.0, |r, l| l $op r)
            }
        }

        impl<T> $trait<&Tensor<T>> for Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                rhs.binary_scalar_map(self.0, |r, l| l $op r)
            }
        }

        impl<T> $trait<&Scalar<T>> for Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Scalar<T>) -> Self::Output {
                self.binary_scalar_map(rhs.0, |l, r| l $op r)
            }
        }

        impl<T> $trait<&Scalar<T>> for &Tensor<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Scalar<T>) -> Self::Output {
                self.binary_scalar_map(rhs.0, |l, r| l $op r)
            }
        }

        impl<T> $trait<Tensor<T>> for &Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                rhs.binary_scalar_map(self.0, |r, l| l $op r)
            }
        }

        impl<T> $trait<&Tensor<T>> for &Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                rhs.binary_scalar_map(self.0, |r, l| l $op r)
            }
        }
    };
}

scalar_tensor_ops!(Add, add, +);
scalar_tensor_ops!(Sub, sub, -);
scalar_tensor_ops!(Mul, mul, *);
scalar_tensor_ops!(Div, div, /);

macro_rules! binary_scalar_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait<Scalar<T>> for Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Scalar<T>;
            fn $method(self, rhs: Scalar<T>) -> Self::Output {
                Scalar(self.0 $op rhs.0)
            }
        }

        impl<T> $trait<&Scalar<T>> for Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Scalar<T>;
            fn $method(self, rhs: &Scalar<T>) -> Self::Output {
                Scalar(self.0 $op rhs.0)
            }
        }

        impl<T> $trait<Scalar<T>> for &Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Scalar<T>;
            fn $method(self, rhs: Scalar<T>) -> Self::Output {
                Scalar(self.0 $op rhs.0)
            }
        }

        impl<T> $trait<&Scalar<T>> for &Scalar<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Scalar<T>;
            fn $method(self, rhs: &Scalar<T>) -> Self::Output {
                Scalar(self.0 $op rhs.0)
            }
        }
    };
}

binary_scalar_ops!(Add, add, +);
binary_scalar_ops!(Sub, sub, -);
binary_scalar_ops!(Mul, mul, *);
binary_scalar_ops!(Div, div, /);
