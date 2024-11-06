use crate::{
    core::{errors::EmptyTensorError, iters::Indexer, utils::cast_usize},
    Tensor,
};
use anyhow::Result;
use num_traits::FromPrimitive;
use std::{
    iter::{Product, Sum},
    ops::Div,
};

impl<T> Tensor<T>
where
    T: Copy,
{
    pub fn sum(&self) -> Result<T>
    where
        T: Sum<T>,
    {
        let sum = if self.is_contiguous() {
            self.data_contiguous().iter().copied().sum()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .sum()
        };

        Ok(sum)
    }

    pub fn mean(&self) -> Result<T>
    where
        T: Sum<T> + Div<T, Output = T> + FromPrimitive,
    {
        let numel = self.numel();
        let numel_casted = cast_usize(numel)?;

        Ok(self.sum()? / numel_casted)
    }

    pub fn product(&self) -> Result<T>
    where
        T: Product<T>,
    {
        let product = if self.is_contiguous() {
            self.data_contiguous().iter().copied().product()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .product()
        };

        Ok(product)
    }

    pub fn max(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        let max = if self.is_contiguous() {
            self.data_contiguous()
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
        };

        max.ok_or(EmptyTensorError::ReduceMax.into())
    }

    pub fn min(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        let min = if self.is_contiguous() {
            self.data_contiguous()
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        };

        min.ok_or(EmptyTensorError::ReduceMin.into())
    }

    pub fn sum_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<Tensor<T>>
    where
        T: Sum<T>,
    {
        self.reduce(dimensions, Tensor::sum, keepdims)
    }

    pub fn mean_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<Tensor<T>>
    where
        T: Sum<T> + Div<T, Output = T> + FromPrimitive,
    {
        self.reduce(dimensions, Tensor::mean, keepdims)
    }

    pub fn product_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<Tensor<T>>
    where
        T: Product<T>,
    {
        self.reduce(dimensions, Tensor::product, keepdims)
    }

    pub fn max_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<Tensor<T>>
    where
        T: PartialOrd,
    {
        self.reduce(dimensions, Tensor::max, keepdims)
    }

    pub fn min_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<Tensor<T>>
    where
        T: PartialOrd,
    {
        self.reduce(dimensions, Tensor::min, keepdims)
    }
}
