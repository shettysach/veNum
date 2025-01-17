use crate::{conv::Mode, core::iters::Strider, Tensor};
use anyhow::Result;
use num_traits::FromPrimitive;
use std::{
    iter::{Product, Sum},
    ops::Div,
};

impl<T> Tensor<T>
where
    T: Copy + Product<T> + Sum<T> + PartialOrd + Default,
{
    pub fn pool_1d(
        &self,
        f: impl Fn(&Tensor<T>, &[usize], bool) -> Result<Tensor<T>>,
        pool_sizes: &[usize; 1],
        strides: &[usize; 1],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        let i_first = self.rank() - 1;
        let input_width = self.shape.sizes[i_first];
        let input_sizes = &[input_width];

        let range_fn = mode.range_fn(input_sizes, pool_sizes)?;
        let output_sizes = mode.output_sizes(input_sizes, pool_sizes, strides);
        let output_width = output_sizes[0];

        let sizes = [&self.shape.sizes[..i_first], &output_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for iter_index in Strider::new(&output_sizes, strides) {
            let (input_ranges, _) = range_fn(input_sizes, pool_sizes, &iter_index);

            let input_slice = match input_ranges {
                Some(input_ranges) => &self.slice_dims(&[i_first], &input_ranges)?,
                None => self,
            };
            let aggregate = f(input_slice, &[i_first], keepdims)?;

            for (index, &value) in aggregate.data_contiguous().iter().enumerate() {
                let offset = (index * output_width) + index;
                data[offset] = value
            }
        }

        Ok(Tensor::init(data, &sizes))
    }

    pub fn pool_2d(
        &self,
        f: impl Fn(&Tensor<T>, &[usize], bool) -> Result<Tensor<T>>,
        pool_sizes: &[usize; 2],
        strides: &[usize; 2],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        let n = self.rank();
        let input_dims = &[n - 2, n - 1];
        let input_sizes = &[
            self.shape.sizes[input_dims[0]],
            self.shape.sizes[input_dims[1]],
        ];

        let output_sizes = mode.output_sizes(input_sizes, pool_sizes, strides);
        let output_product = output_sizes.iter().product::<usize>();
        let range_fn = mode.range_fn(input_sizes, pool_sizes)?;

        let sizes = [&self.shape.sizes[..input_dims[0]], &output_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for iter_index in Strider::new(&output_sizes, strides) {
            let (input_ranges, _) = range_fn(input_sizes, pool_sizes, &iter_index);

            let input_slice = match input_ranges {
                Some(input_ranges) => &self.slice_dims(input_dims, &input_ranges)?,
                None => self,
            };
            let reduction = f(input_slice, input_dims, keepdims)?;

            for (index, &value) in reduction.data_contiguous().iter().enumerate() {
                let offset = (index * output_product)
                    + (iter_index[0] / strides[0] * output_sizes[1])
                    + iter_index[1] / strides[1];

                data[offset] = value
            }
        }

        Ok(Tensor::init(data, &sizes))
    }

    // 1D

    pub fn max_pool_1d(
        self,
        pool_sizes: &[usize; 1],
        strides: &[usize; 1],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        self.pool_1d(Tensor::max_dims, pool_sizes, strides, mode, keepdims)
    }

    pub fn min_pool_1d(
        self,
        pool_sizes: &[usize; 1],
        strides: &[usize; 1],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        self.pool_1d(Tensor::min_dims, pool_sizes, strides, mode, keepdims)
    }

    pub fn sum_pool_1d(
        self,
        pool_sizes: &[usize; 1],
        strides: &[usize; 1],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        self.pool_1d(Tensor::sum_dims, pool_sizes, strides, mode, keepdims)
    }

    pub fn mean_pool_1d(
        self,
        pool_sizes: &[usize; 1],
        strides: &[usize; 1],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>>
    where
        T: Div<Output = T> + FromPrimitive,
    {
        self.pool_1d(Tensor::mean_dims, pool_sizes, strides, mode, keepdims)
    }

    // 2D

    pub fn max_pool_2d(
        self,
        pool_sizes: &[usize; 2],
        strides: &[usize; 2],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        self.pool_2d(Tensor::max_dims, pool_sizes, strides, mode, keepdims)
    }

    pub fn min_pool_2d(
        self,
        pool_sizes: &[usize; 2],
        strides: &[usize; 2],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        self.pool_2d(Tensor::min_dims, pool_sizes, strides, mode, keepdims)
    }

    pub fn sum_pool_2d(
        self,
        pool_sizes: &[usize; 2],
        strides: &[usize; 2],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>> {
        self.pool_2d(Tensor::sum_dims, pool_sizes, strides, mode, keepdims)
    }

    pub fn mean_pool_2d(
        self,
        pool_sizes: &[usize; 2],
        strides: &[usize; 2],
        mode: Mode,
        keepdims: bool,
    ) -> Result<Tensor<T>>
    where
        T: Div<Output = T> + FromPrimitive,
    {
        self.pool_2d(Tensor::mean_dims, pool_sizes, strides, mode, keepdims)
    }
}
