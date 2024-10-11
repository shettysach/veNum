use crate::{
    core::{iters::Strider, shape::Shape},
    Tensor,
};
use anyhow::Result;
use std::{iter::Sum, ops::Mul};

pub enum Mode {
    Valid,
    Full,
    Same,
}

impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + Default,
{
    pub fn correlate_1d(
        &self,
        kernel: &Tensor<T>,
        strides: &[usize; 1],
        mode: Mode,
    ) -> Result<Tensor<T>> {
        let i_first = self.rank() - 1;
        let input_width = self.shape.sizes[i_first];
        let input_conv_sizes = &[input_width];

        let k_first = kernel.rank() - 1;
        let kernel_width = kernel.shape.sizes[k_first];
        let kernel_conv_sizes = &[kernel_width];

        let prod_sum_fn = mode.product_sum_fn(input_conv_sizes, kernel_conv_sizes)?;

        let output_sizes = mode.output_sizes(input_conv_sizes, kernel_conv_sizes, strides);
        let output_width = output_sizes[0];

        let sizes = [&self.shape.sizes[..i_first], &output_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for iter_index in (0..output_width).step_by(strides[0]) {
            let product_sum = prod_sum_fn(
                (self, kernel),
                (input_conv_sizes, kernel_conv_sizes),
                (&[i_first], &[k_first]),
                &[iter_index],
            )?;

            for (index, &value) in product_sum.data_contiguous().iter().enumerate() {
                let offset = (index * output_width) + index;
                data[offset] = value
            }
        }

        Ok(Tensor::init(data, &sizes))
    }

    pub fn correlate_2d(
        &self,
        kernel: &Tensor<T>,
        strides: &[usize; 2],
        mode: Mode,
    ) -> Result<Tensor<T>> {
        let n = self.rank();
        let input_conv_dims = &[n - 2, n - 1];
        let input_conv_sizes = &[
            self.shape.sizes[input_conv_dims[0]],
            self.shape.sizes[input_conv_dims[1]],
        ];

        let kn = kernel.rank();
        let kernel_conv_dims = &[kn - 2, kn - 1];
        let kernel_conv_sizes = &[
            kernel.shape.sizes[kernel_conv_dims[0]],
            kernel.shape.sizes[kernel_conv_dims[1]],
        ];

        let prod_sum_fn = mode.product_sum_fn(input_conv_sizes, kernel_conv_sizes)?;

        let output_sizes = mode.output_sizes(input_conv_sizes, kernel_conv_sizes, strides);
        let output_conv_sizes = &[output_sizes[0], output_sizes[1]];
        let output_conv_product = output_conv_sizes.iter().product::<usize>();

        let sizes = [&self.shape.sizes[..input_conv_dims[0]], output_conv_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for iter_index in Strider::new(output_conv_sizes, strides) {
            let product_sum = prod_sum_fn(
                (self, kernel),
                (input_conv_sizes, kernel_conv_sizes),
                (input_conv_dims, kernel_conv_dims),
                &iter_index,
            )?;

            // Pointer arithmetic
            for (index, &value) in product_sum.data_contiguous().iter().enumerate() {
                let offset = (index * output_conv_product)
                    + (iter_index[0] / strides[0] * output_conv_sizes[1])
                    + iter_index[1] / strides[1];

                data[offset] = value
            }
        }

        Ok(Tensor::init(data, &sizes))
    }

    pub fn convolve_1d(
        &self,
        kernel: &Tensor<T>,
        strides: &[usize; 1],
        mode: Mode,
    ) -> Result<Tensor<T>> {
        self.correlate_1d(&kernel.flip_all()?, strides, mode)
    }

    pub fn convolve_2d(
        &self,
        kernel: &Tensor<T>,
        strides: &[usize; 2],
        mode: Mode,
    ) -> Result<Tensor<T>> {
        self.correlate_2d(&kernel.flip_all()?, strides, mode)
    }
}

pub type ProductSumFn<T> = fn(
    (&Tensor<T>, &Tensor<T>),
    (&[usize], &[usize]),
    (&[usize], &[usize]),
    &[usize],
) -> Result<Tensor<T>>;

impl Mode {
    fn output_sizes(
        &self,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
        strides: &[usize],
    ) -> Vec<usize> {
        let output_size_fn: fn(usize, usize, usize) -> usize = match self {
            Mode::Valid => |i, k, s| i.abs_diff(k) / s + 1,
            Mode::Full => |i, k, s| (i + k) / s - 1,
            Mode::Same => |i, _, s| i / s,
        };

        input_sizes
            .iter()
            .zip(kernel_sizes)
            .zip(strides)
            .map(|((&i, &k), &s)| output_size_fn(i, k, s))
            .collect()
    }

    fn product_sum_fn<T>(
        &self,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
    ) -> Result<ProductSumFn<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        Ok(match self {
            Mode::Valid => {
                if Shape::conv_larger_input(input_sizes, kernel_sizes)? {
                    Mode::valid_input_product_sum::<T>
                } else {
                    Mode::valid_kernel_product_sum::<T>
                }
            }
            Mode::Full => Mode::full_product_sum::<T>,
            Mode::Same => Mode::same_product_sum::<T>,
        })
    }

    fn valid_input_product_sum<T>(
        tensors: (&Tensor<T>, &Tensor<T>),
        sizes: (&[usize], &[usize]),
        dims: (&[usize], &[usize]),
        indices: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let ranges = indices
            .iter()
            .zip(sizes.1)
            .map(|(&index, &kernel_size)| (index, index + kernel_size))
            .collect::<Vec<(usize, usize)>>();

        let input_slice = tensors.0.slice_dims(dims.0, &ranges)?;
        (input_slice * tensors.1)?.sum_dims(dims.0, true)
    }

    fn valid_kernel_product_sum<T>(
        tensors: (&Tensor<T>, &Tensor<T>),
        sizes: (&[usize], &[usize]),
        dims: (&[usize], &[usize]),
        indices: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let ranges = indices
            .iter()
            .zip(sizes.0)
            .zip(sizes.1)
            .map(|((index, input_size), kernel_size)| {
                let start = kernel_size - input_size - index;
                let end = start + input_size;

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let kernel_slice = tensors.1.slice_dims(dims.0, &ranges)?;
        (tensors.0 * kernel_slice)?.sum_dims(dims.0, true)
    }

    fn full_product_sum<T>(
        tensors: (&Tensor<T>, &Tensor<T>),
        sizes: (&[usize], &[usize]),
        dims: (&[usize], &[usize]),
        indices: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let input_ranges = indices
            .iter()
            .zip(sizes.0)
            .zip(sizes.1)
            .map(|((&index, &input_size), &kernel_size)| {
                let start = index.saturating_sub(kernel_size - 1);
                let end = (index + 1).min(input_size);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let kernel_ranges = indices
            .iter()
            .zip(sizes.0)
            .zip(sizes.1)
            .map(|((&index, &input_size), &kernel_size)| {
                let start = (kernel_size - 1).saturating_sub(index);
                let range = input_size - index.saturating_sub(kernel_size - 1);
                let end = kernel_size.min(start + range);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let input_slice = tensors.0.slice_dims(dims.0, &input_ranges)?;
        let kernel_slice = tensors.1.slice_dims(dims.1, &kernel_ranges)?;
        (input_slice * kernel_slice)?.sum_dims(dims.0, true)
    }

    fn same_product_sum<T>(
        tensors: (&Tensor<T>, &Tensor<T>),
        sizes: (&[usize], &[usize]),
        dims: (&[usize], &[usize]),
        indices: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let input_ranges = indices
            .iter()
            .zip(sizes.0)
            .zip(sizes.1)
            .map(|((&index, &input_size), &kernel_size)| {
                let start = index.saturating_sub(1);
                let end = (index + kernel_size - 1).min(input_size);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let kernel_ranges = indices
            .iter()
            .zip(sizes.0)
            .zip(sizes.1)
            .map(|((&index, &input_size), &kernel_size)| {
                let start = 1_usize.saturating_sub(index);
                let range = input_size - index.saturating_sub(1);
                let end = kernel_size.min(start + range);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let input_slice = tensors.0.slice_dims(dims.0, &input_ranges)?;
        let kernel_slice = tensors.1.slice_dims(dims.1, &kernel_ranges)?;
        (input_slice * kernel_slice)?.sum_dims(dims.0, true)
    }
}
