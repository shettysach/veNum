use crate::{
    core::{indexer::IndexIterator, shape::Shape},
    Res, Tensor,
};
use std::{iter::Sum, ops::Mul};

pub enum Slide {
    Valid,
    Full,
    Same,
}

impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + Default,
{
    pub fn convolve_1d(&self, kernel: &Tensor<T>, mode: Slide) -> Res<Tensor<T>> {
        self.correlate_1d(&kernel.flip_all()?, mode)
    }

    pub fn convolve_2d(&self, kernel: &Tensor<T>, mode: Slide) -> Res<Tensor<T>> {
        self.correlate_2d(&kernel.flip_all()?, mode)
    }

    pub fn correlate_1d(&self, kernel: &Tensor<T>, mode: Slide) -> Res<Tensor<T>> {
        let i_first = self.ndims() - 1;
        let input_sizes = self.sizes();

        let input_width = input_sizes[i_first];
        let input_conv_sizes = &[input_width];

        let k_first = kernel.ndims() - 1;
        let kernel_width = kernel.sizes()[k_first];
        let kernel_conv_sizes = &[kernel_width];

        let prod_sum_fn = mode.product_sum_function(input_conv_sizes, kernel_conv_sizes)?;
        let output_sizes = mode.output_sizes(input_conv_sizes, kernel_conv_sizes);
        let output_width = output_sizes[0];

        let sizes = [&input_sizes[..i_first], &output_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for index in 0..output_width {
            let product_sum = prod_sum_fn(
                self,
                kernel,
                input_conv_sizes,
                kernel_conv_sizes,
                &[i_first],
                &[k_first],
                &[index],
            )?;

            product_sum
                .data_contiguous()
                .iter()
                .enumerate()
                .for_each(|(index, &value)| {
                    let offset = (index * output_width) + index;
                    data[offset] = value
                });
        }

        Tensor::new(&data, &sizes)
    }

    // NOTE: Can generalize for higher dimensions

    pub fn correlate_2d(&self, kernel: &Tensor<T>, mode: Slide) -> Res<Tensor<T>> {
        let n = self.ndims();
        let (i_second, i_first) = (n - 2, n - 1);
        let input_sizes = self.sizes();
        let input_conv_sizes = &[input_sizes[i_second], input_sizes[i_first]];

        let kn = kernel.ndims();
        let (k_second, k_first) = (kn - 2, kn - 1);
        let kernel_sizes = kernel.sizes();
        let kernel_conv_sizes = &[kernel_sizes[k_second], kernel_sizes[k_first]];

        let prod_sum_fn = mode.product_sum_function(input_conv_sizes, kernel_conv_sizes)?;
        let output_sizes = mode.output_sizes(input_conv_sizes, kernel_conv_sizes);
        let output_conv_sizes = &[output_sizes[0], output_sizes[1]];

        // TODO: Better concatenation
        let sizes = [&input_sizes[..i_second], output_conv_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        let shape = Shape::new(output_conv_sizes);
        let output_conv_product = output_conv_sizes[0] * output_conv_sizes[1];

        for iter_index in IndexIterator::new(&shape) {
            let product_sum = prod_sum_fn(
                self,
                kernel,
                input_conv_sizes,
                kernel_conv_sizes,
                &[i_second, i_first],
                &[k_second, k_first],
                &iter_index,
            )?;

            product_sum
                .data_contiguous()
                .iter()
                .enumerate()
                .for_each(|(index, &value)| {
                    let offset = (index * output_conv_product)
                        + (iter_index[0] * output_conv_sizes[1])
                        + iter_index[1];
                    data[offset] = value
                });
        }

        Tensor::new(&data, &sizes)
    }
}

pub type ProductSumFn<T> = fn(
    &Tensor<T>,
    &Tensor<T>,
    &[usize],
    &[usize],
    &[usize],
    &[usize],
    &[usize],
) -> Result<Tensor<T>, String>;

impl Slide {
    fn output_sizes(&self, input_sizes: &[usize], kernel_sizes: &[usize]) -> Vec<usize> {
        let output_size: fn(usize, usize) -> usize = match self {
            Slide::Valid => |i, k| i.abs_diff(k) + 1,
            Slide::Full => |i, k| i + k - 1,
            Slide::Same => |i, _| i,
        };

        input_sizes
            .iter()
            .zip(kernel_sizes)
            .map(|(&i, &k)| output_size(i, k))
            .collect()
    }

    fn product_sum_function<T>(
        &self,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
    ) -> Res<ProductSumFn<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        Ok(match self {
            Slide::Valid => {
                if Shape::greater_input(input_sizes, kernel_sizes)? {
                    Slide::valid_input_product_sum::<T>
                } else {
                    Slide::valid_kernel_product_sum::<T>
                }
            }
            Slide::Full => Slide::full_product_sum::<T>,
            Slide::Same => Slide::same_product_sum::<T>,
        })
    }

    fn valid_input_product_sum<T>(
        input: &Tensor<T>,
        kernel: &Tensor<T>,
        _input_sizes: &[usize],
        kernel_sizes: &[usize],
        input_dimensions: &[usize],
        _kernel_dimensions: &[usize],
        indices: &[usize],
    ) -> Res<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let ranges = indices
            .iter()
            .zip(kernel_sizes)
            .map(|(&ind, &ker)| (ind, ind + ker))
            .collect::<Vec<(usize, usize)>>();

        let input_slice = input.slice_dims(input_dimensions, &ranges)?;
        (input_slice * kernel)?.sum_dims(input_dimensions, true)
    }

    fn valid_kernel_product_sum<T>(
        input: &Tensor<T>,
        kernel: &Tensor<T>,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
        input_dimensions: &[usize],
        kernel_dimensions: &[usize],
        indices: &[usize],
    ) -> Res<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let ranges = indices
            .iter()
            .zip(kernel_sizes)
            .zip(input_sizes)
            .map(|((ind, ker), inp)| {
                let start = ker - inp - ind;
                let end = start + inp;

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let kernel_slice = kernel.slice_dims(kernel_dimensions, &ranges)?;
        (input * kernel_slice)?.sum_dims(input_dimensions, true)
    }

    fn full_product_sum<T>(
        input: &Tensor<T>,
        kernel: &Tensor<T>,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
        input_dimensions: &[usize],
        kernel_dimensions: &[usize],
        indices: &[usize],
    ) -> Res<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let input_ranges = indices
            .iter()
            .zip(kernel_sizes)
            .zip(input_sizes)
            .map(|((&ind, &ker), &inp)| {
                let start = ind.saturating_sub(ker - 1);
                let end = (ind + 1).min(inp);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let kernel_ranges = indices
            .iter()
            .zip(kernel_sizes)
            .zip(input_sizes)
            .map(|((&ind, &ker), &inp)| {
                let start = (ker - 1).saturating_sub(ind);
                let range = inp - ind.saturating_sub(ker - 1);
                let end = ker.min(start + range);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let input_slice = input.slice_dims(input_dimensions, &input_ranges)?;
        let kernel_slice = kernel.slice_dims(kernel_dimensions, &kernel_ranges)?;
        (input_slice * kernel_slice)?.sum_dims(input_dimensions, true)
    }

    fn same_product_sum<T>(
        input: &Tensor<T>,
        kernel: &Tensor<T>,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
        input_dimensions: &[usize],
        kernel_dimensions: &[usize],
        indices: &[usize],
    ) -> Res<Tensor<T>>
    where
        T: Copy + Mul<Output = T> + Sum<T>,
    {
        let input_ranges = indices
            .iter()
            .zip(kernel_sizes)
            .zip(input_sizes)
            .map(|((&ind, &ker), &inp)| {
                let start = ind.saturating_sub(1);
                let end = (ind + ker - 1).min(inp);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let kernel_ranges = indices
            .iter()
            .zip(kernel_sizes)
            .zip(input_sizes)
            .map(|((&ind, &ker), &inp)| {
                let start = 1_usize.saturating_sub(ind);
                let range = inp - ind.saturating_sub(1);
                let end = ker.min(start + range);

                (start, end)
            })
            .collect::<Vec<(usize, usize)>>();

        let input_slice = input.slice_dims(input_dimensions, &input_ranges)?;
        let kernel_slice = kernel.slice_dims(kernel_dimensions, &kernel_ranges)?;
        (input_slice * kernel_slice)?.sum_dims(input_dimensions, true)
    }
}
