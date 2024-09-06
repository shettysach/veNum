use crate::{core::shape::Shape, Res, Tensor};
use std::{iter::Sum, ops::Mul};

pub enum Slide {
    Valid,
    Full,
    Same,
}

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

    fn input_padding<T>(
        &self,
        input: &Tensor<T>,
        dimensions: &[usize],
        kernel_sizes: &[usize],
    ) -> Res<Tensor<T>>
    where
        T: Copy + Default,
    {
        match self {
            Slide::Valid => Ok(input.clone()),
            Slide::Full => {
                let padding = kernel_sizes
                    .iter()
                    .map(|k| (k - 1, k - 1))
                    .collect::<Vec<(usize, usize)>>();
                input.pad_dimensions(T::default(), dimensions, &padding)
            }
            Slide::Same => {
                let padding = vec![(1, 1); dimensions.len()];
                input.pad_dimensions(T::default(), dimensions, &padding)
            }
        }
    }
}

// TODO: Handle cases where kernel > input

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
        let first = self.ndims() - 1;
        let input_sizes = self.sizes();

        let width = input_sizes[first];
        let input_convolution_sizes = &[width];

        let kernel_width = kernel.sizes()[kernel.ndims() - 1];
        let kernel_convolution_sizes = &[kernel_width];

        Shape::valid_convolution(input_convolution_sizes, kernel_convolution_sizes, &mode)?;
        let output_sizes = mode.output_sizes(input_convolution_sizes, kernel_convolution_sizes);
        let output_width = output_sizes[0];

        let sizes = [&input_sizes[..first], &output_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        let input = mode.input_padding(self, &[first], &[kernel_width])?;

        for lhs_index in 0..output_width {
            let slice =
                input.slice_dimensions(&[first], &[(lhs_index, lhs_index + kernel_width)])?;

            let product_sum = (slice * kernel)?.sum_dimensions(&[first], true)?;

            product_sum
                .data_contiguous()
                .iter()
                .enumerate()
                .for_each(|(index, &value)| {
                    let offset = (index * output_width) + lhs_index;
                    data[offset] = value
                });
        }

        Tensor::new(&data, &sizes)
    }

    pub fn correlate_2d(&self, kernel: &Tensor<T>, mode: Slide) -> Res<Tensor<T>> {
        let n = self.ndims();
        let (first, second) = (n - 1, n - 2);

        let input_sizes = self.sizes();
        let (height, width) = (input_sizes[second], input_sizes[first]);
        let input_convolution_sizes = &[height, width];

        let kn = kernel.ndims();
        let sizes = kernel.sizes();
        let (kernel_height, kernel_width) = (sizes[kn - 2], sizes[kn - 1]);
        let kernel_convolution_sizes = &[kernel_height, kernel_width];

        Shape::valid_convolution(input_convolution_sizes, kernel_convolution_sizes, &mode)?;
        let output_sizes = mode.output_sizes(input_convolution_sizes, kernel_convolution_sizes);
        let (output_height, output_width) = (output_sizes[0], output_sizes[1]);

        // TODO: Better concatenation
        let sizes = [&input_sizes[..second], &[output_height, output_width]].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        let input = mode.input_padding(self, &[first, second], &[kernel_width, kernel_height])?;

        for lhs_index in 0..output_height {
            for rhs_index in 0..output_width {
                let slice = input.slice_dimensions(
                    &[first, second],
                    &[
                        (rhs_index, rhs_index + kernel_width),
                        (lhs_index, lhs_index + kernel_height),
                    ],
                )?;

                let product_sum = (slice * kernel)?.sum_dimensions(&[first, second], true)?;

                product_sum
                    .data_contiguous()
                    .iter()
                    .enumerate()
                    .for_each(|(index, &value)| {
                        let offset = (index * output_height * output_width)
                            + (lhs_index * output_width)
                            + rhs_index;
                        data[offset] = value
                    });
            }
        }

        Tensor::new(&data, &sizes)
    }
}
