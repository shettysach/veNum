use crate::{
    core::{errors::MatmulShapeError, iters::Slicer, shape::Shape},
    Tensor,
};
use anyhow::Result;
use std::{iter::Sum, ops::Mul};

impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + Default,
{
    pub fn matmul(&self, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        match (self.rank(), rhs.rank()) {
            (0, _) | (_, 0) => Err(MatmulShapeError::Matmul0d.into()),
            (1, 1) => Tensor::scalar(self.mul(rhs)?.sum()?),
            (2, 2) => self.matmul_2d(rhs),
            (_, _) => self.matmul_nd(rhs),
        }
    }

    fn matmul_2d(&self, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        let (n1, n2) = (self.sizes()[1], rhs.sizes()[0]);

        if n1 != n2 {
            return Err(MatmulShapeError::Matmul2d { n1, n2 }.into());
        }

        let rhs = rhs.transpose(1, 0)?.to_contiguous()?;
        let (m, l) = (self.sizes()[0], rhs.sizes()[0]);

        let (lhs_iter, rhs_iter) = (
            Slicer::new(&self.shape.sizes, &[0], false),
            Slicer::new(&rhs.shape.sizes, &[0], false).collect::<Vec<_>>(),
        );
        let mut data = Vec::with_capacity(m * l);

        for lhs_slice in lhs_iter {
            let row = &self.slicer(&lhs_slice)?;

            for rhs_slice in rhs_iter.iter() {
                let column = rhs.slicer(rhs_slice)?;
                let product_sum = (row * column)?.sum()?;

                data.push(product_sum);
            }
        }

        Ok(Tensor::init(data, &[m, l]))
    }

    fn matmul_nd(&self, rhs: &Tensor<T>) -> Result<Tensor<T>> {
        let max_dims = self.rank().max(rhs.rank());
        let (lhs, rhs) = (self.unsqueeze(max_dims)?, rhs.unsqueeze(max_dims)?);

        let (first, second) = (max_dims - 1, max_dims - 2);
        let (n1, n2) = (lhs.sizes()[first], rhs.sizes()[second]);

        if n1 != n2 {
            return Err(MatmulShapeError::MatmulNd { n1, n2 }.into());
        }

        let rhs = rhs.transpose(second, first)?.to_contiguous()?;
        let (m, l) = (lhs.sizes()[second], rhs.sizes()[second]);

        let broadcast = &Shape::broadcast(&lhs.sizes()[..second], &rhs.sizes()[..second])?;
        let (lhs, rhs) = (
            lhs.expand(&[broadcast.as_slice(), &[m, n1]].concat())?
                .into_contiguous()?,
            rhs.expand(&[broadcast.as_slice(), &[l, n1]].concat())?
                .into_contiguous()?,
        );

        let slice_dim = &[second];
        let (lhs_iter, rhs_iter) = (
            Slicer::new(&lhs.shape.sizes, slice_dim, false),
            Slicer::new(&rhs.shape.sizes, slice_dim, false).collect::<Vec<_>>(),
        );

        let sizes = [broadcast.as_slice(), &[m, l]].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for (lhs_i, lhs_slice) in lhs_iter.enumerate() {
            let row = &lhs.slicer(&lhs_slice)?;

            for (rhs_i, rhs_slice) in rhs_iter.iter().enumerate() {
                let column = rhs.slicer(rhs_slice)?;
                let product_sum = (row * column)?.sum_dims(&[first, second], true)?;

                // Pointer arithmetic
                for (iter_i, &value) in product_sum.data_contiguous().iter().enumerate() {
                    let offset = (iter_i * m * l) + (lhs_i * l) + rhs_i;
                    data[offset] = value
                }
            }
        }

        Ok(Tensor::init(data, &sizes))
    }
}
