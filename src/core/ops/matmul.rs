use crate::{
    core::{errors::MatmulShapeError, iters::Slicer, shape::Shape, utils::Res},
    Tensor,
};
use std::{iter::Sum, ops::Mul};

impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + Default,
{
    pub fn matmul(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        match (self.ndims(), rhs.ndims()) {
            (0, _) | (_, 0) => Err(MatmulShapeError::Matmul0d.into()),
            (1, 1) => Tensor::scalar(self.mul(rhs)?.sum()?),
            (2, 2) => self.matmul_2d(rhs),
            (_, _) => self.matmul_nd(rhs),
        }
    }

    fn matmul_2d(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
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

    fn matmul_nd(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        let max_dimensions = self.ndims().max(rhs.ndims());
        let (lhs, rhs) = (
            self.unsqueeze(max_dimensions)?,
            rhs.unsqueeze(max_dimensions)?,
        );

        let (first, second) = (max_dimensions - 1, max_dimensions - 2);
        let (n1, n2) = (lhs.sizes()[first], rhs.sizes()[second]);

        if n1 != n2 {
            return Err(MatmulShapeError::MatmulNd { n1, n2 }.into());
        }

        let rhs = rhs.transpose(second, first)?.to_contiguous()?;
        let (m, l) = (lhs.sizes()[second], rhs.sizes()[second]);
        let ml = m * l;

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

        for (li, lhs_slice) in lhs_iter.enumerate() {
            let row = &lhs.slicer(&lhs_slice)?;

            for (ri, rhs_slice) in rhs_iter.iter().enumerate() {
                let column = rhs.slicer(rhs_slice)?;
                let product_sum = (row * column)?.sum_dims(&[first, second], true)?;

                for (i, &value) in product_sum.data_contiguous().iter().enumerate() {
                    let offset = (i * ml) + (li * l) + ri;
                    data[offset] = value
                }
            }
        }

        Ok(Tensor::init(data, &sizes))
    }
}
