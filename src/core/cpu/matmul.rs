use crate::{
    core::{shape::Shape, slicer::SliceIterator},
    Res, Tensor,
};
use std::{iter::Sum, ops::Mul};

impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + Default,
{
    pub fn matmul(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        match (self.ndims(), rhs.ndims()) {
            (0, _) | (_, 0) => Err("Cannot matmul 0d tensor".to_string()),
            (1, 1) => Tensor::scalar((self * rhs)?.sum()?),
            (2, 2) => self.matmul_2d(rhs),
            (_, _) => self.matmul_nd(rhs),
        }
    }

    pub fn matmul_2d(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        let (n1, n2) = (self.sizes()[1], rhs.sizes()[0]);

        if n1 != n2 {
            return Err(format!(
                "Cannot be matrix multiplied. Shapes: m x n1 and n2 x l, n1 ({}) != n2 ({}).",
                n1, n2
            ));
        }

        let rhs = rhs.transpose(1, 0)?.to_contiguous()?;
        let (m, l) = (self.sizes()[0], rhs.sizes()[0]);

        let (lhs_iter, rhs_iter) = (
            SliceIterator::new(&self.shape, &[0], false),
            SliceIterator::new(&rhs.shape, &[0], false).collect::<Vec<_>>(),
        );
        let mut data = Vec::with_capacity(m * l);

        for lhs_index in lhs_iter {
            let lhs_row = self.single_slice(&lhs_index)?;

            for rhs_index in rhs_iter.iter() {
                let rhs_row = rhs.single_slice(rhs_index)?;
                let prodsum = (&lhs_row * &rhs_row)?.sum()?;

                data.push(prodsum);
            }
        }

        Tensor::new(&data, &[m, l])
    }

    pub fn matmul_nd(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        let max_ndims = self.ndims().max(rhs.ndims());
        let (lhs, rhs) = (self.unsqueeze(max_ndims)?, rhs.unsqueeze(max_ndims)?);

        let (dim_1, dim_2) = (max_ndims - 1, max_ndims - 2);
        let (n1, n2) = (lhs.sizes()[dim_1], rhs.sizes()[dim_2]);

        if n1 != n2 {
            return Err(format!(
                "Cannot be matrix multiplied. Shapes: m1 x n1 and m2 x n2 x l, n1 ({}) != n2 ({}).",
                n1, n2
            ));
        }

        let rhs = rhs.transpose(dim_2, dim_1)?.to_contiguous()?;
        let (m, l) = (lhs.sizes()[dim_2], rhs.sizes()[dim_2]);
        let ml = m * l;

        // TODO: Better concatenation
        let (lhs_sizes, rhs_sizes) = (&lhs.sizes()[..dim_2], &rhs.sizes()[..dim_2]);
        let broadcast = &Shape::broadcast(lhs_sizes, rhs_sizes)?;
        let broadcast_ndims = broadcast.iter().product::<usize>();

        let (lhs, rhs) = (
            lhs.expand(&[broadcast, &vec![m, n1][..]].concat())?
                .into_contiguous()?,
            rhs.expand(&[broadcast, &vec![l, n1][..]].concat())?
                .into_contiguous()?,
        );

        let slice_dim = &[dim_2];
        let (lhs_iter, rhs_iter) = (
            SliceIterator::new(&self.shape, slice_dim, false),
            SliceIterator::new(&rhs.shape, slice_dim, false).collect::<Vec<_>>(),
        );

        let mut data = vec![T::default(); ml * broadcast_ndims];

        for (li, lhs_slice) in lhs_iter.enumerate() {
            let lhs_row = lhs.single_slice(&lhs_slice)?;

            for (ri, rhs_slice) in rhs_iter.iter().enumerate() {
                let rhs_row = rhs.single_slice(rhs_slice)?;
                let prodsum = (&lhs_row * &rhs_row)?.sum_dimensions(&[dim_1, dim_2], true)?;

                prodsum
                    .data_contiguous()
                    .iter()
                    .enumerate()
                    .for_each(|(i, p)| {
                        let index = (i * ml) + (li * m) + ri;
                        data[index] = *p
                    });
            }
        }

        let sizes = &[broadcast, &vec![m, l][..]].concat();
        Tensor::new(&data, sizes)
    }
}
