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
            (1, 1) => Tensor::scalar(self.mul(rhs)?.sum()?),
            (2, 2) => self.matmul_2d(rhs),
            (_, _) => self.matmul_nd(rhs),
        }
    }

    // O(n^3)

    fn matmul_2d(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        let (n1, n2) = (self.sizes()[1], rhs.sizes()[0]);

        if n1 != n2 {
            return Err(format!(
                "Cannot be matrix multiplied. 
                Shapes: m x n1 and n2 x l, n1 ({}) != n2 ({}).",
                n1, n2
            ));
        }

        let rhs = rhs.transpose(1, 0)?.to_contiguous()?;
        let (m, l) = (self.sizes()[0], rhs.sizes()[0]);

        let (lhs_iter, rhs_iter) = (
            SliceIterator::new(&self.shape.sizes, &[0], false),
            SliceIterator::new(&rhs.shape.sizes, &[0], false).collect::<Vec<_>>(),
        );
        let mut data = Vec::with_capacity(m * l);

        for lhs_index in lhs_iter {
            let row = self.slicer(&lhs_index)?;

            for rhs_index in rhs_iter.iter() {
                let column = rhs.slicer(rhs_index)?;
                let product = (&row * &column)?;
                let product_sum = product.sum()?;

                data.push(product_sum);
            }
        }

        Tensor::init(&data, &[m, l])
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
            return Err(format!(
                "Cannot be matrix multiplied. 
                Shapes: m1 x n1 and m2 x n2 x l, n1 ({}) != n2 ({}).",
                n1, n2
            ));
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
            SliceIterator::new(&lhs.shape.sizes, slice_dim, false),
            SliceIterator::new(&rhs.shape.sizes, slice_dim, false).collect::<Vec<_>>(),
        );

        let sizes = [broadcast.as_slice(), &[m, l]].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for (lhs_index, lhs_slice) in lhs_iter.enumerate() {
            let row = lhs.slicer(&lhs_slice)?;

            for (rhs_index, rhs_slice) in rhs_iter.iter().enumerate() {
                let column = rhs.slicer(rhs_slice)?;
                let product = (&row * &column)?;
                let product_sum = product.sum_dims(&[first, second], true)?;

                for (index, &value) in product_sum.data_contiguous().iter().enumerate() {
                    let offset = (index * ml) + (lhs_index * l) + rhs_index;
                    data[offset] = value
                }
            }
        }

        Tensor::init(&data, &sizes)
    }
}
