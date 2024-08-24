use crate::{
    core::{shape::Shape, slicer::SliceIterator},
    Res, Tensor,
};
use std::{iter::Sum, ops::Mul};

impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + std::fmt::Debug,
{
    pub fn matmul(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        match (self.ndims(), rhs.ndims()) {
            (0, _) | (_, 0) => Err("Cannot matmul 0d tensor".to_string()),
            (1, _) | (_, 1) => Tensor::scalar((self * rhs)?.sum()),
            (2, 2) => self.matmul_2d(rhs),
            (_, _) => self.matmul_nd(rhs),
        }
    }

    pub fn matmul_2d(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        let n1 = self.sizes()[1];
        let n2 = rhs.sizes()[0];

        if n1 != n2 {
            return Err(format!(
                "Cannot be matrix multiplied. Shapes: m1 x n1 and m2 x n2 x l, n1 ({}) != n2 ({}).",
                n1, n2
            ));
        }

        let rhs = rhs.transpose(0, 1)?.to_contiguous()?;

        let m = self.sizes()[0];
        let l = rhs.sizes()[0];

        let lhs_iter = SliceIterator::new(&self.shape, &[0]);
        let mut rhs_iter = SliceIterator::new(&rhs.shape, &[0]);
        let mut data = Vec::with_capacity(m * l);

        for lhs_index in lhs_iter {
            let lhs_row = self.single_slice(&lhs_index)?;

            for rhs_index in &mut rhs_iter {
                let rhs_row = rhs.single_slice(&rhs_index)?;

                let product = (&lhs_row * &rhs_row)?.sum();
                data.push(product);
            }
        }

        Tensor::new(&data, &[m, l])
    }

    pub fn matmul_nd(&self, rhs: &Tensor<T>) -> Res<Tensor<T>> {
        let dim_1 = self.ndims() - 1;
        let n1 = self.sizes()[dim_1];

        let dim_2 = rhs.ndims() - 2;
        let n2 = rhs.sizes()[dim_2];

        if n1 != n2 {
            return Err(format!(
                "Cannot be matrix multiplied. Shapes: m1 x n1 and m2 x n2 x l, n1 ({}) != n2 ({}).",
                n1, n2
            ));
        }

        let rhs = rhs.transpose(dim_2, dim_1)?.to_contiguous()?;

        let m = self.sizes()[dim_2];
        let l = rhs.sizes()[dim_2];

        let lhs_sizes = &self.sizes()[..dim_2];
        let rhs_sizes = &rhs.sizes()[..dim_2];
        let broadcast = &Shape::broadcast(lhs_sizes, rhs_sizes)?;

        let lhs = self
            .reshape_then_expand(&[broadcast, &vec![m, n1][..]].concat())?
            .contiguous_if_not()?;
        let rhs = rhs
            .reshape_then_expand(&[broadcast, &vec![l, n1][..]].concat())?
            .contiguous_if_not()?;

        let dimension = &[dim_2];
        let lhs_iter = SliceIterator::new(&lhs.shape, dimension);
        let rhs_iter = SliceIterator::new(&rhs.shape, dimension);

        let b = broadcast.iter().product::<usize>();
        let mut data = vec![Vec::with_capacity(m * l); b];

        for lhs_index in lhs_iter {
            let lhs_row = lhs.single_slice(&lhs_index)?;

            for rhs_index in rhs_iter.clone() {
                let rhs_row = rhs.single_slice(&rhs_index)?;
                let product = (&lhs_row * &rhs_row)?.sum_dimensions(&[0])?;

                product
                    .data_contiguous()
                    .iter()
                    .enumerate()
                    .for_each(|(i, prod)| data[i].push(*prod));
            }
        }

        let data = data.into_iter().flatten().collect::<Vec<T>>();
        let sizes = &[broadcast, &vec![m, l][..]].concat();
        Tensor::new(&data, sizes)
    }
}
