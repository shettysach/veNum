use crate::core::shape::Shape;

//pub(crate) struct IndexIterator<'a> {
pub struct IndexIterator<'a> {
    shape: &'a Shape,
    indices: Vec<usize>,
    exhausted: bool,
}

impl<'a> IndexIterator<'a> {
    pub(crate) fn new(shape: &'a Shape) -> Self {
        let indices = vec![0; shape.numdims()];
        IndexIterator {
            shape,
            exhausted: !shape.valid_indices(&indices),
            indices,
        }
    }
}

impl<'a> Iterator for IndexIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let next = self.indices.clone();
        for i in (0..self.shape.numdims()).rev() {
            self.indices[i] += 1;

            if self.indices[i] < self.shape.sizes[i] {
                break;
            }

            self.indices[i] = 0;
        }

        self.exhausted = self.indices.iter().all(|i| *i == 0);
        Some(next)
    }
}
