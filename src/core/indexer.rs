use crate::core::shape::Shape;

pub(crate) struct IndexIterator<'a> {
    shape: &'a Shape,
    indices: Vec<usize>,
    current: usize,
    maximum: usize,
}

impl<'a> IndexIterator<'a> {
    pub(crate) fn new(shape: &'a Shape) -> Self {
        IndexIterator {
            shape,
            indices: vec![0; shape.ndims()],
            current: 0,
            maximum: shape.numel(),
        }
    }
}

impl<'a> Iterator for IndexIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.maximum {
            return None;
        };

        let next = self.indices.clone();

        for i in (0..self.shape.ndims()).rev() {
            self.indices[i] += 1;

            if self.indices[i] != self.shape.sizes[i] {
                break;
            }

            self.indices[i] = 0;
        }

        self.current += 1;
        Some(next)
    }
}
