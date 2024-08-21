use crate::core::shape::Shape;

#[derive(Clone)]
pub struct SliceIterator<'a> {
    shape: &'a Shape,
    indices: Vec<Option<usize>>,
    dimensions: &'a [usize],
    exhausted: bool,
}

impl<'a> SliceIterator<'a> {
    pub(crate) fn new(shape: &'a Shape, dimensions: &'a [usize]) -> Self {
        let indices = (0..shape.numdims())
            .map(|d| {
                if dimensions.contains(&d) {
                    Some(0)
                } else {
                    None
                }
            })
            .collect();
        let exhausted = shape.sizes.is_empty();

        SliceIterator {
            shape,
            exhausted,
            dimensions,
            indices,
        }
    }
}

impl<'a> Iterator for SliceIterator<'a> {
    type Item = Vec<Option<usize>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        };

        self.exhausted = self
            .dimensions
            .iter()
            .all(|&d| self.indices[d] == Some(self.shape.sizes[d] - 1));

        let next = self.indices.clone();

        for &d in self.dimensions.iter().rev() {
            let i = self.indices[d].unwrap();
            self.indices[d] = Some(i + 1);

            if i + 1 < self.shape.sizes[d] {
                break;
            }

            self.indices[d] = Some(0);
        }

        Some(next)
    }
}
