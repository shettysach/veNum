use crate::core::shape::Shape;

#[derive(Clone)]
pub struct SliceIterator<'a> {
    shape: &'a Shape,
    indices: Vec<Option<usize>>,
    dimensions: &'a [usize],
    current: usize,
    maximum: usize,
}

impl<'a> SliceIterator<'a> {
    pub(crate) fn new(shape: &'a Shape, dimensions: &'a [usize]) -> Self {
        let indices = (0..shape.ndims())
            .map(|d| {
                if dimensions.contains(&d) {
                    Some(0)
                } else {
                    None
                }
            })
            .collect();
        let current = 0;
        let maximum = dimensions.iter().map(|&d| shape.sizes[d]).product();

        SliceIterator {
            shape,
            indices,
            dimensions,
            current,
            maximum,
        }
    }
}

impl<'a> Iterator for SliceIterator<'a> {
    type Item = Vec<Option<usize>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.maximum {
            return None;
        }

        self.current += 1;

        let next = self.indices.clone();

        for &d in self.dimensions.iter().rev() {
            let i = self.indices[d].unwrap() + 1;

            if i != self.shape.sizes[d] {
                self.indices[d] = Some(i);
                break;
            }

            self.indices[d] = Some(0);
        }

        Some(next)
    }
}
