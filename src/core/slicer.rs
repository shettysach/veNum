use crate::core::shape::Shape;

#[derive(Clone)]
pub struct SliceIterator<'a> {
    shape: &'a Shape,
    indices: Vec<Option<usize>>,
    current: usize,
    maximum: usize,
}

impl<'a> SliceIterator<'a> {
    pub(crate) fn new(shape: &'a Shape, dimensions: &'a [usize], keepdims: bool) -> Self {
        let mut maximum = 1;
        let indices = if keepdims {
            (0..shape.ndims())
                .map(|d| {
                    (!dimensions.contains(&d)).then(|| {
                        maximum *= shape.sizes[d];
                        0
                    })
                })
                .collect()
        } else {
            (0..shape.ndims())
                .map(|d| {
                    dimensions.contains(&d).then(|| {
                        maximum *= shape.sizes[d];
                        0
                    })
                })
                .collect()
        };

        SliceIterator {
            shape,
            indices,
            current: 0,
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

        for (d, slice_index) in self.indices.iter_mut().enumerate().rev() {
            if let Some(slice_index) = slice_index.as_mut() {
                *slice_index += 1;

                if *slice_index < self.shape.sizes[d] {
                    break;
                }

                *slice_index = 0;
            }
        }

        Some(next)
    }
}
