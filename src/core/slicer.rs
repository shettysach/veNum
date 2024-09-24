pub struct Slicer<'a> {
    sizes: &'a [usize],
    indices: Vec<Option<usize>>,
    current: usize,
    maximum: usize,
}

impl<'a> Slicer<'a> {
    pub(crate) fn new(sizes: &'a [usize], dimensions: &'a [usize], keepdims: bool) -> Self {
        let mut maximum = 1;
        let indices = if keepdims {
            (0..sizes.len())
                .map(|d| {
                    (!dimensions.contains(&d)).then(|| {
                        maximum *= sizes[d];
                        0
                    })
                })
                .collect()
        } else {
            (0..sizes.len())
                .map(|d| {
                    dimensions.contains(&d).then(|| {
                        maximum *= sizes[d];
                        0
                    })
                })
                .collect()
        };

        Slicer {
            sizes,
            indices,
            current: 0,
            maximum,
        }
    }
}

impl<'a> Iterator for Slicer<'a> {
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

                if *slice_index < self.sizes[d] {
                    break;
                }

                *slice_index = 0;
            }
        }

        Some(next)
    }
}
