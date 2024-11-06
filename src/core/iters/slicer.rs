pub struct Slicer<'a> {
    sizes: &'a [usize],
    indices: Vec<Option<usize>>,
    current: usize,
    maximum: usize,
}

impl<'a> Slicer<'a> {
    pub(crate) fn new(sizes: &'a [usize], dimensions: &'a [usize], keepdims: bool) -> Self {
        let keepdim_fn: fn(&[usize], usize) -> bool = if keepdims {
            |dimensions, d| !dimensions.contains(&d)
        } else {
            |dimensions, d| dimensions.contains(&d)
        };

        let mut maximum = 1;
        let indices = (0..sizes.len())
            .map(|d| {
                keepdim_fn(dimensions, d).then(|| {
                    maximum *= sizes[d];
                    0
                })
            })
            .collect();

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

        let next = self.indices.clone();

        for (d, slice_index) in self.indices.iter_mut().enumerate().rev() {
            if let Some(slice_index) = slice_index.as_mut() {
                *slice_index += 1;

                if *slice_index >= self.sizes[d] {
                    *slice_index = 0;
                } else {
                    break;
                }
            }
        }

        self.current += 1;
        Some(next)
    }
}
