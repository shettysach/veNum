use crate::Tensor;
use prettytable::format::{self, TableFormat};
use prettytable::{Cell, Row, Table};
use std::any::type_name;
use std::fmt::{Debug, Display, Formatter, Result};

impl<T: Debug + Copy> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let num_dims = self.numdims();
        let sizes = self.sizes();
        let dtype = type_name::<T>();

        f.debug_struct("Tensor")
            .field("dtype", &dtype)
            .field("numdims", &num_dims)
            .field("shape", &sizes)
            .finish()
    }
}

impl<T: Display + Debug + Copy> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let n = self.shape.numdims();

        if (1..=10).contains(&n) {
            let table_format = &format::consts::FORMAT_BOX_CHARS;

            let mut table = match n {
                1 => {
                    let row = tensor_1d(self);
                    let mut table = Table::init(vec![row]);
                    table.set_format(**table_format);
                    table
                }
                2 => tensor_2d(self, 0, 0, table_format),
                _ => {
                    if n % 2 == 1 {
                        let row = tensor_odd_d(n, self, 0, 0, table_format);
                        let mut table = Table::init(vec![row]);
                        table.set_format(**table_format);
                        table
                    } else {
                        tensor_even_d(n, self, 0, 0, table_format)
                    }
                }
            };
            table.set_format(**table_format);
            write!(f, "{}", table)?;
        }
        let _ = write!(f, "{:?}", self);
        writeln!(f)
    }
}

fn tensor_1d<T>(tensor: &Tensor<T>) -> Row
where
    T: Copy,
    T: Display,
{
    let sizes = tensor.sizes();
    let strides = tensor.strides();

    (0..sizes[0])
        .map(|index| {
            let index = index * strides[0] + tensor.offset();
            let element = tensor.data[index];
            let element_format = &format!("{:.2}", element);
            Cell::from(&element_format)
        })
        .collect()
}

fn tensor_2d<T>(
    tensor: &Tensor<T>,
    stride_offset: usize,
    dimension_offset: usize,
    table_format: &TableFormat,
) -> Table
where
    T: Copy,
    T: Display,
{
    let sizes = tensor.sizes();
    let strides = tensor.strides();
    let tensor = tensor.ravel();

    let rows = (0..sizes[dimension_offset])
        .map(|index_0| {
            (0..sizes[dimension_offset + 1])
                .map(|index1| {
                    let index = index_0 * strides[dimension_offset]
                        + index1 * strides[dimension_offset + 1]
                        + tensor.offset()
                        + stride_offset;
                    let element = tensor.data[index];
                    let element_format = &format!("{:.2}", element);
                    Cell::from(&element_format)
                })
                .collect()
        })
        .collect();

    let mut table = Table::init(rows);
    table.set_format(*table_format);
    table
}

fn tensor_odd_d<T>(
    n: usize,
    tensor: &Tensor<T>,
    stride_offset: usize,
    dimension_offset: usize,
    table_format: &TableFormat,
) -> Row
where
    T: Copy,
    T: Display,
{
    let size = tensor.sizes()[dimension_offset];
    let stride = tensor.strides()[dimension_offset];

    let row: Vec<Table> = if n == 3 {
        (0..size)
            .map(|index| {
                let offset = index * stride + stride_offset;
                tensor_2d(tensor, offset, dimension_offset + 1, table_format)
            })
            .collect()
    } else {
        (0..size)
            .map(|index| {
                let offset = index * stride + stride_offset;
                tensor_even_d(n - 1, tensor, offset, dimension_offset + 1, table_format)
            })
            .collect()
    };

    Row::from(row)
}

fn tensor_even_d<T>(
    n: usize,
    tensor: &Tensor<T>,
    stride_offset: usize,
    dimension_offset: usize,
    table_format: &TableFormat,
) -> Table
where
    T: Copy,
    T: Display,
{
    let sizes = tensor.sizes();
    let strides = tensor.strides();

    let rows = (0..sizes[dimension_offset])
        .map(|index| {
            let offset = index * strides[dimension_offset] + stride_offset;
            tensor_odd_d(n - 1, tensor, offset, dimension_offset + 1, table_format)
        })
        .collect();

    let mut table = Table::init(rows);
    table.set_format(*table_format);
    table
}