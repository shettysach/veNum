use crate::Tensor;
use prettytable::{
    format::{self, TableFormat},
    {Cell, Row, Table},
};
use std::{
    any::type_name,
    fmt::{Debug, Display, Formatter, Result},
};

impl<T: Debug + Copy> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Tensor")
            .field("dtype", &type_name::<T>())
            .field("dims", &self.numdims())
            .field("elems", &self.numel())
            .field("shape", &self.sizes())
            .finish()
    }
}

impl<T: Display + Debug + Copy> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let n = self.shape.numdims();

        if (1..=8).contains(&n) {
            let table_format = &format::consts::FORMAT_BOX_CHARS;

            let table = if n % 2 == 1 {
                let row = odd_dimensions(n, self, 0, table_format);
                let mut table = Table::init(vec![row]);
                table.set_format(**table_format);
                table
            } else {
                even_dimensions(n, self, 0, table_format)
            };

            write!(f, "{}", table)?;
        }

        let _ = write!(f, "{:?}", self);
        writeln!(f)
    }
}

fn odd_dimensions<T>(
    n: usize,
    tensor: &Tensor<T>,
    stride_offset: usize,
    table_format: &TableFormat,
) -> Row
where
    T: Copy + Display,
{
    let dim = tensor.numdims() - n;
    let size = tensor.sizes()[dim];
    let stride = tensor.strides()[dim];

    if n == 1 {
        let offset = tensor.offset() + stride_offset;
        Row::from(
            (0..size)
                .map(|index| {
                    let index = stride.offset(index, size) + offset;
                    let element = tensor.data[index];
                    let element = &format!("{:.2}", element);
                    Cell::from(&element)
                })
                .collect::<Vec<Cell>>(),
        )
    } else {
        Row::from(
            (0..size)
                .map(|index| {
                    let offset = stride.offset(index, size) + stride_offset;
                    even_dimensions(n - 1, tensor, offset, table_format)
                })
                .collect::<Vec<Table>>(),
        )
    }
}

fn even_dimensions<T>(
    n: usize,
    tensor: &Tensor<T>,
    stride_offset: usize,
    table_format: &TableFormat,
) -> Table
where
    T: Copy + Display,
{
    let dim = tensor.numdims() - n;
    let size = tensor.sizes()[dim];
    let stride = tensor.strides()[dim];

    let rows = (0..size)
        .map(|index| {
            let offset = stride.offset(index, size) + stride_offset;
            odd_dimensions(n - 1, tensor, offset, table_format)
        })
        .collect();

    let mut table = Table::init(rows);
    table.set_format(*table_format);
    table
}
