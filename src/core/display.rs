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
            .field("dims", &self.rank())
            .field("elems", &self.numel())
            .field("shape", &self.sizes())
            .finish()
    }
}

impl<T: Display + Debug + Copy> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let n = self.shape.rank();

        if (1..=8).contains(&n) {
            let style = &format::consts::FORMAT_BOX_CHARS;
            let precision = 2;

            let table = if n % 2 == 1 {
                let row = odd_dimensions(n, self, 0, style, precision);
                let mut table = Table::init(vec![row]);
                table.set_format(**style);
                table
            } else {
                even_dimensions(n, self, 0, style, precision)
            };

            write!(f, "{}", table)?;
        }

        writeln!(f, "{:?}", self)
    }
}

fn odd_dimensions<T>(
    n: usize,
    tensor: &Tensor<T>,
    stride_offset: usize,
    style: &TableFormat,
    precision: usize,
) -> Row
where
    T: Copy + Display,
{
    let dim = tensor.rank() - n;
    let size = tensor.sizes()[dim];
    let stride = tensor.strides()[dim];

    if n == 1 {
        let offset = tensor.offset() + stride_offset;
        Row::from(
            (0..size)
                .map(|index| {
                    let index = stride.offset(index, size) + offset;
                    let element = tensor.data[index];
                    let element = &format!("{:.precision$}", element);
                    Cell::from(&element)
                })
                .collect::<Vec<Cell>>(),
        )
    } else {
        Row::from(
            (0..size)
                .map(|index| {
                    let offset = stride.offset(index, size) + stride_offset;
                    even_dimensions(n - 1, tensor, offset, style, precision)
                })
                .collect::<Vec<Table>>(),
        )
    }
}

fn even_dimensions<T>(
    n: usize,
    tensor: &Tensor<T>,
    stride_offset: usize,
    style: &TableFormat,
    precision: usize,
) -> Table
where
    T: Copy + Display,
{
    let dim = tensor.rank() - n;
    let size = tensor.sizes()[dim];
    let stride = tensor.strides()[dim];

    let rows = (0..size)
        .map(|index| {
            let offset = stride.offset(index, size) + stride_offset;
            odd_dimensions(n - 1, tensor, offset, style, precision)
        })
        .collect();

    let mut table = Table::init(rows);
    table.set_format(*style);
    table
}
