use crate::core::errors::UsizeCastError;
use num_traits::FromPrimitive;
use prettytable::{format::TableFormat, Table};
use std::any::type_name;

/// Type alias for ease of use.
pub(crate) type Res<U> = Result<U, Box<dyn std::error::Error>>;

pub(crate) fn cast_usize<T>(value: usize) -> Result<T, UsizeCastError>
where
    T: FromPrimitive,
{
    T::from_usize(value).ok_or(UsizeCastError {
        value,
        dtype: type_name::<T>(),
    })
}

pub(crate) trait WithStyle {
    fn with_style(self, style: &TableFormat) -> Self;
}

impl WithStyle for Table {
    fn with_style(mut self, table_format: &TableFormat) -> Table {
        self.set_format(*table_format);
        self
    }
}
