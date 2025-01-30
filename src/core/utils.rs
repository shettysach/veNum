use anyhow::Result;
use num_traits::FromPrimitive;
use std::any::type_name;

use crate::core::errors::UsizeCastError;

pub(crate) fn cast_to_usize<T>(value: usize) -> Result<T>
where
    T: FromPrimitive,
{
    T::from_usize(value).ok_or(
        UsizeCastError {
            value,
            dtype: type_name::<T>(),
        }
        .into(),
    )
}
