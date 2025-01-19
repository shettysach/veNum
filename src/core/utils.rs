use crate::core::errors::UsizeCastError;
use num_traits::FromPrimitive;
use std::any::type_name;

pub(crate) fn cast_usize<T>(value: usize) -> Result<T, UsizeCastError>
where
    T: FromPrimitive,
{
    T::from_usize(value).ok_or(UsizeCastError {
        value,
        dtype: type_name::<T>(),
    })
}
