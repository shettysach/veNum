pub trait One {
    fn one() -> Self;
}

macro_rules! one {
    ($type:ident, $one:tt) => {
        impl One for $type {
            fn one() -> $type {
                $one
            }
        }
    };
}

one!(u8, 1);
one!(u16, 1);
one!(u32, 1);
one!(u64, 1);
one!(usize, 1);

one!(i8, 1);
one!(i16, 1);
one!(i32, 1);
one!(i64, 1);
one!(isize, 1);

one!(f32, 1.0);
one!(f64, 1.0);
one!(bool, true);
