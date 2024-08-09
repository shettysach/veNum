use crate::Tensor;

pub fn print_2d<T>(a: &Tensor<T>)
where
    T: std::default::Default,
    T: std::fmt::Display,
    T: std::marker::Copy,
{
    for i in 0..a.sizes()[0] {
        for j in 0..a.sizes()[1] {
            print!(" {:5.2} ", a.index_element(&[i, j]))
        }
        println!()
    }
    println!();
}
