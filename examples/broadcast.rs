use std::error::Error;
use venum::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let a = Tensor::linspace(1, 5, 5)?;
    println!("a");
    println!("{}", &a);

    let b = Tensor::linspace(1, 5, 5)?.view(&[5, 1])?;
    println!("b");
    println!("{}", &b);

    let prod = (&a * &b)?;
    println!("a * b");
    println!("{}", &prod);

    Ok(())
}
