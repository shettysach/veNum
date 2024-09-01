use venum::{Res, Tensor};

fn main() -> Res<()> {
    let a = Tensor::linspace(1, 5, 5)?;
    println!("a");
    println!("{}", &a);

    let b = Tensor::linspace(1, 5, 5)?.reshape(&[5, 1])?;
    println!("b");
    println!("{}", &b);

    let prod = (&a * &b)?;
    println!("a * b");
    println!("{}", &prod);

    Ok(())
}
