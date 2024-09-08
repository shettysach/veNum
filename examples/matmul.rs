use venum::{Res, Tensor};

fn main() -> Res<()> {
    let a = Tensor::arange(0, 48, 1)?.view(&[4, 2, 3, 2])?;
    let b = Tensor::arange(0, 12, 1)?.view(&[2, 2, 3])?;
    println!("{}", a);
    println!("{}", b);

    let c = &a.matmul(&b)?;
    println!("{}", c);

    Ok(())
}
