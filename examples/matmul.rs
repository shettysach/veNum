use venum::{Res, Tensor};

fn main() -> Res<()> {
    let a = Tensor::arange(0, 30, 1)?.view(&[5, 3, 2])?;
    let b = Tensor::arange(0, 6, 1)?.view(&[2, 3])?;
    println!("{}", a);
    println!("{}", b);

    let c = &a.matmul(&b)?;
    println!("{}", c);

    Ok(())
}
