use venum::Tensor;

fn main() -> anyhow::Result<()> {
    let t = Tensor::arange(0, 27, 1)?.view(&[3, 3, 3])?.flip(&[0])?;
    println!("{}", t);

    let z = Tensor::arange(0, 9, 1)?.view(&[3, 3])?;
    println!("{}", z);

    println!("{}", t.matmul(&z)?);

    Ok(())
}
