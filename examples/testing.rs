use venum::Tensor;

fn main() -> anyhow::Result<()> {
    let a = Tensor::arange(0, 27, 1)?.view(&[3, 3, 3])?;
    println!("{}", a);

    let x = a.index_dims(&[2], &[3])?;
    println!("{x}");

    Ok(())
}
