use anyhow::Result;
use venum::Tensor;

fn main() -> Result<()> {
    let tensor = Tensor::arange(0, 9, 1)?.view(&[3, 3])?.flip(&[0])?;
    println!("{}", tensor);

    Ok(())
}
