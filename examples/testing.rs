use anyhow::Result;
use venum::Tensor;

fn main() -> Result<()> {
    let t = Tensor::arange(0, 27, 1)?.view(&[3, 3, 3])?.flip(&[0])?;
    println!("{}", t);

    Ok(())
}
