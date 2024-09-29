use std::error::Error;
use venum::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let tensor = Tensor::arange(0, 256, 1)?.view(&[4, 4, 4, 4])?;
    let eye = Tensor::eye(4)?;
    let res = &tensor.matmul(&eye)?;

    println!("{}", tensor);
    println!("{}", eye);
    println!("{}", res);

    Ok(())
}
