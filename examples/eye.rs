use venum::{Res, Tensor};

fn main() -> Res<()> {
    let tensor = Tensor::arange(0, 256, 1)?.view(&[3, 4, 4, 4])?;
    let eye = Tensor::eye(4)?;
    let res = &tensor.matmul(&eye)?;

    println!("{}", tensor);
    println!("{}", eye);
    println!("{}", res);

    Ok(())
}
