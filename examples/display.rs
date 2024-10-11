use anyhow::Result;
use venum::Tensor;

fn main() -> Result<()> {
    let tensor_1d = Tensor::<u16>::linspace(0, 2, 3)?;
    println!("{}", tensor_1d);

    let tensor_2d = Tensor::<f32>::linspace(0.0, 8.0, 9)?.view(&[3, 3])?;
    println!("{}", tensor_2d);

    let tensor_3d = Tensor::<i32>::linspace(0, 26, 27)?.view(&[3, 3, 3])?;
    println!("{}", tensor_3d);

    let tensor_4d = Tensor::<u32>::linspace(0, 80, 81)?.view(&[3; 4])?;
    println!("{}", tensor_4d);

    Ok(())
}
