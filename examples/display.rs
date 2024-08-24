use venum::{Res, Tensor};

fn main() -> Res<()> {
    let now = std::time::Instant::now();

    let tensor_1d = Tensor::<u16>::linspace(0, 2, 3)?;
    println!("{}", tensor_1d);

    let tensor_2d = Tensor::<f32>::linspace(0.0, 8.0, 9)?.reshape(&[3, 3])?;
    println!("{}", tensor_2d);

    let tensor_3d = Tensor::<i32>::linspace(0, 26, 27)?.reshape(&[3, 3, 3])?;
    println!("{}", tensor_3d);

    let tensor_4d = Tensor::<u32>::linspace(0, 80, 81)?.reshape(&[3; 4])?;
    println!("{}", tensor_4d);

    let end = now.elapsed();
    println!("{:?}", end);

    Ok(())
}
