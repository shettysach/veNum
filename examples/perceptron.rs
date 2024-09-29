use std::error::Error;
use venum::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let x = Tensor::<f32>::new_1d(&[1.0, 0.0, 1.0, 0.0])?;
    let w = Tensor::<f32>::new_1d(&[0.2, 0.3, -0.4, -0.1])?;
    let b = 0.1;

    let z = ((w * x)? + b)?;
    let y = z.unary_map(|v| if v > 0.0 { 1 } else { 0 })?;

    println!("{}", y);

    Ok(())
}
