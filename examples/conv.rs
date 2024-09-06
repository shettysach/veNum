use std::ops::Div;
use venum::{Res, Slide, Tensor};

fn main() -> Res<()> {
    //let input = Tensor::<f32>::arange(0.0, 9.0, 1.0)?.view(&[3, 3])?;
    let input = Tensor::<f32>::arange(0.0, 4.0, 1.0)?.view(&[2, 2])?;

    let kernel = Tensor::<f32>::new(
        &[
            1.0, 2.0, 1.0, // Gaussian
            2.0, 4.0, 2.0, // Blur
            1.0, 2.0, 1.0, // Kernel
        ],
        &[3, 3],
    )?
    .div(16.0)?;

    //let valid = input.correlate_2d(&kernel, Slide::Valid)?.squeeze()?;
    //println!("Valid: \n{}", valid);

    let full = input.correlate_2d(&kernel, Slide::Full)?.squeeze()?;
    println!("Full: \n{}", full);

    let same = input.correlate_2d(&kernel, Slide::Same)?.squeeze()?;
    println!("Same: \n{}", same);

    Ok(())
}
