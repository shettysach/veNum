use anyhow::Result;
use venum::{conv::Mode, Tensor};

fn main() -> Result<()> {
    let input = Tensor::arange(0.0, 32.0, 1.0)?.view(&[2, 4, 4])?;
    let kernel = Tensor::ones(4)?.view(&[2, 2])?;
    let strides = &[1, 1];

    let valid = input.correlate_2d(&kernel, strides, Mode::Valid)?;
    println!("Valid: \n{}", valid);

    let full = input.correlate_2d(&kernel, strides, Mode::Full)?;
    println!("Full: \n{}", full);

    let same = input.correlate_2d(&kernel, strides, Mode::Same)?;
    println!("Same: \n{}", same);

    Ok(())
}
