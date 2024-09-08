use venum::{Res, Slide, Tensor};

fn main() -> Res<()> {
    let input = Tensor::arange(0.0, 32.0, 1.0)?.view(&[2, 4, 4])?;
    let kernel = Tensor::ones(&[4, 4])?;

    let valid = input.correlate_2d(&kernel, Slide::Valid)?;
    println!("Valid: \n{}", valid);

    let full = input.correlate_2d(&kernel, Slide::Full)?;
    println!("Full: \n{}", full);

    let same = input.correlate_2d(&kernel, Slide::Same)?;
    println!("Same: \n{}", same);

    Ok(())
}
