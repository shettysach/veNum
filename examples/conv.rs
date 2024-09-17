use venum::{conv::Mode, Res, Tensor};

fn main() -> Res<()> {
    let input = Tensor::arange(0, 32, 1)?.view(&[2, 4, 4])?;
    let kernel = Tensor::ones(9)?.view(&[3, 3])?;

    let valid = input.correlate_2d(&kernel, Mode::Valid)?;
    println!("Valid: \n{}", valid);

    let full = input.correlate_2d(&kernel, Mode::Full)?;
    println!("Full: \n{}", full);

    let same = input.correlate_2d(&kernel, Mode::Same)?;
    println!("Same: \n{}", same);

    Ok(())
}
