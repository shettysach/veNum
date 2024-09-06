use venum::{Res, Tensor};

fn main() -> Res<()> {
    let tensor = Tensor::arange(1, 9, 1)?.view(&[2, 2, 2])?;
    println!("{}", tensor);

    let padded = tensor.pad(0, &[(1, 1)])?;
    println!("{}", padded);

    let padded_dimension = tensor.pad_dimensions(0, &[2], &[(1, 1)])?;
    println!("{}", padded_dimension);

    Ok(())
}
