use venum::{Res, Tensor};

fn main() -> Res<()> {
    let a = Tensor::new_1d(&[true, true, false])?;
    let b = Tensor::new_1d(&[false, true, false])?;

    let c = a.binary_tensor_map(&b, |l, r| l || r)?;
    let d = a.binary_tensor_map(&b, |l, r| l && r)?;

    println!("{}", c);
    println!("{}", d);

    Ok(())
}
