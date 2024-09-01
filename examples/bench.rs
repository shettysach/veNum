use venum::{Res, Tensor};

fn main() -> Res<()> {
    let a = Tensor::arange(0, 120, 1)?.view(&[5, 2, 2, 3, 2])?;
    let b = Tensor::arange(0, 6, 1)?.view(&[2, 3])?;

    for _ in 0..10 {
        let now = std::time::Instant::now();

        let _c = &a.matmul(&b)?;

        let end = now.elapsed();
        println!("{}", _c);
        println!("{:?}", end);
    }
    Ok(())
}
