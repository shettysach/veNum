use venum::{Res, Tensor};

fn main() -> Res<()> {
    let a = Tensor::arange(0.0, 120.0, 1.0)?.view(&[5, 2, 2, 3, 2])?;
    let b = Tensor::arange(0.0, 6.0, 1.0)?.view(&[2, 3])?;
    println!("{}", b);

    for _ in 0..10 {
        let now = std::time::Instant::now();

        let _c = &a.matmul(&b)?;

        let end = now.elapsed();
        println!("{:?}", end);
    }

    Ok(())
}
