use std::error::Error;
use venum::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let a = Tensor::arange(0, 1_000_000, 1)?.view(&[1000, 1000])?;
    let b = Tensor::arange(0, 1_000_000, 1)?.view(&[1000, 1000])?;

    for _ in 0..10 {
        let now = std::time::Instant::now();

        let _c = &a.matmul(&b)?;

        let end = now.elapsed();
        println!("{:?}", end);
    }

    Ok(())
}
