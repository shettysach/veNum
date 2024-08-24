use venum::{Res, Tensor};

fn main() -> Res<()> {
    let a = Tensor::arange(0, 30, 1)?.view(&[5, 3, 2])?;
    let b = Tensor::arange(0, 30, 1)?.view(&[5, 2, 3])?;

    println!("a\n{}", a);
    println!("b\n{}", b);

    for _i in 0..10 {
        let now = std::time::Instant::now();
        let c = &a.matmul(&b)?;
        let end = now.elapsed();

        println!("a × b\n{}", c);
        println!("{:.?}\n", end);
    }

    Ok(())
}
