use anyhow::Result;
use venum::{conv::Mode, Tensor};

fn main() -> Result<()> {
    let input = Tensor::arange(0.0, 32.0, 1.0)?.view(&[2, 4, 4])?;
    let kernel = Tensor::ones(4)?.view(&[2, 2])?;
    let strides = &[1, 1];

    println!("Input: \n{}", input);
    println!("Kernel: \n{}", kernel);

    let same_conv = input.correlate_2d(&kernel, strides, Mode::Same)?;
    println!("Same conv: \n{}", same_conv);

    let pool = &[2, 2];
    println!("Pool sizes: \n{:?}\n", pool);

    let max_pool = same_conv.pool_2d(Tensor::max_dims, pool, strides, Mode::Valid, true)?;
    println!("Max pool: \n{}", max_pool);

    let avg_pool = same_conv.pool_2d(Tensor::mean_dims, pool, strides, Mode::Valid, true)?;
    println!("Avg pool: \n{}", avg_pool);

    Ok(())
}
