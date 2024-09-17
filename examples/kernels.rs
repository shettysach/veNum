use image::{ImageBuffer, ImageError, RgbImage};
use std::{error::Error, fmt::Display, ops::Div, path::Path};
use venum::{conv::Mode, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let (image, width, height) = read_image("assets/venom.png")?;
    let image_tensor = Tensor::new(&image, &[3, height as usize, width as usize])?;

    // Gaussian blur

    let blur_kernel = Tensor::new(
        &[
            1.0, 2.0, 1.0, // Gaussian
            2.0, 4.0, 2.0, // Blur
            1.0, 2.0, 1.0, // Kernel
        ],
        &[3, 3],
    )?
    .div(16.0)?;

    let blur = image_tensor.correlate_2d(&blur_kernel, Mode::Same)?;
    write_image("assets/blur.png", width, height, &blur.data())?;

    // Sharpen

    let sharpen_kernel = Tensor::new(
        &[
            0.0, -1.0, 0.0, // Sharpen
            -1.0, 5.0, -1.0, // Features
            0.0, -1.0, 0.0, // Kernel
        ],
        &[3, 3],
    )?;

    let sharpen = image_tensor.correlate_2d(&sharpen_kernel, Mode::Same)?;
    write_image("assets/sharpen.png", width, height, &sharpen.data())?;

    // Edge

    let edge_kernel = Tensor::new(
        &[
            0.0, -1.0, 0.0, // Laplacian
            -1.0, 4.0, -1.0, // Edge Detection
            0.0, -1.0, 0.0, // Kernel
        ],
        &[3, 3],
    )?;

    let edge = image_tensor.correlate_2d(&edge_kernel, Mode::Same)?;
    write_image("assets/edge.png", width, height, &edge.data())?;

    Ok(())
}

fn read_image<P>(path: P) -> Result<(Vec<f32>, u32, u32), ImageError>
where
    P: AsRef<Path>,
{
    let img = image::open(path)?;
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let channel_size = (width * height) as usize;
    let channel_size_2 = channel_size * 2;
    let mut f32_data = vec![0.0; channel_size * 3];

    for (i, pixel) in rgb_img.pixels().enumerate() {
        f32_data[i] = pixel[0] as f32;
        f32_data[i + channel_size] = pixel[1] as f32;
        f32_data[i + channel_size_2] = pixel[2] as f32;
    }

    Ok((f32_data, width, height))
}

fn write_image<P>(path: P, width: u32, height: u32, data: &[f32]) -> Result<(), ImageError>
where
    P: AsRef<Path> + Display,
{
    let channel_size = (width * height) as usize;
    let channel_size_2 = channel_size * 2;
    let mut u8_data = Vec::with_capacity(channel_size * 3);

    for i in 0..channel_size {
        u8_data.push(data[i] as u8);
        u8_data.push(data[i + channel_size] as u8);
        u8_data.push(data[i + channel_size_2] as u8);
    }

    let img: RgbImage =
        ImageBuffer::from_raw(width, height, u8_data).expect("Error saving output image.");

    println!("Image saved at {}", path);
    img.save(path)?;

    Ok(())
}