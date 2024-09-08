use image::{ImageBuffer, ImageError, RgbImage};
use std::{error::Error, fmt::Display, ops::Div, path::Path};
use venum::{Slide, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let (image, width, height) = read_image("assets/image.png")?;
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

    let blur = image_tensor.correlate_2d(&blur_kernel, Slide::Same)?;
    write_image("assets/blur.png", width, height, blur.data_contiguous())?;

    // Sharpen

    let sharpen_kernel = Tensor::new(
        &[
            0.0, -1.0, 0.0, // Sharpen
            -1.0, 5.0, -1.0, // Features
            0.0, -1.0, 0.0, // Kernel
        ],
        &[3, 3],
    )?;

    let sharpen = image_tensor.correlate_2d(&sharpen_kernel, Slide::Same)?;
    write_image(
        "assets/sharpen.png",
        width,
        height,
        sharpen.data_contiguous(),
    )?;

    // Edge

    let edge_kernel = Tensor::new(
        &[
            0.0, -1.0, 0.0, // Laplacian
            -1.0, 4.0, -1.0, // Edge Detection
            0.0, -1.0, 0.0, // Kernel
        ],
        &[3, 3],
    )?;

    let edge = image_tensor.correlate_2d(&edge_kernel, Slide::Same)?;
    write_image("assets/edge.png", width, height, edge.data_contiguous())?;

    // Colours

    let red: Vec<f32> = [[1.0; 9], [0.0; 9], [0.0; 9]].concat();
    let green: Vec<f32> = [[0.0; 9], [1.0; 9], [0.0; 9]].concat();
    let blue: Vec<f32> = [[0.0; 9], [0.0; 9], [1.0; 9]].concat();

    for (i, colour) in [&red, &green, &blue].iter().enumerate() {
        let kernel = Tensor::new(colour, &[3, 3, 3])?;

        let colour = image_tensor.correlate_2d(&kernel, Slide::Same)?;
        write_image(
            format!("assets/colour_{i}.png"),
            width,
            height,
            colour.data_contiguous(),
        )?;
    }

    Ok(())
}

fn read_image<P>(path: P) -> Result<(Vec<f32>, u32, u32), ImageError>
where
    P: AsRef<Path>,
{
    let img = image::open(path)?;
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let mut r_channel = Vec::with_capacity((width * height) as usize);
    let mut g_channel = Vec::with_capacity((width * height) as usize);
    let mut b_channel = Vec::with_capacity((width * height) as usize);

    for pixel in rgb_img.pixels() {
        r_channel.push(pixel[0] as f32); // Red
        g_channel.push(pixel[1] as f32); // Green
        b_channel.push(pixel[2] as f32); // Blue
    }

    let mut output = Vec::with_capacity((width * height * 3) as usize);
    output.extend(r_channel);
    output.extend(g_channel);
    output.extend(b_channel);

    Ok((output, width, height))
}

fn write_image<P>(path: P, width: u32, height: u32, data: &[f32]) -> Result<(), ImageError>
where
    P: AsRef<Path> + Display,
{
    let mut u8_data = Vec::with_capacity((width * height * 3) as usize);
    let chunk_size = data.len() / 3;

    for i in 0..chunk_size {
        u8_data.push(data[i] as u8);
        u8_data.push(data[i + chunk_size] as u8);
        u8_data.push(data[i + 2 * chunk_size] as u8);
    }

    let img: RgbImage =
        ImageBuffer::from_raw(width, height, u8_data).expect("Error saving output image.");

    println!("Image saved as {}", path);
    img.save(path)?;

    Ok(())
}
