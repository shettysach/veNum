use venum::Tensor;

fn main() {
    let tensor_0d = Tensor::<u8>::new(&[], &[0]);
    println!("{}", tensor_0d);

    let tensor_1d = Tensor::<u16>::linspace(0, 2, 3);
    println!("{}", tensor_1d);

    let tensor_2d = Tensor::<f32>::linspace(0.0, 8.0, 9).reshape(&[3, 3]);
    println!("{}", tensor_2d);

    let tensor_3d = Tensor::<i32>::linspace(0, 26, 27).reshape(&[3, 3, 3]);
    println!("{}", tensor_3d);

    let tensor_4d = Tensor::<u32>::linspace(0, 80, 81).reshape(&[3; 4]);
    println!("{}", tensor_4d);

    let tensor_6d = Tensor::<u8>::new(&Vec::from_iter(0..64), &[2; 6]);
    println!("{}", tensor_6d);
}
