use venum::Tensor;

fn main() {
    let tensor_3d = Tensor::<i32>::linspace(0, 26, 27).reshape(&[3, 3, 3]);
    println!("{}", tensor_3d);

    let sums = tensor_3d.sum_dimensions(&[0, 1]);
    println!("{}", sums);
}
