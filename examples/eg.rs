use venum::Tensor;

fn main() {
    let tensor = Tensor::<i32>::arange(0, 4 * 5 * 6, 1)
        .unwrap()
        .reshape(&[4, 5, 6])
        .unwrap();
    println!("{}", tensor);

    let sums = tensor.sum_dimensions(&[0]).unwrap().squeeze().unwrap();
    println!("{}", sums);
}
