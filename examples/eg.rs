use venum::Tensor;

fn main() {
    let a = Tensor::arange(1, 28, 1).view(&[3, 3, 3]);
    let a_flipped = a.flip(&[0, 1, 2]);

    println!("{}", a);
    println!("{}", a_flipped);

    let a2 = a_flipped.to_contiguous().slice(&[(0, 2), (0, 2), (0, 2)]);
    println!("{}", a2);
}
