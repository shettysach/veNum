use venum::Tensor;

fn main() {
    let a = Tensor::linspace(0, 8, 4).reshape(&[2, 2, 1]);
    let b = Tensor::linspace(0, 8, 4).reshape(&[1, 2, 2]);
    println!("{}", &a);
    println!("{}", &b);

    let c = a + b;
    println!("{}", &c);
}
