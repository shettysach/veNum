use venum::Tensor;

fn main() {
    let a = Tensor::linspace(1, 5, 5).slice(&[(2, 5)]);
    println!("a");
    println!("{}", &a);

    let b = Tensor::linspace(1, 5, 5).reshape(&[5, 1]).slice(&[(1, 4)]);
    println!("b");
    println!("{}", &b);

    println!("a * b");
    let prod = &a * &b;
    println!("{}", &prod);
}
