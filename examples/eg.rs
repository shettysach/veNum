use venum::{utils::print_2d, Tensor};

fn main() {
    let a = Tensor::<f32>::linspace(1.0, 1.5, 5).reshape(&[5, 1]);
    println!("a");
    print_2d(&a);

    let b = Tensor::<f32>::new(&[5.0, 10.0, 15.0], &[1, 3]);
    println!("b");
    print_2d(&b);

    println!("c");
    let c = &a / &b;
    print_2d(&c);
}
