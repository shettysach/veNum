use venum::Tensor;

fn main() {
    let a = Tensor::arange(1, 28, 1).view(&[3, 3, 3]);
    println!("{}", a);
    println!("{}", a.slice(&[(1, 0), (0, 2)]));

    let a_flipped = a.flip(&[0, 1, 2]);
    println!("{}", a_flipped);
    println!("{}", a_flipped.slice(&[(1, 0), (0, 2)]));

    let a = Tensor::arange(1, 19, 1).view(&[3, 6]);
    println!("{}", a);
    println!("{}", a.slice(&[(1, 0), (0, 2)]));

    let a_flipped = a.flip(&[0, 1, 2]);
    println!("{}", a_flipped);
    println!("{}", a_flipped.slice(&[(1, 0), (0, 2)]));
}
