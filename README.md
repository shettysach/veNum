<div align="center">
    <img src="assets/icon.png" width="33%">
</div>

- Stands for vectorized N-dimensional numerical arrays. Tensor / NdArray library.
- Currently capable of creating CPU Tensors of type T and performing 
    - broadcasted addition, subtraction multiplication and division
    - Nd matrix multiplication
    - 1d and 2d convolution / correlation.
- Capable of performing transformations such as 
    - viewing / reshaping
    - expanding 
    - padding
    - squeezing, unsqueezing 
    - flipping / reversing 
    - permutating / transposing
    - indexing and slicing 
- Plan on adding autograd, support for other backends and expanding the library.

- Clone the repo and run examples
```bash
cargo run --example <example_name> --release
```

##### credits

- [kurtschelfthout/tensorken](https://github.com/kurtschelfthout/tensorken)
- [huggingface/candle](https://github.com/huggingface/candle)
- [minitorch/minitorch](https://github.com/minitorch/minitorch)
- [nreHieW/r-nn](https://github.com/nreHieW/r-nn)
- [assets/image.png](https://pixelblock.tumblr.com/post/33847269942/solid-snake-the-legendary-mercenary-former)

##### resources

- [pytorch-internals by ezyang](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [tensorken articles by Kurt Schelfthout](https://getcode.substack.com/p/fun-and-hackable-tensors-in-rust)
- [MiniTorch](https://minitorch.github.io/)
- [Convolutions](https://youtu.be/Lakz2MoHy6o?si=hsYi2IzxUwv3LOkW)
