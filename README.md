<div align="center">
    <img src="assets/icon.png" width="33%">
</div>

- Stands for vectorized N-dimensional numerical arrays. Tensor / Nd Array library.
- Currently capable of creating CPU Tensors of type T and performing 
    - broadcasted algebraic operations
    - Nd matrix multiplication (naive)
    - 1d and 2d convolution / cross-correlation (naive) with strides
    - reduction operations such as sum, product, max, min
    - transformations such as view/reshape, permute/transpose, flip, expand, pad, slice, squeeze, unsqueeze

- Clone the repo and run examples
```bash
cargo run -r --example <example_name>
```
- Use as library
```bash
cargo add --git https://github.com/shettysach/veNum
```

##### credits

- [kurtschelfthout/tensorken](https://github.com/kurtschelfthout/tensorken)
- [huggingface/candle](https://github.com/huggingface/candle)
- [minitorch/minitorch](https://github.com/minitorch/minitorch)
- [nreHieW/r-nn](https://github.com/nreHieW/r-nn)
- [assets/venom.png](https://www.reddit.com/r/metalgearsolid/comments/2xn8f2/i_heard_yall_like_sprites/)

##### resources

- [pytorch-internals by ezyang](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [tensorken articles by Kurt Schelfthout](https://getcode.substack.com/p/fun-and-hackable-tensors-in-rust)
- [MiniTorch](https://minitorch.github.io/)
- [Convolutions](https://youtu.be/Lakz2MoHy6o?si=hsYi2IzxUwv3LOkW)
