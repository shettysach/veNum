/*!
```console
               __
__   _____  /\ \ \_   _ _ __ ___
\ \ / / _ \/  \/ / | | | '_ ` _ \
 \ V /  __/ /\  /| |_| | | | | | |
  \_/ \___\_\ \/  \__,_|_| |_| |_|
```

Vectorized N-dimensional numerical arrays.
*/

mod core;
pub use core::Slide;
pub use core::Tensor;
pub type Res<U> = Result<U, String>;
