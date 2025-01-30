use thiserror::Error;

// --- Shape ---

#[derive(Error, Debug)]
#[error("Data length ({data_length}) does not match size of tensor ({tensor_size}).")]
pub(crate) struct InvalidDataLengthError {
    pub data_length: usize,
    pub tensor_size: usize,
}

#[derive(Error, Debug)]
#[error("Tensor of shape {current_shape:?} cannot be viewed/reshaped to {new_shape:?}.")]
pub(crate) struct ReshapeError {
    pub current_shape: Vec<usize>,
    pub new_shape: Vec<usize>,
}

#[derive(Error, Debug)]
pub(crate) enum EmptyTensorError {
    #[error("Strides are empty. Unable to view.")]
    View,

    #[error("Strides are empty. Unable to slice.")]
    Slice,

    #[error("Empty tensor. No max.")]
    ReduceMax,

    #[error("Empty tensor. No min.")]
    ReduceMin,
}

#[derive(Error, Debug)]
#[error("Shape is not contiguous. Use `to_contiguous()` or an alternate function.")]
pub(crate) struct NonContiguousError;

#[derive(Error, Debug)]
#[error("Size {size} cannot be expaned to size {expansion}. To be expanded, size should be 1.")]
pub(crate) struct ExpansionError {
    pub size: usize,
    pub expansion: usize,
}

#[derive(Error, Debug)]
#[error("Current rank ({current}) is greater than unsqueezed rank ({unsqueezed}).")]
pub(crate) struct UnsqueezeError {
    pub current: usize,
    pub unsqueezed: usize,
}

#[derive(Error, Debug)]
#[error("Transpose requires at least two dimensions.")]
pub(crate) struct TransposeError;

#[derive(Error, Debug)]
#[error("Shapes {lhs_sizes:?} and {rhs_sizes:?} cannot broadcasted together.")]
pub(crate) struct BroadcastError {
    pub lhs_sizes: Vec<usize>,
    pub rhs_sizes: Vec<usize>,
}

// --- Index, Range, Dims ---

#[derive(Error, Debug)]
pub(crate) enum IndexError {
    #[error("Index {index:?} is out of range for dimension {dimension}, of size {size}.")]
    OutOfRange {
        index: usize,
        dimension: usize,
        size: usize,
    },

    #[error("Number of indices ({num_indices}) does not match the number of dimensions {num_dimensions}.")]
    IndicesLength {
        num_indices: usize,
        num_dimensions: usize,
    },
}

#[derive(Error, Debug)]
pub(crate) enum DimensionError {
    #[error("Dimension {dimension} is greater than max range of dimensions, {dim_range}.")]
    OutOfRange { dimension: usize, dim_range: usize },

    #[error("Dimension {0} repeats.")]
    Repetition(usize),
}

#[derive(Error, Debug)]
pub(crate) enum RangeError {
    #[error("{range:?} is out of range for dimension {dimension}, of size {size}.")]
    OutOfRange {
        range: (usize, usize),
        dimension: usize,
        size: usize,
    },

    #[error("Range start index {0} is greater than range end index {1}.")]
    GreaterStartRange(usize, usize),
}

// --- Matmul ---

#[derive(Error, Debug)]
pub(crate) enum MatmulShapeError {
    #[error("Cannot matrix mutltiply with 0d tensor.")]
    Matmul0d,

    #[error("Cannot be matrix multiplied. [m x n1] @ [n2 x l], n1 ({n1}) != n2 ({n2}).")]
    Matmul2d { n1: usize, n2: usize },

    #[error("Cannot be matrix multiplied. [m1 x n1] @ [m2 x n2 x l], n1 ({n1}) != n2 ({n2}).")]
    MatmulNd { n1: usize, n2: usize },
}

// --- Conv ---

#[derive(Error, Debug)]
#[error(
    "Neither are all input sizes {0:?} >= the kernel sizes {1:?}, nor are all kernel sizes {1:?} >= the input sizes {0:?}.", input_sizes, kernel_sizes
)]
pub(crate) struct ValidConvShapeError {
    pub input_sizes: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
}

// --- Misc ---

#[derive(Error, Debug)]
#[error("Cannot convert {value} from `usize` to type {dtype}.")]
pub(crate) struct UsizeCastError {
    pub value: usize,
    pub dtype: &'static str,
}

#[derive(Error, Debug)]
pub(crate) enum ArangeError {
    #[error("Step size cannot be zero.")]
    Zero,

    #[error("Step size is positive, but start > end.")]
    Positive,

    #[error("Step size is negative, but end > start.")]
    Negative,

    #[error("Step size cannot compared with zero.")]
    Comparison,
}
