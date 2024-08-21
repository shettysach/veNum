#[cfg(test)]
mod core_tests {
    use crate::Tensor;

    #[test]
    fn same_storage() {
        let a = Tensor::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 9]);
        let b = a.reshape(&[3, 3]);

        let a_data_ptr: *const Vec<i32> = std::sync::Arc::as_ptr(&a.data);
        let b_data_ptr: *const Vec<i32> = std::sync::Arc::as_ptr(&b.data);
        assert_eq!(a_data_ptr, b_data_ptr);
    }

    #[test]
    fn contiguous() {
        let a = Tensor::arange(1, 28, 1);
        let a = a.reshape(&[3, 3, 3]);

        let flip_0 = a.flip(&[0]);
        let flip_01 = a.flip(&[0, 1]);
        let flip_all = a.flip(&[0, 1, 2]);

        assert!(a.is_contiguous());
        assert!(flip_all.is_contiguous());

        assert!(!flip_0.is_contiguous());
        assert!(!flip_01.is_contiguous());
    }
}
