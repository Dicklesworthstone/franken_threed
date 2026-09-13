//! Bit-exact parity and edge-case integration tests for runtime SIMD / scalar batch compose boundaries.
//!
//! Validates:
//! 1. Bit-exact equality between `f3d_batch_compose_scalar` and `f3d_batch_compose_simd`
//!    across tail counts N = 0, 1, 2, 3, 7, 8, 9 and larger N = 257.
//! 2. Preservation of -0.0, quiet NaN, +/-Infinity, subnormal numbers, authored non-unit
//!    quaternions, and negative scaling.
//! 3. Length mismatch error handling for truncated or non-multiple slice inputs.

use f3d_runtime::simd_compose::{
    batch_compose_f64_scalar, batch_compose_f64_simd, f3d_batch_compose_scalar,
    f3d_batch_compose_simd, SimdComposeError,
};

/// Deterministic float generator covering standard floats, negative values,
/// signed zeros (+0.0, -0.0), subnormals, quiet NaNs, and infinities.
fn deterministic_test_float(index: usize, component: usize) -> f64 {
    let step = (index.wrapping_mul(31) + component.wrapping_mul(17)) % 27;
    match step {
        0 => 0.0,
        1 => -0.0,
        2 => 1.0,
        3 => -1.0,
        4 => 0.5,
        5 => -1.5,
        6 => f64::NAN,
        7 => f64::INFINITY,
        8 => f64::NEG_INFINITY,
        9 => f64::from_bits(1),           // smallest positive subnormal (5e-324)
        10 => -f64::from_bits(1),         // smallest negative subnormal
        11 => f64::MIN_POSITIVE * 0.5,    // subnormal
        12 => 2.0,
        13 => -0.25,
        14 => 0.1,
        15 => 0.2,
        16 => 0.3,
        17 => 0.4,
        18 => -2.0,
        19 => 3.5,
        20 => 0.0001,
        21 => -100.0,
        22 => 42.0,
        23 => -0.0,
        24 => 0.0,
        25 => 1.41421356,
        _ => (index as f64) * 0.1 - (component as f64) * 0.2,
    }
}

fn generate_packed_inputs(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut positions = Vec::with_capacity(n * 3);
    let mut quaternions = Vec::with_capacity(n * 4);
    let mut scales = Vec::with_capacity(n * 3);

    for i in 0..n {
        positions.push(deterministic_test_float(i, 1));
        positions.push(deterministic_test_float(i, 2));
        positions.push(deterministic_test_float(i, 3));

        quaternions.push(deterministic_test_float(i, 4));
        quaternions.push(deterministic_test_float(i, 5));
        quaternions.push(deterministic_test_float(i, 6));
        quaternions.push(deterministic_test_float(i, 7));

        scales.push(deterministic_test_float(i, 8));
        scales.push(deterministic_test_float(i, 9));
        scales.push(deterministic_test_float(i, 10));
    }

    (positions, quaternions, scales)
}

fn assert_f64_slice_bitwise_match(scalar: &[f64], simd: &[f64], n: usize) {
    assert_eq!(scalar.len(), simd.len(), "Output length mismatch at N={n}");
    assert_eq!(scalar.len(), n * 16, "Expected 16*N outputs at N={n}");

    for i in 0..scalar.len() {
        let s = scalar[i];
        let v = simd[i];
        if s.is_nan() && v.is_nan() {
            continue;
        }
        assert_eq!(
            s.to_bits(),
            v.to_bits(),
            "Bitwise mismatch at N={n}, index {i}: scalar={s} (0x{s_bits:016x}), simd={v} (0x{v_bits:016x})",
            s_bits = s.to_bits(),
            v_bits = v.to_bits()
        );
    }
}

#[test]
fn test_simd_compose_parity_over_tail_lengths() {
    let test_sizes = [0, 1, 2, 3, 7, 8, 9, 257];

    for &n in &test_sizes {
        let (positions, quaternions, scales) = generate_packed_inputs(n);

        let scalar_out = batch_compose_f64_scalar(&positions, &quaternions, &scales)
            .unwrap_or_else(|e| panic!("scalar batch_compose failed at N={n}: {e}"));
        let simd_out = batch_compose_f64_simd(&positions, &quaternions, &scales)
            .unwrap_or_else(|e| panic!("simd batch_compose failed at N={n}: {e}"));

        assert_f64_slice_bitwise_match(&scalar_out, &simd_out, n);

        // Also test the f3d_ wrapper entry point
        let f3d_scalar_out = f3d_batch_compose_scalar(&positions, &quaternions, &scales)
            .unwrap_or_else(|e| panic!("f3d_batch_compose_scalar failed at N={n}: {e}"));
        let f3d_simd_out = f3d_batch_compose_simd(&positions, &quaternions, &scales)
            .unwrap_or_else(|e| panic!("f3d_batch_compose_simd failed at N={n}: {e}"));

        assert_f64_slice_bitwise_match(&f3d_scalar_out, &f3d_simd_out, n);
    }
}

#[test]
fn test_simd_compose_special_values_exact_parity() {
    // Explicit vector testing -0.0, quiet NaN, +/-Inf, subnormals, non-unit quaternions, negative scale
    let positions = vec![
        -0.0, 0.0, -0.0,                           // item 0: signed zeros
        10.0, 20.0, 30.0,                           // item 1: normal
        f64::NAN, f64::INFINITY, f64::NEG_INFINITY, // item 2: NaN and Inf
        f64::from_bits(1), f64::MIN_POSITIVE * 0.5, -f64::from_bits(1), // item 3: subnormals
    ];
    let quaternions = vec![
        0.1, 0.2, 0.3, 0.4,                         // item 0: non-unit quaternion (0.1^2 + 0.2^2 + 0.3^2 + 0.4^2 != 1)
        -0.0, 0.0, 1.0, -0.0,                       // item 1: signed zeros
        f64::NAN, 0.0, 0.0, 1.0,                    // item 2: NaN
        f64::from_bits(1), 0.0, 0.0, 1.0,           // item 3: subnormal
    ];
    let scales = vec![
        -1.5, -2.0, -0.5,                           // item 0: negative scale
        1.0, 1.0, 1.0,                              // item 1: unit scale
        f64::INFINITY, -0.0, 2.0,                   // item 2: Inf and -0.0
        f64::from_bits(1), 1.0, -1.0,               // item 3: subnormal and negative
    ];

    let scalar_out = f3d_batch_compose_scalar(&positions, &quaternions, &scales)
        .expect("scalar compose should succeed");
    let simd_out = f3d_batch_compose_simd(&positions, &quaternions, &scales)
        .expect("simd compose should succeed");

    assert_f64_slice_bitwise_match(&scalar_out, &simd_out, 4);
}

#[test]
fn test_simd_compose_length_mismatch_errors() {
    // 1. positions not multiple of 3
    let err_pos = batch_compose_f64_scalar(&[1.0, 2.0], &[0.0, 0.0, 0.0, 1.0], &[1.0, 1.0, 1.0]);
    assert_eq!(err_pos, Err(SimdComposeError::PositionsNotMultipleOf3 { len: 2 }));
    let err_pos_simd = batch_compose_f64_simd(&[1.0, 2.0], &[0.0, 0.0, 0.0, 1.0], &[1.0, 1.0, 1.0]);
    assert_eq!(err_pos, err_pos_simd);

    // 2. quaternions not multiple of 4
    let err_quat = batch_compose_f64_scalar(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
    assert_eq!(err_quat, Err(SimdComposeError::QuaternionsNotMultipleOf4 { len: 3 }));
    let err_quat_simd = batch_compose_f64_simd(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
    assert_eq!(err_quat, err_quat_simd);

    // 3. scales not multiple of 3
    let err_scale = batch_compose_f64_scalar(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0, 1.0], &[1.0, 1.0]);
    assert_eq!(err_scale, Err(SimdComposeError::ScalesNotMultipleOf3 { len: 2 }));
    let err_scale_simd = batch_compose_f64_simd(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0, 1.0], &[1.0, 1.0]);
    assert_eq!(err_scale, err_scale_simd);

    // 4. Count mismatch between positions and quaternions
    let err_count = batch_compose_f64_scalar(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 2 positions
        &[0.0, 0.0, 0.0, 1.0],           // 1 quaternion
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // 2 scales
    );
    assert_eq!(
        err_count,
        Err(SimdComposeError::CountMismatch {
            positions_count: 2,
            quaternions_count: 1,
            scales_count: 2,
        })
    );
    let err_count_simd = batch_compose_f64_simd(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[0.0, 0.0, 0.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    assert_eq!(err_count, err_count_simd);
}
