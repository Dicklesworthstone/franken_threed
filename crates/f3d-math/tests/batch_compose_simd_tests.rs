//! Bit-exact parity tests between [`Matrix4::batch_compose_simd`] and the scalar
//! reference [`Matrix4::batch_compose`].
//!
//! # Performance Non-Claim
//! No acceleration, speedup, or crossover claims are made for this native SIMD kernel.
//! The release Wasm build does not enable `simd128` (no RUSTFLAGS change), so there is
//! no browser SIMD claim. Empirical crossover measurements and browser acceleration
//! validation are deferred to dedicated benchmark gates.

use f3d_math::matrix4::{BatchComposeError, Matrix4};
use f3d_math::quaternion::Quaternion;
use f3d_math::vector3::Vector3;

/// Deterministically generates float components covering edge cases:
/// - non-unit quaternions
/// - negative scales
/// - signed zero (+0.0 vs -0.0) in positions, scales, and quaternions
/// - NaN and +/-Infinity components
fn deterministic_float(index: usize, component: usize) -> f64 {
    let step = (index.wrapping_mul(31) + component.wrapping_mul(17)) % 25;
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
        9 => -0.0,
        10 => 2.0,
        11 => -0.25,
        12 => 0.1,
        13 => 0.2,
        14 => 0.3,
        15 => 0.4,
        16 => -2.0,
        17 => 3.5,
        18 => 0.0001,
        19 => -100.0,
        20 => 42.0,
        21 => -0.0,
        22 => 0.0,
        23 => 1.41421356,
        _ => (index as f64) * 0.1 - (component as f64) * 0.2,
    }
}

/// Generates deterministic input batches of size `n`.
fn generate_deterministic_inputs(n: usize) -> (Vec<Vector3>, Vec<Quaternion>, Vec<Vector3>) {
    let mut positions = Vec::with_capacity(n);
    let mut quaternions = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(n);

    for i in 0..n {
        let p = Vector3::new(
            deterministic_float(i, 1),
            deterministic_float(i, 2),
            deterministic_float(i, 3),
        );
        // Authored non-unit quaternions (not normalized)
        let q = Quaternion::new(
            deterministic_float(i, 4),
            deterministic_float(i, 5),
            deterministic_float(i, 6),
            deterministic_float(i, 7),
        );
        // Includes positive, negative, signed zero, and infinite scaling
        let s = Vector3::new(
            deterministic_float(i, 8),
            deterministic_float(i, 9),
            deterministic_float(i, 10),
        );
        positions.push(p);
        quaternions.push(q);
        scales.push(s);
    }

    (positions, quaternions, scales)
}

/// Asserts that two `Matrix4` instances match across all 16 elements:
/// bit-exact (`to_bits()` equal), except that when both elements are NaN they count as equal.
fn assert_matrix4_bitwise_match(scalar: &Matrix4, simd: &Matrix4, n: usize, item: usize) {
    for elem in 0..16 {
        let s = scalar.elements[elem];
        let v = simd.elements[elem];
        if s.is_nan() && v.is_nan() {
            continue;
        }
        assert_eq!(
            s.to_bits(),
            v.to_bits(),
            "Mismatch at N={}, item={}, element {}: scalar={} (0x{:016x}), simd={} (0x{:016x})",
            n, item, elem, s, s.to_bits(), v, v.to_bits()
        );
    }
}

#[test]
fn test_batch_compose_simd_parity_all_sizes() {
    let test_sizes = [0, 1, 3, 4, 5, 7, 8, 9, 257];

    for &n in &test_sizes {
        let (positions, quaternions, scales) = generate_deterministic_inputs(n);

        let mut scalar_outputs = vec![Matrix4::zero(); n];
        let mut simd_outputs = vec![Matrix4::zero(); n];

        let scalar_res = Matrix4::batch_compose(&positions, &quaternions, &scales, &mut scalar_outputs);
        let simd_res = Matrix4::batch_compose_simd(&positions, &quaternions, &scales, &mut simd_outputs);

        assert_eq!(scalar_res, Ok(()), "scalar batch_compose failed at N={}", n);
        assert_eq!(simd_res, Ok(()), "simd batch_compose_simd failed at N={}", n);

        for i in 0..n {
            assert_matrix4_bitwise_match(&scalar_outputs[i], &simd_outputs[i], n, i);
        }
    }
}

#[test]
fn test_batch_compose_simd_length_mismatch_untouched_outputs() {
    let sentinel = Matrix4::from_elements([
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]);

    let (positions, quaternions, scales) = generate_deterministic_inputs(8);

    // 1. Quaternions slice too short
    let mut outputs = vec![sentinel; 8];
    let err_simd = Matrix4::batch_compose_simd(&positions, &quaternions[..6], &scales, &mut outputs);
    let err_scalar = Matrix4::batch_compose(&positions, &quaternions[..6], &scales, &mut outputs);
    assert_eq!(err_simd, err_scalar);
    assert_eq!(
        err_simd,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 8,
            quaternions_len: 6,
            scales_len: 8,
            outputs_len: 8,
        })
    );
    for out in &outputs {
        assert_eq!(out, &sentinel);
    }

    // 2. Scales slice too short
    let mut outputs = vec![sentinel; 8];
    let err_simd = Matrix4::batch_compose_simd(&positions, &quaternions, &scales[..5], &mut outputs);
    let err_scalar = Matrix4::batch_compose(&positions, &quaternions, &scales[..5], &mut outputs);
    assert_eq!(err_simd, err_scalar);
    assert_eq!(
        err_simd,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 8,
            quaternions_len: 8,
            scales_len: 5,
            outputs_len: 8,
        })
    );
    for out in &outputs {
        assert_eq!(out, &sentinel);
    }

    // 3. Outputs slice too short
    let mut short_outputs = vec![sentinel; 4];
    let err_simd = Matrix4::batch_compose_simd(&positions, &quaternions, &scales, &mut short_outputs);
    let err_scalar = Matrix4::batch_compose(&positions, &quaternions, &scales, &mut short_outputs);
    assert_eq!(err_simd, err_scalar);
    assert_eq!(
        err_simd,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 8,
            quaternions_len: 8,
            scales_len: 8,
            outputs_len: 4,
        })
    );
    for out in &short_outputs {
        assert_eq!(out, &sentinel);
    }

    // 4. Positions slice too short
    let mut outputs = vec![sentinel; 8];
    let err_simd = Matrix4::batch_compose_simd(&positions[..3], &quaternions, &scales, &mut outputs);
    let err_scalar = Matrix4::batch_compose(&positions[..3], &quaternions, &scales, &mut outputs);
    assert_eq!(err_simd, err_scalar);
    assert_eq!(
        err_simd,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 3,
            quaternions_len: 8,
            scales_len: 8,
            outputs_len: 8,
        })
    );
    for out in &outputs {
        assert_eq!(out, &sentinel);
    }
}
