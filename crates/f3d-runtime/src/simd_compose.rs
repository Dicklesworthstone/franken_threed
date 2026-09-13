//! Browser and runtime conversion boundary for batch matrix composition.
//!
//! Exposes scalar and native SIMD matrix composition over flat `f64` slice boundaries
//! suitable for WebAssembly and JavaScript consumption.
//!
//! # Architecture & Non-Claim (§5.1)
//! - **Kernel Reuse**: Reuses [`Matrix4::batch_compose`] (scalar reference) and
//!   [`Matrix4::batch_compose_simd`]. No math logic is modified or duplicated here.
//! - **Shared Conversion**: Both scalar and SIMD entry points execute identical input validation,
//!   chunking, struct instantiation, and output flattening.
//! - **Non-Claim**: No acceleration, speedup, or crossover claims are made. This boundary provides
//!   the executable bridge for empirical browser measurements. Standard Wasm builds do not enable
//!   `simd128` by default.

use f3d_math::matrix4::{BatchComposeError, Matrix4};
use f3d_math::quaternion::Quaternion;
use f3d_math::vector3::Vector3;

/// Errors occurring during flat `f64` slice batch matrix composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdComposeError {
    /// Positions slice length is not a multiple of 3.
    PositionsNotMultipleOf3 { len: usize },
    /// Quaternions slice length is not a multiple of 4.
    QuaternionsNotMultipleOf4 { len: usize },
    /// Scales slice length is not a multiple of 3.
    ScalesNotMultipleOf3 { len: usize },
    /// Derived element counts (positions/3, quaternions/4, scales/3) do not match.
    CountMismatch {
        positions_count: usize,
        quaternions_count: usize,
        scales_count: usize,
    },
    /// Underlying math crate batch composition error.
    MathBatchError(BatchComposeError),
}

impl core::fmt::Display for SimdComposeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PositionsNotMultipleOf3 { len } => {
                write!(f, "positions length {len} is not a multiple of 3")
            }
            Self::QuaternionsNotMultipleOf4 { len } => {
                write!(f, "quaternions length {len} is not a multiple of 4")
            }
            Self::ScalesNotMultipleOf3 { len } => {
                write!(f, "scales length {len} is not a multiple of 3")
            }
            Self::CountMismatch {
                positions_count,
                quaternions_count,
                scales_count,
            } => {
                write!(
                    f,
                    "batch compose count mismatch: positions={positions_count}, quaternions={quaternions_count}, scales={scales_count}"
                )
            }
            Self::MathBatchError(err) => write!(f, "math batch compose error: {err}"),
        }
    }
}

impl std::error::Error for SimdComposeError {}

/// Execution mode for batch matrix composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComposeMode {
    Scalar,
    Simd,
}

/// Composes a batch of matrices using the scalar reference [`Matrix4::batch_compose`].
///
/// Input slices must contain packed float components:
/// - `positions`: $3N$ floats `[x0, y0, z0, x1, y1, z1, ...]`
/// - `quaternions`: $4N$ floats `[x0, y0, z0, w0, x1, y1, z1, w1, ...]`
/// - `scales`: $3N$ floats `[x0, y0, z0, x1, y1, z1, ...]`
///
/// Returns $16N$ floats in column-major order matching Three.js `Matrix4.elements`.
pub fn batch_compose_f64_scalar(
    positions: &[f64],
    quaternions: &[f64],
    scales: &[f64],
) -> Result<Vec<f64>, SimdComposeError> {
    batch_compose_f64_impl(positions, quaternions, scales, ComposeMode::Scalar)
}

/// Composes a batch of matrices using the SIMD kernel [`Matrix4::batch_compose_simd`].
///
/// Input slices must contain packed float components:
/// - `positions`: $3N$ floats `[x0, y0, z0, x1, y1, z1, ...]`
/// - `quaternions`: $4N$ floats `[x0, y0, z0, w0, x1, y1, z1, w1, ...]`
/// - `scales`: $3N$ floats `[x0, y0, z0, x1, y1, z1, ...]`
///
/// Returns $16N$ floats in column-major order matching Three.js `Matrix4.elements`.
pub fn batch_compose_f64_simd(
    positions: &[f64],
    quaternions: &[f64],
    scales: &[f64],
) -> Result<Vec<f64>, SimdComposeError> {
    batch_compose_f64_impl(positions, quaternions, scales, ComposeMode::Simd)
}

/// Shared conversion and dispatch implementation.
fn batch_compose_f64_impl(
    positions: &[f64],
    quaternions: &[f64],
    scales: &[f64],
    mode: ComposeMode,
) -> Result<Vec<f64>, SimdComposeError> {
    if positions.len() % 3 != 0 {
        return Err(SimdComposeError::PositionsNotMultipleOf3 {
            len: positions.len(),
        });
    }
    if quaternions.len() % 4 != 0 {
        return Err(SimdComposeError::QuaternionsNotMultipleOf4 {
            len: quaternions.len(),
        });
    }
    if scales.len() % 3 != 0 {
        return Err(SimdComposeError::ScalesNotMultipleOf3 {
            len: scales.len(),
        });
    }

    let n_pos = positions.len() / 3;
    let n_quat = quaternions.len() / 4;
    let n_scale = scales.len() / 3;

    if n_pos != n_quat || n_pos != n_scale {
        return Err(SimdComposeError::CountMismatch {
            positions_count: n_pos,
            quaternions_count: n_quat,
            scales_count: n_scale,
        });
    }

    let n = n_pos;
    if n == 0 {
        return Ok(Vec::new());
    }

    let pos_vec: Vec<Vector3> = positions
        .chunks_exact(3)
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();

    let quat_vec: Vec<Quaternion> = quaternions
        .chunks_exact(4)
        .map(|c| Quaternion::new(c[0], c[1], c[2], c[3]))
        .collect();

    let scale_vec: Vec<Vector3> = scales
        .chunks_exact(3)
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();

    let mut out_mats = vec![Matrix4::zero(); n];

    match mode {
        ComposeMode::Scalar => {
            Matrix4::batch_compose(&pos_vec, &quat_vec, &scale_vec, &mut out_mats)
                .map_err(SimdComposeError::MathBatchError)?;
        }
        ComposeMode::Simd => {
            Matrix4::batch_compose_simd(&pos_vec, &quat_vec, &scale_vec, &mut out_mats)
                .map_err(SimdComposeError::MathBatchError)?;
        }
    }

    let mut flat_outputs = Vec::with_capacity(n * 16);
    for mat in &out_mats {
        flat_outputs.extend_from_slice(&mat.elements);
    }

    Ok(flat_outputs)
}

// ============================================================================
// Wasm / Native Entry Points
// ============================================================================

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
/// WebAssembly export: executes scalar [`Matrix4::batch_compose`] over flat `f64` slices.
pub fn f3d_batch_compose_scalar(
    positions: &[f64],
    quaternions: &[f64],
    scales: &[f64],
) -> Result<Vec<f64>, wasm_bindgen::JsValue> {
    batch_compose_f64_scalar(positions, quaternions, scales)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
/// WebAssembly export: executes SIMD [`Matrix4::batch_compose_simd`] over flat `f64` slices.
pub fn f3d_batch_compose_simd(
    positions: &[f64],
    quaternions: &[f64],
    scales: &[f64],
) -> Result<Vec<f64>, wasm_bindgen::JsValue> {
    batch_compose_f64_simd(positions, quaternions, scales)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native host export matching the Wasm signature for host test verification.
pub fn f3d_batch_compose_scalar(
    positions: &[f64],
    quaternions: &[f64],
    scales: &[f64],
) -> Result<Vec<f64>, SimdComposeError> {
    batch_compose_f64_scalar(positions, quaternions, scales)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native host export matching the Wasm signature for host test verification.
pub fn f3d_batch_compose_simd(
    positions: &[f64],
    quaternions: &[f64],
    scales: &[f64],
) -> Result<Vec<f64>, SimdComposeError> {
    batch_compose_f64_simd(positions, quaternions, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_batch_compose() {
        let res_scalar = batch_compose_f64_scalar(&[], &[], &[]).expect("empty should succeed");
        let res_simd = batch_compose_f64_simd(&[], &[], &[]).expect("empty should succeed");
        assert!(res_scalar.is_empty());
        assert!(res_simd.is_empty());
    }

    #[test]
    fn test_single_item_compose_bit_exact() {
        let positions = [1.0, 2.0, 3.0];
        let quaternions = [0.0, 0.0, 0.0, 1.0];
        let scales = [1.0, 1.0, 1.0];

        let s_out = batch_compose_f64_scalar(&positions, &quaternions, &scales).unwrap();
        let v_out = batch_compose_f64_simd(&positions, &quaternions, &scales).unwrap();

        assert_eq!(s_out.len(), 16);
        assert_eq!(v_out.len(), 16);
        for i in 0..16 {
            assert_eq!(s_out[i].to_bits(), v_out[i].to_bits());
        }
    }
}
