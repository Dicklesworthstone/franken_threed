//! Seamless conversions between `f3d-math` f64 simulation types and `f3d-core` GPU wire layouts.
//!
//! Architectural Seam:
//! - Coordinates with RusticRobin's `f3d-core::layout` module.
//! - Preserves canonical single source of truth for GPU layouts (`AffineRows`, `ProjectiveMat4`).
//! - Does NOT duplicate wire structs or byte-packing serialization in `f3d-math`.

use crate::matrix4::Matrix4;
use f3d_core::layout::{AffineRows, LayoutError, ProjectiveMat4};

impl Matrix4 {
    /// Converts this `f64` matrix into a 48-byte packed `AffineRows` GPU wire record.
    ///
    /// Invariant: Exact structural check at full `f64` precision before narrowing.
    /// Uses `AffineRows::from_column_major_f64(&self.elements)`. If any perspective
    /// element is non-zero in `f64` (even subnormal e.g. `1e-100`) or `e[15] != 1.0`,
    /// returns `Err(LayoutError::NonAffineMatrix)`.
    #[inline]
    pub fn to_affine_rows(&self) -> Result<AffineRows, LayoutError> {
        AffineRows::from_column_major_f64(&self.elements)
    }

    /// Reconstructs an `f64` `Matrix4` from a 48-byte packed `AffineRows` GPU record.
    #[inline]
    pub fn from_affine_rows(affine: &AffineRows) -> Self {
        let e_f32 = affine.to_column_major();
        let mut e_f64 = [0.0f64; 16];
        for i in 0..16 {
            e_f64[i] = e_f32[i] as f64;
        }
        Self::from_elements(e_f64)
    }

    /// Converts this `f64` matrix into a 64-byte `ProjectiveMat4` GPU wire record.
    #[inline]
    pub fn to_projective_mat4(&self) -> ProjectiveMat4 {
        ProjectiveMat4::from_elements_f64(&self.elements)
    }

    /// Reconstructs an `f64` `Matrix4` from a 64-byte `ProjectiveMat4` GPU record.
    pub fn from_projective_mat4(proj: &ProjectiveMat4) -> Self {
        let mut e_f64 = [0.0f64; 16];
        for i in 0..16 {
            e_f64[i] = proj.elements[i] as f64;
        }
        Self::from_elements(e_f64)
    }
}
