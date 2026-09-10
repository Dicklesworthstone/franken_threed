//! Seamless conversions between `f3d-math` f64 simulation types and `f3d-core` GPU wire layouts.
//!
//! Architectural Seam:
//! - Coordinates with RusticRobin's `f3d-core::layout` module.
//! - Preserves canonical single source of truth for GPU layouts (`AffineRows`, `ProjectiveMat4`).
//! - Does NOT duplicate wire structs or byte-packing serialization in `f3d-math`.

use crate::matrix4::Matrix4;
use crate::narrowing::{NarrowingError, NarrowingTolerance};
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

    /// Converts this `f64` matrix into a 48-byte packed `AffineRows` GPU record, verifying
    /// that precision loss on each affine element does not exceed `max_abs_err` or `max_rel_err`.
    ///
    /// Rejects non-affine matrices (`!self.is_affine()`) with `Err(NarrowingError::NonAffineMatrix)`.
    pub fn to_affine_rows_checked(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<AffineRows, NarrowingError> {
        self.to_affine_rows_with_policy(max_abs_err, max_rel_err, false)
    }

    /// Strict conversion to `AffineRows`: rejects non-affine matrices, non-finite values (`NaN`, `Infinity`),
    /// and precision loss exceeding tolerance.
    pub fn to_affine_rows_strict(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<AffineRows, NarrowingError> {
        self.to_affine_rows_with_policy(max_abs_err, max_rel_err, true)
    }

    /// Converts to `AffineRows` using explicit `NarrowingTolerance` parameters and strict non-finite control.
    pub fn to_affine_rows_with_tolerance(
        &self,
        tolerance: NarrowingTolerance,
        strict: bool,
    ) -> Result<AffineRows, NarrowingError> {
        self.to_affine_rows_with_policy(tolerance.max_abs_err, tolerance.max_rel_err, strict)
    }

    /// Internal policy helper for `AffineRows` checked narrowing.
    pub fn to_affine_rows_with_policy(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
        strict: bool,
    ) -> Result<AffineRows, NarrowingError> {
        if !self.is_affine() {
            return Err(NarrowingError::NonAffineMatrix);
        }
        let e_f32 = self.to_f32_with_policy(max_abs_err, max_rel_err, strict)?;
        AffineRows::from_column_major(&e_f32).map_err(|_| NarrowingError::NonAffineMatrix)
    }

    /// Converts this `f64` matrix into a 64-byte `ProjectiveMat4` GPU wire record, verifying
    /// that precision loss on each element does not exceed `max_abs_err` or `max_rel_err`.
    pub fn to_projective_mat4_checked(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<ProjectiveMat4, NarrowingError> {
        self.to_projective_mat4_with_policy(max_abs_err, max_rel_err, false)
    }

    /// Strict conversion to `ProjectiveMat4`: rejects non-finite values (`NaN`, `Infinity`)
    /// and precision loss exceeding tolerance.
    pub fn to_projective_mat4_strict(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<ProjectiveMat4, NarrowingError> {
        self.to_projective_mat4_with_policy(max_abs_err, max_rel_err, true)
    }

    /// Converts to `ProjectiveMat4` using explicit `NarrowingTolerance` parameters and strict non-finite control.
    pub fn to_projective_mat4_with_tolerance(
        &self,
        tolerance: NarrowingTolerance,
        strict: bool,
    ) -> Result<ProjectiveMat4, NarrowingError> {
        self.to_projective_mat4_with_policy(tolerance.max_abs_err, tolerance.max_rel_err, strict)
    }

    /// Internal policy helper for `ProjectiveMat4` checked narrowing.
    pub fn to_projective_mat4_with_policy(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
        strict: bool,
    ) -> Result<ProjectiveMat4, NarrowingError> {
        let e_f32 = self.to_f32_with_policy(max_abs_err, max_rel_err, strict)?;
        Ok(ProjectiveMat4::from_elements(e_f32))
    }
}

