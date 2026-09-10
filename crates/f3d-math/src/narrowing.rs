//! Checked `f64` to `f32` wire narrowing policy and precision loss validation (§5.1, §6.7).
//!
//! Provides explicit tolerance checking (both relative and absolute) and strict
//! non-finite value rejection (`NaN`, `+Infinity`, `-Infinity`) when preparing `f64`
//! mathematical simulation values for GPU wire layouts (`AffineRows`, `ProjectiveMat4`).

use core::fmt;

/// Tolerance parameters for checked `f64` -> `f32` narrowing.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NarrowingTolerance {
    /// Maximum allowable absolute error `|val_f64 - (val_f32 as f64)|`.
    pub max_abs_err: f64,
    /// Maximum allowable relative error `|val_f64 - (val_f32 as f64)| / |val_f64|`.
    pub max_rel_err: f64,
}

impl NarrowingTolerance {
    /// Constructs a new `NarrowingTolerance` with explicit absolute and relative bounds.
    #[inline]
    pub const fn new(max_abs_err: f64, max_rel_err: f64) -> Self {
        Self {
            max_abs_err,
            max_rel_err,
        }
    }

    /// Strict exact tolerance: requires exact dyadic representation with zero loss.
    pub const EXACT: Self = Self {
        max_abs_err: 0.0,
        max_rel_err: 0.0,
    };

    /// Standard single-precision float tolerance matching `f32::EPSILON` (~1.192e-7).
    pub const DEFAULT_FLOAT: Self = Self {
        max_abs_err: 1e-7,
        max_rel_err: 1.1920928955078125e-7, // 2^-23
    };

    /// Permissive tolerance suitable for coarse transforms where 1e-4 absolute error is acceptable.
    pub const PERMISSIVE: Self = Self {
        max_abs_err: 1e-4,
        max_rel_err: 1e-4,
    };
}

impl From<(f64, f64)> for NarrowingTolerance {
    #[inline]
    fn from((max_abs_err, max_rel_err): (f64, f64)) -> Self {
        Self::new(max_abs_err, max_rel_err)
    }
}

/// Errors occurring during checked `f64` to `f32` narrowing.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NarrowingError {
    /// Value is non-finite (`NaN`, `+Infinity`, or `-Infinity`) and strict validation is enabled.
    NonFinite {
        /// Component index (e.g. 0..3 for Vector3, 0..15 for Matrix4).
        index: usize,
        /// The raw non-finite `f64` value.
        value: f64,
    },
    /// Precision loss exceeded both the allowable absolute and relative tolerances.
    PrecisionLossExceeded {
        /// Component index where tolerance was exceeded.
        index: usize,
        /// Original `f64` value.
        original: f64,
        /// Narrowed `f32` value (represented as `f64`).
        narrowed: f64,
        /// Measured absolute difference `|original - narrowed|`.
        abs_diff: f64,
        /// Measured relative difference `abs_diff / |original|`.
        rel_diff: f64,
        /// Configured maximum absolute error.
        max_abs_err: f64,
        /// Configured maximum relative error.
        max_rel_err: f64,
    },
    /// A matrix expected to be affine (row 3 = `[0, 0, 0, 1]`) contains non-affine perspective components.
    NonAffineMatrix,
}

impl fmt::Display for NarrowingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { index, value } => {
                write!(
                    f,
                    "non-finite value {value} at index {index} rejected by strict narrowing policy"
                )
            }
            Self::PrecisionLossExceeded {
                index,
                original,
                narrowed,
                abs_diff,
                rel_diff,
                max_abs_err,
                max_rel_err,
            } => {
                write!(
                    f,
                    "narrowing precision loss exceeded at index {index}: original {original} -> narrowed {narrowed} (abs_diff {abs_diff} > max_abs {max_abs_err}, rel_diff {rel_diff} > max_rel {max_rel_err})"
                )
            }
            Self::NonAffineMatrix => {
                write!(
                    f,
                    "matrix is non-affine: perspective row 3 elements are non-zero or scale is invalid"
                )
            }
        }
    }
}

impl core::error::Error for NarrowingError {}

/// Narrows a single `f64` value to `f32`, checking precision loss against absolute and relative tolerances.
///
/// If `strict` is `true`, rejects non-finite values (`NaN`, `+Infinity`, `-Infinity`, and finite values
/// that overflow `f32::MAX`).
///
/// Returns `Ok(val as f32)` if `abs_diff <= max_abs_err || rel_diff <= max_rel_err`.
/// Otherwise returns `Err(NarrowingError::PrecisionLossExceeded)`.
#[inline]
pub fn check_narrow_f64(
    val: f64,
    index: usize,
    max_abs_err: f64,
    max_rel_err: f64,
    strict: bool,
) -> Result<f32, NarrowingError> {
    if !val.is_finite() {
        if strict {
            return Err(NarrowingError::NonFinite { index, value: val });
        }
        return Ok(val as f32);
    }

    let narrowed = val as f32;
    if !narrowed.is_finite() {
        // Finite f64 overflowed f32 range
        if strict {
            return Err(NarrowingError::NonFinite { index, value: val });
        }
        return Err(NarrowingError::PrecisionLossExceeded {
            index,
            original: val,
            narrowed: narrowed as f64,
            abs_diff: f64::INFINITY,
            rel_diff: f64::INFINITY,
            max_abs_err,
            max_rel_err,
        });
    }

    let widened = narrowed as f64;
    let abs_diff = (val - widened).abs();
    let abs_val = val.abs();
    let rel_diff = if abs_val == 0.0 { 0.0 } else { abs_diff / abs_val };

    if abs_diff > max_abs_err && rel_diff > max_rel_err {
        return Err(NarrowingError::PrecisionLossExceeded {
            index,
            original: val,
            narrowed: widened,
            abs_diff,
            rel_diff,
            max_abs_err,
            max_rel_err,
        });
    }

    Ok(narrowed)
}
