//! # `f3d-math`: Fixed-size 3D Math Primitives for FrankenThreeD
//!
//! Provides `Vector3`, `Quaternion`, and `Matrix4` with `f64` public semantics
//! matching Three.js r186 (source commit `148ef33ecb6d2502ff796d4554abd1549c95d519`).
//!
//! ## Invariants
//! - Operation ordering and formula precision strictly match Three.js r186 math modules.
//! - Non-unit authored quaternions are preserved without premature normalization in `compose`.
//! - Singular matrix inversion (`determinant == 0.0`) yields an all-zero matrix.
//! - First-party code uses `#![forbid(unsafe_code)]`.
//! - Seamless conversion to `f3d-core::layout` GPU wire representations (`AffineRows`, `ProjectiveMat4`).

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod box3;
pub mod color;
pub mod conversion;
pub mod euler;
pub mod jsnum;
pub mod line3;
pub mod matrix3;
pub mod matrix4;
pub mod narrowing;
pub mod plane;
pub mod quaternion;
pub mod ray;
pub mod sphere;
pub mod triangle;
pub mod vector2;
pub mod vector3;
pub mod vector4;

pub use box3::Box3;
pub use color::{
    linear_to_srgb, srgb_to_linear, Color, ColorSpace, Hsl, StyleOutcome, COLOR_NAMES,
    LINEAR_DISPLAY_P3_TO_LINEAR_SRGB, LINEAR_DISPLAY_P3_TO_XYZ, LINEAR_REC709_TO_XYZ,
    LINEAR_SRGB_TO_LINEAR_DISPLAY_P3, XYZ_TO_LINEAR_DISPLAY_P3, XYZ_TO_LINEAR_REC709,
};
pub use euler::{Euler, EulerOrder};
pub use jsnum::{
    js_max, js_max_slice, js_min, js_min_slice, js_rem, js_round, js_shift_left, js_shift_right,
    js_shift_unsigned_right, js_sign, js_trunc, js_unsigned_shift_right, to_int32, to_uint32,
};
pub use line3::Line3;
pub use matrix3::Matrix3;
pub use matrix4::{BatchComposeError, CoordinateSystem, Matrix4};
pub use narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};
pub use plane::Plane;
pub use quaternion::Quaternion;
pub use ray::Ray;
pub use sphere::Sphere;
pub use triangle::Triangle;
pub use vector2::Vector2;
pub use vector3::Vector3;
pub use vector4::Vector4;


