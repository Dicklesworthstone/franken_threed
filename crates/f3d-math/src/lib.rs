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

pub mod conversion;
pub mod matrix4;
pub mod narrowing;
pub mod quaternion;
pub mod vector3;

pub use matrix4::{CoordinateSystem, Matrix4};
pub use narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};
pub use quaternion::Quaternion;
pub use vector3::Vector3;

