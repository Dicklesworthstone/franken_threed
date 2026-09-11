//! 4D quaternion primitive with `f64` public semantics matching Three.js r186 `Quaternion`.

use core::fmt;
use crate::matrix4::Matrix4;
use crate::narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};


/// A quaternion represented by components `(x, y, z, w)`.
///
/// Note: Quaternions are not automatically normalized on construction or in `compose`
/// to preserve authored non-unit quaternions and match Three.js r186 exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Quaternion {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
    /// W (scalar) component.
    pub w: f64,
}

impl Quaternion {
    /// Constructs a new quaternion with explicit `(x, y, z, w)` components.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// Constructs the identity quaternion `(0.0, 0.0, 0.0, 1.0)`.
    #[inline]
    pub const fn identity() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    /// Sets the components of this quaternion.
    #[inline]
    pub fn set(&mut self, x: f64, y: f64, z: f64, w: f64) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
        self
    }

    /// Copies components from another quaternion into this instance.
    #[inline]
    pub fn copy(&mut self, q: &Self) -> &mut Self {
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        self
    }

    /// Computes the dot product with quaternion `v`.
    #[inline]
    pub fn dot(&self, v: &Self) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z + self.w * v.w
    }

    /// Computes the squared Euclidean length of the 4D quaternion.
    #[inline]
    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Computes the Euclidean length of the 4D quaternion.
    #[inline]
    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    /// Normalizes this quaternion.
    ///
    /// If length is zero, resets to identity `(0.0, 0.0, 0.0, 1.0)` matching Three.js r186.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let l = self.length();
        if l == 0.0 {
            self.x = 0.0;
            self.y = 0.0;
            self.z = 0.0;
            self.w = 1.0;
        } else {
            let inv_l = 1.0 / l;
            self.x *= inv_l;
            self.y *= inv_l;
            self.z *= inv_l;
            self.w *= inv_l;
        }
        self
    }

    /// Inverts this quaternion via conjugate (assumes unit length, matching Three.js r186).
    #[inline]
    pub fn invert(&mut self) -> &mut Self {
        self.conjugate()
    }

    /// Returns the rotational conjugate of this quaternion (`x = -x, y = -y, z = -z`).
    #[inline]
    pub fn conjugate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self
    }

    /// Multiplies quaternion `a` by `b` and stores the result in this instance (`self = a * b`).
    #[inline]
    pub fn multiply_quaternions(&mut self, a: &Self, b: &Self) -> &mut Self {
        let qax = a.x;
        let qay = a.y;
        let qaz = a.z;
        let qaw = a.w;

        let qbx = b.x;
        let qby = b.y;
        let qbz = b.z;
        let qbw = b.w;

        self.x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
        self.y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
        self.z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
        self.w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;

        self
    }

    /// Multiplies this quaternion by `q` (`self = self * q`).
    #[inline]
    pub fn multiply(&mut self, q: &Self) -> &mut Self {
        let a = *self;
        self.multiply_quaternions(&a, q)
    }

    /// Premultiplies this quaternion by `q` (`self = q * self`).
    #[inline]
    pub fn premultiply(&mut self, q: &Self) -> &mut Self {
        let b = *self;
        self.multiply_quaternions(q, &b)
    }

    /// Extracts rotation from a 4x4 matrix assuming the upper 3x3 is pure rotation (unscaled).
    ///
    /// Evaluates using the standard Shoemake algorithm matching Three.js r186.
    pub fn set_from_rotation_matrix(&mut self, m: &Matrix4) -> &mut Self {
        let te = &m.elements;
        let m11 = te[0];
        let m12 = te[4];
        let m13 = te[8];
        let m21 = te[1];
        let m22 = te[5];
        let m23 = te[9];
        let m31 = te[2];
        let m32 = te[6];
        let m33 = te[10];

        let trace = m11 + m22 + m33;

        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            self.w = 0.25 / s;
            self.x = (m32 - m23) * s;
            self.y = (m13 - m31) * s;
            self.z = (m21 - m12) * s;
        } else if m11 > m22 && m11 > m33 {
            let s = 2.0 * (1.0 + m11 - m22 - m33).sqrt();
            self.w = (m32 - m23) / s;
            self.x = 0.25 * s;
            self.y = (m12 + m21) / s;
            self.z = (m13 + m31) / s;
        } else if m22 > m33 {
            let s = 2.0 * (1.0 + m22 - m11 - m33).sqrt();
            self.w = (m13 - m31) / s;
            self.x = (m12 + m21) / s;
            self.y = 0.25 * s;
            self.z = (m23 + m32) / s;
        } else {
            let s = 2.0 * (1.0 + m33 - m11 - m22).sqrt();
            self.w = (m21 - m12) / s;
            self.x = (m13 + m31) / s;
            self.y = (m23 + m32) / s;
            self.z = 0.25 * s;
        }

        self
    }

    /// Sets this quaternion from Euler angles matching Three.js r186 `Quaternion.setFromEuler`.
    pub fn set_from_euler(&mut self, euler: &crate::euler::Euler) -> &mut Self {
        let x = euler.x;
        let y = euler.y;
        let z = euler.z;
        let order = euler.order;

        let c1 = (x / 2.0).cos();
        let c2 = (y / 2.0).cos();
        let c3 = (z / 2.0).cos();

        let s1 = (x / 2.0).sin();
        let s2 = (y / 2.0).sin();
        let s3 = (z / 2.0).sin();

        match order {
            crate::euler::EulerOrder::XYZ => {
                self.x = s1 * c2 * c3 + c1 * s2 * s3;
                self.y = c1 * s2 * c3 - s1 * c2 * s3;
                self.z = c1 * c2 * s3 + s1 * s2 * c3;
                self.w = c1 * c2 * c3 - s1 * s2 * s3;
            }
            crate::euler::EulerOrder::YXZ => {
                self.x = s1 * c2 * c3 + c1 * s2 * s3;
                self.y = c1 * s2 * c3 - s1 * c2 * s3;
                self.z = c1 * c2 * s3 - s1 * s2 * c3;
                self.w = c1 * c2 * c3 + s1 * s2 * s3;
            }
            crate::euler::EulerOrder::ZXY => {
                self.x = s1 * c2 * c3 - c1 * s2 * s3;
                self.y = c1 * s2 * c3 + s1 * c2 * s3;
                self.z = c1 * c2 * s3 + s1 * s2 * c3;
                self.w = c1 * c2 * c3 - s1 * s2 * s3;
            }
            crate::euler::EulerOrder::ZYX => {
                self.x = s1 * c2 * c3 - c1 * s2 * s3;
                self.y = c1 * s2 * c3 + s1 * c2 * s3;
                self.z = c1 * c2 * s3 - s1 * s2 * c3;
                self.w = c1 * c2 * c3 + s1 * s2 * s3;
            }
            crate::euler::EulerOrder::YZX => {
                self.x = s1 * c2 * c3 + c1 * s2 * s3;
                self.y = c1 * s2 * c3 - s1 * c2 * s3;
                self.z = c1 * c2 * s3 - s1 * s2 * c3;
                self.w = c1 * c2 * c3 - s1 * s2 * s3;
            }
            crate::euler::EulerOrder::XZY => {
                self.x = s1 * c2 * c3 - c1 * s2 * s3;
                self.y = c1 * s2 * c3 - s1 * c2 * s3;
                self.z = c1 * c2 * s3 + s1 * s2 * c3;
                self.w = c1 * c2 * c3 + s1 * s2 * s3;
            }
        }

        self
    }

    /// Returns components as a fixed-size 4-element array `[x, y, z, w]`.
    #[inline]
    pub const fn to_array(&self) -> [f64; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Constructs a `Quaternion` from an array `[x, y, z, w]`.
    #[inline]
    pub const fn from_array(a: [f64; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
    }

    /// Narrows this `f64` quaternion into `[f32; 4]`, verifying that precision loss does not
    /// exceed `max_abs_err` or `max_rel_err`. Non-finite values (`NaN`, `Infinity`) are preserved.
    pub fn to_f32_checked(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 4], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, false)
    }

    /// Strict narrowing: rejects non-finite components (`NaN`, `Infinity`) and checks tolerance.
    pub fn to_f32_strict(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 4], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, true)
    }

    /// Narrows this quaternion with explicit `NarrowingTolerance` parameters.
    pub fn to_f32_with_tolerance(
        &self,
        tolerance: NarrowingTolerance,
        strict: bool,
    ) -> Result<[f32; 4], NarrowingError> {
        self.to_f32_with_policy(tolerance.max_abs_err, tolerance.max_rel_err, strict)
    }

    /// Checked narrowing helper evaluating each component with optional strict non-finite rejection.
    pub fn to_f32_with_policy(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
        strict: bool,
    ) -> Result<[f32; 4], NarrowingError> {
        let x = check_narrow_f64(self.x, 0, max_abs_err, max_rel_err, strict)?;
        let y = check_narrow_f64(self.y, 1, max_abs_err, max_rel_err, strict)?;
        let z = check_narrow_f64(self.z, 2, max_abs_err, max_rel_err, strict)?;
        let w = check_narrow_f64(self.w, 3, max_abs_err, max_rel_err, strict)?;
        Ok([x, y, z, w])
    }
}


impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quaternion({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl Default for Quaternion {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}
