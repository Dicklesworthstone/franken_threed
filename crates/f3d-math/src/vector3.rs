//! 3D vector primitive with `f64` public semantics matching Three.js r186 `Vector3`.

use core::fmt;
use crate::matrix3::Matrix3;
use crate::matrix4::Matrix4;
use crate::narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};
use crate::quaternion::Quaternion;


/// A 3D vector represented by double-precision `f64` components `(x, y, z)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vector3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vector3 {
    /// Constructs a new `Vector3` with given `(x, y, z)` components.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Constructs a zero `Vector3` `(0.0, 0.0, 0.0)`.
    #[inline]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Constructs a vector filled with ones `(1.0, 1.0, 1.0)`.
    #[inline]
    pub const fn one() -> Self {
        Self { x: 1.0, y: 1.0, z: 1.0 }
    }

    /// Sets the components of this vector.
    #[inline]
    pub fn set(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self
    }

    /// Sets this vector's components from the specified column (0..=3) of a 4x4 matrix.
    ///
    /// Matches Three.js r186 `Vector3.setFromMatrixColumn(m, index)`.
    #[inline]
    pub fn set_from_matrix_column(&mut self, m: &Matrix4, index: usize) -> &mut Self {
        let offset = index * 4;
        self.x = m.elements[offset];
        self.y = m.elements[offset + 1];
        self.z = m.elements[offset + 2];
        self
    }

    /// Sets this vector's components from the specified column (0..=2) of a 3x3 matrix.
    ///
    /// Matches Three.js r186 `Vector3.setFromMatrix3Column(m, index)`.
    #[inline]
    pub fn set_from_matrix3_column(&mut self, m: &Matrix3, index: usize) -> &mut Self {
        let offset = index * 3;
        self.x = m.elements[offset];
        self.y = m.elements[offset + 1];
        self.z = m.elements[offset + 2];
        self
    }

    /// Sets this vector's components from the translation column (elements 12, 13, 14) of a 4x4 matrix.
    ///
    /// Matches Three.js r186 `Vector3.setFromMatrixPosition(m)`.
    #[inline]
    pub fn set_from_matrix_position(&mut self, m: &Matrix4) -> &mut Self {
        self.x = m.elements[12];
        self.y = m.elements[13];
        self.z = m.elements[14];
        self
    }

    /// Sets this vector's components to the scale factors extracted from the basis vectors of a 4x4 matrix.
    ///
    /// Matches Three.js r186 `Vector3.setFromMatrixScale(m)`.
    #[inline]
    pub fn set_from_matrix_scale(&mut self, m: &Matrix4) -> &mut Self {
        let sx = (m.elements[0] * m.elements[0] + m.elements[1] * m.elements[1] + m.elements[2] * m.elements[2]).sqrt();
        let sy = (m.elements[4] * m.elements[4] + m.elements[5] * m.elements[5] + m.elements[6] * m.elements[6]).sqrt();
        let sz = (m.elements[8] * m.elements[8] + m.elements[9] * m.elements[9] + m.elements[10] * m.elements[10]).sqrt();
        self.x = sx;
        self.y = sy;
        self.z = sz;
        self
    }

    /// Copies components from another vector into this instance.
    #[inline]
    pub fn copy(&mut self, v: &Self) -> &mut Self {
        self.x = v.x;
        self.y = v.y;
        self.z = v.z;
        self
    }

    /// Adds vector `v` to this instance.
    #[inline]
    pub fn add(&mut self, v: &Self) -> &mut Self {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
        self
    }

    /// Subtracts vector `v` from this instance.
    #[inline]
    pub fn sub(&mut self, v: &Self) -> &mut Self {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
        self
    }

    /// Sets this vector to `a + b`.
    ///
    /// Matches Three.js r186 `Vector3.addVectors(a, b)`.
    #[inline]
    pub fn add_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.x = a.x + b.x;
        self.y = a.y + b.y;
        self.z = a.z + b.z;
        self
    }

    /// Sets this vector to `a - b`.
    ///
    /// Matches Three.js r186 `Vector3.subVectors(a, b)`.
    #[inline]
    pub fn sub_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.x = a.x - b.x;
        self.y = a.y - b.y;
        self.z = a.z - b.z;
        self
    }

    /// Adds `v * s` to this vector.
    ///
    /// Matches Three.js r186 `Vector3.addScaledVector(v, s)`.
    #[inline]
    pub fn add_scaled_vector(&mut self, v: &Self, s: f64) -> &mut Self {
        self.x += v.x * s;
        self.y += v.y * s;
        self.z += v.z * s;
        self
    }

    /// Inverts all components of this vector (`self = -self`).
    ///
    /// Matches Three.js r186 `Vector3.negate()`.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self
    }

    /// Multiplies all components of this vector by scalar `s`.
    #[inline]
    pub fn multiply_scalar(&mut self, s: f64) -> &mut Self {
        self.x *= s;
        self.y *= s;
        self.z *= s;
        self
    }

    /// Computes the dot product with vector `v`.
    #[inline]
    pub fn dot(&self, v: &Self) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Sets this vector to the cross product of `a` and `b`.
    #[inline]
    pub fn cross_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        let ax = a.x;
        let ay = a.y;
        let az = a.z;
        let bx = b.x;
        let by = b.y;
        let bz = b.z;

        self.x = ay * bz - az * by;
        self.y = az * bx - ax * bz;
        self.z = ax * by - ay * bx;
        self
    }

    /// Calculates the cross product of this vector with vector `v` and stores the result in `self`.
    ///
    /// Matches Three.js r186 `Vector3.cross(v)`.
    #[inline]
    pub fn cross(&mut self, v: &Self) -> &mut Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        self.x = y * v.z - z * v.y;
        self.y = z * v.x - x * v.z;
        self.z = x * v.y - y * v.x;
        self
    }

    /// Computes the squared Euclidean length.
    #[inline]
    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Computes the Euclidean length.
    #[inline]
    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    /// Computes the squared Euclidean distance between this vector and vector `v`.
    ///
    /// Matches Three.js r186 `Vector3.distanceToSquared(v)`.
    #[inline]
    pub fn distance_to_squared(&self, v: &Self) -> f64 {
        let dx = self.x - v.x;
        let dy = self.y - v.y;
        let dz = self.z - v.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Computes the Euclidean distance between this vector and vector `v`.
    ///
    /// Matches Three.js r186 `Vector3.distanceTo(v)`.
    #[inline]
    pub fn distance_to(&self, v: &Self) -> f64 {
        self.distance_to_squared(v).sqrt()
    }

    /// Computes the Manhattan (L1) distance between this vector and vector `v`.
    ///
    /// Matches Three.js r186 `Vector3.manhattanDistanceTo(v)`.
    #[inline]
    pub fn manhattan_distance_to(&self, v: &Self) -> f64 {
        (self.x - v.x).abs() + (self.y - v.y).abs() + (self.z - v.z).abs()
    }

    /// Divides all components of this vector by scalar `s` matching Three.js r186 `divideScalar`.
    #[inline]
    pub fn divide_scalar(&mut self, s: f64) -> &mut Self {
        self.multiply_scalar(1.0 / s)
    }

    /// Converts this vector to a unit vector matching Three.js r186 `normalize()`.
    ///
    /// Upstream implementation: `this.divideScalar( this.length() || 1 )`.
    /// In JavaScript `length || 1` evaluates to `1` when length is zero or `NaN`.
    /// This preserves signed zeros (e.g. `-0.0`) and preserves non-NaN components when one component is `NaN`.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let l = self.length();
        let denom = if l == 0.0 || l.is_nan() { 1.0 } else { l };
        self.divide_scalar(denom)
    }

    /// Linearly interpolates between this vector and vector `v` by factor `alpha`.
    ///
    /// Matches Three.js r186 `Vector3.lerp(v, alpha)`:
    /// `self += (v - self) * alpha`.
    #[inline]
    pub fn lerp(&mut self, v: &Self, alpha: f64) -> &mut Self {
        self.x += (v.x - self.x) * alpha;
        self.y += (v.y - self.y) * alpha;
        self.z += (v.z - self.z) * alpha;
        self
    }

    /// Linearly interpolates between `v1` and `v2` by factor `alpha` and stores in `self`.
    ///
    /// Matches Three.js r186 `Vector3.lerpVectors(v1, v2, alpha)`.
    #[inline]
    pub fn lerp_vectors(&mut self, v1: &Self, v2: &Self, alpha: f64) -> &mut Self {
        self.x = v1.x + (v2.x - v1.x) * alpha;
        self.y = v1.y + (v2.y - v1.y) * alpha;
        self.z = v1.z + (v2.z - v1.z) * alpha;
        self
    }

    /// Returns the angle between this vector and vector `v` in radians.
    ///
    /// Matches Three.js r186 `Vector3.angleTo(v)`:
    /// Returns `PI / 2` when either vector length is zero.
    #[inline]
    pub fn angle_to(&self, v: &Self) -> f64 {
        let denominator = (self.length_sq() * v.length_sq()).sqrt();
        if denominator == 0.0 {
            return core::f64::consts::FRAC_PI_2;
        }
        let theta = self.dot(v) / denominator;
        theta.clamp(-1.0, 1.0).acos()
    }

    /// Transforms this vector by a 4x4 matrix, including division by the perspective component `w`.
    ///
    /// Preserves exact Three.js r186 operation ordering:
    /// `w = 1.0 / (e[3]*x + e[7]*y + e[11]*z + e[15])`
    /// `x = (e[0]*x + e[4]*y + e[8]*z + e[12]) * w`
    #[inline]
    pub fn apply_matrix4(&mut self, m: &Matrix4) -> &mut Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let e = &m.elements;

        let denom = e[3] * x + e[7] * y + e[11] * z + e[15];
        let w = 1.0 / denom;

        self.x = (e[0] * x + e[4] * y + e[8] * z + e[12]) * w;
        self.y = (e[1] * x + e[5] * y + e[9] * z + e[13]) * w;
        self.z = (e[2] * x + e[6] * y + e[10] * z + e[14]) * w;

        self
    }

    /// Projects this vector from world space into normalized device coordinates (NDC)
    /// given the camera's inverse world matrix (view matrix) and projection matrix.
    ///
    /// Matches Three.js r186 `Vector3.project(camera)`:
    /// `this.applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix)`.
    #[inline]
    pub fn project(&mut self, matrix_world_inverse: &Matrix4, projection_matrix: &Matrix4) -> &mut Self {
        self.apply_matrix4(matrix_world_inverse);
        self.apply_matrix4(projection_matrix);
        self
    }

    /// Unprojects this vector from normalized device coordinates (NDC) into world space
    /// given the camera's inverse projection matrix and world matrix.
    ///
    /// Matches Three.js r186 `Vector3.unproject(camera)`:
    /// `this.applyMatrix4(camera.projectionMatrixInverse).applyMatrix4(camera.matrixWorld)`.
    #[inline]
    pub fn unproject(&mut self, projection_matrix_inverse: &Matrix4, matrix_world: &Matrix4) -> &mut Self {
        self.apply_matrix4(projection_matrix_inverse);
        self.apply_matrix4(matrix_world);
        self
    }

    /// Projects this vector directly using a combined view-projection matrix.
    ///
    /// Evaluates `self.apply_matrix4(view_projection)`.
    #[inline]
    pub fn project_view_projection(&mut self, view_projection: &Matrix4) -> &mut Self {
        self.apply_matrix4(view_projection)
    }

    /// Transforms this direction vector by a 4x4 matrix ignoring translation (`w = 0.0`).
    #[inline]
    pub fn transform_direction(&mut self, m: &Matrix4) -> &mut Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let e = &m.elements;

        self.x = e[0] * x + e[4] * y + e[8] * z;
        self.y = e[1] * x + e[5] * y + e[9] * z;
        self.z = e[2] * x + e[6] * y + e[10] * z;

        self.normalize()
    }

    /// Multiplies this vector by 3x3 matrix `m`.
    ///
    /// Matches Three.js r186 `Vector3.applyMatrix3(m)`.
    #[inline]
    pub fn apply_matrix3(&mut self, m: &Matrix3) -> &mut Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let e = &m.elements;

        self.x = e[0] * x + e[3] * y + e[6] * z;
        self.y = e[1] * x + e[4] * y + e[7] * z;
        self.z = e[2] * x + e[5] * y + e[8] * z;

        self
    }

    /// Multiplies this vector by the given normal matrix and normalizes the result.
    ///
    /// Matches Three.js r186 `Vector3.applyNormalMatrix(m)`.
    #[inline]
    pub fn apply_normal_matrix(&mut self, m: &Matrix3) -> &mut Self {
        self.apply_matrix3(m).normalize()
    }

    /// Applies rotation from a quaternion to this vector.
    ///
    /// Assumes unit quaternion. Evaluates `v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)`.
    #[inline]
    pub fn apply_quaternion(&mut self, q: &Quaternion) -> &mut Self {
        let vx = self.x;
        let vy = self.y;
        let vz = self.z;
        let qx = q.x;
        let qy = q.y;
        let qz = q.z;
        let qw = q.w;

        let tx = 2.0 * (qy * vz - qz * vy);
        let ty = 2.0 * (qz * vx - qx * vz);
        let tz = 2.0 * (qx * vy - qy * vx);

        self.x = vx + qw * tx + (qy * tz - qz * ty);
        self.y = vy + qw * ty + (qz * tx - qx * tz);
        self.z = vz + qw * tz + (qx * ty - qy * tx);

        self
    }

    /// Returns components as a fixed-size 3-element array.
    #[inline]
    pub const fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Constructs a `Vector3` from a fixed-size 3-element array.
    #[inline]
    pub const fn from_array(a: [f64; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }

    /// Narrows this `f64` vector into `[f32; 3]`, verifying that precision loss does not
    /// exceed `max_abs_err` or `max_rel_err`.
    ///
    /// Non-finite values (`NaN`, `Infinity`) are preserved.
    pub fn to_f32_checked(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 3], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, false)
    }

    /// Strict narrowing: rejects non-finite components (`NaN`, `Infinity`) and checks tolerance.
    pub fn to_f32_strict(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 3], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, true)
    }

    /// Narrows this vector with explicit `NarrowingTolerance` parameters.
    pub fn to_f32_with_tolerance(
        &self,
        tolerance: NarrowingTolerance,
        strict: bool,
    ) -> Result<[f32; 3], NarrowingError> {
        self.to_f32_with_policy(tolerance.max_abs_err, tolerance.max_rel_err, strict)
    }

    /// Checked narrowing helper evaluating each component with optional strict non-finite rejection.
    pub fn to_f32_with_policy(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
        strict: bool,
    ) -> Result<[f32; 3], NarrowingError> {
        let x = check_narrow_f64(self.x, 0, max_abs_err, max_rel_err, strict)?;
        let y = check_narrow_f64(self.y, 1, max_abs_err, max_rel_err, strict)?;
        let z = check_narrow_f64(self.z, 2, max_abs_err, max_rel_err, strict)?;
        Ok([x, y, z])
    }
}


impl fmt::Display for Vector3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector3({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Default for Vector3 {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}
