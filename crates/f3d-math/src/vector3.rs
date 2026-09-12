//! 3D vector primitive with `f64` public semantics matching Three.js r186 `Vector3`.

use core::fmt;
use crate::color::Color;
use crate::euler::Euler;
use crate::jsnum::{js_max, js_min, js_round, js_trunc};
use crate::matrix3::Matrix3;
use crate::matrix4::Matrix4;
use crate::narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};
use crate::quaternion::Quaternion;

#[inline]
fn js_clamp(value: f64, min: f64, max: f64) -> f64 {
    js_max(min, js_min(max, value))
}


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

    /// Sets the `x` component of this vector.
    ///
    /// Matches Three.js r186 `Vector3.setX(x)`.
    #[inline]
    pub fn set_x(&mut self, x: f64) -> &mut Self {
        self.x = x;
        self
    }

    /// Sets the `y` component of this vector.
    ///
    /// Matches Three.js r186 `Vector3.setY(y)`.
    #[inline]
    pub fn set_y(&mut self, y: f64) -> &mut Self {
        self.y = y;
        self
    }

    /// Sets the `z` component of this vector.
    ///
    /// Matches Three.js r186 `Vector3.setZ(z)`.
    #[inline]
    pub fn set_z(&mut self, z: f64) -> &mut Self {
        self.z = z;
        self
    }

    /// Sets all components of this vector to `scalar`.
    ///
    /// Matches Three.js r186 `Vector3.setScalar(scalar)`.
    #[inline]
    pub fn set_scalar(&mut self, scalar: f64) -> &mut Self {
        self.x = scalar;
        self.y = scalar;
        self.z = scalar;
        self
    }

    /// Sets the vector component by index (`0` for `x`, `1` for `y`, `2` for `z`).
    ///
    /// Matches Three.js r186 `Vector3.setComponent(index, value)`.
    /// Panics if index >= 3.
    #[inline]
    pub fn set_component(&mut self, index: usize, value: f64) -> &mut Self {
        match index {
            0 => self.x = value,
            1 => self.y = value,
            2 => self.z = value,
            _ => panic!("THREE.Vector3: index is out of range: {index}"),
        }
        self
    }

    /// Returns the vector component by index (`0` for `x`, `1` for `y`, `2` for `z`).
    ///
    /// Matches Three.js r186 `Vector3.getComponent(index)`.
    /// Panics if index >= 3.
    #[inline]
    pub fn get_component(&self, index: usize) -> f64 {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("THREE.Vector3: index is out of range: {index}"),
        }
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

    /// Sets the vector components from the given spherical coordinates.
    ///
    /// Matches Three.js r186 `Vector3.setFromSphericalCoords(radius, phi, theta)`.
    pub fn set_from_spherical_coords(&mut self, radius: f64, phi: f64, theta: f64) -> &mut Self {
        let sin_phi_radius = phi.sin() * radius;

        self.x = sin_phi_radius * theta.sin();
        self.y = phi.cos() * radius;
        self.z = sin_phi_radius * theta.cos();

        self
    }

    /// Sets the vector components from the given cylindrical coordinates.
    ///
    /// Matches Three.js r186 `Vector3.setFromCylindricalCoords(radius, theta, y)`.
    pub fn set_from_cylindrical_coords(&mut self, radius: f64, theta: f64, y: f64) -> &mut Self {
        self.x = radius * theta.sin();
        self.y = y;
        self.z = radius * theta.cos();

        self
    }

    /// Sets this vector's components from the angles of `e`.
    ///
    /// Matches Three.js r186 `Vector3.setFromEuler(e)`.
    #[inline]
    pub fn set_from_euler(&mut self, e: &Euler) -> &mut Self {
        self.x = e.x;
        self.y = e.y;
        self.z = e.z;
        self
    }

    /// Sets this vector's components from the RGB channels of `c`.
    ///
    /// Matches Three.js r186 `Vector3.setFromColor(c)`.
    #[inline]
    pub fn set_from_color(&mut self, c: &Color) -> &mut Self {
        self.x = c.r;
        self.y = c.g;
        self.z = c.b;
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

    /// Adds scalar `s` to all components of this vector.
    ///
    /// Matches Three.js r186 `Vector3.addScalar(s)`.
    #[inline]
    pub fn add_scalar(&mut self, s: f64) -> &mut Self {
        self.x += s;
        self.y += s;
        self.z += s;
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

    /// Subtracts scalar `s` from all components of this vector.
    ///
    /// Matches Three.js r186 `Vector3.subScalar(s)`.
    #[inline]
    pub fn sub_scalar(&mut self, s: f64) -> &mut Self {
        self.x -= s;
        self.y -= s;
        self.z -= s;
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

    /// Multiplies this vector component-wise by `v`.
    ///
    /// Matches Three.js r186 `Vector3.multiply(v)`.
    #[inline]
    pub fn multiply(&mut self, v: &Self) -> &mut Self {
        self.x *= v.x;
        self.y *= v.y;
        self.z *= v.z;
        self
    }

    /// Multiplies vectors `a` and `b` component-wise and stores the result in `self`.
    ///
    /// Matches Three.js r186 `Vector3.multiplyVectors(a, b)`.
    #[inline]
    pub fn multiply_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.x = a.x * b.x;
        self.y = a.y * b.y;
        self.z = a.z * b.z;
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

    /// Computes the Manhattan (taxicab) length: `|x| + |y| + |z|`.
    ///
    /// Matches Three.js r186 `Vector3.manhattanLength()`.
    #[inline]
    pub fn manhattan_length(&self) -> f64 {
        self.x.abs() + self.y.abs() + self.z.abs()
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

    /// Divides this vector component-wise by `v`.
    ///
    /// Matches Three.js r186 `Vector3.divide(v)`.
    #[inline]
    pub fn divide(&mut self, v: &Self) -> &mut Self {
        self.x /= v.x;
        self.y /= v.y;
        self.z /= v.z;
        self
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

    /// Sets the length of this vector to `length`.
    ///
    /// Matches Three.js r186 `Vector3.setLength(length)`.
    /// If the vector length is zero, it remains zero.
    #[inline]
    pub fn set_length(&mut self, length: f64) -> &mut Self {
        self.normalize().multiply_scalar(length)
    }

    /// Clamps the length of this vector between `min` and `max`.
    ///
    /// Matches Three.js r186 `Vector3.clampLength(min, max)`.
    /// Clamping is performed using ECMAScript `Math.max(min, Math.min(max, length))`.
    /// Preserves the `length || 1` zero-division guard matching upstream.
    pub fn clamp_length(&mut self, min: f64, max: f64) -> &mut Self {
        let length = self.length();
        let denom = if length == 0.0 || length.is_nan() { 1.0 } else { length };
        let clamped = js_clamp(length, min, max);
        self.divide_scalar(denom).multiply_scalar(clamped)
    }

    /// Replaces components with the component-wise minimum with vector `v` using ECMAScript `Math.min`.
    ///
    /// Matches Three.js r186 `Vector3.min(v)`.
    #[inline]
    pub fn min(&mut self, v: &Self) -> &mut Self {
        self.x = js_min(self.x, v.x);
        self.y = js_min(self.y, v.y);
        self.z = js_min(self.z, v.z);
        self
    }

    /// Replaces components with the component-wise maximum with vector `v` using ECMAScript `Math.max`.
    ///
    /// Matches Three.js r186 `Vector3.max(v)`.
    #[inline]
    pub fn max(&mut self, v: &Self) -> &mut Self {
        self.x = js_max(self.x, v.x);
        self.y = js_max(self.y, v.y);
        self.z = js_max(self.z, v.z);
        self
    }

    /// Clamps each component between `min` and `max` vectors using ECMAScript clamp.
    ///
    /// Matches Three.js r186 `Vector3.clamp(min, max)`.
    #[inline]
    pub fn clamp(&mut self, min: &Self, max: &Self) -> &mut Self {
        self.x = js_clamp(self.x, min.x, max.x);
        self.y = js_clamp(self.y, min.y, max.y);
        self.z = js_clamp(self.z, min.z, max.z);
        self
    }

    /// Clamps each component between scalar `min_val` and `max_val` using ECMAScript clamp.
    ///
    /// Matches Three.js r186 `Vector3.clampScalar(minVal, maxVal)`.
    #[inline]
    pub fn clamp_scalar(&mut self, min_val: f64, max_val: f64) -> &mut Self {
        self.x = js_clamp(self.x, min_val, max_val);
        self.y = js_clamp(self.y, min_val, max_val);
        self.z = js_clamp(self.z, min_val, max_val);
        self
    }

    /// Rounds components down to the nearest integer matching ECMAScript `Math.floor`.
    ///
    /// Matches Three.js r186 `Vector3.floor()`.
    #[inline]
    pub fn floor(&mut self) -> &mut Self {
        self.x = self.x.floor();
        self.y = self.y.floor();
        self.z = self.z.floor();
        self
    }

    /// Rounds components up to the nearest integer matching ECMAScript `Math.ceil`.
    ///
    /// Matches Three.js r186 `Vector3.ceil()`.
    #[inline]
    pub fn ceil(&mut self) -> &mut Self {
        self.x = self.x.ceil();
        self.y = self.y.ceil();
        self.z = self.z.ceil();
        self
    }

    /// Rounds components to the nearest integer matching ECMAScript `Math.round`.
    ///
    /// Matches Three.js r186 `Vector3.round()`.
    #[inline]
    pub fn round(&mut self) -> &mut Self {
        self.x = js_round(self.x);
        self.y = js_round(self.y);
        self.z = js_round(self.z);
        self
    }

    /// Truncates fractional parts toward zero matching ECMAScript `Math.trunc`.
    ///
    /// Matches Three.js r186 `Vector3.roundToZero()`.
    #[inline]
    pub fn round_to_zero(&mut self) -> &mut Self {
        self.x = js_trunc(self.x);
        self.y = js_trunc(self.y);
        self.z = js_trunc(self.z);
        self
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

    /// Projects this vector onto the given vector `v`.
    ///
    /// Matches Three.js r186 `Vector3.projectOnVector(v)`.
    /// If `v.length_sq() == 0.0`, resets this vector to zero `(0, 0, 0)`.
    /// Uses `dot / length_sq` scaling without normalizing `v`.
    pub fn project_on_vector(&mut self, v: &Self) -> &mut Self {
        let vx = v.x;
        let vy = v.y;
        let vz = v.z;
        let denominator = vx * vx + vy * vy + vz * vz;

        if denominator == 0.0 {
            return self.set(0.0, 0.0, 0.0);
        }

        let scalar = (vx * self.x + vy * self.y + vz * self.z) / denominator;

        self.x = vx * scalar;
        self.y = vy * scalar;
        self.z = vz * scalar;
        self
    }

    /// Projects this vector onto a plane specified by `plane_normal`.
    ///
    /// Matches Three.js r186 `Vector3.projectOnPlane(planeNormal)`.
    /// Subtracts the projection of this vector onto `plane_normal` from this vector.
    pub fn project_on_plane(&mut self, plane_normal: &Self) -> &mut Self {
        let mut projected = *self;
        projected.project_on_vector(plane_normal);
        self.sub(&projected)
    }

    /// Reflects this vector off a plane orthogonal to `normal`.
    ///
    /// Matches Three.js r186 `Vector3.reflect(normal)`.
    /// Evaluates `self - 2 * (self.dot(normal)) * normal`.
    /// Note: Does not normalize `normal`, faithfully matching Three.js r186.
    pub fn reflect(&mut self, normal: &Self) -> &mut Self {
        let nx = normal.x;
        let ny = normal.y;
        let nz = normal.z;
        let factor = 2.0 * (self.x * nx + self.y * ny + self.z * nz);

        self.x -= nx * factor;
        self.y -= ny * factor;
        self.z -= nz * factor;
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

    /// Applies a rotation specified by an axis and an angle to this vector.
    ///
    /// Matches Three.js r186 `Vector3.applyAxisAngle(axis, angle)`.
    pub fn apply_axis_angle(&mut self, axis: &Self, angle: f64) -> &mut Self {
        let mut q = Quaternion::identity();
        q.set_from_axis_angle(axis, angle);
        self.apply_quaternion(&q)
    }

    /// Applies the rotation specified by `euler` to this vector.
    ///
    /// Matches Three.js r186 `Vector3.applyEuler(euler)`.
    #[inline]
    pub fn apply_euler(&mut self, euler: &Euler) -> &mut Self {
        let mut q = Quaternion::identity();
        q.set_from_euler(euler);
        self.apply_quaternion(&q)
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

    /// Checks strict per-component equality with `v`.
    ///
    /// Matches Three.js r186 `Vector3.equals(v)` (`===` per component: `NaN != NaN`, `-0.0 == +0.0`).
    #[inline]
    pub fn equals(&self, v: &Self) -> bool {
        (v.x == self.x) && (v.y == self.y) && (v.z == self.z)
    }

    /// Reads 3 components from slice `array` starting at `offset` into `self`.
    ///
    /// Requires `array.len() >= offset + 3`.
    /// Matches Three.js r186 `Vector3.fromArray(array, offset)`.
    ///
    /// # Panics
    /// Panics if `offset + 3 > array.len()`. If `offset < array.len()`, components
    /// read before the out-of-bounds index will be updated on `self` before panic (partial write).
    #[inline]
    pub fn from_slice_offset(&mut self, array: &[f64], offset: usize) -> &mut Self {
        self.x = array[offset];
        self.y = array[offset + 1];
        self.z = array[offset + 2];
        self
    }

    /// Writes 3 components of `self` into mutable slice `array` starting at `offset`.
    ///
    /// Requires `array.len() >= offset + 3`.
    /// Matches Three.js r186 `Vector3.toArray(array, offset)`.
    ///
    /// # Panics
    /// Panics if `offset + 3 > array.len()`. If `offset < array.len()`, elements
    /// written before the out-of-bounds index will remain modified in `array` before panic (partial write).
    #[inline]
    pub fn to_slice_offset<'a>(&self, array: &'a mut [f64], offset: usize) -> &'a mut [f64] {
        array[offset] = self.x;
        array[offset + 1] = self.y;
        array[offset + 2] = self.z;
        array
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
