//! 4D vector primitive with `f64` public semantics matching Three.js r186 `Vector4`.

use crate::jsnum::{js_max, js_min, js_round, js_trunc};
use crate::matrix4::Matrix4;
use crate::narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};
use core::fmt;

#[inline]
fn js_clamp(value: f64, min: f64, max: f64) -> f64 {
    js_max(min, js_min(max, value))
}

/// A 4D vector represented by double-precision `f64` components `(x, y, z, w)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vector4 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
    /// W component.
    pub w: f64,
}

impl Vector4 {
    /// Constructs a new `Vector4` with given `(x, y, z, w)` components.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// Constructs an all-zero `Vector4` `(0.0, 0.0, 0.0, 0.0)`.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    /// Constructs a vector filled with ones `(1.0, 1.0, 1.0, 1.0)`.
    #[inline]
    pub const fn one() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 1.0,
        }
    }

    /// Sets the components of this vector matching `Vector4.set(x, y, z, w)`.
    #[inline]
    pub fn set(&mut self, x: f64, y: f64, z: f64, w: f64) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
        self
    }

    /// Sets all components to `scalar` matching `Vector4.setScalar(scalar)`.
    #[inline]
    pub fn set_scalar(&mut self, scalar: f64) -> &mut Self {
        self.x = scalar;
        self.y = scalar;
        self.z = scalar;
        self.w = scalar;
        self
    }

    /// Sets the x component of this vector matching Three.js `Vector4.setX(x)`.
    #[inline]
    pub fn set_x(&mut self, x: f64) -> &mut Self {
        self.x = x;
        self
    }

    /// Sets the y component of this vector matching Three.js `Vector4.setY(y)`.
    #[inline]
    pub fn set_y(&mut self, y: f64) -> &mut Self {
        self.y = y;
        self
    }

    /// Sets the z component of this vector matching Three.js `Vector4.setZ(z)`.
    #[inline]
    pub fn set_z(&mut self, z: f64) -> &mut Self {
        self.z = z;
        self
    }

    /// Sets the w component of this vector matching Three.js `Vector4.setW(w)`.
    #[inline]
    pub fn set_w(&mut self, w: f64) -> &mut Self {
        self.w = w;
        self
    }

    /// Alias for `z` component matching Three.js `Vector4.width` getter.
    #[inline]
    pub fn width(&self) -> f64 {
        self.z
    }
    /// Alias for setting `z` component matching Three.js `Vector4.width` setter.
    #[inline]
    pub fn set_width(&mut self, value: f64) -> &mut Self {
        self.z = value;
        self
    }

    /// Alias for `w` component matching Three.js `Vector4.height` getter.
    #[inline]
    pub fn height(&self) -> f64 {
        self.w
    }
    /// Alias for setting `w` component matching Three.js `Vector4.height` setter.
    #[inline]
    pub fn set_height(&mut self, value: f64) -> &mut Self {
        self.w = value;
        self
    }

    /// Sets component by index (0=x, 1=y, 2=z, 3=w) matching Three.js `Vector4.setComponent`.
    #[inline]
    pub fn set_component(&mut self, index: usize, value: f64) -> &mut Self {
        match index {
            0 => self.x = value,
            1 => self.y = value,
            2 => self.z = value,
            3 => self.w = value,
            _ => panic!("THREE.Vector4: index is out of range: {index}"),
        }
        self
    }

    /// Returns component by index (0=x, 1=y, 2=z, 3=w) matching Three.js `Vector4.getComponent`.
    #[inline]
    pub fn get_component(&self, index: usize) -> f64 {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            _ => panic!("THREE.Vector4: index is out of range: {index}"),
        }
    }

    /// Copies components from vector `v` matching Three.js `Vector4.copy(v)`.
    #[inline]
    pub fn copy(&mut self, v: &Self) -> &mut Self {
        self.x = v.x;
        self.y = v.y;
        self.z = v.z;
        self.w = v.w;
        self
    }

    /// Adds vector `v` to this instance.
    #[inline]
    pub fn add(&mut self, v: &Self) -> &mut Self {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
        self.w += v.w;
        self
    }

    /// Adds scalar `s` to all components of this instance.
    #[inline]
    pub fn add_scalar(&mut self, s: f64) -> &mut Self {
        self.x += s;
        self.y += s;
        self.z += s;
        self.w += s;
        self
    }

    /// Sets this vector to `a + b` matching Three.js `Vector4.addVectors(a, b)`.
    #[inline]
    pub fn add_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.x = a.x + b.x;
        self.y = a.y + b.y;
        self.z = a.z + b.z;
        self.w = a.w + b.w;
        self
    }

    /// Adds `v * s` to this vector matching Three.js `Vector4.addScaledVector(v, s)`.
    #[inline]
    pub fn add_scaled_vector(&mut self, v: &Self, s: f64) -> &mut Self {
        self.x += v.x * s;
        self.y += v.y * s;
        self.z += v.z * s;
        self.w += v.w * s;
        self
    }

    /// Subtracts vector `v` from this instance.
    #[inline]
    pub fn sub(&mut self, v: &Self) -> &mut Self {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
        self.w -= v.w;
        self
    }

    /// Subtracts scalar `s` from all components of this instance.
    #[inline]
    pub fn sub_scalar(&mut self, s: f64) -> &mut Self {
        self.x -= s;
        self.y -= s;
        self.z -= s;
        self.w -= s;
        self
    }

    /// Sets this vector to `a - b` matching Three.js `Vector4.subVectors(a, b)`.
    #[inline]
    pub fn sub_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.x = a.x - b.x;
        self.y = a.y - b.y;
        self.z = a.z - b.z;
        self.w = a.w - b.w;
        self
    }

    /// Multiplies this vector componentwise by `v`.
    #[inline]
    pub fn multiply(&mut self, v: &Self) -> &mut Self {
        self.x *= v.x;
        self.y *= v.y;
        self.z *= v.z;
        self.w *= v.w;
        self
    }

    /// Multiplies all components by scalar `scalar`.
    #[inline]
    pub fn multiply_scalar(&mut self, scalar: f64) -> &mut Self {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
        self
    }

    /// Transforms this vector by 4x4 matrix `m`, preserving projective `w`.
    ///
    /// Matches Three.js r186 `Vector4.applyMatrix4(m)`.
    #[inline]
    pub fn apply_matrix4(&mut self, m: &Matrix4) -> &mut Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let w = self.w;
        let e = &m.elements;

        self.x = e[0] * x + e[4] * y + e[8] * z + e[12] * w;
        self.y = e[1] * x + e[5] * y + e[9] * z + e[13] * w;
        self.z = e[2] * x + e[6] * y + e[10] * z + e[14] * w;
        self.w = e[3] * x + e[7] * y + e[11] * z + e[15] * w;
        self
    }

    /// Divides this vector componentwise by `v`.
    #[inline]
    pub fn divide(&mut self, v: &Self) -> &mut Self {
        self.x /= v.x;
        self.y /= v.y;
        self.z /= v.z;
        self.w /= v.w;
        self
    }

    /// Divides all components by `scalar`.
    #[inline]
    pub fn divide_scalar(&mut self, scalar: f64) -> &mut Self {
        self.multiply_scalar(1.0 / scalar)
    }

    /// Sets this vector from matrix translation column matching Three.js `Vector4.setFromMatrixPosition`.
    #[inline]
    pub fn set_from_matrix_position(&mut self, m: &Matrix4) -> &mut Self {
        let e = &m.elements;
        self.x = e[12];
        self.y = e[13];
        self.z = e[14];
        self.w = e[15];
        self
    }

    /// Componentwise minimum matching ECMAScript `Math.min` (preserves `-0.0 < +0.0` and NaN).
    #[inline]
    pub fn min(&mut self, v: &Self) -> &mut Self {
        self.x = js_min(self.x, v.x);
        self.y = js_min(self.y, v.y);
        self.z = js_min(self.z, v.z);
        self.w = js_min(self.w, v.w);
        self
    }

    /// Componentwise maximum matching ECMAScript `Math.max` (preserves `+0.0 > -0.0` and NaN).
    #[inline]
    pub fn max(&mut self, v: &Self) -> &mut Self {
        self.x = js_max(self.x, v.x);
        self.y = js_max(self.y, v.y);
        self.z = js_max(self.z, v.z);
        self.w = js_max(self.w, v.w);
        self
    }

    /// Clamps each component between `min` and `max` vectors.
    #[inline]
    pub fn clamp(&mut self, min: &Self, max: &Self) -> &mut Self {
        self.x = js_clamp(self.x, min.x, max.x);
        self.y = js_clamp(self.y, min.y, max.y);
        self.z = js_clamp(self.z, min.z, max.z);
        self.w = js_clamp(self.w, min.w, max.w);
        self
    }

    /// Clamps each component between `min_val` and `max_val` scalars.
    #[inline]
    pub fn clamp_scalar(&mut self, min_val: f64, max_val: f64) -> &mut Self {
        self.x = js_clamp(self.x, min_val, max_val);
        self.y = js_clamp(self.y, min_val, max_val);
        self.z = js_clamp(self.z, min_val, max_val);
        self.w = js_clamp(self.w, min_val, max_val);
        self
    }

    /// Clamps the Euclidean length between `min` and `max` matching Three.js `Vector4.clampLength`.
    #[inline]
    pub fn clamp_length(&mut self, min: f64, max: f64) -> &mut Self {
        let length = self.length();
        let denom = if length == 0.0 || length.is_nan() {
            1.0
        } else {
            length
        };
        let clamped = js_clamp(length, min, max);
        self.divide_scalar(denom);
        self.multiply_scalar(clamped);
        self
    }

    /// Rounds components down to the nearest integer matching Three.js `Vector4.floor()`.
    #[inline]
    pub fn floor(&mut self) -> &mut Self {
        self.x = self.x.floor();
        self.y = self.y.floor();
        self.z = self.z.floor();
        self.w = self.w.floor();
        self
    }

    /// Rounds components up to the nearest integer matching Three.js `Vector4.ceil()`.
    #[inline]
    pub fn ceil(&mut self) -> &mut Self {
        self.x = self.x.ceil();
        self.y = self.y.ceil();
        self.z = self.z.ceil();
        self.w = self.w.ceil();
        self
    }

    /// Rounds components to the nearest integer matching ECMAScript `Math.round`.
    #[inline]
    pub fn round(&mut self) -> &mut Self {
        self.x = js_round(self.x);
        self.y = js_round(self.y);
        self.z = js_round(self.z);
        self.w = js_round(self.w);
        self
    }

    /// Truncates fractional parts toward zero matching ECMAScript `Math.trunc`.
    #[inline]
    pub fn round_to_zero(&mut self) -> &mut Self {
        self.x = js_trunc(self.x);
        self.y = js_trunc(self.y);
        self.z = js_trunc(self.z);
        self.w = js_trunc(self.w);
        self
    }

    /// Inverts all components of this vector (`self = -self`) matching Three.js `Vector4.negate()`.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self.w = -self.w;
        self
    }

    /// Computes the dot product with vector `v` matching Three.js `Vector4.dot(v)`.
    #[inline]
    pub fn dot(&self, v: &Self) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z + self.w * v.w
    }

    /// Computes the squared Euclidean length matching Three.js `Vector4.lengthSq()`.
    #[inline]
    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Computes the Euclidean length matching Three.js `Vector4.length()`.
    #[inline]
    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    /// Computes the Manhattan (L1) length matching Three.js `Vector4.manhattanLength()`.
    #[inline]
    pub fn manhattan_length(&self) -> f64 {
        self.x.abs() + self.y.abs() + self.z.abs() + self.w.abs()
    }

    /// Converts to unit vector matching Three.js `Vector4.normalize()`. Preserves signed zero signs.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let l = self.length();
        let denom = if l == 0.0 || l.is_nan() { 1.0 } else { l };
        self.divide_scalar(denom)
    }

    /// Sets this vector to the same direction with length `length` matching Three.js `Vector4.setLength(length)`.
    #[inline]
    pub fn set_length(&mut self, length: f64) -> &mut Self {
        self.normalize();
        self.multiply_scalar(length)
    }

    /// Linearly interpolates between this vector and vector `v` by factor `alpha` matching Three.js `Vector4.lerp(v, alpha)`.
    #[inline]
    pub fn lerp(&mut self, v: &Self, alpha: f64) -> &mut Self {
        self.x += (v.x - self.x) * alpha;
        self.y += (v.y - self.y) * alpha;
        self.z += (v.z - self.z) * alpha;
        self.w += (v.w - self.w) * alpha;
        self
    }

    /// Linearly interpolates between `v1` and `v2` by factor `alpha` and stores in `self` matching Three.js `Vector4.lerpVectors(v1, v2, alpha)`.
    #[inline]
    pub fn lerp_vectors(&mut self, v1: &Self, v2: &Self, alpha: f64) -> &mut Self {
        self.x = v1.x + (v2.x - v1.x) * alpha;
        self.y = v1.y + (v2.y - v1.y) * alpha;
        self.z = v1.z + (v2.z - v1.z) * alpha;
        self.w = v1.w + (v2.w - v1.w) * alpha;
        self
    }

    /// Returns `true` if this vector strictly equals `v` matching Three.js `Vector4.equals(v)`.
    #[inline]
    pub fn equals(&self, v: &Self) -> bool {
        self.x == v.x && self.y == v.y && self.z == v.z && self.w == v.w
    }

    /// Returns components as a fixed-size 4-element array `[x, y, z, w]`.
    #[inline]
    pub const fn to_array(&self) -> [f64; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Constructs a `Vector4` from a fixed-size 4-element array `[x, y, z, w]`.
    #[inline]
    pub const fn from_array(a: [f64; 4]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
            w: a[3],
        }
    }

    /// Reads 4 components from slice `array` starting at `offset` into `self`.
    ///
    /// Requires `array.len() >= offset + 4`.
    ///
    /// # Panics
    /// Panics if `offset + 4 > array.len()`. If `offset < array.len()`, components
    /// read before the out-of-bounds index will be updated on `self` before panic (partial write).
    #[inline]
    pub fn from_slice_offset(&mut self, array: &[f64], offset: usize) -> &mut Self {
        self.x = array[offset];
        self.y = array[offset + 1];
        self.z = array[offset + 2];
        self.w = array[offset + 3];
        self
    }

    /// Writes 4 components of `self` into mutable slice `array` starting at `offset`.
    ///
    /// Requires `array.len() >= offset + 4`.
    ///
    /// # Panics
    /// Panics if `offset + 4 > array.len()`. If `offset < array.len()`, elements
    /// written before the out-of-bounds index will remain modified in `array` before panic (partial write).
    #[inline]
    pub fn to_slice_offset(&self, array: &mut [f64], offset: usize) {
        array[offset] = self.x;
        array[offset + 1] = self.y;
        array[offset + 2] = self.z;
        array[offset + 3] = self.w;
    }

    /// Narrows this `f64` vector into `[f32; 4]`, verifying that precision loss does not
    /// exceed `max_abs_err` or `max_rel_err`.
    ///
    /// Non-finite values (`NaN`, `Infinity`) are preserved.
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

    /// Narrows this vector with explicit `NarrowingTolerance` parameters.
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

/// Default constructor for `Vector4`: `constructor(x=0, y=0, z=0, w=1)` -> `(0.0, 0.0, 0.0, 1.0)`.
impl Default for Vector4 {
    #[inline]
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl fmt::Display for Vector4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl core::ops::Index<usize> for Vector4 {
    type Output = f64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vector4 index out of range: {index}"),
        }
    }
}

impl core::ops::IndexMut<usize> for Vector4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vector4 index out of range: {index}"),
        }
    }
}
