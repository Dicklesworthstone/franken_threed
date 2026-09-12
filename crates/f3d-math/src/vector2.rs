//! 2D vector primitive with `f64` public semantics matching Three.js r186 `Vector2`.

use core::fmt;
use crate::jsnum::{js_max, js_min, js_round, js_trunc};
use crate::matrix3::Matrix3;
use crate::narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};

/// Clamps value between `min` and `max` matching Three.js `MathUtils.clamp`.
#[inline]
fn clamp(value: f64, min: f64, max: f64) -> f64 {
    js_max(min, js_min(max, value))
}

/// A 2D vector represented by double-precision `f64` components `(x, y)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vector2 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
}

impl Vector2 {
    /// Constructs a new `Vector2` with given `(x, y)` components.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Constructs a zero `Vector2` `(0.0, 0.0)`.
    #[inline]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Constructs a vector filled with ones `(1.0, 1.0)`.
    #[inline]
    pub const fn one() -> Self {
        Self { x: 1.0, y: 1.0 }
    }

    /// Sets the components of this vector.
    ///
    /// Matches Three.js r186 `Vector2.set(x, y)`.
    #[inline]
    pub fn set(&mut self, x: f64, y: f64) -> &mut Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Sets both components to scalar `scalar`.
    ///
    /// Matches Three.js r186 `Vector2.setScalar(scalar)`.
    #[inline]
    pub fn set_scalar(&mut self, scalar: f64) -> &mut Self {
        self.x = scalar;
        self.y = scalar;
        self
    }

    /// Sets the x component of this vector.
    ///
    /// Matches Three.js r186 `Vector2.setX(x)`.
    #[inline]
    pub fn set_x(&mut self, x: f64) -> &mut Self {
        self.x = x;
        self
    }

    /// Sets the y component of this vector.
    ///
    /// Matches Three.js r186 `Vector2.setY(y)`.
    #[inline]
    pub fn set_y(&mut self, y: f64) -> &mut Self {
        self.y = y;
        self
    }

    /// Sets the vector component by index (`0` for `x`, `1` for `y`).
    ///
    /// Matches Three.js r186 `Vector2.setComponent(index, value)`.
    /// Panics if index >= 2.
    #[inline]
    pub fn set_component(&mut self, index: usize, value: f64) -> &mut Self {
        match index {
            0 => self.x = value,
            1 => self.y = value,
            _ => panic!("Vector2: index is out of range: {index}"),
        }
        self
    }

    /// Returns the vector component by index (`0` for `x`, `1` for `y`).
    ///
    /// Matches Three.js r186 `Vector2.getComponent(index)`.
    /// Panics if index >= 2.
    #[inline]
    pub fn get_component(&self, index: usize) -> f64 {
        match index {
            0 => self.x,
            1 => self.y,
            _ => panic!("Vector2: index is out of range: {index}"),
        }
    }

    /// Copies components from another vector into this instance.
    ///
    /// Matches Three.js r186 `Vector2.copy(v)`.
    #[inline]
    pub fn copy(&mut self, v: &Self) -> &mut Self {
        self.x = v.x;
        self.y = v.y;
        self
    }

    /// Adds vector `v` to this vector.
    ///
    /// Matches Three.js r186 `Vector2.add(v)`.
    #[inline]
    pub fn add(&mut self, v: &Self) -> &mut Self {
        self.x += v.x;
        self.y += v.y;
        self
    }

    /// Adds scalar `s` to all components of this vector.
    ///
    /// Matches Three.js r186 `Vector2.addScalar(s)`.
    #[inline]
    pub fn add_scalar(&mut self, s: f64) -> &mut Self {
        self.x += s;
        self.y += s;
        self
    }

    /// Sets this vector to the sum of `a` and `b`.
    ///
    /// Matches Three.js r186 `Vector2.addVectors(a, b)`.
    #[inline]
    pub fn add_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.x = a.x + b.x;
        self.y = a.y + b.y;
        self
    }

    /// Adds vector `v` scaled by factor `s` to this vector.
    ///
    /// Matches Three.js r186 `Vector2.addScaledVector(v, s)`.
    #[inline]
    pub fn add_scaled_vector(&mut self, v: &Self, s: f64) -> &mut Self {
        self.x += v.x * s;
        self.y += v.y * s;
        self
    }

    /// Subtracts vector `v` from this vector.
    ///
    /// Matches Three.js r186 `Vector2.sub(v)`.
    #[inline]
    pub fn sub(&mut self, v: &Self) -> &mut Self {
        self.x -= v.x;
        self.y -= v.y;
        self
    }

    /// Subtracts scalar `s` from all components of this vector.
    ///
    /// Matches Three.js r186 `Vector2.subScalar(s)`.
    #[inline]
    pub fn sub_scalar(&mut self, s: f64) -> &mut Self {
        self.x -= s;
        self.y -= s;
        self
    }

    /// Sets this vector to the difference of `a` and `b` (`a - b`).
    ///
    /// Matches Three.js r186 `Vector2.subVectors(a, b)`.
    #[inline]
    pub fn sub_vectors(&mut self, a: &Self, b: &Self) -> &mut Self {
        self.x = a.x - b.x;
        self.y = a.y - b.y;
        self
    }

    /// Multiplies this vector component-wise by `v`.
    ///
    /// Matches Three.js r186 `Vector2.multiply(v)`.
    #[inline]
    pub fn multiply(&mut self, v: &Self) -> &mut Self {
        self.x *= v.x;
        self.y *= v.y;
        self
    }

    /// Multiplies all components of this vector by scalar `s`.
    ///
    /// Matches Three.js r186 `Vector2.multiplyScalar(s)`.
    #[inline]
    pub fn multiply_scalar(&mut self, s: f64) -> &mut Self {
        self.x *= s;
        self.y *= s;
        self
    }

    /// Divides this vector component-wise by `v`.
    ///
    /// Matches Three.js r186 `Vector2.divide(v)`.
    #[inline]
    pub fn divide(&mut self, v: &Self) -> &mut Self {
        self.x /= v.x;
        self.y /= v.y;
        self
    }

    /// Divides all components of this vector by scalar `s` matching Three.js r186 `divideScalar`.
    #[inline]
    pub fn divide_scalar(&mut self, s: f64) -> &mut Self {
        self.multiply_scalar(1.0 / s)
    }

    /// Multiplies this vector (with implicit 1 as 3rd component) by 3x3 matrix `m`.
    ///
    /// Matches Three.js r186 `Vector2.applyMatrix3(m)`:
    /// `x = e[0] * x + e[3] * y + e[6]`
    /// `y = e[1] * x + e[4] * y + e[7]`
    #[inline]
    pub fn apply_matrix3(&mut self, m: &Matrix3) -> &mut Self {
        let x = self.x;
        let y = self.y;
        let e = &m.elements;

        self.x = e[0] * x + e[3] * y + e[6];
        self.y = e[1] * x + e[4] * y + e[7];
        self
    }

    /// Inverts all components of this vector (`self = -self`).
    ///
    /// Matches Three.js r186 `Vector2.negate()`.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self
    }

    /// Computes the dot product with vector `v`.
    ///
    /// Matches Three.js r186 `Vector2.dot(v)`.
    #[inline]
    pub fn dot(&self, v: &Self) -> f64 {
        self.x * v.x + self.y * v.y
    }

    /// Computes the 2D cross product (perpendicular dot product) with vector `v`.
    ///
    /// Matches Three.js r186 `Vector2.cross(v)`:
    /// `self.x * v.y - self.y * v.x`.
    #[inline]
    pub fn cross(&self, v: &Self) -> f64 {
        self.x * v.y - self.y * v.x
    }

    /// Computes the squared Euclidean length.
    ///
    /// Matches Three.js r186 `Vector2.lengthSq()`.
    #[inline]
    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Computes the Euclidean length.
    ///
    /// Matches Three.js r186 `Vector2.length()`.
    #[inline]
    pub fn length(&self) -> f64 {
        self.length_sq().sqrt()
    }

    /// Computes the Manhattan (L1) length of this vector.
    ///
    /// Matches Three.js r186 `Vector2.manhattanLength()`.
    #[inline]
    pub fn manhattan_length(&self) -> f64 {
        self.x.abs() + self.y.abs()
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

    /// Computes the angle in radians of this vector with respect to the positive x-axis.
    ///
    /// Matches Three.js r186 `Vector2.angle()`:
    /// Evaluates `atan2(-y, -x) + PI`.
    #[inline]
    pub fn angle(&self) -> f64 {
        (-self.y).atan2(-self.x) + core::f64::consts::PI
    }

    /// Returns the angle between this vector and vector `v` in radians.
    ///
    /// Matches Three.js r186 `Vector2.angleTo(v)`:
    /// Returns `PI / 2` when either vector length is zero.
    #[inline]
    pub fn angle_to(&self, v: &Self) -> f64 {
        let denominator = (self.length_sq() * v.length_sq()).sqrt();
        if denominator == 0.0 {
            return core::f64::consts::FRAC_PI_2;
        }
        let theta = self.dot(v) / denominator;
        clamp(theta, -1.0, 1.0).acos()
    }

    /// Computes the squared Euclidean distance between this vector and vector `v`.
    ///
    /// Matches Three.js r186 `Vector2.distanceToSquared(v)`.
    #[inline]
    pub fn distance_to_squared(&self, v: &Self) -> f64 {
        let dx = self.x - v.x;
        let dy = self.y - v.y;
        dx * dx + dy * dy
    }

    /// Computes the Euclidean distance between this vector and vector `v`.
    ///
    /// Matches Three.js r186 `Vector2.distanceTo(v)`.
    #[inline]
    pub fn distance_to(&self, v: &Self) -> f64 {
        self.distance_to_squared(v).sqrt()
    }

    /// Computes the Manhattan (L1) distance between this vector and vector `v`.
    ///
    /// Matches Three.js r186 `Vector2.manhattanDistanceTo(v)`.
    #[inline]
    pub fn manhattan_distance_to(&self, v: &Self) -> f64 {
        (self.x - v.x).abs() + (self.y - v.y).abs()
    }

    /// Sets this vector to the same direction with length `length`.
    ///
    /// Matches Three.js r186 `Vector2.setLength(length)`.
    #[inline]
    pub fn set_length(&mut self, length: f64) -> &mut Self {
        self.normalize().multiply_scalar(length)
    }

    /// Linearly interpolates between this vector and vector `v` by factor `alpha`.
    ///
    /// Matches Three.js r186 `Vector2.lerp(v, alpha)`:
    /// `self += (v - self) * alpha`.
    #[inline]
    pub fn lerp(&mut self, v: &Self, alpha: f64) -> &mut Self {
        self.x += (v.x - self.x) * alpha;
        self.y += (v.y - self.y) * alpha;
        self
    }

    /// Linearly interpolates between `v1` and `v2` by factor `alpha` and stores in `self`.
    ///
    /// Matches Three.js r186 `Vector2.lerpVectors(v1, v2, alpha)`.
    #[inline]
    pub fn lerp_vectors(&mut self, v1: &Self, v2: &Self, alpha: f64) -> &mut Self {
        self.x = v1.x + (v2.x - v1.x) * alpha;
        self.y = v1.y + (v2.y - v1.y) * alpha;
        self
    }

    /// Rotates this vector around the given center by angle in radians.
    ///
    /// Matches Three.js r186 `Vector2.rotateAround(center, angle)`.
    #[inline]
    pub fn rotate_around(&mut self, center: &Self, angle: f64) -> &mut Self {
        let c = angle.cos();
        let s = angle.sin();
        let x = self.x - center.x;
        let y = self.y - center.y;
        self.x = x * c - y * s + center.x;
        self.y = x * s + y * c + center.y;
        self
    }

    /// Replaces components with the component-wise minimum with vector `v` using ECMAScript `Math.min`.
    ///
    /// Matches Three.js r186 `Vector2.min(v)`.
    #[inline]
    pub fn min(&mut self, v: &Self) -> &mut Self {
        self.x = js_min(self.x, v.x);
        self.y = js_min(self.y, v.y);
        self
    }

    /// Replaces components with the component-wise maximum with vector `v` using ECMAScript `Math.max`.
    ///
    /// Matches Three.js r186 `Vector2.max(v)`.
    #[inline]
    pub fn max(&mut self, v: &Self) -> &mut Self {
        self.x = js_max(self.x, v.x);
        self.y = js_max(self.y, v.y);
        self
    }

    /// Clamps components between `min` and `max`.
    ///
    /// Matches Three.js r186 `Vector2.clamp(min, max)`.
    #[inline]
    pub fn clamp(&mut self, min: &Self, max: &Self) -> &mut Self {
        self.x = clamp(self.x, min.x, max.x);
        self.y = clamp(self.y, min.y, max.y);
        self
    }

    /// Clamps components between scalar `min_val` and `max_val`.
    ///
    /// Matches Three.js r186 `Vector2.clampScalar(minVal, maxVal)`.
    #[inline]
    pub fn clamp_scalar(&mut self, min_val: f64, max_val: f64) -> &mut Self {
        self.x = clamp(self.x, min_val, max_val);
        self.y = clamp(self.y, min_val, max_val);
        self
    }

    /// Clamps the length of this vector between `min` and `max`.
    ///
    /// Matches Three.js r186 `Vector2.clampLength(min, max)`.
    #[inline]
    pub fn clamp_length(&mut self, min: f64, max: f64) -> &mut Self {
        let length = self.length();
        let denom = if length == 0.0 || length.is_nan() { 1.0 } else { length };
        self.divide_scalar(denom).multiply_scalar(clamp(length, min, max))
    }

    /// Rounds components down to the nearest integer.
    ///
    /// Matches Three.js r186 `Vector2.floor()`.
    #[inline]
    pub fn floor(&mut self) -> &mut Self {
        self.x = self.x.floor();
        self.y = self.y.floor();
        self
    }

    /// Rounds components up to the nearest integer.
    ///
    /// Matches Three.js r186 `Vector2.ceil()`.
    #[inline]
    pub fn ceil(&mut self) -> &mut Self {
        self.x = self.x.ceil();
        self.y = self.y.ceil();
        self
    }

    /// Rounds components to the nearest integer matching ECMAScript `Math.round`.
    ///
    /// Matches Three.js r186 `Vector2.round()`.
    #[inline]
    pub fn round(&mut self) -> &mut Self {
        self.x = js_round(self.x);
        self.y = js_round(self.y);
        self
    }

    /// Truncates fractional parts toward zero matching ECMAScript `Math.trunc`.
    ///
    /// Matches Three.js r186 `Vector2.roundToZero()`.
    #[inline]
    pub fn round_to_zero(&mut self) -> &mut Self {
        self.x = js_trunc(self.x);
        self.y = js_trunc(self.y);
        self
    }

    /// Returns `true` if this vector strictly equals `v` matching Three.js r186 `Vector2.equals(v)`.
    #[inline]
    pub fn equals(&self, v: &Self) -> bool {
        self.x == v.x && self.y == v.y
    }

    /// Returns components as a fixed-size 2-element array.
    #[inline]
    pub const fn to_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }

    /// Constructs a `Vector2` from a fixed-size 2-element array.
    #[inline]
    pub const fn from_array(a: [f64; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }

    /// Reads 2 components from slice `array` starting at `offset` into `self`.
    ///
    /// Equivalent to Three.js `Vector2.fromArray(array, offset)` on valid slices.
    ///
    /// Requires `array.len() >= offset + 2`.
    ///
    /// # Panics
    /// Panics if `offset + 2 > array.len()`. If `offset < array.len()`, components
    /// read before the out-of-bounds index will be updated on `self` before panic (partial write).
    #[inline]
    pub fn from_slice_offset(&mut self, array: &[f64], offset: usize) -> &mut Self {
        self.x = array[offset];
        self.y = array[offset + 1];
        self
    }

    /// Writes 2 components of `self` into mutable slice `array` starting at `offset`.
    ///
    /// Equivalent to Three.js `Vector2.toArray(array, offset)` on valid slices.
    ///
    /// Requires `array.len() >= offset + 2`.
    ///
    /// # Panics
    /// Panics if `offset + 2 > array.len()`. If `offset < array.len()`, elements
    /// written before the out-of-bounds index will remain modified in `array` before panic (partial write).
    #[inline]
    pub fn to_slice_offset(&self, array: &mut [f64], offset: usize) {
        array[offset] = self.x;
        array[offset + 1] = self.y;
    }

    /// Narrows this `f64` vector into `[f32; 2]`, verifying that precision loss does not
    /// exceed `max_abs_err` or `max_rel_err`.
    ///
    /// Non-finite values (`NaN`, `Infinity`) are preserved.
    pub fn to_f32_checked(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 2], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, false)
    }

    /// Strict narrowing: rejects non-finite components (`NaN`, `Infinity`) and checks tolerance.
    pub fn to_f32_strict(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 2], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, true)
    }

    /// Narrows this vector with explicit `NarrowingTolerance` parameters.
    pub fn to_f32_with_tolerance(
        &self,
        tolerance: NarrowingTolerance,
        strict: bool,
    ) -> Result<[f32; 2], NarrowingError> {
        self.to_f32_with_policy(tolerance.max_abs_err, tolerance.max_rel_err, strict)
    }

    /// Checked narrowing helper evaluating each component with optional strict non-finite rejection.
    pub fn to_f32_with_policy(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
        strict: bool,
    ) -> Result<[f32; 2], NarrowingError> {
        let x = check_narrow_f64(self.x, 0, max_abs_err, max_rel_err, strict)?;
        let y = check_narrow_f64(self.y, 1, max_abs_err, max_rel_err, strict)?;
        Ok([x, y])
    }
}

impl fmt::Display for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector2({}, {})", self.x, self.y)
    }
}

impl Default for Vector2 {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}
