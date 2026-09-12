//! 3D axis-aligned bounding box (AABB) with `f64` public semantics matching Three.js r186 `Box3`.

use core::fmt;
use crate::jsnum::{js_max, js_min};
use crate::matrix4::Matrix4;
use crate::plane::Plane;
use crate::sphere::Sphere;
use crate::triangle::Triangle;
use crate::vector3::Vector3;

/// Axis-aligned bounding box (AABB) represented by minimum and maximum corner points.
///
/// Follows Three.js r186 empty conventions:
/// - An empty box has `min = (+Infinity, +Infinity, +Infinity)` and `max = (-Infinity, -Infinity, -Infinity)`.
/// - `is_empty()` returns `true` when any `max` component is strictly less than its `min` component.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Box3 {
    /// Lower boundary corner point.
    pub min: Vector3,
    /// Upper boundary corner point.
    pub max: Vector3,
}

impl Box3 {
    /// Constructs a bounding box with specified `min` and `max` corners.
    #[inline]
    pub const fn new(min: Vector3, max: Vector3) -> Self {
        Self { min, max }
    }

    /// Constructs an empty bounding box matching Three.js r186 default constructor:
    /// `min = (+Infinity, +Infinity, +Infinity)`, `max = (-Infinity, -Infinity, -Infinity)`.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            min: Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    /// Resets this bounding box to the empty state (+inf / -inf).
    ///
    /// Matches Three.js r186 `Box3.makeEmpty()`.
    #[inline]
    pub fn make_empty(&mut self) -> &mut Self {
        self.min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        self.max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        self
    }

    /// Returns `true` if this box is empty (i.e. contains zero points).
    ///
    /// Evaluates `max.x < min.x || max.y < min.y || max.z < min.z` matching Three.js r186 `Box3.isEmpty()`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.max.x < self.min.x || self.max.y < self.min.y || self.max.z < self.min.z
    }

    /// Sets the lower and upper bounds of this box.
    ///
    /// Matches Three.js r186 `Box3.set(min, max)`.
    #[inline]
    pub fn set(&mut self, min: Vector3, max: Vector3) -> &mut Self {
        self.min = min;
        self.max = max;
        self
    }

    /// Copies the values of `other` to this box.
    ///
    /// Matches Three.js r186 `Box3.copy(box)`.
    #[inline]
    pub fn copy(&mut self, other: &Self) -> &mut Self {
        self.min = other.min;
        self.max = other.max;
        self
    }

    /// Computes the bounding box enclosing the given points.
    ///
    /// Resets this box to empty, then expands by each point.
    /// Matches Three.js r186 `Box3.setFromPoints(points)`.
    pub fn set_from_points(&mut self, points: &[Vector3]) -> &mut Self {
        self.make_empty();
        for p in points {
            self.expand_by_point(p);
        }
        self
    }

    /// Sets the upper and lower bounds of this box so it encloses the position data
    /// in the given buffer.
    ///
    /// Missing trailing coordinates evaluate to `NaN` per Three.js r186 `_vector.fromArray(array, i)`.
    /// Matches Three.js r186 `Box3.setFromArray(array)`.
    pub fn set_from_buffer(&mut self, buffer: &[f64]) -> &mut Self {
        self.make_empty();
        for chunk in buffer.chunks(3) {
            let x = chunk[0];
            let y = if chunk.len() > 1 { chunk[1] } else { f64::NAN };
            let z = if chunk.len() > 2 { chunk[2] } else { f64::NAN };
            self.expand_by_point(&Vector3::new(x, y, z));
        }
        self
    }

    /// Sets the upper and lower bounds of this box so it encloses the position data
    /// in the given array.
    ///
    /// Missing trailing coordinates evaluate to `NaN` per Three.js r186 `_vector.fromArray(array, i)`.
    /// Matches Three.js r186 `Box3.setFromArray(array)`.
    #[inline]
    pub fn set_from_array(&mut self, array: &[f64]) -> &mut Self {
        self.set_from_buffer(array)
    }

    /// Centers this box on the given center vector and sets dimensions to the given size values.
    ///
    /// Matches Three.js r186 `Box3.setFromCenterAndSize(center, size)`.
    #[inline]
    pub fn set_from_center_and_size(&mut self, center: &Vector3, size: &Vector3) -> &mut Self {
        let half_size = Vector3::new(size.x * 0.5, size.y * 0.5, size.z * 0.5);
        self.min = Vector3::new(center.x - half_size.x, center.y - half_size.y, center.z - half_size.z);
        self.max = Vector3::new(center.x + half_size.x, center.y + half_size.y, center.z + half_size.z);
        self
    }

    /// Expands the boundaries of this box to include the given point.
    ///
    /// Evaluates `min = Math.min(min, point)` and `max = Math.max(max, point)` component-wise
    /// using ECMAScript semantics (propagating NaN and respecting signed zero).
    /// Matches Three.js r186 `Box3.expandByPoint(point)`.
    #[inline]
    pub fn expand_by_point(&mut self, point: &Vector3) -> &mut Self {
        self.min.x = js_min(self.min.x, point.x);
        self.min.y = js_min(self.min.y, point.y);
        self.min.z = js_min(self.min.z, point.z);

        self.max.x = js_max(self.max.x, point.x);
        self.max.y = js_max(self.max.y, point.y);
        self.max.z = js_max(self.max.z, point.z);
        self
    }

    /// Expands this box equilaterally by subtracting vector from min and adding vector to max.
    ///
    /// Matches Three.js r186 `Box3.expandByVector(vector)`.
    #[inline]
    pub fn expand_by_vector(&mut self, vector: &Vector3) -> &mut Self {
        self.min.sub(vector);
        self.max.add(vector);
        self
    }

    /// Expands each dimension of the box by scalar.
    ///
    /// Matches Three.js r186 `Box3.expandByScalar(scalar)`.
    #[inline]
    pub fn expand_by_scalar(&mut self, scalar: f64) -> &mut Self {
        self.min.x -= scalar;
        self.min.y -= scalar;
        self.min.z -= scalar;
        self.max.x += scalar;
        self.max.y += scalar;
        self.max.z += scalar;
        self
    }

    /// Returns `true` if the given point lies inside or on the boundaries of this box.
    ///
    /// Matches Three.js r186 `Box3.containsPoint(point)`.
    #[inline]
    pub fn contains_point(&self, point: &Vector3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Returns `true` if this box entirely encloses the given box.
    ///
    /// Matches Three.js r186 `Box3.containsBox(box)`.
    #[inline]
    pub fn contains_box(&self, other: &Box3) -> bool {
        self.min.x <= other.min.x
            && other.max.x <= self.max.x
            && self.min.y <= other.min.y
            && other.max.y <= self.max.y
            && self.min.z <= other.min.z
            && other.max.z <= self.max.z
    }

    /// Returns `true` if this bounding box intersects the given bounding box.
    ///
    /// Matches Three.js r186 `Box3.intersectsBox(box)`:
    /// Uses 6 splitting planes to test overlap.
    #[inline]
    pub fn intersects_box(&self, other: &Box3) -> bool {
        other.max.x >= self.min.x
            && other.min.x <= self.max.x
            && other.max.y >= self.min.y
            && other.min.y <= self.max.y
            && other.max.z >= self.min.z
            && other.min.z <= self.max.z
    }

    /// Returns `true` if this bounding box intersects the given sphere.
    ///
    /// Matches Three.js r186 `Box3.intersectsSphere(sphere)`:
    /// Finds the closest point on the AABB to the sphere center and tests if distance <= radius.
    #[inline]
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        let closest = self.clamp_point(&sphere.center);
        closest.distance_to_squared(&sphere.center) <= sphere.radius * sphere.radius
    }

    /// Returns `true` if the given plane intersects with this bounding box.
    ///
    /// Matches Three.js r186 `Box3.intersectsPlane(plane)`.
    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        let min_val;
        let max_val;

        if plane.normal.x > 0.0 {
            min_val = plane.normal.x * self.min.x;
            max_val = plane.normal.x * self.max.x;
        } else {
            min_val = plane.normal.x * self.max.x;
            max_val = plane.normal.x * self.min.x;
        }

        let (min_val, max_val) = if plane.normal.y > 0.0 {
            (min_val + plane.normal.y * self.min.y, max_val + plane.normal.y * self.max.y)
        } else {
            (min_val + plane.normal.y * self.max.y, max_val + plane.normal.y * self.min.y)
        };

        let (min_val, max_val) = if plane.normal.z > 0.0 {
            (min_val + plane.normal.z * self.min.z, max_val + plane.normal.z * self.max.z)
        } else {
            (min_val + plane.normal.z * self.max.z, max_val + plane.normal.z * self.min.z)
        };

        min_val <= -plane.constant && max_val >= -plane.constant
    }

    /// Returns `true` if the given triangle intersects with this bounding box using SAT.
    ///
    /// Matches Three.js r186 `Box3.intersectsTriangle(triangle)`.
    pub fn intersects_triangle(&self, triangle: &Triangle) -> bool {
        if self.is_empty() {
            return false;
        }

        let center = self.get_center();
        let mut extents = self.max;
        extents.sub(&center);

        let mut v0 = triangle.a;
        v0.sub(&center);
        let mut v1 = triangle.b;
        v1.sub(&center);
        let mut v2 = triangle.c;
        v2.sub(&center);

        let mut f0 = v1;
        f0.sub(&v0);
        let mut f1 = v2;
        f1.sub(&v1);
        let mut f2 = v0;
        f2.sub(&v2);

        // 9 axes formed by cross products of AABB face normals with triangle edges
        let axes = [
            Vector3::new(0.0, -f0.z, f0.y),
            Vector3::new(0.0, -f1.z, f1.y),
            Vector3::new(0.0, -f2.z, f2.y),
            Vector3::new(f0.z, 0.0, -f0.x),
            Vector3::new(f1.z, 0.0, -f1.x),
            Vector3::new(f2.z, 0.0, -f2.x),
            Vector3::new(-f0.y, f0.x, 0.0),
            Vector3::new(-f1.y, f1.x, 0.0),
            Vector3::new(-f2.y, f2.x, 0.0),
        ];

        for axis in &axes {
            if !sat_for_axis(axis, &v0, &v1, &v2, &extents) {
                return false;
            }
        }

        // 3 face normals of AABB
        let aabb_normals = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        for axis in &aabb_normals {
            if !sat_for_axis(axis, &v0, &v1, &v2, &extents) {
                return false;
            }
        }

        // Face normal of triangle
        let mut tri_normal = f0;
        tri_normal.cross(&f1);
        sat_for_axis(&tri_normal, &v0, &v1, &v2, &extents)
    }

    /// Clamps the given point to the boundaries of this box.
    ///
    /// Matches Three.js r186 `Box3.clampPoint(point, target)`.
    #[inline]
    pub fn clamp_point(&self, point: &Vector3) -> Vector3 {
        Vector3::new(
            clamp_scalar(point.x, self.min.x, self.max.x),
            clamp_scalar(point.y, self.min.y, self.max.y),
            clamp_scalar(point.z, self.min.z, self.max.z),
        )
    }

    /// Returns the euclidean distance from any boundary of this box to the specified point.
    ///
    /// Returns `0.0` if the point lies inside the box.
    /// Matches Three.js r186 `Box3.distanceToPoint(point)`.
    #[inline]
    pub fn distance_to_point(&self, point: &Vector3) -> f64 {
        let clamped = self.clamp_point(point);
        clamped.distance_to(point)
    }

    /// Returns the center point of this box.
    ///
    /// Returns `(0.0, 0.0, 0.0)` if this box is empty.
    /// Matches Three.js r186 `Box3.getCenter(target)`.
    #[inline]
    pub fn get_center(&self) -> Vector3 {
        if self.is_empty() {
            Vector3::zero()
        } else {
            Vector3::new(
                (self.min.x + self.max.x) * 0.5,
                (self.min.y + self.max.y) * 0.5,
                (self.min.z + self.max.z) * 0.5,
            )
        }
    }

    /// Returns the dimensions (width, height, depth) of this box.
    ///
    /// Returns `(0.0, 0.0, 0.0)` if this box is empty.
    /// Matches Three.js r186 `Box3.getSize(target)`.
    #[inline]
    pub fn get_size(&self) -> Vector3 {
        if self.is_empty() {
            Vector3::zero()
        } else {
            Vector3::new(
                self.max.x - self.min.x,
                self.max.y - self.min.y,
                self.max.z - self.min.z,
            )
        }
    }

    /// Returns a bounding sphere that encloses this bounding box.
    ///
    /// Matches Three.js r186 `Box3.getBoundingSphere(target)`.
    #[inline]
    pub fn get_bounding_sphere<'a>(&self, target: &'a mut Sphere) -> &'a mut Sphere {
        if self.is_empty() {
            target.make_empty();
        } else {
            target.center = self.get_center();
            target.radius = self.get_size().length() * 0.5;
        }
        target
    }

    /// Returns a point as a proportion of this box's width, height, and depth.
    ///
    /// Preserves direct divide-by-zero to `f64::INFINITY`, `f64::NEG_INFINITY`, or `f64::NAN`.
    /// Matches Three.js r186 `Box3.getParameter(point, target)`.
    #[inline]
    pub fn get_parameter<'a>(&self, point: &Vector3, target: &'a mut Vector3) -> &'a mut Vector3 {
        target.set(
            (point.x - self.min.x) / (self.max.x - self.min.x),
            (point.y - self.min.y) / (self.max.y - self.min.y),
            (point.z - self.min.z) / (self.max.z - self.min.z),
        )
    }

    /// Computes the union of this box and another, expanding bounds to enclose both.
    ///
    /// Evaluates `min = min(min, other.min)` and `max = max(max, other.max)`.
    /// Matches Three.js r186 `Box3.union(box)`.
    #[inline]
    pub fn union(&mut self, other: &Box3) -> &mut Self {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);

        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
        self
    }

    /// Computes the intersection of this box and another.
    ///
    /// Sets this box to empty if there is no overlap.
    /// Matches Three.js r186 `Box3.intersect(box)`.
    #[inline]
    pub fn intersect(&mut self, other: &Box3) -> &mut Self {
        self.min.x = self.min.x.max(other.min.x);
        self.min.y = self.min.y.max(other.min.y);
        self.min.z = self.min.z.max(other.min.z);

        self.max.x = self.max.x.min(other.max.x);
        self.max.y = self.max.y.min(other.max.y);
        self.max.z = self.max.z.min(other.max.z);

        if self.is_empty() {
            self.make_empty();
        }
        self
    }

    /// Transforms this bounding box by a 4x4 matrix using the 8-corner method.
    ///
    /// Matches Three.js r186 `Box3.applyMatrix4(matrix)`:
    /// - An empty box transforms into an empty box without modification.
    /// - For non-empty boxes, all 8 corners are transformed by the matrix (including perspective divide),
    ///   and a new bounding box is constructed from the transformed corners.
    pub fn apply_matrix4(&mut self, matrix: &Matrix4) -> &mut Self {
        if self.is_empty() {
            return self;
        }

        let min = self.min;
        let max = self.max;

        let mut points = [
            Vector3::new(min.x, min.y, min.z), // 000
            Vector3::new(min.x, min.y, max.z), // 001
            Vector3::new(min.x, max.y, min.z), // 010
            Vector3::new(min.x, max.y, max.z), // 011
            Vector3::new(max.x, min.y, min.z), // 100
            Vector3::new(max.x, min.y, max.z), // 101
            Vector3::new(max.x, max.y, min.z), // 110
            Vector3::new(max.x, max.y, max.z), // 111
        ];

        for p in &mut points {
            p.apply_matrix4(matrix);
        }

        self.set_from_points(&points);
        self
    }

    /// Translates this bounding box by the specified offset vector.
    ///
    /// Matches Three.js r186 `Box3.translate(offset)`.
    #[inline]
    pub fn translate(&mut self, offset: &Vector3) -> &mut Self {
        self.min.add(offset);
        self.max.add(offset);
        self
    }

    /// Returns `true` if this box equals another box component-wise.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max
    }
}

impl Default for Box3 {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for Box3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Box3(min: {}, max: {})", self.min, self.max)
    }
}

#[inline]
fn clamp_scalar(value: f64, min: f64, max: f64) -> f64 {
    js_max(min, js_min(max, value))
}

#[inline]
fn sat_for_axis(axis: &Vector3, v0: &Vector3, v1: &Vector3, v2: &Vector3, extents: &Vector3) -> bool {
    let r = extents.x * axis.x.abs() + extents.y * axis.y.abs() + extents.z * axis.z.abs();
    let p0 = v0.dot(axis);
    let p1 = v1.dot(axis);
    let p2 = v2.dot(axis);
    let max_p = p0.max(p1).max(p2);
    let min_p = p0.min(p1).min(p2);
    (-max_p).max(min_p) <= r
}
