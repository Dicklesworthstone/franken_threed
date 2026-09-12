//! 3D analytical bounding sphere with `f64` public semantics matching Three.js r186 `Sphere`.

use core::fmt;
use crate::box3::Box3;
use crate::matrix4::Matrix4;
use crate::plane::Plane;
use crate::vector3::Vector3;

/// Analytical 3D sphere defined by a center point and a radius.
///
/// Follows Three.js r186 empty conventions:
/// - An empty sphere has `center = (0, 0, 0)` and `radius = -1.0`.
/// - `is_empty()` returns `true` if `radius < 0.0`.
/// - A sphere with `radius == 0.0` is not empty (it contains its center point).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sphere {
    /// Center of the sphere.
    pub center: Vector3,
    /// Radius of the sphere. Negative values indicate an empty sphere.
    pub radius: f64,
}

impl Sphere {
    /// Constructs a new sphere with specified center and radius.
    #[inline]
    pub const fn new(center: Vector3, radius: f64) -> Self {
        Self { center, radius }
    }

    /// Constructs an empty sphere matching Three.js r186:
    /// `center = (0.0, 0.0, 0.0)`, `radius = -1.0`.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            center: Vector3::zero(),
            radius: -1.0,
        }
    }

    /// Sets center and radius of this sphere.
    ///
    /// Matches Three.js r186 `Sphere.set(center, radius)`.
    #[inline]
    pub fn set(&mut self, center: Vector3, radius: f64) -> &mut Self {
        self.center = center;
        self.radius = radius;
        self
    }

    /// Resets this sphere to the empty state (`radius = -1.0`).
    ///
    /// Matches Three.js r186 `Sphere.makeEmpty()`.
    #[inline]
    pub fn make_empty(&mut self) -> &mut Self {
        self.center = Vector3::zero();
        self.radius = -1.0;
        self
    }

    /// Returns `true` if this sphere is empty (`radius < 0.0`).
    ///
    /// Matches Three.js r186 `Sphere.isEmpty()`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.radius < 0.0
    }

    /// Computes the bounding sphere enclosing a slice of points.
    ///
    /// If `optional_center` is provided, that center is used; otherwise the center of
    /// the axis-aligned bounding box enclosing the points is computed.
    /// Matches Three.js r186 `Sphere.setFromPoints(points, optionalCenter)`.
    pub fn set_from_points(&mut self, points: &[Vector3], optional_center: Option<Vector3>) -> &mut Self {
        if let Some(center) = optional_center {
            self.center = center;
        } else {
            let mut box3 = Box3::empty();
            box3.set_from_points(points);
            self.center = box3.get_center();
        }

        let mut max_radius_sq = 0.0f64;
        for p in points {
            max_radius_sq = max_radius_sq.max(self.center.distance_to_squared(p));
        }

        self.radius = max_radius_sq.sqrt();
        self
    }

    /// Returns `true` if this sphere contains the given point (distance <= radius).
    ///
    /// Matches Three.js r186 `Sphere.containsPoint(point)`.
    #[inline]
    pub fn contains_point(&self, point: &Vector3) -> bool {
        point.distance_to_squared(&self.center) <= self.radius * self.radius
    }

    /// Returns the signed distance from the sphere boundary to the specified point.
    ///
    /// Negative distances indicate the point is inside the sphere.
    /// Matches Three.js r186 `Sphere.distanceToPoint(point)`.
    #[inline]
    pub fn distance_to_point(&self, point: &Vector3) -> f64 {
        point.distance_to(&self.center) - self.radius
    }

    /// Returns `true` if this sphere intersects with another sphere.
    ///
    /// Matches Three.js r186 `Sphere.intersectsSphere(sphere)`.
    #[inline]
    pub fn intersects_sphere(&self, other: &Sphere) -> bool {
        let radius_sum = self.radius + other.radius;
        self.center.distance_to_squared(&other.center) <= radius_sum * radius_sum
    }

    /// Returns `true` if this sphere intersects with the given bounding box.
    ///
    /// Matches Three.js r186 `Sphere.intersectsBox(box)`.
    #[inline]
    pub fn intersects_box(&self, box3: &Box3) -> bool {
        box3.intersects_sphere(self)
    }

    /// Returns `true` if this sphere intersects with the given plane.
    ///
    /// Matches Three.js r186 `Sphere.intersectsPlane(plane)`.
    #[inline]
    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        plane.distance_to_point(&self.center).abs() <= self.radius
    }

    /// Clamps a point to the surface of the sphere if outside; points inside are returned unmodified.
    ///
    /// Matches Three.js r186 `Sphere.clampPoint(point, target)`.
    pub fn clamp_point(&self, point: &Vector3) -> Vector3 {
        let delta_length_sq = self.center.distance_to_squared(point);
        if delta_length_sq > self.radius * self.radius {
            let mut target = *point;
            target.sub(&self.center).normalize().multiply_scalar(self.radius).add(&self.center);
            target
        } else {
            *point
        }
    }

    /// Returns the minimum axis-aligned bounding box enclosing this sphere.
    ///
    /// Empty spheres return an empty box.
    /// Matches Three.js r186 `Sphere.getBoundingBox(target)`.
    #[inline]
    pub fn get_bounding_box(&self) -> Box3 {
        if self.is_empty() {
            Box3::empty()
        } else {
            let r = Vector3::new(self.radius, self.radius, self.radius);
            let mut min = self.center;
            min.sub(&r);
            let mut max = self.center;
            max.add(&r);
            Box3::new(min, max)
        }
    }

    /// Transforms this sphere by the given 4x4 transformation matrix.
    ///
    /// Evaluates `center.apply_matrix4(matrix)` and `radius *= matrix.get_max_scale_on_axis()`.
    /// Matches Three.js r186 `Sphere.applyMatrix4(matrix)`.
    pub fn apply_matrix4(&mut self, matrix: &Matrix4) -> &mut Self {
        self.center.apply_matrix4(matrix);
        self.radius *= matrix.get_max_scale_on_axis();
        self
    }

    /// Translates this sphere's center by the given offset vector.
    ///
    /// Matches Three.js r186 `Sphere.translate(offset)`.
    #[inline]
    pub fn translate(&mut self, offset: &Vector3) -> &mut Self {
        self.center.add(offset);
        self
    }

    /// Expands this sphere to enclose the given point.
    ///
    /// Matches Three.js r186 `Sphere.expandByPoint(point)`.
    pub fn expand_by_point(&mut self, point: &Vector3) -> &mut Self {
        if self.is_empty() {
            self.center = *point;
            self.radius = 0.0;
            return self;
        }

        let mut v = *point;
        v.sub(&self.center);

        let length_sq = v.length_sq();
        if length_sq > self.radius * self.radius {
            let length = length_sq.sqrt();
            let delta = (length - self.radius) * 0.5;
            self.center.add_scaled_vector(&v, delta / length);
            self.radius += delta;
        }

        self
    }

    /// Expands this sphere to enclose both this instance and another sphere.
    ///
    /// Matches Three.js r186 `Sphere.union(sphere)`.
    pub fn union(&mut self, other: &Sphere) -> &mut Self {
        if other.is_empty() {
            return self;
        }
        if self.is_empty() {
            self.center = other.center;
            self.radius = other.radius;
            return self;
        }

        let d2 = self.center.distance_to_squared(&other.center);
        let r_diff = other.radius - self.radius;

        // If one sphere encloses the other
        if r_diff * r_diff >= d2 {
            if r_diff >= 0.0 {
                self.center = other.center;
                self.radius = other.radius;
            }
            return self;
        }

        let d = d2.sqrt();
        let radius = (d + self.radius + other.radius) * 0.5;

        let mut v = other.center;
        v.sub(&self.center);
        self.center.add_scaled_vector(&v, (radius - self.radius) / d);
        self.radius = radius;

        self
    }

    /// Returns `true` if this sphere equals another sphere component-wise.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.center == other.center && self.radius == other.radius
    }
}

impl Default for Sphere {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for Sphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sphere(center: {}, radius: {})", self.center, self.radius)
    }
}
