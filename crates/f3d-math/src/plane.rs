//! 2D infinite plane in 3D space in Hessian normal form matching Three.js r186 `Plane`.

use core::fmt;
use crate::box3::Box3;
use crate::line3::Line3;
use crate::matrix3::Matrix3;
use crate::matrix4::Matrix4;
use crate::sphere::Sphere;
use crate::vector3::Vector3;

/// A 2D plane extending infinitely in 3D space represented in Hessian normal form
/// by a unit length normal vector and a constant signed distance from the origin.
///
/// Follows Three.js r186:
/// - Normal vector `normal` (unit length).
/// - Constant `constant`: signed distance from the origin (`dot(normal, x) + constant = 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Plane {
    /// Unit length vector defining the normal of the plane.
    pub normal: Vector3,
    /// Signed distance from the origin to the plane.
    pub constant: f64,
}

impl Plane {
    /// Constructs a plane with given normal and constant.
    #[inline]
    pub const fn new(normal: Vector3, constant: f64) -> Self {
        Self { normal, constant }
    }

    /// Sets normal and constant components.
    #[inline]
    pub fn set(&mut self, normal: Vector3, constant: f64) -> &mut Self {
        self.normal = normal;
        self.constant = constant;
        self
    }

    /// Sets normal `(x, y, z)` components and constant `w`.
    #[inline]
    pub fn set_components(&mut self, x: f64, y: f64, z: f64, w: f64) -> &mut Self {
        self.normal.set(x, y, z);
        self.constant = w;
        self
    }

    /// Sets the plane from a normal vector and a coplanar point.
    ///
    /// Evaluates `normal = *normal` and `constant = -point.dot(normal)`.
    /// Matches Three.js r186 `Plane.setFromNormalAndCoplanarPoint(normal, point)`.
    #[inline]
    pub fn set_from_normal_and_coplanar_point(&mut self, normal: &Vector3, point: &Vector3) -> &mut Self {
        self.normal = *normal;
        self.constant = -point.dot(&self.normal);
        self
    }

    /// Sets the plane from three coplanar points in counter-clockwise order.
    ///
    /// Evaluates `normal = normalize((c - b) x (a - b))` and `constant = -a.dot(normal)`.
    /// Matches Three.js r186 `Plane.setFromCoplanarPoints(a, b, c)`.
    pub fn set_from_coplanar_points(&mut self, a: &Vector3, b: &Vector3, c: &Vector3) -> &mut Self {
        let mut v1 = *c;
        v1.sub(b);
        let mut v2 = *a;
        v2.sub(b);
        v1.cross(&v2).normalize();
        self.set_from_normal_and_coplanar_point(&v1, a);
        self
    }

    /// Normalizes the plane normal and adjusts constant accordingly.
    ///
    /// Matches Three.js r186 `Plane.normalize()`.
    pub fn normalize(&mut self) -> &mut Self {
        let inv_len = 1.0 / self.normal.length();
        self.normal.multiply_scalar(inv_len);
        self.constant *= inv_len;
        self
    }

    /// Negates both the normal vector and the constant.
    ///
    /// Matches Three.js r186 `Plane.negate()`.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.constant = -self.constant;
        self.normal.negate();
        self
    }

    /// Returns the signed distance from the given point to this plane.
    ///
    /// Positive if the point is in front of the plane (direction of normal), negative if behind.
    /// Matches Three.js r186 `Plane.distanceToPoint(point)`.
    #[inline]
    pub fn distance_to_point(&self, point: &Vector3) -> f64 {
        self.normal.dot(point) + self.constant
    }

    /// Returns the signed distance from the given sphere to this plane.
    ///
    /// Matches Three.js r186 `Plane.distanceToSphere(sphere)`.
    #[inline]
    pub fn distance_to_sphere(&self, sphere: &Sphere) -> f64 {
        self.distance_to_point(&sphere.center) - sphere.radius
    }

    /// Projects the given point onto this plane.
    ///
    /// Evaluates `point - normal * distance_to_point(point)`.
    /// Matches Three.js r186 `Plane.projectPoint(point, target)`.
    #[inline]
    pub fn project_point(&self, point: &Vector3) -> Vector3 {
        let d = self.distance_to_point(point);
        let mut target = *point;
        target.add_scaled_vector(&self.normal, -d);
        target
    }

    /// Returns a coplanar point by projecting the normal at origin onto the plane.
    ///
    /// Evaluates `normal * -constant`.
    /// Matches Three.js r186 `Plane.coplanarPoint(target)`.
    #[inline]
    pub fn coplanar_point(&self) -> Vector3 {
        let mut target = self.normal;
        target.multiply_scalar(-self.constant);
        target
    }

    /// Computes the intersection point between a 3D line segment and this plane.
    ///
    /// - Returns `None` if the line is parallel and not coplanar, or if `clamp_to_line` is true and `t < 0 || t > 1`.
    /// - Returns `Some(line.start)` if the line is coplanar (`distance_to_point(line.start) == 0.0`).
    /// Matches Three.js r186 `Plane.intersectLine(line, target, clampToLine)`.
    pub fn intersect_line(&self, line: &Line3, clamp_to_line: bool) -> Option<Vector3> {
        let direction = line.delta();
        let denominator = self.normal.dot(&direction);

        if denominator == 0.0 {
            if self.distance_to_point(&line.start) == 0.0 {
                return Some(line.start);
            }
            return None;
        }

        let t = -(line.start.dot(&self.normal) + self.constant) / denominator;
        if clamp_to_line && (t < 0.0 || t > 1.0) {
            return None;
        }

        let mut target = line.start;
        target.add_scaled_vector(&direction, t);
        Some(target)
    }

    /// Returns `true` if the given line segment passes through the plane.
    ///
    /// Matches Three.js r186 `Plane.intersectsLine(line)`.
    #[inline]
    pub fn intersects_line(&self, line: &Line3) -> bool {
        let start_sign = self.distance_to_point(&line.start);
        let end_sign = self.distance_to_point(&line.end);
        (start_sign < 0.0 && end_sign > 0.0) || (end_sign < 0.0 && start_sign > 0.0)
    }

    /// Returns `true` if the given bounding box intersects with the plane.
    ///
    /// Matches Three.js r186 `Plane.intersectsBox(box)`.
    #[inline]
    pub fn intersects_box(&self, box3: &Box3) -> bool {
        box3.intersects_plane(self)
    }

    /// Returns `true` if the given sphere intersects with the plane.
    ///
    /// Matches Three.js r186 `Plane.intersectsSphere(sphere)`.
    #[inline]
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        sphere.intersects_plane(self)
    }

    /// Transforms this plane with a 4x4 affine matrix.
    ///
    /// An optional pre-computed normal matrix (inverse transpose of upper 3x3) can be passed.
    /// If `None`, it is derived from `matrix.get_normal_matrix()`.
    /// Matches Three.js r186 `Plane.applyMatrix4(matrix, optionalNormalMatrix)`.
    pub fn apply_matrix4(&mut self, matrix: &Matrix4, optional_normal_matrix: Option<&Matrix3>) -> &mut Self {
        let normal_matrix = match optional_normal_matrix {
            Some(nm) => *nm,
            None => {
                let mut nm = Matrix3::identity();
                nm.get_normal_matrix(matrix);
                nm
            }
        };

        let mut ref_point = self.coplanar_point();
        ref_point.apply_matrix4(matrix);

        let mut normal = self.normal;
        normal.apply_matrix3(&normal_matrix).normalize();

        self.constant = -ref_point.dot(&normal);
        self.normal = normal;
        self
    }

    /// Translates this plane along an offset vector.
    ///
    /// Note: only affects `constant`; does not change `normal`.
    /// Matches Three.js r186 `Plane.translate(offset)`.
    #[inline]
    pub fn translate(&mut self, offset: &Vector3) -> &mut Self {
        self.constant -= offset.dot(&self.normal);
        self
    }

    /// Returns `true` if this plane equals another component-wise.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.normal == other.normal && self.constant == other.constant
    }
}

impl Default for Plane {
    #[inline]
    fn default() -> Self {
        Self {
            normal: Vector3::new(1.0, 0.0, 0.0),
            constant: 0.0,
        }
    }
}

impl fmt::Display for Plane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Plane(normal: {}, constant: {})", self.normal, self.constant)
    }
}
