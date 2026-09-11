//! 3D Triangle matching Three.js r186 `Triangle`.

use core::fmt;
use crate::box3::Box3;
use crate::plane::Plane;
use crate::vector3::Vector3;

/// A geometric triangle in 3D space defined by three corner points `(a, b, c)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Triangle {
    /// First corner vertex.
    pub a: Vector3,
    /// Second corner vertex.
    pub b: Vector3,
    /// Third corner vertex.
    pub c: Vector3,
}

impl Triangle {
    /// Constructs a new triangle with specified vertices `a, b, c`.
    #[inline]
    pub const fn new(a: Vector3, b: Vector3, c: Vector3) -> Self {
        Self { a, b, c }
    }

    /// Sets the triangle's vertices by copying the given values.
    #[inline]
    pub fn set(&mut self, a: Vector3, b: Vector3, c: Vector3) -> &mut Self {
        self.a = a;
        self.b = b;
        self.c = c;
        self
    }

    /// Computes the unit normal vector of a triangle from three vertices.
    ///
    /// If the triangle is degenerate (zero area), returns `(0, 0, 0)`.
    /// Matches Three.js r186 `Triangle.getNormal(a, b, c, target)`.
    pub fn get_normal_of(a: &Vector3, b: &Vector3, c: &Vector3) -> Vector3 {
        let mut target = *c;
        target.sub(b);
        let mut v0 = *a;
        v0.sub(b);
        target.cross(&v0);

        let target_len_sq = target.length_sq();
        if target_len_sq > 0.0 {
            target.multiply_scalar(1.0 / target_len_sq.sqrt());
            target
        } else {
            Vector3::zero()
        }
    }

    /// Computes the unit normal vector of this triangle.
    ///
    /// Returns `(0, 0, 0)` if degenerate.
    /// Matches Three.js r186 `Triangle.getNormal(target)`.
    #[inline]
    pub fn get_normal(&self) -> Vector3 {
        Self::get_normal_of(&self.a, &self.b, &self.c)
    }

    /// Computes the barycentric coordinates `(u_a, u_b, u_c)` for `point` relative to triangle `(a, b, c)`.
    ///
    /// Coordinates sum to 1: `point = u_a * a + u_b * b + u_c * c`.
    /// Returns `None` if the triangle is degenerate (collinear or singular).
    /// Matches Three.js r186 `Triangle.getBarycoord(point, a, b, c, target)`.
    pub fn get_barycoord_of(point: &Vector3, a: &Vector3, b: &Vector3, c: &Vector3) -> Option<Vector3> {
        let mut v0 = *c;
        v0.sub(a);
        let mut v1 = *b;
        v1.sub(a);
        let mut v2 = *point;
        v2.sub(a);

        let dot00 = v0.dot(&v0);
        let dot01 = v0.dot(&v1);
        let dot02 = v0.dot(&v2);
        let dot11 = v1.dot(&v1);
        let dot12 = v1.dot(&v2);

        let denom = dot00 * dot11 - dot01 * dot01;
        if denom == 0.0 {
            return None;
        }

        let inv_denom = 1.0 / denom;
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        Some(Vector3::new(1.0 - u - v, v, u))
    }

    /// Computes the barycentric coordinates for `point` on this triangle.
    ///
    /// Returns `None` if the triangle is degenerate.
    /// Matches Three.js r186 `Triangle.getBarycoord(point, target)`.
    #[inline]
    pub fn get_barycoord(&self, point: &Vector3) -> Option<Vector3> {
        Self::get_barycoord_of(point, &self.a, &self.b, &self.c)
    }

    /// Returns `true` if `point`, when projected onto the triangle's plane, lies within the triangle.
    ///
    /// Returns `false` for degenerate triangles.
    /// Matches Three.js r186 `Triangle.containsPoint(point, a, b, c)`.
    pub fn contains_point_of(point: &Vector3, a: &Vector3, b: &Vector3, c: &Vector3) -> bool {
        match Self::get_barycoord_of(point, a, b, c) {
            None => false,
            Some(bary) => bary.x >= 0.0 && bary.y >= 0.0 && (bary.x + bary.y) <= 1.0,
        }
    }

    /// Returns `true` if `point`, when projected onto this triangle's plane, lies within the triangle.
    ///
    /// Matches Three.js r186 `Triangle.containsPoint(point)`.
    #[inline]
    pub fn contains_point(&self, point: &Vector3) -> bool {
        Self::contains_point_of(point, &self.a, &self.b, &self.c)
    }

    /// Returns `true` if the triangle is oriented towards `direction`.
    ///
    /// Strictly front facing (`dot < 0.0`).
    /// Matches Three.js r186 `Triangle.isFrontFacing(a, b, c, direction)`.
    pub fn is_front_facing_of(a: &Vector3, b: &Vector3, c: &Vector3, direction: &Vector3) -> bool {
        let mut v0 = *c;
        v0.sub(b);
        let mut v1 = *a;
        v1.sub(b);
        v0.cross(&v1).dot(direction) < 0.0
    }

    /// Returns `true` if this triangle is oriented towards `direction`.
    ///
    /// Matches Three.js r186 `Triangle.isFrontFacing(direction)`.
    #[inline]
    pub fn is_front_facing(&self, direction: &Vector3) -> bool {
        Self::is_front_facing_of(&self.a, &self.b, &self.c, direction)
    }

    /// Computes the area of this triangle.
    ///
    /// Matches Three.js r186 `Triangle.getArea()`.
    pub fn get_area(&self) -> f64 {
        let mut v0 = self.c;
        v0.sub(&self.b);
        let mut v1 = self.a;
        v1.sub(&self.b);
        v0.cross(&v1).length() * 0.5
    }

    /// Computes the midpoint of this triangle.
    ///
    /// Matches Three.js r186 `Triangle.getMidpoint(target)`.
    #[inline]
    pub fn get_midpoint(&self) -> Vector3 {
        Vector3::new(
            (self.a.x + self.b.x + self.c.x) / 3.0,
            (self.a.y + self.b.y + self.c.y) / 3.0,
            (self.a.z + self.b.z + self.c.z) / 3.0,
        )
    }

    /// Computes the plane containing this triangle.
    ///
    /// Matches Three.js r186 `Triangle.getPlane(target)`.
    pub fn get_plane(&self) -> Plane {
        let mut plane = Plane::default();
        plane.set_from_coplanar_points(&self.a, &self.b, &self.c);
        plane
    }

    /// Finds the closest point on this triangle to `p` using the Voronoi region walk method.
    ///
    /// Implementation from Christer Ericson's Real-Time Collision Detection (Section 5.1.5).
    /// Matches Three.js r186 `Triangle.closestPointToPoint(p, target)`.
    pub fn closest_point_to_point(&self, p: &Vector3) -> Vector3 {
        let a = self.a;
        let b = self.b;
        let c = self.c;

        let mut vab = b;
        vab.sub(&a);
        let mut vac = c;
        vac.sub(&a);
        let mut vap = *p;
        vap.sub(&a);

        let d1 = vab.dot(&vap);
        let d2 = vac.dot(&vap);
        if d1 <= 0.0 && d2 <= 0.0 {
            // vertex region of A; barycentric coords (1, 0, 0)
            return a;
        }

        let mut vbp = *p;
        vbp.sub(&b);
        let d3 = vab.dot(&vbp);
        let d4 = vac.dot(&vbp);
        if d3 >= 0.0 && d4 <= d3 {
            // vertex region of B; barycentric coords (0, 1, 0)
            return b;
        }

        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            let mut target = a;
            target.add_scaled_vector(&vab, v);
            return target;
        }

        let mut vcp = *p;
        vcp.sub(&c);
        let d5 = vab.dot(&vcp);
        let d6 = vac.dot(&vcp);
        if d6 >= 0.0 && d5 <= d6 {
            // vertex region of C; barycentric coords (0, 0, 1)
            return c;
        }

        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            let mut target = a;
            target.add_scaled_vector(&vac, w);
            return target;
        }

        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let mut vbc = c;
            vbc.sub(&b);
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            let mut target = b;
            target.add_scaled_vector(&vbc, w);
            return target;
        }

        // face region
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;

        let mut target = a;
        target.add_scaled_vector(&vab, v);
        target.add_scaled_vector(&vac, w);
        target
    }

    /// Returns `true` if this triangle intersects with the given box.
    ///
    /// Matches Three.js r186 `Triangle.intersectsBox(box)`.
    #[inline]
    pub fn intersects_box(&self, box3: &Box3) -> bool {
        box3.intersects_triangle(self)
    }

    /// Returns `true` if this triangle equals another component-wise.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c
    }
}

impl Default for Triangle {
    #[inline]
    fn default() -> Self {
        Self {
            a: Vector3::zero(),
            b: Vector3::zero(),
            c: Vector3::zero(),
        }
    }
}

impl fmt::Display for Triangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Triangle(a: {}, b: {}, c: {})", self.a, self.b, self.c)
    }
}
