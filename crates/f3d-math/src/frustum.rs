//! 3D Frustum with `f64` public semantics matching Three.js r186 `Frustum`.
//!
//! Frustums define a camera's viewing volume enclosed by 6 planes (Right, Left,
//! Bottom, Top, Far, Near). They are used for visibility determination and view-frustum culling.

use core::fmt;
use crate::box3::Box3;
use crate::matrix4::{CoordinateSystem, Matrix4};
use crate::plane::Plane;
use crate::sphere::Sphere;
use crate::vector3::Vector3;

/// 3D frustum enclosed by six boundary planes matching Three.js r186 `Frustum`.
///
/// Planes order:
/// - `planes[0]`: Right plane
/// - `planes[1]`: Left plane
/// - `planes[2]`: Bottom plane
/// - `planes[3]`: Top plane
/// - `planes[4]`: Far plane
/// - `planes[5]`: Near plane
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Frustum {
    /// The six bounding planes enclosing the frustum.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Constructs a new frustum from six explicit planes.
    #[inline]
    pub const fn new(p0: Plane, p1: Plane, p2: Plane, p3: Plane, p4: Plane, p5: Plane) -> Self {
        Self {
            planes: [p0, p1, p2, p3, p4, p5],
        }
    }

    /// Sets the frustum planes by copying the six given planes.
    #[inline]
    pub fn set(
        &mut self,
        p0: &Plane,
        p1: &Plane,
        p2: &Plane,
        p3: &Plane,
        p4: &Plane,
        p5: &Plane,
    ) -> &mut Self {
        self.planes[0] = *p0;
        self.planes[1] = *p1;
        self.planes[2] = *p2;
        self.planes[3] = *p3;
        self.planes[4] = *p4;
        self.planes[5] = *p5;
        self
    }

    /// Copies the planes from another frustum into this instance.
    #[inline]
    pub fn copy(&mut self, source: &Self) -> &mut Self {
        self.planes = source.planes;
        self
    }

    /// Sets the frustum planes from a projection matrix matching Three.js r186 `setFromProjectionMatrix`.
    ///
    /// Preserves exact formula and scalar operation order:
    /// - Plane 0 (Right):  `row3 - row0` normalized
    /// - Plane 1 (Left):   `row3 + row0` normalized
    /// - Plane 2 (Bottom): `row3 + row1` normalized
    /// - Plane 3 (Top):    `row3 - row1` normalized
    /// - When `reversed_depth` is `true`:
    ///   - Plane 4 (Far):  `row2` normalized
    ///   - Plane 5 (Near): `row3 - row2` normalized
    /// - When `reversed_depth` is `false`:
    ///   - Plane 4 (Far):  `row3 - row2` normalized
    ///   - Plane 5 (Near): `row3 + row2` normalized (WebGL [-1, 1]) or `row2` normalized (WebGPU [0, 1])
    pub fn set_from_projection_matrix(
        &mut self,
        m: &Matrix4,
        coordinate_system: CoordinateSystem,
        reversed_depth: bool,
    ) -> &mut Self {
        let me = &m.elements;
        let me0 = me[0];   let me1 = me[1];   let me2 = me[2];   let me3 = me[3];
        let me4 = me[4];   let me5 = me[5];   let me6 = me[6];   let me7 = me[7];
        let me8 = me[8];   let me9 = me[9];   let me10 = me[10]; let me11 = me[11];
        let me12 = me[12]; let me13 = me[13]; let me14 = me[14]; let me15 = me[15];

        self.planes[0].set_components(me3 - me0, me7 - me4, me11 - me8, me15 - me12).normalize();
        self.planes[1].set_components(me3 + me0, me7 + me4, me11 + me8, me15 + me12).normalize();
        self.planes[2].set_components(me3 + me1, me7 + me5, me11 + me9, me15 + me13).normalize();
        self.planes[3].set_components(me3 - me1, me7 - me5, me11 - me9, me15 - me13).normalize();

        if reversed_depth {
            self.planes[4].set_components(me2, me6, me10, me14).normalize();
            self.planes[5].set_components(me3 - me2, me7 - me6, me11 - me10, me15 - me14).normalize();
        } else {
            self.planes[4].set_components(me3 - me2, me7 - me6, me11 - me10, me15 - me14).normalize();

            match coordinate_system {
                CoordinateSystem::WebGL => {
                    self.planes[5].set_components(me3 + me2, me7 + me6, me11 + me10, me15 + me14).normalize();
                }
                CoordinateSystem::WebGPU => {
                    self.planes[5].set_components(me2, me6, me10, me14).normalize();
                }
            }
        }

        self
    }

    /// Constructs a frustum from a projection matrix matching Three.js r186.
    #[inline]
    pub fn from_projection_matrix(
        m: &Matrix4,
        coordinate_system: CoordinateSystem,
        reversed_depth: bool,
    ) -> Self {
        let mut f = Self::default();
        f.set_from_projection_matrix(m, coordinate_system, reversed_depth);
        f
    }

    /// Returns `true` if the given point lies within the frustum matching Three.js r186 `containsPoint`.
    ///
    /// Evaluates `distance_to_point(point) < 0.0` across all 6 planes.
    #[inline]
    pub fn contains_point(&self, point: &Vector3) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(point) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the given bounding sphere intersects this frustum matching Three.js r186 `intersectsSphere`.
    ///
    /// Evaluates `distance_to_point(center) < -radius` across all 6 planes.
    #[inline]
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        let neg_radius = -sphere.radius;
        for plane in &self.planes {
            let distance = plane.distance_to_point(&sphere.center);
            if distance < neg_radius {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the given bounding box intersects this frustum matching Three.js r186 `intersectsBox`.
    ///
    /// For each plane, tests the corner point at maximum distance along the plane normal:
    /// `vx = if normal.x > 0.0 { max.x } else { min.x }`
    /// If `distance_to_point(v) < 0.0`, the box is outside the half-space and separated.
    #[inline]
    pub fn intersects_box(&self, b: &Box3) -> bool {
        for plane in &self.planes {
            let vx = if plane.normal.x > 0.0 { b.max.x } else { b.min.x };
            let vy = if plane.normal.y > 0.0 { b.max.y } else { b.min.y };
            let vz = if plane.normal.z > 0.0 { b.max.z } else { b.min.z };
            let v = Vector3::new(vx, vy, vz);

            if plane.distance_to_point(&v) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Returns the Right plane (index 0).
    #[inline]
    pub fn right(&self) -> &Plane {
        &self.planes[0]
    }

    /// Returns the Left plane (index 1).
    #[inline]
    pub fn left(&self) -> &Plane {
        &self.planes[1]
    }

    /// Returns the Bottom plane (index 2).
    #[inline]
    pub fn bottom(&self) -> &Plane {
        &self.planes[2]
    }

    /// Returns the Top plane (index 3).
    #[inline]
    pub fn top(&self) -> &Plane {
        &self.planes[3]
    }

    /// Returns the Far plane (index 4).
    #[inline]
    pub fn far(&self) -> &Plane {
        &self.planes[4]
    }

    /// Returns the Near plane (index 5).
    #[inline]
    pub fn near(&self) -> &Plane {
        &self.planes[5]
    }
}

impl Default for Frustum {
    /// Constructs a default frustum with six default planes matching Three.js `new Frustum()`.
    #[inline]
    fn default() -> Self {
        Self {
            planes: [Plane::default(); 6],
        }
    }
}

impl fmt::Display for Frustum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frustum(R: {}, L: {}, B: {}, T: {}, Far: {}, Near: {})",
            self.planes[0],
            self.planes[1],
            self.planes[2],
            self.planes[3],
            self.planes[4],
            self.planes[5]
        )
    }
}
