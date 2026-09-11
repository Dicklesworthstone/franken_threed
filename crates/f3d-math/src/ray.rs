//! 3D Ray matching Three.js r186 `Ray`.

use core::fmt;
use crate::box3::Box3;
use crate::jsnum::{js_max, js_min};
use crate::matrix4::Matrix4;
use crate::plane::Plane;
use crate::sphere::Sphere;
use crate::vector3::Vector3;

/// A 3D ray emitted from an origin in a normalized direction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ray {
    /// Origin of the ray.
    pub origin: Vector3,
    /// Direction vector of the ray.
    pub direction: Vector3,
}

impl Ray {
    /// Constructs a new ray with specified origin and direction.
    #[inline]
    pub const fn new(origin: Vector3, direction: Vector3) -> Self {
        Self { origin, direction }
    }

    /// Sets origin and direction by value.
    #[inline]
    pub fn set(&mut self, origin: Vector3, direction: Vector3) -> &mut Self {
        self.origin = origin;
        self.direction = direction;
        self
    }

    /// Returns a vector located at distance `t` along this ray: `origin + direction * t`.
    ///
    /// Matches Three.js r186 `Ray.at(t, target)`.
    #[inline]
    pub fn at(&self, t: f64) -> Vector3 {
        let mut target = self.origin;
        target.add_scaled_vector(&self.direction, t);
        target
    }

    /// Adjusts the direction to point at the given vector in world space.
    ///
    /// Matches Three.js r186 `Ray.lookAt(v)`.
    #[inline]
    pub fn look_at(&mut self, v: &Vector3) -> &mut Self {
        self.direction = *v;
        self.direction.sub(&self.origin).normalize();
        self
    }

    /// Shifts the origin along the direction by distance `t`.
    ///
    /// Matches Three.js r186 `Ray.recast(t)`.
    #[inline]
    pub fn recast(&mut self, t: f64) -> &mut Self {
        self.origin = self.at(t);
        self
    }

    /// Returns the point along this ray closest to the given point.
    ///
    /// Points behind the ray origin clamp to `origin`.
    /// Matches Three.js r186 `Ray.closestPointToPoint(point, target)`.
    pub fn closest_point_to_point(&self, point: &Vector3) -> Vector3 {
        let mut diff = *point;
        diff.sub(&self.origin);

        let direction_distance = diff.dot(&self.direction);
        if direction_distance < 0.0 {
            return self.origin;
        }

        let mut target = self.origin;
        target.add_scaled_vector(&self.direction, direction_distance);
        target
    }

    /// Returns the squared distance of closest approach between this ray and `point`.
    ///
    /// Matches Three.js r186 `Ray.distanceSqToPoint(point)`.
    pub fn distance_sq_to_point(&self, point: &Vector3) -> f64 {
        let mut diff = *point;
        diff.sub(&self.origin);

        let direction_distance = diff.dot(&self.direction);
        if direction_distance < 0.0 {
            return self.origin.distance_to_squared(point);
        }

        let mut on_ray = self.origin;
        on_ray.add_scaled_vector(&self.direction, direction_distance);
        on_ray.distance_to_squared(point)
    }

    /// Returns the distance of closest approach between this ray and `point`.
    ///
    /// Matches Three.js r186 `Ray.distanceToPoint(point)`.
    #[inline]
    pub fn distance_to_point(&self, point: &Vector3) -> f64 {
        self.distance_sq_to_point(point).sqrt()
    }

    /// Returns the squared distance between this ray and the given line segment `[v0, v1]`.
    ///
    /// If `optional_point_on_ray` is provided, it receives the closest point on this ray.
    /// If `optional_point_on_segment` is provided, it receives the closest point on the segment.
    /// Matches Three.js r186 `Ray.distanceSqToSegment(v0, v1, optionalPointOnRay, optionalPointOnSegment)`.
    pub fn distance_sq_to_segment(
        &self,
        v0: &Vector3,
        v1: &Vector3,
        optional_point_on_ray: Option<&mut Vector3>,
        optional_point_on_segment: Option<&mut Vector3>,
    ) -> f64 {
        let mut seg_center = *v0;
        seg_center.add(v1).multiply_scalar(0.5);

        let mut seg_dir = *v1;
        seg_dir.sub(v0).normalize();

        let mut diff = self.origin;
        diff.sub(&seg_center);

        let seg_extent = v0.distance_to(v1) * 0.5;
        let a01 = -self.direction.dot(&seg_dir);
        let b0 = diff.dot(&self.direction);
        let b1 = -diff.dot(&seg_dir);
        let c = diff.length_sq();
        let det = (1.0 - a01 * a01).abs();

        let mut s0: f64;
        let mut s1: f64;
        let sqr_dist: f64;

        if det > 0.0 {
            // The ray and segment are not parallel.
            s0 = a01 * b1 - b0;
            s1 = a01 * b0 - b1;
            let ext_det = seg_extent * det;

            if s0 >= 0.0 {
                if s1 >= -ext_det {
                    if s1 <= ext_det {
                        // region 0: Minimum at interior points of ray and segment.
                        let inv_det = 1.0 / det;
                        s0 *= inv_det;
                        s1 *= inv_det;
                        sqr_dist = s0 * (s0 + a01 * s1 + 2.0 * b0)
                            + s1 * (a01 * s0 + s1 + 2.0 * b1)
                            + c;
                    } else {
                        // region 1
                        s1 = seg_extent;
                        s0 = js_max(0.0, -(a01 * s1 + b0));
                        sqr_dist = -s0 * s0 + s1 * (s1 + 2.0 * b1) + c;
                    }
                } else {
                    // region 5
                    s1 = -seg_extent;
                    s0 = js_max(0.0, -(a01 * s1 + b0));
                    sqr_dist = -s0 * s0 + s1 * (s1 + 2.0 * b1) + c;
                }
            } else if s1 <= -ext_det {
                // region 4
                s0 = js_max(0.0, -(-a01 * seg_extent + b0));
                s1 = if s0 > 0.0 {
                    -seg_extent
                } else {
                    js_min(js_max(-seg_extent, -b1), seg_extent)
                };
                sqr_dist = -s0 * s0 + s1 * (s1 + 2.0 * b1) + c;
            } else if s1 <= ext_det {
                // region 3
                s0 = 0.0;
                s1 = js_min(js_max(-seg_extent, -b1), seg_extent);
                sqr_dist = s1 * (s1 + 2.0 * b1) + c;
            } else {
                // region 2
                s0 = js_max(0.0, -(a01 * seg_extent + b0));
                s1 = if s0 > 0.0 {
                    seg_extent
                } else {
                    js_min(js_max(-seg_extent, -b1), seg_extent)
                };
                sqr_dist = -s0 * s0 + s1 * (s1 + 2.0 * b1) + c;
            }
        } else {
            // Ray and segment are parallel.
            s1 = if a01 > 0.0 { -seg_extent } else { seg_extent };
            s0 = js_max(0.0, -(a01 * s1 + b0));
            sqr_dist = -s0 * s0 + s1 * (s1 + 2.0 * b1) + c;
        }

        if let Some(point_on_ray) = optional_point_on_ray {
            *point_on_ray = self.origin;
            point_on_ray.add_scaled_vector(&self.direction, s0);
        }

        if let Some(point_on_segment) = optional_point_on_segment {
            *point_on_segment = seg_center;
            point_on_segment.add_scaled_vector(&seg_dir, s1);
        }

        sqr_dist
    }

    /// Intersects this ray with a sphere, returning the intersection point.
    ///
    /// - Returns `None` if the sphere is empty (`radius < 0.0`), if the ray misses, or if the sphere is behind the ray.
    /// - If the ray origin is inside the sphere, returns the exit point along the ray.
    /// Matches Three.js r186 `Ray.intersectSphere(sphere, target)`.
    pub fn intersect_sphere(&self, sphere: &Sphere) -> Option<Vector3> {
        if sphere.is_empty() {
            return None;
        }

        let mut v = sphere.center;
        v.sub(&self.origin);

        let tca = v.dot(&self.direction);
        let d2 = v.dot(&v) - tca * tca;
        let radius2 = sphere.radius * sphere.radius;

        if d2 > radius2 {
            return None;
        }

        let thc = (radius2 - d2).sqrt();
        let t0 = tca - thc;
        let t1 = tca + thc;

        if t1 < 0.0 {
            return None;
        }

        if t0 < 0.0 {
            return Some(self.at(t1));
        }

        Some(self.at(t0))
    }

    /// Returns `true` if this ray intersects with the given sphere.
    ///
    /// Matches Three.js r186 `Ray.intersectsSphere(sphere)`.
    #[inline]
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        if sphere.is_empty() {
            return false;
        }
        self.distance_sq_to_point(&sphere.center) <= sphere.radius * sphere.radius
    }

    /// Computes distance from ray origin to plane. Returns `None` if parallel or behind ray.
    ///
    /// Matches Three.js r186 `Ray.distanceToPlane(plane)`.
    pub fn distance_to_plane(&self, plane: &Plane) -> Option<f64> {
        let denominator = plane.normal.dot(&self.direction);
        if denominator == 0.0 {
            if plane.distance_to_point(&self.origin) == 0.0 {
                return Some(0.0);
            }
            return None;
        }

        let t = -(self.origin.dot(&plane.normal) + plane.constant) / denominator;
        if t >= 0.0 {
            Some(t)
        } else {
            None
        }
    }

    /// Intersects this ray with a plane, returning the intersection point.
    ///
    /// Matches Three.js r186 `Ray.intersectPlane(plane, target)`.
    #[inline]
    pub fn intersect_plane(&self, plane: &Plane) -> Option<Vector3> {
        let t = self.distance_to_plane(plane)?;
        Some(self.at(t))
    }

    /// Returns `true` if this ray intersects with the given plane.
    ///
    /// Matches Three.js r186 `Ray.intersectsPlane(plane)`.
    pub fn intersects_plane(&self, plane: &Plane) -> bool {
        let dist_to_point = plane.distance_to_point(&self.origin);
        if dist_to_point == 0.0 {
            return true;
        }
        let denominator = plane.normal.dot(&self.direction);
        denominator * dist_to_point < 0.0
    }

    /// Intersects this ray with a bounding box using the slab method.
    ///
    /// Preserves exact Three.js r186 NaN handling and edge cases.
    /// Matches Three.js r186 `Ray.intersectBox(box, target)`.
    pub fn intersect_box(&self, box3: &Box3) -> Option<Vector3> {
        if box3.is_empty() {
            return None;
        }

        let invdirx = 1.0 / self.direction.x;
        let invdiry = 1.0 / self.direction.y;
        let invdirz = 1.0 / self.direction.z;

        let origin = self.origin;

        let (mut tmin, mut tmax) = if invdirx >= 0.0 {
            ((box3.min.x - origin.x) * invdirx, (box3.max.x - origin.x) * invdirx)
        } else {
            ((box3.max.x - origin.x) * invdirx, (box3.min.x - origin.x) * invdirx)
        };

        let (tymin, tymax) = if invdiry >= 0.0 {
            ((box3.min.y - origin.y) * invdiry, (box3.max.y - origin.y) * invdiry)
        } else {
            ((box3.max.y - origin.y) * invdiry, (box3.min.y - origin.y) * invdiry)
        };

        if (tmin > tymax) || (tymin > tmax) {
            return None;
        }

        if tymin > tmin || tmin.is_nan() {
            tmin = tymin;
        }

        if tymax < tmax || tmax.is_nan() {
            tmax = tymax;
        }

        let (tzmin, tzmax) = if invdirz >= 0.0 {
            ((box3.min.z - origin.z) * invdirz, (box3.max.z - origin.z) * invdirz)
        } else {
            ((box3.max.z - origin.z) * invdirz, (box3.min.z - origin.z) * invdirz)
        };

        if (tmin > tzmax) || (tzmin > tmax) {
            return None;
        }

        if tzmin > tmin || tmin.is_nan() {
            tmin = tzmin;
        }

        if tzmax < tmax || tmax.is_nan() {
            tmax = tzmax;
        }

        if tmax < 0.0 {
            return None;
        }

        Some(self.at(if tmin >= 0.0 { tmin } else { tmax }))
    }

    /// Returns `true` if this ray intersects with the given bounding box.
    ///
    /// Matches Three.js r186 `Ray.intersectsBox(box)`.
    #[inline]
    pub fn intersects_box(&self, box3: &Box3) -> bool {
        self.intersect_box(box3).is_some()
    }

    /// Intersects this ray with a triangle.
    ///
    /// Uses the watertight ray/triangle intersection algorithm (Woop, Benthin, Wald 2013).
    /// Respects `backface_culling`.
    /// Matches Three.js r186 `Ray.intersectTriangle(a, b, c, backfaceCulling, target)`.
    pub fn intersect_triangle(
        &self,
        a: &Vector3,
        b: &Vector3,
        c: &Vector3,
        backface_culling: bool,
    ) -> Option<Vector3> {
        let origin = self.origin;
        let direction = self.direction;

        let dx = direction.x;
        let dy = direction.y;
        let dz = direction.z;

        let aox = a.x - origin.x;
        let aoy = a.y - origin.y;
        let aoz = a.z - origin.z;

        let box_x = b.x - origin.x;
        let boy = b.y - origin.y;
        let boz = b.z - origin.z;

        let cox = c.x - origin.x;
        let coy = c.y - origin.y;
        let coz = c.z - origin.z;

        let adx = dx.abs();
        let ady = dy.abs();
        let adz = dz.abs();

        let (dkx, dky, dkz);
        let (akx, aky, akz);
        let (bkx, bky, bkz);
        let (ckx, cky, ckz);

        if adx >= ady && adx >= adz {
            dkz = dx; akz = aox; bkz = box_x; ckz = cox;
            if dx >= 0.0 {
                dkx = dy; dky = dz;
                akx = aoy; aky = aoz; bkx = boy; bky = boz; ckx = coy; cky = coz;
            } else {
                dkx = dz; dky = dy;
                akx = aoz; aky = aoy; bkx = boz; bky = boy; ckx = coz; cky = coy;
            }
        } else if ady >= adz {
            dkz = dy; akz = aoy; bkz = boy; ckz = coy;
            if dy >= 0.0 {
                dkx = dz; dky = dx;
                akx = aoz; aky = aox; bkx = boz; bky = box_x; ckx = coz; cky = cox;
            } else {
                dkx = dx; dky = dz;
                akx = aox; aky = aoz; bkx = box_x; bky = boz; ckx = cox; cky = coz;
            }
        } else {
            dkz = dz; akz = aoz; bkz = boz; ckz = coz;
            if dz >= 0.0 {
                dkx = dx; dky = dy;
                akx = aox; aky = aoy; bkx = box_x; bky = boy; ckx = cox; cky = coy;
            } else {
                dkx = dy; dky = dx;
                akx = aoy; aky = aox; bkx = boy; bky = box_x; ckx = coy; cky = cox;
            }
        }

        if dkz == 0.0 {
            return None;
        }

        let sx = dkx / dkz;
        let sy = dky / dkz;
        let sz = 1.0 / dkz;

        let ax = akx - sx * akz;
        let ay = aky - sy * akz;
        let bx = bkx - sx * bkz;
        let by = bky - sy * bkz;
        let cx = ckx - sx * ckz;
        let cy = cky - sy * ckz;

        let u = cx * by - cy * bx;
        let v = ax * cy - ay * cx;
        let w = bx * ay - by * ax;

        if backface_culling {
            if u < 0.0 || v < 0.0 || w < 0.0 {
                return None;
            }
        } else {
            if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
                return None;
            }
        }

        let det = u + v + w;
        if det == 0.0 {
            return None;
        }

        let t_scaled = sz * (u * akz + v * bkz + w * ckz);
        if if det > 0.0 { t_scaled < 0.0 } else { t_scaled > 0.0 } {
            return None;
        }

        Some(self.at(t_scaled / det))
    }

    /// Transforms this ray with a 4x4 transformation matrix.
    ///
    /// Evaluates `origin.apply_matrix4(m)` and `direction.transform_direction(m)`.
    /// Matches Three.js r186 `Ray.applyMatrix4(matrix4)`.
    pub fn apply_matrix4(&mut self, matrix: &Matrix4) -> &mut Self {
        self.origin.apply_matrix4(matrix);
        self.direction.transform_direction(matrix);
        self
    }

    /// Returns `true` if this ray equals another component-wise.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.origin == other.origin && self.direction == other.direction
    }
}

impl Default for Ray {
    #[inline]
    fn default() -> Self {
        Self {
            origin: Vector3::zero(),
            direction: Vector3::new(0.0, 0.0, -1.0),
        }
    }
}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ray(origin: {}, direction: {})", self.origin, self.direction)
    }
}
