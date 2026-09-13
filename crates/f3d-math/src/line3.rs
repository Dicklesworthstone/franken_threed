//! 3D analytical line segment matching Three.js r186 `Line3`.

use core::fmt;
use crate::matrix4::Matrix4;
use crate::vector3::Vector3;

#[inline]
fn js_clamp(value: f64, min: f64, max: f64) -> f64 {
    crate::jsnum::js_max(min, crate::jsnum::js_min(max, value))
}

/// An analytical line segment in 3D space represented by a start and end point.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Line3 {
    /// Start of the line segment.
    pub start: Vector3,
    /// End of the line segment.
    pub end: Vector3,
}

impl Line3 {
    /// Constructs a new line segment with given start and end points.
    #[inline]
    pub const fn new(start: Vector3, end: Vector3) -> Self {
        Self { start, end }
    }

    /// Sets the start and end points.
    #[inline]
    pub fn set(&mut self, start: Vector3, end: Vector3) -> &mut Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Copies the values of the given line segment to this instance.
    #[inline]
    pub fn copy(&mut self, line: &Line3) -> &mut Self {
        self.start = line.start;
        self.end = line.end;
        self
    }

    /// Returns the center point of the line segment.
    #[inline]
    pub fn get_center(&self) -> Vector3 {
        Vector3::new(
            (self.start.x + self.end.x) * 0.5,
            (self.start.y + self.end.y) * 0.5,
            (self.start.z + self.end.z) * 0.5,
        )
    }

    /// Returns the delta vector `(end - start)`.
    #[inline]
    pub fn delta(&self) -> Vector3 {
        Vector3::new(
            self.end.x - self.start.x,
            self.end.y - self.start.y,
            self.end.z - self.start.z,
        )
    }

    /// Returns the squared Euclidean distance between start and end.
    #[inline]
    pub fn distance_sq(&self) -> f64 {
        self.start.distance_to_squared(&self.end)
    }

    /// Returns the Euclidean distance between start and end.
    #[inline]
    pub fn distance(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    /// Returns a vector at position `t` along the line segment: `start + delta * t`.
    #[inline]
    pub fn at(&self, t: f64) -> Vector3 {
        let mut d = self.delta();
        d.multiply_scalar(t);
        d.add(&self.start);
        d
    }

    /// Returns the projection parameter `t` for the closest point on the line to `point`.
    pub fn closest_point_to_point_parameter(&self, point: &Vector3, clamp_to_line: bool) -> f64 {
        let mut start_p = *point;
        start_p.sub(&self.start);
        let start_end = self.delta();
        let start_end_sq = start_end.dot(&start_end);
        if start_end_sq == 0.0 {
            return 0.0;
        }
        let t = start_end.dot(&start_p) / start_end_sq;
        if clamp_to_line {
            js_clamp(t, 0.0, 1.0)
        } else {
            t
        }
    }

    /// Returns the closest point on the line to the given point.
    #[inline]
    pub fn closest_point_to_point(&self, point: &Vector3, clamp_to_line: bool) -> Vector3 {
        let t = self.closest_point_to_point_parameter(point, clamp_to_line);
        self.at(t)
    }

    /// Returns the closest squared distance between this line segment and another.
    ///
    /// Optionally writes the closest points on this line segment and the other line segment
    /// into `c1` and `c2` respectively.
    ///
    /// Matches Three.js r186 `Line3.distanceSqToLine3(line, c1, c2)`.
    pub fn distance_sq_to_line3(
        &self,
        line: &Line3,
        c1: Option<&mut Vector3>,
        c2: Option<&mut Vector3>,
    ) -> f64 {
        const EPSILON: f64 = 1e-8 * 1e-8;
        let mut s;
        let mut t;

        let p1 = self.start;
        let p2 = line.start;
        let q1 = self.end;
        let q2 = line.end;

        let mut d1 = q1;
        d1.sub(&p1);
        let mut d2 = q2;
        d2.sub(&p2);
        let mut r = p1;
        r.sub(&p2);

        let a = d1.dot(&d1);
        let e = d2.dot(&d2);
        let f = d2.dot(&r);

        // Check if either or both segments degenerate into points
        if a <= EPSILON && e <= EPSILON {
            // Both segments degenerate into points
            let mut diff = p1;
            diff.sub(&p2);
            let dist_sq = diff.dot(&diff);
            if let Some(c1_out) = c1 {
                *c1_out = diff;
            }
            if let Some(c2_out) = c2 {
                *c2_out = p2;
            }
            return dist_sq;
        }

        if a <= EPSILON {
            // First segment degenerates into a point
            s = 0.0;
            t = f / e;
            t = js_clamp(t, 0.0, 1.0);
        } else {
            let c = d1.dot(&r);
            if e <= EPSILON {
                // Second segment degenerates into a point
                t = 0.0;
                s = js_clamp(-c / a, 0.0, 1.0);
            } else {
                // The general nondegenerate case starts here
                let b = d1.dot(&d2);
                let denom = a * e - b * b;

                if denom != 0.0 {
                    s = js_clamp((b * f - c * e) / denom, 0.0, 1.0);
                } else {
                    s = 0.0;
                }

                t = (b * s + f) / e;

                if t < 0.0 {
                    t = 0.0;
                    s = js_clamp(-c / a, 0.0, 1.0);
                } else if t > 1.0 {
                    t = 1.0;
                    s = js_clamp((b - c) / a, 0.0, 1.0);
                }
            }
        }

        let mut pt1 = p1;
        pt1.add_scaled_vector(&d1, s);
        let mut pt2 = p2;
        pt2.add_scaled_vector(&d2, t);

        let dist_sq = pt1.distance_to_squared(&pt2);

        if let Some(c1_out) = c1 {
            *c1_out = pt1;
        }
        if let Some(c2_out) = c2 {
            *c2_out = pt2;
        }

        dist_sq
    }

    /// Transforms this line segment by a 4x4 matrix.
    #[inline]
    pub fn apply_matrix4(&mut self, matrix: &Matrix4) -> &mut Self {
        self.start.apply_matrix4(matrix);
        self.end.apply_matrix4(matrix);
        self
    }

    /// Returns `true` if this line segment equals another component-wise.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

impl Default for Line3 {
    #[inline]
    fn default() -> Self {
        Self {
            start: Vector3::zero(),
            end: Vector3::zero(),
        }
    }
}

impl fmt::Display for Line3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Line3(start: {}, end: {})", self.start, self.end)
    }
}
