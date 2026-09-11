//! 3D analytical line segment matching Three.js r186 `Line3`.

use core::fmt;
use crate::matrix4::Matrix4;
use crate::vector3::Vector3;

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
            t.clamp(0.0, 1.0)
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
