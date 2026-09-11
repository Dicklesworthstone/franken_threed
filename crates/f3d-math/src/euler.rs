//! 3D Euler rotation angles with `f64` semantics matching Three.js r186 `Euler`.
//!
//! Describes rotational transformations via extrinsic/intrinsic axis rotations
//! in specified axis sequences (`XYZ`, `YXZ`, `ZXY`, `ZYX`, `YZX`, `XZY`).

use crate::matrix4::Matrix4;
use crate::quaternion::Quaternion;
use crate::vector3::Vector3;

/// Supported Euler rotation orders matching Three.js r186.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EulerOrder {
    /// X then Y then Z (Three.js default order).
    #[default]
    XYZ,
    /// Y then X then Z.
    YXZ,
    /// Z then X then Y.
    ZXY,
    /// Z then Y then X.
    ZYX,
    /// Y then Z then X.
    YZX,
    /// X then Z then Y.
    XZY,
}

/// Clamps value between min and max matching Three.js `MathUtils.clamp`.
#[inline]
fn clamp(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// A representation of Euler rotation angles `(x, y, z)` with an explicit `order`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Euler {
    /// Angle of rotation around the X axis in radians.
    pub x: f64,
    /// Angle of rotation around the Y axis in radians.
    pub y: f64,
    /// Angle of rotation around the Z axis in radians.
    pub z: f64,
    /// Axis order in which rotations are applied.
    pub order: EulerOrder,
}

impl Default for Euler {
    #[inline]
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            order: EulerOrder::XYZ,
        }
    }
}

impl Euler {
    /// Default order constant (`EulerOrder::XYZ`).
    pub const DEFAULT_ORDER: EulerOrder = EulerOrder::XYZ;

    /// Gimbal lock singularity threshold matching Three.js r186 `Euler.js`.
    pub const GIMBAL_THRESHOLD: f64 = 0.9999999;

    /// Constructs a new `Euler` angle representation.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, order: EulerOrder) -> Self {
        Self { x, y, z, order }
    }

    /// Sets the components and order of this Euler representation.
    #[inline]
    pub fn set(&mut self, x: f64, y: f64, z: f64, order: EulerOrder) -> &mut Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self.order = order;
        self
    }

    /// Copies components from another Euler instance into this one.
    #[inline]
    pub fn copy(&mut self, euler: &Self) -> &mut Self {
        self.x = euler.x;
        self.y = euler.y;
        self.z = euler.z;
        self.order = euler.order;
        self
    }

    /// Sets Euler angles from a pure rotation matrix matching Three.js r186 `Euler.setFromRotationMatrix`.
    ///
    /// Applies strict numerical clamping between `[-1.0, 1.0]` and singular gimbal lock
    /// threshold testing against `0.9999999` across all six rotation orders.
    pub fn set_from_rotation_matrix(&mut self, m: &Matrix4, order: EulerOrder) -> &mut Self {
        let te = &m.elements;
        let m11 = te[0];
        let m12 = te[4];
        let m13 = te[8];
        let m21 = te[1];
        let m22 = te[5];
        let m23 = te[9];
        let m31 = te[2];
        let m32 = te[6];
        let m33 = te[10];

        match order {
            EulerOrder::XYZ => {
                self.y = clamp(m13, -1.0, 1.0).asin();
                if m13.abs() < Self::GIMBAL_THRESHOLD {
                    self.x = (-m23).atan2(m33);
                    self.z = (-m12).atan2(m11);
                } else {
                    self.x = m32.atan2(m22);
                    self.z = 0.0;
                }
            }
            EulerOrder::YXZ => {
                self.x = (-clamp(m23, -1.0, 1.0)).asin();
                if m23.abs() < Self::GIMBAL_THRESHOLD {
                    self.y = m13.atan2(m33);
                    self.z = m21.atan2(m22);
                } else {
                    self.y = (-m31).atan2(m11);
                    self.z = 0.0;
                }
            }
            EulerOrder::ZXY => {
                self.x = clamp(m32, -1.0, 1.0).asin();
                if m32.abs() < Self::GIMBAL_THRESHOLD {
                    self.y = (-m31).atan2(m33);
                    self.z = (-m12).atan2(m22);
                } else {
                    self.y = 0.0;
                    self.z = m21.atan2(m11);
                }
            }
            EulerOrder::ZYX => {
                self.y = (-clamp(m31, -1.0, 1.0)).asin();
                if m31.abs() < Self::GIMBAL_THRESHOLD {
                    self.x = m32.atan2(m33);
                    self.z = m21.atan2(m11);
                } else {
                    self.x = 0.0;
                    self.z = (-m12).atan2(m22);
                }
            }
            EulerOrder::YZX => {
                self.z = clamp(m21, -1.0, 1.0).asin();
                if m21.abs() < Self::GIMBAL_THRESHOLD {
                    self.x = (-m23).atan2(m22);
                    self.y = (-m31).atan2(m11);
                } else {
                    self.x = 0.0;
                    self.y = m13.atan2(m33);
                }
            }
            EulerOrder::XZY => {
                self.z = (-clamp(m12, -1.0, 1.0)).asin();
                if m12.abs() < Self::GIMBAL_THRESHOLD {
                    self.x = m32.atan2(m22);
                    self.y = m13.atan2(m11);
                } else {
                    self.x = (-m23).atan2(m33);
                    self.y = 0.0;
                }
            }
        }

        self.order = order;
        self
    }

    /// Sets the angles of this Euler instance from a normalized quaternion.
    #[inline]
    pub fn set_from_quaternion(&mut self, q: &Quaternion, order: EulerOrder) -> &mut Self {
        let mut m = Matrix4::identity();
        m.make_rotation_from_quaternion(q);
        self.set_from_rotation_matrix(&m, order)
    }

    /// Sets the angles of this Euler instance from vector components.
    #[inline]
    pub fn set_from_vector3(&mut self, v: &Vector3, order: EulerOrder) -> &mut Self {
        self.set(v.x, v.y, v.z, order)
    }

    /// Returns the Euler angles as a `Vector3`.
    #[inline]
    pub const fn to_vector3(&self) -> Vector3 {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Converts these Euler angles into an equivalent rotation `Quaternion`.
    #[inline]
    pub fn to_quaternion(&self) -> Quaternion {
        let mut q = Quaternion::identity();
        q.set_from_euler(self);
        q
    }

    /// Reorders the Euler rotation angles while preserving the overall rotation.
    pub fn reorder(&mut self, new_order: EulerOrder) -> &mut Self {
        let q = self.to_quaternion();
        self.set_from_quaternion(&q, new_order)
    }

    /// Returns whether this Euler instance has identical angles and order.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.order == other.order
    }
}
