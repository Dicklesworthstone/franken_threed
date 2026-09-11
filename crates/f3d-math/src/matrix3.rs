//! 3x3 matrix primitive with `f64` public semantics matching Three.js r186 `Matrix3`.
//!
//! # Storage and Ordering
//! Values are stored internally in column-major order in `elements: [f64; 9]`:
//! ```text
//! Column 0: elements[0] = n11, elements[1] = n21, elements[2] = n31
//! Column 1: elements[3] = n12, elements[4] = n22, elements[5] = n32
//! Column 2: elements[6] = n13, elements[7] = n23, elements[8] = n33
//! ```
//! The constructor and [`Matrix3::set`] accept arguments in row-major order `(n11, n12, n13, n21, ...)`
//! matching Three.js conventions.

use core::fmt;
use crate::matrix4::Matrix4;

/// A 3x3 matrix represented in column-major order by double-precision `f64` elements.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Matrix3 {
    /// Column-major matrix elements.
    pub elements: [f64; 9],
}

impl Default for Matrix3 {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl fmt::Debug for Matrix3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let te = &self.elements;
        f.debug_struct("Matrix3")
            .field("row0", &[te[0], te[3], te[6]])
            .field("row1", &[te[1], te[4], te[7]])
            .field("row2", &[te[2], te[5], te[8]])
            .finish()
    }
}

impl Matrix3 {
    /// Constructs an identity 3x3 matrix.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            elements: [
                1.0, 0.0, 0.0, // Column 0
                0.0, 1.0, 0.0, // Column 1
                0.0, 0.0, 1.0, // Column 2
            ],
        }
    }

    /// Constructs an all-zero 3x3 matrix.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            elements: [0.0; 9],
        }
    }

    /// Constructs a matrix from direct column-major elements `[f64; 9]`.
    #[inline]
    pub const fn from_elements(elements: [f64; 9]) -> Self {
        Self { elements }
    }

    /// Constructs a new 3x3 matrix with arguments in row-major order matching Three.js `Matrix3.set`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        n11: f64, n12: f64, n13: f64,
        n21: f64, n22: f64, n23: f64,
        n31: f64, n32: f64, n33: f64,
    ) -> Self {
        Self {
            elements: [
                n11, n21, n31, // Column 0
                n12, n22, n32, // Column 1
                n13, n23, n33, // Column 2
            ],
        }
    }

    /// Sets the elements of this matrix with arguments in row-major order.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &mut self,
        n11: f64, n12: f64, n13: f64,
        n21: f64, n22: f64, n23: f64,
        n31: f64, n32: f64, n33: f64,
    ) -> &mut Self {
        let te = &mut self.elements;
        te[0] = n11; te[1] = n21; te[2] = n31;
        te[3] = n12; te[4] = n22; te[5] = n32;
        te[6] = n13; te[7] = n23; te[8] = n33;
        self
    }

    /// Copies elements from another matrix into this one.
    #[inline]
    pub fn copy(&mut self, m: &Self) -> &mut Self {
        self.elements = m.elements;
        self
    }

    /// Extracts the upper-left 3x3 portion of a 4x4 matrix matching Three.js `Matrix3.setFromMatrix4`.
    pub fn set_from_matrix4(&mut self, m: &Matrix4) -> &mut Self {
        let me = &m.elements;
        let te = &mut self.elements;

        te[0] = me[0];
        te[1] = me[1];
        te[2] = me[2];

        te[3] = me[4];
        te[4] = me[5];
        te[5] = me[6];

        te[6] = me[8];
        te[7] = me[9];
        te[8] = me[10];

        self
    }

    /// Computes the determinant of this 3x3 matrix.
    #[inline]
    pub fn determinant(&self) -> f64 {
        let te = &self.elements;
        let a = te[0]; let b = te[1]; let c = te[2];
        let d = te[3]; let e = te[4]; let f = te[5];
        let g = te[6]; let h = te[7]; let i = te[8];

        a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g
    }

    /// Inverts this matrix in place matching Three.js r186 `Matrix3.invert`.
    ///
    /// Invariant: If the determinant is zero (`det == 0.0`), sets this matrix to an all-zero
    /// matrix `[0.0; 9]` and returns without panicking.
    pub fn invert(&mut self) -> &mut Self {
        let te = self.elements;
        let n11 = te[0]; let n21 = te[1]; let n31 = te[2];
        let n12 = te[3]; let n22 = te[4]; let n32 = te[5];
        let n13 = te[6]; let n23 = te[7]; let n33 = te[8];

        let t11 = n33 * n22 - n32 * n23;
        let t12 = n32 * n13 - n33 * n12;
        let t13 = n23 * n12 - n22 * n13;

        let det = n11 * t11 + n21 * t12 + n31 * t13;

        if det == 0.0 {
            self.elements = [0.0; 9];
            return self;
        }

        let det_inv = 1.0 / det;

        self.elements[0] = t11 * det_inv;
        self.elements[1] = (n31 * n23 - n33 * n21) * det_inv;
        self.elements[2] = (n32 * n21 - n31 * n22) * det_inv;

        self.elements[3] = t12 * det_inv;
        self.elements[4] = (n33 * n11 - n31 * n13) * det_inv;
        self.elements[5] = (n31 * n12 - n32 * n11) * det_inv;

        self.elements[6] = t13 * det_inv;
        self.elements[7] = (n21 * n13 - n23 * n11) * det_inv;
        self.elements[8] = (n22 * n11 - n21 * n12) * det_inv;

        self
    }

    /// Safe fallible inverse returning `None` if the matrix is singular (`determinant == 0.0`).
    pub fn try_invert(&self) -> Option<Self> {
        let mut copy = *self;
        copy.invert();
        if copy.elements == [0.0; 9] && self.determinant() == 0.0 {
            None
        } else {
            Some(copy)
        }
    }

    /// Transposes this matrix in place.
    pub fn transpose(&mut self) -> &mut Self {
        self.elements.swap(1, 3);
        self.elements.swap(2, 6);
        self.elements.swap(5, 7);
        self
    }

    /// Computes the normal matrix as the inverse transpose of the upper-left 3x3 of a 4x4 matrix.
    ///
    /// Matches Three.js r186 `Matrix3.getNormalMatrix(matrix4)`:
    /// `this.setFromMatrix4(matrix4).invert().transpose()`.
    pub fn get_normal_matrix(&mut self, matrix4: &Matrix4) -> &mut Self {
        self.set_from_matrix4(matrix4);
        self.invert();
        self.transpose();
        self
    }

    /// Multiplies matrix `a` by matrix `b` and stores the result in this instance (`self = a * b`).
    pub fn multiply_matrices(&mut self, a: &Self, b: &Self) -> &mut Self {
        let ae = &a.elements;
        let be = &b.elements;

        let a11 = ae[0]; let a12 = ae[3]; let a13 = ae[6];
        let a21 = ae[1]; let a22 = ae[4]; let a23 = ae[7];
        let a31 = ae[2]; let a32 = ae[5]; let a33 = ae[8];

        let b11 = be[0]; let b12 = be[3]; let b13 = be[6];
        let b21 = be[1]; let b22 = be[4]; let b23 = be[7];
        let b31 = be[2]; let b32 = be[5]; let b33 = be[8];

        let te = &mut self.elements;
        te[0] = a11 * b11 + a12 * b21 + a13 * b31;
        te[3] = a11 * b12 + a12 * b22 + a13 * b32;
        te[6] = a11 * b13 + a12 * b23 + a13 * b33;

        te[1] = a21 * b11 + a22 * b21 + a23 * b31;
        te[4] = a21 * b12 + a22 * b22 + a23 * b32;
        te[7] = a21 * b13 + a22 * b23 + a23 * b33;

        te[2] = a31 * b11 + a32 * b21 + a33 * b31;
        te[5] = a31 * b12 + a32 * b22 + a33 * b32;
        te[8] = a31 * b13 + a32 * b23 + a33 * b33;

        self
    }

    /// Post-multiplies this matrix by `m` (`self = self * m`).
    #[inline]
    pub fn multiply(&mut self, m: &Self) -> &mut Self {
        let a = *self;
        self.multiply_matrices(&a, m)
    }

    /// Pre-multiplies this matrix by `m` (`self = m * self`).
    #[inline]
    pub fn premultiply(&mut self, m: &Self) -> &mut Self {
        let b = *self;
        self.multiply_matrices(m, &b)
    }

    /// Multiplies every element by a scalar `s`.
    #[inline]
    pub fn multiply_scalar(&mut self, s: f64) -> &mut Self {
        for el in &mut self.elements {
            *el *= s;
        }
        self
    }

    /// Returns `true` if all elements match `other` exactly.
    #[inline]
    pub fn equals(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}
