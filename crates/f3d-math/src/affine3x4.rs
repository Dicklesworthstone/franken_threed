//! Packed 3x4 affine transformation matrix with `f64` public semantics (§7.5, roa.1).
//!
//! Represents an affine 3D transformation as three rows of 4 `f64` components,
//! where row 3 is implicitly `[0.0, 0.0, 0.0, 1.0]`.
//!
//! Layout matching `f3d-core::layout::AffineRows`:
//! ```text
//! Row 0: [e[0], e[4], e[8],  e[12]]  (X-axis basis + translation X)
//! Row 1: [e[1], e[5], e[9],  e[13]]  (Y-axis basis + translation Y)
//! Row 2: [e[2], e[6], e[10], e[14]]  (Z-axis basis + translation Z)
//! ```

use f3d_core::layout::{AffineRows, LayoutError};
use crate::matrix4::Matrix4;
use crate::vector3::Vector3;

/// A packed 3x4 affine matrix represented by three 4-element `f64` rows.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Affine3x4 {
    /// Row 0: X-axis linear basis in xyz, translation X in w.
    pub r0: [f64; 4],
    /// Row 1: Y-axis linear basis in xyz, translation Y in w.
    pub r1: [f64; 4],
    /// Row 2: Z-axis linear basis in xyz, translation Z in w.
    pub r2: [f64; 4],
}

impl Affine3x4 {
    /// Standard 3D identity affine transform.
    pub const IDENTITY: Self = Self {
        r0: [1.0, 0.0, 0.0, 0.0],
        r1: [0.0, 1.0, 0.0, 0.0],
        r2: [0.0, 0.0, 1.0, 0.0],
    };

    /// All-zero affine matrix.
    pub const ZERO: Self = Self {
        r0: [0.0; 4],
        r1: [0.0; 4],
        r2: [0.0; 4],
    };

    /// Constructs an `Affine3x4` from explicit row vectors.
    #[inline]
    pub const fn new(r0: [f64; 4], r1: [f64; 4], r2: [f64; 4]) -> Self {
        Self { r0, r1, r2 }
    }

    /// Returns the standard identity affine transform.
    #[inline]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Returns an all-zero affine matrix.
    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    /// Checked conversion from a column-major `Matrix4`.
    ///
    /// Returns `Err(LayoutError::NonAffineMatrix)` if row 3 is not bitwise `[+0.0, +0.0, +0.0, 1.0]`.
    #[inline]
    pub fn from_matrix4(m: &Matrix4) -> Result<Self, LayoutError> {
        Self::from_column_major(&m.elements)
    }

    /// Unchecked conversion from a column-major `Matrix4` omitting row 3 verification.
    #[inline]
    pub const fn from_matrix4_unchecked(m: &Matrix4) -> Self {
        Self::from_column_major_unchecked(&m.elements)
    }

    /// Checked conversion from a 16-element column-major array.
    ///
    /// Invariant: Exact structural check. Row 3 must be exactly bitwise `[+0.0, +0.0, +0.0, 1.0]`.
    /// This is an intentional packing refusal (with `Matrix4` fallback): because `to_column_major`
    /// expands row 3 with canonical `+0.0`, any `-0.0` signed zero in perspective entries would
    /// lose bitwise exactness upon expansion. Any non-zero entry, subnormal (e.g. `1e-100`),
    /// `-0.0` signed zero in perspective entries, or `e[15] != 1.0` is rejected with
    /// `Err(LayoutError::NonAffineMatrix)`.
    pub fn from_column_major(e: &[f64; 16]) -> Result<Self, LayoutError> {
        // Enforce exact bitwise +0.0 in perspective row entries (rejecting -0.0) and exactly 1.0 in e[15].
        if e[3].to_bits() != 0 || e[7].to_bits() != 0 || e[11].to_bits() != 0 || e[15] != 1.0 {
            return Err(LayoutError::NonAffineMatrix);
        }
        Ok(Self::from_column_major_unchecked(e))
    }

    /// Unchecked conversion from a 16-element column-major array.
    #[inline]
    pub const fn from_column_major_unchecked(e: &[f64; 16]) -> Self {
        Self {
            r0: [e[0], e[4], e[8], e[12]],
            r1: [e[1], e[5], e[9], e[13]],
            r2: [e[2], e[6], e[10], e[14]],
        }
    }

    /// Expands this 3x4 affine matrix into a full 16-element column-major array with implicit row 3 `[0, 0, 0, 1]`.
    #[inline]
    pub const fn to_column_major(&self) -> [f64; 16] {
        [
            self.r0[0], self.r1[0], self.r2[0], 0.0,
            self.r0[1], self.r1[1], self.r2[1], 0.0,
            self.r0[2], self.r1[2], self.r2[2], 0.0,
            self.r0[3], self.r1[3], self.r2[3], 1.0,
        ]
    }

    /// Converts this 3x4 affine matrix into a full `Matrix4`.
    #[inline]
    pub fn to_matrix4(&self) -> Matrix4 {
        Matrix4::from_elements(self.to_column_major())
    }

    /// Narrows this `f64` affine matrix to a 48-byte GPU `AffineRows` record at the GPU boundary.
    #[inline]
    pub const fn to_affine_rows(&self) -> AffineRows {
        AffineRows {
            r0: [self.r0[0] as f32, self.r0[1] as f32, self.r0[2] as f32, self.r0[3] as f32],
            r1: [self.r1[0] as f32, self.r1[1] as f32, self.r1[2] as f32, self.r1[3] as f32],
            r2: [self.r2[0] as f32, self.r2[1] as f32, self.r2[2] as f32, self.r2[3] as f32],
        }
    }

    /// Reconstructs an `f64` `Affine3x4` from a 48-byte GPU `AffineRows` record.
    #[inline]
    pub const fn from_affine_rows(rows: &AffineRows) -> Self {
        Self {
            r0: [rows.r0[0] as f64, rows.r0[1] as f64, rows.r0[2] as f64, rows.r0[3] as f64],
            r1: [rows.r1[0] as f64, rows.r1[1] as f64, rows.r1[2] as f64, rows.r1[3] as f64],
            r2: [rows.r2[0] as f64, rows.r2[1] as f64, rows.r2[2] as f64, rows.r2[3] as f64],
        }
    }

    /// Multiplies affine transform `a` by `b` (`self = a * b`), preserving Three.js scalar operation order.
    ///
    /// # Representation & Checked Composition Contract
    /// In Three.js `Matrix4.multiplyMatrices`, row 3 of the product $C = A \times B$ is evaluated as:
    /// - $c_{30} = a_{30} b_{00} + a_{31} b_{10} + a_{32} b_{20} + a_{33} b_{30} = 0.0 \cdot b_{00} + 0.0 \cdot b_{10} + 0.0 \cdot b_{20} + 1.0 \cdot 0.0$
    /// - $c_{31} = a_{30} b_{01} + a_{31} b_{11} + a_{32} b_{21} + a_{33} b_{31} = 0.0 \cdot b_{01} + 0.0 \cdot b_{11} + 0.0 \cdot b_{21} + 1.0 \cdot 0.0$
    /// - $c_{32} = a_{30} b_{02} + a_{31} b_{12} + a_{32} b_{22} + a_{33} b_{32} = 0.0 \cdot b_{02} + 0.0 \cdot b_{12} + 0.0 \cdot b_{22} + 1.0 \cdot 0.0$
    /// - $c_{33} = a_{30} b_{03} + a_{31} b_{13} + a_{32} b_{23} + a_{33} b_{33} = 0.0 \cdot b_{03} + 0.0 \cdot b_{13} + 0.0 \cdot b_{23} + 1.0 \cdot 1.0$
    ///
    /// When the right-hand matrix `b` contains non-finite entries (`NaN` or `Infinity`), IEEE 754
    /// arithmetic evaluates $0.0 \times \text{nonfinite} = \text{NaN}$, making row 3 of the full
    /// 4x4 matrix product contain `NaN`. In that situation, the full product is no longer affine
    /// and cannot be truthfully represented by the implicit row `[+0.0, +0.0, +0.0, 1.0]`.
    ///
    /// This method checks the actual full product bottom row before mutating `self`. If row 3 is
    /// unrepresentable, it returns `Err(LayoutError::NonAffineMatrix)` leaving `self` completely
    /// unmutated. Callers requiring non-affine product representation must use `Matrix4`.
    pub fn multiply_affines(&mut self, a: &Self, b: &Self) -> Result<&mut Self, LayoutError> {
        // Evaluate actual full product row 3 under Three.js Matrix4.multiplyMatrices arithmetic:
        let c30 = 0.0 * b.r0[0] + 0.0 * b.r1[0] + 0.0 * b.r2[0] + 1.0 * 0.0;
        let c31 = 0.0 * b.r0[1] + 0.0 * b.r1[1] + 0.0 * b.r2[1] + 1.0 * 0.0;
        let c32 = 0.0 * b.r0[2] + 0.0 * b.r1[2] + 0.0 * b.r2[2] + 1.0 * 0.0;
        let c33 = 0.0 * b.r0[3] + 0.0 * b.r1[3] + 0.0 * b.r2[3] + 1.0 * 1.0;

        // Verify that the product's bottom row is representable as canonical [+0.0, +0.0, +0.0, 1.0].
        if c30.to_bits() != 0 || c31.to_bits() != 0 || c32.to_bits() != 0 || c33 != 1.0 {
            return Err(LayoutError::NonAffineMatrix);
        }

        let a00 = a.r0[0]; let a01 = a.r0[1]; let a02 = a.r0[2]; let a03 = a.r0[3];
        let a10 = a.r1[0]; let a11 = a.r1[1]; let a12 = a.r1[2]; let a13 = a.r1[3];
        let a20 = a.r2[0]; let a21 = a.r2[1]; let a22 = a.r2[2]; let a23 = a.r2[3];

        let b00 = b.r0[0]; let b01 = b.r0[1]; let b02 = b.r0[2]; let b03 = b.r0[3];
        let b10 = b.r1[0]; let b11 = b.r1[1]; let b12 = b.r1[2]; let b13 = b.r1[3];
        let b20 = b.r2[0]; let b21 = b.r2[1]; let b22 = b.r2[2]; let b23 = b.r2[3];

        // Compute into locals first: receiver is only mutated after all computations succeed.
        let r0 = [
            a00 * b00 + a01 * b10 + a02 * b20 + a03 * 0.0,
            a00 * b01 + a01 * b11 + a02 * b21 + a03 * 0.0,
            a00 * b02 + a01 * b12 + a02 * b22 + a03 * 0.0,
            a00 * b03 + a01 * b13 + a02 * b23 + a03 * 1.0,
        ];
        let r1 = [
            a10 * b00 + a11 * b10 + a12 * b20 + a13 * 0.0,
            a10 * b01 + a11 * b11 + a12 * b21 + a13 * 0.0,
            a10 * b02 + a11 * b12 + a12 * b22 + a13 * 0.0,
            a10 * b03 + a11 * b13 + a12 * b23 + a13 * 1.0,
        ];
        let r2 = [
            a20 * b00 + a21 * b10 + a22 * b20 + a23 * 0.0,
            a20 * b01 + a21 * b11 + a22 * b21 + a23 * 0.0,
            a20 * b02 + a21 * b12 + a22 * b22 + a23 * 0.0,
            a20 * b03 + a21 * b13 + a22 * b23 + a23 * 1.0,
        ];

        self.r0 = r0;
        self.r1 = r1;
        self.r2 = r2;
        Ok(self)
    }

    /// Post-multiplies this affine transform by `b` (`self = self * b`).
    ///
    /// Returns `Err(LayoutError::NonAffineMatrix)` without mutating `self` if `b` contains
    /// non-finite components that cause the full 4x4 product's bottom row to become `NaN`.
    #[inline]
    pub fn multiply(&mut self, b: &Self) -> Result<&mut Self, LayoutError> {
        let a = *self;
        self.multiply_affines(&a, b)
    }

    /// Pre-multiplies this affine transform by `a` (`self = a * self`).
    ///
    /// Returns `Err(LayoutError::NonAffineMatrix)` without mutating `self` if `self` contains
    /// non-finite components that cause the full 4x4 product's bottom row to become `NaN`.
    #[inline]
    pub fn premultiply(&mut self, a: &Self) -> Result<&mut Self, LayoutError> {
        let b = *self;
        self.multiply_affines(a, &b)
    }

    /// Checked non-mutating composition returning a new `Affine3x4`.
    #[inline]
    pub fn checked_multiply(&self, b: &Self) -> Result<Self, LayoutError> {
        let mut out = Self::zero();
        out.multiply_affines(self, b)?;
        Ok(out)
    }

    /// Transforms a 3D point `(x, y, z)` matching Three.js `Vector3.applyMatrix4`.
    ///
    /// Invariant: Three.js evaluates the homogeneous denominator `w = 1 / (e3*x + e7*y + e11*z + e15)`.
    /// For affine matrices with e3=e7=e11=0 and e15=1, `0.0 * x + 0.0 * y + 0.0 * z + 1.0`
    /// produces `NaN` when any coordinate is `Infinity`, yielding `NaN` point components.
    #[inline]
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let x = p[0]; let y = p[1]; let z = p[2];
        let denom = 0.0 * x + 0.0 * y + 0.0 * z + 1.0;
        let w = 1.0 / denom;
        [
            (self.r0[0] * x + self.r0[1] * y + self.r0[2] * z + self.r0[3]) * w,
            (self.r1[0] * x + self.r1[1] * y + self.r1[2] * z + self.r1[3]) * w,
            (self.r2[0] * x + self.r2[1] * y + self.r2[2] * z + self.r2[3]) * w,
        ]
    }

    /// Transforms a 3D point `Vector3` matching Three.js `Vector3.applyMatrix4`.
    #[inline]
    pub fn transform_point_vector3(&self, p: &Vector3) -> Vector3 {
        let out = self.transform_point([p.x, p.y, p.z]);
        Vector3::new(out[0], out[1], out[2])
    }

    /// Transforms a 3D direction vector `(x, y, z)` ignoring translation (`w = 0.0`).
    #[inline]
    pub fn transform_vector(&self, v: [f64; 3]) -> [f64; 3] {
        let x = v[0]; let y = v[1]; let z = v[2];
        [
            self.r0[0] * x + self.r0[1] * y + self.r0[2] * z,
            self.r1[0] * x + self.r1[1] * y + self.r1[2] * z,
            self.r2[0] * x + self.r2[1] * y + self.r2[2] * z,
        ]
    }

    /// Transforms a 3D direction `Vector3` ignoring translation (`w = 0.0`).
    #[inline]
    pub fn transform_vector_vector3(&self, v: &Vector3) -> Vector3 {
        let out = self.transform_vector([v.x, v.y, v.z]);
        Vector3::new(out[0], out[1], out[2])
    }

    /// Computes the affine determinant (the determinant of the 3x3 linear basis).
    ///
    /// Matches Three.js r186 `Matrix4.determinantAffine()`.
    #[inline]
    pub fn determinant(&self) -> f64 {
        let n11 = self.r0[0]; let n12 = self.r0[1]; let n13 = self.r0[2];
        let n21 = self.r1[0]; let n22 = self.r1[1]; let n23 = self.r1[2];
        let n31 = self.r2[0]; let n32 = self.r2[1]; let n33 = self.r2[2];

        n11 * (n22 * n33 - n23 * n32) -
        n12 * (n21 * n33 - n23 * n31) +
        n13 * (n21 * n32 - n22 * n31)
    }

    /// Returns `true` if this transform contains reflection/negative scale (`determinant < 0.0`).
    #[inline]
    pub fn has_negative_scale(&self) -> bool {
        self.determinant() < 0.0
    }

    /// Returns the translation vector `[tx, ty, tz]`.
    #[inline]
    pub const fn translation(&self) -> [f64; 3] {
        [self.r0[3], self.r1[3], self.r2[3]]
    }

    /// Sets the translation components `(tx, ty, tz)`.
    #[inline]
    pub fn set_translation(&mut self, tx: f64, ty: f64, tz: f64) -> &mut Self {
        self.r0[3] = tx;
        self.r1[3] = ty;
        self.r2[3] = tz;
        self
    }
}

impl Default for Affine3x4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
