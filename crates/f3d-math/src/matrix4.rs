//! 4x4 matrix primitive with `f64` public semantics matching Three.js r186 `Matrix4`.

use core::fmt;
use core::simd::f64x4;
use crate::euler::{Euler, EulerOrder};
use crate::jsnum::js_max;
use crate::matrix3::Matrix3;
use crate::narrowing::{check_narrow_f64, NarrowingError, NarrowingTolerance};
use crate::quaternion::Quaternion;
use crate::vector3::Vector3;


/// A 4x4 matrix stored in column-major order matching Three.js `Matrix4.elements`.
///
/// Element indexing:
/// ```text
/// e[0] = m11, e[4] = m12, e[8]  = m13, e[12] = m14
/// e[1] = m21, e[5] = m22, e[9]  = m23, e[13] = m24
/// e[2] = m31, e[6] = m32, e[10] = m33, e[14] = m34
/// e[3] = m41, e[7] = m42, e[11] = m43, e[15] = m44
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Matrix4 {
    /// 16 double-precision float elements in column-major order.
    pub elements: [f64; 16],
}

impl Matrix4 {
    /// Constructs an identity 4x4 matrix.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            elements: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Constructs a matrix with all elements set to zero.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            elements: [0.0; 16],
        }
    }

    /// Constructs a `Matrix4` from 16 column-major float values.
    #[inline]
    pub const fn from_elements(elements: [f64; 16]) -> Self {
        Self { elements }
    }

    /// Returns the 16 column-major elements.
    #[inline]
    pub const fn to_elements(&self) -> [f64; 16] {
        self.elements
    }

    /// Sets the elements of this matrix from row-major parameters.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &mut self,
        n11: f64, n12: f64, n13: f64, n14: f64,
        n21: f64, n22: f64, n23: f64, n24: f64,
        n31: f64, n32: f64, n33: f64, n34: f64,
        n41: f64, n42: f64, n43: f64, n44: f64,
    ) -> &mut Self {
        let te = &mut self.elements;
        te[0] = n11; te[4] = n12; te[8]  = n13; te[12] = n14;
        te[1] = n21; te[5] = n22; te[9]  = n23; te[13] = n24;
        te[2] = n31; te[6] = n32; te[10] = n33; te[14] = n34;
        te[3] = n41; te[7] = n42; te[11] = n43; te[15] = n44;
        self
    }

    /// Copies elements from matrix `m` into this instance.
    #[inline]
    pub fn copy(&mut self, m: &Self) -> &mut Self {
        self.elements = m.elements;
        self
    }

    /// Multiplies matrix `a` by `b` and stores the result in this instance (`self = a * b`).
    ///
    /// Preserves exact Three.js r186 operation ordering and column-major evaluation.
    #[inline]
    pub fn multiply_matrices(&mut self, a: &Self, b: &Self) -> &mut Self {
        let ae = &a.elements;
        let be = &b.elements;

        let a11 = ae[0]; let a12 = ae[4]; let a13 = ae[8];  let a14 = ae[12];
        let a21 = ae[1]; let a22 = ae[5]; let a23 = ae[9];  let a24 = ae[13];
        let a31 = ae[2]; let a32 = ae[6]; let a33 = ae[10]; let a34 = ae[14];
        let a41 = ae[3]; let a42 = ae[7]; let a43 = ae[11]; let a44 = ae[15];

        let b11 = be[0]; let b12 = be[4]; let b13 = be[8];  let b14 = be[12];
        let b21 = be[1]; let b22 = be[5]; let b23 = be[9];  let b24 = be[13];
        let b31 = be[2]; let b32 = be[6]; let b33 = be[10]; let b34 = be[14];
        let b41 = be[3]; let b42 = be[7]; let b43 = be[11]; let b44 = be[15];

        let te = &mut self.elements;
        te[0]  = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
        te[4]  = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
        te[8]  = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
        te[12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

        te[1]  = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
        te[5]  = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
        te[9]  = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
        te[13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

        te[2]  = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
        te[6]  = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
        te[10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
        te[14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

        te[3]  = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
        te[7]  = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
        te[11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
        te[15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;

        self
    }

    /// Multiplies this matrix by matrix `m` (`self = self * m`).
    #[inline]
    pub fn multiply(&mut self, m: &Self) -> &mut Self {
        let a = *self;
        self.multiply_matrices(&a, m)
    }

    /// Premultiplies this matrix by matrix `m` (`self = m * self`).
    #[inline]
    pub fn premultiply(&mut self, m: &Self) -> &mut Self {
        let b = *self;
        self.multiply_matrices(m, &b)
    }

    /// Multiplies every component of the matrix by scalar `s`.
    #[inline]
    pub fn multiply_scalar(&mut self, s: f64) -> &mut Self {
        for val in &mut self.elements {
            *val *= s;
        }
        self
    }

    /// Transposes this matrix in place.
    #[inline]
    pub fn transpose(&mut self) -> &mut Self {
        let te = &mut self.elements;
        let mut tmp;

        tmp = te[1]; te[1] = te[4]; te[4] = tmp;
        tmp = te[2]; te[2] = te[8]; te[8] = tmp;
        tmp = te[6]; te[6] = te[9]; te[9] = tmp;

        tmp = te[3]; te[3] = te[12]; te[12] = tmp;
        tmp = te[7]; te[7] = te[13]; te[13] = tmp;
        tmp = te[11]; te[11] = te[14]; te[14] = tmp;

        self
    }

    /// Sets the position components of this matrix (`e[12], e[13], e[14]`).
    #[inline]
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.elements[12] = x;
        self.elements[13] = y;
        self.elements[14] = z;
        self
    }

    /// Sets the position components from a `Vector3`.
    #[inline]
    pub fn set_position_vec(&mut self, v: &Vector3) -> &mut Self {
        self.set_position(v.x, v.y, v.z)
    }

    /// Computes and returns the full 4x4 determinant matching Three.js r186 `Matrix4.determinant()`.
    pub fn determinant(&self) -> f64 {
        let te = &self.elements;

        let n11 = te[0]; let n12 = te[4]; let n13 = te[8];  let n14 = te[12];
        let n21 = te[1]; let n22 = te[5]; let n23 = te[9];  let n24 = te[13];
        let n31 = te[2]; let n32 = te[6]; let n33 = te[10]; let n34 = te[14];
        let n41 = te[3]; let n42 = te[7]; let n43 = te[11]; let n44 = te[15];

        let t11 = n23 * n34 - n24 * n33;
        let t12 = n22 * n34 - n24 * n32;
        let t13 = n22 * n33 - n23 * n32;

        let t21 = n21 * n34 - n24 * n31;
        let t22 = n21 * n33 - n23 * n31;
        let t23 = n21 * n32 - n22 * n31;

        n11 * (n42 * t11 - n43 * t12 + n44 * t13) -
        n12 * (n41 * t11 - n43 * t21 + n44 * t22) +
        n13 * (n41 * t12 - n42 * t21 + n44 * t23) -
        n14 * (n41 * t13 - n42 * t22 + n43 * t23)
    }

    /// Computes and returns the affine determinant (assuming bottom row is `[0, 0, 0, 1]`),
    /// matching Three.js r186 `Matrix4.determinantAffine()`.
    pub fn determinant_affine(&self) -> f64 {
        let te = &self.elements;

        let n11 = te[0]; let n12 = te[4]; let n13 = te[8];
        let n21 = te[1]; let n22 = te[5]; let n23 = te[9];
        let n31 = te[2]; let n32 = te[6]; let n33 = te[10];

        n11 * (n22 * n33 - n23 * n32) -
        n12 * (n21 * n33 - n23 * n31) +
        n13 * (n21 * n32 - n22 * n31)
    }

    /// Inverts this matrix in place using the analytic solution.
    ///
    /// Invariant: Matches Three.js r186 singular matrix behavior.
    /// When `determinant == 0.0`, sets all 16 elements to `0.0`.
    pub fn invert(&mut self) -> &mut Self {
        let te = &self.elements;

        let n11 = te[0];  let n21 = te[1];  let n31 = te[2];  let n41 = te[3];
        let n12 = te[4];  let n22 = te[5];  let n32 = te[6];  let n42 = te[7];
        let n13 = te[8];  let n23 = te[9];  let n33 = te[10]; let n43 = te[11];
        let n14 = te[12]; let n24 = te[13]; let n34 = te[14]; let n44 = te[15];

        let t1 = n11 * n22 - n21 * n12;
        let t2 = n11 * n32 - n31 * n12;
        let t3 = n11 * n42 - n41 * n12;
        let t4 = n21 * n32 - n31 * n22;
        let t5 = n21 * n42 - n41 * n22;
        let t6 = n31 * n42 - n41 * n32;
        let t7 = n13 * n24 - n23 * n14;
        let t8 = n13 * n34 - n33 * n14;
        let t9 = n13 * n44 - n43 * n14;
        let t10 = n23 * n34 - n33 * n24;
        let t11 = n23 * n44 - n43 * n24;
        let t12 = n33 * n44 - n43 * n34;

        let det = t1 * t12 - t2 * t11 + t3 * t10 + t4 * t9 - t5 * t8 + t6 * t7;

        if det == 0.0 {
            self.elements = [0.0; 16];
            return self;
        }

        let det_inv = 1.0 / det;

        let out = &mut self.elements;
        out[0]  = ( n22 * t12 - n32 * t11 + n42 * t10) * det_inv;
        out[1]  = ( n31 * t11 - n21 * t12 - n41 * t10) * det_inv;
        out[2]  = ( n24 * t6  - n34 * t5  + n44 * t4 ) * det_inv;
        out[3]  = ( n33 * t5  - n23 * t6  - n43 * t4 ) * det_inv;

        out[4]  = ( n32 * t9  - n12 * t12 - n42 * t8 ) * det_inv;
        out[5]  = ( n11 * t12 - n31 * t9  + n41 * t8 ) * det_inv;
        out[6]  = ( n34 * t3  - n14 * t6  - n44 * t2 ) * det_inv;
        out[7]  = ( n13 * t6  - n33 * t3  + n43 * t2 ) * det_inv;

        out[8]  = ( n12 * t11 - n22 * t9  + n42 * t7 ) * det_inv;
        out[9]  = ( n21 * t9  - n11 * t11 - n41 * t7 ) * det_inv;
        out[10] = ( n14 * t5  - n24 * t3  + n44 * t1 ) * det_inv;
        out[11] = ( n23 * t3  - n13 * t5  - n43 * t1 ) * det_inv;

        out[12] = ( n22 * t8  - n12 * t10 - n32 * t7 ) * det_inv;
        out[13] = ( n11 * t10 - n21 * t8  + n31 * t7 ) * det_inv;
        out[14] = ( n24 * t2  - n14 * t4  - n34 * t1 ) * det_inv;
        out[15] = ( n13 * t4  - n23 * t2  + n33 * t1 ) * det_inv;

        self
    }

    /// Safe fallible inverse returning `None` if the matrix is singular (`determinant == 0.0`).
    pub fn try_invert(&self) -> Option<Self> {
        let mut copy = *self;
        copy.invert();
        if copy.elements == [0.0; 16] && self.determinant() == 0.0 {
            None
        } else {
            Some(copy)
        }
    }

    /// Scales the first three columns of this matrix by components of scale vector `v`.
    #[inline]
    pub fn scale(&mut self, v: &Vector3) -> &mut Self {
        let te = &mut self.elements;
        let x = v.x;
        let y = v.y;
        let z = v.z;

        te[0] *= x; te[4] *= y; te[8]  *= z;
        te[1] *= x; te[5] *= y; te[9]  *= z;
        te[2] *= x; te[6] *= y; te[10] *= z;
        te[3] *= x; te[7] *= y; te[11] *= z;

        self
    }

    /// Composes this transformation matrix from position, rotation quaternion, and scale.
    ///
    /// Invariant: Evaluates Shoemake's quaternion-to-matrix expansion directly without
    /// normalizing `quaternion`, faithfully preserving authored non-unit quaternions and
    /// negative scaling according to Three.js r186.
    pub fn compose(&mut self, position: &Vector3, quaternion: &Quaternion, scale: &Vector3) -> &mut Self {
        let x = quaternion.x;
        let y = quaternion.y;
        let z = quaternion.z;
        let w = quaternion.w;

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;

        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        let sx = scale.x;
        let sy = scale.y;
        let sz = scale.z;

        let te = &mut self.elements;
        te[0]  = (1.0 - (yy + zz)) * sx;
        te[1]  = (xy + wz) * sx;
        te[2]  = (xz - wy) * sx;
        te[3]  = 0.0;

        te[4]  = (xy - wz) * sy;
        te[5]  = (1.0 - (xx + zz)) * sy;
        te[6]  = (yz + wx) * sy;
        te[7]  = 0.0;

        te[8]  = (xz + wy) * sz;
        te[9]  = (yz - wx) * sz;
        te[10] = (1.0 - (xx + yy)) * sz;
        te[11] = 0.0;

        te[12] = position.x;
        te[13] = position.y;
        te[14] = position.z;
        te[15] = 1.0;

        self
    }

    /// Composes a batch of $N$ transformation matrices from caller-provided position,
    /// quaternion, and scale slices into a caller-provided `outputs` slice.
    ///
    /// # Invariants
    /// - Evaluates each transform using the exact scalar operation ordering of [`Matrix4::compose`].
    /// - Validates that all input and output slice lengths match before performing any writes;
    ///   on length mismatch, returns `Err(BatchComposeError::LengthMismatch)` and leaves `outputs` unmodified.
    /// - An empty batch ($N = 0$) is valid and immediately returns `Ok(())`.
    /// - Preserves authored non-unit quaternions, negative scaling, and signed zero without heap allocations.
    pub fn batch_compose(
        positions: &[Vector3],
        quaternions: &[Quaternion],
        scales: &[Vector3],
        outputs: &mut [Matrix4],
    ) -> Result<(), BatchComposeError> {
        let count = positions.len();
        if quaternions.len() != count || scales.len() != count || outputs.len() != count {
            return Err(BatchComposeError::LengthMismatch {
                positions_len: count,
                quaternions_len: quaternions.len(),
                scales_len: scales.len(),
                outputs_len: outputs.len(),
            });
        }

        for i in 0..count {
            outputs[i].compose(&positions[i], &quaternions[i], &scales[i]);
        }

        Ok(())
    }

    /// Composes a batch of $N$ transformation matrices from caller-provided position,
    /// quaternion, and scale slices using 4-lane `core::simd` vector operations on native hosts,
    /// with scalar [`Matrix4::compose`] processing any remaining tail elements.
    ///
    /// # Invariants
    /// - Evaluates each transform using the identical algebraic expression ordering and grouping
    ///   as [`Matrix4::compose`] without reassociation, reordering, or `mul_add` contraction.
    /// - Validates that all input and output slice lengths match before performing any writes;
    ///   on length mismatch, returns `Err(BatchComposeError::LengthMismatch)` and leaves `outputs` unmodified.
    /// - An empty batch ($N = 0$) is valid and immediately returns `Ok(())`.
    /// - Preserves authored non-unit quaternions, negative scaling, and signed zero without heap allocations.
    ///
    /// # Note on Performance Claims
    /// No acceleration, speedup, or crossover claims are made for this native SIMD kernel.
    /// The Wasm release contract currently compiles with default flags and does not enable `simd128`,
    /// so there is no browser SIMD claim. Empirical cross-over benchmarks and browser acceleration
    /// validation are deferred to dedicated benchmark gates.
    pub fn batch_compose_simd(
        positions: &[Vector3],
        quaternions: &[Quaternion],
        scales: &[Vector3],
        outputs: &mut [Matrix4],
    ) -> Result<(), BatchComposeError> {
        let count = positions.len();
        if quaternions.len() != count || scales.len() != count || outputs.len() != count {
            return Err(BatchComposeError::LengthMismatch {
                positions_len: count,
                quaternions_len: quaternions.len(),
                scales_len: scales.len(),
                outputs_len: outputs.len(),
            });
        }

        let chunks = count / 4;
        let simd_end = chunks * 4;

        let mut i = 0;
        while i < simd_end {
            let x = f64x4::from_array([
                quaternions[i].x,
                quaternions[i + 1].x,
                quaternions[i + 2].x,
                quaternions[i + 3].x,
            ]);
            let y = f64x4::from_array([
                quaternions[i].y,
                quaternions[i + 1].y,
                quaternions[i + 2].y,
                quaternions[i + 3].y,
            ]);
            let z = f64x4::from_array([
                quaternions[i].z,
                quaternions[i + 1].z,
                quaternions[i + 2].z,
                quaternions[i + 3].z,
            ]);
            let w = f64x4::from_array([
                quaternions[i].w,
                quaternions[i + 1].w,
                quaternions[i + 2].w,
                quaternions[i + 3].w,
            ]);

            let x2 = x + x;
            let y2 = y + y;
            let z2 = z + z;

            let xx = x * x2;
            let xy = x * y2;
            let xz = x * z2;
            let yy = y * y2;
            let yz = y * z2;
            let zz = z * z2;
            let wx = w * x2;
            let wy = w * y2;
            let wz = w * z2;

            let sx = f64x4::from_array([
                scales[i].x,
                scales[i + 1].x,
                scales[i + 2].x,
                scales[i + 3].x,
            ]);
            let sy = f64x4::from_array([
                scales[i].y,
                scales[i + 1].y,
                scales[i + 2].y,
                scales[i + 3].y,
            ]);
            let sz = f64x4::from_array([
                scales[i].z,
                scales[i + 1].z,
                scales[i + 2].z,
                scales[i + 3].z,
            ]);

            let one = f64x4::splat(1.0);
            let zero = f64x4::splat(0.0);

            let m0  = (one - (yy + zz)) * sx;
            let m1  = (xy + wz) * sx;
            let m2  = (xz - wy) * sx;
            let m3  = zero;

            let m4  = (xy - wz) * sy;
            let m5  = (one - (xx + zz)) * sy;
            let m6  = (yz + wx) * sy;
            let m7  = zero;

            let m8  = (xz + wy) * sz;
            let m9  = (yz - wx) * sz;
            let m10 = (one - (xx + yy)) * sz;
            let m11 = zero;

            let m12 = f64x4::from_array([
                positions[i].x,
                positions[i + 1].x,
                positions[i + 2].x,
                positions[i + 3].x,
            ]);
            let m13 = f64x4::from_array([
                positions[i].y,
                positions[i + 1].y,
                positions[i + 2].y,
                positions[i + 3].y,
            ]);
            let m14 = f64x4::from_array([
                positions[i].z,
                positions[i + 1].z,
                positions[i + 2].z,
                positions[i + 3].z,
            ]);
            let m15 = one;

            let a0 = m0.to_array();
            let a1 = m1.to_array();
            let a2 = m2.to_array();
            let a3 = m3.to_array();
            let a4 = m4.to_array();
            let a5 = m5.to_array();
            let a6 = m6.to_array();
            let a7 = m7.to_array();
            let a8 = m8.to_array();
            let a9 = m9.to_array();
            let a10 = m10.to_array();
            let a11 = m11.to_array();
            let a12 = m12.to_array();
            let a13 = m13.to_array();
            let a14 = m14.to_array();
            let a15 = m15.to_array();

            for lane in 0..4 {
                outputs[i + lane].elements = [
                    a0[lane], a1[lane], a2[lane], a3[lane],
                    a4[lane], a5[lane], a6[lane], a7[lane],
                    a8[lane], a9[lane], a10[lane], a11[lane],
                    a12[lane], a13[lane], a14[lane], a15[lane],
                ];
            }

            i += 4;
        }

        while i < count {
            outputs[i].compose(&positions[i], &quaternions[i], &scales[i]);
            i += 1;
        }

        Ok(())
    }

    /// Extracts the rotation component of matrix `m` into this matrix's rotation component.
    ///
    /// If `m.determinant_affine() == 0.0`, resets this matrix to identity matching Three.js r186.
    /// Note: This method does not support reflection matrices.
    pub fn extract_rotation(&mut self, m: &Self) -> &mut Self {
        if m.determinant_affine() == 0.0 {
            *self = Self::identity();
            return self;
        }

        let me = &m.elements;

        let scale_x = 1.0 / (me[0] * me[0] + me[1] * me[1] + me[2] * me[2]).sqrt();
        let scale_y = 1.0 / (me[4] * me[4] + me[5] * me[5] + me[6] * me[6]).sqrt();
        let scale_z = 1.0 / (me[8] * me[8] + me[9] * me[9] + me[10] * me[10]).sqrt();

        self.elements[0] = me[0] * scale_x;
        self.elements[1] = me[1] * scale_x;
        self.elements[2] = me[2] * scale_x;
        self.elements[3] = 0.0;

        self.elements[4] = me[4] * scale_y;
        self.elements[5] = me[5] * scale_y;
        self.elements[6] = me[6] * scale_y;
        self.elements[7] = 0.0;

        self.elements[8] = me[8] * scale_z;
        self.elements[9] = me[9] * scale_z;
        self.elements[10] = me[10] * scale_z;
        self.elements[11] = 0.0;

        self.elements[12] = 0.0;
        self.elements[13] = 0.0;
        self.elements[14] = 0.0;
        self.elements[15] = 1.0;

        self
    }

    /// Sets the rotation component of this transformation matrix from a quaternion,
    /// with position zero `(0, 0, 0)` and unit scale `(1, 1, 1)` matching Three.js r186.
    #[inline]
    pub fn make_rotation_from_quaternion(&mut self, q: &Quaternion) -> &mut Self {
        self.compose(&Vector3::zero(), q, &Vector3::one())
    }

    /// Sets the rotation component of this transformation matrix from Euler angles,
    /// with the rest of the matrix set to the identity, matching Three.js r186 `Matrix4.makeRotationFromEuler`.
    ///
    /// Evaluates direct trigonometric formulas across all six Euler rotation orders
    /// (`XYZ`, `YXZ`, `ZXY`, `ZYX`, `YZX`, `XZY`) using native `f64::cos` and `f64::sin`
    /// without routing through quaternion composition.
    ///
    /// # Numerics Note
    /// Evaluates native `f64::sin` and `f64::cos` operations, matching Three.js r186 within
    /// floating-point tolerance ($10^{-15}$); no bit-identical browser transcendental claim
    /// is made without a host browser differential proof.
    pub fn make_rotation_from_euler(&mut self, euler: &Euler) -> &mut Self {
        let te = &mut self.elements;

        let x = euler.x;
        let y = euler.y;
        let z = euler.z;
        let a = x.cos();
        let b = x.sin();
        let c = y.cos();
        let d = y.sin();
        let e = z.cos();
        let f = z.sin();

        match euler.order {
            EulerOrder::XYZ => {
                let ae = a * e;
                let af = a * f;
                let be = b * e;
                let bf = b * f;

                te[0] = c * e;
                te[4] = -c * f;
                te[8] = d;

                te[1] = af + be * d;
                te[5] = ae - bf * d;
                te[9] = -b * c;

                te[2] = bf - ae * d;
                te[6] = be + af * d;
                te[10] = a * c;
            }
            EulerOrder::YXZ => {
                let ce = c * e;
                let cf = c * f;
                let de = d * e;
                let df = d * f;

                te[0] = ce + df * b;
                te[4] = de * b - cf;
                te[8] = a * d;

                te[1] = a * f;
                te[5] = a * e;
                te[9] = -b;

                te[2] = cf * b - de;
                te[6] = df + ce * b;
                te[10] = a * c;
            }
            EulerOrder::ZXY => {
                let ce = c * e;
                let cf = c * f;
                let de = d * e;
                let df = d * f;

                te[0] = ce - df * b;
                te[4] = -a * f;
                te[8] = de + cf * b;

                te[1] = cf + de * b;
                te[5] = a * e;
                te[9] = df - ce * b;

                te[2] = -a * d;
                te[6] = b;
                te[10] = a * c;
            }
            EulerOrder::ZYX => {
                let ae = a * e;
                let af = a * f;
                let be = b * e;
                let bf = b * f;

                te[0] = c * e;
                te[4] = be * d - af;
                te[8] = ae * d + bf;

                te[1] = c * f;
                te[5] = bf * d + ae;
                te[9] = af * d - be;

                te[2] = -d;
                te[6] = b * c;
                te[10] = a * c;
            }
            EulerOrder::YZX => {
                let ac = a * c;
                let ad = a * d;
                let bc = b * c;
                let bd = b * d;

                te[0] = c * e;
                te[4] = bd - ac * f;
                te[8] = bc * f + ad;

                te[1] = f;
                te[5] = a * e;
                te[9] = -b * e;

                te[2] = -d * e;
                te[6] = ad * f + bc;
                te[10] = ac - bd * f;
            }
            EulerOrder::XZY => {
                let ac = a * c;
                let ad = a * d;
                let bc = b * c;
                let bd = b * d;

                te[0] = c * e;
                te[4] = -f;
                te[8] = d * e;

                te[1] = ac * f + bd;
                te[5] = a * e;
                te[9] = ad * f - bc;

                te[2] = bc * f - ad;
                te[6] = b * e;
                te[10] = bd * f + ac;
            }
        }

        // bottom row
        te[3] = 0.0;
        te[7] = 0.0;
        te[11] = 0.0;

        // last column
        te[12] = 0.0;
        te[13] = 0.0;
        te[14] = 0.0;
        te[15] = 1.0;

        self
    }

    /// Decomposes this matrix into its position, rotation, and scale components.
    ///
    /// Returns `false` if the affine determinant is zero (matrix is singular), matching Three.js r186.
    pub fn decompose(&self, position: &mut Vector3, quaternion: &mut Quaternion, scale: &mut Vector3) -> bool {
        let te = &self.elements;

        position.x = te[12];
        position.y = te[13];
        position.z = te[14];

        let det = self.determinant_affine();
        if det == 0.0 {
            scale.set(1.0, 1.0, 1.0);
            quaternion.set(0.0, 0.0, 0.0, 1.0);
            return false;
        }

        let mut sx = (te[0] * te[0] + te[1] * te[1] + te[2] * te[2]).sqrt();
        let sy = (te[4] * te[4] + te[5] * te[5] + te[6] * te[6]).sqrt();
        let sz = (te[8] * te[8] + te[9] * te[9] + te[10] * te[10]).sqrt();

        // If determinant is negative, invert one scale component (reflection)
        if det < 0.0 {
            sx = -sx;
        }

        let mut rot_matrix = *self;
        let inv_sx = 1.0 / sx;
        let inv_sy = 1.0 / sy;
        let inv_sz = 1.0 / sz;

        rot_matrix.elements[0] *= inv_sx;
        rot_matrix.elements[1] *= inv_sx;
        rot_matrix.elements[2] *= inv_sx;

        rot_matrix.elements[4] *= inv_sy;
        rot_matrix.elements[5] *= inv_sy;
        rot_matrix.elements[6] *= inv_sy;

        rot_matrix.elements[8]  *= inv_sz;
        rot_matrix.elements[9]  *= inv_sz;
        rot_matrix.elements[10] *= inv_sz;

        quaternion.set_from_rotation_matrix(&rot_matrix);

        scale.x = sx;
        scale.y = sy;
        scale.z = sz;

        true
    }

    /// Returns the maximum scale factor along the three coordinate axes.
    ///
    /// Evaluates `sqrt(max(||col0||^2, ||col1||^2, ||col2||^2))` matching Three.js r186 `Matrix4.getMaxScaleOnAxis()`.
    #[inline]
    pub fn get_max_scale_on_axis(&self) -> f64 {
        let te = &self.elements;
        let scale_x_sq = te[0] * te[0] + te[1] * te[1] + te[2] * te[2];
        let scale_y_sq = te[4] * te[4] + te[5] * te[5] + te[6] * te[6];
        let scale_z_sq = te[8] * te[8] + te[9] * te[9] + te[10] * te[10];

        let max_sq = js_max(js_max(scale_x_sq, scale_y_sq), scale_z_sq);
        max_sq.sqrt()
    }

    /// Checks if this matrix represents an affine transformation.
    ///
    /// Invariant: Exact structural check matching Three.js r186 and `f3d-core::layout`.
    /// Row 3 must be exactly `[0.0, 0.0, 0.0, 1.0]`.
    /// No epsilon is permitted: small perspective terms produce major projective distortions on large coordinates.
    #[inline]
    pub fn is_affine(&self) -> bool {
        self.elements[3] == 0.0
            && self.elements[7] == 0.0
            && self.elements[11] == 0.0
            && self.elements[15] == 1.0
    }

    /// Checks if this matrix represents an affine transformation within an explicit tolerance `eps`.
    #[inline]
    pub fn is_affine_eps(&self, eps: f64) -> bool {
        let te = &self.elements;
        te[3].abs() <= eps
            && te[7].abs() <= eps
            && te[11].abs() <= eps
            && (te[15] - 1.0).abs() <= eps
    }

    /// Constructs a rotation matrix, looking from `eye` towards `target`, oriented by the `up` vector.
    ///
    /// Matches Three.js r186 `Matrix4.lookAt`.
    /// Note: Preserves untouched elements `elements[3]`, `elements[7]`, `elements[11]`,
    /// and column 3 (`elements[12..16]`).
    pub fn look_at(&mut self, eye: &Vector3, target: &Vector3, up: &Vector3) -> &mut Self {
        let mut z = Vector3::new(eye.x - target.x, eye.y - target.y, eye.z - target.z);

        if z.length_sq() == 0.0 {
            // eye and target are in the same position
            z.z = 1.0;
        }

        z.normalize();
        let mut x = Vector3::new(
            up.y * z.z - up.z * z.y,
            up.z * z.x - up.x * z.z,
            up.x * z.y - up.y * z.x,
        );

        if x.length_sq() == 0.0 {
            // up and z are parallel
            if up.z.abs() == 1.0 {
                z.x += 0.0001;
            } else {
                z.z += 0.0001;
            }

            z.normalize();
            x.set(
                up.y * z.z - up.z * z.y,
                up.z * z.x - up.x * z.z,
                up.x * z.y - up.y * z.x,
            );
        }

        x.normalize();
        let y = Vector3::new(
            z.y * x.z - z.z * x.y,
            z.z * x.x - z.x * x.z,
            z.x * x.y - z.y * x.x,
        );

        self.elements[0] = x.x;
        self.elements[4] = y.x;
        self.elements[8] = z.x;

        self.elements[1] = x.y;
        self.elements[5] = y.y;
        self.elements[9] = z.y;

        self.elements[2] = x.z;
        self.elements[6] = y.z;
        self.elements[10] = z.z;

        self
    }

    /// Creates a perspective projection matrix matching Three.js r186 `makePerspective`.
    ///
    /// Preserves exact operation order and supports `CoordinateSystem::WebGL` (depth range [-1, 1]),
    /// `CoordinateSystem::WebGPU` (depth range [0, 1]), and `reversed_depth`.
    #[allow(clippy::too_many_arguments)]
    pub fn make_perspective(
        &mut self,
        left: f64,
        right: f64,
        top: f64,
        bottom: f64,
        near: f64,
        far: f64,
        coordinate_system: CoordinateSystem,
        reversed_depth: bool,
    ) -> &mut Self {
        let x = 2.0 * near / (right - left);
        let y = 2.0 * near / (top - bottom);

        let a = (right + left) / (right - left);
        let b = (top + bottom) / (top - bottom);

        let (c, d) = if reversed_depth {
            (
                near / (far - near),
                (far * near) / (far - near),
            )
        } else {
            match coordinate_system {
                CoordinateSystem::WebGL => (
                    -(far + near) / (far - near),
                    (-2.0 * far * near) / (far - near),
                ),
                CoordinateSystem::WebGPU => (
                    -far / (far - near),
                    (-far * near) / (far - near),
                ),
            }
        };

        let te = &mut self.elements;
        te[0] = x;   te[4] = 0.0; te[8] = a;    te[12] = 0.0;
        te[1] = 0.0; te[5] = y;   te[9] = b;    te[13] = 0.0;
        te[2] = 0.0; te[6] = 0.0; te[10] = c;   te[14] = d;
        te[3] = 0.0; te[7] = 0.0; te[11] = -1.0; te[15] = 0.0;

        self
    }

    /// Creates an orthographic projection matrix matching Three.js r186 `makeOrthographic`.
    ///
    /// Preserves exact operation order and supports `CoordinateSystem::WebGL` (depth range [-1, 1]),
    /// `CoordinateSystem::WebGPU` (depth range [0, 1]), and `reversed_depth`.
    #[allow(clippy::too_many_arguments)]
    pub fn make_orthographic(
        &mut self,
        left: f64,
        right: f64,
        top: f64,
        bottom: f64,
        near: f64,
        far: f64,
        coordinate_system: CoordinateSystem,
        reversed_depth: bool,
    ) -> &mut Self {
        let x = 2.0 / (right - left);
        let y = 2.0 / (top - bottom);

        let a = -(right + left) / (right - left);
        let b = -(top + bottom) / (top - bottom);

        let (c, d) = if reversed_depth {
            (
                1.0 / (far - near),
                far / (far - near),
            )
        } else {
            match coordinate_system {
                CoordinateSystem::WebGL => (
                    -2.0 / (far - near),
                    -(far + near) / (far - near),
                ),
                CoordinateSystem::WebGPU => (
                    -1.0 / (far - near),
                    -near / (far - near),
                ),
            }
        };

        let te = &mut self.elements;
        te[0] = x;   te[4] = 0.0; te[8]  = 0.0; te[12] = a;
        te[1] = 0.0; te[5] = y;   te[9]  = 0.0; te[13] = b;
        te[2] = 0.0; te[6] = 0.0; te[10] = c;   te[14] = d;
        te[3] = 0.0; te[7] = 0.0; te[11] = 0.0; te[15] = 1.0;

        self
    }

    /// Copies the translation component from matrix `m` into this matrix
    /// matching Three.js r186 `Matrix4.copyPosition()`.
    #[inline]
    pub fn copy_position(&mut self, m: &Self) -> &mut Self {
        self.elements[12] = m.elements[12];
        self.elements[13] = m.elements[13];
        self.elements[14] = m.elements[14];
        self
    }

    /// Sets the upper 3x3 elements of this matrix from the given `Matrix3`
    /// matching Three.js r186 `Matrix4.setFromMatrix3()`.
    #[inline]
    pub fn set_from_matrix3(&mut self, m: &Matrix3) -> &mut Self {
        let me = &m.elements;
        self.set(
            me[0], me[3], me[6], 0.0,
            me[1], me[4], me[7], 0.0,
            me[2], me[5], me[8], 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Sets the basis vectors of this matrix matching Three.js r186 `Matrix4.makeBasis()`.
    #[inline]
    pub fn make_basis(
        &mut self,
        x_axis: &Vector3,
        y_axis: &Vector3,
        z_axis: &Vector3,
    ) -> &mut Self {
        self.set(
            x_axis.x, y_axis.x, z_axis.x, 0.0,
            x_axis.y, y_axis.y, z_axis.y, 0.0,
            x_axis.z, y_axis.z, z_axis.z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Extracts the basis vectors of this matrix into the three given vectors
    /// matching Three.js r186 `Matrix4.extractBasis()`.
    ///
    /// If `self.determinant_affine() == 0.0`, resets basis vectors to canonical axes (1,0,0), (0,1,0), (0,0,1).
    pub fn extract_basis(
        &mut self,
        x_axis: &mut Vector3,
        y_axis: &mut Vector3,
        z_axis: &mut Vector3,
    ) -> &mut Self {
        if self.determinant_affine() == 0.0 {
            x_axis.set(1.0, 0.0, 0.0);
            y_axis.set(0.0, 1.0, 0.0);
            z_axis.set(0.0, 0.0, 1.0);
            return self;
        }

        x_axis.set_from_matrix_column(self, 0);
        y_axis.set_from_matrix_column(self, 1);
        z_axis.set_from_matrix_column(self, 2);

        self
    }

    /// Sets this matrix as a translation transformation matching Three.js r186 `Matrix4.makeTranslation()`.
    #[inline]
    pub fn make_translation(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.set(
            1.0, 0.0, 0.0, x,
            0.0, 1.0, 0.0, y,
            0.0, 0.0, 1.0, z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Sets this matrix as a translation transformation from a `Vector3`.
    #[inline]
    pub fn make_translation_vec(&mut self, v: &Vector3) -> &mut Self {
        self.make_translation(v.x, v.y, v.z)
    }

    /// Alias for [`make_translation_vec`](Self::make_translation_vec).
    #[inline]
    pub fn make_translation_v(&mut self, v: &Vector3) -> &mut Self {
        self.make_translation(v.x, v.y, v.z)
    }

    /// Sets this matrix as a rotation around the X axis by `theta` radians
    /// matching Three.js r186 `Matrix4.makeRotationX()`.
    #[inline]
    pub fn make_rotation_x(&mut self, theta: f64) -> &mut Self {
        let c = theta.cos();
        let s = theta.sin();
        self.set(
            1.0, 0.0, 0.0, 0.0,
            0.0, c, -s, 0.0,
            0.0, s, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Sets this matrix as a rotation around the Y axis by `theta` radians
    /// matching Three.js r186 `Matrix4.makeRotationY()`.
    #[inline]
    pub fn make_rotation_y(&mut self, theta: f64) -> &mut Self {
        let c = theta.cos();
        let s = theta.sin();
        self.set(
            c, 0.0, s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Sets this matrix as a rotation around the Z axis by `theta` radians
    /// matching Three.js r186 `Matrix4.makeRotationZ()`.
    #[inline]
    pub fn make_rotation_z(&mut self, theta: f64) -> &mut Self {
        let c = theta.cos();
        let s = theta.sin();
        self.set(
            c, -s, 0.0, 0.0,
            s, c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Sets this matrix as a rotation around the normalized `axis` by `angle` radians
    /// matching Three.js r186 `Matrix4.makeRotationAxis()`.
    #[inline]
    pub fn make_rotation_axis(&mut self, axis: &Vector3, angle: f64) -> &mut Self {
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        let x = axis.x;
        let y = axis.y;
        let z = axis.z;
        let tx = t * x;
        let ty = t * y;

        self.set(
            tx * x + c, tx * y - s * z, tx * z + s * y, 0.0,
            tx * y + s * z, ty * y + c, ty * z - s * x, 0.0,
            tx * z - s * y, ty * z + s * x, t * z * z + c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Sets this matrix as a scale transformation matching Three.js r186 `Matrix4.makeScale()`.
    #[inline]
    pub fn make_scale(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.set(
            x, 0.0, 0.0, 0.0,
            0.0, y, 0.0, 0.0,
            0.0, 0.0, z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Sets this matrix as a scale transformation from a `Vector3`.
    #[inline]
    pub fn make_scale_vec(&mut self, v: &Vector3) -> &mut Self {
        self.make_scale(v.x, v.y, v.z)
    }

    /// Sets this matrix as a shear transformation matching Three.js r186 `Matrix4.makeShear()`.
    #[inline]
    pub fn make_shear(
        &mut self,
        xy: f64,
        xz: f64,
        yx: f64,
        yz: f64,
        zx: f64,
        zy: f64,
    ) -> &mut Self {
        self.set(
            1.0, yx, zx, 0.0,
            xy, 1.0, zy, 0.0,
            xz, yz, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Narrows this `f64` 4x4 matrix into `[f32; 16]`, verifying that precision loss does not
    /// exceed `max_abs_err` or `max_rel_err`.
    ///
    /// Non-finite values (`NaN`, `Infinity`) are preserved.
    pub fn to_f32_checked(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 16], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, false)
    }

    /// Strict narrowing: rejects non-finite components (`NaN`, `Infinity`) and checks tolerance.
    pub fn to_f32_strict(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
    ) -> Result<[f32; 16], NarrowingError> {
        self.to_f32_with_policy(max_abs_err, max_rel_err, true)
    }

    /// Narrows this matrix with explicit `NarrowingTolerance` parameters.
    pub fn to_f32_with_tolerance(
        &self,
        tolerance: NarrowingTolerance,
        strict: bool,
    ) -> Result<[f32; 16], NarrowingError> {
        self.to_f32_with_policy(tolerance.max_abs_err, tolerance.max_rel_err, strict)
    }

    /// Checked narrowing helper evaluating all 16 elements with optional strict non-finite rejection.
    pub fn to_f32_with_policy(
        &self,
        max_abs_err: f64,
        max_rel_err: f64,
        strict: bool,
    ) -> Result<[f32; 16], NarrowingError> {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = check_narrow_f64(self.elements[i], i, max_abs_err, max_rel_err, strict)?;
        }
        Ok(out)
    }

    /// Resets this matrix to the 4x4 identity matrix.
    ///
    /// Matches Three.js r186 `Matrix4.identity()`.
    #[inline]
    pub fn set_identity(&mut self) -> &mut Self {
        *self = Self::identity();
        self
    }

    /// Returns true if all 16 elements strictly equal those of `other`.
    ///
    /// Matches Three.js r186 `Matrix4.equals(matrix)` using strict JS `===` semantics
    /// (`NaN != NaN` returns `false`, `-0.0 == +0.0` returns `true`).
    #[inline]
    #[must_use]
    pub fn equals(&self, other: &Self) -> bool {
        for i in 0..16 {
            if self.elements[i] != other.elements[i] {
                return false;
            }
        }
        true
    }

    /// Sets the elements of this matrix from the given slice in column-major order starting at `offset`.
    ///
    /// Validates bounds upfront to guarantee no partial mutation on out-of-range input.
    /// Matches Three.js r186 `Matrix4.fromArray(array, offset)` in the valid native slice domain.
    #[inline]
    pub fn from_slice_offset(&mut self, array: &[f64], offset: usize) -> &mut Self {
        assert!(offset + 16 <= array.len(), "slice too short for Matrix4 read");
        self.elements.copy_from_slice(&array[offset..offset + 16]);
        self
    }

    /// Writes the elements of this matrix in column-major order into `array` starting at `offset`.
    ///
    /// Validates bounds upfront to guarantee no partial mutation on out-of-range destination.
    /// Matches Three.js r186 `Matrix4.toArray(array, offset)` in the valid native slice domain.
    #[inline]
    pub fn to_slice_offset<'a>(&self, array: &'a mut [f64], offset: usize) -> &'a mut [f64] {
        assert!(offset + 16 <= array.len(), "destination slice too short for Matrix4 write");
        array[offset..offset + 16].copy_from_slice(&self.elements);
        array
    }
}

/// Errors occurring during batch matrix composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BatchComposeError {
    /// Input or output slice lengths do not match.
    LengthMismatch {
        /// Number of position vectors provided.
        positions_len: usize,
        /// Number of rotation quaternions provided.
        quaternions_len: usize,
        /// Number of scale vectors provided.
        scales_len: usize,
        /// Number of output matrices provided.
        outputs_len: usize,
    },
}

impl fmt::Display for BatchComposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                positions_len,
                quaternions_len,
                scales_len,
                outputs_len,
            } => {
                write!(
                    f,
                    "batch compose slice length mismatch: positions={}, quaternions={}, scales={}, outputs={}",
                    positions_len, quaternions_len, scales_len, outputs_len
                )
            }
        }
    }
}

impl core::error::Error for BatchComposeError {}


/// Target coordinate system for projection matrices matching Three.js r186 constants.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoordinateSystem {
    /// WebGL clip space: depth range [-1, 1] (constant value 2000 in Three.js).
    WebGL = 2000,
    /// WebGPU clip space: depth range [0, 1] (constant value 2001 in Three.js).
    WebGPU = 2001,
}

impl CoordinateSystem {
    /// Returns the raw numeric constant matching Three.js.
    #[inline]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }
}

impl fmt::Display for Matrix4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let e = &self.elements;
        write!(
            f,
            "Matrix4([{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}])",
            e[0], e[4], e[8], e[12],
            e[1], e[5], e[9], e[13],
            e[2], e[6], e[10], e[14],
            e[3], e[7], e[11], e[15],
        )
    }
}

impl Default for Matrix4 {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix4_identity_equals_and_array_slice_io_with_oracles() {
        // Node oracle 1: identity resets matrix to 4x4 identity
        let mut m = Matrix4::zero();
        m.set_identity();
        assert_eq!(m, Matrix4::identity());

        // Node oracle 2: equals with strict === semantics
        // Identity vs Identity -> true
        assert!(m.equals(&Matrix4::identity()));

        // -0.0 vs +0.0 -> true
        let mut m_neg_zero = Matrix4::identity();
        m_neg_zero.elements[0] = -0.0;
        let mut m_pos_zero = Matrix4::identity();
        m_pos_zero.elements[0] = 0.0;
        assert!(m_neg_zero.equals(&m_pos_zero));

        // NaN element -> false (NaN !== NaN)
        let mut m_nan = Matrix4::identity();
        m_nan.elements[5] = f64::NAN;
        assert!(!m_nan.equals(&m_nan));
        assert!(!m_nan.equals(&Matrix4::identity()));

        // Node oracle 3: fromArray with offset 4 from a 20-element buffer
        // Buffer: [-1, -2, -3, -4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        let mut buf20 = [-1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..16 {
            buf20[4 + i] = (i + 1) as f64;
        }
        let mut m_from = Matrix4::zero();
        m_from.from_slice_offset(&buf20, 4);
        for i in 0..16 {
            assert_eq!(m_from.elements[i], (i + 1) as f64);
        }

        // Node oracle 4: toArray into a 20-element sentinel buffer at offset 2 preserving sentinels
        let mut dst20 = [0.0; 20];
        dst20[0] = 888.0;
        dst20[1] = 999.0;
        dst20[18] = 777.0;
        dst20[19] = 666.0;

        m_from.to_slice_offset(&mut dst20, 2);
        assert_eq!(dst20[0], 888.0);
        assert_eq!(dst20[1], 999.0);
        for i in 0..16 {
            assert_eq!(dst20[2 + i], (i + 1) as f64);
        }
        assert_eq!(dst20[18], 777.0);
        assert_eq!(dst20[19], 666.0);

        // Out-of-range bounds checks guarantee zero partial mutation in native slice domain
        let mut m_err = Matrix4::identity();
        let short_src = [1.0; 18];
        let res_read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m_err.from_slice_offset(&short_src, 5);
        }));
        assert!(res_read.is_err(), "from_slice_offset must panic on out-of-range read");
        assert_eq!(m_err, Matrix4::identity(), "matrix must have zero partial mutation on read failure");

        let mut short_dst = [10.0; 18];
        let res_write = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m_from.to_slice_offset(&mut short_dst, 5);
        }));
        assert!(res_write.is_err(), "to_slice_offset must panic on out-of-range write");
        assert_eq!(short_dst, [10.0; 18], "destination slice must have zero partial mutation on write failure");
    }
}
