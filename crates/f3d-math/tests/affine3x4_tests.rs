//! Conformance, differential, and regression tests for `Affine3x4` (§7.5, roa.1; Root #13036).
//!
//! Verifies analytical and numerical equivalence against upstream Three.js r186
//! `src/math/Matrix4.js` and `src/math/Vector3.js` (commit `148ef33ecb6d2502ff796d4554abd1549c95d519`).

use f3d_core::layout::{AffineRows, LayoutError};
use f3d_math::affine3x4::Affine3x4;
use f3d_math::matrix4::{CoordinateSystem, Matrix4};
use f3d_math::vector3::Vector3;

#[test]
fn test_affine3x4_identity_and_constructors() {
    let id = Affine3x4::identity();
    assert_eq!(id.r0, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(id.r1, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(id.r2, [0.0, 0.0, 1.0, 0.0]);

    let def = Affine3x4::default();
    assert_eq!(def, id, "default must equal identity");

    let zero = Affine3x4::zero();
    assert_eq!(zero.r0, [0.0; 4]);
    assert_eq!(zero.r1, [0.0; 4]);
    assert_eq!(zero.r2, [0.0; 4]);

    let col_major = id.to_column_major();
    assert_eq!(col_major, [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]);

    let m_id = id.to_matrix4();
    assert_eq!(m_id, Matrix4::identity());
}

#[test]
fn test_affine3x4_checked_conversion_and_non_affine_rejection() {
    // 1. Valid affine matrix converts cleanly
    let mut m = Matrix4::identity();
    m.elements[12] = 10.0;
    m.elements[13] = 20.0;
    m.elements[14] = 30.0;
    let aff = Affine3x4::from_matrix4(&m).expect("valid affine matrix must convert");
    assert_eq!(aff.translation(), [10.0, 20.0, 30.0]);

    // 2. Perspective element e[3] != 0 is rejected (including subnormal 1e-100)
    let mut non_affine_e3 = m;
    non_affine_e3.elements[3] = 1e-100;
    assert_eq!(
        Affine3x4::from_matrix4(&non_affine_e3),
        Err(LayoutError::NonAffineMatrix),
        "subnormal perspective in e[3] must be rejected"
    );

    // 3. Signed zero -0.0 in perspective entries (e[3], e[7], e[11]) must be rejected
    // as an intentional packing refusal (with Matrix4 fallback) so that the sign bit
    // is not silently lost on round-trip expansion to [0.0, 0.0, 0.0, 1.0].
    let mut neg_zero_e3 = m;
    neg_zero_e3.elements[3] = -0.0;
    assert_eq!(
        Affine3x4::from_matrix4(&neg_zero_e3),
        Err(LayoutError::NonAffineMatrix),
        "signed zero -0.0 in e[3] must be rejected (intentional packing refusal)"
    );

    let mut neg_zero_e7 = m;
    neg_zero_e7.elements[7] = -0.0;
    assert_eq!(
        Affine3x4::from_matrix4(&neg_zero_e7),
        Err(LayoutError::NonAffineMatrix),
        "signed zero -0.0 in e[7] must be rejected (intentional packing refusal)"
    );

    let mut neg_zero_e11 = m;
    neg_zero_e11.elements[11] = -0.0;
    assert_eq!(
        Affine3x4::from_matrix4(&neg_zero_e11),
        Err(LayoutError::NonAffineMatrix),
        "signed zero -0.0 in e[11] must be rejected (intentional packing refusal)"
    );

    // 4. Perspective elements e[7] and e[11] non-zero are rejected
    let mut non_affine_e7 = m;
    non_affine_e7.elements[7] = 0.001;
    assert_eq!(Affine3x4::from_matrix4(&non_affine_e7), Err(LayoutError::NonAffineMatrix));

    let mut non_affine_e11 = m;
    non_affine_e11.elements[11] = -1.0;
    assert_eq!(Affine3x4::from_matrix4(&non_affine_e11), Err(LayoutError::NonAffineMatrix));

    // 5. Scale element e[15] != 1.0 is rejected
    let mut non_affine_e15_zero = m;
    non_affine_e15_zero.elements[15] = 0.0;
    assert_eq!(Affine3x4::from_matrix4(&non_affine_e15_zero), Err(LayoutError::NonAffineMatrix));

    let mut non_affine_e15_two = m;
    non_affine_e15_two.elements[15] = 2.0;
    assert_eq!(Affine3x4::from_matrix4(&non_affine_e15_two), Err(LayoutError::NonAffineMatrix));

    // 6. Three.js perspective projection matrix is strictly rejected
    let mut persp = Matrix4::identity();
    persp.make_perspective(-1.0, 1.0, 1.0, -1.0, 0.1, 1000.0, CoordinateSystem::WebGPU, false);
    assert_eq!(
        Affine3x4::from_matrix4(&persp),
        Err(LayoutError::NonAffineMatrix),
        "perspective projection matrix must be rejected"
    );

    // 7. Three.js orthographic projection matrix is affine and must be accepted
    let mut ortho = Matrix4::identity();
    ortho.make_orthographic(-10.0, 10.0, 10.0, -10.0, 0.1, 100.0, CoordinateSystem::WebGPU, false);
    let ortho_aff = Affine3x4::from_matrix4(&ortho);
    assert!(ortho_aff.is_ok(), "orthographic projection matrix must convert cleanly");
}

#[test]
fn test_affine3x4_composition_and_multiplication_oracle() {
    // Independent Three.js r186 Node oracle cross-check:
    // Matrix A:
    //   [ 2,  0.5, 0,  10 ]
    //   [ 0, -1,   1,  20 ]
    //   [ 1,  0,   3,  30 ]
    //   [ 0,  0,   0,   1 ]
    let aff_a = Affine3x4::new(
        [2.0, 0.5, 0.0, 10.0],
        [0.0, -1.0, 1.0, 20.0],
        [1.0, 0.0, 3.0, 30.0],
    );

    // Matrix B:
    //   [ 1, 2, 0, -5 ]
    //   [ 0, 1, 1,  4 ]
    //   [ 0, 0, 1,  2 ]
    //   [ 0, 0, 0,  1 ]
    let aff_b = Affine3x4::new(
        [1.0, 2.0, 0.0, -5.0],
        [0.0, 1.0, 1.0, 4.0],
        [0.0, 0.0, 1.0, 2.0],
    );

    // Expected C = A * B analytical & Three.js Node oracle:
    //   Row 0: [ 2*1 + 0.5*0 + 0*0 + 10*0, 2*2 + 0.5*1 + 0*0 + 10*0, 2*0 + 0.5*1 + 0*1 + 10*0, 2*(-5) + 0.5*4 + 0*2 + 10*1 ]
    //        = [ 2.0, 4.5, 0.5, 2.0 ]
    //   Row 1: [ 0*1 + -1*0 + 1*0 + 20*0, 0*2 + -1*1 + 1*0 + 20*0, 0*0 + -1*1 + 1*1 + 20*0, 0*(-5) + -1*4 + 1*2 + 20*1 ]
    //        = [ 0.0, -1.0, 0.0, 18.0 ]
    //   Row 2: [ 1*1 + 0*0 + 3*0 + 30*0, 1*2 + 0*1 + 3*0 + 30*0, 1*0 + 0*1 + 3*1 + 30*0, 1*(-5) + 0*4 + 3*2 + 30*1 ]
    //        = [ 1.0, 2.0, 3.0, 31.0 ]
    let mut c = Affine3x4::zero();
    c.multiply_affines(&aff_a, &aff_b).expect("valid affine composition must succeed");

    assert_eq!(c.r0, [2.0, 4.5, 0.5, 2.0]);
    assert_eq!(c.r1, [0.0, -1.0, 0.0, 18.0]);
    assert_eq!(c.r2, [1.0, 2.0, 3.0, 31.0]);

    // Test multiply (self = self * b)
    let mut c_mul = aff_a;
    c_mul.multiply(&aff_b).expect("valid affine multiply must succeed");
    assert_eq!(c_mul, c);

    // Test premultiply (self = a * self)
    let mut c_pre = aff_b;
    c_pre.premultiply(&aff_a).expect("valid affine premultiply must succeed");
    assert_eq!(c_pre, c);

    // Test checked_multiply
    let c_checked = aff_a.checked_multiply(&aff_b).expect("checked_multiply must succeed");
    assert_eq!(c_checked, c);

    // Cross-check against Matrix4::multiply_matrices
    let m_a = aff_a.to_matrix4();
    let m_b = aff_b.to_matrix4();
    let mut m_c = Matrix4::zero();
    m_c.multiply_matrices(&m_a, &m_b);
    assert_eq!(c.to_matrix4(), m_c);

    // Check determinant:
    // det(A) = 2*(-1*3 - 1*0) - 0.5*(0*3 - 1*1) + 0 = 2*(-3) - 0.5*(-1) = -6.0 + 0.5 = -5.5
    assert_eq!(aff_a.determinant(), -5.5);
    assert_eq!(aff_b.determinant(), 1.0);
    assert_eq!(c.determinant(), -5.5);
    assert!(aff_a.has_negative_scale());
}

#[test]
fn test_affine3x4_multiply_affines_nonfinite_differential() {
    // In Three.js Matrix4.multiplyMatrices, if LHS translation entry (a14) is Infinity,
    // and RHS is finite (identity), the linear entries evaluate a_i3 * 0.0 = Infinity * 0.0 = NaN,
    // while row 3 evaluates 0.0*b_0j + ... + 1.0*b_3j = [0.0, 0.0, 0.0, 1.0], which is affine!
    // Affine3x4::multiply_affines must evaluate `+ a_i3 * 0.0` in linear entries to match!
    let inf_trans = Affine3x4::new(
        [1.0, 0.0, 0.0, f64::INFINITY],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    );
    let identity = Affine3x4::identity();

    let mut result = Affine3x4::zero();
    result.multiply_affines(&inf_trans, &identity).expect("LHS non-finite with finite RHS remains affine");

    // Linear entry r0[0] must be NaN matching Three.js
    assert!(result.r0[0].is_nan(), "Infinity translation * 0.0 must produce NaN in linear entry matching Three.js");
    assert!(result.r0[1].is_nan());
    assert!(result.r0[2].is_nan());
    assert!(result.r0[3].is_infinite());

    // Cross-check against Matrix4::multiply_matrices
    let m_inf = inf_trans.to_matrix4();
    let m_id = identity.to_matrix4();
    let mut m_res = Matrix4::zero();
    m_res.multiply_matrices(&m_inf, &m_id);
    assert!(m_res.elements[0].is_nan(), "Matrix4::multiply_matrices must also yield NaN");
    assert_eq!(m_res.elements[3], 0.0);
    assert_eq!(m_res.elements[7], 0.0);
    assert_eq!(m_res.elements[11], 0.0);
    assert_eq!(m_res.elements[15], 1.0);
}

#[test]
fn test_affine3x4_composition_rhs_nonfinite_refusal_preserves_receiver() {
    let original = Affine3x4::new(
        [2.0, 0.5, 0.0, 10.0],
        [0.0, -1.0, 1.0, 20.0],
        [1.0, 0.0, 3.0, 30.0],
    );

    // 1. RHS with Infinity in translation (b.r0[3]):
    // In Three.js Matrix4.multiplyMatrices(a, b):
    // row 3 element [15] evaluates ae[3]*be[12] + ae[7]*be[13] + ae[11]*be[14] + ae[15]*be[15]
    // = 0.0 * Inf + 0.0 * 0 + 0.0 * 0 + 1.0 * 1.0 = NaN + 1.0 = NaN.
    // The product is non-affine (row 3 is unrepresentable).
    let mut rhs_inf_trans = Affine3x4::identity();
    rhs_inf_trans.r0[3] = f64::INFINITY;

    let mut receiver = original;
    let res = receiver.multiply(&rhs_inf_trans);
    assert_eq!(
        res,
        Err(LayoutError::NonAffineMatrix),
        "multiply by RHS with Infinity translation must be refused as NonAffineMatrix"
    );
    assert_eq!(
        receiver, original,
        "receiver must remain completely unmutated on composition refusal"
    );

    // Cross-check against Three.js Matrix4::multiply_matrices:
    let m_a = original.to_matrix4();
    let m_b = rhs_inf_trans.to_matrix4();
    let mut m_c = Matrix4::zero();
    m_c.multiply_matrices(&m_a, &m_b);
    assert!(m_c.elements[15].is_nan(), "full Matrix4 product element[15] must be NaN in Three.js");
    assert!(!m_c.is_affine(), "full Matrix4 product is not affine");

    // 2. RHS with Infinity in linear block (b.r0[0]):
    // In Three.js: row 3 element [3] evaluates ae[3]*be[0] + ... = 0.0 * Inf = NaN.
    let mut rhs_inf_linear = Affine3x4::identity();
    rhs_inf_linear.r0[0] = f64::INFINITY;

    let mut receiver2 = original;
    let res2 = receiver2.multiply(&rhs_inf_linear);
    assert_eq!(
        res2,
        Err(LayoutError::NonAffineMatrix),
        "multiply by RHS with Infinity linear component must be refused as NonAffineMatrix"
    );
    assert_eq!(
        receiver2, original,
        "receiver must remain completely unmutated on composition refusal"
    );

    let m_b_lin = rhs_inf_linear.to_matrix4();
    let mut m_c_lin = Matrix4::zero();
    m_c_lin.multiply_matrices(&m_a, &m_b_lin);
    assert!(m_c_lin.elements[3].is_nan(), "full Matrix4 product element[3] must be NaN in Three.js");
    assert!(!m_c_lin.is_affine(), "full Matrix4 product is not affine");

    // 3. RHS with NaN in linear block:
    let mut rhs_nan = Affine3x4::identity();
    rhs_nan.r1[2] = f64::NAN;

    let mut receiver3 = original;
    let res3 = receiver3.multiply(&rhs_nan);
    assert_eq!(
        res3,
        Err(LayoutError::NonAffineMatrix),
        "multiply by RHS with NaN component must be refused as NonAffineMatrix"
    );
    assert_eq!(
        receiver3, original,
        "receiver must remain completely unmutated on composition refusal"
    );

    // 4. Test multiply_affines with separate destination preserving destination on refusal:
    let mut dest = Affine3x4::new(
        [9.0, 8.0, 7.0, 6.0],
        [5.0, 4.0, 3.0, 2.0],
        [1.0, 0.0, -1.0, -2.0],
    );
    let dest_copy = dest;
    let res4 = dest.multiply_affines(&original, &rhs_inf_trans);
    assert_eq!(res4, Err(LayoutError::NonAffineMatrix));
    assert_eq!(dest, dest_copy, "destination matrix must not be modified when multiply_affines fails");

    // 5. Non-mutating checked_multiply also returns Err:
    assert_eq!(original.checked_multiply(&rhs_inf_trans), Err(LayoutError::NonAffineMatrix));
}

#[test]
fn test_affine3x4_point_and_vector_transformation_oracle() {
    let aff = Affine3x4::new(
        [2.0, 4.5, 0.5, 2.0],
        [0.0, -1.0, 0.0, 18.0],
        [1.0, 2.0, 3.0, 31.0],
    );

    let p = [1.0, 2.0, 3.0];
    let p_prime = aff.transform_point(p);
    // x' = 2*1 + 4.5*2 + 0.5*3 + 2  = 2 + 9 + 1.5 + 2 = 14.5
    // y' = 0*1 + -1*2 + 0*3 + 18   = -2 + 18          = 16.0
    // z' = 1*1 + 2*2 + 3*3 + 31    = 1 + 4 + 9 + 31   = 45.0
    assert_eq!(p_prime, [14.5, 16.0, 45.0]);

    let v3_p = Vector3::new(1.0, 2.0, 3.0);
    let v3_p_prime = aff.transform_point_vector3(&v3_p);
    assert_eq!(v3_p_prime, Vector3::new(14.5, 16.0, 45.0));

    // Direction vector transformation (ignores translation):
    let v = [1.0, 2.0, 3.0];
    let v_prime = aff.transform_vector(v);
    // x' = 2*1 + 4.5*2 + 0.5*3 = 12.5
    // y' = 0*1 + -1*2 + 0*3    = -2.0
    // z' = 1*1 + 2*2 + 3*3     = 14.0
    assert_eq!(v_prime, [12.5, -2.0, 14.0]);

    let v3_v = Vector3::new(1.0, 2.0, 3.0);
    let v3_v_prime = aff.transform_vector_vector3(&v3_v);
    assert_eq!(v3_v_prime, Vector3::new(12.5, -2.0, 14.0));

    // Three.js homogeneous denominator differential check:
    // In Vector3.applyMatrix4: w = 1 / (e3*x + e7*y + e11*z + e15).
    // For identity with x = Infinity: 0.0 * Infinity + 1.0 = NaN, w = NaN, result = NaN!
    let id = Affine3x4::identity();
    let inf_pt = id.transform_point([f64::INFINITY, 0.0, 0.0]);
    assert!(inf_pt[0].is_nan(), "point transform on Infinity coordinate must yield NaN matching Three.js applyMatrix4");
    assert!(inf_pt[1].is_nan());
    assert!(inf_pt[2].is_nan());
}

#[test]
fn test_affine3x4_negative_scale_and_shear() {
    // Negative scale matrix (reflection across X axis + scaling)
    let neg_scale = Affine3x4::new(
        [-1.0, 0.0, 0.0, 5.0],
        [0.0, 2.0, 0.0, -3.0],
        [0.0, 0.0, 3.0, 1.0],
    );
    assert!(neg_scale.has_negative_scale());
    assert_eq!(neg_scale.determinant(), -6.0);

    let p = [10.0, 20.0, 30.0];
    let p_transformed = neg_scale.transform_point(p);
    assert_eq!(p_transformed, [-5.0, 37.0, 91.0]);

    // Shear matrix (xy shear = 0.5, xz shear = 0.25)
    let shear = Affine3x4::new(
        [1.0, 0.5, 0.25, 4.0],
        [0.0, 1.0, 0.0, -2.0],
        [0.0, 0.0, 1.0, 8.0],
    );
    assert_eq!(shear.determinant(), 1.0);
    assert!(!shear.has_negative_scale());

    let p_shear = shear.transform_point([2.0, 4.0, 8.0]);
    // x' = 1*2 + 0.5*4 + 0.25*8 + 4 = 2 + 2 + 2 + 4 = 10.0
    // y' = 0*2 + 1*4 + 0*8 - 2      = 4 - 2         = 2.0
    // z' = 0*2 + 0*4 + 1*8 + 8      = 8 + 8         = 16.0
    assert_eq!(p_shear, [10.0, 2.0, 16.0]);
}

#[test]
fn test_affine3x4_to_affine_rows_gpu_boundary() {
    let aff = Affine3x4::new(
        [1.5, 0.25, -2.125, 100.5],
        [0.0, 3.0, 0.5, -50.25],
        [-1.0, 2.0, 4.0, 10.0],
    );

    // Direct conversion to AffineRows (f32)
    let rows: AffineRows = aff.to_affine_rows();
    assert_eq!(rows.r0, [1.5_f32, 0.25_f32, -2.125_f32, 100.5_f32]);
    assert_eq!(rows.r1, [0.0_f32, 3.0_f32, 0.5_f32, -50.25_f32]);
    assert_eq!(rows.r2, [-1.0_f32, 2.0_f32, 4.0_f32, 10.0_f32]);

    // Reconstruct Affine3x4 from AffineRows
    let reconstructed = Affine3x4::from_affine_rows(&rows);
    assert_eq!(reconstructed, aff);
}
