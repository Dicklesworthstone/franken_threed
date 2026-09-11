//! Conformance and regression tests for `Vector4` (§5.1, §6.7; Root #7864).
//!
//! Verifies analytical and numerical equivalence against upstream Three.js r186
//! `src/math/Vector4.js` (commit `148ef33ecb6d2502ff796d4554abd1549c95d519`).

use f3d_math::matrix4::Matrix4;
use f3d_math::narrowing::NarrowingTolerance;
use f3d_math::quaternion::Quaternion;
use f3d_math::vector4::Vector4;

#[test]
fn test_vector4_default_and_constructors() {
    // Three.js r186: constructor(x=0, y=0, z=0, w=1)
    let def = Vector4::default();
    assert_eq!(def.x, 0.0, "default x must be 0.0");
    assert_eq!(def.y, 0.0, "default y must be 0.0");
    assert_eq!(def.z, 0.0, "default z must be 0.0");
    assert_eq!(def.w, 1.0, "default w must be 1.0 matching Three.js r186");

    let zero = Vector4::zero();
    assert_eq!(zero.x, 0.0);
    assert_eq!(zero.y, 0.0);
    assert_eq!(zero.z, 0.0);
    assert_eq!(zero.w, 0.0);

    let one = Vector4::one();
    assert_eq!(one.x, 1.0);
    assert_eq!(one.y, 1.0);
    assert_eq!(one.z, 1.0);
    assert_eq!(one.w, 1.0);

    let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
    assert_eq!(v.w, 4.0);
}

#[test]
fn test_vector4_set_and_component_access() {
    let mut v = Vector4::zero();
    v.set(5.0, 6.0, 7.0, 8.0);
    assert_eq!(v, Vector4::new(5.0, 6.0, 7.0, 8.0));

    v.set_scalar(42.0);
    assert_eq!(v, Vector4::new(42.0, 42.0, 42.0, 42.0));

    v.set_x(1.0).set_y(2.0).set_z(3.0).set_w(4.0);
    assert_eq!(v, Vector4::new(1.0, 2.0, 3.0, 4.0));

    // width and height getters/setters (aliases for z and w)
    assert_eq!(v.width(), 3.0);
    assert_eq!(v.height(), 4.0);
    v.set_width(30.0);
    v.set_height(40.0);
    assert_eq!(v.z, 30.0);
    assert_eq!(v.w, 40.0);

    // Component indexing
    assert_eq!(v.get_component(0), 1.0);
    assert_eq!(v.get_component(1), 2.0);
    assert_eq!(v.get_component(2), 30.0);
    assert_eq!(v.get_component(3), 40.0);

    v.set_component(0, 100.0);
    v.set_component(1, 200.0);
    v.set_component(2, 300.0);
    v.set_component(3, 400.0);
    assert_eq!(v[0], 100.0);
    assert_eq!(v[1], 200.0);
    assert_eq!(v[2], 300.0);
    assert_eq!(v[3], 400.0);

    v[0] = -1.0;
    v[1] = -2.0;
    v[2] = -3.0;
    v[3] = -4.0;
    assert_eq!(v, Vector4::new(-1.0, -2.0, -3.0, -4.0));
}

#[test]
fn test_vector4_basic_arithmetic() {
    let mut a = Vector4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vector4::new(10.0, 20.0, 30.0, 40.0);

    a.add(&b);
    assert_eq!(a, Vector4::new(11.0, 22.0, 33.0, 44.0));

    a.add_scalar(1.0);
    assert_eq!(a, Vector4::new(12.0, 23.0, 34.0, 45.0));

    let mut c = Vector4::zero();
    c.add_vectors(
        &Vector4::new(1.0, 2.0, 3.0, 4.0),
        &Vector4::new(5.0, 6.0, 7.0, 8.0),
    );
    assert_eq!(c, Vector4::new(6.0, 8.0, 10.0, 12.0));

    let mut d = Vector4::new(1.0, 1.0, 1.0, 1.0);
    d.add_scaled_vector(&Vector4::new(2.0, 3.0, 4.0, 5.0), 3.0);
    assert_eq!(d, Vector4::new(7.0, 10.0, 13.0, 16.0));

    let mut e = Vector4::new(10.0, 20.0, 30.0, 40.0);
    e.sub(&Vector4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(e, Vector4::new(9.0, 18.0, 27.0, 36.0));

    e.sub_scalar(5.0);
    assert_eq!(e, Vector4::new(4.0, 13.0, 22.0, 31.0));

    let mut f = Vector4::zero();
    f.sub_vectors(
        &Vector4::new(10.0, 20.0, 30.0, 40.0),
        &Vector4::new(1.0, 2.0, 3.0, 4.0),
    );
    assert_eq!(f, Vector4::new(9.0, 18.0, 27.0, 36.0));

    let mut g = Vector4::new(2.0, 3.0, 4.0, 5.0);
    g.multiply(&Vector4::new(3.0, 4.0, 5.0, 6.0));
    assert_eq!(g, Vector4::new(6.0, 12.0, 20.0, 30.0));

    g.multiply_scalar(2.0);
    assert_eq!(g, Vector4::new(12.0, 24.0, 40.0, 60.0));

    let mut h = Vector4::new(12.0, 24.0, 40.0, 60.0);
    h.divide(&Vector4::new(2.0, 3.0, 4.0, 5.0));
    assert_eq!(h, Vector4::new(6.0, 8.0, 10.0, 12.0));

    h.divide_scalar(2.0);
    assert_eq!(h, Vector4::new(3.0, 4.0, 5.0, 6.0));

    let mut neg = Vector4::new(1.0, -2.0, 3.5, -4.5);
    neg.negate();
    assert_eq!(neg, Vector4::new(-1.0, 2.0, -3.5, 4.5));
}

#[test]
fn test_vector4_apply_matrix4_projective_analytical_oracle() {
    // Pinned Three.js r186 Vector4.applyMatrix4 analytical & oracle test:
    // Matrix:
    //   [ 2    0    1    5 ]
    //   [ 0    3   -1    2 ]
    //   [ 1   -2    4   -3 ]
    //   [ 0.5  1   -0.5  2 ]
    // Vector: [ 1, 2, 3, 4 ]^T
    //
    // Analytical computation:
    //   x' = 2*1 + 0*2 + 1*3 + 5*4       = 2 + 0 + 3 + 20     = 25
    //   y' = 0*1 + 3*2 + -1*3 + 2*4      = 0 + 6 - 3 + 8      = 11
    //   z' = 1*1 + -2*2 + 4*3 + -3*4     = 1 - 4 + 12 - 12    = -3
    //   w' = 0.5*1 + 1*2 + -0.5*3 + 2*4  = 0.5 + 2 - 1.5 + 8  = 9
    // Node oracle:
    //   const m = new Matrix4().set(2,0,1,5, 0,3,-1,2, 1,-2,4,-3, 0.5,1,-0.5,2);
    //   const v = new Vector4(1,2,3,4).applyMatrix4(m);
    //   -> (25, 11, -3, 9)
    let mut m = Matrix4::zero();
    m.set(
        2.0, 0.0, 1.0, 5.0, 0.0, 3.0, -1.0, 2.0, 1.0, -2.0, 4.0, -3.0, 0.5, 1.0, -0.5, 2.0,
    );

    let mut v = Vector4::new(1.0, 2.0, 3.0, 4.0);
    v.apply_matrix4(&m);

    assert_eq!(v.x, 25.0, "applyMatrix4 projective x mismatch");
    assert_eq!(v.y, 11.0, "applyMatrix4 projective y mismatch");
    assert_eq!(v.z, -3.0, "applyMatrix4 projective z mismatch");
    assert_eq!(v.w, 9.0, "applyMatrix4 projective w mismatch");

    // setFromMatrixPosition
    let mut pos = Vector4::zero();
    pos.set_from_matrix_position(&m);
    assert_eq!(pos, Vector4::new(5.0, 2.0, -3.0, 2.0));
}

#[test]
fn test_vector4_dot_length_and_normalize() {
    let a = Vector4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vector4::new(2.0, -1.0, 4.0, -2.0);
    // dot: 1*2 + 2*(-1) + 3*4 + 4*(-2) = 2 - 2 + 12 - 8 = 4
    assert_eq!(a.dot(&b), 4.0);

    // Vector with 3-4-5-like integer length: (1, 2, 2, 4) -> 1 + 4 + 4 + 16 = 25 -> length = 5
    let v = Vector4::new(1.0, 2.0, 2.0, 4.0);
    assert_eq!(v.length_sq(), 25.0);
    assert_eq!(v.length(), 5.0);
    assert_eq!(v.manhattan_length(), 9.0);

    let mut norm = v;
    norm.normalize();
    assert_eq!(norm.x, 0.2);
    assert_eq!(norm.y, 0.4);
    assert_eq!(norm.z, 0.4);
    assert_eq!(norm.w, 0.8);
    assert!((norm.length() - 1.0).abs() < 1e-15);

    let mut scaled = v;
    scaled.set_length(10.0);
    assert_eq!(scaled.x, 2.0);
    assert_eq!(scaled.y, 4.0);
    assert_eq!(scaled.z, 4.0);
    assert_eq!(scaled.w, 8.0);
    assert_eq!(scaled.length(), 10.0);
}

#[test]
fn test_vector4_min_max_clamp() {
    let mut a = Vector4::new(-5.0, 10.0, 3.0, 0.0);
    let min_bound = Vector4::new(-2.0, 0.0, 1.0, -1.0);
    let max_bound = Vector4::new(2.0, 5.0, 4.0, 1.0);

    a.clamp(&min_bound, &max_bound);
    assert_eq!(a, Vector4::new(-2.0, 5.0, 3.0, 0.0));

    let mut b = Vector4::new(-10.0, 20.0, 5.0, -2.0);
    b.clamp_scalar(-1.0, 10.0);
    assert_eq!(b, Vector4::new(-1.0, 10.0, 5.0, -1.0));

    // clamp_length on (1, 2, 2, 4) whose length is 5.0 clamped to [2.0, 4.0]
    let mut len_v = Vector4::new(1.0, 2.0, 2.0, 4.0);
    len_v.clamp_length(2.0, 4.0);
    assert_eq!(len_v.x, 0.8);
    assert_eq!(len_v.y, 1.6);
    assert_eq!(len_v.z, 1.6);
    assert_eq!(len_v.w, 3.2);
    assert_eq!(len_v.length(), 4.0);

    // min / max
    let mut m = Vector4::new(1.0, 5.0, -2.0, 10.0);
    m.min(&Vector4::new(2.0, 3.0, 0.0, 12.0));
    assert_eq!(m, Vector4::new(1.0, 3.0, -2.0, 10.0));

    m.max(&Vector4::new(0.0, 4.0, -1.0, 9.0));
    assert_eq!(m, Vector4::new(1.0, 4.0, -1.0, 10.0));
}

#[test]
fn test_vector4_floor_ceil_round_trunc() {
    let base = Vector4::new(-1.7, 2.5, -0.4, 0.4);

    let mut v_floor = base;
    v_floor.floor();
    assert_eq!(v_floor.x, -2.0);
    assert_eq!(v_floor.y, 2.0);
    assert_eq!(v_floor.z, -1.0);
    assert_eq!(v_floor.w, 0.0);

    let mut v_ceil = base;
    v_ceil.ceil();
    assert_eq!(v_ceil.x, -1.0);
    assert_eq!(v_ceil.y, 3.0);
    assert_eq!(v_ceil.z, -0.0);
    assert!(
        v_ceil.z.is_sign_negative(),
        "ceil(-0.4) must preserve -0.0 matching ECMAScript"
    );
    assert_eq!(v_ceil.w, 1.0);

    let mut v_round = base;
    v_round.round();
    assert_eq!(v_round.x, -2.0);
    assert_eq!(v_round.y, 3.0);
    assert_eq!(v_round.z, -0.0);
    assert!(
        v_round.z.is_sign_negative(),
        "round(-0.4) must preserve -0.0 matching ECMAScript"
    );
    assert_eq!(v_round.w, 0.0);

    let mut v_trunc = base;
    v_trunc.round_to_zero();
    assert_eq!(v_trunc.x, -1.0);
    assert_eq!(v_trunc.y, 2.0);
    assert_eq!(v_trunc.z, -0.0);
    assert!(
        v_trunc.z.is_sign_negative(),
        "trunc(-0.4) must preserve -0.0 matching ECMAScript"
    );
    assert_eq!(v_trunc.w, 0.0);
}

#[test]
fn test_vector4_lerp() {
    let mut a = Vector4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vector4::new(5.0, 6.0, 7.0, 8.0);
    a.lerp(&b, 0.25);
    assert_eq!(a, Vector4::new(2.0, 3.0, 4.0, 5.0));

    let mut c = Vector4::zero();
    c.lerp_vectors(
        &Vector4::new(10.0, 20.0, 30.0, 40.0),
        &Vector4::new(30.0, 60.0, 90.0, 120.0),
        0.5,
    );
    assert_eq!(c, Vector4::new(20.0, 40.0, 60.0, 80.0));
}

#[test]
fn test_vector4_array_conversions_and_display() {
    let v = Vector4::new(1.5, 2.5, 3.5, 4.5);
    let arr = v.to_array();
    assert_eq!(arr, [1.5, 2.5, 3.5, 4.5]);

    let from = Vector4::from_array([10.0, 20.0, 30.0, 40.0]);
    assert_eq!(from, Vector4::new(10.0, 20.0, 30.0, 40.0));

    let slice = [0.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0];
    let mut v_slice = Vector4::zero();
    v_slice.from_slice_offset(&slice, 2);
    assert_eq!(v_slice, Vector4::new(7.0, 8.0, 9.0, 10.0));

    let mut out_slice = [0.0; 6];
    v_slice.to_slice_offset(&mut out_slice, 1);
    assert_eq!(out_slice, [0.0, 7.0, 8.0, 9.0, 10.0, 0.0]);

    #[cfg(feature = "std")]
    {
        let disp = format!("{}", v);
        assert_eq!(disp, "Vector4(1.5, 2.5, 3.5, 4.5)");
    }
}

#[test]
fn test_vector4_narrowing() {
    let v = Vector4::new(1.0, 2.5, -4.25, 0.125);
    let narrowed = v
        .to_f32_checked(1e-7, 1e-7)
        .expect("exact dyadic narrowing must succeed");
    assert_eq!(narrowed, [1.0_f32, 2.5_f32, -4.25_f32, 0.125_f32]);

    let non_finite = Vector4::new(f64::NAN, 1.0, 2.0, 3.0);
    assert!(
        non_finite.to_f32_strict(1e-4, 1e-4).is_err(),
        "strict narrowing must reject NaN"
    );

    let precise = Vector4::new(1.0000000001, 2.0, 3.0, 4.0);
    assert!(
        precise
            .to_f32_with_tolerance(NarrowingTolerance::EXACT, false)
            .is_err(),
        "exact tolerance must reject inexact precision"
    );
}

#[test]
fn test_vector4_negatives_and_edge_cases() {
    // 1. Zero vector normalize: divide by 1.0 (length || 1), no panic, no NaN
    let mut zero_vec = Vector4::zero();
    zero_vec.normalize();
    assert_eq!(zero_vec, Vector4::zero(), "zero normalize must remain zero");

    // 2. Signed zero preservation under normalize: (-0.0, 0.0, -0.0, 0.0) -> (-0.0, 0.0, -0.0, 0.0)
    let mut neg_zero = Vector4::new(-0.0, 0.0, -0.0, 0.0);
    neg_zero.normalize();
    assert!(
        neg_zero.x.is_sign_negative(),
        "normalize must preserve -0.0 sign on x"
    );
    assert!(
        !neg_zero.y.is_sign_negative(),
        "normalize must preserve +0.0 sign on y"
    );
    assert!(
        neg_zero.z.is_sign_negative(),
        "normalize must preserve -0.0 sign on z"
    );
    assert!(
        !neg_zero.w.is_sign_negative(),
        "normalize must preserve +0.0 sign on w"
    );

    // 3. ECMAScript signed zero min / max
    let mut signed_zero = Vector4::new(0.0, -0.0, 5.0, 5.0);
    signed_zero.min(&Vector4::new(-0.0, 0.0, 2.0, 8.0));
    assert!(
        signed_zero.x.is_sign_negative(),
        "js_min(0, -0) must yield -0.0"
    );
    assert!(
        signed_zero.y.is_sign_negative(),
        "js_min(-0, 0) must yield -0.0"
    );
    assert_eq!(signed_zero.z, 2.0);

    let mut signed_max = Vector4::new(0.0, -0.0, 1.0, 1.0);
    signed_max.max(&Vector4::new(-0.0, 0.0, 2.0, 0.0));
    assert!(
        !signed_max.x.is_sign_negative(),
        "js_max(0, -0) must yield +0.0"
    );
    assert!(
        !signed_max.y.is_sign_negative(),
        "js_max(-0, 0) must yield +0.0"
    );
    assert_eq!(signed_max.z, 2.0);

    // 4. NaN propagation in min / max matching JS
    let mut nan_v = Vector4::new(f64::NAN, 1.0, 2.0, 3.0);
    nan_v.min(&Vector4::new(0.0, 0.0, 0.0, 0.0));
    assert!(nan_v.x.is_nan(), "min with NaN must yield NaN");

    let mut nan_max = Vector4::new(f64::NAN, 1.0, 2.0, 3.0);
    nan_max.max(&Vector4::new(0.0, 0.0, 0.0, 0.0));
    assert!(nan_max.x.is_nan(), "max with NaN must yield NaN");

    // 5. clamp_length with zero vector
    let mut clamp_zero = Vector4::zero();
    clamp_zero.clamp_length(1.0, 5.0);
    assert_eq!(clamp_zero.x, 0.0);
    assert_eq!(clamp_zero.y, 0.0);
    assert_eq!(clamp_zero.z, 0.0);
    assert_eq!(clamp_zero.w, 0.0);
}

#[test]
#[should_panic(expected = "THREE.Vector4: index is out of range: 4")]
fn test_vector4_get_component_out_of_bounds() {
    let v = Vector4::zero();
    let _ = v.get_component(4);
}

#[test]
#[should_panic(expected = "THREE.Vector4: index is out of range: 4")]
fn test_vector4_set_component_out_of_bounds() {
    let mut v = Vector4::zero();
    v.set_component(4, 1.0);
}

const EPS_AXIS: f64 = 1e-10;

fn assert_vec4_close(v: &Vector4, expected: [f64; 4], eps: f64, msg: &str) {
    let diff_x = (v.x - expected[0]).abs();
    let diff_y = (v.y - expected[1]).abs();
    let diff_z = (v.z - expected[2]).abs();
    let diff_w = (v.w - expected[3]).abs();
    assert!(
        diff_x <= eps && diff_y <= eps && diff_z <= eps && diff_w <= eps,
        "{msg}: expected {:?}, got {:?} (diffs: [{}, {}, {}, {}] > {eps})",
        expected,
        v.to_array(),
        diff_x,
        diff_y,
        diff_z,
        diff_w,
    );
}

#[test]
fn test_vector4_set_axis_angle_from_quaternion() {
    // 1. Identity quaternion (0, 0, 0, 1) -> [1.0, 0.0, 0.0, 0.0]
    let mut v = Vector4::zero();
    v.set_axis_angle_from_quaternion(&Quaternion::identity());
    assert_eq!(v, Vector4::new(1.0, 0.0, 0.0, 0.0), "identity quaternion");

    // 2. 90 deg around Y: (0, sin(PI/4), 0, cos(PI/4))
    // Node oracle: [0.0, 1.0, 0.0, 1.5707963267948966]
    let half_pi = core::f64::consts::FRAC_PI_4;
    let q_y_90 = Quaternion::new(0.0, half_pi.sin(), 0.0, half_pi.cos());
    v.set_axis_angle_from_quaternion(&q_y_90);
    assert_vec4_close(
        &v,
        [0.0, 1.0, 0.0, core::f64::consts::FRAC_PI_2],
        EPS_AXIS,
        "90 deg around Y",
    );

    // 3. 180 deg around Z: (0, 0, 1, 0)
    // Node oracle: [0.0, 0.0, 1.0, 3.141592653589793]
    let q_z_180 = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    v.set_axis_angle_from_quaternion(&q_z_180);
    assert_vec4_close(
        &v,
        [0.0, 0.0, 1.0, core::f64::consts::PI],
        EPS_AXIS,
        "180 deg around Z",
    );

    // 4. Threshold-adjacent below (s < 0.0001):
    // qw = sqrt(1 - 0.99e-8), s approx 0.0000994987 < 0.0001
    // Node oracle: [1.0, 0.0, 0.0, 0.00019899748904107667]
    let qw_below = (1.0 - 0.99e-8_f64).sqrt();
    let q_below = Quaternion::new(0.0, 0.0000994987, 0.0, qw_below);
    v.set_axis_angle_from_quaternion(&q_below);
    assert_vec4_close(
        &v,
        [1.0, 0.0, 0.0, 0.00019899748904107667],
        EPS_AXIS,
        "quaternion threshold-adjacent s < 0.0001",
    );

    // 5. Threshold-adjacent above (s > 0.0001):
    // qw = sqrt(1 - 1.01e-8), s approx 0.000100498756 > 0.0001
    // Node oracle: [0.0, 1.000000000497436, 0.0, 0.00020099751198460303]
    let qw_above = (1.0 - 1.01e-8_f64).sqrt();
    let q_above = Quaternion::new(0.0, 0.000100498756, 0.0, qw_above);
    v.set_axis_angle_from_quaternion(&q_above);
    assert_vec4_close(
        &v,
        [0.0, 1.000000000497436, 0.0, 0.00020099751198460303],
        EPS_AXIS,
        "quaternion threshold-adjacent s > 0.0001",
    );

    // 6. Non-unit / out-of-domain quaternion (q.w = 2.0):
    // Node oracle: [NaN, NaN, NaN, NaN]
    let q_out = Quaternion::new(0.0, 0.0, 0.0, 2.0);
    v.set_axis_angle_from_quaternion(&q_out);
    assert!(v.x.is_nan(), "out-of-domain q.w=2 must yield NaN on x");
    assert!(v.y.is_nan(), "out-of-domain q.w=2 must yield NaN on y");
    assert!(v.z.is_nan(), "out-of-domain q.w=2 must yield NaN on z");
    assert!(v.w.is_nan(), "out-of-domain q.w=2 must yield NaN on w");

    // 7. Opposite orientation (q.w = -1.0):
    // Node oracle: [1.0, 0.0, 0.0, 2 * PI]
    let q_neg = Quaternion::new(0.0, 0.0, 0.0, -1.0);
    v.set_axis_angle_from_quaternion(&q_neg);
    assert_vec4_close(
        &v,
        [1.0, 0.0, 0.0, 2.0 * core::f64::consts::PI],
        EPS_AXIS,
        "quaternion q.w = -1.0",
    );
}

#[test]
fn test_vector4_set_axis_angle_from_rotation_matrix() {
    let mut v = Vector4::zero();

    // 1. Identity matrix -> [1.0, 0.0, 0.0, 0.0]
    let m_ident = Matrix4::identity();
    v.set_axis_angle_from_rotation_matrix(&m_ident);
    assert_eq!(v, Vector4::new(1.0, 0.0, 0.0, 0.0), "matrix identity");

    // 2. 180-deg rotation: diagonal branch 1 (xx > yy && xx > zz, xx >= epsilon)
    // 180 deg around X: diag [1, -1, -1] -> [1.0, 0.0, 0.0, PI]
    let m_pi_x = Matrix4::from_elements([
        1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_pi_x);
    assert_vec4_close(
        &v,
        [1.0, 0.0, 0.0, core::f64::consts::PI],
        EPS_AXIS,
        "180 deg around X (diag branch 1)",
    );

    // 3. 180-deg rotation: diagonal branch 2 (yy > zz, yy >= epsilon)
    // 180 deg around Y: diag [-1, 1, -1] -> [0.0, 1.0, 0.0, PI]
    let m_pi_y = Matrix4::from_elements([
        -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_pi_y);
    assert_vec4_close(
        &v,
        [0.0, 1.0, 0.0, core::f64::consts::PI],
        EPS_AXIS,
        "180 deg around Y (diag branch 2)",
    );

    // 4. 180-deg rotation: diagonal branch 3 (else, zz >= epsilon)
    // 180 deg around Z: diag [-1, -1, 1] -> [0.0, 0.0, 1.0, PI]
    let m_pi_z = Matrix4::from_elements([
        -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_pi_z);
    assert_vec4_close(
        &v,
        [0.0, 0.0, 1.0, core::f64::consts::PI],
        EPS_AXIS,
        "180 deg around Z (diag branch 3)",
    );

    // 5. 180-deg rotation: diagonal fallback sub-branches (diag < epsilon -> 0.707106781)
    // Branch 5a: xx < epsilon
    // Node oracle: [0.0, 0.707106781, 0.707106781, 3.141592653589793]
    let m_xx_eps = Matrix4::from_elements([
        -0.99, 0.0, 0.0, 0.0, 0.0, -0.995, 0.0, 0.0, 0.0, 0.0, -0.995, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_xx_eps);
    assert_vec4_close(
        &v,
        [0.0, 0.707106781, 0.707106781, core::f64::consts::PI],
        EPS_AXIS,
        "180 deg xx < epsilon fallback",
    );

    // Branch 5b: yy < epsilon
    // Node oracle: [0.707106781, 0.0, 0.707106781, 3.141592653589793]
    let m_yy_eps = Matrix4::from_elements([
        -0.995, 0.0, 0.0, 0.0, 0.0, -0.99, 0.0, 0.0, 0.0, 0.0, -0.995, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_yy_eps);
    assert_vec4_close(
        &v,
        [0.707106781, 0.0, 0.707106781, core::f64::consts::PI],
        EPS_AXIS,
        "180 deg yy < epsilon fallback",
    );

    // Branch 5c: zz < epsilon
    // Node oracle: [0.707106781, 0.707106781, 0.0, 3.141592653589793]
    let m_zz_eps = Matrix4::from_elements([
        -0.995, 0.0, 0.0, 0.0, 0.0, -0.995, 0.0, 0.0, 0.0, 0.0, -0.99, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_zz_eps);
    assert_vec4_close(
        &v,
        [0.707106781, 0.707106781, 0.0, core::f64::consts::PI],
        EPS_AXIS,
        "180 deg zz < epsilon fallback",
    );

    // 6. General rotation: axis = (1, 2, 3).normalize(), angle = PI / 4
    // Node oracle: [0.26726124191242434, 0.5345224838248487, 0.8017837257372731, 0.7853981633974484]
    let m_gen = Matrix4::from_elements([
        0.7280277253875085,
        0.6087885979157627,
        -0.3152016404063445,
        0.0,
        -0.525104821111919,
        0.7907905579903911,
        0.3145079017103789,
        0.0,
        0.4407273056121099,
        -0.06345657129884827,
        0.8953952789951956,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_gen);
    assert_vec4_close(
        &v,
        [
            0.26726124191242434,
            0.5345224838248487,
            0.8017837257372731,
            0.7853981633974484,
        ],
        EPS_AXIS,
        "general rotation (axis 1,2,3 angle pi/4)",
    );

    // 7. Threshold-adjacent singularity cases:
    // Asymmetry below epsilon (|m12 - m21| = 0.008 < 0.01): enters identity singularity branch
    // Node oracle: [1.0, 0.0, 0.0, 0.0]
    let m_sing_below = Matrix4::from_elements([
        1.0, 0.004, 0.0, 0.0, -0.004, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_sing_below);
    assert_eq!(
        v,
        Vector4::new(1.0, 0.0, 0.0, 0.0),
        "singularity threshold below epsilon (0.008 < 0.01)"
    );

    // Asymmetry above epsilon (|m12 - m21| = 0.012 > 0.01): normal branch (m21 - m12 = 0.012, s = 0.012, z = +1.0)
    // Node oracle: [0.0, 0.0, 1.0, 0.0]
    let m_sing_above = Matrix4::from_elements([
        1.0, 0.006, 0.0, 0.0, -0.006, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    v.set_axis_angle_from_rotation_matrix(&m_sing_above);
    assert_vec4_close(
        &v,
        [0.0, 0.0, 1.0, 0.0],
        EPS_AXIS,
        "singularity threshold above epsilon (0.012 > 0.01)",
    );
}
