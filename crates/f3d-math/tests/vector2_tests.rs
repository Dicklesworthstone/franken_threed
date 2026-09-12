//! Tests for `Vector2` with `f64` public semantics against pinned Three.js r186.

use core::f64::consts::PI;
use f3d_math::{Matrix3, NarrowingTolerance, Vector2};

const EPS: f64 = 1e-10;

fn assert_close(a: f64, b: f64, eps: f64, msg: &str) {
    let diff = (a - b).abs();
    assert!(
        diff <= eps,
        "{msg}: expected {b}, got {a} (diff {diff} > {eps})"
    );
}

fn assert_vec2_close(v: &Vector2, expected: [f64; 2], eps: f64, msg: &str) {
    assert_close(v.x, expected[0], eps, &format!("{msg} (x)"));
    assert_close(v.y, expected[1], eps, &format!("{msg} (y)"));
}

#[test]
fn test_vector2_constructor_and_default() {
    let v = Vector2::new(3.5, -4.25);
    assert_eq!(v.x, 3.5);
    assert_eq!(v.y, -4.25);

    let z = Vector2::zero();
    assert_eq!(z.x, 0.0);
    assert_eq!(z.y, 0.0);

    let o = Vector2::one();
    assert_eq!(o.x, 1.0);
    assert_eq!(o.y, 1.0);

    let def: Vector2 = Default::default();
    assert_eq!(def, Vector2::zero());

    let formatted = format!("{v}");
    assert_eq!(formatted, "Vector2(3.5, -4.25)");
}

#[test]
fn test_vector2_set_copy_clone_and_components() {
    let mut v = Vector2::zero();
    v.set(1.0, 2.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);

    v.set_scalar(7.5);
    assert_eq!(v.x, 7.5);
    assert_eq!(v.y, 7.5);

    v.set_x(10.0);
    assert_eq!(v.x, 10.0);
    assert_eq!(v.y, 7.5);

    v.set_y(20.0);
    assert_eq!(v.x, 10.0);
    assert_eq!(v.y, 20.0);

    let mut copied = Vector2::zero();
    copied.copy(&v);
    assert_eq!(copied.x, 10.0);
    assert_eq!(copied.y, 20.0);

    let cloned = v.clone();
    assert_eq!(cloned, v);

    assert_eq!(v.get_component(0), 10.0);
    assert_eq!(v.get_component(1), 20.0);

    v.set_component(0, 100.0);
    v.set_component(1, 200.0);
    assert_eq!(v.x, 100.0);
    assert_eq!(v.y, 200.0);
}

#[test]
fn test_vector2_arithmetic_operations() {
    let mut v = Vector2::new(1.0, 2.0);
    v.add(&Vector2::new(3.0, 4.0));
    assert_eq!(v, Vector2::new(4.0, 6.0));

    v.add_scalar(2.0);
    assert_eq!(v, Vector2::new(6.0, 8.0));

    let mut sum = Vector2::zero();
    sum.add_vectors(&Vector2::new(1.5, 2.5), &Vector2::new(3.5, 4.5));
    assert_eq!(sum, Vector2::new(5.0, 7.0));

    v.add_scaled_vector(&Vector2::new(2.0, 3.0), 0.5);
    assert_eq!(v, Vector2::new(7.0, 9.5));

    v.sub(&Vector2::new(2.0, 1.5));
    assert_eq!(v, Vector2::new(5.0, 8.0));

    v.sub_scalar(1.0);
    assert_eq!(v, Vector2::new(4.0, 7.0));

    let mut diff = Vector2::zero();
    diff.sub_vectors(&Vector2::new(10.0, 20.0), &Vector2::new(3.0, 5.0));
    assert_eq!(diff, Vector2::new(7.0, 15.0));

    let mut m = Vector2::new(3.0, 4.0);
    m.multiply(&Vector2::new(2.0, 0.5));
    assert_eq!(m, Vector2::new(6.0, 2.0));

    m.multiply_scalar(3.0);
    assert_eq!(m, Vector2::new(18.0, 6.0));

    m.divide(&Vector2::new(2.0, 3.0));
    assert_eq!(m, Vector2::new(9.0, 2.0));

    m.divide_scalar(2.0);
    assert_eq!(m, Vector2::new(4.5, 1.0));

    m.negate();
    assert_eq!(m, Vector2::new(-4.5, -1.0));
}

#[test]
fn test_vector2_dot_cross_length_and_distance() {
    let a = Vector2::new(2.0, 3.0);
    let b = Vector2::new(4.0, 5.0);

    // dot: 2*4 + 3*5 = 8 + 15 = 23
    assert_eq!(a.dot(&b), 23.0);

    // cross: 2*5 - 3*4 = 10 - 12 = -2
    assert_eq!(a.cross(&b), -2.0);

    let v = Vector2::new(3.0, 4.0);
    assert_eq!(v.length_sq(), 25.0);
    assert_eq!(v.length(), 5.0);
    assert_eq!(v.manhattan_length(), 7.0);

    let p1 = Vector2::new(1.0, 2.0);
    let p2 = Vector2::new(4.0, 6.0);
    // dx = 3, dy = 4
    assert_eq!(p1.distance_to_squared(&p2), 25.0);
    assert_eq!(p1.distance_to(&p2), 5.0);
    assert_eq!(p1.manhattan_distance_to(&p2), 7.0);

    let mut resized = Vector2::new(3.0, 4.0);
    resized.set_length(10.0);
    assert_vec2_close(&resized, [6.0, 8.0], EPS, "set_length 10");
    assert_close(resized.length(), 10.0, EPS, "set_length magnitude");
}

#[test]
fn test_vector2_normalize_oracle_and_signed_zero() {
    let mut v = Vector2::new(3.0, 4.0);
    v.normalize();
    assert_vec2_close(&v, [0.6, 0.8], EPS, "normalized unit vector");
    assert_close(v.length(), 1.0, EPS, "normalized length is 1");

    // Zero-length vector normalize: Three.js evaluates `divideScalar(length || 1)`
    let mut zero_vec = Vector2::zero();
    zero_vec.normalize();
    assert_eq!(zero_vec.x, 0.0);
    assert_eq!(zero_vec.y, 0.0);

    // Signed zero preservation on zero normalize:
    let mut neg_zero = Vector2::new(-0.0, 0.0);
    neg_zero.normalize();
    assert!(neg_zero.x.is_sign_negative(), "normalized -0.0 x preserves negative sign");
    assert!(neg_zero.y.is_sign_positive(), "normalized +0.0 y preserves positive sign");

    // NaN component preservation:
    let mut nan_vec = Vector2::new(0.0, f64::NAN);
    nan_vec.normalize();
    assert_eq!(nan_vec.x, 0.0, "non-NaN component preserved on normalize");
    assert!(nan_vec.y.is_nan(), "NaN component preserved on normalize");
}

#[test]
fn test_vector2_angle_and_angle_to_signed_zero_oracle() {
    // Verified against Node oracle running Three.js r186 Vector2.angle()
    assert_close(Vector2::new(1.0, 0.0).angle(), 0.0, EPS, "angle (1, 0)");
    assert_close(Vector2::new(0.0, 1.0).angle(), PI / 2.0, EPS, "angle (0, 1)");
    assert_close(Vector2::new(-1.0, 0.0).angle(), PI, EPS, "angle (-1, 0)");
    assert_close(Vector2::new(0.0, -1.0).angle(), 1.5 * PI, EPS, "angle (0, -1)");

    // Signed-zero angles matching Three.js r186 atan2(-y, -x) + PI:
    assert_close(Vector2::new(0.0, 0.0).angle(), 0.0, EPS, "angle (0, 0)");
    assert_close(Vector2::new(-0.0, 0.0).angle(), PI, EPS, "angle (-0, 0)");
    assert_close(Vector2::new(0.0, -0.0).angle(), 2.0 * PI, EPS, "angle (0, -0)");
    assert_close(Vector2::new(-0.0, -0.0).angle(), PI, EPS, "angle (-0, -0)");

    // Angle between vectors:
    let va = Vector2::new(1.0, 0.0);
    let vb = Vector2::new(0.0, 5.0);
    assert_close(va.angle_to(&vb), PI / 2.0, EPS, "angle_to perpendicular");

    let v_par = Vector2::new(2.0, 0.0);
    assert_close(va.angle_to(&v_par), 0.0, EPS, "angle_to parallel");

    let v_opp = Vector2::new(-1.0, 0.0);
    assert_close(va.angle_to(&v_opp), PI, EPS, "angle_to opposite");

    // Zero vector angle_to returns PI / 2 per Three.js r186
    assert_close(va.angle_to(&Vector2::zero()), PI / 2.0, EPS, "angle_to zero vector");
}

#[test]
fn test_vector2_lerp_and_lerp_vectors() {
    let mut v = Vector2::new(1.0, 2.0);
    v.lerp(&Vector2::new(5.0, 10.0), 0.25);
    assert_vec2_close(&v, [2.0, 4.0], EPS, "lerp 0.25");

    let mut r = Vector2::zero();
    r.lerp_vectors(&Vector2::new(10.0, 20.0), &Vector2::new(30.0, 60.0), 0.5);
    assert_vec2_close(&r, [20.0, 40.0], EPS, "lerp_vectors 0.5");
}

#[test]
fn test_vector2_rotate_around_center() {
    // Rotated around center (3, 4) by PI / 3:
    // Matches Node oracle: x = 3.7320508075688767, y = 1.2679491924311224
    let mut v = Vector2::new(1.0, 2.0);
    v.rotate_around(&Vector2::new(3.0, 4.0), PI / 3.0);
    assert_close(v.x, 3.7320508075688767, EPS, "rotate_around x");
    assert_close(v.y, 1.2679491924311224, EPS, "rotate_around y");

    // 90-degree rotation around origin:
    let mut p = Vector2::new(1.0, 0.0);
    p.rotate_around(&Vector2::zero(), PI / 2.0);
    assert_vec2_close(&p, [0.0, 1.0], EPS, "rotate 90 deg around origin");
}

#[test]
fn test_vector2_min_max_clamp() {
    // Signed zero min / max per ECMAScript spec:
    let mut v_min = Vector2::new(-0.0, 5.0);
    v_min.min(&Vector2::new(0.0, 3.0));
    assert!(v_min.x.is_sign_negative(), "min(-0, 0) preserves -0.0");
    assert_eq!(v_min.y, 3.0);

    let mut v_max = Vector2::new(-0.0, 5.0);
    v_max.max(&Vector2::new(0.0, 3.0));
    assert!(v_max.x.is_sign_positive(), "max(-0, 0) yields +0.0");
    assert_eq!(v_max.y, 5.0);

    let mut c = Vector2::new(-5.0, 15.0);
    c.clamp(&Vector2::new(0.0, 0.0), &Vector2::new(10.0, 10.0));
    assert_eq!(c, Vector2::new(0.0, 10.0));

    let mut cs = Vector2::new(-2.0, 8.0);
    cs.clamp_scalar(0.0, 5.0);
    assert_eq!(cs, Vector2::new(0.0, 5.0));

    // clamp_length up: (3, 4) length 5 clamped to [6, 10] -> length 6
    let mut cl_up = Vector2::new(3.0, 4.0);
    cl_up.clamp_length(6.0, 10.0);
    assert_vec2_close(&cl_up, [3.6, 4.8], EPS, "clamp_length up");
    assert_close(cl_up.length(), 6.0, EPS, "clamp_length up magnitude");

    // clamp_length down: (3, 4) length 5 clamped to [1, 2] -> length 2
    let mut cl_down = Vector2::new(3.0, 4.0);
    cl_down.clamp_length(1.0, 2.0);
    assert_vec2_close(&cl_down, [1.2, 1.6], EPS, "clamp_length down");
    assert_close(cl_down.length(), 2.0, EPS, "clamp_length down magnitude");
}

#[test]
fn test_vector2_rounding_and_truncation() {
    let mut f = Vector2::new(1.9, -1.1);
    f.floor();
    assert_eq!(f, Vector2::new(1.0, -2.0));

    let mut c = Vector2::new(1.1, -1.9);
    c.ceil();
    assert_eq!(c, Vector2::new(2.0, -1.0));

    // Rounding with ECMAScript round-half-up and -0.0 for [-0.5, 0.0):
    let mut r = Vector2::new(-0.5, 1.5);
    r.round();
    assert!(r.x.is_sign_negative() && r.x == 0.0, "round(-0.5) must be -0.0");
    assert_eq!(r.y, 2.0);

    // Truncation preserving signed zero:
    let mut t = Vector2::new(-0.9, 1.9);
    t.round_to_zero();
    assert!(t.x.is_sign_negative() && t.x == 0.0, "trunc(-0.9) must be -0.0");
    assert_eq!(t.y, 1.0);
}

#[test]
fn test_vector2_apply_matrix3() {
    // 3x3 matrix in column-major order:
    // [ 2  0  5 ]
    // [ 0  3  7 ]
    // [ 0  0  1 ]
    // elements: [e0, e1, e2, e3, e4, e5, e6, e7, e8]
    // = [2, 0, 0, 0, 3, 0, 5, 7, 1]
    let m = Matrix3::from_elements([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 5.0, 7.0, 1.0]);
    let mut v = Vector2::new(4.0, 6.0);
    v.apply_matrix3(&m);
    // x = 2*4 + 0*6 + 5 = 13
    // y = 0*4 + 3*6 + 7 = 25
    assert_eq!(v, Vector2::new(13.0, 25.0));
}

#[test]
fn test_vector2_equals_and_array_conversion() {
    let a = Vector2::new(1.0, 2.0);
    let b = Vector2::new(1.0, 2.0);
    let c = Vector2::new(1.0, 3.0);
    assert!(a.equals(&b));
    assert!(!a.equals(&c));

    let arr = a.to_array();
    assert_eq!(arr, [1.0, 2.0]);
    let from = Vector2::from_array([5.0, 6.0]);
    assert_eq!(from, Vector2::new(5.0, 6.0));
}

#[test]
fn test_vector2_f32_narrowing() {
    let v = Vector2::new(1.5, 2.25);
    let tol = NarrowingTolerance::EXACT;
    let f32_arr = v.to_f32_with_tolerance(tol, true).expect("narrowing exact f32");
    assert_eq!(f32_arr, [1.5f32, 2.25f32]);

    // Strict rejects NaN:
    let nan_v = Vector2::new(1.0, f64::NAN);
    assert!(nan_v.to_f32_strict(1e-4, 1e-4).is_err(), "strict narrowing must reject NaN");

    // Non-strict preserves NaN:
    let preserved = nan_v.to_f32_checked(1e-4, 1e-4).expect("checked narrowing preserves NaN");
    assert_eq!(preserved[0], 1.0f32);
    assert!(preserved[1].is_nan());
}

#[test]
fn test_vector2_negative_divide_by_zero_counterexample() {
    // In Three.js / JS: 1 / 0 = Infinity.
    let mut v = Vector2::new(2.0, -3.0);
    v.divide_scalar(0.0);
    assert!(v.x.is_infinite() && v.x.is_sign_positive(), "divide by +0.0 yields +Inf for positive numerator");
    assert!(v.y.is_infinite() && v.y.is_sign_negative(), "divide by +0.0 yields -Inf for negative numerator");

    let mut z = Vector2::zero();
    z.divide_scalar(0.0);
    assert!(z.x.is_nan(), "0 / 0 yields NaN");
    assert!(z.y.is_nan(), "0 / 0 yields NaN");
}

#[test]
fn test_vector2_negative_scaling_counterexample() {
    let mut v = Vector2::new(3.0, -4.0);
    v.multiply_scalar(-2.5);
    assert_eq!(v, Vector2::new(-7.5, 10.0));
}

#[test]
#[should_panic(expected = "Vector2: index is out of range: 2")]
fn test_vector2_get_component_out_of_range_panics() {
    let v = Vector2::new(1.0, 2.0);
    let _ = v.get_component(2);
}

#[test]
#[should_panic(expected = "Vector2: index is out of range: 2")]
fn test_vector2_set_component_out_of_range_panics() {
    let mut v = Vector2::new(1.0, 2.0);
    v.set_component(2, 5.0);
}

#[test]
fn test_vector2_slice_offset_valid_and_out_of_range_behavior() {
    // 1. Valid slice with nonzero offset from sentinel buffer: matches Three.js Vector2.fromArray(src, 2)
    let src = [99.0, 99.0, 1.5, -2.5, 99.0];
    let mut v = Vector2::zero();
    v.from_slice_offset(&src, 2);
    assert_eq!(v.x, 1.5, "v.x after from_slice_offset(src, 2)");
    assert_eq!(v.y, -2.5, "v.y after from_slice_offset(src, 2)");

    // 2. to_slice_offset into sentinel buffer at nonzero offset: matches Three.js Vector2.toArray(dst, 1)
    let mut dst = [99.0, 0.0, 0.0, 99.0];
    v.to_slice_offset(&mut dst, 1);
    assert_eq!(dst, [99.0, 1.5, -2.5, 99.0], "dst buffer with sentinels preserved");

    // 3. Native out-of-range behavior: panics on short slice; partial write occurs before out-of-bounds index
    let short_src = [7.0];
    let mut v_partial = Vector2::new(100.0, 200.0);
    let res_src = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v_partial.from_slice_offset(&short_src, 0);
    }));
    assert!(res_src.is_err(), "must panic when src slice is too short");
    assert_eq!(v_partial.x, 7.0, "self.x updated before panic on offset + 1");
    assert_eq!(v_partial.y, 200.0, "self.y untouched");

    let mut short_dst = [0.0];
    let res_dst = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v.to_slice_offset(&mut short_dst, 0);
    }));
    assert!(res_dst.is_err(), "must panic when dst slice is too short");
    assert_eq!(short_dst[0], 1.5, "dst[0] updated before panic on offset + 1");
}
