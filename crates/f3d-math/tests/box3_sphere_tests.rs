//! Analytical tests for Box3 and Sphere primitives matching Three.js r186.

use f3d_math::box3::Box3;
use f3d_math::matrix4::Matrix4;
use f3d_math::plane::Plane;
use f3d_math::sphere::Sphere;
use f3d_math::vector3::Vector3;

const EPS: f64 = 1e-10;

fn assert_close(a: f64, b: f64, eps: f64, msg: &str) {
    let diff = (a - b).abs();
    assert!(diff <= eps, "{msg}: expected {b}, got {a} (diff {diff} > {eps})");
}

fn assert_vec_close(v: &Vector3, expected: [f64; 3], eps: f64, msg: &str) {
    assert_close(v.x, expected[0], eps, &format!("{msg} (x)"));
    assert_close(v.y, expected[1], eps, &format!("{msg} (y)"));
    assert_close(v.z, expected[2], eps, &format!("{msg} (z)"));
}

#[test]
fn test_box3_empty_convention_and_predicates() {
    let b = Box3::empty();

    // Verify empty convention (+inf / -inf)
    assert!(b.min.x.is_infinite() && b.min.x.is_sign_positive(), "min.x is +inf");
    assert!(b.min.y.is_infinite() && b.min.y.is_sign_positive(), "min.y is +inf");
    assert!(b.min.z.is_infinite() && b.min.z.is_sign_positive(), "min.z is +inf");
    assert!(b.max.x.is_infinite() && b.max.x.is_sign_negative(), "max.x is -inf");
    assert!(b.max.y.is_infinite() && b.max.y.is_sign_negative(), "max.y is -inf");
    assert!(b.max.z.is_infinite() && b.max.z.is_sign_negative(), "max.z is -inf");

    assert!(b.is_empty(), "empty box is_empty() is true");

    // Center and size on empty box must return (0, 0, 0)
    assert_vec_close(&b.get_center(), [0.0, 0.0, 0.0], EPS, "empty box center");
    assert_vec_close(&b.get_size(), [0.0, 0.0, 0.0], EPS, "empty box size");

    // Empty box contains no points and intersects no boxes
    assert!(!b.contains_point(&Vector3::zero()), "empty box contains no points");
    assert!(!b.intersects_box(&b), "empty box does not intersect itself");

    // Transforming an empty box leaves it empty
    let mut m = Matrix4::identity();
    m.elements[12] = 100.0;
    let mut b_trans = b;
    b_trans.apply_matrix4(&m);
    assert!(b_trans.is_empty(), "empty box transformed is still empty");
}

#[test]
fn test_box3_set_and_set_from_points_and_buffer() {
    let points = [
        Vector3::new(1.0, 2.0, 3.0),
        Vector3::new(-4.0, 5.0, 0.0),
        Vector3::new(2.0, -1.0, 8.0),
    ];

    let mut b1 = Box3::empty();
    b1.set_from_points(&points);

    // Expected min: (-4, -1, 0), max: (2, 5, 8)
    assert_vec_close(&b1.min, [-4.0, -1.0, 0.0], EPS, "set_from_points min");
    assert_vec_close(&b1.max, [2.0, 5.0, 8.0], EPS, "set_from_points max");

    // Center: ((-4 + 2)/2, (-1 + 5)/2, (0 + 8)/2) = (-1, 2, 4)
    assert_vec_close(&b1.get_center(), [-1.0, 2.0, 4.0], EPS, "box center");

    // Size: (2 - (-4), 5 - (-1), 8 - 0) = (6, 6, 8)
    assert_vec_close(&b1.get_size(), [6.0, 6.0, 8.0], EPS, "box size");

    // set_from_buffer with trailing coordinate ignored
    let buffer = [
        1.0, 2.0, 3.0,
        -4.0, 5.0, 0.0,
        2.0, -1.0, 8.0,
        999.0, // extra partial coordinate ignored
    ];
    let mut b2 = Box3::empty();
    b2.set_from_buffer(&buffer);

    assert!(b1.equals(&b2), "set_from_buffer matches set_from_points");

    // Empty slice yields an empty box
    let mut b_empty_slice = Box3::new(Vector3::zero(), Vector3::new(1.0, 1.0, 1.0));
    b_empty_slice.set_from_points(&[]);
    assert!(b_empty_slice.is_empty(), "set_from_points(&[]) must yield an empty box");
}

#[test]
fn test_box3_expand_contains_and_clamp() {
    let mut b = Box3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 10.0, 10.0));

    // contains_point
    assert!(b.contains_point(&Vector3::new(5.0, 5.0, 5.0)), "inside point");
    assert!(b.contains_point(&Vector3::new(0.0, 5.0, 10.0)), "boundary point");
    assert!(!b.contains_point(&Vector3::new(11.0, 5.0, 5.0)), "outside point");

    // expand_by_point
    b.expand_by_point(&Vector3::new(-2.0, 12.0, 5.0));
    assert_vec_close(&b.min, [-2.0, 0.0, 0.0], EPS, "expanded min");
    assert_vec_close(&b.max, [10.0, 12.0, 10.0], EPS, "expanded max");

    // expand_by_vector
    b.expand_by_vector(&Vector3::new(1.0, 2.0, 3.0));
    assert_vec_close(&b.min, [-3.0, -2.0, -3.0], EPS, "vector expanded min");
    assert_vec_close(&b.max, [11.0, 14.0, 13.0], EPS, "vector expanded max");

    // clamp_point
    let clamped = b.clamp_point(&Vector3::new(20.0, -5.0, 5.0));
    assert_vec_close(&clamped, [11.0, -2.0, 5.0], EPS, "clamped point");

    // distance_to_point
    let unit_box = Box3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 10.0, 10.0));
    assert_close(unit_box.distance_to_point(&Vector3::new(15.0, 0.0, 0.0)), 5.0, EPS, "distance to point");
    assert_close(unit_box.distance_to_point(&Vector3::new(5.0, 5.0, 5.0)), 0.0, EPS, "inside distance is 0");
}

#[test]
fn test_box3_union_and_intersect() {
    let mut b1 = Box3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(4.0, 4.0, 4.0));
    let b2 = Box3::new(Vector3::new(2.0, 2.0, 2.0), Vector3::new(6.0, 6.0, 6.0));

    // Union
    let mut b_union = b1;
    b_union.union(&b2);
    assert_vec_close(&b_union.min, [0.0, 0.0, 0.0], EPS, "union min");
    assert_vec_close(&b_union.max, [6.0, 6.0, 6.0], EPS, "union max");

    // Intersect
    let mut b_inter = b1;
    b_inter.intersect(&b2);
    assert_vec_close(&b_inter.min, [2.0, 2.0, 2.0], EPS, "intersect min");
    assert_vec_close(&b_inter.max, [4.0, 4.0, 4.0], EPS, "intersect max");

    // Disjoint intersect yields empty box
    let b_disjoint = Box3::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(12.0, 12.0, 12.0));
    b1.intersect(&b_disjoint);
    assert!(b1.is_empty(), "disjoint intersect produces empty box");

    // Union with empty box preserves original
    let mut b_empty = Box3::empty();
    b_empty.union(&b2);
    assert_vec_close(&b_empty.min, [2.0, 2.0, 2.0], EPS, "union empty with box min");
    assert_vec_close(&b_empty.max, [6.0, 6.0, 6.0], EPS, "union empty with box max");
}

#[test]
fn test_box3_apply_matrix4_8_corner_non_uniform_scale_and_rotation() {
    // Initial box centered at origin
    let mut b = Box3::new(Vector3::new(-1.0, -2.0, -3.0), Vector3::new(1.0, 2.0, 3.0));

    // Non-uniform scale (2, 3, 4) and translation (10, 20, 30)
    let mut m = Matrix4::identity();
    m.elements[0] = 2.0;
    m.elements[5] = 3.0;
    m.elements[10] = 4.0;
    m.elements[12] = 10.0;
    m.elements[13] = 20.0;
    m.elements[14] = 30.0;

    b.apply_matrix4(&m);

    // Analytical bounds:
    // x in [10 - 2, 10 + 2] = [8, 12]
    // y in [20 - 6, 20 + 6] = [14, 26]
    // z in [30 - 12, 30 + 12] = [18, 42]
    assert_vec_close(&b.min, [8.0, 14.0, 18.0], EPS, "transformed min with non-uniform scale");
    assert_vec_close(&b.max, [12.0, 26.0, 42.0], EPS, "transformed max with non-uniform scale");

    // 90 degree rotation around Z axis: (x, y) -> (-y, x)
    // For box [-1, 1] x [-2, 2], new x is [-2, 2], new y is [-1, 1]
    let mut b_rot = Box3::new(Vector3::new(-1.0, -2.0, -3.0), Vector3::new(1.0, 2.0, 3.0));
    let mut m_rot = Matrix4::identity();
    m_rot.elements[0] = 0.0;
    m_rot.elements[1] = 1.0;
    m_rot.elements[4] = -1.0;
    m_rot.elements[5] = 0.0;

    b_rot.apply_matrix4(&m_rot);
    assert_vec_close(&b_rot.min, [-2.0, -1.0, -3.0], EPS, "rotated 90 deg Z min");
    assert_vec_close(&b_rot.max, [2.0, 1.0, 3.0], EPS, "rotated 90 deg Z max");
}

#[test]
fn test_sphere_empty_convention_and_set_from_points() {
    let s = Sphere::empty();
    assert!(s.is_empty(), "empty sphere is_empty is true");
    assert_eq!(s.radius, -1.0, "empty sphere radius is -1");
    assert_vec_close(&s.center, [0.0, 0.0, 0.0], EPS, "empty sphere center");

    // Points on unit circle/sphere
    let points = [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, -1.0),
    ];

    let mut s_points = Sphere::empty();
    s_points.set_from_points(&points, None);

    assert_vec_close(&s_points.center, [0.0, 0.0, 0.0], EPS, "sphere center from points");
    assert_close(s_points.radius, 1.0, EPS, "sphere radius from points");

    // set_from_points with explicit center
    let offset_center = Vector3::new(10.0, 0.0, 0.0);
    let pts = [Vector3::new(13.0, 4.0, 0.0)]; // dist = sqrt(3^2 + 4^2) = 5.0
    let mut s_explicit = Sphere::empty();
    s_explicit.set_from_points(&pts, Some(offset_center));

    assert_vec_close(&s_explicit.center, [10.0, 0.0, 0.0], EPS, "explicit center");
    assert_close(s_explicit.radius, 5.0, EPS, "explicit radius 5.0");

    // Empty slice yields center = (0, 0, 0), radius = 0.0
    let mut s_empty_slice = Sphere::new(Vector3::new(1.0, 2.0, 3.0), 5.0);
    s_empty_slice.set_from_points(&[], None);
    assert_close(s_empty_slice.radius, 0.0, EPS, "empty points radius 0");
    assert_vec_close(&s_empty_slice.center, [0.0, 0.0, 0.0], EPS, "empty points center (0,0,0)");
}

#[test]
fn test_sphere_contains_distance_clamp_and_bounding_box() {
    let s = Sphere::new(Vector3::new(0.0, 0.0, 0.0), 5.0);

    // contains_point
    assert!(s.contains_point(&Vector3::new(3.0, 4.0, 0.0)), "on surface point");
    assert!(s.contains_point(&Vector3::new(1.0, 1.0, 1.0)), "inside point");
    assert!(!s.contains_point(&Vector3::new(4.0, 4.0, 0.0)), "outside point");

    // distance_to_point
    assert_close(s.distance_to_point(&Vector3::new(0.0, 8.0, 0.0)), 3.0, EPS, "distance outside");
    assert_close(s.distance_to_point(&Vector3::new(0.0, 2.0, 0.0)), -3.0, EPS, "distance inside");

    // clamp_point
    let clamped_outside = s.clamp_point(&Vector3::new(0.0, 10.0, 0.0));
    assert_vec_close(&clamped_outside, [0.0, 5.0, 0.0], EPS, "clamped to surface");

    let clamped_inside = s.clamp_point(&Vector3::new(0.0, 3.0, 0.0));
    assert_vec_close(&clamped_inside, [0.0, 3.0, 0.0], EPS, "inside unchanged");

    // get_bounding_box
    let bb = s.get_bounding_box();
    assert_vec_close(&bb.min, [-5.0, -5.0, -5.0], EPS, "bounding box min");
    assert_vec_close(&bb.max, [5.0, 5.0, 5.0], EPS, "bounding box max");

    // Empty sphere bounding box is empty
    let empty_bb = Sphere::empty().get_bounding_box();
    assert!(empty_bb.is_empty(), "empty sphere bounding box is empty");
}

#[test]
fn test_sphere_apply_matrix4_get_max_scale_on_axis() {
    let mut s = Sphere::new(Vector3::new(1.0, 2.0, 3.0), 2.0);

    // Non-uniform scale (2.0, 5.0, 3.0) and translation (10.0, 20.0, 30.0)
    let mut m = Matrix4::identity();
    m.elements[0] = 2.0;
    m.elements[5] = 5.0;
    m.elements[10] = 3.0;
    m.elements[12] = 10.0;
    m.elements[13] = 20.0;
    m.elements[14] = 30.0;

    assert_close(m.get_max_scale_on_axis(), 5.0, EPS, "max scale on axis");

    s.apply_matrix4(&m);

    // Center transformed: (1*2 + 10, 2*5 + 20, 3*3 + 30) = (12, 30, 39)
    assert_vec_close(&s.center, [12.0, 30.0, 39.0], EPS, "transformed sphere center");

    // Radius scaled by max scale on axis: 2.0 * 5.0 = 10.0
    assert_close(s.radius, 10.0, EPS, "transformed sphere radius");
}

#[test]
fn test_box3_and_sphere_intersections() {
    let b = Box3::new(Vector3::new(-2.0, -2.0, -2.0), Vector3::new(2.0, 2.0, 2.0));

    // Sphere 1: concentric inside
    let s1 = Sphere::new(Vector3::new(0.0, 0.0, 0.0), 1.0);
    assert!(b.intersects_sphere(&s1), "concentric sphere intersects");
    assert!(s1.intersects_box(&b), "reciprocal intersects_box");

    // Sphere 2: touches face (center at (3,0,0), radius 1.0 -> dist to box is 1.0 <= 1.0)
    let s2 = Sphere::new(Vector3::new(3.0, 0.0, 0.0), 1.0);
    assert!(b.intersects_sphere(&s2), "touching face sphere intersects");

    // Sphere 3: outside face (center at (4,0,0), radius 1.0 -> dist is 2.0 > 1.0)
    let s3 = Sphere::new(Vector3::new(4.0, 0.0, 0.0), 1.0);
    assert!(!b.intersects_sphere(&s3), "separated face sphere does not intersect");

    // Sphere 4: outside corner (center at (3,3,0), radius 1.0 -> dist to (2,2,0) is sqrt(2) ≈ 1.414 > 1.0)
    let s4 = Sphere::new(Vector3::new(3.0, 3.0, 0.0), 1.0);
    assert!(!b.intersects_sphere(&s4), "outside corner sphere does not intersect");

    // Sphere 5: overlaps corner (radius 1.5 >= sqrt(2))
    let s5 = Sphere::new(Vector3::new(3.0, 3.0, 0.0), 1.5);
    assert!(b.intersects_sphere(&s5), "overlapping corner sphere intersects");

    // Sphere-sphere intersections
    let sa = Sphere::new(Vector3::new(0.0, 0.0, 0.0), 2.0);
    let sb = Sphere::new(Vector3::new(3.0, 0.0, 0.0), 2.0); // dist=3, r_sum=4 -> intersect
    let sc = Sphere::new(Vector3::new(5.0, 0.0, 0.0), 2.0); // dist=5, r_sum=4 -> separate

    assert!(sa.intersects_sphere(&sb), "overlapping spheres intersect");
    assert!(!sa.intersects_sphere(&sc), "disjoint spheres do not intersect");
}

#[test]
fn test_sphere_union_and_expand() {
    let mut s = Sphere::new(Vector3::new(0.0, 0.0, 0.0), 1.0);

    // Expand by point (3, 0, 0)
    s.expand_by_point(&Vector3::new(3.0, 0.0, 0.0));
    // Center shifts to (1, 0, 0), radius expands to 2.0 (covers [-1, 3])
    assert_vec_close(&s.center, [1.0, 0.0, 0.0], EPS, "expand_by_point center");
    assert_close(s.radius, 2.0, EPS, "expand_by_point radius");

    // Union of two disjoint spheres
    let s1 = Sphere::new(Vector3::new(0.0, 0.0, 0.0), 1.0);
    let s2 = Sphere::new(Vector3::new(4.0, 0.0, 0.0), 1.0);
    let mut s_union = s1;
    s_union.union(&s2);

    // Expected enclosing sphere: spans [-1, 5], center (2, 0, 0), radius 3.0
    assert_vec_close(&s_union.center, [2.0, 0.0, 0.0], EPS, "union center");
    assert_close(s_union.radius, 3.0, EPS, "union radius");

    // Union with enclosing sphere
    let mut s_small = Sphere::new(Vector3::new(0.5, 0.0, 0.0), 0.5);
    s_small.union(&s1); // s1 encloses s_small
    assert_vec_close(&s_small.center, [0.0, 0.0, 0.0], EPS, "enclosing union center");
    assert_close(s_small.radius, 1.0, EPS, "enclosing union radius");
}

#[test]
fn test_upstream_r186_empty_and_inverted_intersection_parity() {
    let origin = Vector3::zero();
    let s_default = Sphere::default();
    let mut s_empty = Sphere::default();
    s_empty.make_empty();

    // 1. new Sphere().containsPoint(origin) = true
    assert!(s_default.contains_point(&origin), "new Sphere().containsPoint(origin) must be true");
    assert!(s_empty.contains_point(&origin), "emptySphere.containsPoint(origin) must be true");

    // 2. emptySphere.intersectsSphere(itself) = true
    assert!(s_empty.intersects_sphere(&s_empty), "emptySphere.intersectsSphere(itself) must be true");
    assert!(s_default.intersects_sphere(&s_default), "default Sphere intersects itself");

    // 3. box[-10,10].intersectsSphere(new Sphere()) = true (and symmetric)
    let b_10 = Box3::new(Vector3::new(-10.0, -10.0, -10.0), Vector3::new(10.0, 10.0, 10.0));
    assert!(b_10.intersects_sphere(&s_default), "box[-10,10].intersectsSphere(new Sphere()) must be true");
    assert!(s_default.intersects_box(&b_10), "new Sphere().intersectsBox(box[-10,10]) must be true");
    assert!(b_10.intersects_sphere(&s_empty), "box[-10,10].intersectsSphere(emptySphere) must be true");
    assert!(s_empty.intersects_box(&b_10), "emptySphere.intersectsBox(box[-10,10]) must be true");

    // 4. box[-10,10].intersectsBox(inverted min=1 max=-1) = true (and symmetric)
    let inv = Box3::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(-1.0, -1.0, -1.0));
    assert!(b_10.intersects_box(&inv), "box[-10,10].intersectsBox(inv) must be true");
    assert!(inv.intersects_box(&b_10), "inv.intersectsBox(box[-10,10]) must be true");

    // Authored negative radius arithmetic parity
    let s_neg = Sphere::new(Vector3::zero(), -5.0);
    assert!(s_neg.contains_point(&origin), "Sphere(0, -5).containsPoint(0) must be true (0 <= 25)");
    assert!(s_neg.contains_point(&Vector3::new(4.0, 0.0, 0.0)), "Sphere(0, -5).containsPoint(4,0,0) must be true (16 <= 25)");
    assert!(!s_neg.contains_point(&Vector3::new(6.0, 0.0, 0.0)), "Sphere(0, -5).containsPoint(6,0,0) must be false (36 <= 25)");
    assert!(s_neg.intersects_sphere(&s_neg), "Sphere(0, -5).intersectsSphere(itself) must be true (0 <= 100)");

    // Plane intersection arithmetic parity (no is_empty guards)
    let plane_origin = Plane::new(Vector3::new(0.0, 1.0, 0.0), 0.0);
    assert!(!s_empty.intersects_plane(&plane_origin), "emptySphere.intersectsPlane(plane) is false (|0| <= -1 is false)");
    assert!(b_10.intersects_plane(&plane_origin), "box[-10,10].intersectsPlane(plane) is true");
    assert!(!inv.intersects_plane(&plane_origin), "inverted box intersectsPlane(plane) is false");

    // Inverted box clamp_point evaluates Math.max(min, Math.min(max, value))
    let inv_clamp = Box3::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(-1.0, -2.0, -3.0));
    let clamped_inv = inv_clamp.clamp_point(&origin);
    assert_vec_close(&clamped_inv, [1.0, 2.0, 3.0], EPS, "inverted box clamp_point returns min bound");

    // NaN point coordinates in clamp_point propagate NaN per ECMA Math.max/min
    let nan_pt = Vector3::new(f64::NAN, 0.0, 0.0);
    let clamped_nan_pt = b_10.clamp_point(&nan_pt);
    assert!(clamped_nan_pt.x.is_nan(), "clamped NaN x is NaN");
    assert_close(clamped_nan_pt.y, 0.0, EPS, "clamped y remains 0.0");
    assert_close(clamped_nan_pt.z, 0.0, EPS, "clamped z remains 0.0");

    let inv_clamped_nan = inv_clamp.clamp_point(&nan_pt);
    assert!(inv_clamped_nan.x.is_nan(), "inverted clamped NaN x is NaN");
    assert_close(inv_clamped_nan.y, 2.0, EPS, "inverted clamped y is 2.0");
    assert_close(inv_clamped_nan.z, 3.0, EPS, "inverted clamped z is 3.0");

    // NaN box bounds in clamp_point propagate NaN
    let nan_box = Box3::new(Vector3::new(f64::NAN, -5.0, -5.0), Vector3::new(5.0, 5.0, 5.0));
    let clamped_nan_box = nan_box.clamp_point(&origin);
    assert!(clamped_nan_box.x.is_nan(), "nan_box clamped x is NaN");
    assert_close(clamped_nan_box.y, 0.0, EPS, "nan_box clamped y is 0.0");
    assert_close(clamped_nan_box.z, 0.0, EPS, "nan_box clamped z is 0.0");

    // NaN sphere center in predicates returns false
    let s_nan = Sphere::new(Vector3::new(f64::NAN, 0.0, 0.0), 1.0);
    assert!(!b_10.intersects_sphere(&s_nan), "box intersects_sphere with NaN center is false");
    assert!(!inv_clamp.intersects_sphere(&s_nan), "inv intersects_sphere with NaN center is false");
    assert!(!s_nan.intersects_sphere(&s_default), "NaN sphere intersects_sphere is false");
    assert!(!s_nan.contains_point(&origin), "NaN sphere contains_point is false");
}
