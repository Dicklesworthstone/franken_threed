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

    // set_from_buffer matching points
    let buffer = [
        1.0, 2.0, 3.0,
        -4.0, 5.0, 0.0,
        2.0, -1.0, 8.0,
    ];
    let mut b2 = Box3::empty();
    b2.set_from_buffer(&buffer);

    assert!(b1.equals(&b2), "set_from_buffer matches set_from_points");

    // Incomplete trailing coordinate produces NaN on missing components per Three.js r186
    let buffer_trailing = [
        1.0, 2.0, 3.0,
        -4.0, 5.0, 0.0,
        2.0, -1.0, 8.0,
        999.0, // incomplete trailing coordinate
    ];
    let mut b_trail = Box3::empty();
    b_trail.set_from_buffer(&buffer_trailing);
    assert_close(b_trail.min.x, -4.0, EPS, "trail min x");
    assert_close(b_trail.max.x, 999.0, EPS, "trail max x");
    assert!(b_trail.min.y.is_nan(), "trail min y is NaN");
    assert!(b_trail.max.y.is_nan(), "trail max y is NaN");
    assert!(b_trail.min.z.is_nan(), "trail min z is NaN");
    assert!(b_trail.max.z.is_nan(), "trail max z is NaN");

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
fn test_sphere_set_from_points_propagates_nan_radius() {
    // Pinned r186 Math.max preserves a NaN distance regardless of point order.
    let nan_point = Vector3::new(f64::NAN, 2.0, 3.0);
    let finite_point = Vector3::new(3.0, 4.0, 5.0);
    let center = Vector3::new(1.0, 1.0, 1.0);
    for points in [[nan_point, finite_point], [finite_point, nan_point]] {
        for optional_center in [None, Some(center)] {
            let mut sphere = Sphere::empty();
            sphere.set_from_points(&points, optional_center);
            assert!(sphere.radius.is_nan(), "NaN point must propagate with center {optional_center:?}");
            if optional_center.is_some() {
                assert_eq!(sphere.center, center);
            } else {
                assert!(sphere.center.x.is_nan());
                assert_eq!(sphere.center.y, 3.0);
                assert_eq!(sphere.center.z, 4.0);
            }
        }
    }

    let nan_center = Vector3::new(f64::NAN, 1.0, 1.0);
    let mut sphere = Sphere::empty();
    sphere.set_from_points(&[finite_point], Some(nan_center));
    assert!(sphere.radius.is_nan(), "NaN optional center makes the finite point's distance NaN");
    assert!(sphere.center.x.is_nan());
    assert_eq!(sphere.center.y, 1.0);
    assert_eq!(sphere.center.z, 1.0);

    // With no points, upstream never evaluates a distance and keeps radius zero.
    sphere.set_from_points(&[], Some(nan_center));
    assert_eq!(sphere.radius, 0.0);
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

#[test]
fn test_box3_missing_methods_node_parity() {
    // No-claim line: scalar parity only.
    // Derived Node oracle values verified against Three.js r186 Box3.js.

    // 1. set_from_center_and_size
    let mut b1 = Box3::empty();
    let center1 = Vector3::new(1.0, 2.0, 3.0);
    let size1 = Vector3::new(4.0, 6.0, 8.0);
    let ret1 = b1.set_from_center_and_size(&center1, &size1);
    assert_vec_close(&ret1.min, [-1.0, -1.0, -1.0], EPS, "b1 min");
    assert_vec_close(&ret1.max, [3.0, 5.0, 7.0], EPS, "b1 max");
    assert_vec_close(&b1.min, [-1.0, -1.0, -1.0], EPS, "mutated b1 min");
    assert_vec_close(&b1.max, [3.0, 5.0, 7.0], EPS, "mutated b1 max");

    let mut b1b = Box3::empty();
    let center1b = Vector3::new(-5.0, 10.0, -2.5);
    let size1b = Vector3::new(0.0, 5.0, 10.0);
    b1b.set_from_center_and_size(&center1b, &size1b);
    assert_vec_close(&b1b.min, [-5.0, 7.5, -7.5], EPS, "b1b min with zero width");
    assert_vec_close(&b1b.max, [-5.0, 12.5, 2.5], EPS, "b1b max with zero width");

    // 2. copy
    let mut b_copy = Box3::empty();
    let ret_copy = b_copy.copy(&b1);
    assert_vec_close(&ret_copy.min, [-1.0, -1.0, -1.0], EPS, "copy min");
    assert_vec_close(&ret_copy.max, [3.0, 5.0, 7.0], EPS, "copy max");
    assert!(b_copy.equals(&b1), "copied box equals source box");

    // 3. set_from_array
    let arr = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];
    let mut b_arr = Box3::new(Vector3::zero(), Vector3::zero());
    let ret_arr = b_arr.set_from_array(&arr);
    assert_vec_close(&ret_arr.min, [-4.0, -8.0, -6.0], EPS, "set_from_array min");
    assert_vec_close(&ret_arr.max, [7.0, 5.0, 9.0], EPS, "set_from_array max");

    // Empty array resets to empty box
    let mut b_arr_empty = Box3::new(Vector3::zero(), Vector3::new(1.0, 1.0, 1.0));
    b_arr_empty.set_from_array(&[]);
    assert!(b_arr_empty.is_empty(), "set_from_array(&[]) produces empty box");
    assert!(b_arr_empty.min.x.is_infinite() && b_arr_empty.min.x.is_sign_positive(), "min +inf");
    assert!(b_arr_empty.max.x.is_infinite() && b_arr_empty.max.x.is_sign_negative(), "max -inf");

    // Incomplete trailing coordinates produce NaN on missing components per Three.js r186
    let mut b_inc1 = Box3::empty();
    b_inc1.set_from_array(&[1.0]);
    assert_close(b_inc1.min.x, 1.0, EPS, "inc1 min x");
    assert_close(b_inc1.max.x, 1.0, EPS, "inc1 max x");
    assert!(b_inc1.min.y.is_nan() && b_inc1.max.y.is_nan(), "inc1 y is NaN");
    assert!(b_inc1.min.z.is_nan() && b_inc1.max.z.is_nan(), "inc1 z is NaN");

    let mut b_inc2 = Box3::empty();
    b_inc2.set_from_array(&[1.0, 2.0]);
    assert_close(b_inc2.min.x, 1.0, EPS, "inc2 min x");
    assert_close(b_inc2.max.x, 1.0, EPS, "inc2 max x");
    assert_close(b_inc2.min.y, 2.0, EPS, "inc2 min y");
    assert_close(b_inc2.max.y, 2.0, EPS, "inc2 max y");
    assert!(b_inc2.min.z.is_nan() && b_inc2.max.z.is_nan(), "inc2 z is NaN");

    let mut b_inc4 = Box3::empty();
    b_inc4.set_from_array(&[1.0, 2.0, 3.0, 4.0]);
    assert_close(b_inc4.min.x, 1.0, EPS, "inc4 min x");
    assert_close(b_inc4.max.x, 4.0, EPS, "inc4 max x");
    assert!(b_inc4.min.y.is_nan() && b_inc4.max.y.is_nan(), "inc4 y is NaN");
    assert!(b_inc4.min.z.is_nan() && b_inc4.max.z.is_nan(), "inc4 z is NaN");

    let mut b_inc5 = Box3::empty();
    b_inc5.set_from_array(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_close(b_inc5.min.x, 1.0, EPS, "inc5 min x");
    assert_close(b_inc5.max.x, 4.0, EPS, "inc5 max x");
    assert_close(b_inc5.min.y, 2.0, EPS, "inc5 min y");
    assert_close(b_inc5.max.y, 5.0, EPS, "inc5 max y");
    assert!(b_inc5.min.z.is_nan() && b_inc5.max.z.is_nan(), "inc5 z is NaN");

    let arr_trailing = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 999.0, 888.0];
    let mut b_arr_trail = Box3::empty();
    b_arr_trail.set_from_array(&arr_trailing);
    assert_close(b_arr_trail.min.x, -4.0, EPS, "trail min x");
    assert_close(b_arr_trail.max.x, 999.0, EPS, "trail max x");
    assert_close(b_arr_trail.min.y, -8.0, EPS, "trail min y");
    assert_close(b_arr_trail.max.y, 888.0, EPS, "trail max y");
    assert!(b_arr_trail.min.z.is_nan() && b_arr_trail.max.z.is_nan(), "trail z is NaN");

    // NaN coordinates propagate per ECMAScript Math.min/Math.max
    let mut b_nan1 = Box3::empty();
    b_nan1.set_from_array(&[f64::NAN, 2.0, 3.0]);
    assert!(b_nan1.min.x.is_nan() && b_nan1.max.x.is_nan(), "NaN x propagates");
    assert_close(b_nan1.min.y, 2.0, EPS, "nan1 y min");
    assert_close(b_nan1.max.y, 2.0, EPS, "nan1 y max");
    assert_close(b_nan1.min.z, 3.0, EPS, "nan1 z min");
    assert_close(b_nan1.max.z, 3.0, EPS, "nan1 z max");

    let mut b_nan2 = Box3::empty();
    b_nan2.set_from_array(&[1.0, f64::NAN, 3.0]);
    assert_close(b_nan2.min.x, 1.0, EPS, "nan2 x min");
    assert_close(b_nan2.max.x, 1.0, EPS, "nan2 x max");
    assert!(b_nan2.min.y.is_nan() && b_nan2.max.y.is_nan(), "NaN y propagates");
    assert_close(b_nan2.min.z, 3.0, EPS, "nan2 z min");
    assert_close(b_nan2.max.z, 3.0, EPS, "nan2 z max");

    // Direct oracles from NavyAspen review 19852
    let mut b_na1 = Box3::empty();
    b_na1.set_from_array(&[5.0]);
    assert_close(b_na1.min.x, 5.0, EPS, "na1 min x");
    assert_close(b_na1.max.x, 5.0, EPS, "na1 max x");
    assert!(b_na1.min.y.is_nan() && b_na1.max.y.is_nan(), "na1 y is NaN");
    assert!(b_na1.min.z.is_nan() && b_na1.max.z.is_nan(), "na1 z is NaN");

    let mut b_na2 = Box3::empty();
    b_na2.set_from_array(&[5.0, 6.0]);
    assert_close(b_na2.min.x, 5.0, EPS, "na2 min x");
    assert_close(b_na2.max.x, 5.0, EPS, "na2 max x");
    assert_close(b_na2.min.y, 6.0, EPS, "na2 min y");
    assert_close(b_na2.max.y, 6.0, EPS, "na2 max y");
    assert!(b_na2.min.z.is_nan() && b_na2.max.z.is_nan(), "na2 z is NaN");

    let mut b_na3 = Box3::empty();
    b_na3.set_from_array(&[1.0, 2.0, 3.0, 9.0]);
    assert_close(b_na3.min.x, 1.0, EPS, "na3 min x");
    assert_close(b_na3.max.x, 9.0, EPS, "na3 max x");
    assert!(b_na3.min.y.is_nan() && b_na3.max.y.is_nan(), "na3 y is NaN");
    assert!(b_na3.min.z.is_nan() && b_na3.max.z.is_nan(), "na3 z is NaN");

    let mut b_na4 = Box3::empty();
    b_na4.set_from_array(&[1.0, 2.0, 3.0, 9.0, 8.0]);
    assert_close(b_na4.min.x, 1.0, EPS, "na4 min x");
    assert_close(b_na4.max.x, 9.0, EPS, "na4 max x");
    assert_close(b_na4.min.y, 2.0, EPS, "na4 min y");
    assert_close(b_na4.max.y, 8.0, EPS, "na4 max y");
    assert!(b_na4.min.z.is_nan() && b_na4.max.z.is_nan(), "na4 z is NaN");

    let mut b_na5 = Box3::empty();
    b_na5.set_from_array(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 100.0]);
    assert_close(b_na5.min.x, -4.0, EPS, "na5 min x");
    assert_close(b_na5.max.x, 100.0, EPS, "na5 max x");
    assert!(b_na5.min.y.is_nan() && b_na5.max.y.is_nan(), "na5 y is NaN");
    assert!(b_na5.min.z.is_nan() && b_na5.max.z.is_nan(), "na5 z is NaN");

    let mut b_na6 = Box3::empty();
    b_na6.set_from_array(&[f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(b_na6.min.x.is_nan() && b_na6.max.x.is_nan(), "na6 x is NaN");
    assert_close(b_na6.min.y, 1.0, EPS, "na6 min y");
    assert_close(b_na6.max.y, 4.0, EPS, "na6 max y");
    assert_close(b_na6.min.z, 2.0, EPS, "na6 min z");
    assert_close(b_na6.max.z, 5.0, EPS, "na6 max z");

    // Signed zero handling per ECMAScript Math.min/Math.max
    let mut b_zero = Box3::empty();
    b_zero.set_from_array(&[-0.0, 0.0, -0.0]);
    assert!(b_zero.min.x == 0.0 && b_zero.min.x.is_sign_negative(), "min.x is -0.0");
    assert!(b_zero.min.y == 0.0 && b_zero.min.y.is_sign_positive(), "min.y is +0.0");
    assert!(b_zero.min.z == 0.0 && b_zero.min.z.is_sign_negative(), "min.z is -0.0");
    assert!(b_zero.max.x == 0.0 && b_zero.max.x.is_sign_negative(), "max.x is -0.0");
    assert!(b_zero.max.y == 0.0 && b_zero.max.y.is_sign_positive(), "max.y is +0.0");
    assert!(b_zero.max.z == 0.0 && b_zero.max.z.is_sign_negative(), "max.z is -0.0");

    // 4. get_bounding_sphere
    let b_sphere1 = Box3::new(Vector3::new(-2.0, -3.0, -4.0), Vector3::new(2.0, 3.0, 4.0));
    let mut s1 = Sphere::empty();
    let ret_s1 = b_sphere1.get_bounding_sphere(&mut s1);
    assert_vec_close(&ret_s1.center, [0.0, 0.0, 0.0], EPS, "bounding sphere center");
    assert_close(ret_s1.radius, 29.0_f64.sqrt(), EPS, "bounding sphere radius sqrt(29)");
    assert_vec_close(&s1.center, [0.0, 0.0, 0.0], EPS, "mutated target sphere center");
    assert_close(s1.radius, 29.0_f64.sqrt(), EPS, "mutated target sphere radius");

    let b_sphere2 = Box3::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(5.0, 8.0, 15.0));
    let mut s2 = Sphere::empty();
    b_sphere2.get_bounding_sphere(&mut s2);
    assert_vec_close(&s2.center, [3.0, 5.0, 9.0], EPS, "b_sphere2 center");
    assert_close(s2.radius, 7.0, EPS, "b_sphere2 radius 7.0");

    let b_sphere_empty = Box3::empty();
    let mut s_empty_target = Sphere::new(Vector3::new(10.0, 20.0, 30.0), 50.0);
    b_sphere_empty.get_bounding_sphere(&mut s_empty_target);
    assert!(s_empty_target.is_empty(), "empty box bounding sphere is empty");
    assert_vec_close(&s_empty_target.center, [0.0, 0.0, 0.0], EPS, "empty box sphere center (0,0,0)");
    assert_close(s_empty_target.radius, -1.0, EPS, "empty box sphere radius -1.0");

    // 5. get_parameter
    let b_param = Box3::new(Vector3::new(0.0, 10.0, 20.0), Vector3::new(10.0, 30.0, 40.0));
    let mut param_target = Vector3::zero();

    // Interior point
    let ret_p1 = b_param.get_parameter(&Vector3::new(5.0, 20.0, 35.0), &mut param_target);
    assert_vec_close(ret_p1, [0.5, 0.5, 0.75], EPS, "interior parameter");
    assert_vec_close(&param_target, [0.5, 0.5, 0.75], EPS, "interior parameter target");

    // Boundary point
    b_param.get_parameter(&Vector3::new(0.0, 10.0, 20.0), &mut param_target);
    assert_vec_close(&param_target, [0.0, 0.0, 0.0], EPS, "boundary parameter");

    // Outside point
    b_param.get_parameter(&Vector3::new(15.0, 5.0, 50.0), &mut param_target);
    assert_vec_close(&param_target, [1.5, -0.25, 1.5], EPS, "outside parameter");

    // Degenerate box with width 0 (divide by zero behavior)
    let b_degen = Box3::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(2.0, 4.0, 8.0));
    // point.x == min.x: (2 - 2) / (2 - 2) = 0.0 / 0.0 = NaN
    b_degen.get_parameter(&Vector3::new(2.0, 2.0, 4.0), &mut param_target);
    assert!(param_target.x.is_nan(), "0.0 / 0.0 parameter is NaN");
    assert_close(param_target.y, 0.5, EPS, "degen y is 0.5");
    assert_close(param_target.z, 0.5, EPS, "degen z is 0.5");

    // point.x > min.x: (5 - 2) / (2 - 2) = 3.0 / 0.0 = +Infinity
    b_degen.get_parameter(&Vector3::new(5.0, 2.0, 4.0), &mut param_target);
    assert!(param_target.x.is_infinite() && param_target.x.is_sign_positive(), "3.0 / 0.0 parameter is +inf");
    assert_close(param_target.y, 0.5, EPS, "degen y is 0.5");
    assert_close(param_target.z, 0.5, EPS, "degen z is 0.5");

    // point.x < min.x: (-1 - 2) / (2 - 2) = -3.0 / 0.0 = -Infinity
    b_degen.get_parameter(&Vector3::new(-1.0, 2.0, 4.0), &mut param_target);
    assert!(param_target.x.is_infinite() && param_target.x.is_sign_negative(), "-3.0 / 0.0 parameter is -inf");
    assert_close(param_target.y, 0.5, EPS, "degen y is 0.5");
    assert_close(param_target.z, 0.5, EPS, "degen z is 0.5");
}
