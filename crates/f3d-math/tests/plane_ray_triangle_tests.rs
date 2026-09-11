//! Analytical tests for Plane, Ray, and Triangle matching Three.js r186.

use f3d_math::box3::Box3;
use f3d_math::line3::Line3;
use f3d_math::matrix4::Matrix4;
use f3d_math::plane::Plane;
use f3d_math::ray::Ray;
use f3d_math::sphere::Sphere;
use f3d_math::triangle::Triangle;
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
fn test_plane_set_from_coplanar_points_and_normal_point() {
    // CCW points on XY plane: a=(1, 0, 0), b=(0, 0, 0), c=(0, 1, 0)
    // (c - b) x (a - b) = (0, 1, 0) x (1, 0, 0) = (0, 0, -1)
    let a = Vector3::new(1.0, 0.0, 0.0);
    let b = Vector3::new(0.0, 0.0, 0.0);
    let c = Vector3::new(0.0, 1.0, 0.0);

    let mut plane = Plane::default();
    plane.set_from_coplanar_points(&a, &b, &c);
    assert_vec_close(&plane.normal, [0.0, 0.0, -1.0], EPS, "coplanar points normal");
    assert_close(plane.constant, 0.0, EPS, "coplanar points constant");

    // set_from_normal_and_coplanar_point: plane z = 5
    let normal = Vector3::new(0.0, 0.0, 1.0);
    let point = Vector3::new(0.0, 0.0, 5.0);
    plane.set_from_normal_and_coplanar_point(&normal, &point);

    assert_vec_close(&plane.normal, [0.0, 0.0, 1.0], EPS, "normal vector");
    assert_close(plane.constant, -5.0, EPS, "constant for z=5 plane");

    // distance_to_point: (0, 0, 8) -> 8 - 5 = 3
    assert_close(plane.distance_to_point(&Vector3::new(0.0, 0.0, 8.0)), 3.0, EPS, "point in front");
    assert_close(plane.distance_to_point(&Vector3::new(0.0, 0.0, 2.0)), -3.0, EPS, "point behind");

    // project_point: (3, 4, 8) -> (3, 4, 5)
    let projected = plane.project_point(&Vector3::new(3.0, 4.0, 8.0));
    assert_vec_close(&projected, [3.0, 4.0, 5.0], EPS, "projected point on plane");

    // distance_to_sphere: center (0, 0, 8), radius 2.0 -> distance 3 - 2 = 1.0
    let sphere = Sphere::new(Vector3::new(0.0, 0.0, 8.0), 2.0);
    assert_close(plane.distance_to_sphere(&sphere), 1.0, EPS, "distance to sphere");
}

#[test]
fn test_plane_intersect_line_and_project_point() {
    let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), -5.0);

    // Line crossing plane from z=0 to z=10
    let line1 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 10.0));
    let hit1 = plane.intersect_line(&line1, true);
    assert!(hit1.is_some(), "line intersects plane");
    assert_vec_close(&hit1.unwrap(), [0.0, 0.0, 5.0], EPS, "line intersection point");
    assert!(plane.intersects_line(&line1), "intersects_line predicate true");

    // Line segment ending before plane (z=0 to z=4)
    let line2 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 4.0));
    assert!(plane.intersect_line(&line2, true).is_none(), "clamped line does not intersect");
    let unclamped = plane.intersect_line(&line2, false);
    assert!(unclamped.is_some(), "unclamped line intersects infinite line");
    assert_vec_close(&unclamped.unwrap(), [0.0, 0.0, 5.0], EPS, "unclamped intersection point");
    assert!(!plane.intersects_line(&line2), "intersects_line predicate false");

    // Coplanar line: z=5
    let line_coplanar = Line3::new(Vector3::new(1.0, 2.0, 5.0), Vector3::new(3.0, 4.0, 5.0));
    let hit_coplanar = plane.intersect_line(&line_coplanar, true);
    assert!(hit_coplanar.is_some(), "coplanar line returns start");
    assert_vec_close(&hit_coplanar.unwrap(), [1.0, 2.0, 5.0], EPS, "coplanar returns start");

    // Parallel non-coplanar line: z=0
    let line_parallel = Line3::new(Vector3::new(1.0, 2.0, 0.0), Vector3::new(3.0, 4.0, 0.0));
    assert!(plane.intersect_line(&line_parallel, true).is_none(), "parallel line returns none");
}

#[test]
fn test_plane_apply_matrix4_and_translate() {
    let mut plane = Plane::new(Vector3::new(1.0, 0.0, 0.0), 0.0);

    // Translate by (3, 4, 5)
    plane.translate(&Vector3::new(3.0, 4.0, 5.0));
    assert_vec_close(&plane.normal, [1.0, 0.0, 0.0], EPS, "normal unchanged after translate");
    assert_close(plane.constant, -3.0, EPS, "constant updated after translate");
    assert_close(plane.distance_to_point(&Vector3::new(3.0, 0.0, 0.0)), 0.0, EPS, "translated plane distance 0");

    // Transform plane z = 2 (normal = (0, 0, 1), constant = -2)
    // by Matrix4: translation (10, 0, 0) and scale 2 on all axes
    let mut plane_z = Plane::new(Vector3::new(0.0, 0.0, 1.0), -2.0);
    let mut m = Matrix4::identity();
    m.elements[0] = 2.0;
    m.elements[5] = 2.0;
    m.elements[10] = 2.0;
    m.elements[12] = 10.0;

    plane_z.apply_matrix4(&m, None);
    // Scaled z=2 becomes z=4; normal remains (0, 0, 1), constant = -4
    assert_vec_close(&plane_z.normal, [0.0, 0.0, 1.0], EPS, "transformed plane normal");
    assert_close(plane_z.constant, -4.0, EPS, "transformed plane constant");
}

#[test]
fn test_ray_at_closest_point_and_distance() {
    let ray = Ray::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 1.0, 0.0));

    // at
    assert_vec_close(&ray.at(5.0), [1.0, 7.0, 3.0], EPS, "ray at 5.0");

    // closest_point_to_point
    assert_vec_close(&ray.closest_point_to_point(&Vector3::new(1.0, 5.0, 3.0)), [1.0, 5.0, 3.0], EPS, "closest on ray");
    // point behind ray origin clamps to origin
    assert_vec_close(&ray.closest_point_to_point(&Vector3::new(1.0, -5.0, 3.0)), [1.0, 2.0, 3.0], EPS, "point behind clamps to origin");

    // distance_sq_to_point
    assert_close(ray.distance_sq_to_point(&Vector3::new(4.0, 5.0, 3.0)), 9.0, EPS, "distance squared");
    assert_close(ray.distance_to_point(&Vector3::new(4.0, 5.0, 3.0)), 3.0, EPS, "distance");
}

#[test]
fn test_ray_intersect_sphere() {
    let ray = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let sphere = Sphere::new(Vector3::new(5.0, 0.0, 0.0), 2.0);

    // Front intersection at t = 3.0
    let hit = ray.intersect_sphere(&sphere);
    assert!(hit.is_some(), "sphere hit");
    assert_vec_close(&hit.unwrap(), [3.0, 0.0, 0.0], EPS, "sphere hit point");
    assert!(ray.intersects_sphere(&sphere), "intersects_sphere predicate");

    // Origin inside sphere (at x=4): returns exit point at x=7
    let ray_inside = Ray::new(Vector3::new(4.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let exit_hit = ray_inside.intersect_sphere(&sphere);
    assert!(exit_hit.is_some(), "inside sphere exit hit");
    assert_vec_close(&exit_hit.unwrap(), [7.0, 0.0, 0.0], EPS, "inside sphere exit point");

    // Ray directed away
    let ray_away = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0));
    assert!(ray_away.intersect_sphere(&sphere).is_none(), "ray pointing away misses");

    // Ray missing sphere
    let ray_miss = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
    assert!(ray_miss.intersect_sphere(&sphere).is_none(), "ray missing sphere");
    assert!(!ray_miss.intersects_sphere(&sphere), "intersects_sphere false on miss");

    // Empty sphere
    assert!(ray.intersect_sphere(&Sphere::empty()).is_none(), "empty sphere returns none");
}

#[test]
fn test_ray_intersect_plane() {
    let ray = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
    let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), -10.0); // z = 10

    let hit = ray.intersect_plane(&plane);
    assert!(hit.is_some(), "plane hit");
    assert_vec_close(&hit.unwrap(), [0.0, 0.0, 10.0], EPS, "plane hit point");
    assert!(ray.intersects_plane(&plane), "intersects_plane true");

    // Plane behind ray
    let plane_behind = Plane::new(Vector3::new(0.0, 0.0, 1.0), 10.0); // z = -10
    assert!(ray.intersect_plane(&plane_behind).is_none(), "plane behind ray returns none");
    assert!(!ray.intersects_plane(&plane_behind), "intersects_plane false for plane behind");
}

#[test]
fn test_ray_intersect_box_hit_and_miss() {
    let box3 = Box3::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));

    // Hit from outside
    let ray_hit = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
    let hit = ray_hit.intersect_box(&box3);
    assert!(hit.is_some(), "box hit from outside");
    assert_vec_close(&hit.unwrap(), [0.0, 0.0, -1.0], EPS, "box hit entrance point");
    assert!(ray_hit.intersects_box(&box3), "intersects_box true");

    // Hit from inside
    let ray_inside = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
    let hit_inside = ray_inside.intersect_box(&box3);
    assert!(hit_inside.is_some(), "box hit from inside");
    assert_vec_close(&hit_inside.unwrap(), [0.0, 0.0, 1.0], EPS, "box hit exit point");

    // MISS 1: Ray misses in Y
    let ray_miss_y = Ray::new(Vector3::new(0.0, 2.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
    assert!(ray_miss_y.intersect_box(&box3).is_none(), "ray missing box in Y returns none");
    assert!(!ray_miss_y.intersects_box(&box3), "intersects_box false on miss");

    // MISS 2: Ray pointing away
    let ray_away = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, -1.0));
    assert!(ray_away.intersect_box(&box3).is_none(), "ray pointing away returns none");

    // MISS 3: Ray parallel to faces outside box bounds (invdirx=inf, invdiry=inf, origin=(2, 2, -5))
    let ray_parallel = Ray::new(Vector3::new(2.0, 2.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
    assert!(ray_parallel.intersect_box(&box3).is_none(), "parallel ray outside box returns none");

    // Empty box
    assert!(ray_hit.intersect_box(&Box3::empty()).is_none(), "empty box returns none");
}

#[test]
fn test_ray_intersect_triangle_hit_backface_culled_and_miss() {
    let a = Vector3::new(1.0, 1.0, 0.0);
    let b = Vector3::new(0.0, 1.0, 1.0);
    let c = Vector3::new(1.0, 0.0, 1.0);

    // Front-facing hit (normal points toward negative octant, ray direction (1, 1, 1) enters front)
    let mut dir = Vector3::new(1.0, 1.0, 1.0);
    dir.normalize();
    let ray = Ray::new(Vector3::zero(), dir);

    // Non-culled hit
    let hit = ray.intersect_triangle(&a, &b, &c, false);
    assert!(hit.is_some(), "front face non-culled hit");
    assert_vec_close(&hit.unwrap(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 1e-6, "triangle hit point");

    // Culled hit: ray hitting from backface with backfaceCulling = true
    let mut back_dir = Vector3::new(-1.0, -1.0, -1.0);
    back_dir.normalize();
    let back_ray = Ray::new(Vector3::new(2.0, 2.0, 2.0), back_dir);
    let hit_culled = back_ray.intersect_triangle(&a, &b, &c, true);
    assert!(hit_culled.is_none(), "backface hit culled when backfaceCulling is true");

    let hit_not_culled = back_ray.intersect_triangle(&a, &b, &c, false);
    assert!(hit_not_culled.is_some(), "backface hit detected when backfaceCulling is false");
    assert_vec_close(&hit_not_culled.unwrap(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 1e-6, "backface hit point");

    // MISS: Ray pointing wrong direction
    let mut miss_dir = Vector3::new(-1.0, -1.0, -1.0);
    miss_dir.normalize();
    let miss_ray = Ray::new(Vector3::zero(), miss_dir);
    assert!(miss_ray.intersect_triangle(&a, &b, &c, false).is_none(), "ray pointing away misses triangle");

    // MISS: Degenerate collinear triangle
    let deg_a = Vector3::new(0.0, 0.0, 0.0);
    let deg_b = Vector3::new(1.0, 1.0, 1.0);
    let deg_c = Vector3::new(2.0, 2.0, 2.0);
    assert!(ray.intersect_triangle(&deg_a, &deg_b, &deg_c, false).is_none(), "degenerate triangle returns none");
}

#[test]
fn test_triangle_get_normal_and_barycoord_and_degenerate() {
    let a = Vector3::new(0.0, 0.0, 0.0);
    let b = Vector3::new(2.0, 0.0, 0.0);
    let c = Vector3::new(0.0, 2.0, 0.0);
    let tri = Triangle::new(a, b, c);

    // Normal: CCW (c - b) x (a - b) = (-2, 2, 0) x (-2, 0, 0) = (0, 0, 4) -> (0, 0, 1)
    let normal = tri.get_normal();
    assert_vec_close(&normal, [0.0, 0.0, 1.0], EPS, "triangle normal");

    // Area: 0.5 * 2 * 2 = 2.0
    assert_close(tri.get_area(), 2.0, EPS, "triangle area");

    // Midpoint: (2/3, 2/3, 0)
    assert_vec_close(&tri.get_midpoint(), [2.0 / 3.0, 2.0 / 3.0, 0.0], EPS, "triangle midpoint");

    // Barycentric coordinates:
    // a -> (1, 0, 0)
    let bary_a = tri.get_barycoord(&a);
    assert!(bary_a.is_some(), "barycoord of a");
    assert_vec_close(&bary_a.unwrap(), [1.0, 0.0, 0.0], EPS, "barycoord a");

    // b -> (0, 1, 0)
    let bary_b = tri.get_barycoord(&b);
    assert!(bary_b.is_some(), "barycoord of b");
    assert_vec_close(&bary_b.unwrap(), [0.0, 1.0, 0.0], EPS, "barycoord b");

    // c -> (0, 0, 1)
    let bary_c = tri.get_barycoord(&c);
    assert!(bary_c.is_some(), "barycoord of c");
    assert_vec_close(&bary_c.unwrap(), [0.0, 0.0, 1.0], EPS, "barycoord c");

    // contains_point
    assert!(tri.contains_point(&Vector3::new(0.5, 0.5, 0.0)), "inside point");
    assert!(tri.contains_point(&Vector3::new(0.0, 0.0, 0.0)), "vertex point");
    assert!(!tri.contains_point(&Vector3::new(2.0, 2.0, 0.0)), "outside point");

    // Degenerate triangle (collinear)
    let deg_tri = Triangle::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 0.0),
        Vector3::new(2.0, 2.0, 0.0),
    );
    assert_vec_close(&deg_tri.get_normal(), [0.0, 0.0, 0.0], EPS, "degenerate normal is zero");
    assert!(deg_tri.get_barycoord(&Vector3::new(1.0, 1.0, 0.0)).is_none(), "degenerate barycoord is None");
    assert!(!deg_tri.contains_point(&Vector3::new(1.0, 1.0, 0.0)), "degenerate contains_point is false");
}

#[test]
fn test_triangle_closest_point_region_walk_and_front_facing() {
    let tri = Triangle::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(4.0, 0.0, 0.0),
        Vector3::new(0.0, 4.0, 0.0),
    );

    // Region A: closest is vertex A
    assert_vec_close(&tri.closest_point_to_point(&Vector3::new(-2.0, -2.0, 0.0)), [0.0, 0.0, 0.0], EPS, "region A");

    // Region B: closest is vertex B
    assert_vec_close(&tri.closest_point_to_point(&Vector3::new(6.0, -1.0, 0.0)), [4.0, 0.0, 0.0], EPS, "region B");

    // Region C: closest is vertex C
    assert_vec_close(&tri.closest_point_to_point(&Vector3::new(-1.0, 6.0, 0.0)), [0.0, 4.0, 0.0], EPS, "region C");

    // Region AB: closest is on edge AB
    assert_vec_close(&tri.closest_point_to_point(&Vector3::new(2.0, -3.0, 0.0)), [2.0, 0.0, 0.0], EPS, "region AB");

    // Region AC: closest is on edge AC
    assert_vec_close(&tri.closest_point_to_point(&Vector3::new(-3.0, 2.0, 0.0)), [0.0, 2.0, 0.0], EPS, "region AC");

    // Region BC: closest is on edge BC (hypotenuse x + y = 4)
    assert_vec_close(&tri.closest_point_to_point(&Vector3::new(3.0, 3.0, 0.0)), [2.0, 2.0, 0.0], EPS, "region BC");

    // Face region: closest is projection onto triangle plane
    assert_vec_close(&tri.closest_point_to_point(&Vector3::new(1.0, 1.0, 5.0)), [1.0, 1.0, 0.0], EPS, "face region");

    // is_front_facing: normal is (0, 0, 1). Direction (0, 0, -1) points against normal -> front facing
    assert!(tri.is_front_facing(&Vector3::new(0.0, 0.0, -1.0)), "opposing direction is front facing");
    assert!(!tri.is_front_facing(&Vector3::new(0.0, 0.0, 1.0)), "co-aligned direction is not front facing");
}

#[test]
fn test_box3_and_sphere_intersects_plane_and_triangle() {
    let plane_z0 = Plane::new(Vector3::new(0.0, 0.0, 1.0), 0.0); // z = 0

    // Box intersecting plane z=0
    let box_crossing = Box3::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
    assert!(box_crossing.intersects_plane(&plane_z0), "box crossing plane intersects");
    assert!(plane_z0.intersects_box(&box_crossing), "reciprocal plane intersects box");

    // Box separated from plane z=0
    let box_separated = Box3::new(Vector3::new(2.0, 2.0, 2.0), Vector3::new(4.0, 4.0, 4.0));
    assert!(!box_separated.intersects_plane(&plane_z0), "separated box does not intersect");
    assert!(!plane_z0.intersects_box(&box_separated), "reciprocal plane separated box");

    // Sphere intersecting plane z=0
    let sphere_crossing = Sphere::new(Vector3::new(0.0, 0.0, 3.0), 4.0); // dist=3 <= r=4
    assert!(sphere_crossing.intersects_plane(&plane_z0), "sphere crossing plane");
    assert!(plane_z0.intersects_sphere(&sphere_crossing), "reciprocal plane crossing sphere");

    // Sphere separated from plane z=0
    let sphere_separated = Sphere::new(Vector3::new(0.0, 0.0, 5.0), 2.0); // dist=5 > r=2
    assert!(!sphere_separated.intersects_plane(&plane_z0), "sphere separated from plane");
    assert!(!plane_z0.intersects_sphere(&sphere_separated), "reciprocal plane separated sphere");

    // Triangle intersecting box
    let tri_crossing = Triangle::new(
        Vector3::new(-2.0, 0.0, 0.0),
        Vector3::new(2.0, 0.0, 0.0),
        Vector3::new(0.0, 2.0, 0.0),
    );
    assert!(box_crossing.intersects_triangle(&tri_crossing), "triangle crossing box intersects");
    assert!(tri_crossing.intersects_box(&box_crossing), "reciprocal triangle intersects box");

    // Triangle far away from box
    let tri_far = Triangle::new(
        Vector3::new(10.0, 10.0, 10.0),
        Vector3::new(12.0, 10.0, 10.0),
        Vector3::new(10.0, 12.0, 10.0),
    );
    assert!(!box_crossing.intersects_triangle(&tri_far), "distant triangle does not intersect box");
    assert!(!tri_far.intersects_box(&box_crossing), "reciprocal distant triangle");
}
