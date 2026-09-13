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

    // (b-a) cross (c-a) points toward the positive octant. The origin ray
    // therefore hits the back face, as verified against pinned Three.js Ray.
    let mut dir = Vector3::new(1.0, 1.0, 1.0);
    dir.normalize();
    let ray = Ray::new(Vector3::zero(), dir);

    // Non-culled hit
    let hit = ray.intersect_triangle(&a, &b, &c, false);
    assert!(hit.is_some(), "back face non-culled hit");
    assert_vec_close(&hit.unwrap(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 1e-6, "triangle hit point");
    assert!(ray.intersect_triangle(&a, &b, &c, true).is_none(),
        "back face must be culled");

    // Approaching from the positive octant hits the front face and survives culling.
    let mut front_dir = Vector3::new(-1.0, -1.0, -1.0);
    front_dir.normalize();
    let front_ray = Ray::new(Vector3::new(2.0, 2.0, 2.0), front_dir);
    let front_hit = front_ray.intersect_triangle(&a, &b, &c, true);
    assert!(front_hit.is_some(), "front face survives backface culling");
    assert_vec_close(&front_hit.unwrap(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 1e-6, "front face hit point");

    let hit_not_culled = front_ray.intersect_triangle(&a, &b, &c, false);
    assert!(hit_not_culled.is_some(), "front face hit without culling");
    assert_vec_close(&hit_not_culled.unwrap(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 1e-6, "front face hit point");

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

#[test]
fn test_fixture_tri1_pixel_center_ndc_mapping_and_containment() {
    // Fixture tri1 in NDC space:
    // NDC vertices (-1,-1), (0,-1), (0,1) with z=0
    let tri1 = Triangle::new(
        Vector3::new(-1.0, -1.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    );

    // Viewport dimensions: 64x64
    let width = 64.0_f64;
    let height = 64.0_f64;

    // Pinned mapping from pixel-center coordinates (px, py) to WebGPU/Three.js NDC:
    // x_ndc = (px / width) * 2.0 - 1.0
    // y_ndc = 1.0 - (py / height) * 2.0
    let pixel_center_to_ndc = |px: f64, py: f64| -> Vector3 {
        let x_ndc = (px / width) * 2.0 - 1.0;
        let y_ndc = 1.0 - (py / height) * 2.0;
        Vector3::new(x_ndc, y_ndc, 0.0)
    };

    // Sample Point 1: Left triangle interior at pixel (24, 32), center (24.5, 32.5)
    // Expected NDC: x = (24.5/64)*2 - 1 = -0.234375, y = 1 - (32.5/64)*2 = -0.015625
    let p_inside = pixel_center_to_ndc(24.5, 32.5);
    assert_vec_close(&p_inside, [-0.234375, -0.015625, 0.0], EPS, "p_inside NDC mapping");
    assert!(
        tri1.contains_point(&p_inside),
        "pixel center (24.5, 32.5) at NDC (-0.234375, -0.015625) must be inside tri1"
    );
    let bary_inside = tri1.get_barycoord(&p_inside).expect("tri1 is non-degenerate");
    assert!(bary_inside.x > 0.0 && bary_inside.y > 0.0 && bary_inside.z > 0.0);
    assert_close(bary_inside.x + bary_inside.y + bary_inside.z, 1.0, EPS, "barycentric sum");

    // Sample Point 2: Right side / outside tri1 at pixel (56, 32), center (56.5, 32.5)
    // Expected NDC: x = (56.5/64)*2 - 1 = +0.765625, y = 1 - (32.5/64)*2 = -0.015625
    let p_outside_right = pixel_center_to_ndc(56.5, 32.5);
    assert_vec_close(&p_outside_right, [0.765625, -0.015625, 0.0], EPS, "p_outside_right NDC mapping");
    assert!(
        !tri1.contains_point(&p_outside_right),
        "pixel center (56.5, 32.5) at NDC (0.765625, -0.015625) must be outside tri1"
    );

    // Sample Point 3: Background / top-left at pixel (2, 2), center (2.5, 2.5)
    // Expected NDC: x = (2.5/64)*2 - 1 = -0.921875, y = 1 - (2.5/64)*2 = +0.921875
    let p_outside_bg = pixel_center_to_ndc(2.5, 2.5);
    assert_vec_close(&p_outside_bg, [-0.921875, 0.921875, 0.0], EPS, "p_outside_bg NDC mapping");
    assert!(
        !tri1.contains_point(&p_outside_bg),
        "pixel center (2.5, 2.5) at NDC (-0.921875, 0.921875) must be outside tri1"
    );
}

#[test]
fn test_line3_distance_sq_triangle_interpolation_and_copy() {
    // 1. Line3 copy round-trip
    let l_orig = Line3::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
    let mut l_copy = Line3::default();
    l_copy.copy(&l_orig);
    assert!(l_copy.equals(&l_orig), "line3 copy equals");
    assert_vec_close(&l_copy.start, [1.0, 2.0, 3.0], EPS, "line3 copy start");
    assert_vec_close(&l_copy.end, [4.0, 5.0, 6.0], EPS, "line3 copy end");

    // 2. Line3 distance_sq_to_line3:
    // a) Parallel segments
    let l_par1 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0));
    let l_par2 = Line3::new(Vector3::new(0.0, 1.0, 0.0), Vector3::new(2.0, 1.0, 0.0));
    let mut c1 = Vector3::zero();
    let mut c2 = Vector3::zero();
    let d_par = l_par1.distance_sq_to_line3(&l_par2, Some(&mut c1), Some(&mut c2));
    assert_close(d_par, 1.0, EPS, "parallel distance_sq");
    assert_vec_close(&c1, [0.0, 0.0, 0.0], EPS, "parallel c1");
    assert_vec_close(&c2, [0.0, 1.0, 0.0], EPS, "parallel c2");

    // b) Skew segments
    let l_skew1 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0));
    let l_skew2 = Line3::new(Vector3::new(1.0, 1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
    let d_skew = l_skew1.distance_sq_to_line3(&l_skew2, Some(&mut c1), Some(&mut c2));
    assert_close(d_skew, 1.0, EPS, "skew distance_sq");
    assert_vec_close(&c1, [1.0, 0.0, 0.0], EPS, "skew c1");
    assert_vec_close(&c2, [1.0, 1.0, 0.0], EPS, "skew c2");

    // c) Segments sharing an endpoint
    let l_share1 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let l_share2 = Line3::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 0.0));
    let d_share = l_share1.distance_sq_to_line3(&l_share2, Some(&mut c1), Some(&mut c2));
    assert_close(d_share, 0.0, EPS, "shared endpoint distance_sq");
    assert_vec_close(&c1, [1.0, 0.0, 0.0], EPS, "shared endpoint c1");
    assert_vec_close(&c2, [1.0, 0.0, 0.0], EPS, "shared endpoint c2");

    // d) Degenerate zero-length segment
    let l_deg1 = Line3::new(Vector3::new(1.0, 2.0, 0.0), Vector3::new(1.0, 2.0, 0.0));
    let l_seg2 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0));
    let d_deg = l_deg1.distance_sq_to_line3(&l_seg2, Some(&mut c1), Some(&mut c2));
    assert_close(d_deg, 4.0, EPS, "degenerate segment distance_sq");
    assert_vec_close(&c1, [1.0, 2.0, 0.0], EPS, "degenerate segment c1");
    assert_vec_close(&c2, [1.0, 0.0, 0.0], EPS, "degenerate segment c2");

    // d2) Double-degenerate segments (both zero-length): verifies surprising c1 = p1 - p2 behavior
    let l_both_deg1 = Line3::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 2.0, 3.0));
    let l_both_deg2 = Line3::new(Vector3::new(4.0, 6.0, 3.0), Vector3::new(4.0, 6.0, 3.0));
    let d_both_deg = l_both_deg1.distance_sq_to_line3(&l_both_deg2, Some(&mut c1), Some(&mut c2));
    assert_close(d_both_deg, 25.0, EPS, "double degenerate distance_sq");
    assert_vec_close(&c1, [-3.0, -4.0, 0.0], EPS, "double degenerate c1 (diff)");
    assert_vec_close(&c2, [4.0, 6.0, 3.0], EPS, "double degenerate c2 (p2)");

    // d3) Endpoint clamp case (general branch where t < 0 clamps t=0 and re-clamps s)
    let l_clamp1 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0));
    let l_clamp2 = Line3::new(Vector3::new(3.0, 1.0, 0.0), Vector3::new(3.0, 3.0, 0.0));
    let d_clamp = l_clamp1.distance_sq_to_line3(&l_clamp2, Some(&mut c1), Some(&mut c2));
    assert_close(d_clamp, 2.0, EPS, "endpoint clamp distance_sq");
    assert_vec_close(&c1, [2.0, 0.0, 0.0], EPS, "endpoint clamp c1");
    assert_vec_close(&c2, [3.0, 1.0, 0.0], EPS, "endpoint clamp c2");

    // e) None targets call
    let d_none = l_skew1.distance_sq_to_line3(&l_skew2, None, None);
    assert_close(d_none, 1.0, EPS, "distance_sq with None targets");

    // 3. Triangle copy round-trip
    let t_orig = Triangle::new(
        Vector3::new(1.0, 2.0, 3.0),
        Vector3::new(4.0, 5.0, 6.0),
        Vector3::new(7.0, 8.0, 9.0),
    );
    let mut t_copy = Triangle::default();
    t_copy.copy(&t_orig);
    assert!(t_copy.equals(&t_orig), "triangle copy equals");
    assert_vec_close(&t_copy.a, [1.0, 2.0, 3.0], EPS, "triangle copy a");
    assert_vec_close(&t_copy.b, [4.0, 5.0, 6.0], EPS, "triangle copy b");
    assert_vec_close(&t_copy.c, [7.0, 8.0, 9.0], EPS, "triangle copy c");

    // 4. Triangle set_from_points_and_indices with non-sequential indices
    let points = [
        Vector3::new(10.0, 11.0, 12.0),
        Vector3::new(20.0, 21.0, 22.0),
        Vector3::new(30.0, 31.0, 32.0),
        Vector3::new(40.0, 41.0, 42.0),
        Vector3::new(50.0, 51.0, 52.0),
    ];
    let mut tri_indexed = Triangle::default();
    tri_indexed.set_from_points_and_indices(&points, 3, 0, 4);
    assert_vec_close(&tri_indexed.a, [40.0, 41.0, 42.0], EPS, "indexed a (index 3)");
    assert_vec_close(&tri_indexed.b, [10.0, 11.0, 12.0], EPS, "indexed b (index 0)");
    assert_vec_close(&tri_indexed.c, [50.0, 51.0, 52.0], EPS, "indexed c (index 4)");

    // 5. Triangle get_interpolation:
    // P = (0, 0, 0), (2, 0, 0), (0, 2, 0)
    let p1 = Vector3::new(0.0, 0.0, 0.0);
    let p2 = Vector3::new(2.0, 0.0, 0.0);
    let p3 = Vector3::new(0.0, 2.0, 0.0);
    let v1 = Vector3::new(10.0, 20.0, 30.0);
    let v2 = Vector3::new(40.0, 50.0, 60.0);
    let v3 = Vector3::new(70.0, 80.0, 90.0);
    let tri_interp = Triangle::new(p1, p2, p3);

    // a) Inside point: (0.5, 0.5, 0.0) -> bary (0.5, 0.25, 0.25) -> (32.5, 42.5, 52.5)
    let mut target = Vector3::zero();
    let pt_in = Vector3::new(0.5, 0.5, 0.0);
    let res_in = tri_interp.get_interpolation(&pt_in, &v1, &v2, &v3, &mut target);
    assert!(res_in.is_some(), "inside interpolation is Some");
    assert_vec_close(&target, [32.5, 42.5, 52.5], EPS, "inside interpolated target");
    assert_vec_close(&res_in.unwrap(), [32.5, 42.5, 52.5], EPS, "inside return value");

    // Static get_interpolation_of on inside point
    let mut target_static = Vector3::zero();
    let res_static = Triangle::get_interpolation_of(&pt_in, &p1, &p2, &p3, &v1, &v2, &v3, &mut target_static);
    assert!(res_static.is_some(), "static interpolation is Some");
    assert_vec_close(&target_static, [32.5, 42.5, 52.5], EPS, "static interpolated target");

    // b) Outside point: (2.0, 2.0, 0.0) -> bary (-1.0, 1.0, 1.0) -> (100.0, 110.0, 120.0)
    let pt_out = Vector3::new(2.0, 2.0, 0.0);
    let res_out = tri_interp.get_interpolation(&pt_out, &v1, &v2, &v3, &mut target);
    assert!(res_out.is_some(), "outside interpolation is Some");
    assert_vec_close(&target, [100.0, 110.0, 120.0], EPS, "outside interpolated target");

    // c) Degenerate (collinear) triangle: points (0,0,0), (1,1,1), (2,2,2)
    let dp1 = Vector3::new(0.0, 0.0, 0.0);
    let dp2 = Vector3::new(1.0, 1.0, 1.0);
    let dp3 = Vector3::new(2.0, 2.0, 2.0);
    let mut deg_target = Vector3::new(9.0, 9.0, 9.0);
    let pt_deg = Vector3::new(0.5, 0.5, 0.5);
    let res_deg = Triangle::get_interpolation_of(&pt_deg, &dp1, &dp2, &dp3, &v1, &v2, &v3, &mut deg_target);
    assert!(res_deg.is_none(), "degenerate interpolation is None");
    assert_vec_close(&deg_target, [0.0, 0.0, 0.0], EPS, "degenerate target zeroed");
}

#[test]
fn test_line3_closest_point_to_point_parameter_signed_zero_clamp() {
    // Line from (1, 1, 1) to (0, 0, 0): delta is (-1, -1, -1)
    let line = Line3::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(0.0, 0.0, 0.0));
    // Point at start (1, 1, 1): start_p = (0, 0, 0)
    // delta.dot(start_p) = (-1*0) + (-1*0) + (-1*0) = -0.0 + -0.0 + -0.0 = -0.0
    // delta_sq = 3.0
    // t = -0.0 / 3.0 = -0.0
    let point = Vector3::new(1.0, 1.0, 1.0);

    // Unclamped parameter evaluates to -0.0
    let t_unclamped = line.closest_point_to_point_parameter(&point, false);
    assert_eq!(t_unclamped.to_bits(), (-0.0f64).to_bits(), "unclamped t must be -0.0");
    assert!(t_unclamped.is_sign_negative(), "unclamped t has negative sign");

    // Clamped parameter must clamp -0.0 to +0.0 matching upstream Three.js Line3.js:154 / MathUtils.clamp
    let t_clamped = line.closest_point_to_point_parameter(&point, true);
    assert_eq!(t_clamped.to_bits(), (0.0f64).to_bits(), "clamped t must be +0.0 (bits 0)");
    assert!(!t_clamped.is_sign_negative(), "clamped t has positive sign");

    // Negative t outside [0, 1] also clamps to +0.0
    let line2 = Line3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let point2 = Vector3::new(-0.5, 0.0, 0.0);
    let t2_clamped = line2.closest_point_to_point_parameter(&point2, true);
    assert_eq!(t2_clamped.to_bits(), (0.0f64).to_bits(), "clamped negative t is +0.0 (bits 0)");
}
