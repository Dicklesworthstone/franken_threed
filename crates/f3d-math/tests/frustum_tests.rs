//! Conformance, differential, and regression tests for `Frustum` (§7.5, roa.1).
//!
//! Verifies analytical and numerical equivalence against upstream Three.js r186
//! `src/math/Frustum.js` (commit `148ef33ecb6d2502ff796d4554abd1549c95d519`).

use f3d_math::box3::Box3;
use f3d_math::frustum::Frustum;
use f3d_math::matrix4::{CoordinateSystem, Matrix4};
use f3d_math::plane::Plane;
use f3d_math::sphere::Sphere;
use f3d_math::vector3::Vector3;

#[test]
fn test_frustum_constructors_and_defaults() {
    let def = Frustum::default();
    for i in 0..6 {
        assert_eq!(def.planes[i], Plane::default());
    }

    let p0 = Plane::new(Vector3::new(1.0, 0.0, 0.0), 10.0);
    let p1 = Plane::new(Vector3::new(-1.0, 0.0, 0.0), 10.0);
    let p2 = Plane::new(Vector3::new(0.0, 1.0, 0.0), 10.0);
    let p3 = Plane::new(Vector3::new(0.0, -1.0, 0.0), 10.0);
    let p4 = Plane::new(Vector3::new(0.0, 0.0, 1.0), 10.0);
    let p5 = Plane::new(Vector3::new(0.0, 0.0, -1.0), 10.0);

    let f = Frustum::new(p0, p1, p2, p3, p4, p5);
    assert_eq!(*f.right(), p0);
    assert_eq!(*f.left(), p1);
    assert_eq!(*f.bottom(), p2);
    assert_eq!(*f.top(), p3);
    assert_eq!(*f.far(), p4);
    assert_eq!(*f.near(), p5);

    // Test set
    let mut f_set = Frustum::default();
    f_set.set(&p0, &p1, &p2, &p3, &p4, &p5);
    assert_eq!(f_set, f);

    // Test copy
    let mut f_copy = Frustum::default();
    f_copy.copy(&f);
    assert_eq!(f_copy, f);

    // Display formatting
    let s = format!("{}", f);
    assert!(s.starts_with("Frustum("));
}

#[test]
fn test_frustum_set_from_projection_matrix_webgpu_non_reversed() {
    // Standard perspective projection: frustum from z = -1 to z = -100, fov = 90 deg (left=-1, right=1, top=1, bottom=-1)
    let mut proj = Matrix4::identity();
    proj.make_perspective(-1.0, 1.0, 1.0, -1.0, 1.0, 100.0, CoordinateSystem::WebGPU, false);

    let mut frustum = Frustum::default();
    frustum.set_from_projection_matrix(&proj, CoordinateSystem::WebGPU, false);

    // Expected planes matching Three.js r186 Node oracle:
    // Plane 0 (Right): normal ~ (-sqrt(0.5), 0, -sqrt(0.5)), constant ~ 0
    let inv_sqrt2 = (0.5_f64).sqrt();
    assert!((frustum.planes[0].normal.x - (-inv_sqrt2)).abs() < 1e-12);
    assert_eq!(frustum.planes[0].normal.y, 0.0);
    assert!((frustum.planes[0].normal.z - (-inv_sqrt2)).abs() < 1e-12);
    assert_eq!(frustum.planes[0].constant, 0.0);

    // Plane 1 (Left): normal ~ (sqrt(0.5), 0, -sqrt(0.5)), constant ~ 0
    assert!((frustum.planes[1].normal.x - inv_sqrt2).abs() < 1e-12);
    assert_eq!(frustum.planes[1].normal.y, 0.0);
    assert!((frustum.planes[1].normal.z - (-inv_sqrt2)).abs() < 1e-12);
    assert_eq!(frustum.planes[1].constant, 0.0);

    // Plane 2 (Bottom): normal ~ (0, sqrt(0.5), -sqrt(0.5)), constant ~ 0
    assert_eq!(frustum.planes[2].normal.x, 0.0);
    assert!((frustum.planes[2].normal.y - inv_sqrt2).abs() < 1e-12);
    assert!((frustum.planes[2].normal.z - (-inv_sqrt2)).abs() < 1e-12);
    assert_eq!(frustum.planes[2].constant, 0.0);

    // Plane 3 (Top): normal ~ (0, -sqrt(0.5), -sqrt(0.5)), constant ~ 0
    assert_eq!(frustum.planes[3].normal.x, 0.0);
    assert!((frustum.planes[3].normal.y - (-inv_sqrt2)).abs() < 1e-12);
    assert!((frustum.planes[3].normal.z - (-inv_sqrt2)).abs() < 1e-12);
    assert_eq!(frustum.planes[3].constant, 0.0);

    // Plane 4 (Far): normal = (0, 0, 1), constant ~ 100
    assert_eq!(frustum.planes[4].normal.x, 0.0);
    assert_eq!(frustum.planes[4].normal.y, 0.0);
    assert!((frustum.planes[4].normal.z - 1.0).abs() < 1e-12);
    assert!((frustum.planes[4].constant - 100.0).abs() < 1e-10);

    // Plane 5 (Near): normal = (0, 0, -1), constant = -1
    assert_eq!(frustum.planes[5].normal.x, 0.0);
    assert_eq!(frustum.planes[5].normal.y, 0.0);
    assert_eq!(frustum.planes[5].normal.z, -1.0);
    assert_eq!(frustum.planes[5].constant, -1.0);

    // Factory method equivalence
    let f_factory = Frustum::from_projection_matrix(&proj, CoordinateSystem::WebGPU, false);
    assert_eq!(f_factory, frustum);

    // Point containment
    assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -50.0)), "center must be inside");
    assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -1.0)), "point on near plane must be inside");
    assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -99.99)), "point just inside far plane");
    assert!(!frustum.contains_point(&Vector3::new(0.0, 0.0, -0.5)), "point in front of near plane is outside");
    assert!(!frustum.contains_point(&Vector3::new(0.0, 0.0, -150.0)), "point beyond far plane is outside");
    assert!(!frustum.contains_point(&Vector3::new(100.0, 0.0, -50.0)), "point outside right plane");
    assert!(!frustum.contains_point(&Vector3::new(-100.0, 0.0, -50.0)), "point outside left plane");
    assert!(!frustum.contains_point(&Vector3::new(0.0, 100.0, -50.0)), "point outside top plane");
    assert!(!frustum.contains_point(&Vector3::new(0.0, -100.0, -50.0)), "point outside bottom plane");
}

#[test]
fn test_frustum_set_from_projection_matrix_webgl_non_reversed() {
    let mut proj = Matrix4::identity();
    proj.make_perspective(-1.0, 1.0, 1.0, -1.0, 1.0, 100.0, CoordinateSystem::WebGL, false);

    let mut frustum = Frustum::default();
    frustum.set_from_projection_matrix(&proj, CoordinateSystem::WebGL, false);

    // WebGL depth range [-1, 1]:
    // Near plane uses row3 + row2 normalized -> normal = (0, 0, -1), constant ~ -1.0
    assert_eq!(frustum.planes[5].normal.x, 0.0);
    assert_eq!(frustum.planes[5].normal.y, 0.0);
    assert!((frustum.planes[5].normal.z - (-1.0)).abs() < 1e-12);
    assert!((frustum.planes[5].constant - (-1.0)).abs() < 1e-12);

    // Far plane uses row3 - row2 normalized -> normal = (0, 0, 1), constant ~ 100.0
    assert_eq!(frustum.planes[4].normal.x, 0.0);
    assert_eq!(frustum.planes[4].normal.y, 0.0);
    assert!((frustum.planes[4].normal.z - 1.0).abs() < 1e-12);
    assert!((frustum.planes[4].constant - 100.0).abs() < 1e-10);

    assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -50.0)));
    assert!(!frustum.contains_point(&Vector3::new(0.0, 0.0, -0.5)));
    assert!(!frustum.contains_point(&Vector3::new(0.0, 0.0, -150.0)));
}

#[test]
fn test_frustum_set_from_projection_matrix_reversed_depth() {
    for coord in [CoordinateSystem::WebGL, CoordinateSystem::WebGPU] {
        let mut proj = Matrix4::identity();
        proj.make_perspective(-1.0, 1.0, 1.0, -1.0, 1.0, 100.0, coord, true);

        let mut frustum = Frustum::default();
        frustum.set_from_projection_matrix(&proj, coord, true);

        // Under reversed depth:
        // Plane 4 (Far) uses row2 normalized -> normal = (0, 0, 1), constant ~ 100.0
        assert!((frustum.planes[4].normal.z - 1.0).abs() < 1e-12);
        assert!((frustum.planes[4].constant - 100.0).abs() < 1e-10);

        // Plane 5 (Near) uses row3 - row2 normalized -> normal = (0, 0, -1), constant = -1.0
        assert!((frustum.planes[5].normal.z - (-1.0)).abs() < 1e-12);
        assert!((frustum.planes[5].constant - (-1.0)).abs() < 1e-10);

        assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -50.0)));
        assert!(!frustum.contains_point(&Vector3::new(0.0, 0.0, -0.5)));
        assert!(!frustum.contains_point(&Vector3::new(0.0, 0.0, -150.0)));
    }
}

#[test]
fn test_frustum_orthographic() {
    let mut proj = Matrix4::identity();
    proj.make_orthographic(-10.0, 10.0, 10.0, -10.0, 1.0, 100.0, CoordinateSystem::WebGPU, false);

    let mut frustum = Frustum::default();
    frustum.set_from_projection_matrix(&proj, CoordinateSystem::WebGPU, false);

    // Box volume [-10, 10] x [-10, 10] x [-100, -1]
    assert_eq!(frustum.planes[0].normal, Vector3::new(-1.0, 0.0, 0.0));
    assert_eq!(frustum.planes[0].constant, 10.0);

    assert_eq!(frustum.planes[1].normal, Vector3::new(1.0, 0.0, 0.0));
    assert_eq!(frustum.planes[1].constant, 10.0);

    assert_eq!(frustum.planes[2].normal, Vector3::new(0.0, 1.0, 0.0));
    assert_eq!(frustum.planes[2].constant, 10.0);

    assert_eq!(frustum.planes[3].normal, Vector3::new(0.0, -1.0, 0.0));
    assert_eq!(frustum.planes[3].constant, 10.0);

    // Exact Three.js r186 floating-point normalization oracle values:
    assert_eq!(frustum.planes[4].normal, Vector3::new(0.0, 0.0, 0.9999999999999999));
    assert_eq!(frustum.planes[4].constant, 99.99999999999999);

    assert_eq!(frustum.planes[5].normal, Vector3::new(0.0, 0.0, -0.9999999999999999));
    assert_eq!(frustum.planes[5].constant, -0.9999999999999999);

    assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -50.0)));
    assert!(frustum.contains_point(&Vector3::new(9.0, 9.0, -50.0)));
    assert!(!frustum.contains_point(&Vector3::new(11.0, 0.0, -50.0)));
    assert!(!frustum.contains_point(&Vector3::new(-11.0, 0.0, -50.0)));
    assert!(!frustum.contains_point(&Vector3::new(0.0, 11.0, -50.0)));
    assert!(!frustum.contains_point(&Vector3::new(0.0, -11.0, -50.0)));
}

#[test]
fn test_frustum_contains_point_and_nonfinite_oracle() {
    let mut proj = Matrix4::identity();
    proj.make_perspective(-1.0, 1.0, 1.0, -1.0, 1.0, 100.0, CoordinateSystem::WebGPU, false);
    let frustum = Frustum::from_projection_matrix(&proj, CoordinateSystem::WebGPU, false);

    assert!(frustum.contains_point(&Vector3::new(0.0, 0.0, -50.0)));

    // Three.js r186 IEEE 754 non-finite behavior:
    // In Three.js: `distanceToPoint(NaN)` evaluates to NaN, and `NaN < 0` is false!
    // Therefore containsPoint(NaN, 0, -50) returns true in Three.js!
    assert!(
        frustum.contains_point(&Vector3::new(f64::NAN, 0.0, -50.0)),
        "NaN coordinate produces NaN distance where `dist < 0` is false, matching Three.js"
    );

    // Infinity coordinate: normal.x * Inf = -0.707 * Inf = -Inf, -Inf < 0 is true, returns false
    assert!(
        !frustum.contains_point(&Vector3::new(f64::INFINITY, 0.0, -50.0)),
        "Infinity coordinate evaluates distance -Inf < 0 returning false"
    );
}

#[test]
fn test_frustum_intersects_sphere() {
    let mut proj = Matrix4::identity();
    proj.make_perspective(-1.0, 1.0, 1.0, -1.0, 1.0, 100.0, CoordinateSystem::WebGPU, false);
    let frustum = Frustum::from_projection_matrix(&proj, CoordinateSystem::WebGPU, false);

    // 1. Sphere completely inside
    let s_inside = Sphere::new(Vector3::new(0.0, 0.0, -50.0), 5.0);
    assert!(frustum.intersects_sphere(&s_inside));

    // 2. Sphere completely outside near plane
    let s_outside_near = Sphere::new(Vector3::new(0.0, 0.0, 10.0), 5.0);
    assert!(!frustum.intersects_sphere(&s_outside_near));

    // 3. Sphere straddling near plane (center at 0,0,0, radius 2.0 reaches z = -2.0 through near plane at z = -1.0)
    let s_straddle_near = Sphere::new(Vector3::new(0.0, 0.0, 0.0), 2.0);
    assert!(frustum.intersects_sphere(&s_straddle_near));

    // 4. Sphere touching boundary plane exactly: center at (0, 0, 0), radius 1.0.
    // Distance to near plane (0,0,-1, -1) is -1.0; neg_radius is -1.0.
    // In Three.js: `distance < negRadius` (-1.0 < -1.0) is false, returning true!
    let s_boundary = Sphere::new(Vector3::new(0.0, 0.0, 0.0), 1.0);
    assert!(frustum.intersects_sphere(&s_boundary), "boundary tangent sphere must intersect");

    // 5. Zero-radius point sphere inside and outside
    let s_point_inside = Sphere::new(Vector3::new(0.0, 0.0, -50.0), 0.0);
    assert!(frustum.intersects_sphere(&s_point_inside));

    let s_point_outside = Sphere::new(Vector3::new(0.0, 0.0, 0.5), 0.0);
    assert!(!frustum.intersects_sphere(&s_point_outside));

    // 6. Empty sphere (center 0,0,0, radius -1.0 matching Three.js default)
    let empty_sphere = Sphere::empty();
    assert!(!frustum.intersects_sphere(&empty_sphere), "empty sphere must not intersect frustum");
}

#[test]
fn test_frustum_intersects_box() {
    let mut proj = Matrix4::identity();
    proj.make_perspective(-1.0, 1.0, 1.0, -1.0, 1.0, 100.0, CoordinateSystem::WebGPU, false);
    let frustum = Frustum::from_projection_matrix(&proj, CoordinateSystem::WebGPU, false);

    // 1. Box completely inside
    let b_inside = Box3::new(Vector3::new(-1.0, -1.0, -55.0), Vector3::new(1.0, 1.0, -45.0));
    assert!(frustum.intersects_box(&b_inside));

    // 2. Box completely outside near plane
    let b_outside_near = Box3::new(Vector3::new(-1.0, -1.0, 10.0), Vector3::new(1.0, 1.0, 20.0));
    assert!(!frustum.intersects_box(&b_outside_near));

    // 3. Box completely outside far plane
    let b_outside_far = Box3::new(Vector3::new(-1.0, -1.0, -200.0), Vector3::new(1.0, 1.0, -150.0));
    assert!(!frustum.intersects_box(&b_outside_far));

    // 4. Box straddling far plane
    let b_straddle_far = Box3::new(Vector3::new(-1.0, -1.0, -110.0), Vector3::new(1.0, 1.0, -90.0));
    assert!(frustum.intersects_box(&b_straddle_far));

    // 5. Box straddling near plane
    let b_straddle_near = Box3::new(Vector3::new(-1.0, -1.0, -2.0), Vector3::new(1.0, 1.0, 0.0));
    assert!(frustum.intersects_box(&b_straddle_near));

    // 6. Boxes completely outside on lateral axes
    let b_outside_right = Box3::new(Vector3::new(100.0, -1.0, -50.0), Vector3::new(110.0, 1.0, -45.0));
    assert!(!frustum.intersects_box(&b_outside_right));

    let b_outside_left = Box3::new(Vector3::new(-110.0, -1.0, -50.0), Vector3::new(-100.0, 1.0, -45.0));
    assert!(!frustum.intersects_box(&b_outside_left));

    let b_outside_top = Box3::new(Vector3::new(-1.0, 100.0, -50.0), Vector3::new(1.0, 110.0, -45.0));
    assert!(!frustum.intersects_box(&b_outside_top));

    let b_outside_bottom = Box3::new(Vector3::new(-1.0, -110.0, -50.0), Vector3::new(1.0, -100.0, -45.0));
    assert!(!frustum.intersects_box(&b_outside_bottom));

    // 7. Empty box (min = +Inf, max = -Inf matching Three.js default)
    // In Three.js: dot products evaluate to NaN, `NaN < 0` is false, returning true!
    let empty_box = Box3::empty();
    assert!(
        frustum.intersects_box(&empty_box),
        "empty box evaluates NaN distance where `dist < 0` is false, returning true matching Three.js r186"
    );
}
