//! Unit and regression tests for Ray distance_sq_to_segment matching Three.js r186.

use f3d_math::ray::Ray;
use f3d_math::vector3::Vector3;

#[test]
fn test_ray_distance_sq_to_segment_all_regions() {
    // Region 0: Interior points of both ray and segment
    let ray = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let v0 = Vector3::new(5.0, 2.0, -3.0);
    let v1 = Vector3::new(5.0, 2.0, 3.0);
    let mut pt_ray = Vector3::zero();
    let mut pt_seg = Vector3::zero();
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 4.0);
    assert_eq!(pt_ray, Vector3::new(5.0, 0.0, 0.0));
    assert_eq!(pt_seg, Vector3::new(5.0, 2.0, 0.0));

    // Region 1: s0 >= 0, s1 > ext_det (closest point at segment end)
    let v0 = Vector3::new(2.0, 3.0, 4.0);
    let v1 = Vector3::new(4.0, 3.0, 2.0);
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 13.0);
    assert_eq!(pt_ray, Vector3::new(4.0, 0.0, 0.0));
    assert_eq!(pt_seg, Vector3::new(4.0, 3.0, 2.0));

    // Region 5: s0 >= 0, s1 < -ext_det (closest point at segment start)
    let v0 = Vector3::new(4.0, 3.0, 2.0);
    let v1 = Vector3::new(2.0, 3.0, 4.0);
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 13.0);
    assert_eq!(pt_ray, Vector3::new(4.0, 0.0, 0.0));
    assert_eq!(pt_seg, Vector3::new(4.0, 3.0, 2.0));

    // Region 4: s0 < 0, s1 <= -ext_det (behind ray origin, segment end)
    let ray = Ray::new(Vector3::new(10.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let v0 = Vector3::new(2.0, 1.0, 0.0);
    let v1 = Vector3::new(4.0, 1.0, 0.0);
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 37.0);
    assert_eq!(pt_ray, Vector3::new(10.0, 0.0, 0.0));
    assert_eq!(pt_seg, Vector3::new(4.0, 1.0, 0.0));

    // Region 3: s0 < 0, -ext_det < s1 <= ext_det (behind ray origin, segment interior)
    let ray = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let v0 = Vector3::new(-5.0, 2.0, -2.0);
    let v1 = Vector3::new(-5.0, 2.0, 2.0);
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 29.0);
    assert_eq!(pt_ray, Vector3::new(0.0, 0.0, 0.0));
    assert_eq!(pt_seg, Vector3::new(-5.0, 2.0, 0.0));

    // Region 2: s0 < 0, s1 > ext_det (behind ray origin, segment start)
    let ray = Ray::new(Vector3::new(5.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let v0 = Vector3::new(2.0, 2.0, 0.0);
    let v1 = Vector3::new(0.0, 2.0, 0.0);
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 13.0);
    assert_eq!(pt_ray, Vector3::new(5.0, 0.0, 0.0));
    assert_eq!(pt_seg, Vector3::new(2.0, 2.0, 0.0));
}

#[test]
fn test_ray_distance_sq_to_segment_parallel_and_anti_parallel() {
    let ray = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));

    // Parallel same direction
    let v0 = Vector3::new(3.0, 4.0, 1.0);
    let v1 = Vector3::new(3.0, 4.0, 6.0);
    let mut pt_ray = Vector3::zero();
    let mut pt_seg = Vector3::zero();
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 25.0);
    assert_eq!(pt_ray, Vector3::new(0.0, 0.0, 6.0));
    assert_eq!(pt_seg, Vector3::new(3.0, 4.0, 6.0));

    // Parallel opposite direction
    let v0_anti = Vector3::new(3.0, 4.0, 6.0);
    let v1_anti = Vector3::new(3.0, 4.0, 1.0);
    let d2_anti = ray.distance_sq_to_segment(&v0_anti, &v1_anti, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2_anti, 25.0);
    assert_eq!(pt_ray, Vector3::new(0.0, 0.0, 6.0));
    assert_eq!(pt_seg, Vector3::new(3.0, 4.0, 6.0));
}

#[test]
fn test_ray_distance_sq_to_segment_degenerate_point_and_intersection() {
    // Degenerate segment: v0 == v1
    let ray = Ray::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(0.0, 1.0, 0.0));
    let v0 = Vector3::new(4.0, 5.0, 6.0);
    let v1 = Vector3::new(4.0, 5.0, 6.0);
    let mut pt_ray = Vector3::zero();
    let mut pt_seg = Vector3::zero();
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 34.0);
    assert_eq!(pt_ray, Vector3::new(1.0, 5.0, 1.0));
    assert_eq!(pt_seg, Vector3::new(4.0, 5.0, 6.0));

    // Exact intersection: d2 == 0.0, points coincide
    let ray = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let v0 = Vector3::new(5.0, -2.0, 0.0);
    let v1 = Vector3::new(5.0, 2.0, 0.0);
    let d2 = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2, 0.0);
    assert_eq!(pt_ray, Vector3::new(5.0, 0.0, 0.0));
    assert_eq!(pt_seg, Vector3::new(5.0, 0.0, 0.0));
}

#[test]
fn test_ray_distance_sq_to_segment_optional_targets_and_nan() {
    let ray = Ray::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let v0 = Vector3::new(5.0, 2.0, -3.0);
    let v1 = Vector3::new(5.0, 2.0, 3.0);

    // Both targets None
    let d2 = ray.distance_sq_to_segment(&v0, &v1, None, None);
    assert_eq!(d2, 4.0);

    // Only ray target
    let mut pt_ray = Vector3::zero();
    let d2_ray = ray.distance_sq_to_segment(&v0, &v1, Some(&mut pt_ray), None);
    assert_eq!(d2_ray, 4.0);
    assert_eq!(pt_ray, Vector3::new(5.0, 0.0, 0.0));

    // Only segment target
    let mut pt_seg = Vector3::zero();
    let d2_seg = ray.distance_sq_to_segment(&v0, &v1, None, Some(&mut pt_seg));
    assert_eq!(d2_seg, 4.0);
    assert_eq!(pt_seg, Vector3::new(5.0, 2.0, 0.0));

    // NaN propagation matching Three.js r186
    let nan_v0 = Vector3::new(f64::NAN, 2.0, 0.0);
    let mut nan_ray = Vector3::zero();
    let mut nan_seg = Vector3::zero();
    let d2_nan = ray.distance_sq_to_segment(&nan_v0, &v1, Some(&mut nan_ray), Some(&mut nan_seg));
    assert!(d2_nan.is_nan());
    assert!(nan_ray.x.is_nan());
    assert!(nan_ray.y.is_nan());
    assert!(nan_ray.z.is_nan());
    assert!(nan_seg.x.is_nan());
    assert!(nan_seg.y.is_nan());
    assert!(nan_seg.z.is_nan());
}

#[test]
fn test_ray_distance_sq_to_segment_diagonal_cases() {
    // Case A: Skew diagonals hitting boundary/clamping (origin closest on ray)
    let mut dir_a = Vector3::new(1.0, 1.0, 1.0);
    dir_a.normalize();
    let ray_a = Ray::new(Vector3::new(1.0, 2.0, 3.0), dir_a);
    let v0_a = Vector3::new(-2.0, 4.0, 1.0);
    let v1_a = Vector3::new(3.0, -1.0, 5.0);
    let mut pt_ray = Vector3::zero();
    let mut pt_seg = Vector3::zero();
    let d2_a = ray_a.distance_sq_to_segment(&v0_a, &v1_a, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2_a, 0.5);
    assert_eq!(pt_ray, Vector3::new(1.0, 2.0, 3.0));
    assert_eq!(pt_seg, Vector3::new(0.5, 1.5, 3.0));

    // Case B: 3D diagonals with interior minimum (Region 0)
    let mut dir_b = Vector3::new(2.0, 3.0, 6.0);
    dir_b.normalize();
    let ray_b = Ray::new(Vector3::new(-1.0, -2.0, -3.0), dir_b);
    let v0_b = Vector3::new(0.0, 10.0, -5.0);
    let v1_b = Vector3::new(10.0, 0.0, 5.0);
    let d2_b = ray_b.distance_sq_to_segment(&v0_b, &v1_b, Some(&mut pt_ray), Some(&mut pt_seg));
    assert_eq!(d2_b, 36.79508196721312);
    assert_eq!(
        pt_ray,
        Vector3::new(1.3442622950819665, 1.5163934426229497, 4.0327868852458995)
    );
    assert_eq!(
        pt_seg,
        Vector3::new(6.286885245901638, 3.7131147540983616, 1.2868852459016382)
    );
}
