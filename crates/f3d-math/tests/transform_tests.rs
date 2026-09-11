//! Focused tests for `f3d-math` transform primitives against pinned Three.js r186 semantics.
//!
//! Evaluates independently derived analytical reference cases to prevent candidate-generated goldens,
//! and tests negative cases (singular matrices, shear, negative determinants, non-unit quaternions,
//! and perspective divide).

use f3d_core::layout::{LayoutError, ProjectiveMat4};
use f3d_math::{CoordinateSystem, Matrix3, Matrix4, NarrowingError, Quaternion, Vector3};


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

fn assert_mat_close(m: &Matrix4, expected: &[f64; 16], eps: f64, msg: &str) {
    for i in 0..16 {
        assert_close(
            m.elements[i],
            expected[i],
            eps,
            &format!("{msg} [element {i}]"),
        );
    }
}

#[test]
fn test_trs_compose_identity_and_components() {
    // Rotation: 180 degrees around Z-axis -> q = (0.0, 0.0, 1.0, 0.0)
    let q = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    let pos = Vector3::new(10.0, 20.0, 30.0);
    let scale = Vector3::new(2.0, 3.0, 4.0);

    let mut m = Matrix4::identity();
    m.compose(&pos, &q, &scale);

    // Analytical expectation:
    // col 0: [-2.0, 0.0, 0.0, 0.0]
    // col 1: [0.0, -3.0, 0.0, 0.0]
    // col 2: [0.0, 0.0, 4.0, 0.0]
    // col 3: [10.0, 20.0, 30.0, 1.0]
    let expected = [
        -2.0, 0.0, 0.0, 0.0,
        0.0, -3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        10.0, 20.0, 30.0, 1.0,
    ];
    assert_mat_close(&m, &expected, EPS, "TRS composition");

    // Apply to point (1, 1, 1): (-2 + 10, -3 + 20, 4 + 30) = (8, 17, 34)
    let mut v = Vector3::new(1.0, 1.0, 1.0);
    v.apply_matrix4(&m);
    assert_vec_close(&v, [8.0, 17.0, 34.0], EPS, "Vector transform by TRS");
}

#[test]
fn test_nontrivial_hierarchy_multiplication() {
    // Parent: translation (10, 0, 0), uniform scale 2.0
    let mut parent = Matrix4::identity();
    let q_id = Quaternion::identity();
    parent.compose(&Vector3::new(10.0, 0.0, 0.0), &q_id, &Vector3::new(2.0, 2.0, 2.0));

    // Child: translation (0, 5, 0), 90 deg rotation around Z
    // q = (0, 0, 1/sqrt(2), 1/sqrt(2))
    let s = (0.5f64).sqrt();
    let q_child = Quaternion::new(0.0, 0.0, s, s);
    let mut child = Matrix4::identity();
    child.compose(&Vector3::new(0.0, 5.0, 0.0), &q_child, &Vector3::one());

    // World = Parent * Child
    let mut world = Matrix4::identity();
    world.multiply_matrices(&parent, &child);

    // Transform local point (1, 0, 0)
    // Child turns (1,0,0) into (0,1,0) + (0,5,0) = (0,6,0)
    // Parent scales by 2 -> (0,12,0) + translates (10,0,0) -> (10,12,0)
    let mut p = Vector3::new(1.0, 0.0, 0.0);
    p.apply_matrix4(&world);
    assert_vec_close(&p, [10.0, 12.0, 0.0], 1e-9, "Hierarchy point transformation");
}

#[test]
fn test_invert_positive_and_roundtrip() {
    let mut parent = Matrix4::identity();
    let q_id = Quaternion::identity();
    parent.compose(&Vector3::new(10.0, 0.0, 0.0), &q_id, &Vector3::new(2.0, 2.0, 2.0));

    let s = (0.5f64).sqrt();
    let q_child = Quaternion::new(0.0, 0.0, s, s);
    let mut child = Matrix4::identity();
    child.compose(&Vector3::new(0.0, 5.0, 0.0), &q_child, &Vector3::one());

    let mut world = Matrix4::identity();
    world.multiply_matrices(&parent, &child);

    let det = world.determinant();
    assert_close(det, 8.0, 1e-9, "Determinant of scaled hierarchy");

    let mut world_inv = world;
    world_inv.invert();

    // Multiplying world * world_inv must yield identity
    let mut ident = Matrix4::identity();
    ident.multiply_matrices(&world, &world_inv);
    assert_mat_close(&ident, &Matrix4::identity().elements, 1e-9, "Inverse * Original == Identity");

    // Applying world_inv to transformed point (10, 12, 0) must return (1, 0, 0)
    let mut p_trans = Vector3::new(10.0, 12.0, 0.0);
    p_trans.apply_matrix4(&world_inv);
    assert_vec_close(&p_trans, [1.0, 0.0, 0.0], 1e-9, "Inverse restores original point");
}

#[test]
fn test_singular_matrix_inversion_behavior_matches_threejs() {
    // Matrix with scale 0 on Y axis has det == 0.0
    let mut m_singular = Matrix4::identity();
    m_singular.compose(
        &Vector3::new(1.0, 2.0, 3.0),
        &Quaternion::identity(),
        &Vector3::new(2.0, 0.0, 2.0),
    );

    assert_close(m_singular.determinant(), 0.0, EPS, "Singular determinant is zero");
    assert_eq!(m_singular.try_invert(), None);

    // Three.js r186 invert() specifically sets all elements to 0.0 on det == 0
    m_singular.invert();
    assert_eq!(
        m_singular.elements,
        [0.0; 16],
        "Singular invert must produce all-zero matrix matching Three.js r186"
    );
}

#[test]
fn test_non_unit_authored_quaternion_preserves_unnormalized_math() {
    // Unnormalized quaternion: norm is 2.0, not 1.0
    let q = Quaternion::new(0.0, 2.0, 0.0, 0.0);
    let mut m = Matrix4::identity();
    m.compose(&Vector3::zero(), &q, &Vector3::one());

    // In Three.js r186:
    // y = 2.0, y2 = 4.0, yy = 8.0.
    // te[0] = (1 - (yy + zz)) * sx = (1 - 8) * 1 = -7.0.
    // te[5] = (1 - (xx + zz)) * sy = 1.0.
    // te[10] = (1 - (xx + yy)) * sz = -7.0.
    // Premature normalization would erroneously yield te[0] == -1.0, te[10] == -1.0.
    assert_close(m.elements[0], -7.0, EPS, "te[0] unnormalized quaternion expansion");
    assert_close(m.elements[5], 1.0, EPS, "te[5] unnormalized quaternion expansion");
    assert_close(m.elements[10], -7.0, EPS, "te[10] unnormalized quaternion expansion");
}

#[test]
fn test_negative_scale_and_reflection() {
    let mut m = Matrix4::identity();
    let pos = Vector3::zero();
    let q = Quaternion::identity();
    let scale = Vector3::new(-1.0, 2.0, 3.0);
    m.compose(&pos, &q, &scale);

    let det = m.determinant();
    assert_close(det, -6.0, EPS, "Reflection produces negative determinant");

    // Invert reflection matrix
    let mut m_inv = m;
    m_inv.invert();

    let mut v = Vector3::new(5.0, 1.0, 2.0);
    v.apply_matrix4(&m);
    assert_vec_close(&v, [-5.0, 2.0, 6.0], EPS, "Reflection scales correctly");

    v.apply_matrix4(&m_inv);
    assert_vec_close(&v, [5.0, 1.0, 2.0], EPS, "Inverted reflection restores original");
}

#[test]
fn test_shear_matrix_and_inversion() {
    let mut m_shear = Matrix4::identity();
    // Shear in XY: x' = x + 0.5 * y
    m_shear.set(
        1.0, 0.5, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    assert_close(m_shear.determinant(), 1.0, EPS, "Shear matrix determinant is 1.0");

    let mut v = Vector3::new(2.0, 4.0, 1.0);
    v.apply_matrix4(&m_shear);
    assert_vec_close(&v, [4.0, 4.0, 1.0], EPS, "Point transformed by shear");

    let mut m_inv = m_shear;
    m_inv.invert();

    v.apply_matrix4(&m_inv);
    assert_vec_close(&v, [2.0, 4.0, 1.0], EPS, "Inverted shear restores original");
}

#[test]
fn test_perspective_divide_in_apply_matrix4() {
    let mut m_proj = Matrix4::zero();
    // Projective matrix where bottom row has perspective divisor:
    // row 0: [2, 0, 0, 0]
    // row 1: [0, 4, 0, 0]
    // row 2: [0, 0, 6, 0]
    // row 3: [0, 0, 2, 0] -> denom = 2.0 * z
    m_proj.set(
        2.0, 0.0, 0.0, 0.0,
        0.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 6.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
    );

    let mut v = Vector3::new(1.0, 1.0, 3.0);
    // denom = 2.0 * 3.0 = 6.0, w = 1/6
    // x' = 2.0 * (1/6) = 1/3
    // y' = 4.0 * (1/6) = 2/3
    // z' = (6.0 * 3.0) * (1/6) = 3.0
    v.apply_matrix4(&m_proj);
    assert_vec_close(&v, [1.0 / 3.0, 2.0 / 3.0, 3.0], EPS, "Perspective divide");
}

#[test]
fn test_wire_conversion_seam_affine_rows() {
    let mut m = Matrix4::identity();
    m.compose(
        &Vector3::new(10.0, 20.0, 30.0),
        &Quaternion::identity(),
        &Vector3::new(2.0, 3.0, 4.0),
    );

    // Should convert cleanly to AffineRows
    let affine = m.to_affine_rows().expect("Should convert to affine rows");
    assert_eq!(affine.r0, [2.0, 0.0, 0.0, 10.0]);
    assert_eq!(affine.r1, [0.0, 3.0, 0.0, 20.0]);
    assert_eq!(affine.r2, [0.0, 0.0, 4.0, 30.0]);

    // Roundtrip back to Matrix4
    let restored = Matrix4::from_affine_rows(&affine);
    assert_mat_close(&restored, &m.elements, EPS, "AffineRows roundtrip");

    // Projective matrix must be rejected by to_affine_rows()
    let mut m_proj = Matrix4::identity();
    m_proj.elements[11] = -1.0; // perspective term
    assert_eq!(m_proj.to_affine_rows(), Err(LayoutError::NonAffineMatrix));

    // But succeeds with ProjectiveMat4
    let proj: ProjectiveMat4 = m_proj.to_projective_mat4();
    assert_eq!(proj.elements[11], -1.0f32, "ProjectiveMat4 elements field preserves perspective value");
    let restored_proj = Matrix4::from_projective_mat4(&proj);
    assert_mat_close(&restored_proj, &m_proj.elements, EPS, "ProjectiveMat4 roundtrip");
}

#[test]
fn test_vector3_normalize_signed_zeros() {
    // Upstream Vector3.js:794: this.divideScalar( this.length() || 1 )
    // When length is 0, (0 || 1) evaluates to 1 in JS, so components are multiplied by (1 / 1 = 1).
    // In IEEE 754: -0.0 * 1.0 = -0.0, and 0.0 * 1.0 = 0.0.
    let mut v = Vector3::new(-0.0, 0.0, -0.0);
    v.normalize();

    assert_eq!(
        v.x.to_bits(),
        (-0.0f64).to_bits(),
        "v.x must preserve negative zero sign bit"
    );
    assert_eq!(
        v.y.to_bits(),
        (0.0f64).to_bits(),
        "v.y must preserve positive zero sign bit"
    );
    assert_eq!(
        v.z.to_bits(),
        (-0.0f64).to_bits(),
        "v.z must preserve negative zero sign bit"
    );
}

#[test]
fn test_vector3_normalize_nan_component_isolation() {
    // Upstream Vector3.js:794: this.divideScalar( this.length() || 1 )
    // If one component is NaN, length is NaN. In JS, (NaN || 1) evaluates to 1!
    // Therefore, divideScalar(1) multiplies components by 1, leaving non-NaN components unaffected.
    let mut v = Vector3::new(f64::NAN, 1.0, -0.0);
    v.normalize();

    assert!(v.x.is_nan(), "v.x must remain NaN");
    assert_eq!(
        v.y.to_bits(),
        (1.0f64).to_bits(),
        "v.y must remain exactly 1.0, not contaminated by NaN in other components"
    );
    assert_eq!(
        v.z.to_bits(),
        (-0.0f64).to_bits(),
        "v.z must retain -0.0, not contaminated by NaN"
    );
}

#[test]
fn test_to_affine_rows_f64_subnormal_and_narrowing_rejection() {
    // 1. Subnormal / tiny non-zero perspective term that would underflow to 0.0 in f32:
    let mut m_tiny = Matrix4::identity();
    m_tiny.elements[3] = 1e-100;
    assert_eq!(
        m_tiny.to_affine_rows(),
        Err(LayoutError::NonAffineMatrix),
        "1e-100 in e[3] must be rejected in f64 before narrowing underflows to 0.0f32"
    );

    // 2. High-precision scale near 1.0 that would round to 1.0 in f32:
    let mut m_near1 = Matrix4::identity();
    m_near1.elements[15] = 1.0 + 2.0f64.powi(-40);
    assert_eq!(
        m_near1.to_affine_rows(),
        Err(LayoutError::NonAffineMatrix),
        "e[15] = 1 + 2^-40 must be rejected in f64 before narrowing rounds to 1.0f32"
    );

    // 3. Small perspective coordinate effect demonstration:
    // With e[3] = 5e-7 and x = 1e7, denom = e[3]*x + e[15] = 5e-7 * 1e7 + 1 = 6.0.
    // Discarding e[3] changes the result by a factor of 6!
    let mut m_persp = Matrix4::identity();
    m_persp.elements[3] = 5e-7;
    assert_eq!(
        m_persp.to_affine_rows(),
        Err(LayoutError::NonAffineMatrix),
        "e[3] = 5e-7 must not be silently discarded as affine"
    );

    let mut pt = Vector3::new(1e7, 2.0, 3.0);
    pt.apply_matrix4(&m_persp);
    // x' = 1e7 / 6.0
    assert_close(pt.x, 1e7 / 6.0, 1e-3, "Point coordinate divided by 6 under 5e-7 perspective");
}

#[test]
fn test_decompose_reference_cases() {
    // 1. Non-uniform scale + real 3D rotation decompose roundtrip:
    // Position: (1.5, -2.5, 3.75)
    // Non-uniform scale: (2.0, 3.0, 4.0)
    // Oblique rotation: 60° around normalized axis (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
    // half_angle = 30° -> sin(30°) = 0.5, cos(30°) = sqrt(3)/2
    // q = (0.5/sqrt(3), 0.5/sqrt(3), 0.5/sqrt(3), sqrt(3)/2)
    let inv_sqrt3 = 1.0 / 3.0f64.sqrt();
    let q = Quaternion::new(
        0.5 * inv_sqrt3,
        0.5 * inv_sqrt3,
        0.5 * inv_sqrt3,
        0.5 * 3.0f64.sqrt(),
    );
    let pos = Vector3::new(1.5, -2.5, 3.75);
    let scale = Vector3::new(2.0, 3.0, 4.0);

    let mut m = Matrix4::identity();
    m.compose(&pos, &q, &scale);

    let mut dec_pos = Vector3::zero();
    let mut dec_q = Quaternion::identity();
    let mut dec_scale = Vector3::zero();

    let success = m.decompose(&mut dec_pos, &mut dec_q, &mut dec_scale);
    assert!(success, "decompose must succeed for valid affine matrix");
    assert_vec_close(&dec_pos, [1.5, -2.5, 3.75], 1e-12, "decomposed position");
    assert_vec_close(&dec_scale, [2.0, 3.0, 4.0], 1e-12, "decomposed scale");
    let dot = dec_q.dot(&q).abs();
    assert!(
        (dot - 1.0).abs() < 1e-12,
        "decomposed quaternion must be equivalent to q (|dot| == 1, got dot = {dot})"
    );

    // 2. Negative determinant flips sx (Three.js Matrix4.js:1094-1095):
    // scale.x = -2.0 produces negative determinant (-24.0 < 0)
    let scale_refl = Vector3::new(-2.0, 3.0, 4.0);
    let mut m_refl = Matrix4::identity();
    m_refl.compose(&pos, &q, &scale_refl);

    let det_refl = m_refl.determinant_affine();
    assert!(det_refl < 0.0, "Reflection matrix has negative affine determinant");

    let success_refl = m_refl.decompose(&mut dec_pos, &mut dec_q, &mut dec_scale);
    assert!(success_refl, "decompose must succeed for negative determinant matrix");
    assert_vec_close(&dec_pos, [1.5, -2.5, 3.75], 1e-12, "negative det position");
    assert_vec_close(&dec_scale, [-2.0, 3.0, 4.0], 1e-12, "negative det flips sx to -2.0");
    let dot_refl = dec_q.dot(&q).abs();
    assert!(
        (dot_refl - 1.0).abs() < 1e-12,
        "negative det rotation preserved (|dot| == 1, got dot = {dot_refl})"
    );

    // 3. Singular matrix (det == 0) yields unit scale and identity quaternion (Three.js Matrix4.js:1081-1088):
    let scale_zero = Vector3::new(2.0, 0.0, 4.0);
    let mut m_singular = Matrix4::identity();
    m_singular.compose(&pos, &q, &scale_zero);

    assert_close(m_singular.determinant_affine(), 0.0, 1e-12, "singular affine det is 0.0");

    // Initialize with dirty values to verify explicit reset
    let mut dec_pos_sing = Vector3::zero();
    let mut dec_q_sing = Quaternion::new(42.0, 42.0, 42.0, 42.0);
    let mut dec_scale_sing = Vector3::new(99.0, 99.0, 99.0);

    let success_sing = m_singular.decompose(&mut dec_pos_sing, &mut dec_q_sing, &mut dec_scale_sing);
    assert!(!success_sing, "singular decompose must return false per Three.js r186 fallback");
    assert_vec_close(&dec_pos_sing, [1.5, -2.5, 3.75], 1e-12, "singular decompose preserves position");
    assert_vec_close(&dec_scale_sing, [1.0, 1.0, 1.0], 1e-12, "singular decompose sets unit scale");
    assert_eq!(
        dec_q_sing,
        Quaternion::identity(),
        "singular decompose sets identity quaternion"
    );
}

#[test]
fn test_make_perspective_ndc_and_reversal() {
    let left = -1.0;
    let right = 1.0;
    let bottom = -1.0;
    let top = 1.0;
    let near = 2.0;
    let far = 10.0;

    // 1. WebGL coordinate system (depth range [-1, 1]):
    let mut m_webgl = Matrix4::identity();
    m_webgl.make_perspective(left, right, top, bottom, near, far, CoordinateSystem::WebGL, false);

    // Near plane point (0, 0, -near)
    let mut pt_near_gl = Vector3::new(0.0, 0.0, -near);
    pt_near_gl.apply_matrix4(&m_webgl);
    assert_close(pt_near_gl.z, -1.0, 1e-12, "WebGL perspective near plane maps to z = -1.0");

    // Far plane point (0, 0, -far)
    let mut pt_far_gl = Vector3::new(0.0, 0.0, -far);
    pt_far_gl.apply_matrix4(&m_webgl);
    assert_close(pt_far_gl.z, 1.0, 1e-12, "WebGL perspective far plane maps to z = 1.0");

    // 2. WebGPU coordinate system (depth range [0, 1]):
    let mut m_webgpu = Matrix4::identity();
    m_webgpu.make_perspective(left, right, top, bottom, near, far, CoordinateSystem::WebGPU, false);

    // Near plane point (0, 0, -near)
    let mut pt_near_gpu = Vector3::new(0.0, 0.0, -near);
    pt_near_gpu.apply_matrix4(&m_webgpu);
    assert_close(pt_near_gpu.z, 0.0, 1e-12, "WebGPU perspective near plane maps to z = 0.0");

    // Far plane point (0, 0, -far)
    let mut pt_far_gpu = Vector3::new(0.0, 0.0, -far);
    pt_far_gpu.apply_matrix4(&m_webgpu);
    assert_close(pt_far_gpu.z, 1.0, 1e-12, "WebGPU perspective far plane maps to z = 1.0");

    // 3. Reversed depth swaps near and far to [1.0, 0.0]:
    let mut m_rev = Matrix4::identity();
    m_rev.make_perspective(left, right, top, bottom, near, far, CoordinateSystem::WebGPU, true);

    let mut pt_near_rev = Vector3::new(0.0, 0.0, -near);
    pt_near_rev.apply_matrix4(&m_rev);
    assert_close(pt_near_rev.z, 1.0, 1e-12, "Reversed depth near plane maps to z = 1.0");

    let mut pt_far_rev = Vector3::new(0.0, 0.0, -far);
    pt_far_rev.apply_matrix4(&m_rev);
    assert_close(pt_far_rev.z, 0.0, 1e-12, "Reversed depth far plane maps to z = 0.0");

    // 4. Layout rejection / retention:
    // Perspective matrices have non-zero perspective coefficients (e[11] = -1.0, e[15] = 0.0)
    // and MUST be rejected by to_affine_rows().
    assert_eq!(
        m_webgpu.to_affine_rows(),
        Err(LayoutError::NonAffineMatrix),
        "Perspective matrix must be rejected by to_affine_rows"
    );
    assert_eq!(
        m_rev.to_affine_rows(),
        Err(LayoutError::NonAffineMatrix),
        "Reversed perspective matrix must be rejected by to_affine_rows"
    );

    // But must be retained byte-exactly by to_projective_mat4():
    let proj = m_webgpu.to_projective_mat4();
    for i in 0..16 {
        assert_eq!(
            proj.elements[i],
            m_webgpu.elements[i] as f32,
            "ProjectiveMat4 element {i} matches perspective matrix element byte-for-byte"
        );
    }
    let bytes = proj.to_bytes();
    assert_eq!(bytes.len(), 64, "ProjectiveMat4 serializes to 64 bytes");
    let restored_proj = ProjectiveMat4::from_bytes(&bytes);
    assert_eq!(proj, restored_proj, "ProjectiveMat4 byte deserialization roundtrips");
}

#[test]
fn test_make_orthographic_ndc_and_reversal() {
    let left = -5.0;
    let right = 5.0;
    let bottom = -5.0;
    let top = 5.0;
    let near = 2.0;
    let far = 10.0;

    // 1. WebGL coordinate system (depth range [-1, 1]):
    let mut m_webgl = Matrix4::identity();
    m_webgl.make_orthographic(left, right, top, bottom, near, far, CoordinateSystem::WebGL, false);

    let mut pt_near_gl = Vector3::new(0.0, 0.0, -near);
    pt_near_gl.apply_matrix4(&m_webgl);
    assert_close(pt_near_gl.z, -1.0, 1e-12, "WebGL orthographic near plane maps to z = -1.0");

    let mut pt_far_gl = Vector3::new(0.0, 0.0, -far);
    pt_far_gl.apply_matrix4(&m_webgl);
    assert_close(pt_far_gl.z, 1.0, 1e-12, "WebGL orthographic far plane maps to z = 1.0");

    // 2. WebGPU coordinate system (depth range [0, 1]):
    let mut m_webgpu = Matrix4::identity();
    m_webgpu.make_orthographic(left, right, top, bottom, near, far, CoordinateSystem::WebGPU, false);

    let mut pt_near_gpu = Vector3::new(0.0, 0.0, -near);
    pt_near_gpu.apply_matrix4(&m_webgpu);
    assert_close(pt_near_gpu.z, 0.0, 1e-12, "WebGPU orthographic near plane maps to z = 0.0");

    let mut pt_far_gpu = Vector3::new(0.0, 0.0, -far);
    pt_far_gpu.apply_matrix4(&m_webgpu);
    assert_close(pt_far_gpu.z, 1.0, 1e-12, "WebGPU orthographic far plane maps to z = 1.0");

    // 3. Reversed depth swaps near and far to [1.0, 0.0]:
    let mut m_rev = Matrix4::identity();
    m_rev.make_orthographic(left, right, top, bottom, near, far, CoordinateSystem::WebGPU, true);

    let mut pt_near_rev = Vector3::new(0.0, 0.0, -near);
    pt_near_rev.apply_matrix4(&m_rev);
    assert_close(pt_near_rev.z, 1.0, 1e-12, "Reversed depth orthographic near plane maps to z = 1.0");

    let mut pt_far_rev = Vector3::new(0.0, 0.0, -far);
    pt_far_rev.apply_matrix4(&m_rev);
    assert_close(pt_far_rev.z, 0.0, 1e-12, "Reversed depth orthographic far plane maps to z = 0.0");

    // 4. Layout behavior: Orthographic matrices have row 3 = [0, 0, 0, 1], so they are affine:
    assert!(m_webgpu.is_affine(), "Orthographic matrix is structurally affine");
    assert!(m_webgpu.to_affine_rows().is_ok(), "Orthographic matrix converts to AffineRows");

    let proj = m_webgpu.to_projective_mat4();
    for i in 0..16 {
        assert_eq!(
            proj.elements[i],
            m_webgpu.elements[i] as f32,
            "ProjectiveMat4 element {i} matches orthographic matrix element byte-for-byte"
        );
    }
}

#[test]
fn test_projective_mat4_transform_homogeneous_matches_apply_matrix4() {
    let left = -1.0;
    let right = 1.0;
    let bottom = -1.0;
    let top = 1.0;
    let near = 2.0;
    let far = 10.0;

    let mut m_webgpu = Matrix4::identity();
    m_webgpu.make_perspective(left, right, top, bottom, near, far, CoordinateSystem::WebGPU, false);

    let proj = m_webgpu.to_projective_mat4();

    // Test both dyadic and non-dyadic points inside the frustum:
    let test_points = [
        Vector3::new(0.5, -0.25, -4.0),
        Vector3::new(0.3, 0.7, -3.5),
        Vector3::new(-0.8, -0.6, -8.0),
    ];

    for pt in &test_points {
        // 1. Vector3::apply_matrix4 (f64 reference with perspective divide)
        let mut v = *pt;
        v.apply_matrix4(&m_webgpu);

        // 2. ProjectiveMat4::transform_homogeneous (f32 GPU wire representation)
        let h = proj.transform_homogeneous([pt.x as f32, pt.y as f32, pt.z as f32, 1.0]);
        let inv_w = 1.0 / h[3];
        let h_ndc = [h[0] * inv_w, h[1] * inv_w, h[2] * inv_w];

        // 3. Agreement within 1e-6 accounting for f32 narrowing precision
        assert_close(
            h_ndc[0] as f64,
            v.x,
            1e-6,
            &format!("ProjectiveMat4 X matches Vector3::apply_matrix4 for point ({}, {}, {})", pt.x, pt.y, pt.z),
        );
        assert_close(
            h_ndc[1] as f64,
            v.y,
            1e-6,
            &format!("ProjectiveMat4 Y matches Vector3::apply_matrix4 for point ({}, {}, {})", pt.x, pt.y, pt.z),
        );
        assert_close(
            h_ndc[2] as f64,
            v.z,
            1e-6,
            &format!("ProjectiveMat4 Z matches Vector3::apply_matrix4 for point ({}, {}, {})", pt.x, pt.y, pt.z),
        );
    }
}

#[test]
fn test_vector3_checked_narrowing_policy() {
    // 1. Exact value in f32:
    // 0.5 = 2^-1, -0.25 = -2^-2, 1024.0 = 2^10. Exactly representable with 0 diff.
    let v_exact = Vector3::new(0.5, -0.25, 1024.0);
    let narrowed = v_exact.to_f32_checked(0.0, 0.0).expect("exact f32 passes with 0 tolerance");
    assert_eq!(narrowed, [0.5f32, -0.25f32, 1024.0f32]);
    let strict_exact = v_exact.to_f32_strict(0.0, 0.0).expect("strict exact passes");
    assert_eq!(strict_exact, [0.5f32, -0.25f32, 1024.0f32]);

    // 2. Inexact value in f32:
    // 1.0 + 2^-30 is not representable in f32 (24-bit mantissa).
    // In f32, 1.0 + 2^-30 rounds to 1.0f32.
    // Absolute diff = 2^-30 = 9.313225746154785e-10.
    // Relative diff = 2^-30 / (1.0 + 2^-30) ~= 9.31322573748e-10.
    let v_inexact = Vector3::new(1.0 + 2.0f64.powi(-30), 0.0, 0.0);
    // Should fail when tolerance is 1e-12:
    let err = v_inexact.to_f32_checked(1e-12, 1e-12).expect_err("should reject precision loss");
    match err {
        NarrowingError::PrecisionLossExceeded { index, original, narrowed, abs_diff, max_abs_err, .. } => {
            assert_eq!(index, 0);
            assert_eq!(original, 1.0 + 2.0f64.powi(-30));
            assert_eq!(narrowed, 1.0);
            assert_close(abs_diff, 2.0f64.powi(-30), 1e-18, "abs_diff equals 2^-30");
            assert_eq!(max_abs_err, 1e-12);
        }
        _ => panic!("unexpected error variant: {err:?}"),
    }
    // Should pass when tolerance is relaxed to 1e-8:
    let pass = v_inexact.to_f32_checked(1e-8, 1e-8).expect("passes with 1e-8 tolerance");
    assert_eq!(pass[0], 1.0f32);

    // 3. Non-finite values: NaN
    let v_nan = Vector3::new(f64::NAN, 1.0, 2.0);
    // Permissive mode preserves NaN:
    let perm_nan = v_nan.to_f32_checked(1e-6, 1e-6).expect("permissive allows NaN");
    assert!(perm_nan[0].is_nan());
    assert_eq!(perm_nan[1], 1.0f32);
    // Strict mode rejects NaN:
    let strict_nan_err = v_nan.to_f32_strict(1e-6, 1e-6).expect_err("strict mode rejects NaN");
    match strict_nan_err {
        NarrowingError::NonFinite { index, value } => {
            assert_eq!(index, 0);
            assert!(value.is_nan());
        }
        _ => panic!("unexpected error variant: {strict_nan_err:?}"),
    }

    // 4. Non-finite values: Infinity
    let v_inf = Vector3::new(1.0, f64::INFINITY, -f64::INFINITY);
    // Permissive mode preserves Infinity:
    let perm_inf = v_inf.to_f32_checked(1e-6, 1e-6).expect("permissive allows Infinity");
    assert_eq!(perm_inf[1], f32::INFINITY);
    assert_eq!(perm_inf[2], -f32::INFINITY);
    // Strict mode rejects Infinity:
    let strict_inf_err = v_inf.to_f32_strict(1e-6, 1e-6).expect_err("strict mode rejects +Infinity");
    match strict_inf_err {
        NarrowingError::NonFinite { index, value } => {
            assert_eq!(index, 1);
            assert_eq!(value, f64::INFINITY);
        }
        _ => panic!("unexpected error variant: {strict_inf_err:?}"),
    }
}

#[test]
fn test_matrix4_checked_narrowing_and_conversions() {
    // 1. Exact affine matrix in f32:
    let mut m_exact = Matrix4::identity();
    m_exact.elements[12] = 2.0;
    m_exact.elements[13] = -4.5;
    m_exact.elements[14] = 0.125;
    let affine = m_exact
        .to_affine_rows_checked(0.0, 0.0)
        .expect("exact affine matrix converts with 0.0 tolerance");
    let proj = m_exact
        .to_projective_mat4_checked(0.0, 0.0)
        .expect("exact projective matrix converts with 0.0 tolerance");
    assert_eq!(affine.r0[3], 2.0f32);
    assert_eq!(affine.r1[3], -4.5f32);
    assert_eq!(affine.r2[3], 0.125f32);
    assert_eq!(proj.elements[12], 2.0f32);
    assert_eq!(proj.elements[13], -4.5f32);
    assert_eq!(proj.elements[14], 0.125f32);

    // 2. Inexact element:
    let mut m_inexact = Matrix4::identity();
    m_inexact.elements[0] = 1.0 + 2.0f64.powi(-30);
    let affine_err = m_inexact
        .to_affine_rows_checked(1e-12, 1e-12)
        .expect_err("should reject precision loss in affine element");
    match affine_err {
        NarrowingError::PrecisionLossExceeded { index, original, .. } => {
            assert_eq!(index, 0);
            assert_eq!(original, 1.0 + 2.0f64.powi(-30));
        }
        _ => panic!("unexpected: {affine_err:?}"),
    }
    let proj_err = m_inexact
        .to_projective_mat4_checked(1e-12, 1e-12)
        .expect_err("should reject precision loss in projective element");
    match proj_err {
        NarrowingError::PrecisionLossExceeded { index, .. } => assert_eq!(index, 0),
        _ => panic!("unexpected: {proj_err:?}"),
    }

    // 3. Strict non-finite rejection: NaN
    let mut m_nan = Matrix4::identity();
    m_nan.elements[5] = f64::NAN;
    let strict_err = m_nan
        .to_affine_rows_strict(1e-6, 1e-6)
        .expect_err("strict affine rejects NaN");
    match strict_err {
        NarrowingError::NonFinite { index, value } => {
            assert_eq!(index, 5);
            assert!(value.is_nan());
        }
        _ => panic!("unexpected: {strict_err:?}"),
    }
    let proj_nan_err = m_nan
        .to_projective_mat4_strict(1e-6, 1e-6)
        .expect_err("strict projective rejects NaN");
    match proj_nan_err {
        NarrowingError::NonFinite { index, value } => {
            assert_eq!(index, 5);
            assert!(value.is_nan());
        }
        _ => panic!("unexpected: {proj_nan_err:?}"),
    }

    // 4. Strict non-finite rejection: Infinity
    let mut m_inf = Matrix4::identity();
    m_inf.elements[10] = f64::INFINITY;
    let inf_err = m_inf
        .to_affine_rows_strict(1e-6, 1e-6)
        .expect_err("strict affine rejects Infinity");
    match inf_err {
        NarrowingError::NonFinite { index, value } => {
            assert_eq!(index, 10);
            assert_eq!(value, f64::INFINITY);
        }
        _ => panic!("unexpected: {inf_err:?}"),
    }

    // 5. Non-affine matrix rejection before narrowing:
    // Perspective matrix (e[11] = -1.0, e[15] = 0.0)
    let mut m_persp = Matrix4::identity();
    m_persp.make_perspective(-1.0, 1.0, 1.0, -1.0, 2.0, 10.0, CoordinateSystem::WebGPU, false);
    let non_affine_err = m_persp
        .to_affine_rows_checked(1e-4, 1e-4)
        .expect_err("perspective matrix must be rejected by to_affine_rows_checked");
    assert_eq!(non_affine_err, NarrowingError::NonAffineMatrix);

    // But to_projective_mat4_checked retains it:
    let proj_persp = m_persp
        .to_projective_mat4_checked(1e-6, 1e-6)
        .expect("perspective matrix succeeds in to_projective_mat4_checked");
    assert_eq!(proj_persp.elements[11], -1.0f32);
    assert_eq!(proj_persp.elements[15], 0.0f32);

    // Subnormal perspective term (1e-100) must also be rejected by to_affine_rows_checked
    // even though it would underflow to 0.0 in f32:
    let mut m_subnormal = Matrix4::identity();
    m_subnormal.elements[11] = 1e-100;
    assert_eq!(
        m_subnormal.to_affine_rows_checked(1e-4, 1e-4),
        Err(NarrowingError::NonAffineMatrix),
        "f64 affine shape check runs before narrowing"
    );
}

#[test]
fn test_vector3_cross_and_anti_commutativity() {
    let ex = Vector3::new(1.0, 0.0, 0.0);
    let ey = Vector3::new(0.0, 1.0, 0.0);
    let ez = Vector3::new(0.0, 0.0, 1.0);

    // Standard basis cross products
    let mut v = ex;
    v.cross(&ey);
    assert_vec_close(&v, [0.0, 0.0, 1.0], EPS, "ex cross ey == ez");

    let mut v = ey;
    v.cross(&ez);
    assert_vec_close(&v, [1.0, 0.0, 0.0], EPS, "ey cross ez == ex");

    let mut v = ez;
    v.cross(&ex);
    assert_vec_close(&v, [0.0, 1.0, 0.0], EPS, "ez cross ex == ey");

    // Anti-commutativity
    let mut v = ey;
    v.cross(&ex);
    assert_vec_close(&v, [0.0, 0.0, -1.0], EPS, "ey cross ex == -ez");

    // Arbitrary vectors: a = (2, 3, 4), b = (5, 6, 7)
    // a x b = (3*7 - 4*6, 4*5 - 2*7, 2*6 - 3*5) = (21-24, 20-14, 12-15) = (-3, 6, -3)
    let mut a = Vector3::new(2.0, 3.0, 4.0);
    let b = Vector3::new(5.0, 6.0, 7.0);
    a.cross(&b);
    assert_vec_close(&a, [-3.0, 6.0, -3.0], EPS, "arbitrary cross product");

    // Self cross product is zero
    let mut s = Vector3::new(2.0, 3.0, 4.0);
    let s_clone = s;
    s.cross(&s_clone);
    assert_vec_close(&s, [0.0, 0.0, 0.0], EPS, "self cross product is zero");
}

#[test]
fn test_vector3_lerp_and_lerp_vectors() {
    let v1 = Vector3::new(10.0, 20.0, 30.0);
    let v2 = Vector3::new(30.0, 40.0, 50.0);

    // alpha = 0.0
    let mut v = v1;
    v.lerp(&v2, 0.0);
    assert_vec_close(&v, [10.0, 20.0, 30.0], EPS, "lerp alpha=0.0");

    // alpha = 0.5
    let mut v = v1;
    v.lerp(&v2, 0.5);
    assert_vec_close(&v, [20.0, 30.0, 40.0], EPS, "lerp alpha=0.5");

    // alpha = 1.0
    let mut v = v1;
    v.lerp(&v2, 1.0);
    assert_vec_close(&v, [30.0, 40.0, 50.0], EPS, "lerp alpha=1.0");

    // lerp_vectors with alpha = 0.25: 10 + (30-10)*0.25 = 15
    let mut out = Vector3::zero();
    out.lerp_vectors(&v1, &v2, 0.25);
    assert_vec_close(&out, [15.0, 25.0, 35.0], EPS, "lerp_vectors alpha=0.25");

    // Extrapolation with alpha = -0.5: 10 + 20*(-0.5) = 0
    let mut ext = v1;
    ext.lerp(&v2, -0.5);
    assert_vec_close(&ext, [0.0, 10.0, 20.0], EPS, "lerp extrapolation alpha=-0.5");
}

#[test]
fn test_vector3_angle_to_analytical() {
    let ex = Vector3::new(1.0, 0.0, 0.0);
    let ey = Vector3::new(0.0, 1.0, 0.0);
    let ez = Vector3::new(0.0, 0.0, 1.0);

    // Orthogonal: PI / 2
    assert_close(ex.angle_to(&ey), core::f64::consts::FRAC_PI_2, EPS, "ex angle_to ey == PI/2");
    assert_close(ey.angle_to(&ez), core::f64::consts::FRAC_PI_2, EPS, "ey angle_to ez == PI/2");

    // Collinear same direction: 0
    assert_close(ex.angle_to(&ex), 0.0, EPS, "ex angle_to ex == 0");
    let ex2 = Vector3::new(5.0, 0.0, 0.0);
    assert_close(ex.angle_to(&ex2), 0.0, EPS, "ex angle_to scaled ex == 0");

    // Collinear opposite direction: PI
    let neg_ex = Vector3::new(-1.0, 0.0, 0.0);
    assert_close(ex.angle_to(&neg_ex), core::f64::consts::PI, EPS, "ex angle_to -ex == PI");

    // 45 degrees: (1, 1, 0) and (1, 0, 0) -> cos(theta) = 1/sqrt(2) -> PI/4
    let diag = Vector3::new(1.0, 1.0, 0.0);
    assert_close(diag.angle_to(&ex), core::f64::consts::FRAC_PI_4, EPS, "(1,1,0) angle_to (1,0,0) == PI/4");

    // Zero vector fallback matching Three.js r186: if denominator == 0 return PI / 2
    let zero = Vector3::zero();
    assert_close(zero.angle_to(&ex), core::f64::consts::FRAC_PI_2, EPS, "zero angle_to ex == PI/2");
    assert_close(ex.angle_to(&zero), core::f64::consts::FRAC_PI_2, EPS, "ex angle_to zero == PI/2");
}

#[test]
fn test_vector3_distance_methods_pythagorean() {
    // 3-4-12 -> 13 Pythagorean quadruple: 3^2 + 4^2 + 12^2 = 9 + 16 + 144 = 169 = 13^2
    let a = Vector3::new(1.0, 2.0, 3.0);
    let b = Vector3::new(4.0, 6.0, 15.0);

    assert_close(a.distance_to_squared(&b), 169.0, EPS, "distance_to_squared 3-4-12");
    assert_close(a.distance_to(&b), 13.0, EPS, "distance_to 3-4-12");
    assert_close(a.manhattan_distance_to(&b), 19.0, EPS, "manhattan_distance_to 3-4-12");

    // Symmetry
    assert_close(b.distance_to(&a), 13.0, EPS, "distance_to symmetry");
    assert_close(b.manhattan_distance_to(&a), 19.0, EPS, "manhattan symmetry");

    // Zero distance to self
    assert_close(a.distance_to(&a), 0.0, EPS, "distance to self is 0");
    assert_close(a.distance_to_squared(&a), 0.0, EPS, "distance squared to self is 0");
    assert_close(a.manhattan_distance_to(&a), 0.0, EPS, "manhattan to self is 0");
}

#[test]
fn test_vector3_set_from_matrix_columns_scale_and_position() {
    let mut m4 = Matrix4::zero();
    for col in 0..4 {
        for row in 0..4 {
            m4.elements[col * 4 + row] = (col * 4 + row + 1) as f64;
        }
    }

    let mut v = Vector3::zero();
    v.set_from_matrix_column(&m4, 0);
    assert_vec_close(&v, [1.0, 2.0, 3.0], EPS, "m4 column 0");

    v.set_from_matrix_column(&m4, 1);
    assert_vec_close(&v, [5.0, 6.0, 7.0], EPS, "m4 column 1");

    v.set_from_matrix_column(&m4, 2);
    assert_vec_close(&v, [9.0, 10.0, 11.0], EPS, "m4 column 2");

    v.set_from_matrix_column(&m4, 3);
    assert_vec_close(&v, [13.0, 14.0, 15.0], EPS, "m4 column 3");

    v.set_from_matrix_position(&m4);
    assert_vec_close(&v, [13.0, 14.0, 15.0], EPS, "m4 position column");

    // Matrix3 column extraction
    let mut m3 = Matrix3::zero();
    for col in 0..3 {
        for row in 0..3 {
            m3.elements[col * 3 + row] = ((col * 3 + row + 1) * 10) as f64;
        }
    }
    v.set_from_matrix3_column(&m3, 0);
    assert_vec_close(&v, [10.0, 20.0, 30.0], EPS, "m3 column 0");

    v.set_from_matrix3_column(&m3, 1);
    assert_vec_close(&v, [40.0, 50.0, 60.0], EPS, "m3 column 1");

    v.set_from_matrix3_column(&m3, 2);
    assert_vec_close(&v, [70.0, 80.0, 90.0], EPS, "m3 column 2");

    // set_from_matrix_scale
    let mut m_scale = Matrix4::identity();
    m_scale.elements[0] = 2.0;
    m_scale.elements[5] = -3.0;
    m_scale.elements[10] = 4.0;
    v.set_from_matrix_scale(&m_scale);
    assert_vec_close(&v, [2.0, 3.0, 4.0], EPS, "set_from_matrix_scale norms");
}

#[test]
fn test_vector3_apply_matrix4_perspective_divide_analytical() {
    let m = Matrix4::from_elements([
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, -1.0,
        0.0, 0.0, 5.0, 0.0,
    ]);

    let mut v = Vector3::new(2.0, 4.0, -2.0);
    v.apply_matrix4(&m);
    assert_vec_close(&v, [2.0, 6.0, -1.5], EPS, "apply_matrix4 perspective divide");
}

#[test]
fn test_vector3_apply_quaternion_analytical() {
    let s = (0.5f64).sqrt();
    let qz_90 = Quaternion::new(0.0, 0.0, s, s);
    let mut v = Vector3::new(1.0, 0.0, 0.0);
    v.apply_quaternion(&qz_90);
    assert_vec_close(&v, [0.0, 1.0, 0.0], EPS, "rot 90 deg around Z");

    let qx_90 = Quaternion::new(s, 0.0, 0.0, s);
    let mut v = Vector3::new(0.0, 1.0, 0.0);
    v.apply_quaternion(&qx_90);
    assert_vec_close(&v, [0.0, 0.0, 1.0], EPS, "rot 90 deg around X");

    let qy_180 = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let mut v = Vector3::new(1.0, 2.0, 3.0);
    v.apply_quaternion(&qy_180);
    assert_vec_close(&v, [-1.0, 2.0, -3.0], EPS, "rot 180 deg around Y");
}

#[test]
fn test_vector3_project_and_unproject_analytical() {
    let mut matrix_world = Matrix4::identity();
    matrix_world.elements[14] = 10.0;

    let mut matrix_world_inv = Matrix4::identity();
    matrix_world_inv.elements[14] = -10.0;

    let mut proj = Matrix4::identity();
    proj.make_perspective(-1.0, 1.0, 1.0, -1.0, 1.0, 100.0, CoordinateSystem::WebGPU, false);

    let proj_inv = proj.try_invert().expect("perspective projection is invertible");

    let original = Vector3::new(2.5, -1.25, 5.0);
    let mut p = original;
    p.project(&matrix_world_inv, &proj);

    let expected_z_ndc = 80.0 / 99.0;
    assert_vec_close(&p, [0.5, -0.25, expected_z_ndc], 1e-9, "project to NDC");

    p.unproject(&proj_inv, &matrix_world);
    assert_vec_close(&p, [original.x, original.y, original.z], 1e-9, "unproject roundtrip");
}




