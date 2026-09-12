//! Focused tests for `f3d-math` transform primitives against pinned Three.js r186 semantics.
//!
//! Evaluates independently derived analytical reference cases to prevent candidate-generated goldens,
//! and tests negative cases (singular matrices, shear, negative determinants, non-unit quaternions,
//! and perspective divide).

use f3d_core::layout::{LayoutError, ProjectiveMat4};
use f3d_math::{
    BatchComposeError, CoordinateSystem, Euler, EulerOrder, Matrix3, Matrix4, NarrowingError,
    Quaternion, Vector3,
};

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

#[test]
fn test_batch_compose_matches_per_item_compose_scalar_exact() {
    let positions = [
        Vector3::zero(),
        Vector3::new(10.0, 20.0, 30.0),
        Vector3::new(-5.0, 12.5, 0.25),
        Vector3::new(-0.0, 0.0, -10.0), // explicit -0.0 translation
    ];
    let quaternions = [
        Quaternion::identity(),
        Quaternion::new(0.0, 0.0, 1.0, 0.0), // 180 deg around Z
        Quaternion::new(0.1, 0.2, 0.3, 0.4), // non-unit authored quaternion preserved
        Quaternion::new((0.5f64).sqrt(), 0.0, 0.0, (0.5f64).sqrt()), // 90 deg around X
    ];
    let scales = [
        Vector3::one(),
        Vector3::new(2.0, 3.0, 4.0),
        Vector3::new(-1.5, 2.0, -0.5), // negative scaling preserved
        Vector3::new(0.5, 1.0, 2.0),
    ];

    let mut individual_outputs = [Matrix4::identity(); 4];
    for i in 0..4 {
        individual_outputs[i].compose(&positions[i], &quaternions[i], &scales[i]);
    }

    let mut batch_outputs = [Matrix4::zero(); 4];
    Matrix4::batch_compose(&positions, &quaternions, &scales, &mut batch_outputs)
        .expect("batch_compose should succeed");

    for i in 0..4 {
        assert_eq!(
            batch_outputs[i].elements.map(f64::to_bits),
            individual_outputs[i].elements.map(f64::to_bits),
            "batch compose item {i} must bitwise equal individual compose (preserving signed zero)"
        );
    }
}

#[test]
fn test_batch_compose_valid_empty_input() {
    let mut outputs: [Matrix4; 0] = [];
    let res = Matrix4::batch_compose(&[], &[], &[], &mut outputs);
    assert!(res.is_ok(), "empty batch compose must succeed without writes");
}

#[test]
fn test_batch_compose_length_mismatch_no_partial_writes() {
    let positions = [
        Vector3::new(1.0, 2.0, 3.0),
        Vector3::new(4.0, 5.0, 6.0),
        Vector3::new(7.0, 8.0, 9.0),
    ];
    let quaternions = [
        Quaternion::identity(),
        Quaternion::identity(),
        Quaternion::identity(),
    ];
    let scales = [
        Vector3::one(),
        Vector3::one(),
        Vector3::one(),
    ];

    let sentinel = Matrix4::from_elements([42.0; 16]);

    // Case 1: quaternions slice too short (2 vs 3)
    let mut outputs1 = [sentinel; 3];
    let err1 = Matrix4::batch_compose(&positions, &quaternions[..2], &scales, &mut outputs1);
    assert_eq!(
        err1,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 3,
            quaternions_len: 2,
            scales_len: 3,
            outputs_len: 3,
        })
    );
    for (i, m) in outputs1.iter().enumerate() {
        assert_eq!(
            m.elements, sentinel.elements,
            "output1 {i} must remain untouched on length mismatch"
        );
    }

    // Case 2: outputs slice too short (2 vs 3)
    let mut outputs_short = [sentinel; 2];
    let err2 = Matrix4::batch_compose(&positions, &quaternions, &scales, &mut outputs_short);
    assert_eq!(
        err2,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 3,
            quaternions_len: 3,
            scales_len: 3,
            outputs_len: 2,
        })
    );
    for (i, m) in outputs_short.iter().enumerate() {
        assert_eq!(
            m.elements, sentinel.elements,
            "output_short {i} must remain untouched on length mismatch"
        );
    }

    // Case 3: scales slice too short (1 vs 3)
    let mut outputs3 = [sentinel; 3];
    let err3 = Matrix4::batch_compose(&positions, &quaternions, &scales[..1], &mut outputs3);
    assert_eq!(
        err3,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 3,
            quaternions_len: 3,
            scales_len: 1,
            outputs_len: 3,
        })
    );
    for (i, m) in outputs3.iter().enumerate() {
        assert_eq!(
            m.elements, sentinel.elements,
            "output3 {i} must remain untouched on length mismatch"
        );
    }

    // Case 4: positions slice too short (2 vs 3)
    let mut outputs4 = [sentinel; 3];
    let err4 = Matrix4::batch_compose(&positions[..2], &quaternions, &scales, &mut outputs4);
    assert_eq!(
        err4,
        Err(BatchComposeError::LengthMismatch {
            positions_len: 2,
            quaternions_len: 3,
            scales_len: 3,
            outputs_len: 3,
        })
    );
    for (i, m) in outputs4.iter().enumerate() {
        assert_eq!(
            m.elements, sentinel.elements,
            "output4 {i} must remain untouched on length mismatch"
        );
    }
}

#[test]
fn test_batch_compose_swapped_inputs_differentiate_per_item() {
    let pos_a = Vector3::new(10.0, 0.0, 0.0);
    let pos_b = Vector3::new(0.0, 20.0, 0.0);
    let quat_a = Quaternion::identity();
    let quat_b = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    let scale_a = Vector3::new(1.0, 2.0, 3.0);
    let scale_b = Vector3::new(3.0, 2.0, 1.0);

    let mut normal_outputs = [Matrix4::identity(); 2];
    Matrix4::batch_compose(
        &[pos_a, pos_b],
        &[quat_a, quat_b],
        &[scale_a, scale_b],
        &mut normal_outputs,
    )
    .expect("normal compose succeeds");

    // Swapped positions: pos_b with quat_a/scale_a, pos_a with quat_b/scale_b
    let mut swapped_outputs = [Matrix4::identity(); 2];
    Matrix4::batch_compose(
        &[pos_b, pos_a],
        &[quat_a, quat_b],
        &[scale_a, scale_b],
        &mut swapped_outputs,
    )
    .expect("swapped compose succeeds");

    assert_ne!(
        normal_outputs[0].elements, swapped_outputs[0].elements,
        "swapping per-item input must produce distinct matrix"
    );
    assert_ne!(
        normal_outputs[1].elements, swapped_outputs[1].elements,
        "swapping per-item input must produce distinct matrix"
    );
}

#[test]
fn test_make_rotation_from_euler_identity_all_orders() {
    let orders = [
        EulerOrder::XYZ,
        EulerOrder::YXZ,
        EulerOrder::ZXY,
        EulerOrder::ZYX,
        EulerOrder::YZX,
        EulerOrder::XZY,
    ];
    for &order in &orders {
        let euler = Euler::new(0.0, 0.0, 0.0, order);
        let mut m = Matrix4::zero();
        m.make_rotation_from_euler(&euler);
        assert_eq!(
            m,
            Matrix4::identity(),
            "make_rotation_from_euler on (0,0,0) must yield identity for {order:?}"
        );
        assert!(m.is_affine(), "identity rotation must be affine");
    }
}

#[test]
fn test_make_rotation_from_euler_oracle_all_six_orders() {
    // Exact column-major elements produced by pinned Three.js r186 Matrix4.makeRotationFromEuler
    // for euler angles (x = 0.3, y = -0.5, z = 0.7) across all six EulerOrder variants.
    // Trigonometric evaluation uses native f64::sin and f64::cos tested against Three.js within
    // floating-point tolerance (1e-15); no bit-identical browser transcendental claim is made
    // without a host browser differential proof.
    let test_cases: [(EulerOrder, [f64; 16]); 6] = [
        (
            EulerOrder::XYZ,
            [
                0.6712121661589577, 0.5070818727544463, 0.5406867876359134, 0.0,
                -0.5653542083811438, 0.8219543695041275, -0.06903356805788474, 0.0,
                -0.479425538604203, -0.2593433800522308, 0.8383866435942036, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        (
            EulerOrder::YXZ,
            [
                0.5799394465903427, 0.6154446635582734, 0.5337584700837362, 0.0,
                -0.673716999184971, 0.7306816499355124, -0.11049765362538344, 0.0,
                -0.45801271084729195, -0.29552020666133955, 0.8383866435942036, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        (
            EulerOrder::ZXY,
            [
                0.7624848857275728, 0.4569914175773167, 0.45801271084729195, 0.0,
                -0.6154446635582734, 0.7306816499355124, 0.29552020666133955, 0.0,
                -0.19961128508842896, -0.5072111697391846, 0.8383866435942036, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        (
            EulerOrder::ZYX,
            [
                0.6712121661589577, 0.5653542083811438, 0.479425538604203, 0.0,
                -0.7238074543621006, 0.6394089303668974, 0.2593433800522308, 0.0,
                -0.15992809950116813, -0.5210862105571308, 0.8383866435942036, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        (
            EulerOrder::YZX,
            [
                0.6712121661589577, 0.644217687237691, 0.3666848775860826, 0.0,
                -0.6817834387942662, 0.7306816499355124, -0.035716509255276974, 0.0,
                -0.29093911834963826, -0.22602632124962302, 0.9296593631628186, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        (
            EulerOrder::XZY,
            [
                0.6712121661589577, 0.39842357030019004, 0.6250863033449456, 0.0,
                -0.644217687237691, 0.7306816499355124, 0.22602632124962302, 0.0,
                -0.3666848775860826, -0.5544032693597385, 0.7471139240255885, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
    ];

    for (order, expected) in test_cases {
        let euler = Euler::new(0.3, -0.5, 0.7, order);
        let mut m = Matrix4::zero();
        m.make_rotation_from_euler(&euler);

        assert_mat_close(
            &m,
            &expected,
            1e-15,
            &format!("make_rotation_from_euler oracle match for {order:?}"),
        );
        assert!(m.is_affine(), "rotation matrix must be affine for {order:?}");
    }
}

#[test]
fn test_make_rotation_from_euler_elementary_axes() {
    use core::f64::consts::FRAC_PI_2;

    let orders = [
        EulerOrder::XYZ,
        EulerOrder::YXZ,
        EulerOrder::ZXY,
        EulerOrder::ZYX,
        EulerOrder::YZX,
        EulerOrder::XZY,
    ];

    for &order in &orders {
        // Rotation around X by 90 degrees: Y -> Z, Z -> -Y
        let ex = Euler::new(FRAC_PI_2, 0.0, 0.0, order);
        let mut mx = Matrix4::zero();
        mx.make_rotation_from_euler(&ex);
        let mut v_y = Vector3::new(0.0, 1.0, 0.0);
        v_y.apply_matrix4(&mx);
        assert_vec_close(&v_y, [0.0, 0.0, 1.0], 1e-12, "X 90deg rotation (0,1,0)");
        let mut v_z = Vector3::new(0.0, 0.0, 1.0);
        v_z.apply_matrix4(&mx);
        assert_vec_close(&v_z, [0.0, -1.0, 0.0], 1e-12, "X 90deg rotation (0,0,1)");

        // Rotation around Y by 90 degrees: Z -> X, X -> -Z
        let ey = Euler::new(0.0, FRAC_PI_2, 0.0, order);
        let mut my = Matrix4::zero();
        my.make_rotation_from_euler(&ey);
        let mut v_z2 = Vector3::new(0.0, 0.0, 1.0);
        v_z2.apply_matrix4(&my);
        assert_vec_close(&v_z2, [1.0, 0.0, 0.0], 1e-12, "Y 90deg rotation (0,0,1)");
        let mut v_x = Vector3::new(1.0, 0.0, 0.0);
        v_x.apply_matrix4(&my);
        assert_vec_close(&v_x, [0.0, 0.0, -1.0], 1e-12, "Y 90deg rotation (1,0,0)");

        // Rotation around Z by 90 degrees: X -> Y, Y -> -X
        let ez = Euler::new(0.0, 0.0, FRAC_PI_2, order);
        let mut mz = Matrix4::zero();
        mz.make_rotation_from_euler(&ez);
        let mut v_x2 = Vector3::new(1.0, 0.0, 0.0);
        v_x2.apply_matrix4(&mz);
        assert_vec_close(&v_x2, [0.0, 1.0, 0.0], 1e-12, "Z 90deg rotation (1,0,0)");
        let mut v_y2 = Vector3::new(0.0, 1.0, 0.0);
        v_y2.apply_matrix4(&mz);
        assert_vec_close(&v_y2, [-1.0, 0.0, 0.0], 1e-12, "Z 90deg rotation (0,1,0)");
    }
}

#[test]
fn test_make_rotation_from_euler_roundtrip_set_from_rotation_matrix() {
    let orders = [
        EulerOrder::XYZ,
        EulerOrder::YXZ,
        EulerOrder::ZXY,
        EulerOrder::ZYX,
        EulerOrder::YZX,
        EulerOrder::XZY,
    ];
    let angles = [
        (0.2, 0.3, 0.4),
        (-0.5, 0.6, -0.7),
        (0.8, -0.4, 0.2),
        (-0.1, -0.2, -0.3),
    ];

    for &order in &orders {
        for &(x, y, z) in &angles {
            let e1 = Euler::new(x, y, z, order);
            let mut m = Matrix4::zero();
            m.make_rotation_from_euler(&e1);

            let mut e2 = Euler::default();
            e2.set_from_rotation_matrix(&m, order);

            assert_close(e1.x, e2.x, 1e-12, &format!("Roundtrip X for {order:?}"));
            assert_close(e1.y, e2.y, 1e-12, &format!("Roundtrip Y for {order:?}"));
            assert_close(e1.z, e2.z, 1e-12, &format!("Roundtrip Z for {order:?}"));
        }
    }
}

#[test]
fn test_get_max_scale_on_axis_finite_cases() {
    // 1. Identity matrix: all axes length 1.0
    let m_id = Matrix4::identity();
    assert_close(m_id.get_max_scale_on_axis(), 1.0, EPS, "identity max scale");

    // 2. Pure scale matrix (2, 5, 3)
    let mut m_scale = Matrix4::identity();
    m_scale.elements[0] = 2.0;
    m_scale.elements[5] = 5.0;
    m_scale.elements[10] = 3.0;
    assert_close(m_scale.get_max_scale_on_axis(), 5.0, EPS, "diagonal scale max");

    // 3. Negative scales (-4, 2, -3) -> norms are (4, 2, 3) -> max 4
    let mut m_neg = Matrix4::identity();
    m_neg.elements[0] = -4.0;
    m_neg.elements[5] = 2.0;
    m_neg.elements[10] = -3.0;
    assert_close(m_neg.get_max_scale_on_axis(), 4.0, EPS, "negative scale max");

    // 4. Upstream Three.js Matrix4.tests.js test vector:
    // set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    // col 0: (1, 5, 9) -> norm^2 = 1 + 25 + 81 = 107
    // col 1: (2, 6, 10) -> norm^2 = 4 + 36 + 100 = 140
    // col 2: (3, 7, 11) -> norm^2 = 9 + 49 + 121 = 179
    // max norm = sqrt(179)
    let mut m_three = Matrix4::zero();
    m_three.set(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    );
    let expected_three = (3.0 * 3.0 + 7.0 * 7.0 + 11.0 * 11.0f64).sqrt();
    assert_close(
        m_three.get_max_scale_on_axis(),
        expected_three,
        1e-12,
        "upstream Three.js test vector match",
    );
}

#[test]
fn test_get_max_scale_on_axis_nan_propagation_defect_regression() {
    // Upstream Three.js evaluates Math.sqrt(Math.max(scaleXSq, scaleYSq, scaleZSq)).
    // Under ECMAScript semantics, Math.max propagates NaN if ANY argument is NaN.
    // Rust's f64::max drops NaN (IEEE 754-2008 maxNum), which caused get_max_scale_on_axis
    // to incorrectly return a finite float when axis elements contained NaN.

    // 1. Single NaN in any 3x3 basis column component must strictly propagate NaN.
    let axis_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10];
    for &idx in &axis_indices {
        let mut m = Matrix4::identity();
        m.elements[idx] = f64::NAN;
        assert!(
            m.get_max_scale_on_axis().is_nan(),
            "element [{idx}] = NaN must propagate NaN per Three.js r186 Math.max"
        );
    }

    // 2. All 3x3 axis components NaN must return NaN.
    let mut m_all = Matrix4::zero();
    for &idx in &axis_indices {
        m_all.elements[idx] = f64::NAN;
    }
    assert!(
        m_all.get_max_scale_on_axis().is_nan(),
        "all axis components NaN must return NaN"
    );

    // 3. Translation/perspective column components (te[12], te[13], te[14], te[15]) containing NaN
    // do not affect axis vectors (columns 0, 1, 2 unaffected), so max scale remains 1.0 matching Three.js.
    let mut m_trans = Matrix4::identity();
    m_trans.elements[12] = f64::NAN;
    m_trans.elements[13] = f64::NAN;
    m_trans.elements[14] = f64::NAN;
    assert_close(
        m_trans.get_max_scale_on_axis(),
        1.0,
        EPS,
        "translation column NaN does not affect axis scale",
    );
}

#[test]
fn test_quaternion_slerp_endpoints_and_symmetry() {
    // Vector from Three.js r186 unit/src/math/Quaternion.tests.js slerpTestSkeleton
    let a = Quaternion::new(
        0.6753410084407496,
        0.4087830051091744,
        0.32856700410659473,
        0.5185120064806223,
    );
    let b = Quaternion::new(
        0.6602792107657797,
        0.43647413932562285,
        0.35119011210236006,
        0.5001871596632682,
    );

    // t = 0 yields exactly a
    let mut s0 = a;
    s0.slerp(&b, 0.0);
    assert_close(s0.x, a.x, 1e-15, "slerp @ t=0 (x)");
    assert_close(s0.y, a.y, 1e-15, "slerp @ t=0 (y)");
    assert_close(s0.z, a.z, 1e-15, "slerp @ t=0 (z)");
    assert_close(s0.w, a.w, 1e-15, "slerp @ t=0 (w)");

    // t = 1 yields exactly b
    let mut s1 = a;
    s1.slerp(&b, 1.0);
    assert_close(s1.x, b.x, 1e-15, "slerp @ t=1 (x)");
    assert_close(s1.y, b.y, 1e-15, "slerp @ t=1 (y)");
    assert_close(s1.z, b.z, 1e-15, "slerp @ t=1 (z)");
    assert_close(s1.w, b.w, 1e-15, "slerp @ t=1 (w)");

    // t = 0.5: symmetry dotA == dotB and unit length
    let mut s05 = a;
    s05.slerp(&b, 0.5);
    let dot_a = s05.dot(&a);
    let dot_b = s05.dot(&b);
    assert_close(dot_a, dot_b, 1e-14, "slerp symmetry @ t=0.5");
    assert_close(s05.length(), 1.0, 1e-14, "slerp unit length @ t=0.5");

    // Analytical values matching Three.js r186 exactly
    assert_close(s05.x, 0.6679638643334043, 1e-14, "slerp @ t=0.5 (x)");
    assert_close(s05.y, 0.4227258770367419, 1e-14, "slerp @ t=0.5 (y)");
    assert_close(s05.z, 0.3399568107922252, 1e-14, "slerp @ t=0.5 (z)");
    assert_close(s05.w, 0.5094668542940572, 1e-14, "slerp @ t=0.5 (w)");

    // t = 0.25
    let mut s025 = a;
    s025.slerp(&b, 0.25);
    assert!(s025.dot(&a) > s025.dot(&b), "closer to a at t=0.25");
    assert_close(s025.x, 0.6716910906690009, 1e-14, "slerp @ t=0.25 (x)");
    assert_close(s025.y, 0.4157783681645695, 1e-14, "slerp @ t=0.25 (y)");
    assert_close(s025.z, 0.3342811445626951, 1e-14, "slerp @ t=0.25 (z)");
    assert_close(s025.w, 0.5140190110026578, 1e-14, "slerp @ t=0.25 (w)");

    // t = 0.75
    let mut s075 = a;
    s075.slerp(&b, 0.75);
    assert!(s075.dot(&a) < s075.dot(&b), "closer to b at t=0.75");
    assert_close(s075.x, 0.6641597584206161, 1e-14, "slerp @ t=0.75 (x)");
    assert_close(s075.y, 0.4296247320992852, 1e-14, "slerp @ t=0.75 (y)");
    assert_close(s075.z, 0.34559334955203336, 1e-14, "slerp @ t=0.75 (z)");
    assert_close(s075.w, 0.5048560602871799, 1e-14, "slerp @ t=0.75 (w)");
}

#[test]
fn test_quaternion_slerp_canonical_axes_and_diagonals() {
    let d = (0.5f64).sqrt();

    // 1. Diagonal X/Z from orthogonal axes
    let mut q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let q2 = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    q1.slerp(&q2, 0.5);
    assert_close(q1.x, d, 1e-15, "X/Z diagonal (x)");
    assert_close(q1.y, 0.0, 1e-15, "X/Z diagonal (y)");
    assert_close(q1.z, d, 1e-15, "X/Z diagonal (z)");
    assert_close(q1.w, 0.0, 1e-15, "X/Z diagonal (w)");

    // 2. W-unit from diagonals
    let mut q3 = Quaternion::new(0.0, d, 0.0, d);
    let q4 = Quaternion::new(0.0, -d, 0.0, d);
    q3.slerp(&q4, 0.5);
    assert_close(q3.x, 0.0, 1e-15, "W-unit (x)");
    assert_close(q3.y, 0.0, 1e-15, "W-unit (y)");
    assert_close(q3.z, 0.0, 1e-15, "W-unit (z)");
    assert_close(q3.w, 1.0, 1e-15, "W-unit (w)");
}

#[test]
fn test_quaternion_slerp_antipodal_path() {
    // Quaternions q and -q represent identical orientation; dot product is -1.0.
    // slerp must negate the target to take the shortest zero-length arc.
    let q = Quaternion::new(0.0, 0.0, 0.0, 1.0);
    let minus_q = Quaternion::new(0.0, 0.0, 0.0, -1.0);

    let mut res = q;
    res.slerp(&minus_q, 0.5);
    assert_close(res.x, 0.0, 1e-15, "antipodal slerp (x)");
    assert_close(res.y, 0.0, 1e-15, "antipodal slerp (y)");
    assert_close(res.z, 0.0, 1e-15, "antipodal slerp (z)");
    assert_close(res.w, 1.0, 1e-15, "antipodal slerp (w)");

    // At t=1, slerp to -q yields +q (since -q is flipped to +q before interpolating)
    let mut res1 = q;
    res1.slerp(&minus_q, 1.0);
    assert_close(res1.x, 0.0, 1e-15, "antipodal slerp t=1 (x)");
    assert_close(res1.y, 0.0, 1e-15, "antipodal slerp t=1 (y)");
    assert_close(res1.z, 0.0, 1e-15, "antipodal slerp t=1 (z)");
    assert_close(res1.w, 1.0, 1e-15, "antipodal slerp t=1 (w)");
}

#[test]
fn test_quaternion_slerp_small_angles_lerp_normalize() {
    // dot >= 0.9995 triggers linear interpolation followed by normalization.
    let q1 = Quaternion::new(0.0, 0.0, 0.0, 1.0);
    let q2 = Quaternion::new(0.001, 0.0, 0.0, (1.0 - 0.001 * 0.001f64).sqrt());
    assert!(q1.dot(&q2) >= 0.9995, "dot must be >= 0.9995");

    let mut res = q1;
    res.slerp(&q2, 0.5);
    assert_close(res.length(), 1.0, 1e-14, "small-angle slerp must normalize result");
    assert!(res.x > 0.0 && res.x < 0.001, "x must interpolate intermediate value");
}

#[test]
fn test_quaternion_slerp_nonunit_authored_and_extrapolation() {
    // Non-unit quaternions at dot < 0.9995 (orthogonal non-unit)
    let g = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    let h = Quaternion::new(0.0, 2.0, 0.0, 0.0);

    let mut res0 = g;
    res0.slerp(&h, 0.0);
    assert_close(res0.x, 2.0, 1e-15, "nonunit slerp @ t=0 preserves authored magnitude");
    assert_close(res0.y, 0.0, 1e-15, "nonunit slerp @ t=0 (y)");

    let mut res1 = g;
    res1.slerp(&h, 1.0);
    assert_close(res1.x, 0.0, 1e-15, "nonunit slerp @ t=1 (x)");
    assert_close(res1.y, 2.0, 1e-15, "nonunit slerp @ t=1 preserves authored magnitude");

    // Extrapolation: t = 2.0
    // q_a is identity (0 deg rotation); q_b has half-angle PI/4, representing a 90 deg rotation around Y.
    // Extrapolating to t = 2.0 produces a 180 deg rotation around Y: [0, sin(PI/2), 0, cos(PI/2)] = [0, 1, 0, ~0].
    let q_a = Quaternion::identity();
    let q_b = Quaternion::new(0.0, (core::f64::consts::FRAC_PI_4).sin(), 0.0, (core::f64::consts::FRAC_PI_4).cos());
    let mut ext = q_a;
    ext.slerp(&q_b, 2.0);
    assert_close(ext.x, 0.0, 1e-15, "extrapolated slerp (x)");
    assert_close(ext.y, 1.0, 1e-15, "extrapolated slerp to 180 deg around Y (y)");
    assert_close(ext.z, 0.0, 1e-15, "extrapolated slerp (z)");
    assert_close(ext.w, 2.220446049250313e-16, 1e-15, "extrapolated slerp cos(PI/2) (w)");
}

#[test]
fn test_quaternion_slerp_nonunit_same_input_t0_normalizes_matching_pinned_r186() {
    // When slerping between identical non-unit quaternions, dot product is >= 0.9995 (e.g. 2*2 = 4.0).
    // Pinned Three.js r186 Quaternion.js:746-756 executes the small-angle branch (lerp then normalize),
    // with NO early return at t=0.
    // Consequently, authored non-unit quaternions with dot >= 0.9995 are normalized even at t=0.
    let mut a = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    let b = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    a.slerp(&b, 0.0);

    assert_close(a.x, 1.0, 1e-15, "nonunit same-input slerp @ t=0 must normalize to 1.0 per r186");
    assert_close(a.y, 0.0, 1e-15, "y must remain 0");
    assert_close(a.z, 0.0, 1e-15, "z must remain 0");
    assert_close(a.w, 0.0, 1e-15, "w must remain 0");
}

#[test]
fn test_quaternion_slerp_nan_propagation() {
    let mut q = Quaternion::identity();
    let q_target = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    q.slerp(&q_target, f64::NAN);
    assert!(q.x.is_nan(), "slerp with NaN t propagates NaN to x");
    assert!(q.y.is_nan(), "slerp with NaN t propagates NaN to y");
    assert!(q.z.is_nan(), "slerp with NaN t propagates NaN to z");
    assert!(q.w.is_nan(), "slerp with NaN t propagates NaN to w");
}

#[test]
fn test_quaternion_slerp_quaternions_and_equals() {
    let qa = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let qb = Quaternion::new(0.0, 0.0, 1.0, 0.0);
    let mut target = Quaternion::identity();
    target.slerp_quaternions(&qa, &qb, 0.5);

    let d = (0.5f64).sqrt();
    let expected = Quaternion::new(d, 0.0, d, 0.0);
    assert_close(target.x, expected.x, 1e-15, "slerp_quaternions (x)");
    assert_close(target.z, expected.z, 1e-15, "slerp_quaternions (z)");

    let mut direct = qa;
    direct.slerp(&qb, 0.5);
    assert!(target.equals(&direct), "slerp_quaternions must equal copy().slerp()");
}

#[test]
fn test_quaternion_slerp_flat() {
    let src0 = [10.0, 0.6753410084407496, 0.4087830051091744, 0.32856700410659473, 0.5185120064806223];
    let src1 = [20.0, 0.6602792107657797, 0.43647413932562285, 0.35119011210236006, 0.5001871596632682];
    let mut dst = [0.0; 6];

    // Offsets: src0 starts at index 1, src1 starts at index 1, dst starts at index 2
    Quaternion::slerp_flat(&mut dst, 2, &src0, 1, &src1, 1, 0.5);
    assert_close(dst[2], 0.6679638643334043, 1e-14, "slerp_flat (x)");
    assert_close(dst[3], 0.4227258770367419, 1e-14, "slerp_flat (y)");
    assert_close(dst[4], 0.3399568107922252, 1e-14, "slerp_flat (z)");
    assert_close(dst[5], 0.5094668542940572, 1e-14, "slerp_flat (w)");

    // Identical quaternions early return path in slerp_flat
    let identical_src = [0.0, 1.0, 2.0, 3.0, 4.0];
    let mut dst_ident = [0.0; 5];
    Quaternion::slerp_flat(&mut dst_ident, 1, &identical_src, 1, &identical_src, 1, 0.5);
    assert_eq!(dst_ident[1], 1.0, "identical slerp_flat preserves x");
    assert_eq!(dst_ident[2], 2.0, "identical slerp_flat preserves y");
    assert_eq!(dst_ident[3], 3.0, "identical slerp_flat preserves z");
    assert_eq!(dst_ident[4], 4.0, "identical slerp_flat preserves w");
}

#[test]
fn test_quaternion_angle_to_cases() {
    let id = Quaternion::identity();
    let rot_y_pi = Euler::new(0.0, core::f64::consts::PI, 0.0, EulerOrder::XYZ).to_quaternion();
    let rot_y_2pi = Euler::new(0.0, core::f64::consts::PI * 2.0, 0.0, EulerOrder::XYZ).to_quaternion();
    let rot_y_half_pi = Euler::new(0.0, core::f64::consts::FRAC_PI_2, 0.0, EulerOrder::XYZ).to_quaternion();

    // 1. Same orientation: angle = 0
    assert_close(id.angle_to(&id), 0.0, 1e-15, "angleTo(self) == 0");

    // 2. 180 degrees rotation around Y: angle = PI
    assert_close(id.angle_to(&rot_y_pi), core::f64::consts::PI, 1e-14, "angleTo(180 deg) == PI");

    // 3. 360 degrees rotation around Y: angle = 0
    assert_close(id.angle_to(&rot_y_2pi), 0.0, 1e-14, "angleTo(360 deg) == 0");

    // 4. 90 degrees rotation around Y: angle = PI / 2
    assert_close(id.angle_to(&rot_y_half_pi), core::f64::consts::FRAC_PI_2, 1e-14, "angleTo(90 deg) == PI/2");

    // 5. Antipodal quaternion: angle = 0
    let minus_id = Quaternion::new(0.0, 0.0, 0.0, -1.0);
    assert_close(id.angle_to(&minus_id), 0.0, 1e-15, "angleTo(-q) == 0");

    // 6. Non-unit or out-of-range dot clamped to [-1, 1] without NaN
    let q_big1 = Quaternion::new(0.0, 0.0, 0.0, 2.0);
    let q_big2 = Quaternion::new(0.0, 0.0, 0.0, 2.0);
    assert_close(q_big1.angle_to(&q_big2), 0.0, 1e-15, "clamped dot prevents acos(>1) NaN");

    // 7. NaN dot propagates NaN
    let q_nan = Quaternion::new(f64::NAN, 0.0, 0.0, 1.0);
    assert!(id.angle_to(&q_nan).is_nan(), "angleTo with NaN propagates NaN");
}

#[test]
fn test_quaternion_rotate_towards_cases() {
    let id = Quaternion::identity();
    let rot_y_pi = Euler::new(0.0, core::f64::consts::PI, 0.0, EulerOrder::XYZ).to_quaternion();

    // 1. step = 0: no rotation
    let mut q1 = id;
    q1.rotate_towards(&rot_y_pi, 0.0);
    assert_close(q1.x, 0.0, 1e-15, "step 0 (x)");
    assert_close(q1.y, 0.0, 1e-15, "step 0 (y)");
    assert_close(q1.z, 0.0, 1e-15, "step 0 (z)");
    assert_close(q1.w, 1.0, 1e-15, "step 0 (w)");

    // 2. step >= angle (overshoot): clamps to target
    let mut q2 = id;
    q2.rotate_towards(&rot_y_pi, core::f64::consts::PI * 2.0);
    assert_close(q2.x, rot_y_pi.x, 1e-15, "overshoot clamp (x)");
    assert_close(q2.y, rot_y_pi.y, 1e-15, "overshoot clamp (y)");
    assert_close(q2.z, rot_y_pi.z, 1e-15, "overshoot clamp (z)");
    assert_close(q2.w, rot_y_pi.w, 1e-15, "overshoot clamp (w)");

    // 3. step = PI / 2 (halfway to 180 deg): rotates 90 deg around Y
    let mut q3 = id;
    q3.rotate_towards(&rot_y_pi, core::f64::consts::FRAC_PI_2);
    let d = (0.5f64).sqrt();
    assert_close(q3.x, 0.0, 1e-15, "half step (x)");
    assert_close(q3.y, d, 1e-15, "half step (y)");
    assert_close(q3.z, 0.0, 1e-15, "half step (z)");
    assert_close(q3.w, d, 1e-15, "half step (w)");
    assert_close(q3.angle_to(&id), core::f64::consts::FRAC_PI_2, 1e-14, "angle from start is PI/2");
    assert_close(q3.angle_to(&rot_y_pi), core::f64::consts::FRAC_PI_2, 1e-14, "angle to target is PI/2");

    // 4. angle == 0 early return: non-unit quaternion is not normalized
    let mut q_nonunit = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    let q_same = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    q_nonunit.rotate_towards(&q_same, 1.0);
    assert_eq!(q_nonunit.x, 2.0, "angle 0 early return preserves non-unit x");
    assert_eq!(q_nonunit.y, 0.0, "angle 0 early return preserves non-unit y");
    assert_eq!(q_nonunit.z, 0.0, "angle 0 early return preserves non-unit z");
    assert_eq!(q_nonunit.w, 0.0, "angle 0 early return preserves non-unit w");
}

#[test]
fn test_quaternion_set_from_axis_angle_cases() {
    let mut q = Quaternion::identity();

    // 1. Ordinary rotations matching pinned Three.js r186
    // Y-axis 90 degrees: [0, sin(PI/4), 0, cos(PI/4)]
    let axis_y = Vector3::new(0.0, 1.0, 0.0);
    q.set_from_axis_angle(&axis_y, core::f64::consts::FRAC_PI_2);
    assert_close(q.x, 0.0, 1e-15, "axis_angle y 90 (x)");
    assert_close(q.y, 0.7071067811865475, 1e-15, "axis_angle y 90 (y)");
    assert_close(q.z, 0.0, 1e-15, "axis_angle y 90 (z)");
    assert_close(q.w, 0.7071067811865476, 1e-15, "axis_angle y 90 (w)");

    // X-axis 180 degrees: [sin(PI/2), 0, 0, cos(PI/2)]
    let axis_x = Vector3::new(1.0, 0.0, 0.0);
    q.set_from_axis_angle(&axis_x, core::f64::consts::PI);
    assert_close(q.x, 1.0, 1e-15, "axis_angle x 180 (x)");
    assert_close(q.y, 0.0, 1e-15, "axis_angle x 180 (y)");
    assert_close(q.z, 0.0, 1e-15, "axis_angle x 180 (z)");
    assert_close(q.w, 6.123233995736766e-17, 1e-15, "axis_angle x 180 (w)");

    // 2. Authored non-unit axis (no implicit normalization beyond source)
    // axis (0, 2, 0) with angle PI/2 scales components by 2 without normalizing
    let axis_nonunit = Vector3::new(0.0, 2.0, 0.0);
    q.set_from_axis_angle(&axis_nonunit, core::f64::consts::FRAC_PI_2);
    assert_close(q.x, 0.0, 1e-15, "nonunit axis_angle (x)");
    assert_close(q.y, 1.414213562373095, 1e-15, "nonunit axis_angle preserves magnitude (y)");
    assert_close(q.z, 0.0, 1e-15, "nonunit axis_angle (z)");
    assert_close(q.w, 0.7071067811865476, 1e-15, "nonunit axis_angle (w)");

    // 3. Zero angle
    q.set_from_axis_angle(&axis_x, 0.0);
    assert_close(q.x, 0.0, 1e-15, "zero angle (x)");
    assert_close(q.y, 0.0, 1e-15, "zero angle (y)");
    assert_close(q.z, 0.0, 1e-15, "zero angle (z)");
    assert_close(q.w, 1.0, 1e-15, "zero angle (w)");

    // 4. Signed zero angle: sin(-0.0) produces -0.0 for x, y, z
    q.set_from_axis_angle(&axis_x, -0.0);
    assert_eq!(q.x.to_bits(), (-0.0f64).to_bits(), "signed zero angle preserves -0.0 on x");
    assert_eq!(q.y.to_bits(), (-0.0f64).to_bits(), "signed zero angle preserves -0.0 on y");
    assert_eq!(q.z.to_bits(), (-0.0f64).to_bits(), "signed zero angle preserves -0.0 on z");
    assert_close(q.w, 1.0, 1e-15, "signed zero angle w is 1.0");

    // 5. NaN propagation
    let axis_nan = Vector3::new(f64::NAN, 0.0, 0.0);
    q.set_from_axis_angle(&axis_nan, core::f64::consts::FRAC_PI_2);
    assert!(q.x.is_nan(), "NaN axis x propagates NaN");
    assert_close(q.y, 0.0, 1e-15, "NaN axis y remains 0");
    assert_close(q.z, 0.0, 1e-15, "NaN axis z remains 0");
    assert_close(q.w, 0.7071067811865476, 1e-15, "NaN axis w computed from angle");

    q.set_from_axis_angle(&axis_y, f64::NAN);
    assert!(q.x.is_nan(), "NaN angle propagates NaN to x");
    assert!(q.y.is_nan(), "NaN angle propagates NaN to y");
    assert!(q.z.is_nan(), "NaN angle propagates NaN to z");
    assert!(q.w.is_nan(), "NaN angle propagates NaN to w");
}

#[test]
fn test_quaternion_set_from_unit_vectors_cases() {
    let mut q = Quaternion::identity();

    // 1. Ordinary rotation: 90 deg from X to Y around Z
    let v_x = Vector3::new(1.0, 0.0, 0.0);
    let v_y = Vector3::new(0.0, 1.0, 0.0);
    q.set_from_unit_vectors(&v_x, &v_y);
    assert_close(q.x, 0.0, 1e-15, "ordinary x to y (x)");
    assert_close(q.y, 0.0, 1e-15, "ordinary x to y (y)");
    assert_close(q.z, 0.7071067811865475, 1e-15, "ordinary x to y (z)");
    assert_close(q.w, 0.7071067811865475, 1e-15, "ordinary x to y (w)");

    // 2. Identical vectors: identity rotation
    q.set_from_unit_vectors(&v_y, &v_y);
    assert_close(q.x, 0.0, 1e-15, "identical vectors (x)");
    assert_close(q.y, 0.0, 1e-15, "identical vectors (y)");
    assert_close(q.z, 0.0, 1e-15, "identical vectors (z)");
    assert_close(q.w, 1.0, 1e-15, "identical vectors (w)");

    // 3. Antipodal branch 1: |v_from.x| > |v_from.z|
    // v_from = (1, 0, 0), v_to = (-1, 0, 0).
    // |1| > |0| -> x = -v_from.y = -0.0, y = v_from.x = 1.0, z = 0.0, w = 0.0
    let v_neg_x = Vector3::new(-1.0, 0.0, 0.0);
    q.set_from_unit_vectors(&v_x, &v_neg_x);
    assert_eq!(q.x.to_bits(), (-0.0f64).to_bits(), "antipodal branch 1 preserves -0.0 on x");
    assert_close(q.y, 1.0, 1e-15, "antipodal branch 1 (y)");
    assert_close(q.z, 0.0, 1e-15, "antipodal branch 1 (z)");
    assert_close(q.w, 0.0, 1e-15, "antipodal branch 1 (w)");

    // Antipodal branch 1 with general (x, y):
    let v_xy = Vector3::new(0.6, 0.8, 0.0);
    let v_neg_xy = Vector3::new(-0.6, -0.8, 0.0);
    q.set_from_unit_vectors(&v_xy, &v_neg_xy);
    assert_close(q.x, -0.8, 1e-15, "antipodal branch 1 general (x)");
    assert_close(q.y, 0.6, 1e-15, "antipodal branch 1 general (y)");
    assert_close(q.z, 0.0, 1e-15, "antipodal branch 1 general (z)");
    assert_close(q.w, 0.0, 1e-15, "antipodal branch 1 general (w)");

    // 4. Antipodal branch 2: |v_from.x| <= |v_from.z|
    // v_from = (0, 0, 1), v_to = (0, 0, -1).
    // |0| <= |1| -> x = 0.0, y = -v_from.z = -1.0, z = v_from.y = 0.0, w = 0.0
    let v_z = Vector3::new(0.0, 0.0, 1.0);
    let v_neg_z = Vector3::new(0.0, 0.0, -1.0);
    q.set_from_unit_vectors(&v_z, &v_neg_z);
    assert_close(q.x, 0.0, 1e-15, "antipodal branch 2 (x)");
    assert_close(q.y, -1.0, 1e-15, "antipodal branch 2 (y)");
    assert_close(q.z, 0.0, 1e-15, "antipodal branch 2 (z)");
    assert_close(q.w, 0.0, 1e-15, "antipodal branch 2 (w)");

    // 5. Near threshold tests (r < 1e-8 vs r >= 1e-8)
    // Below threshold: dot = -1.0 + 0.5e-8 -> r = 0.5e-8 < 1e-8 (enters antipodal branch)
    let dot_below = -1.0 + 0.5e-8;
    let v_to_below = Vector3::new(dot_below, (1.0 - dot_below * dot_below).sqrt(), 0.0);
    q.set_from_unit_vectors(&v_x, &v_to_below);
    assert_eq!(q.x.to_bits(), (-0.0f64).to_bits(), "near threshold below (x is -0.0)");
    assert_close(q.y, 1.0, 1e-15, "near threshold below triggers antipodal branch (y)");
    assert_close(q.z, 0.0, 1e-15, "near threshold below (z)");
    assert_close(q.w, 0.0, 1e-15, "near threshold below (w)");

    // Above threshold: dot = -1.0 + 2.0e-8 -> r = 2.0e-8 >= 1e-8 (regular cross product branch)
    let dot_above = -1.0 + 2.0e-8;
    let v_to_above = Vector3::new(dot_above, (1.0 - dot_above * dot_above).sqrt(), 0.0);
    q.set_from_unit_vectors(&v_x, &v_to_above);
    assert_close(q.x, 0.0, 1e-15, "near threshold above (x)");
    assert_close(q.y, 0.0, 1e-15, "near threshold above (y)");
    assert_close(q.z, 0.999999995, 1e-14, "near threshold above regular branch (z)");
    assert_close(q.w, 0.00010000000002879373, 1e-14, "near threshold above regular branch (w)");

    // 6. Non-unit authored inputs (no implicit normalization of inputs beyond source)
    // v_from = (2, 0, 0), v_to = (0, 2, 0). Upstream yields normalized [0, 0, 4/sqrt(17), 1/sqrt(17)]
    let v_from_nonunit = Vector3::new(2.0, 0.0, 0.0);
    let v_to_nonunit = Vector3::new(0.0, 2.0, 0.0);
    q.set_from_unit_vectors(&v_from_nonunit, &v_to_nonunit);
    assert_close(q.x, 0.0, 1e-15, "nonunit inputs (x)");
    assert_close(q.y, 0.0, 1e-15, "nonunit inputs (y)");
    assert_close(q.z, 0.9701425001453319, 1e-15, "nonunit inputs 4/sqrt(17) (z)");
    assert_close(q.w, 0.24253562503633297, 1e-15, "nonunit inputs 1/sqrt(17) (w)");

    // 7. NaN propagation
    let v_nan = Vector3::new(f64::NAN, 0.0, 0.0);
    q.set_from_unit_vectors(&v_nan, &v_y);
    assert!(q.x.is_nan(), "NaN input propagates NaN to x");
    assert!(q.y.is_nan(), "NaN input propagates NaN to y");
    assert!(q.z.is_nan(), "NaN input propagates NaN to z");
    assert!(q.w.is_nan(), "NaN input propagates NaN to w");
}

#[test]
fn test_matrix4_look_at_cases() {
    let mut m = Matrix4::identity();

    // 1. Ordinary camera basis looking down -Z axis
    let eye = Vector3::new(0.0, 0.0, 5.0);
    let target = Vector3::new(0.0, 0.0, 0.0);
    let up = Vector3::new(0.0, 1.0, 0.0);
    m.look_at(&eye, &target, &up);
    let expected_ordinary = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert_mat_close(&m, &expected_ordinary, 1e-15, "look_at ordinary");

    // 2. Diagonal look_at orientation
    let eye_diag = Vector3::new(1.0, 2.0, 3.0);
    let target_diag = Vector3::new(4.0, 5.0, 6.0);
    m.look_at(&eye_diag, &target_diag, &up);
    let expected_diag = [
        -0.7071067811865475, 0.0, 0.7071067811865475, 0.0,
        -0.40824829046386296, 0.8164965809277259, -0.40824829046386296, 0.0,
        -0.5773502691896257, -0.5773502691896257, -0.5773502691896257, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert_mat_close(&m, &expected_diag, 1e-14, "look_at diagonal");

    // 3. Coincident eye and target: sets z.z = 1.0 and resolves to identity upper 3x3
    m.look_at(&eye_diag, &eye_diag, &up);
    assert_mat_close(&m, &expected_ordinary, 1e-15, "look_at coincident eye/target");

    // 4. Parallel up and z with |up.z| == 1.0
    let up_z1 = Vector3::new(0.0, 0.0, 1.0);
    m.look_at(&eye, &target, &up_z1);
    let expected_parallel_z1 = [
        0.0, 0.9999999999999999, 0.0, 0.0,
        -0.9999999949999999, 0.0, 0.0000999999995, 0.0,
        0.00009999999950000001, 0.0, 0.999999995, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert_mat_close(&m, &expected_parallel_z1, 1e-14, "look_at parallel up |up.z| == 1");

    // 5. Parallel up and z with |up.z| != 1.0 (looking along +Y with up=(0,1,0))
    let eye_y = Vector3::new(0.0, 5.0, 0.0);
    m.look_at(&eye_y, &target, &up);
    let expected_parallel_y = [
        0.9999999999999999, 0.0, 0.0, 0.0,
        0.0, 0.0000999999995, -0.9999999949999999, 0.0,
        0.0, 0.999999995, 0.00009999999950000001, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert_mat_close(&m, &expected_parallel_y, 1e-14, "look_at parallel up |up.z| != 1");

    // 6. Source-defined untouched matrix elements:
    // look_at only writes elements [0..=2], [4..=6], [8..=10].
    // elements [3], [7], [11], and [12..=15] must be left untouched.
    let mut m_untouched = Matrix4::from_elements([
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]);
    m_untouched.look_at(&eye, &target, &up);
    // Columns 0, 1, 2 top 3 elements are updated to ordinary rotation:
    assert_close(m_untouched.elements[0], 1.0, 1e-15, "col 0 x");
    assert_close(m_untouched.elements[1], 0.0, 1e-15, "col 0 y");
    assert_close(m_untouched.elements[2], 0.0, 1e-15, "col 0 z");
    assert_close(m_untouched.elements[4], 0.0, 1e-15, "col 1 x");
    assert_close(m_untouched.elements[5], 1.0, 1e-15, "col 1 y");
    assert_close(m_untouched.elements[6], 0.0, 1e-15, "col 1 z");
    assert_close(m_untouched.elements[8], 0.0, 1e-15, "col 2 x");
    assert_close(m_untouched.elements[9], 0.0, 1e-15, "col 2 y");
    assert_close(m_untouched.elements[10], 1.0, 1e-15, "col 2 z");
    // Assert untouched elements:
    assert_eq!(m_untouched.elements[3], 4.0, "elements[3] preserved untouched");
    assert_eq!(m_untouched.elements[7], 8.0, "elements[7] preserved untouched");
    assert_eq!(m_untouched.elements[11], 12.0, "elements[11] preserved untouched");
    assert_eq!(m_untouched.elements[12], 13.0, "elements[12] preserved untouched");
    assert_eq!(m_untouched.elements[13], 14.0, "elements[13] preserved untouched");
    assert_eq!(m_untouched.elements[14], 15.0, "elements[14] preserved untouched");
    assert_eq!(m_untouched.elements[15], 16.0, "elements[15] preserved untouched");
}

#[test]
fn test_matrix4_extract_rotation_cases() {
    let mut rot = Matrix4::identity();
    rot.make_rotation_from_euler(&Euler::new(0.2, 0.4, 0.6, EulerOrder::XYZ));

    // 1. Pure rotation extraction
    let mut dst = Matrix4::zero();
    dst.extract_rotation(&rot);
    assert_mat_close(&dst, &rot.elements, 1e-15, "extract_rotation pure");

    // 2. Nonuniform scale: rot * scale(2, 3, 4) extracts pure rot
    let mut scaled = rot;
    scaled.scale(&Vector3::new(2.0, 3.0, 4.0));
    dst.extract_rotation(&scaled);
    assert_mat_close(&dst, &rot.elements, 1e-14, "extract_rotation nonuniform scale");

    // 3. Negative scale (reflection): rot * scale(-2, 3, 4)
    // Upstream divides by Euclidean length (positive), preserving the negative sign without reflection normalization
    let mut neg_scaled = rot;
    neg_scaled.scale(&Vector3::new(-2.0, 3.0, 4.0));
    dst.extract_rotation(&neg_scaled);
    let expected_neg = [
        -rot.elements[0], -rot.elements[1], -rot.elements[2], 0.0,
        rot.elements[4], rot.elements[5], rot.elements[6], 0.0,
        rot.elements[8], rot.elements[9], rot.elements[10], 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert_mat_close(&dst, &expected_neg, 1e-14, "extract_rotation negative scale");

    // 4. Zero scale / singular matrix (determinant_affine == 0): resets to identity
    let mut zero_scaled = rot;
    zero_scaled.scale(&Vector3::new(0.0, 3.0, 4.0));
    let mut dirty_dst = Matrix4::from_elements([
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]);
    dirty_dst.extract_rotation(&zero_scaled);
    assert_mat_close(&dirty_dst, &Matrix4::identity().elements, 1e-15, "extract_rotation zero scale resets to identity");

    // 5. Overwrites all 16 elements on valid matrix:
    // Unlike look_at, extract_rotation explicitly writes all 16 elements (col 3 is [0,0,0,1], row 3 is [0,0,0,1])
    let mut dirty_dst2 = Matrix4::from_elements([
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]);
    dirty_dst2.extract_rotation(&rot);
    assert_mat_close(&dirty_dst2, &rot.elements, 1e-15, "extract_rotation overwrites all 16 elements");
}

