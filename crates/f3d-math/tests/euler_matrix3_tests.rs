//! Comprehensive tests for Euler rotation angles and Matrix3 matching Three.js r186.

use core::f64::consts::PI;
use f3d_math::euler::{Euler, EulerOrder};
use f3d_math::matrix3::Matrix3;
use f3d_math::matrix4::Matrix4;
use f3d_math::vector2::Vector2;
use f3d_math::vector3::Vector3;

#[test]
fn test_euler_quaternion_roundtrip_all_orders() {
    let orders = [
        EulerOrder::XYZ,
        EulerOrder::YXZ,
        EulerOrder::ZXY,
        EulerOrder::ZYX,
        EulerOrder::YZX,
        EulerOrder::XZY,
    ];

    // Non-gimbal angles in radians
    let test_angles = [
        (0.3, 0.4, 0.5),
        (-0.6, 0.7, -0.8),
        (1.1, -0.9, 0.4),
        (-0.2, -0.3, -0.4),
    ];

    for &order in &orders {
        for &(x, y, z) in &test_angles {
            let e1 = Euler::new(x, y, z, order);
            let q = e1.to_quaternion();

            let mut e2 = Euler::default();
            e2.set_from_quaternion(&q, order);

            assert!(
                (e1.x - e2.x).abs() < 1e-12,
                "Euler X mismatch for order {order:?}: expected {x}, got {}",
                e2.x
            );
            assert!(
                (e1.y - e2.y).abs() < 1e-12,
                "Euler Y mismatch for order {order:?}: expected {y}, got {}",
                e2.y
            );
            assert!(
                (e1.z - e2.z).abs() < 1e-12,
                "Euler Z mismatch for order {order:?}: expected {z}, got {}",
                e2.z
            );
        }
    }
}

#[test]
fn test_euler_matrix4_roundtrip_all_orders() {
    let orders = [
        EulerOrder::XYZ,
        EulerOrder::YXZ,
        EulerOrder::ZXY,
        EulerOrder::ZYX,
        EulerOrder::YZX,
        EulerOrder::XZY,
    ];

    let test_angles = [
        (0.25, 0.35, 0.45),
        (-0.5, 0.6, -0.7),
        (0.8, -0.4, 0.3),
    ];

    for &order in &orders {
        for &(x, y, z) in &test_angles {
            let e1 = Euler::new(x, y, z, order);
            let q = e1.to_quaternion();
            let mut m = Matrix4::identity();
            m.make_rotation_from_quaternion(&q);

            let mut e2 = Euler::default();
            e2.set_from_rotation_matrix(&m, order);

            assert!(
                (e1.x - e2.x).abs() < 1e-12,
                "Matrix4 roundtrip X mismatch for order {order:?}"
            );
            assert!(
                (e1.y - e2.y).abs() < 1e-12,
                "Matrix4 roundtrip Y mismatch for order {order:?}"
            );
            assert!(
                (e1.z - e2.z).abs() < 1e-12,
                "Matrix4 roundtrip Z mismatch for order {order:?}"
            );

            // Direct make_rotation_from_euler verification
            let mut m_direct = Matrix4::identity();
            m_direct.make_rotation_from_euler(&e1);

            for i in 0..16 {
                assert!(
                    (m.elements[i] - m_direct.elements[i]).abs() < 1e-12,
                    "make_rotation_from_euler mismatch with quaternion composition at {i} for {order:?}"
                );
            }

            let mut e3 = Euler::default();
            e3.set_from_rotation_matrix(&m_direct, order);

            assert!(
                (e1.x - e3.x).abs() < 1e-12,
                "make_rotation_from_euler roundtrip X mismatch for order {order:?}"
            );
            assert!(
                (e1.y - e3.y).abs() < 1e-12,
                "make_rotation_from_euler roundtrip Y mismatch for order {order:?}"
            );
            assert!(
                (e1.z - e3.z).abs() < 1e-12,
                "make_rotation_from_euler roundtrip Z mismatch for order {order:?}"
            );
        }
    }
}

#[test]
fn test_euler_gimbal_lock_pos_neg_half_pi_all_orders() {
    // In Euler representations, gimbal lock occurs when the intermediate rotation axis
    // aligns the outer and inner axes. Upstream Three.js Euler.js checks:
    // abs(m) < 0.9999999 (Euler::GIMBAL_THRESHOLD).
    //
    // Tested axes:
    // - XYZ: pitch around Y at +-PI/2
    // - YXZ: pitch around X at +-PI/2
    // - ZXY: pitch around X at +-PI/2
    // - ZYX: pitch around Y at +-PI/2
    // - YZX: pitch around Z at +-PI/2
    // - XZY: pitch around Z at +-PI/2
    let gimbal_cases = [
        (EulerOrder::XYZ, 0.2, PI / 2.0, 0.4),
        (EulerOrder::XYZ, -0.3, -PI / 2.0, 0.5),
        (EulerOrder::YXZ, PI / 2.0, 0.2, 0.4),
        (EulerOrder::YXZ, -PI / 2.0, -0.3, 0.5),
        (EulerOrder::ZXY, PI / 2.0, 0.2, 0.4),
        (EulerOrder::ZXY, -PI / 2.0, -0.3, 0.5),
        (EulerOrder::ZYX, 0.2, PI / 2.0, 0.4),
        (EulerOrder::ZYX, -0.3, -PI / 2.0, 0.5),
        (EulerOrder::YZX, 0.2, 0.4, PI / 2.0),
        (EulerOrder::YZX, -0.3, 0.5, -PI / 2.0),
        (EulerOrder::XZY, 0.2, 0.4, PI / 2.0),
        (EulerOrder::XZY, -0.3, 0.5, -PI / 2.0),
    ];

    for &(order, x, y, z) in &gimbal_cases {
        let e1 = Euler::new(x, y, z, order);
        let q1 = e1.to_quaternion();

        let mut m1 = Matrix4::identity();
        m1.make_rotation_from_quaternion(&q1);

        // set_from_rotation_matrix must exercise the gimbal lock fallback path
        let mut e2 = Euler::default();
        e2.set_from_rotation_matrix(&m1, order);

        let q2 = e2.to_quaternion();
        let mut m2 = Matrix4::identity();
        m2.make_rotation_from_quaternion(&q2);

        // 1. Quaternions must represent the exact same orientation (|q1 . q2| == 1.0)
        let dot = (q1.dot(&q2)).abs();
        assert!(
            (dot - 1.0).abs() < 1e-12,
            "Gimbal lock orientation equivalence failed for {order:?} (dot was {dot})"
        );

        // 2. Resulting rotation matrix must match m1 within floating-point tolerance
        for i in 0..16 {
            let diff = (m1.elements[i] - m2.elements[i]).abs();
            assert!(
                diff < 1e-7,
                "Gimbal lock matrix difference at index {i} for {order:?}: {diff}"
            );
        }
    }
}

#[test]
fn test_matrix3_normal_matrix_non_uniform_scale() {
    // 1. Pure non-uniform scale: Matrix4 with scale (2, 3, 4)
    // The upper-left 3x3 is diag(2, 3, 4).
    // The normal matrix is the inverse transpose:
    // (diag(2, 3, 4)^-1)^T = diag(1/2, 1/3, 1/4) = diag(0.5, 0.3333333333333333, 0.25)
    let mut m4_scale = Matrix4::identity();
    m4_scale.scale(&Vector3::new(2.0, 3.0, 4.0));

    let mut normal_mat = Matrix3::identity();
    normal_mat.get_normal_matrix(&m4_scale);

    let te = normal_mat.elements;
    // Diagonal elements:
    assert_eq!(te[0], 0.5, "Normal matrix X scale (1/2)");
    assert_eq!(te[4], 1.0 / 3.0, "Normal matrix Y scale (1/3)");
    assert_eq!(te[8], 0.25, "Normal matrix Z scale (1/4)");

    // Off-diagonal elements must all be exactly 0.0
    assert_eq!(te[1], 0.0);
    assert_eq!(te[2], 0.0);
    assert_eq!(te[3], 0.0);
    assert_eq!(te[5], 0.0);
    assert_eq!(te[6], 0.0);
    assert_eq!(te[7], 0.0);

    // 2. Full transformation with translation, rotation, and non-uniform scale
    // Normal matrix must be invariant to translation and satisfy:
    // NormalMatrix^T * Upper3x3(Matrix4) == Identity
    let position = Vector3::new(10.0, -20.0, 30.0);
    let rotation = Euler::new(0.5, 0.3, 0.2, EulerOrder::XYZ).to_quaternion();
    let scale = Vector3::new(2.0, 4.0, 8.0);

    let mut m4_full = Matrix4::identity();
    m4_full.compose(&position, &rotation, &scale);

    let mut norm_full = Matrix3::identity();
    norm_full.get_normal_matrix(&m4_full);

    // Check that translation does not affect normal matrix:
    // compose with same rotation and scale but zero position
    let mut m4_no_trans = Matrix4::identity();
    m4_no_trans.compose(&Vector3::zero(), &rotation, &scale);

    let mut norm_no_trans = Matrix3::identity();
    norm_no_trans.get_normal_matrix(&m4_no_trans);

    assert_eq!(
        norm_full.elements, norm_no_trans.elements,
        "Translation must not affect the normal matrix"
    );

    // Verify mathematical invariant: N^T * M33 = I
    let mut m33 = Matrix3::identity();
    m33.set_from_matrix4(&m4_full);

    let mut nt = norm_full;
    nt.transpose();

    let mut prod = Matrix3::identity();
    prod.multiply_matrices(&nt, &m33);

    let pe = prod.elements;
    // Diagonal must be 1.0 within 1e-12
    assert!((pe[0] - 1.0).abs() < 1e-12, "prod[0] should be 1.0, got {}", pe[0]);
    assert!((pe[4] - 1.0).abs() < 1e-12, "prod[4] should be 1.0, got {}", pe[4]);
    assert!((pe[8] - 1.0).abs() < 1e-12, "prod[8] should be 1.0, got {}", pe[8]);

    // Off-diagonals must be 0.0 within 1e-12
    for &idx in &[1, 2, 3, 5, 6, 7] {
        assert!(
            pe[idx].abs() < 1e-12,
            "Off-diagonal prod[{idx}] should be 0.0, got {}",
            pe[idx]
        );
    }
}

#[test]
fn test_matrix3_singular_invert_yields_zeros_matching_r186() {
    // Upstream Three.js Matrix3.invert:
    // if ( det === 0 ) return this.set( 0, 0, 0, 0, 0, 0, 0, 0, 0 );

    // 1. Matrix with linearly dependent rows (det == 0)
    let mut singular = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );
    assert_eq!(singular.determinant(), 0.0, "Determinant of singular matrix must be 0");
    singular.invert();
    assert_eq!(
        singular.elements,
        [0.0; 9],
        "Inverting a singular matrix must produce all zeros matching Three.js r186"
    );

    // 2. All-zero matrix
    let mut zero_mat = Matrix3::zero();
    zero_mat.invert();
    assert_eq!(zero_mat.elements, [0.0; 9]);

    // 3. Fallible try_invert returns None for singular, Some for invertible
    let singular_test = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );
    assert!(singular_test.try_invert().is_none(), "try_invert on singular must return None");

    let invertible = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 4.0,
    );
    let inv = invertible.try_invert().expect("Invertible matrix should succeed");
    assert_eq!(inv.elements[0], 1.0);
    assert_eq!(inv.elements[4], 0.5);
    assert_eq!(inv.elements[8], 0.25);
}

#[test]
fn test_matrix3_basic_operations_and_transpose() {
    // Row-major constructor stores in column-major order:
    // row 0: 1, 2, 3
    // row 1: 4, 5, 6
    // row 2: 7, 8, 0
    let m = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 0.0,
    );
    assert_eq!(m.elements, [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 0.0]);

    // Determinant:
    // 1*(5*0 - 8*6) - 2*(4*0 - 7*6) + 3*(4*8 - 7*5) = 1*(-48) - 2*(-42) + 3*(32 - 35)
    // = -48 + 84 - 9 = 27
    assert_eq!(m.determinant(), 27.0);

    // Transpose in place:
    let mut mt = m;
    mt.transpose();
    assert_eq!(mt.elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0]);

    // Matrix multiplication with identity:
    let mut prod = Matrix3::identity();
    prod.multiply(&m);
    assert_eq!(prod.elements, m.elements);

    // Invert roundtrip: M * M^-1 == Identity
    let mut m_inv = m;
    m_inv.invert();
    let mut check_id = Matrix3::identity();
    check_id.multiply_matrices(&m, &m_inv);

    for i in 0..9 {
        let expected = if i == 0 || i == 4 || i == 8 { 1.0 } else { 0.0 };
        assert!(
            (check_id.elements[i] - expected).abs() < 1e-12,
            "M * M^-1 mismatch at index {i}"
        );
    }
}

#[test]
fn test_euler_reorder() {
    let mut e = Euler::new(0.3, 0.5, 0.7, EulerOrder::XYZ);
    let q_orig = e.to_quaternion();

    e.reorder(EulerOrder::ZYX);
    assert_eq!(e.order, EulerOrder::ZYX);

    let q_reordered = e.to_quaternion();
    let dot = (q_orig.dot(&q_reordered)).abs();
    assert!(
        (dot - 1.0).abs() < 1e-12,
        "Reordering must preserve orientation"
    );
}

#[test]
fn test_matrix3_transform_make_translate_rotate_scale_uv_basis() {
    // 1. make_translation(2.0, 3.0) and make_translation_vec(&Vector2)
    // Derived from Node Matrix3.js: new Matrix3().makeTranslation(2, 3).elements -> [1,0,0,0,1,0,2,3,1]
    let mut m_trans = Matrix3::identity();
    m_trans.make_translation(2.0, 3.0);
    assert_eq!(m_trans.elements, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 1.0]);

    let v_trans = Vector2::new(2.0, 3.0);
    let mut m_trans_v = Matrix3::identity();
    m_trans_v.make_translation_vec(&v_trans);
    assert_eq!(m_trans_v.elements, m_trans.elements);

    // 2. make_rotation(PI / 6.0)
    // Derived from Node Matrix3.js: new Matrix3().makeRotation(Math.PI / 6).elements
    // -> [0.8660254037844387, 0.49999999999999994, 0, -0.49999999999999994, 0.8660254037844387, 0, 0, 0, 1]
    let mut m_rot = Matrix3::identity();
    m_rot.make_rotation(PI / 6.0);
    let expected_rot = [
        0.8660254037844387,
        0.49999999999999994,
        0.0,
        -0.49999999999999994,
        0.8660254037844387,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    for i in 0..9 {
        assert!(
            (m_rot.elements[i] - expected_rot[i]).abs() < 1e-14,
            "make_rotation mismatch at index {i}"
        );
    }

    // 3. make_scale(2.0, 3.0)
    // Derived from Node Matrix3.js: new Matrix3().makeScale(2, 3).elements -> [2,0,0,0,3,0,0,0,1]
    let mut m_scale = Matrix3::identity();
    m_scale.make_scale(2.0, 3.0);
    assert_eq!(m_scale.elements, [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);

    // 4. Chained translate(2, 3) then rotate(PI / 6) then scale(2, 3) on non-identity start matrix
    // Start: Matrix3.set(1, 2, 3, 4, 5, 6, 7, 8, 9)
    // Derived from Node Matrix3.js:
    // m = new Matrix3().set(1, 2, 3, 4, 5, 6, 7, 8, 9); m.translate(2, 3); m.rotate(Math.PI / 6); m.scale(2, 3);
    // -> [50.98076211353316, 42.45190528383291, 7, 60.17691453623979, 48.34421012924618, 8, 69.37306695894642, 54.23651497465943, 9]
    let mut m_chain = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );
    m_chain.translate(2.0, 3.0);
    m_chain.rotate(PI / 6.0);
    m_chain.scale(2.0, 3.0);
    let expected_chain = [
        50.98076211353316,
        42.45190528383291,
        7.0,
        60.17691453623979,
        48.34421012924618,
        8.0,
        69.37306695894642,
        54.23651497465943,
        9.0,
    ];
    for i in 0..9 {
        assert!(
            (m_chain.elements[i] - expected_chain[i]).abs() < 1e-12,
            "chained transform mismatch at index {i}: expected {}, got {}",
            expected_chain[i],
            m_chain.elements[i]
        );
    }

    // 5. set_uv_transform(0.1, 0.2, 2.0, 3.0, PI / 4.0, 0.5, 0.5)
    // Derived from Node Matrix3.js:
    // new Matrix3().setUvTransform(0.1, 0.2, 2, 3, Math.PI / 4, 0.5, 0.5).elements
    // -> [1.4142135623730951, -2.1213203435596424, 0, 1.414213562373095, 2.121320343559643, 0, -0.814213562373095, 0.6999999999999998, 1]
    let mut m_uv = Matrix3::identity();
    m_uv.set_uv_transform(0.1, 0.2, 2.0, 3.0, PI / 4.0, 0.5, 0.5);
    let expected_uv = [
        1.4142135623730951,
        -2.1213203435596424,
        0.0,
        1.414213562373095,
        2.121320343559643,
        0.0,
        -0.814213562373095,
        0.6999999999999998,
        1.0,
    ];
    for i in 0..9 {
        assert!(
            (m_uv.elements[i] - expected_uv[i]).abs() < 1e-14,
            "set_uv_transform mismatch at index {i}: expected {}, got {}",
            expected_uv[i],
            m_uv.elements[i]
        );
    }

    // 6. extract_basis of a known matrix yields its three columns
    // Derived from Node Matrix3.js:
    // new Matrix3().set(1, 2, 3, 4, 5, 6, 7, 8, 9).extractBasis(x, y, z)
    // -> x = [1, 4, 7], y = [2, 5, 8], z = [3, 6, 9]
    let mut m_basis = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );
    let mut x_axis = Vector3::zero();
    let mut y_axis = Vector3::zero();
    let mut z_axis = Vector3::zero();
    m_basis.extract_basis(&mut x_axis, &mut y_axis, &mut z_axis);
    assert_eq!(x_axis, Vector3::new(1.0, 4.0, 7.0));
    assert_eq!(y_axis, Vector3::new(2.0, 5.0, 8.0));
    assert_eq!(z_axis, Vector3::new(3.0, 6.0, 9.0));

    // No-claim line: scalar parity only.
}

#[test]
fn test_matrix3_array_and_identity_parity() {
    // No-claim line: scalar parity only.
    // Note: slice-bounds behavior is a native API contract, not out-of-range Three.js parity.
    // Derived Node oracle values verified against Three.js r186 Matrix3.js.

    // 1. Matrix3 from_array offset 2 of a 12-element buffer
    let buf12 = [999.0, 888.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 777.0];
    let mut m = Matrix3::zero();
    let ret_m = m.from_array(&buf12, 2);
    let expected_elements = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(ret_m.elements, expected_elements);
    assert_eq!(m.elements, expected_elements);

    // 2. Matrix3 to_array into a 12-element sentinel buffer at offset 1
    let mut out_buf12 = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0];
    m.to_array(&mut out_buf12, 1);
    let expected_out_buf12 = [100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1100.0, 1200.0];
    assert_eq!(out_buf12, expected_out_buf12);

    // 3. Matrix3 transpose_into_array of set(1..9) -> exact 9 values
    let m_trans = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );
    let mut r = [-1.0; 9];
    m_trans.transpose_into_array(&mut r);
    let expected_r = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(r, expected_r);

    // 4. Matrix3 set_identity on a filled matrix
    let mut m_fill = Matrix3::new(
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0,
    );
    m_fill.set_identity();
    assert_eq!(m_fill, Matrix3::identity());
    assert_eq!(m_fill.elements, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}
