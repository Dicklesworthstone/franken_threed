use f3d_math::Matrix4;
use f3d_math::deformation::{
    build_joint_palette_into, morph_attribute_into, skin_attribute_into,
    DeformationError, SkinAttribute, Skinning,
};

fn translated(x: f64) -> Matrix4 {
    let mut matrix = Matrix4::identity();
    matrix.elements[12] = x;
    matrix
}

#[test]
fn absolute_and_relative_targets_preserve_authored_weights() {
    let base = [2.0, 4.0, 6.0];
    let a = [4.0, 8.0, 12.0];
    let b = [8.0, 4.0, 0.0];
    let mut out = [0.0; 3];
    morph_attribute_into(&base, &[&a, &b], &[0.5, -0.25], false, &mut out).unwrap();
    assert_eq!(out, [1.5, 6.0, 10.5]);
    morph_attribute_into(&base, &[&a, &b], &[0.5, -0.25], true, &mut out).unwrap();
    assert_eq!(out, [2.0, 7.0, 12.0]);
    morph_attribute_into(&base, &[&a], &[2.0], false, &mut out).unwrap();
    assert_eq!(out, [6.0, 12.0, 18.0]);
}

#[test]
fn no_targets_is_identity_and_bad_input_does_not_write() {
    let mut out = [9.0; 3];
    morph_attribute_into(&[1.0, 2.0, 3.0], &[], &[], false, &mut out).unwrap();
    assert_eq!(out, [1.0, 2.0, 3.0]);
    let before = out;
    assert!(morph_attribute_into(&[1.0; 3], &[&[1.0; 2]], &[1.0], false, &mut out).is_err());
    assert_eq!(out, before);
    assert!(morph_attribute_into(&[1.0; 3], &[&[1.0; 3]], &[], false, &mut out).is_err());
    assert!(morph_attribute_into(&[f64::NAN; 3], &[], &[], false, &mut out).is_err());
    assert_eq!(out, before);
}

#[test]
fn bind_space_skinning_preserves_blended_homogeneous_coordinate() {
    let mut bind = translated(10.0);
    bind.elements[0] = 2.0;
    let mut inverse = translated(-5.0);
    inverse.elements[0] = 0.5;
    let palette = [translated(3.0)];
    let skin = Skinning {
        joint_indices: &[0, u32::MAX, u32::MAX, u32::MAX],
        joint_weights: &[2.0, 0.0, 0.0, 0.0],
        palette: &palette, bind_matrix: &bind, bind_matrix_inverse: &inverse,
    };
    let mut out = [0.0; 3];
    skin_attribute_into(&[1.0, 0.0, 0.0], &skin, SkinAttribute::Position, &mut out).unwrap();
    assert_eq!(out, [5.0, 0.0, 0.0]);
    skin_attribute_into(&[1.0, 0.0, 0.0], &skin, SkinAttribute::Direction, &mut out).unwrap();
    assert_eq!(out, [2.0, 0.0, 0.0]);
}

#[test]
fn four_joints_blend_and_zero_weights_do_not_become_identity() {
    let identity = Matrix4::identity();
    let palette = [translated(1.0), translated(2.0), translated(3.0), translated(4.0)];
    let mut skin = Skinning {
        joint_indices: &[0, 1, 2, 3], joint_weights: &[0.25; 4], palette: &palette,
        bind_matrix: &identity, bind_matrix_inverse: &identity,
    };
    let mut out = [0.0; 3];
    skin_attribute_into(&[1.0, 2.0, 3.0], &skin, SkinAttribute::Position, &mut out).unwrap();
    assert_eq!(out, [3.5, 2.0, 3.0]);
    skin.joint_weights = &[0.0; 4];
    skin_attribute_into(&[1.0, 2.0, 3.0], &skin, SkinAttribute::Position, &mut out).unwrap();
    assert_eq!(out, [0.0; 3]);
}

#[test]
fn active_invalid_joint_and_nonfinite_palette_are_rejected_before_writes() {
    let identity = Matrix4::identity();
    let mut skin = Skinning {
        joint_indices: &[0, 0, 0, 0], joint_weights: &[1.0, 0.0, 0.0, 0.0], palette: &[],
        bind_matrix: &identity, bind_matrix_inverse: &identity,
    };
    let mut out = [9.0; 3];
    assert!(matches!(skin_attribute_into(&[1.0; 3], &skin, SkinAttribute::Position, &mut out),
        Err(DeformationError::JointOutOfBounds { .. })));
    assert_eq!(out, [9.0; 3]);
    let palette = [translated(f64::INFINITY)];
    skin.palette = &palette;
    assert!(skin_attribute_into(&[1.0; 3], &skin, SkinAttribute::Position, &mut out).is_err());
    assert_eq!(out, [9.0; 3]);
}

#[test]
fn morphing_precedes_skinning() {
    let mut morphed = [0.0; 3];
    morph_attribute_into(&[1.0, 0.0, 0.0], &[&[3.0, 0.0, 0.0]], &[0.5], false, &mut morphed).unwrap();
    let identity = Matrix4::identity();
    let palette = [translated(3.0)];
    let skin = Skinning {
        joint_indices: &[0; 4], joint_weights: &[1.0, 0.0, 0.0, 0.0], palette: &palette,
        bind_matrix: &identity, bind_matrix_inverse: &identity,
    };
    let mut out = [0.0; 3];
    skin_attribute_into(&morphed, &skin, SkinAttribute::Position, &mut out).unwrap();
    assert_eq!(out, [5.0, 0.0, 0.0]);
}

#[test]
fn palette_uses_world_times_inverse_bind_in_joint_order() {
    let mut scaled = translated(10.0);
    scaled.elements[0] = 2.0;
    let mut out = [Matrix4::zero(); 2];
    build_joint_palette_into(&[scaled, translated(4.0)], &[translated(-3.0), translated(-1.0)], &mut out).unwrap();
    assert_eq!(out[0].elements[12], 4.0);
    assert_eq!(out[0].elements[0], 2.0);
    assert_eq!(out[1].elements[12], 3.0);
    let before = out;
    assert!(build_joint_palette_into(&[scaled], &[], &mut out).is_err());
    assert_eq!(out, before);
}

#[test]
fn finite_inputs_cannot_silently_produce_infinite_outputs() {
    let mut out = [0.0; 3];
    assert!(morph_attribute_into(&[f64::MAX; 3], &[&[f64::MAX; 3]], &[2.0], true, &mut out).is_err());
    let identity = Matrix4::identity();
    let palette = [translated(f64::MAX)];
    let skin = Skinning {
        joint_indices: &[0; 4], joint_weights: &[1.0, 0.0, 0.0, 0.0], palette: &palette,
        bind_matrix: &identity, bind_matrix_inverse: &identity,
    };
    assert!(skin_attribute_into(&[f64::MAX, 0.0, 0.0], &skin, SkinAttribute::Position, &mut out).is_err());
}
