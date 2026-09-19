use f3d_math::{Matrix4, deformation::Skinning};
use f3d_runtime::deformation::{
    DeformedMeshError, GeometryDeformation, MorphTargets, build_deformed_mesh_packet,
    build_deformed_mesh_submission, deform_geometry,
};
use f3d_runtime::mesh::{DynamicMeshInput, MeshDepthOptions, build_mesh_submission};

const POSITIONS: [f64; 9] = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0];
const BASE: [f32; 9] = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0];
const NORMALS: [f64; 9] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn input() -> GeometryDeformation<'static> {
    GeometryDeformation {
        positions: &POSITIONS, normals: None, morph: MorphTargets::default(), skin: None,
    }
}

fn mesh(positions: &[f32]) -> DynamicMeshInput<'_> {
    DynamicMeshInput::try_from_raw(
        positions, &[2, 0, 1], &IDENTITY, &IDENTITY, &[0.2, 0.4, 0.6, 1.0], 8, 8, true,
    ).unwrap()
}

#[test]
fn position_morphs_do_not_attenuate_unmorphed_normals_and_bounds_follow_pose() {
    let mut source = input();
    source.normals = Some(&NORMALS);
    let delta = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let targets: [&[f64]; 1] = [&delta];
    source.morph = MorphTargets { positions: &targets, normals: &[], weights: &[0.5], relative: true };
    let geometry = deform_geometry(&source).unwrap();
    assert_eq!(geometry.positions(), &[0.0, -0.5, 0.0, 1.0, -0.5, 0.0, 0.5, 0.5, 0.0]);
    assert_eq!(geometry.normals().unwrap(), &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert_eq!(geometry.bounds(), Some(([0.0, -0.5, 0.0], [1.0, 0.5, 0.0])));
    assert_eq!(source.positions, &POSITIONS);
}

#[test]
fn morph_and_skin_normals_reach_upload_attributes() {
    let mut source = input();
    source.normals = Some(&NORMALS);
    let targets: [&[f64]; 1] = [&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
    source.morph = MorphTargets { positions: &[], normals: &targets, weights: &[1.0], relative: false };
    let identity = Matrix4::identity();
    let mut joint = identity;
    joint.elements[12] = 10.0;
    joint.elements[0] = 2.0;
    let palette = [joint];
    source.skin = Some(Skinning {
        joint_indices: &[0; 12], joint_weights: &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        palette: &palette, bind_matrix: &identity, bind_matrix_inverse: &identity,
    });
    let geometry = deform_geometry(&source).unwrap();
    assert_eq!(geometry.positions()[0], 9.0);
    assert_eq!(geometry.normals().unwrap(), &[2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
}

#[test]
fn mesh_adapter_preserves_all_existing_render_state() {
    let colors = [0.25f32; 12];
    let depth = MeshDepthOptions::new(true, false, 3).unwrap();
    let template = mesh(&BASE).with_cull(2, 1).unwrap().with_depth(depth).unwrap()
        .with_color_write(false).with_vertex_colors(&colors).unwrap();
    let geometry = deform_geometry(&input()).unwrap();
    let result = geometry.mesh_input(&template).unwrap();
    assert_eq!(result, template);
    assert_eq!(result.indices(), &[2, 0, 1]);
}

#[test]
fn deformed_draw_packet_equals_explicit_final_geometry_packet() {
    let delta = [0.25; 9];
    let targets: [&[f64]; 1] = [&delta];
    let mut source = input();
    source.morph = MorphTargets { positions: &targets, normals: &[], weights: &[1.0], relative: true };
    let template = mesh(&BASE).with_cull(2, 0).unwrap().with_depth_options(true, true, 2).unwrap();
    let actual = build_deformed_mesh_submission(&template, &source).unwrap().encode().unwrap();
    let final_positions = [-0.25, -0.25, 0.25, 0.75, -0.25, 0.25, 0.25, 0.75, 0.25];
    let expected_input = mesh(&final_positions).with_cull(2, 0).unwrap().with_depth_options(true, true, 2).unwrap();
    let expected = build_mesh_submission(&expected_input).unwrap().encode().unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn flat_host_entrypoint_skins_morphed_positions_into_real_packet() {
    let mut joint = IDENTITY;
    joint[12] = 0.25;
    let weights = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let actual = build_deformed_mesh_packet(
        &POSITIONS, &[0.25; 9], &[1.0], true, &[0; 12], &weights, &joint,
        &IDENTITY, &IDENTITY, &[2, 0, 1], &IDENTITY, &IDENTITY, &[0.2, 0.4, 0.6, 1.0], 8, 8, true,
    ).unwrap();
    let final_positions = [0.0, -0.25, 0.25, 1.0, -0.25, 0.25, 0.5, 0.75, 0.25];
    let expected = build_mesh_submission(&mesh(&final_positions)).unwrap().encode().unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn final_narrowing_rejects_overflow_but_does_not_round_before_morphing() {
    let huge = [f64::from(f32::MAX) * 2.0; 3];
    let mut source = GeometryDeformation { positions: &huge, ..input() };
    assert!(matches!(deform_geometry(&source), Err(DeformedMeshError::UploadRange { .. })));
    let targets: [&[f64]; 1] = [&[0.0; 3]];
    source.morph = MorphTargets { positions: &targets, normals: &[], weights: &[0.5], relative: false };
    assert_eq!(deform_geometry(&source).unwrap().positions(), &[f32::MAX; 3]);
}

#[test]
fn frames_restart_from_base_and_empty_geometry_has_no_bounds() {
    let targets: [&[f64]; 1] = [&[1.0; 9]];
    let mut source = input();
    source.morph = MorphTargets { positions: &targets, normals: &[], weights: &[0.5], relative: true };
    let a = deform_geometry(&source).unwrap();
    let b = deform_geometry(&source).unwrap();
    assert_eq!(a, b);
    assert_eq!(source.positions, &POSITIONS);
    let empty = deform_geometry(&GeometryDeformation { positions: &[], ..input() }).unwrap();
    assert!(empty.positions().is_empty());
    assert_eq!(empty.bounds(), None);
}

#[test]
fn malformed_parallel_attributes_and_skin_bindings_fail_cleanly() {
    let mut source = input();
    source.normals = Some(&[1.0; 3]);
    assert!(deform_geometry(&source).is_err());
    source = input();
    source.morph.weights = &[f64::NAN];
    assert!(deform_geometry(&source).is_err());
    source = input();
    let targets: [&[f64]; 1] = [&[1.0; 3]];
    source.morph = MorphTargets { positions: &targets, normals: &[], weights: &[1.0], relative: true };
    assert!(deform_geometry(&source).is_err());
    assert!(build_deformed_mesh_packet(
        &POSITIONS, &[], &[], true, &[0; 12], &[0.25; 12], &[1.0; 15],
        &IDENTITY, &IDENTITY, &[], &IDENTITY, &IDENTITY, &[1.0; 4], 8, 8, true,
    ).is_err());
    assert!(build_deformed_mesh_packet(
        &POSITIONS, &[0.0; 8], &[1.0], true, &[], &[], &[], &[], &[],
        &[], &IDENTITY, &IDENTITY, &[1.0; 4], 8, 8, true,
    ).is_err());
}
