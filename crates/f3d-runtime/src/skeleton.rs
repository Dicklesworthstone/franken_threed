//! Skeleton palettes connecting solved hierarchy state to the existing deformer.
//!
//! Joint order is independent of scene order. A -1 joint slot uses the identity
//! world transform, as a missing bone does in pinned Three.js Skeleton.update.
//! Mesh bind matrices are separate from the skeleton's inverse-bind matrices.
//! Nothing here advances an animation clock or updates the source hierarchy.
#![forbid(unsafe_code)]

use core::fmt;
use f3d_math::{Matrix4, deformation::{Skinning, build_joint_palette_into}};
use crate::{
    deformation::{DeformedGeometry, DeformedMeshError, GeometryDeformation, MorphTargets, deform_geometry},
    gpu_host::GpuSubmissionPacket,
    hierarchy::{HierarchyError, TransformHierarchy},
    mesh::{DynamicMeshInput, build_mesh_submission},
};

/// A palette, hierarchy receipt, or downstream deformation failed validation.
#[derive(Debug)]
pub enum SkeletonError {
    /// Flat matrix arrays must contain complete matrices and matching counts.
    Shape { field: &'static str, expected: usize, actual: usize },
    /// A joint is neither a missing-bone sentinel nor a valid hierarchy node.
    JointNode { joint: usize, node: i32, node_count: usize },
    /// A hierarchy receipt was stale or had not been solved.
    Hierarchy(HierarchyError),
    /// Existing morph/skin or render-packet validation failed.
    Deformation(DeformedMeshError),
    /// An owned transport or scratch allocation could not be satisfied.
    Allocation,
}

impl fmt::Display for SkeletonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { field, expected, actual } => write!(f, "{field}: expected {expected} elements, got {actual}"),
            Self::JointNode { joint, node, node_count } => write!(f, "joint {joint} refers to node {node}, outside hierarchy of {node_count} nodes"),
            Self::Hierarchy(error) => fmt::Display::fmt(error, f),
            Self::Deformation(error) => fmt::Display::fmt(error, f),
            Self::Allocation => write!(f, "could not allocate skeleton palette"),
        }
    }
}
impl std::error::Error for SkeletonError {}
impl From<HierarchyError> for SkeletonError {
    fn from(error: HierarchyError) -> Self { Self::Hierarchy(error) }
}
impl From<DeformedMeshError> for SkeletonError {
    fn from(error: DeformedMeshError) -> Self { Self::Deformation(error) }
}

fn length(field: &'static str, expected: usize, actual: usize) -> Result<(), SkeletonError> {
    if expected != actual { return Err(SkeletonError::Shape { field, expected, actual }); }
    Ok(())
}
fn matrix(values: &[f64]) -> Matrix4 {
    let mut elements = [0.0; 16];
    elements.copy_from_slice(values);
    Matrix4::from_elements(elements)
}

/// Produce the public Float32 bone-matrix upload bank from packed f64 matrices.
///
/// Each output is f32(world * inverse_bind), NOT f32(world) * f32(inverse_bind).
/// This reuses the scalar Matrix4 kernel, including full projective matrices and
/// IEEE exceptional values. It follows Skeleton.update's numeric boundary, not
/// the finite-only admission rules of the downstream CPU geometry deformer.
/// The returned allocation is owned; it never borrows Wasm linear memory.
///
/// # Errors
/// Rejects malformed/mismatched flat banks and allocation failure before output
/// is exposed. Empty banks are valid. Inputs are never changed.
pub fn batch_skeleton_palette_f32(worlds: &[f64], inverses: &[f64]) -> Result<Vec<f32>, SkeletonError> {
    length("complete world matrices", worlds.len() / 16 * 16, worlds.len())?;
    length("inverse-bind matrices", worlds.len(), inverses.len())?;
    let mut result = Vec::new();
    result.try_reserve_exact(worlds.len()).map_err(|_| SkeletonError::Allocation)?;
    for (world, inverse) in worlds.chunks_exact(16).zip(inverses.chunks_exact(16)) {
        let mut product = Matrix4::zero();
        product.multiply_matrices(&matrix(world), &matrix(inverse));
        result.extend(product.elements.map(|value| value as f32));
    }
    Ok(result)
}

/// Gather joints from a current solved hierarchy and construct a finite f64
/// palette using the existing deformation kernel. Reordered, repeated, missing,
/// and manually owned world-transform slots are supported. No inverse-bind
/// calculation, pose reset, normalization, or hierarchy solve is implicit.
///
/// This returns an owned snapshot. Use the immediate deformation/submission
/// functions below when the receipt must be checked at the point of use.
///
/// # Errors
/// Rejects stale/unsolved revisions, invalid mappings, mismatched inverse binds,
/// non-finite matrices/results, and allocation failures.
pub fn skeleton_palette_from_hierarchy(
    hierarchy: &TransformHierarchy, revision: u64,
    joint_nodes: &[i32], inverse_bind: &[Matrix4],
) -> Result<Vec<Matrix4>, SkeletonError> {
    let bank = hierarchy.world_matrices(revision)?;
    length("inverse-bind count", joint_nodes.len(), inverse_bind.len())?;
    for (joint, &node) in joint_nodes.iter().enumerate() {
        if node < -1 || node >= 0 && node as usize >= bank.len() {
            return Err(SkeletonError::JointNode { joint, node, node_count: bank.len() });
        }
    }
    let mut worlds = Vec::new();
    let mut palette = Vec::new();
    worlds.try_reserve_exact(joint_nodes.len()).map_err(|_| SkeletonError::Allocation)?;
    palette.try_reserve_exact(joint_nodes.len()).map_err(|_| SkeletonError::Allocation)?;
    for &node in joint_nodes {
        worlds.push(if node == -1 { Matrix4::identity() } else { bank[node as usize] });
    }
    palette.resize(joint_nodes.len(), Matrix4::zero());
    build_joint_palette_into(&worlds, inverse_bind, &mut palette)
        .map_err(DeformedMeshError::from)?;
    Ok(palette)
}

/// Joint mapping plus geometry binding; inverse binds remain in joint order.
#[derive(Clone, Copy, Debug)]
pub struct HierarchySkinning<'a> {
    /// Stable hierarchy node per joint; -1 denotes a missing bone.
    pub joint_nodes: &'a [i32],
    /// Bind-time inverse world matrices, one per joint.
    pub inverse_bind: &'a [Matrix4],
    /// Four palette indices per geometry vertex, not hierarchy node indices.
    pub joint_indices: &'a [u32],
    /// Four authored, unnormalized weights per vertex.
    pub joint_weights: &'a [f64],
    /// Mesh-local to bind-space matrix.
    pub bind_matrix: &'a Matrix4,
    /// Current bind-space to mesh-local matrix (caller owns attached/detached mode).
    pub bind_matrix_inverse: &'a Matrix4,
}

/// Immutable base geometry, current morph influences, and hierarchy skin binding.
#[derive(Clone, Copy, Debug)]
pub struct HierarchyGeometry<'a> {
    /// Flat xyz base positions; repeated evaluations never accumulate deformation.
    pub positions: &'a [f64],
    /// Optional base xyz normals.
    pub normals: Option<&'a [f64]>,
    /// Morph targets evaluated before skinning.
    pub morph: MorphTargets<'a>,
    /// Current mesh binding and joint mapping.
    pub skin: HierarchySkinning<'a>,
}

/// Check the solved receipt, gather the palette, morph, skin, and narrow at the
/// existing upload boundary in one synchronous call. The hierarchy is borrowed
/// throughout, so it cannot mutate between receipt validation and consumption.
pub fn deform_hierarchy_geometry(
    hierarchy: &TransformHierarchy, revision: u64, input: &HierarchyGeometry<'_>,
) -> Result<DeformedGeometry, SkeletonError> {
    let binding = &input.skin;
    let palette = skeleton_palette_from_hierarchy(hierarchy, revision, binding.joint_nodes, binding.inverse_bind)?;
    Ok(deform_geometry(&GeometryDeformation {
        positions: input.positions, normals: input.normals, morph: input.morph,
        skin: Some(Skinning {
            joint_indices: binding.joint_indices, joint_weights: binding.joint_weights,
            palette: &palette, bind_matrix: binding.bind_matrix,
            bind_matrix_inverse: binding.bind_matrix_inverse,
        }),
    })?)
}

/// Emit a real existing mesh-render packet from hierarchy-owned joint transforms.
/// Preserves the template's indices, depth/cull/color-write state and transforms.
/// This basic-material route uses deformed positions; normal-aware callers use
/// `deform_hierarchy_geometry` and its normal output instead.
pub fn build_hierarchy_mesh_submission(
    template: &DynamicMeshInput<'_>, hierarchy: &TransformHierarchy,
    revision: u64, input: &HierarchyGeometry<'_>,
) -> Result<GpuSubmissionPacket, SkeletonError> {
    let geometry = deform_hierarchy_geometry(hierarchy, revision, input)?;
    Ok(build_mesh_submission(&geometry.mesh_input(template)?)
        .map_err(DeformedMeshError::from)?)
}

/// Coarse Wasm boundary for retained Three.js skeletons, with no JS math copy.
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn f3d_batch_skeleton_palette(worlds: &[f64], inverses: &[f64]) -> Result<Vec<f32>, wasm_bindgen::JsValue> {
    batch_skeleton_palette_f32(worlds, inverses)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn translation(x: f64) -> Matrix4 {
        let mut matrix = Matrix4::identity(); matrix.elements[12] = x; matrix
    }
    fn forest() -> TransformHierarchy {
        let locals: Vec<f64> = [translation(2.0), translation(3.0), translation(7.0)]
            .into_iter().flat_map(|m| m.elements).collect();
        TransformHierarchy::new(&[-1, 0, -1], &locals, None, None).unwrap()
    }
    #[test]
    fn upload_multiplies_before_narrowing() {
        let world = translation(16_777_217.0);
        let inverse = translation(-16_777_216.0);
        assert_eq!(batch_skeleton_palette_f32(&world.elements, &inverse.elements).unwrap()[12], 1.0);
    }
    #[test]
    fn upload_validates_both_shapes() {
        assert!(batch_skeleton_palette_f32(&[0.0; 15], &[0.0; 15]).is_err());
        assert!(batch_skeleton_palette_f32(&[0.0; 16], &[0.0; 32]).is_err());
        assert!(batch_skeleton_palette_f32(&[], &[]).unwrap().is_empty());
    }
    #[test]
    fn upload_keeps_projective_matrix_math() {
        let world = Matrix4::from_elements(core::array::from_fn(|i| i as f64 - 7.0));
        let inverse = Matrix4::from_elements(core::array::from_fn(|i| (i * i) as f64 / 3.0));
        let mut expected = Matrix4::zero(); expected.multiply_matrices(&world, &inverse);
        assert_eq!(batch_skeleton_palette_f32(&world.elements, &inverse.elements).unwrap(),
            expected.elements.map(|v| v as f32));
    }
    #[test]
    fn upload_preserves_exceptional_values_instead_of_repairing_them() {
        let mut world = Matrix4::identity(); world.elements[0] = f64::NAN;
        let actual = batch_skeleton_palette_f32(&world.elements, &Matrix4::identity().elements).unwrap();
        assert!(actual[0].is_nan());
    }
    #[test]
    fn joint_order_missing_and_repeated_bones() {
        let mut hierarchy = forest(); let receipt = hierarchy.solve().revision;
        let inverse = [translation(-1.0); 4];
        let palette = skeleton_palette_from_hierarchy(&hierarchy, receipt, &[2, 1, -1, 1], &inverse).unwrap();
        assert_eq!(palette.iter().map(|m| m.elements[12]).collect::<Vec<_>>(), [6.0, 4.0, -1.0, 4.0]);
    }
    #[test]
    fn unsolved_and_obsolete_receipts_reject() {
        let mut hierarchy = forest();
        assert!(matches!(skeleton_palette_from_hierarchy(&hierarchy, hierarchy.revision(), &[], &[]),
            Err(SkeletonError::Hierarchy(HierarchyError::Unsolved { .. }))));
        let receipt = hierarchy.solve().revision;
        hierarchy.set_local_matrices(&[0], &translation(4.0).elements).unwrap();
        assert!(matches!(skeleton_palette_from_hierarchy(&hierarchy, receipt, &[], &[]),
            Err(SkeletonError::Hierarchy(HierarchyError::Stale { .. }))));
    }
    #[test]
    fn invalid_joint_map_and_inverse_count_reject() {
        let mut hierarchy = forest(); let receipt = hierarchy.solve().revision;
        for node in [-2, 3, i32::MAX] {
            assert!(matches!(skeleton_palette_from_hierarchy(&hierarchy, receipt, &[node], &[Matrix4::identity()]),
                Err(SkeletonError::JointNode { .. })));
        }
        assert!(skeleton_palette_from_hierarchy(&hierarchy, receipt, &[0], &[]).is_err());
    }
    #[test]
    fn manual_world_boundary_drives_descendant_joints() {
        let locals: Vec<f64> = [translation(2.0), translation(3.0)].into_iter().flat_map(|m| m.elements).collect();
        let worlds: Vec<f64> = [translation(40.0), Matrix4::identity()].into_iter().flat_map(|m| m.elements).collect();
        let mut hierarchy = TransformHierarchy::new(&[-1, 0], &locals, Some(&worlds), Some(&[0, 1])).unwrap();
        let receipt = hierarchy.solve().revision;
        let palette = skeleton_palette_from_hierarchy(&hierarchy, receipt, &[1], &[translation(-3.0)]).unwrap();
        assert_eq!(palette[0].elements[12], 40.0);
    }
    #[test]
    fn finite_geometry_path_rejects_nonfinite_palette() {
        let mut hierarchy = forest(); let receipt = hierarchy.solve().revision;
        let inverse = translation(f64::INFINITY);
        assert!(skeleton_palette_from_hierarchy(&hierarchy, receipt, &[0], &[inverse]).is_err());
    }
    #[test]
    fn hierarchy_morph_skin_normals_and_packet_are_connected() {
        let mut hierarchy = forest(); let receipt = hierarchy.solve().revision;
        let identity = Matrix4::identity();
        let inverse_bind = [translation(-3.0)];
        let positions = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0];
        let normals: [f64; 9] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let delta = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let targets: [&[f64]; 1] = [&delta];
        let input = HierarchyGeometry {
            positions: &positions, normals: Some(&normals),
            morph: MorphTargets { positions: &targets, normals: &[], weights: &[0.5], relative: true },
            skin: HierarchySkinning { joint_nodes: &[1], inverse_bind: &inverse_bind,
                joint_indices: &[0; 12], joint_weights: &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                bind_matrix: &identity, bind_matrix_inverse: &identity },
        };
        let geometry = deform_hierarchy_geometry(&hierarchy, receipt, &input).unwrap();
        assert_eq!(geometry.positions()[0], 2.5);
        assert_eq!(geometry.normals().unwrap(), normals.map(|v| v as f32));
        assert_eq!(geometry.bounds().unwrap().0, [2.5, 0.0, 0.0]);
        let base: Vec<f32> = positions.map(|v| v as f32).to_vec();
        let template = DynamicMeshInput::try_from_raw(&base, &[], &identity.elements,
            &identity.elements, &[1.0, 0.0, 0.0, 1.0], 32, 32, false).unwrap();
        let packet = build_hierarchy_mesh_submission(&template, &hierarchy, receipt, &input).unwrap().encode().unwrap();
        assert_eq!(&packet[..4], b"F3DP");
        assert_eq!(deform_hierarchy_geometry(&hierarchy, receipt, &input).unwrap().positions(), geometry.positions());
    }
}
