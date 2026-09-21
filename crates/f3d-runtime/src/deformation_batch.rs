//! Coarse retained-mesh deformation boundary, feeding the existing mesh renderer.
//!
//! Layout rows are [vertex_count, morph_count, joint_count, flags]. Bit 0 means
//! relative morph targets, bit 1 means skinning. All banks are concatenated in
//! row order; morph positions are target-major within a row. Skinned rows have
//! four indices/weights per vertex, 16 values per palette joint, and 32 bind
//! values (bind followed by inverse bind). Other rows consume no skin banks.
//! Palettes are already evaluated, in skeleton order: this API neither samples
//! animation nor changes hierarchy/bind state. Output is packed xyz f32, ready
//! for the same draw-range, depth, culling, and vertex-color packet paths as rigid
//! geometry. There is no JSON, per-vertex host call, or JavaScript math fallback.
#![forbid(unsafe_code)]

use f3d_math::{Matrix4, deformation::Skinning};
use crate::deformation::{GeometryDeformation, MorphTargets, deform_geometry};

/// Relative morph-target flag in a four-word layout row.
pub const RELATIVE_MORPHS: u32 = 1;
/// Skinning flag in a four-word layout row.
pub const SKINNED: u32 = 2;

/// Parallel flat banks for a complete synchronous geometry snapshot.
#[derive(Clone, Copy, Debug)]
pub struct DeformationBatch<'a> {
    /// Four unsigned words per mesh; see the module-level wire contract.
    pub layout: &'a [u32],
    /// Mesh-major xyz base positions, before morphing and skinning.
    pub positions: &'a [f64],
    /// Mesh-major, target-major xyz morph positions/deltas.
    pub morph_positions: &'a [f64],
    /// Authored, unnormalized weights, one per position target.
    pub morph_weights: &'a [f64],
    /// Four palette indices per skinned vertex.
    pub joint_indices: &'a [u32],
    /// Four authored, unnormalized weights per skinned vertex.
    pub joint_weights: &'a [f64],
    /// Evaluated column-major joint palettes; not joint world matrices.
    pub joint_palettes: &'a [f64],
    /// Bind then inverse-bind matrix for each skinned mesh (32 values).
    pub bind_matrices: &'a [f64],
}

fn add(total: &mut usize, amount: usize) -> Result<(), String> {
    *total = total.checked_add(amount).ok_or("deformation batch size overflow")?;
    Ok(())
}
fn product(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or_else(|| "deformation batch size overflow".to_owned())
}
fn matrix(values: &[f64]) -> Matrix4 {
    let mut elements = [0.0; 16];
    elements.copy_from_slice(values);
    Matrix4::from_elements(elements)
}

/// Evaluate every row using the existing morph-then-skin kernel. Validation and
/// arithmetic failure never expose a partially evaluated batch. Each row starts
/// from immutable base attributes, so repeated calls cannot accumulate a pose.
///
/// # Errors
/// Rejects unknown flags, malformed/trailing banks, overflow, invalid geometry,
/// palettes or active joint indices, non-finite results, and allocation failure.
/// Empty batches and empty individual meshes are valid.
pub fn deform_position_batch(input: &DeformationBatch<'_>) -> Result<Vec<f32>, String> {
    if input.layout.len() % 4 != 0 {
        return Err("deformation layout needs four words per mesh".to_owned());
    }
    // Validate the entire wire shape before taking any sub-slices. Checked
    // products matter on wasm32 even when individual counts fit a u32.
    let (mut positions, mut morphs, mut weights, mut influences, mut palettes, mut binds) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for (mesh, row) in input.layout.chunks_exact(4).enumerate() {
        let vertices = usize::try_from(row[0]).map_err(|_| "vertex count exceeds host size")?;
        let targets = usize::try_from(row[1]).map_err(|_| "morph count exceeds host size")?;
        let joints = usize::try_from(row[2]).map_err(|_| "joint count exceeds host size")?;
        if row[3] & !(RELATIVE_MORPHS | SKINNED) != 0 {
            return Err(format!("mesh {mesh}: unknown deformation flags {}", row[3]));
        }
        if row[3] & SKINNED == 0 && joints != 0 {
            return Err(format!("mesh {mesh}: unskinned row has palette joints"));
        }
        let xyz = product(vertices, 3)?;
        add(&mut positions, xyz)?;
        add(&mut morphs, product(xyz, targets)?)?;
        add(&mut weights, targets)?;
        if row[3] & SKINNED != 0 {
            add(&mut influences, product(vertices, 4)?)?;
            add(&mut palettes, product(joints, 16)?)?;
            add(&mut binds, 32)?;
        }
    }
    for (name, actual, expected) in [
        ("positions", input.positions.len(), positions),
        ("morph positions", input.morph_positions.len(), morphs),
        ("morph weights", input.morph_weights.len(), weights),
        ("joint indices", input.joint_indices.len(), influences),
        ("joint weights", input.joint_weights.len(), influences),
        ("joint palettes", input.joint_palettes.len(), palettes),
        ("bind matrices", input.bind_matrices.len(), binds),
    ] {
        if actual != expected {
            return Err(format!("{name}: expected {expected} elements, got {actual}"));
        }
    }
    let mut output = Vec::new();
    output.try_reserve_exact(positions).map_err(|_| "could not allocate deformed batch")?;
    let (mut p, mut m, mut w, mut j, mut k, mut b) = (0, 0, 0, 0, 0, 0);
    for (mesh, row) in input.layout.chunks_exact(4).enumerate() {
        let xyz = row[0] as usize * 3;
        let target_count = row[1] as usize;
        let mut targets = Vec::new();
        targets.try_reserve_exact(target_count).map_err(|_| "could not allocate morph views")?;
        for (let_target, _) in (0..target_count).enumerate() {
            let start = m + let_target * xyz;
            targets.push(&input.morph_positions[start..start + xyz]);
        }
        let mut palette = Vec::new();
        let bind;
        let inverse;
        let skin = if row[3] & SKINNED != 0 {
            let count = row[2] as usize;
            palette.try_reserve_exact(count).map_err(|_| "could not allocate joint palette")?;
            for values in input.joint_palettes[k..k + count * 16].chunks_exact(16) {
                palette.push(matrix(values));
            }
            bind = matrix(&input.bind_matrices[b..b + 16]);
            inverse = matrix(&input.bind_matrices[b + 16..b + 32]);
            let count = row[0] as usize * 4;
            Some(Skinning {
                joint_indices: &input.joint_indices[j..j + count],
                joint_weights: &input.joint_weights[j..j + count],
                palette: &palette, bind_matrix: &bind, bind_matrix_inverse: &inverse,
            })
        } else { None };
        let geometry = deform_geometry(&GeometryDeformation {
            positions: &input.positions[p..p + xyz], normals: None,
            morph: MorphTargets {
                positions: &targets, normals: &[],
                weights: &input.morph_weights[w..w + target_count],
                relative: row[3] & RELATIVE_MORPHS != 0,
            },
            skin,
        }).map_err(|error| format!("mesh {mesh}: {error}"))?;
        output.extend_from_slice(geometry.positions());
        p += xyz; m += xyz * target_count; w += target_count;
        if row[3] & SKINNED != 0 {
            j += row[0] as usize * 4; k += row[2] as usize * 16; b += 32;
        }
    }
    Ok(output)
}

/// One compiled call per mixed rigid/morphed/skinned mesh batch.
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn f3d_deform_position_batch(
    layout: &[u32], positions: &[f64], morph_positions: &[f64], morph_weights: &[f64],
    joint_indices: &[u32], joint_weights: &[f64], joint_palettes: &[f64], bind_matrices: &[f64],
) -> Result<Vec<f32>, wasm_bindgen::JsValue> {
    deform_position_batch(&DeformationBatch {
        layout, positions, morph_positions, morph_weights,
        joint_indices, joint_weights, joint_palettes, bind_matrices,
    }).map_err(|error| wasm_bindgen::JsValue::from_str(&error))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn empty() -> DeformationBatch<'static> {
        DeformationBatch { layout: &[], positions: &[], morph_positions: &[], morph_weights: &[],
            joint_indices: &[], joint_weights: &[], joint_palettes: &[], bind_matrices: &[] }
    }
    fn translated(x: f64) -> [f64; 16] {
        let mut values = Matrix4::identity().elements; values[12] = x; values
    }
    #[test]
    fn mixed_batch_offsets_morph_order_and_repeated_frames() {
        let binds: Vec<f64> = [translated(0.0), translated(0.0), translated(2.0), translated(-2.0)]
            .into_iter().flatten().collect();
        let palettes: Vec<f64> = [translated(4.0), translated(3.0)].into_iter().flatten().collect();
        let input = DeformationBatch {
            layout: &[1,0,0,0, 1,1,0,1, 1,1,1,2, 1,0,1,2],
            positions: &[1.0,0.0,0.0, 2.0,0.0,0.0, 1.0,0.0,0.0, 1.0,0.0,0.0],
            morph_positions: &[6.0,0.0,0.0, 3.0,0.0,0.0], morph_weights: &[0.5,0.5],
            joint_indices: &[0,u32::MAX,u32::MAX,u32::MAX, 0,0,0,0],
            joint_weights: &[1.0,0.0,0.0,0.0, 2.0,0.0,0.0,0.0],
            joint_palettes: &palettes, bind_matrices: &binds,
        };
        let expected = [1.0,0.0,0.0, 5.0,0.0,0.0, 6.0,0.0,0.0, 8.0,0.0,0.0];
        assert_eq!(deform_position_batch(&input).unwrap(), expected);
        assert_eq!(deform_position_batch(&input).unwrap(), expected);
    }
    #[test]
    fn empty_rows_and_zero_vertex_targets_consume_their_weights() {
        assert!(deform_position_batch(&empty()).unwrap().is_empty());
        let input = DeformationBatch { layout: &[0,1,0,1, 1,1,0,1], positions: &[2.0,0.0,0.0],
            morph_positions: &[4.0,0.0,0.0], morph_weights: &[9.0,0.5], ..empty() };
        assert_eq!(deform_position_batch(&input).unwrap(), [4.0,0.0,0.0]);
    }
    #[test]
    fn malformed_layout_flags_and_trailing_banks_reject() {
        for layout in [&[1,0,0][..], &[0,0,0,4], &[0,0,1,0], &[u32::MAX,u32::MAX,0,1]] {
            assert!(deform_position_batch(&DeformationBatch { layout, ..empty() }).is_err());
        }
        assert!(deform_position_batch(&DeformationBatch { positions: &[1.0], ..empty() }).is_err());
        assert!(deform_position_batch(&DeformationBatch { joint_weights: &[1.0], ..empty() }).is_err());
        assert!(deform_position_batch(&DeformationBatch { bind_matrices: &[1.0], ..empty() }).is_err());
    }
    #[test]
    fn invalid_later_mesh_rejects_the_whole_result() {
        let input = DeformationBatch { layout: &[1,0,0,0, 1,0,0,0],
            positions: &[1.0,0.0,0.0, f64::NAN,0.0,0.0], ..empty() };
        assert!(deform_position_batch(&input).unwrap_err().starts_with("mesh 1:"));
    }
    #[test]
    fn upload_overflow_and_active_bad_joints_reject() {
        assert!(deform_position_batch(&DeformationBatch { layout: &[1,0,0,0],
            positions: &[1e100,0.0,0.0], ..empty() }).is_err());
        let binds: Vec<f64> = [translated(0.0); 2].into_iter().flatten().collect();
        let input = DeformationBatch { layout: &[1,0,0,2], positions: &[1.0,0.0,0.0],
            joint_indices: &[0; 4], joint_weights: &[1.0,0.0,0.0,0.0], bind_matrices: &binds, ..empty() };
        assert!(deform_position_batch(&input).is_err());
    }
}
