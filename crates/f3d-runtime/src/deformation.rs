//! Morph/skin evaluation connected to the existing dynamic mesh renderer.
//!
//! This CPU/Wasm route produces real vertex uploads through `FrameSession` and
//! the existing GPU packet encoder. It does not replace animation sampling,
//! loader ownership, GPU skinning, or the host's scene traversal. Callers supply
//! the current pose and immutable base attributes on every evaluation.

use core::fmt;

use f3d_math::Matrix4;
use f3d_math::deformation::{
    DeformationError, SkinAttribute, Skinning, morph_attribute_into, skin_attribute_into,
};

use crate::gpu_host::GpuSubmissionPacket;
use crate::mesh::{DynamicMeshInput, MeshPacketError, build_mesh_submission};

/// A failed deformation or render-packet build. No partially built mesh escapes.
#[derive(Debug)]
pub enum DeformedMeshError {
    /// Invalid deformation attributes or arithmetic.
    Deformation(DeformationError),
    /// Existing mesh validation or packet encoding failure.
    Mesh(MeshPacketError),
    /// Scratch/output allocation could not be satisfied.
    Allocation,
    /// A final upload component cannot be represented as finite f32.
    UploadRange { attribute: &'static str, index: usize },
}

impl fmt::Display for DeformedMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deformation(e) => fmt::Display::fmt(e, f),
            Self::Mesh(e) => fmt::Display::fmt(e, f),
            Self::Allocation => write!(f, "could not allocate deformation buffers"),
            Self::UploadRange { attribute, index } =>
                write!(f, "{attribute} component {index} is outside finite f32 upload range"),
        }
    }
}

impl core::error::Error for DeformedMeshError {}

impl From<DeformationError> for DeformedMeshError {
    fn from(value: DeformationError) -> Self { Self::Deformation(value) }
}

impl From<MeshPacketError> for DeformedMeshError {
    fn from(value: MeshPacketError) -> Self { Self::Mesh(value) }
}

/// Parallel morph attributes. A missing target family leaves that base attribute
/// unchanged; every present family must have one target per shared influence.
#[derive(Clone, Copy, Debug, Default)]
pub struct MorphTargets<'a> {
    /// Absolute position targets or relative position deltas.
    pub positions: &'a [&'a [f64]],
    /// Absolute normal targets or relative normal deltas.
    pub normals: &'a [&'a [f64]],
    /// Shared, unnormalized authored influences.
    pub weights: &'a [f64],
    /// Whether target values are deltas rather than absolute attributes.
    pub relative: bool,
}

/// Current pose plus immutable base geometry. All attributes are flat xyz arrays.
#[derive(Clone, Copy, Debug)]
pub struct GeometryDeformation<'a> {
    /// Base positions, before any morphing or skinning.
    pub positions: &'a [f64],
    /// Optional base normals, before any morphing or skinning.
    pub normals: Option<&'a [f64]>,
    /// Current morph targets and influences.
    pub morph: MorphTargets<'a>,
    /// Current skin palette/binding, or no skeletal deformation.
    pub skin: Option<Skinning<'a>>,
}

/// Fully evaluated, upload-ready attributes and local-space bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformedGeometry {
    positions: Vec<f32>,
    normals: Option<Vec<f32>>,
    bounds: Option<([f32; 3], [f32; 3])>,
}

fn checked_length(field: &'static str, actual: usize, expected: usize) -> Result<(), DeformedMeshError> {
    if actual != expected {
        return Err(DeformationError::Length { field, expected, actual }.into());
    }
    Ok(())
}

fn scratch(len: usize) -> Result<Vec<f64>, DeformedMeshError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| DeformedMeshError::Allocation)?;
    values.resize(len, 0.0);
    Ok(values)
}

fn evaluate_attribute(
    base: &[f64], targets: &[&[f64]], morph: &MorphTargets<'_>,
    skin: Option<&Skinning<'_>>, kind: SkinAttribute, name: &'static str,
) -> Result<Vec<f32>, DeformedMeshError> {
    let mut values = scratch(base.len())?;
    let weights = if targets.is_empty() { &[][..] } else { morph.weights };
    morph_attribute_into(base, targets, weights, morph.relative, &mut values)?;
    if let Some(skin) = skin {
        let mut skinned = scratch(base.len())?;
        skin_attribute_into(&values, skin, kind, &mut skinned)?;
        values = skinned;
    }
    let mut upload = Vec::new();
    upload.try_reserve_exact(values.len()).map_err(|_| DeformedMeshError::Allocation)?;
    for (index, value) in values.into_iter().enumerate() {
        let narrowed = value as f32;
        if !narrowed.is_finite() {
            return Err(DeformedMeshError::UploadRange { attribute: name, index });
        }
        upload.push(narrowed);
    }
    Ok(upload)
}

/// Evaluate morphs, then skinning, then narrow once at the GPU upload boundary.
///
/// Bounds are recomputed from the final upload positions, not the undeformed
/// bind pose. Source arrays remain unchanged, so repeated frames do not
/// accumulate deformation. Normal magnitudes are preserved for the material's
/// later normal transformation/normalization.
///
/// # Errors
/// Rejects malformed/non-finite attributes, bad palettes, allocation failure,
/// and non-finite final f32 values. Does not expose partially evaluated output.
pub fn deform_geometry(input: &GeometryDeformation<'_>) -> Result<DeformedGeometry, DeformedMeshError> {
    if input.positions.len() % 3 != 0 {
        return Err(DeformationError::IncompleteVertex { components: input.positions.len() }.into());
    }
    if let Some(normals) = input.normals {
        checked_length("base normals", normals.len(), input.positions.len())?;
    } else if !input.morph.normals.is_empty() {
        return Err(DeformationError::Length {
            field: "base normals", expected: input.positions.len(), actual: 0,
        }.into());
    }
    if let Some(index) = input.morph.weights.iter().position(|x| !x.is_finite()) {
        return Err(DeformationError::NonFinite { field: "morph weights", index }.into());
    }
    if input.morph.positions.is_empty() && input.morph.normals.is_empty() {
        checked_length("unused morph weights", input.morph.weights.len(), 0)?;
    }
    for (field, targets) in [("position morphs", input.morph.positions), ("normal morphs", input.morph.normals)] {
        if !targets.is_empty() {
            checked_length(field, targets.len(), input.morph.weights.len())?;
            for target in targets {
                checked_length(field, target.len(), input.positions.len())?;
            }
        }
    }
    let positions = evaluate_attribute(
        input.positions, input.morph.positions, &input.morph, input.skin.as_ref(),
        SkinAttribute::Position, "position",
    )?;
    let normals = input.normals.map(|base| evaluate_attribute(
        base, input.morph.normals, &input.morph, input.skin.as_ref(),
        SkinAttribute::Direction, "normal",
    )).transpose()?;
    let mut bounds: Option<([f32; 3], [f32; 3])> = None;
    for vertex in positions.chunks_exact(3) {
        if let Some((min, max)) = &mut bounds {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        } else {
            let point = [vertex[0], vertex[1], vertex[2]];
            bounds = Some((point, point));
        }
    }
    Ok(DeformedGeometry { positions, normals, bounds })
}

impl DeformedGeometry {
    /// Final mesh-local positions in GPU upload precision.
    pub fn positions(&self) -> &[f32] { &self.positions }

    /// Deformed normals for normal-aware material paths; absent when unauthored.
    pub fn normals(&self) -> Option<&[f32]> { self.normals.as_deref() }

    /// Inclusive local-space (minimum, maximum), or None for empty geometry.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> { self.bounds }

    /// Replace a mesh's positions while preserving indices, transforms, color,
    /// depth, culling, color-write mask and vertex colors. This basic-material
    /// adapter does not consume normals; normal-aware callers use `normals()`.
    /// The returned input can use existing single/batched, canvas/offscreen APIs.
    ///
    /// # Errors
    /// Rejects vertex-count mismatch or invalid existing mesh parameters.
    pub fn mesh_input<'a>(
        &'a self, template: &DynamicMeshInput<'a>,
    ) -> Result<DynamicMeshInput<'a>, DeformedMeshError> {
        checked_length("deformed positions", self.positions.len(), template.positions().len())?;
        let model_view: [f64; 16] = core::array::from_fn(|i| f64::from(template.model_view()[i]));
        let projection: [f64; 16] = core::array::from_fn(|i| f64::from(template.projection()[i]));
        let mut mesh = DynamicMeshInput::try_from_raw(
            &self.positions, template.indices(), &model_view, &projection,
            template.color(), template.width(), template.height(), template.webgl_depth(),
        )?.with_color_write(template.color_write());
        if let Some((cull, winding)) = template.cull() { mesh = mesh.with_cull(cull, winding)?; }
        if let Some(depth) = template.depth() { mesh = mesh.with_depth(depth)?; }
        if let Some(colors) = template.vertex_colors() { mesh = mesh.with_vertex_colors(colors)?; }
        Ok(mesh)
    }
}

/// Build an actual offscreen draw/upload packet for the supplied current pose.
///
/// # Errors
/// Returns deformation or existing render-packet validation errors.
pub fn build_deformed_mesh_submission(
    template: &DynamicMeshInput<'_>, input: &GeometryDeformation<'_>,
) -> Result<GpuSubmissionPacket, DeformedMeshError> {
    checked_length("base positions", input.positions.len(), template.positions().len())?;
    let geometry = deform_geometry(input)?;
    Ok(build_mesh_submission(&geometry.mesh_input(template)?)?)
}

fn matrix(values: &[f64], field: &'static str) -> Result<Matrix4, DeformedMeshError> {
    checked_length(field, values.len(), 16)?;
    let mut elements = [0.0; 16];
    elements.copy_from_slice(values);
    Ok(Matrix4::from_elements(elements))
}

/// Flat-buffer entry point for hosts without Rust slice-of-slices support.
///
/// Morph positions are target-major: each target contains `positions.len()`
/// components. Palettes contain column-major 16-component matrices. Pass all
/// five skin buffers empty to disable skinning. The basic-material packet uses
/// deformed positions; use `deform_geometry` for normal-aware rendering.
///
/// # Errors
/// Returns a diagnostic string for malformed input or failed packet encoding.
#[allow(clippy::too_many_arguments)]
pub fn build_deformed_mesh_packet(
    positions: &[f64], morph_positions: &[f64], morph_weights: &[f64], relative: bool,
    joint_indices: &[u32], joint_weights: &[f64], joint_palette: &[f64],
    bind_matrix: &[f64], bind_matrix_inverse: &[f64],
    indices: &[u32], model_view: &[f64], projection: &[f64], color: &[f32],
    width: u32, height: u32, webgl_depth: bool,
) -> Result<Vec<u8>, String> {
    let build = || -> Result<Vec<u8>, DeformedMeshError> {
        let expected = positions.len().checked_mul(morph_weights.len())
            .ok_or(DeformationError::SizeOverflow)?;
        checked_length("flat morph positions", morph_positions.len(), expected)?;
        let mut targets = Vec::new();
        targets.try_reserve_exact(morph_weights.len()).map_err(|_| DeformedMeshError::Allocation)?;
        for target in 0..morph_weights.len() {
            let start = target * positions.len();
            targets.push(&morph_positions[start..start + positions.len()]);
        }
        let has_skin = !joint_indices.is_empty() || !joint_weights.is_empty()
            || !joint_palette.is_empty() || !bind_matrix.is_empty() || !bind_matrix_inverse.is_empty();
        let mut palette = Vec::new();
        let bind;
        let inverse;
        let skin = if has_skin {
            if joint_palette.len() % 16 != 0 {
                return Err(DeformationError::Length {
                    field: "flat joint palette", expected: joint_palette.len() / 16 * 16,
                    actual: joint_palette.len(),
                }.into());
            }
            bind = matrix(bind_matrix, "bind matrix")?;
            inverse = matrix(bind_matrix_inverse, "inverse bind matrix")?;
            palette.try_reserve_exact(joint_palette.len() / 16).map_err(|_| DeformedMeshError::Allocation)?;
            for values in joint_palette.chunks_exact(16) { palette.push(matrix(values, "joint matrix")?); }
            Some(Skinning { joint_indices, joint_weights, palette: &palette,
                bind_matrix: &bind, bind_matrix_inverse: &inverse })
        } else { None };
        let geometry = deform_geometry(&GeometryDeformation {
            positions, normals: None,
            morph: MorphTargets { positions: &targets, normals: &[], weights: morph_weights, relative },
            skin,
        })?;
        let input = DynamicMeshInput::try_from_raw(
            geometry.positions(), indices, model_view, projection, color, width, height, webgl_depth,
        )?;
        build_mesh_submission(&input)?.encode()
            .map_err(|e| MeshPacketError::EncodeError(format!("{e:?}")).into())
    };
    build().map_err(|e| e.to_string())
}

/// Wasm export of [`build_deformed_mesh_packet`], with the same flat-buffer ABI.
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn f3d_build_deformed_mesh_packet(
    positions: &[f64], morph_positions: &[f64], morph_weights: &[f64], relative: bool,
    joint_indices: &[u32], joint_weights: &[f64], joint_palette: &[f64],
    bind_matrix: &[f64], bind_matrix_inverse: &[f64],
    indices: &[u32], model_view: &[f64], projection: &[f64], color: &[f32],
    width: u32, height: u32, webgl_depth: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_deformed_mesh_packet(
        positions, morph_positions, morph_weights, relative, joint_indices, joint_weights,
        joint_palette, bind_matrix, bind_matrix_inverse, indices, model_view, projection,
        color, width, height, webgl_depth,
    ).map_err(|e| wasm_bindgen::JsValue::from_str(&e))
}

/// Native counterpart of the Wasm export, for host-side callers and tests.
#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
pub use build_deformed_mesh_packet as f3d_build_deformed_mesh_packet;
