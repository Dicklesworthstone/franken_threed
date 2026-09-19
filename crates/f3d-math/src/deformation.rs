//! Allocation-free morphing and linear-blend skinning for dynamic geometry.
//!
//! Morphing precedes skinning. The formulas follow the pinned r186 shader chunks
//! `morphtarget_vertex`, `skinning_vertex`, and `skinnormal_vertex`. Arithmetic
//! stays in f64; the runtime narrows only the final upload. This is a CPU
//! execution path, not a claim of GPU skinning or bit-identical shader rounding.

use core::fmt;

use crate::Matrix4;

/// Invalid geometry, palette, or deformation output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeformationError {
    /// A parallel attribute or output has the wrong number of components.
    Length { field: &'static str, expected: usize, actual: usize },
    /// A position/direction attribute does not contain complete xyz triples.
    IncompleteVertex { components: usize },
    /// An input or computed output contains a non-finite component.
    NonFinite { field: &'static str, index: usize },
    /// An active skin influence refers outside the supplied joint palette.
    JointOutOfBounds { influence: usize, joint: u32, joint_count: usize },
    /// Four influences per vertex cannot be represented by this platform.
    SizeOverflow,
}

impl fmt::Display for DeformationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { field, expected, actual } =>
                write!(f, "{field}: expected {expected} components, got {actual}"),
            Self::IncompleteVertex { components } =>
                write!(f, "attribute has {components} components, not complete xyz triples"),
            Self::NonFinite { field, index } =>
                write!(f, "non-finite {field} component at {index}"),
            Self::JointOutOfBounds { influence, joint, joint_count } =>
                write!(f, "influence {influence} uses joint {joint}, palette has {joint_count}"),
            Self::SizeOverflow => write!(f, "deformation attribute size overflow"),
        }
    }
}

impl core::error::Error for DeformationError {}

fn length(field: &'static str, actual: usize, expected: usize) -> Result<(), DeformationError> {
    if actual != expected {
        return Err(DeformationError::Length { field, expected, actual });
    }
    Ok(())
}

fn finite(field: &'static str, values: &[f64]) -> Result<(), DeformationError> {
    if let Some(index) = values.iter().position(|x| !x.is_finite()) {
        return Err(DeformationError::NonFinite { field, index });
    }
    Ok(())
}

/// Blend a flat attribute into caller-owned storage without allocating.
///
/// Absolute targets use `base * (1 - sum(weights)) + sum(target * weight)`;
/// relative targets use `base + sum(delta * weight)`. Weights are deliberately
/// not normalized or clamped: negative and extrapolating influences are valid.
/// Works for positions, normals, or other consistently sized attributes.
///
/// Input/shape errors leave output untouched. On arithmetic overflow, callers
/// must discard the output (some earlier components may have been written).
/// Normals are not normalized here; normalization belongs after transformation.
///
/// # Errors
/// Rejects mismatched lengths, non-finite inputs, and non-finite results.
pub fn morph_attribute_into(
    base: &[f64],
    targets: &[&[f64]],
    weights: &[f64],
    relative: bool,
    output: &mut [f64],
) -> Result<(), DeformationError> {
    length("morph weights", weights.len(), targets.len())?;
    length("morph output", output.len(), base.len())?;
    finite("base attribute", base)?;
    finite("morph weights", weights)?;
    for target in targets {
        length("morph target", target.len(), base.len())?;
        finite("morph target", target)?;
    }
    let base_weight = if relative { 1.0 } else { 1.0 - weights.iter().sum::<f64>() };
    finite("morph base weight", &[base_weight])?;
    for (index, (value, destination)) in base.iter().zip(output).enumerate() {
        let mut result = value * base_weight;
        for (target, &weight) in targets.iter().zip(weights) {
            if weight != 0.0 {
                result += target[index] * weight;
            }
        }
        if !result.is_finite() {
            return Err(DeformationError::NonFinite { field: "morph output", index });
        }
        *destination = result;
    }
    Ok(())
}

/// Borrowed four-influence skin data and column-major joint transforms.
///
/// `palette[j]` is `joint_world[j] * inverse_bind[j]`, in joint-index order,
/// not scene traversal order. The mesh bind matrices remain separate. This
/// supports both attached and detached bind modes when the caller supplies the
/// corresponding current inverse bind matrix.
#[derive(Clone, Copy, Debug)]
pub struct Skinning<'a> {
    /// Four joint indices per vertex.
    pub joint_indices: &'a [u32],
    /// Four authored weights per vertex; never implicitly normalized.
    pub joint_weights: &'a [f64],
    /// Joint-world multiplied by inverse-bind matrices.
    pub palette: &'a [Matrix4],
    /// Mesh-local to bind-space transform.
    pub bind_matrix: &'a Matrix4,
    /// Current bind-space to mesh-local transform.
    pub bind_matrix_inverse: &'a Matrix4,
}

/// Whether translation should act on an attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkinAttribute {
    /// Positions use homogeneous w = 1.
    Position,
    /// Normals/tangents use homogeneous w = 0 and are not normalized here.
    Direction,
}

fn transform(matrix: &Matrix4, v: [f64; 4]) -> [f64; 4] {
    let e = &matrix.elements;
    core::array::from_fn(|row| {
        e[row] * v[0] + e[4 + row] * v[1] + e[8 + row] * v[2] + e[12 + row] * v[3]
    })
}

/// Skin an xyz attribute into caller-owned storage, without allocating.
///
/// Evaluates `bind_inverse * sum(weight * joint * bind * attribute)`. The
/// blended homogeneous w is preserved through the inverse bind transform;
/// dividing by w or replacing it with 1 would break authored non-unit weights.
/// Zero-weight influences do not dereference their joint index. Input/shape
/// errors leave output untouched; discard output after arithmetic overflow.
///
/// # Errors
/// Rejects malformed attributes, active out-of-bounds joints, non-finite inputs,
/// and non-finite results. Does not repair or normalize authored skin weights.
pub fn skin_attribute_into(
    attribute: &[f64],
    skin: &Skinning<'_>,
    kind: SkinAttribute,
    output: &mut [f64],
) -> Result<(), DeformationError> {
    if attribute.len() % 3 != 0 {
        return Err(DeformationError::IncompleteVertex { components: attribute.len() });
    }
    let influences = (attribute.len() / 3).checked_mul(4).ok_or(DeformationError::SizeOverflow)?;
    length("skin output", output.len(), attribute.len())?;
    length("joint indices", skin.joint_indices.len(), influences)?;
    length("joint weights", skin.joint_weights.len(), influences)?;
    finite("skin attribute", attribute)?;
    finite("joint weights", skin.joint_weights)?;
    finite("bind matrix", &skin.bind_matrix.elements)?;
    finite("inverse bind matrix", &skin.bind_matrix_inverse.elements)?;
    for matrix in skin.palette {
        finite("joint palette", &matrix.elements)?;
    }
    for (influence, (&joint, &weight)) in skin.joint_indices.iter().zip(skin.joint_weights).enumerate() {
        if weight != 0.0 && (joint as usize) >= skin.palette.len() {
            return Err(DeformationError::JointOutOfBounds {
                influence, joint, joint_count: skin.palette.len(),
            });
        }
    }
    let w = if kind == SkinAttribute::Position { 1.0 } else { 0.0 };
    for (vertex, (source, destination)) in attribute.chunks_exact(3).zip(output.chunks_exact_mut(3)).enumerate() {
        let bound = transform(skin.bind_matrix, [source[0], source[1], source[2], w]);
        let mut blended = [0.0; 4];
        for lane in 0..4 {
            let offset = vertex * 4 + lane;
            let weight = skin.joint_weights[offset];
            if weight != 0.0 {
                let joint = &skin.palette[skin.joint_indices[offset] as usize];
                let transformed = transform(joint, bound);
                for component in 0..4 {
                    blended[component] += transformed[component] * weight;
                }
            }
        }
        let result = transform(skin.bind_matrix_inverse, blended);
        for component in 0..3 {
            if !result[component].is_finite() {
                return Err(DeformationError::NonFinite {
                    field: "skin output", index: vertex * 3 + component,
                });
            }
        }
        destination.copy_from_slice(&result[..3]);
    }
    Ok(())
}

/// Build a reusable joint palette from joint-world and inverse-bind matrices.
///
/// Inputs must already be in the skin's joint-index order. No scene traversal or
/// allocation occurs. Shape/non-finite input errors leave output untouched;
/// discard output after arithmetic overflow.
///
/// # Errors
/// Rejects length mismatches and non-finite inputs/results.
pub fn build_joint_palette_into(
    joint_world: &[Matrix4],
    inverse_bind: &[Matrix4],
    output: &mut [Matrix4],
) -> Result<(), DeformationError> {
    length("inverse bind palette", inverse_bind.len(), joint_world.len())?;
    length("joint palette output", output.len(), joint_world.len())?;
    for matrix in joint_world.iter().chain(inverse_bind) {
        finite("joint matrix", &matrix.elements)?;
    }
    for ((world, inverse), destination) in joint_world.iter().zip(inverse_bind).zip(output) {
        let mut result = Matrix4::zero();
        result.multiply_matrices(world, inverse);
        finite("joint palette output", &result.elements)?;
        *destination = result;
    }
    Ok(())
}
