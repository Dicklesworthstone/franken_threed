//! WebGPU usage-scope hazard analysis, whole-buffer subresource rules, and automatic pass splitting.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{GraphError, HazardError};
use crate::pass::{Draw, LoadOp, Pass, PassId, PassKind, StoreOp};
use crate::resource::{ResourceAccess, ResourceKind, SubresourceRange};

/// Validates usage-scope rules for a single pass according to WebGPU specifications (§8.5, [S51]).
///
/// Invariants:
/// 1. Whole-buffer rule: For usage scopes, a buffer is a whole subresource. Disjoint byte offsets
///    in ONE buffer cannot legalize incompatible usages (e.g. uniform read + storage write).
/// 2. Render-pass scope: A writable attachment subresource cannot simultaneously be sampled in the same render pass (§8.5).
/// 3. Compute-dispatch scope: Writable aliases within a single compute dispatch are strictly forbidden.
/// 4. Copy separation: Copies cannot be inside a render pass and cannot have overlapping endpoints.
pub fn validate_pass_hazards(pass: &Pass) -> Result<(), HazardError> {
    match pass.kind {
        PassKind::Render => validate_render_pass_hazards(pass),
        PassKind::Compute => validate_compute_pass_hazards(pass),
        PassKind::Copy => validate_copy_pass_hazards(pass),
    }
}

/// Determines if two buffer ranges overlap conservatively according to WebGPU binding rules (§8.5, [S51]).
///
/// Invariant: Half-open intervals `[offset, offset + size)` overlap if `start_a < end_b && start_b < end_a`.
/// If either offset or size is unknown (`None`), zero-sized, or if `offset + size` overflows `u64`,
/// this conservatively returns `true` (potentially overlapping).
fn buffer_ranges_overlap(
    offset_a: Option<u64>,
    size_a: Option<u64>,
    offset_b: Option<u64>,
    size_b: Option<u64>,
) -> bool {
    let (Some(oa), Some(sa), Some(ob), Some(sb)) = (offset_a, size_a, offset_b, size_b) else {
        return true;
    };
    if sa == 0 || sb == 0 {
        return true;
    }
    let (Some(end_a), Some(end_b)) = (oa.checked_add(sa), ob.checked_add(sb)) else {
        return true;
    };
    oa < end_b && ob < end_a
}

/// Validates usage scopes within a WebGPU Render pass.
fn validate_render_pass_hazards(pass: &Pass) -> Result<(), HazardError> {
    // 1. Enforce that copies are never inside a render pass (§6.7)
    if !pass.copies.is_empty() {
        return Err(HazardError::CopyInRenderPass {
            pass_id: pass.id.get(),
        });
    }

    // 2. Collect all attachment uses
    let mut attachment_uses = Vec::new();
    for ca in &pass.color_attachments {
        attachment_uses.extend(ca.resource_uses(f3d_core::ownership::DataVersion::INITIAL));
    }
    if let Some(ref dsa) = pass.depth_stencil_attachment {
        attachment_uses.push(dsa.to_resource_use(f3d_core::ownership::DataVersion::INITIAL));
    }

    // 3. Collect all draw uses
    let mut all_uses = attachment_uses.clone();
    for draw in &pass.draws {
        all_uses.extend(draw.uses.clone());
    }

    // 4. Pairwise check across all uses in the render pass
    for i in 0..all_uses.len() {
        for j in (i + 1)..all_uses.len() {
            let u1 = &all_uses[i];
            let u2 = &all_uses[j];

            if u1.resource_id != u2.resource_id {
                continue;
            }

            // A. Whole-buffer rule check
            if u1.kind == ResourceKind::Buffer && u2.kind == ResourceKind::Buffer {
                if !u1.access.is_compatible_with(&u2.access) {
                    return Err(HazardError::WholeBufferConflict {
                        buffer_id: u1.resource_id.get(),
                        access_a: u1.access,
                        access_b: u2.access,
                        offset_a: u1.byte_offset,
                        offset_b: u2.byte_offset,
                    });
                }
            }

            // B. Texture subresource overlap check
            if (u1.kind == ResourceKind::Texture || u1.kind == ResourceKind::CanvasOutput)
                && (u2.kind == ResourceKind::Texture || u2.kind == ResourceKind::CanvasOutput)
            {
                if u1.subresource.overlaps(&u2.subresource) {
                    // Attachment vs Sampled conflict (writable attachments only; ReadOnlyDepthStencil is compatible, §8.5)
                    if (u1.access.is_write() && u1.access.is_attachment() && u2.access.is_sampled())
                        || (u1.access.is_sampled() && u2.access.is_write() && u2.access.is_attachment())
                    {
                        return Err(HazardError::AttachmentSamplingConflict {
                            texture_id: u1.resource_id.get(),
                            pass_id: pass.id.get(),
                            subresource: u1.subresource.clone(),
                        });
                    }

                    // Multiple writers conflict
                    if u1.access.is_write() && u2.access.is_write() {
                        return Err(HazardError::MultipleWriters {
                            resource_id: u1.resource_id.get(),
                            pass_id: pass.id.get(),
                            subresource: u1.subresource.clone(),
                        });
                    }
                }
            }
        }
    }

    // 5. Per-draw writable alias check (§8.5, [S51], WebGPU storage exception)
    for draw in &pass.draws {
        let uses = &draw.uses;
        for i in 0..uses.len() {
            for j in (i + 1)..uses.len() {
                let u1 = &uses[i];
                let u2 = &uses[j];

                if u1.resource_id != u2.resource_id {
                    continue;
                }

                if u1.kind == ResourceKind::Buffer && u2.kind == ResourceKind::Buffer {
                    if (u1.access.is_write() || u2.access.is_write())
                        && buffer_ranges_overlap(
                            u1.byte_offset,
                            u1.byte_size,
                            u2.byte_offset,
                            u2.byte_size,
                        )
                    {
                        return Err(HazardError::DrawWritableAlias {
                            resource_id: u1.resource_id.get(),
                            draw_id: draw.draw_id,
                            subresource: SubresourceRange::WholeBuffer,
                        });
                    }
                }
            }
        }
    }

    // 6. Enforce render bundle state reset invariant (§8.5, AGENTS.md):
    // In WebGPU, executing a render bundle invalidates all cached pipeline and bind group state.
    // A direct draw following a render bundle cannot assume warm state; it must explicitly rebind.
    let mut last_was_bundle = false;
    for draw in &pass.draws {
        if draw.is_bundle() {
            last_was_bundle = true;
        } else if draw.is_direct() {
            if last_was_bundle && draw.assumes_warm_state() {
                return Err(HazardError::BundleDirectDrawRequiresRebind {
                    pass_id: pass.id.get(),
                    draw_id: draw.draw_id,
                });
            }
            last_was_bundle = false;
        }
    }

    Ok(())
}

/// Validates usage scopes within a WebGPU Compute pass.
///
/// Invariant: In compute, usage scopes are tracked per dispatch (§8.5, [S51]).
/// Writable aliases within a single compute dispatch are rejected.
fn validate_compute_pass_hazards(pass: &Pass) -> Result<(), HazardError> {
    if pass.draws.iter().any(Draw::is_bundle) {
        return Err(HazardError::BundleInNonRenderPass {
            pass_id: pass.id.get(),
            pass_kind: PassKind::Compute,
        });
    }

    // Per-dispatch usage-scope hazard check (§8.5, [S51], WebGPU storage exception)
    for dispatch in &pass.dispatches {
        let uses = &dispatch.uses;

        // 1. Validate explicit group-0 bindings if present (§6.1, root 21419 / 21498)
        for (b_idx, b) in dispatch.bindings.iter().enumerate() {
            if b.use_index >= uses.len() {
                return Err(HazardError::InvalidUseIndex {
                    dispatch_id: dispatch.dispatch_id,
                    binding_index: b.binding_index,
                    use_index: b.use_index,
                    uses_len: uses.len(),
                });
            }

            let u = &uses[b.use_index];

            if u.kind != ResourceKind::Buffer {
                return Err(HazardError::NonBufferBinding {
                    dispatch_id: dispatch.dispatch_id,
                    binding_index: b.binding_index,
                    resource_id: u.resource_id.get(),
                });
            }

            match u.access {
                ResourceAccess::UniformBuffer
                | ResourceAccess::StorageBufferRead
                | ResourceAccess::StorageBufferWrite => {}
                other => {
                    return Err(HazardError::IncompatibleAccess {
                        dispatch_id: dispatch.dispatch_id,
                        binding_index: b.binding_index,
                        access: other,
                    });
                }
            }

            if let Some(0) = u.byte_size {
                return Err(HazardError::UnspecifiedBindingSize {
                    dispatch_id: dispatch.dispatch_id,
                    binding_index: b.binding_index,
                    resource_id: u.resource_id.get(),
                });
            }

            for other_b in &dispatch.bindings[b_idx + 1..] {
                if b.binding_index == other_b.binding_index {
                    return Err(HazardError::DuplicateBindingIndex {
                        dispatch_id: dispatch.dispatch_id,
                        binding_index: b.binding_index,
                    });
                }
            }
        }

        // 2. Reject repeated writable use_index across distinct bindings (root 21773).
        // A repeated read-only use_index (UniformBuffer or StorageBufferRead) is legal.
        for (i, b1) in dispatch.bindings.iter().enumerate() {
            for b2 in &dispatch.bindings[i + 1..] {
                if b1.use_index == b2.use_index {
                    let u = &uses[b1.use_index];
                    if u.access.is_write() {
                        return Err(HazardError::ComputeWritableAlias {
                            resource_id: u.resource_id.get(),
                            dispatch_id: dispatch.dispatch_id,
                            subresource: SubresourceRange::WholeBuffer,
                        });
                    }
                }
            }
        }

        // 3. Pair-loop over dispatch.uses for distinct usages targeting the same resource
        for i in 0..uses.len() {
            for j in (i + 1)..uses.len() {
                let u1 = &uses[i];
                let u2 = &uses[j];

                if u1.resource_id != u2.resource_id {
                    continue;
                }

                // Buffer checks within a single compute dispatch
                if u1.kind == ResourceKind::Buffer && u2.kind == ResourceKind::Buffer {
                    // 1. Overlapping writable ranges are rejected as aliases
                    if (u1.access.is_write() || u2.access.is_write())
                        && buffer_ranges_overlap(
                            u1.byte_offset,
                            u1.byte_size,
                            u2.byte_offset,
                            u2.byte_size,
                        )
                    {
                        return Err(HazardError::ComputeWritableAlias {
                            resource_id: u1.resource_id.get(),
                            dispatch_id: dispatch.dispatch_id,
                            subresource: SubresourceRange::WholeBuffer,
                        });
                    }

                    // 2. Disjoint ranges with incompatible roles are whole-buffer conflicts
                    if !u1.access.is_compatible_with(&u2.access) {
                        return Err(HazardError::WholeBufferConflict {
                            buffer_id: u1.resource_id.get(),
                            access_a: u1.access,
                            access_b: u2.access,
                            offset_a: u1.byte_offset,
                            offset_b: u2.byte_offset,
                        });
                    }
                }

                // Texture writable alias check
                if u1.kind == ResourceKind::Texture && u2.kind == ResourceKind::Texture {
                    if u1.subresource.overlaps(&u2.subresource)
                        && (u1.access.is_write() || u2.access.is_write())
                    {
                        return Err(HazardError::ComputeWritableAlias {
                            resource_id: u1.resource_id.get(),
                            dispatch_id: dispatch.dispatch_id,
                            subresource: u1.subresource.clone(),
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validates usage scopes within a WebGPU Copy pass.
fn validate_copy_pass_hazards(pass: &Pass) -> Result<(), HazardError> {
    if pass.draws.iter().any(Draw::is_bundle) {
        return Err(HazardError::BundleInCopyPass {
            pass_id: pass.id.get(),
        });
    }
    for copy in &pass.copies {
        let (src, dst) = copy.all_uses(f3d_core::ownership::DataVersion::INITIAL);
        if src.resource_id == dst.resource_id && src.subresource.overlaps(&dst.subresource) {
            return Err(HazardError::OverlappingCopyEndpoints {
                resource_id: src.resource_id.get(),
            });
        }
    }
    Ok(())
}

/// Whether sequential output-preserving passes can resolve cross-draw buffer usage conflicts.
///
/// Color attachment slots apply to every draw. Without per-draw output declarations,
/// sampling an attachment cannot justify removing that attachment from a later segment.
#[must_use]
pub fn can_split_pass(pass: &Pass) -> bool {
    output_preserving_split(pass).is_some()
}

/// Build maximal legal draw groups, preserving every output and its attachment location.
fn output_preserving_split(pass: &Pass) -> Option<Vec<Pass>> {
    if pass.kind != PassKind::Render
        || !pass.dispatches.is_empty()
        || !matches!(
            validate_pass_hazards(pass),
            Err(HazardError::WholeBufferConflict { .. })
        )
    {
        return None;
    }

    let mut ranges = Vec::new();
    let mut start = 0;
    let mut inherited_draw_state = false;
    for (idx, draw) in pass.draws.iter().enumerate() {
        let conflicts_with_prefix = draw.uses.iter().any(|current| {
            current.kind == ResourceKind::Buffer
                && pass.draws[start..idx]
                    .iter()
                    .flat_map(|prior| &prior.uses)
                    .any(|prior| {
                        prior.kind == ResourceKind::Buffer
                            && prior.resource_id == current.resource_id
                            && !prior.access.is_compatible_with(&current.access)
                    })
        });
        if conflicts_with_prefix {
            // The graph cannot materialize implicit viewport/scissor or binding
            // state. Every introduced boundary must be safe, including later ones.
            if draw.assumes_warm_state || inherited_draw_state {
                return None;
            }
            ranges.push(start..idx);
            start = idx;
        }
        inherited_draw_state |=
            draw.viewport.is_some() || draw.scissor.is_some() || draw.scissor_test_enabled;
    }
    if ranges.is_empty() {
        return None;
    }
    ranges.push(start..pass.draws.len());

    // Clone shared pass metadata once; do not clone the entire draw list once
    // per segment when a pass changes buffer roles repeatedly.
    let mut template = pass.clone();
    template.draws.clear();
    let mut segments = Vec::with_capacity(ranges.len());
    let last_index = ranges.len() - 1;
    for (index, range) in ranges.into_iter().enumerate() {
        let mut segment = template.clone();
        segment.draws.extend_from_slice(&pass.draws[range]);
        for attachment in &mut segment.color_attachments {
            if index > 0 {
                attachment.load_op = LoadOp::Load;
            }
            if index < last_index {
                attachment.store_op = StoreOp::Store;
                // Publish the resolve only at the source pass's final boundary.
                attachment.resolve_target = None;
            }
        }

        if let Some(attachment) = &mut segment.depth_stencil_attachment {
            for (read_only, load, store) in [
                (
                    attachment.depth_read_only,
                    &mut attachment.depth_load_op,
                    &mut attachment.depth_store_op,
                ),
                (
                    attachment.stencil_read_only,
                    &mut attachment.stencil_load_op,
                    &mut attachment.stencil_store_op,
                ),
            ] {
                if !read_only {
                    match (*load, *store) {
                        (Some(_), Some(_)) => {
                            if index > 0 {
                                *load = Some(LoadOp::Load);
                            }
                            if index < last_index {
                                *store = Some(StoreOp::Store);
                            }
                        }
                        (None, None) => {}
                        _ => return None,
                    }
                }
            }
        }

        // A within-draw conflict, attachment feedback, or other independent
        // hazard must not disappear just because another usage was split.
        validate_pass_hazards(&segment).ok()?;
        segments.push(segment);
    }
    Some(segments)
}

/// Splits a conflicting render pass into sequential legal passes.
///
/// Invariant: Preserves draw ordering and exact attachment semantics.
/// Does NOT silently drop sampled outputs or depth/stencil state!
/// Later passes load the preceding pass's stored contents. Additional IDs are
/// contiguous starting at `next_pass_id`; exhaustion and ID collisions return
/// structured graph errors rather than wrapping or producing a malformed graph.
pub fn split_pass_on_hazard(pass: &Pass, next_pass_id: PassId) -> Result<Vec<Pass>, GraphError> {
    let original_error = match validate_pass_hazards(pass) {
        Ok(()) => return Ok(vec![pass.clone()]),
        Err(error) => error,
    };
    let Some(mut segments) = output_preserving_split(pass) else {
        return Err(original_error.into());
    };
    let additional_count =
        u32::try_from(segments.len() - 1).map_err(|_| GraphError::PassIdOverflow)?;
    let last_id = next_pass_id
        .get()
        .checked_add(additional_count - 1)
        .ok_or(GraphError::PassIdOverflow)?;
    if (next_pass_id.get()..=last_id).contains(&pass.id.get()) {
        return Err(GraphError::DuplicatePassId {
            pass_id: pass.id.get(),
        });
    }
    let mut previous_id = pass.id;
    for (index, segment) in segments.iter_mut().enumerate() {
        segment.name = alloc::format!("{}_part{}", pass.name, index + 1);
        if index > 0 {
            // The complete range was checked above, so every offset fits u32.
            segment.id = PassId::new(next_pass_id.get() + (index as u32 - 1));
            segment.dependencies = vec![previous_id];
            previous_id = segment.id;
        }
    }
    Ok(segments)
}
