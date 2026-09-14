//! WebGPU usage-scope hazard analysis, whole-buffer subresource rules, and automatic pass splitting.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::HazardError;
use crate::pass::{ColorAttachment, Draw, LoadOp, Pass, PassId, PassKind, StoreOp};
use crate::resource::{ResourceAccess, ResourceId, ResourceKind, SubresourceRange};

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
        attachment_uses.push(ca.to_resource_use(f3d_core::ownership::DataVersion::INITIAL));
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

/// Evaluates whether a pass has hazards that can be resolved by pass splitting without dropping outputs.
///
/// Invariant: Splitting is legal ONLY when draws can be partitioned such that:
/// 1. Earlier draws write to an intermediate target.
/// 2. Later draws sample that intermediate target and output to distinct targets.
/// 3. Later draws do NOT attempt to write to the sampled target (unsupported feedback loop).
/// 4. Depth/stencil state is preserved across split segments without discarding depth outputs.
#[must_use]
pub fn can_split_pass(pass: &Pass) -> bool {
    if pass.kind != PassKind::Render || pass.draws.len() < 2 {
        return false;
    }
    let split_idx = match find_attachment_sampling_split_point(pass) {
        Some(idx) if idx > 0 => idx,
        _ => return false,
    };

    let (_, second_draws) = pass.draws.split_at(split_idx);

    let mut attachment_ids = Vec::new();
    for ca in &pass.color_attachments {
        attachment_ids.push(ca.target_id);
    }
    if let Some(ref dsa) = pass.depth_stencil_attachment {
        attachment_ids.push(dsa.target_id);
    }

    let sampled_ids: Vec<ResourceId> = second_draws
        .iter()
        .flat_map(|d| d.uses.iter())
        .filter(|u| u.access.is_sampled() && attachment_ids.contains(&u.resource_id))
        .map(|u| u.resource_id)
        .collect();

    // If all color attachments in the pass are sampled in second_draws,
    // second_draws has no separate legal destination; dropping them is illegal feedback.
    let remaining_color_targets: Vec<&ColorAttachment> = pass
        .color_attachments
        .iter()
        .filter(|ca| !sampled_ids.contains(&ca.target_id))
        .collect();

    if remaining_color_targets.is_empty() && !pass.color_attachments.is_empty() {
        return false;
    }

    // If depth/stencil is sampled in second_draws and depth or stencil is writable, that is also unsupported feedback.
    if let Some(ref dsa) = pass.depth_stencil_attachment {
        if sampled_ids.contains(&dsa.target_id) && (!dsa.depth_read_only || !dsa.stencil_read_only) {
            return false;
        }
    }

    true
}

/// Finds the draw index where an attachment is first sampled.
fn find_attachment_sampling_split_point(pass: &Pass) -> Option<usize> {
    let mut attachment_ids = Vec::new();
    for ca in &pass.color_attachments {
        attachment_ids.push(ca.target_id);
    }
    if let Some(ref dsa) = pass.depth_stencil_attachment {
        // Read-only depth-stencil attachments can legally be sampled without hazard (§8.5).
        // Only writable depth/stencil attachments can create attachment-sampling conflicts.
        if !dsa.depth_read_only || !dsa.stencil_read_only {
            attachment_ids.push(dsa.target_id);
        }
    }

    for (draw_idx, draw) in pass.draws.iter().enumerate() {
        for u in &draw.uses {
            if u.access.is_sampled() && attachment_ids.contains(&u.resource_id) {
                return Some(draw_idx);
            }
        }
    }
    None
}

/// Splits a conflicting render pass into two sequential legal passes.
///
/// Invariant: Preserves draw ordering and exact attachment semantics.
/// Does NOT silently drop sampled outputs or depth/stencil state!
/// The second pass loads existing contents (`LoadOp::Load`) from the first pass's store (`StoreOp::Store`).
pub fn split_pass_on_hazard(pass: &Pass, next_pass_id: PassId) -> Result<Vec<Pass>, HazardError> {
    if pass.kind != PassKind::Render {
        return validate_pass_hazards(pass).map(|()| vec![pass.clone()]);
    }

    // Check if initial pass is already legal
    if validate_pass_hazards(pass).is_ok() {
        return Ok(vec![pass.clone()]);
    }

    // If the pass cannot be split without dropping outputs or resolving unsupported feedback,
    // reject with the precise hazard error.
    if !can_split_pass(pass) {
        return Err(validate_pass_hazards(pass).unwrap_err());
    }

    let split_idx = find_attachment_sampling_split_point(pass).unwrap();
    let (first_draws, second_draws) = pass.draws.split_at(split_idx);

    let mut attachment_ids = Vec::new();
    for ca in &pass.color_attachments {
        attachment_ids.push(ca.target_id);
    }
    if let Some(ref dsa) = pass.depth_stencil_attachment {
        attachment_ids.push(dsa.target_id);
    }

    let sampled_ids: Vec<ResourceId> = second_draws
        .iter()
        .flat_map(|d| d.uses.iter())
        .filter(|u| u.access.is_sampled() && attachment_ids.contains(&u.resource_id))
        .map(|u| u.resource_id)
        .collect();

    // Pass 1: Executes draws before the sampling conflict, stores all results
    let mut pass1 = pass.clone();
    pass1.name = alloc::format!("{}_part1", pass.name);
    pass1.draws = first_draws.to_vec();
    for ca in &mut pass1.color_attachments {
        ca.store_op = StoreOp::Store;
    }
    if let Some(ref mut dsa) = pass1.depth_stencil_attachment {
        if !dsa.depth_read_only {
            dsa.depth_store_op = Some(StoreOp::Store);
        }
        if !dsa.stencil_read_only && dsa.stencil_store_op.is_some() {
            dsa.stencil_store_op = Some(StoreOp::Store);
        }
    }

    // Pass 2: Loads results from pass 1, depends on pass 1, executes remaining draws
    let mut pass2 = Pass::new_render(next_pass_id, alloc::format!("{}_part2", pass.name));
    pass2.dependencies = vec![pass1.id];
    pass2.draws = second_draws.to_vec();

    // Preserve color attachments that are NOT sampled in pass 2 (distinct legal destinations)
    for ca in &pass.color_attachments {
        if !sampled_ids.contains(&ca.target_id) {
            let mut ca2 = ca.clone();
            ca2.load_op = LoadOp::Load;
            ca2.store_op = StoreOp::Store;
            pass2.color_attachments.push(ca2);
        }
    }

    // PRESERVE depth/stencil attachment in Pass 2!
    // Read-only depth-stencil attachments can legally be sampled in Pass 2 and must not be dropped.
    // Read-only attachments retain absent (None) load/store operations.
    if let Some(ref dsa) = pass.depth_stencil_attachment {
        let is_sampled = sampled_ids.contains(&dsa.target_id);
        let is_read_only = dsa.depth_read_only && dsa.stencil_read_only;
        if !is_sampled || is_read_only {
            let mut dsa2 = dsa.clone();
            if !dsa2.depth_read_only {
                dsa2.depth_load_op = Some(LoadOp::Load);
                dsa2.depth_store_op = Some(StoreOp::Store);
            }
            if !dsa2.stencil_read_only && dsa2.stencil_load_op.is_some() {
                dsa2.stencil_load_op = Some(LoadOp::Load);
                dsa2.stencil_store_op = Some(StoreOp::Store);
            }
            pass2.depth_stencil_attachment = Some(dsa2);
        }
    }

    // Strict validation of both split passes
    validate_pass_hazards(&pass1)?;
    validate_pass_hazards(&pass2)?;

    Ok(vec![pass1, pass2])
}
