//! Execution plan representations and pre-compiled bridge segments (§6.1, §6.3).

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use f3d_core::layout::{aligned_bytes_per_row, validate_copy_bytes_per_row};
use f3d_core::ownership::{DataVersion, Epoch};

use crate::canvas::{CanvasEpochTracker, CanvasId};
use crate::error::GraphError;
use crate::pass::{
    ColorAttachment, CopyCommand, DepthStencilAttachment, Dispatch, Draw, Pass, PassId, PassKind,
};
use crate::resource::{ResourceId, ResourceUse};
use crate::schedule::PassGraph;

/// An individual scheduled segment within the compiled execution plan.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlanSegment {
    /// ID of the original or split pass.
    pub pass_id: PassId,
    /// Diagnostic name.
    pub name: String,
    /// Execution kind (Render, Compute, Copy).
    pub kind: PassKind,
    /// Color attachment configurations.
    pub color_attachments: Vec<ColorAttachment>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
    /// Draw call buckets.
    pub draws: Vec<Draw>,
    /// Compute dispatches.
    pub dispatches: Vec<Dispatch>,
    /// Copy operations.
    pub copies: Vec<CopyCommand>,
    /// All versioned resources read by this segment.
    pub versioned_reads: Vec<(ResourceId, DataVersion)>,
    /// All versioned resources written by this segment.
    pub versioned_writes: Vec<(ResourceId, DataVersion)>,
}

impl PlanSegment {
    /// Create a plan segment from a validated pass.
    pub fn from_pass(pass: &Pass) -> Self {
        let mut versioned_reads = Vec::new();
        let mut versioned_writes = Vec::new();

        for u in pass.all_uses() {
            if u.access.is_write() {
                if !versioned_writes.iter().any(|&(id, _)| id == u.resource_id) {
                    versioned_writes.push((u.resource_id, u.version));
                }
            } else if !versioned_reads.iter().any(|&(id, _)| id == u.resource_id) {
                versioned_reads.push((u.resource_id, u.version));
            }
        }

        Self {
            pass_id: pass.id,
            name: pass.name.clone(),
            kind: pass.kind,
            color_attachments: pass.color_attachments.clone(),
            depth_stencil_attachment: pass.depth_stencil_attachment.clone(),
            draws: pass.draws.clone(),
            dispatches: pass.dispatches.clone(),
            copies: pass.copies.clone(),
            versioned_reads,
            versioned_writes,
        }
    }

    /// Collect all resource uses declared in this segment.
    pub fn all_uses(&self) -> Vec<ResourceUse> {
        let mut uses = Vec::new();
        for ca in &self.color_attachments {
            uses.push(ca.to_resource_use(DataVersion::INITIAL));
        }
        if let Some(ref dsa) = self.depth_stencil_attachment {
            uses.push(dsa.to_resource_use(DataVersion::INITIAL));
        }
        for draw in &self.draws {
            uses.extend(draw.uses.clone());
        }
        for dispatch in &self.dispatches {
            uses.extend(dispatch.uses.clone());
        }
        for copy in &self.copies {
            let (src, dst) = copy.all_uses(DataVersion::INITIAL);
            uses.push(src);
            uses.push(dst);
        }
        uses
    }

    /// The unique PassId identifying this segment.
    #[inline]
    #[must_use]
    pub const fn pass_id(&self) -> PassId {
        self.pass_id
    }

    /// Diagnostic name of this pass segment.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Pass execution category (Render, Compute, Copy).
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> PassKind {
        self.kind
    }

    /// Returns `true` if this segment encodes a Render pass.
    #[inline]
    #[must_use]
    pub const fn is_render(&self) -> bool {
        matches!(self.kind, PassKind::Render)
    }

    /// Returns `true` if this segment encodes a Compute pass.
    #[inline]
    #[must_use]
    pub const fn is_compute(&self) -> bool {
        matches!(self.kind, PassKind::Compute)
    }

    /// Returns `true` if this segment encodes a Copy pass.
    #[inline]
    #[must_use]
    pub const fn is_copy(&self) -> bool {
        matches!(self.kind, PassKind::Copy)
    }

    /// Slice of color attachments configured for this segment.
    #[inline]
    #[must_use]
    pub fn color_attachments(&self) -> &[ColorAttachment] {
        &self.color_attachments
    }

    /// Primary color attachment (first attachment), if any.
    #[inline]
    #[must_use]
    pub fn primary_color_attachment(&self) -> Option<&ColorAttachment> {
        self.color_attachments.first()
    }

    /// Optional depth/stencil attachment configured for this segment.
    #[inline]
    #[must_use]
    pub fn depth_stencil_attachment(&self) -> Option<&DepthStencilAttachment> {
        self.depth_stencil_attachment.as_ref()
    }

    /// Slice of draw commands contained in this segment.
    #[inline]
    #[must_use]
    pub fn draws(&self) -> &[Draw] {
        &self.draws
    }

    /// Number of draw commands in this segment.
    #[inline]
    #[must_use]
    pub fn draw_count(&self) -> usize {
        self.draws.len()
    }

    /// First draw command in this segment, if any.
    #[inline]
    #[must_use]
    pub fn first_draw(&self) -> Option<&Draw> {
        self.draws.first()
    }

    /// Slice of compute dispatches contained in this segment.
    #[inline]
    #[must_use]
    pub fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
    }

    /// Number of compute dispatches in this segment.
    #[inline]
    #[must_use]
    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    /// Slice of copy commands contained in this segment.
    #[inline]
    #[must_use]
    pub fn copies(&self) -> &[CopyCommand] {
        &self.copies
    }

    /// Number of copy commands in this segment.
    #[inline]
    #[must_use]
    pub fn copy_count(&self) -> usize {
        self.copies.len()
    }

    /// First copy command in this segment, if any.
    #[inline]
    #[must_use]
    pub fn first_copy(&self) -> Option<&CopyCommand> {
        self.copies.first()
    }

    /// Versioned resources read by this segment.
    #[inline]
    #[must_use]
    pub fn versioned_reads(&self) -> &[(ResourceId, DataVersion)] {
        &self.versioned_reads
    }

    /// Versioned resources written by this segment.
    #[inline]
    #[must_use]
    pub fn versioned_writes(&self) -> &[(ResourceId, DataVersion)] {
        &self.versioned_writes
    }
}

/// The compiled, ordered execution plan submitted to the bridge encoder (§6.1, §6.4).
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionPlan {
    /// Ordered list of executable plan segments.
    pub segments: Vec<PlanSegment>,
    /// Output epoch of any canvas outputs bound to this plan.
    pub canvas_epoch: Option<Epoch>,
    /// Number of passes produced after any automatic hazard splits.
    pub pass_count: usize,
    /// Number of split passes introduced during compilation.
    pub split_count: usize,
    /// Diagnostic descriptions of reasons why passes were split.
    pub split_reasons: Vec<String>,
}

impl ExecutionPlan {
    /// Returns the number of segments in the plan.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns `true` if the plan has no segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Read-only slice of segments.
    #[must_use]
    pub fn segments(&self) -> &[PlanSegment] {
        &self.segments
    }

    /// Captured canvas output epoch for the plan, if canvas swapchain is bound.
    #[inline]
    #[must_use]
    pub const fn canvas_epoch(&self) -> Option<Epoch> {
        self.canvas_epoch
    }

    /// Total number of pass segments in the plan.
    #[inline]
    #[must_use]
    pub const fn pass_count(&self) -> usize {
        self.pass_count
    }

    /// Number of split passes introduced during compilation.
    #[inline]
    #[must_use]
    pub const fn split_count(&self) -> usize {
        self.split_count
    }

    /// Diagnostic descriptions of reasons why passes were split.
    #[inline]
    #[must_use]
    pub fn split_reasons(&self) -> &[String] {
        &self.split_reasons
    }

    /// Validates canvas freshness at the execution/publication boundary.
    ///
    /// Compares the plan's captured canvas epoch against the tracker's current active epoch.
    pub fn validate_execution(&self, tracker: &CanvasEpochTracker) -> Result<(), GraphError> {
        for segment in &self.segments {
            for ca in &segment.color_attachments {
                if let Some(epoch) = ca.canvas_epoch {
                    let canvas_id = tracker
                        .find_by_resource(ca.target_id)
                        .unwrap_or_else(|| CanvasId::new(ca.target_id.get()));
                    tracker.validate_canvas_access(canvas_id, epoch)?;
                } else if let Some(canvas_id) = tracker.find_by_resource(ca.target_id) {
                    return Err(GraphError::Canvas(crate::error::CanvasError::CanvasNotAcquired {
                        canvas_id: canvas_id.get(),
                    }));
                }
            }
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Canonical Bridge Plans (Coordinating with ChartreuseFern's bridge cases)
// -----------------------------------------------------------------------------

/// Constructs an executable single-pass bridge plan rendering a triangle to canvas swapchain.
///
/// Invariant: Passes through `PassGraph::compile` to enforce validation rather than bypassing it.
pub fn build_single_pass_bridge_plan(
    canvas_resource_id: ResourceId,
    pipeline_id: u32,
    vertex_buffer_id: ResourceId,
    vertex_count: u32,
    version: DataVersion,
    canvas_epoch: Option<Epoch>,
    canvas_tracker: Option<&CanvasEpochTracker>,
) -> Result<ExecutionPlan, GraphError> {
    let mut graph = PassGraph::new();
    let pass_id = PassId::new(1);
    let mut pass = Pass::new_render(pass_id, "bridge_single_pass");

    // Color attachment targeting canvas swapchain with captured epoch
    if let Some(epoch) = canvas_epoch {
        pass.color_attachments.push(ColorAttachment::new_canvas(
            canvas_resource_id,
            [0.0, 0.0, 0.0, 1.0],
            epoch,
        ));
    } else {
        pass.color_attachments.push(ColorAttachment::new_clear(
            canvas_resource_id,
            [0.0, 0.0, 0.0, 1.0],
        ));
    }

    // Single draw using vertex buffer
    let draw = Draw::new(
        0,
        pipeline_id,
        vertex_count,
        0,
        vec![ResourceUse::buffer_vertex(
            vertex_buffer_id,
            version,
            Some(0),
            None,
        )],
    );
    pass.draws.push(draw);
    graph.add_pass(pass)?;

    graph.compile(canvas_tracker)
}

/// Constructs an executable two-target bridge plan with WebGPU copy pitch alignment (§8.5):
/// Pass 1: Offscreen target render pass
/// Pass 2: Canvas swapchain render pass
/// Pass 3: Copy texture to readback buffer (bytesPerRow aligned to 256 bytes)
///
/// Invariant: Passes through `PassGraph::compile` to enforce validation rather than bypassing it.
pub fn build_two_target_bridge_plan(
    offscreen_texture_id: ResourceId,
    canvas_resource_id: ResourceId,
    readback_buffer_id: ResourceId,
    pipeline_offscreen: u32,
    pipeline_canvas: u32,
    vertex_buffer_id: ResourceId,
    vertex_count: u32,
    width: u32,
    height: u32,
    version: DataVersion,
    canvas_epoch: Option<Epoch>,
    canvas_tracker: Option<&CanvasEpochTracker>,
) -> Result<ExecutionPlan, GraphError> {
    // Calculate 256-byte aligned copy bytes_per_row using core layout helper
    let bytes_per_row = aligned_bytes_per_row(width).map_err(|_| GraphError::InvalidPass {
        pass_id: 3,
        reason: String::from("bytes_per_row calculation overflow or alignment violation"),
    })?;
    validate_copy_bytes_per_row(bytes_per_row).map_err(|_| GraphError::InvalidPass {
        pass_id: 3,
        reason: String::from("unaligned bytes_per_row violating COPY_BYTES_PER_ROW_ALIGNMENT"),
    })?;

    let mut graph = PassGraph::new();

    // 1. Offscreen render pass
    let p1_id = PassId::new(1);
    let mut p1 = Pass::new_render(p1_id, "bridge_offscreen_pass");
    p1.color_attachments.push(ColorAttachment::new_clear(
        offscreen_texture_id,
        [0.0, 0.0, 0.0, 1.0],
    ));
    p1.draws.push(Draw::new(
        0,
        pipeline_offscreen,
        vertex_count,
        0,
        vec![ResourceUse::buffer_vertex(
            vertex_buffer_id,
            version,
            Some(0),
            None,
        )],
    ));
    graph.add_pass(p1)?;

    // 2. Canvas swapchain render pass
    let p2_id = PassId::new(2);
    let mut p2 = Pass::new_render(p2_id, "bridge_canvas_pass");
    if let Some(epoch) = canvas_epoch {
        p2.color_attachments.push(ColorAttachment::new_canvas(
            canvas_resource_id,
            [0.0, 0.0, 0.0, 1.0],
            epoch,
        ));
    } else {
        p2.color_attachments.push(ColorAttachment::new_clear(
            canvas_resource_id,
            [0.0, 0.0, 0.0, 1.0],
        ));
    }
    p2.draws.push(Draw::new(
        0,
        pipeline_canvas,
        vertex_count,
        0,
        vec![ResourceUse::buffer_vertex(
            vertex_buffer_id,
            version,
            Some(0),
            None,
        )],
    ));
    graph.add_pass(p2)?;

    // 3. Copy pass (copies texture to readback buffer, isolated from render passes)
    let p3_id = PassId::new(3);
    let mut p3 = Pass::new_copy(p3_id, "bridge_readback_copy_pass");
    p3.dependencies.push(p1_id); // Copy must wait for offscreen render pass to complete
    p3.copies.push(CopyCommand::TextureToBuffer {
        texture_id: offscreen_texture_id,
        buffer_id: readback_buffer_id,
        width,
        height,
        bytes_per_row,
    });
    graph.add_pass(p3)?;

    graph.compile(canvas_tracker)
}

/// Constructs a plan demonstrating the Red-A / Blue-B per-use versioning schedule (§6.7).
///
/// Contains 4 ordered execution segments:
/// 1. Render pass A (`pass_red_target_a`): clears attachment `target_a_id` to `[0.0, 0.0, 0.0, 1.0]`
///    and draws Red with dynamic uniform offset `red_offset` at `red_version`.
/// 2. Render pass B (`pass_blue_target_b`): clears attachment `target_b_id` to `[0.0, 0.0, 0.0, 1.0]`
///    and draws Blue with dynamic uniform offset `blue_offset` at `blue_version`. Depends on pass 1.
/// 3. Copy pass A (`copy_readback_a`): copies `target_a_id` texture to readback buffer `readback_a_id`
///    (64x64, `bytes_per_row` aligned via `aligned_bytes_per_row(64)`). Depends on pass 1.
/// 4. Copy pass B (`copy_readback_b`): copies `target_b_id` texture to readback buffer `readback_b_id`
///    (64x64, `bytes_per_row` aligned via `aligned_bytes_per_row(64)`). Depends on pass 2.
///
/// Both passes coexist in the same submission schedule without in-place overwrite.
/// Enforces compilation through `PassGraph::compile`.
pub fn build_red_a_blue_b_plan(
    shared_uniform_buffer_id: ResourceId,
    target_a_id: ResourceId,
    target_b_id: ResourceId,
    readback_a_id: ResourceId,
    readback_b_id: ResourceId,
    red_version: DataVersion,
    blue_version: DataVersion,
    red_offset: u32,
    blue_offset: u32,
    pipeline_id: u32,
) -> Result<ExecutionPlan, GraphError> {
    let bytes_per_row = aligned_bytes_per_row(64).map_err(|_| GraphError::InvalidPass {
        pass_id: 3,
        reason: String::from("bytes_per_row calculation overflow or alignment violation"),
    })?;
    validate_copy_bytes_per_row(bytes_per_row).map_err(|_| GraphError::InvalidPass {
        pass_id: 3,
        reason: String::from("unaligned bytes_per_row violating COPY_BYTES_PER_ROW_ALIGNMENT"),
    })?;

    let mut graph = PassGraph::new();

    let p1_id = PassId::new(1);
    let mut p1 = Pass::new_render(p1_id, "pass_red_target_a");
    p1.color_attachments.push(ColorAttachment::new_clear(
        target_a_id,
        [0.0, 0.0, 0.0, 1.0],
    ));
    p1.draws.push(Draw::new(
        0,
        pipeline_id,
        3,
        red_offset,
        vec![ResourceUse::buffer_uniform(
            shared_uniform_buffer_id,
            red_version,
            Some(red_offset as u64),
            Some(256),
        )],
    ));
    graph.add_pass(p1)?;

    let p2_id = PassId::new(2);
    let mut p2 = Pass::new_render(p2_id, "pass_blue_target_b");
    p2.dependencies.push(p1_id);
    p2.color_attachments.push(ColorAttachment::new_clear(
        target_b_id,
        [0.0, 0.0, 0.0, 1.0],
    ));
    p2.draws.push(Draw::new(
        0,
        pipeline_id,
        3,
        blue_offset,
        vec![ResourceUse::buffer_uniform(
            shared_uniform_buffer_id,
            blue_version,
            Some(blue_offset as u64),
            Some(256),
        )],
    ));
    graph.add_pass(p2)?;

    let p3_id = PassId::new(3);
    let mut p3 = Pass::new_copy(p3_id, "copy_readback_a");
    p3.dependencies.push(p1_id);
    p3.copies.push(CopyCommand::TextureToBuffer {
        texture_id: target_a_id,
        buffer_id: readback_a_id,
        width: 64,
        height: 64,
        bytes_per_row,
    });
    graph.add_pass(p3)?;

    let p4_id = PassId::new(4);
    let mut p4 = Pass::new_copy(p4_id, "copy_readback_b");
    p4.dependencies.push(p2_id);
    p4.copies.push(CopyCommand::TextureToBuffer {
        texture_id: target_b_id,
        buffer_id: readback_b_id,
        width: 64,
        height: 64,
        bytes_per_row,
    });
    graph.add_pass(p4)?;

    graph.compile(None)
}

/// Constructs a plan demonstrating the Red-A / Blue-B render passes without readback copies.
///
/// Standardized with black clear `[0.0, 0.0, 0.0, 1.0]`.
pub fn build_red_a_blue_b_render_plan(
    shared_uniform_buffer_id: ResourceId,
    target_a_id: ResourceId,
    target_b_id: ResourceId,
    red_version: DataVersion,
    blue_version: DataVersion,
    red_offset: u32,
    blue_offset: u32,
    pipeline_id: u32,
) -> Result<ExecutionPlan, GraphError> {
    let mut graph = PassGraph::new();

    let p1_id = PassId::new(1);
    let mut p1 = Pass::new_render(p1_id, "pass_red_target_a");
    p1.color_attachments.push(ColorAttachment::new_clear(
        target_a_id,
        [0.0, 0.0, 0.0, 1.0],
    ));
    p1.draws.push(Draw::new(
        0,
        pipeline_id,
        3,
        red_offset,
        vec![ResourceUse::buffer_uniform(
            shared_uniform_buffer_id,
            red_version,
            Some(red_offset as u64),
            Some(256),
        )],
    ));
    graph.add_pass(p1)?;

    let p2_id = PassId::new(2);
    let mut p2 = Pass::new_render(p2_id, "pass_blue_target_b");
    p2.dependencies.push(p1_id);
    p2.color_attachments.push(ColorAttachment::new_clear(
        target_b_id,
        [0.0, 0.0, 0.0, 1.0],
    ));
    p2.draws.push(Draw::new(
        0,
        pipeline_id,
        3,
        blue_offset,
        vec![ResourceUse::buffer_uniform(
            shared_uniform_buffer_id,
            blue_version,
            Some(blue_offset as u64),
            Some(256),
        )],
    ));
    graph.add_pass(p2)?;

    graph.compile(None)
}
