//! Pass representations, attachment operations, draw/dispatch buckets, and copy commands.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use f3d_core::ownership::{DataVersion, Epoch};

use crate::resource::{ResourceAccess, ResourceId, ResourceKind, ResourceUse, SubresourceRange};

/// Strongly typed 32-bit handle for a pass in the pass graph.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct PassId(pub u32);

impl PassId {
    /// Construct a pass ID from a raw `u32`.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Extract the raw `u32` value.
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for PassId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PassId({})", self.0)
    }
}

impl fmt::Display for PassId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pass#{}", self.0)
    }
}

/// Category of execution pass.
///
/// Invariant: Copy commands cannot be inserted inside a render pass as though they were draws (§6.7).
/// Copies reside strictly in dedicated `Copy` passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PassKind {
    /// Encodes a WebGPU render pass with color/depth attachments and draw calls.
    Render,
    /// Encodes a WebGPU compute pass with dispatches.
    Compute,
    /// Encodes copy operations (buffer-to-buffer, texture-to-buffer, etc.).
    Copy,
}

/// Explicit load operation for an attachment (§8.5).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoadOp {
    /// Clear the attachment to the specified clear color or depth value.
    #[default]
    Clear,
    /// Preserve existing attachment contents from previous passes.
    Load,
    /// Discard or undefined prior contents.
    DontCare,
}

/// Explicit store operation for an attachment (§8.5).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StoreOp {
    /// Store the rendered results into the target texture/buffer.
    #[default]
    Store,
    /// Discard rendered results at the end of the pass (e.g. transient depth).
    Discard,
}

/// Color attachment configuration with explicit load/store/clear/resolve semantics.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorAttachment {
    /// Target texture or canvas output ID.
    pub target_id: ResourceId,
    /// Target subresource range within the texture.
    pub view_subresource: SubresourceRange,
    /// Explicit load operation.
    pub load_op: LoadOp,
    /// Explicit store operation.
    pub store_op: StoreOp,
    /// Clear color `[r, g, b, a]` when `load_op == LoadOp::Clear`.
    pub clear_color: [f32; 4],
    /// Optional resolve target for multisampled attachments.
    pub resolve_target: Option<ResourceId>,
    /// Optional captured canvas output epoch (if this attachment is a canvas swapchain).
    pub canvas_epoch: Option<Epoch>,
}

impl ColorAttachment {
    /// Construct a standard color attachment that clears to a color.
    pub fn new_clear(target_id: ResourceId, clear_color: [f32; 4]) -> Self {
        Self {
            target_id,
            view_subresource: SubresourceRange::full_texture(),
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color,
            resolve_target: None,
            canvas_epoch: None,
        }
    }

    /// Construct a standard color attachment that loads existing contents.
    pub fn new_load(target_id: ResourceId) -> Self {
        Self {
            target_id,
            view_subresource: SubresourceRange::full_texture(),
            load_op: LoadOp::Load,
            store_op: StoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            resolve_target: None,
            canvas_epoch: None,
        }
    }

    /// Construct a canvas swapchain color attachment with captured epoch.
    pub fn new_canvas(target_id: ResourceId, clear_color: [f32; 4], epoch: Epoch) -> Self {
        Self {
            target_id,
            view_subresource: SubresourceRange::full_texture(),
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color,
            resolve_target: None,
            canvas_epoch: Some(epoch),
        }
    }

    /// Produce the declared resource usage for this attachment.
    pub fn to_resource_use(&self, version: DataVersion) -> ResourceUse {
        let (kind, ver) = if let Some(epoch) = self.canvas_epoch {
            (ResourceKind::CanvasOutput, DataVersion::new(epoch.get()))
        } else {
            (ResourceKind::Texture, version)
        };
        ResourceUse {
            resource_id: self.target_id,
            kind,
            version: ver,
            subresource: self.view_subresource.clone(),
            access: ResourceAccess::ColorAttachment,
            byte_offset: None,
            byte_size: None,
            canvas_epoch: self.canvas_epoch,
        }
    }

    /// Target resource identifier.
    #[inline]
    #[must_use]
    pub const fn target_id(&self) -> ResourceId {
        self.target_id
    }

    /// Explicit load operation.
    #[inline]
    #[must_use]
    pub const fn load_op(&self) -> LoadOp {
        self.load_op
    }

    /// Explicit store operation.
    #[inline]
    #[must_use]
    pub const fn store_op(&self) -> StoreOp {
        self.store_op
    }

    /// Clear color RGBA values normalized to [0.0, 1.0].
    #[inline]
    #[must_use]
    pub const fn clear_color(&self) -> [f32; 4] {
        self.clear_color
    }

    /// Optional resolve target for multisampled attachments.
    #[inline]
    #[must_use]
    pub const fn resolve_target(&self) -> Option<ResourceId> {
        self.resolve_target
    }

    /// Optional captured canvas output epoch (Some if targeting canvas swapchain).
    #[inline]
    #[must_use]
    pub const fn canvas_epoch(&self) -> Option<Epoch> {
        self.canvas_epoch
    }

    /// Returns `true` if this attachment targets a canvas presentation surface.
    #[inline]
    #[must_use]
    pub const fn is_canvas(&self) -> bool {
        self.canvas_epoch.is_some()
    }
}

/// Depth/stencil attachment configuration with explicit operations.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthStencilAttachment {
    /// Target depth/stencil texture ID.
    pub target_id: ResourceId,
    /// Target subresource range.
    pub view_subresource: SubresourceRange,
    /// Depth load operation.
    pub depth_load_op: Option<LoadOp>,
    /// Depth store operation.
    pub depth_store_op: Option<StoreOp>,
    /// Depth clear value (typically 1.0).
    pub depth_clear_value: f32,
    /// Whether depth writes are disabled.
    pub depth_read_only: bool,
    /// Stencil load operation.
    pub stencil_load_op: Option<LoadOp>,
    /// Stencil store operation.
    pub stencil_store_op: Option<StoreOp>,
    /// Stencil clear value.
    pub stencil_clear_value: u32,
    /// Whether stencil writes are disabled.
    pub stencil_read_only: bool,
}

impl DepthStencilAttachment {
    /// Standard depth-only clearing attachment.
    pub fn new_depth_clear(target_id: ResourceId, clear_value: f32) -> Self {
        Self {
            target_id,
            view_subresource: SubresourceRange::full_texture(),
            depth_load_op: Some(LoadOp::Clear),
            depth_store_op: Some(StoreOp::Store),
            depth_clear_value: clear_value,
            depth_read_only: false,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_clear_value: 0,
            stencil_read_only: true,
        }
    }

    /// Produce the declared resource usage for this depth/stencil attachment.
    pub fn to_resource_use(&self, version: DataVersion) -> ResourceUse {
        let access = if self.depth_read_only && self.stencil_read_only {
            ResourceAccess::ReadOnlyDepthStencil
        } else {
            ResourceAccess::DepthStencilAttachment
        };
        ResourceUse {
            resource_id: self.target_id,
            kind: ResourceKind::Texture,
            version,
            subresource: self.view_subresource.clone(),
            access,
            byte_offset: None,
            byte_size: None,
            canvas_epoch: None,
        }
    }

    /// Target depth/stencil resource identifier.
    #[inline]
    #[must_use]
    pub const fn target_id(&self) -> ResourceId {
        self.target_id
    }

    /// Depth load operation.
    #[inline]
    #[must_use]
    pub const fn depth_load_op(&self) -> Option<LoadOp> {
        self.depth_load_op
    }

    /// Depth store operation.
    #[inline]
    #[must_use]
    pub const fn depth_store_op(&self) -> Option<StoreOp> {
        self.depth_store_op
    }

    /// Depth clear value (typically 1.0).
    #[inline]
    #[must_use]
    pub const fn depth_clear_value(&self) -> f32 {
        self.depth_clear_value
    }

    /// Whether depth writes are disabled.
    #[inline]
    #[must_use]
    pub const fn depth_read_only(&self) -> bool {
        self.depth_read_only
    }

    /// Stencil load operation.
    #[inline]
    #[must_use]
    pub const fn stencil_load_op(&self) -> Option<LoadOp> {
        self.stencil_load_op
    }

    /// Stencil store operation.
    #[inline]
    #[must_use]
    pub const fn stencil_store_op(&self) -> Option<StoreOp> {
        self.stencil_store_op
    }

    /// Stencil clear value.
    #[inline]
    #[must_use]
    pub const fn stencil_clear_value(&self) -> u32 {
        self.stencil_clear_value
    }

    /// Whether stencil writes are disabled.
    #[inline]
    #[must_use]
    pub const fn stencil_read_only(&self) -> bool {
        self.stencil_read_only
    }
}

/// Category of draw command within a render pass (§8.5, AGENTS.md).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DrawKind {
    /// Direct non-indexed or indexed draw command (`draw` / `drawIndexed`).
    Direct,
    /// Pre-recorded WebGPU render bundle execution (`executeBundles`).
    Bundle {
        /// Unique bundle identifier.
        bundle_id: u32,
    },
}

/// Specification of a pre-recorded WebGPU render bundle execution (§8.5, AGENTS.md).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderBundle {
    /// Unique bundle identifier.
    pub bundle_id: u32,
    /// Bound pipeline ID encapsulated within the bundle.
    pub pipeline_id: u32,
    /// Resources used by commands recorded within this bundle.
    pub uses: Vec<ResourceUse>,
}

impl RenderBundle {
    /// Construct a new render bundle specification.
    pub fn new(bundle_id: u32, pipeline_id: u32, uses: Vec<ResourceUse>) -> Self {
        Self {
            bundle_id,
            pipeline_id,
            uses,
        }
    }

    /// Convert this render bundle into an executable `Draw` command.
    pub fn to_draw(&self, draw_id: u32) -> Draw {
        Draw::new_bundle(draw_id, self.bundle_id, self.pipeline_id, self.uses.clone())
    }
}

/// Individual draw command or bundle execution within a render pass.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Draw {
    /// Unique draw index within the pass.
    pub draw_id: u32,
    /// Pipeline or shader variant ID.
    pub pipeline_id: u32,
    /// Number of vertices to draw.
    pub vertex_count: u32,
    /// Number of instances.
    pub instance_count: u32,
    /// Index of the first vertex.
    pub first_vertex: u32,
    /// Index of the first instance.
    pub first_instance: u32,
    /// Dynamic uniform buffer offset.
    pub uniform_dynamic_offset: u32,
    /// Resources used by this specific draw call.
    pub uses: Vec<ResourceUse>,
    /// Command category: direct draw or pre-recorded bundle execution.
    pub kind: DrawKind,
    /// Whether this draw assumes warm/inherited pipeline and binding state from a preceding draw.
    ///
    /// Invariant: WebGPU specification states that executing render bundles (`executeBundles`)
    /// resets all pipeline and bind group state on the render pass encoder.
    /// A direct draw following a bundle cannot assume warm state (`assumes_warm_state = true`)
    /// and must explicitly rebind (`assumes_warm_state = false`).
    pub assumes_warm_state: bool,
    /// Active viewport rectangle `[x, y, width, height]` for this draw call (§8.5).
    /// `None` indicates the draw inherits the default pass/render-target viewport.
    /// `Some([x, y, width, height])` explicitly sets the viewport, allowing explicit zero-size viewports.
    pub viewport: Option<[u32; 4]>,
    /// Active scissor rectangle `[x, y, width, height]` for this draw call (§8.5).
    /// When `scissor_test_enabled` is true, this is the active clipping rectangle.
    /// When `scissor_test_enabled` is false, this is either `None` (unspecified) or `Some([0, 0, width, height])`
    /// representing the full attachment scissor needed to clear clipping after a previous scissor draw.
    pub scissor: Option<[u32; 4]>,
    /// Whether scissor test is enabled for this draw call.
    pub scissor_test_enabled: bool,
}

impl Draw {
    /// Construct a simple non-indexed direct draw command.
    pub fn new(
        draw_id: u32,
        pipeline_id: u32,
        vertex_count: u32,
        uniform_dynamic_offset: u32,
        uses: Vec<ResourceUse>,
    ) -> Self {
        Self {
            draw_id,
            pipeline_id,
            vertex_count,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
            uniform_dynamic_offset,
            uses,
            kind: DrawKind::Direct,
            assumes_warm_state: false,
            viewport: None,
            scissor: None,
            scissor_test_enabled: false,
        }
    }

    /// Construct a pre-recorded WebGPU render bundle execution command (§8.5, AGENTS.md).
    pub fn new_bundle(
        draw_id: u32,
        bundle_id: u32,
        pipeline_id: u32,
        uses: Vec<ResourceUse>,
    ) -> Self {
        Self {
            draw_id,
            pipeline_id,
            vertex_count: 0,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
            uniform_dynamic_offset: 0,
            uses,
            kind: DrawKind::Bundle { bundle_id },
            assumes_warm_state: false,
            viewport: None,
            scissor: None,
            scissor_test_enabled: false,
        }
    }

    /// Construct a direct draw that explicitly assumes warm state (inheriting pipeline/bindings from previous draw).
    pub fn new_warm_state(
        draw_id: u32,
        pipeline_id: u32,
        vertex_count: u32,
        uniform_dynamic_offset: u32,
        uses: Vec<ResourceUse>,
    ) -> Self {
        Self {
            draw_id,
            pipeline_id,
            vertex_count,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
            uniform_dynamic_offset,
            uses,
            kind: DrawKind::Direct,
            assumes_warm_state: true,
            viewport: None,
            scissor: None,
            scissor_test_enabled: false,
        }
    }

    /// Mark whether this draw assumes warm state from prior draws.
    #[must_use]
    pub fn with_assumes_warm_state(mut self, warm: bool) -> Self {
        self.assumes_warm_state = warm;
        self
    }

    /// Sets whether rebind is required (if false, assumes warm state).
    #[must_use]
    pub fn with_rebind_required(mut self, rebind: bool) -> Self {
        self.assumes_warm_state = !rebind;
        self
    }

    /// Draw index within the pass.
    #[inline]
    #[must_use]
    pub const fn draw_id(&self) -> u32 {
        self.draw_id
    }

    /// Bound pipeline ID.
    #[inline]
    #[must_use]
    pub const fn pipeline_id(&self) -> u32 {
        self.pipeline_id
    }

    /// Number of vertices to draw.
    #[inline]
    #[must_use]
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Number of instances.
    #[inline]
    #[must_use]
    pub const fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Index of the first vertex.
    #[inline]
    #[must_use]
    pub const fn first_vertex(&self) -> u32 {
        self.first_vertex
    }

    /// Index of the first instance.
    #[inline]
    #[must_use]
    pub const fn first_instance(&self) -> u32 {
        self.first_instance
    }

    /// Dynamic uniform buffer offset.
    #[inline]
    #[must_use]
    pub const fn uniform_dynamic_offset(&self) -> u32 {
        self.uniform_dynamic_offset
    }

    /// Slice of all resources used by this draw call.
    #[inline]
    #[must_use]
    pub fn uses(&self) -> &[ResourceUse] {
        &self.uses
    }

    /// Extract the vertex buffer resource ID bound to this draw, if any.
    #[must_use]
    pub fn vertex_buffer(&self) -> Option<ResourceId> {
        self.uses
            .iter()
            .find(|u| u.access == ResourceAccess::VertexBuffer)
            .map(|u| u.resource_id)
    }

    /// Extract the raw vertex buffer identifier, returning 0 if no vertex buffer is bound.
    #[inline]
    #[must_use]
    pub fn vertex_buffer_id(&self) -> u32 {
        self.vertex_buffer().map_or(0, |id| id.get())
    }

    /// Returns `true` if this command is a pre-recorded render bundle execution.
    #[inline]
    #[must_use]
    pub const fn is_bundle(&self) -> bool {
        matches!(self.kind, DrawKind::Bundle { .. })
    }

    /// Returns `true` if this command is a direct draw.
    #[inline]
    #[must_use]
    pub const fn is_direct(&self) -> bool {
        matches!(self.kind, DrawKind::Direct)
    }

    /// Returns the bundle ID if this is a bundle execution command.
    #[inline]
    #[must_use]
    pub const fn bundle_id(&self) -> Option<u32> {
        match self.kind {
            DrawKind::Bundle { bundle_id } => Some(bundle_id),
            DrawKind::Direct => None,
        }
    }

    /// Returns the draw command category.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> DrawKind {
        self.kind
    }

    /// Returns `true` if this draw assumes warm/inherited pipeline and binding state.
    #[inline]
    #[must_use]
    pub const fn assumes_warm_state(&self) -> bool {
        self.assumes_warm_state
    }

    /// Returns `true` if this direct draw explicitly rebinds its state (does not assume warm state).
    #[inline]
    #[must_use]
    pub const fn rebind_required(&self) -> bool {
        !self.assumes_warm_state
    }

    /// Configure the active viewport rectangle for this draw call (§8.5).
    /// `None` indicates the draw inherits the default pass/render-target viewport.
    #[must_use]
    pub const fn with_viewport(mut self, viewport: Option<[u32; 4]>) -> Self {
        self.viewport = viewport;
        self
    }

    /// Configure an explicit active viewport rectangle `[x, y, width, height]` for this draw call (§8.5).
    #[must_use]
    pub const fn with_viewport_rect(mut self, viewport: [u32; 4]) -> Self {
        self.viewport = Some(viewport);
        self
    }

    /// Configure the active scissor rectangle and test enable flag (§8.5).
    #[must_use]
    pub const fn with_scissor(mut self, scissor: Option<[u32; 4]>, enabled: bool) -> Self {
        self.scissor = scissor;
        self.scissor_test_enabled = enabled;
        self
    }

    /// Configure an explicit active scissor rectangle `[x, y, width, height]` with testing enabled (§8.5).
    #[must_use]
    pub const fn with_scissor_rect(mut self, scissor: [u32; 4]) -> Self {
        self.scissor = Some(scissor);
        self.scissor_test_enabled = true;
        self
    }

    /// Configure an explicit full-attachment scissor rectangle `[0, 0, width, height]` to clear clipping (§8.5).
    #[must_use]
    pub const fn with_full_attachment_scissor(mut self, width: u32, height: u32) -> Self {
        self.scissor = Some([0, 0, width, height]);
        self.scissor_test_enabled = false;
        self
    }

    /// Configure direct draw parameters `[vertex_count, instance_count, first_vertex, first_instance]` in WebGPU order (§8.5).
    #[must_use]
    pub const fn with_range(mut self, range: [u32; 4]) -> Self {
        self.vertex_count = range[0];
        self.instance_count = range[1];
        self.first_vertex = range[2];
        self.first_instance = range[3];
        self
    }

    /// Active viewport rectangle `[x, y, width, height]` for this draw call.
    #[inline]
    #[must_use]
    pub const fn viewport(&self) -> Option<[u32; 4]> {
        self.viewport
    }

    /// Active viewport rectangle converted to WebGPU coordinates `[x, y, width, height, min_depth, max_depth]`.
    #[inline]
    #[must_use]
    pub const fn viewport_f32(&self) -> Option<[f32; 6]> {
        if let Some(vp) = self.viewport {
            Some([vp[0] as f32, vp[1] as f32, vp[2] as f32, vp[3] as f32, 0.0, 1.0])
        } else {
            None
        }
    }

    /// Active scissor rectangle `[x, y, width, height]` for this draw call.
    #[inline]
    #[must_use]
    pub const fn scissor(&self) -> Option<[u32; 4]> {
        self.scissor
    }

    /// Whether scissor test is enabled for this draw call.
    #[inline]
    #[must_use]
    pub const fn scissor_test_enabled(&self) -> bool {
        self.scissor_test_enabled
    }
}

/// Explicit group-0 WGSL binding reference on a compute dispatch (§6.1, §8.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeBufferBinding {
    /// WGSL `@binding(index)` in `@group(0)`.
    pub binding_index: u32,
    /// Direct index into `Dispatch::uses`.
    pub use_index: usize,
}

impl ComputeBufferBinding {
    /// Construct a new group-0 compute buffer binding referencing a dispatch usage index.
    #[inline]
    #[must_use]
    pub const fn new(binding_index: u32, use_index: usize) -> Self {
        Self {
            binding_index,
            use_index,
        }
    }
}

/// Individual compute dispatch within a compute pass.
///
/// Invariant: Per-dispatch usage-scope rules forbid writable binding aliases (§8.5, [S51]).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dispatch {
    /// Unique dispatch index within the pass.
    pub dispatch_id: u32,
    /// Pipeline ID.
    pub pipeline_id: u32,
    /// Workgroup dimensions `[x, y, z]`.
    pub workgroups: [u32; 3],
    /// Resources bound to this specific dispatch.
    pub uses: Vec<ResourceUse>,
    /// Explicit group-0 buffer binding map referencing indices into `uses`.
    pub bindings: Vec<ComputeBufferBinding>,
}

impl Dispatch {
    /// Construct a compute dispatch with no explicit bindings.
    pub fn new(
        dispatch_id: u32,
        pipeline_id: u32,
        workgroups: [u32; 3],
        uses: Vec<ResourceUse>,
    ) -> Self {
        Self {
            dispatch_id,
            pipeline_id,
            workgroups,
            uses,
            bindings: Vec::new(),
        }
    }

    /// Construct a compute dispatch with explicit group-0 buffer bindings.
    pub fn new_with_bindings(
        dispatch_id: u32,
        pipeline_id: u32,
        workgroups: [u32; 3],
        uses: Vec<ResourceUse>,
        bindings: Vec<ComputeBufferBinding>,
    ) -> Self {
        Self {
            dispatch_id,
            pipeline_id,
            workgroups,
            uses,
            bindings,
        }
    }

    /// Builder method to attach explicit group-0 buffer bindings.
    #[must_use]
    pub fn with_bindings(mut self, bindings: Vec<ComputeBufferBinding>) -> Self {
        self.bindings = bindings;
        self
    }
}

/// Copy commands strictly isolated from render passes (§6.7, §8.5).
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CopyCommand {
    /// Copy bytes from buffer to buffer.
    BufferToBuffer {
        /// Source buffer ID.
        src: ResourceId,
        /// Source byte offset.
        src_offset: u64,
        /// Destination buffer ID.
        dst: ResourceId,
        /// Destination byte offset.
        dst_offset: u64,
        /// Byte size to copy.
        size: u64,
    },
    /// Copy subresource from texture to texture.
    TextureToTexture {
        /// Source texture ID.
        src: ResourceId,
        /// Source mip level.
        src_mip: u32,
        /// Source array layer.
        src_layer: u32,
        /// Destination texture ID.
        dst: ResourceId,
        /// Destination mip level.
        dst_mip: u32,
        /// Destination array layer.
        dst_layer: u32,
        /// Pixel extent `[width, height, depth]`.
        extent: [u32; 3],
    },
    /// Readback or copy from texture to buffer (e.g. ChartreuseFern's bridge case).
    TextureToBuffer {
        /// Source texture ID.
        texture_id: ResourceId,
        /// Destination readback buffer ID.
        buffer_id: ResourceId,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Bytes per row alignment padding.
        bytes_per_row: u32,
    },
    /// Upload from staging buffer to texture.
    BufferToTexture {
        /// Source buffer ID.
        buffer_id: ResourceId,
        /// Destination texture ID.
        texture_id: ResourceId,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Bytes per row alignment padding.
        bytes_per_row: u32,
    },
}

impl CopyCommand {
    /// Extract the resources used by this copy command.
    pub fn all_uses(&self, version: DataVersion) -> (ResourceUse, ResourceUse) {
        match self {
            Self::BufferToBuffer { src, src_offset, dst, dst_offset, size } => (
                ResourceUse {
                    resource_id: *src,
                    kind: ResourceKind::Buffer,
                    version,
                    subresource: SubresourceRange::WholeBuffer,
                    access: ResourceAccess::CopySrc,
                    byte_offset: Some(*src_offset),
                    byte_size: Some(*size),
                    canvas_epoch: None,
                },
                ResourceUse {
                    resource_id: *dst,
                    kind: ResourceKind::Buffer,
                    version,
                    subresource: SubresourceRange::WholeBuffer,
                    access: ResourceAccess::CopyDst,
                    byte_offset: Some(*dst_offset),
                    byte_size: Some(*size),
                    canvas_epoch: None,
                },
            ),
            Self::TextureToTexture { src, src_mip, src_layer, dst, dst_mip, dst_layer, .. } => (
                ResourceUse {
                    resource_id: *src,
                    kind: ResourceKind::Texture,
                    version,
                    subresource: SubresourceRange::single_mip_layer(*src_mip, *src_layer, crate::resource::TextureAspect::All),
                    access: ResourceAccess::CopySrc,
                    byte_offset: None,
                    byte_size: None,
                    canvas_epoch: None,
                },
                ResourceUse {
                    resource_id: *dst,
                    kind: ResourceKind::Texture,
                    version,
                    subresource: SubresourceRange::single_mip_layer(*dst_mip, *dst_layer, crate::resource::TextureAspect::All),
                    access: ResourceAccess::CopyDst,
                    byte_offset: None,
                    byte_size: None,
                    canvas_epoch: None,
                },
            ),
            Self::TextureToBuffer { texture_id, buffer_id, .. } => (
                ResourceUse {
                    resource_id: *texture_id,
                    kind: ResourceKind::Texture,
                    version,
                    subresource: SubresourceRange::full_texture(),
                    access: ResourceAccess::CopySrc,
                    byte_offset: None,
                    byte_size: None,
                    canvas_epoch: None,
                },
                ResourceUse {
                    resource_id: *buffer_id,
                    kind: ResourceKind::Buffer,
                    version,
                    subresource: SubresourceRange::WholeBuffer,
                    access: ResourceAccess::CopyDst,
                    byte_offset: None,
                    byte_size: None,
                    canvas_epoch: None,
                },
            ),
            Self::BufferToTexture { buffer_id, texture_id, .. } => (
                ResourceUse {
                    resource_id: *buffer_id,
                    kind: ResourceKind::Buffer,
                    version,
                    subresource: SubresourceRange::WholeBuffer,
                    access: ResourceAccess::CopySrc,
                    byte_offset: None,
                    byte_size: None,
                    canvas_epoch: None,
                },
                ResourceUse {
                    resource_id: *texture_id,
                    kind: ResourceKind::Texture,
                    version,
                    subresource: SubresourceRange::full_texture(),
                    access: ResourceAccess::CopyDst,
                    byte_offset: None,
                    byte_size: None,
                    canvas_epoch: None,
                },
            ),
        }
    }

    /// Bytes per row alignment padding, if applicable to this copy command.
    #[must_use]
    pub const fn bytes_per_row(&self) -> Option<u32> {
        match *self {
            Self::TextureToBuffer { bytes_per_row, .. }
            | Self::BufferToTexture { bytes_per_row, .. } => Some(bytes_per_row),
            Self::BufferToBuffer { .. } | Self::TextureToTexture { .. } => None,
        }
    }

    /// Returns `true` if this command copies a texture to a staging readback buffer.
    #[inline]
    #[must_use]
    pub const fn is_texture_to_buffer(&self) -> bool {
        matches!(self, Self::TextureToBuffer { .. })
    }

    /// Deconstructs a `TextureToBuffer` command into `(texture_id, buffer_id, width, height, bytes_per_row)`.
    #[must_use]
    pub const fn as_texture_to_buffer(&self) -> Option<(ResourceId, ResourceId, u32, u32, u32)> {
        match *self {
            Self::TextureToBuffer {
                texture_id,
                buffer_id,
                width,
                height,
                bytes_per_row,
            } => Some((texture_id, buffer_id, width, height, bytes_per_row)),
            _ => None,
        }
    }
}

/// A fully specified execution pass in the pass graph.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pass {
    /// Unique pass ID.
    pub id: PassId,
    /// Diagnostic name.
    pub name: String,
    /// Execution category.
    pub kind: PassKind,
    /// Color attachments (for Render passes).
    pub color_attachments: Vec<ColorAttachment>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
    /// Draw buckets (for Render passes).
    pub draws: Vec<Draw>,
    /// Dispatches (for Compute passes).
    pub dispatches: Vec<Dispatch>,
    /// Copy operations (for Copy passes).
    pub copies: Vec<CopyCommand>,
    /// Explicit order constraints (passes that must finish before this pass runs).
    pub dependencies: Vec<PassId>,
    /// Previous-frame history references.
    pub history_dependencies: Vec<ResourceId>,
}

impl Pass {
    /// Create a new Render pass.
    pub fn new_render(id: PassId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            kind: PassKind::Render,
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            draws: Vec::new(),
            dispatches: Vec::new(),
            copies: Vec::new(),
            dependencies: Vec::new(),
            history_dependencies: Vec::new(),
        }
    }

    /// Create a new Compute pass.
    pub fn new_compute(id: PassId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            kind: PassKind::Compute,
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            draws: Vec::new(),
            dispatches: Vec::new(),
            copies: Vec::new(),
            dependencies: Vec::new(),
            history_dependencies: Vec::new(),
        }
    }

    /// Create a new Copy pass.
    pub fn new_copy(id: PassId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            kind: PassKind::Copy,
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            draws: Vec::new(),
            dispatches: Vec::new(),
            copies: Vec::new(),
            dependencies: Vec::new(),
            history_dependencies: Vec::new(),
        }
    }

    /// Add a color attachment.
    pub fn with_color_attachment(mut self, attachment: ColorAttachment) -> Self {
        self.color_attachments.push(attachment);
        self
    }

    /// Add a depth/stencil attachment.
    pub fn with_depth_stencil_attachment(mut self, attachment: DepthStencilAttachment) -> Self {
        self.depth_stencil_attachment = Some(attachment);
        self
    }

    /// Add a draw call.
    pub fn with_draw(mut self, draw: Draw) -> Self {
        self.draws.push(draw);
        self
    }

    /// Add a pre-recorded render bundle execution to this pass (§8.5, AGENTS.md).
    pub fn with_bundle(mut self, bundle: RenderBundle) -> Self {
        let draw_id = self.draws.len() as u32;
        self.draws.push(bundle.to_draw(draw_id));
        self
    }

    /// Add a pre-recorded render bundle execution directly with parameters (§8.5, AGENTS.md).
    pub fn with_bundle_draw(
        mut self,
        bundle_id: u32,
        pipeline_id: u32,
        uses: Vec<ResourceUse>,
    ) -> Self {
        let draw_id = self.draws.len() as u32;
        self.draws.push(Draw::new_bundle(draw_id, bundle_id, pipeline_id, uses));
        self
    }

    /// Add a compute dispatch.
    pub fn with_dispatch(mut self, dispatch: Dispatch) -> Self {
        self.dispatches.push(dispatch);
        self
    }

    /// Add a copy command.
    pub fn with_copy(mut self, copy: CopyCommand) -> Self {
        self.copies.push(copy);
        self
    }

    /// Add an explicit dependency on another pass.
    pub fn with_dependency(mut self, dep: PassId) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Collect all resource uses declared across all commands and attachments in this pass.
    pub fn all_uses(&self) -> Vec<ResourceUse> {
        let mut uses = Vec::new();
        // Color attachments write at default version
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
}
