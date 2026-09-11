//! Logical render-context stack, nested scratch isolation, and source-ordered frame protocol.
//!
//! # Architecture & Guarantees (§6.3, §6.7, §5.5, §8.2, 2v8.4)
//!
//! - **Source-Ordered Execution Protocol (§6.3)**: There is no global application tick.
//!   Rendering follows the exact call and effect order of the application. A browser presentation
//!   frame may contain zero, one, or multiple logical render passes.
//! - **Reentrant Logical Render Contexts & Active-Pass Nesting (§6.7)**: When nested rendering occurs
//!   (e.g. shadow maps, environment maps, reflection passes, material `onBeforeRender` callbacks),
//!   an active outer pass cannot execute simultaneously on WebGPU encoders. The frame protocol:
//!   1. Splits the active outer pass into an "outer prefix" pass, committing in-progress draws and
//!      setting `StoreOp::Store`.
//!   2. Pushes the nested context onto [`RenderContextStack`].
//!   3. Executes the nested pass(es) targeting the sub-renderer's target.
//!   4. Pops the nested context and resumes the outer pass on the outer target with [`LoadOp::Load`],
//!      preserving earlier draws without re-clearing the framebuffer.
//! - **Unwinding & Panic Restoration Guard**: Both [`RenderContextStack::with_nested_context`] and
//!   [`FrameSession::with_nested_render`] employ safe RAII guards whose `Drop` implementations
//!   unconditionally restore the exact entry depth, entry context, and scratch watermark, even if the
//!   callback panics, returns an error, or unbalances the stack via manual pushes/pops.
//! - **CPU Metadata vs. GPU Command State (§5.1, NO-CLAIM)**:
//!   - *Saved CPU Metadata*: `camera_projection` (double-precision `[f64; 16]`), `camera_epoch`, `viewport`,
//!     `scissor`, `scissor_test_enabled`, and `scratch_watermark` preserve application-level camera and rect state.
//!     Camera projection is maintained as `f64` until the admitted GPU upload boundary.
//!   - *GPU Command State*: `target_id`, `is_canvas`, and pass `load_op` / `clear_color` directly govern
//!     WebGPU render pass encoder operations lowered into [`GpuCommand::RenderPass`].
//! - **Scratch Isolation & Ownership Validation (§5.5, §8.2)**: Dynamic uniform parameter slices are
//!   recorded into [`f3d_core::ownership::PerUseByteBuffer`]. Calling [`FrameSession::record_direct_draw`]
//!   strictly validates that the supplied [`UseRecord`] belongs to the active session buffer via
//!   [`PerUseByteBuffer::get_slice`], preventing foreign, stale, or forged records.
//! - **Callback Multiplicity (§5.5)**: When an application callback or material update mutates parameters
//!   between draws in a single submission, two distinct slices with distinct dynamic offsets are allocated
//!   rather than overwriting existing slice data.
//! - **Callable Interface for Bridge Lowering**: Exposes [`FrameSession::build_submission_packet`],
//!   which compiles the accumulated pass graph, lowers it via [`crate::gpu_host::lower_plan`],
//!   and emits a verified [`crate::gpu_host::GpuSubmissionPacket`].
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use f3d_core::{
    handle::{Handle, MaterialDomain},
    ownership::{DataVersion, Epoch, OwnershipError, PerUseByteBuffer, UseRecord},
};
use f3d_graph::{
    error::{CanvasError, GraphError},
    pass::{ColorAttachment, Draw, LoadOp, Pass, PassId, StoreOp},
    resource::{ResourceAccess, ResourceId, ResourceUse},
    schedule::PassGraph,
    CanvasEpochTracker, CanvasFormat, CanvasId,
};

use crate::gpu_host::{
    lower_plan, GpuCommand, GpuSubmissionPacket, PlanLoweringError,
    LOAD_OP_CLEAR, LOAD_OP_LOAD, PASS_FLAG_NEW_PASS, PASS_FLAG_NONE,
    STORE_OP_STORE, TARGET_CANVAS, TARGET_OFFSCREEN,
};

/// Default maximum allowable depth for nested render contexts to prevent unbounded recursion.
pub const DEFAULT_MAX_CONTEXT_DEPTH: usize = 16;

/// Default uniform buffer binding identifier for material parameters in bridge submission.
pub const DEFAULT_MATERIAL_UNIFORM_BUFFER_ID: u32 = 1;

/// Standard identity 4x4 matrix in column-major order (double precision).
pub const IDENTITY_4X4: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Encapsulates the logical render state at a given nesting level (§6.7).
///
/// # CPU Metadata vs GPU Command State (§5.1, NO-CLAIM)
/// - **Saved CPU Metadata**: `camera_projection`, `camera_epoch`, `viewport`, `scissor`, and
///   `scissor_test_enabled` represent application-visible state saved on the context stack.
///   Three.js maintains camera matrices as double-precision (`f64`) values; `camera_projection`
///   is stored as `[f64; 16]` to preserve precision until the admitted upload/quantization boundary.
/// - **GPU Command State**: `target_id`, `is_canvas`, and the pass `clear_color` and `load_op`
///   directly govern WebGPU render pass encoder operations lowered into [`GpuCommand::RenderPass`].
#[derive(Clone, Debug, PartialEq)]
pub struct RenderContext {
    /// Render target resource ID (texture or swapchain).
    pub target_id: ResourceId,
    /// Whether this context targets the active browser canvas swapchain.
    pub is_canvas: bool,
    /// Target attachment width in pixels (§8.5).
    pub target_width: u32,
    /// Target attachment height in pixels (§8.5).
    pub target_height: u32,
    /// Active viewport rectangle `[x, y, width, height]` (CPU metadata).
    /// `None` indicates the default full pass attachment bounds.
    pub viewport: Option<[u32; 4]>,
    /// Whether an explicit reset of the viewport to full attachment was requested (§8.5).
    pub viewport_reset: bool,
    /// Active scissor rectangle `[x, y, width, height]` (CPU metadata).
    /// `None` indicates scissor testing is disabled (full attachment).
    pub scissor: Option<[u32; 4]>,
    /// Whether scissor clipping is active (CPU metadata).
    pub scissor_test_enabled: bool,
    /// Semantic epoch of the camera matrix/state for this context (CPU metadata).
    pub camera_epoch: Epoch,
    /// Semantic epoch of the actively acquired canvas swapchain output texture (§8.5, [S48]).
    ///
    /// This is strictly separated from `camera_epoch`: camera matrices can be updated without
    /// invalidating the canvas swapchain, and canvas interval acquisition advances monotonically
    /// per presentation frame independent of camera state.
    pub canvas_output_epoch: Option<Epoch>,
    /// Observable camera projection matrix (16 floats, column-major, double precision).
    pub camera_projection: [f64; 16],
    /// High-water mark in the parent scratch buffer when this context was activated.
    pub scratch_watermark: usize,
}

impl RenderContext {
    /// Create a new render context targeting the canvas swapchain without active interval tracking.
    #[must_use]
    pub fn new_canvas(target_id: ResourceId, width: u32, height: u32, camera_epoch: Epoch) -> Self {
        Self {
            target_id,
            is_canvas: true,
            target_width: width,
            target_height: height,
            viewport: None,
            viewport_reset: false,
            scissor: None,
            scissor_test_enabled: false,
            camera_epoch,
            canvas_output_epoch: None,
            camera_projection: IDENTITY_4X4,
            scratch_watermark: 0,
        }
    }

    /// Create a new render context targeting the canvas swapchain with an actively acquired canvas output epoch (§8.5).
    #[must_use]
    pub fn new_canvas_acquired(
        target_id: ResourceId,
        width: u32,
        height: u32,
        camera_epoch: Epoch,
        canvas_output_epoch: Epoch,
    ) -> Self {
        Self {
            target_id,
            is_canvas: true,
            target_width: width,
            target_height: height,
            viewport: None,
            viewport_reset: false,
            scissor: None,
            scissor_test_enabled: false,
            camera_epoch,
            canvas_output_epoch: Some(canvas_output_epoch),
            camera_projection: IDENTITY_4X4,
            scratch_watermark: 0,
        }
    }

    /// Configure the acquired canvas output epoch.
    #[must_use]
    pub const fn with_canvas_output_epoch(mut self, epoch: Epoch) -> Self {
        self.canvas_output_epoch = Some(epoch);
        self
    }

    /// Create a new render context targeting an offscreen texture (e.g. shadow map, reflection target).
    #[must_use]
    pub fn new_offscreen(target_id: ResourceId, width: u32, height: u32, camera_epoch: Epoch) -> Self {
        Self {
            target_id,
            is_canvas: false,
            target_width: width,
            target_height: height,
            viewport: None,
            viewport_reset: false,
            scissor: None,
            scissor_test_enabled: false,
            camera_epoch,
            canvas_output_epoch: None,
            camera_projection: IDENTITY_4X4,
            scratch_watermark: 0,
        }
    }

    /// Configure an explicit active viewport rectangle `[x, y, width, height]`.
    #[must_use]
    pub const fn with_viewport(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.viewport = Some([x, y, width, height]);
        self.viewport_reset = false;
        self
    }

    /// Configure the active viewport rectangle (or `None` for default full pass).
    #[must_use]
    pub const fn with_optional_viewport(mut self, viewport: Option<[u32; 4]>) -> Self {
        self.viewport = viewport;
        self.viewport_reset = false;
        self
    }

    /// Configure an explicit active scissor rectangle `[x, y, width, height]`.
    #[must_use]
    pub const fn with_scissor(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.scissor = Some([x, y, width, height]);
        self
    }

    /// Configure the active scissor rectangle (or `None` for disabled/full).
    #[must_use]
    pub const fn with_optional_scissor(mut self, scissor: Option<[u32; 4]>) -> Self {
        self.scissor = scissor;
        self
    }

    /// Configure whether scissor testing is enabled.
    #[must_use]
    pub const fn with_scissor_test(mut self, enabled: bool) -> Self {
        self.scissor_test_enabled = enabled;
        self
    }

    /// Configure the camera projection matrix in double precision.
    #[must_use]
    pub const fn with_camera_projection(mut self, projection: [f64; 16]) -> Self {
        self.camera_projection = projection;
        self
    }

    /// Configure the scratch watermark.
    #[must_use]
    pub const fn with_scratch_watermark(mut self, watermark: usize) -> Self {
        self.scratch_watermark = watermark;
        self
    }
}

/// Logical stack of render contexts supporting bounded reentrancy and guaranteed restoration (§6.7).
#[derive(Debug)]
pub struct RenderContextStack {
    current: RenderContext,
    stack: Vec<RenderContext>,
    max_depth: usize,
}

impl RenderContextStack {
    /// Initialize the stack with a root render context and default maximum depth limit.
    #[must_use]
    pub fn new(root: RenderContext) -> Self {
        Self::with_max_depth(root, DEFAULT_MAX_CONTEXT_DEPTH)
    }

    /// Initialize the stack with an explicit maximum recursion depth limit.
    #[must_use]
    pub fn with_max_depth(root: RenderContext, max_depth: usize) -> Self {
        Self {
            current: root,
            stack: Vec::new(),
            max_depth: max_depth.max(1),
        }
    }

    /// Reference to the currently active render context.
    #[inline]
    #[must_use]
    pub const fn current(&self) -> &RenderContext {
        &self.current
    }

    /// Mutable reference to the currently active render context.
    #[inline]
    pub fn current_mut(&mut self) -> &mut RenderContext {
        &mut self.current
    }

    /// Current nesting depth (0 = root level).
    #[inline]
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Returns `true` if currently at the root context level (not nested).
    #[inline]
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.stack.is_empty()
    }

    /// Maximum nesting depth limit configured for this stack.
    #[inline]
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Configure the maximum recursion depth limit.
    pub fn set_max_depth(&mut self, max_depth: usize) {
        self.max_depth = max_depth.max(1);
    }

    /// Push a nested render context onto the stack, preserving the outer context.
    ///
    /// # Errors
    /// Returns [`FrameError::MaxNestingDepthExceeded`] if the stack depth exceeds the configured limit.
    pub fn push(&mut self, nested: RenderContext) -> Result<(), FrameError> {
        if self.stack.len() >= self.max_depth {
            return Err(FrameError::MaxNestingDepthExceeded {
                depth: self.stack.len(),
                max: self.max_depth,
            });
        }
        let outer = core::mem::replace(&mut self.current, nested);
        self.stack.push(outer);
        Ok(())
    }

    /// Pop the active nested render context and restore the previous outer context.
    ///
    /// # Errors
    /// Returns [`FrameError::ContextStackUnderflow`] if attempting to pop the root context.
    pub fn pop(&mut self) -> Result<RenderContext, FrameError> {
        let prev = self.stack.pop().ok_or(FrameError::ContextStackUnderflow)?;
        let finished = core::mem::replace(&mut self.current, prev);
        Ok(finished)
    }

    /// Execute a closure within a nested render context with guaranteed RAII restoration.
    ///
    /// Uses [`ContextStackGuard`] to guarantee that upon return—whether normal, error, or panic—
    /// the stack depth and active context are restored to their exact entry values, even if the
    /// callback pushed or popped extra frames.
    pub fn with_nested_context<R, F>(&mut self, nested: RenderContext, f: F) -> Result<R, FrameError>
    where
        F: FnOnce(&mut RenderContextStack) -> Result<R, FrameError>,
    {
        let mut guard = ContextStackGuard::new(self, nested)?;
        f(guard.stack)
    }
}

/// RAII guard that restores the exact context stack entry state on drop (both normal exit and unwind).
struct ContextStackGuard<'a> {
    stack: &'a mut RenderContextStack,
    entry_context: RenderContext,
    saved_stack: Option<Vec<RenderContext>>,
}

impl<'a> ContextStackGuard<'a> {
    fn new(stack: &'a mut RenderContextStack, nested: RenderContext) -> Result<Self, FrameError> {
        if stack.stack.len() >= stack.max_depth {
            return Err(FrameError::MaxNestingDepthExceeded {
                depth: stack.stack.len(),
                max: stack.max_depth,
            });
        }
        let saved_stack = Some(stack.stack.clone());
        let entry_context = stack.current.clone();
        stack.stack.push(entry_context.clone());
        stack.current = nested;
        Ok(Self {
            stack,
            entry_context,
            saved_stack,
        })
    }
}

impl Drop for ContextStackGuard<'_> {
    fn drop(&mut self) {
        // Restore entry stack and entry context unconditionally and idempotently
        if let Some(saved) = self.saved_stack.take() {
            self.stack.stack = saved;
            self.stack.current = self.entry_context.clone();
        }
    }
}

/// Scratch memory arena for dynamic uniform parameters and scratch buffers with high-water mark isolation.
#[derive(Debug)]
pub struct FrameScratch {
    byte_buffer: PerUseByteBuffer<MaterialDomain>,
    scratch_bytes: Vec<u8>,
    alignment: u64,
}

impl FrameScratch {
    /// Construct a scratch arena with the specified uniform buffer offset alignment (e.g. 256).
    pub fn new(alignment: u64) -> Result<Self, FrameError> {
        let byte_buffer = PerUseByteBuffer::new(alignment)?;
        Ok(Self {
            byte_buffer,
            scratch_bytes: Vec::new(),
            alignment,
        })
    }

    /// Offset alignment requirement configured for uniform buffer slices.
    #[inline]
    #[must_use]
    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Append an immutable material parameter slice to the authoritative byte buffer arena.
    pub fn append_material_slice(
        &mut self,
        handle: Handle<MaterialDomain>,
        version: DataVersion,
        epoch: Epoch,
        bytes: &[u8],
    ) -> Result<UseRecord<MaterialDomain>, FrameError> {
        let record = self.byte_buffer.append_slice(handle, version, epoch, bytes)?;
        Ok(record)
    }

    /// Allocate raw scratch bytes and return the start byte offset.
    pub fn allocate_scratch(&mut self, bytes: &[u8]) -> usize {
        let start = self.scratch_bytes.len();
        self.scratch_bytes.extend_from_slice(bytes);
        start
    }

    /// Current length of the scratch staging buffer in bytes.
    #[inline]
    #[must_use]
    pub fn scratch_len(&self) -> usize {
        self.scratch_bytes.len()
    }

    /// Read-only slice of the scratch staging buffer.
    #[inline]
    #[must_use]
    pub fn scratch_bytes(&self) -> &[u8] {
        &self.scratch_bytes
    }

    /// Truncate scratch staging bytes back to a previous watermark, discarding nested scratch data.
    pub fn truncate_scratch(&mut self, watermark: usize) {
        if watermark < self.scratch_bytes.len() {
            self.scratch_bytes.truncate(watermark);
        }
    }

    /// Reference to the underlying material uniform byte buffer.
    #[inline]
    #[must_use]
    pub const fn byte_buffer(&self) -> &PerUseByteBuffer<MaterialDomain> {
        &self.byte_buffer
    }

    /// Mutable reference to the underlying material uniform byte buffer.
    #[inline]
    pub fn byte_buffer_mut(&mut self) -> &mut PerUseByteBuffer<MaterialDomain> {
        &mut self.byte_buffer
    }
}

/// In-progress pass being assembled within the current render context.
#[derive(Clone, Debug, PartialEq)]
struct ActivePass {
    id: PassId,
    name: String,
    color_attachment: ColorAttachment,
    draws: Vec<Draw>,
    dependency: Option<PassId>,
    scissor_used: bool,
}

/// Saved outer pass descriptor for reentrant pass resumption (§6.7).
#[derive(Clone, Debug, PartialEq)]
struct OuterResumeState {
    target_id: ResourceId,
    is_canvas: bool,
    target_width: u32,
    target_height: u32,
    viewport: Option<[u32; 4]>,
    viewport_reset: bool,
    scissor: Option<[u32; 4]>,
    scissor_test_enabled: bool,
    scissor_used: bool,
    camera_epoch: Epoch,
    canvas_output_epoch: Option<Epoch>,
    name: String,
    prefix_pass_id: PassId,
}

/// Orchestrates the source-ordered execution of logical frame operations, passes, and context switches.
#[derive(Debug)]
pub struct FrameSession {
    context_stack: RenderContextStack,
    scratch: FrameScratch,
    pass_graph: PassGraph,
    active_pass: Option<ActivePass>,
    uniform_buffer_id: u32,
    next_pass_id: u32,
    next_draw_id: u32,
    last_completed_pass_id: Option<PassId>,
    canvas_tracker: Option<CanvasEpochTracker>,
}

impl FrameSession {
    /// Initialize a new frame session with a root render context and uniform alignment (e.g. 256).
    pub fn new(root_context: RenderContext, uniform_alignment: u64) -> Result<Self, FrameError> {
        let context_stack = RenderContextStack::new(root_context);
        let scratch = FrameScratch::new(uniform_alignment)?;
        let pass_graph = PassGraph::new();
        Ok(Self {
            context_stack,
            scratch,
            pass_graph,
            active_pass: None,
            uniform_buffer_id: DEFAULT_MATERIAL_UNIFORM_BUFFER_ID,
            next_pass_id: 1,
            next_draw_id: 1,
            last_completed_pass_id: None,
            canvas_tracker: None,
        })
    }

    /// Associate an optional [`CanvasEpochTracker`] with this session for canvas interval freshness validation.
    #[must_use]
    pub fn with_canvas_tracker(mut self, tracker: CanvasEpochTracker) -> Self {
        self.canvas_tracker = Some(tracker);
        self
    }

    /// Set or replace the session's active [`CanvasEpochTracker`].
    pub fn set_canvas_tracker(&mut self, tracker: CanvasEpochTracker) {
        self.canvas_tracker = Some(tracker);
    }

    /// Reference to the session's configured [`CanvasEpochTracker`], if any.
    #[must_use]
    pub const fn canvas_tracker(&self) -> Option<&CanvasEpochTracker> {
        self.canvas_tracker.as_ref()
    }

    /// Mutable reference to the session's configured [`CanvasEpochTracker`], if any.
    #[must_use]
    pub fn canvas_tracker_mut(&mut self) -> Option<&mut CanvasEpochTracker> {
        self.canvas_tracker.as_mut()
    }

    /// Ends the host's canvas acquisition interval on the stored tracker.
    ///
    /// The host calls this when the acquired texture expires. Neither packet
    /// compilation nor `device.queue.submit` expires it: several packets may
    /// render to the same canvas within one synchronous rendering interval.
    pub fn end_canvas_interval(
        &mut self,
        canvas_id: CanvasId,
        epoch: Epoch,
    ) -> Result<(), FrameError> {
        let tracker = self.canvas_tracker.as_mut().ok_or_else(|| {
            FrameError::Graph(GraphError::Canvas(CanvasError::CanvasNotAcquired {
                canvas_id: canvas_id.get(),
            }))
        })?;
        tracker
            .end_frame_interval(canvas_id, epoch)
            .map_err(|err| FrameError::Graph(GraphError::Canvas(err)))
    }

    /// Configure the uniform buffer resource ID used for material parameter uploads.
    #[must_use]
    pub const fn with_uniform_buffer_id(mut self, buffer_id: u32) -> Self {
        self.uniform_buffer_id = buffer_id;
        self
    }

    /// Reference to the render context stack.
    #[inline]
    #[must_use]
    pub const fn context_stack(&self) -> &RenderContextStack {
        &self.context_stack
    }

    /// Mutable reference to the render context stack.
    #[inline]
    pub fn context_stack_mut(&mut self) -> &mut RenderContextStack {
        &mut self.context_stack
    }

    /// Configure the active viewport rectangle `[x, y, width, height]` on the current context (§8.5).
    pub fn set_viewport(&mut self, x: u32, y: u32, width: u32, height: u32) {
        let ctx = self.context_stack.current_mut();
        ctx.viewport = Some([x, y, width, height]);
        ctx.viewport_reset = false;
    }

    /// Reset the viewport to the default full-pass attachment bounds (§8.5).
    pub fn reset_viewport(&mut self) {
        let ctx = self.context_stack.current_mut();
        ctx.viewport = None;
        ctx.viewport_reset = true;
    }

    /// Configure the active scissor rectangle `[x, y, width, height]` on the current context (§8.5).
    pub fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.context_stack.current_mut().scissor = Some([x, y, width, height]);
        if let Some(active) = &mut self.active_pass {
            active.scissor_used = true;
        }
    }

    /// Reset the scissor rectangle to unspecified (§8.5).
    pub fn reset_scissor(&mut self) {
        self.context_stack.current_mut().scissor = None;
        if let Some(active) = &mut self.active_pass {
            active.scissor_used = true;
        }
    }

    /// Enable or disable scissor testing on the current context (§8.5).
    pub fn set_scissor_test(&mut self, enabled: bool) {
        self.context_stack.current_mut().scissor_test_enabled = enabled;
        if enabled {
            if let Some(active) = &mut self.active_pass {
                active.scissor_used = true;
            }
        }
    }

    /// Active viewport rectangle `[x, y, width, height]` of the current render context, or `None` if default.
    #[inline]
    #[must_use]
    pub fn viewport(&self) -> Option<[u32; 4]> {
        self.context_stack.current().viewport
    }

    /// Active scissor rectangle `[x, y, width, height]` of the current render context, or `None` if disabled.
    #[inline]
    #[must_use]
    pub fn scissor(&self) -> Option<[u32; 4]> {
        self.context_stack.current().scissor
    }

    /// Whether scissor test is enabled on the current render context.
    #[inline]
    #[must_use]
    pub fn scissor_test_enabled(&self) -> bool {
        self.context_stack.current().scissor_test_enabled
    }

    /// Reference to the frame scratch arena.
    #[inline]
    #[must_use]
    pub const fn scratch(&self) -> &FrameScratch {
        &self.scratch
    }

    /// Mutable reference to the frame scratch arena.
    #[inline]
    pub fn scratch_mut(&mut self) -> &mut FrameScratch {
        &mut self.scratch
    }

    /// Reference to the accumulated pass graph.
    #[inline]
    #[must_use]
    pub const fn pass_graph(&self) -> &PassGraph {
        &self.pass_graph
    }

    /// Mutable reference to the accumulated pass graph.
    #[inline]
    pub fn pass_graph_mut(&mut self) -> &mut PassGraph {
        &mut self.pass_graph
    }

    /// Current nesting depth of the render context.
    #[inline]
    #[must_use]
    pub fn depth(&self) -> usize {
        self.context_stack.depth()
    }

    /// Begin a new render pass targeting the current context's render target.
    ///
    /// Constructs a [`ColorAttachment`] configured for canvas or offscreen rendering based on the
    /// active context state.
    ///
    /// # Errors
    /// Returns [`FrameError::PassAlreadyActive`] if another pass has not been closed via [`Self::end_render_pass`].
    pub fn begin_render_pass(&mut self, name: &str, clear_color: [f32; 4]) -> Result<PassId, FrameError> {
        if let Some(active) = &self.active_pass {
            return Err(FrameError::PassAlreadyActive {
                pass_id: active.id.get(),
            });
        }

        let pass_id = PassId::new(self.next_pass_id);
        self.next_pass_id = self
            .next_pass_id
            .checked_add(1)
            .ok_or(FrameError::Graph(GraphError::PassIdOverflow))?;

        let ctx = self.context_stack.current();
        let color_attachment = match (ctx.is_canvas, ctx.canvas_output_epoch) {
            (true, Some(epoch)) => ColorAttachment::new_canvas(ctx.target_id, clear_color, epoch),
            _ => ColorAttachment::new_clear(ctx.target_id, clear_color),
        };

        let dependency = self.last_completed_pass_id;
        let scissor_used = ctx.scissor_test_enabled || ctx.scissor.is_some();

        self.active_pass = Some(ActivePass {
            id: pass_id,
            name: name.to_string(),
            color_attachment,
            draws: Vec::new(),
            dependency,
            scissor_used,
        });

        Ok(pass_id)
    }

    /// Snapshot a material parameter block into the immutable per-use byte buffer (§5.5, §8.2).
    ///
    /// Allocates an aligned slice with an immutable [`UseRecord`]. If an intervening callback
    /// mutates parameters between draws, a new distinct slice with distinct dynamic offset is returned.
    pub fn snapshot_material_use(
        &mut self,
        handle: Handle<MaterialDomain>,
        version: DataVersion,
        epoch: Epoch,
        bytes: &[u8],
    ) -> Result<UseRecord<MaterialDomain>, FrameError> {
        let record = self.scratch.append_material_slice(handle, version, epoch, bytes)?;
        Ok(record)
    }

    /// Record a direct draw command with explicit range parameters `[vertex_count, instance_count, first_vertex, first_instance]`
    /// in WebGPU order (§8.5).
    ///
    /// If a `material_record` is provided:
    /// 1. Validates that the record belongs to the active session's scratch buffer via
    ///    [`PerUseByteBuffer::get_slice`] to reject foreign, stale, or forged records (§5.5).
    /// 2. Binds the dynamic uniform offset obtained from [`UseRecord::byte_offset`].
    ///
    /// # Errors
    /// - Returns [`FrameError::NoActivePass`] if no pass is currently open.
    /// - Returns [`FrameError::Ownership`] if `material_record` is foreign or invalid.
    /// - Returns [`FrameError::DrawIdOverflow`] if the draw ID counter exceeds `u32::MAX`.
    pub fn record_direct_draw_with_range(
        &mut self,
        pipeline_id: u32,
        vertex_buffer_id: u32,
        range: [u32; 4],
        material_record: Option<UseRecord<MaterialDomain>>,
    ) -> Result<u32, FrameError> {
        let active = self.active_pass.as_mut().ok_or(FrameError::NoActivePass)?;

        let draw_id = self.next_draw_id;
        self.next_draw_id = self
            .next_draw_id
            .checked_add(1)
            .ok_or(FrameError::DrawIdOverflow)?;

        let (uniform_offset, mut uses) = if let Some(rec) = material_record {
            // Validate that this UseRecord was allocated by this session's buffer
            self.scratch.byte_buffer.get_slice(&rec)?;

            let offset = u32::try_from(rec.byte_offset()).map_err(|_| {
                FrameError::Ownership(OwnershipError::VersionOverflow {
                    current: rec.byte_offset(),
                })
            })?;
            let u_use = ResourceUse::buffer_uniform(
                ResourceId::new(self.uniform_buffer_id),
                rec.version(),
                Some(rec.byte_offset()),
                Some(rec.byte_length()),
            );
            (offset, alloc::vec![u_use])
        } else {
            (0, Vec::new())
        };

        if vertex_buffer_id != 0 {
            uses.push(ResourceUse {
                resource_id: ResourceId::new(vertex_buffer_id),
                kind: f3d_graph::resource::ResourceKind::Buffer,
                version: DataVersion::INITIAL,
                subresource: f3d_graph::resource::SubresourceRange::WholeBuffer,
                access: ResourceAccess::VertexBuffer,
                byte_offset: Some(0),
                byte_size: None,
                canvas_epoch: None,
            });
        }

        let ctx = self.context_stack.current();
        let viewport = match ctx.viewport {
            Some(vp) => Some(vp),
            None if ctx.viewport_reset => Some([0, 0, ctx.target_width, ctx.target_height]),
            None => None,
        };
        if ctx.scissor_test_enabled || ctx.scissor.is_some() {
            active.scissor_used = true;
        }
        let scissor = if ctx.scissor_test_enabled {
            Some(ctx.scissor.unwrap_or([0, 0, ctx.target_width, ctx.target_height]))
        } else if active.scissor_used || ctx.scissor.is_some() {
            Some([0, 0, ctx.target_width, ctx.target_height])
        } else {
            None
        };
        let draw = Draw::new(draw_id, pipeline_id, range[0], uniform_offset, uses)
            .with_range(range)
            .with_viewport(viewport)
            .with_scissor(scissor, ctx.scissor_test_enabled);
        active.draws.push(draw);
        Ok(draw_id)
    }

    /// Record a direct draw command in the active render pass with default single-instance parameters `[vertex_count, 1, 0, 0]`.
    ///
    /// Delegates directly to [`Self::record_direct_draw_with_range`].
    pub fn record_direct_draw(
        &mut self,
        pipeline_id: u32,
        vertex_buffer_id: u32,
        vertex_count: u32,
        material_record: Option<UseRecord<MaterialDomain>>,
    ) -> Result<u32, FrameError> {
        self.record_direct_draw_with_range(
            pipeline_id,
            vertex_buffer_id,
            [vertex_count, 1, 0, 0],
            material_record,
        )
    }

    /// Record a pre-recorded render bundle execution command in the active render pass (§8.5, 2v8.2).
    ///
    /// Validates any uniform buffer [`ResourceUse`] referencing the session's material uniform buffer
    /// against the authoritative scratch buffer for alignment, bounds, and allocated slice identity (§5.5).
    ///
    /// # Errors
    /// - Returns [`FrameError::NoActivePass`] if no pass is currently open.
    /// - Returns [`FrameError::Ownership`] if a uniform buffer use is unaligned, out-of-bounds, or unallocated.
    /// - Returns [`FrameError::DrawIdOverflow`] if the draw ID counter exceeds `u32::MAX`.
    pub fn record_bundle_draw(
        &mut self,
        bundle_id: u32,
        pipeline_id: u32,
        uses: Vec<ResourceUse>,
    ) -> Result<u32, FrameError> {
        let active = self.active_pass.as_mut().ok_or(FrameError::NoActivePass)?;

        // Validate uniform buffer resource uses referencing the session's material uniform buffer
        for u in &uses {
            if u.resource_id == ResourceId::new(self.uniform_buffer_id)
                && u.access == ResourceAccess::UniformBuffer
            {
                let offset = u.byte_offset.ok_or_else(|| {
                    FrameError::Ownership(OwnershipError::SliceOutOfBounds {
                        offset: 0,
                        length: 0,
                        buffer_len: self.scratch.byte_buffer.total_bytes(),
                    })
                })?;

                if self.scratch.alignment != 0 && offset % self.scratch.alignment != 0 {
                    return Err(FrameError::Ownership(OwnershipError::InvalidAlignment {
                        alignment: self.scratch.alignment,
                    }));
                }

                let len = u.byte_size.unwrap_or(0);
                let buffer_len = self.scratch.byte_buffer.total_bytes();
                let end = offset.checked_add(len).ok_or_else(|| {
                    FrameError::Ownership(OwnershipError::SliceOutOfBounds {
                        offset,
                        length: len,
                        buffer_len,
                    })
                })?;

                if end > buffer_len as u64 {
                    return Err(FrameError::Ownership(OwnershipError::SliceOutOfBounds {
                        offset,
                        length: len,
                        buffer_len,
                    }));
                }

                // Verify that the referenced offset corresponds to an allocated slice record in this session
                let matching = self
                    .scratch
                    .byte_buffer
                    .records()
                    .iter()
                    .find(|r| r.byte_offset() == offset);
                match matching {
                    Some(rec) => {
                        self.scratch.byte_buffer.get_slice(rec)?;
                        if let Some(req_len) = u.byte_size {
                            if req_len != rec.byte_length() {
                                return Err(FrameError::Ownership(OwnershipError::SliceOutOfBounds {
                                    offset,
                                    length: req_len,
                                    buffer_len,
                                }));
                            }
                        }
                    }
                    None => {
                        return Err(FrameError::Ownership(OwnershipError::SliceOutOfBounds {
                            offset,
                            length: len,
                            buffer_len,
                        }));
                    }
                }
            }
        }

        let draw_id = self.next_draw_id;
        self.next_draw_id = self
            .next_draw_id
            .checked_add(1)
            .ok_or(FrameError::DrawIdOverflow)?;

        let ctx = self.context_stack.current();
        let viewport = match ctx.viewport {
            Some(vp) => Some(vp),
            None if ctx.viewport_reset => Some([0, 0, ctx.target_width, ctx.target_height]),
            None => None,
        };
        if ctx.scissor_test_enabled || ctx.scissor.is_some() {
            active.scissor_used = true;
        }
        let scissor = if ctx.scissor_test_enabled {
            Some(ctx.scissor.unwrap_or([0, 0, ctx.target_width, ctx.target_height]))
        } else if active.scissor_used || ctx.scissor.is_some() {
            Some([0, 0, ctx.target_width, ctx.target_height])
        } else {
            None
        };
        let draw = Draw::new_bundle(draw_id, bundle_id, pipeline_id, uses)
            .with_viewport(viewport)
            .with_scissor(scissor, ctx.scissor_test_enabled);
        active.draws.push(draw);
        Ok(draw_id)
    }

    /// End the active render pass and register it in the pass graph.
    ///
    /// # Errors
    /// Returns [`FrameError::NoActivePass`] if no pass is currently open.
    pub fn end_render_pass(&mut self) -> Result<PassId, FrameError> {
        let active = self.active_pass.take().ok_or(FrameError::NoActivePass)?;
        let mut pass = Pass::new_render(active.id, active.name)
            .with_color_attachment(active.color_attachment);

        for draw in active.draws {
            pass = pass.with_draw(draw);
        }

        if let Some(dep) = active.dependency {
            pass = pass.with_dependency(dep);
        }

        self.pass_graph.add_pass(pass)?;
        self.last_completed_pass_id = Some(active.id);
        Ok(active.id)
    }

    /// Resumes a previously split outer render pass with [`LoadOp::Load`] on the outer target (§6.7).
    fn resume_outer_pass(&mut self, resume: &OuterResumeState) -> Result<PassId, FrameError> {
        let current_ctx = self.context_stack.current_mut();
        current_ctx.target_width = resume.target_width;
        current_ctx.target_height = resume.target_height;
        current_ctx.viewport = resume.viewport;
        current_ctx.viewport_reset = resume.viewport_reset;
        current_ctx.scissor = resume.scissor;
        current_ctx.scissor_test_enabled = resume.scissor_test_enabled;

        let resumed_id = PassId::new(self.next_pass_id);
        self.next_pass_id = self
            .next_pass_id
            .checked_add(1)
            .ok_or(FrameError::Graph(GraphError::PassIdOverflow))?;

        let mut resumed_color_attachment = match (resume.is_canvas, resume.canvas_output_epoch) {
            (true, Some(epoch)) => {
                ColorAttachment::new_canvas(resume.target_id, [0.0, 0.0, 0.0, 1.0], epoch)
            }
            _ => ColorAttachment::new_load(resume.target_id),
        };
        resumed_color_attachment.load_op = LoadOp::Load;
        resumed_color_attachment.store_op = StoreOp::Store;

        let dep = self.last_completed_pass_id.or(Some(resume.prefix_pass_id));
        let scissor_used = resume.scissor_used || resume.scissor_test_enabled || resume.scissor.is_some();

        self.active_pass = Some(ActivePass {
            id: resumed_id,
            name: format!("{}_resumed", resume.name),
            color_attachment: resumed_color_attachment,
            draws: Vec::new(),
            dependency: dep,
            scissor_used,
        });

        Ok(resumed_id)
    }

    /// Execute reentrant rendering in a nested render context (§6.7).
    ///
    /// # Active-Pass Nesting & Resumption Flow
    /// If an outer render pass is currently active (the standard `onBeforeRender` callback pattern):
    /// 1. The in-progress outer pass is ended as an "outer prefix" pass with `StoreOp::Store`,
    ///    preserving all draws issued before the callback.
    /// 2. The nested context is pushed with a [`SessionNestingGuard`] that guarantees restoration
    ///    on normal return, error, or panic.
    /// 3. The nested closure `f` executes its render operations.
    /// 4. On return, the outer context is restored and the outer pass is **resumed** on the outer target
    ///    with [`LoadOp::Load`], preserving earlier contents and adding a pass dependency on the nested pass.
    ///
    /// If no pass was open at entry, nested passes execute independently without prefix/resume passes.
    pub fn with_nested_render<R, F>(
        &mut self,
        nested_context: RenderContext,
        f: F,
    ) -> Result<R, FrameError>
    where
        F: FnOnce(&mut FrameSession) -> Result<R, FrameError>,
    {
        // 0. Pre-admission check: reject if nesting depth would exceed max_depth BEFORE splitting or mutating outer pass
        if self.context_stack.depth() >= self.context_stack.max_depth() {
            return Err(FrameError::MaxNestingDepthExceeded {
                depth: self.context_stack.depth(),
                max: self.context_stack.max_depth(),
            });
        }

        // 1. If an outer pass is active, split and commit it as an outer prefix pass
        let outer_resume_state = if self.active_pass.is_some() {
            let active = self.active_pass.take().unwrap();
            let prefix_id = active.id;
            let mut pass = Pass::new_render(prefix_id, active.name.clone())
                .with_color_attachment(active.color_attachment.clone());

            for draw in active.draws {
                pass = pass.with_draw(draw);
            }
            if let Some(dep) = active.dependency {
                pass = pass.with_dependency(dep);
            }

            self.pass_graph.add_pass(pass)?;
            self.last_completed_pass_id = Some(prefix_id);

            let ctx = self.context_stack.current();
            let scissor_used = active.scissor_used || ctx.scissor_test_enabled || ctx.scissor.is_some();
            Some(OuterResumeState {
                target_id: ctx.target_id,
                is_canvas: ctx.is_canvas,
                target_width: ctx.target_width,
                target_height: ctx.target_height,
                viewport: ctx.viewport,
                viewport_reset: ctx.viewport_reset,
                scissor: ctx.scissor,
                scissor_test_enabled: ctx.scissor_test_enabled,
                scissor_used,
                camera_epoch: ctx.camera_epoch,
                canvas_output_epoch: ctx.canvas_output_epoch,
                name: active.name,
                prefix_pass_id: prefix_id,
            })
        } else {
            None
        };

        // 2. Install the RAII nesting guard for unwind, error, and ancestor safety
        let mut guard = SessionNestingGuard::new(self, nested_context, outer_resume_state)?;

        // 3. Execute nested closure
        let result = f(guard.session);

        match result {
            Ok(val) => {
                // If nested callback left an active pass open, cleanly end it; propagate any error
                if guard.session.active_pass.is_some() {
                    guard.session.end_render_pass()?;
                }

                // Restore context stack ancestors idempotently
                guard.restore_context_stack();

                // 4. Resume outer pass if one was open at entry
                if let Some(resume) = guard.outer_resume_state.take() {
                    guard.session.resume_outer_pass(&resume)?;
                }

                guard.completed = true;
                Ok(val)
            }
            Err(err) => {
                // Discard any incomplete pass left open inside the failing nested scope
                guard.session.active_pass = None;
                // Roll back scratch staging bytes to outer watermark
                guard.session.scratch.truncate_scratch(guard.entry_watermark);
                // Restore context stack ancestors idempotently
                guard.restore_context_stack();

                // Resume outer pass so the outer caller's active pass is not stranded
                if let Some(resume) = guard.outer_resume_state.take() {
                    let _ = guard.session.resume_outer_pass(&resume);
                }

                guard.completed = true;
                Err(err)
            }
        }
    }

    /// Compiles the accumulated pass graph with an optional [`CanvasEpochTracker`], lowers it
    /// via bridge lowering, and packages the commands into a verified [`GpuSubmissionPacket`].
    ///
    /// When rendering to a canvas swapchain with captured [`Epoch`]s, the [`CanvasEpochTracker`]
    /// validates that the canvas texture was actively acquired for the current frame interval
    /// and rejects stale or unacquired epochs (§8.5, [S48]). If `canvas_tracker` is `None`,
    /// any tracker previously configured on the session via [`Self::with_canvas_tracker`] is used.
    ///
    /// If material uniform parameters were recorded via [`Self::snapshot_material_use`],
    /// an initial [`GpuCommand::WriteBuffer`] is prepended to upload the complete uniform arena
    /// before any render passes execute, ensuring queue-ordering integrity (§5.5, §8.2).
    pub fn build_submission_packet_with_tracker(
        &mut self,
        canvas_tracker: Option<&CanvasEpochTracker>,
    ) -> Result<GpuSubmissionPacket, FrameError> {
        if self.active_pass.is_some() {
            self.end_render_pass()?;
        }

        let tracker = canvas_tracker.or(self.canvas_tracker.as_ref());
        let plan = self.pass_graph.compile(tracker)?;
        let lowered_commands = lower_plan(&plan)?;

        let mut packet = GpuSubmissionPacket::new();

        let uniform_bytes = self.scratch.byte_buffer.as_bytes();
        if !uniform_bytes.is_empty() {
            packet.push(GpuCommand::WriteBuffer {
                buffer_id: self.uniform_buffer_id,
                offset: 0,
                data: uniform_bytes.to_vec(),
            });
        }

        for cmd in lowered_commands {
            packet.push(cmd);
        }

        Ok(packet)
    }

    /// Compiles the accumulated pass graph without overriding the canvas tracker, lowers it
    /// via bridge lowering, and packages the commands into a verified [`GpuSubmissionPacket`].
    ///
    /// Equivalent to calling [`Self::build_submission_packet_with_tracker`] with `None`.
    pub fn build_submission_packet(&mut self) -> Result<GpuSubmissionPacket, FrameError> {
        self.build_submission_packet_with_tracker(None)
    }
}

/// Safe RAII guard ensuring rollback of scratch, discard of incomplete passes, restoration
/// of context stack ancestors and context, and resumption of outer passes on error or panic
/// during [`FrameSession::with_nested_render`].
struct SessionNestingGuard<'a> {
    session: &'a mut FrameSession,
    entry_context: RenderContext,
    saved_stack: Option<Vec<RenderContext>>,
    entry_watermark: usize,
    outer_resume_state: Option<OuterResumeState>,
    completed: bool,
}

impl<'a> SessionNestingGuard<'a> {
    fn new(
        session: &'a mut FrameSession,
        nested: RenderContext,
        outer_resume_state: Option<OuterResumeState>,
    ) -> Result<Self, FrameError> {
        let entry_context = session.context_stack.current().clone();
        let saved_stack = Some(session.context_stack.stack.clone());
        let entry_watermark = session.scratch.scratch_len();

        let mut nested_with_watermark = nested;
        nested_with_watermark.scratch_watermark = entry_watermark;

        session.context_stack.push(nested_with_watermark)?;

        Ok(Self {
            session,
            entry_context,
            saved_stack,
            entry_watermark,
            outer_resume_state,
            completed: false,
        })
    }

    fn restore_context_stack(&mut self) {
        if let Some(saved) = self.saved_stack.take() {
            self.session.context_stack.stack = saved;
            self.session.context_stack.current = self.entry_context.clone();
        }
    }
}

impl Drop for SessionNestingGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            // Discard any incomplete pass left open inside the failing nested scope
            self.session.active_pass = None;
            // Roll back scratch staging bytes to outer watermark
            self.session.scratch.truncate_scratch(self.entry_watermark);
            // Restore context stack ancestors idempotently
            self.restore_context_stack();

            // If an outer pass was split, resume it even on panic/unwind so active pass is not stranded
            if let Some(resume) = self.outer_resume_state.take() {
                let _ = self.session.resume_outer_pass(&resume);
            }
        }
    }
}

/// Errors arising during logical frame rendering, context switching, or packet synthesis.
#[derive(Clone, Debug, PartialEq)]
pub enum FrameError {
    /// Context stack underflow: attempted to pop the root render context.
    ContextStackUnderflow,
    /// Nesting depth exceeded the maximum recursion bound.
    MaxNestingDepthExceeded {
        /// Current depth attempted.
        depth: usize,
        /// Maximum allowed depth.
        max: usize,
    },
    /// Operation requires an active render pass, but none was open.
    NoActivePass,
    /// Operation attempted to begin a pass while another pass was already open.
    PassAlreadyActive {
        /// Identifier of the currently active pass.
        pass_id: u32,
    },
    /// Draw ID counter overflowed u32::MAX.
    DrawIdOverflow,
    /// Invalid viewport dimensions specified.
    InvalidViewport {
        /// Viewport coordinates `[x, y, width, height]`.
        rect: [u32; 4],
    },
    /// Invalid scissor rectangle specified.
    InvalidScissor {
        /// Scissor coordinates `[x, y, width, height]`.
        rect: [u32; 4],
    },
    /// Attempted an invalid context transition.
    InvalidNestingState {
        /// Diagnostic description of the illegal transition.
        detail: String,
    },
    /// Core ownership, versioning, or buffer allocation error.
    Ownership(OwnershipError),
    /// Pass graph compilation, scheduling, or cycle error.
    Graph(GraphError),
    /// Bridge plan lowering error.
    PlanLowering(PlanLoweringError),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextStackUnderflow => write!(f, "Attempted to pop the root render context"),
            Self::MaxNestingDepthExceeded { depth, max } => {
                write!(f, "Render context depth {depth} exceeded maximum limit {max}")
            }
            Self::NoActivePass => write!(f, "No render pass is currently active"),
            Self::PassAlreadyActive { pass_id } => {
                write!(f, "Render pass #{pass_id} is already active")
            }
            Self::DrawIdOverflow => write!(f, "Draw ID counter overflowed u32::MAX"),
            Self::InvalidViewport { rect } => {
                write!(f, "Invalid viewport rectangle {rect:?}")
            }
            Self::InvalidScissor { rect } => {
                write!(f, "Invalid scissor rectangle {rect:?}")
            }
            Self::InvalidNestingState { detail } => {
                write!(f, "Invalid nesting state: {detail}")
            }
            Self::Ownership(err) => write!(f, "Ownership error: {err}"),
            Self::Graph(err) => write!(f, "Graph error: {err}"),
            Self::PlanLowering(err) => write!(f, "Plan lowering error: {err}"),
        }
    }
}

impl core::error::Error for FrameError {}

impl From<OwnershipError> for FrameError {
    fn from(err: OwnershipError) -> Self {
        Self::Ownership(err)
    }
}

impl From<GraphError> for FrameError {
    fn from(err: GraphError) -> Self {
        Self::Graph(err)
    }
}

impl From<CanvasError> for FrameError {
    fn from(err: CanvasError) -> Self {
        Self::Graph(GraphError::Canvas(err))
    }
}

impl From<PlanLoweringError> for FrameError {
    fn from(err: PlanLoweringError) -> Self {
        Self::PlanLowering(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;

    #[test]
    fn test_context_stack_push_pop_and_state_restoration() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::new(10))
            .with_viewport(0, 0, 800, 600)
            .with_scissor(10, 10, 780, 580)
            .with_scissor_test(true)
            .with_camera_projection([1.5; 16]);

        let mut stack = RenderContextStack::new(root.clone());
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_root());
        assert_eq!(stack.current(), &root);

        // Push nested context for offscreen shadow map
        let shadow_ctx = RenderContext::new_offscreen(ResourceId::new(2), 1024, 1024, Epoch::new(11))
            .with_viewport(0, 0, 1024, 1024)
            .with_scissor(0, 0, 1024, 1024)
            .with_scissor_test(false)
            .with_camera_projection([2.0; 16]);

        stack.push(shadow_ctx.clone()).expect("push shadow context");
        assert_eq!(stack.depth(), 1);
        assert!(!stack.is_root());
        assert_eq!(stack.current().target_id, ResourceId::new(2));
        assert!(!stack.current().is_canvas);
        assert_eq!(stack.current().viewport, Some([0, 0, 1024, 1024]));
        assert_eq!(stack.current().camera_epoch, Epoch::new(11));
        assert_eq!(stack.current().camera_projection, [2.0; 16]);

        // Pop shadow context: outer context must be restored verbatim
        let popped = stack.pop().expect("pop shadow context");
        assert_eq!(popped, shadow_ctx);
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_root());
        assert_eq!(stack.current(), &root);

        // Popping root context must underflow cleanly
        let underflow_err = stack.pop().expect_err("pop root must underflow");
        assert_eq!(underflow_err, FrameError::ContextStackUnderflow);
    }

    #[test]
    fn test_context_stack_raii_guard_restores_on_unwind_and_unbalanced_stack() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 1920, 1080, Epoch::new(1))
            .with_camera_projection([1.0; 16]);
        let mut stack = RenderContextStack::new(root.clone());

        let nested = RenderContext::new_offscreen(ResourceId::new(99), 256, 256, Epoch::new(2));

        // 1. Error return: guard restores entry state even if closure pushed extra frames
        let result: Result<(), FrameError> = stack.with_nested_context(nested.clone(), |s| {
            assert_eq!(s.depth(), 1);
            // Intentionally unbalance stack by pushing another frame
            let extra = RenderContext::new_offscreen(ResourceId::new(100), 128, 128, Epoch::ZERO);
            s.push(extra).expect("push extra");
            assert_eq!(s.depth(), 2);
            Err(FrameError::InvalidNestingState {
                detail: String::from("simulated error"),
            })
        });

        assert!(result.is_err());
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_root());
        assert_eq!(stack.current(), &root);

        // 2. Unwind (panic) safety via std::panic::catch_unwind
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), FrameError> = stack.with_nested_context(nested, |s| {
                let extra = RenderContext::new_offscreen(ResourceId::new(101), 64, 64, Epoch::ZERO);
                s.push(extra).expect("push extra");
                panic!("simulated nested panic");
            });
        }));

        assert!(panic_result.is_err());
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_root());
        assert_eq!(stack.current(), &root);
    }

    #[test]
    fn test_context_stack_max_depth_enforcement() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 100, 100, Epoch::ZERO);
        let mut stack = RenderContextStack::with_max_depth(root.clone(), 2);

        let ctx1 = RenderContext::new_offscreen(ResourceId::new(2), 100, 100, Epoch::ZERO);
        let ctx2 = RenderContext::new_offscreen(ResourceId::new(3), 100, 100, Epoch::ZERO);
        let ctx3 = RenderContext::new_offscreen(ResourceId::new(4), 100, 100, Epoch::ZERO);

        stack.push(ctx1).expect("depth 1 ok");
        stack.push(ctx2).expect("depth 2 ok");
        assert_eq!(stack.depth(), 2);

        let err = stack.push(ctx3).expect_err("depth 3 must exceed max_depth 2");
        assert_eq!(err, FrameError::MaxNestingDepthExceeded { depth: 2, max: 2 });
        assert_eq!(stack.depth(), 2);
    }

    #[test]
    fn test_nested_scratch_isolation_and_rollback() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root.clone(), 256).expect("session init");

        // Outer scratch write
        let outer_offset = session.scratch_mut().allocate_scratch(b"outer_data_bytes");
        assert_eq!(outer_offset, 0);
        assert_eq!(session.scratch().scratch_len(), 16);

        let nested_ctx = RenderContext::new_offscreen(ResourceId::new(2), 256, 256, Epoch::ZERO);

        // Nested execution that fails must roll back scratch to outer watermark (16 bytes)
        let fail_result: Result<(), FrameError> = session.with_nested_render(nested_ctx, |s| {
            let nested_offset = s.scratch_mut().allocate_scratch(b"nested_scratch_that_aborts");
            assert_eq!(nested_offset, 16);
            assert_eq!(s.scratch().scratch_len(), 42);
            Err(FrameError::InvalidNestingState {
                detail: String::from("abort nested pass"),
            })
        });

        assert!(fail_result.is_err());
        assert_eq!(session.depth(), 0);
        assert_eq!(session.context_stack().current(), &root);
        // Scratch bytes must be cleanly rolled back to outer watermark
        assert_eq!(session.scratch().scratch_len(), 16);
        assert_eq!(session.scratch().scratch_bytes(), b"outer_data_bytes");
    }

    #[test]
    fn test_callback_multiplicity_allocates_distinct_uniform_slices() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root, 256).expect("session init");

        let mat_handle = Handle::<MaterialDomain>::new(10, NonZeroU32::new(1).unwrap());
        let red_bytes = [255u8, 0, 0, 255];
        let blue_bytes = [0u8, 0, 255, 255];

        let red_ver = DataVersion::INITIAL;
        let blue_ver = red_ver.checked_next().unwrap();

        // 1. Snapshot red material before draw A
        let rec_a = session
            .snapshot_material_use(mat_handle, red_ver, Epoch::ZERO, &red_bytes)
            .expect("snapshot red");

        // 2. Intervening callback mutates material to blue before draw B
        let rec_b = session
            .snapshot_material_use(mat_handle, blue_ver, Epoch::ZERO, &blue_bytes)
            .expect("snapshot blue");

        // Distinct slices must be allocated with distinct dynamic offsets
        assert_ne!(rec_a.byte_offset(), rec_b.byte_offset());
        assert_eq!(rec_a.byte_offset() % 256, 0);
        assert_eq!(rec_b.byte_offset() % 256, 0);
        assert_eq!(rec_a.version(), red_ver);
        assert_eq!(rec_b.version(), blue_ver);

        // Verify that the byte buffer contains both slices without collision or overwrite
        let slice_a = session.scratch().byte_buffer().get_slice(&rec_a).expect("get slice a");
        let slice_b = session.scratch().byte_buffer().get_slice(&rec_b).expect("get slice b");
        assert_eq!(slice_a, &red_bytes);
        assert_eq!(slice_b, &blue_bytes);
    }

    #[test]
    fn test_record_direct_draw_rejects_foreign_use_record() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root, 256).expect("session init");

        // Create a foreign PerUseByteBuffer with its own unique store_id
        let mut foreign_store = PerUseByteBuffer::<MaterialDomain>::new(256).expect("foreign store");
        let mat_handle = Handle::<MaterialDomain>::new(5, NonZeroU32::new(1).unwrap());
        let foreign_rec = foreign_store
            .append_slice(mat_handle, DataVersion::INITIAL, Epoch::ZERO, &[1, 2, 3, 4])
            .expect("append foreign");

        session.begin_render_pass("pass1", [0.0, 0.0, 0.0, 1.0]).expect("begin pass");

        // Passing foreign_rec must be rejected by checked get_slice ownership validation
        let err = session
            .record_direct_draw(1, 0, 3, Some(foreign_rec))
            .expect_err("foreign record must be rejected");

        match err {
            FrameError::Ownership(OwnershipError::ForeignSliceRecord { expected_store, actual_store }) => {
                assert_eq!(expected_store, session.scratch().byte_buffer().store_id());
                assert_eq!(actual_store, foreign_store.store_id());
            }
            other => panic!("expected ForeignSliceRecord, got {other:?}"),
        }
    }

    #[test]
    fn test_record_direct_draw_rejects_draw_id_overflow() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root, 256).expect("session init");
        session.next_draw_id = u32::MAX;

        session.begin_render_pass("pass_overflow", [0.0, 0.0, 0.0, 1.0]).expect("begin pass");
        let err = session
            .record_direct_draw(1, 0, 3, None)
            .expect_err("overflow draw ID must error");
        assert_eq!(err, FrameError::DrawIdOverflow);
    }

    #[test]
    fn test_active_pass_nesting_ordered_prefix_inner_resume_flow() {
        let target_a = ResourceId::new(10);
        let target_b = ResourceId::new(20);

        let root_ctx = RenderContext::new_canvas(target_a, 1920, 1080, Epoch::new(1))
            .with_viewport(0, 0, 1920, 1080)
            .with_camera_projection([1.0; 16]);
        let mut session = FrameSession::new(root_ctx.clone(), 256).expect("session init");

        let mat_handle = Handle::<MaterialDomain>::new(10, NonZeroU32::new(1).unwrap());
        let red_bytes = [255u8, 0, 0, 255];
        let blue_bytes = [0u8, 0, 255, 255];

        let rec_red = session
            .snapshot_material_use(mat_handle, DataVersion::INITIAL, Epoch::new(1), &red_bytes)
            .expect("red snapshot");

        // 1. Begin outer pass on Target A and record Draw 1 (Red)
        session.begin_render_pass("outer_scene", [0.0, 0.0, 0.0, 1.0]).expect("begin outer");
        session.record_direct_draw(1, 0, 3, Some(rec_red)).expect("draw 1 in outer");

        // 2. While outer pass is active, an onBeforeRender callback triggers nested rendering to Target B
        let shadow_ctx = RenderContext::new_offscreen(target_b, 512, 512, Epoch::new(2))
            .with_viewport(0, 0, 512, 512)
            .with_camera_projection([2.0; 16]);

        session
            .with_nested_render(shadow_ctx, |s| {
                // Assert nested context is active and outer state is saved
                assert_eq!(s.depth(), 1);
                assert_eq!(s.context_stack().current().target_id, target_b);
                assert_eq!(s.context_stack().current().camera_projection, [2.0; 16]);

                // Nested pass on Target B with Draw 2 (Blue)
                let rec_blue = s
                    .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::new(2), &blue_bytes)
                    .expect("blue snapshot");
                s.begin_render_pass("nested_shadow", [0.0, 0.0, 0.0, 1.0]).expect("begin shadow");
                s.record_direct_draw(2, 0, 6, Some(rec_blue)).expect("draw 2 in shadow");
                s.end_render_pass().expect("end shadow");
                Ok(())
            })
            .expect("nested render succeeds");

        // 3. Post-condition: context is restored to Target A and outer pass is RESUMED with LoadOp::Load
        assert_eq!(session.depth(), 0);
        assert_eq!(session.context_stack().current(), &root_ctx);
        assert!(session.active_pass.is_some());

        let active_resumed = session.active_pass.as_ref().unwrap();
        assert_eq!(active_resumed.name, "outer_scene_resumed");
        assert_eq!(active_resumed.color_attachment.load_op, LoadOp::Load);
        assert_eq!(active_resumed.color_attachment.store_op, StoreOp::Store);

        // 4. Continue outer rendering in resumed pass with Draw 3 (Red)
        session.record_direct_draw(1, 0, 3, Some(rec_red)).expect("draw 3 in resumed outer");
        session.end_render_pass().expect("end resumed outer");

        // 5. Compile plan and verify topological scheduling and load semantics
        let plan = session.pass_graph().compile(None).expect("compile plan");
        assert_eq!(plan.segment_count(), 3);

        let seg0 = &plan.segments()[0]; // outer_scene prefix (LoadOp::Clear)
        let seg1 = &plan.segments()[1]; // nested_shadow (LoadOp::Clear)
        let seg2 = &plan.segments()[2]; // outer_scene_resumed (LoadOp::Load)

        assert_eq!(seg0.name(), "outer_scene");
        assert_eq!(seg0.primary_color_attachment().unwrap().target_id(), target_a);
        assert_eq!(seg0.primary_color_attachment().unwrap().load_op, LoadOp::Clear);
        assert_eq!(seg0.draws().len(), 1);

        assert_eq!(seg1.name(), "nested_shadow");
        assert_eq!(seg1.primary_color_attachment().unwrap().target_id(), target_b);
        assert_eq!(seg1.primary_color_attachment().unwrap().load_op, LoadOp::Clear);
        assert_eq!(seg1.draws().len(), 1);

        assert_eq!(seg2.name(), "outer_scene_resumed");
        assert_eq!(seg2.primary_color_attachment().unwrap().target_id(), target_a);
        assert_eq!(seg2.primary_color_attachment().unwrap().load_op, LoadOp::Load);
        assert_eq!(seg2.draws().len(), 1);

        // 6. Build packet: lower_plan succeeds and uniform buffer contains both Red and Blue data
        let packet = session.build_submission_packet().expect("build submission packet");
        let encoded = packet.encode().expect("encode packet");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_nested_render_propagates_callback_error() {
        let root = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root.clone(), 256).expect("session init");

        let nested_ctx = RenderContext::new_offscreen(ResourceId::new(2), 256, 256, Epoch::ZERO);

        // A callback error propagates after restoring the enclosing context.
        let err: FrameError = session
            .with_nested_render(nested_ctx, |s| {
                s.begin_render_pass("valid_pass", [0.0, 0.0, 0.0, 1.0]).expect("begin");
                Err::<(), FrameError>(FrameError::InvalidNestingState {
                    detail: String::from("explicit error to test propagation"),
                })
            })
            .expect_err("error must propagate");

        assert_eq!(
            err,
            FrameError::InvalidNestingState {
                detail: String::from("explicit error to test propagation")
            }
        );
        assert_eq!(session.depth(), 0);
        assert_eq!(session.context_stack().current(), &root);
    }

    #[test]
    fn test_context_stack_restores_ancestors_when_popped_below_entry_depth() {
        let ctx0 = RenderContext::new_canvas(ResourceId::new(1), 100, 100, Epoch::ZERO);
        let ctx1 = RenderContext::new_offscreen(ResourceId::new(2), 100, 100, Epoch::ZERO);
        let ctx2 = RenderContext::new_offscreen(ResourceId::new(3), 100, 100, Epoch::ZERO);
        let ctx3 = RenderContext::new_offscreen(ResourceId::new(4), 100, 100, Epoch::ZERO);

        let mut stack = RenderContextStack::new(ctx0.clone());
        stack.push(ctx1.clone()).unwrap();
        stack.push(ctx2.clone()).unwrap();
        assert_eq!(stack.depth(), 2);

        let res = stack.with_nested_context(ctx3, |s| {
            s.pop().unwrap(); // pops ctx2
            s.pop().unwrap(); // pops ctx1
            s.pop().unwrap(); // pops ctx0
            Ok(())
        });
        assert!(res.is_ok());

        // Ancestors must be exactly restored, not lost via no-op truncate
        assert_eq!(stack.depth(), 2);
        assert_eq!(stack.current(), &ctx2);
        let popped = stack.pop().unwrap();
        assert_eq!(popped, ctx2);
        assert_eq!(stack.current(), &ctx1);
        assert_eq!(stack.depth(), 1);
        let popped_root = stack.pop().unwrap();
        assert_eq!(popped_root, ctx1);
        assert_eq!(stack.current(), &ctx0);
        assert_eq!(stack.depth(), 0);
        assert!(stack.is_root());
    }

    #[test]
    fn test_nested_render_resumes_outer_pass_when_nested_callback_errors() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(1), 100, 100, Epoch::ZERO);
        let mut session = FrameSession::new(root_ctx, 256).unwrap();

        session.begin_render_pass("main", [0.0, 0.0, 0.0, 1.0]).unwrap();
        session.record_direct_draw(1, 0, 3, None).unwrap(); // Draw 1 in main prefix

        let nested_ctx = RenderContext::new_offscreen(ResourceId::new(2), 100, 100, Epoch::ZERO);
        let res: Result<(), FrameError> = session.with_nested_render(nested_ctx, |s| {
            s.begin_render_pass("inner", [0.0, 0.0, 0.0, 1.0]).unwrap();
            s.record_direct_draw(2, 0, 3, None).unwrap();
            s.end_render_pass().unwrap();
            Err(FrameError::InvalidNestingState {
                detail: String::from("aborted"),
            })
        });
        assert!(res.is_err());

        // Outer pass must not be dropped; must be resumed with LoadOp::Load
        assert!(session.active_pass.is_some());
        let active = session.active_pass.as_ref().unwrap();
        assert_eq!(active.name, "main_resumed");
        assert_eq!(active.color_attachment.load_op, LoadOp::Load);

        // Recording subsequent draw into resumed pass must succeed
        session.record_direct_draw(1, 0, 3, None).unwrap(); // Draw 3 in main_resumed
        session.end_render_pass().unwrap();

        // Verify pass graph has 3 segments: main prefix, inner, and main_resumed
        let plan = session.pass_graph().compile(None).unwrap();
        assert_eq!(plan.segment_count(), 3);
        assert_eq!(plan.segments()[0].name(), "main");
        assert_eq!(plan.segments()[1].name(), "inner");
        assert_eq!(plan.segments()[2].name(), "main_resumed");
    }

    #[test]
    fn test_record_bundle_draw_validates_uniform_buffer_offsets() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root_ctx, 256).unwrap();

        let mat_handle = Handle::<MaterialDomain>::new(10, NonZeroU32::new(1).unwrap());
        let rec = session
            .snapshot_material_use(mat_handle, DataVersion::INITIAL, Epoch::ZERO, &[1, 2, 3, 4])
            .unwrap();

        session.begin_render_pass("pass_bundle", [0.0, 0.0, 0.0, 1.0]).unwrap();

        // 1. Valid bundle draw with matched uniform use succeeds
        let valid_use = ResourceUse::buffer_uniform(
            ResourceId::new(session.uniform_buffer_id),
            rec.version(),
            Some(rec.byte_offset()),
            Some(rec.byte_length()),
        );
        let draw_id = session.record_bundle_draw(1, 10, alloc::vec![valid_use]).unwrap();
        assert_eq!(draw_id, 1);

        // 2. Unaligned uniform offset is rejected with InvalidAlignment
        let unaligned_use = ResourceUse::buffer_uniform(
            ResourceId::new(session.uniform_buffer_id),
            rec.version(),
            Some(rec.byte_offset() + 1), // unaligned (alignment is 256)
            Some(rec.byte_length()),
        );
        let err = session
            .record_bundle_draw(2, 10, alloc::vec![unaligned_use])
            .unwrap_err();
        assert_eq!(
            err,
            FrameError::Ownership(OwnershipError::InvalidAlignment { alignment: 256 })
        );

        // 3. Out-of-bounds uniform offset is rejected with SliceOutOfBounds
        let oob_use = ResourceUse::buffer_uniform(
            ResourceId::new(session.uniform_buffer_id),
            rec.version(),
            Some(1024), // beyond buffer length
            Some(4),
        );
        let err = session
            .record_bundle_draw(3, 10, alloc::vec![oob_use])
            .unwrap_err();
        match err {
            FrameError::Ownership(OwnershipError::SliceOutOfBounds { offset, .. }) => {
                assert_eq!(offset, 1024);
            }
            other => panic!("expected SliceOutOfBounds, got {other:?}"),
        }

        // 4. Non-uniform uses (e.g. vertex buffer) are not checked against uniform buffer
        let vb_use = ResourceUse {
            resource_id: ResourceId::new(99),
            kind: f3d_graph::resource::ResourceKind::Buffer,
            version: DataVersion::INITIAL,
            subresource: f3d_graph::resource::SubresourceRange::WholeBuffer,
            access: ResourceAccess::VertexBuffer,
            byte_offset: Some(0),
            byte_size: None,
            canvas_epoch: None,
        };
        let vb_draw = session.record_bundle_draw(4, 10, alloc::vec![vb_use]).unwrap();
        assert_eq!(vb_draw, 2);
    }

    #[test]
    fn test_session_nesting_restores_ancestors_when_popped_below_entry_depth() {
        let ctx0 = RenderContext::new_canvas(ResourceId::new(1), 100, 100, Epoch::ZERO);
        let ctx1 = RenderContext::new_offscreen(ResourceId::new(2), 100, 100, Epoch::ZERO);
        let ctx2 = RenderContext::new_offscreen(ResourceId::new(3), 100, 100, Epoch::ZERO);
        let ctx3 = RenderContext::new_offscreen(ResourceId::new(4), 100, 100, Epoch::ZERO);

        let mut session = FrameSession::new(ctx0.clone(), 256).unwrap();
        session.context_stack_mut().push(ctx1.clone()).unwrap();
        session.context_stack_mut().push(ctx2.clone()).unwrap();
        assert_eq!(session.depth(), 2);

        let res: Result<(), FrameError> = session.with_nested_render(ctx3, |s| {
            s.context_stack_mut().pop().unwrap();
            s.context_stack_mut().pop().unwrap();
            s.context_stack_mut().pop().unwrap();
            Ok(())
        });
        assert!(res.is_ok());

        assert_eq!(session.depth(), 2);
        assert_eq!(session.context_stack().current(), &ctx2);
        let popped = session.context_stack_mut().pop().unwrap();
        assert_eq!(popped, ctx2);
        assert_eq!(session.context_stack().current(), &ctx1);
    }

    #[test]
    fn test_nested_render_pre_admission_rejects_at_max_depth_preserving_active_outer_pass() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(10), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root_ctx, 256).unwrap();
        session.context_stack_mut().set_max_depth(1);

        // Begin outer pass and record Draw 1
        session.begin_render_pass("main_pass", [0.0, 0.0, 0.0, 1.0]).unwrap();
        session.record_direct_draw(1, 0, 3, None).unwrap();

        // Push an offscreen context manually to reach depth 1 (the maximum allowed)
        let intermediate_ctx = RenderContext::new_offscreen(ResourceId::new(20), 100, 100, Epoch::ZERO);
        session.context_stack_mut().push(intermediate_ctx).unwrap();
        assert_eq!(session.depth(), 1);

        // Attempt with_nested_render while depth == max_depth: pre-admission check must reject
        let nested_ctx = RenderContext::new_offscreen(ResourceId::new(30), 100, 100, Epoch::ZERO);
        let err = session
            .with_nested_render(nested_ctx, |_| Ok(()))
            .expect_err("must reject when depth >= max_depth");

        assert_eq!(err, FrameError::MaxNestingDepthExceeded { depth: 1, max: 1 });

        // Outer pass must NOT be stranded as None; it was not split or dropped
        assert!(session.active_pass.is_some());
        assert_eq!(session.active_pass.as_ref().unwrap().name, "main_pass");

        // Popping the manually pushed frame brings depth back to 0
        session.context_stack_mut().pop().unwrap();
        assert_eq!(session.depth(), 0);

        // Recording a subsequent draw in the intact outer pass must succeed
        session.record_direct_draw(1, 0, 3, None).unwrap();
        let finished_id = session.end_render_pass().unwrap();
        assert_eq!(finished_id.get(), 1);

        // Pass graph has exactly 1 pass with 2 draws
        let plan = session.pass_graph().compile(None).unwrap();
        assert_eq!(plan.segment_count(), 1);
        assert_eq!(plan.segments()[0].draws().len(), 2);
    }

    #[test]
    fn test_nested_render_packet_three_openers_source_order_and_load_resume() {
        let canvas_target = ResourceId::new(10);
        let offscreen_target = ResourceId::new(11);

        // 1. FrameSession on canvas target 10 with registered and acquired canvas swapchain epoch
        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            CanvasId::new(10),
            canvas_target,
            800,
            600,
            CanvasFormat::Bgra8Unorm,
        );
        let canvas_output = tracker
            .begin_frame_acquire(CanvasId::new(10))
            .expect("acquire canvas");

        let root_ctx = RenderContext::new_canvas_acquired(
            canvas_target,
            800,
            600,
            Epoch::new(1),
            canvas_output.epoch,
        );
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        let mat_handle = Handle::<MaterialDomain>::new(10, NonZeroU32::new(1).unwrap());
        let red_bytes = [255u8, 0, 0, 255];
        let inner_bytes = [128u8, 128, 128, 255];
        let blue_bytes = [0u8, 0, 255, 255];

        let rec_red = session
            .snapshot_material_use(mat_handle, DataVersion::INITIAL, Epoch::new(1), &red_bytes)
            .expect("snapshot red");

        // 2. Outer pass clear plus one red draw
        session
            .begin_render_pass("main_scene", [0.0, 0.0, 0.0, 1.0])
            .expect("begin main_scene");
        session
            .record_direct_draw(1, 0, 3, Some(rec_red))
            .expect("record red draw");

        // 3. with_nested_render onto offscreen target 11 with one draw
        let nested_ctx = RenderContext::new_offscreen(offscreen_target, 256, 256, Epoch::new(2));
        let rec_inner = session
            .with_nested_render(nested_ctx, |s| {
                let rec = s
                    .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::new(2), &inner_bytes)
                    .expect("snapshot inner");
                s.begin_render_pass("offscreen_shadow", [0.1, 0.1, 0.1, 1.0])
                    .expect("begin shadow");
                s.record_direct_draw(2, 0, 6, Some(rec))
                    .expect("record inner draw");
                s.end_render_pass().expect("end shadow");
                Ok(rec)
            })
            .expect("nested render succeeds");

        // 4. Outer resume with one blue draw
        let rec_blue = session
            .snapshot_material_use(mat_handle, DataVersion::new(3), Epoch::new(1), &blue_bytes)
            .expect("snapshot blue");
        session
            .record_direct_draw(1, 0, 3, Some(rec_blue))
            .expect("record blue draw");

        // 5. Build submission packet passing the actual tracker at packet compilation
        let packet = session
            .build_submission_packet_with_tracker(Some(&tracker))
            .expect("build submission packet");

        // 6. Assert segment dependency edges: inner before resume
        let plan = session
            .pass_graph()
            .compile(Some(&tracker))
            .expect("compile plan");
        assert_eq!(plan.segment_count(), 3);
        let seg0 = &plan.segments()[0]; // main_scene prefix
        let seg1 = &plan.segments()[1]; // offscreen_shadow inner
        let seg2 = &plan.segments()[2]; // main_scene_resumed

        assert_eq!(seg0.name(), "main_scene");
        assert_eq!(seg1.name(), "offscreen_shadow");
        assert_eq!(seg2.name(), "main_scene_resumed");

        // Topological schedule must place inner (seg1) strictly before resume (seg2)
        assert!(seg1.pass_id().get() < seg2.pass_id().get());
        assert_eq!(seg0.primary_color_attachment().unwrap().target_id(), canvas_target);
        assert_eq!(seg0.primary_color_attachment().unwrap().load_op, LoadOp::Clear);
        assert_eq!(seg1.primary_color_attachment().unwrap().target_id(), offscreen_target);
        assert_eq!(seg1.primary_color_attachment().unwrap().load_op, LoadOp::Clear);
        assert_eq!(seg2.primary_color_attachment().unwrap().target_id(), canvas_target);
        assert_eq!(seg2.primary_color_attachment().unwrap().load_op, LoadOp::Load);

        // 7. Assert through lower_plan that the packet carries the prepended uniform WriteBuffer
        // and three RenderPass first-draw openers in source order with flags NEW_PASS and >0 vertices (§5.5, §8.5).
        assert_eq!(packet.commands().len(), 4);
        match &packet.commands()[0] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, DEFAULT_MATERIAL_UNIFORM_BUFFER_ID);
                assert_eq!(*offset, 0);
                let red_off = usize::try_from(rec_red.byte_offset()).expect("fixture offset fits usize");
                let inner_off = usize::try_from(rec_inner.byte_offset()).expect("fixture offset fits usize");
                let blue_off = usize::try_from(rec_blue.byte_offset()).expect("fixture offset fits usize");
                assert_eq!(&data[red_off..red_off + 4], &red_bytes);
                assert_eq!(&data[inner_off..inner_off + 4], &inner_bytes);
                assert_eq!(&data[blue_off..blue_off + 4], &blue_bytes);
            }
            other => panic!("expected WriteBuffer at index 0, got {other:?}"),
        }
        let render_passes: Vec<&GpuCommand> = packet
            .commands()
            .iter()
            .filter(|cmd| matches!(cmd, GpuCommand::RenderPass { .. }))
            .collect();
        assert_eq!(render_passes.len(), 3);

        assert_render_pass_first_draw(render_passes[0], TARGET_CANVAS, 10, LOAD_OP_CLEAR, 1, 3, rec_red.byte_offset() as u32);
        assert_render_pass_first_draw(render_passes[1], TARGET_OFFSCREEN, 11, LOAD_OP_CLEAR, 2, 6, rec_inner.byte_offset() as u32);
        assert_render_pass_first_draw(render_passes[2], TARGET_CANVAS, 10, LOAD_OP_LOAD, 1, 3, rec_blue.byte_offset() as u32);

        let pass0_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[0])).unwrap();
        let pass1_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[1])).unwrap();
        let pass2_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[2])).unwrap();

        assert_eq!(pass0_idx, 1);
        assert_eq!(pass1_idx, 2);
        assert_eq!(pass2_idx, 3);

        // 8. Assert draws keep source order: red -> inner -> blue
        assert!(rec_red.byte_offset() < rec_inner.byte_offset());
        assert!(rec_inner.byte_offset() < rec_blue.byte_offset());

        // 9. Binary wire encoding succeeds
        let encoded = packet.encode().expect("encode packet");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_canvas_output_epoch_independent_from_camera_epoch_freshness() {
        let canvas_res_id = ResourceId::new(20);
        let canvas_id = CanvasId::new(20);

        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            canvas_id,
            canvas_res_id,
            800,
            600,
            CanvasFormat::Bgra8Unorm,
        );

        // Frame 1: Camera matrix epoch is 42, canvas acquires epoch 1
        let frame1_output = tracker.begin_frame_acquire(canvas_id).expect("acquire frame 1");
        assert_eq!(frame1_output.epoch, Epoch::new(1));

        let camera_epoch = Epoch::new(42);
        let ctx_frame1 = RenderContext::new_canvas_acquired(
            canvas_res_id,
            800,
            600,
            camera_epoch,
            frame1_output.epoch,
        );
        let mut session1 = FrameSession::new(ctx_frame1, 256).expect("session 1 init");
        session1
            .begin_render_pass("canvas_pass_f1", [0.0, 0.0, 0.0, 1.0])
            .expect("begin pass 1");
        session1
            .record_direct_draw(1, 0, 3, None)
            .expect("record draw 1");

        // Positive case: Freshly acquired epoch 1 compiles successfully with tracker
        let packet1 = session1
            .build_submission_packet_with_tracker(Some(&tracker))
            .expect("compile frame 1 packet");
        assert!(!packet1.commands().is_empty());

        // Frame interval advances: Frame 2 acquires epoch 2
        // Meanwhile, camera state epoch remains unchanged (still 42)
        let frame2_output = tracker.begin_frame_acquire(canvas_id).expect("acquire frame 2");
        assert_eq!(frame2_output.epoch, Epoch::new(2));

        // Rejection case: Attempting to compile frame 1 (which captured stale epoch 1)
        // against the advanced tracker (current epoch 2) must be rejected with CanvasCachedAcrossEpochs
        let stale_res = session1.build_submission_packet_with_tracker(Some(&tracker));
        match stale_res {
            Err(FrameError::Graph(GraphError::Canvas(CanvasError::CanvasCachedAcrossEpochs {
                canvas_id: id,
                cached_epoch,
                current_epoch,
            }))) => {
                assert_eq!(id, 20);
                assert_eq!(cached_epoch, 1);
                assert_eq!(current_epoch, 2);
            }
            other => panic!("expected CanvasCachedAcrossEpochs error, got {other:?}"),
        }

        // Positive case: Fresh Frame 2 session with the same camera epoch (42) and freshly acquired canvas epoch (2)
        // compiles and lowers successfully
        let ctx_frame2 = RenderContext::new_canvas_acquired(
            canvas_res_id,
            800,
            600,
            camera_epoch,
            frame2_output.epoch,
        );
        let mut session2 = FrameSession::new(ctx_frame2, 256).expect("session 2 init");
        session2
            .begin_render_pass("canvas_pass_f2", [0.0, 0.0, 0.0, 1.0])
            .expect("begin pass 2");
        session2
            .record_direct_draw(1, 0, 3, None)
            .expect("record draw 2");

        let packet2 = session2
            .build_submission_packet_with_tracker(Some(&tracker))
            .expect("compile frame 2 packet");
        assert!(!packet2.commands().is_empty());

        // Camera epoch mutation independence: camera matrix changes from 42 to 100 without altering canvas output freshness
        let new_camera_epoch = Epoch::new(100);
        let ctx_cam_update = RenderContext::new_canvas_acquired(
            canvas_res_id,
            800,
            600,
            new_camera_epoch,
            frame2_output.epoch,
        );
        let mut session3 = FrameSession::new(ctx_cam_update, 256).expect("session 3 init");
        session3
            .begin_render_pass("canvas_pass_cam_update", [0.0, 0.0, 0.0, 1.0])
            .expect("begin pass 3");
        session3
            .record_direct_draw(1, 0, 3, None)
            .expect("record draw 3");

        let packet3 = session3
            .build_submission_packet_with_tracker(Some(&tracker))
            .expect("compile with updated camera epoch");
        assert!(!packet3.commands().is_empty());
    }

    #[test]
    fn test_canvas_packets_share_interval_until_host_expiry() {
        let canvas_res_id = ResourceId::new(30);
        let canvas_id = CanvasId::new(30);

        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            canvas_id,
            canvas_res_id,
            800,
            600,
            CanvasFormat::Bgra8Unorm,
        );

        // Frame interval 1: acquire epoch 1
        let frame1_output = tracker.begin_frame_acquire(canvas_id).expect("acquire frame 1");
        assert_eq!(frame1_output.epoch, Epoch::new(1));

        let camera_epoch = Epoch::new(42);
        let ctx1 = RenderContext::new_canvas_acquired(
            canvas_res_id,
            800,
            600,
            camera_epoch,
            frame1_output.epoch,
        );

        // Session 1 is configured with the tracker
        let mut session1 = FrameSession::new(ctx1, 256)
            .expect("session 1 init")
            .with_canvas_tracker(tracker);

        session1
            .begin_render_pass("session1_canvas_pass", [0.0, 0.0, 0.0, 1.0])
            .expect("begin pass 1");
        session1
            .record_direct_draw(1, 0, 3, None)
            .expect("record draw 1");

        // Session 1 compiles successfully with its stored tracker
        let packet1 = session1
            .build_submission_packet()
            .expect("compile session 1 packet");
        assert!(!packet1.commands().is_empty());

        // Session 2 is constructed targeting the same canvas interval and epoch
        let ctx2 = RenderContext::new_canvas_acquired(
            canvas_res_id,
            800,
            600,
            camera_epoch,
            frame1_output.epoch,
        );
        let mut session2 = FrameSession::new(ctx2, 256).expect("session 2 init");
        session2
            .begin_render_pass("session2_canvas_pass", [0.0, 0.0, 0.0, 1.0])
            .expect("begin pass 2");
        session2
            .record_direct_draw(1, 0, 3, None)
            .expect("record draw 2");

        // A second packet is legal in the same interval. A native queue.submit
        // does not seal this tracker (actual multiple-submit pixels are checked
        // in tests/fixtures/gpu_bridge/canvas_lifetime_test.js).
        let packet2 = session2
            .build_submission_packet_with_tracker(session1.canvas_tracker())
            .expect("second packet in the same interval");
        assert!(!packet2.commands().is_empty());

        session1
            .end_canvas_interval(canvas_id, frame1_output.epoch)
            .expect("host expires the acquired canvas texture");

        // Only explicit host expiry rejects further packets against this epoch.
        let stale_res = session2.build_submission_packet_with_tracker(session1.canvas_tracker());
        match stale_res {
            Err(FrameError::Graph(GraphError::Canvas(CanvasError::CanvasIntervalEnded {
                canvas_id: id,
                epoch: ended_epoch,
            }))) => {
                assert_eq!(id, 30);
                assert_eq!(ended_epoch, 1);
            }
            other => panic!("expected CanvasIntervalEnded error, got {other:?}"),
        }

        // Advance to a new frame interval via canvas_tracker_mut: acquire epoch 2
        let tracker_mut = session1.canvas_tracker_mut().expect("tracker present");
        let frame2_output = tracker_mut.begin_frame_acquire(canvas_id).expect("acquire frame 2");
        assert_eq!(frame2_output.epoch, Epoch::new(2));

        // Session 3 constructed for the new interval compiles successfully
        let ctx3 = RenderContext::new_canvas_acquired(
            canvas_res_id,
            800,
            600,
            camera_epoch,
            frame2_output.epoch,
        );
        let mut session3 = FrameSession::new(ctx3, 256).expect("session 3 init");
        session3
            .begin_render_pass("session3_canvas_pass", [0.0, 0.0, 0.0, 1.0])
            .expect("begin pass 3");
        session3
            .record_direct_draw(1, 0, 3, None)
            .expect("record draw 3");

        let packet3 = session3
            .build_submission_packet_with_tracker(session1.canvas_tracker())
            .expect("new epoch compiles again");
        assert!(!packet3.commands().is_empty());
    }

    #[test]
    fn test_canvas_nested_render_resumes_outer_canvas_pass_on_callback_error_with_tracker() {
        let canvas_target = ResourceId::new(10);
        let offscreen_target = ResourceId::new(11);
        let canvas_id = CanvasId::new(10);

        // 1. Tracker-acquired canvas swapchain epoch
        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            canvas_id,
            canvas_target,
            800,
            600,
            CanvasFormat::Bgra8Unorm,
        );
        let canvas_output = tracker
            .begin_frame_acquire(canvas_id)
            .expect("acquire canvas");
        assert_eq!(canvas_output.epoch, Epoch::new(1));

        let camera_epoch = Epoch::new(1);
        let root_ctx = RenderContext::new_canvas_acquired(
            canvas_target,
            800,
            600,
            camera_epoch,
            canvas_output.epoch,
        );
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        let mat_handle = Handle::<MaterialDomain>::new(10, NonZeroU32::new(1).unwrap());
        let red_bytes = [255u8, 0, 0, 255];
        let inner_bytes = [0u8, 255, 0, 255];
        let blue_bytes = [0u8, 0, 255, 255];

        let rec_red = session
            .snapshot_material_use(mat_handle, DataVersion::INITIAL, camera_epoch, &red_bytes)
            .expect("snapshot red");

        // 2. Outer canvas pass with red draw
        session
            .begin_render_pass("canvas_main", [0.0, 0.0, 0.0, 1.0])
            .expect("begin canvas_main");
        session
            .record_direct_draw(1, 0, 3, Some(rec_red))
            .expect("record red draw");

        // 3. with_nested_render onto offscreen 11 that commits its inner pass then returns Err
        let nested_ctx = RenderContext::new_offscreen(offscreen_target, 256, 256, Epoch::new(2));
        let mut rec_inner_opt = None;
        let res: Result<(), FrameError> = session.with_nested_render(nested_ctx, |s| {
            let rec = s
                .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::new(2), &inner_bytes)
                .expect("snapshot inner");
            rec_inner_opt = Some(rec);
            s.begin_render_pass("offscreen_shadow", [0.1, 0.1, 0.1, 1.0])
                .expect("begin inner pass");
            s.record_direct_draw(2, 0, 6, rec_inner_opt)
                .expect("record inner draw");
            s.end_render_pass().expect("commit inner pass");
            Err(FrameError::InvalidNestingState {
                detail: String::from("simulated nested callback failure after inner commit"),
            })
        });
        assert!(res.is_err());
        let rec_inner = rec_inner_opt.expect("inner record snapshotted");

        // 4. Assert outer canvas pass is resumed with LoadOp::Load and the same canvas_output_epoch
        assert!(session.active_pass.is_some());
        let active = session.active_pass.as_ref().unwrap();
        assert_eq!(active.name, "canvas_main_resumed");
        assert_eq!(active.color_attachment.target_id, canvas_target);
        assert_eq!(active.color_attachment.load_op, LoadOp::Load);
        assert_eq!(active.color_attachment.canvas_epoch, Some(canvas_output.epoch));
        assert_eq!(
            session.context_stack().current().canvas_output_epoch,
            Some(canvas_output.epoch)
        );

        // Record blue draw into resumed canvas pass
        let rec_blue = session
            .snapshot_material_use(mat_handle, DataVersion::new(3), camera_epoch, &blue_bytes)
            .expect("snapshot blue");
        session
            .record_direct_draw(1, 0, 3, Some(rec_blue))
            .expect("record blue draw in resumed pass");

        // 5. Assert build_submission_packet_with_tracker still compiles against the tracker (fresh epoch)
        let packet = session
            .build_submission_packet_with_tracker(Some(&tracker))
            .expect("compile submission packet with tracker");

        // 6. Assert lowered openers are canvas clear, offscreen clear, canvas load in source order
        // with flags NEW_PASS and >0 vertices, preceded by the uniform WriteBuffer (§5.5, §8.5).
        assert_eq!(packet.commands().len(), 4);
        match &packet.commands()[0] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, DEFAULT_MATERIAL_UNIFORM_BUFFER_ID);
                assert_eq!(*offset, 0);
                let red_off = usize::try_from(rec_red.byte_offset()).expect("fixture offset fits usize");
                let inner_off = usize::try_from(rec_inner.byte_offset()).expect("fixture offset fits usize");
                let blue_off = usize::try_from(rec_blue.byte_offset()).expect("fixture offset fits usize");
                assert_eq!(&data[red_off..red_off + 4], &red_bytes);
                assert_eq!(&data[inner_off..inner_off + 4], &inner_bytes);
                assert_eq!(&data[blue_off..blue_off + 4], &blue_bytes);
            }
            other => panic!("expected WriteBuffer at index 0, got {other:?}"),
        }
        let render_passes: Vec<&GpuCommand> = packet
            .commands()
            .iter()
            .filter(|cmd| matches!(cmd, GpuCommand::RenderPass { .. }))
            .collect();
        assert_eq!(render_passes.len(), 3);

        assert_render_pass_first_draw(render_passes[0], TARGET_CANVAS, 10, LOAD_OP_CLEAR, 1, 3, rec_red.byte_offset() as u32);
        assert_render_pass_first_draw(render_passes[1], TARGET_OFFSCREEN, 11, LOAD_OP_CLEAR, 2, 6, rec_inner.byte_offset() as u32);
        assert_render_pass_first_draw(render_passes[2], TARGET_CANVAS, 10, LOAD_OP_LOAD, 1, 3, rec_blue.byte_offset() as u32);

        let pass0_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[0])).unwrap();
        let pass1_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[1])).unwrap();
        let pass2_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[2])).unwrap();

        assert_eq!(pass0_idx, 1);
        assert_eq!(pass1_idx, 2);
        assert_eq!(pass2_idx, 3);
    }

    fn assert_render_pass_first_draw(
        cmd: &GpuCommand,
        expected_type: u32,
        expected_id: u32,
        expected_load: u32,
        expected_pipeline: u32,
        expected_vertex_count: u32,
        expected_offset: u32,
    ) {
        match cmd {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                load_op,
                store_op,
                pass_flags,
                pipeline_id,
                vertex_count,
                uniform_dynamic_offset,
                ..
            } => {
                assert_eq!(*target_type, expected_type);
                assert_eq!(*target_id, expected_id);
                assert_eq!(*load_op, expected_load);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*pipeline_id, expected_pipeline);
                assert_eq!(*vertex_count, expected_vertex_count);
                assert_eq!(*uniform_dynamic_offset, expected_offset);
            }
            other => panic!("expected RenderPass first-draw opener, got {other:?}"),
        }
    }

    #[test]
    fn test_canvas_nested_render_resumes_outer_canvas_pass_on_ok_with_tracker() {
        let canvas_target = ResourceId::new(10);
        let offscreen_target = ResourceId::new(11);
        let canvas_id = CanvasId::new(10);

        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            canvas_id,
            canvas_target,
            800,
            600,
            CanvasFormat::Bgra8Unorm,
        );
        let canvas_output = tracker.begin_frame_acquire(canvas_id).expect("acquire canvas");
        let camera_epoch = Epoch::new(1);
        let root_ctx = RenderContext::new_canvas_acquired(
            canvas_target,
            800,
            600,
            camera_epoch,
            canvas_output.epoch,
        );
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        let mat_handle = Handle::<MaterialDomain>::new(10, NonZeroU32::new(1).unwrap());
        let rec_red = session
            .snapshot_material_use(mat_handle, DataVersion::INITIAL, camera_epoch, &[255, 0, 0, 255])
            .expect("snapshot red");

        session.begin_render_pass("canvas_main", [0.0, 0.0, 0.0, 1.0]).expect("begin canvas_main");
        session.record_direct_draw(1, 0, 3, Some(rec_red)).expect("record red draw");

        // with_nested_render on offscreen 11 returns Ok
        let nested_ctx = RenderContext::new_offscreen(offscreen_target, 256, 256, Epoch::new(2));
        let rec_inner = session
            .with_nested_render(nested_ctx, |s| {
                let rec = s
                    .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::new(2), &[0, 255, 0, 255])
                    .expect("snapshot inner");
                s.begin_render_pass("offscreen_shadow", [0.1, 0.1, 0.1, 1.0]).expect("begin shadow");
                s.record_direct_draw(2, 0, 6, Some(rec)).expect("record inner draw");
                s.end_render_pass().expect("commit shadow");
                Ok(rec)
            })
            .expect("nested render succeeds");

        // Assert outer canvas pass resumes with LoadOp::Load and preserved epoch
        assert!(session.active_pass.is_some());
        let active = session.active_pass.as_ref().unwrap();
        assert_eq!(active.name, "canvas_main_resumed");
        assert_eq!(active.color_attachment.target_id, canvas_target);
        assert_eq!(active.color_attachment.load_op, LoadOp::Load);
        assert_eq!(active.color_attachment.canvas_epoch, Some(canvas_output.epoch));
        assert_eq!(session.context_stack().current().canvas_output_epoch, Some(canvas_output.epoch));

        let rec_blue = session
            .snapshot_material_use(mat_handle, DataVersion::new(3), camera_epoch, &[0, 0, 255, 255])
            .expect("snapshot blue");
        session.record_direct_draw(1, 0, 3, Some(rec_blue)).expect("record blue draw");

        // Compile with tracker yields 1 WriteBuffer and 3 first-draw openers: CANVAS CLEAR NEW_PASS / OFFSCREEN CLEAR NEW_PASS / CANVAS LOAD NEW_PASS
        let packet = session
            .build_submission_packet_with_tracker(Some(&tracker))
            .expect("compile submission packet with tracker");
        assert_eq!(packet.commands().len(), 4);
        match &packet.commands()[0] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, DEFAULT_MATERIAL_UNIFORM_BUFFER_ID);
                assert_eq!(*offset, 0);
                let red_off = usize::try_from(rec_red.byte_offset()).expect("fixture offset fits usize");
                let inner_off = usize::try_from(rec_inner.byte_offset()).expect("fixture offset fits usize");
                let blue_off = usize::try_from(rec_blue.byte_offset()).expect("fixture offset fits usize");
                assert_eq!(&data[red_off..red_off + 4], &[255, 0, 0, 255]);
                assert_eq!(&data[inner_off..inner_off + 4], &[0, 255, 0, 255]);
                assert_eq!(&data[blue_off..blue_off + 4], &[0, 0, 255, 255]);
            }
            other => panic!("expected WriteBuffer at index 0, got {other:?}"),
        }
        let render_passes: Vec<&GpuCommand> = packet
            .commands()
            .iter()
            .filter(|cmd| matches!(cmd, GpuCommand::RenderPass { .. }))
            .collect();
        assert_eq!(render_passes.len(), 3);

        assert_render_pass_first_draw(render_passes[0], TARGET_CANVAS, 10, LOAD_OP_CLEAR, 1, 3, rec_red.byte_offset() as u32);
        assert_render_pass_first_draw(render_passes[1], TARGET_OFFSCREEN, 11, LOAD_OP_CLEAR, 2, 6, rec_inner.byte_offset() as u32);
        assert_render_pass_first_draw(render_passes[2], TARGET_CANVAS, 10, LOAD_OP_LOAD, 1, 3, rec_blue.byte_offset() as u32);

        let pass0_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[0])).unwrap();
        let pass1_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[1])).unwrap();
        let pass2_idx = packet.commands().iter().position(|cmd| std::ptr::eq(cmd, render_passes[2])).unwrap();

        assert_eq!(pass0_idx, 1);
        assert_eq!(pass1_idx, 2);
        assert_eq!(pass2_idx, 3);
    }

    #[test]
    fn test_nested_render_viewport_and_scissor_preservation_and_restoration() {
        let canvas_target = ResourceId::new(10);
        let offscreen_target = ResourceId::new(11);
        let canvas_id = CanvasId::new(10);

        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            canvas_id,
            canvas_target,
            800,
            600,
            CanvasFormat::Bgra8Unorm,
        );
        let canvas_output = tracker.begin_frame_acquire(canvas_id).expect("acquire canvas");
        let camera_epoch = Epoch::new(1);
        let root_ctx = RenderContext::new_canvas_acquired(
            canvas_target,
            800,
            600,
            camera_epoch,
            canvas_output.epoch,
        )
        .with_viewport(0, 0, 800, 600)
        .with_scissor(0, 0, 400, 600)
        .with_scissor_test(true);

        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        // 1. Begin outer pass and record outer prefix draw (left half scissor)
        session.begin_render_pass("canvas_outer", [0.0, 0.0, 0.0, 1.0]).expect("begin outer");
        let outer_draw_id = session.record_direct_draw(1, 0, 3, None).expect("record outer draw");

        // 2. Execute with_nested_render targeting offscreen texture 11 with distinct viewport and scissor
        let nested_ctx = RenderContext::new_offscreen(offscreen_target, 256, 256, Epoch::new(2))
            .with_viewport(0, 0, 256, 256)
            .with_scissor(32, 32, 128, 128)
            .with_scissor_test(true);

        let nested_draw_id = session
            .with_nested_render(nested_ctx, |s| {
                assert_eq!(s.viewport(), Some([0, 0, 256, 256]));
                assert_eq!(s.scissor(), Some([32, 32, 128, 128]));
                assert!(s.scissor_test_enabled());

                s.begin_render_pass("offscreen_nested", [0.2, 0.2, 0.2, 1.0]).expect("begin nested");
                let id = s.record_direct_draw(2, 0, 6, None).expect("record nested draw");
                s.end_render_pass().expect("commit nested pass");
                Ok(id)
            })
            .expect("nested render ok");

        // 3. Resumed outer pass is open: verify outer context viewport/scissor were restored
        assert_eq!(session.viewport(), Some([0, 0, 800, 600]));
        assert_eq!(session.scissor(), Some([0, 0, 400, 600]));
        assert!(session.scissor_test_enabled());

        // Update scissor to right half for resumed draw
        session.set_scissor(400, 0, 400, 600);
        let resumed_draw_id = session.record_direct_draw(1, 0, 3, None).expect("record resumed draw");

        // End outer resumed pass so it is committed to pass_graph before compilation
        session.end_render_pass().expect("commit outer resumed pass");

        // 4. Verify pass graph compilation preserves distinct viewport/scissor on each draw
        let plan = session.pass_graph.compile(Some(&tracker)).expect("compile plan");
        assert_eq!(plan.segment_count(), 3);

        // Segment 0: outer prefix draw
        let seg0_draws = plan.segments()[0].draws();
        assert_eq!(seg0_draws.len(), 1);
        assert_eq!(seg0_draws[0].draw_id, outer_draw_id);
        assert_eq!(seg0_draws[0].viewport(), Some([0, 0, 800, 600]));
        assert_eq!(seg0_draws[0].scissor(), Some([0, 0, 400, 600]));
        assert!(seg0_draws[0].scissor_test_enabled());

        // Segment 1: nested offscreen draw
        let seg1_draws = plan.segments()[1].draws();
        assert_eq!(seg1_draws.len(), 1);
        assert_eq!(seg1_draws[0].draw_id, nested_draw_id);
        assert_eq!(seg1_draws[0].viewport(), Some([0, 0, 256, 256]));
        assert_eq!(seg1_draws[0].scissor(), Some([32, 32, 128, 128]));
        assert!(seg1_draws[0].scissor_test_enabled());

        // Segment 2: resumed outer draw
        let seg2_draws = plan.segments()[2].draws();
        assert_eq!(seg2_draws.len(), 1);
        assert_eq!(seg2_draws[0].draw_id, resumed_draw_id);
        assert_eq!(seg2_draws[0].viewport(), Some([0, 0, 800, 600]));
        assert_eq!(seg2_draws[0].scissor(), Some([400, 0, 400, 600]));
        assert!(seg2_draws[0].scissor_test_enabled());
    }

    #[test]
    fn test_nested_render_viewport_scissor_snapshot_not_overwritten_by_subsequent_mutation() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO)
            .with_viewport(0, 0, 800, 600)
            .with_scissor(10, 20, 100, 200)
            .with_scissor_test(true);
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        session.begin_render_pass("main", [0.0, 0.0, 0.0, 1.0]).expect("begin pass");
        let draw_a_id = session.record_direct_draw(1, 0, 3, None).expect("draw A");

        // Mutate context AFTER recording draw A
        session.set_viewport(999, 999, 50, 50);
        session.set_scissor(888, 888, 40, 40);
        session.set_scissor_test(false);

        let draw_b_id = session.record_direct_draw(2, 0, 6, None).expect("draw B");

        // Mutate context again AFTER recording draw B
        session.set_viewport(111, 222, 333, 444);
        session.set_scissor(555, 666, 777, 888);
        session.set_scissor_test(true);

        session.end_render_pass().expect("end pass");

        let plan = session.pass_graph.compile(None).expect("compile plan");
        let draws = plan.segments()[0].draws();
        assert_eq!(draws.len(), 2);

        // Draw A must have preserved its snapshot from when it was recorded
        assert_eq!(draws[0].draw_id, draw_a_id);
        assert_eq!(draws[0].viewport(), Some([0, 0, 800, 600]));
        assert_eq!(draws[0].scissor(), Some([10, 20, 100, 200]));
        assert!(draws[0].scissor_test_enabled());

        // Draw B must have preserved its snapshot from when it was recorded
        assert_eq!(draws[1].draw_id, draw_b_id);
        assert_eq!(draws[1].viewport(), Some([999, 999, 50, 50]));
        // Scissor disabled after enable provides full-attachment scissor to clear clipping
        assert_eq!(draws[1].scissor(), Some([0, 0, 800, 600]));
        assert!(!draws[1].scissor_test_enabled());
    }

    #[test]
    fn test_viewport_near_neighbor_default_full_vs_explicit_zero_size() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        session.begin_render_pass("viewport_pass", [0.0, 0.0, 0.0, 1.0]).expect("begin");

        // Draw 0: default unspecified viewport (None)
        let id0 = session.record_direct_draw(1, 0, 3, None).expect("draw 0");
        assert_eq!(session.viewport(), None);

        // Draw 1: explicit zero-size viewport (Some([0, 0, 0, 0])) - near-neighbor case
        session.set_viewport(0, 0, 0, 0);
        let id1 = session.record_direct_draw(1, 0, 3, None).expect("draw 1");
        assert_eq!(session.viewport(), Some([0, 0, 0, 0]));

        // Draw 2: explicit non-zero viewport (Some([0, 0, 800, 600]))
        session.set_viewport(0, 0, 800, 600);
        let id2 = session.record_direct_draw(1, 0, 3, None).expect("draw 2");
        assert_eq!(session.viewport(), Some([0, 0, 800, 600]));

        // Draw 3: reset to default viewport (None on session CPU metadata, explicit full target on emitted draw to clear prior narrow GPU state)
        session.reset_viewport();
        let id3 = session.record_direct_draw(1, 0, 3, None).expect("draw 3");
        assert_eq!(session.viewport(), None);

        session.end_render_pass().expect("end");

        let plan = session.pass_graph.compile(None).expect("compile");
        let draws = plan.segments()[0].draws();
        assert_eq!(draws.len(), 4);

        assert_eq!(draws[0].draw_id, id0);
        assert_eq!(draws[0].viewport(), None);

        assert_eq!(draws[1].draw_id, id1);
        assert_eq!(draws[1].viewport(), Some([0, 0, 0, 0]));
        assert_ne!(draws[0].viewport(), draws[1].viewport());

        assert_eq!(draws[2].draw_id, id2);
        assert_eq!(draws[2].viewport(), Some([0, 0, 800, 600]));

        assert_eq!(draws[3].draw_id, id3);
        assert_eq!(draws[3].viewport(), Some([0, 0, 800, 600]));
        assert_ne!(draws[1].viewport(), draws[3].viewport());
    }

    #[test]
    fn test_scissor_disable_after_enable_restores_full_attachment_scissor() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(1), 1024, 768, Epoch::ZERO);
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        session.begin_render_pass("scissor_pass", [0.0, 0.0, 0.0, 1.0]).expect("begin");

        // Draw 0: scissor never enabled in pass -> None
        let id0 = session.record_direct_draw(1, 0, 3, None).expect("draw 0");
        assert_eq!(session.scissor(), None);
        assert!(!session.scissor_test_enabled());

        // Draw 1: scissor enabled with clipping box
        session.set_scissor(50, 60, 200, 300);
        session.set_scissor_test(true);
        let id1 = session.record_direct_draw(1, 0, 3, None).expect("draw 1");
        assert_eq!(session.scissor(), Some([50, 60, 200, 300]));
        assert!(session.scissor_test_enabled());

        // Draw 2: scissor disabled after enable -> emits full attachment [0, 0, 1024, 768]
        session.set_scissor_test(false);
        let id2 = session.record_direct_draw(1, 0, 3, None).expect("draw 2");
        assert!(!session.scissor_test_enabled());

        session.end_render_pass().expect("end");

        let plan = session.pass_graph.compile(None).expect("compile");
        let draws = plan.segments()[0].draws();
        assert_eq!(draws.len(), 3);

        // Draw 0: never enabled
        assert_eq!(draws[0].draw_id, id0);
        assert_eq!(draws[0].scissor(), None);
        assert!(!draws[0].scissor_test_enabled());

        // Draw 1: actively clipped
        assert_eq!(draws[1].draw_id, id1);
        assert_eq!(draws[1].scissor(), Some([50, 60, 200, 300]));
        assert!(draws[1].scissor_test_enabled());

        // Draw 2: disabled, carries full-attachment restore rect to clear clipping
        assert_eq!(draws[2].draw_id, id2);
        assert_eq!(draws[2].scissor(), Some([0, 0, 1024, 768]));
        assert!(!draws[2].scissor_test_enabled());
        assert_ne!(draws[0].scissor(), draws[2].scissor());
    }

    #[test]
    fn test_scissor_narrow_then_reset_while_enabled_then_disabled_after_reset() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(1), 1024, 768, Epoch::ZERO);
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        session.begin_render_pass("scissor_leak_pass", [0.0, 0.0, 0.0, 1.0]).expect("begin");

        // 1. Draw 0: set narrow scissor, enable scissor test, draw
        session.set_scissor(50, 60, 200, 300);
        session.set_scissor_test(true);
        let id0 = session.record_direct_draw(1, 0, 3, None).expect("draw 0");
        assert_eq!(session.scissor(), Some([50, 60, 200, 300]));
        assert!(session.scissor_test_enabled());

        // 2. Draw 1: reset scissor while enabled -> must resolve to full-attachment rectangle to clear narrow clipping
        session.reset_scissor();
        assert_eq!(session.scissor(), None); // CPU metadata reflects reset
        assert!(session.scissor_test_enabled()); // scissor test remains enabled
        let id1 = session.record_direct_draw(1, 0, 3, None).expect("draw 1");

        // 3. Draw 2: disable scissor test after reset -> must also carry full-attachment restore rect so lower_plan clears clipping
        session.set_scissor_test(false);
        assert!(!session.scissor_test_enabled());
        let id2 = session.record_direct_draw(1, 0, 3, None).expect("draw 2");

        // 4. Draw 3: subsequent draw while disabled -> lower_plan dedups and emits no scissor commands
        let id3 = session.record_direct_draw(1, 0, 3, None).expect("draw 3");

        session.end_render_pass().expect("end");

        // Verify Draw descriptors in compiled plan
        let plan = session.pass_graph.compile(None).expect("compile plan");
        let draws = plan.segments()[0].draws();
        assert_eq!(draws.len(), 4);

        assert_eq!(draws[0].draw_id, id0);
        assert_eq!(draws[0].scissor(), Some([50, 60, 200, 300]));
        assert!(draws[0].scissor_test_enabled());

        assert_eq!(draws[1].draw_id, id1);
        assert_eq!(draws[1].scissor(), Some([0, 0, 1024, 768]));
        assert!(draws[1].scissor_test_enabled());

        assert_eq!(draws[2].draw_id, id2);
        assert_eq!(draws[2].scissor(), Some([0, 0, 1024, 768]));
        assert!(!draws[2].scissor_test_enabled());

        assert_eq!(draws[3].draw_id, id3);
        assert_eq!(draws[3].scissor(), Some([0, 0, 1024, 768]));
        assert!(!draws[3].scissor_test_enabled());

        // Verify lowered commands emitted by lower_plan
        let commands = lower_plan(&plan).expect("lower plan");
        let scissor_commands: Vec<&GpuCommand> = commands
            .iter()
            .filter(|cmd| matches!(cmd, GpuCommand::SetScissorRect { .. }))
            .collect();

        // Must emit SetScissorRect for narrow box, then SetScissorRect for full attachment box,
        // then SetScissorRect on disable, and no further redundant scissor commands once disabled (deduped).
        assert_eq!(scissor_commands.len(), 3);
        match scissor_commands[0] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 50);
                assert_eq!(*y, 60);
                assert_eq!(*width, 200);
                assert_eq!(*height, 300);
            }
            other => panic!("expected narrow SetScissorRect, got {other:?}"),
        }
        match scissor_commands[1] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
                assert_eq!(*width, 1024);
                assert_eq!(*height, 768);
            }
            other => panic!("expected full SetScissorRect on reset, got {other:?}"),
        }
        match scissor_commands[2] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
                assert_eq!(*width, 1024);
                assert_eq!(*height, 768);
            }
            other => panic!("expected full SetScissorRect on disable, got {other:?}"),
        }
    }

    #[test]
    fn test_record_direct_draw_with_range_and_opcode11_lowering() {
        let root_ctx = RenderContext::new_canvas(ResourceId::new(1), 800, 600, Epoch::ZERO);
        let mut session = FrameSession::new(root_ctx, 256).expect("session init");

        session
            .begin_render_pass("range_pass", [0.0, 0.0, 0.0, 1.0])
            .expect("begin pass");

        // 1. Draw 0: standard record_direct_draw (delegates to [count, 1, 0, 0])
        let id0 = session
            .record_direct_draw(100, 2, 3, None)
            .expect("record direct draw default");

        // 2. Draw 1: record_direct_draw_with_range with nonzero starts and two instances [3, 2, 3, 5]
        let id1 = session
            .record_direct_draw_with_range(100, 2, [3, 2, 3, 5], None)
            .expect("record direct draw with range");

        // 3. Draw 2: record_direct_draw_with_range with zero instances [4, 0, 0, 0] (negative zero retained)
        let id2 = session
            .record_direct_draw_with_range(100, 2, [4, 0, 0, 0], None)
            .expect("record direct draw zero instances");

        session.end_render_pass().expect("end pass");

        // Assert compiled Graph Draw records
        let plan = session.pass_graph.compile(None).expect("compile plan");
        let draws = plan.segments()[0].draws();
        assert_eq!(draws.len(), 3);

        // Draw 0: default shape (vertex_count: 3, instance_count: 1, first_vertex: 0, first_instance: 0)
        assert_eq!(draws[0].draw_id, id0);
        assert_eq!(draws[0].vertex_count, 3);
        assert_eq!(draws[0].instance_count(), 1);
        assert_eq!(draws[0].first_vertex(), 0);
        assert_eq!(draws[0].first_instance(), 0);

        // Draw 1: positive parameters preserved in Graph
        assert_eq!(draws[1].draw_id, id1);
        assert_eq!(draws[1].vertex_count, 3);
        assert_eq!(draws[1].instance_count(), 2);
        assert_eq!(draws[1].first_vertex(), 3);
        assert_eq!(draws[1].first_instance(), 5);

        // Draw 2: zero instance retained in Graph
        assert_eq!(draws[2].draw_id, id2);
        assert_eq!(draws[2].vertex_count, 4);
        assert_eq!(draws[2].instance_count(), 0);
        assert_eq!(draws[2].first_vertex(), 0);
        assert_eq!(draws[2].first_instance(), 0);

        // Assert lowered GpuCommands through lower_plan
        let commands = lower_plan(&plan).expect("lower plan");

        // Draw 0: default range [3, 1, 0, 0] does not emit SetDrawParameters (first draw directly opens pass)
        // Draw 1: non-default range [3, 2, 3, 5] emits SetDrawParameters (opcode 11) followed by RenderPass draw
        // Draw 2: zero instance [4, 0, 0, 0] emits SetDrawParameters (opcode 11) followed by RenderPass draw
        let draw_param_cmds: Vec<&GpuCommand> = commands
            .iter()
            .filter(|cmd| matches!(cmd, GpuCommand::SetDrawParameters { .. }))
            .collect();
        assert_eq!(draw_param_cmds.len(), 2);

        // First SetDrawParameters is for Draw 1
        match draw_param_cmds[0] {
            GpuCommand::SetDrawParameters {
                instance_count,
                first_vertex,
                first_instance,
            } => {
                assert_eq!(*instance_count, 2);
                assert_eq!(*first_vertex, 3);
                assert_eq!(*first_instance, 5);
            }
            other => panic!("expected SetDrawParameters for draw 1, got {other:?}"),
        }

        // Second SetDrawParameters is for Draw 2 (zero instance preserved)
        match draw_param_cmds[1] {
            GpuCommand::SetDrawParameters {
                instance_count,
                first_vertex,
                first_instance,
            } => {
                assert_eq!(*instance_count, 0);
                assert_eq!(*first_vertex, 0);
                assert_eq!(*first_instance, 0);
            }
            other => panic!("expected SetDrawParameters for draw 2, got {other:?}"),
        }

        // Verify full command sequence order:
        // 0: RenderPass (opener for Draw 0, vertex_count: 3, PASS_FLAG_NEW_PASS)
        // 1: SetDrawParameters (for Draw 1: [2, 3, 5])
        // 2: RenderPass (Draw 1, vertex_count: 3, PASS_FLAG_NONE)
        // 3: SetDrawParameters (for Draw 2: [0, 0, 0])
        // 4: RenderPass (Draw 2, vertex_count: 4, PASS_FLAG_NONE)
        assert_eq!(commands.len(), 5);
        assert!(matches!(
            commands[0],
            GpuCommand::RenderPass {
                vertex_count: 3,
                pass_flags: PASS_FLAG_NEW_PASS,
                ..
            }
        ));
        assert!(matches!(
            commands[1],
            GpuCommand::SetDrawParameters {
                instance_count: 2,
                first_vertex: 3,
                first_instance: 5,
            }
        ));
        assert!(matches!(
            commands[2],
            GpuCommand::RenderPass {
                vertex_count: 3,
                pass_flags: PASS_FLAG_NONE,
                ..
            }
        ));
        assert!(matches!(
            commands[3],
            GpuCommand::SetDrawParameters {
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }
        ));
        assert!(matches!(
            commands[4],
            GpuCommand::RenderPass {
                vertex_count: 4,
                pass_flags: PASS_FLAG_NONE,
                ..
            }
        ));
    }
}
