//! First-frame WebGPU host bridge, negotiation, checked packet submission,
//! and error-scope discipline.
//!
//! # Architecture & Guarantees
//! - **Single Asupersync Runtime**: Polled futures and task ownership remain in the
//!   single application runtime; no secondary executor.
//! - **Pre-Device Negotiation**: Required features and limits are validated against
//!   the adapter before `requestDevice`. Missing required capabilities fail
//!   cleanly and immediately with structured errors.
//! - **Error-Scope Discipline**: Scope discipline is enforced in `bridge_runtime.js`
//!   only (via `WebGpuBridgeHost::withErrorScopes`). `ErrorScopeTracker` provides
//!   an audited reference model for scope nesting and concurrency serialization.
//! - **Canvas Texture Freshness**: Canvas swapchain textures are acquired fresh each
//!   render interval and never cached across frames.
//! - **Queue Ordering & Snapshot Semantics**: Per-use versioned slices or separate
//!   buffers are used so that multiple draws in a single submission (such as Red-A / Blue-B)
//!   remain distinct without queue-write hazard overwrites.
//! - **Coarse Checked Packet**: High-throughput binary command stream with validated
//!   offsets and bounds. No `eval` and no `new Function`.
//!
//! # Scope & Limitations (§5.1, NO-CLAIM)
//! - **Readback Publication Gate**: The readback publication gate ([`ReadbackPublicationState`]
//!   and [`try_publish_readback`]) is exposed to browser callers as [`gpu_bridge_try_publish_readback`].
//!   The browser bridge (`bridge_runtime.js`) decodes and attaches `epochHi` and `epochLo` words onto readback
//!   buffers. JavaScript callers supply the readback epoch words alongside the region's current epoch words to
//!   `gpu_bridge_try_publish_readback`, which reconstructs both [`Epoch`] values using [`Epoch::from_words`]
//!   and returns the product freshness gate decision without reimplementing the logic in JavaScript.
//! - **Render Bundle Execution & State Invalidation (§8.5, 2v8.2)**: Pre-recorded bundles
//!   are recorded via `OPCODE_RECORD_BUNDLE` and executed via `OPCODE_EXECUTE_BUNDLES`.
//!   In accordance with the WebGPU specification, executing bundles clears render pass state
//!   (pipeline, bind groups, vertex buffers), mandating explicit state rebinding before any
//!   subsequent direct draw in the same pass.
//! - **Generational Handle Slot Table & Publication Gate (§6.5, vqa.6)**: GPU resource identities
//!   are tracked in a generational slot table backed by [`f3d_core::Handle`]. When slots are released
//!   and reused, their generation counter advances. Stale handles from prior incarnations are rejected
//!   on verification ([`gpu_bridge_check_resource_handle`]), preventing ABA publication defects.
//!   Zero-valued generations are strictly rejected via [`HandleError::InvalidGeneration`].
//! - **Linear Memory Borrow Guards (§6.6, §13.1, vqa.6)**: Active borrows of WebAssembly linear memory
//!   are guarded by [`f3d_core::ownership::BorrowScope`]. Attempting to grow memory while a borrow is active
//!   is strictly blocked ([`gpu_bridge_try_grow_memory`]), and exiting requires the exact matching [`BorrowToken`].
//! - **AffineRows Wire Layout Validation (§6.1, §6.2, vqa.6)**: Host and bridge matrices are verified
//!   against [`f3d_core::layout::AffineRows::from_column_major`]. Matrices containing perspective components
//!   or invalid row 3 values are rejected ([`gpu_bridge_validate_affine_rows`]).
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::num::NonZeroU32;
use std::sync::Mutex;

use f3d_core::{
    error::HandleError,
    handle::{Domain, Handle, MaterialDomain},
    layout::{
        AFFINE_ROWS_BYTES, AffineRows, COLOR_UNIFORM_BYTES,
        DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT, LayoutError, VERTEX_POS_UV_STRIDE,
        VertexPosUv, WGSL_AFFINE_ROWS_DECLARATION,
    },
    ownership::{
        BorrowScope, BorrowToken, DataVersion, Epoch, OwnershipError, PerUseByteBuffer, RegionState,
    },
};
use f3d_graph::{
    CanvasEpochTracker, CanvasFormat, CanvasId, DrawKind, ExecutionPlan, LoadOp, PassKind,
    ResourceAccess, ResourceId, StoreOp,
};
use f3d_math::{Matrix4, Quaternion, Vector3};

use crate::frame::{FrameSession, RenderContext};

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

/// Device limits profile required by the renderer or bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimitsRecord {
    /// Maximum 2D texture dimension in pixels.
    pub max_texture_dimension_2d: u32,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: u64,
    /// Maximum number of active bind groups.
    pub max_bind_groups: u32,
    /// Minimum required alignment for uniform buffer dynamic offsets.
    pub min_uniform_buffer_offset_alignment: u32,
    /// Minimum required alignment for storage buffer dynamic offsets.
    pub min_storage_buffer_offset_alignment: u32,
}

impl Default for DeviceLimitsRecord {
    fn default() -> Self {
        Self {
            max_texture_dimension_2d: 8192,
            max_buffer_size: 268435456, // 256 MiB
            max_bind_groups: 4,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
        }
    }
}

/// Minimum requirements profile negotiated before device creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuRequiredProfile {
    /// List of required WebGPU feature names.
    pub required_features: Vec<String>,
    /// Minimum device limits required by the workload.
    pub min_limits: DeviceLimitsRecord,
}

impl Default for GpuRequiredProfile {
    fn default() -> Self {
        Self {
            required_features: Vec::new(),
            min_limits: DeviceLimitsRecord::default(),
        }
    }
}

/// Structured negotiation failure when adapter capabilities cannot satisfy requirements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NegotiationError {
    /// Adapter is missing a mandatory WebGPU feature flag.
    MissingRequiredFeature {
        /// Name of the missing feature.
        feature: String,
    },
    /// Adapter limit does not meet the minimum requested threshold.
    InsufficientLimit {
        /// Name of the insufficient WebGPU limit.
        limit_name: String,
        /// Minimum requested value.
        requested: u64,
        /// Actual limit available on the adapter.
        available: u64,
    },
    /// No compatible WebGPU adapter is available.
    AdapterUnavailable,
}

impl core::fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingRequiredFeature { feature } => {
                write!(f, "Required WebGPU feature '{feature}' is not supported by the adapter")
            }
            Self::InsufficientLimit { limit_name, requested, available } => {
                write!(
                    f,
                    "Insufficient limit for '{limit_name}': requested {requested}, available {available}"
                )
            }
            Self::AdapterUnavailable => write!(f, "No compatible WebGPU adapter available"),
        }
    }
}

/// Validated negotiation profile recording enabled features and bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NegotiatedDeviceProfile {
    /// Vendor name string of the underlying GPU adapter.
    pub adapter_vendor: String,
    /// Architecture description string of the underlying GPU adapter.
    pub adapter_architecture: String,
    /// Whether the adapter is a CPU fallback/software rasterizer.
    pub is_fallback_adapter: bool,
    /// List of successfully enabled WebGPU features.
    pub enabled_features: Vec<String>,
    /// Negotiated device limits applied during device creation.
    pub limits: DeviceLimitsRecord,
    /// Host's preferred swapchain canvas format (e.g. bgra8unorm or rgba8unorm).
    pub preferred_canvas_format: String,
}

impl GpuRequiredProfile {
    /// Negotiates against adapter capabilities before requesting a device.
    pub fn negotiate(
        &self,
        adapter_features: &[String],
        adapter_limits: &DeviceLimitsRecord,
        adapter_vendor: &str,
        adapter_architecture: &str,
        is_fallback_adapter: bool,
        preferred_canvas_format: &str,
    ) -> Result<NegotiatedDeviceProfile, NegotiationError> {
        // 1. Verify all required features exist in adapter features
        for req in &self.required_features {
            if !adapter_features.iter().any(|f| f == req) {
                return Err(NegotiationError::MissingRequiredFeature {
                    feature: req.clone(),
                });
            }
        }

        // 2. Verify limits: maxima must be >= requested, minimum alignments must be <= requested
        if (adapter_limits.max_texture_dimension_2d as u64) < (self.min_limits.max_texture_dimension_2d as u64) {
            return Err(NegotiationError::InsufficientLimit {
                limit_name: String::from("maxTextureDimension2D"),
                requested: self.min_limits.max_texture_dimension_2d as u64,
                available: adapter_limits.max_texture_dimension_2d as u64,
            });
        }
        if adapter_limits.max_buffer_size < self.min_limits.max_buffer_size {
            return Err(NegotiationError::InsufficientLimit {
                limit_name: String::from("maxBufferSize"),
                requested: self.min_limits.max_buffer_size,
                available: adapter_limits.max_buffer_size,
            });
        }
        if (adapter_limits.max_bind_groups as u64) < (self.min_limits.max_bind_groups as u64) {
            return Err(NegotiationError::InsufficientLimit {
                limit_name: String::from("maxBindGroups"),
                requested: self.min_limits.max_bind_groups as u64,
                available: adapter_limits.max_bind_groups as u64,
            });
        }
        // Alignments: an adapter requiring e.g. 512-byte alignment cannot satisfy code requiring 256
        if adapter_limits.min_uniform_buffer_offset_alignment > self.min_limits.min_uniform_buffer_offset_alignment {
            return Err(NegotiationError::InsufficientLimit {
                limit_name: String::from("minUniformBufferOffsetAlignment"),
                requested: self.min_limits.min_uniform_buffer_offset_alignment as u64,
                available: adapter_limits.min_uniform_buffer_offset_alignment as u64,
            });
        }
        if adapter_limits.min_storage_buffer_offset_alignment > self.min_limits.min_storage_buffer_offset_alignment {
            return Err(NegotiationError::InsufficientLimit {
                limit_name: String::from("minStorageBufferOffsetAlignment"),
                requested: self.min_limits.min_storage_buffer_offset_alignment as u64,
                available: adapter_limits.min_storage_buffer_offset_alignment as u64,
            });
        }

        Ok(NegotiatedDeviceProfile {
            adapter_vendor: adapter_vendor.to_string(),
            adapter_architecture: adapter_architecture.to_string(),
            is_fallback_adapter,
            enabled_features: self.required_features.clone(),
            limits: self.min_limits,
            preferred_canvas_format: preferred_canvas_format.to_string(),
        })
    }
}

/// Error scope kind per WebGPU specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorScopeKind {
    /// Validation error scope catching pipeline and descriptor errors.
    Validation,
    /// Out-of-memory error scope catching buffer and texture allocation failures.
    OutOfMemory,
    /// Internal error scope catching unexpected GPU driver failures.
    Internal,
}

impl ErrorScopeKind {
    /// Returns the WebGPU string identifier for this error scope kind.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::OutOfMemory => "out-of-memory",
            Self::Internal => "internal",
        }
    }
}

/// Tracks error scope nesting state and serializes access.
///
/// Note: Runtime error-scope discipline on the browser WebGPU device is enforced in
/// `bridge_runtime.js` only (via `WebGpuBridgeHost::withErrorScopes`).
/// `ErrorScopeTracker` provides an audited reference model for scope nesting and concurrency serialization.
#[derive(Default, Debug)]
pub struct ErrorScopeTracker {
    active_depth: u32,
    in_flight_operation: Option<u64>,
}

impl ErrorScopeTracker {
    /// Creates a new error scope tracker with zero depth.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_depth: 0,
            in_flight_operation: None,
        }
    }

    /// Attempts to begin an error scope block for an operation ID.
    /// Fails if another operation already holds the error scope stack.
    pub fn begin_scope(&mut self, operation_id: u64) -> Result<(), &'static str> {
        if let Some(active) = self.in_flight_operation {
            if active != operation_id {
                return Err("error-scope serialization violation: concurrent operation attempted to interleave error scopes");
            }
        }
        self.active_depth += 1;
        self.in_flight_operation = Some(operation_id);
        Ok(())
    }

    /// Ends one error scope level. When depth reaches 0, releases operation ownership.
    pub fn end_scope(&mut self, operation_id: u64) -> Result<bool, &'static str> {
        match self.in_flight_operation {
            Some(active) if active == operation_id => {
                if self.active_depth == 0 {
                    return Err("unbalanced popErrorScope: depth already zero");
                }
                self.active_depth -= 1;
                if self.active_depth == 0 {
                    self.in_flight_operation = None;
                    Ok(true) // all scopes ended
                } else {
                    Ok(false)
                }
            }
            _ => Err("error-scope ownership mismatch on pop"),
        }
    }

    /// Returns whether any error scope is currently active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active_depth > 0
    }
}

// -----------------------------------------------------------------------------
// Coarse Checked Packet Format
// -----------------------------------------------------------------------------

/// Four-byte magic header identifying the F3D binary GPU packet format (`F3DP`).
pub const PACKET_MAGIC: [u8; 4] = *b"F3DP";
/// Packet protocol version number (1).
pub const PACKET_VERSION: u16 = 1;

/// Opcode for creating a GPU buffer.
pub const OPCODE_CREATE_BUFFER: u16 = 1;
/// Opcode for writing data to an existing GPU buffer.
pub const OPCODE_WRITE_BUFFER: u16 = 2;
/// Opcode for creating a render pipeline from WGSL shader text.
pub const OPCODE_CREATE_PIPELINE: u16 = 3;
/// Opcode for encoding a complete render pass to a texture or canvas target.
pub const OPCODE_RENDER_PASS: u16 = 4;
/// Opcode for copying texture contents to a readable staging buffer.
pub const OPCODE_COPY_TEXTURE_TO_BUFFER: u16 = 5;
/// Opcode for creating a GPU texture.
pub const OPCODE_CREATE_TEXTURE: u16 = 6;
/// Opcode for pre-recording a GPURenderBundle from a pipeline, vertex buffer, and uniforms.
pub const OPCODE_RECORD_BUNDLE: u16 = 7;
/// Opcode for executing pre-recorded render bundles within a render pass.
pub const OPCODE_EXECUTE_BUNDLES: u16 = 8;
/// Opcode for configuring the active viewport within an open render pass (`renderPassEncoder.setViewport`).
pub const OPCODE_SET_VIEWPORT: u16 = 9;
/// Opcode for configuring the active scissor rectangle within an open render pass (`renderPassEncoder.setScissorRect`).
pub const OPCODE_SET_SCISSOR_RECT: u16 = 10;
/// Opcode for configuring draw call parameters (instance count, first vertex, first instance)
/// immediately preceding a direct render pass draw (`renderPassEncoder.draw`).
pub const OPCODE_SET_DRAW_PARAMETERS: u16 = 11;

/// GPUBufferUsage flag: map for CPU reading.
pub const BUFFER_USAGE_MAP_READ: u32 = 1;
/// GPUBufferUsage flag: copy source.
pub const BUFFER_USAGE_COPY_SRC: u32 = 4;
/// GPUBufferUsage flag: copy destination.
pub const BUFFER_USAGE_COPY_DST: u32 = 8;
/// GPUBufferUsage flag: vertex buffer.
pub const BUFFER_USAGE_VERTEX: u32 = 32;
/// GPUBufferUsage flag: uniform buffer.
pub const BUFFER_USAGE_UNIFORM: u32 = 64;

/// GPUTextureUsage flag: copy source.
pub const TEXTURE_USAGE_COPY_SRC: u32 = 1;
/// GPUTextureUsage flag: copy destination.
pub const TEXTURE_USAGE_COPY_DST: u32 = 2;
/// GPUTextureUsage flag: texture binding in shaders.
pub const TEXTURE_USAGE_TEXTURE_BINDING: u32 = 4;
/// GPUTextureUsage flag: storage binding in shaders.
pub const TEXTURE_USAGE_STORAGE_BINDING: u32 = 8;
/// GPUTextureUsage flag: render attachment target.
pub const TEXTURE_USAGE_RENDER_ATTACHMENT: u32 = 16;

/// Target kind flag indicating an offscreen render target texture.
pub const TARGET_OFFSCREEN: u32 = 0;
/// Target kind flag indicating the canvas swapchain texture.
pub const TARGET_CANVAS: u32 = 1;

/// Load operation: clear attachment to clear color.
pub const LOAD_OP_CLEAR: u32 = 0;
/// Load operation: preserve existing attachment contents.
pub const LOAD_OP_LOAD: u32 = 1;
/// Load operation: undefined prior contents (discard/dontcare).
pub const LOAD_OP_DONT_CARE: u32 = 2;

/// Store operation: store rendered results to attachment target.
pub const STORE_OP_STORE: u32 = 0;
/// Store operation: discard rendered results at pass end.
pub const STORE_OP_DISCARD: u32 = 1;

/// Pass flag: no special pass boundary behavior.
pub const PASS_FLAG_NONE: u32 = 0;
/// Pass flag: indicates this command begins a new logical render pass.
pub const PASS_FLAG_NEW_PASS: u32 = 1;

/// Packs target kind, load op, store op, and pass boundary flags into a 32-bit wire integer.
///
/// Panics if target_kind > 1, load_op > 2, store_op > 1, or flags > 1.
#[inline]
pub const fn pack_target_type(target_kind: u32, load_op: u32, store_op: u32, flags: u32) -> u32 {
    assert!(target_kind <= 1, "target_kind must be 0 (TARGET_OFFSCREEN) or 1 (TARGET_CANVAS)");
    assert!(load_op <= 2, "load_op must be 0 (Clear), 1 (Load), or 2 (DontCare)");
    assert!(store_op <= 1, "store_op must be 0 (Store) or 1 (Discard)");
    assert!(flags <= 1, "pass_flags can only set bit 0 (PASS_FLAG_NEW_PASS)");
    (target_kind & 0xFF) | ((load_op & 0xFF) << 8) | ((store_op & 0xFF) << 16) | ((flags & 0xFF) << 24)
}

/// Validates that a raw 32-bit target_type integer contains only legal target kinds, load/store ops, and flags.
#[inline]
pub const fn validate_target_type(target_type: u32) -> bool {
    let kind = unpack_target_kind(target_type);
    let load = unpack_load_op(target_type);
    let store = unpack_store_op(target_type);
    let flags = unpack_pass_flags(target_type);
    kind <= 1 && load <= 2 && store <= 1 && flags <= 1
}

/// Extracts the target kind (TARGET_OFFSCREEN or TARGET_CANVAS) from a packed target_type integer.
#[inline]
pub const fn unpack_target_kind(target_type: u32) -> u32 {
    target_type & 0xFF
}

/// Extracts the load operation (LOAD_OP_CLEAR, LOAD_OP_LOAD, LOAD_OP_DONT_CARE) from a packed target_type integer.
#[inline]
pub const fn unpack_load_op(target_type: u32) -> u32 {
    (target_type >> 8) & 0xFF
}

/// Extracts the store operation (STORE_OP_STORE, STORE_OP_DISCARD) from a packed target_type integer.
#[inline]
pub const fn unpack_store_op(target_type: u32) -> u32 {
    (target_type >> 16) & 0xFF
}

/// Extracts the pass flags from a packed target_type integer.
#[inline]
pub const fn unpack_pass_flags(target_type: u32) -> u32 {
    (target_type >> 24) & 0xFF
}

/// Target format code indicating the canvas pipeline should adopt the host's negotiated preferredCanvasFormat.
pub const TARGET_FORMAT_PREFERRED_CANVAS: u32 = 0;
/// Target format code for standard bgra8unorm swapchain format.
pub const TARGET_FORMAT_BGRA8UNORM: u32 = 1;
/// Target format code for standard rgba8unorm swapchain and offscreen format.
pub const TARGET_FORMAT_RGBA8UNORM: u32 = 2;

/// High-level typed commands serialized into the checked packet.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuCommand {
    /// Command to allocate a new GPU buffer.
    CreateBuffer {
        /// Unique integer identifier for the buffer.
        buffer_id: u32,
        /// Size of the buffer in bytes (must be multiple of 4).
        size: u32,
        /// Bitfield of GPUBufferUsage flags.
        usage: u32,
    },
    /// Command to upload raw byte payload into a buffer slice.
    WriteBuffer {
        /// Target buffer identifier.
        buffer_id: u32,
        /// Byte offset within the target buffer.
        offset: u32,
        /// Raw byte payload to upload.
        data: Vec<u8>,
    },
    /// Command to allocate a new 2D GPU texture.
    CreateTexture {
        /// Unique integer identifier for the texture.
        texture_id: u32,
        /// Texture width in pixels.
        width: u32,
        /// Texture height in pixels.
        height: u32,
        /// Format code (1 = bgra8unorm, 2 = rgba8unorm).
        format: u32,
        /// Bitfield of GPUTextureUsage flags.
        usage: u32,
    },
    /// Command to compile and create a render pipeline from WGSL shader text.
    CreatePipeline {
        /// Unique integer identifier for the pipeline.
        pipeline_id: u32,
        /// Complete WGSL shader source code.
        wgsl_code: String,
        /// Color attachment format code (0 = preferred canvas, 1 = bgra8unorm, 2 = rgba8unorm).
        target_format: u32,
        /// Whether the pipeline consumes a vertex buffer at location 0.
        has_vertex_buffer: bool,
        /// Whether the pipeline consumes a dynamic uniform buffer at binding 0.
        has_uniform_buffer: bool,
        /// Explicit uniform buffer binding byte size (e.g. 48 for AffineRows, 16 for ColorUniform; defaults to 48 if 0 and has_uniform_buffer is true).
        uniform_size: u32,
        /// Explicit vertex array byte stride (e.g. 20 for position+uv; defaults to 20 if 0 and has_vertex_buffer is true).
        vertex_stride: u32,
    },
    /// Command to encode and execute a complete render pass.
    RenderPass {
        /// Target kind (0 = offscreen texture, 1 = canvas texture).
        target_type: u32,
        /// Texture identifier when target_type is TARGET_OFFSCREEN.
        target_id: u32,
        /// Clear color [R, G, B, A] normalized to [0.0, 1.0].
        clear_color: [f32; 4],
        /// Pipeline identifier to bind for rendering.
        pipeline_id: u32,
        /// Vertex buffer identifier (or 0 if vertex index generation is used).
        vertex_buffer_id: u32,
        /// Number of vertices to draw.
        vertex_count: u32,
        /// Dynamic offset for the uniform buffer binding (must be multiple of alignment).
        uniform_dynamic_offset: u32,
        /// Uniform buffer identifier bound to group 0 (defaults to 1 if 0).
        uniform_buffer_id: u32,
        /// Explicit load operation (0 = Clear, 1 = Load, 2 = DontCare).
        load_op: u32,
        /// Explicit store operation (0 = Store, 1 = Discard).
        store_op: u32,
        /// Pass boundary flags (0 = none, 1 = new pass boundary).
        pass_flags: u32,
    },
    /// Command to copy texture pixel data into a map-readable staging buffer.
    CopyTextureToBuffer {
        /// Source texture identifier.
        texture_id: u32,
        /// Destination staging buffer identifier.
        buffer_id: u32,
        /// Width in pixels to copy.
        width: u32,
        /// Height in pixels to copy.
        height: u32,
        /// Semantic epoch of the source texture state region.
        epoch: Epoch,
    },
    /// Command to pre-record a GPURenderBundle from a pipeline, vertex buffer, and optional uniform buffer.
    RecordBundle {
        /// Unique integer identifier for the recorded bundle.
        bundle_id: u32,
        /// Pipeline identifier to bind within the bundle.
        pipeline_id: u32,
        /// Vertex buffer identifier (or 0 if vertex index generation is used).
        vertex_buffer_id: u32,
        /// Number of vertices to draw.
        vertex_count: u32,
        /// Dynamic offset for the uniform buffer binding.
        uniform_dynamic_offset: u32,
        /// Uniform buffer identifier bound to group 0 (defaults to 1 if 0).
        uniform_buffer_id: u32,
        /// Color attachment target format code (0 = preferred canvas, 1 = bgra8unorm, 2 = rgba8unorm).
        target_format: u32,
    },
    /// Command to execute a list of pre-recorded GPURenderBundle objects inside an active render pass.
    ExecuteBundles {
        /// List of bundle identifiers to execute in sequence.
        bundle_ids: Vec<u32>,
    },
    /// Command to set the active viewport within an open render pass.
    SetViewport {
        /// X coordinate of the top-left corner of the viewport in pixels.
        x: f32,
        /// Y coordinate of the top-left corner of the viewport in pixels.
        y: f32,
        /// Width of the viewport in pixels.
        width: f32,
        /// Height of the viewport in pixels.
        height: f32,
        /// Minimum depth value in normalized coordinates [0.0, 1.0].
        min_depth: f32,
        /// Maximum depth value in normalized coordinates [0.0, 1.0].
        max_depth: f32,
    },
    /// Command to set the active scissor rectangle within an open render pass.
    SetScissorRect {
        /// X coordinate of the top-left corner of the scissor rectangle in pixels.
        x: u32,
        /// Y coordinate of the top-left corner of the scissor rectangle in pixels.
        y: u32,
        /// Width of the scissor rectangle in pixels.
        width: u32,
        /// Height of the scissor rectangle in pixels.
        height: u32,
    },
    /// Command to configure draw parameters (instance count, first vertex, first instance)
    /// for the immediately following direct render pass draw.
    SetDrawParameters {
        /// Number of instances to draw.
        instance_count: u32,
        /// Index of the first vertex to draw.
        first_vertex: u32,
        /// Index of the first instance to draw.
        first_instance: u32,
    },
}

/// Structured encoding failure when packet dimensions or data lengths exceed binary bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PacketEncodeError {
    /// Number of serialized commands exceeds u32::MAX.
    CommandCountOverflow {
        /// Total command count attempted.
        count: usize,
    },
    /// Accumulated data payload offset exceeds u32::MAX or requested bounds.
    DataPayloadOverflow {
        /// Current payload byte offset.
        offset: usize,
        /// Length of the chunk attempted to append.
        length: usize,
    },
    /// Individual command payload length exceeds u32::MAX.
    CommandDataOverflow {
        /// Index of the offending command.
        command_index: usize,
        /// Byte length of the command data.
        length: usize,
    },
}

impl core::fmt::Display for PacketEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CommandCountOverflow { count } => {
                write!(f, "Command count {count} exceeds u32::MAX")
            }
            Self::DataPayloadOverflow { offset, length } => {
                write!(f, "Data payload overflow at offset {offset} with chunk length {length}")
            }
            Self::CommandDataOverflow { command_index, length } => {
                write!(f, "Command {command_index} data length {length} exceeds u32::MAX")
            }
        }
    }
}

/// Serializes and validates a batch of GPU commands into a compact binary packet.
#[derive(Default, Debug)]
pub struct GpuSubmissionPacket {
    commands: Vec<GpuCommand>,
}

impl GpuSubmissionPacket {
    /// Creates a new empty GPU submission packet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Appends a GPU command to the packet batch.
    pub fn push(&mut self, cmd: GpuCommand) {
        self.commands.push(cmd);
    }

    /// Serializes commands into bytes:
    /// [Header: 4 magic, 2 ver, 2 flags, 4 cmd_count, 4 data_len]
    /// [Command records...]
    /// [Data payload block...]
    pub fn encode(&self) -> Result<Vec<u8>, PacketEncodeError> {
        self.encode_bounded(u32::MAX as usize)
    }

    /// Serializes commands with a specified maximum data payload bound (for testing and defensive limits).
    pub fn encode_bounded(&self, max_payload_len: usize) -> Result<Vec<u8>, PacketEncodeError> {
        let cmd_count = u32::try_from(self.commands.len()).map_err(|_| PacketEncodeError::CommandCountOverflow {
            count: self.commands.len(),
        })?;

        let mut command_records = Vec::new();
        let mut data_payload = Vec::new();

        for (cmd_idx, cmd) in self.commands.iter().enumerate() {
            match cmd {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    command_records.extend_from_slice(&OPCODE_CREATE_BUFFER.to_le_bytes());
                    command_records.extend_from_slice(&buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&size.to_le_bytes());
                    command_records.extend_from_slice(&usage.to_le_bytes());
                }
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    let data_len = u32::try_from(data.len()).map_err(|_| PacketEncodeError::CommandDataOverflow {
                        command_index: cmd_idx,
                        length: data.len(),
                    })?;
                    let current_len = data_payload.len();
                    if current_len.checked_add(data.len()).map_or(true, |sum| sum > max_payload_len) {
                        return Err(PacketEncodeError::DataPayloadOverflow {
                            offset: current_len,
                            length: data.len(),
                        });
                    }
                    let data_offset = u32::try_from(current_len).map_err(|_| PacketEncodeError::DataPayloadOverflow {
                        offset: current_len,
                        length: data.len(),
                    })?;
                    data_payload.extend_from_slice(data);

                    command_records.extend_from_slice(&OPCODE_WRITE_BUFFER.to_le_bytes());
                    command_records.extend_from_slice(&buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&offset.to_le_bytes());
                    command_records.extend_from_slice(&data_offset.to_le_bytes());
                    command_records.extend_from_slice(&data_len.to_le_bytes());
                }
                GpuCommand::CreateTexture { texture_id, width, height, format, usage } => {
                    command_records.extend_from_slice(&OPCODE_CREATE_TEXTURE.to_le_bytes());
                    command_records.extend_from_slice(&texture_id.to_le_bytes());
                    command_records.extend_from_slice(&width.to_le_bytes());
                    command_records.extend_from_slice(&height.to_le_bytes());
                    command_records.extend_from_slice(&format.to_le_bytes());
                    command_records.extend_from_slice(&usage.to_le_bytes());
                }
                GpuCommand::CreatePipeline {
                    pipeline_id,
                    wgsl_code,
                    target_format,
                    has_vertex_buffer,
                    has_uniform_buffer,
                    uniform_size,
                    vertex_stride,
                } => {
                    let bytes = wgsl_code.as_bytes();
                    let code_len = u32::try_from(bytes.len()).map_err(|_| PacketEncodeError::CommandDataOverflow {
                        command_index: cmd_idx,
                        length: bytes.len(),
                    })?;
                    let current_len = data_payload.len();
                    if current_len.checked_add(bytes.len()).map_or(true, |sum| sum > max_payload_len) {
                        return Err(PacketEncodeError::DataPayloadOverflow {
                            offset: current_len,
                            length: bytes.len(),
                        });
                    }
                    let code_offset = u32::try_from(current_len).map_err(|_| PacketEncodeError::DataPayloadOverflow {
                        offset: current_len,
                        length: bytes.len(),
                    })?;
                    data_payload.extend_from_slice(bytes);

                    command_records.extend_from_slice(&OPCODE_CREATE_PIPELINE.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *has_vertex_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&(if *has_uniform_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&uniform_size.to_le_bytes());
                    command_records.extend_from_slice(&vertex_stride.to_le_bytes());
                }
                GpuCommand::RenderPass {
                    target_type,
                    target_id,
                    clear_color,
                    pipeline_id,
                    vertex_buffer_id,
                    vertex_count,
                    uniform_dynamic_offset,
                    uniform_buffer_id,
                    load_op,
                    store_op,
                    pass_flags,
                } => {
                    command_records.extend_from_slice(&OPCODE_RENDER_PASS.to_le_bytes());
                    let packed_target = pack_target_type(*target_type, *load_op, *store_op, *pass_flags);
                    command_records.extend_from_slice(&packed_target.to_le_bytes());
                    command_records.extend_from_slice(&target_id.to_le_bytes());
                    for c in clear_color {
                        command_records.extend_from_slice(&c.to_le_bytes());
                    }
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&vertex_buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&vertex_count.to_le_bytes());
                    command_records.extend_from_slice(&uniform_dynamic_offset.to_le_bytes());
                    command_records.extend_from_slice(&uniform_buffer_id.to_le_bytes());
                }
                GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, epoch } => {
                    command_records.extend_from_slice(&OPCODE_COPY_TEXTURE_TO_BUFFER.to_le_bytes());
                    command_records.extend_from_slice(&texture_id.to_le_bytes());
                    command_records.extend_from_slice(&buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&width.to_le_bytes());
                    command_records.extend_from_slice(&height.to_le_bytes());
                    let (epoch_hi, epoch_lo) = epoch.to_words();
                    command_records.extend_from_slice(&epoch_hi.to_le_bytes());
                    command_records.extend_from_slice(&epoch_lo.to_le_bytes());
                }
                GpuCommand::RecordBundle {
                    bundle_id,
                    pipeline_id,
                    vertex_buffer_id,
                    vertex_count,
                    uniform_dynamic_offset,
                    uniform_buffer_id,
                    target_format,
                } => {
                    command_records.extend_from_slice(&OPCODE_RECORD_BUNDLE.to_le_bytes());
                    command_records.extend_from_slice(&bundle_id.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&vertex_buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&vertex_count.to_le_bytes());
                    command_records.extend_from_slice(&uniform_dynamic_offset.to_le_bytes());
                    command_records.extend_from_slice(&uniform_buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                }
                GpuCommand::ExecuteBundles { bundle_ids } => {
                    let count = u32::try_from(bundle_ids.len()).map_err(|_| {
                        PacketEncodeError::CommandDataOverflow {
                            command_index: cmd_idx,
                            length: bundle_ids.len() * 4,
                        }
                    })?;
                    command_records.extend_from_slice(&OPCODE_EXECUTE_BUNDLES.to_le_bytes());
                    command_records.extend_from_slice(&count.to_le_bytes());
                    for id in bundle_ids {
                        command_records.extend_from_slice(&id.to_le_bytes());
                    }
                }
                GpuCommand::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => {
                    command_records.extend_from_slice(&OPCODE_SET_VIEWPORT.to_le_bytes());
                    command_records.extend_from_slice(&x.to_le_bytes());
                    command_records.extend_from_slice(&y.to_le_bytes());
                    command_records.extend_from_slice(&width.to_le_bytes());
                    command_records.extend_from_slice(&height.to_le_bytes());
                    command_records.extend_from_slice(&min_depth.to_le_bytes());
                    command_records.extend_from_slice(&max_depth.to_le_bytes());
                }
                GpuCommand::SetScissorRect {
                    x,
                    y,
                    width,
                    height,
                } => {
                    command_records.extend_from_slice(&OPCODE_SET_SCISSOR_RECT.to_le_bytes());
                    command_records.extend_from_slice(&x.to_le_bytes());
                    command_records.extend_from_slice(&y.to_le_bytes());
                    command_records.extend_from_slice(&width.to_le_bytes());
                    command_records.extend_from_slice(&height.to_le_bytes());
                }
                GpuCommand::SetDrawParameters {
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    command_records.extend_from_slice(&OPCODE_SET_DRAW_PARAMETERS.to_le_bytes());
                    command_records.extend_from_slice(&instance_count.to_le_bytes());
                    command_records.extend_from_slice(&first_vertex.to_le_bytes());
                    command_records.extend_from_slice(&first_instance.to_le_bytes());
                }
            }
        }

        let total_data_len = u32::try_from(data_payload.len()).map_err(|_| PacketEncodeError::DataPayloadOverflow {
            offset: data_payload.len(),
            length: 0,
        })?;

        let mut output = Vec::with_capacity(16 + command_records.len() + data_payload.len());
        output.extend_from_slice(&PACKET_MAGIC);
        output.extend_from_slice(&PACKET_VERSION.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes()); // flags
        output.extend_from_slice(&cmd_count.to_le_bytes());
        output.extend_from_slice(&total_data_len.to_le_bytes());
        output.extend_from_slice(&command_records);
        output.extend_from_slice(&data_payload);
        Ok(output)
    }

    /// Returns a slice of the recorded GPU commands.
    #[must_use]
    pub fn commands(&self) -> &[GpuCommand] {
        &self.commands
    }

    /// Consumes the packet and returns the vector of recorded GPU commands.
    #[must_use]
    pub fn into_commands(self) -> Vec<GpuCommand> {
        self.commands
    }

    /// Creates a submission packet pre-populated with a batch of GPU commands.
    #[must_use]
    pub fn from_commands(commands: Vec<GpuCommand>) -> Self {
        Self { commands }
    }
}

/// Structured error during lowering of an execution plan to GPU commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanLoweringError {
    /// Render segment is missing a required color attachment.
    MissingColorAttachment {
        /// Diagnostic name of the pass segment.
        segment_name: String,
    },
    /// Render segment contains zero draw commands.
    MissingDrawCommand {
        /// Diagnostic name of the pass segment.
        segment_name: String,
    },
    /// Render segment configures multiple color attachments (MRT not yet supported by bridge lowering).
    UnsupportedMultipleColorAttachments {
        /// Diagnostic name of the pass segment.
        segment_name: String,
        /// Number of configured color attachments.
        count: usize,
    },
    /// Render segment configures a depth/stencil attachment (depth not yet supported by bridge lowering).
    UnsupportedDepthStencilAttachment {
        /// Diagnostic name of the pass segment.
        segment_name: String,
    },
    /// Execution plan contains an unsupported pass kind for bridge lowering.
    UnsupportedPassKind {
        /// Diagnostic name of the pass segment.
        segment_name: String,
    },
    /// Execution plan contains an unsupported copy command variant.
    UnsupportedCopyCommand {
        /// Diagnostic reason describing the unsupported copy operation.
        reason: String,
    },
}

impl core::fmt::Display for PlanLoweringError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingColorAttachment { segment_name } => {
                write!(
                    f,
                    "Render segment '{segment_name}' is missing a required color attachment"
                )
            }
            Self::MissingDrawCommand { segment_name } => {
                write!(
                    f,
                    "Render segment '{segment_name}' contains zero draw commands"
                )
            }
            Self::UnsupportedMultipleColorAttachments { segment_name, count } => {
                write!(
                    f,
                    "Render segment '{segment_name}' configures {count} color attachments; MRT is not supported by bridge lowering"
                )
            }
            Self::UnsupportedDepthStencilAttachment { segment_name } => {
                write!(
                    f,
                    "Render segment '{segment_name}' configures a depth/stencil attachment; depth is not supported by bridge lowering"
                )
            }
            Self::UnsupportedPassKind { segment_name } => {
                write!(
                    f,
                    "Pass segment '{segment_name}' has an unsupported pass kind for bridge lowering"
                )
            }
            Self::UnsupportedCopyCommand { reason } => {
                write!(
                    f,
                    "Unsupported copy command during bridge lowering: {reason}"
                )
            }
        }
    }
}

impl core::error::Error for PlanLoweringError {}

/// Lowers a compiled `f3d_graph::ExecutionPlan` into a sequence of executable `GpuCommand` items.
///
/// Consumes typed accessors on `ExecutionPlan`, `PlanSegment`, `ColorAttachment`, `Draw`, and `CopyCommand`.
///
/// # Architecture & Invariants (§6.1, §6.7)
/// - **Render Segments**: Primary color attachments map to `TARGET_CANVAS` (if swapchain epoch is bound)
///   or `TARGET_OFFSCREEN`. Clear colors and explicit `LoadOp` / `StoreOp` semantics are preserved verbatim.
///   The opening command of each render pass segment receives `PASS_FLAG_NEW_PASS` to enforce pass boundaries.
///   Each draw is lowered with its bound pipeline, vertex buffer ID (or 0), vertex count, 256-byte aligned
///   dynamic uniform offset, and uniform buffer ID (from declared uniform resource use, defaulting to 1).
/// - **Copy Segments**: `CopyCommand::TextureToBuffer` commands are lowered into `GpuCommand::CopyTextureToBuffer`
///   with source texture ID, destination readback buffer ID, width, and height.
/// - **Compute Segments**: Lowering compute dispatches is not yet supported and returns a structured error.
pub fn lower_plan(plan: &ExecutionPlan) -> Result<Vec<GpuCommand>, PlanLoweringError> {
    let mut commands = Vec::new();

    for segment in plan.segments() {
        match segment.kind() {
            PassKind::Render => {
                if segment.color_attachments().len() > 1 {
                    return Err(PlanLoweringError::UnsupportedMultipleColorAttachments {
                        segment_name: segment.name().to_string(),
                        count: segment.color_attachments().len(),
                    });
                }
                if segment.depth_stencil_attachment().is_some() {
                    return Err(PlanLoweringError::UnsupportedDepthStencilAttachment {
                        segment_name: segment.name().to_string(),
                    });
                }
                let Some(ca) = segment.primary_color_attachment() else {
                    return Err(PlanLoweringError::MissingColorAttachment {
                        segment_name: segment.name().to_string(),
                    });
                };
                let target_type = if ca.is_canvas() {
                    TARGET_CANVAS
                } else {
                    TARGET_OFFSCREEN
                };
                let target_id = ca.target_id().get();
                let clear_color = ca.clear_color();
                let load_op = match ca.load_op() {
                    LoadOp::Clear => LOAD_OP_CLEAR,
                    LoadOp::Load => LOAD_OP_LOAD,
                    LoadOp::DontCare => LOAD_OP_DONT_CARE,
                };
                let store_op = match ca.store_op() {
                    StoreOp::Store => STORE_OP_STORE,
                    StoreOp::Discard => STORE_OP_DISCARD,
                };

                if segment.draws().is_empty() {
                    if ca.load_op() == LoadOp::Clear {
                        commands.push(GpuCommand::RenderPass {
                            target_type,
                            target_id,
                            clear_color,
                            pipeline_id: 0,
                            vertex_buffer_id: 0,
                            vertex_count: 0,
                            uniform_dynamic_offset: 0,
                            uniform_buffer_id: 0,
                            load_op,
                            store_op,
                            pass_flags: PASS_FLAG_NEW_PASS,
                        });
                        continue;
                    } else {
                        return Err(PlanLoweringError::MissingDrawCommand {
                            segment_name: segment.name().to_string(),
                        });
                    }
                }

                let mut is_first_command_in_segment = true;
                let mut current_viewport: Option<[u32; 4]> = None;
                let mut current_scissor: Option<[u32; 4]> = None;

                // When the first draw in a render pass is a bundle draw, emit an initial
                // GpuCommand::RenderPass with vertex_count: 0 to open the render pass on the target
                // with its clear color and load/store semantics. Subsequent bundle executes and direct draws
                // occur inside this same pass without re-clearing.
                if matches!(segment.draws().first().map(|d| d.kind()), Some(DrawKind::Bundle { .. })) {
                    commands.push(GpuCommand::RenderPass {
                        target_type,
                        target_id,
                        clear_color,
                        pipeline_id: 0,
                        vertex_buffer_id: 0,
                        vertex_count: 0,
                        uniform_dynamic_offset: 0,
                        uniform_buffer_id: 0,
                        load_op,
                        store_op,
                        pass_flags: PASS_FLAG_NEW_PASS,
                    });
                    is_first_command_in_segment = false;
                }

                for draw in segment.draws() {
                    let needs_viewport_update = match draw.viewport() {
                        Some(vp) => current_viewport != Some(vp),
                        None => false,
                    };

                    let (needs_scissor_update, target_scissor, next_current_scissor) =
                        if draw.scissor_test_enabled() {
                            if let Some(sc) = draw.scissor() {
                                if current_scissor != Some(sc) {
                                    (true, Some(sc), Some(sc))
                                } else {
                                    (false, None, current_scissor)
                                }
                            } else {
                                (false, None, current_scissor)
                            }
                        } else if current_scissor.is_some() {
                            // Scissor test was disabled after being enabled in this pass.
                            // Emit full-attachment scissor to clear clipping (WebGPU has no disableScissor).
                            if let Some(full) = draw.scissor() {
                                (true, Some(full), None)
                            } else {
                                (false, None, current_scissor)
                            }
                        } else {
                            (false, None, current_scissor)
                        };

                    let needs_draw_parameters = matches!(draw.kind(), DrawKind::Direct)
                        && (draw.instance_count() != 1
                            || draw.first_vertex() != 0
                            || draw.first_instance() != 0);

                    // If a viewport, scissor, or draw parameters command must precede this draw,
                    // ensure the render pass is open first (WebGPU requires an active render pass encoder
                    // before setViewport/setScissorRect/draw commands).
                    if (needs_viewport_update || needs_scissor_update || needs_draw_parameters)
                        && is_first_command_in_segment
                    {
                        commands.push(GpuCommand::RenderPass {
                            target_type,
                            target_id,
                            clear_color,
                            pipeline_id: 0,
                            vertex_buffer_id: 0,
                            vertex_count: 0,
                            uniform_dynamic_offset: 0,
                            uniform_buffer_id: 0,
                            load_op,
                            store_op,
                            pass_flags: PASS_FLAG_NEW_PASS,
                        });
                        is_first_command_in_segment = false;
                    }

                    if needs_viewport_update {
                        let vp = draw.viewport().unwrap();
                        commands.push(GpuCommand::SetViewport {
                            x: vp[0] as f32,
                            y: vp[1] as f32,
                            width: vp[2] as f32,
                            height: vp[3] as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        });
                        current_viewport = Some(vp);
                    }

                    if needs_scissor_update {
                        let sc = target_scissor.unwrap();
                        commands.push(GpuCommand::SetScissorRect {
                            x: sc[0],
                            y: sc[1],
                            width: sc[2],
                            height: sc[3],
                        });
                        current_scissor = next_current_scissor;
                    }

                    match draw.kind() {
                        DrawKind::Bundle { bundle_id } => {
                            commands.push(GpuCommand::ExecuteBundles {
                                bundle_ids: alloc::vec![bundle_id],
                            });
                        }
                        DrawKind::Direct => {
                            if needs_draw_parameters {
                                commands.push(GpuCommand::SetDrawParameters {
                                    instance_count: draw.instance_count(),
                                    first_vertex: draw.first_vertex(),
                                    first_instance: draw.first_instance(),
                                });
                            }

                            let uniform_buffer_id = draw
                                .uses()
                                .iter()
                                .find(|u| u.access == ResourceAccess::UniformBuffer)
                                .map_or(1, |u| u.resource_id.get());

                            let pass_flags = if is_first_command_in_segment {
                                is_first_command_in_segment = false;
                                PASS_FLAG_NEW_PASS
                            } else {
                                PASS_FLAG_NONE
                            };

                            commands.push(GpuCommand::RenderPass {
                                target_type,
                                target_id,
                                clear_color,
                                pipeline_id: draw.pipeline_id(),
                                vertex_buffer_id: draw.vertex_buffer_id(),
                                vertex_count: draw.vertex_count(),
                                uniform_dynamic_offset: draw.uniform_dynamic_offset(),
                                uniform_buffer_id,
                                load_op,
                                store_op,
                                pass_flags,
                            });
                        }
                    }
                }
            }
            PassKind::Copy => {
                for copy in segment.copies() {
                    if let Some((tex, buf, width, height, _pitch)) = copy.as_texture_to_buffer() {
                        commands.push(GpuCommand::CopyTextureToBuffer {
                            texture_id: tex.get(),
                            buffer_id: buf.get(),
                            width,
                            height,
                            epoch: plan.canvas_epoch().unwrap_or(Epoch::ZERO),
                        });
                    } else {
                        return Err(PlanLoweringError::UnsupportedCopyCommand {
                            reason: String::from(
                                "Only TextureToBuffer copy commands are currently supported by bridge lowering",
                            ),
                        });
                    }
                }
            }
            PassKind::Compute => {
                return Err(PlanLoweringError::UnsupportedPassKind {
                    segment_name: segment.name().to_string(),
                });
            }
        }
    }

    Ok(commands)
}

/// Builds a real WGSL triangle submission packet using `f3d_core::layout::AffineRows`
/// and `f3d_core::layout::WGSL_AFFINE_ROWS_DECLARATION`.
pub fn build_triangle_submission() -> GpuSubmissionPacket {
    // Register resource IDs in the generational slot table (§6.5, vqa.6)
    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vertex_buffer_id
        table.register(3);   // readback_buffer_id
        table.register(10);  // target_texture_id
        table.register(100); // pipeline_id
    });

    let mut packet = GpuSubmissionPacket::new();

    // 1. Create uniform buffer with AffineRows identity transform
    let uniform_buffer_id = 1;
    let identity_affine = AffineRows::identity();
    let affine_bytes = identity_affine.to_bytes().to_vec();

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT as u32,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: uniform_buffer_id,
        offset: 0,
        data: affine_bytes,
    });

    // 2. Vertex buffer: 3 vertices using typed VertexPosUv records (no raw float slices)
    let vertex_buffer_id = 2;
    let vertices = [
        VertexPosUv::new([0.0, 0.5, 0.0], [0.5, 1.0]),
        VertexPosUv::new([-0.5, -0.5, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.5, -0.5, 0.0], [1.0, 0.0]),
    ];
    let mut vertex_bytes = Vec::with_capacity(vertices.len() * VertexPosUv::BYTE_SIZE);
    for v in &vertices {
        vertex_bytes.extend_from_slice(&v.to_bytes());
    }
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vertex_buffer_id,
        size: vertex_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vertex_buffer_id,
        offset: 0,
        data: vertex_bytes,
    });

    // 3. Targets and readback buffer (64x64)
    let target_texture_id = 10;
    let readback_buffer_id = 20;
    let bytes_per_row = 256u32; // 64 * 4 = 256, naturally aligned
    let readback_size = bytes_per_row * 64;

    packet.push(GpuCommand::CreateTexture {
        texture_id: target_texture_id,
        width: 64,
        height: 64,
        format: 2, // rgba8unorm
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // 4. Construct WGSL shader incorporating RusticRobin's canonical AffineRows definition
    let wgsl_source = format!(
        "{}\n\
@group(0) @binding(0)\n\
var<uniform> model: AffineRows;\n\
\n\
struct VertexInput {{\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
}};\n\
\n\
struct VertexOutput {{\n\
    @builtin(position) clip_pos: vec4<f32>,\n\
    @location(0) uv: vec2<f32>,\n\
}};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> VertexOutput {{\n\
    var out: VertexOutput;\n\
    let transformed = transform_affine_point(model, in.position);\n\
    out.clip_pos = vec4<f32>(transformed, 1.0);\n\
    out.uv = in.uv;\n\
    return out;\n\
}}\n\
\n\
@fragment\n\
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{\n\
    return vec4<f32>(in.uv.x, in.uv.y, 1.0 - in.uv.x, 1.0);\n\
}}\n",
        WGSL_AFFINE_ROWS_DECLARATION
    );

    // 5. Pipeline for offscreen target (rgba8unorm = 2)
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: 100,
        wgsl_code: wgsl_source.clone(),
        target_format: 2,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: AFFINE_ROWS_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 6. Pipeline for canvas target (uses TARGET_FORMAT_PREFERRED_CANVAS to dynamically match host preferredCanvasFormat)
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: 101,
        wgsl_code: wgsl_source,
        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: AFFINE_ROWS_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 7. Render pass to offscreen target
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_texture_id,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id: 100,
        vertex_buffer_id,
        vertex_count: 3,
        uniform_dynamic_offset: 0,
        uniform_buffer_id,
        load_op: LOAD_OP_CLEAR,
        store_op: STORE_OP_STORE,
        pass_flags: PASS_FLAG_NEW_PASS,
    });

    // 8. Render pass to canvas swapchain (acquires fresh currentTexture per interval)
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_CANVAS,
        target_id: 0,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id: 101,
        vertex_buffer_id,
        vertex_count: 3,
        uniform_dynamic_offset: 0,
        uniform_buffer_id,
        load_op: LOAD_OP_CLEAR,
        store_op: STORE_OP_STORE,
        pass_flags: PASS_FLAG_NEW_PASS,
    });

    // 9. Copy offscreen texture to readback buffer
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_texture_id,
        buffer_id: readback_buffer_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

/// Helper to build a complete Red-A / Blue-B submission packet proving snapshot isolation
/// across two draws in the SAME queue submission.
///
/// Consumes GrayFox's `PerUseByteBuffer` from `f3d-core::ownership`:
/// - Red slice appended at `DataVersion::new(1)` (offset 0)
/// - Blue slice appended at `DataVersion::new(2)` (offset 256, 256-byte aligned)
///
/// In variant `per_use_versioned = true`:
/// - Pass A dynamically binds offset 0 and renders to Target A
/// - Pass B dynamically binds offset 256 and renders to Target B
/// Result: Target A is pure Red, Target B is pure Blue.
///
/// In variant `per_use_versioned = false` (regression / wrong impl toggle):
/// - Both passes bind offset 0
/// Result: Both passes read the second write (Blue), proving the test detects the queue hazard!
pub fn build_red_a_blue_b_submission(per_use_versioned: bool) -> GpuSubmissionPacket {
    let mut packet = GpuSubmissionPacket::new();

    let mat_handle = Handle::<MaterialDomain>::from_raw(1, 1).expect("valid material handle");
    let mut byte_buf = PerUseByteBuffer::<MaterialDomain>::new(256).expect("alignment 256");

    let red_bytes = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    let blue_bytes = [0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();

    let rec_a = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &red_bytes)
        .expect("append red slice");
    let rec_b = byte_buf
        .append_slice(mat_handle, DataVersion::new(2), Epoch::ZERO, &blue_bytes)
        .expect("append blue slice");

    let uniform_buffer_id = 1;
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: byte_buf.as_bytes().len() as u32,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: uniform_buffer_id,
        offset: 0,
        data: byte_buf.as_bytes().to_vec(),
    });

    let target_a = 30;
    let target_b = 31;
    let readback_a = 40;
    let readback_b = 41;
    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;

    packet.push(GpuCommand::CreateTexture {
        texture_id: target_a,
        width: 64,
        height: 64,
        format: 2, // rgba8unorm
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_b,
        width: 64,
        height: 64,
        format: 2, // rgba8unorm
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_a,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_b,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    let flat_shader = "\
struct ColorUniform {\n\
    color: vec4<f32>,\n\
};\n\
@group(0) @binding(0)\n\
var<uniform> u: ColorUniform;\n\
\n\
@vertex\n\
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {\n\
    var pos = array<vec2<f32>, 3>(\n\
        vec2<f32>(-1.0, -1.0),\n\
        vec2<f32>( 3.0, -1.0),\n\
        vec2<f32>(-1.0,  3.0)\n\
    );\n\
    return vec4<f32>(pos[idx], 0.0, 1.0);\n\
}\n\
\n\
@fragment\n\
fn fs_main() -> @location(0) vec4<f32> {\n\
    return u.color;\n\
}\n";

    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: 200,
        wgsl_code: flat_shader.to_string(),
        target_format: 2, // rgba8unorm
        has_vertex_buffer: false,
        has_uniform_buffer: true,
        uniform_size: COLOR_UNIFORM_BYTES as u32,
        vertex_stride: 0,
    });

    let offset_a = rec_a.byte_offset() as u32;
    let offset_b = if per_use_versioned {
        rec_b.byte_offset() as u32
    } else {
        rec_a.byte_offset() as u32
    };

    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_a,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id: 200,
        vertex_buffer_id: 0,
        vertex_count: 3,
        uniform_dynamic_offset: offset_a,
        uniform_buffer_id,
        load_op: LOAD_OP_CLEAR,
        store_op: STORE_OP_STORE,
        pass_flags: PASS_FLAG_NEW_PASS,
    });

    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_b,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id: 200,
        vertex_buffer_id: 0,
        vertex_count: 3,
        uniform_dynamic_offset: offset_b,
        uniform_buffer_id,
        load_op: LOAD_OP_CLEAR,
        store_op: STORE_OP_STORE,
        pass_flags: PASS_FLAG_NEW_PASS,
    });

    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_a,
        buffer_id: readback_a,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_b,
        buffer_id: readback_b,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

/// Helper to build a bundle-then-direct-draw submission packet demonstrating WebGPU bundle execution
/// and state invalidation (§8.5, 2v8.2).
///
/// Records:
/// 1. Uniform buffer: Offset 0 = Green `[0.0, 1.0, 0.0, 1.0]`, Offset 256 = Blue `[0.0, 0.0, 1.0, 1.0]`.
/// 2. Vertex buffer 1: Triangle 1 covering the left NDC half `[-1.0, 0.0]`.
/// 3. Vertex buffer 2: Triangle 2 covering the right NDC half `[0.0, 1.0]`.
/// 4. Pipeline 200: Flat color uniform with `VertexPosUv` vertex layout.
/// 5. `RecordBundle` (bundle ID 1): Pre-recorded bundle drawing Triangle 1 with Green (dynamic offset 0).
/// 6. `ExecuteBundles([1])`: Draws Green Triangle 1; per WebGPU specification, clears pass state.
/// 7. `ExecuteBundles([])`: Empty bundle sequence; reinforces spec-mandated pass state clearing.
/// 8. `RenderPass` (direct draw): Rebinds pipeline, bind group (offset 256 = Blue), vertex buffer 2, draws Triangle 2.
/// 9. `CopyTextureToBuffer`: Copies offscreen target to readback staging buffer.
///
/// Result: Both Green (bundle) and Blue (direct draw) are simultaneously visible in the readback buffer.
pub fn build_bundle_then_direct_draw_submission() -> GpuSubmissionPacket {
    // Register resource IDs in the generational slot table (§6.5, vqa.6)
    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vb_bundle_id
        table.register(3);   // vb_direct_id
        table.register(10);  // target_texture_id
        table.register(20);  // readback_buffer_id
        table.register(200); // pipeline_id
    });

    let mut packet = GpuSubmissionPacket::new();

    // 1. Uniform buffer with Green at offset 0 and Blue at offset 256
    let uniform_buffer_id = 1;
    let mut uniform_bytes = Vec::with_capacity(512);
    let green_color = [0.0f32, 1.0f32, 0.0f32, 1.0f32];
    for c in green_color {
        uniform_bytes.extend_from_slice(&c.to_le_bytes());
    }
    uniform_bytes.resize(256, 0);
    let blue_color = [0.0f32, 0.0f32, 1.0f32, 1.0f32];
    for c in blue_color {
        uniform_bytes.extend_from_slice(&c.to_le_bytes());
    }
    uniform_bytes.resize(512, 0);

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: 512,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: uniform_buffer_id,
        offset: 0,
        data: uniform_bytes,
    });

    // 2. Vertex buffer 1: Triangle 1 (left side)
    let vb_bundle_id = 2;
    let tri1 = [
        VertexPosUv::new([-1.0, -1.0, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([0.0, 1.0, 0.0], [0.5, 1.0]),
    ];
    let mut tri1_bytes = Vec::with_capacity(tri1.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri1 {
        tri1_bytes.extend_from_slice(&v.to_bytes());
    }
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb_bundle_id,
        size: tri1_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb_bundle_id,
        offset: 0,
        data: tri1_bytes,
    });

    // 3. Vertex buffer 2: Triangle 2 (right side)
    let vb_direct_id = 3;
    let tri2 = [
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([1.0, -1.0, 0.0], [1.0, 0.0]),
        VertexPosUv::new([1.0, 1.0, 0.0], [1.0, 1.0]),
    ];
    let mut tri2_bytes = Vec::with_capacity(tri2.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri2 {
        tri2_bytes.extend_from_slice(&v.to_bytes());
    }
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb_direct_id,
        size: tri2_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb_direct_id,
        offset: 0,
        data: tri2_bytes,
    });

    // 4. Target texture and readback buffer (64x64)
    let target_texture_id = 10;
    let readback_buffer_id = 20;
    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;

    packet.push(GpuCommand::CreateTexture {
        texture_id: target_texture_id,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // 5. Shader and Pipeline: Flat color uniform with VertexPosUv layout
    let flat_shader = "\
struct ColorUniform {\n\
    color: vec4<f32>,\n\
};\n\
@group(0) @binding(0)\n\
var<uniform> u: ColorUniform;\n\
\n\
struct VertexInput {\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(in.position, 1.0);\n\
}\n\
\n\
@fragment\n\
fn fs_main() -> @location(0) vec4<f32> {\n\
    return u.color;\n\
}\n";

    let pipeline_id = 200;
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: flat_shader.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: COLOR_UNIFORM_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 6. Record RenderBundle 1: draws Triangle 1 with Green (offset 0)
    let bundle_id = 1;
    packet.push(GpuCommand::RecordBundle {
        bundle_id,
        pipeline_id,
        vertex_buffer_id: vb_bundle_id,
        vertex_count: 3,
        uniform_dynamic_offset: 0,
        uniform_buffer_id,
        target_format: TARGET_FORMAT_RGBA8UNORM,
    });

    // 7. Open Render Pass on target texture with Black clear color (vertex_count = 0)
    // WebGPU spec & oracle sequence: pass begins once with clear, bundles execute,
    // empty bundle resets pass state, and direct draw rebinds inside the same pass.
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_texture_id,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id: 0,
        vertex_buffer_id: 0,
        vertex_count: 0,
        uniform_dynamic_offset: 0,
        uniform_buffer_id: 0,
        load_op: LOAD_OP_CLEAR,
        store_op: STORE_OP_STORE,
        pass_flags: PASS_FLAG_NEW_PASS,
    });

    // 8. Execute Bundle 1 inside the render pass (draws Green triangle)
    packet.push(GpuCommand::ExecuteBundles {
        bundle_ids: alloc::vec![bundle_id],
    });

    // 9. Execute empty bundle sequence (spec mandates this also resets state)
    packet.push(GpuCommand::ExecuteBundles {
        bundle_ids: Vec::new(),
    });

    // 10. Direct Draw inside same pass: draws Triangle 2 with Blue (offset 256)
    // Pass state was reset by executeBundles, forcing explicit re-bind of pipeline,
    // bind group, and vertex buffer without re-clearing the render pass.
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_texture_id,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id,
        vertex_buffer_id: vb_direct_id,
        vertex_count: 3,
        uniform_dynamic_offset: 256,
        uniform_buffer_id,
        load_op: LOAD_OP_CLEAR,
        store_op: STORE_OP_STORE,
        pass_flags: PASS_FLAG_NONE,
    });

    // 11. Copy offscreen target to readback buffer
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_texture_id,
        buffer_id: readback_buffer_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

/// Builds an AffineRows transform submission packet demonstrating GPU uniform upload,
/// WGSL row-dot-product evaluation (§6.1, §6.2, 05.6, vqa.2), and readback validation.
///
/// Records:
/// 1. Constructs a [`Matrix4`] with translation `(+0.5, 0.0, 0.0)` and scale `0.5` (`(0.5, 0.5, 1.0)`).
/// 2. Narrows the matrix to [`AffineRows`] via [`Matrix4::to_affine_rows_checked`].
/// 3. Uploads `AffineRows` as a 48-byte uniform into buffer 1 (padded to 256B for dynamic offset alignment).
/// 4. Creates vertex buffer 2 with a centered triangle:
///    - Top vertex: `(0.0, 0.5, 0.0)`
///    - Bottom-left vertex: `(-0.5, -0.5, 0.0)`
///    - Bottom-right vertex: `(0.5, -0.5, 0.0)`
/// 5. Compiles pipeline 200 with the canonical WGSL `transform_affine_point` helper from `CONTRACT.md`.
/// 6. Executes a render pass clearing 64x64 target texture 10 to Black `[0, 0, 0, 1]`, drawing the transformed triangle in Green `[0, 1, 0, 1]`.
/// 7. Copies target texture 10 to readback staging buffer 20 (64x64 RGBA8Unorm, 256-byte row pitch).
///
/// Hand-computed screen coordinates (64x64 viewport):
/// - After transform `P' = (0.5 * P.x + 0.5, 0.5 * P.y, P.z)`:
///   - Top vertex: NDC `(0.5, 0.25)` -> Screen `(48, 24)`
///   - Bottom-left: NDC `(0.25, -0.25)` -> Screen `(40, 40)`
///   - Bottom-right: NDC `(0.75, -0.25)` -> Screen `(56, 40)`
/// - Sample points:
///   - Transformed center `(48, 32)`: inside triangle -> Green `[0, 255, 0, 255]`
///   - Untransformed center `(32, 32)`: outside triangle -> Black `[0, 0, 0, 255]`
///   - Left half `(16, 32)`: outside triangle -> Black `[0, 0, 0, 255]`
///   - Right border `(60, 32)`: outside triangle -> Black `[0, 0, 0, 255]`
///
/// Registers resource IDs (buffers 1, 2, 20, texture 10, pipeline 200) in the global generational
/// slot table at generation 1 (§6.5, vqa.6). Packet building performs schedule synthesis
/// and command encoding only, leaving submission to the bridge caller (Mail 7117).
#[must_use]
pub fn build_affine_rows_transform_submission() -> GpuSubmissionPacket {
    // Register resource IDs in the generational slot table (§6.5, vqa.6)
    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vertex_buffer_id
        table.register(10);  // target_texture_id
        table.register(20);  // readback_buffer_id
        table.register(200); // pipeline_id
    });

    let mut packet = GpuSubmissionPacket::new();

    // 1. Build transform matrix with translation (+0.5, 0, 0) and scale 0.5
    let mut matrix = Matrix4::identity();
    matrix.compose(
        &Vector3::new(0.5, 0.0, 0.0),
        &Quaternion::identity(),
        &Vector3::new(0.5, 0.5, 1.0),
    );
    let affine_rows = matrix
        .to_affine_rows_checked(1e-9, 1e-9)
        .expect("Matrix4 must narrow to AffineRows cleanly");

    // 2. Uniform buffer with 48-byte AffineRows record at offset 0
    let uniform_buffer_id = 1;
    let mut uniform_bytes = Vec::with_capacity(256);
    uniform_bytes.extend_from_slice(&affine_rows.to_bytes());
    uniform_bytes.resize(256, 0);

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: 256,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: uniform_buffer_id,
        offset: 0,
        data: uniform_bytes,
    });

    // 3. Vertex buffer with centered triangle
    let vertex_buffer_id = 2;
    let tri = [
        VertexPosUv::new([0.0, 0.5, 0.0], [0.5, 1.0]),
        VertexPosUv::new([-0.5, -0.5, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.5, -0.5, 0.0], [1.0, 0.0]),
    ];
    let mut tri_bytes = Vec::with_capacity(tri.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri {
        tri_bytes.extend_from_slice(&v.to_bytes());
    }
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vertex_buffer_id,
        size: tri_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vertex_buffer_id,
        offset: 0,
        data: tri_bytes,
    });

    // 4. Target texture and readback buffer (64x64)
    let target_texture_id = 10;
    let readback_buffer_id = 20;
    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;

    packet.push(GpuCommand::CreateTexture {
        texture_id: target_texture_id,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // 5. WGSL Shader with three row dot-products (§6.1, CONTRACT.md)
    let affine_shader = "\
struct AffineRows {\n\
    r0: vec4<f32>,\n\
    r1: vec4<f32>,\n\
    r2: vec4<f32>,\n\
};\n\
\n\
fn transform_affine_point(m: AffineRows, p: vec3<f32>) -> vec3<f32> {\n\
    let v = vec4<f32>(p, 1.0);\n\
    return vec3<f32>(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));\n\
}\n\
\n\
@group(0) @binding(0)\n\
var<uniform> transform: AffineRows;\n\
\n\
struct VertexInput {\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
};\n\
\n\
struct VertexOutput {\n\
    @builtin(position) clip_position: vec4<f32>,\n\
    @location(0) uv: vec2<f32>,\n\
};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> VertexOutput {\n\
    var out: VertexOutput;\n\
    let transformed = transform_affine_point(transform, in.position);\n\
    out.clip_position = vec4<f32>(transformed, 1.0);\n\
    out.uv = in.uv;\n\
    return out;\n\
}\n\
\n\
@fragment\n\
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {\n\
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);\n\
}\n";

    let pipeline_id = 200;
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: affine_shader.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: AFFINE_ROWS_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 6. Render Pass on target texture clearing to Black and drawing the green triangle
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_texture_id,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id,
        vertex_buffer_id,
        vertex_count: 3,
        uniform_dynamic_offset: 0,
        uniform_buffer_id,
        load_op: LOAD_OP_CLEAR,
        store_op: STORE_OP_STORE,
        pass_flags: PASS_FLAG_NEW_PASS,
    });

    // 7. Copy target texture to readback buffer
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_texture_id,
        buffer_id: readback_buffer_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

/// Helper to build a nested-pass reentrant execution submission packet (§6.7, 2v8.4).
///
/// Constructs a 3-pass workload using [`FrameSession`] and lowered via [`lower_plan`],
/// matching byte-for-byte with `renderDirectNestedPassReference` in `oracle_reference.js`:
/// 1. Pass 1 (Target 10): Outer prefix pass clearing to Black `[0, 0, 0, 1]` with a Red draw on the left half (tri1, offset 0).
/// 2. Pass 2 (Target 11): Nested inner pass on Target 11 clearing to Black with a Green draw (tri1, offset 512).
/// 3. Pass 3 (Target 10): Resumed outer pass using `LoadOp::Load` with a Blue draw on the right half (tri2, offset 256).
/// 4. CopyTextureToBuffer: Copies Target 10 to readback Buffer 20.
///
/// Ground truth pixel expectations on 64x64 target:
/// - (24, 32): Inside left triangle -> Red `[255, 0, 0, 255]`.
/// - (56, 32): Inside right triangle -> Blue `[0, 0, 255, 255]`.
/// - (2, 2): Untouched background -> Black `[0, 0, 0, 255]`.
///
/// Registers resource IDs (buffers 1, 2, 3, 20, textures 10, 11, pipeline 200) in the global
/// generational slot table at generation 1 (§6.5, vqa.6). Packet construction performs schedule
/// synthesis and encoding only, leaving frame submission to the bridge caller (Mail 7117).
pub fn build_nested_pass_submission() -> GpuSubmissionPacket {
    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vb1_id (tri1)
        table.register(3);   // vb2_id (tri2)
        table.register(10);  // target_texture_10_id
        table.register(11);  // target_texture_11_id
        table.register(20);  // readback_buffer_id
        table.register(200); // pipeline_id
    });

    let mut packet = GpuSubmissionPacket::new();

    let uniform_buffer_id = 1;
    let vb1_id = 2;
    let vb2_id = 3;
    let target_10 = 10;
    let target_11 = 11;
    let readback_buffer_id = 20;
    let pipeline_id = 200;

    // 1. Vertex buffer 1: Triangle 1 (left side, covers x in [-1, 0], samples at (24, 32))
    // Matches renderDirectNestedPassReference tri1Data in oracle_reference.js
    let tri1 = [
        VertexPosUv::new([-1.0, -1.0, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([0.0, 1.0, 0.0], [0.5, 1.0]),
    ];
    let mut tri1_bytes = Vec::with_capacity(tri1.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri1 {
        tri1_bytes.extend_from_slice(&v.to_bytes());
    }

    // 2. Vertex buffer 2: Triangle 2 (right side, covers x in [0, 1], samples at (56, 32))
    // Matches renderDirectNestedPassReference tri2Data in oracle_reference.js
    let tri2 = [
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([1.0, -1.0, 0.0], [1.0, 0.0]),
        VertexPosUv::new([1.0, 1.0, 0.0], [1.0, 1.0]),
    ];
    let mut tri2_bytes = Vec::with_capacity(tri2.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri2 {
        tri2_bytes.extend_from_slice(&v.to_bytes());
    }

    // 3. Resource creation commands (matching oracle_reference.js device allocations)
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: 768,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb1_id,
        size: tri1_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb1_id,
        offset: 0,
        data: tri1_bytes,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb2_id,
        size: tri2_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb2_id,
        offset: 0,
        data: tri2_bytes,
    });

    packet.push(GpuCommand::CreateTexture {
        texture_id: target_10,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_11,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    let flat_shader = "\
struct ColorUniform {\n\
    color: vec4<f32>,\n\
};\n\
@group(0) @binding(0)\n\
var<uniform> u: ColorUniform;\n\
\n\
struct VertexInput {\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(in.position, 1.0);\n\
}\n\
\n\
@fragment\n\
fn fs_main() -> @location(0) vec4<f32> {\n\
    return u.color;\n\
}\n";

    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: flat_shader.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: COLOR_UNIFORM_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 4. Build pass structure through FrameSession and lower_plan
    let root_ctx = RenderContext::new_offscreen(
        ResourceId::new(target_10),
        64,
        64,
        Epoch::ZERO,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .expect("session init must succeed")
        .with_uniform_buffer_id(uniform_buffer_id);

    let mat_handle = Handle::<MaterialDomain>::from_raw(1, 1).expect("valid material handle");
    let red_bytes = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    let blue_bytes = [0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    let green_bytes = [0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();

    // Snapshot order matches oracle_reference.js layout:
    // Offset 0: Red [1.0, 0.0, 0.0, 1.0] (Pass 1 - Target 10 left triangle)
    // Offset 256: Blue [0.0, 0.0, 1.0, 1.0] (Pass 3 - Target 10 right triangle)
    // Offset 512: Green [0.0, 1.0, 0.0, 1.0] (Pass 2 - Target 11 nested pass)
    let rec_red = session
        .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, &red_bytes)
        .expect("red material snapshot");
    let rec_blue = session
        .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::ZERO, &blue_bytes)
        .expect("blue material snapshot");
    let rec_green = session
        .snapshot_material_use(mat_handle, DataVersion::new(3), Epoch::ZERO, &green_bytes)
        .expect("green material snapshot");

    // Pass 1: Outer prefix pass on Target 10 with clear to Black and Red draw on left half (offset 0)
    session.begin_render_pass("outer_prefix", [0.0, 0.0, 0.0, 1.0]).expect("begin outer prefix pass");
    session.record_direct_draw(pipeline_id, vb1_id, 3, Some(rec_red)).expect("record red draw");

    // Pass 2: Nested inner pass on Target 11 with Green draw (offset 512, draws tri1 on target 11)
    let nested_ctx = RenderContext::new_offscreen(
        ResourceId::new(target_11),
        64,
        64,
        Epoch::ZERO,
    );
    session
        .with_nested_render(nested_ctx, |s| {
            s.begin_render_pass("nested_pass", [0.0, 0.0, 0.0, 1.0]).expect("begin nested pass");
            s.record_direct_draw(pipeline_id, vb1_id, 3, Some(rec_green)).expect("record green draw");
            s.end_render_pass().expect("end nested pass");
            Ok(())
        })
        .expect("with_nested_render must succeed");

    // Pass 3: Outer resumed pass on Target 10 (with LoadOp::Load) and Blue draw on right half (offset 256, draws tri2 on target 10)
    session.record_direct_draw(pipeline_id, vb2_id, 3, Some(rec_blue)).expect("record blue draw");
    session.end_render_pass().expect("end outer resumed pass");

    let session_packet = session.build_submission_packet().expect("build session packet must succeed");

    for cmd in session_packet.into_commands() {
        packet.push(cmd);
    }

    // 5. Copy Target 10 to Readback Buffer 20
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_10,
        buffer_id: readback_buffer_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

/// Constructs a 14-command nested-pass reentrant submission packet targeting a live canvas presentation surface (§6.7, §8.5, 2v8.4).
///
/// Executes three logical passes:
/// 1. Outer prefix pass on Canvas Target 10: Clear to Black and Red draw on left half (`tri1`, dynamic offset 0).
/// 2. Nested inner pass on Offscreen Target 11: Clear to Black and Green draw (`tri1`, dynamic offset 512).
/// 3. Outer resumed pass on Canvas Target 10: Resume with `LoadOp::Load` and Blue draw on right half (`tri2`, dynamic offset 256).
///
/// This packet copies Offscreen Target 11 to Readback Buffer 20. The browser
/// counterexample suite separately copies the actual canvas texture with `COPY_SRC`
/// before yielding, checking the outer red/blue pixels as well as the inner target.
///
/// ### Resource Registration & Generational Slot Table (§6.5, vqa.6)
/// Registers resource IDs in the global generational slot table at generation 1:
/// - Buffers: 1 (uniform), 2 (tri1), 3 (tri2), 20 (readback staging)
/// - Textures: 10 (canvas presentation surface), 11 (offscreen nested render target)
/// - Pipelines: 200 (offscreen pipeline), 201 (canvas presentation pipeline)
///
/// ### Submission and Canvas Interval Boundaries
/// Packet construction (`build_nested_canvas_pass_submission` and its binary export
/// `gpu_bridge_build_nested_canvas_pass_packet`) performs schedule synthesis and command
/// encoding only. It registers resource handles and prepares the binary command buffer,
/// but does not submit work or end the canvas interval.
///
/// The bridge caller submits work with `device.queue.submit()`. Multiple submissions
/// may use the same live canvas texture; `FrameSession::end_canvas_interval` belongs
/// at actual host texture expiry, not at each queue submission.
pub fn build_nested_canvas_pass_submission() -> GpuSubmissionPacket {
    // Register resource IDs in the generational slot table (§6.5, §8.5, vqa.6, 2v8.4)
    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vb1_id (tri1)
        table.register(3);   // vb2_id (tri2)
        table.register(10);  // target_10 (canvas)
        table.register(11);  // target_11 (offscreen)
        table.register(20);  // readback_buffer_id
        table.register(200); // pipeline_offscreen
        table.register(201); // pipeline_canvas
    });

    let mut packet = GpuSubmissionPacket::new();

    let uniform_buffer_id = 1;
    let vb1_id = 2;
    let vb2_id = 3;
    let target_10 = 10;
    let target_11 = 11;
    let readback_buffer_id = 20;
    let pipeline_offscreen = 200;
    let pipeline_canvas = 201;

    // 1. Vertex buffer 1: Triangle 1 (left side, covers x in [-1, 0], samples at (24, 32))
    let tri1 = [
        VertexPosUv::new([-1.0, -1.0, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([0.0, 1.0, 0.0], [0.5, 1.0]),
    ];
    let mut tri1_bytes = Vec::with_capacity(tri1.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri1 {
        tri1_bytes.extend_from_slice(&v.to_bytes());
    }

    // 2. Vertex buffer 2: Triangle 2 (right side, covers x in [0, 1], samples at (56, 32))
    let tri2 = [
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([1.0, -1.0, 0.0], [1.0, 0.0]),
        VertexPosUv::new([1.0, 1.0, 0.0], [1.0, 1.0]),
    ];
    let mut tri2_bytes = Vec::with_capacity(tri2.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri2 {
        tri2_bytes.extend_from_slice(&v.to_bytes());
    }

    // 3. Resource creation commands (matching oracle device allocations):
    // Command 0: Uniform buffer 1 (768 bytes: 3 * 256B aligned dynamic slots)
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: 768,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });

    // Commands 1 & 2: Vertex buffer 1 (60 bytes)
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb1_id,
        size: tri1_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb1_id,
        offset: 0,
        data: tri1_bytes,
    });

    // Commands 3 & 4: Vertex buffer 2 (60 bytes)
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb2_id,
        size: tri2_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb2_id,
        offset: 0,
        data: tri2_bytes,
    });

    // Command 5: Offscreen texture Target 11 (Canvas Target 10 is acquired via swapchain, not created)
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_11,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    // Command 6: Readback buffer 20 for Target 11
    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    let flat_shader = "\
struct ColorUniform {\n\
    color: vec4<f32>,\n\
};\n\
@group(0) @binding(0)\n\
var<uniform> u: ColorUniform;\n\
\n\
struct VertexInput {\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(in.position, 1.0);\n\
}\n\
\n\
@fragment\n\
fn fs_main() -> @location(0) vec4<f32> {\n\
    return u.color;\n\
}\n";

    // Command 7: Pipeline 200 for offscreen target (RGBA8Unorm)
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: pipeline_offscreen,
        wgsl_code: flat_shader.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: COLOR_UNIFORM_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // Command 8: Pipeline 201 for canvas presentation target (dynamically matches host preferredCanvasFormat)
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: pipeline_canvas,
        wgsl_code: flat_shader.to_string(),
        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: COLOR_UNIFORM_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 4. Build pass structure through FrameSession with CanvasEpochTracker and lower_plan
    let mut tracker = CanvasEpochTracker::new();
    tracker.register_canvas(
        CanvasId::new(target_10),
        ResourceId::new(target_10),
        64,
        64,
        CanvasFormat::Bgra8Unorm,
    );
    let canvas_output = tracker
        .begin_frame_acquire(CanvasId::new(target_10))
        .expect("canvas acquire must succeed");

    let root_ctx = RenderContext::new_canvas_acquired(
        ResourceId::new(target_10),
        64,
        64,
        Epoch::new(1),
        canvas_output.epoch,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .expect("session init must succeed")
        .with_uniform_buffer_id(uniform_buffer_id);

    let mat_handle = Handle::<MaterialDomain>::from_raw(1, 1).expect("valid material handle");
    let red_bytes = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    let blue_bytes = [0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    let green_bytes = [0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();

    // Snapshot order matches oracle_reference.js layout:
    // Offset 0: Red [1.0, 0.0, 0.0, 1.0] (Pass 1 - Canvas Target 10 left triangle)
    // Offset 256: Blue [0.0, 0.0, 1.0, 1.0] (Pass 3 - Canvas Target 10 right triangle)
    // Offset 512: Green [0.0, 1.0, 0.0, 1.0] (Pass 2 - Offscreen Target 11 nested pass)
    let rec_red = session
        .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, &red_bytes)
        .expect("red material snapshot");
    let rec_blue = session
        .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::ZERO, &blue_bytes)
        .expect("blue material snapshot");
    let rec_green = session
        .snapshot_material_use(mat_handle, DataVersion::new(3), Epoch::ZERO, &green_bytes)
        .expect("green material snapshot");

    // Pass 1: Outer prefix pass on Canvas Target 10 with clear to Black and Red draw on left half (offset 0)
    session.begin_render_pass("canvas_prefix", [0.0, 0.0, 0.0, 1.0]).expect("begin canvas prefix pass");
    session.record_direct_draw(pipeline_canvas, vb1_id, 3, Some(rec_red)).expect("record red draw");

    // Pass 2: Nested inner pass on Offscreen Target 11 with Green draw (offset 512, draws tri1 on target 11)
    let nested_ctx = RenderContext::new_offscreen(
        ResourceId::new(target_11),
        64,
        64,
        Epoch::ZERO,
    );
    session
        .with_nested_render(nested_ctx, |s| {
            s.begin_render_pass("nested_pass", [0.0, 0.0, 0.0, 1.0]).expect("begin nested pass");
            s.record_direct_draw(pipeline_offscreen, vb1_id, 3, Some(rec_green)).expect("record green draw");
            s.end_render_pass().expect("end nested pass");
            Ok(())
        })
        .expect("with_nested_render must succeed");

    // Pass 3: Outer resumed pass on Canvas Target 10 (with LoadOp::Load) and Blue draw on right half (offset 256, draws tri2 on target 10)
    session.record_direct_draw(pipeline_canvas, vb2_id, 3, Some(rec_blue)).expect("record blue draw");
    session.end_render_pass().expect("end outer canvas resumed pass");

    let session_packet = session
        .build_submission_packet_with_tracker(Some(&tracker))
        .expect("build session packet must succeed");

    for cmd in session_packet.into_commands() {
        packet.push(cmd);
    }

    // Command 13: Copy Offscreen Target 11 to Readback Buffer 20 (Target 10 has no canvas readback seam)
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_11,
        buffer_id: readback_buffer_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

/// Constructs a 25-command nested-pass reentrant submission packet exercising real viewport and scissor commands (§6.7, §8.5, 2v8.4).
///
/// Executes three logical passes through [`FrameSession::with_nested_render`] lowered via [`lower_plan`]:
/// 1. Outer prefix pass on Target 10: Clear to Black, Viewport [0, 0, 64, 64], Scissor [0, 0, 32, 64] (left half),
///    and Red draw (`tri1`, dynamic offset 0).
/// 2. Nested inner pass on Target 11: Clear to Black, Viewport [16, 16, 32, 32], Scissor [16, 16, 32, 32] (centered sub-rect),
///    and Green draw (`tri1`, dynamic offset 512).
/// 3. Outer resumed pass on Target 10: Resume with `LoadOp::Load`, restored Viewport [0, 0, 64, 64], restored Scissor [32, 0, 32, 64] (right half),
///    and Blue draw (`tri2`, dynamic offset 256).
///
/// Copies Target 10 to Readback Buffer 20 and Target 11 to Readback Buffer 21.
///
/// ### Distinct Geometric and Sample Points:
/// - Target 10:
///   - (24, 32): Inside `tri1` and inside Pass 1 scissor [0, 0, 32, 64] -> Red `[255, 0, 0, 255]`.
///   - (56, 32): Inside `tri2` and inside Pass 3 scissor [32, 0, 32, 64] -> Blue `[0, 0, 255, 255]`.
///   - (2, 2): Untouched background outside `tri1`/`tri2` -> Black `[0, 0, 0, 255]`.
/// - Target 11:
///   - (24, 32): Inside `tri1` and inside centered scissor [16, 16, 32, 32] -> Green `[0, 255, 0, 255]`.
///   - (8, 8): Outside centered scissor [16, 16, 32, 32] -> Black `[0, 0, 0, 255]`.
///   - (56, 32): Outside centered scissor [16, 16, 32, 32] -> Black `[0, 0, 0, 255]`.
///
/// If scissor clipping were omitted or restored incorrectly (e.g. Pass 1 scissor leaking into Pass 3),
/// the right half would be clipped out and (56, 32) would remain Black.
///
/// Registers resource IDs in the global generational slot table at generation 1 (§6.5, vqa.6):
/// - Buffers: 1 (uniform), 2 (tri1), 3 (tri2), 20 (readback target 10), 21 (readback target 11)
/// - Textures: 10 (target 10), 11 (target 11)
/// - Pipeline: 200 (flat color pipeline)
pub fn build_nested_viewport_scissor_submission() -> GpuSubmissionPacket {
    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vb1_id (tri1)
        table.register(3);   // vb2_id (tri2)
        table.register(10);  // target_10
        table.register(11);  // target_11
        table.register(20);  // readback_buffer_10_id
        table.register(21);  // readback_buffer_11_id
        table.register(200); // pipeline_id
    });

    let mut packet = GpuSubmissionPacket::new();

    let uniform_buffer_id = 1;
    let vb1_id = 2;
    let vb2_id = 3;
    let target_10 = 10;
    let target_11 = 11;
    let readback_buffer_10_id = 20;
    let readback_buffer_11_id = 21;
    let pipeline_id = 200;

    // 1. Vertex buffer 1: Triangle 1 (left side, covers x in [-1, 0], samples at (24, 32))
    let tri1 = [
        VertexPosUv::new([-1.0, -1.0, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([0.0, 1.0, 0.0], [0.5, 1.0]),
    ];
    let mut tri1_bytes = Vec::with_capacity(tri1.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri1 {
        tri1_bytes.extend_from_slice(&v.to_bytes());
    }

    // 2. Vertex buffer 2: Triangle 2 (right side, covers x in [0, 1], samples at (56, 32))
    let tri2 = [
        VertexPosUv::new([0.0, -1.0, 0.0], [0.5, 0.0]),
        VertexPosUv::new([1.0, -1.0, 0.0], [1.0, 0.0]),
        VertexPosUv::new([1.0, 1.0, 0.0], [1.0, 1.0]),
    ];
    let mut tri2_bytes = Vec::with_capacity(tri2.len() * VertexPosUv::BYTE_SIZE);
    for v in &tri2 {
        tri2_bytes.extend_from_slice(&v.to_bytes());
    }

    // 3. Resource creations (matching oracle device allocations): 10 commands (indices 0..9)
    // Command 0: Uniform buffer 1 (768 bytes: 3 * 256B dynamic slots)
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: 768,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });

    // Commands 1 & 2: Vertex buffer 1
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb1_id,
        size: tri1_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb1_id,
        offset: 0,
        data: tri1_bytes,
    });

    // Commands 3 & 4: Vertex buffer 2
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb2_id,
        size: tri2_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb2_id,
        offset: 0,
        data: tri2_bytes,
    });

    // Command 5: Target Texture 10
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_10,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    // Command 6: Target Texture 11
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_11,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    // Command 7: Readback buffer 20 (for Target 10)
    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_10_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // Command 8: Readback buffer 21 (for Target 11)
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_11_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    let flat_shader = "\
struct ColorUniform {\n\
    color: vec4<f32>,\n\
};\n\
@group(0) @binding(0)\n\
var<uniform> u: ColorUniform;\n\
\n\
struct VertexInput {\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(in.position, 1.0);\n\
}\n\
\n\
@fragment\n\
fn fs_main() -> @location(0) vec4<f32> {\n\
    return u.color;\n\
}\n";

    // Command 9: Pipeline 200
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: flat_shader.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: COLOR_UNIFORM_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 4. Build pass structure through FrameSession with with_nested_render (§6.7, §8.5)
    let root_ctx = RenderContext::new_offscreen(
        ResourceId::new(target_10),
        64,
        64,
        Epoch::ZERO,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .expect("session init must succeed")
        .with_uniform_buffer_id(uniform_buffer_id);

    let mat_handle = Handle::<MaterialDomain>::from_raw(1, 1).expect("valid material handle");
    let red_bytes = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    let blue_bytes = [0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    let green_bytes = [0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();

    // Snapshot order:
    // Slot 0 (offset 0): Red [1.0, 0.0, 0.0, 1.0] (Pass 1 - Target 10 left triangle)
    // Slot 1 (offset 256): Blue [0.0, 0.0, 1.0, 1.0] (Pass 3 - Target 10 right triangle)
    // Slot 2 (offset 512): Green [0.0, 1.0, 0.0, 1.0] (Pass 2 - Target 11 nested pass)
    let rec_red = session
        .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, &red_bytes)
        .expect("red material snapshot");
    let rec_blue = session
        .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::ZERO, &blue_bytes)
        .expect("blue material snapshot");
    let rec_green = session
        .snapshot_material_use(mat_handle, DataVersion::new(3), Epoch::ZERO, &green_bytes)
        .expect("green material snapshot");

    // Pass 1: Outer prefix on Target 10: Clear to Black, Viewport [0, 0, 64, 64], Scissor [0, 0, 32, 64]
    session.set_viewport(0, 0, 64, 64);
    session.set_scissor(0, 0, 32, 64);
    session.set_scissor_test(true);
    session.begin_render_pass("outer_prefix", [0.0, 0.0, 0.0, 1.0]).expect("begin outer prefix pass");
    session.record_direct_draw(pipeline_id, vb1_id, 3, Some(rec_red)).expect("record red draw");

    // Pass 2: Nested inner pass on Target 11: Clear to Black, Viewport [16, 16, 32, 32], Scissor [16, 16, 32, 32]
    let nested_ctx = RenderContext::new_offscreen(
        ResourceId::new(target_11),
        64,
        64,
        Epoch::ZERO,
    );
    session
        .with_nested_render(nested_ctx, |s| {
            s.set_viewport(16, 16, 32, 32);
            s.set_scissor(16, 16, 32, 32);
            s.set_scissor_test(true);
            s.begin_render_pass("nested_pass", [0.0, 0.0, 0.0, 1.0])?;
            s.record_direct_draw(pipeline_id, vb1_id, 3, Some(rec_green))?;
            s.end_render_pass()?;
            Ok(())
        })
        .expect("with_nested_render must succeed");

    // Pass 3: Outer resumed on Target 10: Viewport restored to [0, 0, 64, 64], set scissor to right half [32, 0, 32, 64]
    session.set_scissor(32, 0, 32, 64);
    session.record_direct_draw(pipeline_id, vb2_id, 3, Some(rec_blue)).expect("record blue draw");
    session.end_render_pass().expect("end outer resumed pass");

    let session_packet = session
        .build_submission_packet()
        .expect("build session packet must succeed");

    for cmd in session_packet.into_commands() {
        packet.push(cmd);
    }

    // Command 23: Copy Target 10 to Readback Buffer 20
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_10,
        buffer_id: readback_buffer_10_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    // Command 24: Copy Target 11 to Readback Buffer 21
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_11,
        buffer_id: readback_buffer_11_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

/// Builds a real WGSL instanced range submission packet testing `instance_count`, `first_vertex`, and `first_instance` (§8.5, 2v8.5).
///
/// Registers resource IDs in the generational slot table at generation 1 (§6.5, vqa.6):
/// - Buffers: 2 (vertex buffer), 20 (readback buffer)
/// - Texture: 10 (target texture)
/// - Pipeline: 200 (instanced color pipeline)
pub fn build_draw_parameters_submission() -> GpuSubmissionPacket {
    with_global_resource_table(|table| {
        table.register(2);   // vb2_id
        table.register(10);  // target_10
        table.register(20);  // readback_buffer_20_id
        table.register(200); // pipeline_id
    });

    let mut packet = GpuSubmissionPacket::new();

    let vb2_id = 2;
    let target_10 = 10;
    let readback_buffer_20_id = 20;
    let pipeline_id = 200;

    // 1. Vertex buffer 2: 6 vertices (stride 20: 12 bytes position + 8 bytes UV)
    // First 3 vertices: offscreen / degenerate [3.0, 3.0, 0.0]
    // Next 3 vertices: centered triangle [-0.2, -0.5, 0.0], [0.2, -0.5, 0.0], [0.0, 0.5, 0.0]
    let vertices = [
        VertexPosUv::new([3.0, 3.0, 0.0], [0.0, 0.0]),
        VertexPosUv::new([3.0, 3.0, 0.0], [0.0, 0.0]),
        VertexPosUv::new([3.0, 3.0, 0.0], [0.0, 0.0]),
        VertexPosUv::new([-0.2, -0.5, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.2, -0.5, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.0, 0.5, 0.0], [0.0, 0.0]),
    ];
    let mut vb_bytes = Vec::with_capacity(vertices.len() * VertexPosUv::BYTE_SIZE);
    for v in &vertices {
        vb_bytes.extend_from_slice(&v.to_bytes());
    }

    // Command 0: CreateBuffer 2
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb2_id,
        size: vb_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });

    // Command 1: WriteBuffer 2
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb2_id,
        offset: 0,
        data: vb_bytes.clone(),
    });

    // Command 2: CreateTexture 10 (64x64, rgba8unorm)
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_10,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    // Command 3: CreateBuffer 20 (readback, 256 * 64 = 16384 bytes)
    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_20_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // Command 4: CreatePipeline 200
    let instanced_shader = "\
struct VertexInput {\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
    @builtin(instance_index) instance_idx: u32,\n\
};\n\
\n\
struct VertexOutput {\n\
    @builtin(position) position: vec4<f32>,\n\
    @location(0) color: vec4<f32>,\n\
};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> VertexOutput {\n\
    var out: VertexOutput;\n\
    var offset_x: f32 = 0.0;\n\
    var col: vec4<f32> = vec4<f32>(0.0, 0.0, 1.0, 1.0);\n\
    if (in.instance_idx == 5u) {\n\
        offset_x = -0.5;\n\
        col = vec4<f32>(1.0, 0.0, 0.0, 1.0);\n\
    } else if (in.instance_idx == 6u) {\n\
        offset_x = 0.5;\n\
        col = vec4<f32>(0.0, 1.0, 0.0, 1.0);\n\
    } else {\n\
        offset_x = 10.0;\n\
        col = vec4<f32>(0.0, 0.0, 1.0, 1.0);\n\
    }\n\
    out.position = vec4<f32>(in.position.x + offset_x, in.position.y, in.position.z, 1.0);\n\
    out.color = col;\n\
    return out;\n\
}\n\
\n\
@fragment\n\
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {\n\
    return in.color;\n\
}\n";

    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: instanced_shader.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: false,
        uniform_size: 0,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // 2. Build pass structure through FrameSession with record_direct_draw_with_range (§6.7, §8.5, 2v8.5)
    let root_ctx = RenderContext::new_offscreen(
        ResourceId::new(target_10),
        64,
        64,
        Epoch::ZERO,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .expect("session init must succeed");

    session
        .begin_render_pass("draw_params_pass", [0.0, 0.0, 0.0, 1.0])
        .expect("begin draw_params_pass must succeed");

    session
        .record_direct_draw_with_range(pipeline_id, vb2_id, [3, 2, 3, 5], None)
        .expect("record draw with range must succeed");

    session
        .end_render_pass()
        .expect("end draw_params_pass must succeed");

    let session_packet = session
        .build_submission_packet()
        .expect("build session packet must succeed");

    for cmd in session_packet.into_commands() {
        packet.push(cmd);
    }

    // 3. Copy Target 10 to Readback Buffer 20
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_10,
        buffer_id: readback_buffer_20_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

// -----------------------------------------------------------------------------
// Compiled Wasm Exports
// -----------------------------------------------------------------------------

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a static WGSL triangle first-frame submission packet and returns the raw binary bytes.
pub fn gpu_bridge_build_triangle_packet() -> Vec<u8> {
    build_triangle_submission()
        .encode()
        .expect("static triangle packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a static WGSL triangle first-frame submission packet and returns the raw binary bytes (canonical alias).
pub fn f3d_build_first_frame_packet() -> Vec<u8> {
    build_triangle_submission()
        .encode()
        .expect("static triangle packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a Red-A / Blue-B snapshot isolation submission packet with optional per-use versioning.
pub fn gpu_bridge_build_red_blue_packet(per_use_versioned: bool) -> Vec<u8> {
    build_red_a_blue_b_submission(per_use_versioned)
        .encode()
        .expect("static red-blue packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a Red-A / Blue-B snapshot isolation submission packet with optional per-use versioning (canonical alias).
pub fn f3d_build_red_a_blue_b_packet(per_use_versioned: bool) -> Vec<u8> {
    build_red_a_blue_b_submission(per_use_versioned)
        .encode()
        .expect("static red-blue packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a bundle-then-direct-draw submission packet and returns the raw binary bytes.
pub fn gpu_bridge_build_bundle_direct_draw_packet() -> Vec<u8> {
    build_bundle_then_direct_draw_submission()
        .encode()
        .expect("static bundle-direct-draw packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a bundle-then-direct-draw submission packet and returns the raw binary bytes (canonical alias).
pub fn f3d_build_bundle_direct_draw_packet() -> Vec<u8> {
    gpu_bridge_build_bundle_direct_draw_packet()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_bundle_direct_draw_packet` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_build_bundle_direct_draw_packet() -> Vec<u8> {
    build_bundle_then_direct_draw_submission()
        .encode()
        .expect("static bundle-direct-draw packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_bundle_direct_draw_packet` (canonical alias).
#[must_use]
pub fn f3d_build_bundle_direct_draw_packet() -> Vec<u8> {
    gpu_bridge_build_bundle_direct_draw_packet()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes an AffineRows transform submission packet and returns the raw binary bytes (§6.1, §6.2, 05.6, vqa.2).
pub fn gpu_bridge_build_affine_rows_transform_packet() -> Vec<u8> {
    build_affine_rows_transform_submission()
        .encode()
        .expect("static affine-rows transform packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes an AffineRows transform submission packet and returns the raw binary bytes (canonical alias).
pub fn f3d_build_affine_rows_transform_packet() -> Vec<u8> {
    gpu_bridge_build_affine_rows_transform_packet()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_affine_rows_transform_packet` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_build_affine_rows_transform_packet() -> Vec<u8> {
    build_affine_rows_transform_submission()
        .encode()
        .expect("static affine-rows transform packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_transform_packet` (canonical alias).
#[must_use]
pub fn f3d_build_affine_rows_transform_packet() -> Vec<u8> {
    gpu_bridge_build_affine_rows_transform_packet()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a nested-pass reentrant execution packet (§6.7, 2v8.4).
pub fn gpu_bridge_build_nested_pass_packet() -> Vec<u8> {
    build_nested_pass_submission()
        .encode()
        .expect("static nested pass packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a nested-pass reentrant execution packet (canonical alias).
pub fn f3d_build_nested_pass_packet() -> Vec<u8> {
    gpu_bridge_build_nested_pass_packet()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_nested_pass_packet` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_build_nested_pass_packet() -> Vec<u8> {
    build_nested_pass_submission()
        .encode()
        .expect("static nested pass packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_nested_pass_packet` (canonical alias).
#[must_use]
pub fn f3d_build_nested_pass_packet() -> Vec<u8> {
    gpu_bridge_build_nested_pass_packet()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a nested-pass canvas-variant reentrant execution packet (§6.7, §8.5, 2v8.4).
///
/// Registers resource IDs (buffers 1, 2, 3, 20, textures 10, 11, pipelines 200, 201) in the
/// generational slot table at generation 1. Does not submit the frame (Mail 7117); submission
/// is performed by the bridge caller.
pub fn gpu_bridge_build_nested_canvas_pass_packet() -> Vec<u8> {
    build_nested_canvas_pass_submission()
        .encode()
        .expect("static nested canvas pass packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a nested-pass canvas-variant reentrant execution packet (canonical alias).
///
/// Registers resource IDs (buffers 1, 2, 3, 20, textures 10, 11, pipelines 200, 201) in the
/// generational slot table at generation 1. Does not submit the frame (Mail 7117).
pub fn f3d_build_nested_canvas_pass_packet() -> Vec<u8> {
    gpu_bridge_build_nested_canvas_pass_packet()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_nested_canvas_pass_packet` for host verification and unit tests.
///
/// Registers resource IDs (buffers 1, 2, 3, 20, textures 10, 11, pipelines 200, 201) in the
/// generational slot table at generation 1. Does not submit the frame (Mail 7117).
#[must_use]
pub fn gpu_bridge_build_nested_canvas_pass_packet() -> Vec<u8> {
    build_nested_canvas_pass_submission()
        .encode()
        .expect("static nested canvas pass packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_nested_canvas_pass_packet` (canonical alias).
///
/// Registers resource IDs (buffers 1, 2, 3, 20, textures 10, 11, pipelines 200, 201) in the
/// generational slot table at generation 1. Does not submit the frame (Mail 7117).
#[must_use]
pub fn f3d_build_nested_canvas_pass_packet() -> Vec<u8> {
    gpu_bridge_build_nested_canvas_pass_packet()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a nested-pass viewport/scissor execution packet (§6.7, §8.5, 2v8.4).
///
/// Registers resource IDs (buffers 1, 2, 3, 20, 21, textures 10, 11, pipeline 200) in the
/// generational slot table at generation 1.
pub fn gpu_bridge_build_nested_viewport_scissor_packet() -> Vec<u8> {
    build_nested_viewport_scissor_submission()
        .encode()
        .expect("static nested viewport/scissor packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a nested-pass viewport/scissor execution packet (canonical alias).
pub fn f3d_build_nested_viewport_scissor_packet() -> Vec<u8> {
    gpu_bridge_build_nested_viewport_scissor_packet()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_nested_viewport_scissor_packet` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_build_nested_viewport_scissor_packet() -> Vec<u8> {
    build_nested_viewport_scissor_submission()
        .encode()
        .expect("static nested viewport/scissor packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_nested_viewport_scissor_packet` (canonical alias).
#[must_use]
pub fn f3d_build_nested_viewport_scissor_packet() -> Vec<u8> {
    gpu_bridge_build_nested_viewport_scissor_packet()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a draw parameters execution packet (§6.7, §8.5, 2v8.5).
///
/// Registers resource IDs (buffers 2, 20, texture 10, pipeline 200) in the
/// generational slot table at generation 1.
pub fn gpu_bridge_build_draw_parameters_packet() -> Vec<u8> {
    build_draw_parameters_submission()
        .encode()
        .expect("static draw parameters packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a draw parameters execution packet (canonical alias).
pub fn f3d_build_draw_parameters_packet() -> Vec<u8> {
    gpu_bridge_build_draw_parameters_packet()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_draw_parameters_packet` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_build_draw_parameters_packet() -> Vec<u8> {
    build_draw_parameters_submission()
        .encode()
        .expect("static draw parameters packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_draw_parameters_packet` (canonical alias).
#[must_use]
pub fn f3d_build_draw_parameters_packet() -> Vec<u8> {
    gpu_bridge_build_draw_parameters_packet()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Evaluates readback epoch freshness against a target region epoch in the browser (§5.8, 58j.3).
///
/// Reconstructs both [`Epoch`] values from their `(high, low)` 32-bit word pairs
/// using [`Epoch::from_words`] and returns `true` if `readback_epoch == region_epoch`,
/// or `false` to discard the stale readback.
pub fn gpu_bridge_try_publish_readback(
    readback_epoch_hi: u32,
    readback_epoch_lo: u32,
    region_epoch_hi: u32,
    region_epoch_lo: u32,
) -> bool {
    try_publish_readback_words(
        readback_epoch_hi,
        readback_epoch_lo,
        region_epoch_hi,
        region_epoch_lo,
    )
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Evaluates readback epoch freshness against a target region epoch (canonical alias).
pub fn f3d_try_publish_readback(
    readback_epoch_hi: u32,
    readback_epoch_lo: u32,
    region_epoch_hi: u32,
    region_epoch_lo: u32,
) -> bool {
    gpu_bridge_try_publish_readback(
        readback_epoch_hi,
        readback_epoch_lo,
        region_epoch_hi,
        region_epoch_lo,
    )
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_try_publish_readback` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_try_publish_readback(
    readback_epoch_hi: u32,
    readback_epoch_lo: u32,
    region_epoch_hi: u32,
    region_epoch_lo: u32,
) -> bool {
    try_publish_readback_words(
        readback_epoch_hi,
        readback_epoch_lo,
        region_epoch_hi,
        region_epoch_lo,
    )
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Validates a resource handle index and generation against the bridge slot table (§6.5, vqa.6).
///
/// Returns `true` if the handle's generation matches the slot's current active generation.
/// Returns `false` if the handle is stale (older generation after slot reuse), vacant, or
/// if generation is 0 (rejected by [`HandleError::InvalidGeneration`]).
pub fn gpu_bridge_check_resource_handle(index: u32, generation: u32) -> bool {
    check_resource_handle(index, generation).unwrap_or(false)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Validates a resource handle index and generation against the bridge slot table (canonical alias).
pub fn f3d_check_resource_handle(index: u32, generation: u32) -> bool {
    gpu_bridge_check_resource_handle(index, generation)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Advances a resource slot's generation (simulating release and reuse) for ABA testing.
pub fn gpu_bridge_advance_resource_generation(index: u32) -> u32 {
    advance_resource_slot_generation(index).unwrap_or(0)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Advances a resource slot's generation (canonical alias).
pub fn f3d_advance_resource_generation(index: u32) -> u32 {
    gpu_bridge_advance_resource_generation(index)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_check_resource_handle` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_check_resource_handle(index: u32, generation: u32) -> bool {
    check_resource_handle(index, generation).unwrap_or(false)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_check_resource_handle` (canonical alias).
#[must_use]
pub fn f3d_check_resource_handle(index: u32, generation: u32) -> bool {
    gpu_bridge_check_resource_handle(index, generation)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_advance_resource_generation`.
#[must_use]
pub fn gpu_bridge_advance_resource_generation(index: u32) -> u32 {
    advance_resource_slot_generation(index).unwrap_or(0)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_advance_resource_generation` (canonical alias).
#[must_use]
pub fn f3d_advance_resource_generation(index: u32) -> u32 {
    gpu_bridge_advance_resource_generation(index)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Enters a linear memory borrow scope, returning the unique [`BorrowToken`] numeric value (§6.6, §13.1, vqa.6).
/// Returns `0` if entering fails (e.g. re-entrant borrow while already active).
pub fn gpu_bridge_borrow_enter() -> u64 {
    borrow_enter()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Enters a linear memory borrow scope (canonical alias).
pub fn f3d_borrow_enter() -> u64 {
    gpu_bridge_borrow_enter()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Exits an active linear memory borrow scope given its token (§6.6, §13.1, vqa.6).
/// Returns `true` if exit succeeded, or `false` if the token mismatched or no borrow was active.
pub fn gpu_bridge_borrow_exit(token: u64) -> bool {
    borrow_exit(token)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Exits an active linear memory borrow scope (canonical alias).
pub fn f3d_borrow_exit(token: u64) -> bool {
    gpu_bridge_borrow_exit(token)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Attempts to record linear memory growth of `pages` WebAssembly memory pages (§6.6, §13.1, vqa.6).
/// Returns `true` if growth succeeded, or `false` if growth was blocked by an active borrow.
pub fn gpu_bridge_try_grow_memory(pages: u32) -> bool {
    try_grow_memory(pages)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Attempts to record linear memory growth (canonical alias).
pub fn f3d_try_grow_memory(pages: u32) -> bool {
    gpu_bridge_try_grow_memory(pages)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Returns the current linear memory growth generation counter (§6.6, §13.1, vqa.6).
/// Hosts inspect this counter to detect Detached ArrayBuffer conditions and validate cached views.
pub fn gpu_bridge_borrow_growth_generation() -> u64 {
    borrow_growth_generation()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Returns the current linear memory growth generation counter (canonical alias).
pub fn f3d_borrow_growth_generation() -> u64 {
    gpu_bridge_borrow_growth_generation()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Validates an affine transform payload against GPU wire layout specifications (§6.1, §6.2, vqa.6).
/// Accepts either a 48-byte `AffineRows` row-major wire record (with translation) or a 64-byte column-major `Matrix4`.
/// Returns `0` if valid, or a non-zero code mapped from [`LayoutError`] variants on failure.
pub fn gpu_bridge_validate_affine_rows(bytes: Vec<u8>) -> i32 {
    validate_affine_rows(&bytes)
        .map(|()| LAYOUT_VALIDATION_OK)
        .unwrap_or_else(|err| layout_error_to_code(&err))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Validates an affine matrix or wire record payload (canonical alias).
pub fn f3d_validate_affine_rows(bytes: Vec<u8>) -> i32 {
    gpu_bridge_validate_affine_rows(bytes)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_borrow_enter`.
#[must_use]
pub fn gpu_bridge_borrow_enter() -> u64 {
    borrow_enter()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_borrow_enter` (canonical alias).
#[must_use]
pub fn f3d_borrow_enter() -> u64 {
    gpu_bridge_borrow_enter()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_borrow_exit`.
#[must_use]
pub fn gpu_bridge_borrow_exit(token: u64) -> bool {
    borrow_exit(token)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_borrow_exit` (canonical alias).
#[must_use]
pub fn f3d_borrow_exit(token: u64) -> bool {
    gpu_bridge_borrow_exit(token)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_try_grow_memory`.
#[must_use]
pub fn gpu_bridge_try_grow_memory(pages: u32) -> bool {
    try_grow_memory(pages)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_try_grow_memory` (canonical alias).
#[must_use]
pub fn f3d_try_grow_memory(pages: u32) -> bool {
    gpu_bridge_try_grow_memory(pages)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_borrow_growth_generation`.
#[must_use]
pub fn gpu_bridge_borrow_growth_generation() -> u64 {
    borrow_growth_generation()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_borrow_growth_generation` (canonical alias).
#[must_use]
pub fn f3d_borrow_growth_generation() -> u64 {
    gpu_bridge_borrow_growth_generation()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_validate_affine_rows` (validates 48-byte `AffineRows` or 64-byte `Matrix4`).
#[must_use]
pub fn gpu_bridge_validate_affine_rows(bytes: Vec<u8>) -> i32 {
    validate_affine_rows(&bytes)
        .map(|()| LAYOUT_VALIDATION_OK)
        .unwrap_or_else(|err| layout_error_to_code(&err))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_validate_affine_rows` (canonical alias).
#[must_use]
pub fn f3d_validate_affine_rows(bytes: Vec<u8>) -> i32 {
    gpu_bridge_validate_affine_rows(bytes)
}

// -----------------------------------------------------------------------------
// Readback Publication Gate (§5.8, 58j.3)
// -----------------------------------------------------------------------------

/// State tracking a published readback payload guarded by epoch freshness.
///
/// If a state region's epoch advances while an asynchronous readback (e.g. `mapAsync`)
/// is in flight, attempting to publish the readback will detect the stale epoch and
/// discard the data without overwriting the region's newer state.
///
/// For browser callers, [`gpu_bridge_try_publish_readback`] provides the callable gate
/// accepting `(high, low)` epoch words and evaluating freshness.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadbackPublicationState {
    /// The epoch under which the current data was published.
    pub published_epoch: Option<Epoch>,
    /// The published raw byte payload, if any.
    pub data: Option<Vec<u8>>,
}

impl ReadbackPublicationState {
    /// Creates a new, empty publication state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the published epoch, or `None` if no readback has been published.
    #[must_use]
    pub fn published_epoch(&self) -> Option<Epoch> {
        self.published_epoch
    }

    /// Returns the published data slice, or `None` if no readback has been published.
    #[must_use]
    pub fn data(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }

    /// Attempts to publish readback bytes guarded by the target region's current epoch.
    ///
    /// Mirrors `PublishedState::try_publish`. If `readback_epoch` does not match
    /// `target_region.current_epoch()`, the readback is considered stale and discarded,
    /// returning `false`. When epochs match, stores the data and returns `true`.
    pub fn try_publish_readback<D: f3d_core::handle::Domain>(
        &mut self,
        target_region: &RegionState<D>,
        readback_epoch: Epoch,
        data: Vec<u8>,
    ) -> bool {
        if readback_epoch != target_region.current_epoch() {
            false
        } else {
            self.published_epoch = Some(readback_epoch);
            self.data = Some(data);
            true
        }
    }
}

/// Freestanding publication gate verifying readback epoch freshness against a target region state.
///
/// Returns `true` if `readback_epoch` matches `target_region.current_epoch()`, or `false` to discard.
#[must_use]
pub fn try_publish_readback<D: f3d_core::handle::Domain>(
    target_region: &RegionState<D>,
    readback_epoch: Epoch,
) -> bool {
    readback_epoch == target_region.current_epoch()
}

/// Reconstructs readback and region [`Epoch`]s from their `(high, low)` 32-bit word pairs
/// and evaluates whether the readback's epoch matches the region's current epoch.
#[must_use]
pub fn try_publish_readback_words(
    readback_epoch_hi: u32,
    readback_epoch_lo: u32,
    region_epoch_hi: u32,
    region_epoch_lo: u32,
) -> bool {
    let readback_epoch = Epoch::from_words(readback_epoch_hi, readback_epoch_lo);
    let region_epoch = Epoch::from_words(region_epoch_hi, region_epoch_lo);
    readback_epoch == region_epoch
}

// -----------------------------------------------------------------------------
// Generational Resource Handle Slot Table & Publication Gate (§6.5, vqa.6)
// -----------------------------------------------------------------------------

/// Domain marker for GPU bridge resource handles (§6.5, vqa.6).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct GpuResourceDomain;

impl Domain for GpuResourceDomain {
    const DOMAIN_NAME: &'static str = "gpu_resource";
}

/// A generational slot tracking allocation state and current generation stamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceSlot {
    generation: NonZeroU32,
    active: bool,
}

impl ResourceSlot {
    /// Creates a new active slot at the given generation.
    #[must_use]
    pub const fn new(generation: NonZeroU32) -> Self {
        Self {
            generation,
            active: true,
        }
    }

    /// Returns the current generational counter for this slot.
    #[must_use]
    pub const fn generation(&self) -> NonZeroU32 {
        self.generation
    }

    /// Returns whether this slot is currently allocated and active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

/// Generational slot table backing GPU bridge resource handle verification (§6.5, vqa.6).
///
/// Ensures ABA safety: when a resource slot is released and subsequently reused,
/// its generation stamp increments. Older handles with stale generation stamps
/// return `false` on verification, preventing collisions and dangling handle publication.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceSlotTable {
    slots: Vec<Option<ResourceSlot>>,
}

impl ResourceSlotTable {
    /// Creates an empty resource slot table.
    #[must_use]
    pub const fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Registers a resource ID into the slot table at initial generation 1 if new,
    /// or reactivates an existing slot. Returns the typed [`Handle`].
    pub fn register(&mut self, index: u32) -> Handle<GpuResourceDomain> {
        let idx = index as usize;
        if idx >= self.slots.len() {
            self.slots.resize(idx + 1, None);
        }
        let initial_gen = NonZeroU32::new(1).expect("1 is non-zero");
        match &mut self.slots[idx] {
            Some(slot) => {
                slot.active = true;
                Handle::<GpuResourceDomain>::new(index, slot.generation)
            }
            None => {
                let slot = ResourceSlot {
                    generation: initial_gen,
                    active: true,
                };
                self.slots[idx] = Some(slot);
                Handle::<GpuResourceDomain>::new(index, initial_gen)
            }
        }
    }

    /// Releases a resource slot and advances its generation counter for subsequent reuse (preventing ABA).
    /// Returns the new generation value.
    pub fn release_and_advance(&mut self, index: u32) -> Result<u32, HandleError> {
        let idx = index as usize;
        if idx >= self.slots.len() {
            return Err(HandleError::IndexOutOfBounds {
                index,
                capacity: self.slots.len() as u32,
            });
        }
        let slot = self.slots[idx].as_mut().ok_or(HandleError::SlotVacant { index })?;
        if !slot.active {
            return Err(HandleError::SlotVacant { index });
        }
        let next_gen = slot
            .generation
            .get()
            .checked_add(1)
            .and_then(NonZeroU32::new)
            .ok_or(HandleError::GenerationOverflow { index })?;
        slot.generation = next_gen;
        slot.active = false;
        Ok(next_gen.get())
    }

    /// Reallocates a previously released slot at its new incremented generation stamp.
    pub fn reallocate(&mut self, index: u32) -> Result<Handle<GpuResourceDomain>, HandleError> {
        let idx = index as usize;
        if idx >= self.slots.len() {
            return Err(HandleError::IndexOutOfBounds {
                index,
                capacity: self.slots.len() as u32,
            });
        }
        let slot = self.slots[idx].as_mut().ok_or(HandleError::SlotVacant { index })?;
        slot.active = true;
        Ok(Handle::<GpuResourceDomain>::new(index, slot.generation))
    }

    /// Validates a handle index and generation against the slot table.
    ///
    /// - If `generation == 0`, returns `Err(HandleError::InvalidGeneration)`.
    /// - If the slot is active and `handle.generation() == slot.generation`, returns `Ok(true)` (fresh).
    /// - If the handle's generation is older than the slot's current generation (slot reused after release), returns `Ok(false)` (stale ABA handle rejected).
    /// - If the slot is vacant, unallocated, or inactive, returns `Ok(false)`.
    pub fn check_handle(&self, index: u32, generation: u32) -> Result<bool, HandleError> {
        // Generation zero is strictly rejected via HandleError (§6.5)
        let handle = Handle::<GpuResourceDomain>::from_raw(index, generation)?;

        let idx = index as usize;
        if idx >= self.slots.len() {
            return Ok(false);
        }
        let Some(slot) = &self.slots[idx] else {
            return Ok(false);
        };
        if !slot.active {
            return Ok(false);
        }
        // Handle generation must match the slot's current active generation.
        // If older (stale after reuse) or mismatch, returns false.
        Ok(handle.generation() == slot.generation)
    }
}

static GLOBAL_RESOURCE_SLOT_TABLE: Mutex<Option<ResourceSlotTable>> = Mutex::new(None);

/// Executes a closure with mutable access to the global resource slot table.
pub fn with_global_resource_table<R>(f: impl FnOnce(&mut ResourceSlotTable) -> R) -> R {
    let mut guard = match GLOBAL_RESOURCE_SLOT_TABLE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let table = guard.get_or_insert_with(ResourceSlotTable::new);
    f(table)
}

/// Validates a resource handle index and generation against the global resource slot table.
///
/// Returns `Ok(true)` if fresh, `Ok(false)` if stale (older generation after slot reuse),
/// and `Err(HandleError::InvalidGeneration)` if `generation == 0`.
pub fn check_resource_handle(index: u32, generation: u32) -> Result<bool, HandleError> {
    with_global_resource_table(|table| table.check_handle(index, generation))
}

/// Advances a slot's generation (simulating release and reuse) in the global slot table.
pub fn advance_resource_slot_generation(index: u32) -> Result<u32, HandleError> {
    with_global_resource_table(|table| {
        let new_gen = table.release_and_advance(index)?;
        table.reallocate(index)?;
        Ok(new_gen)
    })
}

// -----------------------------------------------------------------------------
// Linear Memory Borrow Scope State (§6.6, §13.1, vqa.6)
// -----------------------------------------------------------------------------

static GLOBAL_BORROW_SCOPE: Mutex<Option<BorrowScope>> = Mutex::new(None);

/// Executes a closure with mutable access to the global Wasm linear memory borrow scope.
pub fn with_global_borrow_scope<R>(f: impl FnOnce(&mut BorrowScope) -> R) -> R {
    let mut guard = match GLOBAL_BORROW_SCOPE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let scope = guard.get_or_insert_with(BorrowScope::new);
    f(scope)
}

/// Returns the current growth generation counter from the global linear memory borrow scope (§6.6, §13.1, vqa.6).
#[must_use]
pub fn borrow_growth_generation() -> u64 {
    with_global_borrow_scope(|scope| scope.growth_generation())
}

/// Enters a linear memory borrow scope, returning the unique [`BorrowToken`] numeric value.
///
/// Returns `0` if entering fails due to re-entrancy while a borrow is already active.
/// Panics on counter exhaustion ([`OwnershipError::VersionOverflow`]).
#[must_use]
pub fn borrow_enter() -> u64 {
    with_global_borrow_scope(|scope| match scope.enter() {
        Ok(tok) => tok.get(),
        Err(OwnershipError::BorrowScopeReentry { .. }) => 0,
        Err(OwnershipError::VersionOverflow { current }) => {
            panic!("linear memory borrow scope token counter overflowed: current={current}");
        }
        Err(other) => panic!("unexpected borrow_enter failure: {other:?}"),
    })
}

/// Exits an active linear memory borrow scope given its token.
///
/// Returns `true` if exit succeeded.
/// Returns `false` if the token mismatched or no borrow was active.
#[must_use]
pub fn borrow_exit(token: u64) -> bool {
    with_global_borrow_scope(|scope| match scope.exit(BorrowToken::new(token)) {
        Ok(()) => true,
        Err(OwnershipError::BorrowTokenMismatch { .. } | OwnershipError::BorrowScopeNotActive) => {
            false
        }
        Err(other) => panic!("unexpected borrow_exit failure: {other:?}"),
    })
}

/// Attempts to record linear memory growth of `pages` WebAssembly memory pages.
///
/// When `pages == 0`, acts as a size query / probe that checks whether growth is permitted
/// without advancing `growth_generation`.
///
/// Returns `true` if growth succeeded or `pages == 0` query passed.
/// Returns `false` if growth was blocked by an active borrow scope.
#[must_use]
pub fn try_grow_memory(pages: u32) -> bool {
    with_global_borrow_scope(|scope| {
        if pages == 0 {
            !scope.is_growth_blocked()
        } else {
            match scope.record_growth(pages) {
                Ok(_) => true,
                Err(OwnershipError::LinearMemoryGrowthBlocked { .. }) => false,
                Err(OwnershipError::VersionOverflow { current }) => {
                    panic!("linear memory growth generation counter overflowed: current={current}");
                }
                Err(other) => panic!("unexpected record_growth failure: {other:?}"),
            }
        }
    })
}

// -----------------------------------------------------------------------------
// AffineRows GPU Wire Layout Validation (§6.1, §6.2, vqa.6)
// -----------------------------------------------------------------------------

/// Status code for successful layout validation.
pub const LAYOUT_VALIDATION_OK: i32 = 0;
/// Status code indicating buffer slice is smaller than required layout size.
pub const LAYOUT_VALIDATION_BUFFER_TOO_SMALL: i32 = 1;
/// Status code indicating a matrix contains non-affine perspective elements or invalid scale.
pub const LAYOUT_VALIDATION_NON_AFFINE_MATRIX: i32 = 2;
/// Status code indicating an unaligned buffer offset.
pub const LAYOUT_VALIDATION_UNALIGNED_OFFSET: i32 = 3;
/// Status code indicating an incompatible target GPU layout.
pub const LAYOUT_VALIDATION_INCOMPATIBLE_TARGET: i32 = 4;
/// Status code indicating texture row-pitch violates the 256-byte alignment rule.
pub const LAYOUT_VALIDATION_UNALIGNED_BYTES_PER_ROW: i32 = 5;
/// Status code indicating a writeBuffer copy violates 4-byte alignment.
pub const LAYOUT_VALIDATION_UNALIGNED_WRITE_BUFFER: i32 = 6;
/// Status code indicating arithmetic overflow in layout calculation.
pub const LAYOUT_VALIDATION_CALCULATION_OVERFLOW: i32 = 7;

/// Maps a [`LayoutError`] to a stable numeric status code for the JS/Wasm boundary.
#[must_use]
pub fn layout_error_to_code(err: &LayoutError) -> i32 {
    match err {
        LayoutError::BufferTooSmall { .. } => LAYOUT_VALIDATION_BUFFER_TOO_SMALL,
        LayoutError::NonAffineMatrix => LAYOUT_VALIDATION_NON_AFFINE_MATRIX,
        LayoutError::UnalignedOffset { .. } => LAYOUT_VALIDATION_UNALIGNED_OFFSET,
        LayoutError::IncompatibleTargetLayout { .. } => LAYOUT_VALIDATION_INCOMPATIBLE_TARGET,
        LayoutError::UnalignedBytesPerRow { .. } => LAYOUT_VALIDATION_UNALIGNED_BYTES_PER_ROW,
        LayoutError::UnalignedWriteBuffer { .. } => LAYOUT_VALIDATION_UNALIGNED_WRITE_BUFFER,
        LayoutError::CalculationOverflow => LAYOUT_VALIDATION_CALCULATION_OVERFLOW,
    }
}

/// Validates an affine transform payload against GPU wire layout specifications (§6.1, §6.2, vqa.6).
///
/// Accepts two canonical layout shapes:
/// - **48-byte wire record** ([`AffineRows`], 12 little-endian floats):
///   Three row-major `vec4<f32>` rows (`r0, r1, r2`) where translation is stored in each row's
///   fourth component (`r0.w = tx`, `r1.w = ty`, `r2.w = tz`).
///   Row 3 is implicitly `[0.0, 0.0, 0.0, 1.0]` by structural construction.
///   Decoded via [`AffineRows::from_bytes`] and validated to ensure all 12 float elements are finite.
///   Returns `Err(LayoutError::NonAffineMatrix)` if any float is non-finite (`NaN` or `+/-Inf`).
/// - **64-byte matrix** (Three.js `Matrix4.elements`, 16 little-endian floats):
///   Full 4x4 column-major matrix, verified via [`AffineRows::from_column_major`].
///   Row 3 must be strictly affine: `e[3] == 0.0 && e[7] == 0.0 && e[11] == 0.0 && e[15] == 1.0`.
///   Returns `Err(LayoutError::NonAffineMatrix)` on non-zero perspective coefficients, non-unit homogeneous scale,
///   or non-finite float elements.
///
/// Any other buffer length is rejected:
/// - Less than 48 bytes: returns `Err(LayoutError::BufferTooSmall { required: 48, provided: len })`.
/// - Between 49 and 63 bytes: returns `Err(LayoutError::BufferTooSmall { required: 64, provided: len })`.
/// - Greater than 64 bytes: returns `Err(LayoutError::IncompatibleTargetLayout { target_size: 64, source_size: len })`.
///
/// Returns `Ok(())` on success, or [`LayoutError`] on failure.
pub fn validate_affine_rows(bytes: &[u8]) -> Result<(), LayoutError> {
    if bytes.len() == AFFINE_ROWS_BYTES {
        let mut arr = [0u8; AFFINE_ROWS_BYTES];
        arr.copy_from_slice(bytes);
        let rows = AffineRows::from_bytes(&arr);
        if !rows.r0.iter().all(|x| x.is_finite())
            || !rows.r1.iter().all(|x| x.is_finite())
            || !rows.r2.iter().all(|x| x.is_finite())
        {
            return Err(LayoutError::NonAffineMatrix);
        }
        Ok(())
    } else if bytes.len() == 64 {
        let mut elements = [0.0f32; 16];
        for i in 0..16 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
            elements[i] = f32::from_le_bytes(b);
        }
        if !elements.iter().all(|x| x.is_finite()) {
            return Err(LayoutError::NonAffineMatrix);
        }
        AffineRows::from_column_major(&elements)?;
        Ok(())
    } else if bytes.len() < AFFINE_ROWS_BYTES {
        Err(LayoutError::BufferTooSmall {
            required: AFFINE_ROWS_BYTES,
            provided: bytes.len(),
        })
    } else if bytes.len() < 64 {
        Err(LayoutError::BufferTooSmall {
            required: 64,
            provided: bytes.len(),
        })
    } else {
        Err(LayoutError::IncompatibleTargetLayout {
            target_size: 64,
            source_size: bytes.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only mutex serializing tests that mutate, reset, advance, or assert `GLOBAL_RESOURCE_SLOT_TABLE`.
    /// Tolerates lock poisoning across test failures (§6.5, peer review 7202).
    static TEST_SLOT_TABLE_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn negotiation_succeeds_when_profile_supported() {
        let req = GpuRequiredProfile {
            required_features: vec![String::from("depth-clip-control")],
            min_limits: DeviceLimitsRecord::default(),
        };
        let adapter_features = vec![String::from("depth-clip-control"), String::from("indirect-first-instance")];
        let adapter_limits = DeviceLimitsRecord::default();

        let result = req.negotiate(
            &adapter_features,
            &adapter_limits,
            "Apple",
            "Apple M5",
            false,
            "bgra8unorm",
        );
        assert!(result.is_ok());
        let prof = result.unwrap();
        assert_eq!(prof.adapter_vendor, "Apple");
        assert_eq!(prof.enabled_features, vec![String::from("depth-clip-control")]);
    }

    #[test]
    fn negotiation_fails_when_feature_missing() {
        let req = GpuRequiredProfile {
            required_features: vec![String::from("shader-f16")],
            min_limits: DeviceLimitsRecord::default(),
        };
        let adapter_features = vec![String::from("depth-clip-control")];
        let adapter_limits = DeviceLimitsRecord::default();

        let result = req.negotiate(
            &adapter_features,
            &adapter_limits,
            "Generic",
            "Software",
            true,
            "rgba8unorm",
        );
        match result {
            Err(NegotiationError::MissingRequiredFeature { feature }) => {
                assert_eq!(feature, "shader-f16");
            }
            _ => panic!("Expected MissingRequiredFeature error"),
        }
    }

    #[test]
    fn negotiation_fails_when_limit_insufficient() {
        let mut req = GpuRequiredProfile::default();
        req.min_limits.max_buffer_size = 536870912; // 512 MiB requested

        let adapter_features = Vec::new();
        let mut adapter_limits = DeviceLimitsRecord::default();
        adapter_limits.max_buffer_size = 268435456; // only 256 MiB available

        let result = req.negotiate(
            &adapter_features,
            &adapter_limits,
            "Vendor",
            "Device",
            false,
            "bgra8unorm",
        );
        match result {
            Err(NegotiationError::InsufficientLimit { limit_name, requested, available }) => {
                assert_eq!(limit_name, "maxBufferSize");
                assert_eq!(requested, 536870912);
                assert_eq!(available, 268435456);
            }
            _ => panic!("Expected InsufficientLimit error"),
        }
    }

    #[test]
    fn negotiation_fails_when_storage_alignment_insufficient() {
        let mut req = GpuRequiredProfile::default();
        req.min_limits.min_storage_buffer_offset_alignment = 256;

        let adapter_features = Vec::new();
        let mut adapter_limits = DeviceLimitsRecord::default();
        adapter_limits.min_storage_buffer_offset_alignment = 512; // 512 > 256 fails alignment requirement

        let result = req.negotiate(
            &adapter_features,
            &adapter_limits,
            "Vendor",
            "Device",
            false,
            "rgba8unorm",
        );
        match result {
            Err(NegotiationError::InsufficientLimit { limit_name, requested, available }) => {
                assert_eq!(limit_name, "minStorageBufferOffsetAlignment");
                assert_eq!(requested, 256);
                assert_eq!(available, 512);
            }
            _ => panic!("Expected InsufficientLimit for minStorageBufferOffsetAlignment"),
        }
    }

    #[test]
    fn error_scope_serialization_tracker() {
        let mut tracker = ErrorScopeTracker::new();
        assert!(!tracker.is_active());

        // Operation 1 opens scopes
        assert!(tracker.begin_scope(1).is_ok());
        assert!(tracker.begin_scope(1).is_ok());
        assert!(tracker.is_active());

        // Operation 2 attempts to interleave -> fails
        let conflict = tracker.begin_scope(2);
        assert!(conflict.is_err(), "Concurrent task must not interleave error scopes");

        // Operation 1 pops scopes
        assert_eq!(tracker.end_scope(1), Ok(false));
        assert_eq!(tracker.end_scope(1), Ok(true));
        assert!(!tracker.is_active());

        // Now Operation 2 can acquire
        assert!(tracker.begin_scope(2).is_ok());
        assert_eq!(tracker.end_scope(2), Ok(true));
    }

    #[test]
    fn triangle_submission_packet_encoded_valid() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let packet = build_triangle_submission();
        let encoded = packet.encode().expect("valid packet encoding");
        assert_eq!(&encoded[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes([encoded[4], encoded[5]]), PACKET_VERSION);
    }

    #[test]
    fn red_a_blue_b_submission_uses_per_use_byte_buffer() {
        let valid_packet = build_red_a_blue_b_submission(true);
        let encoded_valid = valid_packet.encode().expect("valid packet encoding");
        assert!(!encoded_valid.is_empty());

        let invalid_packet = build_red_a_blue_b_submission(false);
        let encoded_invalid = invalid_packet.encode().expect("valid packet encoding");
        assert!(!encoded_invalid.is_empty());

        // In the valid packet, the second pass dynamic offset is 256.
        // In the invalid packet, the offset is 0.
        assert_ne!(encoded_valid, encoded_invalid);
    }

    #[test]
    fn packet_encode_fails_when_payload_exceeds_bounds() {
        let mut packet = GpuSubmissionPacket::new();
        packet.push(GpuCommand::WriteBuffer {
            buffer_id: 1,
            offset: 0,
            data: vec![0u8; 100],
        });
        // Bounded encoding with max_payload_len = 50 fails with DataPayloadOverflow
        let res = packet.encode_bounded(50);
        match res {
            Err(PacketEncodeError::DataPayloadOverflow { offset, length }) => {
                assert_eq!(offset, 0);
                assert_eq!(length, 100);
            }
            _ => panic!("Expected DataPayloadOverflow error, got {res:?}"),
        }
    }

    #[test]
    fn lower_plan_red_a_blue_b_produces_identical_commands() {
        use f3d_graph::ResourceId;

        let shared_buf = ResourceId::new(1);
        let target_a = ResourceId::new(30);
        let target_b = ResourceId::new(31);
        let readback_a = ResourceId::new(40);
        let readback_b = ResourceId::new(41);
        let pipeline_id = 200;

        let plan = f3d_graph::build_red_a_blue_b_plan(
            shared_buf,
            target_a,
            target_b,
            readback_a,
            readback_b,
            DataVersion::new(1),
            DataVersion::new(2),
            0,
            256,
            pipeline_id,
        )
        .expect("build_red_a_blue_b_plan should compile");

        let lowered = lower_plan(&plan).expect("lower plan");

        let submission = build_red_a_blue_b_submission(true);
        let expected_execution_cmds: Vec<GpuCommand> = submission
            .commands()
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    GpuCommand::RenderPass { .. } | GpuCommand::CopyTextureToBuffer { .. }
                )
            })
            .cloned()
            .collect();

        assert_eq!(lowered, expected_execution_cmds);
        assert_eq!(lowered.len(), 4);
    }

    #[test]
    fn lower_plan_precompiled_red_blue_plan_lowers_renders() {
        use f3d_graph::ResourceId;

        let plan = f3d_graph::build_red_a_blue_b_render_plan(
            ResourceId::new(1),
            ResourceId::new(30),
            ResourceId::new(31),
            DataVersion::new(1),
            DataVersion::new(2),
            0,
            256,
            200,
        )
        .expect("build_red_a_blue_b_render_plan");

        let lowered = lower_plan(&plan).expect("lower plan");
        assert_eq!(lowered.len(), 2);
        match &lowered[0] {
            GpuCommand::RenderPass {
                target_id,
                pipeline_id,
                uniform_dynamic_offset,
                uniform_buffer_id,
                clear_color,
                ..
            } => {
                assert_eq!(*target_id, 30);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*uniform_buffer_id, 1);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
            }
            _ => panic!("Expected RenderPass command"),
        }
        match &lowered[1] {
            GpuCommand::RenderPass {
                target_id,
                pipeline_id,
                uniform_dynamic_offset,
                uniform_buffer_id,
                clear_color,
                ..
            } => {
                assert_eq!(*target_id, 31);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*uniform_dynamic_offset, 256);
                assert_eq!(*uniform_buffer_id, 1);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
            }
            _ => panic!("Expected RenderPass command"),
        }
    }

    #[test]
    fn try_publish_readback_discards_stale_epoch() {
        use f3d_core::{
            handle::{Handle, MaterialDomain},
            ownership::Author,
        };

        let handle = Handle::<MaterialDomain>::from_raw(10, 1).expect("valid handle");
        let mut region = RegionState::new(handle);
        assert_eq!(region.current_epoch(), Epoch::ZERO);

        let mut pub_state = ReadbackPublicationState::new();

        // 1. Publishing with matching Epoch::ZERO succeeds
        let data_epoch_0 = vec![1, 2, 3, 4];
        let ok = pub_state.try_publish_readback(&region, Epoch::ZERO, data_epoch_0.clone());
        assert!(ok);
        assert_eq!(pub_state.data(), Some(data_epoch_0.as_slice()));
        assert_eq!(pub_state.published_epoch(), Some(Epoch::ZERO));

        // 2. Region epoch advances while async mapAsync was in flight
        let next_epoch = region.publish(Author::Js, Epoch::ZERO).expect("advance epoch");
        assert_eq!(region.current_epoch(), next_epoch);
        assert_ne!(next_epoch, Epoch::ZERO);

        // 3. Publishing stale readback with older Epoch::ZERO is rejected and discarded
        let stale_data = vec![9, 9, 9, 9];
        let rejected = pub_state.try_publish_readback(&region, Epoch::ZERO, stale_data);
        assert!(!rejected);
        // Stale data was not published; prior data remains
        assert_eq!(pub_state.data(), Some(data_epoch_0.as_slice()));

        // 4. Publishing with updated fresh epoch succeeds
        let fresh_data = vec![5, 6, 7, 8];
        let ok_fresh = pub_state.try_publish_readback(&region, next_epoch, fresh_data.clone());
        assert!(ok_fresh);
        assert_eq!(pub_state.data(), Some(fresh_data.as_slice()));
        assert_eq!(pub_state.published_epoch(), Some(next_epoch));

        // 5. Freestanding helper also rejects stale epoch
        assert!(!try_publish_readback(&region, Epoch::ZERO));
        assert!(try_publish_readback(&region, next_epoch));
    }

    #[test]
    fn lower_plan_rejects_render_pass_without_color_attachment() {
        use f3d_graph::{Pass, PassId};

        let p = Pass::new_render(PassId::new(1), "empty_pass");
        let seg = f3d_graph::PlanSegment::from_pass(&p);
        let plan = ExecutionPlan {
            segments: vec![seg],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        match lower_plan(&plan) {
            Err(PlanLoweringError::MissingColorAttachment { segment_name }) => {
                assert_eq!(segment_name, "empty_pass");
            }
            _ => panic!("Expected MissingColorAttachment error"),
        }
    }

    #[test]
    fn lower_plan_rejects_render_pass_without_draws() {
        use f3d_graph::{ColorAttachment, Pass, PassId, ResourceId};

        let mut p = Pass::new_render(PassId::new(1), "no_draws_pass");
        p.color_attachments.push(ColorAttachment::new_load(ResourceId::new(10)));
        let seg = f3d_graph::PlanSegment::from_pass(&p);
        let plan = ExecutionPlan {
            segments: vec![seg],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        match lower_plan(&plan) {
            Err(PlanLoweringError::MissingDrawCommand { segment_name }) => {
                assert_eq!(segment_name, "no_draws_pass");
            }
            _ => panic!("Expected MissingDrawCommand error"),
        }
    }

    #[test]
    fn lower_plan_permits_zero_draw_pass_with_clear() {
        use f3d_graph::{ColorAttachment, Pass, PassId, ResourceId};

        let mut p = Pass::new_render(PassId::new(1), "empty_clear_pass");
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.2, 0.4, 0.6, 1.0],
        ));
        let seg = f3d_graph::PlanSegment::from_pass(&p);
        let plan = ExecutionPlan {
            segments: vec![seg],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("empty clear pass must be permitted");
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                clear_color,
                vertex_count,
                load_op,
                store_op,
                pass_flags,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*clear_color, [0.2, 0.4, 0.6, 1.0]);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
            }
            other => panic!("expected RenderPass clear opener, got {other:?}"),
        }
    }

    #[test]
    fn create_texture_usage_includes_copy_src_and_excludes_texture_binding() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        // 1. Verify GpuCommand structured commands for triangle target 10
        let triangle_packet = build_triangle_submission();
        let triangle_create_tex = triangle_packet
            .commands()
            .iter()
            .find_map(|cmd| match cmd {
                GpuCommand::CreateTexture {
                    texture_id, usage, ..
                } if *texture_id == 10 => Some(*usage),
                _ => None,
            })
            .expect("triangle submission must contain CreateTexture for target 10");

        assert_eq!(
            triangle_create_tex & TEXTURE_USAGE_COPY_SRC,
            TEXTURE_USAGE_COPY_SRC,
            "Target 10 command must include COPY_SRC (bit 1)"
        );
        assert_eq!(
            triangle_create_tex & TEXTURE_USAGE_RENDER_ATTACHMENT,
            TEXTURE_USAGE_RENDER_ATTACHMENT,
            "Target 10 command must include RENDER_ATTACHMENT (bit 16)"
        );
        assert_eq!(
            triangle_create_tex & TEXTURE_USAGE_TEXTURE_BINDING,
            0,
            "Target 10 command must not include TEXTURE_BINDING (bit 4)"
        );
        assert_eq!(triangle_create_tex, 17);

        // 2. Verify encoded binary packet for triangle target 10
        let encoded_triangle = triangle_packet
            .encode()
            .expect("triangle packet encoding must succeed");
        let mut cursor = 16;
        let mut found_triangle_tex_usage = None;
        for _ in 0..triangle_packet.commands().len() {
            let opcode =
                u16::from_le_bytes([encoded_triangle[cursor], encoded_triangle[cursor + 1]]);
            cursor += 2;
            match opcode {
                OPCODE_CREATE_BUFFER => cursor += 12,
                OPCODE_WRITE_BUFFER => cursor += 16,
                OPCODE_CREATE_TEXTURE => {
                    let tex_id = u32::from_le_bytes(
                        encoded_triangle[cursor..cursor + 4]
                            .try_into()
                            .expect("tex_id slice"),
                    );
                    let usage = u32::from_le_bytes(
                        encoded_triangle[cursor + 16..cursor + 20]
                            .try_into()
                            .expect("usage slice"),
                    );
                    if tex_id == 10 {
                        found_triangle_tex_usage = Some(usage);
                    }
                    cursor += 20;
                }
                OPCODE_CREATE_PIPELINE => cursor += 32,
                OPCODE_RENDER_PASS => cursor += 44,
                OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                _ => panic!("unknown opcode {opcode}"),
            }
        }
        let encoded_usage_10 =
            found_triangle_tex_usage.expect("encoded CreateTexture for target 10 must exist");
        assert_eq!(
            encoded_usage_10 & TEXTURE_USAGE_COPY_SRC,
            TEXTURE_USAGE_COPY_SRC,
            "Encoded target 10 must include COPY_SRC (bit 1)"
        );
        assert_eq!(
            encoded_usage_10 & TEXTURE_USAGE_RENDER_ATTACHMENT,
            TEXTURE_USAGE_RENDER_ATTACHMENT,
            "Encoded target 10 must include RENDER_ATTACHMENT (bit 16)"
        );
        assert_eq!(
            encoded_usage_10 & TEXTURE_USAGE_TEXTURE_BINDING,
            0,
            "Encoded target 10 must not include TEXTURE_BINDING (bit 4)"
        );
        assert_eq!(encoded_usage_10, 17);

        // 3. Verify Red-A / Blue-B targets 30 and 31 (both command and encoded bytes)
        let red_blue_packet = build_red_a_blue_b_submission(true);
        for target_id in [30, 31] {
            let usage = red_blue_packet
                .commands()
                .iter()
                .find_map(|cmd| match cmd {
                    GpuCommand::CreateTexture {
                        texture_id, usage, ..
                    } if *texture_id == target_id => Some(*usage),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    panic!("red_blue submission must contain CreateTexture for target {target_id}")
                });

            assert_eq!(
                usage & TEXTURE_USAGE_COPY_SRC,
                TEXTURE_USAGE_COPY_SRC,
                "Target {target_id} command must include COPY_SRC (bit 1)"
            );
            assert_eq!(
                usage & TEXTURE_USAGE_RENDER_ATTACHMENT,
                TEXTURE_USAGE_RENDER_ATTACHMENT,
                "Target {target_id} command must include RENDER_ATTACHMENT (bit 16)"
            );
            assert_eq!(
                usage & TEXTURE_USAGE_TEXTURE_BINDING,
                0,
                "Target {target_id} command must not include TEXTURE_BINDING (bit 4)"
            );
            assert_eq!(usage, 17);
        }

        let encoded_red_blue = red_blue_packet
            .encode()
            .expect("red-blue packet encoding must succeed");
        let mut cursor = 16;
        let mut found_targets: Vec<(u32, u32)> = Vec::new();
        for _ in 0..red_blue_packet.commands().len() {
            let opcode =
                u16::from_le_bytes([encoded_red_blue[cursor], encoded_red_blue[cursor + 1]]);
            cursor += 2;
            match opcode {
                OPCODE_CREATE_BUFFER => cursor += 12,
                OPCODE_WRITE_BUFFER => cursor += 16,
                OPCODE_CREATE_TEXTURE => {
                    let tex_id = u32::from_le_bytes(
                        encoded_red_blue[cursor..cursor + 4]
                            .try_into()
                            .expect("tex_id slice"),
                    );
                    let usage = u32::from_le_bytes(
                        encoded_red_blue[cursor + 16..cursor + 20]
                            .try_into()
                            .expect("usage slice"),
                    );
                    found_targets.push((tex_id, usage));
                    cursor += 20;
                }
                OPCODE_CREATE_PIPELINE => cursor += 32,
                OPCODE_RENDER_PASS => cursor += 44,
                OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                _ => panic!("unknown opcode {opcode}"),
            }
        }
        for target_id in [30, 31] {
            let (_, enc_usage) = found_targets
                .iter()
                .find(|(id, _)| *id == target_id)
                .unwrap_or_else(|| panic!("missing encoded target {target_id}"));
            assert_eq!(
                enc_usage & TEXTURE_USAGE_COPY_SRC,
                TEXTURE_USAGE_COPY_SRC,
                "Encoded target {target_id} must include COPY_SRC (bit 1)"
            );
            assert_eq!(
                enc_usage & TEXTURE_USAGE_RENDER_ATTACHMENT,
                TEXTURE_USAGE_RENDER_ATTACHMENT,
                "Encoded target {target_id} must include RENDER_ATTACHMENT (bit 16)"
            );
            assert_eq!(
                enc_usage & TEXTURE_USAGE_TEXTURE_BINDING,
                0,
                "Encoded target {target_id} must not include TEXTURE_BINDING (bit 4)"
            );
            assert_eq!(*enc_usage, 17);
        }
    }

    #[test]
    fn try_publish_readback_word_round_trip() {
        use f3d_core::{
            handle::{Handle, MaterialDomain},
            ownership::Author,
        };

        // 1. Epoch::ZERO words (0, 0)
        let (zero_hi, zero_lo) = Epoch::ZERO.to_words();
        assert_eq!(zero_hi, 0);
        assert_eq!(zero_lo, 0);
        assert!(gpu_bridge_try_publish_readback(zero_hi, zero_lo, zero_hi, zero_lo));
        assert!(try_publish_readback_words(0, 0, 0, 0));

        // 2. Arbitrary multi-word epoch reconstruction and round-trip
        let epoch_a = Epoch::new(0x1234_5678_9ABC_DEF0);
        let (a_hi, a_lo) = epoch_a.to_words();
        assert_eq!(a_hi, 0x1234_5678);
        assert_eq!(a_lo, 0x9ABC_DEF0);
        let reconstructed_a = Epoch::from_words(a_hi, a_lo);
        assert_eq!(epoch_a, reconstructed_a);

        // Matching words -> accepted
        assert!(gpu_bridge_try_publish_readback(a_hi, a_lo, a_hi, a_lo));
        assert!(try_publish_readback_words(a_hi, a_lo, a_hi, a_lo));

        // High word mismatch -> rejected
        assert!(!gpu_bridge_try_publish_readback(a_hi, a_lo, a_hi + 1, a_lo));
        assert!(!gpu_bridge_try_publish_readback(a_hi + 1, a_lo, a_hi, a_lo));

        // Low word mismatch -> rejected
        assert!(!gpu_bridge_try_publish_readback(a_hi, a_lo, a_hi, a_lo + 1));
        assert!(!gpu_bridge_try_publish_readback(a_hi, a_lo + 1, a_hi, a_lo));

        // 3. Integration with RegionState publication progression
        let handle = Handle::<MaterialDomain>::from_raw(1, 1).expect("valid handle");
        let mut region = RegionState::new(handle);
        let (reg_hi0, reg_lo0) = region.current_epoch().to_words();
        assert!(gpu_bridge_try_publish_readback(reg_hi0, reg_lo0, reg_hi0, reg_lo0));

        // Advance epoch via region.publish
        let next_epoch = region.publish(Author::Js, Epoch::ZERO).expect("advance epoch");
        let (reg_hi1, reg_lo1) = next_epoch.to_words();
        assert_ne!((reg_hi0, reg_lo0), (reg_hi1, reg_lo1));

        // Stale readback (reg_hi0, reg_lo0) against advanced region (reg_hi1, reg_lo1) -> rejected
        assert!(!gpu_bridge_try_publish_readback(reg_hi0, reg_lo0, reg_hi1, reg_lo1));

        // Fresh readback (reg_hi1, reg_lo1) against advanced region (reg_hi1, reg_lo1) -> accepted
        assert!(gpu_bridge_try_publish_readback(reg_hi1, reg_lo1, reg_hi1, reg_lo1));
    }

    #[test]
    fn encode_bundle_commands_and_packet_builder() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut packet = GpuSubmissionPacket::new();

        // 1. RecordBundle command
        packet.push(GpuCommand::RecordBundle {
            bundle_id: 42,
            pipeline_id: 100,
            vertex_buffer_id: 2,
            vertex_count: 3,
            uniform_dynamic_offset: 256,
            uniform_buffer_id: 1,
            target_format: TARGET_FORMAT_RGBA8UNORM,
        });

        // 2. ExecuteBundles with 2 bundle IDs
        packet.push(GpuCommand::ExecuteBundles {
            bundle_ids: vec![42, 43],
        });

        // 3. ExecuteBundles with empty list (spec-mandated state reset)
        packet.push(GpuCommand::ExecuteBundles {
            bundle_ids: Vec::new(),
        });

        let encoded = packet.encode().expect("bundle commands packet encoding");
        // Header: 16 bytes
        // RecordBundle: 2 (opcode) + 28 = 30 bytes
        // ExecuteBundles (2): 2 (opcode) + 4 (count) + 8 = 14 bytes
        // ExecuteBundles (0): 2 (opcode) + 4 (count) + 0 = 6 bytes
        // Total: 16 + 30 + 14 + 6 = 66 bytes
        assert_eq!(encoded.len(), 66);

        // Verify header
        assert_eq!(&encoded[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes([encoded[4], encoded[5]]), PACKET_VERSION);
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 3); // 3 commands
        assert_eq!(u32::from_le_bytes(encoded[12..16].try_into().unwrap()), 0); // 0 data payload

        // Verify Command 0: RecordBundle (opcode 7)
        let mut cursor = 16;
        let op0 = u16::from_le_bytes([encoded[cursor], encoded[cursor + 1]]);
        assert_eq!(op0, OPCODE_RECORD_BUNDLE);
        cursor += 2;
        let b_id = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        let p_id = u32::from_le_bytes(encoded[cursor + 4..cursor + 8].try_into().unwrap());
        let vb_id = u32::from_le_bytes(encoded[cursor + 8..cursor + 12].try_into().unwrap());
        let v_count = u32::from_le_bytes(encoded[cursor + 12..cursor + 16].try_into().unwrap());
        let dyn_off = u32::from_le_bytes(encoded[cursor + 16..cursor + 20].try_into().unwrap());
        let ub_id = u32::from_le_bytes(encoded[cursor + 20..cursor + 24].try_into().unwrap());
        let fmt = u32::from_le_bytes(encoded[cursor + 24..cursor + 28].try_into().unwrap());
        assert_eq!(b_id, 42);
        assert_eq!(p_id, 100);
        assert_eq!(vb_id, 2);
        assert_eq!(v_count, 3);
        assert_eq!(dyn_off, 256);
        assert_eq!(ub_id, 1);
        assert_eq!(fmt, TARGET_FORMAT_RGBA8UNORM);
        cursor += 28;

        // Verify Command 1: ExecuteBundles with 2 IDs (opcode 8)
        let op1 = u16::from_le_bytes([encoded[cursor], encoded[cursor + 1]]);
        assert_eq!(op1, OPCODE_EXECUTE_BUNDLES);
        cursor += 2;
        let count1 = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        assert_eq!(count1, 2);
        cursor += 4;
        let id1_0 = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        let id1_1 = u32::from_le_bytes(encoded[cursor + 4..cursor + 8].try_into().unwrap());
        assert_eq!(id1_0, 42);
        assert_eq!(id1_1, 43);
        cursor += 8;

        // Verify Command 2: ExecuteBundles with 0 IDs (opcode 8)
        let op2 = u16::from_le_bytes([encoded[cursor], encoded[cursor + 1]]);
        assert_eq!(op2, OPCODE_EXECUTE_BUNDLES);
        cursor += 2;
        let count2 = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        assert_eq!(count2, 0);
        cursor += 4;
        assert_eq!(cursor, 66);

        // Verify positive builder and native export
        let positive_packet = build_bundle_then_direct_draw_submission();
        assert_eq!(positive_packet.commands.len(), 15);

        // Command 9: RecordBundle (opcode 7)
        match &positive_packet.commands[9] {
            GpuCommand::RecordBundle {
                bundle_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                target_format,
                ..
            } => {
                assert_eq!(*bundle_id, 1);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*target_format, TARGET_FORMAT_RGBA8UNORM);
            }
            other => panic!("expected RecordBundle at command 9, got {other:?}"),
        }

        // Command 10: RenderPass opener (vertex_count: 0) MUST precede ExecuteBundles
        match &positive_packet.commands[10] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                vertex_count,
                clear_color,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected RenderPass opener at command 10, got {other:?}"),
        }

        // Command 11: ExecuteBundles ([1])
        match &positive_packet.commands[11] {
            GpuCommand::ExecuteBundles { bundle_ids } => {
                assert_eq!(bundle_ids, &vec![1]);
            }
            other => panic!("expected ExecuteBundles at command 11, got {other:?}"),
        }

        // Command 12: ExecuteBundles ([])
        match &positive_packet.commands[12] {
            GpuCommand::ExecuteBundles { bundle_ids } => {
                assert!(bundle_ids.is_empty());
            }
            other => panic!("expected ExecuteBundles at command 12, got {other:?}"),
        }

        // Command 13: RenderPass direct draw (vertex_count: 3)
        match &positive_packet.commands[13] {
            GpuCommand::RenderPass {
                target_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                ..
            } => {
                assert_eq!(*target_id, 10);
                assert_eq!(*vertex_buffer_id, 3);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 256);
            }
            other => panic!("expected RenderPass direct draw at command 13, got {other:?}"),
        }

        // Command 14: CopyTextureToBuffer
        match &positive_packet.commands[14] {
            GpuCommand::CopyTextureToBuffer {
                texture_id,
                buffer_id,
                ..
            } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*buffer_id, 20);
            }
            other => panic!("expected CopyTextureToBuffer at command 14, got {other:?}"),
        }

        let positive_bytes = positive_packet.encode().expect("encode bundle positive packet");
        assert!(!positive_bytes.is_empty());
        let exported_bytes = gpu_bridge_build_bundle_direct_draw_packet();
        assert_eq!(positive_bytes, exported_bytes);
    }

    #[test]
    fn lower_plan_bundle_then_direct_draw() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            resource::ResourceId,
            schedule::PassGraph,
        };

        let target_tex = ResourceId::new(10);
        let mut pass = Pass::new_render(PassId::new(1), "bundle_pass");
        pass.color_attachments
            .push(ColorAttachment::new_clear(target_tex, [0.0, 0.0, 0.0, 1.0]));

        // Draw 0: Render bundle execution
        let bundle_draw = Draw::new_bundle(0, 42, 100, vec![]);
        pass.draws.push(bundle_draw);

        // Draw 1: Direct draw explicitly requiring rebind after bundle
        let direct_draw = Draw::new(1, 100, 3, 0, vec![]).with_rebind_required(true);
        pass.draws.push(direct_draw);

        let mut graph = PassGraph::new();
        graph.add_pass(pass).expect("add pass");
        let plan = graph.compile(None).expect("compile plan with bundle then direct draw");

        let commands = lower_plan(&plan).expect("lower plan with bundle then direct draw");
        // Single-pass command shape: pass opener (vertex_count: 0) -> execute bundles -> direct draw
        assert_eq!(commands.len(), 3);

        match &commands[0] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                vertex_count,
                clear_color,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected RenderPass opener as first command, got {other:?}"),
        }

        match &commands[1] {
            GpuCommand::ExecuteBundles { bundle_ids } => {
                assert_eq!(bundle_ids, &vec![42]);
            }
            other => panic!("expected ExecuteBundles as second command, got {other:?}"),
        }

        match &commands[2] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                pipeline_id,
                vertex_count,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 100);
                assert_eq!(*vertex_count, 3);
            }
            other => panic!("expected RenderPass as third command, got {other:?}"),
        }

        // Packet encoding of lowered commands succeeds
        let packet = GpuSubmissionPacket::from_commands(commands);
        let encoded = packet.encode().expect("packet encode must succeed");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn bundle_then_direct_draw_single_pass_command_shape() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let packet = build_bundle_then_direct_draw_submission();
        let commands = packet.commands();

        // Must contain setup resources + pass opener + bundle draw + empty bundle + direct draw + copy
        assert_eq!(commands.len(), 15);

        // Command 10: Render pass opener with vertex_count: 0 and clear color [0, 0, 0, 1]
        match &commands[10] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                clear_color,
                vertex_count,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
                assert_eq!(*vertex_count, 0);
            }
            other => panic!("expected RenderPass opener at index 10, got {other:?}"),
        }

        // Command 11: Execute Bundle 1
        match &commands[11] {
            GpuCommand::ExecuteBundles { bundle_ids } => {
                assert_eq!(bundle_ids, &vec![1]);
            }
            other => panic!("expected ExecuteBundles([1]) at index 11, got {other:?}"),
        }

        // Command 12: Execute empty bundle sequence (WebGPU spec state reset)
        match &commands[12] {
            GpuCommand::ExecuteBundles { bundle_ids } => {
                assert!(bundle_ids.is_empty());
            }
            other => panic!("expected ExecuteBundles([]) at index 12, got {other:?}"),
        }

        // Command 13: Direct draw inside the same pass with explicit rebind (offset 256)
        match &commands[13] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                uniform_buffer_id,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 3);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 256);
                assert_eq!(*uniform_buffer_id, 1);
            }
            other => panic!("expected RenderPass direct draw at index 13, got {other:?}"),
        }

        // Command 14: CopyTextureToBuffer
        match &commands[14] {
            GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, .. } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*buffer_id, 20);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer at index 14, got {other:?}"),
        }

        // Encode and verify non-empty binary bytes
        let encoded = packet.encode().expect("encode must succeed");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn generational_handle_slot_table_fresh_stale_and_zero_generation() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        use f3d_core::error::HandleError;

        // Clean global table state before running test
        with_global_resource_table(|table| *table = ResourceSlotTable::new());

        // 1. Building submission packets registers their resources in the generational slot table:
        // - Triangle: buffers 1, 2, 3; texture 10; pipeline 100
        // - Bundle: buffers 1, 2, 3, 20; texture 10; pipeline 200
        // - Affine transform: buffers 1, 2, 20; texture 10; pipeline 200
        // - Nested pass: buffers 1, 2, 3, 20; textures 10, 11; pipeline 200
        // - Nested canvas pass: buffers 1, 2, 3, 20; textures 10, 11; pipelines 200, 201
        // - Nested viewport/scissor pass: buffers 1, 2, 3, 20; textures 10, 11; pipeline 200
        let _triangle_packet = build_triangle_submission();
        let _bundle_packet = build_bundle_then_direct_draw_submission();
        let _affine_packet = build_affine_rows_transform_submission();
        let _nested_packet = build_nested_pass_submission();
        let _nested_canvas_packet = build_nested_canvas_pass_submission();
        let _viewport_packet = build_nested_viewport_scissor_submission();

        // Verify that binary exports also trigger slot table registration identically
        let _affine_bytes = gpu_bridge_build_affine_rows_transform_packet();
        let _nested_bytes = gpu_bridge_build_nested_pass_packet();
        let _canvas_bytes = gpu_bridge_build_nested_canvas_pass_packet();
        let _viewport_bytes = gpu_bridge_build_nested_viewport_scissor_packet();

        // 2. Fresh generation: all registered resource IDs answer truthfully at generation 1
        // Buffers: 1 (uniform), 2 (vertex 1), 3 (vertex 2 / readback staging), 20 (readback target 10), 21 (readback target 11)
        for buffer_id in [1, 2, 3, 20, 21] {
            assert_eq!(check_resource_handle(buffer_id, 1), Ok(true));
            assert!(gpu_bridge_check_resource_handle(buffer_id, 1));
            assert!(f3d_check_resource_handle(buffer_id, 1));
        }

        // Textures: 10 (canvas / target A), 11 (offscreen nested target)
        for texture_id in [10, 11] {
            assert_eq!(check_resource_handle(texture_id, 1), Ok(true));
            assert!(gpu_bridge_check_resource_handle(texture_id, 1));
            assert!(f3d_check_resource_handle(texture_id, 1));
        }

        // Pipelines: 100 (triangle), 200 (direct / offscreen), 201 (canvas presentation)
        for pipeline_id in [100, 200, 201] {
            assert_eq!(check_resource_handle(pipeline_id, 1), Ok(true));
            assert!(gpu_bridge_check_resource_handle(pipeline_id, 1));
            assert!(f3d_check_resource_handle(pipeline_id, 1));
        }

        // 3. Zero generation rejection: generation 0 must be rejected via HandleError::InvalidGeneration
        for id in [1, 2, 3, 10, 11, 20, 21, 100, 200, 201] {
            assert_eq!(
                check_resource_handle(id, 0),
                Err(HandleError::InvalidGeneration { raw_generation: 0 })
            );
            assert!(!gpu_bridge_check_resource_handle(id, 0));
            assert!(!f3d_check_resource_handle(id, 0));
        }

        // 4. Stale after reuse: advance generation of slot 10 (simulating slot release and reuse)
        let new_gen = advance_resource_slot_generation(10).expect("advance generation");
        assert_eq!(new_gen, 2);

        // Old generation 1 is now stale -> returns false (ABA prevented!)
        assert_eq!(check_resource_handle(10, 1), Ok(false));
        assert!(!gpu_bridge_check_resource_handle(10, 1));
        assert!(!f3d_check_resource_handle(10, 1));

        // New generation 2 is fresh -> returns true
        assert_eq!(check_resource_handle(10, 2), Ok(true));
        assert!(gpu_bridge_check_resource_handle(10, 2));
        assert!(f3d_check_resource_handle(10, 2));

        // Also test advance on nested offscreen texture 11 and canvas presentation pipeline 201
        let new_gen_11 = advance_resource_slot_generation(11).expect("advance generation 11");
        assert_eq!(new_gen_11, 2);
        assert_eq!(check_resource_handle(11, 1), Ok(false));
        assert!(!gpu_bridge_check_resource_handle(11, 1));
        assert_eq!(check_resource_handle(11, 2), Ok(true));
        assert!(gpu_bridge_check_resource_handle(11, 2));

        let new_gen_201 = advance_resource_slot_generation(201).expect("advance generation 201");
        assert_eq!(new_gen_201, 2);
        assert_eq!(check_resource_handle(201, 1), Ok(false));
        assert!(!gpu_bridge_check_resource_handle(201, 1));
        assert_eq!(check_resource_handle(201, 2), Ok(true));
        assert!(gpu_bridge_check_resource_handle(201, 2));

        // Re-check zero generation still rejected on advanced slots
        assert_eq!(
            check_resource_handle(10, 0),
            Err(HandleError::InvalidGeneration { raw_generation: 0 })
        );
        assert_eq!(
            check_resource_handle(11, 0),
            Err(HandleError::InvalidGeneration { raw_generation: 0 })
        );
        assert_eq!(
            check_resource_handle(201, 0),
            Err(HandleError::InvalidGeneration { raw_generation: 0 })
        );

        // 5. Isolated table unit tests (verifying ResourceSlotTable methods directly)
        let mut table = ResourceSlotTable::new();
        let h42 = table.register(42);
        assert_eq!(h42.index(), 42);
        assert_eq!(h42.generation().get(), 1);

        // Fresh check on isolated table
        assert_eq!(table.check_handle(42, 1), Ok(true));

        // Zero generation check
        assert_eq!(
            table.check_handle(42, 0),
            Err(HandleError::InvalidGeneration { raw_generation: 0 })
        );

        // Release and advance
        let gen2 = table.release_and_advance(42).expect("release and advance");
        assert_eq!(gen2, 2);

        // Double release yields SlotVacant error (§6.5, peer review 6524)
        assert_eq!(
            table.release_and_advance(42),
            Err(HandleError::SlotVacant { index: 42 })
        );

        // Inactive slot returns false
        assert_eq!(table.check_handle(42, 1), Ok(false));
        assert_eq!(table.check_handle(42, 2), Ok(false));

        // Reallocate slot 42 (slot reused after release)
        let h42_reused = table.reallocate(42).expect("reallocate");
        assert_eq!(h42_reused.generation().get(), 2);

        // Stale handle (gen 1) returns false
        assert_eq!(table.check_handle(42, 1), Ok(false));

        // Current handle (gen 2) returns true
        assert_eq!(table.check_handle(42, 2), Ok(true));

        // Out of bounds index returns false
        assert_eq!(table.check_handle(999, 1), Ok(false));

        // Out of bounds release_and_advance yields IndexOutOfBounds with exact capacity (peer review 6524)
        assert_eq!(
            table.release_and_advance(999),
            Err(HandleError::IndexOutOfBounds { index: 999, capacity: 43 })
        );

        // Reset global table to clean state upon test completion
        with_global_resource_table(|table| *table = ResourceSlotTable::new());
    }

    #[test]
    fn linear_memory_borrow_guards_growth_blocked_and_token_validation() {
        // Reset global scope to clean initial state
        with_global_borrow_scope(|scope| *scope = BorrowScope::new());

        // Sentinel token 0 exit rejected (§6.6, peer review 6585)
        assert!(!gpu_bridge_borrow_exit(0));
        assert!(!f3d_borrow_exit(0));

        // pages == 0 acts as a size query / probe that succeeds when idle without advancing generation (§6.6, peer review 6585)
        let gen0 = gpu_bridge_borrow_growth_generation();
        assert!(gpu_bridge_try_grow_memory(0));
        assert_eq!(gpu_bridge_borrow_growth_generation(), gen0);
        assert!(f3d_try_grow_memory(0));
        assert_eq!(f3d_borrow_growth_generation(), gen0);

        // Initially idle: real growth succeeds and advances growth_generation
        assert!(gpu_bridge_try_grow_memory(1));
        assert_eq!(gpu_bridge_borrow_growth_generation(), gen0 + 1);
        assert!(f3d_try_grow_memory(2));
        assert_eq!(f3d_borrow_growth_generation(), gen0 + 2);

        // Enter borrow scope: returns non-zero token
        let token1 = gpu_bridge_borrow_enter();
        assert_ne!(token1, 0);

        // While borrowed: pages == 0 query returns false and does not advance generation
        let gen_borrowed = gpu_bridge_borrow_growth_generation();
        assert!(!gpu_bridge_try_grow_memory(0));
        assert_eq!(gpu_bridge_borrow_growth_generation(), gen_borrowed);
        assert!(!f3d_try_grow_memory(0));
        assert_eq!(f3d_borrow_growth_generation(), gen_borrowed);

        // While borrowed: memory growth must be blocked (returns false)
        assert!(!gpu_bridge_try_grow_memory(1));
        assert!(!f3d_try_grow_memory(1));
        assert_eq!(gpu_bridge_borrow_growth_generation(), gen_borrowed);

        // Re-entrant enter while borrowed returns 0
        assert_eq!(gpu_bridge_borrow_enter(), 0);

        // Mismatched token exit returns false and leaves scope borrowed
        assert!(!gpu_bridge_borrow_exit(token1 + 999));
        assert!(!f3d_borrow_exit(token1 + 999));
        assert!(!gpu_bridge_try_grow_memory(1));

        // Clean exit with matching token succeeds
        assert!(gpu_bridge_borrow_exit(token1));

        // Double exit after already idle returns false
        assert!(!gpu_bridge_borrow_exit(token1));

        // Now idle again: memory growth and zero-page query succeed
        assert!(gpu_bridge_try_grow_memory(0));
        assert!(gpu_bridge_try_grow_memory(1));
        assert_eq!(gpu_bridge_borrow_growth_generation(), gen_borrowed + 1);

        // Re-enter with f3d alias
        let token2 = f3d_borrow_enter();
        assert_ne!(token2, 0);
        assert_ne!(token2, token1); // Monotonic tokens
        assert!(!f3d_try_grow_memory(0));
        assert!(!f3d_try_grow_memory(1));
        assert!(f3d_borrow_exit(token2));
        assert!(f3d_try_grow_memory(1));

        // Poisoned lock recovery test (§6.6, peer review 6585)
        let current_gen = gpu_bridge_borrow_growth_generation();
        let panic_result = std::panic::catch_unwind(|| {
            with_global_borrow_scope(|_scope| {
                panic!("simulated panic to test poisoned mutex recovery in with_global_borrow_scope");
            });
        });
        assert!(panic_result.is_err());
        // Lock recovery must allow subsequent access without panicking
        assert_eq!(gpu_bridge_borrow_growth_generation(), current_gen);
        assert!(gpu_bridge_try_grow_memory(0));
    }

    #[test]
    fn affine_rows_layout_validation_rejects_perspective_terms() {
        // 1. Valid 64-byte identity matrix: passes (0)
        let mut valid_64 = vec![0u8; 64];
        for &idx in &[0, 5, 10, 15] {
            valid_64[idx * 4..(idx + 1) * 4].copy_from_slice(&1.0f32.to_le_bytes());
        }
        assert_eq!(gpu_bridge_validate_affine_rows(valid_64.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(f3d_validate_affine_rows(valid_64.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(validate_affine_rows(&valid_64), Ok(()));

        // 2. Valid 64-byte matrix with non-zero translation (tx=12.5, ty=-4.0, tz=100.0 at e[12..15]): passes (0)
        let mut valid_64_trans = valid_64.clone();
        valid_64_trans[12 * 4..13 * 4].copy_from_slice(&12.5f32.to_le_bytes());
        valid_64_trans[13 * 4..14 * 4].copy_from_slice(&(-4.0f32).to_le_bytes());
        valid_64_trans[14 * 4..15 * 4].copy_from_slice(&100.0f32.to_le_bytes());
        assert_eq!(gpu_bridge_validate_affine_rows(valid_64_trans.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(f3d_validate_affine_rows(valid_64_trans.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(validate_affine_rows(&valid_64_trans), Ok(()));

        // 3. Non-affine 64-byte matrix with perspective term at e[3]: rejected with code 2 (NonAffineMatrix)
        let mut non_affine_64 = valid_64.clone();
        non_affine_64[3 * 4..4 * 4].copy_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(
            gpu_bridge_validate_affine_rows(non_affine_64.clone()),
            LAYOUT_VALIDATION_NON_AFFINE_MATRIX
        );
        assert_eq!(
            f3d_validate_affine_rows(non_affine_64.clone()),
            LAYOUT_VALIDATION_NON_AFFINE_MATRIX
        );
        assert_eq!(
            validate_affine_rows(&non_affine_64),
            Err(LayoutError::NonAffineMatrix)
        );

        // 4. Non-affine 64-byte matrix with perspective term at e[11]: rejected with code 2 (NonAffineMatrix)
        let mut non_affine_64_e11 = valid_64.clone();
        non_affine_64_e11[11 * 4..12 * 4].copy_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(
            gpu_bridge_validate_affine_rows(non_affine_64_e11.clone()),
            LAYOUT_VALIDATION_NON_AFFINE_MATRIX
        );
        assert_eq!(
            validate_affine_rows(&non_affine_64_e11),
            Err(LayoutError::NonAffineMatrix)
        );

        // 5. Non-affine 64-byte matrix with invalid homogeneous scale at e[15] (must be 1.0): rejected
        let mut bad_scale_64 = valid_64.clone();
        bad_scale_64[15 * 4..16 * 4].copy_from_slice(&2.0f32.to_le_bytes());
        assert_eq!(
            gpu_bridge_validate_affine_rows(bad_scale_64),
            LAYOUT_VALIDATION_NON_AFFINE_MATRIX
        );

        // 6. Non-affine 64-byte matrix with NaN element: rejected with code 2 (NonAffineMatrix)
        let mut nan_64 = valid_64.clone();
        nan_64[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            gpu_bridge_validate_affine_rows(nan_64.clone()),
            LAYOUT_VALIDATION_NON_AFFINE_MATRIX
        );
        assert_eq!(
            validate_affine_rows(&nan_64),
            Err(LayoutError::NonAffineMatrix)
        );

        // 7. Valid 48-byte AffineRows wire record (identity rows r0, r1, r2): passes (0)
        let mut valid_48 = vec![0u8; 48];
        for &idx in &[0, 5, 10] {
            valid_48[idx * 4..(idx + 1) * 4].copy_from_slice(&1.0f32.to_le_bytes());
        }
        assert_eq!(gpu_bridge_validate_affine_rows(valid_48.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(f3d_validate_affine_rows(valid_48.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(validate_affine_rows(&valid_48), Ok(()));

        // 8. Valid 48-byte AffineRows wire record with non-zero translation (tx=12.5 at float 3, ty=-4.0 at float 7, tz=100.0 at float 11): MUST pass (0)
        let mut valid_48_trans = valid_48.clone();
        valid_48_trans[3 * 4..4 * 4].copy_from_slice(&12.5f32.to_le_bytes());
        valid_48_trans[7 * 4..8 * 4].copy_from_slice(&(-4.0f32).to_le_bytes());
        valid_48_trans[11 * 4..12 * 4].copy_from_slice(&100.0f32.to_le_bytes());
        assert_eq!(gpu_bridge_validate_affine_rows(valid_48_trans.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(f3d_validate_affine_rows(valid_48_trans.clone()), LAYOUT_VALIDATION_OK);
        assert_eq!(validate_affine_rows(&valid_48_trans), Ok(()));

        // 9. Non-affine 48-byte AffineRows wire record with NaN translation element: rejected with code 2 (NonAffineMatrix)
        let mut nan_48 = valid_48_trans.clone();
        nan_48[3 * 4..4 * 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            gpu_bridge_validate_affine_rows(nan_48.clone()),
            LAYOUT_VALIDATION_NON_AFFINE_MATRIX
        );
        assert_eq!(
            validate_affine_rows(&nan_48),
            Err(LayoutError::NonAffineMatrix)
        );

        // 10. Buffer too small (< 48 bytes): rejected with code 1 (BufferTooSmall)
        let too_small = vec![0u8; 32];
        assert_eq!(
            gpu_bridge_validate_affine_rows(too_small.clone()),
            LAYOUT_VALIDATION_BUFFER_TOO_SMALL
        );
        assert_eq!(
            validate_affine_rows(&too_small),
            Err(LayoutError::BufferTooSmall { required: 48, provided: 32 })
        );

        // 11. Intermediate invalid length (48 < len < 64): rejected with code 1 (BufferTooSmall for 64)
        let mid_len = vec![0u8; 56];
        assert_eq!(
            gpu_bridge_validate_affine_rows(mid_len.clone()),
            LAYOUT_VALIDATION_BUFFER_TOO_SMALL
        );
        assert_eq!(
            validate_affine_rows(&mid_len),
            Err(LayoutError::BufferTooSmall { required: 64, provided: 56 })
        );

        // 12. Incompatible oversized buffer (> 64 bytes): rejected with code 4 (IncompatibleTargetLayout)
        let oversized = vec![0u8; 72];
        assert_eq!(
            gpu_bridge_validate_affine_rows(oversized.clone()),
            LAYOUT_VALIDATION_INCOMPATIBLE_TARGET
        );
        assert_eq!(
            validate_affine_rows(&oversized),
            Err(LayoutError::IncompatibleTargetLayout { target_size: 64, source_size: 72 })
        );
    }

    #[test]
    fn affine_rows_transform_packet_and_pixel_coordinates() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let packet = build_affine_rows_transform_submission();
        assert_eq!(packet.commands.len(), 9);

        // Command 0: Create uniform buffer 1 (size 256)
        match &packet.commands[0] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 1);
                assert_eq!(*size, 256);
                assert_eq!(*usage, BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer for uniform, got {other:?}"),
        }

        // Command 1: Write uniform buffer 1 (offset 0, 256 bytes, first 48 bytes are AffineRows)
        match &packet.commands[1] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 1);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 256);
                // Decode first 48 bytes as AffineRows
                let mut arr = [0u8; 48];
                arr.copy_from_slice(&data[..48]);
                let rows = AffineRows::from_bytes(&arr);
                // r0: [sx, 0, 0, tx] = [0.5, 0.0, 0.0, 0.5]
                assert_eq!(rows.r0, [0.5, 0.0, 0.0, 0.5]);
                // r1: [0, sy, 0, ty] = [0.0, 0.5, 0.0, 0.0]
                assert_eq!(rows.r1, [0.0, 0.5, 0.0, 0.0]);
                // r2: [0, 0, sz, tz] = [0.0, 0.0, 1.0, 0.0]
                assert_eq!(rows.r2, [0.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected WriteBuffer for uniform, got {other:?}"),
        }

        // Command 2: Create vertex buffer 2
        match &packet.commands[2] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 2);
                assert_eq!(*size, 60); // 3 * 20 bytes
                assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer for vertex, got {other:?}"),
        }

        // Command 3: Write vertex buffer 2
        match &packet.commands[3] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 2);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 60);
            }
            other => panic!("expected WriteBuffer for vertex, got {other:?}"),
        }

        // Command 4: Create target texture 10 (64x64 RGBA8Unorm)
        match &packet.commands[4] {
            GpuCommand::CreateTexture { texture_id, width, height, format, .. } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(*format, TARGET_FORMAT_RGBA8UNORM);
            }
            other => panic!("expected CreateTexture, got {other:?}"),
        }

        // Command 5: Create readback buffer 20 (size 16384)
        match &packet.commands[5] {
            GpuCommand::CreateBuffer { buffer_id, size, .. } => {
                assert_eq!(*buffer_id, 20);
                assert_eq!(*size, 16384);
            }
            other => panic!("expected CreateBuffer for readback, got {other:?}"),
        }

        // Command 6: Create pipeline 200 with uniform_size 48 and vertex_stride 20
        match &packet.commands[6] {
            GpuCommand::CreatePipeline { pipeline_id, uniform_size, vertex_stride, wgsl_code, .. } => {
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*uniform_size, 48);
                assert_eq!(*vertex_stride, 20);
                assert!(wgsl_code.contains("transform_affine_point"));
                assert!(wgsl_code.contains("dot(m.r0, v)"));
                assert!(wgsl_code.contains("dot(m.r1, v)"));
                assert!(wgsl_code.contains("dot(m.r2, v)"));
            }
            other => panic!("expected CreatePipeline, got {other:?}"),
        }

        // Command 7: RenderPass drawing 3 vertices
        match &packet.commands[7] {
            GpuCommand::RenderPass { target_id, pipeline_id, vertex_buffer_id, vertex_count, uniform_buffer_id, uniform_dynamic_offset, clear_color, .. } => {
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_buffer_id, 1);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected RenderPass, got {other:?}"),
        }

        // Command 8: CopyTextureToBuffer
        match &packet.commands[8] {
            GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, .. } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*buffer_id, 20);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer, got {other:?}"),
        }

        // Binary packet encoding assertions
        let wasm_bytes = gpu_bridge_build_affine_rows_transform_packet();
        let alias_bytes = f3d_build_affine_rows_transform_packet();
        assert_eq!(wasm_bytes, alias_bytes);
        assert!(!wasm_bytes.is_empty());
        assert_eq!(&wasm_bytes[0..4], &PACKET_MAGIC); // Packet header magic

        // Verify hand-computed screen pixel coordinates (64x64 framebuffer):
        // Transform: P' = (0.5 * P.x + 0.5, 0.5 * P.y, P.z)
        // Untransformed vertices:
        //   v0 = (0.0, 0.5, 0.0)
        //   v1 = (-0.5, -0.5, 0.0)
        //   v2 = (0.5, -0.5, 0.0)
        // Transformed vertices in NDC [-1, 1]:
        //   v0' = (0.5 * 0.0 + 0.5, 0.5 * 0.5) = (0.5, 0.25)
        //   v1' = (0.5 * -0.5 + 0.5, 0.5 * -0.5) = (0.25, -0.25)
        //   v2' = (0.5 * 0.5 + 0.5, 0.5 * -0.5) = (0.75, -0.25)
        // Screen pixel mapping on 64x64:
        //   px = round((x_ndc + 1.0) * 0.5 * 64 - 0.5)
        //   py = round((1.0 - y_ndc) * 0.5 * 64 - 0.5)
        let ndc_to_pixel = |x: f32, y: f32| -> (u32, u32) {
            let px = ((x + 1.0) * 0.5 * 64.0 - 0.5).round() as u32;
            let py = ((1.0 - y) * 0.5 * 64.0 - 0.5).round() as u32;
            (px, py)
        };
        assert_eq!(ndc_to_pixel(0.5, 0.25), (48, 24)); // Top vertex
        assert_eq!(ndc_to_pixel(0.25, -0.25), (40, 40)); // Bottom-left vertex
        assert_eq!(ndc_to_pixel(0.75, -0.25), (56, 40)); // Bottom-right vertex
        assert_eq!(ndc_to_pixel(0.5, 0.0), (48, 32)); // Center of transformed triangle (interior)
        assert_eq!(ndc_to_pixel(0.0, 0.0), (32, 32)); // Untransformed center (origin, outside transformed triangle)
    }

    #[test]
    fn test_pack_unpack_target_type_backwards_compatibility() {
        // 1. Raw 0 (Offscreen) and raw 1 (Canvas) must unpack with Clear, Store, and No-Flag
        assert_eq!(unpack_target_kind(TARGET_OFFSCREEN), TARGET_OFFSCREEN);
        assert_eq!(unpack_load_op(TARGET_OFFSCREEN), LOAD_OP_CLEAR);
        assert_eq!(unpack_store_op(TARGET_OFFSCREEN), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(TARGET_OFFSCREEN), PASS_FLAG_NONE);

        assert_eq!(unpack_target_kind(TARGET_CANVAS), TARGET_CANVAS);
        assert_eq!(unpack_load_op(TARGET_CANVAS), LOAD_OP_CLEAR);
        assert_eq!(unpack_store_op(TARGET_CANVAS), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(TARGET_CANVAS), PASS_FLAG_NONE);

        // 2. Packing default values produces identical raw integer values
        assert_eq!(
            pack_target_type(TARGET_OFFSCREEN, LOAD_OP_CLEAR, STORE_OP_STORE, PASS_FLAG_NONE),
            0
        );
        assert_eq!(
            pack_target_type(TARGET_CANVAS, LOAD_OP_CLEAR, STORE_OP_STORE, PASS_FLAG_NONE),
            1
        );

        // 3. Packing explicit LoadOp::Load, StoreOp::Store, and PASS_FLAG_NEW_PASS round-trips exactly
        let packed = pack_target_type(TARGET_OFFSCREEN, LOAD_OP_LOAD, STORE_OP_STORE, PASS_FLAG_NEW_PASS);
        assert_eq!(unpack_target_kind(packed), TARGET_OFFSCREEN);
        assert_eq!(unpack_load_op(packed), LOAD_OP_LOAD);
        assert_eq!(unpack_store_op(packed), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(packed), PASS_FLAG_NEW_PASS);
        assert_ne!(packed, 0);

        // 4. Packing DontCare and Discard flags round-trips
        let packed_discard = pack_target_type(TARGET_CANVAS, LOAD_OP_DONT_CARE, STORE_OP_DISCARD, PASS_FLAG_NONE);
        assert_eq!(unpack_target_kind(packed_discard), TARGET_CANVAS);
        assert_eq!(unpack_load_op(packed_discard), LOAD_OP_DONT_CARE);
        assert_eq!(unpack_store_op(packed_discard), STORE_OP_DISCARD);
        assert_eq!(unpack_pass_flags(packed_discard), PASS_FLAG_NONE);

        // 5. Positive and negative validation checks for validate_target_type (§6.7, 2v8.4)
        assert!(validate_target_type(TARGET_OFFSCREEN));
        assert!(validate_target_type(TARGET_CANVAS));
        assert!(validate_target_type(packed));
        assert!(validate_target_type(packed_discard));

        assert!(!validate_target_type(2)); // target_kind > 1
        assert!(!validate_target_type(0x0300)); // load_op > 2
        assert!(!validate_target_type(0x020000)); // store_op > 1
        assert!(!validate_target_type(0x02000000)); // pass_flags has bit 1 set (reserved)
        assert!(!validate_target_type(0xFF000000)); // unknown upper flag bits
    }

    #[test]
    fn test_lower_plan_multi_segment_load_store_pass_boundaries() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        // Pass 0: Prefix pass on Target 10 with Clear
        let mut pass0 = Pass::new_render(PassId::new(1), "outer_prefix");
        pass0.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.1, 0.2, 0.3, 1.0],
        ));
        pass0.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));

        // Pass 1: Nested pass on Target 20 with Clear
        let mut pass1 = Pass::new_render(PassId::new(2), "nested_shadow");
        pass1.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(20),
            [0.4, 0.5, 0.6, 1.0],
        ));
        pass1.draws.push(Draw::new(2, 200, 6, 0, Vec::new()));

        // Pass 2: Resumed pass on Target 10 with LoadOp::Load, containing 2 draws
        let mut pass2 = Pass::new_render(PassId::new(3), "outer_resumed");
        pass2.color_attachments.push(ColorAttachment::new_load(ResourceId::new(10)));
        pass2.draws.push(Draw::new(3, 100, 3, 0, Vec::new()));
        pass2.draws.push(Draw::new(4, 100, 3, 256, Vec::new()));

        let plan = ExecutionPlan {
            segments: vec![
                PlanSegment::from_pass(&pass0),
                PlanSegment::from_pass(&pass1),
                PlanSegment::from_pass(&pass2),
            ],
            canvas_epoch: None,
            pass_count: 3,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("lower multi-segment plan");
        assert_eq!(commands.len(), 4);

        // Command 0 (Pass 0 Draw 1): Target 10, Clear, New Pass
        match &commands[0] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                clear_color,
                pipeline_id,
                vertex_count,
                load_op,
                store_op,
                pass_flags,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*clear_color, [0.1, 0.2, 0.3, 1.0]);
                assert_eq!(*pipeline_id, 100);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
            }
            other => panic!("expected RenderPass at command 0, got {other:?}"),
        }

        // Command 1 (Pass 1 Draw 1): Target 20, Clear, New Pass
        match &commands[1] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                clear_color,
                pipeline_id,
                vertex_count,
                load_op,
                store_op,
                pass_flags,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 20);
                assert_eq!(*clear_color, [0.4, 0.5, 0.6, 1.0]);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_count, 6);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
            }
            other => panic!("expected RenderPass at command 1, got {other:?}"),
        }

        // Command 2 (Pass 2 Draw 1): Target 10, LOAD, New Pass
        match &commands[2] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                pipeline_id,
                vertex_count,
                uniform_dynamic_offset,
                load_op,
                store_op,
                pass_flags,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 100);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*load_op, LOAD_OP_LOAD);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
            }
            other => panic!("expected RenderPass at command 2, got {other:?}"),
        }

        // Command 3 (Pass 2 Draw 2): Target 10, LOAD, SAME Pass (pass_flags == NONE)
        match &commands[3] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                pipeline_id,
                vertex_count,
                uniform_dynamic_offset,
                load_op,
                store_op,
                pass_flags,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 100);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 256);
                assert_eq!(*load_op, LOAD_OP_LOAD);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NONE);
            }
            other => panic!("expected RenderPass at command 3, got {other:?}"),
        }

        // Binary wire encoding verification: ensure 44-byte records with correctly packed target_type
        let packet = GpuSubmissionPacket::from_commands(commands);
        let encoded = packet.encode().expect("encode packet");
        assert_eq!(encoded.len(), 16 + 4 * (2 + 44));

        let check_cmd = |offset: usize| -> (u32, u32) {
            let opcode = u16::from_le_bytes([encoded[offset], encoded[offset + 1]]);
            assert_eq!(opcode, OPCODE_RENDER_PASS);
            let raw_target_type = u32::from_le_bytes(encoded[offset + 2..offset + 6].try_into().unwrap());
            let target_id = u32::from_le_bytes(encoded[offset + 6..offset + 10].try_into().unwrap());
            (raw_target_type, target_id)
        };

        // Command 0 at offset 16
        let (raw0, tid0) = check_cmd(16);
        assert_eq!(tid0, 10);
        assert_eq!(unpack_target_kind(raw0), TARGET_OFFSCREEN);
        assert_eq!(unpack_load_op(raw0), LOAD_OP_CLEAR);
        assert_eq!(unpack_store_op(raw0), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(raw0), PASS_FLAG_NEW_PASS);

        // Command 1 at offset 16 + 46 = 62
        let (raw1, tid1) = check_cmd(62);
        assert_eq!(tid1, 20);
        assert_eq!(unpack_target_kind(raw1), TARGET_OFFSCREEN);
        assert_eq!(unpack_load_op(raw1), LOAD_OP_CLEAR);
        assert_eq!(unpack_store_op(raw1), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(raw1), PASS_FLAG_NEW_PASS);

        // Command 2 at offset 62 + 46 = 108
        let (raw2, tid2) = check_cmd(108);
        assert_eq!(tid2, 10);
        assert_eq!(unpack_target_kind(raw2), TARGET_OFFSCREEN);
        assert_eq!(unpack_load_op(raw2), LOAD_OP_LOAD);
        assert_eq!(unpack_store_op(raw2), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(raw2), PASS_FLAG_NEW_PASS);

        // Command 3 at offset 108 + 46 = 154
        let (raw3, tid3) = check_cmd(154);
        assert_eq!(tid3, 10);
        assert_eq!(unpack_target_kind(raw3), TARGET_OFFSCREEN);
        assert_eq!(unpack_load_op(raw3), LOAD_OP_LOAD);
        assert_eq!(unpack_store_op(raw3), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(raw3), PASS_FLAG_NONE);
    }

    #[test]
    fn test_lower_plan_rejects_multiple_color_attachments() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut p = Pass::new_render(PassId::new(1), "mrt_pass");
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(11),
            [1.0, 0.0, 0.0, 1.0],
        ));
        p.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        match lower_plan(&plan) {
            Err(PlanLoweringError::UnsupportedMultipleColorAttachments {
                segment_name,
                count,
            }) => {
                assert_eq!(segment_name, "mrt_pass");
                assert_eq!(count, 2);
            }
            other => panic!("Expected UnsupportedMultipleColorAttachments error, got {other:?}"),
        }
    }

    #[test]
    fn test_lower_plan_rejects_depth_stencil_attachment() {
        use f3d_graph::{
            pass::{ColorAttachment, DepthStencilAttachment, Draw, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut p = Pass::new_render(PassId::new(1), "depth_pass");
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));
        p.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(99),
            1.0,
        ));
        p.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        match lower_plan(&plan) {
            Err(PlanLoweringError::UnsupportedDepthStencilAttachment { segment_name }) => {
                assert_eq!(segment_name, "depth_pass");
            }
            other => panic!("Expected UnsupportedDepthStencilAttachment error, got {other:?}"),
        }
    }

    #[test]
    fn test_nested_pass_packet_and_pixel_coordinates() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let wasm_bytes = gpu_bridge_build_nested_pass_packet();
        let alias_bytes = f3d_build_nested_pass_packet();
        assert_eq!(wasm_bytes, alias_bytes);
        assert!(!wasm_bytes.is_empty());

        // Verify header
        assert_eq!(&wasm_bytes[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes(wasm_bytes[4..6].try_into().unwrap()), PACKET_VERSION);
        let cmd_count = u32::from_le_bytes(wasm_bytes[8..12].try_into().unwrap());

        // Verify structured commands
        let submission = build_nested_pass_submission();
        let commands = submission.commands();
        assert_eq!(cmd_count, commands.len() as u32);

        // Verify resource creations
        match &commands[0] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 1);
                assert_eq!(*size, 768);
                assert_eq!(*usage, BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer at 0, got {other:?}"),
        }

        // Verify 3 distinct RenderPass draws matching the three passes (Root 7734)
        let render_draws: Vec<&GpuCommand> = commands
            .iter()
            .filter(|cmd| matches!(cmd, GpuCommand::RenderPass { vertex_count, .. } if *vertex_count == 3))
            .collect();
        assert_eq!(render_draws.len(), 3);

        // Pass 1: Target 10, Clear to Black, Draw Red left triangle (dynamic offset 0, vb 2)
        match render_draws[0] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                clear_color,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                uniform_buffer_id,
                load_op,
                store_op,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*uniform_buffer_id, 1);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
            }
            other => panic!("expected RenderPass draw 0, got {other:?}"),
        }

        // Pass 2: Target 11, Nested pass, Clear to Black, Draw Green triangle (dynamic offset 512, vb 2)
        match render_draws[1] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                clear_color,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                uniform_buffer_id,
                load_op,
                store_op,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 11);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 512);
                assert_eq!(*uniform_buffer_id, 1);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
            }
            other => panic!("expected RenderPass draw 1, got {other:?}"),
        }

        // Pass 3: Target 10, Outer resume with LoadOp::Load, Draw Blue right triangle (dynamic offset 256, vb 3)
        match render_draws[2] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                uniform_buffer_id,
                load_op,
                store_op,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 3);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 256);
                assert_eq!(*uniform_buffer_id, 1);
                assert_eq!(*load_op, LOAD_OP_LOAD);
                assert_eq!(*store_op, STORE_OP_STORE);
            }
            other => panic!("expected RenderPass draw 2, got {other:?}"),
        }

        // Verify CopyTextureToBuffer of Target 10 to buffer 20
        let copy_cmd = commands
            .iter()
            .find(|cmd| matches!(cmd, GpuCommand::CopyTextureToBuffer { .. }))
            .expect("expected CopyTextureToBuffer command");
        match copy_cmd {
            GpuCommand::CopyTextureToBuffer {
                texture_id,
                buffer_id,
                width,
                height,
                ..
            } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*buffer_id, 20);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer, got {other:?}"),
        }

        // Verify strict relative source order: draw0 < draw1 < draw2 < copy
        let draw0_idx = commands.iter().position(|cmd| std::ptr::eq(cmd, render_draws[0])).unwrap();
        let draw1_idx = commands.iter().position(|cmd| std::ptr::eq(cmd, render_draws[1])).unwrap();
        let draw2_idx = commands.iter().position(|cmd| std::ptr::eq(cmd, render_draws[2])).unwrap();
        let copy_idx = commands.iter().position(|cmd| std::ptr::eq(cmd, copy_cmd)).unwrap();
        assert!(draw0_idx < draw1_idx);
        assert!(draw1_idx < draw2_idx);
        assert!(draw2_idx < copy_idx);

        // Verify binary wire format: scan commands dynamically from offset 16 per Root 7734
        let total_data_len = u32::from_le_bytes(wasm_bytes[12..16].try_into().unwrap());
        let mut cursor = 16usize;
        let mut render_pass_index = 0;
        for _ in 0..cmd_count {
            let opcode = u16::from_le_bytes(wasm_bytes[cursor..cursor + 2].try_into().unwrap());
            cursor += 2;
            match opcode {
                OPCODE_CREATE_BUFFER => cursor += 12,
                OPCODE_WRITE_BUFFER => cursor += 16,
                OPCODE_CREATE_TEXTURE => cursor += 20,
                OPCODE_CREATE_PIPELINE => cursor += 32,
                OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                OPCODE_SET_VIEWPORT => cursor += 24,
                OPCODE_SET_SCISSOR_RECT => cursor += 16,
                OPCODE_SET_DRAW_PARAMETERS => cursor += 12,
                OPCODE_EXECUTE_BUNDLES => {
                    let bundle_count = u32::from_le_bytes(wasm_bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                    cursor += 4 + bundle_count * 4;
                }
                OPCODE_RENDER_PASS => {
                    // Exactly 44-byte record
                    let packed_target = u32::from_le_bytes(wasm_bytes[cursor..cursor + 4].try_into().unwrap());
                    let target_id = u32::from_le_bytes(wasm_bytes[cursor + 4..cursor + 8].try_into().unwrap());
                    let pipeline_id = u32::from_le_bytes(wasm_bytes[cursor + 24..cursor + 28].try_into().unwrap());
                    let vertex_buffer_id = u32::from_le_bytes(wasm_bytes[cursor + 28..cursor + 32].try_into().unwrap());
                    let vertex_count = u32::from_le_bytes(wasm_bytes[cursor + 32..cursor + 36].try_into().unwrap());
                    let dynamic_offset = u32::from_le_bytes(wasm_bytes[cursor + 36..cursor + 40].try_into().unwrap());
                    let uniform_buffer_id = u32::from_le_bytes(wasm_bytes[cursor + 40..cursor + 44].try_into().unwrap());

                    assert!(validate_target_type(packed_target));

                    // When this record represents a draw with vertices, verify its target and uniforms
                    if vertex_count > 0 {
                        assert_eq!(pipeline_id, 200);
                        assert_eq!(vertex_count, 3);
                        assert_eq!(uniform_buffer_id, 1);

                        match render_pass_index {
                            0 => {
                                // Pass 1: Target 10, Clear, Store, dynamic offset 0, vb 2
                                assert_eq!(target_id, 10);
                                assert_eq!(unpack_target_kind(packed_target), TARGET_OFFSCREEN);
                                assert_eq!(unpack_load_op(packed_target), LOAD_OP_CLEAR);
                                assert_eq!(unpack_store_op(packed_target), STORE_OP_STORE);
                                assert_eq!(dynamic_offset, 0);
                                assert_eq!(vertex_buffer_id, 2);
                            }
                            1 => {
                                // Pass 2: Target 11, Clear, Store, dynamic offset 512, vb 2
                                assert_eq!(target_id, 11);
                                assert_eq!(unpack_target_kind(packed_target), TARGET_OFFSCREEN);
                                assert_eq!(unpack_load_op(packed_target), LOAD_OP_CLEAR);
                                assert_eq!(unpack_store_op(packed_target), STORE_OP_STORE);
                                assert_eq!(dynamic_offset, 512);
                                assert_eq!(vertex_buffer_id, 2);
                            }
                            2 => {
                                // Pass 3: Target 10, LOAD, Store, dynamic offset 256, vb 3
                                assert_eq!(target_id, 10);
                                assert_eq!(unpack_target_kind(packed_target), TARGET_OFFSCREEN);
                                assert_eq!(unpack_load_op(packed_target), LOAD_OP_LOAD);
                                assert_eq!(unpack_store_op(packed_target), STORE_OP_STORE);
                                assert_eq!(dynamic_offset, 256);
                                assert_eq!(vertex_buffer_id, 3);
                            }
                            _ => panic!("unexpected render pass index {render_pass_index}"),
                        }
                        render_pass_index += 1;
                    }
                    cursor += 44;
                }
                other => panic!("unexpected opcode {other} at cursor {cursor}"),
            }
        }
        assert_eq!(render_pass_index, 3);
        assert_eq!(cursor, wasm_bytes.len() - total_data_len as usize);

        // Mathematical verification of pixel sample points on 64x64 offscreen target
        // Viewport mapping: pixel center (px + 0.5, py + 0.5) to NDC (x_ndc, y_ndc)
        let pixel_to_ndc = |px: u32, py: u32| -> [f32; 2] {
            let x = ((px as f32 + 0.5) / 64.0) * 2.0 - 1.0;
            let y = 1.0 - ((py as f32 + 0.5) / 64.0) * 2.0;
            [x, y]
        };

        let ndc_to_pixel = |x: f32, y: f32| -> (u32, u32) {
            let px = ((x + 1.0) * 0.5 * 64.0 - 0.5).round() as u32;
            let py = ((1.0 - y) * 0.5 * 64.0 - 0.5).round() as u32;
            (px, py)
        };

        // Cross-product point-in-triangle test (counter-clockwise orientation)
        let point_in_tri = |p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> bool {
            let cross = |p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]| -> f32 {
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            };
            let d1 = cross(a, b, p);
            let d2 = cross(b, c, p);
            let d3 = cross(c, a, p);
            let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
            let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
            !(has_neg && has_pos)
        };

        // Tri 1: (-1, -1) -> (0, -1) -> (0, 1) (covers x in [-1, 0], y in [-1, 2x+1])
        let tri1_a = [-1.0f32, -1.0f32];
        let tri1_b = [0.0f32, -1.0f32];
        let tri1_c = [0.0f32, 1.0f32];

        // Tri 2: (0, -1) -> (1, -1) -> (1, 1) (covers x in [0, 1], y in [-1, 2x-1])
        let tri2_a = [0.0f32, -1.0f32];
        let tri2_b = [1.0f32, -1.0f32];
        let tri2_c = [1.0f32, 1.0f32];

        // 1. Pixel (24, 32): inside left triangle (tri1) -> Red [255, 0, 0, 255]
        let p24_32 = pixel_to_ndc(24, 32);
        assert_eq!(ndc_to_pixel(p24_32[0], p24_32[1]), (24, 32));
        assert!(point_in_tri(p24_32, tri1_a, tri1_b, tri1_c));
        assert!(!point_in_tri(p24_32, tri2_a, tri2_b, tri2_c));

        // 2. Pixel (56, 32): inside right triangle (tri2) -> Blue [0, 0, 255, 255]
        let p56_32 = pixel_to_ndc(56, 32);
        assert_eq!(ndc_to_pixel(p56_32[0], p56_32[1]), (56, 32));
        assert!(point_in_tri(p56_32, tri2_a, tri2_b, tri2_c));
        assert!(!point_in_tri(p56_32, tri1_a, tri1_b, tri1_c));

        // 3. Pixel (2, 2): outside both triangles -> Black [0, 0, 0, 255]
        let p2_2 = pixel_to_ndc(2, 2);
        assert_eq!(ndc_to_pixel(p2_2[0], p2_2[1]), (2, 2));
        assert!(!point_in_tri(p2_2, tri1_a, tri1_b, tri1_c));
        assert!(!point_in_tri(p2_2, tri2_a, tri2_b, tri2_c));
    }

    #[test]
    fn test_nested_canvas_pass_packet_and_epoch_attachment() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        // 1. Verify that canonical alias matches primary export byte-for-byte
        let wasm_bytes = gpu_bridge_build_nested_canvas_pass_packet();
        let canonical_bytes = f3d_build_nested_canvas_pass_packet();
        assert_eq!(wasm_bytes, canonical_bytes);
        assert!(wasm_bytes.len() >= 16);

        // 2. Verify command structure in GpuSubmissionPacket
        let packet = build_nested_canvas_pass_submission();
        assert_eq!(packet.commands().len(), 14);

        // Header verification: Magic F3DP, version 1, command count matching packet (Root 7734)
        assert_eq!(&wasm_bytes[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes(wasm_bytes[4..6].try_into().unwrap()), PACKET_VERSION);
        assert_eq!(u16::from_le_bytes(wasm_bytes[6..8].try_into().unwrap()), 0); // flags
        let cmd_count = u32::from_le_bytes(wasm_bytes[8..12].try_into().unwrap());
        assert_eq!(cmd_count, packet.commands().len() as u32);

        // Generational slot table verification:
        // All canvas pass resources (buffers 1, 2, 3, 20, textures 10, 11, pipelines 200, 201) are active at generation 1
        for &id in &[1, 2, 3, 10, 11, 20, 200, 201] {
            assert_eq!(check_resource_handle(id, 1), Ok(true));
            assert!(gpu_bridge_check_resource_handle(id, 1));
            assert!(f3d_check_resource_handle(id, 1));
        }

        // Commands 0..4: Buffers
        match &packet.commands()[0] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 1);
                assert_eq!(*size, 768);
                assert_eq!(*usage, BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer at 0, got {other:?}"),
        }
        match &packet.commands()[1] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 2);
                assert_eq!(*size, 60);
                assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer at 1, got {other:?}"),
        }
        match &packet.commands()[2] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 2);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 60);
            }
            other => panic!("expected WriteBuffer at 2, got {other:?}"),
        }
        match &packet.commands()[3] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 3);
                assert_eq!(*size, 60);
                assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer at 3, got {other:?}"),
        }
        match &packet.commands()[4] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 3);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 60);
            }
            other => panic!("expected WriteBuffer at 4, got {other:?}"),
        }

        // Command 5: Offscreen texture Target 11 (Canvas Target 10 is NOT created as an offscreen texture)
        match &packet.commands()[5] {
            GpuCommand::CreateTexture { texture_id, width, height, format, usage } => {
                assert_eq!(*texture_id, 11);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(*format, TARGET_FORMAT_RGBA8UNORM);
                assert_eq!(*usage, TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC);
            }
            other => panic!("expected CreateTexture at 5, got {other:?}"),
        }

        // Command 6: Readback buffer 20 for Target 11
        match &packet.commands()[6] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 20);
                assert_eq!(*size, 16384);
                assert_eq!(*usage, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer at 6, got {other:?}"),
        }

        // Command 7: Pipeline 200 for offscreen target (RGBA8Unorm)
        match &packet.commands()[7] {
            GpuCommand::CreatePipeline { pipeline_id, target_format, .. } => {
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*target_format, TARGET_FORMAT_RGBA8UNORM);
            }
            other => panic!("expected CreatePipeline at 7, got {other:?}"),
        }

        // Command 8: Pipeline 201 for canvas target (PREFERRED_CANVAS)
        match &packet.commands()[8] {
            GpuCommand::CreatePipeline { pipeline_id, target_format, .. } => {
                assert_eq!(*pipeline_id, 201);
                assert_eq!(*target_format, TARGET_FORMAT_PREFERRED_CANVAS);
            }
            other => panic!("expected CreatePipeline at 8, got {other:?}"),
        }

        // Command 9: WriteBuffer (uniform arena from FrameSession)
        match &packet.commands()[9] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 1);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 528);
                // Verify uniform snapshot values:
                // Red [1.0, 0.0, 0.0, 1.0] at 0
                assert_eq!(f32::from_le_bytes(data[0..4].try_into().unwrap()), 1.0);
                assert_eq!(f32::from_le_bytes(data[4..8].try_into().unwrap()), 0.0);
                assert_eq!(f32::from_le_bytes(data[8..12].try_into().unwrap()), 0.0);
                assert_eq!(f32::from_le_bytes(data[12..16].try_into().unwrap()), 1.0);
                // Blue [0.0, 0.0, 1.0, 1.0] at 256
                assert_eq!(f32::from_le_bytes(data[256..260].try_into().unwrap()), 0.0);
                assert_eq!(f32::from_le_bytes(data[260..264].try_into().unwrap()), 0.0);
                assert_eq!(f32::from_le_bytes(data[264..268].try_into().unwrap()), 1.0);
                assert_eq!(f32::from_le_bytes(data[268..272].try_into().unwrap()), 1.0);
                // Green [0.0, 1.0, 0.0, 1.0] at 512
                assert_eq!(f32::from_le_bytes(data[512..516].try_into().unwrap()), 0.0);
                assert_eq!(f32::from_le_bytes(data[516..520].try_into().unwrap()), 1.0);
                assert_eq!(f32::from_le_bytes(data[520..524].try_into().unwrap()), 0.0);
                assert_eq!(f32::from_le_bytes(data[524..528].try_into().unwrap()), 1.0);
            }
            other => panic!("expected WriteBuffer at 9, got {other:?}"),
        }

        // Command 10: RenderPass Pass 1 (Canvas Target 10, Clear, NEW_PASS, Pipeline 201, vb1, offset 0)
        match &packet.commands()[10] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                load_op,
                store_op,
                pass_flags,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                uniform_buffer_id,
                ..
            } => {
                assert_eq!(*target_type, TARGET_CANVAS);
                assert_eq!(*target_id, 10);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*pipeline_id, 201);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*uniform_buffer_id, 1);
            }
            other => panic!("expected RenderPass at 10, got {other:?}"),
        }

        // Command 11: RenderPass Pass 2 (Offscreen Target 11, Clear, NEW_PASS, Pipeline 200, vb1, offset 512)
        match &packet.commands()[11] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                load_op,
                store_op,
                pass_flags,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                uniform_buffer_id,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 11);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 512);
                assert_eq!(*uniform_buffer_id, 1);
            }
            other => panic!("expected RenderPass at 11, got {other:?}"),
        }

        // Command 12: RenderPass Pass 3 (Canvas Target 10, LOAD, NEW_PASS, Pipeline 201, vb2, offset 256)
        match &packet.commands()[12] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                load_op,
                store_op,
                pass_flags,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                uniform_dynamic_offset,
                uniform_buffer_id,
                ..
            } => {
                assert_eq!(*target_type, TARGET_CANVAS);
                assert_eq!(*target_id, 10);
                assert_eq!(*load_op, LOAD_OP_LOAD);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*pipeline_id, 201);
                assert_eq!(*vertex_buffer_id, 3);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 256);
                assert_eq!(*uniform_buffer_id, 1);
            }
            other => panic!("expected RenderPass at 12, got {other:?}"),
        }

        // Command 13: CopyTextureToBuffer (Target 11 -> Buffer 20)
        match &packet.commands()[13] {
            GpuCommand::CopyTextureToBuffer {
                texture_id,
                buffer_id,
                width,
                height,
                ..
            } => {
                assert_eq!(*texture_id, 11);
                assert_eq!(*buffer_id, 20);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer at 13, got {other:?}"),
        }

        // 3. Verify wire encoding: iterate commands from offset 16 and verify 44-byte RenderPass records
        let mut cursor = 16usize;
        let mut render_pass_index = 0;
        for _ in 0..14 {
            let opcode = u16::from_le_bytes(wasm_bytes[cursor..cursor + 2].try_into().unwrap());
            cursor += 2;
            match opcode {
                OPCODE_CREATE_BUFFER => cursor += 12,
                OPCODE_WRITE_BUFFER => cursor += 16,
                OPCODE_CREATE_TEXTURE => cursor += 20,
                OPCODE_CREATE_PIPELINE => cursor += 32,
                OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                OPCODE_RENDER_PASS => {
                    let packed_target = u32::from_le_bytes(wasm_bytes[cursor..cursor + 4].try_into().unwrap());
                    let target_id = u32::from_le_bytes(wasm_bytes[cursor + 4..cursor + 8].try_into().unwrap());
                    let pipeline_id = u32::from_le_bytes(wasm_bytes[cursor + 24..cursor + 28].try_into().unwrap());
                    let vertex_buffer_id = u32::from_le_bytes(wasm_bytes[cursor + 28..cursor + 32].try_into().unwrap());
                    let vertex_count = u32::from_le_bytes(wasm_bytes[cursor + 32..cursor + 36].try_into().unwrap());
                    let dynamic_offset = u32::from_le_bytes(wasm_bytes[cursor + 36..cursor + 40].try_into().unwrap());
                    let uniform_buffer_id = u32::from_le_bytes(wasm_bytes[cursor + 40..cursor + 44].try_into().unwrap());

                    assert!(validate_target_type(packed_target));
                    assert_eq!(vertex_count, 3);
                    assert_eq!(uniform_buffer_id, 1);

                    match render_pass_index {
                        0 => {
                            // Pass 1: Canvas Target 10, Clear, Store, NewPass, pipeline 201, offset 0, vb 2
                            assert_eq!(target_id, 10);
                            assert_eq!(unpack_target_kind(packed_target), TARGET_CANVAS);
                            assert_eq!(unpack_load_op(packed_target), LOAD_OP_CLEAR);
                            assert_eq!(unpack_store_op(packed_target), STORE_OP_STORE);
                            assert_eq!(unpack_pass_flags(packed_target), PASS_FLAG_NEW_PASS);
                            assert_eq!(pipeline_id, 201);
                            assert_eq!(dynamic_offset, 0);
                            assert_eq!(vertex_buffer_id, 2);
                        }
                        1 => {
                            // Pass 2: Offscreen Target 11, Clear, Store, NewPass, pipeline 200, offset 512, vb 2
                            assert_eq!(target_id, 11);
                            assert_eq!(unpack_target_kind(packed_target), TARGET_OFFSCREEN);
                            assert_eq!(unpack_load_op(packed_target), LOAD_OP_CLEAR);
                            assert_eq!(unpack_store_op(packed_target), STORE_OP_STORE);
                            assert_eq!(unpack_pass_flags(packed_target), PASS_FLAG_NEW_PASS);
                            assert_eq!(pipeline_id, 200);
                            assert_eq!(dynamic_offset, 512);
                            assert_eq!(vertex_buffer_id, 2);
                        }
                        2 => {
                            // Pass 3: Canvas Target 10, LOAD, Store, NewPass, pipeline 201, offset 256, vb 3
                            assert_eq!(target_id, 10);
                            assert_eq!(unpack_target_kind(packed_target), TARGET_CANVAS);
                            assert_eq!(unpack_load_op(packed_target), LOAD_OP_LOAD);
                            assert_eq!(unpack_store_op(packed_target), STORE_OP_STORE);
                            assert_eq!(unpack_pass_flags(packed_target), PASS_FLAG_NEW_PASS);
                            assert_eq!(pipeline_id, 201);
                            assert_eq!(dynamic_offset, 256);
                            assert_eq!(vertex_buffer_id, 3);
                        }
                        _ => panic!("unexpected render pass index {render_pass_index}"),
                    }
                    render_pass_index += 1;
                    cursor += 44;
                }
                other => panic!("unexpected opcode {other} at cursor {cursor}"),
            }
        }
        assert_eq!(render_pass_index, 3);

        // 4. Assert underlying ExecutionPlan segment and attachment epoch stamping
        let target_10 = 10;
        let target_11 = 11;
        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            CanvasId::new(target_10),
            ResourceId::new(target_10),
            64,
            64,
            CanvasFormat::Bgra8Unorm,
        );
        let canvas_output = tracker
            .begin_frame_acquire(CanvasId::new(target_10))
            .expect("canvas acquire");

        let root_ctx = RenderContext::new_canvas_acquired(
            ResourceId::new(target_10),
            64,
            64,
            Epoch::new(1),
            canvas_output.epoch,
        );
        let mut session = FrameSession::new(root_ctx, 256)
            .expect("session init")
            .with_uniform_buffer_id(1);

        let mat_handle = Handle::<MaterialDomain>::from_raw(1, 1).expect("valid handle");
        let red_bytes = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
        let blue_bytes = [0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
        let green_bytes = [0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();

        let rec_red = session
            .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, &red_bytes)
            .expect("snapshot red");
        let rec_blue = session
            .snapshot_material_use(mat_handle, DataVersion::new(2), Epoch::ZERO, &blue_bytes)
            .expect("snapshot blue");
        let rec_green = session
            .snapshot_material_use(mat_handle, DataVersion::new(3), Epoch::ZERO, &green_bytes)
            .expect("snapshot green");

        session.begin_render_pass("canvas_prefix", [0.0, 0.0, 0.0, 1.0]).expect("begin canvas prefix");
        session.record_direct_draw(201, 2, 3, Some(rec_red)).expect("draw red");

        let nested_ctx = RenderContext::new_offscreen(
            ResourceId::new(target_11),
            64,
            64,
            Epoch::ZERO,
        );
        session
            .with_nested_render(nested_ctx, |s| {
                s.begin_render_pass("nested_pass", [0.0, 0.0, 0.0, 1.0]).expect("begin nested pass");
                s.record_direct_draw(200, 2, 3, Some(rec_green)).expect("draw green");
                s.end_render_pass().expect("end nested pass");
                Ok(())
            })
            .expect("nested render");

        session.record_direct_draw(201, 3, 3, Some(rec_blue)).expect("draw blue");
        session.end_render_pass().expect("end resumed pass");

        let plan = session.pass_graph().compile(Some(&tracker)).expect("compile plan");
        assert_eq!(plan.canvas_epoch(), Some(canvas_output.epoch));
        assert_eq!(plan.segment_count(), 3);

        let seg0 = &plan.segments()[0];
        let seg1 = &plan.segments()[1];
        let seg2 = &plan.segments()[2];

        assert_eq!(seg0.name(), "canvas_prefix");
        assert_eq!(seg1.name(), "nested_pass");
        assert_eq!(seg2.name(), "canvas_prefix_resumed");

        let ca0 = seg0.primary_color_attachment().unwrap();
        assert!(ca0.is_canvas());
        assert_eq!(ca0.target_id(), ResourceId::new(10));
        assert_eq!(ca0.load_op(), LoadOp::Clear);
        assert_eq!(ca0.store_op(), StoreOp::Store);
        assert_eq!(ca0.canvas_epoch(), Some(canvas_output.epoch));

        let ca1 = seg1.primary_color_attachment().unwrap();
        assert!(!ca1.is_canvas());
        assert_eq!(ca1.target_id(), ResourceId::new(11));
        assert_eq!(ca1.load_op(), LoadOp::Clear);
        assert_eq!(ca1.store_op(), StoreOp::Store);
        assert_eq!(ca1.canvas_epoch(), None);

        let ca2 = seg2.primary_color_attachment().unwrap();
        assert!(ca2.is_canvas());
        assert_eq!(ca2.target_id(), ResourceId::new(10));
        assert_eq!(ca2.load_op(), LoadOp::Load);
        assert_eq!(ca2.store_op(), StoreOp::Store);
        assert_eq!(ca2.canvas_epoch(), Some(canvas_output.epoch));

        // 5. Mathematical verification of pixel sample points on 64x64 targets:
        // Target 11 (offscreen, readback buffer 20):
        // Only nested_pass executes on Target 11, drawing tri1 with Green [0, 255, 0, 255].
        // (24, 32) is inside tri1 -> Green
        // (56, 32) is outside tri1 -> Black (clear color)
        // (2, 2) is outside tri1 -> Black (clear color)
        let pixel_to_ndc = |px: u32, py: u32| -> [f32; 2] {
            let x = ((px as f32 + 0.5) / 64.0) * 2.0 - 1.0;
            let y = 1.0 - ((py as f32 + 0.5) / 64.0) * 2.0;
            [x, y]
        };

        let point_in_tri = |p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> bool {
            let cross = |p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]| -> f32 {
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            };
            let d1 = cross(a, b, p);
            let d2 = cross(b, c, p);
            let d3 = cross(c, a, p);
            let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
            let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
            !(has_neg && has_pos)
        };

        let tri1_a = [-1.0f32, -1.0f32];
        let tri1_b = [0.0f32, -1.0f32];
        let tri1_c = [0.0f32, 1.0f32];

        let tri2_a = [0.0f32, -1.0f32];
        let tri2_b = [1.0f32, -1.0f32];
        let tri2_c = [1.0f32, 1.0f32];

        // Target 11 samples:
        let p24_32 = pixel_to_ndc(24, 32);
        assert!(point_in_tri(p24_32, tri1_a, tri1_b, tri1_c));
        assert!(!point_in_tri(p24_32, tri2_a, tri2_b, tri2_c));

        let p56_32 = pixel_to_ndc(56, 32);
        assert!(!point_in_tri(p56_32, tri1_a, tri1_b, tri1_c));
        assert!(point_in_tri(p56_32, tri2_a, tri2_b, tri2_c));

        let p2_2 = pixel_to_ndc(2, 2);
        assert!(!point_in_tri(p2_2, tri1_a, tri1_b, tri1_c));
        assert!(!point_in_tri(p2_2, tri2_a, tri2_b, tri2_c));

        // Note: For Canvas Target 10 (presented to swapchain):
        // (24, 32) is drawn Red in Pass 1 and preserved via LoadOp::Load -> Red [255, 0, 0, 255]
        // (56, 32) is drawn Blue in Pass 3 -> Blue [0, 0, 255, 255]
        // (2, 2) is unpainted -> Black [0, 0, 0, 255]
    }

    #[test]
    fn test_nested_viewport_scissor_packet_and_command_stream() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        let wasm_bytes = gpu_bridge_build_nested_viewport_scissor_packet();
        let alias_bytes = f3d_build_nested_viewport_scissor_packet();
        assert_eq!(wasm_bytes, alias_bytes);
        assert!(!wasm_bytes.is_empty());

        // 1. Verify packet binary header
        assert_eq!(&wasm_bytes[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes(wasm_bytes[4..6].try_into().unwrap()), PACKET_VERSION);
        let cmd_count = u32::from_le_bytes(wasm_bytes[8..12].try_into().unwrap());
        assert_eq!(cmd_count, 25);

        // 2. Verify structured commands
        let submission = build_nested_viewport_scissor_submission();
        let commands = submission.commands();
        assert_eq!(commands.len(), 25);

        // Setup commands: 0-9
        assert!(matches!(&commands[0], GpuCommand::CreateBuffer { buffer_id: 1, size: 768, .. }));
        assert!(matches!(&commands[1], GpuCommand::CreateBuffer { buffer_id: 2, size: 60, .. }));
        assert!(matches!(&commands[2], GpuCommand::WriteBuffer { buffer_id: 2, offset: 0, .. }));
        assert!(matches!(&commands[3], GpuCommand::CreateBuffer { buffer_id: 3, size: 60, .. }));
        assert!(matches!(&commands[4], GpuCommand::WriteBuffer { buffer_id: 3, offset: 0, .. }));
        assert!(matches!(&commands[5], GpuCommand::CreateTexture { texture_id: 10, width: 64, height: 64, .. }));
        assert!(matches!(&commands[6], GpuCommand::CreateTexture { texture_id: 11, width: 64, height: 64, .. }));
        assert!(matches!(&commands[7], GpuCommand::CreateBuffer { buffer_id: 20, size: 16384, .. }));
        assert!(matches!(&commands[8], GpuCommand::CreateBuffer { buffer_id: 21, size: 16384, .. }));
        assert!(matches!(&commands[9], GpuCommand::CreatePipeline { pipeline_id: 200, target_format: TARGET_FORMAT_RGBA8UNORM, .. }));

        // Command 10: WriteBuffer for uniform buffer 1 (768 bytes, emitted by FrameSession)
        assert!(matches!(&commands[10], GpuCommand::WriteBuffer { buffer_id: 1, offset: 0, .. }));

        // Pass 1 (Outer prefix on Target 10):
        // Command 11: Opener with vertex_count: 0 and clear color [0, 0, 0, 1]
        match &commands[11] {
            GpuCommand::RenderPass { target_id, clear_color, vertex_count, pass_flags, load_op, .. } => {
                assert_eq!(*target_id, 10);
                assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
            }
            other => panic!("expected RenderPass opener at index 11, got {other:?}"),
        }

        // Command 12: SetViewport [0, 0, 64, 64]
        match &commands[12] {
            GpuCommand::SetViewport { x, y, width, height, min_depth, max_depth } => {
                assert_eq!(*x, 0.0);
                assert_eq!(*y, 0.0);
                assert_eq!(*width, 64.0);
                assert_eq!(*height, 64.0);
                assert_eq!(*min_depth, 0.0);
                assert_eq!(*max_depth, 1.0);
            }
            other => panic!("expected SetViewport at index 12, got {other:?}"),
        }

        // Command 13: SetScissorRect [0, 0, 32, 64] (left half)
        match &commands[13] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
                assert_eq!(*width, 32);
                assert_eq!(*height, 64);
            }
            other => panic!("expected SetScissorRect at index 13, got {other:?}"),
        }

        // Command 14: Direct draw red tri1
        match &commands[14] {
            GpuCommand::RenderPass { target_id, pipeline_id, vertex_buffer_id, vertex_count, uniform_dynamic_offset, pass_flags, .. } => {
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*pass_flags, PASS_FLAG_NONE);
            }
            other => panic!("expected RenderPass draw at index 14, got {other:?}"),
        }

        // Pass 2 (Nested inner pass on Target 11):
        // Command 15: Opener with vertex_count: 0
        match &commands[15] {
            GpuCommand::RenderPass { target_id, vertex_count, pass_flags, load_op, .. } => {
                assert_eq!(*target_id, 11);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
            }
            other => panic!("expected RenderPass opener at index 15, got {other:?}"),
        }

        // Command 16: SetViewport [16, 16, 32, 32]
        match &commands[16] {
            GpuCommand::SetViewport { x, y, width, height, .. } => {
                assert_eq!(*x, 16.0);
                assert_eq!(*y, 16.0);
                assert_eq!(*width, 32.0);
                assert_eq!(*height, 32.0);
            }
            other => panic!("expected SetViewport at index 16, got {other:?}"),
        }

        // Command 17: SetScissorRect [16, 16, 32, 32]
        match &commands[17] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 16);
                assert_eq!(*y, 16);
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
            }
            other => panic!("expected SetScissorRect at index 17, got {other:?}"),
        }

        // Command 18: Direct draw green tri1
        match &commands[18] {
            GpuCommand::RenderPass { target_id, pipeline_id, vertex_buffer_id, vertex_count, uniform_dynamic_offset, pass_flags, .. } => {
                assert_eq!(*target_id, 11);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 512);
                assert_eq!(*pass_flags, PASS_FLAG_NONE);
            }
            other => panic!("expected RenderPass draw at index 18, got {other:?}"),
        }

        // Pass 3 (Outer resumed pass on Target 10 with LoadOp::Load):
        // Command 19: Opener with LoadOp::Load
        match &commands[19] {
            GpuCommand::RenderPass { target_id, vertex_count, pass_flags, load_op, .. } => {
                assert_eq!(*target_id, 10);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*load_op, LOAD_OP_LOAD);
            }
            other => panic!("expected RenderPass opener at index 19, got {other:?}"),
        }

        // Command 20: SetViewport [0, 0, 64, 64] (restored outer viewport!)
        match &commands[20] {
            GpuCommand::SetViewport { x, y, width, height, .. } => {
                assert_eq!(*x, 0.0);
                assert_eq!(*y, 0.0);
                assert_eq!(*width, 64.0);
                assert_eq!(*height, 64.0);
            }
            other => panic!("expected restored SetViewport at index 20, got {other:?}"),
        }

        // Command 21: SetScissorRect [32, 0, 32, 64] (resumed right-half scissor!)
        match &commands[21] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 32);
                assert_eq!(*y, 0);
                assert_eq!(*width, 32);
                assert_eq!(*height, 64);
            }
            other => panic!("expected SetScissorRect at index 21, got {other:?}"),
        }

        // Command 22: Direct draw blue tri2
        match &commands[22] {
            GpuCommand::RenderPass { target_id, pipeline_id, vertex_buffer_id, vertex_count, uniform_dynamic_offset, pass_flags, .. } => {
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 3);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*uniform_dynamic_offset, 256);
                assert_eq!(*pass_flags, PASS_FLAG_NONE);
            }
            other => panic!("expected RenderPass draw at index 22, got {other:?}"),
        }

        // Command 23: Copy Texture 10 to Buffer 20
        match &commands[23] {
            GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, .. } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*buffer_id, 20);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer at index 23, got {other:?}"),
        }

        // Command 24: Copy Texture 11 to Buffer 21
        match &commands[24] {
            GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, .. } => {
                assert_eq!(*texture_id, 11);
                assert_eq!(*buffer_id, 21);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer at index 24, got {other:?}"),
        }

        // 3. Binary wire cursor walkthrough (matching bridge_runtime.js decoder)
        let mut cursor = 16usize; // past header
        let mut opcodes = Vec::new();
        for _ in 0..cmd_count {
            let op = u16::from_le_bytes(wasm_bytes[cursor..cursor + 2].try_into().unwrap());
            cursor += 2;
            opcodes.push(op);
            match op {
                OPCODE_CREATE_BUFFER => cursor += 12,
                OPCODE_WRITE_BUFFER => cursor += 16,
                OPCODE_CREATE_TEXTURE => cursor += 20,
                OPCODE_CREATE_PIPELINE => cursor += 32,
                OPCODE_RENDER_PASS => cursor += 44,
                OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                OPCODE_SET_VIEWPORT => {
                    let x = f32::from_le_bytes(wasm_bytes[cursor..cursor + 4].try_into().unwrap());
                    let y = f32::from_le_bytes(wasm_bytes[cursor + 4..cursor + 8].try_into().unwrap());
                    let w = f32::from_le_bytes(wasm_bytes[cursor + 8..cursor + 12].try_into().unwrap());
                    let h = f32::from_le_bytes(wasm_bytes[cursor + 12..cursor + 16].try_into().unwrap());
                    let min_d = f32::from_le_bytes(wasm_bytes[cursor + 16..cursor + 20].try_into().unwrap());
                    let max_d = f32::from_le_bytes(wasm_bytes[cursor + 20..cursor + 24].try_into().unwrap());
                    assert!(w > 0.0 && h > 0.0);
                    assert_eq!(min_d, 0.0);
                    assert_eq!(max_d, 1.0);
                    let _ = (x, y);
                    cursor += 24;
                }
                OPCODE_SET_SCISSOR_RECT => {
                    let x = u32::from_le_bytes(wasm_bytes[cursor..cursor + 4].try_into().unwrap());
                    let y = u32::from_le_bytes(wasm_bytes[cursor + 4..cursor + 8].try_into().unwrap());
                    let w = u32::from_le_bytes(wasm_bytes[cursor + 8..cursor + 12].try_into().unwrap());
                    let h = u32::from_le_bytes(wasm_bytes[cursor + 12..cursor + 16].try_into().unwrap());
                    assert!(w > 0 && h > 0);
                    let _ = (x, y);
                    cursor += 16;
                }
                other => panic!("unexpected opcode {other} at cursor {cursor}"),
            }
        }
        assert_eq!(opcodes.len(), 25);
        assert_eq!(
            opcodes,
            vec![
                OPCODE_CREATE_BUFFER,
                OPCODE_CREATE_BUFFER,
                OPCODE_WRITE_BUFFER,
                OPCODE_CREATE_BUFFER,
                OPCODE_WRITE_BUFFER,
                OPCODE_CREATE_TEXTURE,
                OPCODE_CREATE_TEXTURE,
                OPCODE_CREATE_BUFFER,
                OPCODE_CREATE_BUFFER,
                OPCODE_CREATE_PIPELINE,
                OPCODE_WRITE_BUFFER,
                // Pass 1:
                OPCODE_RENDER_PASS,
                OPCODE_SET_VIEWPORT,
                OPCODE_SET_SCISSOR_RECT,
                OPCODE_RENDER_PASS,
                // Pass 2:
                OPCODE_RENDER_PASS,
                OPCODE_SET_VIEWPORT,
                OPCODE_SET_SCISSOR_RECT,
                OPCODE_RENDER_PASS,
                // Pass 3:
                OPCODE_RENDER_PASS,
                OPCODE_SET_VIEWPORT,
                OPCODE_SET_SCISSOR_RECT,
                OPCODE_RENDER_PASS,
                // Copies:
                OPCODE_COPY_TEXTURE_TO_BUFFER,
                OPCODE_COPY_TEXTURE_TO_BUFFER,
            ]
        );

        // 4. Mathematical validation of sample points and scissor bounds:
        let in_rect = |px: u32, py: u32, r: [u32; 4]| -> bool {
            px >= r[0] && px < r[0] + r[2] && py >= r[1] && py < r[1] + r[3]
        };

        // Target 10: Left-half scissor [0, 0, 32, 64] vs Right-half scissor [32, 0, 32, 64]
        let left_scissor = [0, 0, 32, 64];
        let right_scissor = [32, 0, 32, 64];
        assert!(in_rect(24, 32, left_scissor));
        assert!(!in_rect(56, 32, left_scissor));
        assert!(!in_rect(24, 32, right_scissor));
        assert!(in_rect(56, 32, right_scissor));

        // Negative check: If outer prefix scissor [0, 0, 32, 64] were improperly retained into Pass 3,
        // right-side sample point (56, 32) would be outside the scissor rectangle and clipped to Black.
        assert!(!in_rect(56, 32, left_scissor));

        // Target 11: Centered sub-rect scissor [16, 16, 32, 32]
        let inner_scissor = [16, 16, 32, 32];
        assert!(in_rect(24, 32, inner_scissor)); // Green pixel inside centered scissor
        assert!(!in_rect(8, 8, inner_scissor));   // Untouched background outside centered scissor
        assert!(!in_rect(56, 32, inner_scissor)); // Right-half background outside centered scissor
    }

    #[test]
    fn test_lower_plan_viewport_scissor_deduplication_and_restoration() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut pass0 = Pass::new_render(PassId::new(1), "pass_with_changing_scissor");
        pass0.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));

        // Draw 0: viewport [0, 0, 100, 100], scissor [0, 0, 50, 100]
        pass0.draws.push(
            Draw::new(1, 100, 3, 0, Vec::new())
                .with_viewport(Some([0, 0, 100, 100]))
                .with_scissor(Some([0, 0, 50, 100]), true),
        );

        // Draw 1: same viewport [0, 0, 100, 100], changed scissor [50, 0, 50, 100]
        pass0.draws.push(
            Draw::new(2, 100, 3, 256, Vec::new())
                .with_viewport(Some([0, 0, 100, 100]))
                .with_scissor(Some([50, 0, 50, 100]), true),
        );

        // Draw 2: identical viewport [0, 0, 100, 100] and identical scissor [50, 0, 50, 100] -> zero state changes
        pass0.draws.push(
            Draw::new(3, 100, 3, 512, Vec::new())
                .with_viewport(Some([0, 0, 100, 100]))
                .with_scissor(Some([50, 0, 50, 100]), true),
        );

        // Pass 1 on Target 20: changed viewport, scissor disabled
        let mut pass1 = Pass::new_render(PassId::new(2), "pass_target_20");
        pass1.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(20),
            [0.0, 0.0, 0.0, 1.0],
        ));
        pass1.draws.push(
            Draw::new(4, 100, 3, 0, Vec::new())
                .with_viewport(Some([10, 10, 40, 40]))
                .with_scissor(None, false),
        );

        let plan = ExecutionPlan {
            segments: vec![
                PlanSegment::from_pass(&pass0),
                PlanSegment::from_pass(&pass1),
            ],
            canvas_epoch: None,
            pass_count: 2,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("lower plan must succeed");

        // Pass 0:
        // 0: RenderPass opener (vertex_count: 0, new pass)
        // 1: SetViewport [0, 0, 100, 100]
        // 2: SetScissorRect [0, 0, 50, 100]
        // 3: Draw 0 (vertex_count: 3)
        // 4: SetScissorRect [50, 0, 50, 100] (viewport was NOT re-emitted!)
        // 5: Draw 1 (vertex_count: 3)
        // 6: Draw 2 (vertex_count: 3, neither viewport nor scissor re-emitted!)
        // Pass 1:
        // 7: RenderPass opener on target 20 (vertex_count: 0, new pass)
        // 8: SetViewport [10, 10, 40, 40]
        // 9: Draw 3 (vertex_count: 3, no scissor emitted)
        assert_eq!(commands.len(), 10);

        assert!(matches!(&commands[0], GpuCommand::RenderPass { vertex_count: 0, pass_flags: PASS_FLAG_NEW_PASS, .. }));
        assert!(matches!(&commands[1], GpuCommand::SetViewport { width: 100.0, height: 100.0, .. }));
        assert!(matches!(&commands[2], GpuCommand::SetScissorRect { x: 0, width: 50, .. }));
        assert!(matches!(&commands[3], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
        assert!(matches!(&commands[4], GpuCommand::SetScissorRect { x: 50, width: 50, .. }));
        assert!(matches!(&commands[5], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
        assert!(matches!(&commands[6], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
        assert!(matches!(&commands[7], GpuCommand::RenderPass { target_id: 20, vertex_count: 0, pass_flags: PASS_FLAG_NEW_PASS, .. }));
        assert!(matches!(&commands[8], GpuCommand::SetViewport { x: 10.0, y: 10.0, width: 40.0, height: 40.0, .. }));
        assert!(matches!(&commands[9], GpuCommand::RenderPass { target_id: 20, vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
    }

    #[test]
    fn test_lower_plan_near_neighbor_default_vs_explicit_zero_viewport_and_scissor_restoration() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, DrawKind, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        // Pass 0 on Target 10:
        // Tests:
        // - Draw 0: default viewport (None), scissor disabled (None) -> MUST NOT emit SetViewport, MUST NOT emit dummy opener.
        //   Direct draw is emitted with PASS_FLAG_NEW_PASS directly opening the pass!
        // - Draw 1: explicit zero viewport Some([0, 0, 0, 0]) -> MUST emit SetViewport(0, 0, 0, 0).
        // - Draw 2: explicit non-zero viewport Some([0, 0, 800, 600]) and enabled scissor Some([10, 20, 100, 200]) -> emits SetViewport and SetScissorRect.
        // - Draw 3: scissor disabled after enabled, with full-attachment rect Some([0, 0, 800, 600]) -> MUST emit SetScissorRect(0, 0, 800, 600) to clear clipping.
        // - Draw 4: bundle execution with active viewport/scissor -> inherits state without extra sets.
        let mut pass0 = Pass::new_render(PassId::new(1), "pass_near_neighbor");
        pass0.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));

        // Draw 0: default unspecified viewport (None), no scissor (None)
        pass0.draws.push(
            Draw::new(1, 100, 3, 0, Vec::new())
                .with_viewport(None)
                .with_scissor(None, false),
        );

        // Draw 1: near-neighbor explicit zero-size viewport Some([0, 0, 0, 0])
        pass0.draws.push(
            Draw::new(2, 100, 3, 256, Vec::new())
                .with_viewport(Some([0, 0, 0, 0]))
                .with_scissor(None, false),
        );

        // Draw 2: non-zero viewport and active scissor test
        pass0.draws.push(
            Draw::new(3, 100, 3, 512, Vec::new())
                .with_viewport(Some([0, 0, 800, 600]))
                .with_scissor(Some([10, 20, 100, 200]), true),
        );

        // Draw 3: scissor test disabled after enable -> provides full-attachment scissor to clear clipping
        pass0.draws.push(
            Draw::new(4, 100, 3, 768, Vec::new())
                .with_viewport(Some([0, 0, 800, 600]))
                .with_full_attachment_scissor(800, 600),
        );

        // Draw 4: pre-recorded bundle execution inheriting active viewport and cleared scissor
        pass0.draws.push(
            Draw::new_bundle(5, 42, 100, Vec::new())
                .with_viewport(Some([0, 0, 800, 600]))
                .with_scissor(None, false),
        );

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&pass0)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("lower plan must succeed");

        // Expected command stream:
        // Command 0: Draw 0 direct draw (pass_flags: PASS_FLAG_NEW_PASS, no dummy opener!)
        // Command 1: SetViewport [0, 0, 0, 0] (explicit zero size!)
        // Command 2: Draw 1 direct draw (pass_flags: PASS_FLAG_NONE)
        // Command 3: SetViewport [0, 0, 800, 600]
        // Command 4: SetScissorRect [10, 20, 100, 200]
        // Command 5: Draw 2 direct draw (pass_flags: PASS_FLAG_NONE)
        // Command 6: SetScissorRect [0, 0, 800, 600] (full-attachment restore to clear clipping!)
        // Command 7: Draw 3 direct draw (pass_flags: PASS_FLAG_NONE)
        // Command 8: ExecuteBundles [42] (bundle inherits viewport and restored scissor without redundant sets!)
        assert_eq!(commands.len(), 9);

        // Command 0: Draw 0 directly opens pass
        match &commands[0] {
            GpuCommand::RenderPass { vertex_count, pass_flags, load_op, .. } => {
                assert_eq!(*vertex_count, 3);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
            }
            other => panic!("expected RenderPass draw opening pass at 0, got {other:?}"),
        }

        // Command 1: Explicit zero viewport
        match &commands[1] {
            GpuCommand::SetViewport { x, y, width, height, min_depth, max_depth } => {
                assert_eq!(*x, 0.0);
                assert_eq!(*y, 0.0);
                assert_eq!(*width, 0.0);
                assert_eq!(*height, 0.0);
                assert_eq!(*min_depth, 0.0);
                assert_eq!(*max_depth, 1.0);
            }
            other => panic!("expected SetViewport(0,0,0,0) at 1, got {other:?}"),
        }

        // Command 2: Draw 1
        assert!(matches!(&commands[2], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));

        // Command 3: SetViewport 800x600
        match &commands[3] {
            GpuCommand::SetViewport { width, height, .. } => {
                assert_eq!(*width, 800.0);
                assert_eq!(*height, 600.0);
            }
            other => panic!("expected SetViewport at 3, got {other:?}"),
        }

        // Command 4: SetScissorRect clipping box
        match &commands[4] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 10);
                assert_eq!(*y, 20);
                assert_eq!(*width, 100);
                assert_eq!(*height, 200);
            }
            other => panic!("expected SetScissorRect at 4, got {other:?}"),
        }

        // Command 5: Draw 2
        assert!(matches!(&commands[5], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));

        // Command 6: SetScissorRect full attachment restore
        match &commands[6] {
            GpuCommand::SetScissorRect { x, y, width, height } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
                assert_eq!(*width, 800);
                assert_eq!(*height, 600);
            }
            other => panic!("expected full-attachment SetScissorRect at 6, got {other:?}"),
        }

        // Command 7: Draw 3
        assert!(matches!(&commands[7], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));

        // Command 8: ExecuteBundles
        match &commands[8] {
            GpuCommand::ExecuteBundles { bundle_ids } => {
                assert_eq!(bundle_ids, &alloc::vec![42]);
            }
            other => panic!("expected ExecuteBundles at 8, got {other:?}"),
        }
    }

    #[test]
    fn test_draw_parameters_packet_and_command_stream() {
        let packet = build_draw_parameters_submission();
        let commands = packet.commands();
        assert_eq!(commands.len(), 9);

        // Command 0: CreateBuffer vb2 (120 bytes)
        match &commands[0] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 2);
                assert_eq!(*size, 120);
                assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer at 0, got {other:?}"),
        }

        // Command 1: WriteBuffer vb2
        match &commands[1] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 2);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 120);
            }
            other => panic!("expected WriteBuffer at 1, got {other:?}"),
        }

        // Command 2: CreateTexture target 10
        match &commands[2] {
            GpuCommand::CreateTexture { texture_id, width, height, format, usage } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(*format, TARGET_FORMAT_RGBA8UNORM);
                assert_eq!(*usage, TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC);
            }
            other => panic!("expected CreateTexture at 2, got {other:?}"),
        }

        // Command 3: CreateBuffer readback 20
        match &commands[3] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, 20);
                assert_eq!(*size, 16384);
                assert_eq!(*usage, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer at 3, got {other:?}"),
        }

        // Command 4: CreatePipeline 200
        match &commands[4] {
            GpuCommand::CreatePipeline {
                pipeline_id,
                target_format,
                has_vertex_buffer,
                has_uniform_buffer,
                uniform_size,
                vertex_stride,
                wgsl_code,
            } => {
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*target_format, TARGET_FORMAT_RGBA8UNORM);
                assert!(*has_vertex_buffer);
                assert!(!*has_uniform_buffer);
                assert_eq!(*uniform_size, 0);
                assert_eq!(*vertex_stride, 20);
                assert!(wgsl_code.contains("@builtin(instance_index)"));
                assert!(wgsl_code.contains("instance_idx == 5u"));
                assert!(wgsl_code.contains("instance_idx == 6u"));
            }
            other => panic!("expected CreatePipeline at 4, got {other:?}"),
        }

        // Command 5: RenderPass opener (vertex_count: 0, pass_flags: PASS_FLAG_NEW_PASS)
        match &commands[5] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                vertex_count,
                pass_flags,
                load_op,
                store_op,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
                assert_eq!(*store_op, STORE_OP_STORE);
            }
            other => panic!("expected RenderPass opener at 5, got {other:?}"),
        }

        // Command 6: SetDrawParameters (instance_count: 2, first_vertex: 3, first_instance: 5)
        match &commands[6] {
            GpuCommand::SetDrawParameters {
                instance_count,
                first_vertex,
                first_instance,
            } => {
                assert_eq!(*instance_count, 2);
                assert_eq!(*first_vertex, 3);
                assert_eq!(*first_instance, 5);
            }
            other => panic!("expected SetDrawParameters at 6, got {other:?}"),
        }

        // Command 7: RenderPass direct draw (vertex_count: 3, pass_flags: PASS_FLAG_NONE)
        match &commands[7] {
            GpuCommand::RenderPass {
                target_type,
                target_id,
                pipeline_id,
                vertex_buffer_id,
                vertex_count,
                pass_flags,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 200);
                assert_eq!(*vertex_buffer_id, 2);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*pass_flags, PASS_FLAG_NONE);
            }
            other => panic!("expected RenderPass draw at 7, got {other:?}"),
        }

        // Command 8: CopyTextureToBuffer (10 -> 20)
        match &commands[8] {
            GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, .. } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*buffer_id, 20);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer at 8, got {other:?}"),
        }

        // Encode and verify binary wire representation
        let wire_bytes = packet.encode().expect("encoding must succeed");
        assert!(wire_bytes.len() >= 16);
        assert_eq!(&wire_bytes[0..4], &PACKET_MAGIC);
        let version = u16::from_le_bytes(wire_bytes[4..6].try_into().unwrap());
        let flags = u16::from_le_bytes(wire_bytes[6..8].try_into().unwrap());
        let cmd_count = u32::from_le_bytes(wire_bytes[8..12].try_into().unwrap());
        let total_data_len = u32::from_le_bytes(wire_bytes[12..16].try_into().unwrap());
        assert_eq!(version, 1);
        assert_eq!(flags, 0);
        assert_eq!(cmd_count, 9);
        assert_eq!(total_data_len as usize, 944); // 120 bytes vb2 data + 824 bytes CreatePipeline WGSL source

        // Iterate through wire records and inspect the exact bytes for opcode 11
        let mut cursor = 16usize;
        let mut found_draw_parameters = false;
        for _ in 0..cmd_count {
            let opcode = u16::from_le_bytes(wire_bytes[cursor..cursor + 2].try_into().unwrap());
            cursor += 2;
            match opcode {
                OPCODE_CREATE_BUFFER => cursor += 12,
                OPCODE_WRITE_BUFFER => cursor += 16,
                OPCODE_CREATE_TEXTURE => cursor += 20,
                OPCODE_CREATE_PIPELINE => cursor += 32,
                OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                OPCODE_SET_VIEWPORT => cursor += 24,
                OPCODE_SET_SCISSOR_RECT => cursor += 16,
                OPCODE_RENDER_PASS => cursor += 44,
                OPCODE_EXECUTE_BUNDLES => {
                    let count = u32::from_le_bytes(wire_bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                    cursor += 4 + count * 4;
                }
                OPCODE_SET_DRAW_PARAMETERS => {
                    let instance_count = u32::from_le_bytes(wire_bytes[cursor..cursor + 4].try_into().unwrap());
                    let first_vertex = u32::from_le_bytes(wire_bytes[cursor + 4..cursor + 8].try_into().unwrap());
                    let first_instance = u32::from_le_bytes(wire_bytes[cursor + 8..cursor + 12].try_into().unwrap());
                    assert_eq!(instance_count, 2);
                    assert_eq!(first_vertex, 3);
                    assert_eq!(first_instance, 5);
                    cursor += 12;
                    found_draw_parameters = true;
                }
                other => panic!("unexpected opcode {other} at cursor {cursor}"),
            }
        }
        assert!(found_draw_parameters, "wire stream must contain opcode 11 SetDrawParameters");

        // Mathematical NDC Point-In-Triangle Proof:
        // Triangle base (vertices 3..5): [-0.2, -0.5], [0.2, -0.5], [0.0, 0.5] (CCW winding)
        // Instance 5 offset x = -0.5 -> [-0.7, -0.5], [-0.3, -0.5], [-0.5, 0.5] (Red)
        // Instance 6 offset x = +0.5 -> [0.3, -0.5], [0.7, -0.5], [0.5, 0.5] (Green)
        fn point_in_tri(px: u32, py: u32, v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> bool {
            let ndc_x = (2.0 * px as f32 + 1.0) / 64.0 - 1.0;
            let ndc_y = 1.0 - (2.0 * py as f32 + 1.0) / 64.0;
            let cross = |a: [f32; 2], b: [f32; 2]| -> f32 {
                (b[0] - a[0]) * (ndc_y - a[1]) - (b[1] - a[1]) * (ndc_x - a[0])
            };
            let c0 = cross(v0, v1);
            let c1 = cross(v1, v2);
            let c2 = cross(v2, v0);
            (c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0) || (c0 <= 0.0 && c1 <= 0.0 && c2 <= 0.0)
        }

        let tri_inst5_v0 = [-0.7f32, -0.5f32];
        let tri_inst5_v1 = [-0.3f32, -0.5f32];
        let tri_inst5_v2 = [-0.5f32, 0.5f32];

        let tri_inst6_v0 = [0.3f32, -0.5f32];
        let tri_inst6_v1 = [0.7f32, -0.5f32];
        let tri_inst6_v2 = [0.5f32, 0.5f32];

        // Pixel (16, 32): Inside Instance 5 (Red), outside Instance 6
        assert!(point_in_tri(16, 32, tri_inst5_v0, tri_inst5_v1, tri_inst5_v2));
        assert!(!point_in_tri(16, 32, tri_inst6_v0, tri_inst6_v1, tri_inst6_v2));

        // Pixel (48, 32): Inside Instance 6 (Green), outside Instance 5
        assert!(!point_in_tri(48, 32, tri_inst5_v0, tri_inst5_v1, tri_inst5_v2));
        assert!(point_in_tri(48, 32, tri_inst6_v0, tri_inst6_v1, tri_inst6_v2));

        // Pixel (32, 32) [screen center]: Outside both instances -> Black
        assert!(!point_in_tri(32, 32, tri_inst5_v0, tri_inst5_v1, tri_inst5_v2));
        assert!(!point_in_tri(32, 32, tri_inst6_v0, tri_inst6_v1, tri_inst6_v2));

        // Pixel (2, 2) [top-left corner]: Outside both instances -> Black
        assert!(!point_in_tri(2, 2, tri_inst5_v0, tri_inst5_v1, tri_inst5_v2));
        assert!(!point_in_tri(2, 2, tri_inst6_v0, tri_inst6_v1, tri_inst6_v2));
    }

    #[test]
    fn test_lower_plan_draw_parameters_omitted_when_default() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            plan::PlanSegment,
        };

        let mut pass = Pass::new_render(PassId::new(1), "pass_default");
        pass.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));

        // Default draw: instance_count: 1, first_vertex: 0, first_instance: 0
        let draw = Draw::new(1, 100, 3, 0, Vec::new());
        assert_eq!(draw.instance_count(), 1);
        assert_eq!(draw.first_vertex(), 0);
        assert_eq!(draw.first_instance(), 0);
        pass.draws.push(draw);

        let plan = ExecutionPlan {
            segments: alloc::vec![PlanSegment::from_pass(&pass)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("lower plan must succeed");

        // Exactly 1 command: direct RenderPass draw with PASS_FLAG_NEW_PASS
        // SetDrawParameters is omitted; dummy opener is omitted.
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            GpuCommand::RenderPass { vertex_count, pass_flags, .. } => {
                assert_eq!(*vertex_count, 3);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
            }
            other => panic!("expected RenderPass draw at 0, got {other:?}"),
        }
    }

    #[test]
    fn test_lower_plan_draw_parameters_zero_instance_count() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            plan::PlanSegment,
        };

        let mut pass = Pass::new_render(PassId::new(1), "pass_zero_instances");
        pass.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));

        let mut draw = Draw::new(1, 100, 3, 0, Vec::new());
        draw.instance_count = 0; // Zero instances requested
        assert_eq!(draw.instance_count(), 0);
        pass.draws.push(draw);

        let plan = ExecutionPlan {
            segments: alloc::vec![PlanSegment::from_pass(&pass)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("lower plan must succeed");

        // 3 commands:
        // 0: Opener (vertex_count: 0, pass_flags: PASS_FLAG_NEW_PASS)
        // 1: SetDrawParameters (instance_count: 0, preserved, not coerced to 1!)
        // 2: RenderPass draw (vertex_count: 3, pass_flags: PASS_FLAG_NONE)
        assert_eq!(commands.len(), 3);
        assert!(matches!(&commands[0], GpuCommand::RenderPass { vertex_count: 0, pass_flags: PASS_FLAG_NEW_PASS, .. }));
        match &commands[1] {
            GpuCommand::SetDrawParameters { instance_count, first_vertex, first_instance } => {
                assert_eq!(*instance_count, 0);
                assert_eq!(*first_vertex, 0);
                assert_eq!(*first_instance, 0);
            }
            other => panic!("expected SetDrawParameters at 1, got {other:?}"),
        }
        assert!(matches!(&commands[2], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
    }

    #[test]
    fn test_lower_plan_draw_parameters_resets_on_subsequent_draw() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            plan::PlanSegment,
        };

        let mut pass = Pass::new_render(PassId::new(1), "pass_subsequent_reset");
        pass.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));

        // Draw 0: non-default parameters (instance_count: 4, first_vertex: 10, first_instance: 2)
        let mut draw0 = Draw::new(1, 100, 3, 0, Vec::new());
        draw0.instance_count = 4;
        draw0.first_vertex = 10;
        draw0.first_instance = 2;
        pass.draws.push(draw0);

        // Draw 1: default parameters (instance_count: 1, first_vertex: 0, first_instance: 0)
        let draw1 = Draw::new(2, 100, 3, 0, Vec::new());
        pass.draws.push(draw1);

        let plan = ExecutionPlan {
            segments: alloc::vec![PlanSegment::from_pass(&pass)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("lower plan must succeed");

        // 4 commands:
        // 0: Opener (pass_flags: PASS_FLAG_NEW_PASS)
        // 1: SetDrawParameters { 4, 10, 2 }
        // 2: Draw 0 direct draw (pass_flags: PASS_FLAG_NONE)
        // 3: Draw 1 direct draw (pass_flags: PASS_FLAG_NONE) -- NO SetDrawParameters emitted for Draw 1!
        assert_eq!(commands.len(), 4);
        assert!(matches!(&commands[0], GpuCommand::RenderPass { vertex_count: 0, pass_flags: PASS_FLAG_NEW_PASS, .. }));
        assert!(matches!(&commands[1], GpuCommand::SetDrawParameters { instance_count: 4, first_vertex: 10, first_instance: 2 }));
        assert!(matches!(&commands[2], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
        assert!(matches!(&commands[3], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
    }

    #[test]
    fn test_lower_plan_draw_parameters_on_second_draw_after_default_first() {
        use f3d_graph::{
            pass::{ColorAttachment, Draw, Pass, PassId},
            plan::PlanSegment,
        };

        let mut pass = Pass::new_render(PassId::new(1), "pass_default_then_nondefault");
        pass.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));

        // Draw 0: default parameters
        let draw0 = Draw::new(1, 100, 3, 0, Vec::new());
        pass.draws.push(draw0);

        // Draw 1: non-default parameters (3, 6, 1)
        let mut draw1 = Draw::new(2, 100, 3, 0, Vec::new());
        draw1.instance_count = 3;
        draw1.first_vertex = 6;
        draw1.first_instance = 1;
        pass.draws.push(draw1);

        let plan = ExecutionPlan {
            segments: alloc::vec![PlanSegment::from_pass(&pass)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("lower plan must succeed");

        // 3 commands:
        // 0: Draw 0 directly opens pass (pass_flags: PASS_FLAG_NEW_PASS)
        // 1: SetDrawParameters { 3, 6, 1 }
        // 2: Draw 1 direct draw (pass_flags: PASS_FLAG_NONE)
        assert_eq!(commands.len(), 3);
        assert!(matches!(&commands[0], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NEW_PASS, .. }));
        assert!(matches!(&commands[1], GpuCommand::SetDrawParameters { instance_count: 3, first_vertex: 6, first_instance: 1 }));
        assert!(matches!(&commands[2], GpuCommand::RenderPass { vertex_count: 3, pass_flags: PASS_FLAG_NONE, .. }));
    }
}
