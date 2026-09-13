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
        VertexPosUv, WGSL_AFFINE_ROWS_DECLARATION, WGSL_COLOR_UNIFORM_DECLARATION,
    },
    ownership::{
        BorrowScope, BorrowToken, DataVersion, Epoch, OwnershipError, PerUseByteBuffer, RegionState,
    },
};
use f3d_graph::{
    CanvasEpochTracker, CanvasFormat, CanvasId, CopyCommand, DrawKind, ExecutionPlan, LoadOp, PassKind,
    ResourceAccess, ResourceId, ResourceUse, StoreOp,
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
/// Opcode for creating a render pipeline with depth/stencil state.
pub const OPCODE_CREATE_PIPELINE_DEPTH: u16 = 12;
/// Opcode for encoding a complete render pass with a depth attachment.
pub const OPCODE_RENDER_PASS_DEPTH: u16 = 13;
/// Opcode for creating a render pipeline with face culling and front-face winding state.
pub const OPCODE_CREATE_PIPELINE_CULL: u16 = 14;
/// Opcode for creating a render pipeline with depth state, face culling, and front-face winding state.
pub const OPCODE_CREATE_PIPELINE_DEPTH_CULL: u16 = 15;
/// Opcode for creating a render pipeline with depth state, face culling, front-face winding state, and color write mask.
pub const OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR: u16 = 16;
/// Opcode for uploading pixel data to a 2D GPU texture.
pub const OPCODE_WRITE_TEXTURE: u16 = 17;
/// Opcode for creating a textured render pipeline with texture and sampler bindings.
pub const OPCODE_CREATE_PIPELINE_TEXTURED: u16 = 18;
/// Record repeated draws with distinct dynamic offsets in one render bundle.
pub const OPCODE_RECORD_BUNDLE_BATCH: u16 = 19;
/// Opcode for copying raw bytes from one GPU buffer to another.
pub const OPCODE_COPY_BUFFER_TO_BUFFER: u16 = 20;
/// Opcode for compiling and creating a compute pipeline with explicit bind group layout.
pub const OPCODE_CREATE_COMPUTE_PIPELINE: u16 = 21;
/// Opcode for executing a compute pass dispatching workgroups with explicit buffer bindings.
pub const OPCODE_DISPATCH_COMPUTE: u16 = 22;

/// Compute binding type: uniform buffer (`GPUBufferBindingType.uniform`).
pub const BINDING_TYPE_UNIFORM: u32 = 0;
/// Compute binding type: read-only storage buffer (`GPUBufferBindingType.read-only-storage`).
pub const BINDING_TYPE_STORAGE_READ: u32 = 1;
/// Compute binding type: read-write storage buffer (`GPUBufferBindingType.storage`).
pub const BINDING_TYPE_STORAGE_READ_WRITE: u32 = 2;

/// Sampler filter mode: nearest-neighbor filtering.
pub const SAMPLER_FILTER_NEAREST: u32 = 0;
/// Sampler filter mode: linear filtering.
pub const SAMPLER_FILTER_LINEAR: u32 = 1;

/// Sampler address mode: clamp to edge.
pub const ADDRESS_MODE_CLAMP_TO_EDGE: u32 = 0;
/// Sampler address mode: repeat.
pub const ADDRESS_MODE_REPEAT: u32 = 1;

/// Face culling mode: do not cull any faces (Three.js DoubleSide).
pub const CULL_MODE_NONE: u32 = 0;
/// Face culling mode: cull front-facing polygons.
pub const CULL_MODE_FRONT: u32 = 1;
/// Face culling mode: cull back-facing polygons (Three.js FrontSide / BackSide after winding resolution).
pub const CULL_MODE_BACK: u32 = 2;

/// Front-facing winding order: counter-clockwise (default).
pub const FRONT_FACE_CCW: u32 = 0;
/// Front-facing winding order: clockwise (inverted/reflected or flipSided).
pub const FRONT_FACE_CW: u32 = 1;

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
/// GPUBufferUsage flag: storage buffer (`GPUBufferUsage.STORAGE`).
pub const BUFFER_USAGE_STORAGE: u32 = 128;

/// Default persistent buffer ID for compute AffineRows storage.
pub const COMPUTE_AFFINE_BUFFER_ID: u32 = 701;
/// Default persistent buffer ID for compute input points storage.
pub const COMPUTE_INPUT_POINTS_BUFFER_ID: u32 = 702;
/// Default persistent buffer ID for compute output points storage.
pub const COMPUTE_OUTPUT_POINTS_BUFFER_ID: u32 = 703;
/// Default persistent buffer ID for compute readback staging.
pub const COMPUTE_READBACK_BUFFER_ID: u32 = 704;
/// Default persistent pipeline ID for AffineRows compute pipeline.
pub const COMPUTE_PIPELINE_ID: u32 = 300;

/// Persistent source buffer ID for AffineRows storage uploads (workload b / tmt.5).
pub const AFFINE_ROWS_STORAGE_SRC_BUFFER_ID: u32 = 610;
/// Persistent destination buffer slot 0 for AffineRows storage uploads (workload b / tmt.5).
pub const AFFINE_ROWS_STORAGE_DST_SLOT_0: u32 = 611;
/// Persistent destination buffer slot 1 for AffineRows storage uploads (workload b / tmt.5).
pub const AFFINE_ROWS_STORAGE_DST_SLOT_1: u32 = 612;
/// Default persistent storage buffer ID alias (workload b / tmt.5).
pub const AFFINE_ROWS_STORAGE_BUFFER_ID: u32 = AFFINE_ROWS_STORAGE_SRC_BUFFER_ID;

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
/// Target format code for standard depth24plus format.
pub const TARGET_FORMAT_DEPTH24PLUS: u32 = 3;
/// Target format code for standard depth32float format.
pub const TARGET_FORMAT_DEPTH32FLOAT: u32 = 4;

/// Depth comparison function: Never.
pub const DEPTH_COMPARE_NEVER: u32 = 1;
/// Depth comparison function: Less.
pub const DEPTH_COMPARE_LESS: u32 = 2;
/// Depth comparison function: Equal.
pub const DEPTH_COMPARE_EQUAL: u32 = 3;
/// Depth comparison function: LessEqual.
pub const DEPTH_COMPARE_LESS_EQUAL: u32 = 4;
/// Depth comparison function: Greater.
pub const DEPTH_COMPARE_GREATER: u32 = 5;
/// Depth comparison function: NotEqual.
pub const DEPTH_COMPARE_NOT_EQUAL: u32 = 6;
/// Depth comparison function: GreaterEqual.
pub const DEPTH_COMPARE_GREATER_EQUAL: u32 = 7;
/// Depth comparison function: Always.
pub const DEPTH_COMPARE_ALWAYS: u32 = 8;

/// Packs depth load op, depth store op, and depth read-only flag into a 32-bit integer.
#[inline]
pub const fn pack_depth_ops(load_op: u32, store_op: u32, read_only: bool) -> u32 {
    let ro_bit = if read_only { 1u32 } else { 0u32 };
    (load_op & 0xFF) | ((store_op & 0xFF) << 8) | ((ro_bit & 0xFF) << 16)
}

/// Extracts depth load operation from packed depth ops.
#[inline]
pub const fn unpack_depth_load_op(packed: u32) -> u32 {
    packed & 0xFF
}

/// Extracts depth store operation from packed depth ops.
#[inline]
pub const fn unpack_depth_store_op(packed: u32) -> u32 {
    (packed >> 8) & 0xFF
}

/// Extracts depth read-only flag from packed depth ops.
#[inline]
pub const fn unpack_depth_read_only(packed: u32) -> bool {
    ((packed >> 16) & 0xFF) != 0
}

/// Layout descriptor for an individual resource binding within a compute pipeline layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuComputeBindingLayout {
    /// Zero-based binding index in group 0 (`@binding(index)`).
    pub binding_index: u32,
    /// Binding buffer type (0 = uniform, 1 = read-only storage, 2 = read-write storage).
    pub binding_type: u32,
    /// Minimum binding size in bytes (0 if unconstrained).
    pub min_binding_size: u32,
}

/// Explicit buffer binding parameter for a compute dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBufferBinding {
    /// Zero-based binding index in group 0 (`@binding(index)`).
    pub binding_index: u32,
    /// Allocated GPU buffer identifier.
    pub buffer_id: u32,
    /// Byte offset within the bound buffer (must be multiple of 4; device limits checked on host).
    pub offset: u64,
    /// Byte size slice bound to this binding (must be multiple of 4, non-zero).
    pub size: u64,
    /// Binding buffer type (0 = uniform, 1 = read-only storage, 2 = read-write storage).
    pub binding_type: u32,
    /// Semantic epoch of the bound buffer state region.
    pub epoch: Epoch,
    /// Per-use buffer data version.
    pub data_version: DataVersion,
}

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
    /// Record one bundle with `draw_count` draws and checked dynamic-offset progression.
    RecordBundleBatch {
        /// Bundle and source resource identifiers, as in `RecordBundle`.
        bundle_id: u32,
        pipeline_id: u32,
        vertex_buffer_id: u32,
        vertex_count: u32,
        uniform_dynamic_offset: u32,
        uniform_buffer_id: u32,
        target_format: u32,
        /// Zero records a legal empty bundle.
        draw_count: u32,
        /// Byte stride added to the uniform offset after each draw.
        dynamic_offset_stride: u32,
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
    /// Command to compile and create a render pipeline from WGSL shader text with depth testing/writing state.
    CreatePipelineDepth {
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
        /// Explicit uniform buffer binding byte size.
        uniform_size: u32,
        /// Explicit vertex array byte stride.
        vertex_stride: u32,
        /// Depth texture format code (3 = depth24plus, 4 = depth32float).
        depth_format: u32,
        /// Whether depth writes are enabled.
        depth_write_enabled: bool,
        /// Depth comparison function code (1..=8).
        depth_compare: u32,
    },
    /// Command to compile and create a render pipeline with explicit face culling and front-face winding state.
    CreatePipelineCull {
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
        /// Explicit uniform buffer binding byte size.
        uniform_size: u32,
        /// Explicit vertex array byte stride.
        vertex_stride: u32,
        /// Face culling mode (0 = None, 1 = Front, 2 = Back).
        cull_mode: u32,
        /// Front-facing winding order (0 = CCW, 1 = CW).
        front_face: u32,
    },
    /// Command to compile and create a render pipeline with depth state, explicit face culling, and front-face winding state.
    CreatePipelineDepthCull {
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
        /// Explicit uniform buffer binding byte size.
        uniform_size: u32,
        /// Explicit vertex array byte stride.
        vertex_stride: u32,
        /// Depth texture format code (3 = depth24plus, 4 = depth32float).
        depth_format: u32,
        /// Whether depth writes are enabled.
        depth_write_enabled: bool,
        /// Depth comparison function code (1..=8).
        depth_compare: u32,
        /// Face culling mode (0 = None, 1 = Front, 2 = Back).
        cull_mode: u32,
        /// Front-facing winding order (0 = CCW, 1 = CW).
        front_face: u32,
    },
    /// Command to compile and create a render pipeline with depth state, explicit face culling, front-face winding state, and color write mask.
    CreatePipelineDepthCullColor {
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
        /// Explicit uniform buffer binding byte size.
        uniform_size: u32,
        /// Explicit vertex array byte stride.
        vertex_stride: u32,
        /// Depth texture format code (3 = depth24plus, 4 = depth32float).
        depth_format: u32,
        /// Whether depth writes are enabled.
        depth_write_enabled: bool,
        /// Depth comparison function code (1..=8).
        depth_compare: u32,
        /// Face culling mode (0 = None, 1 = Front, 2 = Back).
        cull_mode: u32,
        /// Front-facing winding order (0 = CCW, 1 = CW).
        front_face: u32,
        /// Color write mask (0 = None, 0xF = All, etc.).
        write_mask: u32,
    },
    /// Command to encode and execute a complete render pass with a depth attachment.
    RenderPassDepth {
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
        /// Explicit load operation for color (0 = Clear, 1 = Load, 2 = DontCare).
        load_op: u32,
        /// Explicit store operation for color (0 = Store, 1 = Discard).
        store_op: u32,
        /// Pass boundary flags (0 = none, 1 = new pass boundary).
        pass_flags: u32,
        /// Depth texture identifier.
        depth_target_id: u32,
        /// Explicit load operation for depth (0 = Clear, 1 = Load, 2 = DontCare).
        depth_load_op: u32,
        /// Explicit store operation for depth (0 = Store, 1 = Discard).
        depth_store_op: u32,
        /// Clear depth value normalized to [0.0, 1.0] (typically 1.0).
        depth_clear_value: f32,
        /// Whether depth attachment is read-only.
        depth_read_only: bool,
    },
    /// Command to upload raw pixel data into a 2D GPU texture slice.
    WriteTexture {
        texture_id: u32,
        width: u32,
        height: u32,
        bytes_per_row: u32,
        data: Vec<u8>,
    },
    /// Command to compile and create a textured render pipeline with texture and sampler bindings.
    CreatePipelineTextured {
        pipeline_id: u32,
        wgsl_code: String,
        target_format: u32,
        has_vertex_buffer: bool,
        has_uniform_buffer: bool,
        uniform_size: u32,
        vertex_stride: u32,
        texture_id: u32,
        sampler_filter: u32,
        address_mode: u32,
    },
    /// Command to copy raw bytes from one GPU buffer to another.
    CopyBufferToBuffer {
        /// Source buffer identifier.
        source_buffer_id: u32,
        /// Byte offset within the source buffer (must be multiple of 4).
        source_offset: u64,
        /// Destination buffer identifier.
        destination_buffer_id: u32,
        /// Byte offset within the destination buffer (must be multiple of 4).
        destination_offset: u64,
        /// Byte size to copy (must be multiple of 4).
        size: u64,
        /// Semantic epoch of the source buffer state region.
        epoch: Epoch,
    },
    /// Command to compile and create a compute pipeline with explicit bind group layout.
    CreateComputePipeline {
        /// Unique integer identifier for the pipeline.
        pipeline_id: u32,
        /// Complete WGSL shader source code.
        wgsl_code: String,
        /// Entry point function name (e.g. "main").
        entry_point: String,
        /// Explicit bind group 0 layout entries.
        bindings: Vec<GpuComputeBindingLayout>,
    },
    /// Command to execute a compute pass dispatching workgroups with explicit buffer bindings.
    DispatchCompute {
        /// Unique integer identifier for the compute pipeline.
        pipeline_id: u32,
        /// Number of workgroups to dispatch in the X dimension.
        workgroup_count_x: u32,
        /// Number of workgroups to dispatch in the Y dimension.
        workgroup_count_y: u32,
        /// Number of workgroups to dispatch in the Z dimension.
        workgroup_count_z: u32,
        /// Explicit buffer bindings bound to group 0.
        bindings: Vec<GpuBufferBinding>,
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
    /// Texture data length does not match `bytes_per_row * height`.
    TextureLengthMismatch { expected: usize, actual: usize },
    /// Input dimensions, parameter array lengths, or buffer sizing calculation overflow.
    InvalidDimensions(String),
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
            Self::TextureLengthMismatch { expected, actual } => {
                write!(f, "Texture data length mismatch: expected {expected} (bytes_per_row * height), got {actual}")
            }
            Self::InvalidDimensions(msg) => {
                write!(f, "Invalid dimensions or parameters: {msg}")
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
                GpuCommand::RecordBundleBatch {
                    bundle_id, pipeline_id, vertex_buffer_id, vertex_count,
                    uniform_dynamic_offset, uniform_buffer_id, target_format,
                    draw_count, dynamic_offset_stride,
                } => {
                    if *draw_count > 0 {
                        (draw_count - 1).checked_mul(*dynamic_offset_stride)
                            .and_then(|span| uniform_dynamic_offset.checked_add(span))
                            .ok_or_else(|| PacketEncodeError::InvalidDimensions(
                                "bundle dynamic offset overflow".to_string(),
                            ))?;
                    }
                    command_records.extend_from_slice(&OPCODE_RECORD_BUNDLE_BATCH.to_le_bytes());
                    for value in [bundle_id, pipeline_id, vertex_buffer_id, vertex_count,
                        uniform_dynamic_offset, uniform_buffer_id, target_format,
                        draw_count, dynamic_offset_stride] {
                        command_records.extend_from_slice(&value.to_le_bytes());
                    }
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
                GpuCommand::CreatePipelineDepth {
                    pipeline_id,
                    wgsl_code,
                    target_format,
                    has_vertex_buffer,
                    has_uniform_buffer,
                    uniform_size,
                    vertex_stride,
                    depth_format,
                    depth_write_enabled,
                    depth_compare,
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

                    command_records.extend_from_slice(&OPCODE_CREATE_PIPELINE_DEPTH.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *has_vertex_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&(if *has_uniform_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&uniform_size.to_le_bytes());
                    command_records.extend_from_slice(&vertex_stride.to_le_bytes());
                    command_records.extend_from_slice(&depth_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *depth_write_enabled { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&depth_compare.to_le_bytes());
                }
                GpuCommand::CreatePipelineCull {
                    pipeline_id,
                    wgsl_code,
                    target_format,
                    has_vertex_buffer,
                    has_uniform_buffer,
                    uniform_size,
                    vertex_stride,
                    cull_mode,
                    front_face,
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

                    command_records.extend_from_slice(&OPCODE_CREATE_PIPELINE_CULL.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *has_vertex_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&(if *has_uniform_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&uniform_size.to_le_bytes());
                    command_records.extend_from_slice(&vertex_stride.to_le_bytes());
                    command_records.extend_from_slice(&cull_mode.to_le_bytes());
                    command_records.extend_from_slice(&front_face.to_le_bytes());
                }
                GpuCommand::CreatePipelineDepthCull {
                    pipeline_id,
                    wgsl_code,
                    target_format,
                    has_vertex_buffer,
                    has_uniform_buffer,
                    uniform_size,
                    vertex_stride,
                    depth_format,
                    depth_write_enabled,
                    depth_compare,
                    cull_mode,
                    front_face,
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

                    command_records.extend_from_slice(&OPCODE_CREATE_PIPELINE_DEPTH_CULL.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *has_vertex_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&(if *has_uniform_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&uniform_size.to_le_bytes());
                    command_records.extend_from_slice(&vertex_stride.to_le_bytes());
                    command_records.extend_from_slice(&depth_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *depth_write_enabled { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&depth_compare.to_le_bytes());
                    command_records.extend_from_slice(&cull_mode.to_le_bytes());
                    command_records.extend_from_slice(&front_face.to_le_bytes());
                }
                GpuCommand::CreatePipelineDepthCullColor {
                    pipeline_id,
                    wgsl_code,
                    target_format,
                    has_vertex_buffer,
                    has_uniform_buffer,
                    uniform_size,
                    vertex_stride,
                    depth_format,
                    depth_write_enabled,
                    depth_compare,
                    cull_mode,
                    front_face,
                    write_mask,
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

                    command_records.extend_from_slice(&OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *has_vertex_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&(if *has_uniform_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&uniform_size.to_le_bytes());
                    command_records.extend_from_slice(&vertex_stride.to_le_bytes());
                    command_records.extend_from_slice(&depth_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *depth_write_enabled { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&depth_compare.to_le_bytes());
                    command_records.extend_from_slice(&cull_mode.to_le_bytes());
                    command_records.extend_from_slice(&front_face.to_le_bytes());
                    command_records.extend_from_slice(&write_mask.to_le_bytes());
                }
                GpuCommand::RenderPassDepth {
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
                    depth_target_id,
                    depth_load_op,
                    depth_store_op,
                    depth_clear_value,
                    depth_read_only,
                } => {
                    command_records.extend_from_slice(&OPCODE_RENDER_PASS_DEPTH.to_le_bytes());
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
                    command_records.extend_from_slice(&depth_target_id.to_le_bytes());
                    let packed_depth = pack_depth_ops(*depth_load_op, *depth_store_op, *depth_read_only);
                    command_records.extend_from_slice(&packed_depth.to_le_bytes());
                    command_records.extend_from_slice(&depth_clear_value.to_le_bytes());
                }
                GpuCommand::WriteTexture {
                    texture_id, width, height, bytes_per_row, data,
                } => {
                    let expected_len = (*bytes_per_row as usize).checked_mul(*height as usize).ok_or(
                        PacketEncodeError::CommandDataOverflow { command_index: cmd_idx, length: data.len() },
                    )?;
                    if data.len() != expected_len {
                        return Err(PacketEncodeError::TextureLengthMismatch { expected: expected_len, actual: data.len() });
                    }
                    let data_len = u32::try_from(data.len()).map_err(|_| PacketEncodeError::CommandDataOverflow { command_index: cmd_idx, length: data.len() })?;
                    let current_len = data_payload.len();
                    if current_len.checked_add(data.len()).map_or(true, |sum| sum > max_payload_len) {
                        return Err(PacketEncodeError::DataPayloadOverflow { offset: current_len, length: data.len() });
                    }
                    let data_offset = u32::try_from(current_len).map_err(|_| PacketEncodeError::DataPayloadOverflow { offset: current_len, length: data.len() })?;
                    data_payload.extend_from_slice(data);

                    command_records.extend_from_slice(&OPCODE_WRITE_TEXTURE.to_le_bytes());
                    command_records.extend_from_slice(&texture_id.to_le_bytes());
                    command_records.extend_from_slice(&width.to_le_bytes());
                    command_records.extend_from_slice(&height.to_le_bytes());
                    command_records.extend_from_slice(&bytes_per_row.to_le_bytes());
                    command_records.extend_from_slice(&data_offset.to_le_bytes());
                    command_records.extend_from_slice(&data_len.to_le_bytes());
                }
                GpuCommand::CreatePipelineTextured {
                    pipeline_id, wgsl_code, target_format, has_vertex_buffer, has_uniform_buffer,
                    uniform_size, vertex_stride, texture_id, sampler_filter, address_mode,
                } => {
                    let bytes = wgsl_code.as_bytes();
                    let code_len = u32::try_from(bytes.len()).map_err(|_| PacketEncodeError::CommandDataOverflow { command_index: cmd_idx, length: bytes.len() })?;
                    let current_len = data_payload.len();
                    if current_len.checked_add(bytes.len()).map_or(true, |sum| sum > max_payload_len) {
                        return Err(PacketEncodeError::DataPayloadOverflow { offset: current_len, length: bytes.len() });
                    }
                    let code_offset = u32::try_from(current_len).map_err(|_| PacketEncodeError::DataPayloadOverflow { offset: current_len, length: bytes.len() })?;
                    data_payload.extend_from_slice(bytes);

                    command_records.extend_from_slice(&OPCODE_CREATE_PIPELINE_TEXTURED.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *has_vertex_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&(if *has_uniform_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&uniform_size.to_le_bytes());
                    command_records.extend_from_slice(&vertex_stride.to_le_bytes());
                    command_records.extend_from_slice(&texture_id.to_le_bytes());
                    command_records.extend_from_slice(&sampler_filter.to_le_bytes());
                    command_records.extend_from_slice(&address_mode.to_le_bytes());
                }
                GpuCommand::CopyBufferToBuffer {
                    source_buffer_id,
                    source_offset,
                    destination_buffer_id,
                    destination_offset,
                    size,
                    epoch,
                } => {
                    if source_buffer_id == destination_buffer_id {
                        return Err(PacketEncodeError::InvalidDimensions(
                            "Source and destination buffer IDs must be distinct".to_string(),
                        ));
                    }
                    if source_offset % 4 != 0 || destination_offset % 4 != 0 || size % 4 != 0 {
                        return Err(PacketEncodeError::InvalidDimensions(
                            "Buffer-to-buffer copy offsets and size must be multiples of 4"
                                .to_string(),
                        ));
                    }
                    source_offset.checked_add(*size).ok_or_else(|| {
                        PacketEncodeError::InvalidDimensions(
                            "source offset + size overflow".to_string(),
                        )
                    })?;
                    destination_offset.checked_add(*size).ok_or_else(|| {
                        PacketEncodeError::InvalidDimensions(
                            "destination offset + size overflow".to_string(),
                        )
                    })?;
                    command_records.extend_from_slice(&OPCODE_COPY_BUFFER_TO_BUFFER.to_le_bytes());
                    command_records.extend_from_slice(&source_buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&source_offset.to_le_bytes());
                    command_records.extend_from_slice(&destination_buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&destination_offset.to_le_bytes());
                    command_records.extend_from_slice(&size.to_le_bytes());
                    command_records.extend_from_slice(&epoch.get().to_le_bytes());
                }
                GpuCommand::CreateComputePipeline {
                    pipeline_id,
                    wgsl_code,
                    entry_point,
                    bindings,
                } => {
                    let code_bytes = wgsl_code.as_bytes();
                    let code_len = u32::try_from(code_bytes.len()).map_err(|_| {
                        PacketEncodeError::CommandDataOverflow {
                            command_index: cmd_idx,
                            length: code_bytes.len(),
                        }
                    })?;
                    let ep_bytes = entry_point.as_bytes();
                    let ep_len = u32::try_from(ep_bytes.len()).map_err(|_| {
                        PacketEncodeError::CommandDataOverflow {
                            command_index: cmd_idx,
                            length: ep_bytes.len(),
                        }
                    })?;

                    let current_len = data_payload.len();
                    let needed_payload = code_bytes.len().checked_add(ep_bytes.len()).ok_or_else(|| {
                        PacketEncodeError::DataPayloadOverflow {
                            offset: current_len,
                            length: code_bytes.len(),
                        }
                    })?;
                    if current_len.checked_add(needed_payload).map_or(true, |sum| sum > max_payload_len) {
                        return Err(PacketEncodeError::DataPayloadOverflow {
                            offset: current_len,
                            length: needed_payload,
                        });
                    }

                    let code_offset = u32::try_from(current_len).map_err(|_| {
                        PacketEncodeError::DataPayloadOverflow {
                            offset: current_len,
                            length: code_bytes.len(),
                        }
                    })?;
                    data_payload.extend_from_slice(code_bytes);

                    let ep_offset = u32::try_from(data_payload.len()).map_err(|_| {
                        PacketEncodeError::DataPayloadOverflow {
                            offset: data_payload.len(),
                            length: ep_bytes.len(),
                        }
                    })?;
                    data_payload.extend_from_slice(ep_bytes);

                    let binding_count = u32::try_from(bindings.len()).map_err(|_| {
                        PacketEncodeError::InvalidDimensions("binding count exceeds u32::MAX".to_string())
                    })?;

                    for (i, b) in bindings.iter().enumerate() {
                        if b.binding_type > BINDING_TYPE_STORAGE_READ_WRITE {
                            return Err(PacketEncodeError::InvalidDimensions(format!(
                                "Invalid binding_type {} at binding index {}",
                                b.binding_type, b.binding_index
                            )));
                        }
                        for other in &bindings[i + 1..] {
                            if b.binding_index == other.binding_index {
                                return Err(PacketEncodeError::InvalidDimensions(format!(
                                    "Duplicate binding index {} in CreateComputePipeline",
                                    b.binding_index
                                )));
                            }
                        }
                    }

                    command_records.extend_from_slice(&OPCODE_CREATE_COMPUTE_PIPELINE.to_le_bytes());
                    command_records.extend_from_slice(&0u16.to_le_bytes()); // _pad0
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&ep_offset.to_le_bytes());
                    command_records.extend_from_slice(&ep_len.to_le_bytes());
                    command_records.extend_from_slice(&binding_count.to_le_bytes());
                    command_records.extend_from_slice(&0u32.to_le_bytes()); // _pad1

                    for b in bindings {
                        command_records.extend_from_slice(&b.binding_index.to_le_bytes());
                        command_records.extend_from_slice(&b.binding_type.to_le_bytes());
                        command_records.extend_from_slice(&b.min_binding_size.to_le_bytes());
                        command_records.extend_from_slice(&0u32.to_le_bytes()); // _pad
                    }
                }
                GpuCommand::DispatchCompute {
                    pipeline_id,
                    workgroup_count_x,
                    workgroup_count_y,
                    workgroup_count_z,
                    bindings,
                } => {
                    let binding_count = u32::try_from(bindings.len()).map_err(|_| {
                        PacketEncodeError::InvalidDimensions("binding count exceeds u32::MAX".to_string())
                    })?;

                    for (i, b) in bindings.iter().enumerate() {
                        if b.buffer_id == 0 {
                            return Err(PacketEncodeError::InvalidDimensions(format!(
                                "buffer_id must be non-zero for binding {}",
                                b.binding_index
                            )));
                        }
                        if b.binding_type > BINDING_TYPE_STORAGE_READ_WRITE {
                            return Err(PacketEncodeError::InvalidDimensions(format!(
                                "Invalid binding_type {} for binding {}",
                                b.binding_type, b.binding_index
                            )));
                        }
                        if b.size == 0 {
                            return Err(PacketEncodeError::InvalidDimensions(format!(
                                "Binding {} size must be non-zero",
                                b.binding_index
                            )));
                        }
                        if b.binding_type != BINDING_TYPE_UNIFORM && b.size % 4 != 0 {
                            return Err(PacketEncodeError::InvalidDimensions(format!(
                                "Storage binding {} size {} must be a multiple of 4",
                                b.binding_index, b.size
                            )));
                        }
                        b.offset.checked_add(b.size).ok_or_else(|| {
                            PacketEncodeError::InvalidDimensions(format!(
                                "Binding {} offset + size overflow",
                                b.binding_index
                            ))
                        })?;
                        for other in &bindings[i + 1..] {
                            if b.binding_index == other.binding_index {
                                return Err(PacketEncodeError::InvalidDimensions(format!(
                                    "Duplicate binding index {} in DispatchCompute",
                                    b.binding_index
                                )));
                            }
                        }
                    }

                    command_records.extend_from_slice(&OPCODE_DISPATCH_COMPUTE.to_le_bytes());
                    command_records.extend_from_slice(&0u16.to_le_bytes()); // _pad0
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&workgroup_count_x.to_le_bytes());
                    command_records.extend_from_slice(&workgroup_count_y.to_le_bytes());
                    command_records.extend_from_slice(&workgroup_count_z.to_le_bytes());
                    command_records.extend_from_slice(&binding_count.to_le_bytes());

                    for b in bindings {
                        command_records.extend_from_slice(&b.binding_index.to_le_bytes());
                        command_records.extend_from_slice(&b.buffer_id.to_le_bytes());
                        command_records.extend_from_slice(&b.offset.to_le_bytes());
                        command_records.extend_from_slice(&b.size.to_le_bytes());
                        command_records.extend_from_slice(&b.binding_type.to_le_bytes());
                        command_records.extend_from_slice(&0u32.to_le_bytes()); // _pad
                        command_records.extend_from_slice(&b.epoch.get().to_le_bytes());
                        command_records.extend_from_slice(&b.data_version.get().to_le_bytes());
                    }
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
    /// Render segment configures a stencil operation (stencil not yet supported by bridge lowering).
    UnsupportedStencilAttachment {
        /// Diagnostic name of the pass segment.
        segment_name: String,
    },
    /// Render segment configures render bundle draws with a depth attachment (bundle depth not yet supported by bridge lowering).
    UnsupportedBundleDepthAttachment {
        /// Diagnostic name of the pass segment.
        segment_name: String,
    },
    /// Render segment configures an invalid or malformed depth attachment.
    InvalidDepthAttachment {
        /// Diagnostic name of the pass segment.
        segment_name: String,
        /// Diagnostic reason describing why the depth attachment is malformed.
        reason: String,
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
    /// Execution plan contains an invalid copy command.
    InvalidCopyCommand {
        /// Diagnostic reason describing why the copy command is invalid.
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
            Self::UnsupportedStencilAttachment { segment_name } => {
                write!(
                    f,
                    "Render segment '{segment_name}' configures a stencil operation; stencil is not supported by bridge lowering"
                )
            }
            Self::UnsupportedBundleDepthAttachment { segment_name } => {
                write!(
                    f,
                    "Render segment '{segment_name}' configures render bundle draws with a depth attachment; bundle depth is not supported by bridge lowering"
                )
            }
            Self::InvalidDepthAttachment { segment_name, reason } => {
                write!(
                    f,
                    "Render segment '{segment_name}' configures an invalid depth attachment: {reason}"
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
            Self::InvalidCopyCommand { reason } => {
                write!(
                    f,
                    "Invalid copy command during bridge lowering: {reason}"
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
                let depth_attachment = segment.depth_stencil_attachment();
                if let Some(dsa) = depth_attachment {
                    if dsa.target_id().get() == 0 {
                        return Err(PlanLoweringError::InvalidDepthAttachment {
                            segment_name: segment.name().to_string(),
                            reason: "Depth attachment target ID must not be zero".to_string(),
                        });
                    }
                    if dsa.depth_clear_value().is_nan()
                        || dsa.depth_clear_value() < 0.0
                        || dsa.depth_clear_value() > 1.0
                    {
                        return Err(PlanLoweringError::InvalidDepthAttachment {
                            segment_name: segment.name().to_string(),
                            reason: alloc::format!(
                                "Depth clear value must be within [0.0, 1.0], got {}",
                                dsa.depth_clear_value()
                            ),
                        });
                    }
                    if dsa.depth_read_only() && dsa.depth_load_op() == Some(LoadOp::Clear) {
                        return Err(PlanLoweringError::InvalidDepthAttachment {
                            segment_name: segment.name().to_string(),
                            reason: "Read-only depth attachment cannot specify LoadOp::Clear"
                                .to_string(),
                        });
                    }
                    if !dsa.depth_read_only() && dsa.depth_load_op().is_none() {
                        return Err(PlanLoweringError::InvalidDepthAttachment {
                            segment_name: segment.name().to_string(),
                            reason: "Writable depth attachment requires explicit depth_load_op"
                                .to_string(),
                        });
                    }
                    if !dsa.depth_read_only() && dsa.depth_store_op().is_none() {
                        return Err(PlanLoweringError::InvalidDepthAttachment {
                            segment_name: segment.name().to_string(),
                            reason: "Writable depth attachment requires explicit depth_store_op"
                                .to_string(),
                        });
                    }
                    if dsa.stencil_load_op.is_some()
                        || dsa.stencil_store_op.is_some()
                        || !dsa.stencil_read_only
                        || dsa.stencil_clear_value != 0
                    {
                        return Err(PlanLoweringError::UnsupportedStencilAttachment {
                            segment_name: segment.name().to_string(),
                        });
                    }
                    if segment.draws().iter().any(|d| matches!(d.kind(), DrawKind::Bundle { .. })) {
                        return Err(PlanLoweringError::UnsupportedBundleDepthAttachment {
                            segment_name: segment.name().to_string(),
                        });
                    }
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

                let (depth_target_id, depth_load_op, depth_store_op, depth_clear_value, depth_read_only) =
                    if let Some(dsa) = depth_attachment {
                        let d_read_only = dsa.depth_read_only();
                        let d_load = match dsa.depth_load_op() {
                            Some(LoadOp::Clear) => LOAD_OP_CLEAR,
                            Some(LoadOp::Load) => LOAD_OP_LOAD,
                            Some(LoadOp::DontCare) => LOAD_OP_DONT_CARE,
                            None => {
                                if d_read_only {
                                    LOAD_OP_LOAD
                                } else {
                                    LOAD_OP_CLEAR
                                }
                            }
                        };
                        let d_store = match dsa.depth_store_op() {
                            Some(StoreOp::Store) => STORE_OP_STORE,
                            Some(StoreOp::Discard) => STORE_OP_DISCARD,
                            None => STORE_OP_STORE,
                        };
                        (
                            dsa.target_id().get(),
                            d_load,
                            d_store,
                            dsa.depth_clear_value(),
                            d_read_only,
                        )
                    } else {
                        (0, LOAD_OP_CLEAR, STORE_OP_STORE, 1.0f32, false)
                    };

                let mut emit_render_pass = |commands: &mut Vec<GpuCommand>,
                                            pipeline_id: u32,
                                            vertex_buffer_id: u32,
                                            vertex_count: u32,
                                            uniform_dynamic_offset: u32,
                                            uniform_buffer_id: u32,
                                            pass_flags: u32| {
                    if depth_attachment.is_some() {
                        commands.push(GpuCommand::RenderPassDepth {
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
                            depth_target_id,
                            depth_load_op,
                            depth_store_op,
                            depth_clear_value,
                            depth_read_only,
                        });
                    } else {
                        commands.push(GpuCommand::RenderPass {
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
                        });
                    }
                };

                if segment.draws().is_empty() {
                    let has_color_clear = ca.load_op() == LoadOp::Clear;
                    let has_depth_clear = depth_attachment
                        .and_then(|d| d.depth_load_op())
                        .map_or(false, |op| op == LoadOp::Clear);
                    if has_color_clear || has_depth_clear {
                        emit_render_pass(&mut commands, 0, 0, 0, 0, 0, PASS_FLAG_NEW_PASS);
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
                // pass opening command with vertex_count: 0 to open the render pass on the target
                // with its clear color and load/store semantics. Subsequent bundle executes and direct draws
                // occur inside this same pass without re-clearing.
                if matches!(segment.draws().first().map(|d| d.kind()), Some(DrawKind::Bundle { .. })) {
                    emit_render_pass(&mut commands, 0, 0, 0, 0, 0, PASS_FLAG_NEW_PASS);
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
                        emit_render_pass(&mut commands, 0, 0, 0, 0, 0, PASS_FLAG_NEW_PASS);
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

                            emit_render_pass(
                                &mut commands,
                                draw.pipeline_id(),
                                draw.vertex_buffer_id(),
                                draw.vertex_count(),
                                draw.uniform_dynamic_offset(),
                                uniform_buffer_id,
                                pass_flags,
                            );
                        }
                    }
                }
            }
            PassKind::Copy => {
                for copy in segment.copies() {
                    match copy {
                        CopyCommand::TextureToBuffer {
                            texture_id,
                            buffer_id,
                            width,
                            height,
                            ..
                        } => {
                            commands.push(GpuCommand::CopyTextureToBuffer {
                                texture_id: texture_id.get(),
                                buffer_id: buffer_id.get(),
                                width: *width,
                                height: *height,
                                epoch: plan.canvas_epoch().unwrap_or(Epoch::ZERO),
                            });
                        }
                        CopyCommand::BufferToBuffer {
                            src,
                            src_offset,
                            dst,
                            dst_offset,
                            size,
                        } => {
                            if src == dst {
                                return Err(PlanLoweringError::InvalidCopyCommand {
                                    reason: format!(
                                        "Source buffer {} and destination buffer {} must be distinct",
                                        src.get(),
                                        dst.get()
                                    ),
                                });
                            }
                            if src_offset % 4 != 0 {
                                return Err(PlanLoweringError::InvalidCopyCommand {
                                    reason: format!(
                                        "Source offset {src_offset} must be a multiple of 4"
                                    ),
                                });
                            }
                            if dst_offset % 4 != 0 {
                                return Err(PlanLoweringError::InvalidCopyCommand {
                                    reason: format!(
                                        "Destination offset {dst_offset} must be a multiple of 4"
                                    ),
                                });
                            }
                            if size % 4 != 0 {
                                return Err(PlanLoweringError::InvalidCopyCommand {
                                    reason: format!("Copy size {size} must be a multiple of 4"),
                                });
                            }
                            src_offset.checked_add(*size).ok_or_else(|| {
                                PlanLoweringError::InvalidCopyCommand {
                                    reason: format!(
                                        "Source offset {src_offset} + size {size} overflow"
                                    ),
                                }
                            })?;
                            dst_offset.checked_add(*size).ok_or_else(|| {
                                PlanLoweringError::InvalidCopyCommand {
                                    reason: format!(
                                        "Destination offset {dst_offset} + size {size} overflow"
                                    ),
                                }
                            })?;
                            commands.push(GpuCommand::CopyBufferToBuffer {
                                source_buffer_id: src.get(),
                                source_offset: *src_offset,
                                destination_buffer_id: dst.get(),
                                destination_offset: *dst_offset,
                                size: *size,
                                epoch: plan.canvas_epoch().unwrap_or(Epoch::ZERO),
                            });
                        }
                        other => {
                            return Err(PlanLoweringError::UnsupportedCopyCommand {
                                reason: format!("Unsupported copy command variant: {other:?}"),
                            });
                        }
                    }
                }
            }
            // PassKind::Compute lowering from high-level ExecutionPlan awaits explicit WGSL
            // binding index lowering in the next slice (do not infer binding indices from resource IDs;
            // packet-level compute execution is supported directly via GpuCommand::CreateComputePipeline
            // and GpuCommand::DispatchCompute).
            PassKind::Compute => {
                return Err(PlanLoweringError::UnsupportedPassKind {
                    segment_name: segment.name().to_string(),
                });
            }
        }
    }

    Ok(commands)
}

fn triangle_vertex_bytes() -> Vec<u8> {
    let vertices = [
        VertexPosUv::new([0.0, 0.5, 0.0], [0.5, 1.0]),
        VertexPosUv::new([-0.5, -0.5, 0.0], [0.0, 0.0]),
        VertexPosUv::new([0.5, -0.5, 0.0], [1.0, 0.0]),
    ];
    let mut vertex_bytes = Vec::with_capacity(vertices.len() * VertexPosUv::BYTE_SIZE);
    for v in &vertices {
        vertex_bytes.extend_from_slice(&v.to_bytes());
    }
    vertex_bytes
}

fn triangle_wgsl_source() -> String {
    format!(
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
    )
}

/// Builds a real WGSL triangle submission packet using `f3d_core::layout::AffineRows`
/// and `f3d_core::layout::WGSL_AFFINE_ROWS_DECLARATION`.
pub fn build_triangle_submission() -> GpuSubmissionPacket {
    // Register resource IDs in the generational slot table (§6.5, vqa.6)
    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vertex_buffer_id
        table.register(20);  // readback_buffer_id
        table.register(10);  // target_texture_id
        table.register(100); // offscreen pipeline_id
        table.register(101); // canvas pipeline_id
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
    let vertex_bytes = triangle_vertex_bytes();
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
    let wgsl_source = triangle_wgsl_source();

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

/// Packs a flat slice of affine transform rows (`12 * N` f32 values) into 256-byte aligned uniform buffer bytes.
///
/// Each 12-float transform occupies 48 bytes followed by 208 zero-padding bytes.
/// Returns an owned [`Vec<u8>`] without holding active linear memory borrows.
pub fn pack_affine_rows_uniform_bytes(affine_rows: &[f32]) -> Result<Vec<u8>, PacketEncodeError> {
    if affine_rows.is_empty() || affine_rows.len() % 12 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "affine_rows length {} must be non-empty and a multiple of 12",
            affine_rows.len()
        )));
    }
    let n_draws = u32::try_from(affine_rows.len() / 12).map_err(|_| {
        PacketEncodeError::InvalidDimensions("transform count exceeds u32::MAX".to_string())
    })?;
    let uniform_buffer_size = n_draws.checked_mul(256).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("uniform buffer size overflow".to_string())
    })? as usize;

    let mut uniform_bytes = vec![0u8; uniform_buffer_size];
    for (i, chunk) in affine_rows.chunks_exact(12).enumerate() {
        let offset = i * 256;
        for (j, &val) in chunk.iter().enumerate() {
            uniform_bytes[offset + j * 4..offset + j * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    Ok(uniform_bytes)
}

/// Packs a flat slice of affine transform rows (`12 * N` f32 values) into packed 48-byte storage buffer records.
///
/// Reuses [`AffineRows::to_bytes`] for each 12-float record, packing records contiguously
/// into an exact `48 * N` byte buffer (no 256-byte uniform padding).
/// Preserves exact bit patterns including `-0.0` and NaN payloads.
pub fn pack_affine_rows_storage_bytes(affine_rows: &[f32]) -> Result<Vec<u8>, PacketEncodeError> {
    if affine_rows.is_empty() || affine_rows.len() % 12 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "affine_rows length {} must be non-empty and a multiple of 12",
            affine_rows.len()
        )));
    }
    let n_records = u32::try_from(affine_rows.len() / 12).map_err(|_| {
        PacketEncodeError::InvalidDimensions("transform count exceeds u32::MAX".to_string())
    })?;
    let storage_buffer_size = n_records.checked_mul(48).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("storage buffer size overflow".to_string())
    })? as usize;

    let mut storage_bytes = Vec::with_capacity(storage_buffer_size);
    for chunk in affine_rows.chunks_exact(12) {
        let affine = AffineRows::new(
            [chunk[0], chunk[1], chunk[2], chunk[3]],
            [chunk[4], chunk[5], chunk[6], chunk[7]],
            [chunk[8], chunk[9], chunk[10], chunk[11]],
        );
        storage_bytes.extend_from_slice(&affine.to_bytes());
    }
    Ok(storage_bytes)
}

/// Builds a [`GpuSubmissionPacket`] initializing persistent buffers for AffineRows storage uploads
/// (workload b, bead f3d-03-reference-profiles-and-bridge-tmt.5).
///
/// Allocates outside the repeated frame measurement:
/// 1. Source buffer 610 (`COPY_SRC | COPY_DST`) of size `48 * N`.
/// 2. Destination slot 611 (`MAP_READ | COPY_DST`) of size `48 * N`.
/// 3. Destination slot 612 (`MAP_READ | COPY_DST`) of size `48 * N`.
pub fn build_affine_rows_storage_upload_init_submission(
    expected_records: u32,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if expected_records == 0 {
        return Err(PacketEncodeError::InvalidDimensions(
            "expected_records must be greater than zero".to_string(),
        ));
    }
    let buffer_size = expected_records.checked_mul(48).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("storage upload buffer size overflow".to_string())
    })?;

    with_global_resource_table(|table| {
        table.register(AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
        table.register(AFFINE_ROWS_STORAGE_DST_SLOT_0);
        table.register(AFFINE_ROWS_STORAGE_DST_SLOT_1);
    });

    let mut packet = GpuSubmissionPacket::new();
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: AFFINE_ROWS_STORAGE_SRC_BUFFER_ID,
        size: buffer_size,
        usage: BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: AFFINE_ROWS_STORAGE_DST_SLOT_0,
        size: buffer_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: AFFINE_ROWS_STORAGE_DST_SLOT_1,
        size: buffer_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });
    Ok(packet)
}

/// Builds a [`GpuSubmissionPacket`] for a single frame update in the persistent AffineRows storage upload loop
/// (workload b, bead f3d-03-reference-profiles-and-bridge-tmt.5).
///
/// Invariants:
/// - Reuses [`pack_affine_rows_storage_bytes`] for bit-exact float layout preservation (-0.0, NaN payloads).
/// - No buffer creation or re-creation per frame (reallocates nothing).
/// - Emits exactly two commands:
///   1. `GpuCommand::WriteBuffer` writing packed `48 * N` bytes to source buffer 610 at offset 0.
///   2. `GpuCommand::CopyBufferToBuffer` copying `48 * N` bytes from buffer 610 to selected destination slot (611 or 612).
pub fn build_affine_rows_storage_upload_frame_submission(
    affine_rows: &[f32],
    expected_records: u32,
    destination_slot: u32,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if expected_records == 0 {
        return Err(PacketEncodeError::InvalidDimensions(
            "expected_records must be greater than zero".to_string(),
        ));
    }
    if destination_slot != AFFINE_ROWS_STORAGE_DST_SLOT_0
        && destination_slot != AFFINE_ROWS_STORAGE_DST_SLOT_1
    {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "destination_slot {destination_slot} must be {} or {}",
            AFFINE_ROWS_STORAGE_DST_SLOT_0, AFFINE_ROWS_STORAGE_DST_SLOT_1
        )));
    }
    let expected_floats = (expected_records as usize)
        .checked_mul(12)
        .ok_or_else(|| {
            PacketEncodeError::InvalidDimensions("expected_records * 12 overflow".to_string())
        })?;
    if affine_rows.len() != expected_floats {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "record count mismatch: expected {expected_records} records ({expected_floats} floats), got {} floats",
            affine_rows.len()
        )));
    }

    let storage_bytes = pack_affine_rows_storage_bytes(affine_rows)?;
    let total_bytes = storage_bytes.len() as u64;

    let mut packet = GpuSubmissionPacket::new();
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: AFFINE_ROWS_STORAGE_SRC_BUFFER_ID,
        offset: 0,
        data: storage_bytes,
    });
    packet.push(GpuCommand::CopyBufferToBuffer {
        source_buffer_id: AFFINE_ROWS_STORAGE_SRC_BUFFER_ID,
        source_offset: 0,
        destination_buffer_id: destination_slot,
        destination_offset: 0,
        size: total_bytes,
        epoch: Epoch::ZERO,
    });
    Ok(packet)
}

/// Individual callback operation emitted by the storage upload callback planner (workload b / tmt.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageUploadCallbackOp {
    /// Host `f3dHost.writeBuffer(buffer_id, offset, bytes)` callback.
    WriteBuffer {
        /// Target buffer identifier (e.g. source buffer 610).
        buffer_id: u32,
        /// Byte offset within target buffer.
        offset: u32,
        /// Owned packed byte payload.
        bytes: Vec<u8>,
    },
    /// Host `f3dHost.copyBufferToBuffer(src, src_off, dst, dst_off, size)` callback.
    CopyBufferToBuffer {
        /// Source buffer identifier (e.g. source buffer 610).
        src: u32,
        /// Byte offset within source buffer (Number in JS).
        src_off: u32,
        /// Destination buffer identifier (e.g. destination slot 611 or 612).
        dst: u32,
        /// Byte offset within destination buffer (Number in JS).
        dst_off: u32,
        /// Number of bytes to copy (Number in JS).
        size: u32,
    },
}

/// Maps a storage upload frame [`GpuSubmissionPacket`] to an ordered list of callback operations (workload b / tmt.5).
///
/// Converts:
/// - `GpuCommand::WriteBuffer` to `writeBuffer(buffer_id, offset, bytes)`
/// - `GpuCommand::CopyBufferToBuffer` to `copyBufferToBuffer(src, src_off, dst, dst_off, size)`
pub fn map_storage_upload_submission_to_callback_ops(
    packet: GpuSubmissionPacket,
) -> Result<Vec<StorageUploadCallbackOp>, PacketEncodeError> {
    let mut ops = Vec::with_capacity(packet.commands().len());
    for cmd in packet.into_commands() {
        match cmd {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                ops.push(StorageUploadCallbackOp::WriteBuffer {
                    buffer_id,
                    offset,
                    bytes: data,
                });
            }
            GpuCommand::CopyBufferToBuffer {
                source_buffer_id,
                source_offset,
                destination_buffer_id,
                destination_offset,
                size,
                ..
            } => {
                let src_off = u32::try_from(source_offset).map_err(|_| {
                    PacketEncodeError::InvalidDimensions("source_offset exceeds u32".to_string())
                })?;
                let dst_off = u32::try_from(destination_offset).map_err(|_| {
                    PacketEncodeError::InvalidDimensions("destination_offset exceeds u32".to_string())
                })?;
                let copy_size = u32::try_from(size).map_err(|_| {
                    PacketEncodeError::InvalidDimensions("copy size exceeds u32".to_string())
                })?;
                ops.push(StorageUploadCallbackOp::CopyBufferToBuffer {
                    src: source_buffer_id,
                    src_off,
                    dst: destination_buffer_id,
                    dst_off,
                    size: copy_size,
                });
            }
            other => {
                return Err(PacketEncodeError::InvalidDimensions(format!(
                    "unexpected command in storage upload frame: {other:?}"
                )));
            }
        }
    }
    Ok(ops)
}

/// Pure helper that maps [`build_affine_rows_storage_upload_frame_submission`] to an ordered callback op list (workload b / tmt.5).
///
/// Guards and packed bytes are identical to the bulk frame submission, ensuring the GPU work matches by construction.
pub fn build_affine_rows_storage_upload_callback_ops(
    affine_rows: &[f32],
    expected_records: u32,
    destination_slot: u32,
) -> Result<Vec<StorageUploadCallbackOp>, PacketEncodeError> {
    let packet = build_affine_rows_storage_upload_frame_submission(
        affine_rows,
        expected_records,
        destination_slot,
    )?;
    map_storage_upload_submission_to_callback_ops(packet)
}

/// Builds an offscreen submission packet drawing `N` triangles with `N` distinct
/// AffineRows transforms at 256-byte dynamic offsets in a single offscreen render pass.
///
/// Uses the canonical triangle UV gradient shader and geometry from `build_triangle_submission`.
pub fn build_affine_rows_batch_submission(
    affine_rows: &[f32],
    width: u32,
    height: u32,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if width == 0 || height == 0 {
        return Err(PacketEncodeError::InvalidDimensions(
            "width and height must be non-zero".to_string(),
        ));
    }

    let uniform_bytes = pack_affine_rows_uniform_bytes(affine_rows)?;
    let uniform_buffer_size = u32::try_from(uniform_bytes.len()).map_err(|_| {
        PacketEncodeError::InvalidDimensions("uniform buffer size exceeds u32::MAX".to_string())
    })?;
    let n_draws = (affine_rows.len() / 12) as u32;

    let unpadded_bytes_per_row = width.checked_mul(4).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("width * 4 overflow".to_string())
    })?;
    let bytes_per_row = unpadded_bytes_per_row
        .checked_add(255)
        .map(|v| (v / 256) * 256)
        .ok_or_else(|| {
            PacketEncodeError::InvalidDimensions("bytes_per_row alignment overflow".to_string())
        })?;
    let readback_size = bytes_per_row.checked_mul(height).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("readback size overflow".to_string())
    })?;

    with_global_resource_table(|table| {
        table.register(1);   // uniform_buffer_id
        table.register(2);   // vertex_buffer_id
        table.register(10);  // target_texture_id
        table.register(20);  // readback_buffer_id
        table.register(100); // pipeline_id
    });

    let mut packet = GpuSubmissionPacket::new();
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: 1,
        size: uniform_buffer_size,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: 1,
        offset: 0,
        data: uniform_bytes,
    });

    let vertex_bytes = triangle_vertex_bytes();
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: 2,
        size: vertex_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: 2,
        offset: 0,
        data: vertex_bytes,
    });

    packet.push(GpuCommand::CreateTexture {
        texture_id: 10,
        width,
        height,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: 20,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: 100,
        wgsl_code: triangle_wgsl_source(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: AFFINE_ROWS_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    push_affine_rows_batch_draw_and_readback_commands(&mut packet, n_draws, width, height);

    Ok(packet)
}

#[inline]
fn push_affine_rows_batch_draw_and_readback_commands(
    packet: &mut GpuSubmissionPacket,
    n_draws: u32,
    width: u32,
    height: u32,
) {
    for i in 0..n_draws {
        packet.push(GpuCommand::RenderPass {
            target_type: TARGET_OFFSCREEN,
            target_id: 10,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            pipeline_id: 100,
            vertex_buffer_id: 2,
            vertex_count: 3,
            uniform_dynamic_offset: i * 256,
            uniform_buffer_id: 1,
            load_op: if i == 0 { LOAD_OP_CLEAR } else { LOAD_OP_LOAD },
            store_op: STORE_OP_STORE,
            pass_flags: if i == 0 { PASS_FLAG_NEW_PASS } else { PASS_FLAG_NONE },
        });
    }

    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: 10,
        buffer_id: 20,
        width,
        height,
        epoch: Epoch::ZERO,
    });
}

/// Builds a [`GpuSubmissionPacket`] for updating an already initialized AffineRows scene.
///
/// Emits only:
/// 1. `WriteBuffer` to uniform buffer 1 (offset 0) with updated packed affine transforms.
/// 2. `N` per-draw `RenderPass` commands reusing existing pipeline 100, vertex buffer 2,
///    uniform buffer 1, and offscreen target 10.
/// 3. `CopyTextureToBuffer` from target texture 10 to readback buffer 20.
///
/// Note: `expected_draws` and dimensions correspond to the caller's previously initialized scene shape.
/// This function cannot verify actual GPU allocation capacity and does not claim to; the caller
/// must execute this packet on the same isolated `GpuBridge` instance where resources 1, 2, 10, 20,
/// and 100 were previously created without shape or device changes.
/// Performs no resource creation or global generational table registration.
pub fn build_affine_rows_batch_frame_submission(
    affine_rows: &[f32],
    width: u32,
    height: u32,
    expected_draws: u32,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    let uniform_bytes = affine_rows_frame_bytes(affine_rows, width, height, expected_draws)?;

    let mut packet = GpuSubmissionPacket::new();
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: 1,
        offset: 0,
        data: uniform_bytes,
    });

    push_affine_rows_batch_draw_and_readback_commands(&mut packet, expected_draws, width, height);

    Ok(packet)
}

fn affine_rows_frame_bytes(
    affine_rows: &[f32], width: u32, height: u32, expected_draws: u32,
) -> Result<Vec<u8>, PacketEncodeError> {
    if width == 0 || height == 0 {
        return Err(PacketEncodeError::InvalidDimensions(
            "width and height must be non-zero".to_string(),
        ));
    }
    if expected_draws == 0 {
        return Err(PacketEncodeError::InvalidDimensions(
            "expected_draws must be greater than zero".to_string(),
        ));
    }
    if affine_rows.is_empty() || affine_rows.len() % 12 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "affine_rows length {} must be non-empty and a multiple of 12",
            affine_rows.len()
        )));
    }
    let n_draws = u32::try_from(affine_rows.len() / 12).map_err(|_| {
        PacketEncodeError::InvalidDimensions("transform count exceeds u32::MAX".to_string())
    })?;
    if n_draws != expected_draws {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "draw count mismatch: expected {expected_draws} draws, got {n_draws}"
        )));
    }

    let unpadded_bytes_per_row = width.checked_mul(4).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("width * 4 overflow".to_string())
    })?;
    let bytes_per_row = unpadded_bytes_per_row
        .checked_add(255)
        .map(|v| (v / 256) * 256)
        .ok_or_else(|| {
            PacketEncodeError::InvalidDimensions("bytes_per_row alignment overflow".to_string())
        })?;
    let _readback_size = bytes_per_row.checked_mul(height).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("readback size overflow".to_string())
    })?;

    pack_affine_rows_uniform_bytes(affine_rows)
}

/// Update an initialized affine scene, replay one bundle, then issue a direct tail.
///
/// Uses the same fixed resources and caller-owned shape/device contract as
/// [`build_affine_rows_batch_frame_submission`]. With `record_bundle`, records
/// bundle 200 once; otherwise its pipeline, buffers, draw count and offsets must
/// still match the previous recording. Updating buffer contents does not rerecord it.
pub fn build_affine_rows_bundle_submission(
    affine_rows: &[f32], width: u32, height: u32, expected_draws: u32,
    tail_draws: u32, record_bundle: bool,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if tail_draws == 0 || tail_draws >= expected_draws {
        return Err(PacketEncodeError::InvalidDimensions(
            "bundle replay needs a nonempty prefix and direct tail".to_string(),
        ));
    }
    let data = affine_rows_frame_bytes(affine_rows, width, height, expected_draws)?;
    let prefix = expected_draws - tail_draws;
    let mut packet = GpuSubmissionPacket::new();
    packet.push(GpuCommand::WriteBuffer { buffer_id: 1, offset: 0, data });
    if record_bundle {
        packet.push(GpuCommand::RecordBundleBatch {
            bundle_id: 200, pipeline_id: 100, vertex_buffer_id: 2, vertex_count: 3,
            uniform_dynamic_offset: 0, uniform_buffer_id: 1,
            target_format: TARGET_FORMAT_RGBA8UNORM,
            draw_count: prefix, dynamic_offset_stride: 256,
        });
    }
    // Open and clear the actual target before replay; zero vertices issue no draw.
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN, target_id: 10, clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id: 100, vertex_buffer_id: 2, vertex_count: 0,
        uniform_dynamic_offset: 0, uniform_buffer_id: 1,
        load_op: LOAD_OP_CLEAR, store_op: STORE_OP_STORE, pass_flags: PASS_FLAG_NEW_PASS,
    });
    packet.push(GpuCommand::ExecuteBundles { bundle_ids: vec![200] });
    for index in prefix..expected_draws {
        packet.push(GpuCommand::RenderPass {
            target_type: TARGET_OFFSCREEN, target_id: 10, clear_color: [0.0, 0.0, 0.0, 1.0],
            pipeline_id: 100, vertex_buffer_id: 2, vertex_count: 3,
            uniform_dynamic_offset: index * 256, uniform_buffer_id: 1,
            load_op: LOAD_OP_LOAD, store_op: STORE_OP_STORE, pass_flags: PASS_FLAG_NONE,
        });
    }
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: 10, buffer_id: 20, width, height, epoch: Epoch::ZERO,
    });
    Ok(packet)
}

/// Builds a [`GpuSubmissionPacket`] for buffer-to-buffer copy testing and host verification.
///
/// Emits:
/// 1. `CreateBuffer` for source buffer 610 (`COPY_SRC | COPY_DST`) with size `data.len()`.
/// 2. `WriteBuffer` uploading `data` to buffer 610 at offset 0.
/// 3. `CreateBuffer` for destination buffer 611 (`MAP_READ | COPY_DST`) with size `data.len()`.
/// 4. `CopyBufferToBuffer` (opcode 20) copying `size` bytes from `610` at `source_offset`
///    to `611` at `destination_offset` with `Epoch::ZERO`.
pub fn build_buffer_copy_submission(
    data: &[u8],
    source_offset: u32,
    destination_offset: u32,
    size: u32,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if data.is_empty() {
        return Err(PacketEncodeError::InvalidDimensions(
            "data must be non-empty".to_string(),
        ));
    }
    if data.len() % 4 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "data length {} must be a multiple of 4",
            data.len()
        )));
    }
    let buffer_size = u32::try_from(data.len()).map_err(|_| {
        PacketEncodeError::InvalidDimensions("data length exceeds u32::MAX".to_string())
    })?;

    if source_offset % 4 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "source_offset {source_offset} must be a multiple of 4"
        )));
    }
    if destination_offset % 4 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "destination_offset {destination_offset} must be a multiple of 4"
        )));
    }
    if size % 4 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "size {size} must be a multiple of 4"
        )));
    }

    let src_end = source_offset.checked_add(size).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("source_offset + size overflow".to_string())
    })?;
    if src_end > buffer_size {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "source_offset + size ({src_end}) exceeds buffer size ({buffer_size})"
        )));
    }

    let dst_end = destination_offset.checked_add(size).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("destination_offset + size overflow".to_string())
    })?;
    if dst_end > buffer_size {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "destination_offset + size ({dst_end}) exceeds buffer size ({buffer_size})"
        )));
    }

    with_global_resource_table(|table| {
        table.register(610);
        table.register(611);
    });

    let mut packet = GpuSubmissionPacket::new();
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: 610,
        size: buffer_size,
        usage: BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: 610,
        offset: 0,
        data: data.to_vec(),
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: 611,
        size: buffer_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::CopyBufferToBuffer {
        source_buffer_id: 610,
        source_offset: source_offset as u64,
        destination_buffer_id: 611,
        destination_offset: destination_offset as u64,
        size: size as u64,
        epoch: Epoch::ZERO,
    });

    Ok(packet)
}

/// Builds a submission packet opening a zero-draw canvas render pass immediately followed by
/// an opcode 20 buffer-to-buffer copy.
///
/// Reuses `build_buffer_copy_submission` and inserts a zero-draw canvas `RenderPass`
/// directly before the trailing `CopyBufferToBuffer` command.
pub fn build_render_then_copy_submission(
    data: &[u8],
    source_offset: u32,
    destination_offset: u32,
    size: u32,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    let mut packet = build_buffer_copy_submission(data, source_offset, destination_offset, size)?;
    let insert_idx = packet.commands.len().saturating_sub(1);
    packet.commands.insert(
        insert_idx,
        GpuCommand::RenderPass {
            target_type: TARGET_CANVAS,
            target_id: 0,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            pipeline_id: 0,
            vertex_buffer_id: 0,
            vertex_count: 0,
            uniform_dynamic_offset: 0,
            uniform_buffer_id: 0,
            load_op: LOAD_OP_CLEAR,
            store_op: STORE_OP_STORE,
            pass_flags: PASS_FLAG_NEW_PASS,
        },
    );
    Ok(packet)
}

/// Builds a [`GpuSubmissionPacket`] executing a real compute shader that transforms input points
/// with packed AffineRows storage and copies transformed points to a map-readable staging buffer.
///
/// Supported input contracts:
/// - Points: Packed 4-float vectors (`[x, y, z, w]`). Multiples of 3 or non-multiples of 4 are rejected.
/// - Affine transforms: Either 1 transform (broadcast to all points) or exactly 1 transform per point.
///   All other cardinalities are rejected.
///
/// Execution pipeline:
/// 1. Registers resource IDs in the generational slot table:
///    - Buffer 701: AffineRows storage (`STORAGE | COPY_DST`)
///    - Buffer 702: Input points storage (`STORAGE | COPY_DST`)
///    - Buffer 703: Output points storage (`STORAGE | COPY_SRC`)
///    - Buffer 704: Readback staging buffer (`MAP_READ | COPY_DST`)
///    - Pipeline 300: Compute pipeline
/// 2. Creates and populates storage buffers via `WriteBuffer`.
/// 3. Compiles compute pipeline with explicit bindings:
///    - Binding 0: Read-only AffineRows storage (`array<AffineRows>`)
///    - Binding 1: Read-only Input points storage (`array<vec4<f32>>`)
///    - Binding 2: Read-write Output points storage (`array<vec4<f32>>`)
///    Embeds canonical [`WGSL_AFFINE_ROWS_DECLARATION`] from schema (§6.1, vqa.2).
/// 4. Dispatches compute pass with workgroup size (64, 1, 1).
/// 5. Copies output storage buffer 703 to readback staging buffer 704 via `CopyBufferToBuffer` (opcode 20).
pub fn build_affine_rows_compute_submission(
    affine_rows: &[f32],
    points: &[f32],
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if points.is_empty() || points.len() % 4 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "points length {} must be non-empty and a multiple of 4 (packed [x, y, z, w] vec4 floats)",
            points.len()
        )));
    }
    let num_points = points.len() / 4;
    let mut point_bytes = Vec::with_capacity(points.len() * 4);
    for &val in points {
        point_bytes.extend_from_slice(&val.to_le_bytes());
    }

    let affine_bytes = pack_affine_rows_storage_bytes(affine_rows)?;
    let num_transforms = affine_rows.len() / 12;
    if num_transforms != 1 && num_transforms != num_points {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "affine transform count ({num_transforms}) must be either 1 (broadcast to all points) or match point count ({num_points})"
        )));
    }
    let affine_size = affine_bytes.len() as u32;
    let points_size = u32::try_from(point_bytes.len()).map_err(|_| {
        PacketEncodeError::InvalidDimensions("points byte length exceeds u32::MAX".to_string())
    })?;

    with_global_resource_table(|table| {
        table.register(COMPUTE_AFFINE_BUFFER_ID);
        table.register(COMPUTE_INPUT_POINTS_BUFFER_ID);
        table.register(COMPUTE_OUTPUT_POINTS_BUFFER_ID);
        table.register(COMPUTE_READBACK_BUFFER_ID);
        table.register(COMPUTE_PIPELINE_ID);
    });

    let compute_shader = [
        WGSL_AFFINE_ROWS_DECLARATION,
        "\n\
@group(0) @binding(0)\n\
var<storage, read> affine_transforms: array<AffineRows>;\n\
\n\
@group(0) @binding(1)\n\
var<storage, read> in_points: array<vec4<f32>>;\n\
\n\
@group(0) @binding(2)\n\
var<storage, read_write> out_points: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n\
    let idx = global_id.x;\n\
    if (idx >= arrayLength(&in_points)) {\n\
        return;\n\
    }\n\
    let p = in_points[idx].xyz;\n\
    var transform_idx = 0u;\n\
    if (arrayLength(&affine_transforms) > 1u) {\n\
        transform_idx = idx;\n\
    }\n\
    let transformed = transform_affine_point(affine_transforms[transform_idx], p);\n\
    out_points[idx] = vec4<f32>(transformed, 1.0);\n\
}\n",
    ]
    .concat();

    let mut packet = GpuSubmissionPacket::new();

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_AFFINE_BUFFER_ID,
        size: affine_size,
        usage: BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: COMPUTE_AFFINE_BUFFER_ID,
        offset: 0,
        data: affine_bytes,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_INPUT_POINTS_BUFFER_ID,
        size: points_size,
        usage: BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: COMPUTE_INPUT_POINTS_BUFFER_ID,
        offset: 0,
        data: point_bytes,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_OUTPUT_POINTS_BUFFER_ID,
        size: points_size,
        usage: BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_READBACK_BUFFER_ID,
        size: points_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::CreateComputePipeline {
        pipeline_id: COMPUTE_PIPELINE_ID,
        wgsl_code: compute_shader,
        entry_point: "main".to_string(),
        bindings: alloc::vec![
            GpuComputeBindingLayout {
                binding_index: 0,
                binding_type: BINDING_TYPE_STORAGE_READ,
                min_binding_size: 48,
            },
            GpuComputeBindingLayout {
                binding_index: 1,
                binding_type: BINDING_TYPE_STORAGE_READ,
                min_binding_size: 16,
            },
            GpuComputeBindingLayout {
                binding_index: 2,
                binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                min_binding_size: 16,
            },
        ],
    });

    let workgroup_count_x = ((num_points as u32).saturating_add(63)) / 64;
    packet.push(GpuCommand::DispatchCompute {
        pipeline_id: COMPUTE_PIPELINE_ID,
        workgroup_count_x: workgroup_count_x.max(1),
        workgroup_count_y: 1,
        workgroup_count_z: 1,
        bindings: alloc::vec![
            GpuBufferBinding {
                binding_index: 0,
                buffer_id: COMPUTE_AFFINE_BUFFER_ID,
                offset: 0,
                size: affine_size as u64,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            },
            GpuBufferBinding {
                binding_index: 1,
                buffer_id: COMPUTE_INPUT_POINTS_BUFFER_ID,
                offset: 0,
                size: points_size as u64,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            },
            GpuBufferBinding {
                binding_index: 2,
                buffer_id: COMPUTE_OUTPUT_POINTS_BUFFER_ID,
                offset: 0,
                size: points_size as u64,
                binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            },
        ],
    });

    packet.push(GpuCommand::CopyBufferToBuffer {
        source_buffer_id: COMPUTE_OUTPUT_POINTS_BUFFER_ID,
        source_offset: 0,
        destination_buffer_id: COMPUTE_READBACK_BUFFER_ID,
        destination_offset: 0,
        size: points_size as u64,
        epoch: Epoch::ZERO,
    });

    Ok(packet)
}

/// Builds a [`GpuSubmissionPacket`] with two sequential compute dispatches demonstrating
/// independent per-use buffer slices or deliberate input aliasing.
///
/// Execution pipeline:
/// 1. Registers resources in the generational slot table:
///    - Buffer 701: AffineRows storage (512 bytes: Slice A at 0..256, Slice B at 256..512)
///    - Buffer 702: Input points storage
///    - Buffer 703: Output points storage (2 * slice_stride bytes: Slice A at 0, Slice B at slice_stride)
///    - Buffer 704: Readback staging buffer (2 * slice_stride bytes)
///    - Pipeline 300: Compute pipeline
/// 2. Dispatches pass 1 (Matrix A): binds Slice A (offset 0), version 1.
/// 3. Dispatches pass 2 (Matrix B or aliased Matrix A):
///    - If `aliased`: binds Slice A (offset 0), version 1 (hazard counterexample).
///    - If not `aliased`: binds Slice B (offset 256), version 2 (independent per-use slice).
/// 4. Copies both output slices from Buffer 703 to Readback Buffer 704 via `CopyBufferToBuffer` (opcode 20).
pub fn build_two_dispatch_affine_compute_submission(
    matrix_a: &[f32],
    matrix_b: &[f32],
    points: &[f32],
    aliased: bool,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if matrix_a.len() != 12 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "matrix_a length {} must be exactly 12 floats",
            matrix_a.len()
        )));
    }
    if matrix_b.len() != 12 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "matrix_b length {} must be exactly 12 floats",
            matrix_b.len()
        )));
    }
    if points.is_empty() || points.len() % 4 != 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "points length {} must be non-empty and a multiple of 4 (packed [x, y, z, w] vec4 floats)",
            points.len()
        )));
    }

    let num_points = points.len() / 4;
    let points_byte_len = u32::try_from(points.len() * 4).map_err(|_| {
        PacketEncodeError::InvalidDimensions("points byte length exceeds u32::MAX".to_string())
    })?;
    let output_slice_stride = ((points_byte_len.saturating_add(255)) / 256) * 256;
    let total_output_size = output_slice_stride.checked_mul(2).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("output buffer size overflow".to_string())
    })?;

    let packed_a = pack_affine_rows_storage_bytes(matrix_a)?;
    let packed_b = pack_affine_rows_storage_bytes(matrix_b)?;
    let mut affine_storage = Vec::with_capacity(512);
    affine_storage.extend_from_slice(&packed_a);
    affine_storage.resize(256, 0);
    affine_storage.extend_from_slice(&packed_b);
    affine_storage.resize(512, 0);

    let mut point_bytes = Vec::with_capacity(points.len() * 4);
    for &val in points {
        point_bytes.extend_from_slice(&val.to_le_bytes());
    }

    with_global_resource_table(|table| {
        table.register(COMPUTE_AFFINE_BUFFER_ID);
        table.register(COMPUTE_INPUT_POINTS_BUFFER_ID);
        table.register(COMPUTE_OUTPUT_POINTS_BUFFER_ID);
        table.register(COMPUTE_READBACK_BUFFER_ID);
        table.register(COMPUTE_PIPELINE_ID);
    });

    let compute_shader = [
        WGSL_AFFINE_ROWS_DECLARATION,
        "\n\
@group(0) @binding(0)\n\
var<storage, read> affine_transforms: array<AffineRows>;\n\
\n\
@group(0) @binding(1)\n\
var<storage, read> in_points: array<vec4<f32>>;\n\
\n\
@group(0) @binding(2)\n\
var<storage, read_write> out_points: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n\
    let idx = global_id.x;\n\
    if (idx >= arrayLength(&in_points)) {\n\
        return;\n\
    }\n\
    let p = in_points[idx].xyz;\n\
    let transformed = transform_affine_point(affine_transforms[0], p);\n\
    out_points[idx] = vec4<f32>(transformed, 1.0);\n\
}\n",
    ]
    .concat();

    let mut packet = GpuSubmissionPacket::new();

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_AFFINE_BUFFER_ID,
        size: 512,
        usage: BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: COMPUTE_AFFINE_BUFFER_ID,
        offset: 0,
        data: affine_storage,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_INPUT_POINTS_BUFFER_ID,
        size: points_byte_len,
        usage: BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: COMPUTE_INPUT_POINTS_BUFFER_ID,
        offset: 0,
        data: point_bytes,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_OUTPUT_POINTS_BUFFER_ID,
        size: total_output_size,
        usage: BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: COMPUTE_READBACK_BUFFER_ID,
        size: total_output_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::CreateComputePipeline {
        pipeline_id: COMPUTE_PIPELINE_ID,
        wgsl_code: compute_shader,
        entry_point: "main".to_string(),
        bindings: alloc::vec![
            GpuComputeBindingLayout {
                binding_index: 0,
                binding_type: BINDING_TYPE_STORAGE_READ,
                min_binding_size: 48,
            },
            GpuComputeBindingLayout {
                binding_index: 1,
                binding_type: BINDING_TYPE_STORAGE_READ,
                min_binding_size: 16,
            },
            GpuComputeBindingLayout {
                binding_index: 2,
                binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                min_binding_size: 16,
            },
        ],
    });

    let workgroup_count_x = ((num_points as u32).saturating_add(63)) / 64;

    // Dispatch 1: binds Slice A (offset 0, DataVersion 1) -> Output Slice A (offset 0)
    packet.push(GpuCommand::DispatchCompute {
        pipeline_id: COMPUTE_PIPELINE_ID,
        workgroup_count_x: workgroup_count_x.max(1),
        workgroup_count_y: 1,
        workgroup_count_z: 1,
        bindings: alloc::vec![
            GpuBufferBinding {
                binding_index: 0,
                buffer_id: COMPUTE_AFFINE_BUFFER_ID,
                offset: 0,
                size: 48,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            },
            GpuBufferBinding {
                binding_index: 1,
                buffer_id: COMPUTE_INPUT_POINTS_BUFFER_ID,
                offset: 0,
                size: points_byte_len as u64,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            },
            GpuBufferBinding {
                binding_index: 2,
                buffer_id: COMPUTE_OUTPUT_POINTS_BUFFER_ID,
                offset: 0,
                size: points_byte_len as u64,
                binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            },
        ],
    });

    // Dispatch 2: binds Slice B (offset 256, DataVersion 2), or Slice A if aliased
    let (matrix_offset, data_version) = if aliased {
        (0u64, DataVersion::new(1))
    } else {
        (256u64, DataVersion::new(2))
    };

    packet.push(GpuCommand::DispatchCompute {
        pipeline_id: COMPUTE_PIPELINE_ID,
        workgroup_count_x: workgroup_count_x.max(1),
        workgroup_count_y: 1,
        workgroup_count_z: 1,
        bindings: alloc::vec![
            GpuBufferBinding {
                binding_index: 0,
                buffer_id: COMPUTE_AFFINE_BUFFER_ID,
                offset: matrix_offset,
                size: 48,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::ZERO,
                data_version,
            },
            GpuBufferBinding {
                binding_index: 1,
                buffer_id: COMPUTE_INPUT_POINTS_BUFFER_ID,
                offset: 0,
                size: points_byte_len as u64,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::ZERO,
                data_version,
            },
            GpuBufferBinding {
                binding_index: 2,
                buffer_id: COMPUTE_OUTPUT_POINTS_BUFFER_ID,
                offset: output_slice_stride as u64,
                size: points_byte_len as u64,
                binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                epoch: Epoch::ZERO,
                data_version,
            },
        ],
    });

    // Copy both output slices to readback staging buffer
    packet.push(GpuCommand::CopyBufferToBuffer {
        source_buffer_id: COMPUTE_OUTPUT_POINTS_BUFFER_ID,
        source_offset: 0,
        destination_buffer_id: COMPUTE_READBACK_BUFFER_ID,
        destination_offset: 0,
        size: total_output_size as u64,
        epoch: Epoch::ZERO,
    });

    Ok(packet)
}

/// Independent mathematical CPU oracle for affine point transformation (test-local).
///
/// Computes `P' = (dot(r0, [P.xyz, 1]), dot(r1, [P.xyz, 1]), dot(r2, [P.xyz, 1]))`.
#[cfg(test)]
#[must_use]
pub(crate) fn cpu_transform_affine_point(affine_12: &[f32], point_4: &[f32]) -> [f32; 4] {
    assert_eq!(affine_12.len(), 12, "affine_12 must have exactly 12 floats");
    assert_eq!(point_4.len(), 4, "point_4 must have exactly 4 floats [x, y, z, w]");
    let x = point_4[0];
    let y = point_4[1];
    let z = point_4[2];
    let tx = affine_12[0] * x + affine_12[1] * y + affine_12[2] * z + affine_12[3];
    let ty = affine_12[4] * x + affine_12[5] * y + affine_12[6] * z + affine_12[7];
    let tz = affine_12[8] * x + affine_12[9] * y + affine_12[10] * z + affine_12[11];
    [tx, ty, tz, 1.0]
}


/// Builds a textured WGSL triangle submission packet using `f3d_core::layout::AffineRows`
/// and sampling from a 2D texture (bead vqa.6).
pub fn build_textured_affine_triangle_packet(
    pixels: &[u8],
    tex_w: u32,
    tex_h: u32,
    affine: &AffineRows,
    target_w: u32,
    target_h: u32,
) -> Result<GpuSubmissionPacket, PacketEncodeError> {
    if tex_w == 0 || tex_h == 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "texture dimensions must be non-zero: {tex_w}x{tex_h}"
        )));
    }
    if target_w == 0 || target_h == 0 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "target dimensions must be non-zero: {target_w}x{target_h}"
        )));
    }

    let tex_bytes_per_row = tex_w.checked_mul(4).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("texture row pitch overflow (tex_w * 4)".to_string())
    })?;
    let expected_pixel_len = (tex_bytes_per_row as usize)
        .checked_mul(tex_h as usize)
        .ok_or_else(|| {
            PacketEncodeError::InvalidDimensions("texture data size overflow (bytes_per_row * tex_h)".to_string())
        })?;
    if pixels.len() != expected_pixel_len {
        return Err(PacketEncodeError::TextureLengthMismatch {
            expected: expected_pixel_len,
            actual: pixels.len(),
        });
    }

    let unpadded_target_row = target_w.checked_mul(4).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("target row unpadded size overflow (target_w * 4)".to_string())
    })?;
    let target_bytes_per_row = ((unpadded_target_row.checked_add(255).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("target row alignment padding overflow".to_string())
    })?) / 256)
        .checked_mul(256)
        .ok_or_else(|| {
            PacketEncodeError::InvalidDimensions("target aligned row bytes overflow".to_string())
        })?;
    let readback_size = target_bytes_per_row.checked_mul(target_h).ok_or_else(|| {
        PacketEncodeError::InvalidDimensions("target readback size overflow (bytes_per_row * target_h)".to_string())
    })?;

    let uniform_buffer_id = 1;
    let vertex_buffer_id = 2;
    let target_texture_id = 10;
    let source_texture_id = 11;
    let readback_buffer_id = 20;
    let pipeline_id = 100;

    with_global_resource_table(|table| {
        for id in [
            uniform_buffer_id,
            vertex_buffer_id,
            target_texture_id,
            source_texture_id,
            readback_buffer_id,
            pipeline_id,
        ] {
            table.register(id);
        }
    });

    let mut packet = GpuSubmissionPacket::new();

    // 1. Uniform buffer: AffineRows transform
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT as u32,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: uniform_buffer_id,
        offset: 0,
        data: affine.to_bytes().to_vec(),
    });

    // 2. Vertex buffer: 3 vertices matching existing triangle builder (VertexPosUv)
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

    // 3. Target texture and readback buffer (target_w x target_h)
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_texture_id,
        width: target_w,
        height: target_h,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // 4. Source texture to sample from and pixel upload via WriteTexture
    packet.push(GpuCommand::CreateTexture {
        texture_id: source_texture_id,
        width: tex_w,
        height: tex_h,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_TEXTURE_BINDING | TEXTURE_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteTexture {
        texture_id: source_texture_id,
        width: tex_w,
        height: tex_h,
        bytes_per_row: tex_bytes_per_row,
        data: pixels.to_vec(),
    });

    // 5. WGSL Shader with AffineRows transform and texture sampling
    let wgsl_source = format!(
        "{WGSL_AFFINE_ROWS_DECLARATION}\n\
@group(0) @binding(0) var<uniform> model: AffineRows;\n\
@group(0) @binding(1) var tex: texture_2d<f32>;\n\
@group(0) @binding(2) var samp: sampler;\n\
struct VertexInput {{ @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, }};\n\
struct VertexOutput {{ @builtin(position) clip_pos: vec4<f32>, @location(0) uv: vec2<f32>, }};\n\
@vertex fn vs_main(in: VertexInput) -> VertexOutput {{\n\
    var out: VertexOutput;\n\
    let transformed = transform_affine_point(model, in.position);\n\
    out.clip_pos = vec4<f32>(transformed, 1.0);\n\
    out.uv = in.uv;\n\
    return out;\n\
}}\n\
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{\n\
    return textureSample(tex, samp, in.uv);\n\
}}\n"
    );

    // 6. Textured pipeline for offscreen target (rgba8unorm)
    packet.push(GpuCommand::CreatePipelineTextured {
        pipeline_id,
        wgsl_code: wgsl_source,
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: AFFINE_ROWS_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
        texture_id: source_texture_id,
        sampler_filter: SAMPLER_FILTER_NEAREST,
        address_mode: ADDRESS_MODE_CLAMP_TO_EDGE,
    });

    // 7. Render pass to offscreen target
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

    // 8. Copy offscreen texture to readback buffer
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_texture_id,
        buffer_id: readback_buffer_id,
        width: target_w,
        height: target_h,
        epoch: Epoch::ZERO,
    });

    Ok(packet)
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

    let flat_shader = [
        WGSL_COLOR_UNIFORM_DECLARATION,
        "\n\
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
    return u.rgba;\n\
}\n",
    ]
    .concat();

    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: 200,
        wgsl_code: flat_shader,
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
    let flat_shader = [
        WGSL_COLOR_UNIFORM_DECLARATION,
        "\n\
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
    return u.rgba;\n\
}\n",
    ]
    .concat();

    let pipeline_id = 200;
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: flat_shader,
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

/// Builds a bundle-then-direct-draw submission packet that warms the decoder's
/// `passState` cache before executing bundles (§6.7, §8.2, vqa.7).
///
/// Unlike [`build_bundle_then_direct_draw_submission`], which only issues direct draw
/// after executing bundles, this submission executes a direct draw of Triangle 2
/// *before* [`GpuCommand::ExecuteBundles`] (filling the decoder cache), then executes
/// bundles, and then executes the identical direct draw again.
///
/// Sequence:
/// 1. Create uniform buffer 1 (Green at 0, Blue at 256) and upload colors.
/// 2. Create vertex buffer 2 (Triangle 1, left) and upload vertices.
/// 3. Create vertex buffer 3 (Triangle 2, right) and upload vertices.
/// 4. Create 64x64 target texture 10 and readback buffer 20.
/// 5. Create flat color pipeline 200 (`VertexPosUv` layout, `ColorUniform`).
/// 6. Record RenderBundle 1 (Triangle 1, Green at offset 0, 3 vertices).
/// 7. RenderPass opener: clear black, vertex_count 0, `PASS_FLAG_NEW_PASS`.
/// 8. Direct draw of Triangle 2 before bundle: pipeline 200, uniform 1 offset 256, vb 3,
///    vertex_count 3, `PASS_FLAG_NONE`. This fills the decoder cache.
/// 9. Execute bundles: `[1]` if `!empty_bundle_list`, else `[]`.
/// 10. Identical direct draw again: same pipeline, uniform, offset, vb, vertex_count and `PASS_FLAG_NONE`.
/// 11. CopyTextureToBuffer, texture 10 -> buffer 20, 64x64.
pub fn build_bundle_then_direct_warm_cache_submission(empty_bundle_list: bool) -> GpuSubmissionPacket {
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
    let flat_shader = [
        WGSL_COLOR_UNIFORM_DECLARATION,
        "\n\
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
    return u.rgba;\n\
}\n",
    ]
    .concat();

    let pipeline_id = 200;
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: flat_shader,
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

    // 7. RenderPass opener: clear black, vertex_count 0, PASS_FLAG_NEW_PASS
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

    // 8. Direct draw of the right triangle BEFORE any bundle: pipeline 200, uniform 1 offset 256,
    // vb 3, vertex_count 3, PASS_FLAG_NONE (same fields as the existing step-10 draw). This fills the decoder cache.
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

    // 9. ExecuteBundles([1]) when empty_bundle_list=false; ExecuteBundles([]) when true.
    let bundle_ids = if empty_bundle_list {
        alloc::vec![]
    } else {
        alloc::vec![bundle_id]
    };
    packet.push(GpuCommand::ExecuteBundles { bundle_ids });

    // 10. The identical direct draw again: same pipeline, uniform, offset, vb, vertex_count and PASS_FLAG_NONE.
    // A decoder that skipped its reset would believe that state is still bound.
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

    // 11. CopyTextureToBuffer, texture 10 -> buffer 20, 64x64.
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
    let affine_shader = [
        WGSL_AFFINE_ROWS_DECLARATION,
        "\n\
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
}\n",
    ]
    .concat();

    let pipeline_id = 200;
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: affine_shader,
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

/// Builds an AffineRows layout counterexample submission packet (§6.1, §6.2, vqa.7).
///
/// Demonstrates the silent layout corruption between packed [`AffineRows`] (48 bytes,
/// 3 × `vec4<f32>` rows with translation in the 4th component of each row) and standard
/// WGSL `mat4x3<f32>` (64 bytes, 4 column vectors of `vec3<f32>` each padded to 16 bytes).
///
/// When `wrong_mat4x3_layout` is false (correct AffineRows):
/// - Pipeline uses the canonical [`AffineRows`] struct with row dot-products.
/// - `uniform_size` is 48 ([`AFFINE_ROWS_BYTES`]).
/// - The triangle is scaled by 0.5 and translated by (+0.5, 0, 0), landing at screen
///   center `(48, 32)` (green) with `(32, 32)` outside (black).
///
/// When `wrong_mat4x3_layout` is true (silent mat4x3 corruption):
/// - Pipeline declares `var<uniform> transform: mat4x3<f32>;` and evaluates
///   `transform * vec4<f32>(in.position, 1.0)`.
/// - `uniform_size` is 64 bytes.
/// - The uniform buffer payload is byte-for-byte identical (48 packed AffineRows bytes
///   padded to 256 bytes). Because `mat4x3<f32>` reads columns with 16-byte alignment,
///   the x-translation at byte offset 12 falls into column 0 padding, and column 3
///   (translation) reads the trailing zeros at offset 48..60. Translation is silently lost:
///   the triangle is centered at `(32, 32)` (green) with `(48, 32)` outside (black).
///
/// Only the WGSL code and `uniform_size` differ between the two variants; uniform bytes
/// and all other command parameters are identical.
#[must_use]
pub fn build_affine_rows_layout_counterexample_submission(
    wrong_mat4x3_layout: bool,
) -> GpuSubmissionPacket {
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

    // 2. Uniform buffer with 48-byte AffineRows record at offset 0 (identical bytes for both variants)
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

    // 5. WGSL Shader and uniform_size: only these two fields differ between variants (§6.1, CONTRACT.md)
    let (wgsl_code, uniform_size) = if wrong_mat4x3_layout {
        (
            "\
@group(0) @binding(0)\n\
var<uniform> transform: mat4x3<f32>;\n\
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
    let transformed = transform * vec4<f32>(in.position, 1.0);\n\
    out.clip_position = vec4<f32>(transformed, 1.0);\n\
    out.uv = in.uv;\n\
    return out;\n\
}\n\
\n\
@fragment\n\
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {\n\
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);\n\
}\n"
                .to_string(),
            64u32,
        )
    } else {
        (
            [
                WGSL_AFFINE_ROWS_DECLARATION,
                "\n\
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
}\n",
            ]
            .concat(),
            AFFINE_ROWS_BYTES as u32,
        )
    };

    let pipeline_id = 200;
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code,
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size,
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

    let flat_shader = [
        WGSL_COLOR_UNIFORM_DECLARATION,
        "\n\
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
    return u.rgba;\n\
}\n",
    ]
    .concat();

    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: flat_shader,
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

    let flat_shader = [
        WGSL_COLOR_UNIFORM_DECLARATION,
        "\n\
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
    return u.rgba;\n\
}\n",
    ]
    .concat();

    // Command 7: Pipeline 200 for offscreen target (RGBA8Unorm)
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: pipeline_offscreen,
        wgsl_code: flat_shader.clone(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: true,
        uniform_size: COLOR_UNIFORM_BYTES as u32,
        vertex_stride: VERTEX_POS_UV_STRIDE as u32,
    });

    // Command 8: Pipeline 201 for canvas presentation target (dynamically matches host preferredCanvasFormat)
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id: pipeline_canvas,
        wgsl_code: flat_shader,
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

    let flat_shader = [
        WGSL_COLOR_UNIFORM_DECLARATION,
        "\n\
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
    return u.rgba;\n\
}\n",
    ]
    .concat();

    // Command 9: Pipeline 200
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: flat_shader,
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
/// Encodes a bundle-then-direct warm-cache submission packet and returns the raw binary bytes (§6.7, §8.2, vqa.7).
pub fn f3d_build_bundle_direct_warm_cache_packet(empty_bundle_list: bool) -> Vec<u8> {
    build_bundle_then_direct_warm_cache_submission(empty_bundle_list)
        .encode()
        .expect("static bundle-direct-warm-cache packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_bundle_direct_warm_cache_packet` for host verification and unit tests (§6.7, §8.2, vqa.7).
#[must_use]
pub fn f3d_build_bundle_direct_warm_cache_packet(empty_bundle_list: bool) -> Vec<u8> {
    build_bundle_then_direct_warm_cache_submission(empty_bundle_list)
        .encode()
        .expect("static bundle-direct-warm-cache packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes an AffineRows layout counterexample submission packet and returns the raw binary bytes (§6.1, §6.2, vqa.7).
pub fn f3d_build_affine_rows_layout_counterexample_packet(wrong_mat4x3_layout: bool) -> Vec<u8> {
    build_affine_rows_layout_counterexample_submission(wrong_mat4x3_layout)
        .encode()
        .expect("static affine-rows layout counterexample packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_layout_counterexample_packet` for host verification and unit tests (§6.1, §6.2, vqa.7).
#[must_use]
pub fn f3d_build_affine_rows_layout_counterexample_packet(wrong_mat4x3_layout: bool) -> Vec<u8> {
    build_affine_rows_layout_counterexample_submission(wrong_mat4x3_layout)
        .encode()
        .expect("static affine-rows layout counterexample packet encoding must not fail")
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

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a batch AffineRows transform submission packet and returns the raw binary bytes.
pub fn f3d_build_affine_rows_batch_packet(
    affine_rows: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_affine_rows_batch_submission(affine_rows, width, height)
        .and_then(|p| p.encode())
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes an update packet for an already initialized AffineRows batch scene (wasm-bindgen export).
pub fn f3d_build_affine_rows_batch_frame_packet(
    affine_rows: &[f32],
    width: u32,
    height: u32,
    expected_draws: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_affine_rows_batch_frame_submission(affine_rows, width, height, expected_draws)
        .and_then(|p| p.encode())
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes an update packet for an already initialized AffineRows batch scene into the static borrowed packet slot (wasm-bindgen export).
///
/// Returns `[ptr, len]` as a 2-element array pointing into WebAssembly linear memory.
pub fn f3d_build_affine_rows_batch_frame_packet_borrowed(
    affine_rows: &[f32],
    width: u32,
    height: u32,
    expected_draws: u32,
) -> Result<Vec<u32>, wasm_bindgen::JsValue> {
    build_affine_rows_batch_frame_packet_borrowed(affine_rows, width, height, expected_draws)
        .map(|arr| arr.to_vec())
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Returns the linear memory address (pointer) of the static borrowed frame packet buffer.
pub fn f3d_borrowed_frame_packet_ptr() -> u32 {
    borrowed_frame_packet_ptr()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Returns the byte length of the static borrowed frame packet buffer.
pub fn f3d_borrowed_frame_packet_len() -> u32 {
    borrowed_frame_packet_len()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = f3dHost, js_name = drawCall, catch)]
    fn host_draw_call(draw_index: u32) -> Result<(), JsValue>;

    #[wasm_bindgen(js_namespace = f3dHost, js_name = writeBuffer, catch)]
    fn host_write_buffer(buffer_id: u32, offset: u32, bytes: &js_sys::Uint8Array) -> Result<(), JsValue>;

    #[wasm_bindgen(js_namespace = f3dHost, js_name = copyBufferToBuffer, catch)]
    fn host_copy_buffer_to_buffer(
        src: u32,
        src_off: u32,
        dst: u32,
        dst_off: u32,
        size: u32,
    ) -> Result<(), JsValue>;
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encode an affine bundle replay frame under the initialized scene's fixed resource contract.
pub fn f3d_build_affine_rows_bundle_packet(
    affine_rows: &[f32], width: u32, height: u32, expected_draws: u32,
    tail_draws: u32, record_bundle: bool,
) -> Result<Vec<u8>, JsValue> {
    build_affine_rows_bundle_submission(
        affine_rows, width, height, expected_draws, tail_draws, record_bundle,
    ).and_then(|packet| packet.encode()).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native entry point for the same bundle replay packet encoder used by Wasm.
pub fn f3d_build_affine_rows_bundle_packet(
    affine_rows: &[f32], width: u32, height: u32, expected_draws: u32,
    tail_draws: u32, record_bundle: bool,
) -> Result<Vec<u8>, String> {
    build_affine_rows_bundle_submission(
        affine_rows, width, height, expected_draws, tail_draws, record_bundle,
    ).and_then(|packet| packet.encode()).map_err(|e| e.to_string())
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a buffer-to-buffer copy submission packet and returns the raw binary bytes.
pub fn f3d_build_buffer_copy_packet(
    data: &[u8],
    source_offset: u32,
    destination_offset: u32,
    size: u32,
) -> Result<Vec<u8>, JsValue> {
    build_buffer_copy_submission(data, source_offset, destination_offset, size)
        .and_then(|packet| packet.encode())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_buffer_copy_packet` for host execution and unit tests.
pub fn f3d_build_buffer_copy_packet(
    data: &[u8],
    source_offset: u32,
    destination_offset: u32,
    size: u32,
) -> Result<Vec<u8>, String> {
    build_buffer_copy_submission(data, source_offset, destination_offset, size)
        .and_then(|packet| packet.encode())
        .map_err(|e| e.to_string())
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Canonical export: builds a submission packet with an active canvas render pass followed by a buffer copy.
pub fn f3d_build_render_then_copy_packet(
    data: &[u8],
    source_offset: u32,
    destination_offset: u32,
    size: u32,
) -> Result<Vec<u8>, JsValue> {
    build_render_then_copy_submission(data, source_offset, destination_offset, size)
        .and_then(|packet| packet.encode())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_render_then_copy_packet` for host execution and unit tests.
pub fn f3d_build_render_then_copy_packet(
    data: &[u8],
    source_offset: u32,
    destination_offset: u32,
    size: u32,
) -> Result<Vec<u8>, String> {
    build_render_then_copy_submission(data, source_offset, destination_offset, size)
        .and_then(|packet| packet.encode())
        .map_err(|e| e.to_string())
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Canonical Wasm export: builds an AffineRows compute submission packet transforming points in GPU storage.
pub fn f3d_build_affine_rows_compute_packet(
    affine_rows: &[f32],
    points: &[f32],
) -> Result<Vec<u8>, JsValue> {
    build_affine_rows_compute_submission(affine_rows, points)
        .and_then(|packet| packet.encode())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_compute_packet` for host verification and unit tests.
pub fn f3d_build_affine_rows_compute_packet(
    affine_rows: &[f32],
    points: &[f32],
) -> Result<Vec<u8>, String> {
    build_affine_rows_compute_submission(affine_rows, points)
        .and_then(|packet| packet.encode())
        .map_err(|e| e.to_string())
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Canonical Wasm export: builds a two-dispatch AffineRows compute packet for browser counterexample & verification.
pub fn f3d_build_two_dispatch_affine_compute_packet(
    matrix_a: &[f32],
    matrix_b: &[f32],
    points: &[f32],
    aliased: bool,
) -> Result<Vec<u8>, JsValue> {
    build_two_dispatch_affine_compute_submission(matrix_a, matrix_b, points, aliased)
        .and_then(|packet| packet.encode())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_two_dispatch_affine_compute_packet` for host verification and unit tests.
pub fn f3d_build_two_dispatch_affine_compute_packet(
    matrix_a: &[f32],
    matrix_b: &[f32],
    points: &[f32],
    aliased: bool,
) -> Result<Vec<u8>, String> {
    build_two_dispatch_affine_compute_submission(matrix_a, matrix_b, points, aliased)
        .and_then(|packet| packet.encode())
        .map_err(|e| e.to_string())
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Synchronously loops `count` iterations invoking the imported host callback `f3dHost.drawCall(index)` (Kill Gate 2 / tmt.5).
///
/// Enforces:
/// - Zero live linear memory slices across host callback invocation.
/// - Zero RefCell borrows active during host call.
/// - Zero global table / mutex locks held across the host boundary.
/// - Immediate propagation of thrown host errors to stop iteration.
pub fn f3d_bridge_chatty_draw_loop(count: u32) -> Result<u32, JsValue> {
    for i in 0..count {
        host_draw_call(i)?;
    }
    Ok(count)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Issues persistent AffineRows storage upload frame via host callbacks (workload b / tmt.5).
///
/// Calls [`build_affine_rows_storage_upload_callback_ops`] and invokes imported host callbacks in order:
/// 1. `f3dHost.writeBuffer(buffer_id, offset, bytes)`
/// 2. `f3dHost.copyBufferToBuffer(src, src_off, dst, dst_off, size)`
///
/// Returns the callback count (2) on success.
///
/// Enforces seam invariants:
/// - Zero live linear memory slices across host callback invocation.
/// - Zero RefCell borrows active during host call.
/// - Zero global table / mutex locks held across host boundary.
/// - Bytes passed as an owned copy (in JS Uint8Array).
/// - Uses Number rather than BigInt at the JS boundary.
/// - Immediate propagation of thrown host errors.
pub fn f3d_bridge_callback_storage_upload_frame(
    affine_rows: &[f32],
    expected_records: u32,
    destination_slot: u32,
) -> Result<u32, JsValue> {
    let ops = build_affine_rows_storage_upload_callback_ops(
        affine_rows,
        expected_records,
        destination_slot,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut callback_count = 0u32;
    for op in ops {
        match op {
            StorageUploadCallbackOp::WriteBuffer { buffer_id, offset, bytes } => {
                let js_bytes = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
                js_bytes.copy_from(&bytes);
                drop(bytes);
                host_write_buffer(buffer_id, offset, &js_bytes)?;
                callback_count += 1;
            }
            StorageUploadCallbackOp::CopyBufferToBuffer { src, src_off, dst, dst_off, size } => {
                host_copy_buffer_to_buffer(src, src_off, dst, dst_off, size)?;
                callback_count += 1;
            }
        }
    }
    Ok(callback_count)
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Packs a flat slice of 3x4 affine transforms (`12 * N` f32 values) into 256-byte aligned uniform buffer bytes.
/// Returns an owned `Vec<u8>` for the generated static submission bridge variant.
pub fn f3d_pack_affine_rows_bytes(affine_rows: &[f32]) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    pack_affine_rows_uniform_bytes(affine_rows)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Packs a flat slice of 3x4 affine transforms (`12 * N` f32 values) into packed 48-byte storage buffer records.
/// Returns an owned `Vec<u8>` of length `48 * N`.
pub fn f3d_pack_affine_rows_storage_bytes(
    affine_rows: &[f32],
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    pack_affine_rows_storage_bytes(affine_rows)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Canonical Wasm export: allocates persistent source buffer 610 and destination slots 611 and 612 (workload b / tmt.5).
pub fn f3d_build_affine_rows_storage_upload_init_packet(
    expected_records: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_affine_rows_storage_upload_init_submission(expected_records)
        .and_then(|packet| packet.encode())
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Canonical Wasm export: writes packed 48-byte records into 610 and copies to destination slot 611 or 612 (workload b / tmt.5).
pub fn f3d_build_affine_rows_storage_upload_frame_packet(
    affine_rows: &[f32],
    expected_records: u32,
    destination_slot: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_affine_rows_storage_upload_frame_submission(affine_rows, expected_records, destination_slot)
        .and_then(|packet| packet.encode())
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
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

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_batch_packet`.
pub fn f3d_build_affine_rows_batch_packet(
    affine_rows: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    build_affine_rows_batch_submission(affine_rows, width, height)
        .and_then(|p| p.encode())
        .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_batch_frame_packet` for host execution and unit tests.
pub fn f3d_build_affine_rows_batch_frame_packet(
    affine_rows: &[f32],
    width: u32,
    height: u32,
    expected_draws: u32,
) -> Result<Vec<u8>, String> {
    build_affine_rows_batch_frame_submission(affine_rows, width, height, expected_draws)
        .and_then(|p| p.encode())
        .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_batch_frame_packet_borrowed` for host execution and unit tests.
pub fn f3d_build_affine_rows_batch_frame_packet_borrowed(
    affine_rows: &[f32],
    width: u32,
    height: u32,
    expected_draws: u32,
) -> Result<[u32; 2], String> {
    build_affine_rows_batch_frame_packet_borrowed(affine_rows, width, height, expected_draws)
        .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_borrowed_frame_packet_ptr` for host execution and unit tests.
pub fn f3d_borrowed_frame_packet_ptr() -> u32 {
    borrowed_frame_packet_ptr()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_borrowed_frame_packet_len` for host execution and unit tests.
pub fn f3d_borrowed_frame_packet_len() -> u32 {
    borrowed_frame_packet_len()
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_pack_affine_rows_bytes` for host verification and unit tests.
pub fn f3d_pack_affine_rows_bytes(affine_rows: &[f32]) -> Result<Vec<u8>, String> {
    pack_affine_rows_uniform_bytes(affine_rows).map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_pack_affine_rows_storage_bytes` for host verification and unit tests.
pub fn f3d_pack_affine_rows_storage_bytes(
    affine_rows: &[f32],
) -> Result<Vec<u8>, String> {
    pack_affine_rows_storage_bytes(affine_rows).map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_storage_upload_init_packet` for host verification and unit tests.
pub fn f3d_build_affine_rows_storage_upload_init_packet(
    expected_records: u32,
) -> Result<Vec<u8>, String> {
    build_affine_rows_storage_upload_init_submission(expected_records)
        .and_then(|packet| packet.encode())
        .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_affine_rows_storage_upload_frame_packet` for host verification and unit tests.
pub fn f3d_build_affine_rows_storage_upload_frame_packet(
    affine_rows: &[f32],
    expected_records: u32,
    destination_slot: u32,
) -> Result<Vec<u8>, String> {
    build_affine_rows_storage_upload_frame_submission(affine_rows, expected_records, destination_slot)
        .and_then(|packet| packet.encode())
        .map_err(|e| e.to_string())
}

fn encode_textured_affine_triangle_helper(
    pixels: &[u8],
    tex_w: u32,
    tex_h: u32,
    affine_data: &[f32],
    target_w: u32,
    target_h: u32,
) -> Result<Vec<u8>, PacketEncodeError> {
    if affine_data.len() != 12 {
        return Err(PacketEncodeError::InvalidDimensions(format!(
            "affine_data must have exactly 12 f32 elements (row-major AffineRows), got {}",
            affine_data.len()
        )));
    }
    let affine = AffineRows::new(
        [affine_data[0], affine_data[1], affine_data[2], affine_data[3]],
        [affine_data[4], affine_data[5], affine_data[6], affine_data[7]],
        [affine_data[8], affine_data[9], affine_data[10], affine_data[11]],
    );
    let packet = build_textured_affine_triangle_packet(
        pixels, tex_w, tex_h, &affine, target_w, target_h,
    )?;
    packet.encode()
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a textured affine triangle submission packet and returns the raw binary bytes (§6.1, §6.2, vqa.6).
pub fn gpu_bridge_build_textured_affine_triangle_packet(
    pixels: &[u8],
    tex_w: u32,
    tex_h: u32,
    affine_data: &[f32],
    target_w: u32,
    target_h: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    encode_textured_affine_triangle_helper(pixels, tex_w, tex_h, affine_data, target_w, target_h)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a textured affine triangle submission packet and returns the raw binary bytes (canonical alias).
pub fn f3d_build_textured_affine_triangle_packet(
    pixels: &[u8],
    tex_w: u32,
    tex_h: u32,
    affine_data: &[f32],
    target_w: u32,
    target_h: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    encode_textured_affine_triangle_helper(pixels, tex_w, tex_h, affine_data, target_w, target_h)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_textured_affine_triangle_packet` for host verification and unit tests.
pub fn gpu_bridge_build_textured_affine_triangle_packet(
    pixels: &[u8],
    tex_w: u32,
    tex_h: u32,
    affine_data: &[f32],
    target_w: u32,
    target_h: u32,
) -> Result<Vec<u8>, String> {
    encode_textured_affine_triangle_helper(pixels, tex_w, tex_h, affine_data, target_w, target_h)
        .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_textured_affine_triangle_packet` (canonical alias).
pub fn f3d_build_textured_affine_triangle_packet(
    pixels: &[u8],
    tex_w: u32,
    tex_h: u32,
    affine_data: &[f32],
    target_w: u32,
    target_h: u32,
) -> Result<Vec<u8>, String> {
    encode_textured_affine_triangle_helper(pixels, tex_w, tex_h, affine_data, target_w, target_h)
        .map_err(|e| e.to_string())
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

/// Builds a real WGSL overlapping-triangle depth execution packet (§8.5).
///
/// Tests WebGPU depth attachment lowering, pipeline depth-test/write state, and pass preservation:
/// - Near triangle at $z = 0.2$ (Green `[0.0, 1.0, 0.0, 1.0]`)
/// - Far triangle at $z = 0.8$ (Red `[1.0, 0.0, 0.0, 1.0]`)
/// - Pipeline 100: configured with `depth_compare: Less` (2), `depth_write_enabled: true`, format `depth32float` (4).
/// Builds a self-contained WebGPU submission packet for overlapping-triangle depth verification.
///
/// # Scenarios (§6.1, §6.7, §8.5)
/// - `0`: Near then Far (`depth_compare: Less`, `depth_write_enabled: true`). Near triangle (z=0.2, green) draws first,
///   then Far triangle (z=0.8, red) is rejected by depth test, leaving green pixels.
/// - `1`: Far then Near (`depth_compare: Less`, `depth_write_enabled: true`). Far triangle draws first, then Near triangle
///   overwrites it via depth test (draw-order swap preservation, producing identical green pixels to scenario 0).
/// - `2`: Multi-Pass Depth Persistence. Pass 1 renders Near triangle with `Clear`, `Store`. Pass 2 renders Far triangle
///   with `Load`, `Store` (`LoadOp::Load` on depth). Preserved depth rejects Far triangle, leaving green pixels.
/// - `3`: Planted Negative Control. Near then Far, but with `depth_compare: Always` and `depth_write_enabled: false`.
///   Far triangle overwrites Near, producing red pixels [255, 0, 0, 255] and strictly diverging from depth-enabled reference.
/// - `4`: Readonly Depth. Pass 1 draws Near (Green, z=0.2) with depth_write=true. Pass 2 color Load, depth_read_only=true
///   (load/store/clear omitted), pipeline depth_write_enabled=false, draws Far (Red, z=0.8). Depth test rejects Far -> Green.
/// - `5`: Empty Depth Clear Pass. Pass 1 draws Near (Green, z=0.2). Pass 2 color Load, depth Clear (1.0), zero draws.
///   Pass 3 color Load, depth Load, draws Far (Red, z=0.8). Far passes depth test (< 1.0) and overwrites Near -> Red.
pub fn build_overlapping_depth_submission(scenario: u32) -> GpuSubmissionPacket {
    with_global_resource_table(|table| {
        table.register(2);   // vb_near_id
        table.register(3);   // vb_far_id
        table.register(10);  // color_target_texture_id
        table.register(12);  // depth_target_texture_id
        table.register(20);  // readback_buffer_id
        table.register(100); // pipeline_id
        table.register(101); // pipeline_ro_id
    });

    let mut packet = GpuSubmissionPacket::new();

    let vb_near_id = 2u32;
    let vb_far_id = 3u32;
    let target_color_id = 10u32;
    let target_depth_id = 12u32;
    let readback_buffer_id = 20u32;
    let pipeline_id = 100u32;
    let pipeline_ro_id = 101u32;

    // 1. Vertex buffer: Near triangle at z = 0.2, Green [0.0, 1.0, 0.0, 1.0]
    // 3 vertices * 7 floats (stride 28: pos vec3<f32> at offset 0 + color vec4<f32> at offset 12)
    let near_floats: [f32; 21] = [
        // x,     y,    z,    r,   g,   b,   a
         0.0,   0.5,  0.2,  0.0, 1.0, 0.0, 1.0,
        -0.5,  -0.5,  0.2,  0.0, 1.0, 0.0, 1.0,
         0.5,  -0.5,  0.2,  0.0, 1.0, 0.0, 1.0,
    ];
    let mut near_bytes = Vec::with_capacity(near_floats.len() * 4);
    for f in near_floats {
        near_bytes.extend_from_slice(&f.to_le_bytes());
    }
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb_near_id,
        size: near_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb_near_id,
        offset: 0,
        data: near_bytes.clone(),
    });

    // 2. Vertex buffer: Far triangle at z = 0.8, Red [1.0, 0.0, 0.0, 1.0]
    // 3 vertices * 7 floats (stride 28: pos vec3<f32> at offset 0 + color vec4<f32> at offset 12)
    let far_floats: [f32; 21] = [
        // x,     y,    z,    r,   g,   b,   a
         0.0,   0.6,  0.8,  1.0, 0.0, 0.0, 1.0,
        -0.6,  -0.6,  0.8,  1.0, 0.0, 0.0, 1.0,
         0.6,  -0.6,  0.8,  1.0, 0.0, 0.0, 1.0,
    ];
    let mut far_bytes = Vec::with_capacity(far_floats.len() * 4);
    for f in far_floats {
        far_bytes.extend_from_slice(&f.to_le_bytes());
    }
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: vb_far_id,
        size: far_bytes.len() as u32,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: vb_far_id,
        offset: 0,
        data: far_bytes.clone(),
    });

    // 3. Color target texture (64x64, rgba8unorm)
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_color_id,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    // 4. Depth target texture (64x64, depth32float)
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_depth_id,
        width: 64,
        height: 64,
        format: TARGET_FORMAT_DEPTH32FLOAT,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
    });

    // 5. Readback buffer (64 * 256 bytes)
    let bytes_per_row = 256u32;
    let readback_size = bytes_per_row * 64;
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_buffer_id,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // 6. Depth pipeline with WGSL shader (matching directDepthReference verbatim)
    let depth_shader = "\
struct VertexInput {\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) color: vec4<f32>,\n\
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
    out.position = vec4<f32>(in.position, 1.0);\n\
    out.color = in.color;\n\
    return out;\n\
}\n\
\n\
@fragment\n\
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {\n\
    return in.color;\n\
}\n";

    let is_depth_disabled_negative = scenario == 3;
    let depth_compare = if is_depth_disabled_negative {
        DEPTH_COMPARE_ALWAYS
    } else {
        DEPTH_COMPARE_LESS
    };
    let depth_write_enabled = !is_depth_disabled_negative;

    packet.push(GpuCommand::CreatePipelineDepth {
        pipeline_id,
        wgsl_code: depth_shader.to_string(),
        target_format: TARGET_FORMAT_RGBA8UNORM,
        has_vertex_buffer: true,
        has_uniform_buffer: false,
        uniform_size: 0,
        vertex_stride: 28,
        depth_format: TARGET_FORMAT_DEPTH32FLOAT,
        depth_write_enabled,
        depth_compare,
    });

    if scenario == 4 {
        packet.push(GpuCommand::CreatePipelineDepth {
            pipeline_id: pipeline_ro_id,
            wgsl_code: depth_shader.to_string(),
            target_format: TARGET_FORMAT_RGBA8UNORM,
            has_vertex_buffer: true,
            has_uniform_buffer: false,
            uniform_size: 0,
            vertex_stride: 28,
            depth_format: TARGET_FORMAT_DEPTH32FLOAT,
            depth_write_enabled: false,
            depth_compare: DEPTH_COMPARE_LESS,
        });
    }

    // 7. Build passes using f3d-graph's PassGraph and compile to ExecutionPlan
    use f3d_graph::{
        pass::{ColorAttachment, DepthStencilAttachment, Draw, Pass, PassId},
        PassGraph, ResourceUse,
    };

    let mut graph = PassGraph::new();

    let draw_near = Draw::new(
        1,
        pipeline_id,
        3,
        0,
        vec![
            ResourceUse::buffer_vertex(ResourceId::new(vb_near_id), DataVersion::INITIAL, Some(0), Some(near_bytes.len() as u64)),
        ],
    );

    let draw_far = Draw::new(
        2,
        pipeline_id,
        3,
        0,
        vec![
            ResourceUse::buffer_vertex(ResourceId::new(vb_far_id), DataVersion::INITIAL, Some(0), Some(far_bytes.len() as u64)),
        ],
    );

    if scenario == 2 {
        // Scenario 2: Multi-Pass Depth Persistence
        // Pass 1: Clear color to [0,0,0,1], clear depth to 1.0, Store depth, draw Near (Green, z=0.2)
        let mut pass1 = Pass::new_render(PassId::new(1), "pass_near_clear");
        pass1.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(target_color_id),
            [0.0, 0.0, 0.0, 1.0],
        ));
        pass1.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(target_depth_id),
            1.0,
        ));
        pass1.draws.push(draw_near);
        graph.add_pass(pass1).expect("add pass1");

        // Pass 2: Load color, Load depth (LoadOp::Load), Store depth, draw Far (Red, z=0.8)
        // Depth test rejects Far, preserving earlier Near (Green)
        let mut pass2 = Pass::new_render(PassId::new(2), "pass_far_load");
        pass2.color_attachments.push(ColorAttachment::new_load(
            ResourceId::new(target_color_id),
        ));
        pass2.depth_stencil_attachment = Some(DepthStencilAttachment {
            target_id: ResourceId::new(target_depth_id),
            view_subresource: f3d_graph::resource::SubresourceRange::full_texture(),
            depth_load_op: Some(LoadOp::Load),
            depth_store_op: Some(StoreOp::Store),
            depth_clear_value: 1.0,
            depth_read_only: false,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });
        pass2.draws.push(draw_far);
        graph.add_pass(pass2).expect("add pass2");
    } else if scenario == 4 {
        // Scenario 4: Readonly Depth (depthAttachmentDesc omits load/store/clear per WebGPU spec)
        // Pass 1: Clear color to black, clear depth to 1.0, Store depth, draw Near (Green, z=0.2, depth_write=true)
        let mut pass1 = Pass::new_render(PassId::new(1), "pass_near_clear");
        pass1.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(target_color_id),
            [0.0, 0.0, 0.0, 1.0],
        ));
        pass1.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(target_depth_id),
            1.0,
        ));
        pass1.draws.push(draw_near);
        graph.add_pass(pass1).expect("add pass1");

        // Pass 2: Color Load, Depth read-only (load_op=None, store_op=None, read_only=true)
        // Pipeline 101 has depth_write_enabled=false. Draw Far (Red, z=0.8). Discarded by depth test -> Green
        let mut pass2 = Pass::new_render(PassId::new(2), "pass_far_readonly");
        pass2.color_attachments.push(ColorAttachment::new_load(
            ResourceId::new(target_color_id),
        ));
        pass2.depth_stencil_attachment = Some(DepthStencilAttachment {
            target_id: ResourceId::new(target_depth_id),
            view_subresource: f3d_graph::resource::SubresourceRange::full_texture(),
            depth_load_op: None,
            depth_store_op: None,
            depth_clear_value: 1.0,
            depth_read_only: true,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });
        let draw_far_ro = Draw::new(
            2,
            pipeline_ro_id,
            3,
            0,
            vec![
                ResourceUse::buffer_vertex(ResourceId::new(vb_far_id), DataVersion::INITIAL, Some(0), Some(far_bytes.len() as u64)),
            ],
        );
        pass2.draws.push(draw_far_ro);
        graph.add_pass(pass2).expect("add pass2");
    } else if scenario == 5 {
        // Scenario 5: Empty Depth Clear Pass (proves zero-draw depth clear executes and resets depth)
        // Pass 1: Clear color and depth, draw Near (Green, z=0.2)
        let mut pass1 = Pass::new_render(PassId::new(1), "pass_near_clear");
        pass1.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(target_color_id),
            [0.0, 0.0, 0.0, 1.0],
        ));
        pass1.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(target_depth_id),
            1.0,
        ));
        pass1.draws.push(draw_near);
        graph.add_pass(pass1).expect("add pass1");

        // Pass 2: Color Load, Depth Clear (1.0), ZERO draws!
        let mut pass2 = Pass::new_render(PassId::new(2), "pass_empty_depth_clear");
        pass2.color_attachments.push(ColorAttachment::new_load(
            ResourceId::new(target_color_id),
        ));
        pass2.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(target_depth_id),
            1.0,
        ));
        // Empty draws
        graph.add_pass(pass2).expect("add pass2");

        // Pass 3: Color Load, Depth Load, draw Far (Red, z=0.8)
        // Because Pass 2 cleared depth back to 1.0, Far passes depth test (0.8 < 1.0) and overwrites Near -> Red
        let mut pass3 = Pass::new_render(PassId::new(3), "pass_far_after_clear");
        pass3.color_attachments.push(ColorAttachment::new_load(
            ResourceId::new(target_color_id),
        ));
        pass3.depth_stencil_attachment = Some(DepthStencilAttachment {
            target_id: ResourceId::new(target_depth_id),
            view_subresource: f3d_graph::resource::SubresourceRange::full_texture(),
            depth_load_op: Some(LoadOp::Load),
            depth_store_op: Some(StoreOp::Store),
            depth_clear_value: 1.0,
            depth_read_only: false,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });
        pass3.draws.push(draw_far);
        graph.add_pass(pass3).expect("add pass3");
    } else {
        // Single pass: Clear depth to 1.0, clear color to [0,0,0,1]
        let mut pass = Pass::new_render(PassId::new(1), "pass_depth_draws");
        pass.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(target_color_id),
            [0.0, 0.0, 0.0, 1.0],
        ));
        pass.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(target_depth_id),
            1.0,
        ));
        if scenario == 1 {
            // Scenario 1: Far then Near (Draw-order swap)
            pass.draws.push(draw_far);
            pass.draws.push(draw_near);
        } else {
            // Scenario 0 (Near then Far) and Scenario 3 (Negative control: Near then Far with disabled depth)
            pass.draws.push(draw_near);
            pass.draws.push(draw_far);
        }
        graph.add_pass(pass).expect("add pass");
    }

    let plan = graph.compile(None).expect("depth pass graph must compile");
    let lowered_commands = lower_plan(&plan).expect("depth plan lowering must succeed");
    for cmd in lowered_commands {
        packet.push(cmd);
    }

    // 8. Copy texture to readback buffer
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_color_id,
        buffer_id: readback_buffer_id,
        width: 64,
        height: 64,
        epoch: Epoch::ZERO,
    });

    packet
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes an overlapping-triangle depth execution packet and returns the raw binary bytes.
pub fn gpu_bridge_build_overlapping_depth_packet(scenario: u32) -> Vec<u8> {
    build_overlapping_depth_submission(scenario)
        .encode()
        .expect("static overlapping depth packet encoding must not fail")
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes an overlapping-triangle depth execution packet (canonical alias).
pub fn f3d_build_overlapping_depth_packet(scenario: u32) -> Vec<u8> {
    gpu_bridge_build_overlapping_depth_packet(scenario)
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `gpu_bridge_build_overlapping_depth_packet` for host verification and unit tests.
#[must_use]
pub fn gpu_bridge_build_overlapping_depth_packet(scenario: u32) -> Vec<u8> {
    build_overlapping_depth_submission(scenario)
        .encode()
        .expect("static overlapping depth packet encoding must not fail")
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Native export for `f3d_build_overlapping_depth_packet` (canonical alias).
#[must_use]
pub fn f3d_build_overlapping_depth_packet(scenario: u32) -> Vec<u8> {
    gpu_bridge_build_overlapping_depth_packet(scenario)
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
static GLOBAL_BORROWED_FRAME_PACKET: Mutex<Option<Vec<u8>>> = Mutex::new(None);

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

/// Executes a closure with mutable access to the global borrowed frame packet slot.
pub(crate) fn with_global_borrowed_frame_packet<R>(f: impl FnOnce(&mut Option<Vec<u8>>) -> R) -> R {
    let mut guard = match GLOBAL_BORROWED_FRAME_PACKET.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(&mut guard)
}

/// Executes a closure with read-only access to the global borrowed frame packet slot.
pub fn with_global_borrowed_frame_packet_ref<R>(f: impl FnOnce(Option<&[u8]>) -> R) -> R {
    let guard = match GLOBAL_BORROWED_FRAME_PACKET.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(guard.as_deref())
}

/// Clears the global borrowed frame packet slot, freeing linear memory if held.
///
/// Refuses to clear and returns `false` if an active linear memory borrow scope is held,
/// leaving the slot intact to prevent dangling pointers in active JavaScript views.
/// Returns `true` when cleared successfully.
pub fn clear_borrowed_frame_packet() -> bool {
    let is_borrowed = with_global_borrow_scope(|scope| scope.is_borrowed());
    if is_borrowed {
        return false;
    }
    with_global_borrowed_frame_packet(|slot| {
        *slot = None;
    });
    true
}

/// Returns the linear memory address (pointer) of the global borrowed frame packet buffer, or 0 if empty.
#[must_use]
pub fn borrowed_frame_packet_ptr() -> u32 {
    with_global_borrowed_frame_packet_ref(|slot| {
        slot.map_or(0, |s| (s.as_ptr() as usize) as u32)
    })
}

/// Returns the byte length of the global borrowed frame packet buffer, or 0 if empty.
#[must_use]
pub fn borrowed_frame_packet_len() -> u32 {
    with_global_borrowed_frame_packet_ref(|slot| {
        slot.map_or(0, |s| s.len() as u32)
    })
}

/// Returns the current accumulated borrowed view bytes from copy accounting (§6.6, §13.1).
#[must_use]
pub fn borrow_scope_bytes_view() -> u64 {
    with_global_borrow_scope(|scope| scope.accounting().bytes_view)
}

/// Encodes an update packet for an already initialized AffineRows batch scene into the static borrowed packet slot.
///
/// Refuses to rebuild or reallocate the slot if an active linear memory borrow scope is held,
/// to prevent invalidating active JavaScript typed array views pointing to the buffer.
/// Records the view transfer length in borrow scope copy accounting.
///
/// Returns `[ptr, len]` as WebAssembly linear memory byte offset and byte length.
pub fn build_affine_rows_batch_frame_packet_borrowed(
    affine_rows: &[f32],
    width: u32,
    height: u32,
    expected_draws: u32,
) -> Result<[u32; 2], PacketEncodeError> {
    with_global_borrow_scope(|scope| {
        if scope.is_borrowed() {
            return Err(PacketEncodeError::InvalidDimensions(
                "linear memory borrow scope is active: cannot rebuild borrowed frame packet slot while borrowed".to_string(),
            ));
        }
        Ok(())
    })?;

    let packet = build_affine_rows_batch_frame_submission(affine_rows, width, height, expected_draws)?;
    let encoded = packet.encode()?;

    let len = encoded.len();
    let len_u32 = u32::try_from(len).map_err(|_| PacketEncodeError::DataPayloadOverflow {
        offset: 0,
        length: len,
    })?;

    with_global_borrow_scope(|scope| {
        if scope.is_borrowed() {
            return Err(PacketEncodeError::InvalidDimensions(
                "linear memory borrow scope is active: cannot rebuild borrowed frame packet slot while borrowed".to_string(),
            ));
        }
        scope.accounting_mut().record_view(len as u64);
        Ok(())
    })?;

    let ptr_u32 = with_global_borrowed_frame_packet(|slot| {
        *slot = Some(encoded);
        let slice = slot.as_ref().expect("slot was just populated");
        (slice.as_ptr() as usize) as u32
    });

    Ok([ptr_u32, len_u32])
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

    /// Test-only mutex serializing tests that mutate, reset, advance, or assert `GLOBAL_BORROW_SCOPE` or `GLOBAL_BORROWED_FRAME_PACKET`.
    /// Tolerates lock poisoning across test failures.
    static TEST_BORROW_SCOPE_LOCK: Mutex<()> = Mutex::new(());

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
    fn lower_plan_two_target_bridge_plan_matches_first_frame_commands() {
        use f3d_graph::ResourceId;

        let mut tracker = CanvasEpochTracker::new();
        tracker.register_canvas(
            CanvasId::new(0),
            ResourceId::new(0),
            64,
            64,
            CanvasFormat::Bgra8Unorm,
        );
        let epoch = tracker
            .begin_frame_acquire(CanvasId::new(0))
            .expect("acquire canvas epoch")
            .epoch;

        let plan = f3d_graph::build_two_target_bridge_plan(
            ResourceId::new(10),
            ResourceId::new(0),
            ResourceId::new(20),
            100,
            101,
            ResourceId::new(2),
            3,
            64,
            64,
            DataVersion::INITIAL,
            Some(epoch),
            Some(&tracker),
        )
        .expect("build two-target bridge plan");

        let lowered = lower_plan(&plan).expect("lower plan");

        let expected: Vec<GpuCommand> = build_triangle_submission()
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

        assert_eq!(epoch, Epoch::new(1));
        assert_eq!(lowered.len(), 3);
        assert_eq!(expected.len(), 3);
        assert_eq!(lowered[0], expected[0]);
        assert_eq!(lowered[1], expected[1]);
        assert_eq!(
            lowered[2],
            GpuCommand::CopyTextureToBuffer {
                texture_id: 10,
                buffer_id: 20,
                width: 64,
                height: 64,
                epoch: Epoch::new(1),
            }
        );
        assert_eq!(
            expected[2],
            GpuCommand::CopyTextureToBuffer {
                texture_id: 10,
                buffer_id: 20,
                width: 64,
                height: 64,
                epoch: Epoch::ZERO,
            }
        );
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
    fn bundle_then_direct_warm_cache_command_shape() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        for empty_bundles in [false, true] {
            let packet = build_bundle_then_direct_warm_cache_submission(empty_bundles);
            let commands = packet.commands();
            assert_eq!(commands.len(), 15);

            // Command 0: CreateBuffer uniform (id 1, size 512, UNIFORM | COPY_DST)
            match &commands[0] {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    assert_eq!(*buffer_id, 1);
                    assert_eq!(*size, 512);
                    assert_eq!(*usage, BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST);
                }
                other => panic!("expected CreateBuffer at index 0, got {other:?}"),
            }

            // Command 1: WriteBuffer uniform (512 bytes)
            match &commands[1] {
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    assert_eq!(*buffer_id, 1);
                    assert_eq!(*offset, 0);
                    assert_eq!(data.len(), 512);
                }
                other => panic!("expected WriteBuffer at index 1, got {other:?}"),
            }

            // Command 2: CreateBuffer vb1 (Triangle 1, bundle, id 2)
            match &commands[2] {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    assert_eq!(*buffer_id, 2);
                    assert_eq!(*size, (3 * VertexPosUv::BYTE_SIZE) as u32);
                    assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
                }
                other => panic!("expected CreateBuffer at index 2, got {other:?}"),
            }

            // Command 3: WriteBuffer vb1 (60 bytes)
            match &commands[3] {
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    assert_eq!(*buffer_id, 2);
                    assert_eq!(*offset, 0);
                    assert_eq!(data.len(), 3 * VertexPosUv::BYTE_SIZE);
                }
                other => panic!("expected WriteBuffer at index 3, got {other:?}"),
            }

            // Command 4: CreateBuffer vb2 (Triangle 2, direct, id 3)
            match &commands[4] {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    assert_eq!(*buffer_id, 3);
                    assert_eq!(*size, (3 * VertexPosUv::BYTE_SIZE) as u32);
                    assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
                }
                other => panic!("expected CreateBuffer at index 4, got {other:?}"),
            }

            // Command 5: WriteBuffer vb2 (60 bytes)
            match &commands[5] {
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    assert_eq!(*buffer_id, 3);
                    assert_eq!(*offset, 0);
                    assert_eq!(data.len(), 3 * VertexPosUv::BYTE_SIZE);
                }
                other => panic!("expected WriteBuffer at index 5, got {other:?}"),
            }

            // Command 6: CreateTexture target (id 10, 64x64 RGBA8Unorm)
            match &commands[6] {
                GpuCommand::CreateTexture { texture_id, width, height, format, usage } => {
                    assert_eq!(*texture_id, 10);
                    assert_eq!(*width, 64);
                    assert_eq!(*height, 64);
                    assert_eq!(*format, TARGET_FORMAT_RGBA8UNORM);
                    assert_eq!(*usage, TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC);
                }
                other => panic!("expected CreateTexture at index 6, got {other:?}"),
            }

            // Command 7: CreateBuffer readback (id 20, 256 * 64)
            match &commands[7] {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    assert_eq!(*buffer_id, 20);
                    assert_eq!(*size, 256 * 64);
                    assert_eq!(*usage, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
                }
                other => panic!("expected CreateBuffer at index 7, got {other:?}"),
            }

            // Command 8: CreatePipeline (flat color shader, id 200)
            match &commands[8] {
                GpuCommand::CreatePipeline { pipeline_id, target_format, has_vertex_buffer, has_uniform_buffer, .. } => {
                    assert_eq!(*pipeline_id, 200);
                    assert_eq!(*target_format, TARGET_FORMAT_RGBA8UNORM);
                    assert!(*has_vertex_buffer);
                    assert!(*has_uniform_buffer);
                }
                other => panic!("expected CreatePipeline at index 8, got {other:?}"),
            }

            // Command 9: RecordBundle 1 (Triangle 1, vb 2, offset 0, count 3)
            match &commands[9] {
                GpuCommand::RecordBundle {
                    bundle_id,
                    pipeline_id,
                    vertex_buffer_id,
                    vertex_count,
                    uniform_dynamic_offset,
                    uniform_buffer_id,
                    target_format,
                } => {
                    assert_eq!(*bundle_id, 1);
                    assert_eq!(*pipeline_id, 200);
                    assert_eq!(*vertex_buffer_id, 2);
                    assert_eq!(*vertex_count, 3);
                    assert_eq!(*uniform_dynamic_offset, 0);
                    assert_eq!(*uniform_buffer_id, 1);
                    assert_eq!(*target_format, TARGET_FORMAT_RGBA8UNORM);
                }
                other => panic!("expected RecordBundle at index 9, got {other:?}"),
            }

            // Command 10: RenderPass opener (clear black, vertex_count 0, PASS_FLAG_NEW_PASS)
            match &commands[10] {
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
                    assert_eq!(*target_type, TARGET_OFFSCREEN);
                    assert_eq!(*target_id, 10);
                    assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
                    assert_eq!(*pipeline_id, 0);
                    assert_eq!(*vertex_buffer_id, 0);
                    assert_eq!(*vertex_count, 0);
                    assert_eq!(*uniform_dynamic_offset, 0);
                    assert_eq!(*uniform_buffer_id, 0);
                    assert_eq!(*load_op, LOAD_OP_CLEAR);
                    assert_eq!(*store_op, STORE_OP_STORE);
                    assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                }
                other => panic!("expected RenderPass opener at index 10, got {other:?}"),
            }

            // Command 11: Direct draw of the right triangle BEFORE any bundle:
            // pipeline 200, uniform 1 offset 256, vb 3, vertex_count 3, PASS_FLAG_NONE
            match &commands[11] {
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
                    assert_eq!(*target_type, TARGET_OFFSCREEN);
                    assert_eq!(*target_id, 10);
                    assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
                    assert_eq!(*pipeline_id, 200);
                    assert_eq!(*vertex_buffer_id, 3);
                    assert_eq!(*vertex_count, 3);
                    assert_eq!(*uniform_dynamic_offset, 256);
                    assert_eq!(*uniform_buffer_id, 1);
                    assert_eq!(*load_op, LOAD_OP_CLEAR);
                    assert_eq!(*store_op, STORE_OP_STORE);
                    assert_eq!(*pass_flags, PASS_FLAG_NONE);
                }
                other => panic!("expected RenderPass pre-bundle direct draw at index 11, got {other:?}"),
            }

            // Command 12: ExecuteBundles ([1] if !empty_bundles else [])
            match &commands[12] {
                GpuCommand::ExecuteBundles { bundle_ids } => {
                    if empty_bundles {
                        assert!(bundle_ids.is_empty());
                    } else {
                        assert_eq!(bundle_ids, &vec![1]);
                    }
                }
                other => panic!("expected ExecuteBundles at index 12, got {other:?}"),
            }

            // Command 13: Identical direct draw again:
            // same pipeline, uniform, offset, vb, vertex_count and PASS_FLAG_NONE
            match &commands[13] {
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
                    assert_eq!(*target_type, TARGET_OFFSCREEN);
                    assert_eq!(*target_id, 10);
                    assert_eq!(*clear_color, [0.0, 0.0, 0.0, 1.0]);
                    assert_eq!(*pipeline_id, 200);
                    assert_eq!(*vertex_buffer_id, 3);
                    assert_eq!(*vertex_count, 3);
                    assert_eq!(*uniform_dynamic_offset, 256);
                    assert_eq!(*uniform_buffer_id, 1);
                    assert_eq!(*load_op, LOAD_OP_CLEAR);
                    assert_eq!(*store_op, STORE_OP_STORE);
                    assert_eq!(*pass_flags, PASS_FLAG_NONE);
                }
                other => panic!("expected RenderPass post-bundle direct draw at index 13, got {other:?}"),
            }

            // Command 14: CopyTextureToBuffer, texture 10 -> buffer 20, 64x64
            match &commands[14] {
                GpuCommand::CopyTextureToBuffer {
                    texture_id,
                    buffer_id,
                    width,
                    height,
                    epoch,
                } => {
                    assert_eq!(*texture_id, 10);
                    assert_eq!(*buffer_id, 20);
                    assert_eq!(*width, 64);
                    assert_eq!(*height, 64);
                    assert_eq!(*epoch, Epoch::ZERO);
                }
                other => panic!("expected CopyTextureToBuffer at index 14, got {other:?}"),
            }

            // Direct comparison: pre-bundle (11) and post-bundle (13) carry identical fields
            if let (
                GpuCommand::RenderPass {
                    target_type: t0,
                    target_id: tid0,
                    clear_color: c0,
                    pipeline_id: p0,
                    vertex_buffer_id: vb0,
                    vertex_count: vc0,
                    uniform_dynamic_offset: off0,
                    uniform_buffer_id: ub0,
                    load_op: lo0,
                    store_op: st0,
                    pass_flags: pf0,
                },
                GpuCommand::RenderPass {
                    target_type: t1,
                    target_id: tid1,
                    clear_color: c1,
                    pipeline_id: p1,
                    vertex_buffer_id: vb1,
                    vertex_count: vc1,
                    uniform_dynamic_offset: off1,
                    uniform_buffer_id: ub1,
                    load_op: lo1,
                    store_op: st1,
                    pass_flags: pf1,
                },
            ) = (&commands[11], &commands[13])
            {
                assert_eq!(t0, t1);
                assert_eq!(tid0, tid1);
                assert_eq!(c0, c1);
                assert_eq!(p0, p1);
                assert_eq!(vb0, vb1);
                assert_eq!(vc0, vc1);
                assert_eq!(off0, off1);
                assert_eq!(ub0, ub1);
                assert_eq!(lo0, lo1);
                assert_eq!(st0, st1);
                assert_eq!(pf0, pf1);
            } else {
                panic!("commands 11 and 13 must both be RenderPass");
            }

            // Encode succeeds and encoded opcode sequence matches the command list
            let encoded = packet.encode().expect("encode must succeed");
            assert!(!encoded.is_empty());
            assert_eq!(encoded, f3d_build_bundle_direct_warm_cache_packet(empty_bundles));

            let cmd_count = u32::from_le_bytes(encoded[8..12].try_into().unwrap()) as usize;
            assert_eq!(cmd_count, 15);
            let total_data_len = u32::from_le_bytes(encoded[12..16].try_into().unwrap()) as usize;

            let mut opcodes = Vec::new();
            let mut cursor = 16;
            for _ in 0..cmd_count {
                let opcode = u16::from_le_bytes([encoded[cursor], encoded[cursor + 1]]);
                cursor += 2;
                opcodes.push(opcode);
                match opcode {
                    OPCODE_CREATE_BUFFER => cursor += 12,
                    OPCODE_WRITE_BUFFER => cursor += 16,
                    OPCODE_CREATE_TEXTURE => cursor += 20,
                    OPCODE_CREATE_PIPELINE => cursor += 32,
                    OPCODE_RENDER_PASS => cursor += 44,
                    OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                    OPCODE_RECORD_BUNDLE => cursor += 28,
                    OPCODE_EXECUTE_BUNDLES => {
                        let count = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap()) as usize;
                        cursor += 4 + count * 4;
                    }
                    other => panic!("unknown opcode {other}"),
                }
            }
            assert_eq!(cursor + total_data_len, encoded.len());

            let expected_opcodes = vec![
                OPCODE_CREATE_BUFFER,
                OPCODE_WRITE_BUFFER,
                OPCODE_CREATE_BUFFER,
                OPCODE_WRITE_BUFFER,
                OPCODE_CREATE_BUFFER,
                OPCODE_WRITE_BUFFER,
                OPCODE_CREATE_TEXTURE,
                OPCODE_CREATE_BUFFER,
                OPCODE_CREATE_PIPELINE,
                OPCODE_RECORD_BUNDLE,
                OPCODE_RENDER_PASS,
                OPCODE_RENDER_PASS,
                OPCODE_EXECUTE_BUNDLES,
                OPCODE_RENDER_PASS,
                OPCODE_COPY_TEXTURE_TO_BUFFER,
            ];
            assert_eq!(opcodes, expected_opcodes);
        }
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
        let _borrow_lock = match TEST_BORROW_SCOPE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

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
    fn affine_rows_layout_counterexample_command_shape() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        for wrong_mat4x3 in [false, true] {
            let packet = build_affine_rows_layout_counterexample_submission(wrong_mat4x3);
            let commands = packet.commands();
            assert_eq!(commands.len(), 9);

            // Command 0: CreateBuffer uniform 1 (size 256, UNIFORM | COPY_DST)
            match &commands[0] {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    assert_eq!(*buffer_id, 1);
                    assert_eq!(*size, 256);
                    assert_eq!(*usage, BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST);
                }
                other => panic!("expected CreateBuffer for uniform at index 0, got {other:?}"),
            }

            // Command 1: WriteBuffer uniform 1 (256 bytes, first 48 bytes are AffineRows, MUST BE IDENTICAL for both variants)
            match &commands[1] {
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    assert_eq!(*buffer_id, 1);
                    assert_eq!(*offset, 0);
                    assert_eq!(data.len(), 256);
                    let mut arr = [0u8; 48];
                    arr.copy_from_slice(&data[..48]);
                    let rows = AffineRows::from_bytes(&arr);
                    assert_eq!(rows.r0, [0.5, 0.0, 0.0, 0.5]);
                    assert_eq!(rows.r1, [0.0, 0.5, 0.0, 0.0]);
                    assert_eq!(rows.r2, [0.0, 0.0, 1.0, 0.0]);
                    assert!(data[48..].iter().all(|&b| b == 0));
                }
                other => panic!("expected WriteBuffer for uniform at index 1, got {other:?}"),
            }

            // Command 2: CreateBuffer vb 2 (size 60, VERTEX | COPY_DST)
            match &commands[2] {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    assert_eq!(*buffer_id, 2);
                    assert_eq!(*size, 60);
                    assert_eq!(*usage, BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST);
                }
                other => panic!("expected CreateBuffer for vertex at index 2, got {other:?}"),
            }

            // Command 3: WriteBuffer vb 2 (60 bytes)
            match &commands[3] {
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    assert_eq!(*buffer_id, 2);
                    assert_eq!(*offset, 0);
                    assert_eq!(data.len(), 60);
                }
                other => panic!("expected WriteBuffer for vertex at index 3, got {other:?}"),
            }

            // Command 4: CreateTexture target 10 (64x64 RGBA8Unorm)
            match &commands[4] {
                GpuCommand::CreateTexture { texture_id, width, height, format, usage } => {
                    assert_eq!(*texture_id, 10);
                    assert_eq!(*width, 64);
                    assert_eq!(*height, 64);
                    assert_eq!(*format, TARGET_FORMAT_RGBA8UNORM);
                    assert_eq!(*usage, TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC);
                }
                other => panic!("expected CreateTexture at index 4, got {other:?}"),
            }

            // Command 5: CreateBuffer readback 20 (size 16384)
            match &commands[5] {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    assert_eq!(*buffer_id, 20);
                    assert_eq!(*size, 16384);
                    assert_eq!(*usage, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
                }
                other => panic!("expected CreateBuffer for readback at index 5, got {other:?}"),
            }

            // Command 6: CreatePipeline 200
            // ONLY wgsl_code and uniform_size (64 vs 48) may differ!
            match &commands[6] {
                GpuCommand::CreatePipeline {
                    pipeline_id,
                    wgsl_code,
                    target_format,
                    has_vertex_buffer,
                    has_uniform_buffer,
                    uniform_size,
                    vertex_stride,
                } => {
                    assert_eq!(*pipeline_id, 200);
                    assert_eq!(*target_format, TARGET_FORMAT_RGBA8UNORM);
                    assert!(*has_vertex_buffer);
                    assert!(*has_uniform_buffer);
                    assert_eq!(*vertex_stride, 20);
                    if wrong_mat4x3 {
                        assert_eq!(*uniform_size, 64);
                        assert!(wgsl_code.contains("var<uniform> transform: mat4x3<f32>;"));
                        assert!(wgsl_code.contains("transform * vec4<f32>(in.position, 1.0)"));
                        assert!(!wgsl_code.contains("struct AffineRows"));
                    } else {
                        assert_eq!(*uniform_size, 48);
                        assert!(wgsl_code.contains("struct AffineRows"));
                        assert!(wgsl_code.contains("var<uniform> transform: AffineRows;"));
                        assert!(wgsl_code.contains("transform_affine_point(transform, in.position)"));
                    }
                }
                other => panic!("expected CreatePipeline at index 6, got {other:?}"),
            }

            // Command 7: RenderPass
            match &commands[7] {
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
                    assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                }
                other => panic!("expected RenderPass at index 7, got {other:?}"),
            }

            // Command 8: CopyTextureToBuffer
            match &commands[8] {
                GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, epoch } => {
                    assert_eq!(*texture_id, 10);
                    assert_eq!(*buffer_id, 20);
                    assert_eq!(*width, 64);
                    assert_eq!(*height, 64);
                    assert_eq!(*epoch, Epoch::ZERO);
                }
                other => panic!("expected CopyTextureToBuffer at index 8, got {other:?}"),
            }

            // Binary packet encoding assertions and opcode walker bounded by cmd_count
            let encoded = packet.encode().expect("encode must succeed");
            assert!(!encoded.is_empty());
            assert_eq!(encoded, f3d_build_affine_rows_layout_counterexample_packet(wrong_mat4x3));

            let cmd_count = u32::from_le_bytes(encoded[8..12].try_into().unwrap()) as usize;
            assert_eq!(cmd_count, 9);
            let total_data_len = u32::from_le_bytes(encoded[12..16].try_into().unwrap()) as usize;

            let mut opcodes = Vec::new();
            let mut cursor = 16;
            for _ in 0..cmd_count {
                let opcode = u16::from_le_bytes([encoded[cursor], encoded[cursor + 1]]);
                cursor += 2;
                opcodes.push(opcode);
                match opcode {
                    OPCODE_CREATE_BUFFER => cursor += 12,
                    OPCODE_WRITE_BUFFER => cursor += 16,
                    OPCODE_CREATE_TEXTURE => cursor += 20,
                    OPCODE_CREATE_PIPELINE => cursor += 32,
                    OPCODE_RENDER_PASS => cursor += 44,
                    OPCODE_COPY_TEXTURE_TO_BUFFER => cursor += 24,
                    OPCODE_RECORD_BUNDLE => cursor += 28,
                    OPCODE_EXECUTE_BUNDLES => {
                        let count = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap()) as usize;
                        cursor += 4 + count * 4;
                    }
                    other => panic!("unknown opcode {other}"),
                }
            }
            assert_eq!(cursor + total_data_len, encoded.len());

            let expected_opcodes = vec![
                OPCODE_CREATE_BUFFER,
                OPCODE_WRITE_BUFFER,
                OPCODE_CREATE_BUFFER,
                OPCODE_WRITE_BUFFER,
                OPCODE_CREATE_TEXTURE,
                OPCODE_CREATE_BUFFER,
                OPCODE_CREATE_PIPELINE,
                OPCODE_RENDER_PASS,
                OPCODE_COPY_TEXTURE_TO_BUFFER,
            ];
            assert_eq!(opcodes, expected_opcodes);
        }

        // Cross-variant comparison: Commands 0, 1, 2, 3, 4, 5, 7, 8 must be identical
        let p_false = build_affine_rows_layout_counterexample_submission(false);
        let p_true = build_affine_rows_layout_counterexample_submission(true);
        assert_eq!(p_false.commands[0], p_true.commands[0]);
        assert_eq!(p_false.commands[1], p_true.commands[1]); // Uniform bytes identical
        assert_eq!(p_false.commands[2], p_true.commands[2]);
        assert_eq!(p_false.commands[3], p_true.commands[3]);
        assert_eq!(p_false.commands[4], p_true.commands[4]);
        assert_eq!(p_false.commands[5], p_true.commands[5]);
        assert_eq!(p_false.commands[7], p_true.commands[7]);
        assert_eq!(p_false.commands[8], p_true.commands[8]);

        // Hand-computed raster pixel coordinates proof matching oracle:
        let ndc_to_pixel = |x: f32, y: f32| -> (u32, u32) {
            let px = ((x + 1.0) * 0.5 * 64.0 - 0.5).round() as u32;
            let py = ((1.0 - y) * 0.5 * 64.0 - 0.5).round() as u32;
            (px, py)
        };
        // Correct AffineRows: scaled by 0.5 and translated by +0.5 x
        assert_eq!(ndc_to_pixel(0.5, 0.25), (48, 24)); // Top vertex
        assert_eq!(ndc_to_pixel(0.25, -0.25), (40, 40)); // Bottom-left vertex
        assert_eq!(ndc_to_pixel(0.75, -0.25), (56, 40)); // Bottom-right vertex
        assert_eq!(ndc_to_pixel(0.5, 0.0), (48, 32)); // Center of transformed triangle (interior, green)
        assert_eq!(ndc_to_pixel(0.0, 0.0), (32, 32)); // Origin (outside, black)

        // Wrong mat4x3 layout: translation lost, scaled by 0.5 centered at origin
        assert_eq!(ndc_to_pixel(0.0, 0.25), (32, 24)); // Top vertex
        assert_eq!(ndc_to_pixel(-0.25, -0.25), (24, 40)); // Bottom-left vertex
        assert_eq!(ndc_to_pixel(0.25, -0.25), (40, 40)); // Bottom-right vertex
        assert_eq!(ndc_to_pixel(0.0, 0.0), (32, 32)); // Center of unshifted triangle (interior, green)
        assert_eq!(ndc_to_pixel(0.5, 0.0), (48, 32)); // Shifted center (outside, black)
    }

    #[test]
    fn test_affine_rows_batch_packet() {
        // 1. Validation errors and overflow regressions
        assert!(f3d_build_affine_rows_batch_packet(&[], 64, 64).is_err());
        assert!(f3d_build_affine_rows_batch_packet(&[1.0; 11], 64, 64).is_err());
        assert!(f3d_build_affine_rows_batch_packet(&[1.0; 13], 64, 64).is_err());
        assert!(f3d_build_affine_rows_batch_packet(&[1.0; 12], 0, 64).is_err());
        assert!(f3d_build_affine_rows_batch_packet(&[1.0; 12], 64, 0).is_err());
        // Narrow regression: width = u32::MAX / 4 overflows checked_add(255)
        assert!(f3d_build_affine_rows_batch_packet(&[1.0; 12], u32::MAX / 4, 1).is_err());

        // 2. Distinct 2-transform batch upload & padding proof
        let row0 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let row1 = [0.5f32, 0.0, 0.0, 0.25, 0.0, 0.5, 0.0, -0.25, 0.0, 0.0, 1.0, 0.0];
        let mut row0_bytes = [0u8; 48];
        for (j, &val) in row0.iter().enumerate() {
            row0_bytes[j * 4..j * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
        let mut row1_bytes = [0u8; 48];
        for (j, &val) in row1.iter().enumerate() {
            row1_bytes[j * 4..j * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
        let mut two_rows = Vec::with_capacity(24);
        two_rows.extend_from_slice(&row0);
        two_rows.extend_from_slice(&row1);

        let packet = build_affine_rows_batch_submission(&two_rows, 64, 64).expect("valid 2-row batch");
        assert_eq!(packet.commands.len(), 7 + 2 + 1); // 7 setup + 2 draws + 1 copy = 10 commands

        // Check uniform buffer upload: exact LE bytes for each row and zero padding
        match &packet.commands[1] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 1);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 2 * 256);
                // Exact LE bytes serialization match for each input row
                assert_eq!(&data[0..48], &row0_bytes[..]);
                assert_eq!(&data[256..304], &row1_bytes[..]);
                assert!(data[48..256].iter().all(|&b| b == 0), "slot 0 padding must be zero");
                assert!(data[304..512].iter().all(|&b| b == 0), "slot 1 padding must be zero");
            }
            other => panic!("expected WriteBuffer for uniform, got {other:?}"),
        }

        // Check draws: single render pass, pipeline 100, dynamic offsets
        match &packet.commands[7] {
            GpuCommand::RenderPass { pipeline_id, uniform_dynamic_offset, pass_flags, load_op, .. } => {
                assert_eq!(*pipeline_id, 100);
                assert_eq!(*uniform_dynamic_offset, 0);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*load_op, LOAD_OP_CLEAR);
            }
            other => panic!("expected RenderPass draw 0, got {other:?}"),
        }
        match &packet.commands[8] {
            GpuCommand::RenderPass { pipeline_id, uniform_dynamic_offset, pass_flags, load_op, .. } => {
                assert_eq!(*pipeline_id, 100);
                assert_eq!(*uniform_dynamic_offset, 256);
                assert_eq!(*pass_flags, PASS_FLAG_NONE);
                assert_eq!(*load_op, LOAD_OP_LOAD);
            }
            other => panic!("expected RenderPass draw 1, got {other:?}"),
        }

        // 3. Full 4000-draw batch
        let mut four_thousand_rows = vec![0.0f32; 4000 * 12];
        for i in 0..4000 {
            let offset = i * 12;
            four_thousand_rows[offset..offset + 12].copy_from_slice(&row0);
        }
        let packet_4000 = build_affine_rows_batch_submission(&four_thousand_rows, 256, 256).expect("valid 4000-row batch");
        assert_eq!(packet_4000.commands.len(), 7 + 4000 + 1); // 4008 commands

        match &packet_4000.commands[0] {
            GpuCommand::CreateBuffer { size, .. } => {
                assert_eq!(*size, 4000 * 256);
            }
            other => panic!("expected CreateBuffer, got {other:?}"),
        }

        // 4. Binary encoding check
        let wasm_bytes = f3d_build_affine_rows_batch_packet(&two_rows, 64, 64).unwrap();
        assert_eq!(&wasm_bytes[0..4], &PACKET_MAGIC);
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
    fn test_lower_plan_supports_depth_stencil_attachment() {
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

        let commands = lower_plan(&plan).expect("depth lowering must succeed");
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            GpuCommand::RenderPassDepth {
                target_type,
                target_id,
                pipeline_id,
                vertex_count,
                depth_target_id,
                depth_load_op,
                depth_store_op,
                depth_clear_value,
                depth_read_only,
                ..
            } => {
                assert_eq!(*target_type, TARGET_OFFSCREEN);
                assert_eq!(*target_id, 10);
                assert_eq!(*pipeline_id, 100);
                assert_eq!(*vertex_count, 3);
                assert_eq!(*depth_target_id, 99);
                assert_eq!(*depth_load_op, LOAD_OP_CLEAR);
                assert_eq!(*depth_store_op, STORE_OP_STORE);
                assert_eq!(*depth_clear_value, 1.0);
                assert!(!*depth_read_only);
            }
            other => panic!("Expected RenderPassDepth command, got {other:?}"),
        }
    }

    #[test]
    fn test_lower_plan_readonly_depth_defaults_to_load_op_load() {
        use f3d_graph::{
            pass::{ColorAttachment, DepthStencilAttachment, Draw, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut p = Pass::new_render(PassId::new(1), "readonly_depth_pass");
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));
        let mut dsa = DepthStencilAttachment::new_depth_clear(ResourceId::new(99), 1.0);
        dsa.depth_read_only = true;
        dsa.depth_load_op = None;
        p.depth_stencil_attachment = Some(dsa);
        p.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("readonly depth lowering must succeed");
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            GpuCommand::RenderPassDepth {
                depth_load_op,
                depth_read_only,
                ..
            } => {
                assert_eq!(
                    *depth_load_op,
                    LOAD_OP_LOAD,
                    "depth_read_only=true with None load_op must resolve to LOAD_OP_LOAD (1)"
                );
                assert!(*depth_read_only, "depth_read_only must be true");
            }
            other => panic!("Expected RenderPassDepth command, got {other:?}"),
        }

        let mut packet = GpuSubmissionPacket::new();
        packet.push(commands[0].clone());
        let bytes = packet.encode().expect("binary packet encoding must succeed");
        // Header is 16 bytes. Command 0 opcode is u16 at 16..18 (2 bytes).
        // Command 0 payload fields start at byte 18.
        // packed_depth_ops is at payload offset 48, which is byte 66..70.
        let packed_depth = u32::from_le_bytes(bytes[66..70].try_into().unwrap());
        assert_eq!(unpack_depth_load_op(packed_depth), LOAD_OP_LOAD);
        assert!(unpack_depth_read_only(packed_depth));
        // Byte 66: depth_load_op (u8) = 1 (LOAD_OP_LOAD).
        // Byte 68: depth_read_only (u8) = 1.
        assert_eq!(bytes[66], LOAD_OP_LOAD as u8, "byte 66 (depth_load_op) must be LOAD_OP_LOAD = 1");
        assert_eq!(bytes[68], 1u8, "byte 68 (depth_read_only) must be 1");
    }

    #[test]
    fn test_lower_plan_rejects_stencil_attachment() {
        use f3d_graph::{
            pass::{ColorAttachment, DepthStencilAttachment, Draw, LoadOp, Pass, PassId, StoreOp},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut p = Pass::new_render(PassId::new(1), "stencil_pass");
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));
        p.depth_stencil_attachment = Some(DepthStencilAttachment {
            target_id: ResourceId::new(99),
            view_subresource: f3d_graph::resource::SubresourceRange::full_texture(),
            depth_load_op: Some(LoadOp::Clear),
            depth_store_op: Some(StoreOp::Store),
            depth_clear_value: 1.0,
            depth_read_only: false,
            stencil_load_op: Some(LoadOp::Clear),
            stencil_store_op: Some(StoreOp::Store),
            stencil_clear_value: 0,
            stencil_read_only: false,
        });
        p.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        match lower_plan(&plan) {
            Err(PlanLoweringError::UnsupportedStencilAttachment { segment_name }) => {
                assert_eq!(segment_name, "stencil_pass");
            }
            other => panic!("Expected UnsupportedStencilAttachment error, got {other:?}"),
        }
    }

    #[test]
    fn test_lower_plan_rejects_bundle_depth_attachment() {
        use f3d_graph::{
            pass::{ColorAttachment, DepthStencilAttachment, Draw, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut p = Pass::new_render(PassId::new(1), "bundle_depth_pass");
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));
        p.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(99),
            1.0,
        ));
        p.draws.push(Draw::new_bundle(1, 42, 100, Vec::new()));

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        match lower_plan(&plan) {
            Err(PlanLoweringError::UnsupportedBundleDepthAttachment { segment_name }) => {
                assert_eq!(segment_name, "bundle_depth_pass");
            }
            other => panic!("Expected UnsupportedBundleDepthAttachment error, got {other:?}"),
        }
    }

    #[test]
    fn test_lower_plan_supports_empty_draw_with_depth_clear_and_color_load() {
        use f3d_graph::{
            pass::{ColorAttachment, DepthStencilAttachment, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut p = Pass::new_render(PassId::new(1), "empty_draw_depth_clear");
        p.color_attachments.push(ColorAttachment::new_load(ResourceId::new(10)));
        p.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(
            ResourceId::new(12),
            1.0,
        ));
        // Empty draws: valid depth-clear pass with color load
        assert!(p.draws.is_empty());

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("empty draw with depth clear must lower successfully");
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            GpuCommand::RenderPassDepth {
                target_id,
                vertex_count,
                load_op,
                store_op,
                pass_flags,
                depth_target_id,
                depth_load_op,
                depth_store_op,
                depth_clear_value,
                depth_read_only,
                ..
            } => {
                assert_eq!(*target_id, 10);
                assert_eq!(*vertex_count, 0);
                assert_eq!(*load_op, LOAD_OP_LOAD);
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
                assert_eq!(*depth_target_id, 12);
                assert_eq!(*depth_load_op, LOAD_OP_CLEAR);
                assert_eq!(*depth_store_op, STORE_OP_STORE);
                assert_eq!(*depth_clear_value, 1.0);
                assert!(!*depth_read_only);
            }
            other => panic!("Expected RenderPassDepth, got {other:?}"),
        }
    }

    #[test]
    fn test_lower_plan_rejects_malformed_depth_attachments() {
        use f3d_graph::{
            pass::{ColorAttachment, DepthStencilAttachment, Draw, LoadOp, Pass, PassId, StoreOp},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        // Case 1: Target ID 0
        let mut p1 = Pass::new_render(PassId::new(1), "invalid_target_0");
        p1.color_attachments.push(ColorAttachment::new_clear(ResourceId::new(10), [0.0; 4]));
        p1.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(ResourceId::new(0), 1.0));
        p1.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));
        let plan1 = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p1)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };
        assert!(matches!(
            lower_plan(&plan1),
            Err(PlanLoweringError::InvalidDepthAttachment { reason, .. }) if reason.contains("must not be zero")
        ));

        // Case 2: Depth clear value out of range (1.5)
        let mut p2 = Pass::new_render(PassId::new(2), "invalid_clear_val");
        p2.color_attachments.push(ColorAttachment::new_clear(ResourceId::new(10), [0.0; 4]));
        p2.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(ResourceId::new(12), 1.5));
        p2.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));
        let plan2 = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p2)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };
        assert!(matches!(
            lower_plan(&plan2),
            Err(PlanLoweringError::InvalidDepthAttachment { reason, .. }) if reason.contains("[0.0, 1.0]")
        ));

        // Case 3: Read-only depth attachment specifying LoadOp::Clear
        let mut p3 = Pass::new_render(PassId::new(3), "readonly_clear");
        p3.color_attachments.push(ColorAttachment::new_clear(ResourceId::new(10), [0.0; 4]));
        p3.depth_stencil_attachment = Some(DepthStencilAttachment {
            target_id: ResourceId::new(12),
            view_subresource: f3d_graph::resource::SubresourceRange::full_texture(),
            depth_load_op: Some(LoadOp::Clear),
            depth_store_op: Some(StoreOp::Store),
            depth_clear_value: 1.0,
            depth_read_only: true,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });
        p3.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));
        let plan3 = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p3)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };
        assert!(matches!(
            lower_plan(&plan3),
            Err(PlanLoweringError::InvalidDepthAttachment { reason, .. }) if reason.contains("cannot specify LoadOp::Clear")
        ));

        // Case 4: Color load + Depth load with empty draws (neither clears, no draws -> MissingDrawCommand)
        let mut p4 = Pass::new_render(PassId::new(4), "color_load_depth_load_no_draws");
        p4.color_attachments.push(ColorAttachment::new_load(ResourceId::new(10)));
        p4.depth_stencil_attachment = Some(DepthStencilAttachment {
            target_id: ResourceId::new(12),
            view_subresource: f3d_graph::resource::SubresourceRange::full_texture(),
            depth_load_op: Some(LoadOp::Load),
            depth_store_op: Some(StoreOp::Store),
            depth_clear_value: 1.0,
            depth_read_only: false,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });
        assert!(p4.draws.is_empty());
        let plan4 = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p4)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };
        assert!(matches!(
            lower_plan(&plan4),
            Err(PlanLoweringError::MissingDrawCommand { .. })
        ));
    }

    #[test]
    fn test_lower_plan_depth_read_only_numeric_defaults_and_truth() {
        use f3d_graph::{
            pass::{ColorAttachment, DepthStencilAttachment, Draw, Pass, PassId},
            plan::{ExecutionPlan, PlanSegment},
            resource::ResourceId,
        };

        let mut p = Pass::new_render(PassId::new(1), "readonly_truth_pass");
        p.color_attachments.push(ColorAttachment::new_clear(ResourceId::new(10), [0.0; 4]));
        p.depth_stencil_attachment = Some(DepthStencilAttachment {
            target_id: ResourceId::new(12),
            view_subresource: f3d_graph::resource::SubresourceRange::full_texture(),
            depth_load_op: None,
            depth_store_op: None,
            depth_clear_value: 1.0,
            depth_read_only: true,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });
        p.draws.push(Draw::new(1, 100, 3, 0, Vec::new()));

        let plan = ExecutionPlan {
            segments: vec![PlanSegment::from_pass(&p)],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: Vec::new(),
        };

        let commands = lower_plan(&plan).expect("read-only depth plan must lower successfully");
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            GpuCommand::RenderPassDepth {
                depth_target_id,
                depth_load_op,
                depth_store_op,
                depth_read_only,
                ..
            } => {
                assert_eq!(*depth_target_id, 12);
                assert!(*depth_read_only);
                // Wire carries numeric defaults
                assert_eq!(*depth_load_op, LOAD_OP_LOAD);
                assert_eq!(*depth_store_op, STORE_OP_STORE);
            }
            other => panic!("Expected RenderPassDepth, got {other:?}"),
        }
    }

    #[test]
    fn test_depth_wire_protocol_roundtrip_and_overlapping_submission() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        // 1. Test CreatePipelineDepth binary encoding (44 bytes payload)
        let mut packet = GpuSubmissionPacket::new();
        packet.push(GpuCommand::CreatePipelineDepth {
            pipeline_id: 100,
            wgsl_code: "// test shader".to_string(),
            target_format: TARGET_FORMAT_RGBA8UNORM,
            has_vertex_buffer: true,
            has_uniform_buffer: true,
            uniform_size: 16,
            vertex_stride: 20,
            depth_format: TARGET_FORMAT_DEPTH32FLOAT,
            depth_write_enabled: true,
            depth_compare: DEPTH_COMPARE_LESS,
        });

        let encoded = packet.encode().expect("encoding CreatePipelineDepth");
        assert_eq!(&encoded[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes(encoded[4..6].try_into().unwrap()), PACKET_VERSION);
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 1); // cmd_count = 1

        let op = u16::from_le_bytes(encoded[16..18].try_into().unwrap());
        assert_eq!(op, OPCODE_CREATE_PIPELINE_DEPTH);
        assert_eq!(u32::from_le_bytes(encoded[18..22].try_into().unwrap()), 100); // pipeline_id
        assert_eq!(u32::from_le_bytes(encoded[30..34].try_into().unwrap()), TARGET_FORMAT_RGBA8UNORM);
        assert_eq!(u32::from_le_bytes(encoded[50..54].try_into().unwrap()), TARGET_FORMAT_DEPTH32FLOAT);
        assert_eq!(u32::from_le_bytes(encoded[54..58].try_into().unwrap()), 1); // depth_write_enabled = true
        assert_eq!(u32::from_le_bytes(encoded[58..62].try_into().unwrap()), DEPTH_COMPARE_LESS);

        // 2. Test RenderPassDepth binary encoding (56 bytes payload)
        let mut pass_packet = GpuSubmissionPacket::new();
        pass_packet.push(GpuCommand::RenderPassDepth {
            target_type: TARGET_OFFSCREEN,
            target_id: 10,
            clear_color: [0.1, 0.2, 0.3, 1.0],
            pipeline_id: 100,
            vertex_buffer_id: 2,
            vertex_count: 3,
            uniform_dynamic_offset: 256,
            uniform_buffer_id: 1,
            load_op: LOAD_OP_CLEAR,
            store_op: STORE_OP_STORE,
            pass_flags: PASS_FLAG_NEW_PASS,
            depth_target_id: 12,
            depth_load_op: LOAD_OP_CLEAR,
            depth_store_op: STORE_OP_STORE,
            depth_clear_value: 1.0,
            depth_read_only: false,
        });

        let enc_pass = pass_packet.encode().expect("encoding RenderPassDepth");
        let op_pass = u16::from_le_bytes(enc_pass[16..18].try_into().unwrap());
        assert_eq!(op_pass, OPCODE_RENDER_PASS_DEPTH);
        let packed_color = u32::from_le_bytes(enc_pass[18..22].try_into().unwrap());
        assert_eq!(unpack_target_kind(packed_color), TARGET_OFFSCREEN);
        assert_eq!(unpack_load_op(packed_color), LOAD_OP_CLEAR);
        assert_eq!(unpack_store_op(packed_color), STORE_OP_STORE);
        assert_eq!(unpack_pass_flags(packed_color), PASS_FLAG_NEW_PASS);

        assert_eq!(u32::from_le_bytes(enc_pass[22..26].try_into().unwrap()), 10); // color target
        assert_eq!(u32::from_le_bytes(enc_pass[62..66].try_into().unwrap()), 12); // depth target
        let packed_depth = u32::from_le_bytes(enc_pass[66..70].try_into().unwrap());
        assert_eq!(unpack_depth_load_op(packed_depth), LOAD_OP_CLEAR);
        assert_eq!(unpack_depth_store_op(packed_depth), STORE_OP_STORE);
        assert!(!unpack_depth_read_only(packed_depth));
        let clear_depth = f32::from_le_bytes(enc_pass[70..74].try_into().unwrap());
        assert_eq!(clear_depth, 1.0);

        // 3. Test build_overlapping_depth_submission and canonical exports for all 6 scenarios
        let packet_0 = build_overlapping_depth_submission(0);
        let bytes_0 = packet_0.encode().expect("encode scenario 0");
        assert!(!bytes_0.is_empty());

        let packet_1 = build_overlapping_depth_submission(1);
        let bytes_1 = packet_1.encode().expect("encode scenario 1");
        assert!(!bytes_1.is_empty());

        let packet_2 = build_overlapping_depth_submission(2);
        let bytes_2 = packet_2.encode().expect("encode scenario 2");
        assert!(!bytes_2.is_empty());

        let packet_3 = build_overlapping_depth_submission(3);
        let bytes_3 = packet_3.encode().expect("encode scenario 3");
        assert!(!bytes_3.is_empty());

        let packet_4 = build_overlapping_depth_submission(4);
        let bytes_4 = packet_4.encode().expect("encode scenario 4");
        assert!(!bytes_4.is_empty());

        let packet_5 = build_overlapping_depth_submission(5);
        let bytes_5 = packet_5.encode().expect("encode scenario 5");
        assert!(!bytes_5.is_empty());

        // Verify Scenario 2 has two passes with depth load op on pass 2
        let pass_cmds_2: Vec<_> = packet_2
            .commands()
            .iter()
            .filter(|c| matches!(c, GpuCommand::RenderPassDepth { .. }))
            .collect();
        assert_eq!(pass_cmds_2.len(), 2);
        if let GpuCommand::RenderPassDepth { depth_load_op, load_op, .. } = pass_cmds_2[1] {
            assert_eq!(*depth_load_op, LOAD_OP_LOAD);
            assert_eq!(*load_op, LOAD_OP_LOAD);
        } else {
            panic!("Expected RenderPassDepth for pass 2");
        }

        // Verify Scenario 3 has disabled depth write and compare Always
        let pipeline_3 = packet_3
            .commands()
            .iter()
            .find(|c| matches!(c, GpuCommand::CreatePipelineDepth { .. }))
            .expect("pipeline in scenario 3");
        if let GpuCommand::CreatePipelineDepth { depth_write_enabled, depth_compare, .. } = pipeline_3 {
            assert!(!*depth_write_enabled);
            assert_eq!(*depth_compare, DEPTH_COMPARE_ALWAYS);
        } else {
            panic!("Expected CreatePipelineDepth");
        }

        // Verify Scenario 4 has two passes, pass 2 read-only, and pipeline 101 depth_write_enabled=false
        let pass_cmds_4: Vec<_> = packet_4
            .commands()
            .iter()
            .filter(|c| matches!(c, GpuCommand::RenderPassDepth { .. }))
            .collect();
        assert_eq!(pass_cmds_4.len(), 2);
        if let GpuCommand::RenderPassDepth { depth_read_only, depth_load_op, pipeline_id, .. } = pass_cmds_4[1] {
            assert!(*depth_read_only);
            assert_eq!(*depth_load_op, LOAD_OP_LOAD, "Scenario 4 pass 2 must emit depth_load_op LOAD_OP_LOAD (1)");
            assert_eq!(*pipeline_id, 101);
        } else {
            panic!("Expected RenderPassDepth for pass 2 in scenario 4");
        }
        let pipeline_ro_4 = packet_4
            .commands()
            .iter()
            .find(|c| matches!(c, GpuCommand::CreatePipelineDepth { pipeline_id: 101, .. }))
            .expect("pipeline 101 in scenario 4");
        if let GpuCommand::CreatePipelineDepth { depth_write_enabled, depth_compare, .. } = pipeline_ro_4 {
            assert!(!*depth_write_enabled);
            assert_eq!(*depth_compare, DEPTH_COMPARE_LESS);
        } else {
            panic!("Expected CreatePipelineDepth 101");
        }

        // Verify Scenario 5 has three passes: pass 1 draw, pass 2 zero-draw depth clear, pass 3 draw
        let pass_cmds_5: Vec<_> = packet_5
            .commands()
            .iter()
            .filter(|c| matches!(c, GpuCommand::RenderPassDepth { .. }))
            .collect();
        assert_eq!(pass_cmds_5.len(), 3);
        if let GpuCommand::RenderPassDepth { vertex_count, depth_load_op, .. } = pass_cmds_5[0] {
            assert_eq!(*vertex_count, 3);
            assert_eq!(*depth_load_op, LOAD_OP_CLEAR);
        }
        if let GpuCommand::RenderPassDepth { vertex_count, load_op, depth_load_op, .. } = pass_cmds_5[1] {
            assert_eq!(*vertex_count, 0);
            assert_eq!(*load_op, LOAD_OP_LOAD);
            assert_eq!(*depth_load_op, LOAD_OP_CLEAR);
        }
        if let GpuCommand::RenderPassDepth { vertex_count, load_op, depth_load_op, .. } = pass_cmds_5[2] {
            assert_eq!(*vertex_count, 3);
            assert_eq!(*load_op, LOAD_OP_LOAD);
            assert_eq!(*depth_load_op, LOAD_OP_LOAD);
        }

        // Both canonical aliases return identical bytes across all scenarios
        for s in 0..=5 {
            let canon = f3d_build_overlapping_depth_packet(s);
            let bridge = gpu_bridge_build_overlapping_depth_packet(s);
            assert_eq!(canon, bridge);
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
        let vertex_bytes = match &commands[1] {
            GpuCommand::WriteBuffer { data, .. } => data.as_slice(),
            other => panic!("expected vertex input, got {other:?}"),
        };
        let shader_bytes = match &commands[4] {
            GpuCommand::CreatePipeline { wgsl_code, .. } => wgsl_code.as_bytes(),
            other => panic!("expected shader input, got {other:?}"),
        };
        assert_eq!(total_data_len as usize, vertex_bytes.len() + shader_bytes.len());

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
        assert_eq!(cursor + total_data_len as usize, wire_bytes.len());
        let shader_start = cursor + vertex_bytes.len();
        assert_eq!(&wire_bytes[cursor..shader_start], vertex_bytes);
        assert_eq!(&wire_bytes[shader_start..], shader_bytes);

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

    #[test]
    fn test_write_texture_and_create_pipeline_textured_byte_encoding() {
        let mut packet = GpuSubmissionPacket::new();
        let pixel_data = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        packet.push(GpuCommand::WriteTexture {
            texture_id: 11, width: 2, height: 1, bytes_per_row: 8, data: pixel_data.clone(),
        });
        let encoded = packet.encode().expect("encoding WriteTexture must succeed");
        assert_eq!(&encoded[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes(encoded[4..6].try_into().unwrap()), PACKET_VERSION);
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(encoded[12..16].try_into().unwrap()), 8);

        // Opcode 17 record: 26 bytes (offset 16..42)
        assert_eq!(u16::from_le_bytes(encoded[16..18].try_into().unwrap()), OPCODE_WRITE_TEXTURE);
        assert_eq!(u32::from_le_bytes(encoded[18..22].try_into().unwrap()), 11);
        assert_eq!(u32::from_le_bytes(encoded[22..26].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(encoded[26..30].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(encoded[30..34].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(encoded[34..38].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(encoded[38..42].try_into().unwrap()), 8);
        assert_eq!(&encoded[42..50], &pixel_data[..]);

        // 2. Test CreatePipelineTextured byte encoding
        let mut pipe_packet = GpuSubmissionPacket::new();
        let wgsl_code = "@fragment fn fs() -> @location(0) vec4<f32> { return vec4(1.0); }";
        let code_bytes = wgsl_code.as_bytes();
        pipe_packet.push(GpuCommand::CreatePipelineTextured {
            pipeline_id: 100, wgsl_code: wgsl_code.to_string(), target_format: TARGET_FORMAT_RGBA8UNORM,
            has_vertex_buffer: true, has_uniform_buffer: true, uniform_size: 48, vertex_stride: 20,
            texture_id: 11, sampler_filter: SAMPLER_FILTER_NEAREST, address_mode: ADDRESS_MODE_CLAMP_TO_EDGE,
        });
        let enc_pipe = pipe_packet.encode().expect("encoding CreatePipelineTextured must succeed");
        assert_eq!(&enc_pipe[0..4], &PACKET_MAGIC);
        assert_eq!(u32::from_le_bytes(enc_pipe[8..12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(enc_pipe[12..16].try_into().unwrap()), code_bytes.len() as u32);

        // Opcode 18 record: 46 bytes (offset 16..62)
        assert_eq!(u16::from_le_bytes(enc_pipe[16..18].try_into().unwrap()), OPCODE_CREATE_PIPELINE_TEXTURED);
        assert_eq!(u32::from_le_bytes(enc_pipe[18..22].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(enc_pipe[22..26].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(enc_pipe[26..30].try_into().unwrap()), code_bytes.len() as u32);
        assert_eq!(u32::from_le_bytes(enc_pipe[30..34].try_into().unwrap()), TARGET_FORMAT_RGBA8UNORM);
        assert_eq!(u32::from_le_bytes(enc_pipe[34..38].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(enc_pipe[38..42].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(enc_pipe[42..46].try_into().unwrap()), 48);
        assert_eq!(u32::from_le_bytes(enc_pipe[46..50].try_into().unwrap()), 20);
        assert_eq!(u32::from_le_bytes(enc_pipe[50..54].try_into().unwrap()), 11);
        assert_eq!(u32::from_le_bytes(enc_pipe[54..58].try_into().unwrap()), SAMPLER_FILTER_NEAREST);
        assert_eq!(u32::from_le_bytes(enc_pipe[58..62].try_into().unwrap()), ADDRESS_MODE_CLAMP_TO_EDGE);
        assert_eq!(&enc_pipe[62..62 + code_bytes.len()], code_bytes);
    }

    #[test]
    fn test_write_texture_length_mismatch_error() {
        let mut packet = GpuSubmissionPacket::new();
        packet.push(GpuCommand::WriteTexture {
            texture_id: 11, width: 2, height: 2, bytes_per_row: 8, data: vec![0u8; 12],
        });
        match packet.encode().expect_err("encode must fail on length mismatch") {
            PacketEncodeError::TextureLengthMismatch { expected, actual } => {
                assert_eq!(expected, 16);
                assert_eq!(actual, 12);
            }
            other => panic!("expected TextureLengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_textured_affine_triangle_input_validation_errors() {
        let pixels = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let affine = AffineRows::identity();
        for (tex_w, tex_h, target_w, target_h) in [
            (0, 1, 65, 3), (2, 0, 65, 3), (2, 1, 0, 3), (2, 1, 65, 0),
            (u32::MAX, 1, 65, 3), (2, 1, u32::MAX, 3),
            (2, 1, u32::MAX / 4, 3), (2, 1, 64, u32::MAX),
        ] {
            assert!(matches!(
                build_textured_affine_triangle_packet(&pixels, tex_w, tex_h, &affine, target_w, target_h),
                Err(PacketEncodeError::InvalidDimensions(_))
            ), "accepted invalid dimensions {tex_w}x{tex_h} -> {target_w}x{target_h}");
        }
        assert!(matches!(
            build_textured_affine_triangle_packet(&pixels[..4], 2, 1, &affine, 65, 3),
            Err(PacketEncodeError::TextureLengthMismatch { expected: 8, actual: 4 })
        ));
        for len in [0, 11, 13] {
            assert!(f3d_build_textured_affine_triangle_packet(&pixels, 2, 1, &vec![0.0; len], 65, 3).is_err());
        }
    }

    #[test]
    fn test_textured_affine_triangle_packet_builder_and_exports() {
        // Inputs intentionally differ from the browser's 2x2/64x64 fixture.
        // A 65-pixel target requires 512-byte padded readback rows.
        let pixels = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let affine_data = [1.0f32, 0.0, 0.0, 7.0, 0.0, 2.0, 0.0, -3.0, 0.0, 0.0, 1.0, 0.0];
        let affine = AffineRows::new([1.0, 0.0, 0.0, 7.0], [0.0, 2.0, 0.0, -3.0], [0.0, 0.0, 1.0, 0.0]);
        let packet = build_textured_affine_triangle_packet(&pixels, 2, 1, &affine, 65, 3).unwrap();
        let cmds = packet.commands();
        assert!(matches!(&cmds[1], GpuCommand::WriteBuffer { data, .. }
            if data.as_slice() == affine.to_bytes()));
        assert!(matches!(&cmds[4], GpuCommand::CreateTexture { width: 65, height: 3, .. }));
        assert!(matches!(&cmds[5], GpuCommand::CreateBuffer { size: 1536, .. }));
        assert!(matches!(&cmds[7], GpuCommand::WriteTexture {
            width: 2, height: 1, bytes_per_row: 8, data, ..
        } if data.as_slice() == pixels));
        assert!(matches!(&cmds[10], GpuCommand::CopyTextureToBuffer { width: 65, height: 3, .. }));
        let encoded = packet.encode().unwrap();
        let native = f3d_build_textured_affine_triangle_packet(&pixels, 2, 1, &affine_data, 65, 3).unwrap();
        let alias = gpu_bridge_build_textured_affine_triangle_packet(&pixels, 2, 1, &affine_data, 65, 3).unwrap();
        assert_eq!(encoded, native);
        assert_eq!(native, alias);
    }

    #[test]
    fn test_pack_affine_rows_uniform_bytes_layout_and_validation() {
        // 1 transform: 12 floats = 48 bytes payload + 208 bytes zero padding = 256 bytes
        let floats: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
        let packed = pack_affine_rows_uniform_bytes(&floats).expect("packing 1 transform");
        assert_eq!(packed.len(), 256);
        for i in 0..12 {
            let val = f32::from_le_bytes(packed[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(val, (i + 1) as f32);
        }
        for &b in &packed[48..256] {
            assert_eq!(b, 0);
        }

        // 2 transforms = 512 bytes with 256-byte stride
        let floats2: Vec<f32> = (0..24).map(|i| (i + 1) as f32).collect();
        let packed2 = pack_affine_rows_uniform_bytes(&floats2).expect("packing 2 transforms");
        assert_eq!(packed2.len(), 512);
        let expected_row0: Vec<u8> = floats2[0..12].iter().flat_map(|f| f.to_le_bytes()).collect();
        let expected_row1: Vec<u8> = floats2[12..24].iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(&packed2[0..48], expected_row0.as_slice());
        assert_eq!(&packed2[256..304], expected_row1.as_slice());
        for &b in &packed2[48..256] {
            assert_eq!(b, 0);
        }
        for &b in &packed2[304..512] {
            assert_eq!(b, 0);
        }

        // Export parity on native
        assert_eq!(packed, f3d_pack_affine_rows_bytes(&floats).unwrap());

        // Length validation errors
        assert!(pack_affine_rows_uniform_bytes(&[]).is_err());
        assert!(pack_affine_rows_uniform_bytes(&[1.0; 11]).is_err());
        assert!(pack_affine_rows_uniform_bytes(&[1.0; 13]).is_err());
    }

    #[test]
    fn test_build_affine_rows_batch_frame_submission_commands_and_validation() {
        let floats: Vec<f32> = (0..24).map(|i| (i + 1) as f32).collect(); // 2 draws
        let packet = build_affine_rows_batch_frame_submission(&floats, 64, 64, 2)
            .expect("2-draw frame update submission");
        let cmds = packet.commands();
        // Exactly expected_draws + 2 commands = 4 commands (1 WriteBuffer + 2 RenderPass + 1 CopyTextureToBuffer)
        assert_eq!(cmds.len(), 4);

        // Command 0: WriteBuffer to uniform buffer 1, offset 0
        match &cmds[0] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, 1);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 512); // 2 * 256
                let expected = pack_affine_rows_uniform_bytes(&floats).unwrap();
                assert_eq!(data.as_slice(), expected.as_slice());
            }
            other => panic!("expected WriteBuffer, got {other:?}"),
        }

        // Commands 1..2: RenderPass on target 10, pipeline 100, vb 2, ub 1
        for (i, cmd) in cmds[1..3].iter().enumerate() {
            match cmd {
                GpuCommand::RenderPass {
                    target_type, target_id, pipeline_id, vertex_buffer_id, vertex_count,
                    uniform_buffer_id, uniform_dynamic_offset, load_op, pass_flags, ..
                } => {
                    assert_eq!(*target_type, TARGET_OFFSCREEN);
                    assert_eq!(*target_id, 10);
                    assert_eq!(*pipeline_id, 100);
                    assert_eq!(*vertex_buffer_id, 2);
                    assert_eq!(*vertex_count, 3);
                    assert_eq!(*uniform_buffer_id, 1);
                    assert_eq!(*uniform_dynamic_offset, (i as u32) * 256);
                    assert_eq!(*load_op, if i == 0 { LOAD_OP_CLEAR } else { LOAD_OP_LOAD });
                    assert_eq!(*pass_flags, if i == 0 { PASS_FLAG_NEW_PASS } else { PASS_FLAG_NONE });
                }
                other => panic!("expected RenderPass, got {other:?}"),
            }
        }

        // Command 3: CopyTextureToBuffer 10 -> 20
        match &cmds[3] {
            GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height, .. } => {
                assert_eq!(*texture_id, 10);
                assert_eq!(*buffer_id, 20);
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
            }
            other => panic!("expected CopyTextureToBuffer, got {other:?}"),
        }

        // Public export parity
        let encoded = packet.encode().unwrap();
        let exported = f3d_build_affine_rows_batch_frame_packet(&floats, 64, 64, 2).unwrap();
        assert_eq!(encoded, exported);

        // Validation: compact invalid inputs table
        for (slice, w, h, draws) in [
            (floats.as_slice(), 64, 64, 1),
            (floats.as_slice(), 64, 64, 3),
            (floats.as_slice(), 64, 64, 0),
            (floats.as_slice(), 0, 64, 2),
            (floats.as_slice(), 64, 0, 2),
            (floats.as_slice(), u32::MAX, 64, 2),
            (floats.as_slice(), 64, u32::MAX, 2),
            (&[1.0; 23][..], 64, 64, 2),
        ] {
            assert!(matches!(
                build_affine_rows_batch_frame_submission(slice, w, h, draws),
                Err(PacketEncodeError::InvalidDimensions(_))
            ));
        }
    }

    #[test]
    fn bundle_batch_wire_offsets_and_overflow() {
        let make = |count, base, stride| GpuCommand::RecordBundleBatch {
            bundle_id: 200, pipeline_id: 100, vertex_buffer_id: 2, vertex_count: 3,
            uniform_dynamic_offset: base, uniform_buffer_id: 1, target_format: 2,
            draw_count: count, dynamic_offset_stride: stride,
        };
        let mut packet = GpuSubmissionPacket::new();
        packet.push(make(3996, 0, 256));
        let bytes = packet.encode().unwrap();
        assert_eq!(bytes.len(), 16 + 38);
        assert_eq!(&bytes[16..18], &19u16.to_le_bytes());
        let fields: Vec<u32> = bytes[18..].chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(fields, [200, 100, 2, 3, 0, 1, 2, 3996, 256]);
        for (count, base, stride) in [(2, u32::MAX, 1), (3, 0, u32::MAX)] {
            let mut invalid = GpuSubmissionPacket::new();
            invalid.push(make(count, base, stride));
            assert!(invalid.encode().is_err());
        }
        for count in [0, 1] {
            let mut edge = GpuSubmissionPacket::new();
            edge.push(make(count, u32::MAX, u32::MAX));
            assert!(edge.encode().is_ok()); // No offset progression for zero/one draw.
        }
    }

    #[test]
    fn affine_bundle_replay_preserves_tail_and_skips_recording() {
        let rows = vec![1.0; 4000 * 12];
        let packet = build_affine_rows_bundle_submission(&rows, 256, 256, 4000, 4, true).unwrap();
        let commands = packet.commands();
        assert_eq!(commands.len(), 9);
        assert!(matches!(&commands[0], GpuCommand::WriteBuffer { buffer_id: 1, offset: 0, data }
            if *data == pack_affine_rows_uniform_bytes(&rows).unwrap()));
        assert!(matches!(commands[1], GpuCommand::RecordBundleBatch {
            bundle_id: 200, draw_count: 3996, dynamic_offset_stride: 256, ..
        }));
        assert!(matches!(commands[2], GpuCommand::RenderPass {
            vertex_count: 0, target_id: 10, load_op: LOAD_OP_CLEAR,
            pass_flags: PASS_FLAG_NEW_PASS, ..
        }));
        assert!(matches!(&commands[3], GpuCommand::ExecuteBundles { bundle_ids }
            if bundle_ids == &[200]));
        for (i, command) in commands[4..8].iter().enumerate() {
            assert!(matches!(command, GpuCommand::RenderPass {
                vertex_count: 3, uniform_dynamic_offset, load_op: LOAD_OP_LOAD,
                pass_flags: PASS_FLAG_NONE, ..
            } if *uniform_dynamic_offset == (3996 + i as u32) * 256));
        }
        assert!(matches!(commands[8], GpuCommand::CopyTextureToBuffer {
            texture_id: 10, buffer_id: 20, width: 256, height: 256, ..
        }));
        assert_eq!(packet.encode().unwrap(),
            f3d_build_affine_rows_bundle_packet(&rows, 256, 256, 4000, 4, true).unwrap());
        let replay = build_affine_rows_bundle_submission(&rows, 256, 256, 4000, 4, false).unwrap();
        assert_eq!(replay.commands().len(), 8);
        assert!(!replay.commands().iter().any(|c| matches!(c, GpuCommand::RecordBundleBatch { .. })));
        for (w, h, count, tail) in [(0, 256, 4000, 4), (256, 0, 4000, 4),
            (u32::MAX, 1, 4000, 4), (256, u32::MAX, 4000, 4),
            (256, 256, 3999, 4), (256, 256, 4000, 0), (256, 256, 4000, 4000)] {
            assert!(build_affine_rows_bundle_submission(&rows, w, h, count, tail, false).is_err());
        }
        assert!(build_affine_rows_bundle_submission(&[], 1, 1, 2, 1, false).is_err());
        assert!(build_affine_rows_bundle_submission(&rows[..23], 1, 1, 2, 1, false).is_err());
    }

    #[test]
    fn test_opcode20_buffer_to_buffer_wire_encoding() {
        let mut packet = GpuSubmissionPacket::new();
        packet.push(GpuCommand::CopyBufferToBuffer {
            source_buffer_id: 610,
            source_offset: 16,
            destination_buffer_id: 611,
            destination_offset: 32,
            size: 64,
            epoch: Epoch::from_words(0x1234_5678, 0x9ABC_DEF0),
        });

        let encoded = packet.encode().expect("encoding CopyBufferToBuffer must succeed");
        // Header: 16 bytes (magic 4, ver 2, flags 2, cmd_count 4, data_len 4)
        assert_eq!(&encoded[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes(encoded[4..6].try_into().unwrap()), PACKET_VERSION);
        assert_eq!(u16::from_le_bytes(encoded[6..8].try_into().unwrap()), 0); // flags
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 1); // cmd_count
        assert_eq!(u32::from_le_bytes(encoded[12..16].try_into().unwrap()), 0); // data_len

        // Command record: 42 bytes (offset 16..58)
        assert_eq!(encoded.len(), 58);
        assert_eq!(u16::from_le_bytes(encoded[16..18].try_into().unwrap()), OPCODE_COPY_BUFFER_TO_BUFFER);
        assert_eq!(u32::from_le_bytes(encoded[18..22].try_into().unwrap()), 610);
        assert_eq!(u64::from_le_bytes(encoded[22..30].try_into().unwrap()), 16);
        assert_eq!(u32::from_le_bytes(encoded[30..34].try_into().unwrap()), 611);
        assert_eq!(u64::from_le_bytes(encoded[34..42].try_into().unwrap()), 32);
        assert_eq!(u64::from_le_bytes(encoded[42..50].try_into().unwrap()), 64);
        let expected_epoch = ((0x1234_5678u64) << 32) | (0x9ABC_DEF0u64);
        assert_eq!(u64::from_le_bytes(encoded[50..58].try_into().unwrap()), expected_epoch);

        // Zero-size copy is legal when alignment, distinct handles, and bounds are valid
        let mut zero_copy_packet = GpuSubmissionPacket::new();
        zero_copy_packet.push(GpuCommand::CopyBufferToBuffer {
            source_buffer_id: 1,
            source_offset: 0,
            destination_buffer_id: 2,
            destination_offset: 0,
            size: 0,
            epoch: Epoch::ZERO,
        });
        assert!(zero_copy_packet.encode().is_ok());
    }

    #[test]
    fn test_opcode20_buffer_to_buffer_wire_alignment_and_overflow_validation() {
        // Source and destination buffer IDs must be distinct
        let mut same_buf = GpuSubmissionPacket::new();
        same_buf.push(GpuCommand::CopyBufferToBuffer {
            source_buffer_id: 1,
            source_offset: 0,
            destination_buffer_id: 1,
            destination_offset: 4,
            size: 4,
            epoch: Epoch::ZERO,
        });
        assert!(matches!(
            same_buf.encode(),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));

        for (src_off, dst_off, size) in [
            (1, 0, 4),
            (0, 2, 4),
            (0, 0, 3),
            (u64::MAX - 3, 0, 4),
            (0, u64::MAX - 3, 4),
            (u64::MAX, 0, 0),
        ] {
            let mut packet = GpuSubmissionPacket::new();
            packet.push(GpuCommand::CopyBufferToBuffer {
                source_buffer_id: 1,
                source_offset: src_off,
                destination_buffer_id: 2,
                destination_offset: dst_off,
                size,
                epoch: Epoch::ZERO,
            });
            assert!(
                matches!(packet.encode(), Err(PacketEncodeError::InvalidDimensions(_))),
                "expected encode failure for ({src_off}, {dst_off}, {size})"
            );
        }
    }

    #[test]
    fn test_lower_plan_buffer_to_buffer() {
        use f3d_graph::{CopyCommand, ExecutionPlan, PassId, PassKind, PlanSegment, ResourceId};

        let copy_seg = PlanSegment {
            pass_id: PassId::new(1),
            name: "b2b_copy".to_string(),
            kind: PassKind::Copy,
            color_attachments: vec![],
            depth_stencil_attachment: None,
            draws: vec![],
            dispatches: vec![],
            copies: vec![CopyCommand::BufferToBuffer {
                src: ResourceId::new(610),
                src_offset: 1024,
                dst: ResourceId::new(611),
                dst_offset: 2048,
                size: 512,
            }],
            versioned_reads: vec![],
            versioned_writes: vec![],
        };
        let plan = ExecutionPlan {
            segments: vec![copy_seg],
            canvas_epoch: Some(Epoch::from_words(0, 42)),
            pass_count: 1,
            split_count: 0,
            split_reasons: vec![],
        };

        let lowered = lower_plan(&plan).expect("lowering valid BufferToBuffer must succeed");
        assert_eq!(lowered.len(), 1);
        match &lowered[0] {
            GpuCommand::CopyBufferToBuffer {
                source_buffer_id,
                source_offset,
                destination_buffer_id,
                destination_offset,
                size,
                epoch,
            } => {
                assert_eq!(*source_buffer_id, 610);
                assert_eq!(*source_offset, 1024u64);
                assert_eq!(*destination_buffer_id, 611);
                assert_eq!(*destination_offset, 2048u64);
                assert_eq!(*size, 512u64);
                assert_eq!(*epoch, Epoch::from_words(0, 42));
            }
            other => panic!("expected CopyBufferToBuffer, got {other:?}"),
        }

        for (src_off, dst_off, size) in [
            (1, 0, 4),
            (0, 2, 4),
            (0, 0, 3),
            (u64::MAX - 3, 0, 4),
            (0, u64::MAX - 3, 4),
        ] {
            let invalid_seg = PlanSegment {
                pass_id: PassId::new(1),
                name: "invalid_b2b".to_string(),
                kind: PassKind::Copy,
                color_attachments: vec![],
                depth_stencil_attachment: None,
                draws: vec![],
                dispatches: vec![],
                copies: vec![CopyCommand::BufferToBuffer {
                    src: ResourceId::new(1),
                    src_offset: src_off,
                    dst: ResourceId::new(2),
                    dst_offset: dst_off,
                    size,
                }],
                versioned_reads: vec![],
                versioned_writes: vec![],
            };
            let invalid_plan = ExecutionPlan {
                segments: vec![invalid_seg],
                canvas_epoch: None,
                pass_count: 1,
                split_count: 0,
                split_reasons: vec![],
            };
            assert!(
                matches!(lower_plan(&invalid_plan), Err(PlanLoweringError::InvalidCopyCommand { .. })),
                "expected InvalidCopyCommand for ({src_off}, {dst_off}, {size})"
            );
        }

        // Source and destination buffers must be distinct
        let same_buf_seg = PlanSegment {
            pass_id: PassId::new(1),
            name: "same_buf_b2b".to_string(),
            kind: PassKind::Copy,
            color_attachments: vec![],
            depth_stencil_attachment: None,
            draws: vec![],
            dispatches: vec![],
            copies: vec![CopyCommand::BufferToBuffer {
                src: ResourceId::new(1),
                src_offset: 0,
                dst: ResourceId::new(1),
                dst_offset: 4,
                size: 4,
            }],
            versioned_reads: vec![],
            versioned_writes: vec![],
        };
        let same_buf_plan = ExecutionPlan {
            segments: vec![same_buf_seg],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: vec![],
        };
        assert!(matches!(
            lower_plan(&same_buf_plan),
            Err(PlanLoweringError::InvalidCopyCommand { .. })
        ));

        let unsupp_seg = PlanSegment {
            pass_id: PassId::new(1),
            name: "unsupp_copy".to_string(),
            kind: PassKind::Copy,
            color_attachments: vec![],
            depth_stencil_attachment: None,
            draws: vec![],
            dispatches: vec![],
            copies: vec![CopyCommand::TextureToTexture {
                src: ResourceId::new(1),
                src_mip: 0,
                src_layer: 0,
                dst: ResourceId::new(2),
                dst_mip: 0,
                dst_layer: 0,
                extent: [1, 1, 1],
            }],
            versioned_reads: vec![],
            versioned_writes: vec![],
        };
        let unsupp_plan = ExecutionPlan {
            segments: vec![unsupp_seg],
            canvas_epoch: None,
            pass_count: 1,
            split_count: 0,
            split_reasons: vec![],
        };
        assert!(matches!(
            lower_plan(&unsupp_plan),
            Err(PlanLoweringError::UnsupportedCopyCommand { .. })
        ));
    }

    #[test]
    fn test_build_buffer_copy_packet_and_exports() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let packet = build_buffer_copy_submission(&data, 0, 4, 8)
            .expect("build_buffer_copy_submission must succeed");
        let cmds = packet.commands();
        assert_eq!(cmds.len(), 4);

        assert_eq!(
            cmds[0],
            GpuCommand::CreateBuffer {
                buffer_id: 610,
                size: 16,
                usage: BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST,
            }
        );

        match &cmds[1] {
            GpuCommand::WriteBuffer { buffer_id, offset, data: payload } => {
                assert_eq!(*buffer_id, 610);
                assert_eq!(*offset, 0);
                assert_eq!(payload.as_slice(), &data[..]);
            }
            other => panic!("expected WriteBuffer, got {other:?}"),
        }

        assert_eq!(
            cmds[2],
            GpuCommand::CreateBuffer {
                buffer_id: 611,
                size: 16,
                usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
            }
        );

        assert_eq!(
            cmds[3],
            GpuCommand::CopyBufferToBuffer {
                source_buffer_id: 610,
                source_offset: 0,
                destination_buffer_id: 611,
                destination_offset: 4,
                size: 8,
                epoch: Epoch::ZERO,
            }
        );

        let encoded = packet.encode().expect("encoding packet must succeed");
        let exported = f3d_build_buffer_copy_packet(&data, 0, 4, 8).expect("export must succeed");
        assert_eq!(encoded, exported);

        // Zero-size copy is legal when alignment, distinct handles, and bounds are valid
        assert!(build_buffer_copy_submission(&data, 0, 0, 0).is_ok());

        for (slice, src_off, dst_off, size) in [
            (&[][..], 0, 0, 0),
            (&[1, 2, 3][..], 0, 0, 0),
            (&data[..], 1, 0, 4),
            (&data[..], 0, 2, 4),
            (&data[..], 0, 0, 3),
            (&data[..], 12, 0, 8),
            (&data[..], 0, 12, 8),
            (&data[..], u32::MAX - 3, 0, 4),
            (&data[..], 0, u32::MAX - 3, 4),
        ] {
            assert!(
                matches!(
                    build_buffer_copy_submission(slice, src_off, dst_off, size),
                    Err(PacketEncodeError::InvalidDimensions(_))
                ),
                "expected InvalidDimensions for len={}, src_off={src_off}, dst_off={dst_off}, size={size}",
                slice.len()
            );
        }
    }

    #[test]
    fn test_build_render_then_copy_packet_and_exports() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let packet = build_render_then_copy_submission(&data, 0, 4, 8)
            .expect("build_render_then_copy_submission must succeed");
        let cmds = packet.commands();
        assert_eq!(cmds.len(), 5);

        assert_eq!(
            cmds[0],
            GpuCommand::CreateBuffer {
                buffer_id: 610,
                size: 16,
                usage: BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST,
            }
        );

        match &cmds[1] {
            GpuCommand::WriteBuffer { buffer_id, offset, data: payload } => {
                assert_eq!(*buffer_id, 610);
                assert_eq!(*offset, 0);
                assert_eq!(payload.as_slice(), &data[..]);
            }
            other => panic!("expected WriteBuffer, got {other:?}"),
        }

        assert_eq!(
            cmds[2],
            GpuCommand::CreateBuffer {
                buffer_id: 611,
                size: 16,
                usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
            }
        );

        assert_eq!(
            cmds[3],
            GpuCommand::RenderPass {
                target_type: TARGET_CANVAS,
                target_id: 0,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                pipeline_id: 0,
                vertex_buffer_id: 0,
                vertex_count: 0,
                uniform_dynamic_offset: 0,
                uniform_buffer_id: 0,
                load_op: LOAD_OP_CLEAR,
                store_op: STORE_OP_STORE,
                pass_flags: PASS_FLAG_NEW_PASS,
            }
        );

        assert_eq!(
            cmds[4],
            GpuCommand::CopyBufferToBuffer {
                source_buffer_id: 610,
                source_offset: 0,
                destination_buffer_id: 611,
                destination_offset: 4,
                size: 8,
                epoch: Epoch::ZERO,
            }
        );

        // Explicitly assert pass command directly precedes copy command
        assert!(matches!(cmds[3], GpuCommand::RenderPass { .. }));
        assert!(matches!(cmds[4], GpuCommand::CopyBufferToBuffer { .. }));

        let encoded = packet.encode().expect("encoding packet must succeed");
        let exported = f3d_build_render_then_copy_packet(&data, 0, 4, 8)
            .expect("export must succeed");
        assert_eq!(encoded, exported);
    }

    #[test]
    fn test_pack_affine_rows_storage_bytes_and_exports() {
        let neg_zero = -0.0f32;
        let nan_payload = f32::from_bits(0x7fc0_1234);
        assert!(nan_payload.is_nan());

        // 3 distinct records (36 floats)
        let record0 = [
            1.0f32, 0.0, 0.0, neg_zero,
            0.0, 1.0, 0.0, 2.5,
            0.0, 0.0, 1.0, -10.0,
        ];
        let record1 = [
            2.0f32, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, nan_payload,
            0.0, 0.0, 4.0, 0.0,
        ];
        let record2 = [
            5.0f32, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        let mut three_records = Vec::with_capacity(36);
        three_records.extend_from_slice(&record0);
        three_records.extend_from_slice(&record1);
        three_records.extend_from_slice(&record2);

        let packed = pack_affine_rows_storage_bytes(&three_records)
            .expect("pack_affine_rows_storage_bytes must succeed");
        assert_eq!(packed.len(), 3 * 48);

        // Verify exact expected bytes per record via f3d_core AffineRows::to_bytes
        let a0 = AffineRows::new(
            [record0[0], record0[1], record0[2], record0[3]],
            [record0[4], record0[5], record0[6], record0[7]],
            [record0[8], record0[9], record0[10], record0[11]],
        );
        let a1 = AffineRows::new(
            [record1[0], record1[1], record1[2], record1[3]],
            [record1[4], record1[5], record1[6], record1[7]],
            [record1[8], record1[9], record1[10], record1[11]],
        );
        let a2 = AffineRows::new(
            [record2[0], record2[1], record2[2], record2[3]],
            [record2[4], record2[5], record2[6], record2[7]],
            [record2[8], record2[9], record2[10], record2[11]],
        );

        assert_eq!(&packed[0..48], &a0.to_bytes());
        assert_eq!(&packed[48..96], &a1.to_bytes());
        assert_eq!(&packed[96..144], &a2.to_bytes());

        // Verify exact bit preservation: -0.0 in record0 (offset 12)
        assert_eq!(&packed[12..16], &(-0.0f32).to_le_bytes());
        assert_eq!(&packed[12..16], &[0x00, 0x00, 0x00, 0x80]);

        // Verify exact bit preservation: NaN payload in record1 (offset 48 + 28 = 76)
        assert_eq!(&packed[76..80], &nan_payload.to_le_bytes());
        assert_eq!(&packed[76..80], &[0x34, 0x12, 0xc0, 0x7f]);

        // Native export parity
        let exported = f3d_pack_affine_rows_storage_bytes(&three_records)
            .expect("f3d_pack_affine_rows_storage_bytes export must succeed");
        assert_eq!(packed, exported);

        // Rejection cases
        assert!(matches!(
            pack_affine_rows_storage_bytes(&[]),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        for bad_len in [1, 5, 11, 13, 23, 25] {
            let bad_input = vec![1.0f32; bad_len];
            assert!(
                matches!(
                    pack_affine_rows_storage_bytes(&bad_input),
                    Err(PacketEncodeError::InvalidDimensions(_))
                ),
                "expected rejection for length {bad_len}"
            );
        }

        // 4000 records -> 192,000 bytes
        let large_input = vec![1.0f32; 4000 * 12];
        let large_packed = pack_affine_rows_storage_bytes(&large_input)
            .expect("packing 4000 records must succeed");
        assert_eq!(large_packed.len(), 192_000);
        let expected_single_record = AffineRows::new(
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ).to_bytes();
        for chunk in large_packed.chunks_exact(48) {
            assert_eq!(chunk, &expected_single_record);
        }
    }

    #[test]
    fn test_affine_rows_storage_upload_init_and_frame_packets() {
        let neg_zero = -0.0f32;
        let nan_payload = f32::from_bits(0x7fc0_1234);

        let record0 = [
            1.0f32, 0.0, 0.0, neg_zero,
            0.0, 1.0, 0.0, 2.5,
            0.0, 0.0, 1.0, -10.0,
        ];
        let record1 = [
            2.0f32, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, nan_payload,
            0.0, 0.0, 4.0, 0.0,
        ];
        let record2 = [
            5.0f32, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        let mut three_records = Vec::with_capacity(36);
        three_records.extend_from_slice(&record0);
        three_records.extend_from_slice(&record1);
        three_records.extend_from_slice(&record2);

        // 1. Init packet tests: allocates 610 (COPY_SRC|COPY_DST), 611 (MAP_READ|COPY_DST), 612 (MAP_READ|COPY_DST)
        let init_packet = build_affine_rows_storage_upload_init_submission(3)
            .expect("build_affine_rows_storage_upload_init_submission must succeed");
        let init_cmds = init_packet.commands();
        assert_eq!(init_cmds.len(), 3);

        match &init_cmds[0] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                assert_eq!(*size, 3 * 48);
                assert_eq!(*usage, BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer for slot 610, got {other:?}"),
        }
        match &init_cmds[1] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_DST_SLOT_0);
                assert_eq!(*size, 3 * 48);
                assert_eq!(*usage, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer for slot 611, got {other:?}"),
        }
        match &init_cmds[2] {
            GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_DST_SLOT_1);
                assert_eq!(*size, 3 * 48);
                assert_eq!(*usage, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
            }
            other => panic!("expected CreateBuffer for slot 612, got {other:?}"),
        }

        // Init export parity
        let init_encoded = init_packet.encode().expect("encoding init packet must succeed");
        let init_exported = f3d_build_affine_rows_storage_upload_init_packet(3)
            .expect("init export must succeed");
        assert_eq!(init_encoded, init_exported);

        // 2. Frame packet tests for both slots: slot 611 and slot 612
        let expected_packed = pack_affine_rows_storage_bytes(&three_records)
            .expect("pack_affine_rows_storage_bytes must succeed");

        for &dest_slot in &[AFFINE_ROWS_STORAGE_DST_SLOT_0, AFFINE_ROWS_STORAGE_DST_SLOT_1] {
            let frame_packet = build_affine_rows_storage_upload_frame_submission(&three_records, 3, dest_slot)
                .expect("build_affine_rows_storage_upload_frame_submission must succeed");
            let frame_cmds = frame_packet.commands();
            assert_eq!(frame_cmds.len(), 2);

            // Command 0: WriteBuffer to 610
            match &frame_cmds[0] {
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                    assert_eq!(*offset, 0);
                    assert_eq!(data.len(), 3 * 48);
                    assert_eq!(data.as_slice(), expected_packed.as_slice());
                    // Exact bit preservation of -0.0 and NaN payload in payload
                    assert_eq!(&data[12..16], &[0x00, 0x00, 0x00, 0x80]);
                    assert_eq!(&data[76..80], &[0x34, 0x12, 0xc0, 0x7f]);
                }
                other => panic!("expected WriteBuffer, got {other:?}"),
            }

            // Command 1: CopyBufferToBuffer from 610 to dest_slot
            match &frame_cmds[1] {
                GpuCommand::CopyBufferToBuffer {
                    source_buffer_id,
                    source_offset,
                    destination_buffer_id,
                    destination_offset,
                    size,
                    epoch,
                } => {
                    assert_eq!(*source_buffer_id, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                    assert_eq!(*source_offset, 0);
                    assert_eq!(*destination_buffer_id, dest_slot);
                    assert_eq!(*destination_offset, 0);
                    assert_eq!(*size, 3 * 48);
                    assert_eq!(*epoch, Epoch::ZERO);
                }
                other => panic!("expected CopyBufferToBuffer, got {other:?}"),
            }

            // Frame export parity
            let frame_encoded = frame_packet.encode().expect("encoding frame packet must succeed");
            let frame_exported = f3d_build_affine_rows_storage_upload_frame_packet(&three_records, 3, dest_slot)
                .expect("frame export must succeed");
            assert_eq!(frame_encoded, frame_exported);
        }

        // 3. Rejections and guards
        assert!(matches!(
            build_affine_rows_storage_upload_init_submission(0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        assert!(matches!(
            build_affine_rows_storage_upload_frame_submission(&three_records, 0, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        // Invalid destination slots (not 611 or 612)
        for bad_slot in [0, 610, 613, 999] {
            assert!(
                matches!(
                    build_affine_rows_storage_upload_frame_submission(&three_records, 3, bad_slot),
                    Err(PacketEncodeError::InvalidDimensions(_))
                ),
                "expected rejection for slot {bad_slot}"
            );
        }
        // Count/length mismatch
        assert!(matches!(
            build_affine_rows_storage_upload_frame_submission(&three_records, 2, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        assert!(matches!(
            build_affine_rows_storage_upload_frame_submission(&three_records, 4, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        assert!(matches!(
            build_affine_rows_storage_upload_frame_submission(&[], 1, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));

        // 4. Scale & byte equality for 4000 records -> 192,000 bytes
        let large_init = build_affine_rows_storage_upload_init_submission(4000)
            .expect("init with 4000 records must succeed");
        assert_eq!(large_init.commands().len(), 3);
        match &large_init.commands()[0] {
            GpuCommand::CreateBuffer { buffer_id, size, .. } => {
                assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                assert_eq!(*size, 192_000);
            }
            other => panic!("expected CreateBuffer, got {other:?}"),
        }

        let large_input = vec![1.0f32; 4000 * 12];
        let large_expected = pack_affine_rows_storage_bytes(&large_input)
            .expect("pack_affine_rows_storage_bytes for 4000 records must succeed");
        assert_eq!(large_expected.len(), 192_000);

        let large_frame = build_affine_rows_storage_upload_frame_submission(
            &large_input,
            4000,
            AFFINE_ROWS_STORAGE_DST_SLOT_1,
        ).expect("frame with 4000 records must succeed");
        assert_eq!(large_frame.commands().len(), 2);
        match &large_frame.commands()[0] {
            GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                assert_eq!(*offset, 0);
                assert_eq!(data.len(), 192_000);
                assert_eq!(data.as_slice(), large_expected.as_slice());
            }
            other => panic!("expected WriteBuffer, got {other:?}"),
        }
        match &large_frame.commands()[1] {
            GpuCommand::CopyBufferToBuffer { size, destination_buffer_id, .. } => {
                assert_eq!(*size, 192_000);
                assert_eq!(*destination_buffer_id, AFFINE_ROWS_STORAGE_DST_SLOT_1);
            }
            other => panic!("expected CopyBufferToBuffer, got {other:?}"),
        }
    }

    #[test]
    fn test_affine_rows_storage_upload_callback_ops() {
        let neg_zero = -0.0f32;
        let nan_payload = f32::from_bits(0x7fc0_1234);

        let record0 = [
            1.0f32, 0.0, 0.0, neg_zero,
            0.0, 1.0, 0.0, 2.5,
            0.0, 0.0, 1.0, -10.0,
        ];
        let record1 = [
            2.0f32, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, nan_payload,
            0.0, 0.0, 4.0, 0.0,
        ];
        let record2 = [
            5.0f32, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        let mut three_records = Vec::with_capacity(36);
        three_records.extend_from_slice(&record0);
        three_records.extend_from_slice(&record1);
        three_records.extend_from_slice(&record2);

        let expected_packed = pack_affine_rows_storage_bytes(&three_records)
            .expect("pack_affine_rows_storage_bytes must succeed");

        // 1. Test for both destination slots: 611 and 612
        for &dest_slot in &[AFFINE_ROWS_STORAGE_DST_SLOT_0, AFFINE_ROWS_STORAGE_DST_SLOT_1] {
            let bulk_frame = build_affine_rows_storage_upload_frame_submission(&three_records, 3, dest_slot)
                .expect("bulk frame submission must succeed");
            let ops = build_affine_rows_storage_upload_callback_ops(&three_records, 3, dest_slot)
                .expect("build_affine_rows_storage_upload_callback_ops must succeed");

            assert_eq!(ops.len(), 2);
            assert_eq!(ops.len(), bulk_frame.commands().len());

            // Op 0: WriteBuffer(buffer_id=610, offset=0, bytes)
            match &ops[0] {
                StorageUploadCallbackOp::WriteBuffer { buffer_id, offset, bytes } => {
                    assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                    assert_eq!(*offset, 0);
                    assert_eq!(bytes.len(), 3 * 48);
                    match &bulk_frame.commands()[0] {
                        GpuCommand::WriteBuffer { data, .. } => {
                            assert_eq!(bytes.as_slice(), data.as_slice());
                        }
                        other => panic!("expected WriteBuffer in bulk frame, got {other:?}"),
                    }
                    assert_eq!(bytes.as_slice(), expected_packed.as_slice());
                    // Bit-exact preservation of -0.0 and quiet NaN payload
                    assert_eq!(&bytes[12..16], &[0x00, 0x00, 0x00, 0x80]);
                    assert_eq!(&bytes[76..80], &[0x34, 0x12, 0xc0, 0x7f]);
                }
                other => panic!("expected WriteBuffer op at index 0, got {other:?}"),
            }

            // Op 1: CopyBufferToBuffer(src=610, src_off=0, dst=dest_slot, dst_off=0, size=3*48)
            match &ops[1] {
                StorageUploadCallbackOp::CopyBufferToBuffer { src, src_off, dst, dst_off, size } => {
                    assert_eq!(*src, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                    assert_eq!(*src_off, 0);
                    assert_eq!(*dst, dest_slot);
                    assert_eq!(*dst_off, 0);
                    assert_eq!(*size, 3 * 48);
                    match &bulk_frame.commands()[1] {
                        GpuCommand::CopyBufferToBuffer {
                            source_buffer_id,
                            source_offset,
                            destination_buffer_id,
                            destination_offset,
                            size: bulk_size,
                            ..
                        } => {
                            assert_eq!(*src, *source_buffer_id);
                            assert_eq!(*src_off as u64, *source_offset);
                            assert_eq!(*dst, *destination_buffer_id);
                            assert_eq!(*dst_off as u64, *destination_offset);
                            assert_eq!(*size as u64, *bulk_size);
                        }
                        other => panic!("expected CopyBufferToBuffer in bulk frame, got {other:?}"),
                    }
                }
                other => panic!("expected CopyBufferToBuffer op at index 1, got {other:?}"),
            }
        }

        // 2. Guard rejections: same as bulk frame by construction
        assert!(matches!(
            build_affine_rows_storage_upload_callback_ops(&three_records, 0, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        for bad_slot in [0, 610, 613, 999] {
            assert!(
                matches!(
                    build_affine_rows_storage_upload_callback_ops(&three_records, 3, bad_slot),
                    Err(PacketEncodeError::InvalidDimensions(_))
                ),
                "expected rejection for slot {bad_slot}"
            );
        }
        assert!(matches!(
            build_affine_rows_storage_upload_callback_ops(&three_records, 2, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        assert!(matches!(
            build_affine_rows_storage_upload_callback_ops(&three_records, 4, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));
        assert!(matches!(
            build_affine_rows_storage_upload_callback_ops(&[], 1, AFFINE_ROWS_STORAGE_DST_SLOT_0),
            Err(PacketEncodeError::InvalidDimensions(_))
        ));

        // 3. Byte equality for 4000 records -> 192,000 bytes
        let large_input = vec![1.0f32; 4000 * 12];
        let large_expected = pack_affine_rows_storage_bytes(&large_input)
            .expect("pack_affine_rows_storage_bytes for 4000 records must succeed");
        let large_bulk_frame = build_affine_rows_storage_upload_frame_submission(
            &large_input,
            4000,
            AFFINE_ROWS_STORAGE_DST_SLOT_0,
        ).expect("bulk frame for 4000 records must succeed");

        let large_ops = build_affine_rows_storage_upload_callback_ops(
            &large_input,
            4000,
            AFFINE_ROWS_STORAGE_DST_SLOT_0,
        ).expect("callback ops for 4000 records must succeed");

        assert_eq!(large_ops.len(), 2);
        match &large_ops[0] {
            StorageUploadCallbackOp::WriteBuffer { buffer_id, offset, bytes } => {
                assert_eq!(*buffer_id, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                assert_eq!(*offset, 0);
                assert_eq!(bytes.len(), 192_000);
                assert_eq!(bytes.as_slice(), large_expected.as_slice());
                match &large_bulk_frame.commands()[0] {
                    GpuCommand::WriteBuffer { data, .. } => {
                        assert_eq!(bytes.as_slice(), data.as_slice());
                    }
                    other => panic!("expected WriteBuffer in bulk frame, got {other:?}"),
                }
            }
            other => panic!("expected WriteBuffer op, got {other:?}"),
        }
        match &large_ops[1] {
            StorageUploadCallbackOp::CopyBufferToBuffer { src, src_off, dst, dst_off, size } => {
                assert_eq!(*src, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID);
                assert_eq!(*src_off, 0);
                assert_eq!(*dst, AFFINE_ROWS_STORAGE_DST_SLOT_0);
                assert_eq!(*dst_off, 0);
                assert_eq!(*size, 192_000);
            }
            other => panic!("expected CopyBufferToBuffer op, got {other:?}"),
        }
    }

    #[test]
    fn test_borrowed_frame_packet_slot_and_borrow_scope_invariants() {
        let _borrow_lock = match TEST_BORROW_SCOPE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        clear_borrowed_frame_packet();

        let floats: Vec<f32> = (0..24).map(|i| (i + 1) as f32).collect(); // 2 draws
        let owned = f3d_build_affine_rows_batch_frame_packet(&floats, 64, 64, 2)
            .expect("owned frame packet encode must succeed");

        let view_bytes_0 = borrow_scope_bytes_view();

        // 1. Initial borrowed packet build succeeds
        let [ptr, len] = f3d_build_affine_rows_batch_frame_packet_borrowed(&floats, 64, 64, 2)
            .expect("borrowed frame packet build must succeed");
        assert_ne!(ptr, 0, "borrowed packet pointer must be non-zero");
        assert_eq!(len as usize, owned.len(), "borrowed packet length must match owned length");
        assert_eq!(f3d_borrowed_frame_packet_ptr(), ptr, "scalar getter ptr must match returned ptr");
        assert_eq!(f3d_borrowed_frame_packet_len(), len, "scalar getter len must match returned len");
        assert_eq!(
            borrow_scope_bytes_view(),
            view_bytes_0 + (len as u64),
            "borrow_scope_bytes_view must increase by exactly len after successful borrowed build"
        );

        // 2. Reading bytes from the borrowed slot produces the exact same bytes as owned export
        with_global_borrowed_frame_packet_ref(|slice| {
            assert_eq!(slice, Some(owned.as_slice()), "borrowed slot bytes must equal owned packet bytes");
        });

        // 3. Enter borrow scope: active borrow guard
        let token = f3d_borrow_enter();
        assert_ne!(token, 0, "entering borrow scope must yield non-zero token");

        // 4. While held: a second call to build borrowed packet must be refused (safety against view invalidation)
        let view_bytes_before_refusal = borrow_scope_bytes_view();
        let rebuild_err = f3d_build_affine_rows_batch_frame_packet_borrowed(&floats, 64, 64, 2);
        assert!(
            rebuild_err.is_err(),
            "rebuilding borrowed frame packet while linear memory borrow scope is active must be refused"
        );
        assert_eq!(
            borrow_scope_bytes_view(),
            view_bytes_before_refusal,
            "borrow_scope_bytes_view must remain unchanged after refused build"
        );

        // 5. While held: clear_borrowed_frame_packet must return false and leave slot intact
        assert!(
            !clear_borrowed_frame_packet(),
            "clear_borrowed_frame_packet must return false while linear memory borrow scope is active"
        );
        with_global_borrowed_frame_packet_ref(|slice| {
            assert_eq!(
                slice,
                Some(owned.as_slice()),
                "slot bytes must still equal owned packet after refused clear while borrowed"
            );
        });

        // 6. While held: memory growth must be refused
        assert!(
            !f3d_try_grow_memory(1),
            "memory growth must be blocked while linear memory borrow scope is active"
        );

        // 7. Exit with wrong token must fail and leave scope borrowed
        assert!(
            !f3d_borrow_exit(token + 999_999),
            "exit with mismatched token must fail"
        );
        let view_bytes_before_refusal2 = borrow_scope_bytes_view();
        assert!(
            f3d_build_affine_rows_batch_frame_packet_borrowed(&floats, 64, 64, 2).is_err(),
            "rebuild must still fail after failed mismatched exit"
        );
        assert_eq!(
            borrow_scope_bytes_view(),
            view_bytes_before_refusal2,
            "borrow_scope_bytes_view must remain unchanged after second refused build"
        );
        assert!(
            !clear_borrowed_frame_packet(),
            "clear_borrowed_frame_packet must still return false after mismatched exit"
        );
        with_global_borrowed_frame_packet_ref(|slice| {
            assert_eq!(
                slice,
                Some(owned.as_slice()),
                "slot bytes must still equal owned packet after second refused clear"
            );
        });
        assert!(
            !f3d_try_grow_memory(1),
            "memory growth must still be blocked after failed mismatched exit"
        );

        // 8. Exit with correct token succeeds
        assert!(f3d_borrow_exit(token), "exit with matching token must succeed");

        // 9. After exit: rebuild succeeds and borrow_scope_bytes_view increases by exactly len
        let view_bytes_before_rebuild = borrow_scope_bytes_view();
        let [ptr2, len2] = f3d_build_affine_rows_batch_frame_packet_borrowed(&floats, 64, 64, 2)
            .expect("rebuild after exit must succeed");
        assert_ne!(ptr2, 0);
        assert_eq!(len2 as usize, owned.len());
        assert_eq!(
            borrow_scope_bytes_view(),
            view_bytes_before_rebuild + (len2 as u64),
            "borrow_scope_bytes_view must increase by exactly len2 after second successful build"
        );
        with_global_borrowed_frame_packet_ref(|slice| {
            assert_eq!(slice, Some(owned.as_slice()));
        });

        // 10. Memory growth succeeds when idle, and f3d_borrow_growth_generation increments
        let gen_before = f3d_borrow_growth_generation();
        assert!(f3d_try_grow_memory(1), "idle memory growth must succeed");
        assert_eq!(
            f3d_borrow_growth_generation(),
            gen_before + 1,
            "growth generation must increment by 1 on successful growth"
        );

        // 11. After exit: clear_borrowed_frame_packet returns true and empties the slot
        assert!(
            clear_borrowed_frame_packet(),
            "clear_borrowed_frame_packet must return true after exit when scope is idle"
        );
        with_global_borrowed_frame_packet_ref(|slice| {
            assert_eq!(slice, None, "slot must be None after successful clear");
        });
    }

    #[test]
    fn runtime_shaders_use_schema_generated_wgsl_declarations() {
        let _slot_lock = match TEST_SLOT_TABLE_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        // 1. Verify the six ColorUniform builders use WGSL_COLOR_UNIFORM_DECLARATION
        // and do not contain the drifted field name `color: vec4<f32>`.
        let color_builders: [(&str, GpuSubmissionPacket); 6] = [
            (
                "build_red_a_blue_b_submission",
                build_red_a_blue_b_submission(true),
            ),
            (
                "build_bundle_then_direct_draw_submission",
                build_bundle_then_direct_draw_submission(),
            ),
            (
                "build_bundle_then_direct_warm_cache_submission",
                build_bundle_then_direct_warm_cache_submission(false),
            ),
            (
                "build_nested_pass_submission",
                build_nested_pass_submission(),
            ),
            (
                "build_nested_canvas_pass_submission",
                build_nested_canvas_pass_submission(),
            ),
            (
                "build_nested_viewport_scissor_submission",
                build_nested_viewport_scissor_submission(),
            ),
        ];

        for (name, packet) in &color_builders {
            let mut pipeline_count = 0;
            for cmd in packet.commands() {
                if let GpuCommand::CreatePipeline { wgsl_code, .. } = cmd {
                    pipeline_count += 1;
                    assert!(
                        wgsl_code.contains(WGSL_COLOR_UNIFORM_DECLARATION),
                        "{name} pipeline must contain schema-generated WGSL_COLOR_UNIFORM_DECLARATION"
                    );
                    assert!(
                        !wgsl_code.contains("color: vec4<f32>"),
                        "{name} pipeline must not contain drifted field name 'color: vec4<f32>'"
                    );
                    assert!(
                        wgsl_code.contains("u.rgba"),
                        "{name} pipeline must access schema field 'rgba'"
                    );
                }
            }
            assert!(
                pipeline_count > 0,
                "{name} packet must contain at least one CreatePipeline command"
            );
        }

        // 2. Verify the two AffineRows builders use WGSL_AFFINE_ROWS_DECLARATION
        // and do not contain the drifted field name `color: vec4<f32>`.
        let affine_builders: [(&str, GpuSubmissionPacket); 2] = [
            (
                "build_affine_rows_transform_submission",
                build_affine_rows_transform_submission(),
            ),
            (
                "build_affine_rows_layout_counterexample_submission(false)",
                build_affine_rows_layout_counterexample_submission(false),
            ),
        ];

        for (name, packet) in &affine_builders {
            let mut pipeline_count = 0;
            for cmd in packet.commands() {
                if let GpuCommand::CreatePipeline { wgsl_code, .. } = cmd {
                    pipeline_count += 1;
                    assert!(
                        wgsl_code.contains(WGSL_AFFINE_ROWS_DECLARATION),
                        "{name} pipeline must contain schema-generated WGSL_AFFINE_ROWS_DECLARATION"
                    );
                    assert!(
                        !wgsl_code.contains("color: vec4<f32>"),
                        "{name} pipeline must not contain drifted field name 'color: vec4<f32>'"
                    );
                }
            }
            assert!(
                pipeline_count > 0,
                "{name} packet must contain at least one CreatePipeline command"
            );
        }

        // 3. Verify counterexample wrong variant (true):
        // Must contain `mat4x3<f32>` and must NOT contain `WGSL_AFFINE_ROWS_DECLARATION`.
        let wrong_packet = build_affine_rows_layout_counterexample_submission(true);
        let mut found_wrong_pipeline = false;
        for cmd in wrong_packet.commands() {
            if let GpuCommand::CreatePipeline { wgsl_code, .. } = cmd {
                found_wrong_pipeline = true;
                assert!(
                    wgsl_code.contains("mat4x3<f32>"),
                    "wrong layout pipeline must declare mat4x3<f32>"
                );
                assert!(
                    !wgsl_code.contains(WGSL_AFFINE_ROWS_DECLARATION),
                    "wrong layout pipeline must NOT contain WGSL_AFFINE_ROWS_DECLARATION"
                );
                assert!(
                    !wgsl_code.contains("color: vec4<f32>"),
                    "wrong layout pipeline must not contain drifted field name 'color: vec4<f32>'"
                );
            }
        }
        assert!(
            found_wrong_pipeline,
            "build_affine_rows_layout_counterexample_submission(true) must contain CreatePipeline"
        );
    }

    #[test]
    fn test_compute_opcodes_byte_layout() {
        let mut packet = GpuSubmissionPacket::new();
        let code = "@compute @workgroup_size(64) fn main() {}";
        let entry = "main";
        packet.push(GpuCommand::CreateComputePipeline {
            pipeline_id: 300,
            wgsl_code: code.to_string(),
            entry_point: entry.to_string(),
            bindings: alloc::vec![
                GpuComputeBindingLayout {
                    binding_index: 0,
                    binding_type: BINDING_TYPE_STORAGE_READ,
                    min_binding_size: 48,
                },
                GpuComputeBindingLayout {
                    binding_index: 1,
                    binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                    min_binding_size: 16,
                },
            ],
        });
        packet.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 4,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![
                GpuBufferBinding {
                    binding_index: 0,
                    buffer_id: 701,
                    offset: 0,
                    size: 48,
                    binding_type: BINDING_TYPE_STORAGE_READ,
                    epoch: Epoch::new(100),
                    data_version: DataVersion::new(1),
                },
                GpuBufferBinding {
                    binding_index: 1,
                    buffer_id: 702,
                    offset: 256,
                    size: 64,
                    binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                    epoch: Epoch::new(200),
                    data_version: DataVersion::new(2),
                },
            ],
        });

        let encoded = packet.encode().expect("compute packet encoding must succeed");

        // 1. Validate packet header (16 bytes)
        assert_eq!(&encoded[0..4], b"F3DP");
        assert_eq!(u16::from_le_bytes([encoded[4], encoded[5]]), 1); // version
        assert_eq!(u16::from_le_bytes([encoded[6], encoded[7]]), 0); // flags
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 2); // cmd_count
        let total_data_len = u32::from_le_bytes(encoded[12..16].try_into().unwrap());
        let expected_payload_len = (code.len() + entry.len()) as u32;
        assert_eq!(total_data_len, expected_payload_len);

        // 2. Validate Command 0: CreateComputePipeline (Opcode 21)
        let mut cur = 16;
        let op0 = u16::from_le_bytes([encoded[cur], encoded[cur + 1]]);
        assert_eq!(op0, OPCODE_CREATE_COMPUTE_PIPELINE);
        let pad0 = u16::from_le_bytes([encoded[cur + 2], encoded[cur + 3]]);
        assert_eq!(pad0, 0);
        let pipe_id = u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap());
        assert_eq!(pipe_id, 300);
        let code_off = u32::from_le_bytes(encoded[cur + 8..cur + 12].try_into().unwrap());
        assert_eq!(code_off, 0);
        let code_len = u32::from_le_bytes(encoded[cur + 12..cur + 16].try_into().unwrap());
        assert_eq!(code_len, code.len() as u32);
        let ep_off = u32::from_le_bytes(encoded[cur + 16..cur + 20].try_into().unwrap());
        assert_eq!(ep_off, code.len() as u32);
        let ep_len = u32::from_le_bytes(encoded[cur + 20..cur + 24].try_into().unwrap());
        assert_eq!(ep_len, entry.len() as u32);
        let b_count = u32::from_le_bytes(encoded[cur + 24..cur + 28].try_into().unwrap());
        assert_eq!(b_count, 2);
        let pad1 = u32::from_le_bytes(encoded[cur + 28..cur + 32].try_into().unwrap());
        assert_eq!(pad1, 0);
        cur += 32;

        // Descriptors (2 * 16 bytes)
        assert_eq!(u32::from_le_bytes(encoded[cur..cur + 4].try_into().unwrap()), 0); // binding_index
        assert_eq!(u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap()), BINDING_TYPE_STORAGE_READ);
        assert_eq!(u32::from_le_bytes(encoded[cur + 8..cur + 12].try_into().unwrap()), 48); // min_binding_size
        assert_eq!(u32::from_le_bytes(encoded[cur + 12..cur + 16].try_into().unwrap()), 0); // pad
        cur += 16;

        assert_eq!(u32::from_le_bytes(encoded[cur..cur + 4].try_into().unwrap()), 1); // binding_index
        assert_eq!(u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap()), BINDING_TYPE_STORAGE_READ_WRITE);
        assert_eq!(u32::from_le_bytes(encoded[cur + 8..cur + 12].try_into().unwrap()), 16); // min_binding_size
        assert_eq!(u32::from_le_bytes(encoded[cur + 12..cur + 16].try_into().unwrap()), 0); // pad
        cur += 16;

        // 3. Validate Command 1: DispatchCompute (Opcode 22)
        let op1 = u16::from_le_bytes([encoded[cur], encoded[cur + 1]]);
        assert_eq!(op1, OPCODE_DISPATCH_COMPUTE);
        assert_eq!(u16::from_le_bytes([encoded[cur + 2], encoded[cur + 3]]), 0); // pad0
        assert_eq!(u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap()), 300); // pipeline_id
        assert_eq!(u32::from_le_bytes(encoded[cur + 8..cur + 12].try_into().unwrap()), 4); // wg_x
        assert_eq!(u32::from_le_bytes(encoded[cur + 12..cur + 16].try_into().unwrap()), 1); // wg_y
        assert_eq!(u32::from_le_bytes(encoded[cur + 16..cur + 20].try_into().unwrap()), 1); // wg_z
        assert_eq!(u32::from_le_bytes(encoded[cur + 20..cur + 24].try_into().unwrap()), 2); // binding_count
        cur += 24;

        // Binding record 0 (48 bytes)
        assert_eq!(u32::from_le_bytes(encoded[cur..cur + 4].try_into().unwrap()), 0); // index
        assert_eq!(u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap()), 701); // buffer_id
        assert_eq!(u64::from_le_bytes(encoded[cur + 8..cur + 16].try_into().unwrap()), 0); // offset
        assert_eq!(u64::from_le_bytes(encoded[cur + 16..cur + 24].try_into().unwrap()), 48); // size
        assert_eq!(u32::from_le_bytes(encoded[cur + 24..cur + 28].try_into().unwrap()), BINDING_TYPE_STORAGE_READ);
        assert_eq!(u32::from_le_bytes(encoded[cur + 28..cur + 32].try_into().unwrap()), 0); // pad
        assert_eq!(u64::from_le_bytes(encoded[cur + 32..cur + 40].try_into().unwrap()), 100); // epoch
        assert_eq!(u64::from_le_bytes(encoded[cur + 40..cur + 48].try_into().unwrap()), 1); // data_version
        cur += 48;

        // Binding record 1 (48 bytes)
        assert_eq!(u32::from_le_bytes(encoded[cur..cur + 4].try_into().unwrap()), 1); // index
        assert_eq!(u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap()), 702); // buffer_id
        assert_eq!(u64::from_le_bytes(encoded[cur + 8..cur + 16].try_into().unwrap()), 256); // offset
        assert_eq!(u64::from_le_bytes(encoded[cur + 16..cur + 24].try_into().unwrap()), 64); // size
        assert_eq!(u32::from_le_bytes(encoded[cur + 24..cur + 28].try_into().unwrap()), BINDING_TYPE_STORAGE_READ_WRITE);
        assert_eq!(u32::from_le_bytes(encoded[cur + 28..cur + 32].try_into().unwrap()), 0); // pad
        assert_eq!(u64::from_le_bytes(encoded[cur + 32..cur + 40].try_into().unwrap()), 200); // epoch
        assert_eq!(u64::from_le_bytes(encoded[cur + 40..cur + 48].try_into().unwrap()), 2); // data_version
        cur += 48;

        // 4. Validate Data Payload
        assert_eq!(&encoded[cur..cur + code.len()], code.as_bytes());
        cur += code.len();
        assert_eq!(&encoded[cur..cur + entry.len()], entry.as_bytes());
        cur += entry.len();

        assert_eq!(cur, encoded.len());
    }

    #[test]
    fn test_compute_validation_and_bounds_rejection() {
        // 1. CreateComputePipeline: invalid binding_type (> 2)
        let mut p1 = GpuSubmissionPacket::new();
        p1.push(GpuCommand::CreateComputePipeline {
            pipeline_id: 300,
            wgsl_code: "fn main() {}".to_string(),
            entry_point: "main".to_string(),
            bindings: alloc::vec![GpuComputeBindingLayout {
                binding_index: 0,
                binding_type: 3,
                min_binding_size: 0,
            }],
        });
        assert!(p1.encode().is_err());

        // 2. CreateComputePipeline: duplicate binding_index
        let mut p2 = GpuSubmissionPacket::new();
        p2.push(GpuCommand::CreateComputePipeline {
            pipeline_id: 300,
            wgsl_code: "fn main() {}".to_string(),
            entry_point: "main".to_string(),
            bindings: alloc::vec![
                GpuComputeBindingLayout { binding_index: 0, binding_type: 0, min_binding_size: 0 },
                GpuComputeBindingLayout { binding_index: 0, binding_type: 1, min_binding_size: 0 },
            ],
        });
        assert!(p2.encode().is_err());

        // 3. CreateComputePipeline: data payload overflow
        let mut p3 = GpuSubmissionPacket::new();
        p3.push(GpuCommand::CreateComputePipeline {
            pipeline_id: 300,
            wgsl_code: "fn main() {}".to_string(),
            entry_point: "main".to_string(),
            bindings: alloc::vec![],
        });
        assert!(matches!(p3.encode_bounded(5), Err(PacketEncodeError::DataPayloadOverflow { .. })));

        // 4. DispatchCompute: buffer_id == 0
        let mut p4 = GpuSubmissionPacket::new();
        p4.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![GpuBufferBinding {
                binding_index: 0,
                buffer_id: 0,
                offset: 0,
                size: 16,
                binding_type: 0,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            }],
        });
        assert!(p4.encode().is_err());

        // 5. DispatchCompute: offset alignment is not enforced natively without device profile (root 21381)
        let mut p5 = GpuSubmissionPacket::new();
        p5.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![GpuBufferBinding {
                binding_index: 0,
                buffer_id: 701,
                offset: 3,
                size: 16,
                binding_type: BINDING_TYPE_UNIFORM,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            }],
        });
        assert!(p5.encode().is_ok());

        // 6. DispatchCompute: size == 0
        let mut p6 = GpuSubmissionPacket::new();
        p6.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![GpuBufferBinding {
                binding_index: 0,
                buffer_id: 701,
                offset: 0,
                size: 0,
                binding_type: 0,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            }],
        });
        assert!(p6.encode().is_err());

        // 7. DispatchCompute: storage size % 4 != 0 rejected, but uniform scalar size allowed
        let mut p7_storage = GpuSubmissionPacket::new();
        p7_storage.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![GpuBufferBinding {
                binding_index: 0,
                buffer_id: 701,
                offset: 0,
                size: 15,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            }],
        });
        assert!(p7_storage.encode().is_err());

        // Uniform bindings: scalar f32 size 4 and arbitrary sizes are permitted
        let mut p7_uniform = GpuSubmissionPacket::new();
        p7_uniform.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![
                GpuBufferBinding {
                    binding_index: 0,
                    buffer_id: 701,
                    offset: 0,
                    size: 4, // scalar f32 uniform
                    binding_type: BINDING_TYPE_UNIFORM,
                    epoch: Epoch::ZERO,
                    data_version: DataVersion::new(1),
                },
                GpuBufferBinding {
                    binding_index: 1,
                    buffer_id: 702,
                    offset: 0,
                    size: 1, // 1-byte uniform binding (WebGPU CTS createBindGroup)
                    binding_type: BINDING_TYPE_UNIFORM,
                    epoch: Epoch::ZERO,
                    data_version: DataVersion::new(1),
                },
            ],
        });
        assert!(p7_uniform.encode().is_ok());

        // 8. DispatchCompute: offset + size overflow
        let mut p8 = GpuSubmissionPacket::new();
        p8.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![GpuBufferBinding {
                binding_index: 0,
                buffer_id: 701,
                offset: u64::MAX - 3,
                size: 8,
                binding_type: 0,
                epoch: Epoch::ZERO,
                data_version: DataVersion::new(1),
            }],
        });
        assert!(p8.encode().is_err());

        // 9. DispatchCompute: duplicate binding_index
        let mut p9 = GpuSubmissionPacket::new();
        p9.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![
                GpuBufferBinding {
                    binding_index: 0,
                    buffer_id: 701,
                    offset: 0,
                    size: 16,
                    binding_type: 0,
                    epoch: Epoch::ZERO,
                    data_version: DataVersion::new(1),
                },
                GpuBufferBinding {
                    binding_index: 0,
                    buffer_id: 702,
                    offset: 0,
                    size: 16,
                    binding_type: 0,
                    epoch: Epoch::ZERO,
                    data_version: DataVersion::new(1),
                },
            ],
        });
        assert!(p9.encode().is_err());

        // 10. build_affine_rows_compute_submission input bounds & cardinality
        let valid_affine = [1.0f32; 12];
        let valid_points = [0.0f32; 4];
        assert!(build_affine_rows_compute_submission(&[], &valid_points).is_err());
        assert!(build_affine_rows_compute_submission(&[1.0; 11], &valid_points).is_err());
        assert!(build_affine_rows_compute_submission(&valid_affine, &[]).is_err());
        assert!(build_affine_rows_compute_submission(&valid_affine, &[1.0; 3]).is_err()); // vec3 rejected
        assert!(build_affine_rows_compute_submission(&valid_affine, &[1.0; 5]).is_err()); // non-multiple of 4 rejected
        // Cardinality: 1 transform for 2 points -> OK (broadcast)
        let two_points = [0.0f32; 8];
        assert!(build_affine_rows_compute_submission(&valid_affine, &two_points).is_ok());
        // Cardinality: 2 transforms for 2 points -> OK (1:1)
        let two_affines = [1.0f32; 24];
        assert!(build_affine_rows_compute_submission(&two_affines, &two_points).is_ok());
        // Cardinality: 2 transforms for 3 points -> Err (neither 1 nor 3)
        let three_points = [0.0f32; 12];
        assert!(build_affine_rows_compute_submission(&two_affines, &three_points).is_err());
    }

    #[test]
    fn test_render_pass_serialized_before_compute_commands() {
        // Structural check: verifies serialized command order when interleaving render and compute commands.
        // NOTE: This is a packet serialization structure check, not GPU execution proof.
        // Full GPU pass closure before compute execution requires browser verification.
        let mut packet = GpuSubmissionPacket::new();
        packet.push(GpuCommand::RenderPass {
            target_type: TARGET_CANVAS,
            target_id: 0,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            pipeline_id: 100,
            vertex_buffer_id: 1,
            vertex_count: 3,
            uniform_dynamic_offset: 0,
            uniform_buffer_id: 1,
            load_op: LOAD_OP_CLEAR,
            store_op: STORE_OP_STORE,
            pass_flags: PASS_FLAG_NEW_PASS,
        });
        packet.push(GpuCommand::CreateComputePipeline {
            pipeline_id: 300,
            wgsl_code: "@compute @workgroup_size(64) fn main() {}".to_string(),
            entry_point: "main".to_string(),
            bindings: alloc::vec![],
        });
        packet.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: alloc::vec![],
        });
        packet.push(GpuCommand::CopyBufferToBuffer {
            source_buffer_id: 703,
            source_offset: 0,
            destination_buffer_id: 704,
            destination_offset: 0,
            size: 16,
            epoch: Epoch::ZERO,
        });

        let encoded = packet.encode().expect("mixed render-compute packet encoding must succeed");
        assert_eq!(packet.commands().len(), 4);

        // Command 0 is RenderPass (opcode 4) with STORE_OP_STORE and PASS_FLAG_NEW_PASS
        match &packet.commands()[0] {
            GpuCommand::RenderPass { store_op, pass_flags, .. } => {
                assert_eq!(*store_op, STORE_OP_STORE);
                assert_eq!(*pass_flags, PASS_FLAG_NEW_PASS);
            }
            other => panic!("expected RenderPass, got {other:?}"),
        }

        // Command 1 is CreateComputePipeline (opcode 21)
        assert!(matches!(&packet.commands()[1], GpuCommand::CreateComputePipeline { .. }));

        // Command 2 is DispatchCompute (opcode 22)
        assert!(matches!(&packet.commands()[2], GpuCommand::DispatchCompute { .. }));

        // Command 3 is CopyBufferToBuffer (opcode 20)
        assert!(matches!(&packet.commands()[3], GpuCommand::CopyBufferToBuffer { .. }));

        // Verify encoded byte opcodes in sequence
        let mut cur = 16;
        assert_eq!(u16::from_le_bytes([encoded[cur], encoded[cur + 1]]), OPCODE_RENDER_PASS);
        cur += 46;
        assert_eq!(u16::from_le_bytes([encoded[cur], encoded[cur + 1]]), OPCODE_CREATE_COMPUTE_PIPELINE);
        cur += 32;
        assert_eq!(u16::from_le_bytes([encoded[cur], encoded[cur + 1]]), OPCODE_DISPATCH_COMPUTE);
        cur += 24;
        assert_eq!(u16::from_le_bytes([encoded[cur], encoded[cur + 1]]), OPCODE_COPY_BUFFER_TO_BUFFER);
    }

    #[test]
    fn test_compute_explicit_bindings_roundtrip() {
        let bindings_layout = alloc::vec![
            GpuComputeBindingLayout {
                binding_index: 0,
                binding_type: BINDING_TYPE_STORAGE_READ,
                min_binding_size: 48,
            },
            GpuComputeBindingLayout {
                binding_index: 1,
                binding_type: BINDING_TYPE_STORAGE_READ,
                min_binding_size: 16,
            },
            GpuComputeBindingLayout {
                binding_index: 2,
                binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                min_binding_size: 16,
            },
        ];

        let bindings_dispatch = alloc::vec![
            GpuBufferBinding {
                binding_index: 0,
                buffer_id: 701,
                offset: 0,
                size: 48,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::new(100),
                data_version: DataVersion::new(1),
            },
            GpuBufferBinding {
                binding_index: 1,
                buffer_id: 702,
                offset: 256,
                size: 64,
                binding_type: BINDING_TYPE_STORAGE_READ,
                epoch: Epoch::new(100),
                data_version: DataVersion::new(2),
            },
            GpuBufferBinding {
                binding_index: 2,
                buffer_id: 703,
                offset: 512,
                size: 64,
                binding_type: BINDING_TYPE_STORAGE_READ_WRITE,
                epoch: Epoch::new(200),
                data_version: DataVersion::new(3),
            },
        ];

        let mut packet = GpuSubmissionPacket::new();
        packet.push(GpuCommand::CreateComputePipeline {
            pipeline_id: 300,
            wgsl_code: "@compute @workgroup_size(64) fn main() {}".to_string(),
            entry_point: "main".to_string(),
            bindings: bindings_layout.clone(),
        });
        packet.push(GpuCommand::DispatchCompute {
            pipeline_id: 300,
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            bindings: bindings_dispatch.clone(),
        });

        let encoded = packet.encode().expect("encoding must succeed");

        // Decode CreateComputePipeline
        let mut cur = 16;
        assert_eq!(u16::from_le_bytes([encoded[cur], encoded[cur + 1]]), OPCODE_CREATE_COMPUTE_PIPELINE);
        assert_eq!(u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap()), 300);
        let b_count = u32::from_le_bytes(encoded[cur + 24..cur + 28].try_into().unwrap()) as usize;
        assert_eq!(b_count, 3);
        cur += 32;

        for (i, expected) in bindings_layout.iter().enumerate() {
            let idx = u32::from_le_bytes(encoded[cur..cur + 4].try_into().unwrap());
            let b_type = u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap());
            let min_sz = u32::from_le_bytes(encoded[cur + 8..cur + 12].try_into().unwrap());
            cur += 16;
            assert_eq!(idx, expected.binding_index, "layout {i} index");
            assert_eq!(b_type, expected.binding_type, "layout {i} type");
            assert_eq!(min_sz, expected.min_binding_size, "layout {i} min_size");
        }

        // Decode DispatchCompute
        assert_eq!(u16::from_le_bytes([encoded[cur], encoded[cur + 1]]), OPCODE_DISPATCH_COMPUTE);
        assert_eq!(u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap()), 300);
        assert_eq!(u32::from_le_bytes(encoded[cur + 8..cur + 12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(encoded[cur + 12..cur + 16].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(encoded[cur + 16..cur + 20].try_into().unwrap()), 1);
        let d_count = u32::from_le_bytes(encoded[cur + 20..cur + 24].try_into().unwrap()) as usize;
        assert_eq!(d_count, 3);
        cur += 24;

        for (i, expected) in bindings_dispatch.iter().enumerate() {
            let idx = u32::from_le_bytes(encoded[cur..cur + 4].try_into().unwrap());
            let buf_id = u32::from_le_bytes(encoded[cur + 4..cur + 8].try_into().unwrap());
            let off = u64::from_le_bytes(encoded[cur + 8..cur + 16].try_into().unwrap());
            let sz = u64::from_le_bytes(encoded[cur + 16..cur + 24].try_into().unwrap());
            let b_type = u32::from_le_bytes(encoded[cur + 24..cur + 28].try_into().unwrap());
            let ep = u64::from_le_bytes(encoded[cur + 32..cur + 40].try_into().unwrap());
            let dv = u64::from_le_bytes(encoded[cur + 40..cur + 48].try_into().unwrap());
            cur += 48;

            assert_eq!(idx, expected.binding_index, "dispatch {i} index");
            assert_eq!(buf_id, expected.buffer_id, "dispatch {i} buffer_id");
            assert_eq!(off, expected.offset, "dispatch {i} offset");
            assert_eq!(sz, expected.size, "dispatch {i} size");
            assert_eq!(b_type, expected.binding_type, "dispatch {i} type");
            assert_eq!(ep, expected.epoch.get(), "dispatch {i} epoch");
            assert_eq!(dv, expected.data_version.get(), "dispatch {i} data_version");
        }
    }

    #[test]
    fn test_compute_affine_transformation_cpu_oracle() {
        // Affine transformation:
        // Scale (2.0, 3.0, 4.0), Translation (10.0, 20.0, 30.0)
        // Row 0: [2.0, 0.0, 0.0, 10.0]
        // Row 1: [0.0, 3.0, 0.0, 20.0]
        // Row 2: [0.0, 0.0, 4.0, 30.0]
        let affine_floats = [
            2.0f32, 0.0, 0.0, 10.0,
            0.0, 3.0, 0.0, 20.0,
            0.0, 0.0, 4.0, 30.0,
        ];
        let point = [1.0f32, 2.0, 3.0, 1.0];

        // CPU Oracle computation
        let oracle_out = cpu_transform_affine_point(&affine_floats, &point);
        assert_eq!(oracle_out, [12.0f32, 26.0, 42.0, 1.0]);

        // Struct AffineRows equivalence
        let affine = AffineRows::new(
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
        );
        let p3 = [1.0f32, 2.0, 3.0];
        let p4 = [p3[0], p3[1], p3[2], 1.0];
        let dot_r0 = affine.r0[0] * p4[0] + affine.r0[1] * p4[1] + affine.r0[2] * p4[2] + affine.r0[3] * p4[3];
        let dot_r1 = affine.r1[0] * p4[0] + affine.r1[1] * p4[1] + affine.r1[2] * p4[2] + affine.r1[3] * p4[3];
        let dot_r2 = affine.r2[0] * p4[0] + affine.r2[1] * p4[1] + affine.r2[2] * p4[2] + affine.r2[3] * p4[3];
        assert_eq!([dot_r0, dot_r1, dot_r2, 1.0], oracle_out);

        // Build submission packet and verify structure
        let packet = build_affine_rows_compute_submission(&affine_floats, &point)
            .expect("build_affine_rows_compute_submission must succeed");
        assert_eq!(packet.commands().len(), 9);

        // Verify WGSL code in CreateComputePipeline contains schema WGSL_AFFINE_ROWS_DECLARATION
        let mut found_pipeline = false;
        for cmd in packet.commands() {
            if let GpuCommand::CreateComputePipeline { wgsl_code, entry_point, .. } = cmd {
                found_pipeline = true;
                assert!(wgsl_code.contains(WGSL_AFFINE_ROWS_DECLARATION), "shader must contain schema declaration");
                assert!(wgsl_code.contains("transform_affine_point"), "shader must call transform_affine_point");
                assert_eq!(entry_point, "main");
            }
        }
        assert!(found_pipeline);

        // Verify native export
        let exported_bytes = f3d_build_affine_rows_compute_packet(&affine_floats, &point)
            .expect("f3d_build_affine_rows_compute_packet export must succeed");
        assert!(!exported_bytes.is_empty());
        assert_eq!(&exported_bytes[0..4], b"F3DP");
    }

    #[test]
    fn test_two_dispatch_packet_structure_and_oracle_outputs() {
        // Structural check: verifies multi-dispatch packet structure with non-overlapping buffer slices
        // and distinct data versions, and confirms that the mathematical CPU oracle produces distinct
        // expected outputs for matrices A and B.
        // NOTE: This verifies packet encoding and CPU oracle math, not GPU execution proof.
        // Real GPU execution proof of independent outputs on WebGPU requires browser execution.
        // Matrix A: Scale (2.0, 2.0, 2.0), Translation (1.0, 0.0, 0.0) -> r0=[2,0,0,1], r1=[0,2,0,0], r2=[0,0,2,0]
        let matrix_a_floats = [
            2.0f32, 0.0, 0.0, 1.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
        ];
        // Matrix B: Scale (0.5, 0.5, 0.5), Translation (0.0, 5.0, 0.0) -> r0=[0.5,0,0,0], r1=[0,0.5,0,5], r2=[0,0,0.5,0]
        let matrix_b_floats = [
            0.5f32, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 5.0,
            0.0, 0.0, 0.5, 0.0,
        ];

        let point = [2.0f32, 2.0, 2.0, 1.0];

        // Oracle A and Oracle B produce distinct results for the same point
        let out_a = cpu_transform_affine_point(&matrix_a_floats, &point);
        let out_b = cpu_transform_affine_point(&matrix_b_floats, &point);
        assert_eq!(out_a, [5.0f32, 4.0, 4.0, 1.0]);
        assert_eq!(out_b, [1.0f32, 6.0, 1.0, 1.0]);
        assert_ne!(out_a, out_b, "Independent dispatches must yield distinct oracle outputs");

        // 1. Independent dispatches with per-use versioned slices
        let packet = build_two_dispatch_affine_compute_submission(&matrix_a_floats, &matrix_b_floats, &point, false)
            .expect("two-dispatch submission must succeed");
        let encoded = packet.encode().expect("two-dispatch packet encoding must succeed");
        assert!(!encoded.is_empty());

        let dispatches: Vec<&GpuCommand> = packet.commands().iter()
            .filter(|c| matches!(c, GpuCommand::DispatchCompute { .. }))
            .collect();
        assert_eq!(dispatches.len(), 2);
        if let (
            GpuCommand::DispatchCompute { bindings: b1, .. },
            GpuCommand::DispatchCompute { bindings: b2, .. },
        ) = (dispatches[0], dispatches[1]) {
            assert_eq!(b1[0].offset, 0);
            assert_eq!(b1[0].data_version, DataVersion::new(1));
            assert_eq!(b2[0].offset, 256);
            assert_eq!(b2[0].data_version, DataVersion::new(2));
            assert_ne!(b1[0].offset, b2[0].offset);
            assert_ne!(b1[0].data_version, b2[0].data_version);
        } else {
            panic!("expected two DispatchCompute commands");
        }

        // 2. Aliased dispatches demonstrating input aliasing counterexample
        let aliased_packet = build_two_dispatch_affine_compute_submission(&matrix_a_floats, &matrix_b_floats, &point, true)
            .expect("aliased submission must succeed");
        let dispatches_aliased: Vec<&GpuCommand> = aliased_packet.commands().iter()
            .filter(|c| matches!(c, GpuCommand::DispatchCompute { .. }))
            .collect();
        assert_eq!(dispatches_aliased.len(), 2);
        if let (
            GpuCommand::DispatchCompute { bindings: b1, .. },
            GpuCommand::DispatchCompute { bindings: b2, .. },
        ) = (dispatches_aliased[0], dispatches_aliased[1]) {
            assert_eq!(b1[0].offset, 0);
            assert_eq!(b2[0].offset, 0, "aliased dispatch must reuse slice A offset 0");
            assert_eq!(b1[0].data_version, b2[0].data_version);
        }

        // 3. Verify native export
        let exported = f3d_build_two_dispatch_affine_compute_packet(&matrix_a_floats, &matrix_b_floats, &point, false)
            .expect("export must succeed");
        assert_eq!(&exported[0..4], b"F3DP");
    }

    #[test]
    fn test_existing_opcodes_and_packet_lengths_stay_identical() {
        assert_eq!(OPCODE_CREATE_BUFFER, 1);
        assert_eq!(OPCODE_WRITE_BUFFER, 2);
        assert_eq!(OPCODE_CREATE_PIPELINE, 3);
        assert_eq!(OPCODE_RENDER_PASS, 4);
        assert_eq!(OPCODE_COPY_TEXTURE_TO_BUFFER, 5);
        assert_eq!(OPCODE_CREATE_TEXTURE, 6);
        assert_eq!(OPCODE_RECORD_BUNDLE, 7);
        assert_eq!(OPCODE_EXECUTE_BUNDLES, 8);
        assert_eq!(OPCODE_SET_VIEWPORT, 9);
        assert_eq!(OPCODE_SET_SCISSOR_RECT, 10);
        assert_eq!(OPCODE_SET_DRAW_PARAMETERS, 11);
        assert_eq!(OPCODE_CREATE_PIPELINE_DEPTH, 12);
        assert_eq!(OPCODE_RENDER_PASS_DEPTH, 13);
        assert_eq!(OPCODE_CREATE_PIPELINE_CULL, 14);
        assert_eq!(OPCODE_CREATE_PIPELINE_DEPTH_CULL, 15);
        assert_eq!(OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR, 16);
        assert_eq!(OPCODE_WRITE_TEXTURE, 17);
        assert_eq!(OPCODE_CREATE_PIPELINE_TEXTURED, 18);
        assert_eq!(OPCODE_RECORD_BUNDLE_BATCH, 19);
        assert_eq!(OPCODE_COPY_BUFFER_TO_BUFFER, 20);
        assert_eq!(OPCODE_CREATE_COMPUTE_PIPELINE, 21);
        assert_eq!(OPCODE_DISPATCH_COMPUTE, 22);

        // Verify buffer copy builder packet is unaffected:
        // 16B packet header + 14B CreateBuffer + 22B WriteBuffer + 14B CreateBuffer + 42B CopyBufferToBuffer + 4B payload = 112B
        let copy_packet = build_buffer_copy_submission(&[1, 2, 3, 4], 0, 0, 4)
            .expect("copy builder must succeed");
        let copy_encoded = copy_packet.encode().expect("copy encoding must succeed");
        assert_eq!(copy_encoded.len(), 16 + 14 + 22 + 14 + 42 + 4);
    }
}
