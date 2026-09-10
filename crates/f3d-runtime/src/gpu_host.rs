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
//!   and [`try_publish_readback`]) is currently a Rust-side native primitive only. The browser bridge
//!   (`bridge_runtime.js`) decodes and attaches `epochHi` and `epochLo` words onto readback
//!   buffers, but no browser path feeds these buffers back through `try_publish_readback` in Wasm
//!   today. Stale-readback rejection is verified exclusively via native Rust unit tests and must not
//!   be credited as an end-to-end browser-verified rejection until a Wasm readback consumer is linked.
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use f3d_core::{
    handle::{Handle, MaterialDomain},
    layout::{
        AFFINE_ROWS_BYTES, AffineRows, COLOR_UNIFORM_BYTES,
        DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT, VERTEX_POS_UV_STRIDE, VertexPosUv,
        WGSL_AFFINE_ROWS_DECLARATION,
    },
    ownership::{DataVersion, Epoch, PerUseByteBuffer, RegionState},
};
use f3d_graph::{ExecutionPlan, PassKind, ResourceAccess};

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
                } => {
                    command_records.extend_from_slice(&OPCODE_RENDER_PASS.to_le_bytes());
                    command_records.extend_from_slice(&target_type.to_le_bytes());
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
///   or `TARGET_OFFSCREEN`. Clear colors are preserved verbatim. Each draw is lowered with its bound
///   pipeline, vertex buffer ID (or 0), vertex count, 256-byte aligned dynamic uniform offset,
///   and uniform buffer ID (from declared uniform resource use, defaulting to 1).
/// - **Copy Segments**: `CopyCommand::TextureToBuffer` commands are lowered into `GpuCommand::CopyTextureToBuffer`
///   with source texture ID, destination readback buffer ID, width, and height.
/// - **Compute Segments**: Lowering compute dispatches is not yet supported and returns a structured error.
pub fn lower_plan(plan: &ExecutionPlan) -> Result<Vec<GpuCommand>, PlanLoweringError> {
    let mut commands = Vec::new();

    for segment in plan.segments() {
        match segment.kind() {
            PassKind::Render => {
                let Some(ca) = segment.primary_color_attachment() else {
                    return Err(PlanLoweringError::MissingColorAttachment {
                        segment_name: segment.name().to_string(),
                    });
                };
                if segment.draws().is_empty() {
                    return Err(PlanLoweringError::MissingDrawCommand {
                        segment_name: segment.name().to_string(),
                    });
                }
                for draw in segment.draws() {
                    let uniform_buffer_id = draw
                        .uses()
                        .iter()
                        .find(|u| u.access == ResourceAccess::UniformBuffer)
                        .map_or(1, |u| u.resource_id.get());

                    commands.push(GpuCommand::RenderPass {
                        target_type: if ca.is_canvas() {
                            TARGET_CANVAS
                        } else {
                            TARGET_OFFSCREEN
                        },
                        target_id: ca.target_id().get(),
                        clear_color: ca.clear_color(),
                        pipeline_id: draw.pipeline_id(),
                        vertex_buffer_id: draw.vertex_buffer_id(),
                        vertex_count: draw.vertex_count(),
                        uniform_dynamic_offset: draw.uniform_dynamic_offset(),
                        uniform_buffer_id,
                    });
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

// -----------------------------------------------------------------------------
// Readback Publication Gate (§5.8, 58j.3)
// -----------------------------------------------------------------------------

/// State tracking a published readback payload guarded by epoch freshness.
///
/// If a state region's epoch advances while an asynchronous readback (e.g. `mapAsync`)
/// is in flight, attempting to publish the readback will detect the stale epoch and
/// discard the data without overwriting the region's newer state.
///
/// **Limitation**: This is currently a Rust-side native primitive. The browser bridge
/// attaches `epochHi` and `epochLo` words to readback buffers, but no browser path feeds
/// them back through `try_publish_readback` in Wasm today.
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

#[cfg(test)]
mod tests {
    use super::*;

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
        p.color_attachments.push(ColorAttachment::new_clear(
            ResourceId::new(10),
            [0.0, 0.0, 0.0, 1.0],
        ));
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
    fn create_texture_usage_includes_copy_src_and_excludes_texture_binding() {
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
}
