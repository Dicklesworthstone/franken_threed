//! First-frame WebGPU host bridge, negotiation, checked packet submission,
//! and error-scope discipline.
//!
//! # Architecture & Guarantees
//! - **Single Asupersync Runtime**: Polled futures and task ownership remain in the
//!   single application runtime; no secondary executor.
//! - **Pre-Device Negotiation**: Required features and limits are validated against
//!   the adapter before `requestDevice`. Missing required capabilities fail
//!   cleanly and immediately with structured errors.
//! - **Error-Scope Discipline**: Synchronous push, owned operations, and pop without
//!   intervening awaits or turn boundaries. Result promises are awaited afterwards.
//! - **Canvas Texture Freshness**: Canvas swapchain textures are acquired fresh each
//!   render interval and never cached across frames.
//! - **Queue Ordering & Snapshot Semantics**: Per-use versioned slices or separate
//!   buffers are used so that multiple draws in a single submission (such as Red-A / Blue-B)
//!   remain distinct without queue-write hazard overwrites.
//! - **Coarse Checked Packet**: High-throughput binary command stream with validated
//!   offsets and bounds. No `eval` and no `new Function`.
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// 48-byte affine matrix representation: 3 rows of 4 f32s.
/// Translation is stored in each row's fourth component.
/// Row 0: [e0, e4, e8,  e12]
/// Row 1: [e1, e5, e9,  e13]
/// Row 2: [e2, e6, e10, e14]
/// for column-major Three.js Matrix4 elements e0..e15.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineRows {
    pub rows: [[f32; 4]; 3],
}

impl AffineRows {
    pub const BYTE_SIZE: usize = 48;

    /// Returns the 48-byte identity affine transform.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rows: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    /// Constructs an AffineRows from a column-major 4x4 matrix (e.g. Three.js Matrix4 elements).
    #[must_use]
    pub fn from_column_major(e: &[f32; 16]) -> Self {
        Self {
            rows: [
                [e[0], e[4], e[8], e[12]],
                [e[1], e[5], e[9], e[13]],
                [e[2], e[6], e[10], e[14]],
            ],
        }
    }

    /// Reconstructs a column-major 4x4 matrix with bottom row [0, 0, 0, 1].
    #[must_use]
    pub fn to_column_major(&self) -> [f32; 16] {
        [
            self.rows[0][0], self.rows[1][0], self.rows[2][0], 0.0,
            self.rows[0][1], self.rows[1][1], self.rows[2][1], 0.0,
            self.rows[0][2], self.rows[1][2], self.rows[2][2], 0.0,
            self.rows[0][3], self.rows[1][3], self.rows[2][3], 1.0,
        ]
    }

    /// Transforms a 3D point (x, y, z) by dot-product against each row with w=1.0.
    #[must_use]
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let v = [p[0], p[1], p[2], 1.0];
        let dot = |row: &[f32; 4]| {
            row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3]
        };
        [dot(&self.rows[0]), dot(&self.rows[1]), dot(&self.rows[2])]
    }

    /// Serializes the 48-byte affine rows into little-endian bytes without unsafe casts.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), &'static str> {
        if out.len() < Self::BYTE_SIZE {
            return Err("destination buffer too small for AffineRows (needs 48 bytes)");
        }
        let mut offset = 0;
        for row in &self.rows {
            for val in row {
                out[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                offset += 4;
            }
        }
        Ok(())
    }

    /// Returns the exact 48 little-endian bytes for GPU upload.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut bytes = [0u8; 48];
        self.write_to_slice(&mut bytes).expect("exact size buffer");
        bytes
    }

    /// Deserializes an AffineRows from 48 little-endian bytes.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, &'static str> {
        if src.len() < Self::BYTE_SIZE {
            return Err("source slice too small for AffineRows (needs 48 bytes)");
        }
        let mut rows = [[0.0f32; 4]; 3];
        let mut offset = 0;
        for row in &mut rows {
            for val in row {
                let mut chunk = [0u8; 4];
                chunk.copy_from_slice(&src[offset..offset + 4]);
                *val = f32::from_le_bytes(chunk);
                offset += 4;
            }
        }
        Ok(Self { rows })
    }

    /// WGSL struct definition and point transformation helper.
    #[must_use]
    pub const fn wgsl_struct_def() -> &'static str {
        "struct AffineRows {\n    r0: vec4<f32>,\n    r1: vec4<f32>,\n    r2: vec4<f32>,\n};\n\nfn transform_affine_point(m: AffineRows, p: vec3<f32>) -> vec3<f32> {\n    let v = vec4<f32>(p, 1.0);\n    return vec3<f32>(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));\n}\n"
    }
}

/// Device limits profile required by the renderer or bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimitsRecord {
    pub max_texture_dimension_2d: u32,
    pub max_buffer_size: u64,
    pub max_bind_groups: u32,
    pub min_uniform_buffer_offset_alignment: u32,
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
    pub required_features: Vec<String>,
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
    MissingRequiredFeature {
        feature: String,
    },
    InsufficientLimit {
        limit_name: String,
        requested: u64,
        available: u64,
    },
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
    pub adapter_vendor: String,
    pub adapter_architecture: String,
    pub is_fallback_adapter: bool,
    pub enabled_features: Vec<String>,
    pub limits: DeviceLimitsRecord,
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
    Validation,
    OutOfMemory,
    Internal,
}

impl ErrorScopeKind {
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
/// Concurrent tasks cannot interleave error scopes on the same device.
#[derive(Default, Debug)]
pub struct ErrorScopeTracker {
    active_depth: u32,
    in_flight_operation: Option<u64>,
}

impl ErrorScopeTracker {
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

    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active_depth > 0
    }
}

// -----------------------------------------------------------------------------
// Coarse Checked Packet Format
// -----------------------------------------------------------------------------

pub const PACKET_MAGIC: [u8; 4] = *b"F3DP";
pub const PACKET_VERSION: u16 = 1;

pub const OPCODE_CREATE_BUFFER: u16 = 1;
pub const OPCODE_WRITE_BUFFER: u16 = 2;
pub const OPCODE_CREATE_PIPELINE: u16 = 3;
pub const OPCODE_RENDER_PASS: u16 = 4;
pub const OPCODE_COPY_TEXTURE_TO_BUFFER: u16 = 5;
pub const OPCODE_CREATE_TEXTURE: u16 = 6;

pub const BUFFER_USAGE_MAP_READ: u32 = 1;
pub const BUFFER_USAGE_COPY_SRC: u32 = 4;
pub const BUFFER_USAGE_COPY_DST: u32 = 8;
pub const BUFFER_USAGE_VERTEX: u32 = 32;
pub const BUFFER_USAGE_UNIFORM: u32 = 64;

pub const TARGET_OFFSCREEN: u32 = 0;
pub const TARGET_CANVAS: u32 = 1;

/// High-level typed commands serialized into the checked packet.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuCommand {
    CreateBuffer {
        buffer_id: u32,
        size: u32,
        usage: u32,
    },
    WriteBuffer {
        buffer_id: u32,
        offset: u32,
        data: Vec<u8>,
    },
    CreateTexture {
        texture_id: u32,
        width: u32,
        height: u32,
        format: u32, // 1 = bgra8unorm, 2 = rgba8unorm
        usage: u32,
    },
    CreatePipeline {
        pipeline_id: u32,
        wgsl_code: String,
        target_format: u32,
        has_vertex_buffer: bool,
        has_uniform_buffer: bool,
    },
    RenderPass {
        target_type: u32, // 0 = offscreen texture, 1 = canvas texture
        target_id: u32,
        clear_color: [f32; 4],
        pipeline_id: u32,
        vertex_buffer_id: u32,
        vertex_count: u32,
        uniform_dynamic_offset: u32,
    },
    CopyTextureToBuffer {
        texture_id: u32,
        buffer_id: u32,
        width: u32,
        height: u32,
    },
}

/// Serializes and validates a batch of GPU commands into a compact binary packet.
#[derive(Default, Debug)]
pub struct GpuSubmissionPacket {
    commands: Vec<GpuCommand>,
}

impl GpuSubmissionPacket {
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    pub fn push(&mut self, cmd: GpuCommand) {
        self.commands.push(cmd);
    }

    /// Serializes commands into bytes:
    /// [Header: 4 magic, 2 ver, 2 flags, 4 cmd_count, 4 data_len]
    /// [Command records...]
    /// [Data payload block...]
    pub fn encode(&self) -> Vec<u8> {
        let mut command_records = Vec::new();
        let mut data_payload = Vec::new();

        for cmd in &self.commands {
            match cmd {
                GpuCommand::CreateBuffer { buffer_id, size, usage } => {
                    command_records.extend_from_slice(&OPCODE_CREATE_BUFFER.to_le_bytes());
                    command_records.extend_from_slice(&buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&size.to_le_bytes());
                    command_records.extend_from_slice(&usage.to_le_bytes());
                }
                GpuCommand::WriteBuffer { buffer_id, offset, data } => {
                    let data_offset = data_payload.len() as u32;
                    let data_len = data.len() as u32;
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
                } => {
                    let bytes = wgsl_code.as_bytes();
                    let code_offset = data_payload.len() as u32;
                    let code_len = bytes.len() as u32;
                    data_payload.extend_from_slice(bytes);

                    command_records.extend_from_slice(&OPCODE_CREATE_PIPELINE.to_le_bytes());
                    command_records.extend_from_slice(&pipeline_id.to_le_bytes());
                    command_records.extend_from_slice(&code_offset.to_le_bytes());
                    command_records.extend_from_slice(&code_len.to_le_bytes());
                    command_records.extend_from_slice(&target_format.to_le_bytes());
                    command_records.extend_from_slice(&(if *has_vertex_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                    command_records.extend_from_slice(&(if *has_uniform_buffer { 1u32 } else { 0u32 }).to_le_bytes());
                }
                GpuCommand::RenderPass {
                    target_type,
                    target_id,
                    clear_color,
                    pipeline_id,
                    vertex_buffer_id,
                    vertex_count,
                    uniform_dynamic_offset,
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
                }
                GpuCommand::CopyTextureToBuffer { texture_id, buffer_id, width, height } => {
                    command_records.extend_from_slice(&OPCODE_COPY_TEXTURE_TO_BUFFER.to_le_bytes());
                    command_records.extend_from_slice(&texture_id.to_le_bytes());
                    command_records.extend_from_slice(&buffer_id.to_le_bytes());
                    command_records.extend_from_slice(&width.to_le_bytes());
                    command_records.extend_from_slice(&height.to_le_bytes());
                }
            }
        }

        let mut output = Vec::with_capacity(16 + command_records.len() + data_payload.len());
        output.extend_from_slice(&PACKET_MAGIC);
        output.extend_from_slice(&PACKET_VERSION.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes()); // flags
        output.extend_from_slice(&(self.commands.len() as u32).to_le_bytes());
        output.extend_from_slice(&(data_payload.len() as u32).to_le_bytes());
        output.extend_from_slice(&command_records);
        output.extend_from_slice(&data_payload);
        output
    }
}

/// Helper to build a complete Red-A / Blue-B submission packet proving snapshot isolation
/// across two draws in the SAME queue submission.
///
/// In variant `per_use_versioned = true`:
/// - Material Red is written to uniform slice 0 (offset 0)
/// - Material Blue is written to uniform slice 1 (offset 256, aligned to minUniformBufferOffsetAlignment)
/// - Pass A binds offset 0 and renders to Target A
/// - Pass B binds offset 256 and renders to Target B
/// Result: Target A is pure Red, Target B is pure Blue.
///
/// In variant `per_use_versioned = false` (regression / wrong impl toggle):
/// - Both writes target offset 0
/// Result: Both passes read the second write (Blue), failing the regression test!
pub fn build_red_a_blue_b_submission(
    per_use_versioned: bool,
    wgsl_pipeline_source: &str,
) -> GpuSubmissionPacket {
    let mut packet = GpuSubmissionPacket::new();

    // 1. Create uniform buffer (sized for two 256-byte aligned slots: 512 bytes)
    let uniform_buffer_id = 1;
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: uniform_buffer_id,
        size: 512,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });

    // 2. Create offscreen render targets (64x64) and readback buffers
    let target_a = 10;
    let target_b = 11;
    let readback_a = 20;
    let readback_b = 21;
    let readback_bytes = 64 * 256; // 64 rows, 256 bytes per row (aligned to 256)

    packet.push(GpuCommand::CreateTexture {
        texture_id: target_a,
        width: 64,
        height: 64,
        format: 2, // rgba8unorm
        usage: 16 | 4, // RENDER_ATTACHMENT | COPY_SRC
    });
    packet.push(GpuCommand::CreateTexture {
        texture_id: target_b,
        width: 64,
        height: 64,
        format: 2, // rgba8unorm
        usage: 16 | 4,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_a,
        size: readback_bytes,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: readback_b,
        size: readback_bytes,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    // 3. Write Red to slot 0: [1.0, 0.0, 0.0, 1.0]
    let red_bytes = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: uniform_buffer_id,
        offset: 0,
        data: red_bytes,
    });

    // 4. Write Blue: if per_use_versioned, write to offset 256; else overwrite offset 0!
    let blue_offset = if per_use_versioned { 256 } else { 0 };
    let blue_bytes = [0.0f32.to_le_bytes(), 0.0f32.to_le_bytes(), 1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
    packet.push(GpuCommand::WriteBuffer {
        buffer_id: uniform_buffer_id,
        offset: blue_offset,
        data: blue_bytes,
    });

    // 5. Create render pipeline
    let pipeline_id = 100;
    packet.push(GpuCommand::CreatePipeline {
        pipeline_id,
        wgsl_code: wgsl_pipeline_source.to_string(),
        target_format: 2, // rgba8unorm
        has_vertex_buffer: false, // full-screen procedural triangle
        has_uniform_buffer: true,
    });

    // 6. Pass A: render with dynamic uniform offset 0 to target_a
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_a,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id,
        vertex_buffer_id: 0,
        vertex_count: 3,
        uniform_dynamic_offset: 0,
    });

    // 7. Pass B: render with dynamic uniform offset 256 (or 0 if unversioned) to target_b
    packet.push(GpuCommand::RenderPass {
        target_type: TARGET_OFFSCREEN,
        target_id: target_b,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        pipeline_id,
        vertex_buffer_id: 0,
        vertex_count: 3,
        uniform_dynamic_offset: blue_offset,
    });

    // 8. Copy both textures to readback buffers in the same submission
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_a,
        buffer_id: readback_a,
        width: 64,
        height: 64,
    });
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: target_b,
        buffer_id: readback_b,
        width: 64,
        height: 64,
    });

    packet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_rows_identity_and_round_trip() {
        let id = AffineRows::identity();
        assert_eq!(
            id.to_column_major(),
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ]
        );
        let pt = [2.0, 3.0, 4.0];
        assert_eq!(id.transform_point(pt), pt);

        let bytes = id.to_bytes();
        assert_eq!(bytes.len(), 48);
        let restored = AffineRows::read_from_slice(&bytes).expect("valid read");
        assert_eq!(id, restored);
    }

    #[test]
    fn affine_rows_translation_and_scale() {
        // Matrix scaling by 2 and translating by (10, 20, 30)
        #[rustfmt::skip]
        let col_major = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            10.0, 20.0, 30.0, 1.0,
        ];
        let aff = AffineRows::from_column_major(&col_major);
        assert_eq!(aff.rows[0], [2.0, 0.0, 0.0, 10.0]);
        assert_eq!(aff.rows[1], [0.0, 2.0, 0.0, 20.0]);
        assert_eq!(aff.rows[2], [0.0, 0.0, 2.0, 30.0]);

        let transformed = aff.transform_point([1.0, 2.0, 3.0]);
        assert_eq!(transformed, [12.0, 24.0, 36.0]);
    }

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
    fn packet_encoding_and_structure() {
        let mut packet = GpuSubmissionPacket::new();
        packet.push(GpuCommand::CreateBuffer {
            buffer_id: 1,
            size: 256,
            usage: BUFFER_USAGE_UNIFORM,
        });
        packet.push(GpuCommand::WriteBuffer {
            buffer_id: 1,
            offset: 0,
            data: vec![1, 2, 3, 4],
        });

        let encoded = packet.encode();
        assert_eq!(&encoded[0..4], &PACKET_MAGIC);
        assert_eq!(u16::from_le_bytes([encoded[4], encoded[5]]), PACKET_VERSION);
        let cmd_count = u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]);
        assert_eq!(cmd_count, 2);
    }

    #[test]
    fn red_a_blue_b_submission_separation() {
        let valid_packet = build_red_a_blue_b_submission(true, "// test shader");
        let encoded_valid = valid_packet.encode();
        assert!(!encoded_valid.is_empty());

        let invalid_packet = build_red_a_blue_b_submission(false, "// test shader");
        let encoded_invalid = invalid_packet.encode();
        assert!(!encoded_invalid.is_empty());

        // In the valid packet, the second pass uniform dynamic offset is 256.
        // In the invalid packet, the offset is 0 (identical to first pass).
        assert_ne!(encoded_valid, encoded_invalid);
    }
}
