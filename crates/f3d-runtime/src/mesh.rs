//! Dynamic Three.js Mesh-to-Wasm rendering packet synthesis.
//!
//! # Architecture & Guarantees (§5.1, §6.3, §6.7, §8.5)
//! - **Dynamic Input Processing**: Accepts arbitrary geometry vertex positions and optional triangle
//!   indices from Three.js `BufferGeometry`, rather than hard-coded geometry or captured static scenes.
//! - **f64 to f32 Upload Boundary**: Accepts Three.js double-precision camera and model-view matrices
//!   (`[f64; 16]`) and quantizes element-by-element to 32-bit single-precision (`[f32; 16]`) at the GPU
//!   uniform upload boundary, preserving upstream matrix-update boundaries without premature f64 MVP rounding.
//! - **WebGL-to-WebGPU Depth Conversion**: When the source camera operates under `WebGLCoordinateSystem`
//!   (clip z in `[-1, 1]`), the shader remaps clip coordinates: `clip.z = (clip.z + clip.w) * 0.5`.
//!   WebGPU-native projection matrices (clip z in `[0, 1]`) pass through unremapped.
//! - **Real WGSL Position/Color Pipeline**: Emits a checked 144-byte uniform record (model_view: 64,
//!   projection: 64, color: 16) and 20-byte `VertexPosUv` stride rendered through [`FrameSession`]
//!   and lowered via [`crate::gpu_host::lower_plan`].
//! - **Validation & Error Discipline**: Strictly checks matrix lengths (exactly 16), color lengths
//!   (3 or 4), vertex position alignment (multiple of 3), index bounds, and non-zero dimensions.
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::{string::String, vec::Vec};
use core::fmt;

use f3d_core::{
    handle::{Handle, MaterialDomain},
    layout::{aligned_bytes_per_row, VERTEX_POS_UV_STRIDE, VertexPosUv},
    ownership::{DataVersion, Epoch},
};
use f3d_graph::{
    canvas::{CanvasEpochTracker, CanvasFormat, CanvasId},
    pass::DepthStencilAttachment,
    resource::ResourceId,
};

use crate::frame::{FrameSession, RenderContext};
use crate::gpu_host::{
    with_global_resource_table, GpuCommand, GpuSubmissionPacket, BUFFER_USAGE_COPY_DST,
    BUFFER_USAGE_MAP_READ, BUFFER_USAGE_UNIFORM, BUFFER_USAGE_VERTEX, DEPTH_COMPARE_ALWAYS,
    TARGET_FORMAT_DEPTH24PLUS, TARGET_FORMAT_PREFERRED_CANVAS, TARGET_FORMAT_RGBA8UNORM,
    TEXTURE_USAGE_COPY_SRC, TEXTURE_USAGE_RENDER_ATTACHMENT,
};

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

/// Structured errors encountered during dynamic mesh packet construction or validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshPacketError {
    /// Render target attachment dimensions must be greater than zero.
    ZeroDimensions { width: u32, height: u32 },
    /// Position array length is not a multiple of 3 (x, y, z triplets).
    InvalidPositionLength { len: usize },
    /// Index value exceeds available vertex count.
    IndexOutOfBounds { index: u32, vertex_count: usize },
    /// Matrix array does not have exactly 16 elements.
    InvalidMatrixLength { name: &'static str, len: usize },
    /// Color array does not have 3 or 4 elements.
    InvalidColorLength { len: usize },
    /// Invalid depth comparison operator (must be 1..=8).
    InvalidDepthCompare { value: u32 },
    /// Target dimension alignment, vertex count, or row pitch calculation failed.
    InvalidDimensions(String),
    /// Render session, graph compilation, or plan lowering error.
    SessionError(String),
    /// Packet encoding error.
    EncodeError(String),
}

impl fmt::Display for MeshPacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions { width, height } => {
                write!(f, "target dimensions must be non-zero (got {width}x{height})")
            }
            Self::InvalidPositionLength { len } => {
                write!(f, "positions length must be a multiple of 3 (got {len})")
            }
            Self::IndexOutOfBounds { index, vertex_count } => {
                write!(f, "index {index} out of bounds for vertex count {vertex_count}")
            }
            Self::InvalidMatrixLength { name, len } => {
                write!(f, "{name} matrix must contain exactly 16 elements (got {len})")
            }
            Self::InvalidColorLength { len } => {
                write!(f, "color must contain 3 or 4 elements (got {len})")
            }
            Self::InvalidDepthCompare { value } => {
                write!(f, "depth compare function code must be between 1 and 8 (got {value})")
            }
            Self::InvalidDimensions(msg) => write!(f, "invalid dimensions: {msg}"),
            Self::SessionError(msg) => write!(f, "render session error: {msg}"),
            Self::EncodeError(msg) => write!(f, "packet encode error: {msg}"),
        }
    }
}

impl core::error::Error for MeshPacketError {}

/// Canonical uniform buffer resource identifier for dynamic mesh parameters.
pub const MESH_UNIFORM_BUFFER_ID: u32 = 1;

/// Canonical vertex buffer resource identifier for dynamic mesh geometry.
pub const MESH_VERTEX_BUFFER_ID: u32 = 2;

/// Canonical offscreen render target texture identifier.
pub const MESH_TARGET_TEXTURE_ID: u32 = 10;

/// Canonical offscreen depth texture identifier.
pub const MESH_DEPTH_TEXTURE_ID: u32 = 11;

/// Canonical staging readback buffer resource identifier.
pub const MESH_READBACK_BUFFER_ID: u32 = 20;

/// Canonical pipeline resource identifier for the dynamic mesh render pipeline.
pub const MESH_PIPELINE_ID: u32 = 100;

/// Default pass clear color `[R, G, B, A]` (opaque black).
pub const MESH_CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Canonical canvas presentation target resource identifier.
pub const MESH_CANVAS_TARGET_ID: u32 = 0;

/// Canonical pipeline resource identifier for the canvas presentation mesh render pipeline.
pub const MESH_CANVAS_PIPELINE_ID: u32 = 101;

/// Typed, validated dynamic mesh input parameters ready for GPU schedule synthesis.
///
/// Invariant: Constructible only via [`DynamicMeshInput::try_from_raw`], ensuring
/// unforgeable dimensions, matrix layouts, color ranges, and index bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicMeshInput<'a> {
    positions: &'a [f32],
    indices: &'a [u32],
    model_view: [f32; 16],
    projection: [f32; 16],
    color: [f32; 4],
    width: u32,
    height: u32,
    webgl_depth: bool,
}

impl<'a> DynamicMeshInput<'a> {
    /// Validates raw slices and constructs an unforgeable `DynamicMeshInput`.
    pub fn try_from_raw(
        positions: &'a [f32],
        indices: &'a [u32],
        model_view: &[f64],
        projection: &[f64],
        color: &[f32],
        width: u32,
        height: u32,
        webgl_depth: bool,
    ) -> Result<Self, MeshPacketError> {
        if width == 0 || height == 0 {
            return Err(MeshPacketError::ZeroDimensions { width, height });
        }
        if positions.len() % 3 != 0 {
            return Err(MeshPacketError::InvalidPositionLength { len: positions.len() });
        }
        let vertex_count = positions.len() / 3;

        if !indices.is_empty() {
            // Triangle-list drops incomplete tail: check only indices that form complete triangles
            let num_triangles = indices.len() / 3;
            let effective_index_count = num_triangles * 3;
            for &idx in &indices[..effective_index_count] {
                if (idx as usize) >= vertex_count {
                    return Err(MeshPacketError::IndexOutOfBounds {
                        index: idx,
                        vertex_count,
                    });
                }
            }
        }

        if model_view.len() != 16 {
            return Err(MeshPacketError::InvalidMatrixLength {
                name: "model_view",
                len: model_view.len(),
            });
        }
        if projection.len() != 16 {
            return Err(MeshPacketError::InvalidMatrixLength {
                name: "projection",
                len: projection.len(),
            });
        }

        let mut mv_f32 = [0.0f32; 16];
        for (dst, &src) in mv_f32.iter_mut().zip(model_view.iter()) {
            *dst = src as f32;
        }

        let mut proj_f32 = [0.0f32; 16];
        for (dst, &src) in proj_f32.iter_mut().zip(projection.iter()) {
            *dst = src as f32;
        }

        let color_f32 = match color.len() {
            3 => [color[0], color[1], color[2], 1.0],
            4 => [color[0], color[1], color[2], color[3]],
            other => return Err(MeshPacketError::InvalidColorLength { len: other }),
        };

        Ok(Self {
            positions,
            indices,
            model_view: mv_f32,
            projection: proj_f32,
            color: color_f32,
            width,
            height,
            webgl_depth,
        })
    }

    /// Flattened 3D vertex positions `[x0, y0, z0, x1, ...]`.
    #[inline]
    #[must_use]
    pub fn positions(&self) -> &'a [f32] {
        self.positions
    }

    /// Optional triangle index list. If empty, `positions` is treated as non-indexed triangles.
    #[inline]
    #[must_use]
    pub fn indices(&self) -> &'a [u32] {
        self.indices
    }

    /// Quantized single-precision 4x4 column-major model-view matrix.
    #[inline]
    #[must_use]
    pub fn model_view(&self) -> &[f32; 16] {
        &self.model_view
    }

    /// Quantized single-precision 4x4 column-major projection matrix.
    #[inline]
    #[must_use]
    pub fn projection(&self) -> &[f32; 16] {
        &self.projection
    }

    /// Diffuse base color `[r, g, b, a]` normalized to `[0.0, 1.0]`.
    #[inline]
    #[must_use]
    pub fn color(&self) -> &[f32; 4] {
        &self.color
    }

    /// Target texture width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Target texture height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Whether source camera uses WebGL coordinate system (depth `[-1, 1]`) requiring remap.
    #[inline]
    #[must_use]
    pub fn webgl_depth(&self) -> bool {
        self.webgl_depth
    }
}

/// Depth testing and writing configuration for dynamic mesh render pipeline and pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MeshDepthOptions {
    /// Whether depth testing is enabled.
    pub depth_test: bool,
    /// Effective depth writing flag for the GPU pipeline (resolved by host/adapter).
    pub depth_write: bool,
    /// WebGPU depth comparison operator (wire codes 1..=8: 1=Never, 2=Less, 3=Equal, 4=LessEqual, 5=Greater, 6=NotEqual, 7=GreaterEqual, 8=Always).
    /// Forced to [`DEPTH_COMPARE_ALWAYS`] (8) if `depth_test` is false.
    pub depth_compare: u32,
}

impl MeshDepthOptions {
    /// Constructs and validates depth options.
    ///
    /// # Errors
    /// Returns [`MeshPacketError::InvalidDepthCompare`] if `depth_compare` is not in `1..=8`.
    pub fn new(depth_test: bool, depth_write: bool, depth_compare: u32) -> Result<Self, MeshPacketError> {
        if !(1..=8).contains(&depth_compare) {
            return Err(MeshPacketError::InvalidDepthCompare { value: depth_compare });
        }
        Ok(Self {
            depth_test,
            depth_write,
            depth_compare,
        })
    }

    /// Resolves effective depth write and compare operations according to pinned Three.js r186 WebGPU invariants.
    ///
    /// `depth_write` is accepted directly as the effective GPU pipeline write flag (§6.3, r186 WebGPUPipelineUtils.js:224).
    /// When `depth_test` is false, the comparison operator is forced to [`DEPTH_COMPARE_ALWAYS`] (8).
    #[inline]
    #[must_use]
    pub fn resolve_effective(&self) -> (bool, u32) {
        let effective_compare = if self.depth_test {
            self.depth_compare
        } else {
            DEPTH_COMPARE_ALWAYS
        };
        (self.depth_write, effective_compare)
    }
}

/// Shared internal shader helper with canvas sRGB output flag.
///
/// When `canvas_srgb` is `true` (canvas presentation target), the fragment shader applies
/// the pinned Three.js r186 sRGB OETF transfer (`ColorSpaceFunctions.js:38-48`, exponent 0.41666,
/// threshold 0.0031308) to `uniforms.color.rgb`, leaving alpha unchanged.
///
/// When `canvas_srgb` is `false` (offscreen render target), the shader retains linear-sRGB output
/// matching default upstream `RenderTarget` working space.
fn generate_mesh_wgsl_internal(webgl_depth: bool, canvas_srgb: bool) -> String {
    let depth_remap = if webgl_depth {
        "    clip.z = (clip.z + clip.w) * 0.5;\n"
    } else {
        ""
    };

    let (srgb_fn, fragment_body) = if canvas_srgb {
        (
            "\
fn srgb_transfer_oetf(color: vec3<f32>) -> vec3<f32> {\n\
    let clamped = max(color, vec3<f32>(0.0));\n\
    let a = pow(clamped, vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055);\n\
    let b = color * 12.92;\n\
    return select(a, b, color <= vec3<f32>(0.0031308));\n\
}\n\
\n\
",
            "    let srgb_rgb = srgb_transfer_oetf(uniforms.color.rgb);\n\
    return vec4<f32>(srgb_rgb, uniforms.color.a);\n",
        )
    } else {
        ("", "    return uniforms.color;\n")
    };

    alloc::format!(
        "\
struct MeshUniforms {{\n\
    model_view: mat4x4<f32>,\n\
    projection: mat4x4<f32>,\n\
    color: vec4<f32>,\n\
}};\n\
\n\
@group(0) @binding(0)\n\
var<uniform> uniforms: MeshUniforms;\n\
\n\
struct VertexInput {{\n\
    @location(0) position: vec3<f32>,\n\
    @location(1) uv: vec2<f32>,\n\
}};\n\
\n\
struct VertexOutput {{\n\
    @builtin(position) clip_position: vec4<f32>,\n\
    @location(0) uv: vec2<f32>,\n\
}};\n\
\n\
@vertex\n\
fn vs_main(in: VertexInput) -> VertexOutput {{\n\
    var out: VertexOutput;\n\
    let mv_pos = uniforms.model_view * vec4<f32>(in.position, 1.0);\n\
    var clip = uniforms.projection * mv_pos;\n\
{depth_remap}\
    out.clip_position = clip;\n\
    out.uv = in.uv;\n\
    return out;\n\
}}\n\
\n\
{srgb_fn}\
@fragment\n\
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{\n\
{fragment_body}\
}}\n\
"
    )
}

/// Generates the WGSL shader source code for dynamic offscreen mesh rendering (linear-sRGB output).
///
/// Features a 144-byte uniform buffer with modelView, projection, and color,
/// location 0 position vec3<f32>, location 1 uv vec2<f32>, and conditional
/// WebGL-to-WebGPU depth conversion. Retains linear-sRGB output matching default
/// upstream `RenderTarget` working space.
#[must_use]
pub fn generate_mesh_wgsl(webgl_depth: bool) -> String {
    generate_mesh_wgsl_internal(webgl_depth, false)
}

/// Generates the WGSL shader source code for dynamic canvas mesh rendering (sRGB output).
///
/// Applies the pinned Three.js r186 sRGB OETF transfer (`ColorSpaceFunctions.js:38-48`,
/// exponent 0.41666, threshold 0.0031308) in `fs_main` while keeping alpha unchanged.
#[must_use]
pub fn generate_mesh_canvas_wgsl(webgl_depth: bool) -> String {
    generate_mesh_wgsl_internal(webgl_depth, true)
}

/// Reference CPU evaluation of sRGB OETF transfer matching Three.js r186 `ColorManagement.LinearToSRGB`.
///
/// Evaluates: `v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 0.41666) - 0.055`.
#[inline]
#[must_use]
pub fn srgb_transfer_oetf_cpu(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        f3d_math::color::linear_to_srgb(v as f64) as f32
    }
}

/// Helper: de-indexes vertex positions and formats the padded uniform buffer record.
fn prepare_vertex_and_uniform_data(
    input: &DynamicMeshInput<'_>,
) -> Result<(u32, u32, u32, Vec<u8>, Vec<u8>), MeshPacketError> {
    let (vertex_count, vertex_bytes) = if input.indices.is_empty() {
        let v_count = input.positions.len() / 3;
        let mut v_bytes = Vec::with_capacity(v_count * VertexPosUv::BYTE_SIZE);
        for chunk in input.positions.chunks_exact(3) {
            let vertex = VertexPosUv::new([chunk[0], chunk[1], chunk[2]], [0.0, 0.0]);
            v_bytes.extend_from_slice(&vertex.to_bytes());
        }
        (v_count, v_bytes)
    } else {
        let num_triangles = input.indices.len() / 3;
        let effective_index_count = num_triangles * 3;
        let vertex_count_avail = input.positions.len() / 3;
        let mut v_bytes = Vec::with_capacity(effective_index_count * VertexPosUv::BYTE_SIZE);
        for &idx in &input.indices[..effective_index_count] {
            if (idx as usize) >= vertex_count_avail {
                return Err(MeshPacketError::IndexOutOfBounds {
                    index: idx,
                    vertex_count: vertex_count_avail,
                });
            }
            let base = (idx as usize) * 3;
            let vertex = VertexPosUv::new(
                [input.positions[base], input.positions[base + 1], input.positions[base + 2]],
                [0.0, 0.0],
            );
            v_bytes.extend_from_slice(&vertex.to_bytes());
        }
        (effective_index_count, v_bytes)
    };

    let vertex_count_u32 = u32::try_from(vertex_count)
        .map_err(|_| MeshPacketError::InvalidDimensions("vertex count exceeds u32::MAX".into()))?;
    let raw_vertex_bytes_len = u32::try_from(vertex_bytes.len())
        .map_err(|_| MeshPacketError::InvalidDimensions("vertex buffer bytes exceed u32::MAX".into()))?;
    let vertex_stride_u32 = u32::try_from(VERTEX_POS_UV_STRIDE)
        .map_err(|_| MeshPacketError::InvalidDimensions("vertex stride exceeds u32::MAX".into()))?;

    // WebGPU requires buffer size > 0; allocate minimum 4 bytes for empty geometry
    let vertex_buffer_size = raw_vertex_bytes_len.max(4);
    let vertex_upload_data = if vertex_bytes.is_empty() {
        alloc::vec![0u8; 4]
    } else {
        vertex_bytes
    };

    // Build uniform buffer payload: 144 bytes padded to 256 bytes for WebGPU alignment
    let mut uniform_bytes = Vec::with_capacity(256);
    for val in &input.model_view {
        uniform_bytes.extend_from_slice(&val.to_le_bytes());
    }
    for val in &input.projection {
        uniform_bytes.extend_from_slice(&val.to_le_bytes());
    }
    for val in &input.color {
        uniform_bytes.extend_from_slice(&val.to_le_bytes());
    }
    uniform_bytes.resize(256, 0);

    Ok((
        vertex_count_u32,
        vertex_stride_u32,
        vertex_buffer_size,
        vertex_upload_data,
        uniform_bytes,
    ))
}

/// Builds a verified [`GpuSubmissionPacket`] from typed dynamic mesh inputs for offscreen rendering.
pub fn build_mesh_submission(input: &DynamicMeshInput<'_>) -> Result<GpuSubmissionPacket, MeshPacketError> {
    build_mesh_submission_internal(input, None)
}

/// Builds a verified [`GpuSubmissionPacket`] with depth testing/writing from typed dynamic mesh inputs for offscreen rendering.
pub fn build_mesh_depth_submission(
    input: &DynamicMeshInput<'_>,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    let opts = MeshDepthOptions::new(depth_test, depth_write, depth_compare)?;
    build_mesh_submission_internal(input, Some(opts))
}

fn build_mesh_submission_internal(
    input: &DynamicMeshInput<'_>,
    depth_opts: Option<MeshDepthOptions>,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    // 1. Validate dimensions and calculate aligned readback size BEFORE any slot registration/allocation
    if input.width == 0 || input.height == 0 {
        return Err(MeshPacketError::ZeroDimensions {
            width: input.width,
            height: input.height,
        });
    }
    let bytes_per_row = aligned_bytes_per_row(input.width)
        .map_err(|e| MeshPacketError::InvalidDimensions(alloc::format!("width {}: {e:?}", input.width)))?;
    let readback_size = bytes_per_row
        .checked_mul(input.height)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("readback size calculation overflow".into()))?;

    // 2. Determine vertex count, de-index geometry, and build uniform buffer payload
    let (
        vertex_count_u32,
        vertex_stride_u32,
        vertex_buffer_size,
        vertex_upload_data,
        uniform_bytes,
    ) = prepare_vertex_and_uniform_data(input)?;

    // 3. Register canonical resource IDs in the global generational slot table
    // (Only reached after all validations, layout calculations, and conversions succeed)
    with_global_resource_table(|table| {
        table.register(MESH_UNIFORM_BUFFER_ID);
        table.register(MESH_VERTEX_BUFFER_ID);
        table.register(MESH_TARGET_TEXTURE_ID);
        if depth_opts.is_some() {
            table.register(MESH_DEPTH_TEXTURE_ID);
        }
        table.register(MESH_READBACK_BUFFER_ID);
        table.register(MESH_PIPELINE_ID);
    });

    let mut packet = GpuSubmissionPacket::new();

    // 4. Initial GPU resource allocations
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_UNIFORM_BUFFER_ID,
        size: 256,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_VERTEX_BUFFER_ID,
        size: vertex_buffer_size,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::WriteBuffer {
        buffer_id: MESH_VERTEX_BUFFER_ID,
        offset: 0,
        data: vertex_upload_data,
    });

    packet.push(GpuCommand::CreateTexture {
        texture_id: MESH_TARGET_TEXTURE_ID,
        width: input.width,
        height: input.height,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    if depth_opts.is_some() {
        packet.push(GpuCommand::CreateTexture {
            texture_id: MESH_DEPTH_TEXTURE_ID,
            width: input.width,
            height: input.height,
            format: TARGET_FORMAT_DEPTH24PLUS,
            usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
        });
    }

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_READBACK_BUFFER_ID,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    let wgsl_code = generate_mesh_wgsl(input.webgl_depth);
    if let Some(depth) = depth_opts {
        let (depth_write_enabled, depth_compare) = depth.resolve_effective();
        packet.push(GpuCommand::CreatePipelineDepth {
            pipeline_id: MESH_PIPELINE_ID,
            wgsl_code,
            target_format: TARGET_FORMAT_RGBA8UNORM,
            has_vertex_buffer: true,
            has_uniform_buffer: true,
            uniform_size: 144,
            vertex_stride: vertex_stride_u32,
            depth_format: TARGET_FORMAT_DEPTH24PLUS,
            depth_write_enabled,
            depth_compare,
        });
    } else {
        packet.push(GpuCommand::CreatePipeline {
            pipeline_id: MESH_PIPELINE_ID,
            wgsl_code,
            target_format: TARGET_FORMAT_RGBA8UNORM,
            has_vertex_buffer: true,
            has_uniform_buffer: true,
            uniform_size: 144,
            vertex_stride: vertex_stride_u32,
        });
    }

    // 5. Build pass structure through FrameSession and lower_plan
    let root_ctx = RenderContext::new_offscreen(
        ResourceId::new(MESH_TARGET_TEXTURE_ID),
        input.width,
        input.height,
        Epoch::ZERO,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("FrameSession::new: {e:?}")))?
        .with_uniform_buffer_id(MESH_UNIFORM_BUFFER_ID);

    let mat_handle = Handle::<MaterialDomain>::from_raw(1, 1)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("Handle::from_raw: {e:?}")))?;
    let rec_mat = session
        .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, &uniform_bytes)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("snapshot_material_use: {e:?}")))?;

    let depth_attachment = depth_opts.map(|_| {
        DepthStencilAttachment::new_depth_clear(ResourceId::new(MESH_DEPTH_TEXTURE_ID), 1.0)
    });

    session
        .begin_render_pass_with_depth("mesh_render_pass", MESH_CLEAR_COLOR, depth_attachment)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("begin_render_pass_with_depth: {e:?}")))?;
    session
        .record_direct_draw(
            MESH_PIPELINE_ID,
            MESH_VERTEX_BUFFER_ID,
            vertex_count_u32,
            Some(rec_mat),
        )
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("record_direct_draw: {e:?}")))?;
    session
        .end_render_pass()
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("end_render_pass: {e:?}")))?;

    let session_packet = session
        .build_submission_packet()
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("build_submission_packet: {e:?}")))?;

    for cmd in session_packet.into_commands() {
        packet.push(cmd);
    }

    // 6. Copy target texture to readback buffer
    packet.push(GpuCommand::CopyTextureToBuffer {
        texture_id: MESH_TARGET_TEXTURE_ID,
        buffer_id: MESH_READBACK_BUFFER_ID,
        width: input.width,
        height: input.height,
        epoch: Epoch::ZERO,
    });

    Ok(packet)
}

/// Builds a verified [`GpuSubmissionPacket`] targeting a visible canvas swapchain
/// from typed dynamic mesh inputs.
///
/// # Canvas Pipeline & Epoch Semantics (§5.1, §6.7, §8.5, [S48])
/// - **Dynamically Negotiated Format**: Emits pipeline [`MESH_CANVAS_PIPELINE_ID`] with
///   [`TARGET_FORMAT_PREFERRED_CANVAS`] (format code 0), allowing the browser host to
///   bind its preferred swapchain format (`bgra8unorm` or `rgba8unorm`).
/// - **Per-Submission Canvas Acquisition**: Uses a per-submission [`CanvasEpochTracker`] and
///   [`RenderContext::new_canvas_acquired`] to acquire a single fresh output epoch for this
///   submission. Because the tracker is instantiated per submission call, epoch values are isolated
///   to each packet compile; no claim of globally monotonic epochs across independent calls is made.
/// - **Synchronous Swapchain Texture**: WebGPU swapchain textures are acquired synchronously by the
///   host per frame execution via `context.getCurrentTexture().createView()`; no offscreen texture
///   or readback buffer is allocated.
/// - **Zero New Opcodes**: Emits standard [`GpuCommand::RenderPass`] with `target_type: TARGET_CANVAS`.
pub fn build_mesh_canvas_submission(
    input: &DynamicMeshInput<'_>,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    build_mesh_canvas_submission_internal(input, None)
}

/// Builds a verified [`GpuSubmissionPacket`] with depth testing/writing targeting a visible canvas swapchain
/// from typed dynamic mesh inputs.
pub fn build_mesh_canvas_depth_submission(
    input: &DynamicMeshInput<'_>,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    let opts = MeshDepthOptions::new(depth_test, depth_write, depth_compare)?;
    build_mesh_canvas_submission_internal(input, Some(opts))
}

fn build_mesh_canvas_submission_internal(
    input: &DynamicMeshInput<'_>,
    depth_opts: Option<MeshDepthOptions>,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    // 1. Validate dimensions BEFORE any slot registration/allocation
    if input.width == 0 || input.height == 0 {
        return Err(MeshPacketError::ZeroDimensions {
            width: input.width,
            height: input.height,
        });
    }

    // 2. Determine vertex count, de-index geometry, and build uniform buffer payload
    let (
        vertex_count_u32,
        vertex_stride_u32,
        vertex_buffer_size,
        vertex_upload_data,
        uniform_bytes,
    ) = prepare_vertex_and_uniform_data(input)?;

    // 3. Register canonical resource IDs in the global generational slot table
    with_global_resource_table(|table| {
        table.register(MESH_UNIFORM_BUFFER_ID);
        table.register(MESH_VERTEX_BUFFER_ID);
        table.register(MESH_CANVAS_TARGET_ID);
        if depth_opts.is_some() {
            table.register(MESH_DEPTH_TEXTURE_ID);
        }
        table.register(MESH_CANVAS_PIPELINE_ID);
    });

    let mut packet = GpuSubmissionPacket::new();

    // 4. Initial GPU resource allocations
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_UNIFORM_BUFFER_ID,
        size: 256,
        usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_VERTEX_BUFFER_ID,
        size: vertex_buffer_size,
        usage: BUFFER_USAGE_VERTEX | BUFFER_USAGE_COPY_DST,
    });

    packet.push(GpuCommand::WriteBuffer {
        buffer_id: MESH_VERTEX_BUFFER_ID,
        offset: 0,
        data: vertex_upload_data,
    });

    if depth_opts.is_some() {
        packet.push(GpuCommand::CreateTexture {
            texture_id: MESH_DEPTH_TEXTURE_ID,
            width: input.width,
            height: input.height,
            format: TARGET_FORMAT_DEPTH24PLUS,
            usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
        });
    }

    let wgsl_code = generate_mesh_canvas_wgsl(input.webgl_depth);
    if let Some(depth) = depth_opts {
        let (depth_write_enabled, depth_compare) = depth.resolve_effective();
        packet.push(GpuCommand::CreatePipelineDepth {
            pipeline_id: MESH_CANVAS_PIPELINE_ID,
            wgsl_code,
            target_format: TARGET_FORMAT_PREFERRED_CANVAS,
            has_vertex_buffer: true,
            has_uniform_buffer: true,
            uniform_size: 144,
            vertex_stride: vertex_stride_u32,
            depth_format: TARGET_FORMAT_DEPTH24PLUS,
            depth_write_enabled,
            depth_compare,
        });
    } else {
        packet.push(GpuCommand::CreatePipeline {
            pipeline_id: MESH_CANVAS_PIPELINE_ID,
            wgsl_code,
            target_format: TARGET_FORMAT_PREFERRED_CANVAS,
            has_vertex_buffer: true,
            has_uniform_buffer: true,
            uniform_size: 144,
            vertex_stride: vertex_stride_u32,
        });
    }

    // 5. Build pass structure through FrameSession with CanvasEpochTracker and lower_plan
    let mut tracker = CanvasEpochTracker::new();
    tracker.register_canvas(
        CanvasId::new(MESH_CANVAS_TARGET_ID),
        ResourceId::new(MESH_CANVAS_TARGET_ID),
        input.width,
        input.height,
        CanvasFormat::Bgra8Unorm,
    );
    let canvas_output = tracker
        .begin_frame_acquire(CanvasId::new(MESH_CANVAS_TARGET_ID))
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("canvas acquire: {e:?}")))?;

    let root_ctx = RenderContext::new_canvas_acquired(
        ResourceId::new(MESH_CANVAS_TARGET_ID),
        input.width,
        input.height,
        Epoch::new(1),
        canvas_output.epoch,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("FrameSession::new: {e:?}")))?
        .with_uniform_buffer_id(MESH_UNIFORM_BUFFER_ID);

    let mat_handle = Handle::<MaterialDomain>::from_raw(1, 1)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("Handle::from_raw: {e:?}")))?;
    let rec_mat = session
        .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, &uniform_bytes)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("snapshot_material_use: {e:?}")))?;

    let depth_attachment = depth_opts.map(|_| {
        DepthStencilAttachment::new_depth_clear(ResourceId::new(MESH_DEPTH_TEXTURE_ID), 1.0)
    });

    session
        .begin_render_pass_with_depth("mesh_canvas_render_pass", MESH_CLEAR_COLOR, depth_attachment)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("begin_render_pass_with_depth: {e:?}")))?;
    session
        .record_direct_draw(
            MESH_CANVAS_PIPELINE_ID,
            MESH_VERTEX_BUFFER_ID,
            vertex_count_u32,
            Some(rec_mat),
        )
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("record_direct_draw: {e:?}")))?;
    session
        .end_render_pass()
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("end_render_pass: {e:?}")))?;

    let session_packet = session
        .build_submission_packet_with_tracker(Some(&tracker))
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("build_submission_packet_with_tracker: {e:?}")))?;

    for cmd in session_packet.into_commands() {
        packet.push(cmd);
    }

    Ok(packet)
}

fn build_mesh_packet_impl(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, MeshPacketError> {
    let input = DynamicMeshInput::try_from_raw(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )?;

    let packet = build_mesh_submission(&input)?;
    packet
        .encode()
        .map_err(|e| MeshPacketError::EncodeError(alloc::format!("{e:?}")))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet (wasm-bindgen export).
pub fn f3d_build_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet (canonical bridge alias).
pub fn gpu_bridge_build_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    f3d_build_mesh_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet for host verification and unit tests.
pub fn f3d_build_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, String> {
    build_mesh_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
    .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet (canonical bridge alias).
pub fn gpu_bridge_build_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, String> {
    f3d_build_mesh_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
}

fn build_mesh_depth_packet_impl(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, MeshPacketError> {
    let input = DynamicMeshInput::try_from_raw(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )?;

    let packet = build_mesh_depth_submission(&input, depth_test, depth_write, depth_compare)?;
    packet
        .encode()
        .map_err(|e| MeshPacketError::EncodeError(alloc::format!("{e:?}")))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet with depth state (wasm-bindgen export).
pub fn f3d_build_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_depth_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet with depth state (canonical bridge alias).
pub fn gpu_bridge_build_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    f3d_build_mesh_depth_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet with depth state for host verification and unit tests.
pub fn f3d_build_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, String> {
    build_mesh_depth_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
    .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet with depth state (canonical bridge alias).
pub fn gpu_bridge_build_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, String> {
    f3d_build_mesh_depth_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
}

fn build_mesh_canvas_depth_packet_impl(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, MeshPacketError> {
    let input = DynamicMeshInput::try_from_raw(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )?;

    let packet = build_mesh_canvas_depth_submission(&input, depth_test, depth_write, depth_compare)?;
    packet
        .encode()
        .map_err(|e| MeshPacketError::EncodeError(alloc::format!("{e:?}")))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain with depth state (wasm-bindgen export).
pub fn f3d_build_canvas_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_canvas_depth_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain with depth state (canonical bridge alias).
pub fn gpu_bridge_build_canvas_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    f3d_build_canvas_mesh_depth_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain with depth state for host verification and unit tests.
pub fn f3d_build_canvas_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, String> {
    build_mesh_canvas_depth_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
    .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain with depth state (canonical bridge alias).
pub fn gpu_bridge_build_canvas_mesh_depth_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<Vec<u8>, String> {
    f3d_build_canvas_mesh_depth_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
    )
}

fn build_mesh_canvas_packet_impl(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, MeshPacketError> {
    let input = DynamicMeshInput::try_from_raw(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )?;

    let packet = build_mesh_canvas_submission(&input)?;
    packet
        .encode()
        .map_err(|e| MeshPacketError::EncodeError(alloc::format!("{e:?}")))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain (wasm-bindgen export).
pub fn f3d_build_canvas_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_canvas_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain (canonical bridge alias).
pub fn gpu_bridge_build_canvas_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    f3d_build_canvas_mesh_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain for host verification and unit tests.
pub fn f3d_build_canvas_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, String> {
    build_mesh_canvas_packet_impl(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
    .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a dynamic Three.js Mesh-to-Wasm submission packet targeting a visible canvas swapchain (canonical bridge alias).
pub fn gpu_bridge_build_canvas_mesh_packet(
    positions: &[f32],
    indices: &[u32],
    model_view: &[f64],
    projection: &[f64],
    color: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
) -> Result<Vec<u8>, String> {
    f3d_build_canvas_mesh_packet(
        positions,
        indices,
        model_view,
        projection,
        color,
        width,
        height,
        webgl_depth,
    )
}
