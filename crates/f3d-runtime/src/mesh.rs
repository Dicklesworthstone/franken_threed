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
    BUFFER_USAGE_MAP_READ, BUFFER_USAGE_UNIFORM, BUFFER_USAGE_VERTEX, CULL_MODE_BACK,
    CULL_MODE_FRONT, CULL_MODE_NONE, DEPTH_COMPARE_ALWAYS, DEPTH_COMPARE_LESS, FRONT_FACE_CCW,
    FRONT_FACE_CW, OPCODE_CREATE_PIPELINE_CULL, OPCODE_CREATE_PIPELINE_DEPTH_CULL,
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
    /// Invalid depth test boolean code (must be 0 = false, 1 = true).
    InvalidDepthTest { value: u8 },
    /// Invalid depth write boolean code (must be 0 = false, 1 = true).
    InvalidDepthWrite { value: u8 },
    /// Array length mismatch for depth arrays in batch submission.
    InvalidDepthArrayLength { name: &'static str, expected: usize, actual: usize },
    /// Invalid face culling mode code (must be 0 = None, 1 = Front, 2 = Back).
    InvalidCullMode { value: u8 },
    /// Invalid front face winding code (must be 0 = CCW, 1 = CW).
    InvalidFrontFace { value: u8 },
    /// Array length mismatch for culling arrays in batch submission.
    InvalidCullArrayLength { name: &'static str, expected: usize, actual: usize },
    /// Invalid color write boolean code (must be 0 = false, 1 = true).
    InvalidColorWrite { value: u8 },
    /// Array length mismatch for color write array in batch submission.
    InvalidColorWriteArrayLength { expected: usize, actual: usize },
    /// Target dimension alignment, vertex count, or row pitch calculation failed.
    InvalidDimensions(String),
    /// Render session, graph compilation, or plan lowering error.
    SessionError(String),
    /// Packet encoding error.
    EncodeError(String),
    /// Multi-mesh submission input list is empty.
    EmptyMeshList,
    /// Meshes in a multi-mesh submission have conflicting target dimensions.
    MismatchedDimensions {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
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
            Self::InvalidDepthTest { value } => {
                write!(f, "depth test code must be 0 (false) or 1 (true) (got {value})")
            }
            Self::InvalidDepthWrite { value } => {
                write!(f, "depth write code must be 0 (false) or 1 (true) (got {value})")
            }
            Self::InvalidDepthArrayLength { name, expected, actual } => {
                write!(f, "{name} array length must match mesh count {expected} (got {actual})")
            }
            Self::InvalidCullMode { value } => {
                write!(f, "cull mode code must be between 0 and 2 (got {value})")
            }
            Self::InvalidFrontFace { value } => {
                write!(f, "front face code must be 0 (CCW) or 1 (CW) (got {value})")
            }
            Self::InvalidCullArrayLength { name, expected, actual } => {
                write!(f, "{name} array length must match mesh count {expected} (got {actual})")
            }
            Self::InvalidColorWrite { value } => {
                write!(f, "color write code must be 0 (false) or 1 (true) (got {value})")
            }
            Self::InvalidColorWriteArrayLength { expected, actual } => {
                write!(f, "color_writes array length must match mesh count {expected} (got {actual})")
            }
            Self::InvalidDimensions(msg) => write!(f, "invalid dimensions: {msg}"),
            Self::SessionError(msg) => write!(f, "render session error: {msg}"),
            Self::EncodeError(msg) => write!(f, "packet encode error: {msg}"),
            Self::EmptyMeshList => write!(f, "mesh inputs list must contain at least one mesh"),
            Self::MismatchedDimensions {
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "mismatched mesh dimensions: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
            ),
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
    cull: Option<(u32, u32)>,
    depth: Option<MeshDepthOptions>,
    color_write: bool,
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
            cull: None,
            depth: None,
            color_write: true,
        })
    }

    /// Configures explicit face culling and front-face winding for this mesh.
    pub fn with_cull(mut self, cull_mode: u32, front_face: u32) -> Result<Self, MeshPacketError> {
        if cull_mode > 2 {
            return Err(MeshPacketError::InvalidCullMode { value: cull_mode as u8 });
        }
        if front_face > 1 {
            return Err(MeshPacketError::InvalidFrontFace { value: front_face as u8 });
        }
        self.cull = Some((cull_mode, front_face));
        Ok(self)
    }

    /// Configures explicit depth testing and writing for this mesh.
    ///
    /// # Errors
    /// Returns [`MeshPacketError::InvalidDepthCompare`] if `depth.depth_compare` is not in `1..=8`.
    pub fn with_depth(mut self, depth: MeshDepthOptions) -> Result<Self, MeshPacketError> {
        MeshDepthOptions::new(depth.depth_test, depth.depth_write, depth.depth_compare)?;
        self.depth = Some(depth);
        Ok(self)
    }

    /// Configures explicit depth testing and writing for this mesh with parameter validation.
    pub fn with_depth_options(
        mut self,
        depth_test: bool,
        depth_write: bool,
        depth_compare: u32,
    ) -> Result<Self, MeshPacketError> {
        self.depth = Some(MeshDepthOptions::new(depth_test, depth_write, depth_compare)?);
        Ok(self)
    }

    /// Returns the explicit depth testing and writing options if configured.
    #[inline]
    #[must_use]
    pub fn depth(&self) -> Option<MeshDepthOptions> {
        self.depth
    }

    /// Configures color channel write mask for this mesh (`true` = write color, `false` = depth-only occluder).
    #[inline]
    pub fn with_color_write(mut self, color_write: bool) -> Self {
        self.color_write = color_write;
        self
    }

    /// Returns whether color writes are enabled for this mesh (defaults to true).
    #[inline]
    #[must_use]
    pub fn color_write(&self) -> bool {
        self.color_write
    }

    /// Returns the explicit face culling and front-face winding if configured.
    #[inline]
    #[must_use]
    pub fn cull(&self) -> Option<(u32, u32)> {
        self.cull
    }

    /// Returns face culling mode (defaults to [`CULL_MODE_NONE`] if unconfigured).
    #[inline]
    #[must_use]
    pub fn cull_mode(&self) -> u32 {
        self.cull.map_or(CULL_MODE_NONE, |(cm, _)| cm)
    }

    /// Returns front-facing winding order (defaults to [`FRONT_FACE_CCW`] if unconfigured).
    #[inline]
    #[must_use]
    pub fn front_face(&self) -> u32 {
        self.cull.map_or(FRONT_FACE_CCW, |(_, ff)| ff)
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

/// Maps effective depth testing/writing settings into a deterministic tag in `0..=15`.
///
/// Default WebGPU dynamic mesh depth `(depth_write_enabled = true, depth_compare = DEPTH_COMPARE_LESS (2))`
/// is mapped to `0`, ensuring backward-compatible pipeline IDs matching legacy base pipeline IDs.
/// Remaining 15 combinations of `(depth_write_enabled, depth_compare)` are bijectively mapped into `1..=15`.
#[inline]
#[must_use]
fn compute_depth_tag(depth_write_enabled: bool, depth_compare: u32) -> u32 {
    if depth_write_enabled && depth_compare == DEPTH_COMPARE_LESS {
        0
    } else {
        let raw = (depth_write_enabled as u32) * 8 + (depth_compare.saturating_sub(1));
        if raw < 9 {
            raw + 1
        } else {
            raw
        }
    }
}

/// Shared internal shader helper with canvas sRGB output flag.
///
/// When `output_srgb` is `true` (canvas presentation target), the fragment shader applies
/// the pinned Three.js r186 sRGB OETF transfer (`ColorSpaceFunctions.js:38-48`, exponent 0.41666,
/// threshold 0.0031308) to `uniforms.color.rgb`, leaving alpha unchanged.
///
/// When `output_srgb` is `false` (offscreen render target), the shader retains linear-sRGB output
/// matching default upstream `RenderTarget` working space.
fn generate_mesh_wgsl_internal(webgl_depth: bool, output_srgb: bool) -> String {
    let depth_remap = if webgl_depth {
        "    clip.z = (clip.z + clip.w) * 0.5;\n"
    } else {
        ""
    };

    let (srgb_fn, fragment_body) = if output_srgb {
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
    build_multi_mesh_submission(core::slice::from_ref(input))
}

/// Builds a verified [`GpuSubmissionPacket`] with depth testing/writing from typed dynamic mesh inputs for offscreen rendering.
pub fn build_mesh_depth_submission(
    input: &DynamicMeshInput<'_>,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    build_multi_mesh_depth_submission(core::slice::from_ref(input), depth_test, depth_write, depth_compare)
}

/// Builds a verified [`GpuSubmissionPacket`] from a slice of dynamic mesh inputs for offscreen rendering.
pub fn build_multi_mesh_submission(
    inputs: &[DynamicMeshInput<'_>],
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    build_multi_mesh_submission_internal(inputs, None)
}

/// Builds a verified [`GpuSubmissionPacket`] with depth testing/writing from a slice of dynamic mesh inputs for offscreen rendering.
pub fn build_multi_mesh_depth_submission(
    inputs: &[DynamicMeshInput<'_>],
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    let opts = MeshDepthOptions::new(depth_test, depth_write, depth_compare)?;
    build_multi_mesh_submission_internal(inputs, Some(opts))
}

fn build_multi_mesh_submission_internal(
    inputs: &[DynamicMeshInput<'_>],
    depth_opts: Option<MeshDepthOptions>,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    if inputs.is_empty() {
        return Err(MeshPacketError::EmptyMeshList);
    }
    let first = &inputs[0];
    if first.width == 0 || first.height == 0 {
        return Err(MeshPacketError::ZeroDimensions {
            width: first.width,
            height: first.height,
        });
    }
    for input in &inputs[1..] {
        if input.width != first.width || input.height != first.height {
            return Err(MeshPacketError::MismatchedDimensions {
                expected_width: first.width,
                expected_height: first.height,
                actual_width: input.width,
                actual_height: input.height,
            });
        }
        if input.webgl_depth != first.webgl_depth {
            return Err(MeshPacketError::InvalidDimensions(
                "all meshes in a multi-mesh submission must share the same webgl_depth setting".into(),
            ));
        }
    }

    let bytes_per_row = aligned_bytes_per_row(first.width)
        .map_err(|e| MeshPacketError::InvalidDimensions(alloc::format!("width {}: {e:?}", first.width)))?;
    let readback_size = bytes_per_row
        .checked_mul(first.height)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("readback size calculation overflow".into()))?;

    // 2. Prepare vertex data and uniform payloads for each mesh
    let mut total_vertices: u32 = 0;
    let mut total_vertex_bytes: Vec<u8> = Vec::new();
    let mut mesh_draw_ranges: Vec<(u32, u32)> = Vec::with_capacity(inputs.len());
    let mut mesh_uniform_payloads: Vec<Vec<u8>> = Vec::with_capacity(inputs.len());

    for input in inputs {
        let (v_count, _v_stride, _v_size, v_upload, u_bytes) = prepare_vertex_and_uniform_data(input)?;
        let first_vertex = total_vertices;
        total_vertices = total_vertices
            .checked_add(v_count)
            .ok_or_else(|| MeshPacketError::InvalidDimensions("total vertex count exceeds u32::MAX".into()))?;
        if v_count > 0 {
            total_vertex_bytes.extend_from_slice(&v_upload);
        }
        mesh_draw_ranges.push((first_vertex, v_count));
        mesh_uniform_payloads.push(u_bytes);
    }

    let raw_vertex_bytes_len = u32::try_from(total_vertex_bytes.len())
        .map_err(|_| MeshPacketError::InvalidDimensions("vertex buffer bytes exceed u32::MAX".into()))?;
    let vertex_buffer_size = raw_vertex_bytes_len.max(4);
    let vertex_upload_data = if total_vertex_bytes.is_empty() {
        alloc::vec![0u8; 4]
    } else {
        total_vertex_bytes
    };
    let vertex_stride_u32 = u32::try_from(VERTEX_POS_UV_STRIDE)
        .map_err(|_| MeshPacketError::InvalidDimensions("vertex stride exceeds u32::MAX".into()))?;

    let uniform_buffer_size = u32::try_from(inputs.len())
        .map_err(|_| MeshPacketError::InvalidDimensions("mesh count exceeds u32::MAX".into()))?
        .checked_mul(256)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("uniform buffer size exceeds u32::MAX".into()))?;

    let has_explicit_cull = inputs.iter().any(|i| i.cull().is_some());
    let has_per_mesh_depth = inputs.iter().any(|i| i.depth().is_some());
    let has_color_write_override = inputs.iter().any(|i| !i.color_write());
    let has_depth = depth_opts.is_some()
        || has_color_write_override // Opcode 16 pipelines require a matching depth attachment.
        || inputs.iter().any(|i| i.depth().map_or(false, |d| d.depth_test || d.depth_write));

    let mut unique_pipeline_configs: Vec<(u32, u32, u32, bool, u32, u32)> = Vec::new();
    let mut draw_pipeline_ids = Vec::with_capacity(inputs.len());

    if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
        for input in inputs {
            let (cull_mode, front_face) = input.cull().unwrap_or((CULL_MODE_NONE, FRONT_FACE_CCW));
            let cull_tag = cull_mode * 2 + front_face;
            let write_mask = if input.color_write() { 0xF } else { 0x0 };
            let color_tag = if input.color_write() { 0 } else { 1 };
            if has_depth {
                let (depth_write_enabled, depth_compare) = if let Some(d) = input.depth().or(depth_opts) {
                    d.resolve_effective()
                } else {
                    (false, DEPTH_COMPARE_ALWAYS)
                };
                let depth_tag = compute_depth_tag(depth_write_enabled, depth_compare);
                let pipeline_id = MESH_PIPELINE_ID + cull_tag + depth_tag * 6 + color_tag * 96;
                draw_pipeline_ids.push(pipeline_id);
                let config = (pipeline_id, cull_mode, front_face, depth_write_enabled, depth_compare, write_mask);
                if !unique_pipeline_configs.contains(&config) {
                    unique_pipeline_configs.push(config);
                }
            } else {
                let pipeline_id = MESH_PIPELINE_ID + cull_tag + color_tag * 96;
                draw_pipeline_ids.push(pipeline_id);
                let config = (pipeline_id, cull_mode, front_face, false, 0, write_mask);
                if !unique_pipeline_configs.contains(&config) {
                    unique_pipeline_configs.push(config);
                }
            }
        }
    }

    // 3. Register canonical resource IDs in the global generational slot table
    with_global_resource_table(|table| {
        table.register(MESH_UNIFORM_BUFFER_ID);
        table.register(MESH_VERTEX_BUFFER_ID);
        table.register(MESH_TARGET_TEXTURE_ID);
        if has_depth {
            table.register(MESH_DEPTH_TEXTURE_ID);
        }
        table.register(MESH_READBACK_BUFFER_ID);
        if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
            for &(pipeline_id, ..) in &unique_pipeline_configs {
                table.register(pipeline_id);
            }
        } else {
            table.register(MESH_PIPELINE_ID);
        }
    });

    let mut packet = GpuSubmissionPacket::new();

    // 4. Initial GPU resource allocations
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_UNIFORM_BUFFER_ID,
        size: uniform_buffer_size,
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
        width: first.width,
        height: first.height,
        format: TARGET_FORMAT_RGBA8UNORM,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC,
    });

    if has_depth {
        packet.push(GpuCommand::CreateTexture {
            texture_id: MESH_DEPTH_TEXTURE_ID,
            width: first.width,
            height: first.height,
            format: TARGET_FORMAT_DEPTH24PLUS,
            usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
        });
    }

    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_READBACK_BUFFER_ID,
        size: readback_size,
        usage: BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST,
    });

    let wgsl_code = generate_mesh_wgsl_internal(first.webgl_depth, false);
    if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
        if has_depth {
            for &(pipeline_id, cull_mode, front_face, depth_write_enabled, depth_compare, write_mask) in &unique_pipeline_configs {
                if write_mask != 0xF {
                    packet.push(GpuCommand::CreatePipelineDepthCullColor {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_RGBA8UNORM,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        depth_format: TARGET_FORMAT_DEPTH24PLUS,
                        depth_write_enabled,
                        depth_compare,
                        cull_mode,
                        front_face,
                        write_mask,
                    });
                } else {
                    packet.push(GpuCommand::CreatePipelineDepthCull {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_RGBA8UNORM,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        depth_format: TARGET_FORMAT_DEPTH24PLUS,
                        depth_write_enabled,
                        depth_compare,
                        cull_mode,
                        front_face,
                    });
                }
            }
        } else {
            for &(pipeline_id, cull_mode, front_face, _, _, write_mask) in &unique_pipeline_configs {
                if write_mask != 0xF {
                    packet.push(GpuCommand::CreatePipelineDepthCullColor {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_RGBA8UNORM,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        depth_format: TARGET_FORMAT_DEPTH24PLUS,
                        depth_write_enabled: false,
                        depth_compare: DEPTH_COMPARE_ALWAYS,
                        cull_mode,
                        front_face,
                        write_mask,
                    });
                } else {
                    packet.push(GpuCommand::CreatePipelineCull {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_RGBA8UNORM,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        cull_mode,
                        front_face,
                    });
                }
            }
        }
    } else if let Some(depth) = depth_opts {
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
        first.width,
        first.height,
        Epoch::ZERO,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("FrameSession::new: {e:?}")))?
        .with_uniform_buffer_id(MESH_UNIFORM_BUFFER_ID);

    let mut mat_records = Vec::with_capacity(inputs.len());
    for (i, u_bytes) in mesh_uniform_payloads.iter().enumerate() {
        let mat_handle = Handle::<MaterialDomain>::from_raw(i as u32 + 1, 1)
            .map_err(|e| MeshPacketError::SessionError(alloc::format!("Handle::from_raw: {e:?}")))?;
        let rec_mat = session
            .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, u_bytes)
            .map_err(|e| MeshPacketError::SessionError(alloc::format!("snapshot_material_use: {e:?}")))?;
        mat_records.push(rec_mat);
    }

    let depth_attachment = if has_depth {
        Some(DepthStencilAttachment::new_depth_clear(ResourceId::new(MESH_DEPTH_TEXTURE_ID), 1.0))
    } else {
        None
    };

    session
        .begin_render_pass_with_depth("mesh_render_pass", MESH_CLEAR_COLOR, depth_attachment)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("begin_render_pass_with_depth: {e:?}")))?;

    for (i, &(first_vertex, v_count)) in mesh_draw_ranges.iter().enumerate() {
        let draw_pipeline_id = if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
            draw_pipeline_ids[i]
        } else {
            MESH_PIPELINE_ID
        };
        session
            .record_direct_draw_with_range(
                draw_pipeline_id,
                MESH_VERTEX_BUFFER_ID,
                [v_count, 1, first_vertex, 0],
                Some(mat_records[i]),
            )
            .map_err(|e| MeshPacketError::SessionError(alloc::format!("record_direct_draw_with_range: {e:?}")))?;
    }

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
        width: first.width,
        height: first.height,
        epoch: Epoch::ZERO,
    });

    Ok(packet)
}

/// Builds a verified [`GpuSubmissionPacket`] targeting a visible canvas swapchain
/// from typed dynamic mesh inputs.
pub fn build_mesh_canvas_submission(
    input: &DynamicMeshInput<'_>,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    build_multi_mesh_canvas_submission(core::slice::from_ref(input))
}

/// Builds a verified [`GpuSubmissionPacket`] with depth testing/writing targeting a visible canvas swapchain
/// from typed dynamic mesh inputs.
pub fn build_mesh_canvas_depth_submission(
    input: &DynamicMeshInput<'_>,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    build_multi_mesh_canvas_depth_submission(core::slice::from_ref(input), depth_test, depth_write, depth_compare)
}

/// Builds a verified [`GpuSubmissionPacket`] targeting a visible canvas swapchain
/// from a slice of dynamic mesh inputs.
pub fn build_multi_mesh_canvas_submission(
    inputs: &[DynamicMeshInput<'_>],
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    build_multi_mesh_canvas_submission_internal(inputs, None)
}

/// Builds a verified [`GpuSubmissionPacket`] with depth testing/writing targeting a visible canvas swapchain
/// from a slice of dynamic mesh inputs.
pub fn build_multi_mesh_canvas_depth_submission(
    inputs: &[DynamicMeshInput<'_>],
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    let opts = MeshDepthOptions::new(depth_test, depth_write, depth_compare)?;
    build_multi_mesh_canvas_submission_internal(inputs, Some(opts))
}

fn build_multi_mesh_canvas_submission_internal(
    inputs: &[DynamicMeshInput<'_>],
    depth_opts: Option<MeshDepthOptions>,
) -> Result<GpuSubmissionPacket, MeshPacketError> {
    if inputs.is_empty() {
        return Err(MeshPacketError::EmptyMeshList);
    }
    let first = &inputs[0];
    if first.width == 0 || first.height == 0 {
        return Err(MeshPacketError::ZeroDimensions {
            width: first.width,
            height: first.height,
        });
    }
    for input in &inputs[1..] {
        if input.width != first.width || input.height != first.height {
            return Err(MeshPacketError::MismatchedDimensions {
                expected_width: first.width,
                expected_height: first.height,
                actual_width: input.width,
                actual_height: input.height,
            });
        }
        if input.webgl_depth != first.webgl_depth {
            return Err(MeshPacketError::InvalidDimensions(
                "all meshes in a multi-mesh submission must share the same webgl_depth setting".into(),
            ));
        }
    }

    // 2. Prepare vertex data and uniform payloads for each mesh
    let mut total_vertices: u32 = 0;
    let mut total_vertex_bytes: Vec<u8> = Vec::new();
    let mut mesh_draw_ranges: Vec<(u32, u32)> = Vec::with_capacity(inputs.len());
    let mut mesh_uniform_payloads: Vec<Vec<u8>> = Vec::with_capacity(inputs.len());

    for input in inputs {
        let (v_count, _v_stride, _v_size, v_upload, u_bytes) = prepare_vertex_and_uniform_data(input)?;
        let first_vertex = total_vertices;
        total_vertices = total_vertices
            .checked_add(v_count)
            .ok_or_else(|| MeshPacketError::InvalidDimensions("total vertex count exceeds u32::MAX".into()))?;
        if v_count > 0 {
            total_vertex_bytes.extend_from_slice(&v_upload);
        }
        mesh_draw_ranges.push((first_vertex, v_count));
        mesh_uniform_payloads.push(u_bytes);
    }

    let raw_vertex_bytes_len = u32::try_from(total_vertex_bytes.len())
        .map_err(|_| MeshPacketError::InvalidDimensions("vertex buffer bytes exceed u32::MAX".into()))?;
    let vertex_buffer_size = raw_vertex_bytes_len.max(4);
    let vertex_upload_data = if total_vertex_bytes.is_empty() {
        alloc::vec![0u8; 4]
    } else {
        total_vertex_bytes
    };
    let vertex_stride_u32 = u32::try_from(VERTEX_POS_UV_STRIDE)
        .map_err(|_| MeshPacketError::InvalidDimensions("vertex stride exceeds u32::MAX".into()))?;

    let uniform_buffer_size = u32::try_from(inputs.len())
        .map_err(|_| MeshPacketError::InvalidDimensions("mesh count exceeds u32::MAX".into()))?
        .checked_mul(256)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("uniform buffer size exceeds u32::MAX".into()))?;

    let has_explicit_cull = inputs.iter().any(|i| i.cull().is_some());
    let has_per_mesh_depth = inputs.iter().any(|i| i.depth().is_some());
    let has_color_write_override = inputs.iter().any(|i| !i.color_write());
    let has_depth = depth_opts.is_some()
        || has_color_write_override // Opcode 16 pipelines require a matching depth attachment.
        || inputs.iter().any(|i| i.depth().map_or(false, |d| d.depth_test || d.depth_write));

    let mut unique_pipeline_configs: Vec<(u32, u32, u32, bool, u32, u32)> = Vec::new();
    let mut draw_pipeline_ids = Vec::with_capacity(inputs.len());

    if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
        for input in inputs {
            let (cull_mode, front_face) = input.cull().unwrap_or((CULL_MODE_NONE, FRONT_FACE_CCW));
            let cull_tag = cull_mode * 2 + front_face;
            let write_mask = if input.color_write() { 0xF } else { 0x0 };
            let color_tag = if input.color_write() { 0 } else { 1 };
            if has_depth {
                let (depth_write_enabled, depth_compare) = if let Some(d) = input.depth().or(depth_opts) {
                    d.resolve_effective()
                } else {
                    (false, DEPTH_COMPARE_ALWAYS)
                };
                let depth_tag = compute_depth_tag(depth_write_enabled, depth_compare);
                let pipeline_id = MESH_CANVAS_PIPELINE_ID + cull_tag + depth_tag * 6 + color_tag * 96;
                draw_pipeline_ids.push(pipeline_id);
                let config = (pipeline_id, cull_mode, front_face, depth_write_enabled, depth_compare, write_mask);
                if !unique_pipeline_configs.contains(&config) {
                    unique_pipeline_configs.push(config);
                }
            } else {
                let pipeline_id = MESH_CANVAS_PIPELINE_ID + cull_tag + color_tag * 96;
                draw_pipeline_ids.push(pipeline_id);
                let config = (pipeline_id, cull_mode, front_face, false, 0, write_mask);
                if !unique_pipeline_configs.contains(&config) {
                    unique_pipeline_configs.push(config);
                }
            }
        }
    }

    // 3. Register canonical resource IDs in the global generational slot table
    with_global_resource_table(|table| {
        table.register(MESH_UNIFORM_BUFFER_ID);
        table.register(MESH_VERTEX_BUFFER_ID);
        table.register(MESH_CANVAS_TARGET_ID);
        if has_depth {
            table.register(MESH_DEPTH_TEXTURE_ID);
        }
        if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
            for &(pipeline_id, ..) in &unique_pipeline_configs {
                table.register(pipeline_id);
            }
        } else {
            table.register(MESH_CANVAS_PIPELINE_ID);
        }
    });

    let mut packet = GpuSubmissionPacket::new();

    // 4. Initial GPU resource allocations
    packet.push(GpuCommand::CreateBuffer {
        buffer_id: MESH_UNIFORM_BUFFER_ID,
        size: uniform_buffer_size,
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

    if has_depth {
        packet.push(GpuCommand::CreateTexture {
            texture_id: MESH_DEPTH_TEXTURE_ID,
            width: first.width,
            height: first.height,
            format: TARGET_FORMAT_DEPTH24PLUS,
            usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
        });
    }

    let wgsl_code = generate_mesh_wgsl_internal(first.webgl_depth, true);
    if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
        if has_depth {
            for &(pipeline_id, cull_mode, front_face, depth_write_enabled, depth_compare, write_mask) in &unique_pipeline_configs {
                if write_mask != 0xF {
                    packet.push(GpuCommand::CreatePipelineDepthCullColor {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        depth_format: TARGET_FORMAT_DEPTH24PLUS,
                        depth_write_enabled,
                        depth_compare,
                        cull_mode,
                        front_face,
                        write_mask,
                    });
                } else {
                    packet.push(GpuCommand::CreatePipelineDepthCull {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        depth_format: TARGET_FORMAT_DEPTH24PLUS,
                        depth_write_enabled,
                        depth_compare,
                        cull_mode,
                        front_face,
                    });
                }
            }
        } else {
            for &(pipeline_id, cull_mode, front_face, _, _, write_mask) in &unique_pipeline_configs {
                if write_mask != 0xF {
                    packet.push(GpuCommand::CreatePipelineDepthCullColor {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        depth_format: TARGET_FORMAT_DEPTH24PLUS,
                        depth_write_enabled: false,
                        depth_compare: DEPTH_COMPARE_ALWAYS,
                        cull_mode,
                        front_face,
                        write_mask,
                    });
                } else {
                    packet.push(GpuCommand::CreatePipelineCull {
                        pipeline_id,
                        wgsl_code: wgsl_code.clone(),
                        target_format: TARGET_FORMAT_PREFERRED_CANVAS,
                        has_vertex_buffer: true,
                        has_uniform_buffer: true,
                        uniform_size: 144,
                        vertex_stride: vertex_stride_u32,
                        cull_mode,
                        front_face,
                    });
                }
            }
        }
    } else if let Some(depth) = depth_opts {
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
        first.width,
        first.height,
        CanvasFormat::Bgra8Unorm,
    );
    let canvas_output = tracker
        .begin_frame_acquire(CanvasId::new(MESH_CANVAS_TARGET_ID))
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("canvas acquire: {e:?}")))?;

    let root_ctx = RenderContext::new_canvas_acquired(
        ResourceId::new(MESH_CANVAS_TARGET_ID),
        first.width,
        first.height,
        Epoch::new(1),
        canvas_output.epoch,
    );
    let mut session = FrameSession::new(root_ctx, 256)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("FrameSession::new: {e:?}")))?
        .with_uniform_buffer_id(MESH_UNIFORM_BUFFER_ID);

    let mut mat_records = Vec::with_capacity(inputs.len());
    for (i, u_bytes) in mesh_uniform_payloads.iter().enumerate() {
        let mat_handle = Handle::<MaterialDomain>::from_raw(i as u32 + 1, 1)
            .map_err(|e| MeshPacketError::SessionError(alloc::format!("Handle::from_raw: {e:?}")))?;
        let rec_mat = session
            .snapshot_material_use(mat_handle, DataVersion::new(1), Epoch::ZERO, u_bytes)
            .map_err(|e| MeshPacketError::SessionError(alloc::format!("snapshot_material_use: {e:?}")))?;
        mat_records.push(rec_mat);
    }

    let depth_attachment = if has_depth {
        Some(DepthStencilAttachment::new_depth_clear(ResourceId::new(MESH_DEPTH_TEXTURE_ID), 1.0))
    } else {
        None
    };

    session
        .begin_render_pass_with_depth("mesh_canvas_render_pass", MESH_CLEAR_COLOR, depth_attachment)
        .map_err(|e| MeshPacketError::SessionError(alloc::format!("begin_render_pass_with_depth: {e:?}")))?;

    for (i, &(first_vertex, v_count)) in mesh_draw_ranges.iter().enumerate() {
        let draw_pipeline_id = if has_explicit_cull || has_per_mesh_depth || has_color_write_override {
            draw_pipeline_ids[i]
        } else {
            MESH_CANVAS_PIPELINE_ID
        };
        session
            .record_direct_draw_with_range(
                draw_pipeline_id,
                MESH_VERTEX_BUFFER_ID,
                [v_count, 1, first_vertex, 0],
                Some(mat_records[i]),
            )
            .map_err(|e| MeshPacketError::SessionError(alloc::format!("record_direct_draw_with_range: {e:?}")))?;
    }

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

fn build_mesh_batch_packet_impl(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
    canvas: bool,
) -> Result<Vec<u8>, MeshPacketError> {
    if vertex_counts.is_empty() {
        return Err(MeshPacketError::EmptyMeshList);
    }
    let num_meshes = vertex_counts.len();
    let expected_mv_len = num_meshes
        .checked_mul(16)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("model_views length overflow".into()))?;
    if model_views.len() != expected_mv_len {
        return Err(MeshPacketError::InvalidMatrixLength {
            name: "model_views",
            len: model_views.len(),
        });
    }
    if projection.len() != 16 {
        return Err(MeshPacketError::InvalidMatrixLength {
            name: "projection",
            len: projection.len(),
        });
    }
    let expected_colors_len = num_meshes
        .checked_mul(4)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("colors length overflow".into()))?;
    if colors.len() != expected_colors_len {
        return Err(MeshPacketError::InvalidColorLength { len: colors.len() });
    }

    let mut total_vertices: usize = 0;
    for &vc in vertex_counts {
        total_vertices = total_vertices
            .checked_add(vc as usize)
            .ok_or_else(|| MeshPacketError::InvalidDimensions("total vertex count overflow".into()))?;
    }
    let expected_pos_len = total_vertices
        .checked_mul(3)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("total positions length overflow".into()))?;
    if positions.len() != expected_pos_len {
        return Err(MeshPacketError::InvalidPositionLength { len: positions.len() });
    }

    let mut inputs = Vec::with_capacity(num_meshes);
    let mut current_v_offset: usize = 0;
    for i in 0..num_meshes {
        let v_count = vertex_counts[i] as usize;
        let pos_start = current_v_offset * 3;
        let pos_end = pos_start + v_count * 3;
        let pos_slice = &positions[pos_start..pos_end];
        let mv_slice = &model_views[i * 16..(i + 1) * 16];
        let col_slice = &colors[i * 4..(i + 1) * 4];

        let input = DynamicMeshInput::try_from_raw(
            pos_slice,
            &[],
            mv_slice,
            projection,
            col_slice,
            width,
            height,
            webgl_depth,
        )?;
        inputs.push(input);
        current_v_offset += v_count;
    }

    let packet = if canvas {
        if depth_test || depth_write {
            build_multi_mesh_canvas_depth_submission(&inputs, depth_test, depth_write, depth_compare)?
        } else {
            build_multi_mesh_canvas_submission(&inputs)?
        }
    } else {
        if depth_test || depth_write {
            build_multi_mesh_depth_submission(&inputs, depth_test, depth_write, depth_compare)?
        } else {
            build_multi_mesh_submission(&inputs)?
        }
    };

    packet
        .encode()
        .map_err(|e| MeshPacketError::EncodeError(alloc::format!("{e:?}")))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a batch of dynamic Three.js meshes into a unified submission packet (wasm-bindgen export).
pub fn f3d_build_mesh_batch_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
    canvas: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_batch_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
        canvas,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a batch of dynamic Three.js meshes into a unified submission packet for host verification and unit tests.
pub fn f3d_build_mesh_batch_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
    canvas: bool,
) -> Result<Vec<u8>, String> {
    build_mesh_batch_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
        canvas,
    )
    .map_err(|e| e.to_string())
}

/// Encodes a batch of dynamic Three.js meshes with explicit face culling, front-face winding,
/// and per-mesh depth testing/writing into a unified submission packet.
pub fn build_mesh_batch_cull_depth_color_packet_impl(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    depth_tests: &[u8],
    depth_writes: &[u8],
    depth_compares: &[u32],
    color_writes: &[u8],
    width: u32,
    height: u32,
    webgl_depth: bool,
    canvas: bool,
) -> Result<Vec<u8>, MeshPacketError> {
    let num_meshes = vertex_counts.len();
    if num_meshes == 0 {
        return Err(MeshPacketError::EmptyMeshList);
    }
    let expected_mv_len = num_meshes
        .checked_mul(16)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("model_views length overflow".into()))?;
    if model_views.len() != expected_mv_len {
        return Err(MeshPacketError::InvalidMatrixLength {
            name: "model_views",
            len: model_views.len(),
        });
    }
    if projection.len() != 16 {
        return Err(MeshPacketError::InvalidMatrixLength {
            name: "projection",
            len: projection.len(),
        });
    }
    let expected_colors_len = num_meshes
        .checked_mul(4)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("colors length overflow".into()))?;
    if colors.len() != expected_colors_len {
        return Err(MeshPacketError::InvalidColorLength { len: colors.len() });
    }
    if cull_modes.len() != num_meshes {
        return Err(MeshPacketError::InvalidCullArrayLength {
            name: "cull_modes",
            expected: num_meshes,
            actual: cull_modes.len(),
        });
    }
    if front_faces.len() != num_meshes {
        return Err(MeshPacketError::InvalidCullArrayLength {
            name: "front_faces",
            expected: num_meshes,
            actual: front_faces.len(),
        });
    }
    if depth_tests.len() != num_meshes {
        return Err(MeshPacketError::InvalidDepthArrayLength {
            name: "depth_tests",
            expected: num_meshes,
            actual: depth_tests.len(),
        });
    }
    if depth_writes.len() != num_meshes {
        return Err(MeshPacketError::InvalidDepthArrayLength {
            name: "depth_writes",
            expected: num_meshes,
            actual: depth_writes.len(),
        });
    }
    if depth_compares.len() != num_meshes {
        return Err(MeshPacketError::InvalidDepthArrayLength {
            name: "depth_compares",
            expected: num_meshes,
            actual: depth_compares.len(),
        });
    }
    if color_writes.len() != num_meshes {
        return Err(MeshPacketError::InvalidColorWriteArrayLength {
            expected: num_meshes,
            actual: color_writes.len(),
        });
    }
    for &cm in cull_modes {
        if cm > 2 {
            return Err(MeshPacketError::InvalidCullMode { value: cm });
        }
    }
    for &ff in front_faces {
        if ff > 1 {
            return Err(MeshPacketError::InvalidFrontFace { value: ff });
        }
    }
    for &dt in depth_tests {
        if dt > 1 {
            return Err(MeshPacketError::InvalidDepthTest { value: dt });
        }
    }
    for &dw in depth_writes {
        if dw > 1 {
            return Err(MeshPacketError::InvalidDepthWrite { value: dw });
        }
    }
    for &dc in depth_compares {
        if !(1..=8).contains(&dc) {
            return Err(MeshPacketError::InvalidDepthCompare { value: dc });
        }
    }
    for &cw in color_writes {
        if cw > 1 {
            return Err(MeshPacketError::InvalidColorWrite { value: cw });
        }
    }

    let mut total_vertices: usize = 0;
    for &vc in vertex_counts {
        total_vertices = total_vertices
            .checked_add(vc as usize)
            .ok_or_else(|| MeshPacketError::InvalidDimensions("total vertex count overflow".into()))?;
    }
    let expected_pos_len = total_vertices
        .checked_mul(3)
        .ok_or_else(|| MeshPacketError::InvalidDimensions("total positions length overflow".into()))?;
    if positions.len() != expected_pos_len {
        return Err(MeshPacketError::InvalidPositionLength { len: positions.len() });
    }

    let mut inputs = Vec::with_capacity(num_meshes);
    let mut current_v_offset: usize = 0;
    for i in 0..num_meshes {
        let v_count = vertex_counts[i] as usize;
        let pos_start = current_v_offset * 3;
        let pos_end = pos_start + v_count * 3;
        let pos_slice = &positions[pos_start..pos_end];
        let mv_slice = &model_views[i * 16..(i + 1) * 16];
        let col_slice = &colors[i * 4..(i + 1) * 4];

        let input = DynamicMeshInput::try_from_raw(
            pos_slice,
            &[],
            mv_slice,
            projection,
            col_slice,
            width,
            height,
            webgl_depth,
        )?
        .with_cull(cull_modes[i] as u32, front_faces[i] as u32)?
        .with_depth_options(depth_tests[i] != 0, depth_writes[i] != 0, depth_compares[i])?
        .with_color_write(color_writes[i] != 0);
        inputs.push(input);
        current_v_offset += v_count;
    }

    let packet = if canvas {
        build_multi_mesh_canvas_submission_internal(&inputs, None)?
    } else {
        build_multi_mesh_submission_internal(&inputs, None)?
    };

    packet
        .encode()
        .map_err(|e| MeshPacketError::EncodeError(alloc::format!("{e:?}")))
}

/// Encodes a batch of dynamic Three.js meshes with explicit face culling, front-face winding, and per-mesh depth into a unified submission packet.
pub fn build_mesh_batch_cull_depth_packet_impl(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    depth_tests: &[u8],
    depth_writes: &[u8],
    depth_compares: &[u32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    canvas: bool,
) -> Result<Vec<u8>, MeshPacketError> {
    let color_writes = alloc::vec![1u8; vertex_counts.len()];
    build_mesh_batch_cull_depth_color_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        depth_tests,
        depth_writes,
        depth_compares,
        &color_writes,
        width,
        height,
        webgl_depth,
        canvas,
    )
}

/// Encodes a batch of dynamic Three.js meshes with explicit face culling and front-face winding into a unified submission packet.
pub fn build_mesh_batch_cull_packet_impl(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
    canvas: bool,
) -> Result<Vec<u8>, MeshPacketError> {
    let num_meshes = vertex_counts.len();
    let effective_compare = if !depth_test && !depth_write && !(1..=8).contains(&depth_compare) {
        DEPTH_COMPARE_ALWAYS
    } else {
        depth_compare
    };
    let depth_tests = alloc::vec![if depth_test { 1u8 } else { 0u8 }; num_meshes];
    let depth_writes = alloc::vec![if depth_write { 1u8 } else { 0u8 }; num_meshes];
    let depth_compares = alloc::vec![effective_compare; num_meshes];

    build_mesh_batch_cull_depth_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        &depth_tests,
        &depth_writes,
        &depth_compares,
        width,
        height,
        webgl_depth,
        canvas,
    )
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a batch of dynamic Three.js meshes with explicit face culling and front-face winding into a unified submission packet (wasm-bindgen export).
pub fn f3d_build_mesh_batch_cull_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
    canvas: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_batch_cull_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
        canvas,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a batch of dynamic Three.js meshes with explicit face culling, front-face winding, and per-mesh depth into a unified submission packet (wasm-bindgen export).
pub fn f3d_build_mesh_batch_cull_depth_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    depth_tests: &[u8],
    depth_writes: &[u8],
    depth_compares: &[u32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    canvas: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_batch_cull_depth_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        depth_tests,
        depth_writes,
        depth_compares,
        width,
        height,
        webgl_depth,
        canvas,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen]
/// Encodes a batch of dynamic Three.js meshes with explicit face culling, front-face winding, per-mesh depth, and per-mesh color write into a unified submission packet (wasm-bindgen export).
pub fn f3d_build_mesh_batch_cull_depth_color_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    depth_tests: &[u8],
    depth_writes: &[u8],
    depth_compares: &[u32],
    color_writes: &[u8],
    width: u32,
    height: u32,
    webgl_depth: bool,
    canvas: bool,
) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
    build_mesh_batch_cull_depth_color_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        depth_tests,
        depth_writes,
        depth_compares,
        color_writes,
        width,
        height,
        webgl_depth,
        canvas,
    )
    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a batch of dynamic Three.js meshes with explicit face culling and front-face winding into a unified submission packet for host verification and unit tests.
pub fn f3d_build_mesh_batch_cull_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    width: u32,
    height: u32,
    webgl_depth: bool,
    depth_test: bool,
    depth_write: bool,
    depth_compare: u32,
    canvas: bool,
) -> Result<Vec<u8>, String> {
    build_mesh_batch_cull_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        width,
        height,
        webgl_depth,
        depth_test,
        depth_write,
        depth_compare,
        canvas,
    )
    .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a batch of dynamic Three.js meshes with explicit face culling, front-face winding, and per-mesh depth into a unified submission packet for host verification and unit tests.
pub fn f3d_build_mesh_batch_cull_depth_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    depth_tests: &[u8],
    depth_writes: &[u8],
    depth_compares: &[u32],
    width: u32,
    height: u32,
    webgl_depth: bool,
    canvas: bool,
) -> Result<Vec<u8>, String> {
    build_mesh_batch_cull_depth_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        depth_tests,
        depth_writes,
        depth_compares,
        width,
        height,
        webgl_depth,
        canvas,
    )
    .map_err(|e| e.to_string())
}

#[cfg(not(all(feature = "browser", target_arch = "wasm32")))]
/// Encodes a batch of dynamic Three.js meshes with explicit face culling, front-face winding, per-mesh depth, and per-mesh color write into a unified submission packet for host verification and unit tests.
pub fn f3d_build_mesh_batch_cull_depth_color_packet(
    positions: &[f32],
    vertex_counts: &[u32],
    model_views: &[f64],
    projection: &[f64],
    colors: &[f32],
    cull_modes: &[u8],
    front_faces: &[u8],
    depth_tests: &[u8],
    depth_writes: &[u8],
    depth_compares: &[u32],
    color_writes: &[u8],
    width: u32,
    height: u32,
    webgl_depth: bool,
    canvas: bool,
) -> Result<Vec<u8>, String> {
    build_mesh_batch_cull_depth_color_packet_impl(
        positions,
        vertex_counts,
        model_views,
        projection,
        colors,
        cull_modes,
        front_faces,
        depth_tests,
        depth_writes,
        depth_compares,
        color_writes,
        width,
        height,
        webgl_depth,
        canvas,
    )
    .map_err(|e| e.to_string())
}
