//! Checked GPU wire layout, alignment rules, and byte encoding for FrankenThreeD.
//!
//! Position in architecture: `f3d-core` foundational transport layout.
//! Invariant: "Layout is explicit, generated, and tested."
//!
//! First-party semantic, numerical, compiler, scene, and resource code uses `#![forbid(unsafe_code)]`.
//! This module serializes and deserializes all records safely via byte-level conversions
//! without unchecked memory casts, raw pointer arithmetic, or reliance on undefined memory layouts.

extern crate alloc;

use alloc::string::String;
use core::fmt;

/// Size in bytes of a packed `AffineRows` record on the GPU wire (3 * vec4<f32> = 48 bytes).
pub const AFFINE_ROWS_BYTES: usize = 48;

/// Alignment in bytes of an `AffineRows` record (alignof(vec4<f32>) = 16 bytes).
pub const AFFINE_ROWS_ALIGNMENT: usize = 16;

/// Size in bytes of a full `ProjectiveMat4` record (4 * vec4<f32> = 64 bytes).
pub const PROJECTIVE_MAT4_BYTES: usize = 64;

/// Alignment in bytes of a `ProjectiveMat4` record (16 bytes).
pub const PROJECTIVE_MAT4_ALIGNMENT: usize = 16;

/// Size in bytes of a single RGBA color uniform record (`vec4<f32>` = 16 bytes).
pub const COLOR_UNIFORM_BYTES: usize = 16;

/// Alignment in bytes of a color uniform record (`alignof(vec4<f32>)` = 16 bytes under WGSL rules).
pub const COLOR_UNIFORM_ALIGNMENT: usize = 16;

/// Size in bytes of a material uniform parameter block record (`MaterialParams` = 96 bytes).
pub const MATERIAL_PARAMS_BYTES: usize = 96;

/// Alignment in bytes of a material uniform parameter block record under WGSL uniform rules (16 bytes).
pub const MATERIAL_PARAMS_ALIGNMENT: usize = 16;

/// Size in bytes of a standard mesh uniform record (`MeshUniforms` = 144 bytes).
pub const MESH_UNIFORMS_BYTES: usize = 144;

/// Alignment in bytes of a mesh uniform record under WGSL uniform rules (16 bytes).
pub const MESH_UNIFORMS_ALIGNMENT: usize = 16;

/// Size in bytes of a Toon mesh uniform record (`ToonMeshUniforms` = 304 bytes).
pub const TOON_MESH_UNIFORMS_BYTES: usize = 304;

/// Alignment in bytes of a Toon mesh uniform record under WGSL uniform rules (16 bytes).
pub const TOON_MESH_UNIFORMS_ALIGNMENT: usize = 16;

/// Dynamic uniform buffer offset stride for Toon mesh draws (512 bytes).
///
/// Guaranteed to satisfy WebGPU `minUniformBufferOffsetAlignment` (256 bytes) while
/// enclosing the 304-byte payload.
pub const TOON_MESH_DYNAMIC_OFFSET_STRIDE: usize = 512;

/// Material flag: diffuse texture map is enabled.
pub const MATERIAL_FLAG_MAP: u32 = 1 << 0;
/// Material flag: alpha texture map is enabled.
pub const MATERIAL_FLAG_ALPHA_MAP: u32 = 1 << 1;
/// Material flag: specular / environment map is enabled.
pub const MATERIAL_FLAG_ENV_MAP: u32 = 1 << 2;
/// Material flag: vertex colors are active.
pub const MATERIAL_FLAG_VERTEX_COLORS: u32 = 1 << 3;
/// Material flag: wireframe mode is active.
pub const MATERIAL_FLAG_WIREFRAME: u32 = 1 << 4;
/// Material flag: scene fog is enabled.
pub const MATERIAL_FLAG_FOG: u32 = 1 << 5;
/// Material flag: alpha test threshold is enabled.
pub const MATERIAL_FLAG_ALPHA_TEST: u32 = 1 << 6;
/// Material flag: transparent blending is enabled.
pub const MATERIAL_FLAG_TRANSPARENT: u32 = 1 << 7;

/// Size in bytes of the canonical position + UV vertex record (`vec3<f32>` pos [12B] + `vec2<f32>` uv [8B] = 20 bytes).
pub const VERTEX_POS_UV_BYTES: usize = 20;

/// Byte stride for the canonical position + UV vertex buffer layout (20 bytes).
pub const VERTEX_POS_UV_STRIDE: usize = 20;

/// Alignment in bytes of the canonical position + UV vertex record (4 bytes for f32).
pub const VERTEX_POS_UV_ALIGNMENT: usize = 4;

/// Size in bytes of the position + normal + UV vertex record (`vec3<f32>` pos [12B] + `vec3<f32>` norm [12B] + `vec2<f32>` uv [8B] = 32 bytes).
pub const VERTEX_POS_NORMAL_UV_BYTES: usize = 32;

/// Byte stride for the position + normal + UV vertex buffer layout (32 bytes).
pub const VERTEX_POS_NORMAL_UV_STRIDE: usize = 32;

/// Alignment in bytes of the position + normal + UV vertex record (4 bytes for f32).
pub const VERTEX_POS_NORMAL_UV_ALIGNMENT: usize = 4;

/// Size in bytes of the position + color vertex record (`vec3<f32>` pos [12B] + `vec4<f32>` color [16B] = 28 bytes).
pub const VERTEX_POS_COLOR_BYTES: usize = 28;

/// Byte stride for the position + color vertex buffer layout (28 bytes).
pub const VERTEX_POS_COLOR_STRIDE: usize = 28;

/// Alignment in bytes of the position + color vertex record (4 bytes for f32).
pub const VERTEX_POS_COLOR_ALIGNMENT: usize = 4;

/// Size in bytes of a native WGSL `mat4x3<f32>` (4 columns of vec3<f32>, each padded to 16 bytes = 64 bytes).
///
/// Under WGSL layout rules, `alignof(vec3<f32>) == 16` and `sizeof(vec3<f32>) == 12`, but stride
/// in matrices is rounded to column alignment (16 bytes). Thus `mat4x3` occupies 64 bytes on the wire.
pub const WGSL_MAT4X3_BYTES: usize = 64;

/// Required alignment for WebGPU `writeBuffer` offsets and sizes (4 bytes).
pub const WRITE_BUFFER_ALIGNMENT: usize = 4;

/// Required alignment for WebGPU buffer-to-texture and texture-to-buffer copy `bytesPerRow` (256 bytes).
pub const COPY_BYTES_PER_ROW_ALIGNMENT: usize = 256;

/// Default `minUniformBufferOffsetAlignment` guaranteed by the WebGPU specification (256 bytes).
pub const DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT: usize = 256;

/// Default `minStorageBufferOffsetAlignment` guaranteed by the WebGPU specification (256 bytes).
pub const DEFAULT_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT: usize = 256;

/// Errors occurring during layout verification, byte encoding, or buffer alignment checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LayoutError {
    /// Provided destination or source buffer slice is too small.
    BufferTooSmall {
        /// Bytes required by the record layout.
        required: usize,
        /// Bytes provided in the slice.
        provided: usize,
    },
    /// Buffer offset violates operation or device alignment requirements.
    UnalignedOffset {
        /// The tested byte offset.
        offset: usize,
        /// Required byte alignment.
        required_alignment: usize,
    },
    /// A matrix expected to be affine (row 3 = [0, 0, 0, 1]) contains perspective or non-affine components.
    NonAffineMatrix,
    /// An operation attempted to write or map a layout into an incompatible GPU representation.
    IncompatibleTargetLayout {
        /// Target byte size.
        target_size: usize,
        /// Source byte size.
        source_size: usize,
    },
    /// Texture copy row-pitch violates the 256-byte alignment rule.
    UnalignedBytesPerRow {
        /// Provided bytesPerRow.
        bytes_per_row: u32,
        /// Required alignment (256).
        required_alignment: u32,
    },
    /// Size or offset violates WebGPU writeBuffer 4-byte copy alignment.
    UnalignedWriteBuffer {
        /// Offset or length that failed alignment.
        value: usize,
    },
    /// Arithmetic overflow occurred during layout, stride, or pitch calculation.
    CalculationOverflow,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { required, provided } => {
                write!(f, "buffer too small: required {required} bytes, provided {provided} bytes")
            }
            Self::UnalignedOffset { offset, required_alignment } => {
                write!(f, "unaligned offset {offset}: must be a multiple of {required_alignment}")
            }
            Self::NonAffineMatrix => {
                write!(f, "matrix is non-affine: perspective elements are non-zero or scale is invalid")
            }
            Self::IncompatibleTargetLayout { target_size, source_size } => {
                write!(
                    f,
                    "incompatible GPU layout target: target is {target_size} bytes, source record is {source_size} bytes"
                )
            }
            Self::UnalignedBytesPerRow { bytes_per_row, required_alignment } => {
                write!(
                    f,
                    "unaligned bytesPerRow {bytes_per_row}: must be a multiple of {required_alignment}"
                )
            }
            Self::UnalignedWriteBuffer { value } => {
                write!(f, "writeBuffer parameter {value} must be a multiple of 4 bytes")
            }
            Self::CalculationOverflow => {
                write!(f, "arithmetic overflow during layout, stride, or pitch calculation")
            }
        }
    }
}

impl core::error::Error for LayoutError {}

/// GPU wire representation of an affine 3D transform as three `vec4<f32>` rows (48 bytes total).
///
/// Mapping from Three.js column-major `Matrix4.elements` (`e[0]..e[15]`):
/// ```text
/// Row 0: [e[0], e[4], e[8],  e[12]]
/// Row 1: [e[1], e[5], e[9],  e[13]]
/// Row 2: [e[2], e[6], e[10], e[14]]
/// ```
/// Row 3 is implicitly `[0, 0, 0, 1]` and omitted from the 48-byte record.
/// On the GPU, point transformation is evaluated by row dot-products with `(x, y, z, 1.0)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineRows {
    /// Row 0: X-axis linear basis in xyz, translation X in w.
    pub r0: [f32; 4],
    /// Row 1: Y-axis linear basis in xyz, translation Y in w.
    pub r1: [f32; 4],
    /// Row 2: Z-axis linear basis in xyz, translation Z in w.
    pub r2: [f32; 4],
}

impl AffineRows {
    /// Constructs an `AffineRows` record from explicit row vectors.
    pub const fn new(r0: [f32; 4], r1: [f32; 4], r2: [f32; 4]) -> Self {
        Self { r0, r1, r2 }
    }

    /// Returns the standard 3D identity transform in affine row form.
    pub const fn identity() -> Self {
        Self {
            r0: [1.0, 0.0, 0.0, 0.0],
            r1: [0.0, 1.0, 0.0, 0.0],
            r2: [0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Size in bytes of this layout record (48 bytes).
    pub const fn byte_size() -> usize {
        AFFINE_ROWS_BYTES
    }

    /// Byte alignment of this layout record (16 bytes).
    pub const fn alignment() -> usize {
        AFFINE_ROWS_ALIGNMENT
    }

    /// Converts a Three.js column-major 4x4 matrix into `AffineRows`.
    ///
    /// Invariant: Exact structural check. Row 3 must be exactly `[0.0, 0.0, 0.0, 1.0]`.
    /// If any perspective component is non-zero or scale is not 1.0, returns `Err(LayoutError::NonAffineMatrix)`.
    pub fn from_column_major(e: &[f32; 16]) -> Result<Self, LayoutError> {
        if !is_matrix4_affine(e) {
            return Err(LayoutError::NonAffineMatrix);
        }
        Ok(Self::from_column_major_unchecked(e))
    }

    /// Converts an `f64` Three.js column-major 4x4 matrix into `AffineRows` after verifying affine structure
    /// at full `f64` precision before narrowing to `f32`.
    ///
    /// Returns `Err(LayoutError::NonAffineMatrix)` if any perspective coefficient is non-zero
    /// or if `e[15] != 1.0`.
    pub fn from_column_major_f64(e: &[f64; 16]) -> Result<Self, LayoutError> {
        if !is_matrix4_f64_affine(e) {
            return Err(LayoutError::NonAffineMatrix);
        }
        Ok(Self::from_column_major_f64_unchecked(e))
    }

    /// Converts a Three.js column-major 4x4 matrix into `AffineRows` without verifying row 3.
    pub const fn from_column_major_unchecked(e: &[f32; 16]) -> Self {
        Self {
            r0: [e[0], e[4], e[8], e[12]],
            r1: [e[1], e[5], e[9], e[13]],
            r2: [e[2], e[6], e[10], e[14]],
        }
    }

    /// Converts an `f64` Three.js column-major 4x4 matrix into `AffineRows` without verifying row 3.
    pub const fn from_column_major_f64_unchecked(e: &[f64; 16]) -> Self {
        Self {
            r0: [e[0] as f32, e[4] as f32, e[8] as f32, e[12] as f32],
            r1: [e[1] as f32, e[5] as f32, e[9] as f32, e[13] as f32],
            r2: [e[2] as f32, e[6] as f32, e[10] as f32, e[14] as f32],
        }
    }

    /// Expands `AffineRows` back into a full 16-element column-major 4x4 matrix.
    pub const fn to_column_major(&self) -> [f32; 16] {
        [
            self.r0[0], self.r1[0], self.r2[0], 0.0,
            self.r0[1], self.r1[1], self.r2[1], 0.0,
            self.r0[2], self.r1[2], self.r2[2], 0.0,
            self.r0[3], self.r1[3], self.r2[3], 1.0,
        ]
    }

    /// Transforms a 3D point `(x, y, z)` by evaluating row dot-products with `(x, y, z, 1.0)`.
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        [
            self.r0[0] * x + self.r0[1] * y + self.r0[2] * z + self.r0[3],
            self.r1[0] * x + self.r1[1] * y + self.r1[2] * z + self.r1[3],
            self.r2[0] * x + self.r2[1] * y + self.r2[2] * z + self.r2[3],
        ]
    }

    /// Transforms a 3D direction vector `(x, y, z)` ignoring translation (`w = 0.0`).
    pub fn transform_vector(&self, v: [f32; 3]) -> [f32; 3] {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        [
            self.r0[0] * x + self.r0[1] * y + self.r0[2] * z,
            self.r1[0] * x + self.r1[1] * y + self.r1[2] * z,
            self.r2[0] * x + self.r2[1] * y + self.r2[2] * z,
        ]
    }

    /// Safely writes the 48-byte wire representation into a mutable byte slice in little-endian order.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < AFFINE_ROWS_BYTES {
            return Err(LayoutError::BufferTooSmall {
                required: AFFINE_ROWS_BYTES,
                provided: out.len(),
            });
        }
        let bytes = self.to_bytes();
        out[..AFFINE_ROWS_BYTES].copy_from_slice(&bytes);
        Ok(())
    }

    /// Serializes the record into an exact 48-byte array.
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut out = [0u8; 48];
        let write_f32 = |slice: &mut [u8], offset: usize, val: f32| {
            slice[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        };
        write_f32(&mut out, 0, self.r0[0]);
        write_f32(&mut out, 4, self.r0[1]);
        write_f32(&mut out, 8, self.r0[2]);
        write_f32(&mut out, 12, self.r0[3]);

        write_f32(&mut out, 16, self.r1[0]);
        write_f32(&mut out, 20, self.r1[1]);
        write_f32(&mut out, 24, self.r1[2]);
        write_f32(&mut out, 28, self.r1[3]);

        write_f32(&mut out, 32, self.r2[0]);
        write_f32(&mut out, 36, self.r2[1]);
        write_f32(&mut out, 40, self.r2[2]);
        write_f32(&mut out, 44, self.r2[3]);
        out
    }

    /// Safely reads the 48-byte wire representation from a byte slice in little-endian order.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < AFFINE_ROWS_BYTES {
            return Err(LayoutError::BufferTooSmall {
                required: AFFINE_ROWS_BYTES,
                provided: src.len(),
            });
        }
        let mut b = [0u8; 48];
        b.copy_from_slice(&src[..AFFINE_ROWS_BYTES]);
        Ok(Self::from_bytes(&b))
    }

    /// Deserializes `AffineRows` from an exact 48-byte array.
    pub fn from_bytes(bytes: &[u8; 48]) -> Self {
        let read_f32 = |offset: usize| -> f32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            f32::from_le_bytes(b)
        };
        Self {
            r0: [read_f32(0), read_f32(4), read_f32(8), read_f32(12)],
            r1: [read_f32(16), read_f32(20), read_f32(24), read_f32(28)],
            r2: [read_f32(32), read_f32(36), read_f32(40), read_f32(44)],
        }
    }
}

/// Full projective 4x4 matrix representation retained when a matrix is non-affine (64 bytes).
///
/// Used for perspective camera projections, orthographic projections with perspective shears,
/// and non-affine composite transforms.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProjectiveMat4 {
    /// 16 floats in standard Three.js column-major order `e[0]..e[15]`.
    pub elements: [f32; 16],
}

impl ProjectiveMat4 {
    /// Constructs a `ProjectiveMat4` from 16 column-major float values.
    pub const fn from_elements(elements: [f32; 16]) -> Self {
        Self { elements }
    }

    /// Returns the standard 4x4 identity matrix.
    pub const fn identity() -> Self {
        Self {
            elements: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Size in bytes of this layout record (64 bytes).
    pub const fn byte_size() -> usize {
        PROJECTIVE_MAT4_BYTES
    }

    /// Byte alignment of this layout record (16 bytes).
    pub const fn alignment() -> usize {
        PROJECTIVE_MAT4_ALIGNMENT
    }

    /// Checks if this matrix represents an affine transformation.
    pub fn is_affine(&self) -> bool {
        is_matrix4_affine(&self.elements)
    }

    /// Converts this matrix to `AffineRows`, failing if the matrix is non-affine.
    pub fn to_affine_rows(&self) -> Result<AffineRows, LayoutError> {
        AffineRows::from_column_major(&self.elements)
    }

    /// Transforms a 4D homogeneous vector `(x, y, z, w)` by multiplying `M * v`.
    pub fn transform_homogeneous(&self, v: [f32; 4]) -> [f32; 4] {
        let e = &self.elements;
        [
            e[0] * v[0] + e[4] * v[1] + e[8] * v[2] + e[12] * v[3],
            e[1] * v[0] + e[5] * v[1] + e[9] * v[2] + e[13] * v[3],
            e[2] * v[0] + e[6] * v[1] + e[10] * v[2] + e[14] * v[3],
            e[3] * v[0] + e[7] * v[1] + e[11] * v[2] + e[15] * v[3],
        ]
    }

    /// Safely writes the 64-byte wire representation into a mutable byte slice in little-endian order.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < PROJECTIVE_MAT4_BYTES {
            return Err(LayoutError::BufferTooSmall {
                required: PROJECTIVE_MAT4_BYTES,
                provided: out.len(),
            });
        }
        let bytes = self.to_bytes();
        out[..PROJECTIVE_MAT4_BYTES].copy_from_slice(&bytes);
        Ok(())
    }

    /// Serializes the record into an exact 64-byte array.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for i in 0..16 {
            let b = self.elements[i].to_le_bytes();
            out[i * 4..i * 4 + 4].copy_from_slice(&b);
        }
        out
    }

    /// Safely reads the 64-byte wire representation from a byte slice in little-endian order.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < PROJECTIVE_MAT4_BYTES {
            return Err(LayoutError::BufferTooSmall {
                required: PROJECTIVE_MAT4_BYTES,
                provided: src.len(),
            });
        }
        let mut b = [0u8; 64];
        b.copy_from_slice(&src[..PROJECTIVE_MAT4_BYTES]);
        Ok(Self::from_bytes(&b))
    }

    /// Deserializes `ProjectiveMat4` from an exact 64-byte array.
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let mut elements = [0.0f32; 16];
        for i in 0..16 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
            elements[i] = f32::from_le_bytes(b);
        }
        Self { elements }
    }
    /// Converts an `f64` 16-element column-major matrix to `ProjectiveMat4` by narrowing elements to `f32`.
    pub fn from_elements_f64(e: &[f64; 16]) -> Self {
        let mut elements = [0.0f32; 16];
        for i in 0..16 {
            elements[i] = e[i] as f32;
        }
        Self { elements }
    }
}

/// Checks whether a 4x4 matrix in column-major order represents an affine transformation.
///
/// Invariant: Exact structural check. An affine 4x4 matrix must have row 3 equal to
/// `[0.0, 0.0, 0.0, 1.0]` exactly (`e[3] == 0.0 && e[7] == 0.0 && e[11] == 0.0 && e[15] == 1.0`).
///
/// No epsilon is permitted: even small perspective terms (e.g. `e[3] = 5e-7`)
/// produce significant perspective divide effects on large coordinates (e.g. `x = 1e7 => w = 6`),
/// and silently discarding them distorts geometry. Any non-zero perspective coefficient or `e[15] != 1.0`
/// must be retained in `ProjectiveMat4`.
pub fn is_matrix4_affine(e: &[f32; 16]) -> bool {
    e[3] == 0.0 && e[7] == 0.0 && e[11] == 0.0 && e[15] == 1.0
}

/// Checks whether an `f64` 4x4 matrix in column-major order represents an affine transformation
/// before narrowing to `f32`.
///
/// Invariant: Exact structural test at source precision. If any perspective element is non-zero
/// in `f64` or `e[15] != 1.0`, it cannot be narrowed to an affine record without corrupting
/// the projection.
pub fn is_matrix4_f64_affine(e: &[f64; 16]) -> bool {
    e[3] == 0.0 && e[7] == 0.0 && e[11] == 0.0 && e[15] == 1.0
}

/// Enumeration of distinct GPU matrix memory layout conventions in WebGPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GpuMatrixLayout {
    /// Affine rows representation: 3 rows of `vec4<f32>`, 48 bytes, 16-byte aligned.
    AffineRows48,
    /// Full 4x4 matrix: 4 rows/cols of `vec4<f32>`, 64 bytes, 16-byte aligned.
    ProjectiveMat4x4,
    /// WGSL native `mat4x3<f32>`: 4 columns of `vec3<f32>`, where each `vec3` is padded to 16 bytes.
    /// Total size is 64 bytes (4 * 16 bytes), NOT 48 bytes!
    WgslMat4x3Padded64,
}

impl GpuMatrixLayout {
    /// Expected byte size of this matrix layout under WGSL rules.
    pub const fn byte_size(self) -> usize {
        match self {
            Self::AffineRows48 => AFFINE_ROWS_BYTES,
            Self::ProjectiveMat4x4 => PROJECTIVE_MAT4_BYTES,
            Self::WgslMat4x3Padded64 => WGSL_MAT4X3_BYTES,
        }
    }

    /// Alignment required by this layout under WGSL rules (16 bytes).
    pub const fn alignment(self) -> usize {
        16
    }
}

/// Validates that a target GPU buffer layout can accept tightly-packed affine rows.
///
/// Returns `Err(LayoutError::IncompatibleTargetLayout)` if the target is WGSL `mat4x3`,
/// which requires 64 bytes due to 16-byte column alignment rules, preventing 48-byte packed writes.
pub fn validate_affine_target(target: GpuMatrixLayout) -> Result<(), LayoutError> {
    match target {
        GpuMatrixLayout::AffineRows48 => Ok(()),
        other => Err(LayoutError::IncompatibleTargetLayout {
            target_size: other.byte_size(),
            source_size: AFFINE_ROWS_BYTES,
        }),
    }
}

/// Validates dynamic uniform buffer binding offset alignment.
///
/// WebGPU requires dynamic uniform buffer offsets to be multiples of `minUniformBufferOffsetAlignment` (default 256).
pub fn validate_dynamic_uniform_offset(offset: usize, min_alignment: usize) -> Result<(), LayoutError> {
    if min_alignment == 0 || offset % min_alignment != 0 {
        Err(LayoutError::UnalignedOffset {
            offset,
            required_alignment: min_alignment,
        })
    } else {
        Ok(())
    }
}

/// Validates dynamic storage buffer binding offset alignment.
///
/// WebGPU requires dynamic storage buffer offsets to be multiples of `minStorageBufferOffsetAlignment` (default 256).
pub fn validate_dynamic_storage_offset(offset: usize, min_alignment: usize) -> Result<(), LayoutError> {
    if min_alignment == 0 || offset % min_alignment != 0 {
        Err(LayoutError::UnalignedOffset {
            offset,
            required_alignment: min_alignment,
        })
    } else {
        Ok(())
    }
}

/// Validates storage buffer array stride for an element type with explicit alignment.
///
/// Under WGSL layout rules, storage array stride must be >= element size and a multiple of `element_alignment`.
pub fn validate_storage_array_stride(
    stride: usize,
    element_size: usize,
    element_alignment: usize,
) -> Result<(), LayoutError> {
    if stride < element_size {
        return Err(LayoutError::BufferTooSmall {
            required: element_size,
            provided: stride,
        });
    }
    if element_alignment == 0 || stride % element_alignment != 0 {
        return Err(LayoutError::UnalignedOffset {
            offset: stride,
            required_alignment: element_alignment,
        });
    }
    Ok(())
}

/// Validates storage buffer array stride for a 16-byte aligned composite record (e.g. `AffineRows`).
///
/// Invariant: Composite records containing `vec4` rows require 16-byte stride alignment.
pub fn validate_composite_storage_array_stride(
    stride: usize,
    element_size: usize,
) -> Result<(), LayoutError> {
    validate_storage_array_stride(stride, element_size, 16)
}

/// Validates WebGPU writeBuffer copy alignment (must be a multiple of 4 bytes).
pub fn validate_write_buffer_alignment(offset: usize, size: usize) -> Result<(), LayoutError> {
    if offset % WRITE_BUFFER_ALIGNMENT != 0 {
        return Err(LayoutError::UnalignedWriteBuffer { value: offset });
    }
    if size % WRITE_BUFFER_ALIGNMENT != 0 {
        return Err(LayoutError::UnalignedWriteBuffer { value: size });
    }
    Ok(())
}

/// Validates WebGPU texture copy bytesPerRow alignment (must be a multiple of 256 bytes).
pub fn validate_copy_bytes_per_row(bytes_per_row: u32) -> Result<(), LayoutError> {
    if bytes_per_row % (COPY_BYTES_PER_ROW_ALIGNMENT as u32) != 0 {
        Err(LayoutError::UnalignedBytesPerRow {
            bytes_per_row,
            required_alignment: COPY_BYTES_PER_ROW_ALIGNMENT as u32,
        })
    } else {
        Ok(())
    }
}

/// Computes the WebGPU copy row pitch (`bytesPerRow`) for a 4-byte-per-pixel (e.g. RGBA8) texture of given width,
/// aligned to 256 bytes ([`COPY_BYTES_PER_ROW_ALIGNMENT`]), with checked arithmetic overflow.
///
/// Under WebGPU specification:
/// - Each RGBA8 pixel occupies 4 bytes (`width * 4`).
/// - `bytesPerRow` must be a multiple of 256.
///
/// Returns `Err(LayoutError::CalculationOverflow)` if `width * 4` overflows `u32`
/// or if rounding up to the next 256-byte boundary overflows `u32`.
pub fn aligned_bytes_per_row(width: u32) -> Result<u32, LayoutError> {
    let unpadded = width.checked_mul(4).ok_or(LayoutError::CalculationOverflow)?;
    let align = COPY_BYTES_PER_ROW_ALIGNMENT as u32; // 256
    let remainder = unpadded % align;
    let aligned = if remainder == 0 {
        unpadded
    } else {
        unpadded
            .checked_add(align - remainder)
            .ok_or(LayoutError::CalculationOverflow)?
    };
    Ok(aligned)
}

/// Computes the WebGPU copy row pitch (`bytesPerRow`) with custom byte-per-pixel size,
/// aligned to 256 bytes ([`COPY_BYTES_PER_ROW_ALIGNMENT`]), with checked arithmetic overflow.
pub fn aligned_copy_bytes_per_row(width: u32, bytes_per_pixel: u32) -> Result<u32, LayoutError> {
    let unpadded = width
        .checked_mul(bytes_per_pixel)
        .ok_or(LayoutError::CalculationOverflow)?;
    let align = COPY_BYTES_PER_ROW_ALIGNMENT as u32; // 256
    let remainder = unpadded % align;
    let aligned = if remainder == 0 {
        unpadded
    } else {
        unpadded
            .checked_add(align - remainder)
            .ok_or(LayoutError::CalculationOverflow)?
    };
    Ok(aligned)
}


/// Instance transform record layout for GPU instance buffers (world transform + instance ID).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InstanceRecord {
    /// 48-byte packed affine world transform.
    pub transform: AffineRows,
    /// 32-bit instance index.
    pub instance_id: u32,
}

impl InstanceRecord {
    /// Size in bytes of this instance record with WGSL struct padding (64 bytes).
    pub const BYTE_SIZE: usize = 64;

    /// Alignment in bytes (16 bytes).
    pub const ALIGNMENT: usize = 16;

    /// Constructs a new `InstanceRecord`.
    pub const fn new(transform: AffineRows, instance_id: u32) -> Self {
        Self { transform, instance_id }
    }

    /// Safely writes the 64-byte instance record into a slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        self.transform.write_to_slice(&mut out[..AFFINE_ROWS_BYTES])?;
        out[48..52].copy_from_slice(&self.instance_id.to_le_bytes());
        // Trailing 12 bytes of padding to reach 64-byte alignment
        out[52..64].fill(0);
        Ok(())
    }

    /// Safely reads the instance record from a slice.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: src.len(),
            });
        }
        let transform = AffineRows::read_from_slice(&src[..AFFINE_ROWS_BYTES])?;
        let mut id_bytes = [0u8; 4];
        id_bytes.copy_from_slice(&src[48..52]);
        let instance_id = u32::from_le_bytes(id_bytes);
        Ok(Self { transform, instance_id })
    }
}

/// WebGPU DrawIndirect arguments layout (16 bytes, 4-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DrawIndirectArgs {
    /// Number of vertices to draw.
    pub vertex_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Base vertex index.
    pub first_vertex: u32,
    /// Base instance index.
    pub first_instance: u32,
}

impl DrawIndirectArgs {
    /// Size in bytes (16 bytes).
    pub const BYTE_SIZE: usize = 16;

    /// Writes the indirect draw arguments into a byte slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        out[0..4].copy_from_slice(&self.vertex_count.to_le_bytes());
        out[4..8].copy_from_slice(&self.instance_count.to_le_bytes());
        out[8..12].copy_from_slice(&self.first_vertex.to_le_bytes());
        out[12..16].copy_from_slice(&self.first_instance.to_le_bytes());
        Ok(())
    }
}

/// WebGPU DrawIndexedIndirect arguments layout (20 bytes, 4-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DrawIndexedIndirectArgs {
    /// Number of indices to draw.
    pub index_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// First index in the index buffer.
    pub first_index: u32,
    /// Value added to the vertex index before indexing into the vertex buffer.
    pub base_vertex: i32,
    /// First instance index.
    pub first_instance: u32,
}

impl DrawIndexedIndirectArgs {
    /// Size in bytes (20 bytes).
    pub const BYTE_SIZE: usize = 20;

    /// Writes the indexed indirect draw arguments into a byte slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        out[0..4].copy_from_slice(&self.index_count.to_le_bytes());
        out[4..8].copy_from_slice(&self.instance_count.to_le_bytes());
        out[8..12].copy_from_slice(&self.first_index.to_le_bytes());
        out[12..16].copy_from_slice(&self.base_vertex.to_le_bytes());
        out[16..20].copy_from_slice(&self.first_instance.to_le_bytes());
        Ok(())
    }
}

/// Canonical vertex record for GPU vertex buffers containing 3D position and 2D UV coordinates.
///
/// Memory layout:
/// - `position`: `[f32; 3]` at offset 0 (12 bytes, `shaderLocation: 0, format: "float32x3"`)
/// - `uv`: `[f32; 2]` at offset 12 (8 bytes, `shaderLocation: 1, format: "float32x2"`)
/// Total size: 20 bytes ([`VERTEX_POS_UV_BYTES`]), alignment 4 bytes ([`VERTEX_POS_UV_ALIGNMENT`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexPosUv {
    /// 3D position in model space (`[x, y, z]`).
    pub position: [f32; 3],
    /// 2D texture coordinates (`[u, v]`).
    pub uv: [f32; 2],
}

impl VertexPosUv {
    /// Byte size of this vertex record (20 bytes).
    pub const BYTE_SIZE: usize = VERTEX_POS_UV_BYTES;

    /// Byte stride for vertex buffer layouts (20 bytes).
    pub const STRIDE: usize = VERTEX_POS_UV_STRIDE;

    /// Alignment in bytes (4 bytes).
    pub const ALIGNMENT: usize = VERTEX_POS_UV_ALIGNMENT;

    /// Constructs a new `VertexPosUv` record.
    pub const fn new(position: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, uv }
    }

    /// Serializes the vertex record into an exact 20-byte array in little-endian order.
    pub fn to_bytes(&self) -> [u8; 20] {
        let mut out = [0u8; 20];
        out[0..4].copy_from_slice(&self.position[0].to_le_bytes());
        out[4..8].copy_from_slice(&self.position[1].to_le_bytes());
        out[8..12].copy_from_slice(&self.position[2].to_le_bytes());
        out[12..16].copy_from_slice(&self.uv[0].to_le_bytes());
        out[16..20].copy_from_slice(&self.uv[1].to_le_bytes());
        out
    }

    /// Safely writes the 20-byte wire representation into a mutable byte slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        out[..Self::BYTE_SIZE].copy_from_slice(&self.to_bytes());
        Ok(())
    }

    /// Safely reads the 20-byte vertex record from a slice in little-endian order.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: src.len(),
            });
        }
        let mut b = [0u8; 20];
        b.copy_from_slice(&src[..Self::BYTE_SIZE]);
        Ok(Self::from_bytes(&b))
    }

    /// Deserializes a `VertexPosUv` record from an exact 20-byte array.
    pub fn from_bytes(bytes: &[u8; 20]) -> Self {
        let read_f32 = |offset: usize| -> f32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            f32::from_le_bytes(b)
        };
        Self {
            position: [read_f32(0), read_f32(4), read_f32(8)],
            uv: [read_f32(12), read_f32(16)],
        }
    }
}

/// Canonical vertex record for GPU vertex buffers containing 3D position, 3D surface normal, and 2D UV coordinates.
///
/// Memory layout:
/// - `position`: `[f32; 3]` at offset 0 (12 bytes, `shaderLocation: 0, format: "float32x3"`)
/// - `normal`: `[f32; 3]` at offset 12 (12 bytes, `shaderLocation: 1, format: "float32x3"`)
/// - `uv`: `[f32; 2]` at offset 24 (8 bytes, `shaderLocation: 2, format: "float32x2"`)
/// Total size: 32 bytes ([`VERTEX_POS_NORMAL_UV_BYTES`]), alignment 4 bytes ([`VERTEX_POS_NORMAL_UV_ALIGNMENT`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexPosNormalUv {
    /// 3D position in model space (`[x, y, z]`).
    pub position: [f32; 3],
    /// 3D surface normal vector (`[nx, ny, nz]`).
    pub normal: [f32; 3],
    /// 2D texture coordinates (`[u, v]`).
    pub uv: [f32; 2],
}

impl VertexPosNormalUv {
    /// Byte size of this vertex record (32 bytes).
    pub const BYTE_SIZE: usize = VERTEX_POS_NORMAL_UV_BYTES;

    /// Byte stride for vertex buffer layouts (32 bytes).
    pub const STRIDE: usize = VERTEX_POS_NORMAL_UV_STRIDE;

    /// Alignment in bytes (4 bytes).
    pub const ALIGNMENT: usize = VERTEX_POS_NORMAL_UV_ALIGNMENT;

    /// Constructs a new `VertexPosNormalUv` record.
    pub const fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, normal, uv }
    }

    /// Serializes the vertex record into an exact 32-byte array in little-endian order.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..4].copy_from_slice(&self.position[0].to_le_bytes());
        out[4..8].copy_from_slice(&self.position[1].to_le_bytes());
        out[8..12].copy_from_slice(&self.position[2].to_le_bytes());
        out[12..16].copy_from_slice(&self.normal[0].to_le_bytes());
        out[16..20].copy_from_slice(&self.normal[1].to_le_bytes());
        out[20..24].copy_from_slice(&self.normal[2].to_le_bytes());
        out[24..28].copy_from_slice(&self.uv[0].to_le_bytes());
        out[28..32].copy_from_slice(&self.uv[1].to_le_bytes());
        out
    }

    /// Safely writes the 32-byte wire representation into a mutable byte slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        out[..Self::BYTE_SIZE].copy_from_slice(&self.to_bytes());
        Ok(())
    }

    /// Safely reads the 32-byte vertex record from a slice in little-endian order.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: src.len(),
            });
        }
        let mut b = [0u8; 32];
        b.copy_from_slice(&src[..Self::BYTE_SIZE]);
        Ok(Self::from_bytes(&b))
    }

    /// Deserializes a `VertexPosNormalUv` record from an exact 32-byte array.
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let read_f32 = |offset: usize| -> f32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            f32::from_le_bytes(b)
        };
        Self {
            position: [read_f32(0), read_f32(4), read_f32(8)],
            normal: [read_f32(12), read_f32(16), read_f32(20)],
            uv: [read_f32(24), read_f32(28)],
        }
    }
}

/// Canonical vertex record for GPU vertex buffers containing 3D position and RGBA vertex color.
///
/// Memory layout:
/// - `position`: `[f32; 3]` at offset 0 (12 bytes, `shaderLocation: 0, format: "float32x3"`)
/// - `color`: `[f32; 4]` at offset 12 (16 bytes, `shaderLocation: 1, format: "float32x4"`)
/// Total size: 28 bytes ([`VERTEX_POS_COLOR_BYTES`]), alignment 4 bytes ([`VERTEX_POS_COLOR_ALIGNMENT`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexPosColor {
    /// 3D position in model space (`[x, y, z]`).
    pub position: [f32; 3],
    /// RGBA vertex color (`[r, g, b, a]`).
    pub color: [f32; 4],
}

impl VertexPosColor {
    /// Byte size of this vertex record (28 bytes).
    pub const BYTE_SIZE: usize = VERTEX_POS_COLOR_BYTES;

    /// Byte stride for vertex buffer layouts (28 bytes).
    pub const STRIDE: usize = VERTEX_POS_COLOR_STRIDE;

    /// Alignment in bytes (4 bytes).
    pub const ALIGNMENT: usize = VERTEX_POS_COLOR_ALIGNMENT;

    /// Constructs a new `VertexPosColor` record.
    pub const fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    /// Serializes the vertex record into an exact 28-byte array in little-endian order.
    pub fn to_bytes(&self) -> [u8; 28] {
        let mut out = [0u8; 28];
        out[0..4].copy_from_slice(&self.position[0].to_le_bytes());
        out[4..8].copy_from_slice(&self.position[1].to_le_bytes());
        out[8..12].copy_from_slice(&self.position[2].to_le_bytes());
        out[12..16].copy_from_slice(&self.color[0].to_le_bytes());
        out[16..20].copy_from_slice(&self.color[1].to_le_bytes());
        out[20..24].copy_from_slice(&self.color[2].to_le_bytes());
        out[24..28].copy_from_slice(&self.color[3].to_le_bytes());
        out
    }

    /// Safely writes the 28-byte wire representation into a mutable byte slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        out[..Self::BYTE_SIZE].copy_from_slice(&self.to_bytes());
        Ok(())
    }

    /// Safely reads the 28-byte vertex record from a slice in little-endian order.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: src.len(),
            });
        }
        let mut b = [0u8; 28];
        b.copy_from_slice(&src[..Self::BYTE_SIZE]);
        Ok(Self::from_bytes(&b))
    }

    /// Deserializes a `VertexPosColor` record from an exact 28-byte array.
    pub fn from_bytes(bytes: &[u8; 28]) -> Self {
        let read_f32 = |offset: usize| -> f32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            f32::from_le_bytes(b)
        };
        Self {
            position: [read_f32(0), read_f32(4), read_f32(8)],
            color: [read_f32(12), read_f32(16), read_f32(20), read_f32(24)],
        }
    }
}

/// Field-level layout descriptor for GPU wire data structures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayoutRow {
    /// Name of the record type (e.g. "AffineRows", "VertexPosUv").
    pub record: &'static str,
    /// Name of the struct field or attribute.
    pub field: &'static str,
    /// Byte offset within the record.
    pub offset: usize,
    /// Size in bytes of the field.
    pub size: usize,
    /// Byte alignment of the field.
    pub align: usize,
    /// Corresponding WGSL shader data type.
    pub wgsl_type: &'static str,
}

impl fmt::Display for LayoutRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<24} {:<16} {:>6} {:>6} {:>6}  {:<16}",
            self.record, self.field, self.offset, self.size, self.align, self.wgsl_type
        )
    }
}

/// Helper const function to sort layout rows by ascending byte offset at compile time.
const fn sort_layout_rows<const N: usize>(mut rows: [LayoutRow; N]) -> [LayoutRow; N] {
    let mut i = 0;
    while i < N {
        let mut j = i + 1;
        while j < N {
            if rows[j].offset < rows[i].offset {
                let tmp = rows[i];
                rows[i] = rows[j];
                rows[j] = tmp;
            }
            j += 1;
        }
        i += 1;
    }
    rows
}

/// Shared private definition generating both canonical WGSL shader struct declarations
/// and corresponding compile-time `LayoutRow` catalog slices from a single field list.
///
/// Shader-visible fields are defined once with name, WGSL type, offset, size, and alignment,
/// driving both the WGSL struct declaration and the `LayoutRow` entries.
/// Non-shader-visible padding is listed separately and incorporated into the layout rows.
macro_rules! define_canonical_layout {
    (
        record: $record:expr,
        shader_const: $shader_const:ident,
        layout_rows_const: $layout_const:ident,
        row_count: $row_count:expr,
        fields: [
            $(
                $f_name:expr, $f_wgsl:expr, $f_offset:expr, $f_size:expr, $f_align:expr
            );* $(;)?
        ] $(,)?
    ) => {
        define_canonical_layout! {
            record: $record,
            shader_const: $shader_const,
            layout_rows_const: $layout_const,
            row_count: $row_count,
            fields: [
                $( $f_name, $f_wgsl, $f_offset, $f_size, $f_align );*
            ],
            padding: [],
            extra_shader: "",
        }
    };

    (
        record: $record:expr,
        shader_const: $shader_const:ident,
        layout_rows_const: $layout_const:ident,
        row_count: $row_count:expr,
        fields: [
            $(
                $f_name:expr, $f_wgsl:expr, $f_offset:expr, $f_size:expr, $f_align:expr
            );* $(;)?
        ],
        padding: [
            $(
                $p_name:expr, $p_offset:expr, $p_size:expr, $p_align:expr
            );* $(;)?
        ] $(,)?
    ) => {
        define_canonical_layout! {
            record: $record,
            shader_const: $shader_const,
            layout_rows_const: $layout_const,
            row_count: $row_count,
            fields: [
                $( $f_name, $f_wgsl, $f_offset, $f_size, $f_align );*
            ],
            padding: [
                $( $p_name, $p_offset, $p_size, $p_align );*
            ],
            extra_shader: "",
        }
    };

    (
        record: $record:expr,
        shader_const: $shader_const:ident,
        layout_rows_const: $layout_const:ident,
        row_count: $row_count:expr,
        fields: [
            $(
                $f_name:expr, $f_wgsl:expr, $f_offset:expr, $f_size:expr, $f_align:expr
            );* $(;)?
        ],
        extra_shader: $extra:expr $(,)?
    ) => {
        define_canonical_layout! {
            record: $record,
            shader_const: $shader_const,
            layout_rows_const: $layout_const,
            row_count: $row_count,
            fields: [
                $( $f_name, $f_wgsl, $f_offset, $f_size, $f_align );*
            ],
            padding: [],
            extra_shader: $extra,
        }
    };

    (
        record: $record:expr,
        shader_const: $shader_const:ident,
        layout_rows_const: $layout_const:ident,
        row_count: $row_count:expr,
        fields: [
            $(
                $f_name:expr, $f_wgsl:expr, $f_offset:expr, $f_size:expr, $f_align:expr
            );* $(;)?
        ],
        padding: [
            $(
                $p_name:expr, $p_offset:expr, $p_size:expr, $p_align:expr
            );* $(;)?
        ],
        extra_shader: $extra:expr $(,)?
    ) => {
        /// Canonical WGSL shader declaration.
        pub const $shader_const: &str = concat!(
            "\nstruct ", $record, " {\n",
            $( "    ", $f_name, ": ", $f_wgsl, ",\n", )*
            "};\n",
            $extra
        );

        const $layout_const: [LayoutRow; $row_count] = sort_layout_rows([
            $(
                LayoutRow {
                    record: $record,
                    field: $f_name,
                    offset: $f_offset,
                    size: $f_size,
                    align: $f_align,
                    wgsl_type: $f_wgsl,
                },
            )*
            $(
                LayoutRow {
                    record: $record,
                    field: $p_name,
                    offset: $p_offset,
                    size: $p_size,
                    align: $p_align,
                    wgsl_type: "padding",
                },
            )*
        ]);
    };
}

/// Shared private definition generating canonical WGSL vertex input shader declarations
/// and corresponding compile-time `LayoutRow` catalog slices from a single field list.
///
/// Note: WGSL struct memory layout rules (host-shareable uniform/storage buffers) do NOT apply
/// to packed vertex buffer attributes, where fields are tightly packed according to vertex format.
/// Vertex records are declared as vertex shader stage inputs with explicit `@location(n)` attributes.
macro_rules! define_canonical_vertex_layout {
    (
        record: $record:expr,
        shader_const: $shader_const:ident,
        layout_rows_const: $layout_const:ident,
        row_count: $row_count:expr,
        fields: [
            $(
                $f_loc:expr, $f_name:expr, $f_wgsl:expr, $f_offset:expr, $f_size:expr, $f_align:expr
            );* $(;)?
        ] $(,)?
    ) => {
        define_canonical_vertex_layout! {
            record: $record,
            shader_const: $shader_const,
            layout_rows_const: $layout_const,
            row_count: $row_count,
            fields: [
                $( $f_loc, $f_name, $f_wgsl, $f_offset, $f_size, $f_align );*
            ],
            extra_shader: "",
        }
    };

    (
        record: $record:expr,
        shader_const: $shader_const:ident,
        layout_rows_const: $layout_const:ident,
        row_count: $row_count:expr,
        fields: [
            $(
                $f_loc:expr, $f_name:expr, $f_wgsl:expr, $f_offset:expr, $f_size:expr, $f_align:expr
            );* $(;)?
        ],
        extra_shader: $extra:expr $(,)?
    ) => {
        /// Canonical WGSL vertex input shader declaration.
        pub const $shader_const: &str = concat!(
            "\nstruct ", $record, " {\n",
            $( "    @location(", $f_loc, ") ", $f_name, ": ", $f_wgsl, ",\n", )*
            "};\n",
            $extra
        );

        const $layout_const: [LayoutRow; $row_count] = sort_layout_rows([
            $(
                LayoutRow {
                    record: $record,
                    field: $f_name,
                    offset: $f_offset,
                    size: $f_size,
                    align: $f_align,
                    wgsl_type: $f_wgsl,
                },
            )*
        ]);
    };
}


define_canonical_layout! {
    record: "AffineRows",
    shader_const: WGSL_AFFINE_ROWS_DECLARATION,
    layout_rows_const: AFFINE_ROWS_LAYOUT_ROWS,
    row_count: 3,
    fields: [
        "r0", "vec4<f32>", 0, 16, 16;
        "r1", "vec4<f32>", 16, 16, 16;
        "r2", "vec4<f32>", 32, 16, 16;
    ],
    extra_shader: r#"
fn transform_affine_point(m: AffineRows, p: vec3<f32>) -> vec3<f32> {
    let v = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

fn transform_affine_vector(m: AffineRows, v: vec3<f32>) -> vec3<f32> {
    let d = vec4<f32>(v, 0.0);
    return vec3<f32>(dot(m.r0, d), dot(m.r1, d), dot(m.r2, d));
}

fn affine_to_mat4x4(m: AffineRows) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(m.r0.x, m.r1.x, m.r2.x, 0.0),
        vec4<f32>(m.r0.y, m.r1.y, m.r2.y, 0.0),
        vec4<f32>(m.r0.z, m.r1.z, m.r2.z, 0.0),
        vec4<f32>(m.r0.w, m.r1.w, m.r2.w, 1.0)
    );
}
"#,
}

define_canonical_layout! {
    record: "ProjectiveMat4",
    shader_const: WGSL_PROJECTIVE_MAT4_DECLARATION,
    layout_rows_const: PROJECTIVE_MAT4_LAYOUT_ROWS,
    row_count: 1,
    fields: [
        "elements", "mat4x4<f32>", 0, 64, 16;
    ],
}


/// Canonical GPU uniform parameter block record for materials matching Three.js r186 `MeshBasicMaterial`.
///
/// WGSL uniform buffer address-space alignment rules mandate 16-byte alignment for uniform structs,
/// 16-byte alignment for `vec4<f32>` members, and 4-byte alignment for `f32`/`u32` scalars.
///
/// Memory layout (96 bytes total, 16-byte aligned):
/// - `color`: `[f32; 4]` at offset 0 (16 bytes, `vec4<f32>`, RGBA)
/// - `opacity`: `f32` at offset 16 (4 bytes)
/// - `alpha_test`: `f32` at offset 20 (4 bytes, matching Three.js `alphaTest`)
/// - `_pad0`: `[u8; 8]` at offset 24 (8 bytes alignment padding ensuring `map_transform` starts at offset 32)
/// - `map_transform`: `AffineRows` at offset 32 (48 bytes, 2D UV transform as three `vec4<f32>` rows)
/// - `flags`: `u32` at offset 80 (4 bytes, material feature/capability bitflags)
/// - `_pad1`: `[u8; 12]` at offset 84 (12 bytes trailing alignment padding to 16-byte struct alignment)
/// Total size: 96 bytes ([`MATERIAL_PARAMS_BYTES`]), alignment 16 bytes ([`MATERIAL_PARAMS_ALIGNMENT`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialParams {
    /// Base diffuse/emissive color (`[r, g, b, a]`).
    pub color: [f32; 4],
    /// Material opacity factor in `[0.0, 1.0]`.
    pub opacity: f32,
    /// Alpha test threshold below which fragments are discarded.
    pub alpha_test: f32,
    /// Alignment padding ensuring `map_transform` starts at a 16-byte boundary.
    pub _pad0: [u8; 8],
    /// 2D texture UV map transform (3x3 affine matrix represented as three `vec4<f32>` rows).
    pub map_transform: AffineRows,
    /// Feature and capability bitflags for shader branch specialization.
    pub flags: u32,
    /// Trailing alignment padding to satisfy the 16-byte uniform block alignment constraint.
    pub _pad1: [u8; 12],
}

impl MaterialParams {
    /// Byte size of this material parameter block record (96 bytes).
    pub const BYTE_SIZE: usize = MATERIAL_PARAMS_BYTES;

    /// Alignment in bytes (16 bytes).
    pub const ALIGNMENT: usize = MATERIAL_PARAMS_ALIGNMENT;

    /// Constructs a new `MaterialParams` uniform record.
    pub const fn new(
        color: [f32; 4],
        opacity: f32,
        alpha_test: f32,
        map_transform: AffineRows,
        flags: u32,
    ) -> Self {
        Self {
            color,
            opacity,
            alpha_test,
            _pad0: [0u8; 8],
            map_transform,
            flags,
            _pad1: [0u8; 12],
        }
    }

    /// Constructs a default `MeshBasicMaterial` parameter block matching Three.js r186 defaults:
    /// white opaque color (`[1.0, 1.0, 1.0, 1.0]`), opacity 1.0, alphaTest 0.0, identity UV transform, 0 flags.
    pub const fn basic() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            opacity: 1.0,
            alpha_test: 0.0,
            _pad0: [0u8; 8],
            map_transform: AffineRows::identity(),
            flags: 0,
            _pad1: [0u8; 12],
        }
    }

    /// Checks whether a specific material flag bit is enabled.
    #[inline]
    pub const fn has_flag(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }

    /// Enables or disables a specific material flag bit.
    #[inline]
    pub fn set_flag(&mut self, flag: u32, enable: bool) {
        if enable {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }

    /// Serializes the material parameter record into an exact 96-byte array in little-endian order.
    pub fn to_bytes(&self) -> [u8; MATERIAL_PARAMS_BYTES] {
        let mut out = [0u8; MATERIAL_PARAMS_BYTES];
        out[0..4].copy_from_slice(&self.color[0].to_le_bytes());
        out[4..8].copy_from_slice(&self.color[1].to_le_bytes());
        out[8..12].copy_from_slice(&self.color[2].to_le_bytes());
        out[12..16].copy_from_slice(&self.color[3].to_le_bytes());
        out[16..20].copy_from_slice(&self.opacity.to_le_bytes());
        out[20..24].copy_from_slice(&self.alpha_test.to_le_bytes());
        out[24..32].copy_from_slice(&self._pad0);
        out[32..80].copy_from_slice(&self.map_transform.to_bytes());
        out[80..84].copy_from_slice(&self.flags.to_le_bytes());
        out[84..96].copy_from_slice(&self._pad1);
        out
    }

    /// Safely writes the 96-byte uniform wire representation into a mutable byte slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        out[..Self::BYTE_SIZE].copy_from_slice(&self.to_bytes());
        Ok(())
    }

    /// Safely reads the 96-byte uniform record from a byte slice in little-endian order.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: src.len(),
            });
        }
        let mut b = [0u8; MATERIAL_PARAMS_BYTES];
        b.copy_from_slice(&src[..Self::BYTE_SIZE]);
        Ok(Self::from_bytes(&b))
    }

    /// Deserializes a `MaterialParams` record from an exact 96-byte array.
    pub fn from_bytes(bytes: &[u8; MATERIAL_PARAMS_BYTES]) -> Self {
        let read_f32 = |offset: usize| -> f32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            f32::from_le_bytes(b)
        };
        let color = [
            read_f32(0),
            read_f32(4),
            read_f32(8),
            read_f32(12),
        ];
        let opacity = read_f32(16);
        let alpha_test = read_f32(20);
        let mut _pad0 = [0u8; 8];
        _pad0.copy_from_slice(&bytes[24..32]);
        let mut map_bytes = [0u8; AFFINE_ROWS_BYTES];
        map_bytes.copy_from_slice(&bytes[32..80]);
        let map_transform = AffineRows::from_bytes(&map_bytes);
        let mut flags_bytes = [0u8; 4];
        flags_bytes.copy_from_slice(&bytes[80..84]);
        let flags = u32::from_le_bytes(flags_bytes);
        let mut _pad1 = [0u8; 12];
        _pad1.copy_from_slice(&bytes[84..96]);
        Self {
            color,
            opacity,
            alpha_test,
            _pad0,
            map_transform,
            flags,
            _pad1,
        }
    }
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self::basic()
    }
}

define_canonical_layout! {
    record: "MaterialParams",
    shader_const: WGSL_MATERIAL_PARAMS_DECLARATION,
    layout_rows_const: MATERIAL_PARAMS_LAYOUT_ROWS,
    row_count: 7,
    fields: [
        "color", "vec4<f32>", 0, 16, 16;
        "opacity", "f32", 16, 4, 4;
        "alpha_test", "f32", 20, 4, 4;
        "map_transform", "AffineRows", 32, 48, 16;
        "flags", "u32", 80, 4, 4;
    ],
    padding: [
        "_pad0", 24, 8, 4;
        "_pad1", 84, 12, 4;
    ],
    extra_shader: "",
}

define_canonical_vertex_layout! {
    record: "VertexPosUv",
    shader_const: WGSL_VERTEX_POS_UV_DECLARATION,
    layout_rows_const: VERTEX_POS_UV_LAYOUT_ROWS,
    row_count: 2,
    fields: [
        0, "position", "vec3<f32>", 0, 12, 4;
        1, "uv", "vec2<f32>", 12, 8, 4;
    ],
}

define_canonical_vertex_layout! {
    record: "VertexPosNormalUv",
    shader_const: WGSL_VERTEX_POS_NORMAL_UV_DECLARATION,
    layout_rows_const: VERTEX_POS_NORMAL_UV_LAYOUT_ROWS,
    row_count: 3,
    fields: [
        0, "position", "vec3<f32>", 0, 12, 4;
        1, "normal", "vec3<f32>", 12, 12, 4;
        2, "uv", "vec2<f32>", 24, 8, 4;
    ],
}

define_canonical_vertex_layout! {
    record: "VertexPosColor",
    shader_const: WGSL_VERTEX_POS_COLOR_DECLARATION,
    layout_rows_const: VERTEX_POS_COLOR_LAYOUT_ROWS,
    row_count: 2,
    fields: [
        0, "position", "vec3<f32>", 0, 12, 4;
        1, "color", "vec4<f32>", 12, 16, 4;
    ],
}

define_canonical_layout! {
    record: "InstanceRecord",
    shader_const: WGSL_INSTANCE_RECORD_DECLARATION,
    layout_rows_const: INSTANCE_RECORD_LAYOUT_ROWS,
    row_count: 3,
    fields: [
        "transform", "AffineRows", 0, 48, 16;
        "instance_id", "u32", 48, 4, 4;
    ],
    padding: [
        "_padding", 52, 12, 4;
    ],
}

define_canonical_layout! {
    record: "DrawIndirectArgs",
    shader_const: WGSL_DRAW_INDIRECT_ARGS_DECLARATION,
    layout_rows_const: DRAW_INDIRECT_ARGS_LAYOUT_ROWS,
    row_count: 4,
    fields: [
        "vertex_count", "u32", 0, 4, 4;
        "instance_count", "u32", 4, 4, 4;
        "first_vertex", "u32", 8, 4, 4;
        "first_instance", "u32", 12, 4, 4;
    ],
}

define_canonical_layout! {
    record: "DrawIndexedIndirectArgs",
    shader_const: WGSL_DRAW_INDEXED_INDIRECT_ARGS_DECLARATION,
    layout_rows_const: DRAW_INDEXED_INDIRECT_ARGS_LAYOUT_ROWS,
    row_count: 5,
    fields: [
        "index_count", "u32", 0, 4, 4;
        "instance_count", "u32", 4, 4, 4;
        "first_index", "u32", 8, 4, 4;
        "base_vertex", "i32", 12, 4, 4;
        "first_instance", "u32", 16, 4, 4;
    ],
}

define_canonical_layout! {
    record: "ColorUniform",
    shader_const: WGSL_COLOR_UNIFORM_DECLARATION,
    layout_rows_const: COLOR_UNIFORM_LAYOUT_ROWS,
    row_count: 1,
    fields: [
        "rgba", "vec4<f32>", 0, 16, 16;
    ],
}

define_canonical_layout! {
    record: "MeshUniforms",
    shader_const: WGSL_MESH_UNIFORMS_DECLARATION,
    layout_rows_const: MESH_UNIFORMS_LAYOUT_ROWS,
    row_count: 3,
    fields: [
        "model_view", "mat4x4<f32>", 0, 64, 16;
        "projection", "mat4x4<f32>", 64, 64, 16;
        "color", "vec4<f32>", 128, 16, 16;
    ],
}

/// Canonical GPU uniform parameter block record for Toon mesh rendering matching Three.js r186 `MeshToonMaterial`.
///
/// WGSL uniform buffer address-space alignment rules mandate 16-byte alignment for uniform structs,
/// 16-byte alignment for `mat4x4<f32>`, 16-byte alignment for each column of `mat3x3<f32>` (48 bytes total),
/// and 16-byte alignment for `vec4<f32>` / `vec4<u32>`.
///
/// Memory layout (304 bytes total, 16-byte aligned):
/// - `model_world`: `[f32; 16]` at offset 0 (64 bytes, `mat4x4<f32>`)
/// - `projection`: `[f32; 16]` at offset 64 (64 bytes, `mat4x4<f32>`)
/// - `camera_view`: `[f32; 16]` at offset 128 (64 bytes, `mat4x4<f32>`)
/// - `model_normal_matrix`: `[[f32; 4]; 3]` at offset 192 (48 bytes, 3 columns padded to 16B per WGSL std140/uniform alignment)
/// - `color`: `[f32; 4]` at offset 240 (16 bytes, `vec4<f32>`, surface diffuse / tint)
/// - `light_direction`: `[f32; 4]` at offset 256 (16 bytes, `vec4<f32>`, view-space direction xyz, 0.0 w)
/// - `light_color`: `[f32; 4]` at offset 272 (16 bytes, `vec4<f32>`, rgb = color * intensity, 1.0 a)
/// - `params`: `[u32; 4]` at offset 288 (16 bytes, `vec4<u32>`, side code, flags)
/// Total size: 304 bytes ([`TOON_MESH_UNIFORMS_BYTES`]), alignment 16 bytes ([`TOON_MESH_UNIFORMS_ALIGNMENT`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ToonMeshUniforms {
    /// Object world transform matrix in column-major order (64 bytes).
    pub model_world: [f32; 16],
    /// Camera projection matrix in column-major order (64 bytes).
    pub projection: [f32; 16],
    /// Camera view matrix (`matrixWorldInverse`) in column-major order (64 bytes).
    pub camera_view: [f32; 16],
    /// Normal transformation matrix as 3 columns of `vec3<f32>`, each padded to 16 bytes (48 bytes).
    pub model_normal_matrix: [[f32; 4]; 3],
    /// Surface diffuse / tint color (`[r, g, b, a]`, 16 bytes).
    pub color: [f32; 4],
    /// Directional light view-space direction (`[x, y, z, 0.0]`, 16 bytes).
    pub light_direction: [f32; 4],
    /// Directional light color scaled by intensity (`[r, g, b, 1.0]`, 16 bytes).
    pub light_color: [f32; 4],
    /// Miscellaneous packed parameters (`[side, flags, _reserved0, _reserved1]`, 16 bytes).
    pub params: [u32; 4],
}

impl ToonMeshUniforms {
    /// Byte size of this Toon uniform record (304 bytes).
    pub const BYTE_SIZE: usize = TOON_MESH_UNIFORMS_BYTES;

    /// Alignment in bytes (16 bytes).
    pub const ALIGNMENT: usize = TOON_MESH_UNIFORMS_ALIGNMENT;

    /// Dynamic uniform buffer offset stride (512 bytes).
    pub const DYNAMIC_OFFSET_STRIDE: usize = TOON_MESH_DYNAMIC_OFFSET_STRIDE;

    /// Constructs a new `ToonMeshUniforms` uniform record from explicit fields.
    pub const fn new(
        model_world: [f32; 16],
        projection: [f32; 16],
        camera_view: [f32; 16],
        model_normal_matrix: [[f32; 4]; 3],
        color: [f32; 4],
        light_direction: [f32; 4],
        light_color: [f32; 4],
        params: [u32; 4],
    ) -> Self {
        Self {
            model_world,
            projection,
            camera_view,
            model_normal_matrix,
            color,
            light_direction,
            light_color,
            params,
        }
    }

    /// Converts a 9-element column-major 3x3 normal matrix into explicit 16-byte padded columns.
    pub const fn pad_normal_matrix(m: &[f32; 9]) -> [[f32; 4]; 3] {
        [
            [m[0], m[1], m[2], 0.0],
            [m[3], m[4], m[5], 0.0],
            [m[6], m[7], m[8], 0.0],
        ]
    }

    /// Extracts the unpadded 9-element column-major 3x3 normal matrix from explicit padded columns.
    pub const fn unpad_normal_matrix(&self) -> [f32; 9] {
        [
            self.model_normal_matrix[0][0],
            self.model_normal_matrix[0][1],
            self.model_normal_matrix[0][2],
            self.model_normal_matrix[1][0],
            self.model_normal_matrix[1][1],
            self.model_normal_matrix[1][2],
            self.model_normal_matrix[2][0],
            self.model_normal_matrix[2][1],
            self.model_normal_matrix[2][2],
        ]
    }

    /// Serializes the Toon mesh uniform record into an exact 304-byte array in little-endian order.
    pub fn to_bytes(&self) -> [u8; TOON_MESH_UNIFORMS_BYTES] {
        let mut out = [0u8; TOON_MESH_UNIFORMS_BYTES];
        let write_f32 = |slice: &mut [u8], offset: usize, val: f32| {
            slice[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        };
        let write_u32 = |slice: &mut [u8], offset: usize, val: u32| {
            slice[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        };

        for i in 0..16 {
            write_f32(&mut out, i * 4, self.model_world[i]);
        }
        for i in 0..16 {
            write_f32(&mut out, 64 + i * 4, self.projection[i]);
        }
        for i in 0..16 {
            write_f32(&mut out, 128 + i * 4, self.camera_view[i]);
        }
        for col in 0..3 {
            for row in 0..4 {
                write_f32(&mut out, 192 + col * 16 + row * 4, self.model_normal_matrix[col][row]);
            }
        }
        for i in 0..4 {
            write_f32(&mut out, 240 + i * 4, self.color[i]);
        }
        for i in 0..4 {
            write_f32(&mut out, 256 + i * 4, self.light_direction[i]);
        }
        for i in 0..4 {
            write_f32(&mut out, 272 + i * 4, self.light_color[i]);
        }
        for i in 0..4 {
            write_u32(&mut out, 288 + i * 4, self.params[i]);
        }
        out
    }

    /// Safely writes the 304-byte uniform wire representation into a mutable byte slice.
    pub fn write_to_slice(&self, out: &mut [u8]) -> Result<(), LayoutError> {
        if out.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: out.len(),
            });
        }
        out[..Self::BYTE_SIZE].copy_from_slice(&self.to_bytes());
        Ok(())
    }

    /// Safely reads the 304-byte uniform record from a byte slice in little-endian order.
    pub fn read_from_slice(src: &[u8]) -> Result<Self, LayoutError> {
        if src.len() < Self::BYTE_SIZE {
            return Err(LayoutError::BufferTooSmall {
                required: Self::BYTE_SIZE,
                provided: src.len(),
            });
        }
        let mut b = [0u8; TOON_MESH_UNIFORMS_BYTES];
        b.copy_from_slice(&src[..Self::BYTE_SIZE]);
        Ok(Self::from_bytes(&b))
    }

    /// Deserializes a `ToonMeshUniforms` record from an exact 304-byte array.
    pub fn from_bytes(bytes: &[u8; TOON_MESH_UNIFORMS_BYTES]) -> Self {
        let read_f32 = |offset: usize| -> f32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            f32::from_le_bytes(b)
        };
        let read_u32 = |offset: usize| -> u32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[offset..offset + 4]);
            u32::from_le_bytes(b)
        };

        let mut model_world = [0.0f32; 16];
        for i in 0..16 {
            model_world[i] = read_f32(i * 4);
        }

        let mut projection = [0.0f32; 16];
        for i in 0..16 {
            projection[i] = read_f32(64 + i * 4);
        }

        let mut camera_view = [0.0f32; 16];
        for i in 0..16 {
            camera_view[i] = read_f32(128 + i * 4);
        }

        let mut model_normal_matrix = [[0.0f32; 4]; 3];
        for col in 0..3 {
            for row in 0..4 {
                model_normal_matrix[col][row] = read_f32(192 + col * 16 + row * 4);
            }
        }

        let color = [
            read_f32(240),
            read_f32(244),
            read_f32(248),
            read_f32(252),
        ];

        let light_direction = [
            read_f32(256),
            read_f32(260),
            read_f32(264),
            read_f32(268),
        ];

        let light_color = [
            read_f32(272),
            read_f32(276),
            read_f32(280),
            read_f32(284),
        ];

        let params = [
            read_u32(288),
            read_u32(292),
            read_u32(296),
            read_u32(300),
        ];

        Self {
            model_world,
            projection,
            camera_view,
            model_normal_matrix,
            color,
            light_direction,
            light_color,
            params,
        }
    }
}

impl Default for ToonMeshUniforms {
    fn default() -> Self {
        Self {
            model_world: ProjectiveMat4::identity().elements,
            projection: ProjectiveMat4::identity().elements,
            camera_view: ProjectiveMat4::identity().elements,
            model_normal_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            color: [1.0, 1.0, 1.0, 1.0],
            light_direction: [0.0, 0.0, 1.0, 0.0],
            light_color: [1.0, 1.0, 1.0, 1.0],
            params: [0, 0, 0, 0],
        }
    }
}

define_canonical_layout! {
    record: "ToonMeshUniforms",
    shader_const: WGSL_TOON_MESH_UNIFORMS_DECLARATION,
    layout_rows_const: TOON_MESH_UNIFORMS_LAYOUT_ROWS,
    row_count: 8,
    fields: [
        "model_world", "mat4x4<f32>", 0, 64, 16;
        "projection", "mat4x4<f32>", 64, 64, 16;
        "camera_view", "mat4x4<f32>", 128, 64, 16;
        "model_normal_matrix", "mat3x3<f32>", 192, 48, 16;
        "color", "vec4<f32>", 240, 16, 16;
        "light_direction", "vec4<f32>", 256, 16, 16;
        "light_color", "vec4<f32>", 272, 16, 16;
        "params", "vec4<u32>", 288, 16, 16;
    ],
}

/// Returns standard generated WGSL type declarations and helpers for use in shaders.
pub fn generate_wgsl_declarations() -> String {
    let mut s = String::new();
    s.push_str("// === FrankenThreeD Generated Layouts ===\n");
    s.push_str(WGSL_AFFINE_ROWS_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_PROJECTIVE_MAT4_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_VERTEX_POS_UV_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_VERTEX_POS_NORMAL_UV_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_VERTEX_POS_COLOR_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_INSTANCE_RECORD_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_DRAW_INDIRECT_ARGS_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_DRAW_INDEXED_INDIRECT_ARGS_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_COLOR_UNIFORM_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_MATERIAL_PARAMS_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_MESH_UNIFORMS_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_TOON_MESH_UNIFORMS_DECLARATION.trim());
    s.push('\n');
    s
}

/// Static catalog of GPU wire layouts covering every record type in `f3d-core`.
pub const LAYOUT_TABLE: [LayoutRow; 42] = [
    // 1. AffineRows (48 bytes, 16-byte aligned)
    AFFINE_ROWS_LAYOUT_ROWS[0],
    AFFINE_ROWS_LAYOUT_ROWS[1],
    AFFINE_ROWS_LAYOUT_ROWS[2],

    // 2. ProjectiveMat4 (64 bytes, 16-byte aligned)
    PROJECTIVE_MAT4_LAYOUT_ROWS[0],

    // 3. VertexPosUv (20 bytes, 4-byte aligned)
    VERTEX_POS_UV_LAYOUT_ROWS[0],
    VERTEX_POS_UV_LAYOUT_ROWS[1],

    // 4. VertexPosNormalUv (32 bytes, 4-byte aligned)
    VERTEX_POS_NORMAL_UV_LAYOUT_ROWS[0],
    VERTEX_POS_NORMAL_UV_LAYOUT_ROWS[1],
    VERTEX_POS_NORMAL_UV_LAYOUT_ROWS[2],

    // 5. VertexPosColor (28 bytes, 4-byte aligned)
    VERTEX_POS_COLOR_LAYOUT_ROWS[0],
    VERTEX_POS_COLOR_LAYOUT_ROWS[1],

    // 6. InstanceRecord (64 bytes, 16-byte aligned)
    // Note: `transform` is stored as AffineRows (three vec4<f32> rows), not mat3x4 column-major.
    // Bytes 52..64 constitute 12 bytes of trailing alignment padding to reach the 64-byte struct allocation.
    INSTANCE_RECORD_LAYOUT_ROWS[0],
    INSTANCE_RECORD_LAYOUT_ROWS[1],
    INSTANCE_RECORD_LAYOUT_ROWS[2],

    // 7. DrawIndirectArgs (16 bytes, 4-byte aligned)
    DRAW_INDIRECT_ARGS_LAYOUT_ROWS[0],
    DRAW_INDIRECT_ARGS_LAYOUT_ROWS[1],
    DRAW_INDIRECT_ARGS_LAYOUT_ROWS[2],
    DRAW_INDIRECT_ARGS_LAYOUT_ROWS[3],

    // 8. DrawIndexedIndirectArgs (20 bytes, 4-byte aligned)
    DRAW_INDEXED_INDIRECT_ARGS_LAYOUT_ROWS[0],
    DRAW_INDEXED_INDIRECT_ARGS_LAYOUT_ROWS[1],
    DRAW_INDEXED_INDIRECT_ARGS_LAYOUT_ROWS[2],
    DRAW_INDEXED_INDIRECT_ARGS_LAYOUT_ROWS[3],
    DRAW_INDEXED_INDIRECT_ARGS_LAYOUT_ROWS[4],

    // 9. Color Uniform (16 bytes, 16-byte aligned)
    COLOR_UNIFORM_LAYOUT_ROWS[0],

    // 10. MaterialParams Uniform (96 bytes, 16-byte aligned)
    MATERIAL_PARAMS_LAYOUT_ROWS[0],
    MATERIAL_PARAMS_LAYOUT_ROWS[1],
    MATERIAL_PARAMS_LAYOUT_ROWS[2],
    MATERIAL_PARAMS_LAYOUT_ROWS[3],
    MATERIAL_PARAMS_LAYOUT_ROWS[4],
    MATERIAL_PARAMS_LAYOUT_ROWS[5],
    MATERIAL_PARAMS_LAYOUT_ROWS[6],

    // 11. MeshUniforms Uniform (144 bytes, 16-byte aligned)
    MESH_UNIFORMS_LAYOUT_ROWS[0],
    MESH_UNIFORMS_LAYOUT_ROWS[1],
    MESH_UNIFORMS_LAYOUT_ROWS[2],

    // 12. ToonMeshUniforms Uniform (304 bytes, 16-byte aligned)
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[0],
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[1],
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[2],
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[3],
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[4],
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[5],
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[6],
    TOON_MESH_UNIFORMS_LAYOUT_ROWS[7],
];

/// Returns a fixed slice of [`LayoutRow`] descriptors covering every GPU wire record in the crate.
#[inline]
pub fn layout_table() -> &'static [LayoutRow] {
    &LAYOUT_TABLE
}

/// Formatted text table view of the layout catalog for terminal / CLI diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutTable(pub &'static [LayoutRow]);

impl LayoutTable {
    /// Returns the canonical layout table.
    pub const fn new() -> Self {
        Self(&LAYOUT_TABLE)
    }

    /// Returns the inner slice of layout rows.
    pub const fn as_slice(&self) -> &'static [LayoutRow] {
        self.0
    }
}

impl Default for LayoutTable {
    fn default() -> Self {
        Self::new()
    }
}

impl core::ops::Deref for LayoutTable {
    type Target = [LayoutRow];
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl fmt::Display for LayoutTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<24} {:<16} {:>6} {:>6} {:>6}  {:<16}",
            "RECORD", "FIELD", "OFFSET", "SIZE", "ALIGN", "WGSL_TYPE"
        )?;
        writeln!(
            f,
            "{:<24} {:<16} {:>6} {:>6} {:>6}  {:<16}",
            "------------------------", "----------------", "------", "------", "------", "----------------"
        )?;
        for row in self.0 {
            writeln!(f, "{row}")?;
        }
        Ok(())
    }
}

/// Helper function to dump all layouts formatted as an aligned text table (`--dump-layouts`).
pub fn dump_layouts() -> LayoutTable {
    LayoutTable(layout_table())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_rows_identity_and_byte_roundtrip() {
        let id = AffineRows::identity();
        let bytes = id.to_bytes();
        assert_eq!(bytes.len(), 48);
        let restored = AffineRows::from_bytes(&bytes);
        assert_eq!(id, restored);

        let col_major = id.to_column_major();
        assert_eq!(col_major[0], 1.0);
        assert_eq!(col_major[5], 1.0);
        assert_eq!(col_major[10], 1.0);
        assert_eq!(col_major[15], 1.0);
    }

    #[test]
    fn affine_rows_translation_point_transform() {
        let e = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            10.0, 20.0, 30.0, 1.0,
        ];
        let affine = AffineRows::from_column_major(&e).unwrap();
        assert_eq!(affine.transform_point([1.0, 2.0, 3.0]), [11.0, 22.0, 33.0]);
        assert_eq!(affine.transform_vector([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn non_affine_rejection() {
        let mut e = [0.0; 16];
        e[15] = 1.0;
        e[11] = 0.5; // perspective component
        assert_eq!(AffineRows::from_column_major(&e), Err(LayoutError::NonAffineMatrix));
    }

    #[test]
    fn exact_structural_affine_regression_and_f64_eligibility() {
        // e3 = 5e-7, x = 1e7 => homogeneous w = e3 * x + e15 = 5.0 + 1.0 = 6.0 != 1.0.
        let mut e = [
            1.0f32, 0.0, 0.0, 5e-7,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        assert!(!is_matrix4_affine(&e));
        assert_eq!(AffineRows::from_column_major(&e), Err(LayoutError::NonAffineMatrix));

        let proj = ProjectiveMat4::from_elements(e);
        let h = proj.transform_homogeneous([1e7, 0.0, 0.0, 1.0]);
        assert_eq!(h[3], 6.0);

        // Subnormal perspective term
        e[3] = 1e-40;
        assert!(!is_matrix4_affine(&e));
        assert_eq!(AffineRows::from_column_major(&e), Err(LayoutError::NonAffineMatrix));

        // Near 1 e15
        e[3] = 0.0;
        e[15] = 1.0000001;
        assert!(!is_matrix4_affine(&e));
        assert_eq!(AffineRows::from_column_major(&e), Err(LayoutError::NonAffineMatrix));

        // f64 source matrix with small perspective term before narrowing
        let mut e_f64 = [
            1.0f64, 0.0, 0.0, 5e-7,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        assert!(!is_matrix4_f64_affine(&e_f64));
        assert_eq!(AffineRows::from_column_major_f64(&e_f64), Err(LayoutError::NonAffineMatrix));

        e_f64[3] = 0.0;
        assert!(is_matrix4_f64_affine(&e_f64));
        let affine = AffineRows::from_column_major_f64(&e_f64).expect("valid f64 affine");
        assert_eq!(affine, AffineRows::identity());
    }

    #[test]
    fn wgsl_mat4x3_size_distinction() {
        assert_eq!(GpuMatrixLayout::WgslMat4x3Padded64.byte_size(), 64);
        assert_eq!(GpuMatrixLayout::AffineRows48.byte_size(), 48);
        assert!(validate_affine_target(GpuMatrixLayout::WgslMat4x3Padded64).is_err());
        assert!(validate_affine_target(GpuMatrixLayout::AffineRows48).is_ok());
    }

    #[test]
    fn alignment_validation() {
        assert!(validate_dynamic_uniform_offset(256, 256).is_ok());
        assert!(validate_dynamic_uniform_offset(48, 256).is_err());
        assert!(validate_write_buffer_alignment(0, 48).is_ok());
        assert!(validate_write_buffer_alignment(2, 48).is_err());
        assert!(validate_copy_bytes_per_row(256).is_ok());
        assert!(validate_copy_bytes_per_row(100).is_err());
    }

    #[test]
    fn storage_stride_scalar_vec2_and_composite() {
        // Scalar f32 (size 4, align 4)
        assert!(validate_storage_array_stride(4, 4, 4).is_ok());
        assert!(validate_storage_array_stride(8, 4, 4).is_ok());
        assert_eq!(
            validate_storage_array_stride(6, 4, 4),
            Err(LayoutError::UnalignedOffset { offset: 6, required_alignment: 4 })
        );

        // vec2<f32> (size 8, align 8)
        assert!(validate_storage_array_stride(8, 8, 8).is_ok());
        assert!(validate_storage_array_stride(16, 8, 8).is_ok());
        assert_eq!(
            validate_storage_array_stride(12, 8, 8),
            Err(LayoutError::UnalignedOffset { offset: 12, required_alignment: 8 })
        );

        // Composite record AffineRows (size 48, align 16)
        assert!(validate_composite_storage_array_stride(48, 48).is_ok());
        assert!(validate_composite_storage_array_stride(64, 48).is_ok());
        assert_eq!(
            validate_composite_storage_array_stride(50, 48),
            Err(LayoutError::UnalignedOffset { offset: 50, required_alignment: 16 })
        );
    }

    #[test]
    fn aligned_bytes_per_row_calculations() {
        // Width 32: 32 * 4 = 128 bytes -> rounded up to 256 bytes
        assert_eq!(aligned_bytes_per_row(32), Ok(256));

        // Width 64: 64 * 4 = 256 bytes -> exact multiple
        assert_eq!(aligned_bytes_per_row(64), Ok(256));

        // Width 1: 1 * 4 = 4 bytes -> 256 bytes
        assert_eq!(aligned_bytes_per_row(1), Ok(256));

        // Width 0: 0 bytes -> 0 bytes
        assert_eq!(aligned_bytes_per_row(0), Ok(0));

        // Overflow: width * 4 overflows u32
        assert_eq!(aligned_bytes_per_row(u32::MAX), Err(LayoutError::CalculationOverflow));
        assert_eq!(aligned_bytes_per_row(u32::MAX / 4 + 1), Err(LayoutError::CalculationOverflow));

        // Alignment overflow: unpadded + padding overflows u32
        assert_eq!(aligned_bytes_per_row(1_073_741_823), Err(LayoutError::CalculationOverflow));

        // Custom bytes_per_pixel helper
        assert_eq!(aligned_copy_bytes_per_row(32, 4), Ok(256));
        assert_eq!(aligned_copy_bytes_per_row(32, 8), Ok(256)); // 32 * 8 = 256
        assert_eq!(aligned_copy_bytes_per_row(33, 8), Ok(512)); // 33 * 8 = 264 -> 512
    }

    #[test]
    fn vertex_pos_uv_and_color_uniform_constants_tests() {
        assert_eq!(COLOR_UNIFORM_BYTES, 16);
        assert_eq!(COLOR_UNIFORM_ALIGNMENT, 16);
        assert_eq!(VERTEX_POS_UV_BYTES, 20);
        assert_eq!(VERTEX_POS_UV_STRIDE, 20);
        assert_eq!(VERTEX_POS_UV_ALIGNMENT, 4);

        let v = VertexPosUv::new([0.0, 0.5, 0.0], [0.5, 1.0]);
        assert_eq!(VertexPosUv::BYTE_SIZE, 20);
        assert_eq!(VertexPosUv::STRIDE, 20);
        assert_eq!(VertexPosUv::ALIGNMENT, 4);

        let bytes = v.to_bytes();
        assert_eq!(bytes.len(), 20);
        let restored = VertexPosUv::from_bytes(&bytes);
        assert_eq!(v, restored);

        let mut slice = [0u8; 20];
        v.write_to_slice(&mut slice).expect("write slice ok");
        let from_slice = VertexPosUv::read_from_slice(&slice).expect("read slice ok");
        assert_eq!(v, from_slice);

        let mut small = [0u8; 19];
        assert_eq!(
            v.write_to_slice(&mut small),
            Err(LayoutError::BufferTooSmall { required: 20, provided: 19 })
        );
        assert_eq!(
            VertexPosUv::read_from_slice(&small),
            Err(LayoutError::BufferTooSmall { required: 20, provided: 19 })
        );
    }

    #[test]
    fn vertex_layout_records_and_offset_tests() {
        assert_eq!(core::mem::size_of::<VertexPosNormalUv>(), 32);
        assert_eq!(core::mem::align_of::<VertexPosNormalUv>(), 4);
        assert_eq!(core::mem::offset_of!(VertexPosNormalUv, position), 0);
        assert_eq!(core::mem::offset_of!(VertexPosNormalUv, normal), 12);
        assert_eq!(core::mem::offset_of!(VertexPosNormalUv, uv), 24);
        assert_eq!(VERTEX_POS_NORMAL_UV_BYTES, 32);
        assert_eq!(VERTEX_POS_NORMAL_UV_STRIDE, 32);
        assert_eq!(VERTEX_POS_NORMAL_UV_ALIGNMENT, 4);
        assert_eq!(VertexPosNormalUv::BYTE_SIZE, 32);
        assert_eq!(VertexPosNormalUv::STRIDE, 32);
        assert_eq!(VertexPosNormalUv::ALIGNMENT, 4);

        let v_pnu = VertexPosNormalUv::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.25, 0.75]);
        let bytes_pnu = v_pnu.to_bytes();
        assert_eq!(bytes_pnu.len(), 32);
        let restored_pnu = VertexPosNormalUv::from_bytes(&bytes_pnu);
        assert_eq!(v_pnu, restored_pnu);

        let mut slice_pnu = [0u8; 32];
        v_pnu.write_to_slice(&mut slice_pnu).expect("write slice ok");
        let from_slice_pnu = VertexPosNormalUv::read_from_slice(&slice_pnu).expect("read slice ok");
        assert_eq!(v_pnu, from_slice_pnu);

        let mut small_pnu = [0u8; 31];
        assert_eq!(
            v_pnu.write_to_slice(&mut small_pnu),
            Err(LayoutError::BufferTooSmall { required: 32, provided: 31 })
        );
        assert_eq!(
            VertexPosNormalUv::read_from_slice(&small_pnu),
            Err(LayoutError::BufferTooSmall { required: 32, provided: 31 })
        );

        assert_eq!(core::mem::size_of::<VertexPosColor>(), 28);
        assert_eq!(core::mem::align_of::<VertexPosColor>(), 4);
        assert_eq!(core::mem::offset_of!(VertexPosColor, position), 0);
        assert_eq!(core::mem::offset_of!(VertexPosColor, color), 12);
        assert_eq!(VERTEX_POS_COLOR_BYTES, 28);
        assert_eq!(VERTEX_POS_COLOR_STRIDE, 28);
        assert_eq!(VERTEX_POS_COLOR_ALIGNMENT, 4);
        assert_eq!(VertexPosColor::BYTE_SIZE, 28);
        assert_eq!(VertexPosColor::STRIDE, 28);
        assert_eq!(VertexPosColor::ALIGNMENT, 4);

        let v_pc = VertexPosColor::new([1.0, 2.0, 3.0], [1.0, 0.5, 0.25, 1.0]);
        let bytes_pc = v_pc.to_bytes();
        assert_eq!(bytes_pc.len(), 28);
        let restored_pc = VertexPosColor::from_bytes(&bytes_pc);
        assert_eq!(v_pc, restored_pc);

        let mut slice_pc = [0u8; 28];
        v_pc.write_to_slice(&mut slice_pc).expect("write slice ok");
        let from_slice_pc = VertexPosColor::read_from_slice(&slice_pc).expect("read slice ok");
        assert_eq!(v_pc, from_slice_pc);

        let mut small_pc = [0u8; 27];
        assert_eq!(
            v_pc.write_to_slice(&mut small_pc),
            Err(LayoutError::BufferTooSmall { required: 28, provided: 27 })
        );
        assert_eq!(
            VertexPosColor::read_from_slice(&small_pc),
            Err(LayoutError::BufferTooSmall { required: 28, provided: 27 })
        );
    }

    #[test]
    fn material_params_layout_and_byte_roundtrip_tests() {
        assert_eq!(MATERIAL_PARAMS_BYTES, 96);
        assert_eq!(MATERIAL_PARAMS_ALIGNMENT, 16);
        assert_eq!(core::mem::size_of::<MaterialParams>(), 96);
        assert_eq!(core::mem::offset_of!(MaterialParams, color), 0);
        assert_eq!(core::mem::offset_of!(MaterialParams, opacity), 16);
        assert_eq!(core::mem::offset_of!(MaterialParams, alpha_test), 20);
        assert_eq!(core::mem::offset_of!(MaterialParams, _pad0), 24);
        assert_eq!(core::mem::offset_of!(MaterialParams, map_transform), 32);
        assert_eq!(core::mem::offset_of!(MaterialParams, flags), 80);
        assert_eq!(core::mem::offset_of!(MaterialParams, _pad1), 84);

        let mut mat = MaterialParams::new(
            [0.25, 0.5, 0.75, 1.0],
            0.85,
            0.1,
            AffineRows::identity(),
            MATERIAL_FLAG_MAP | MATERIAL_FLAG_ALPHA_TEST,
        );
        assert!(mat.has_flag(MATERIAL_FLAG_MAP));
        assert!(mat.has_flag(MATERIAL_FLAG_ALPHA_TEST));
        assert!(!mat.has_flag(MATERIAL_FLAG_WIREFRAME));
        mat.set_flag(MATERIAL_FLAG_WIREFRAME, true);
        assert!(mat.has_flag(MATERIAL_FLAG_WIREFRAME));
        mat.set_flag(MATERIAL_FLAG_MAP, false);
        assert!(!mat.has_flag(MATERIAL_FLAG_MAP));

        let bytes = mat.to_bytes();
        assert_eq!(bytes.len(), 96);
        let restored = MaterialParams::from_bytes(&bytes);
        assert_eq!(restored, mat);

        let mut slice_buf = [0u8; 96];
        mat.write_to_slice(&mut slice_buf).expect("write ok");
        let read_back = MaterialParams::read_from_slice(&slice_buf).expect("read ok");
        assert_eq!(read_back, mat);

        // Short buffer error
        let mut short_buf = [0u8; 80];
        assert_eq!(
            mat.write_to_slice(&mut short_buf),
            Err(LayoutError::BufferTooSmall { required: 96, provided: 80 })
        );
        assert_eq!(
            MaterialParams::read_from_slice(&short_buf),
            Err(LayoutError::BufferTooSmall { required: 96, provided: 80 })
        );
    }

    #[test]
    fn layout_table_cross_check_and_display() {
        let table = layout_table();
        assert_eq!(table.len(), 42);

        for row in table {
            let (expected_offset, expected_size) = match (row.record, row.field) {
                ("AffineRows", "r0") => (core::mem::offset_of!(AffineRows, r0), core::mem::size_of::<[f32; 4]>()),
                ("AffineRows", "r1") => (core::mem::offset_of!(AffineRows, r1), core::mem::size_of::<[f32; 4]>()),
                ("AffineRows", "r2") => (core::mem::offset_of!(AffineRows, r2), core::mem::size_of::<[f32; 4]>()),
                ("ProjectiveMat4", "elements") => (core::mem::offset_of!(ProjectiveMat4, elements), core::mem::size_of::<[f32; 16]>()),
                ("VertexPosUv", "position") => (core::mem::offset_of!(VertexPosUv, position), core::mem::size_of::<[f32; 3]>()),
                ("VertexPosUv", "uv") => (core::mem::offset_of!(VertexPosUv, uv), core::mem::size_of::<[f32; 2]>()),
                ("VertexPosNormalUv", "position") => (core::mem::offset_of!(VertexPosNormalUv, position), core::mem::size_of::<[f32; 3]>()),
                ("VertexPosNormalUv", "normal") => (core::mem::offset_of!(VertexPosNormalUv, normal), core::mem::size_of::<[f32; 3]>()),
                ("VertexPosNormalUv", "uv") => (core::mem::offset_of!(VertexPosNormalUv, uv), core::mem::size_of::<[f32; 2]>()),
                ("VertexPosColor", "position") => (core::mem::offset_of!(VertexPosColor, position), core::mem::size_of::<[f32; 3]>()),
                ("VertexPosColor", "color") => (core::mem::offset_of!(VertexPosColor, color), core::mem::size_of::<[f32; 4]>()),
                ("InstanceRecord", "transform") => (core::mem::offset_of!(InstanceRecord, transform), core::mem::size_of::<AffineRows>()),
                ("InstanceRecord", "instance_id") => (core::mem::offset_of!(InstanceRecord, instance_id), core::mem::size_of::<u32>()),
                ("InstanceRecord", "_padding") => (52, 12),
                ("DrawIndirectArgs", "vertex_count") => (core::mem::offset_of!(DrawIndirectArgs, vertex_count), core::mem::size_of::<u32>()),
                ("DrawIndirectArgs", "instance_count") => (core::mem::offset_of!(DrawIndirectArgs, instance_count), core::mem::size_of::<u32>()),
                ("DrawIndirectArgs", "first_vertex") => (core::mem::offset_of!(DrawIndirectArgs, first_vertex), core::mem::size_of::<u32>()),
                ("DrawIndirectArgs", "first_instance") => (core::mem::offset_of!(DrawIndirectArgs, first_instance), core::mem::size_of::<u32>()),
                ("DrawIndexedIndirectArgs", "index_count") => (core::mem::offset_of!(DrawIndexedIndirectArgs, index_count), core::mem::size_of::<u32>()),
                ("DrawIndexedIndirectArgs", "instance_count") => (core::mem::offset_of!(DrawIndexedIndirectArgs, instance_count), core::mem::size_of::<u32>()),
                ("DrawIndexedIndirectArgs", "first_index") => (core::mem::offset_of!(DrawIndexedIndirectArgs, first_index), core::mem::size_of::<u32>()),
                ("DrawIndexedIndirectArgs", "base_vertex") => (core::mem::offset_of!(DrawIndexedIndirectArgs, base_vertex), core::mem::size_of::<i32>()),
                ("DrawIndexedIndirectArgs", "first_instance") => (core::mem::offset_of!(DrawIndexedIndirectArgs, first_instance), core::mem::size_of::<u32>()),
                ("ColorUniform", "rgba") => (0, COLOR_UNIFORM_BYTES),
                ("MaterialParams", "color") => (core::mem::offset_of!(MaterialParams, color), core::mem::size_of::<[f32; 4]>()),
                ("MaterialParams", "opacity") => (core::mem::offset_of!(MaterialParams, opacity), core::mem::size_of::<f32>()),
                ("MaterialParams", "alpha_test") => (core::mem::offset_of!(MaterialParams, alpha_test), core::mem::size_of::<f32>()),
                ("MaterialParams", "_pad0") => (core::mem::offset_of!(MaterialParams, _pad0), core::mem::size_of::<[u8; 8]>()),
                ("MaterialParams", "map_transform") => (core::mem::offset_of!(MaterialParams, map_transform), core::mem::size_of::<AffineRows>()),
                ("MaterialParams", "flags") => (core::mem::offset_of!(MaterialParams, flags), core::mem::size_of::<u32>()),
                ("MaterialParams", "_pad1") => (core::mem::offset_of!(MaterialParams, _pad1), core::mem::size_of::<[u8; 12]>()),
                ("MeshUniforms", "model_view") => (0, 64),
                ("MeshUniforms", "projection") => (64, 64),
                ("MeshUniforms", "color") => (128, 16),
                ("ToonMeshUniforms", "model_world") => (core::mem::offset_of!(ToonMeshUniforms, model_world), core::mem::size_of::<[f32; 16]>()),
                ("ToonMeshUniforms", "projection") => (core::mem::offset_of!(ToonMeshUniforms, projection), core::mem::size_of::<[f32; 16]>()),
                ("ToonMeshUniforms", "camera_view") => (core::mem::offset_of!(ToonMeshUniforms, camera_view), core::mem::size_of::<[f32; 16]>()),
                ("ToonMeshUniforms", "model_normal_matrix") => (core::mem::offset_of!(ToonMeshUniforms, model_normal_matrix), core::mem::size_of::<[[f32; 4]; 3]>()),
                ("ToonMeshUniforms", "color") => (core::mem::offset_of!(ToonMeshUniforms, color), core::mem::size_of::<[f32; 4]>()),
                ("ToonMeshUniforms", "light_direction") => (core::mem::offset_of!(ToonMeshUniforms, light_direction), core::mem::size_of::<[f32; 4]>()),
                ("ToonMeshUniforms", "light_color") => (core::mem::offset_of!(ToonMeshUniforms, light_color), core::mem::size_of::<[f32; 4]>()),
                ("ToonMeshUniforms", "params") => (core::mem::offset_of!(ToonMeshUniforms, params), core::mem::size_of::<[u32; 4]>()),
                (r, f) => panic!("Unknown record/field: {r}.{f}"),
            };

            assert_eq!(
                row.offset, expected_offset,
                "LayoutRow mismatch for field {}.{}: expected offset {}, actual offset {}",
                row.record, row.field, expected_offset, row.offset
            );
            assert_eq!(
                row.size, expected_size,
                "LayoutRow mismatch for field {}.{}: expected size {}, actual size {}",
                row.record, row.field, expected_size, row.size
            );
        }

        // Tiling invariant: for each record type, the catalog rows must tile the full size_of with no gaps or overlaps.
        let record_sizes: &[(&str, usize)] = &[
            ("AffineRows", core::mem::size_of::<AffineRows>()),
            ("ProjectiveMat4", core::mem::size_of::<ProjectiveMat4>()),
            ("VertexPosUv", core::mem::size_of::<VertexPosUv>()),
            ("VertexPosNormalUv", core::mem::size_of::<VertexPosNormalUv>()),
            ("VertexPosColor", core::mem::size_of::<VertexPosColor>()),
            ("InstanceRecord", InstanceRecord::BYTE_SIZE),
            ("DrawIndirectArgs", core::mem::size_of::<DrawIndirectArgs>()),
            ("DrawIndexedIndirectArgs", core::mem::size_of::<DrawIndexedIndirectArgs>()),
            ("ColorUniform", COLOR_UNIFORM_BYTES),
            ("MaterialParams", core::mem::size_of::<MaterialParams>()),
            ("MeshUniforms", MESH_UNIFORMS_BYTES),
            ("ToonMeshUniforms", TOON_MESH_UNIFORMS_BYTES),
        ];

        for &(record_name, expected_total_size) in record_sizes {
            let mut expected_next_offset = 0;
            let mut found_any = false;
            for row in table {
                if row.record == record_name {
                    found_any = true;
                    assert_eq!(
                        row.offset, expected_next_offset,
                        "Gap or overlap detected in record {}: field {} has offset {}, expected {}",
                        record_name, row.field, row.offset, expected_next_offset
                    );
                    expected_next_offset = row.offset + row.size;
                }
            }
            assert!(found_any, "No rows found for record {}", record_name);
            assert_eq!(
                expected_next_offset, expected_total_size,
                "Record {} rows do not tile the full size_of: tiled {} bytes, size_of is {}",
                record_name, expected_next_offset, expected_total_size
            );
        }

        let rendered = format!("{}", dump_layouts());
        assert!(rendered.contains("RECORD"));
        assert!(rendered.contains("AffineRows"));
        assert!(rendered.contains("ColorUniform"));
        assert!(rendered.contains("MaterialParams"));
        assert!(rendered.contains("MeshUniforms"));
        assert!(rendered.contains("ToonMeshUniforms"));
    }

    #[test]
    fn generate_wgsl_declarations_contains_all_records() {
        let decls = generate_wgsl_declarations();
        assert!(decls.contains("struct AffineRows {"));
        assert!(decls.contains("struct ProjectiveMat4 {"));
        assert!(decls.contains("struct VertexPosUv {"));
        assert!(decls.contains("struct VertexPosNormalUv {"));
        assert!(decls.contains("struct VertexPosColor {"));
        assert!(decls.contains("struct InstanceRecord {"));
        assert!(decls.contains("struct DrawIndirectArgs {"));
        assert!(decls.contains("struct DrawIndexedIndirectArgs {"));
        assert!(decls.contains("struct ColorUniform {"));
        assert!(decls.contains("struct MaterialParams {"));
        assert!(decls.contains("struct MeshUniforms {"));
        assert!(decls.contains("struct ToonMeshUniforms {"));
    }

    #[test]
    fn toon_mesh_uniforms_layout_and_byte_roundtrip_tests() {
        assert_eq!(TOON_MESH_UNIFORMS_BYTES, 304);
        assert_eq!(TOON_MESH_UNIFORMS_ALIGNMENT, 16);
        assert_eq!(TOON_MESH_DYNAMIC_OFFSET_STRIDE, 512);
        assert_eq!(core::mem::size_of::<ToonMeshUniforms>(), 304);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, model_world), 0);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, projection), 64);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, camera_view), 128);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, model_normal_matrix), 192);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, color), 240);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, light_direction), 256);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, light_color), 272);
        assert_eq!(core::mem::offset_of!(ToonMeshUniforms, params), 288);

        let default_uniforms = ToonMeshUniforms::default();
        let bytes = default_uniforms.to_bytes();
        assert_eq!(bytes.len(), 304);
        let restored = ToonMeshUniforms::from_bytes(&bytes);
        assert_eq!(restored, default_uniforms);

        let mut slice_buf = [0u8; 304];
        default_uniforms.write_to_slice(&mut slice_buf).expect("write ok");
        let read_back = ToonMeshUniforms::read_from_slice(&slice_buf).expect("read ok");
        assert_eq!(read_back, default_uniforms);

        // Short buffer error
        let mut short_buf = [0u8; 300];
        assert_eq!(
            default_uniforms.write_to_slice(&mut short_buf),
            Err(LayoutError::BufferTooSmall { required: 304, provided: 300 })
        );
        assert_eq!(
            ToonMeshUniforms::read_from_slice(&short_buf),
            Err(LayoutError::BufferTooSmall { required: 304, provided: 300 })
        );

        // Normal matrix padding helper verification
        let mat3_raw = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let padded = ToonMeshUniforms::pad_normal_matrix(&mat3_raw);
        assert_eq!(padded[0], [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(padded[1], [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(padded[2], [7.0, 8.0, 9.0, 0.0]);

        let mut custom = default_uniforms;
        custom.model_normal_matrix = padded;
        assert_eq!(custom.unpad_normal_matrix(), mat3_raw);
    }
}


