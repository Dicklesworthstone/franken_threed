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
    /// Verifies that row 3 is affine `[0.0, 0.0, 0.0, 1.0]`. If perspective components
    /// or projective scaling are present, returns `Err(LayoutError::NonAffineMatrix)`.
    pub fn from_column_major(e: &[f32; 16]) -> Result<Self, LayoutError> {
        if !is_matrix4_affine(e) {
            return Err(LayoutError::NonAffineMatrix);
        }
        Ok(Self::from_column_major_unchecked(e))
    }

    /// Converts a Three.js column-major 4x4 matrix into `AffineRows` without verifying row 3.
    pub const fn from_column_major_unchecked(e: &[f32; 16]) -> Self {
        Self {
            r0: [e[0], e[4], e[8], e[12]],
            r1: [e[1], e[5], e[9], e[13]],
            r2: [e[2], e[6], e[10], e[14]],
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
}

/// Helper function to determine if a 4x4 matrix is affine.
///
/// A matrix is affine if row 3 is `[0, 0, 0, 1]`.
pub fn is_matrix4_affine(e: &[f32; 16]) -> bool {
    let eps = 1e-6f32;
    e[3].abs() <= eps
        && e[7].abs() <= eps
        && e[11].abs() <= eps
        && (e[15] - 1.0f32).abs() <= eps
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

/// Validates storage buffer array stride for an element type.
///
/// Storage buffer arrays must have element stride >= element size and aligned to 16 bytes for matrix/vector composites.
pub fn validate_storage_array_stride(stride: usize, element_size: usize) -> Result<(), LayoutError> {
    if stride < element_size {
        return Err(LayoutError::BufferTooSmall {
            required: element_size,
            provided: stride,
        });
    }
    if stride % 16 != 0 {
        return Err(LayoutError::UnalignedOffset {
            offset: stride,
            required_alignment: 16,
        });
    }
    Ok(())
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

/// Instance transform record layout for GPU instance buffers (world transform + instance ID).
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

/// Canonical WGSL struct declaration for `AffineRows`.
pub const WGSL_AFFINE_ROWS_DECLARATION: &str = r#"
struct AffineRows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
};

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
"#;

/// Canonical WGSL struct declaration for `ProjectiveMat4`.
pub const WGSL_PROJECTIVE_MAT4_DECLARATION: &str = r#"
struct ProjectiveMat4 {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
    col3: vec4<f32>,
};
"#;

/// Returns standard generated WGSL type declarations and helpers for use in shaders.
pub fn generate_wgsl_declarations() -> String {
    let mut s = String::new();
    s.push_str("// === FrankenThreeD Generated Layouts ===\n");
    s.push_str(WGSL_AFFINE_ROWS_DECLARATION.trim());
    s.push_str("\n\n");
    s.push_str(WGSL_PROJECTIVE_MAT4_DECLARATION.trim());
    s.push('\n');
    s
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
}
