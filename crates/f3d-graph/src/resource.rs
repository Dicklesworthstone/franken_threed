//! Typed resource representations, subresource ranges, and access classifications.

extern crate alloc;

use core::fmt;

use f3d_core::ownership::{DataVersion, Epoch};

/// Strongly typed 32-bit handle for a resource in the pass graph.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct ResourceId(pub u32);

impl ResourceId {
    /// Construct a resource ID from a raw `u32`.
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

impl fmt::Debug for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResourceId({})", self.0)
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Res#{}", self.0)
    }
}

/// Resource kind classification in the pass graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceKind {
    /// A linear memory buffer (vertex, index, uniform, storage, copy).
    Buffer,
    /// An offscreen 2D/3D/array/cube texture.
    Texture,
    /// A short-lived swapchain canvas output texture valid only in the current frame interval.
    CanvasOutput,
}

/// Texture aspect mask for subresource tracking.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureAspect {
    /// All aspects (color or combined depth/stencil).
    #[default]
    All,
    /// Color aspect only.
    Color,
    /// Depth aspect only.
    DepthOnly,
    /// Stencil aspect only.
    StencilOnly,
}

impl TextureAspect {
    /// Returns `true` if this aspect overlaps with `other`.
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::All, _) | (_, Self::All) => true,
            (Self::Color, Self::Color) => true,
            (Self::DepthOnly, Self::DepthOnly) => true,
            (Self::StencilOnly, Self::StencilOnly) => true,
            _ => false,
        }
    }
}

/// Subresource range for resource hazard tracking.
///
/// Invariant: In WebGPU, a buffer is a whole subresource. Disjoint byte ranges
/// do not subdivide a buffer into independent subresources for usage-scope validation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SubresourceRange {
    /// Buffer whole-subresource marker.
    WholeBuffer,
    /// Mip-level, array-layer, and aspect range within a texture.
    TextureRange {
        /// Base mip level.
        mip_level_start: u32,
        /// Number of mip levels (0 indicates unbounded/all).
        mip_level_count: u32,
        /// Base array layer.
        array_layer_start: u32,
        /// Number of array layers (0 indicates unbounded/all).
        array_layer_count: u32,
        /// Selected aspect.
        aspect: TextureAspect,
    },
}

impl SubresourceRange {
    /// Full texture coverage (all mips, all layers, all aspects).
    #[must_use]
    pub const fn full_texture() -> Self {
        Self::TextureRange {
            mip_level_start: 0,
            mip_level_count: 0,
            array_layer_start: 0,
            array_layer_count: 0,
            aspect: TextureAspect::All,
        }
    }

    /// Single mip level and single array layer.
    #[must_use]
    pub const fn single_mip_layer(mip: u32, layer: u32, aspect: TextureAspect) -> Self {
        Self::TextureRange {
            mip_level_start: mip,
            mip_level_count: 1,
            array_layer_start: layer,
            array_layer_count: 1,
            aspect,
        }
    }

    /// Returns `true` if two subresource ranges overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::WholeBuffer, Self::WholeBuffer) => true,
            (
                Self::TextureRange {
                    mip_level_start: m0,
                    mip_level_count: mc0,
                    array_layer_start: l0,
                    array_layer_count: lc0,
                    aspect: a0,
                },
                Self::TextureRange {
                    mip_level_start: m1,
                    mip_level_count: mc1,
                    array_layer_start: l1,
                    array_layer_count: lc1,
                    aspect: a1,
                },
            ) => {
                if !a0.overlaps(a1) {
                    return false;
                }
                let mips_overlap = ranges_overlap(*m0, *mc0, *m1, *mc1);
                let layers_overlap = ranges_overlap(*l0, *lc0, *l1, *lc1);
                mips_overlap && layers_overlap
            }
            _ => false,
        }
    }
}

#[inline]
const fn ranges_overlap(start0: u32, count0: u32, start1: u32, count1: u32) -> bool {
    let end0 = if count0 == 0 { u32::MAX } else { start0.saturating_add(count0) };
    let end1 = if count1 == 0 { u32::MAX } else { start1.saturating_add(count1) };
    start0 < end1 && start1 < end0
}

/// Resource access type classifying read-only vs writable usage in WebGPU passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceAccess {
    /// Read uniform buffer (`var<uniform>`).
    UniformBuffer,
    /// Read-only storage buffer (`var<storage, read>`).
    StorageBufferRead,
    /// Writable storage buffer (`var<storage, read_write>`).
    StorageBufferWrite,
    /// Vertex buffer input.
    VertexBuffer,
    /// Index buffer input.
    IndexBuffer,
    /// Sampled texture view (`texture_2d<f32>`, etc.).
    SampledTexture,
    /// Color attachment render target.
    ColorAttachment,
    /// Depth/stencil attachment render target.
    DepthStencilAttachment,
    /// Read-only depth attachment view.
    ReadOnlyDepthStencil,
    /// Storage texture write (`texture_storage_2d<..., write>`).
    StorageTextureWrite,
    /// Copy command source.
    CopySrc,
    /// Copy command destination.
    CopyDst,
}

impl ResourceAccess {
    /// Returns `true` if this access writes to the resource.
    #[must_use]
    pub const fn is_write(&self) -> bool {
        matches!(
            self,
            Self::StorageBufferWrite
                | Self::ColorAttachment
                | Self::DepthStencilAttachment
                | Self::StorageTextureWrite
                | Self::CopyDst
        )
    }

    /// Returns `true` if this access is read-only.
    #[must_use]
    pub const fn is_read(&self) -> bool {
        !self.is_write()
    }

    /// Returns `true` if this access is an attachment operation.
    #[must_use]
    pub const fn is_attachment(&self) -> bool {
        matches!(
            self,
            Self::ColorAttachment | Self::DepthStencilAttachment | Self::ReadOnlyDepthStencil
        )
    }

    /// Returns `true` if this access is a sampled texture operation.
    #[must_use]
    pub const fn is_sampled(&self) -> bool {
        matches!(self, Self::SampledTexture)
    }

    /// Returns `true` if two accesses are compatible within the same render pass usage scope.
    ///
    /// WebGPU Usage-Scope Rules:
    /// - Multiple reads are compatible.
    /// - Two writes are never compatible in the same scope without an intervening pass/barrier.
    /// - Read and write to the same subresource are not compatible (except ReadOnlyDepthStencil).
    /// - An attachment cannot also be sampled in the same render pass.
    #[must_use]
    pub const fn is_compatible_with(&self, other: &Self) -> bool {
        match (self, other) {
            // Attachment and sampled texture never coexist in the same render pass
            (a, b) if (a.is_attachment() && b.is_sampled()) || (a.is_sampled() && b.is_attachment()) => false,
            // Multiple reads are always compatible
            (a, b) if a.is_read() && b.is_read() => true,
            // Writable accesses cannot coexist with any other access to the same subresource
            _ => false,
        }
    }
}

/// A concrete usage of a resource in a draw, dispatch, or copy command.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceUse {
    /// Target resource ID.
    pub resource_id: ResourceId,
    /// Resource kind.
    pub kind: ResourceKind,
    /// Immutable per-use data version.
    pub version: DataVersion,
    /// Target subresource range.
    pub subresource: SubresourceRange,
    /// Declared access type.
    pub access: ResourceAccess,
    /// Optional byte offset (for buffers).
    /// Invariant: Non-zero or distinct offsets do NOT legalize incompatible usages on the same buffer.
    pub byte_offset: Option<u64>,
    /// Optional byte size of the access.
    pub byte_size: Option<u64>,
    /// Optional captured canvas output epoch (for CanvasOutput resources).
    pub canvas_epoch: Option<Epoch>,
}

impl ResourceUse {
    /// Create a uniform buffer read usage.
    pub fn buffer_uniform(
        resource_id: ResourceId,
        version: DataVersion,
        byte_offset: Option<u64>,
        byte_size: Option<u64>,
    ) -> Self {
        Self {
            resource_id,
            kind: ResourceKind::Buffer,
            version,
            subresource: SubresourceRange::WholeBuffer,
            access: ResourceAccess::UniformBuffer,
            byte_offset,
            byte_size,
            canvas_epoch: None,
        }
    }

    /// Create a storage buffer read usage.
    pub fn buffer_storage_read(
        resource_id: ResourceId,
        version: DataVersion,
        byte_offset: Option<u64>,
        byte_size: Option<u64>,
    ) -> Self {
        Self {
            resource_id,
            kind: ResourceKind::Buffer,
            version,
            subresource: SubresourceRange::WholeBuffer,
            access: ResourceAccess::StorageBufferRead,
            byte_offset,
            byte_size,
            canvas_epoch: None,
        }
    }

    /// Create a storage buffer write usage.
    pub fn buffer_storage_write(
        resource_id: ResourceId,
        version: DataVersion,
        byte_offset: Option<u64>,
        byte_size: Option<u64>,
    ) -> Self {
        Self {
            resource_id,
            kind: ResourceKind::Buffer,
            version,
            subresource: SubresourceRange::WholeBuffer,
            access: ResourceAccess::StorageBufferWrite,
            byte_offset,
            byte_size,
            canvas_epoch: None,
        }
    }

    /// Create a vertex buffer read usage.
    pub fn buffer_vertex(
        resource_id: ResourceId,
        version: DataVersion,
        byte_offset: Option<u64>,
        byte_size: Option<u64>,
    ) -> Self {
        Self {
            resource_id,
            kind: ResourceKind::Buffer,
            version,
            subresource: SubresourceRange::WholeBuffer,
            access: ResourceAccess::VertexBuffer,
            byte_offset,
            byte_size,
            canvas_epoch: None,
        }
    }

    /// Create a sampled texture read usage.
    pub fn texture_sampled(
        resource_id: ResourceId,
        version: DataVersion,
        subresource: SubresourceRange,
    ) -> Self {
        Self {
            resource_id,
            kind: ResourceKind::Texture,
            version,
            subresource,
            access: ResourceAccess::SampledTexture,
            byte_offset: None,
            byte_size: None,
            canvas_epoch: None,
        }
    }

    /// Create a color attachment write usage.
    pub fn texture_color_attachment(
        resource_id: ResourceId,
        version: DataVersion,
        subresource: SubresourceRange,
    ) -> Self {
        Self {
            resource_id,
            kind: ResourceKind::Texture,
            version,
            subresource,
            access: ResourceAccess::ColorAttachment,
            byte_offset: None,
            byte_size: None,
            canvas_epoch: None,
        }
    }

    /// Create a canvas output color attachment usage capturing the exact frame epoch.
    pub fn canvas_color_attachment(resource_id: ResourceId, epoch: Epoch) -> Self {
        Self {
            resource_id,
            kind: ResourceKind::CanvasOutput,
            version: DataVersion::new(epoch.get()),
            subresource: SubresourceRange::full_texture(),
            access: ResourceAccess::ColorAttachment,
            byte_offset: None,
            byte_size: None,
            canvas_epoch: Some(epoch),
        }
    }
}
