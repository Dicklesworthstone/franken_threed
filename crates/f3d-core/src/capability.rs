//! Runtime device and browser capability record.
//!
//! Invariant: Capability records represent measured host parameters; unknown values
//! are represented explicitly as unknown and never fabricated or guessed.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// Complete measured host and WebGPU device capability snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CapabilityRecord {
    /// Host browser and OS identification (e.g. "Safari 20 / macOS 25").
    pub host_environment: String,
    /// Whether WebGPU is supported on this host.
    pub webgpu_supported: bool,
    /// GPU adapter information.
    pub adapter: AdapterInfo,
    /// Enforced WebGPU device limits, if probed and available.
    pub limits: Option<DeviceLimits>,
    /// Supported WGSL language feature set.
    pub wgsl_features: Vec<String>,
    /// Preferred canvas presentation format.
    pub preferred_canvas_format: String,
    /// Canvas color space (e.g. "srgb", "display-p3", or "unknown").
    pub color_space: String,
}

impl CapabilityRecord {
    /// Return an unknown / unprobed baseline capability record.
    pub fn unknown() -> Self {
        Self {
            host_environment: String::from("unknown"),
            webgpu_supported: false,
            adapter: AdapterInfo::unknown(),
            limits: None,
            wgsl_features: Vec::new(),
            preferred_canvas_format: String::from("unknown"),
            color_space: String::from("unknown"),
        }
    }
}

/// WebGPU adapter identification metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdapterInfo {
    /// Vendor name or ID string.
    pub vendor: String,
    /// Architecture name.
    pub architecture: String,
    /// Device name or ID.
    pub device: String,
    /// Human-readable driver/adapter description.
    pub description: String,
    /// True if the adapter is a CPU software rasterizer.
    pub is_fallback_adapter: bool,
}

impl AdapterInfo {
    /// Create an unknown adapter record.
    pub fn unknown() -> Self {
        Self {
            vendor: String::from("unknown"),
            architecture: String::from("unknown"),
            device: String::from("unknown"),
            description: String::from("unknown"),
            is_fallback_adapter: false,
        }
    }
}

/// Standard WebGPU resource and alignment limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceLimits {
    /// Maximum 2D texture width/height in pixels.
    pub max_texture_dimension_2d: u32,
    /// Maximum allocated buffer size in bytes.
    pub max_buffer_size: u64,
    /// Maximum bind groups bound at once.
    pub max_bind_groups: u32,
    /// Required byte alignment for dynamic uniform buffer offsets (usually 256).
    pub min_uniform_buffer_offset_alignment: u32,
    /// Required byte alignment for dynamic storage buffer offsets (usually 256).
    pub min_storage_buffer_offset_alignment: u32,
    /// Maximum compute workgroup storage size in bytes.
    pub max_compute_workgroup_storage_size: u32,
}

impl DeviceLimits {
    /// Default standard WebGPU limits per spec baseline.
    pub const fn default_webgpu_limits() -> Self {
        Self {
            max_texture_dimension_2d: 8192,
            max_buffer_size: 268435456, // 256 MiB
            max_bind_groups: 4,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_compute_workgroup_storage_size: 16384,
        }
    }
}
