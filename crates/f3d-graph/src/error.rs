//! Error types for pass graph analysis, usage-scope hazards, and canvas validation.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::pass::PassKind;
use crate::resource::{ResourceAccess, SubresourceRange};

/// Top-level error type for pass graph compilation, scheduling, and validation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GraphError {
    /// A resource usage-scope hazard was detected.
    Hazard(HazardError),
    /// A canvas epoch or interval lifecycle error occurred.
    Canvas(CanvasError),
    /// A cyclic dependency was detected among passes.
    CycleDetected {
        /// Passes involved in the detected dependency cycle.
        cycle: Vec<u32>,
    },
    /// An explicit dependency references an unknown pass ID.
    MissingDependency {
        /// Pass that declared the dependency.
        pass_id: u32,
        /// The missing dependency pass ID.
        dependency_id: u32,
    },
    /// Attempted to access or schedule an unknown pass ID.
    PassNotFound {
        /// The pass ID that was not found.
        pass_id: u32,
    },
    /// A pass contains structurally invalid parameters.
    InvalidPass {
        /// ID of the invalid pass.
        pass_id: u32,
        /// Description of the structural defect.
        reason: String,
    },
    /// Attempted to add a pass with a duplicate ID.
    DuplicatePassId {
        /// The duplicate pass ID.
        pass_id: u32,
    },
    /// Pass ID counter overflowed u32::MAX.
    PassIdOverflow,
    /// The graph contains no executable passes.
    EmptyGraph,
}

impl From<HazardError> for GraphError {
    fn from(err: HazardError) -> Self {
        Self::Hazard(err)
    }
}

impl From<CanvasError> for GraphError {
    fn from(err: CanvasError) -> Self {
        Self::Canvas(err)
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hazard(h) => write!(f, "usage-scope hazard error: {h}"),
            Self::Canvas(c) => write!(f, "canvas error: {c}"),
            Self::CycleDetected { cycle } => {
                write!(f, "dependency cycle detected in pass graph: {cycle:?}")
            }
            Self::MissingDependency {
                pass_id,
                dependency_id,
            } => {
                write!(
                    f,
                    "pass {pass_id} references missing dependency pass {dependency_id}"
                )
            }
            Self::PassNotFound { pass_id } => write!(f, "pass {pass_id} not found in graph"),
            Self::InvalidPass { pass_id, reason } => {
                write!(f, "pass {pass_id} is structurally invalid: {reason}")
            }
            Self::DuplicatePassId { pass_id } => {
                write!(f, "pass {pass_id} already exists in graph; duplicate pass IDs are rejected")
            }
            Self::PassIdOverflow => write!(f, "pass ID allocation overflowed u32::MAX"),
            Self::EmptyGraph => write!(f, "cannot compile empty pass graph"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GraphError {}

/// Usage-scope hazard errors enforcing WebGPU specifications (§8.5, [S51]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HazardError {
    /// Incompatible usages of a single buffer within the same scope.
    ///
    /// Whole-buffer rule invariant: For WebGPU usage scopes, a buffer is a whole subresource.
    /// Disjoint byte offsets in ONE buffer cannot legalize incompatible usages!
    WholeBufferConflict {
        /// ID of the conflicting buffer.
        buffer_id: u32,
        /// First declared access.
        access_a: ResourceAccess,
        /// Incompatible second declared access.
        access_b: ResourceAccess,
        /// Optional byte offset of access A (shows non-overlap does not legalize).
        offset_a: Option<u64>,
        /// Optional byte offset of access B.
        offset_b: Option<u64>,
    },
    /// A texture subresource is simultaneously used as an attachment and sampled in the same render pass.
    AttachmentSamplingConflict {
        /// Texture ID involved in the conflict.
        texture_id: u32,
        /// Pass where the conflict occurred.
        pass_id: u32,
        /// Overlapping subresource range.
        subresource: SubresourceRange,
    },
    /// Writable binding aliases within a single compute dispatch.
    ComputeWritableAlias {
        /// Resource ID with aliasing writable bindings.
        resource_id: u32,
        /// Dispatch index where alias occurred.
        dispatch_id: u32,
        /// Overlapping subresource range.
        subresource: SubresourceRange,
    },
    /// Multiple writes to the same subresource within a single pass without synchronization.
    MultipleWriters {
        /// Resource ID with multiple writes.
        resource_id: u32,
        /// Pass where multiple writes occurred.
        pass_id: u32,
        /// Overlapping subresource range.
        subresource: SubresourceRange,
    },
    /// Copy source and destination overlap or reference the exact same resource.
    OverlappingCopyEndpoints {
        /// Conflicting resource ID.
        resource_id: u32,
    },
    /// An attempt to execute a copy inside a render pass.
    CopyInRenderPass {
        /// Pass where illegal copy was attempted.
        pass_id: u32,
    },
    /// Attempted to execute a render bundle inside a copy pass (§8.5, AGENTS.md).
    BundleInCopyPass {
        /// Pass where illegal bundle was attempted.
        pass_id: u32,
    },
    /// A direct draw following a render bundle assumed warm state instead of rebinding (§8.5, AGENTS.md).
    ///
    /// WebGPU specification invariant: `executeBundles` invalidates/resets all cached
    /// pipeline and bind group state on the render pass encoder. Any direct draw following
    /// a bundle must explicitly rebind; warm-state assumptions are strictly rejected.
    BundleDirectDrawRequiresRebind {
        /// Pass where the violation occurred.
        pass_id: u32,
        /// Draw identifier that assumed warm state without rebinding.
        draw_id: u32,
    },
    /// Attempted to execute a render bundle inside a non-render pass (§8.5).
    BundleInNonRenderPass {
        /// Pass where illegal bundle was attempted.
        pass_id: u32,
        /// Execution category of the invalid pass.
        pass_kind: PassKind,
    },
}

impl fmt::Display for HazardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WholeBufferConflict {
                buffer_id,
                access_a,
                access_b,
                offset_a,
                offset_b,
            } => {
                write!(
                    f,
                    "buffer {buffer_id} has incompatible usages {access_a:?} and {access_b:?} in the same scope; \
                     disjoint byte offsets ({offset_a:?} vs {offset_b:?}) do not legalize the whole buffer"
                )
            }
            Self::AttachmentSamplingConflict {
                texture_id,
                pass_id,
                subresource,
            } => {
                write!(
                    f,
                    "texture {texture_id} in pass {pass_id} conflicts: subresource {subresource:?} \
                     cannot be both an attachment and sampled in the same render pass"
                )
            }
            Self::ComputeWritableAlias {
                resource_id,
                dispatch_id,
                subresource,
            } => {
                write!(
                    f,
                    "compute dispatch {dispatch_id} contains writable alias for resource {resource_id} \
                     at subresource {subresource:?}"
                )
            }
            Self::MultipleWriters {
                resource_id,
                pass_id,
                subresource,
            } => {
                write!(
                    f,
                    "resource {resource_id} in pass {pass_id} has multiple unsynchronized writers \
                     at subresource {subresource:?}"
                )
            }
            Self::OverlappingCopyEndpoints { resource_id } => {
                write!(
                    f,
                    "copy command has overlapping or identical source and destination resource {resource_id}"
                )
            }
            Self::CopyInRenderPass { pass_id } => {
                write!(
                    f,
                    "copy command cannot be inserted inside render pass {pass_id}; copies must be in dedicated Copy passes"
                )
            }
            Self::BundleInCopyPass { pass_id } => {
                write!(
                    f,
                    "render bundle cannot be executed inside copy pass {pass_id}; bundles execute strictly within Render passes"
                )
            }
            Self::BundleDirectDrawRequiresRebind { pass_id, draw_id } => {
                write!(
                    f,
                    "direct draw {draw_id} in pass {pass_id} follows a render bundle and assumes warm state; \
                     bundle execution resets render pass state, requiring explicit rebind"
                )
            }
            Self::BundleInNonRenderPass { pass_id, pass_kind } => {
                write!(
                    f,
                    "render bundle cannot be executed inside {pass_kind:?} pass {pass_id}; bundles execute strictly within Render passes"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HazardError {}

/// Canvas output lifecycle and epoch errors (§8.5, [S48]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CanvasError {
    /// Canvas epoch monotonic counter overflowed `u64::MAX`.
    EpochOverflow {
        /// Target canvas ID.
        canvas_id: u32,
        /// Current epoch before attempted increment.
        current: u64,
    },
    /// Canvas texture access attempted against a stale or mismatched epoch.
    StaleCanvasEpoch {
        /// Target canvas ID.
        canvas_id: u32,
        /// Expected active epoch.
        expected_epoch: u64,
        /// Provided or attempted epoch.
        provided_epoch: u64,
    },
    /// Canvas texture accessed outside an active acquire interval.
    CanvasNotAcquired {
        /// Target canvas ID.
        canvas_id: u32,
    },
    /// Canvas texture used after the frame interval was already submitted.
    CanvasAlreadySubmitted {
        /// Target canvas ID.
        canvas_id: u32,
        /// Submitted epoch.
        epoch: u64,
    },
    /// Attempted to cache a canvas texture across output epochs.
    CanvasCachedAcrossEpochs {
        /// Target canvas ID.
        canvas_id: u32,
        /// Stale cached epoch.
        cached_epoch: u64,
        /// Current active epoch.
        current_epoch: u64,
    },
    /// Canvas has zero dimensions (0x0); rendering is paused per defined policy.
    ZeroSizedCanvasPause {
        /// Target canvas ID.
        canvas_id: u32,
    },
}

impl fmt::Display for CanvasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpochOverflow { canvas_id, current } => {
                write!(
                    f,
                    "canvas {canvas_id} epoch overflowed monotonically after {current}"
                )
            }
            Self::StaleCanvasEpoch {
                canvas_id,
                expected_epoch,
                provided_epoch,
            } => {
                write!(
                    f,
                    "canvas {canvas_id} epoch mismatch: expected {expected_epoch}, got {provided_epoch}"
                )
            }
            Self::CanvasNotAcquired { canvas_id } => {
                write!(
                    f,
                    "canvas {canvas_id} texture not acquired; must acquire before rendering"
                )
            }
            Self::CanvasAlreadySubmitted { canvas_id, epoch } => {
                write!(
                    f,
                    "canvas {canvas_id} texture at epoch {epoch} was already submitted"
                )
            }
            Self::CanvasCachedAcrossEpochs {
                canvas_id,
                cached_epoch,
                current_epoch,
            } => {
                write!(
                    f,
                    "canvas {canvas_id} texture from epoch {cached_epoch} was cached and reused in epoch {current_epoch}; \
                     canvas swapchain textures must be acquired fresh per render interval and never cached"
                )
            }
            Self::ZeroSizedCanvasPause { canvas_id } => {
                write!(
                    f,
                    "canvas {canvas_id} has zero dimensions (0x0); execution paused per policy"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CanvasError {}
