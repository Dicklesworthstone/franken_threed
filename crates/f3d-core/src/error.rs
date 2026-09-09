//! Structured error types for FrankenThreeD core.
//!
//! Invariant: Core library code does not print to stdout or stderr. All diagnostic
//! information is communicated through structured error types and events.

extern crate alloc;

use alloc::string::String;
use core::fmt;

/// Primary error enum for FrankenThreeD operations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum F3dError {
    /// Handle validation error (stale handle, generation mismatch, out of bounds).
    Handle(HandleError),
    /// A requested feature or limit was not satisfied by the host capabilities.
    CapabilityMismatch {
        /// The missing or unsupported feature.
        feature: String,
        /// Concrete rationale for mismatch.
        reason: String,
    },
    /// Feature manifest parsing or semantic validation error.
    ManifestValidation {
        /// Detailed validation failure description.
        message: String,
    },
    /// Struct or buffer layout mismatch between CPU and GPU expectation.
    InvalidLayout {
        /// Expected byte length.
        expected_bytes: usize,
        /// Actual byte length.
        actual_bytes: usize,
        /// Name of the layout type.
        type_name: String,
    },
    /// The GPU device was lost or generation invalidated.
    DeviceLost {
        /// Expected device generation.
        expected_generation: u32,
        /// Actual or current device generation.
        current_generation: u32,
    },
    /// A numeric or scene specialization was refused and fell back to general execution.
    SpecializationRefusal {
        /// Reason the specialization was refused.
        reason: String,
        /// Precise source location triggering refusal.
        span: Option<SourceSpan>,
    },
}

impl fmt::Display for F3dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handle(e) => write!(f, "handle error: {e}"),
            Self::CapabilityMismatch { feature, reason } => {
                write!(f, "capability mismatch for '{feature}': {reason}")
            }
            Self::ManifestValidation { message } => {
                write!(f, "manifest validation failed: {message}")
            }
            Self::InvalidLayout {
                expected_bytes,
                actual_bytes,
                type_name,
            } => {
                write!(
                    f,
                    "layout error in {type_name}: expected {expected_bytes} bytes, got {actual_bytes} bytes"
                )
            }
            Self::DeviceLost {
                expected_generation,
                current_generation,
            } => {
                write!(
                    f,
                    "GPU device lost: expected generation {expected_generation}, current is {current_generation}"
                )
            }
            Self::SpecializationRefusal { reason, span } => match span {
                Some(s) => write!(f, "specialization refused at {s}: {reason}"),
                None => write!(f, "specialization refused: {reason}"),
            },
        }
    }
}

impl core::error::Error for F3dError {}

impl From<HandleError> for F3dError {
    fn from(err: HandleError) -> Self {
        Self::Handle(err)
    }
}

/// Errors occurring during handle resolution or arena access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandleError {
    /// The handle index exceeds the arena capacity.
    IndexOutOfBounds {
        /// The queried index.
        index: u32,
        /// Current arena capacity.
        capacity: u32,
    },
    /// The slot generation did not match the handle generation (stale handle).
    GenerationMismatch {
        /// Slot index.
        index: u32,
        /// Generation recorded on the caller's handle.
        expected_generation: u32,
        /// Current generation active in the arena slot.
        actual_generation: u32,
    },
    /// The slot is currently vacant (item was destroyed or not yet allocated).
    SlotVacant {
        /// Vacant slot index.
        index: u32,
    },
    /// The handle was encoded with an invalid generation value (e.g. zero).
    InvalidGeneration {
        /// Zero or invalid generation value.
        raw_generation: u32,
    },
    /// The device generation recorded on the GPU handle does not match the active device.
    DeviceGenerationMismatch {
        /// Expected device generation on the handle.
        expected_device: u32,
        /// Active GPU device generation.
        current_device: u32,
    },
    /// Generation counter has wrapped or reached retirement threshold.
    GenerationOverflow {
        /// Slot index that reached retirement.
        index: u32,
    },
    /// Attempted an illegal lifecycle transition.
    InvalidLifecycleTransition,
}

impl fmt::Display for HandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, capacity } => {
                write!(f, "index {index} out of bounds (capacity {capacity})")
            }
            Self::GenerationMismatch {
                index,
                expected_generation,
                actual_generation,
            } => {
                write!(
                    f,
                    "slot {index} generation mismatch: handle had generation {expected_generation}, slot has generation {actual_generation}"
                )
            }
            Self::SlotVacant { index } => write!(f, "slot {index} is vacant"),
            Self::InvalidGeneration { raw_generation } => {
                write!(f, "invalid generation {raw_generation} (must be non-zero)")
            }
            Self::DeviceGenerationMismatch {
                expected_device,
                current_device,
            } => {
                write!(
                    f,
                    "device generation mismatch: handle has {expected_device}, active device is {current_device}"
                )
            }
            Self::GenerationOverflow { index } => {
                write!(f, "generation counter overflow in slot {index}; slot retired")
            }
            Self::InvalidLifecycleTransition => {
                write!(f, "invalid lifecycle transition: Retired and Exhausted states cannot be set via public setter")
            }
        }
    }
}

impl core::error::Error for HandleError {}

/// Source file location span for compiler and specialization barrier reporting.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceSpan {
    /// File path or source identifier.
    pub file: String,
    /// 1-indexed starting line number.
    pub start_line: u32,
    /// 1-indexed starting column number.
    pub start_col: u32,
    /// 1-indexed ending line number.
    pub end_line: u32,
    /// 1-indexed ending column number.
    pub end_col: u32,
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}-{}:{}",
            self.file, self.start_line, self.start_col, self.end_line, self.end_col
        )
    }
}
