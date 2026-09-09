//! Feature and route manifest types.
//!
//! Enforces the "Four Promises, Never Conflated" discipline and the "No-Cut Rule" (AGENTS.md).
//! Scope is all features and functionality of Three.js r186 (commit 148ef33ecb6d2502ff796d4554abd1549c95d519).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// Pinned Three.js release string.
pub const PINNED_UPSTREAM_RELEASE: &str = "r186";

/// Pinned Three.js source commit hash.
pub const PINNED_UPSTREAM_COMMIT: &str = "148ef33ecb6d2502ff796d4554abd1549c95d519";

/// Execution routes defining implementation ownership (AGENTS.md / Plan §5.1).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FeatureRoute {
    /// Verified accelerated Rust/Wasm/WebGPU execution.
    SpecializedWebGpu,
    /// Working WebGPU execution when specialization is unavailable; speed measured independently.
    GeneralWebGpu,
    /// Full component functionality — not a Rust rewrite.
    RetainedJs,
    /// Exact backend behavior and functional compatibility; never counted as new renderer acceleration.
    ExactBackend,
}

impl FeatureRoute {
    /// Human-readable route name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SpecializedWebGpu => "Specialized WebGPU",
            Self::GeneralWebGpu => "General WebGPU",
            Self::RetainedJs => "Retained JS / Host Component",
            Self::ExactBackend => "Exact Backend Component",
        }
    }
}

/// Feature compatibility status. Blocking states prevent release closure.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FeatureStatus {
    // === Blocking States (AGENTS.md "The No-Cut Rule") ===
    /// Symbol has not yet been classified into a route.
    Unclassified,
    /// Feature is known but has no working implementation yet.
    Unimplemented,
    /// Implementation exists but has no passing conformance test.
    Untested,
    /// Feature previously passed but has a known regression.
    KnownRegression,
    /// Function or object returns a hollow stub rather than real implementation.
    Stub,
    /// Operation is silently dropped or replaced by a no-op.
    NoOpSubstitute,
    /// The candidate refuses an operation that is valid in the source.
    CandidateRefusalOnValidSource,

    // === Non-Blocking States ===
    /// Implementation passes all acceptance and conformance criteria under its route.
    Verified,
    /// Retained ownership verified against upstream reference.
    Retained,
    /// Prerequisite is missing on host, identical to the reference environment.
    HostBlocked,
}

impl FeatureStatus {
    /// True if this status blocks release closure per the No-Cut Rule.
    pub const fn is_blocking(self) -> bool {
        match self {
            Self::Unclassified
            | Self::Unimplemented
            | Self::Untested
            | Self::KnownRegression
            | Self::Stub
            | Self::NoOpSubstitute
            | Self::CandidateRefusalOnValidSource => true,
            Self::Verified | Self::Retained | Self::HostBlocked => false,
        }
    }
}

/// A single feature or symbol record in the compatibility manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureEntry {
    /// Public symbol name or export path (e.g. "three/src/materials/MeshStandardMaterial.js").
    pub symbol: String,
    /// Category family (e.g. "material", "geometry", "loader", "math", "renderer").
    pub category: String,
    /// Assigned execution route.
    pub route: FeatureRoute,
    /// Current verification status.
    pub status: FeatureStatus,
    /// Optional documentation or barrier rationale.
    pub notes: Option<String>,
}

/// Complete compatibility manifest tracking Three.js r186 feature coverage.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureManifest {
    /// Target upstream release.
    pub upstream_release: String,
    /// Target upstream commit hash.
    pub upstream_commit: String,
    /// Pinned inventory of features.
    pub features: Vec<FeatureEntry>,
}

impl Default for FeatureManifest {
    fn default() -> Self {
        Self::new_pinned()
    }
}

impl FeatureManifest {
    /// Create a new empty manifest pinned to the canonical r186 anchor commit.
    pub fn new_pinned() -> Self {
        Self {
            upstream_release: String::from(PINNED_UPSTREAM_RELEASE),
            upstream_commit: String::from(PINNED_UPSTREAM_COMMIT),
            features: Vec::new(),
        }
    }

    /// Add an entry to the manifest.
    pub fn add_feature(&mut self, entry: FeatureEntry) {
        self.features.push(entry);
    }

    /// True if any registered feature is in a blocking state.
    pub fn has_blocking_features(&self) -> bool {
        self.features.iter().any(|f| f.status.is_blocking())
    }

    /// Count features currently in blocking states.
    pub fn count_blocking_features(&self) -> usize {
        self.features.iter().filter(|f| f.status.is_blocking()).count()
    }

    /// Count features by assigned route.
    pub fn count_by_route(&self, route: FeatureRoute) -> usize {
        self.features.iter().filter(|f| f.route == route).count()
    }
}
