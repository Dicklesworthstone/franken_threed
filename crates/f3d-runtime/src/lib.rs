//! `f3d-runtime`: Asupersync execution core, browser lifecycle, and host protocol for FrankenThreeD.
//!
//! Provides the threadless single-worker browser execution core backed by Asupersync,
//! with non-reentrant microtask pumping, bounded burst yields, and structured probe execution.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod probe;

use alloc::sync::Arc;
pub use probe::{
    BrowserExecutionHarness, BrowserProbeConfig, ProbeEvent, ProbeEventKind, ProbeEventLogger,
    ProbeKind, ProbeSource, ProbeSummary,
};

/// High-level configuration for the FrankenThreeD browser runtime execution core.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BrowserRuntimeConfig {
    /// Maximum microtasks processed before yielding to the host event loop.
    pub microtask_burst_limit: usize,
    /// Probe execution parameters.
    pub probe_config: BrowserProbeConfig,
}

impl Default for BrowserRuntimeConfig {
    fn default() -> Self {
        Self {
            microtask_burst_limit: 32,
            probe_config: BrowserProbeConfig::default(),
        }
    }
}

/// Core runtime wrapper providing browser task admission and lifecycle coordination.
#[derive(Debug)]
pub struct BrowserRuntimeCore {
    config: BrowserRuntimeConfig,
    logger: Arc<ProbeEventLogger>,
}

impl BrowserRuntimeCore {
    /// Creates a new runtime core instance with the specified configuration.
    #[must_use]
    pub fn new(config: BrowserRuntimeConfig) -> Self {
        let logger = Arc::new(ProbeEventLogger::new());
        Self { config, logger }
    }

    /// Returns a reference to the runtime configuration.
    #[must_use]
    pub fn config(&self) -> &BrowserRuntimeConfig {
        &self.config
    }

    /// Returns a clone of the event logger.
    #[must_use]
    pub fn logger(&self) -> Arc<ProbeEventLogger> {
        Arc::clone(&self.logger)
    }

    /// Creates an execution harness configured for this runtime.
    #[must_use]
    pub fn create_harness(&self) -> BrowserExecutionHarness {
        BrowserExecutionHarness::new(self.config.probe_config.clone(), Arc::clone(&self.logger))
    }
}

impl Default for BrowserRuntimeCore {
    fn default() -> Self {
        Self::new(BrowserRuntimeConfig::default())
    }
}
