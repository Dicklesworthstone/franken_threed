//! Browser execution through the upstream Asupersync runtime.
//!
//! The `browser` feature exposes the foundation probe on wasm32. The runtime,
//! task ownership, polling, and wakeup scheduling belong to Asupersync.
#![forbid(unsafe_code)]

pub use asupersync::runtime::{BrowserHostServices, RuntimeBuilder};

pub mod burst;
pub mod publication;

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
mod probe;
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use probe::{
    burst_max_polls_per_turn, burst_polls, pump_turns, reenter_probe, start_probes,
};
