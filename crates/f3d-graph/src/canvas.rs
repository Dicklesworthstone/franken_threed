//! Canvas output lifecycles, fresh swapchain acquisition intervals, and per-canvas output epochs.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use f3d_core::ownership::Epoch;

use crate::error::CanvasError;
use crate::resource::ResourceId;

/// Strongly typed 32-bit handle identifying a canvas instance.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct CanvasId(pub u32);

impl CanvasId {
    /// Construct a canvas ID from raw `u32`.
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

impl fmt::Debug for CanvasId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CanvasId({})", self.0)
    }
}

impl fmt::Display for CanvasId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Canvas#{}", self.0)
    }
}

/// Supported pixel formats for canvas presentation surfaces.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CanvasFormat {
    /// BGRA 8-bit unnormalized (standard WebGPU preferred format on most platforms).
    #[default]
    Bgra8Unorm,
    /// RGBA 8-bit unnormalized.
    Rgba8Unorm,
}

/// Concrete canvas swapchain output resource representation for a single frame interval.
///
/// Invariant: Valid between `begin_frame_acquire` and `end_frame_interval`.
/// Queue submission does not end this interval; multiple submissions may use it.
/// The pass graph refuses to cache or retain canvas textures across output epochs.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CanvasOutput {
    /// Target canvas ID.
    pub canvas_id: CanvasId,
    /// Assigned graph resource ID.
    pub resource_id: ResourceId,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: CanvasFormat,
    /// Output epoch when this texture was acquired.
    pub epoch: Epoch,
}

/// Internal state tracking per canvas instance.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CanvasState {
    canvas_id: CanvasId,
    resource_id: ResourceId,
    width: u32,
    height: u32,
    format: CanvasFormat,
    current_epoch: Epoch,
    is_acquired_in_interval: bool,
    interval_ended: bool,
}

/// Manages per-canvas output epochs and validates interval acquisition freshness (§8.5, [S48]).
///
/// Invariants:
/// 1. Multiple canvases have completely independent output epochs.
/// 2. Canvas swapchain textures must be acquired fresh per render interval and never cached across frames.
/// 3. Zero-sized (0x0) canvases trigger defined pause policies without allocating illegal textures.
/// 4. Resize or reconfiguration invalidates existing attachment plans and advances the epoch.
#[derive(Default, Clone, Debug)]
pub struct CanvasEpochTracker {
    canvases: Vec<CanvasState>,
}

impl CanvasEpochTracker {
    /// Create a new empty canvas epoch tracker.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            canvases: Vec::new(),
        }
    }

    /// Register a new canvas instance with initial dimensions and format.
    ///
    /// If the canvas is already registered:
    /// - If the configuration (`resource_id`, `width`, `height`, `format`) is identical,
    ///   the existing state (including active acquisition interval and epoch) is preserved.
    /// - If any configuration parameter has changed, the canvas surface is reconfigured:
    ///   the acquired interval is closed (`is_acquired_in_interval = false`), and the epoch
    ///   advances monotonically without wrapping, rendering any previous frame output stale.
    pub fn register_canvas(
        &mut self,
        canvas_id: CanvasId,
        resource_id: ResourceId,
        width: u32,
        height: u32,
        format: CanvasFormat,
    ) {
        if let Some(pos) = self.canvases.iter().position(|c| c.canvas_id == canvas_id) {
            let state = &mut self.canvases[pos];
            let is_same_config = state.resource_id == resource_id
                && state.width == width
                && state.height == height
                && state.format == format;

            if !is_same_config {
                state.resource_id = resource_id;
                state.width = width;
                state.height = height;
                state.format = format;
                state.is_acquired_in_interval = false;
                state.interval_ended = false;

                // Advance epoch monotonically without wrapping
                if let Ok(next) = state.current_epoch.checked_next() {
                    state.current_epoch = next;
                } else {
                    state.current_epoch = Epoch::new(u64::MAX);
                }
            }
        } else {
            self.canvases.push(CanvasState {
                canvas_id,
                resource_id,
                width,
                height,
                format,
                current_epoch: Epoch::ZERO,
                is_acquired_in_interval: false,
                interval_ended: false,
            });
        }
    }

    /// Begins the rendering interval by acquiring the current swapchain texture.
    ///
    /// Checks:
    /// 1. If dimensions are zero (0x0), returns `CanvasError::ZeroSizedCanvasPause`.
    /// 2. Advances the canvas output epoch monotonically.
    /// 3. Marks the interval as actively acquired.
    pub fn begin_frame_acquire(&mut self, canvas_id: CanvasId) -> Result<CanvasOutput, CanvasError> {
        let state = self
            .canvases
            .iter_mut()
            .find(|c| c.canvas_id == canvas_id)
            .ok_or(CanvasError::CanvasNotAcquired {
                canvas_id: canvas_id.get(),
            })?;

        // Zero-sized canvas policy: pause rendering without creating illegal textures
        if state.width == 0 || state.height == 0 {
            return Err(CanvasError::ZeroSizedCanvasPause {
                canvas_id: canvas_id.get(),
            });
        }

        // Advance output epoch monotonically
        let next_epoch = state
            .current_epoch
            .checked_next()
            .map_err(|_| CanvasError::EpochOverflow {
                canvas_id: canvas_id.get(),
                current: state.current_epoch.get(),
            })?;
        state.current_epoch = next_epoch;
        state.is_acquired_in_interval = true;
        state.interval_ended = false;

        Ok(CanvasOutput {
            canvas_id,
            resource_id: state.resource_id,
            width: state.width,
            height: state.height,
            format: state.format,
            epoch: state.current_epoch,
        })
    }

    /// Validates that a reference to a canvas texture is currently legal and fresh.
    pub fn validate_canvas_access(
        &self,
        canvas_id: CanvasId,
        provided_epoch: Epoch,
    ) -> Result<(), CanvasError> {
        let state = self
            .canvases
            .iter()
            .find(|c| c.canvas_id == canvas_id)
            .ok_or(CanvasError::CanvasNotAcquired {
                canvas_id: canvas_id.get(),
            })?;

        if state.interval_ended {
            return Err(CanvasError::CanvasIntervalEnded {
                canvas_id: canvas_id.get(),
                epoch: state.current_epoch.get(),
            });
        }

        if !state.is_acquired_in_interval {
            return Err(CanvasError::CanvasNotAcquired {
                canvas_id: canvas_id.get(),
            });
        }

        // Check epoch freshness
        if provided_epoch != state.current_epoch {
            if provided_epoch.get() < state.current_epoch.get() {
                // Cached texture from an earlier epoch!
                return Err(CanvasError::CanvasCachedAcrossEpochs {
                    canvas_id: canvas_id.get(),
                    cached_epoch: provided_epoch.get(),
                    current_epoch: state.current_epoch.get(),
                });
            } else {
                return Err(CanvasError::StaleCanvasEpoch {
                    canvas_id: canvas_id.get(),
                    expected_epoch: state.current_epoch.get(),
                    provided_epoch: provided_epoch.get(),
                });
            }
        }

        Ok(())
    }

    /// Ends the host's rendering interval, preventing use until the next acquire.
    ///
    /// Call when the host expires the acquired canvas texture, not after each
    /// `queue.submit`. Multiple submissions within one interval remain legal.
    pub fn end_frame_interval(&mut self, canvas_id: CanvasId, epoch: Epoch) -> Result<(), CanvasError> {
        self.validate_canvas_access(canvas_id, epoch)?;
        let state = self
            .canvases
            .iter_mut()
            .find(|c| c.canvas_id == canvas_id)
            .unwrap();
        state.is_acquired_in_interval = false;
        state.interval_ended = true;
        Ok(())
    }

    /// Resizes the canvas surface, advancing epoch and invalidating existing attachment plans.
    pub fn resize(
        &mut self,
        canvas_id: CanvasId,
        new_width: u32,
        new_height: u32,
    ) -> Result<Epoch, CanvasError> {
        let state = self
            .canvases
            .iter_mut()
            .find(|c| c.canvas_id == canvas_id)
            .ok_or(CanvasError::CanvasNotAcquired {
                canvas_id: canvas_id.get(),
            })?;

        state.width = new_width;
        state.height = new_height;
        state.is_acquired_in_interval = false;
        state.interval_ended = false;

        let next_epoch = state
            .current_epoch
            .checked_next()
            .map_err(|_| CanvasError::EpochOverflow {
                canvas_id: canvas_id.get(),
                current: state.current_epoch.get(),
            })?;
        state.current_epoch = next_epoch;
        Ok(state.current_epoch)
    }

    /// Returns the current active epoch for a canvas.
    pub fn epoch_of(&self, canvas_id: CanvasId) -> Option<Epoch> {
        self.canvases.iter().find(|c| c.canvas_id == canvas_id).map(|c| c.current_epoch)
    }

    /// Returns the CanvasId associated with a resource ID, if registered.
    #[must_use]
    pub fn find_by_resource(&self, resource_id: ResourceId) -> Option<CanvasId> {
        self.canvases
            .iter()
            .find(|c| c.resource_id == resource_id)
            .map(|c| c.canvas_id)
    }

    /// Test seam: seeds the current epoch of a registered canvas for overflow testing.
    #[doc(hidden)]
    pub fn seed_epoch_for_test(&mut self, canvas_id: CanvasId, epoch: Epoch) {
        if let Some(state) = self.canvases.iter_mut().find(|c| c.canvas_id == canvas_id) {
            state.current_epoch = epoch;
        }
    }
}
