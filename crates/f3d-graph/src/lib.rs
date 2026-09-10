//! `f3d-graph`: WebGPU pass graph, usage-scope hazard tracking, and execution planning.
//!
//! NO-CLAIM: static graph correctness is not browser rendering, whole H1 support, or measured optimization.
//!
//! # Architecture & Guarantees
//! - **Middle Representation (§6.1)**: Bridges the high-level semantic scene/shader definitions
//!   and low-level GPU submissions via an explicit, validated pass graph.
//! - **Whole-Buffer Invariant (§8.5, [S51])**: A buffer is a whole subresource for usage scopes.
//!   Assigning disjoint byte offsets incompatible roles does not legalize the shared buffer.
//! - **Pass Splitting**: Render passes containing attachment versus sampling conflicts are
//!   automatically split into ordered sequential passes where draw equivalence permits.
//! - **Compute Dispatch Scope**: Writable aliases within a single compute dispatch are strictly forbidden.
//! - **Canvas Freshness**: Canvas swapchain textures are valid strictly inside the frame's
//!   acquire-to-submit interval and are never cached across frames. Multiple canvases maintain
//!   distinct, independent output epochs.
//! - **No Silent Drops or Fabricated Barriers**: The graph preserves every valid pass and never
//!   invents synthetic native memory barriers not exposed by WebGPU.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod canvas;
pub mod error;
pub mod hazard;
pub mod pass;
pub mod plan;
pub mod resource;
pub mod schedule;

pub use canvas::{CanvasEpochTracker, CanvasFormat, CanvasId, CanvasOutput};
pub use error::{CanvasError, GraphError, HazardError};
pub use hazard::{can_split_pass, split_pass_on_hazard, validate_pass_hazards};
pub use pass::{
    ColorAttachment, CopyCommand, DepthStencilAttachment, Dispatch, Draw, DrawKind, LoadOp, Pass,
    PassId, PassKind, RenderBundle, StoreOp,
};
pub use plan::{
    ExecutionPlan, PlanSegment, build_red_a_blue_b_plan, build_red_a_blue_b_render_plan,
    build_single_pass_bridge_plan, build_two_target_bridge_plan,
};
pub use resource::{
    ResourceAccess, ResourceId, ResourceKind, ResourceUse, SubresourceRange, TextureAspect,
};
pub use schedule::PassGraph;
