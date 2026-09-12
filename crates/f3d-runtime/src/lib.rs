//! Browser execution through the upstream Asupersync runtime.
//!
//! The `browser` feature exposes the foundation probe on wasm32. The runtime,
//! task ownership, polling, and wakeup scheduling belong to Asupersync.
#![forbid(unsafe_code)]

pub use asupersync::runtime::{BrowserHostServices, RuntimeBuilder};

pub mod burst;
pub mod frame;
pub mod gpu_host;
pub mod mesh;
pub mod publication;

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
mod probe;
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use probe::{
    burst_max_polls_per_turn, burst_polls, pump_turns, reenter_probe, start_probes,
};
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use gpu_host::{
    f3d_advance_resource_generation, f3d_borrow_enter, f3d_borrow_exit,
    f3d_borrow_growth_generation, f3d_build_affine_rows_transform_packet,
    f3d_build_bundle_direct_draw_packet, f3d_build_draw_parameters_packet,
    f3d_build_first_frame_packet, f3d_build_nested_canvas_pass_packet,
    f3d_build_nested_pass_packet, f3d_build_nested_viewport_scissor_packet,
    f3d_build_overlapping_depth_packet,
    f3d_build_red_a_blue_b_packet, f3d_check_resource_handle, f3d_try_grow_memory,
    f3d_try_publish_readback, f3d_validate_affine_rows,
    gpu_bridge_advance_resource_generation, gpu_bridge_borrow_enter, gpu_bridge_borrow_exit,
    gpu_bridge_borrow_growth_generation, gpu_bridge_build_affine_rows_transform_packet,
    gpu_bridge_build_bundle_direct_draw_packet, gpu_bridge_build_draw_parameters_packet,
    gpu_bridge_build_nested_canvas_pass_packet, gpu_bridge_build_nested_pass_packet,
    gpu_bridge_build_nested_viewport_scissor_packet, gpu_bridge_build_overlapping_depth_packet,
    gpu_bridge_build_red_blue_packet,
    gpu_bridge_build_triangle_packet, gpu_bridge_check_resource_handle,
    gpu_bridge_try_grow_memory, gpu_bridge_try_publish_readback,
    gpu_bridge_validate_affine_rows,
};

pub use mesh::{
    build_mesh_canvas_depth_submission, build_mesh_canvas_submission,
    build_mesh_depth_submission, build_mesh_submission,
    f3d_build_canvas_mesh_depth_packet, f3d_build_canvas_mesh_packet,
    f3d_build_mesh_depth_packet, f3d_build_mesh_packet,
    generate_mesh_canvas_wgsl, generate_mesh_wgsl,
    gpu_bridge_build_canvas_mesh_depth_packet, gpu_bridge_build_canvas_mesh_packet,
    gpu_bridge_build_mesh_depth_packet, gpu_bridge_build_mesh_packet,
    srgb_transfer_oetf_cpu,
    DynamicMeshInput, MeshDepthOptions, MeshPacketError,
    MESH_CANVAS_PIPELINE_ID, MESH_CANVAS_TARGET_ID, MESH_CLEAR_COLOR,
    MESH_DEPTH_TEXTURE_ID, MESH_PIPELINE_ID, MESH_READBACK_BUFFER_ID,
    MESH_TARGET_TEXTURE_ID, MESH_UNIFORM_BUFFER_ID, MESH_VERTEX_BUFFER_ID,
};
