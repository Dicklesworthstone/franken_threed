//! Browser execution through the upstream Asupersync runtime.
//!
//! The `browser` feature exposes the foundation probe on wasm32. The runtime,
//! task ownership, polling, and wakeup scheduling belong to Asupersync.
#![forbid(unsafe_code)]

pub use asupersync::runtime::{BrowserHostServices, RuntimeBuilder};

pub mod burst;
pub mod deformation;
pub mod frame;
pub mod gpu_host;
pub mod hierarchy;
pub mod mesh;
pub mod publication;
pub mod simd_compose;
pub mod skeleton;

pub use hierarchy::{HierarchyError, HierarchySolveStats, TransformHierarchy};
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use hierarchy::WasmTransformHierarchy;
pub use skeleton::{
    batch_skeleton_palette_f32, build_hierarchy_mesh_submission,
    deform_hierarchy_geometry, skeleton_palette_from_hierarchy,
    HierarchyGeometry, HierarchySkinning, SkeletonError,
};
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use skeleton::f3d_batch_skeleton_palette;

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
mod probe;
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use probe::{
    burst_max_polls_per_turn, burst_polls, pump_turns, reenter_probe, start_probes,
};
pub use simd_compose::{
    batch_compose_f64_scalar, batch_compose_f64_simd, f3d_batch_compose_scalar,
    f3d_batch_compose_simd, ComposeMode, SimdComposeError,
};
#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use gpu_host::{
    f3d_advance_resource_generation, f3d_borrow_enter, f3d_borrow_exit,
    f3d_borrow_growth_generation, f3d_bridge_callback_storage_upload_frame,
    f3d_bridge_chatty_draw_loop,
    f3d_build_affine_rows_transform_packet,
    f3d_build_bundle_direct_draw_packet, f3d_build_draw_parameters_packet,
    f3d_build_first_frame_packet, f3d_build_nested_canvas_pass_packet,
    f3d_build_nested_pass_packet, f3d_build_nested_viewport_scissor_packet,
    f3d_build_overlapping_depth_packet,
    f3d_build_red_a_blue_b_packet, f3d_build_textured_affine_triangle_packet,
    f3d_check_resource_handle, f3d_try_grow_memory,
    f3d_try_publish_readback, f3d_validate_affine_rows,
    gpu_bridge_advance_resource_generation, gpu_bridge_borrow_enter, gpu_bridge_borrow_exit,
    gpu_bridge_borrow_growth_generation,
    gpu_bridge_build_affine_rows_transform_packet,
    gpu_bridge_build_bundle_direct_draw_packet, gpu_bridge_build_draw_parameters_packet,
    gpu_bridge_build_nested_canvas_pass_packet, gpu_bridge_build_nested_pass_packet,
    gpu_bridge_build_nested_viewport_scissor_packet, gpu_bridge_build_overlapping_depth_packet,
    gpu_bridge_build_red_blue_packet,
    gpu_bridge_build_textured_affine_triangle_packet,
    gpu_bridge_build_triangle_packet, gpu_bridge_check_resource_handle,
    gpu_bridge_try_grow_memory, gpu_bridge_try_publish_readback,
    gpu_bridge_validate_affine_rows,
};

pub use gpu_host::{
    borrow_scope_bytes_view, borrowed_frame_packet_len, borrowed_frame_packet_ptr,
    build_affine_rows_batch_frame_packet_borrowed,
    build_affine_rows_batch_frame_submission, build_affine_rows_batch_submission,
    build_affine_rows_compute_submission,
    build_two_dispatch_affine_compute_submission,
    build_affine_rows_layout_counterexample_submission,
    build_affine_rows_storage_upload_callback_ops,
    build_affine_rows_storage_upload_frame_submission,
    build_affine_rows_storage_upload_init_submission,
    build_buffer_copy_submission, build_bundle_then_direct_warm_cache_submission,
    build_multisample_submission, f3d_build_multisample_packet,
    build_render_then_copy_submission,
    clear_borrowed_frame_packet,
    f3d_borrowed_frame_packet_len, f3d_borrowed_frame_packet_ptr,
    f3d_build_affine_rows_batch_frame_packet,
    f3d_build_affine_rows_batch_packet,
    f3d_build_affine_rows_batch_frame_packet_borrowed,
    f3d_build_affine_rows_bundle_packet,
    f3d_build_affine_rows_compute_packet,
    f3d_build_two_dispatch_affine_compute_packet,
    f3d_build_affine_rows_layout_counterexample_packet,
    f3d_build_affine_rows_storage_upload_frame_packet,
    f3d_build_affine_rows_storage_upload_init_packet,
    f3d_build_buffer_copy_packet, f3d_build_bundle_direct_warm_cache_packet,
    f3d_build_render_then_copy_packet,
    f3d_pack_affine_rows_bytes, f3d_pack_affine_rows_storage_bytes,
    map_storage_upload_submission_to_callback_ops,
    pack_affine_rows_storage_bytes, pack_affine_rows_uniform_bytes,
    with_global_borrowed_frame_packet_ref,
    GpuBufferBinding, GpuComputeBindingLayout, StorageUploadCallbackOp,
    AFFINE_ROWS_STORAGE_BUFFER_ID, AFFINE_ROWS_STORAGE_DST_SLOT_0,
    AFFINE_ROWS_STORAGE_DST_SLOT_1, AFFINE_ROWS_STORAGE_SRC_BUFFER_ID,
    BINDING_TYPE_STORAGE_READ, BINDING_TYPE_STORAGE_READ_WRITE, BINDING_TYPE_UNIFORM,
    BUFFER_USAGE_STORAGE,
    COMPUTE_AFFINE_BUFFER_ID, COMPUTE_INPUT_POINTS_BUFFER_ID,
    COMPUTE_OUTPUT_POINTS_BUFFER_ID, COMPUTE_READBACK_BUFFER_ID, COMPUTE_PIPELINE_ID,
    CULL_MODE_BACK, CULL_MODE_FRONT, CULL_MODE_NONE, FRONT_FACE_CCW, FRONT_FACE_CW,
    OPCODE_COPY_BUFFER_TO_BUFFER, OPCODE_CREATE_COMPUTE_PIPELINE, OPCODE_DISPATCH_COMPUTE,
    OPCODE_CREATE_PIPELINE_CULL, OPCODE_CREATE_PIPELINE_DEPTH_CULL,
    OPCODE_CREATE_PIPELINE_DEPTH_CULL_COLOR,
    OPCODE_CREATE_TEXTURE_MULTISAMPLED, OPCODE_CREATE_PIPELINE_MULTISAMPLED,
    OPCODE_RENDER_PASS_RESOLVE,
};

pub use mesh::{
    build_mesh_canvas_depth_submission, build_mesh_canvas_submission,
    build_mesh_depth_submission, build_mesh_submission,
    build_multi_mesh_canvas_depth_submission, build_multi_mesh_canvas_submission,
    build_multi_mesh_depth_submission, build_multi_mesh_submission,
    f3d_build_canvas_mesh_depth_packet, f3d_build_canvas_mesh_packet,
    f3d_build_mesh_batch_cull_depth_color_packet,
    f3d_build_mesh_batch_cull_depth_packet, f3d_build_mesh_batch_cull_packet,
    f3d_build_mesh_batch_packet,
    f3d_build_mesh_batch_vertex_color_clear_packet,
    f3d_build_mesh_batch_vertex_color_packet,
    f3d_build_mesh_depth_packet, f3d_build_mesh_packet,
    f3d_build_scene_clear_packet,
    generate_mesh_wgsl,
    gpu_bridge_build_canvas_mesh_depth_packet, gpu_bridge_build_canvas_mesh_packet,
    gpu_bridge_build_mesh_depth_packet,
    gpu_bridge_build_mesh_packet,
    build_mesh_batch_cull_depth_color_packet_impl,
    build_mesh_batch_cull_depth_packet_impl,
    build_mesh_batch_vertex_color_clear_packet_impl,
    build_mesh_batch_vertex_color_packet_impl,
    build_scene_clear_packet_impl,
    build_scene_clear_submission,
    DynamicMeshInput, MeshDepthOptions, MeshPacketError,
    MESH_CANVAS_PIPELINE_ID, MESH_CANVAS_TARGET_ID, MESH_CLEAR_COLOR,
    MESH_DEPTH_TEXTURE_ID, MESH_PIPELINE_ID, MESH_READBACK_BUFFER_ID,
    MESH_TARGET_TEXTURE_ID, MESH_UNIFORM_BUFFER_ID, MESH_VERTEX_BUFFER_ID,
};

pub use deformation::{
    build_deformed_mesh_packet, build_deformed_mesh_submission, deform_geometry,
    DeformedGeometry, DeformedMeshError, GeometryDeformation, MorphTargets,
};
pub use deformation::f3d_build_deformed_mesh_packet;
