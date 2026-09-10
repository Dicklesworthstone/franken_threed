//! Direct unit and integration tests for `f3d-graph`.
//!
//! Enforces:
//! 1. Whole-buffer rule: disjoint byte offsets in ONE buffer cannot legalize incompatible usages.
//! 2. Rejection of unsupported feedback loops without silently dropping outputs.
//! 3. Preservation of all color destinations and depth/stencil state during legal pass splitting.
//! 4. Rewiring of downstream consumer dependencies to the final split segment.
//! 5. Compute per-dispatch writable binding alias rejection.
//! 6. Topological order and cycle rejection.
//! 7. Duplicate PassId rejection before mutating graph state, and checked ID overflow.
//! 8. Canvas fresh acquisition interval, captured epoch validation, and stale plan rejection.
//! 9. Execution plan generation and compile validation for ChartreuseFern's bridge cases.
//! 10. WebGPU copy bytes_per_row 256-byte alignment (width 32 -> 256 bytes).
//! 11. Red-A / Blue-B versioned buffer snapshot schedule.

use core::num::NonZeroU32;
use f3d_core::handle::{Handle, MaterialDomain};
use f3d_core::layout::{aligned_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT};
use f3d_core::ownership::{DataVersion, Epoch, PerUseByteBuffer};
use f3d_graph::canvas::{CanvasEpochTracker, CanvasFormat, CanvasId};
use f3d_graph::error::{CanvasError, GraphError, HazardError};
use f3d_graph::hazard::{can_split_pass, split_pass_on_hazard, validate_pass_hazards};
use f3d_graph::pass::{
    ColorAttachment, CopyCommand, DepthStencilAttachment, Dispatch, Draw, LoadOp, Pass, PassId,
    PassKind, StoreOp,
};
use f3d_graph::plan::{
    build_red_a_blue_b_plan, build_red_a_blue_b_render_plan, build_single_pass_bridge_plan,
    build_two_target_bridge_plan,
};
use f3d_graph::resource::{ResourceAccess, ResourceId, ResourceUse, SubresourceRange};
use f3d_graph::schedule::PassGraph;

#[test]
fn positive_legal_read_only_combinations_execute() {
    let mut graph = PassGraph::new();
    let p1_id = PassId::new(1);
    let mut p1 = Pass::new_render(p1_id, "render_scene");

    let target_tex = ResourceId::new(10);
    p1.color_attachments.push(ColorAttachment::new_clear(
        target_tex,
        [0.1, 0.2, 0.3, 1.0],
    ));

    let buf_uniform = ResourceId::new(20);
    let buf_vertex = ResourceId::new(21);
    let tex_sample = ResourceId::new(30);

    // Draw with multiple legal read-only uses
    let draw = Draw::new(
        0,
        100,
        3,
        0,
        vec![
            ResourceUse::buffer_uniform(buf_uniform, DataVersion::INITIAL, Some(0), Some(256)),
            ResourceUse::buffer_vertex(buf_vertex, DataVersion::INITIAL, Some(0), Some(1024)),
            ResourceUse::texture_sampled(
                tex_sample,
                DataVersion::INITIAL,
                SubresourceRange::full_texture(),
            ),
        ],
    );
    p1.draws.push(draw);
    graph.add_pass(p1).expect("add pass");

    let plan = graph.compile(None).expect("legal reads must compile");
    assert_eq!(plan.pass_count, 1);
    assert_eq!(plan.segments.len(), 1);
    assert_eq!(plan.split_count, 0);
}

#[test]
fn negative_whole_buffer_rule_disjoint_offsets_cannot_legalize_conflict() {
    // Invariant (§8.5, [S51]): In WebGPU, a buffer is a whole subresource.
    // Assigning disjoint byte ranges incompatible roles does NOT make the shared buffer legal!
    let shared_buffer = ResourceId::new(50);
    let mut pass = Pass::new_render(PassId::new(1), "illegal_shared_buffer_pass");

    // Draw 0: Uniform read at offset 0..256
    let draw0 = Draw::new(
        0,
        1,
        3,
        0,
        vec![ResourceUse::buffer_uniform(
            shared_buffer,
            DataVersion::INITIAL,
            Some(0),
            Some(256),
        )],
    );

    // Draw 1: Storage buffer write at disjoint offset 256..512 in the SAME render pass
    let draw1 = Draw::new(
        1,
        1,
        3,
        256,
        vec![ResourceUse::buffer_storage_write(
            shared_buffer,
            DataVersion::INITIAL,
            Some(256),
            Some(256),
        )],
    );

    pass.draws.push(draw0);
    pass.draws.push(draw1);

    let err = validate_pass_hazards(&pass).expect_err("disjoint byte offsets cannot legalize conflict");
    match err {
        HazardError::WholeBufferConflict {
            buffer_id,
            access_a,
            access_b,
            offset_a,
            offset_b,
        } => {
            assert_eq!(buffer_id, shared_buffer.get());
            assert_eq!(access_a, ResourceAccess::UniformBuffer);
            assert_eq!(access_b, ResourceAccess::StorageBufferWrite);
            assert_eq!(offset_a, Some(0));
            assert_eq!(offset_b, Some(256));
        }
        other => panic!("expected WholeBufferConflict, got: {other:?}"),
    }
}

#[test]
fn negative_unsupported_feedback_loop_rejected_without_dropping_outputs() {
    // Root Review Defect (1) Counterexample:
    // When a render pass writes to attachment TexA and a later draw samples TexA
    // with NO separate legal destination, it is an unsupported feedback loop.
    // The graph must REJECT the hazard, NOT silently drop TexA or clear outputs to "legalize" it.
    let tex_a = ResourceId::new(100);
    let buf_v = ResourceId::new(102);

    let mut pass = Pass::new_render(PassId::new(1), "feedback_loop_pass");
    pass.color_attachments.push(ColorAttachment::new_clear(
        tex_a,
        [0.0, 0.0, 0.0, 1.0],
    ));

    // Draw 0: Writes to attachment tex_a
    pass.draws.push(Draw::new(
        0,
        1,
        3,
        0,
        vec![ResourceUse::buffer_vertex(
            buf_v,
            DataVersion::INITIAL,
            Some(0),
            Some(64),
        )],
    ));

    // Draw 1: Samples tex_a while rendering to tex_a (only attachment)
    pass.draws.push(Draw::new(
        1,
        2,
        3,
        0,
        vec![
            ResourceUse::buffer_vertex(buf_v, DataVersion::INITIAL, Some(0), Some(64)),
            ResourceUse::texture_sampled(
                tex_a,
                DataVersion::INITIAL,
                SubresourceRange::full_texture(),
            ),
        ],
    ));

    // Must be recognized as an unsupported split: cannot drop the sole destination!
    assert!(!can_split_pass(&pass));

    let split_err = split_pass_on_hazard(&pass, PassId::new(2)).expect_err("must reject feedback loop");
    match split_err {
        HazardError::AttachmentSamplingConflict { texture_id, .. } => {
            assert_eq!(texture_id, tex_a.get());
        }
        other => panic!("expected AttachmentSamplingConflict, got: {other:?}"),
    }

    let mut graph = PassGraph::new();
    graph.add_pass(pass).expect("add pass");
    let compile_err = graph.compile(None).expect_err("compilation must reject unsupported feedback");
    match compile_err {
        GraphError::Hazard(HazardError::AttachmentSamplingConflict { texture_id, .. }) => {
            assert_eq!(texture_id, tex_a.get());
        }
        other => panic!("expected GraphError::Hazard(AttachmentSamplingConflict), got: {other:?}"),
    }
}

#[test]
fn positive_automatic_pass_splitting_preserves_targets_and_depth_stencil() {
    // Root Review Defect (1) Counterexample / Positive:
    // A legal multi-target pass where Draw 0 writes intermediate TexA and Draw 1
    // samples TexA while writing to destination TexB, with active Depth/Stencil.
    // Splitting into Pass1 and Pass2 MUST:
    // 1. Store TexA and TexB in Pass1 (`StoreOp::Store`).
    // 2. Preserve Depth/Stencil in Pass1 with `depth_store_op: Some(StoreOp::Store)`.
    // 3. Preserve TexB in Pass2 with `LoadOp::Load` and `StoreOp::Store`.
    // 4. Preserve Depth/Stencil in Pass2 with `depth_load_op: Some(LoadOp::Load)` and `depth_store_op: Some(StoreOp::Store)`.
    // 5. NOT drop depth/stencil or destinations!
    let tex_intermediate = ResourceId::new(100);
    let tex_destination = ResourceId::new(101);
    let tex_depth = ResourceId::new(102);
    let buf_v = ResourceId::new(103);

    let mut pass = Pass::new_render(PassId::new(1), "multipass_with_depth");
    pass.color_attachments.push(ColorAttachment::new_clear(
        tex_intermediate,
        [0.0, 0.0, 0.0, 1.0],
    ));
    pass.color_attachments.push(ColorAttachment::new_clear(
        tex_destination,
        [0.0, 0.0, 0.0, 1.0],
    ));
    pass.depth_stencil_attachment = Some(DepthStencilAttachment::new_depth_clear(tex_depth, 1.0));

    // Draw 0: Writes to intermediate tex_intermediate
    pass.draws.push(Draw::new(
        0,
        1,
        3,
        0,
        vec![ResourceUse::buffer_vertex(
            buf_v,
            DataVersion::INITIAL,
            Some(0),
            Some(64),
        )],
    ));

    // Draw 1: Samples intermediate tex_intermediate while writing to tex_destination
    pass.draws.push(Draw::new(
        1,
        2,
        3,
        0,
        vec![
            ResourceUse::buffer_vertex(buf_v, DataVersion::INITIAL, Some(0), Some(64)),
            ResourceUse::texture_sampled(
                tex_intermediate,
                DataVersion::INITIAL,
                SubresourceRange::full_texture(),
            ),
        ],
    ));

    assert!(can_split_pass(&pass));

    let mut graph = PassGraph::new();
    graph.add_pass(pass).expect("add pass");
    let plan = graph.compile(None).expect("split pass must compile successfully");

    assert_eq!(plan.split_count, 1);
    assert_eq!(plan.pass_count, 2);
    assert_eq!(plan.segments.len(), 2);

    // Segment 1 (Pass 1): stores all attachments and depth/stencil
    let seg1 = &plan.segments[0];
    assert_eq!(seg1.draws.len(), 1);
    assert_eq!(seg1.draws[0].draw_id, 0);
    assert!(seg1.color_attachments.iter().any(|ca| ca.target_id == tex_intermediate && ca.store_op == StoreOp::Store));
    assert!(seg1.color_attachments.iter().any(|ca| ca.target_id == tex_destination && ca.store_op == StoreOp::Store));
    let dsa1 = seg1.depth_stencil_attachment.as_ref().expect("segment 1 depth attachment");
    assert_eq!(dsa1.target_id, tex_depth);
    assert_eq!(dsa1.depth_store_op, Some(StoreOp::Store));

    // Segment 2 (Pass 2): preserves destination TexB and depth with LoadOp::Load!
    let seg2 = &plan.segments[1];
    assert_eq!(seg2.draws.len(), 1);
    assert_eq!(seg2.draws[0].draw_id, 1);
    // Intermediate TexA is not bound as attachment in pass 2 (it is sampled)
    assert!(!seg2.color_attachments.iter().any(|ca| ca.target_id == tex_intermediate));
    // Destination TexB is PRESERVED with LoadOp::Load and StoreOp::Store
    let ca2_dest = seg2
        .color_attachments
        .iter()
        .find(|ca| ca.target_id == tex_destination)
        .expect("tex_destination must be preserved in pass 2");
    assert_eq!(ca2_dest.load_op, LoadOp::Load);
    assert_eq!(ca2_dest.store_op, StoreOp::Store);

    // Depth/stencil is PRESERVED in pass 2 with LoadOp::Load and StoreOp::Store
    let dsa2 = seg2.depth_stencil_attachment.as_ref().expect("depth must be preserved in pass 2");
    assert_eq!(dsa2.target_id, tex_depth);
    assert_eq!(dsa2.depth_load_op, Some(LoadOp::Load));
    assert_eq!(dsa2.depth_store_op, Some(StoreOp::Store));
}

#[test]
fn test_split_rewires_downstream_consumers_to_final_segment() {
    // Root Review Defect (4) Counterexample:
    // If Pass 1 is split into Part1 (id 1) and Part2 (id 11), a downstream consumer
    // Pass 2 with dependency on Pass 1 must be rewired to depend on Part2 (id 11),
    // NOT run prematurely after Part1.
    let tex_intermediate = ResourceId::new(200);
    let tex_destination = ResourceId::new(201);
    let buf_v = ResourceId::new(202);

    let mut p1 = Pass::new_render(PassId::new(1), "splittable_pass");
    p1.color_attachments.push(ColorAttachment::new_clear(
        tex_intermediate,
        [0.0, 0.0, 0.0, 1.0],
    ));
    p1.color_attachments.push(ColorAttachment::new_clear(
        tex_destination,
        [0.0, 0.0, 0.0, 1.0],
    ));
    p1.draws.push(Draw::new(
        0,
        1,
        3,
        0,
        vec![ResourceUse::buffer_vertex(
            buf_v,
            DataVersion::INITIAL,
            Some(0),
            Some(64),
        )],
    ));
    p1.draws.push(Draw::new(
        1,
        2,
        3,
        0,
        vec![
            ResourceUse::buffer_vertex(buf_v, DataVersion::INITIAL, Some(0), Some(64)),
            ResourceUse::texture_sampled(
                tex_intermediate,
                DataVersion::INITIAL,
                SubresourceRange::full_texture(),
            ),
        ],
    ));

    // Downstream consumer Pass 2 depends on Pass 1
    let mut p2 = Pass::new_render(PassId::new(2), "downstream_consumer");
    p2.dependencies.push(PassId::new(1));
    p2.color_attachments.push(ColorAttachment::new_clear(
        ResourceId::new(205),
        [0.0, 1.0, 0.0, 1.0],
    ));
    p2.draws.push(Draw::new(
        0,
        3,
        3,
        0,
        vec![ResourceUse::buffer_vertex(
            buf_v,
            DataVersion::INITIAL,
            Some(0),
            Some(64),
        )],
    ));

    let mut graph = PassGraph::new();
    graph.add_pass(p1).expect("add p1");
    graph.add_pass(p2).expect("add p2");

    let plan = graph.compile(None).expect("compile with rewired consumer");
    assert_eq!(plan.segments.len(), 3);

    // Topological order must be: Part1 -> Part2 -> DownstreamConsumer
    assert_eq!(plan.segments[0].name, "splittable_pass_part1");
    assert_eq!(plan.segments[1].name, "splittable_pass_part2");
    assert_eq!(plan.segments[2].name, "downstream_consumer");
}

#[test]
fn negative_duplicate_pass_id_rejected_before_mutation() {
    // Root Review Defect (3) Counterexample:
    // Adding a pass with a duplicate PassId must be rejected BEFORE mutating graph state,
    // preventing map overwrite and false cycle reports.
    let mut graph = PassGraph::new();
    let p1 = Pass::new_render(PassId::new(42), "first_pass");
    let p2 = Pass::new_render(PassId::new(42), "duplicate_pass");

    assert!(graph.add_pass(p1).is_ok());
    let err = graph.add_pass(p2).expect_err("duplicate PassId must be rejected");

    assert_eq!(err, GraphError::DuplicatePassId { pass_id: 42 });

    // Graph still compiles cleanly with the single valid pass
    let plan = graph.compile(None).expect("graph must compile without false cycle");
    assert_eq!(plan.segments.len(), 1);
    assert_eq!(plan.segments[0].pass_id, PassId::new(42));
}

#[test]
fn negative_pass_id_overflow_rejected() {
    // Root Review Defect (3) Counterexample:
    // Saturation at u32::MAX is rejected with checked overflow error.
    let mut graph = PassGraph::new();
    let p_max = Pass::new_render(PassId::new(u32::MAX), "max_id_pass");
    let err = graph.add_pass(p_max).expect_err("u32::MAX pass id next_id overflow must be rejected");
    assert_eq!(err, GraphError::PassIdOverflow);
}

#[test]
fn test_canvas_stale_captured_epoch_rejected_on_tracker_advance() {
    // Root Review Defect (2) Counterexample:
    // Validates that canvas output epoch captured at plan creation fails when the tracker
    // advances to a new epoch (e.g. frame 2), preventing stale frame reuse.
    // Also tests:
    // - Positive: fresh e2 plan compiles and validates.
    // - Negative: missing tracker on canvas use cannot bypass guard.
    let canvas_id = CanvasId::new(1);
    let canvas_res = ResourceId::new(300);
    let vbuf = ResourceId::new(301);

    let mut tracker = CanvasEpochTracker::new();
    tracker.register_canvas(canvas_id, canvas_res, 800, 600, CanvasFormat::Bgra8Unorm);

    // Frame 1: Acquire swapchain texture with epoch e1
    let out1 = tracker.begin_frame_acquire(canvas_id).expect("frame 1 acquire");
    let epoch_e1 = out1.epoch;

    // Plan 1 captures epoch e1
    let plan1 = build_single_pass_bridge_plan(
        canvas_res,
        10,
        vbuf,
        3,
        DataVersion::INITIAL,
        Some(epoch_e1),
        Some(&tracker),
    )
    .expect("plan 1 compiles for epoch e1");

    // Execution validation succeeds in frame 1 interval
    assert!(plan1.validate_execution(&tracker).is_ok());

    // Submit frame 1 and advance to Frame 2 (epoch e2)
    tracker.submit_frame(canvas_id, epoch_e1).expect("submit frame 1");
    let out2 = tracker.begin_frame_acquire(canvas_id).expect("frame 2 acquire");
    let epoch_e2 = out2.epoch;
    assert_ne!(epoch_e1, epoch_e2);

    // 1. Validating the old Plan 1 against advanced tracker MUST FAIL with CanvasCachedAcrossEpochs
    let exec_err = plan1.validate_execution(&tracker).expect_err("stale plan1 must fail against frame 2");
    match exec_err {
        GraphError::Canvas(CanvasError::CanvasCachedAcrossEpochs {
            canvas_id: cid,
            cached_epoch,
            current_epoch,
        }) => {
            assert_eq!(cid, canvas_id.get());
            assert_eq!(cached_epoch, epoch_e1.get());
            assert_eq!(current_epoch, epoch_e2.get());
        }
        other => panic!("expected CanvasCachedAcrossEpochs, got: {other:?}"),
    }

    // 2. Attempting to compile a new plan with the stale epoch_e1 against tracker MUST FAIL
    let compile_stale_err = build_single_pass_bridge_plan(
        canvas_res,
        10,
        vbuf,
        3,
        DataVersion::INITIAL,
        Some(epoch_e1),
        Some(&tracker),
    )
    .expect_err("compiling with stale epoch against tracker must fail");
    match compile_stale_err {
        GraphError::Canvas(CanvasError::CanvasCachedAcrossEpochs { .. }) => {}
        other => panic!("expected CanvasCachedAcrossEpochs, got: {other:?}"),
    }

    // 3. Positive: Fresh Plan 2 built with epoch e2 compiles and validates
    let plan2 = build_single_pass_bridge_plan(
        canvas_res,
        10,
        vbuf,
        3,
        DataVersion::INITIAL,
        Some(epoch_e2),
        Some(&tracker),
    )
    .expect("plan 2 compiles for fresh epoch e2");
    assert!(plan2.validate_execution(&tracker).is_ok());

    // 4. Negative: Missing tracker on canvas use CANNOT bypass guard
    let missing_tracker_err = build_single_pass_bridge_plan(
        canvas_res,
        10,
        vbuf,
        3,
        DataVersion::INITIAL,
        Some(epoch_e2),
        None,
    )
    .expect_err("missing tracker on canvas use must fail");
    assert_eq!(
        missing_tracker_err,
        GraphError::Canvas(CanvasError::CanvasNotAcquired {
            canvas_id: canvas_res.get()
        })
    );
}

#[test]
fn test_build_two_target_bridge_plan_rowpitch_alignment_width_32() {
    // Root Review Defect (5) Counterexample & RusticRobin #5619:
    // WebGPU requires texture-to-buffer copies to have bytesPerRow aligned to 256 bytes
    // (COPY_BYTES_PER_ROW_ALIGNMENT). For width 32 with 4 bytes/pixel:
    // unpadded = 32 * 4 = 128 bytes (INVALID).
    // Core helper and plan MUST yield 256 bytes!
    assert_eq!(aligned_bytes_per_row(32), Ok(256));

    let offscreen = ResourceId::new(400);
    let canvas = ResourceId::new(401);
    let readback = ResourceId::new(402);
    let vbuf = ResourceId::new(403);

    let plan = build_two_target_bridge_plan(
        offscreen,
        canvas,
        readback,
        10,
        11,
        vbuf,
        3,
        32, // width 32 gives 256 through the plan
        32, // height 32
        DataVersion::INITIAL,
        None,
        None,
    )
    .expect("two-target plan for width 32 must compile");

    assert_eq!(plan.segments.len(), 3);
    let copy_seg = &plan.segments[2];
    assert_eq!(copy_seg.kind, PassKind::Copy);
    match &copy_seg.copies[0] {
        CopyCommand::TextureToBuffer {
            bytes_per_row,
            width,
            height,
            ..
        } => {
            assert_eq!(*width, 32);
            assert_eq!(*height, 32);
            // Must be aligned up from 128 to 256!
            assert_eq!(*bytes_per_row, 256);
            assert_eq!(*bytes_per_row % (COPY_BYTES_PER_ROW_ALIGNMENT as u32), 0);
        }
        other => panic!("expected TextureToBuffer copy command, got: {other:?}"),
    }
}

#[test]
fn test_build_two_target_bridge_plan_width_overflow_returns_err() {
    // RusticRobin #5619 Regression:
    // build_two_target_bridge_plan with width u32::MAX must return Err(GraphError::InvalidPass)
    let offscreen = ResourceId::new(400);
    let canvas = ResourceId::new(401);
    let readback = ResourceId::new(402);
    let vbuf = ResourceId::new(403);

    let err = build_two_target_bridge_plan(
        offscreen,
        canvas,
        readback,
        10,
        11,
        vbuf,
        3,
        u32::MAX, // overflow width
        32,
        DataVersion::INITIAL,
        None,
        None,
    )
    .expect_err("width u32::MAX must return Err");

    match err {
        GraphError::InvalidPass { pass_id, reason } => {
            assert_eq!(pass_id, 3);
            assert!(reason.contains("bytes_per_row") || reason.contains("overflow"));
        }
        other => panic!("expected InvalidPass error on overflow, got: {other:?}"),
    }
}

#[test]
fn test_consumer_api_execution_plan_accessors_and_bridge_lowering() {
    // Verifies the stable consumer API for ChartreuseFern's gpu_host.rs:
    // Accessors on ExecutionPlan, PlanSegment, ColorAttachment, Draw, and CopyCommand.
    let canvas_id = CanvasId::new(1);
    let canvas_res = ResourceId::new(501);
    let mut tracker = CanvasEpochTracker::new();
    tracker.register_canvas(canvas_id, canvas_res, 800, 600, CanvasFormat::Bgra8Unorm);
    let out = tracker.begin_frame_acquire(canvas_id).expect("frame acquire");

    let offscreen = ResourceId::new(500);
    let readback = ResourceId::new(502);
    let vbuf = ResourceId::new(503);

    let plan = build_two_target_bridge_plan(
        offscreen,
        canvas_res,
        readback,
        100,
        101,
        vbuf,
        3,
        32, // width 32 gives 256 bytes_per_row
        32,
        DataVersion::INITIAL,
        Some(out.epoch),
        Some(&tracker),
    )
    .expect("compile two-target plan with canvas");

    // 1. ExecutionPlan accessors
    assert_eq!(plan.segment_count(), 3);
    assert_eq!(plan.pass_count(), 3);
    assert_eq!(plan.split_count(), 0);
    assert_eq!(plan.canvas_epoch(), Some(out.epoch));
    assert!(!plan.is_empty());
    assert_eq!(plan.segments().len(), 3);

    // 2. PlanSegment 0 (Offscreen Render Pass)
    let seg0 = &plan.segments()[0];
    assert_eq!(seg0.kind(), PassKind::Render);
    assert!(seg0.is_render());
    assert!(!seg0.is_copy());
    assert!(!seg0.is_compute());
    assert_eq!(seg0.name(), "bridge_offscreen_pass");
    assert_eq!(seg0.color_attachments().len(), 1);

    let ca0 = seg0.primary_color_attachment().expect("primary attachment");
    assert_eq!(ca0.target_id(), offscreen);
    assert_eq!(ca0.load_op(), LoadOp::Clear);
    assert_eq!(ca0.store_op(), StoreOp::Store);
    assert_eq!(ca0.clear_color(), [0.0, 0.0, 0.0, 1.0]);
    assert!(!ca0.is_canvas());
    assert_eq!(ca0.canvas_epoch(), None);

    assert_eq!(seg0.draw_count(), 1);
    let draw0 = seg0.first_draw().expect("first draw");
    assert_eq!(draw0.pipeline_id(), 100);
    assert_eq!(draw0.vertex_count(), 3);
    assert_eq!(draw0.uniform_dynamic_offset(), 0);
    assert_eq!(draw0.vertex_buffer(), Some(vbuf));
    assert_eq!(draw0.vertex_buffer_id(), vbuf.get());

    // 3. PlanSegment 1 (Canvas Render Pass)
    let seg1 = &plan.segments()[1];
    assert!(seg1.is_render());
    let ca1 = seg1.primary_color_attachment().expect("canvas attachment");
    assert_eq!(ca1.target_id(), canvas_res);
    assert!(ca1.is_canvas());
    assert_eq!(ca1.canvas_epoch(), Some(out.epoch));
    let draw1 = seg1.first_draw().expect("canvas draw");
    assert_eq!(draw1.pipeline_id(), 101);
    assert_eq!(draw1.vertex_buffer_id(), vbuf.get());

    // 4. PlanSegment 2 (Copy Pass with 256-byte aligned pitch)
    let seg2 = &plan.segments()[2];
    assert!(seg2.is_copy());
    assert_eq!(seg2.copy_count(), 1);
    let copy = seg2.first_copy().expect("copy command");
    assert!(copy.is_texture_to_buffer());
    assert_eq!(copy.bytes_per_row(), Some(256));

    let (src_tex, dst_buf, w, h, pitch) = copy.as_texture_to_buffer().expect("deconstruct copy");
    assert_eq!(src_tex, offscreen);
    assert_eq!(dst_buf, readback);
    assert_eq!(w, 32);
    assert_eq!(h, 32);
    assert_eq!(pitch, 256);
}

#[test]
fn negative_compute_per_dispatch_writable_alias_rejected() {
    let buf_storage = ResourceId::new(70);
    let mut pass = Pass::new_compute(PassId::new(1), "compute_hazard_pass");

    // Single dispatch attempting to bind the same storage buffer twice as writable
    let dispatch = Dispatch::new(
        0,
        500,
        [64, 1, 1],
        vec![
            ResourceUse::buffer_storage_write(
                buf_storage,
                DataVersion::INITIAL,
                Some(0),
                Some(1024),
            ),
            ResourceUse::buffer_storage_write(
                buf_storage,
                DataVersion::INITIAL,
                Some(1024),
                Some(1024),
            ),
        ],
    );
    pass.dispatches.push(dispatch);

    let err = validate_pass_hazards(&pass).expect_err("writable compute aliases must be rejected");
    match err {
        HazardError::ComputeWritableAlias {
            resource_id,
            dispatch_id,
            ..
        } => {
            assert_eq!(resource_id, buf_storage.get());
            assert_eq!(dispatch_id, 0);
        }
        other => panic!("expected ComputeWritableAlias, got: {other:?}"),
    }
}

#[test]
fn negative_dependency_cycle_rejected() {
    let mut graph = PassGraph::new();

    let p1 = Pass::new_render(PassId::new(1), "pass1");
    let p2 = Pass::new_render(PassId::new(2), "pass2");
    let p3 = Pass::new_render(PassId::new(3), "pass3");

    graph.add_pass(p1).unwrap();
    graph.add_pass(p2).unwrap();
    graph.add_pass(p3).unwrap();

    // Cycle: 1 -> 2 -> 3 -> 1
    graph.add_dependency(PassId::new(2), PassId::new(1)).unwrap();
    graph.add_dependency(PassId::new(3), PassId::new(2)).unwrap();
    graph.add_dependency(PassId::new(1), PassId::new(3)).unwrap();

    let err = graph.compile(None).expect_err("cyclic graph must be rejected");
    match err {
        GraphError::CycleDetected { cycle } => {
            assert!(!cycle.is_empty());
        }
        other => panic!("expected CycleDetected, got: {other:?}"),
    }
}

#[test]
fn negative_canvas_texture_cached_across_output_epochs_fails() {
    let canvas_id = CanvasId::new(1);
    let canvas_res = ResourceId::new(999);
    let mut tracker = CanvasEpochTracker::new();
    tracker.register_canvas(canvas_id, canvas_res, 800, 600, CanvasFormat::Bgra8Unorm);

    // Frame 1: Acquire and get epoch 1
    let frame1_output = tracker.begin_frame_acquire(canvas_id).expect("frame 1 acquire");
    let cached_epoch = frame1_output.epoch;
    tracker.submit_frame(canvas_id, cached_epoch).expect("frame 1 submit");

    // Frame 2: Acquire next interval
    let frame2_output = tracker.begin_frame_acquire(canvas_id).expect("frame 2 acquire");
    assert_ne!(cached_epoch, frame2_output.epoch);

    // Attempting to validate or use the cached texture from Frame 1 in Frame 2 fails
    let err = tracker
        .validate_canvas_access(canvas_id, cached_epoch)
        .expect_err("cached canvas texture across epochs must fail");

    match err {
        CanvasError::CanvasCachedAcrossEpochs {
            canvas_id: id,
            cached_epoch: c,
            current_epoch: cur,
        } => {
            assert_eq!(id, canvas_id.get());
            assert_eq!(c, cached_epoch.get());
            assert_eq!(cur, frame2_output.epoch.get());
        }
        other => panic!("expected CanvasCachedAcrossEpochs, got: {other:?}"),
    }
}

#[test]
fn canvas_lifecycle_acquisition_interval_and_zero_size_policy() {
    let canvas_id = CanvasId::new(2);
    let canvas_res = ResourceId::new(998);
    let mut tracker = CanvasEpochTracker::new();

    // 1. Unregistered/unacquired access fails
    let err = tracker
        .validate_canvas_access(canvas_id, Epoch::new(1))
        .expect_err("unacquired canvas fails");
    assert_eq!(err, CanvasError::CanvasNotAcquired { canvas_id: 2 });

    // 2. Zero-sized canvas (0x0) policy: pauses without allocating illegal textures
    tracker.register_canvas(canvas_id, canvas_res, 0, 0, CanvasFormat::Bgra8Unorm);
    let err = tracker
        .begin_frame_acquire(canvas_id)
        .expect_err("0x0 canvas must pause");
    assert_eq!(err, CanvasError::ZeroSizedCanvasPause { canvas_id: 2 });

    // 3. Resize to valid dimensions and acquire
    tracker.resize(canvas_id, 1024, 768).expect("resize ok");
    let output = tracker.begin_frame_acquire(canvas_id).expect("acquire ok");
    assert_eq!(output.width, 1024);
    assert_eq!(output.height, 768);

    // 4. Submit then access again fails with CanvasAlreadySubmitted
    tracker.submit_frame(canvas_id, output.epoch).expect("submit ok");
    let err = tracker
        .validate_canvas_access(canvas_id, output.epoch)
        .expect_err("submitted canvas cannot be reused");
    assert_eq!(
        err,
        CanvasError::CanvasAlreadySubmitted {
            canvas_id: 2,
            epoch: output.epoch.get()
        }
    );
}

#[test]
fn multiple_canvases_maintain_independent_epochs() {
    let mut tracker = CanvasEpochTracker::new();
    let c1 = CanvasId::new(10);
    let c2 = CanvasId::new(20);

    tracker.register_canvas(c1, ResourceId::new(1), 640, 480, CanvasFormat::Bgra8Unorm);
    tracker.register_canvas(c2, ResourceId::new(2), 1920, 1080, CanvasFormat::Rgba8Unorm);

    let out1 = tracker.begin_frame_acquire(c1).unwrap();
    let out2 = tracker.begin_frame_acquire(c2).unwrap();

    // Submit c1 only
    tracker.submit_frame(c1, out1.epoch).unwrap();

    // c1 is submitted, c2 is still valid in its interval
    assert!(tracker.validate_canvas_access(c1, out1.epoch).is_err());
    assert!(tracker.validate_canvas_access(c2, out2.epoch).is_ok());

    // Advancing c1 through resize does not change c2's epoch
    tracker.resize(c1, 800, 600).unwrap();
    assert_eq!(tracker.epoch_of(c2).unwrap(), out2.epoch);
}

#[test]
fn canvas_epoch_overflow_at_u64_max_returns_epoch_overflow_error() {
    let canvas_id = CanvasId::new(42);
    let resource_id = ResourceId::new(100);
    let mut tracker = CanvasEpochTracker::new();
    tracker.register_canvas(canvas_id, resource_id, 800, 600, CanvasFormat::Bgra8Unorm);

    // Seed the tracker at u64::MAX through the test-only seam
    tracker.seed_epoch_for_test(canvas_id, Epoch::new(u64::MAX));

    // 1. begin_frame_acquire attempts monotonic increment beyond u64::MAX and fails with EpochOverflow
    let err = tracker
        .begin_frame_acquire(canvas_id)
        .expect_err("monotonic epoch increment beyond u64::MAX must fail with EpochOverflow");
    assert_eq!(
        err,
        CanvasError::EpochOverflow {
            canvas_id: 42,
            current: u64::MAX,
        }
    );

    // 2. resize also attempts monotonic increment beyond u64::MAX and fails with EpochOverflow
    let err_resize = tracker
        .resize(canvas_id, 1024, 768)
        .expect_err("resize epoch increment beyond u64::MAX must fail with EpochOverflow");
    assert_eq!(
        err_resize,
        CanvasError::EpochOverflow {
            canvas_id: 42,
            current: u64::MAX,
        }
    );
}

#[test]
fn bridge_single_pass_plan_matches_expected_contract() {
    let canvas_res = ResourceId::new(100);
    let vbuf = ResourceId::new(101);
    let plan = build_single_pass_bridge_plan(canvas_res, 10, vbuf, 3, DataVersion::INITIAL, None, None)
        .expect("single pass bridge plan compiles");

    assert_eq!(plan.segment_count(), 1);
    let seg = &plan.segments()[0];
    assert_eq!(seg.kind, PassKind::Render);
    assert_eq!(seg.color_attachments.len(), 1);
    assert_eq!(seg.color_attachments[0].target_id, canvas_res);
    assert_eq!(seg.draws.len(), 1);
    assert_eq!(seg.draws[0].vertex_count, 3);
}

#[test]
fn bridge_two_target_plan_structure_and_copy_isolation() {
    let offscreen = ResourceId::new(10);
    let canvas = ResourceId::new(11);
    let readback = ResourceId::new(12);
    let vbuf = ResourceId::new(13);

    let plan = build_two_target_bridge_plan(
        offscreen,
        canvas,
        readback,
        100,
        101,
        vbuf,
        3,
        64,
        64,
        DataVersion::INITIAL,
        None,
        None,
    )
    .expect("two target bridge plan compiles");

    assert_eq!(plan.segment_count(), 3);
    assert_eq!(plan.segments()[0].kind, PassKind::Render);
    assert_eq!(plan.segments()[0].color_attachments[0].target_id, offscreen);

    assert_eq!(plan.segments()[1].kind, PassKind::Render);
    assert_eq!(plan.segments()[1].color_attachments[0].target_id, canvas);

    // Segment 3 is the isolated copy pass
    assert_eq!(plan.segments()[2].kind, PassKind::Copy);
    assert_eq!(plan.segments()[2].copies.len(), 1);
    match &plan.segments()[2].copies[0] {
        CopyCommand::TextureToBuffer {
            texture_id,
            buffer_id,
            width,
            height,
            bytes_per_row,
        } => {
            assert_eq!(*texture_id, offscreen);
            assert_eq!(*buffer_id, readback);
            assert_eq!(*width, 64);
            assert_eq!(*height, 64);
            assert_eq!(*bytes_per_row, 256);
        }
        other => panic!("expected TextureToBuffer copy command, got: {other:?}"),
    }
}

#[test]
fn red_a_blue_b_plan_preserves_per_use_versions_without_overwrite() {
    let shared_buf = ResourceId::new(50);
    let target_a = ResourceId::new(60);
    let target_b = ResourceId::new(61);

    let red_version = DataVersion::INITIAL;
    let blue_version = red_version.checked_next().unwrap();

    // GrayFox #5657: Allocate real per-use versioned slices via PerUseByteBuffer
    // instead of hard-coded literals, exercising the core ownership allocation contract.
    let mat_handle = Handle::<MaterialDomain>::new(50, NonZeroU32::new(1).unwrap());
    let mut byte_buf = PerUseByteBuffer::<MaterialDomain>::new(256)
        .expect("byte buffer with 256-byte alignment");

    let red_bytes: [u8; 16] = [255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let blue_bytes: [u8; 16] = [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    let rec_red = byte_buf
        .append_slice(mat_handle, red_version, Epoch::ZERO, &red_bytes)
        .expect("append red slice");
    let rec_blue = byte_buf
        .append_slice(mat_handle, blue_version, Epoch::ZERO, &blue_bytes)
        .expect("append blue slice");

    let red_offset = rec_red.byte_offset() as u32;
    let blue_offset = rec_blue.byte_offset() as u32;

    assert_eq!(red_offset, 0);
    assert_eq!(blue_offset, 256);

    let readback_a = ResourceId::new(70);
    let readback_b = ResourceId::new(71);

    let plan = build_red_a_blue_b_plan(
        shared_buf,
        target_a,
        target_b,
        readback_a,
        readback_b,
        red_version,
        blue_version,
        red_offset,
        blue_offset,
        1,
    )
    .expect("red-a blue-b plan compiles");

    assert_eq!(plan.segment_count(), 4);

    // Pass 1: Red at dynamic offset from rec_red, red_version, standardized black clear
    let seg1 = &plan.segments()[0];
    assert_eq!(seg1.kind(), PassKind::Render);
    assert_eq!(
        seg1.primary_color_attachment().unwrap().clear_color(),
        [0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(seg1.draws()[0].uniform_dynamic_offset(), red_offset);
    assert_eq!(seg1.versioned_reads()[0], (shared_buf, red_version));

    // Pass 2: Blue at dynamic offset from rec_blue, blue_version, standardized black clear
    let seg2 = &plan.segments()[1];
    assert_eq!(seg2.kind(), PassKind::Render);
    assert_eq!(
        seg2.primary_color_attachment().unwrap().clear_color(),
        [0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(seg2.draws()[0].uniform_dynamic_offset(), blue_offset);
    assert_eq!(seg2.versioned_reads()[0], (shared_buf, blue_version));

    // Pass 3: Copy texture A to readback buffer A
    let seg3 = &plan.segments()[2];
    assert_eq!(seg3.kind(), PassKind::Copy);
    match seg3.copies()[0] {
        CopyCommand::TextureToBuffer {
            texture_id,
            buffer_id,
            width,
            height,
            bytes_per_row,
        } => {
            assert_eq!(texture_id, target_a);
            assert_eq!(buffer_id, readback_a);
            assert_eq!(width, 64);
            assert_eq!(height, 64);
            assert_eq!(bytes_per_row, 256);
        }
        ref other => panic!("expected TextureToBuffer copy command, got {other:?}"),
    }

    // Pass 4: Copy texture B to readback buffer B
    let seg4 = &plan.segments()[3];
    assert_eq!(seg4.kind(), PassKind::Copy);
    match seg4.copies()[0] {
        CopyCommand::TextureToBuffer {
            texture_id,
            buffer_id,
            width,
            height,
            bytes_per_row,
        } => {
            assert_eq!(texture_id, target_b);
            assert_eq!(buffer_id, readback_b);
            assert_eq!(width, 64);
            assert_eq!(height, 64);
            assert_eq!(bytes_per_row, 256);
        }
        ref other => panic!("expected TextureToBuffer copy command, got {other:?}"),
    }

    // Verify render-only constructor also produces valid segments with black clear
    let render_plan = build_red_a_blue_b_render_plan(
        shared_buf,
        target_a,
        target_b,
        red_version,
        blue_version,
        red_offset,
        blue_offset,
        1,
    )
    .expect("render-only plan compiles");
    assert_eq!(render_plan.segment_count(), 2);
    assert_eq!(
        render_plan.segments()[0]
            .primary_color_attachment()
            .unwrap()
            .clear_color(),
        [0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn copy_in_render_pass_and_overlapping_copy_endpoints_rejected() {
    let buf_a = ResourceId::new(1);
    let mut pass = Pass::new_render(PassId::new(1), "render_with_copy");
    pass.copies.push(CopyCommand::BufferToBuffer {
        src: buf_a,
        src_offset: 0,
        dst: ResourceId::new(2),
        dst_offset: 0,
        size: 64,
    });

    // Copy cannot be inserted in a render pass
    let err = validate_pass_hazards(&pass).expect_err("copy in render pass must fail");
    assert_eq!(err, HazardError::CopyInRenderPass { pass_id: 1 });

    // Overlapping copy endpoints in copy pass
    let mut copy_pass = Pass::new_copy(PassId::new(2), "illegal_copy");
    copy_pass.copies.push(CopyCommand::BufferToBuffer {
        src: buf_a,
        src_offset: 0,
        dst: buf_a,
        dst_offset: 32,
        size: 64,
    });
    let err2 = validate_pass_hazards(&copy_pass).expect_err("overlapping copy must fail");
    assert_eq!(err2, HazardError::OverlappingCopyEndpoints { resource_id: 1 });
}
