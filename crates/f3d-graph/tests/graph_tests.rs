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
    ColorAttachment, ComputeBufferBinding, CopyCommand, DepthStencilAttachment, Dispatch, Draw,
    DrawKind, LoadOp, Pass, PassId, PassKind, RenderBundle, StoreOp,
};
use f3d_graph::plan::{
    build_red_a_blue_b_plan, build_red_a_blue_b_render_plan, build_single_pass_bridge_plan,
    build_two_target_bridge_plan,
};
use f3d_graph::resource::{
    ResourceAccess, ResourceId, ResourceKind, ResourceUse, SubresourceRange, TextureAspect,
};
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
    graph.add_pass(pass.clone()).expect("add pass");
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

    // Final operations belong to the source pass. Only the intermediate segment
    // forces Store so that the next segment can load its contents.
    for (color_store, depth_store, stencil_store) in [
        (StoreOp::Discard, StoreOp::Store, StoreOp::Discard),
        (StoreOp::Store, StoreOp::Discard, StoreOp::Store),
    ] {
        let mut variant = pass.clone();
        variant.color_attachments[1].store_op = color_store;
        let depth = variant.depth_stencil_attachment.as_mut().unwrap();
        depth.depth_store_op = Some(depth_store);
        depth.stencil_read_only = false;
        depth.stencil_load_op = Some(LoadOp::Clear);
        depth.stencil_store_op = Some(stencil_store);
        let mut graph = PassGraph::new();
        graph.add_pass(variant).expect("add mixed-store pass");
        let plan = graph.compile(None).expect("split mixed-store pass");
        assert_eq!(plan.segments.len(), 2);

        let first = &plan.segments[0];
        assert!(first.color_attachments.iter().all(|ca| ca.store_op == StoreOp::Store));
        let first_depth = first.depth_stencil_attachment.as_ref().unwrap();
        assert_eq!(first_depth.depth_store_op, Some(StoreOp::Store));
        assert_eq!(first_depth.stencil_store_op, Some(StoreOp::Store));
        let last = &plan.segments[1];
        let color = last.color_attachments.iter().find(|ca| ca.target_id == tex_destination).unwrap();
        assert_eq!(color.load_op, LoadOp::Load);
        assert_eq!(color.store_op, color_store);
        let depth = last.depth_stencil_attachment.as_ref().unwrap();
        assert_eq!(depth.depth_load_op, Some(LoadOp::Load));
        assert_eq!(depth.stencil_load_op, Some(LoadOp::Load));
        assert_eq!(depth.depth_store_op, Some(depth_store));
        assert_eq!(depth.stencil_store_op, Some(stencil_store));
    }
}

#[test]
fn positive_automatic_pass_splitting_preserves_readonly_depth_matrix() {
    // Tests pass splitting with Color A (intermediate) + Color B (destination) + Read-Only Depth D.
    // Matrix covers depth sampling: neither draw, second draw only, and both draws (including
    // draw 0 sampling D before the color hazard in draw 1, proving legal D sampling is not a false split point).
    // In all variants:
    // 1. Splitting succeeds into 2 segments at draw 1.
    // 2. Both segments retain D with depth_read_only=true and absent (None) load/store ops.
    // 3. Segment 1 stores A+B; Segment 2 drops A, loads B with LoadOp::Load, and stores B.
    let tex_a = ResourceId::new(150);
    let tex_b = ResourceId::new(151);
    let tex_d = ResourceId::new(152);
    let buf_v = ResourceId::new(153);

    for (sample_draw0, sample_draw1) in [(false, false), (false, true), (true, true)] {
        let mut pass = Pass::new_render(PassId::new(1), "split_readonly_depth");
        pass.color_attachments.push(ColorAttachment::new_clear(tex_a, [0.0, 0.0, 0.0, 1.0]));
        pass.color_attachments.push(ColorAttachment::new_clear(tex_b, [0.0, 0.0, 0.0, 1.0]));
        let mut dsa = DepthStencilAttachment::new_depth_clear(tex_d, 1.0);
        dsa.depth_read_only = true;
        dsa.depth_load_op = None;
        dsa.depth_store_op = None;
        pass.depth_stencil_attachment = Some(dsa);

        let mut draw0_uses = vec![ResourceUse::buffer_vertex(buf_v, DataVersion::INITIAL, Some(0), Some(64))];
        if sample_draw0 {
            draw0_uses.push(ResourceUse::texture_sampled(tex_d, DataVersion::INITIAL, SubresourceRange::full_texture()));
        }
        pass.draws.push(Draw::new(0, 1, 3, 0, draw0_uses));

        let mut draw1_uses = vec![
            ResourceUse::buffer_vertex(buf_v, DataVersion::INITIAL, Some(0), Some(64)),
            ResourceUse::texture_sampled(tex_a, DataVersion::INITIAL, SubresourceRange::full_texture()),
        ];
        if sample_draw1 {
            draw1_uses.push(ResourceUse::texture_sampled(tex_d, DataVersion::INITIAL, SubresourceRange::full_texture()));
        }
        pass.draws.push(Draw::new(1, 2, 3, 0, draw1_uses));

        assert!(can_split_pass(&pass));

        let mut graph = PassGraph::new();
        graph.add_pass(pass).expect("add pass");
        let plan = graph.compile(None).expect("must compile split plan");

        assert_eq!(plan.split_count, 1);
        assert_eq!(plan.segments.len(), 2);

        // Segment 1 (draw 0): stores colors, keeps read-only depth with absent ops
        let s1 = &plan.segments[0];
        assert_eq!(s1.draws.len(), 1);
        assert_eq!(s1.draws[0].draw_id, 0);
        assert!(s1.color_attachments.iter().any(|ca| ca.target_id == tex_a && ca.store_op == StoreOp::Store));
        assert!(s1.color_attachments.iter().any(|ca| ca.target_id == tex_b && ca.store_op == StoreOp::Store));
        let d1 = s1.depth_stencil_attachment.as_ref().expect("seg1 depth");
        assert_eq!(d1.target_id, tex_d);
        assert!(d1.depth_read_only);
        assert_eq!(d1.depth_load_op, None);
        assert_eq!(d1.depth_store_op, None);
        assert!(d1.stencil_read_only);
        assert_eq!(d1.stencil_load_op, None);
        assert_eq!(d1.stencil_store_op, None);

        // Segment 2 (draw 1): drops sampled intermediate A, preserves B, preserves read-only depth with absent ops
        let s2 = &plan.segments[1];
        assert_eq!(s2.draws.len(), 1);
        assert_eq!(s2.draws[0].draw_id, 1);
        assert!(!s2.color_attachments.iter().any(|ca| ca.target_id == tex_a));
        let ca_b = s2.color_attachments.iter().find(|ca| ca.target_id == tex_b).expect("seg2 color b");
        assert_eq!(ca_b.load_op, LoadOp::Load);
        assert_eq!(ca_b.store_op, StoreOp::Store);
        let d2 = s2.depth_stencil_attachment.as_ref().expect("seg2 depth");
        assert_eq!(d2.target_id, tex_d);
        assert!(d2.depth_read_only);
        assert_eq!(d2.depth_load_op, None);
        assert_eq!(d2.depth_store_op, None);
        assert!(d2.stencil_read_only);
        assert_eq!(d2.stencil_load_op, None);
        assert_eq!(d2.stencil_store_op, None);
    }
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
    tracker.end_frame_interval(canvas_id, epoch_e1).expect("end frame 1 interval");
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

    // Single dispatch attempting to bind the same storage buffer twice as writable with overlapping ranges [0, 1024) and [512, 1536)
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
                Some(512),
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
fn positive_compute_per_dispatch_disjoint_adjacent_writable_ranges_accepted() {
    // WebGPU storage exception allows disjoint binding ranges in the same dispatch (§8.5, [S51]).
    // Exact-adjacent ranges [0, 1024) and [1024, 2048) do not overlap.
    let buf_storage = ResourceId::new(71);
    let mut pass = Pass::new_compute(PassId::new(1), "compute_adjacent_pass");

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

    validate_pass_hazards(&pass)
        .expect("disjoint adjacent writable ranges in same dispatch must be accepted");
}

#[test]
fn positive_compute_repeated_readonly_use_index_accepted() {
    // Root 21773: Multiple bindings referencing the same read-only use_index are legal (§8.5).
    let buf_storage = ResourceId::new(73);
    let mut pass = Pass::new_compute(PassId::new(1), "compute_readonly_alias_pass");

    let dispatch = Dispatch::new_with_bindings(
        0,
        500,
        [64, 1, 1],
        vec![ResourceUse::buffer_storage_read(
            buf_storage,
            DataVersion::INITIAL,
            Some(0),
            Some(1024),
        )],
        vec![
            ComputeBufferBinding::new(0, 0),
            ComputeBufferBinding::new(1, 0),
        ],
    );
    pass.dispatches.push(dispatch);

    validate_pass_hazards(&pass)
        .expect("repeated read-only compute binding must be accepted");
}

#[test]
fn negative_compute_repeated_writable_use_index_rejected_as_alias() {
    // Root 21773: Multiple bindings referencing the same writable use_index must be rejected as an alias.
    let buf_storage = ResourceId::new(74);
    let mut pass = Pass::new_compute(PassId::new(1), "compute_writable_alias_pass");

    let dispatch = Dispatch::new_with_bindings(
        0,
        500,
        [64, 1, 1],
        vec![ResourceUse::buffer_storage_write(
            buf_storage,
            DataVersion::INITIAL,
            Some(0),
            Some(1024),
        )],
        vec![
            ComputeBufferBinding::new(0, 0),
            ComputeBufferBinding::new(1, 0),
        ],
    );
    pass.dispatches.push(dispatch);

    let err = validate_pass_hazards(&pass)
        .expect_err("repeated writable compute binding must be rejected as an alias");
    match err {
        HazardError::ComputeWritableAlias {
            resource_id,
            dispatch_id,
            subresource,
        } => {
            assert_eq!(resource_id, buf_storage.get());
            assert_eq!(dispatch_id, 0);
            assert_eq!(subresource, SubresourceRange::WholeBuffer);
        }
        other => panic!("expected ComputeWritableAlias, got: {other:?}"),
    }
}

#[test]
fn negative_compute_binding_invalid_use_index_rejected() {
    // Root 21498: use_index >= dispatch.uses.len() must be rejected with InvalidUseIndex.
    let buf = ResourceId::new(75);
    let mut pass = Pass::new_compute(PassId::new(1), "compute_oob_use_pass");

    let dispatch = Dispatch::new_with_bindings(
        0,
        500,
        [64, 1, 1],
        vec![ResourceUse::buffer_uniform(
            buf,
            DataVersion::INITIAL,
            Some(0),
            Some(256),
        )],
        vec![ComputeBufferBinding::new(0, 99)],
    );
    pass.dispatches.push(dispatch);

    let err = validate_pass_hazards(&pass)
        .expect_err("out of bounds use_index must be rejected");
    match err {
        HazardError::InvalidUseIndex {
            dispatch_id,
            binding_index,
            use_index,
            uses_len,
        } => {
            assert_eq!(dispatch_id, 0);
            assert_eq!(binding_index, 0);
            assert_eq!(use_index, 99);
            assert_eq!(uses_len, 1);
        }
        other => panic!("expected InvalidUseIndex, got: {other:?}"),
    }
}

#[test]
fn negative_compute_binding_duplicate_binding_index_rejected() {
    // Root 21498: Duplicate group-0 binding_index must be rejected.
    let buf0 = ResourceId::new(76);
    let buf1 = ResourceId::new(77);
    let mut pass = Pass::new_compute(PassId::new(1), "compute_dup_binding_pass");

    let dispatch = Dispatch::new_with_bindings(
        0,
        500,
        [64, 1, 1],
        vec![
            ResourceUse::buffer_storage_read(
                buf0,
                DataVersion::INITIAL,
                Some(0),
                Some(256),
            ),
            ResourceUse::buffer_storage_read(
                buf1,
                DataVersion::INITIAL,
                Some(0),
                Some(256),
            ),
        ],
        vec![
            ComputeBufferBinding::new(0, 0),
            ComputeBufferBinding::new(0, 1),
        ],
    );
    pass.dispatches.push(dispatch);

    let err = validate_pass_hazards(&pass)
        .expect_err("duplicate binding_index must be rejected");
    match err {
        HazardError::DuplicateBindingIndex {
            dispatch_id,
            binding_index,
        } => {
            assert_eq!(dispatch_id, 0);
            assert_eq!(binding_index, 0);
        }
        other => panic!("expected DuplicateBindingIndex, got: {other:?}"),
    }
}

#[test]
fn negative_compute_binding_incompatible_access_rejected() {
    // Root 21498: Non-compute buffer access (e.g. VertexAttribute) must be rejected.
    let buf = ResourceId::new(78);
    let mut pass = Pass::new_compute(PassId::new(1), "compute_incompatible_access_pass");

    let dispatch = Dispatch::new_with_bindings(
        0,
        500,
        [64, 1, 1],
        vec![ResourceUse::buffer_vertex(
            buf,
            DataVersion::INITIAL,
            Some(0),
            Some(256),
        )],
        vec![ComputeBufferBinding::new(0, 0)],
    );
    pass.dispatches.push(dispatch);

    let err = validate_pass_hazards(&pass)
        .expect_err("vertex buffer access in compute binding must be rejected");
    match err {
        HazardError::IncompatibleAccess {
            dispatch_id,
            binding_index,
            access,
        } => {
            assert_eq!(dispatch_id, 0);
            assert_eq!(binding_index, 0);
            assert_eq!(access, ResourceAccess::VertexBuffer);
        }
        other => panic!("expected IncompatibleAccess, got: {other:?}"),
    }
}

#[test]
fn positive_storage_buffer_reuse_in_separate_draws_accepted() {
    // In WebGPU render passes, storage buffers may be reused and written across separate draws (§8.5).
    let buf_storage = ResourceId::new(72);
    let mut pass = Pass::new_render(PassId::new(1), "render_separate_draws_storage_pass");

    let draw0 = Draw::new(
        0,
        1,
        3,
        0,
        vec![ResourceUse::buffer_storage_write(
            buf_storage,
            DataVersion::INITIAL,
            Some(0),
            Some(1024),
        )],
    );
    let draw1 = Draw::new(
        1,
        1,
        3,
        0,
        vec![ResourceUse::buffer_storage_write(
            buf_storage,
            DataVersion::INITIAL,
            Some(0),
            Some(1024),
        )],
    );
    pass.draws.push(draw0);
    pass.draws.push(draw1);

    validate_pass_hazards(&pass)
        .expect("repeated storage buffer use in separate draws must be accepted");
}

#[test]
fn negative_storage_buffer_overlap_in_same_draw_rejected() {
    // Writable overlapping bindings within a single draw are forbidden (§8.5, [S51]).
    let buf_storage = ResourceId::new(73);
    let mut pass = Pass::new_render(PassId::new(1), "render_draw_overlap_pass");

    let draw = Draw::new(
        0,
        1,
        3,
        0,
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
                Some(512),
                Some(1024),
            ),
        ],
    );
    pass.draws.push(draw);

    let err = validate_pass_hazards(&pass)
        .expect_err("overlapping writable storage bindings in same draw must be rejected");
    match err {
        HazardError::DrawWritableAlias {
            resource_id,
            draw_id,
            ..
        } => {
            assert_eq!(resource_id, buf_storage.get());
            assert_eq!(draw_id, 0);
        }
        other => panic!("expected DrawWritableAlias, got: {other:?}"),
    }
}

#[test]
fn positive_storage_buffer_adjacency_in_same_draw_accepted() {
    // Exact adjacent ranges [0, 1024) and [1024, 2048) in a single draw do not overlap and must be accepted.
    let buf_storage = ResourceId::new(74);
    let mut pass = Pass::new_render(PassId::new(1), "render_draw_adjacent_pass");

    let draw = Draw::new(
        0,
        1,
        3,
        0,
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
    pass.draws.push(draw);

    validate_pass_hazards(&pass)
        .expect("disjoint adjacent storage buffer ranges in same draw must be accepted");
}

#[test]
fn negative_mixed_role_disjoint_offsets_rejected_in_compute_and_render() {
    // Invariant (§8.5, [S51]): Whole-buffer rule forbids incompatible roles (UniformBuffer + StorageBufferWrite,
    // or StorageBufferRead + StorageBufferWrite) even at disjoint offsets, both in render passes and compute passes.

    // A. Render pass with UniformBuffer + StorageBufferWrite at disjoint offsets
    {
        let shared_buffer = ResourceId::new(75);
        let mut pass = Pass::new_render(PassId::new(1), "mixed_render_pass_uniform_write");
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
        let draw1 = Draw::new(
            1,
            1,
            3,
            0,
            vec![ResourceUse::buffer_storage_write(
                shared_buffer,
                DataVersion::INITIAL,
                Some(256),
                Some(1024),
            )],
        );
        pass.draws.push(draw0);
        pass.draws.push(draw1);

        let err = validate_pass_hazards(&pass)
            .expect_err("mixed uniform + storage write in render pass must be rejected");
        match err {
            HazardError::WholeBufferConflict { buffer_id, .. } => {
                assert_eq!(buffer_id, shared_buffer.get());
            }
            other => panic!("expected WholeBufferConflict, got: {other:?}"),
        }
    }

    // B. Render pass with StorageBufferRead + StorageBufferWrite at disjoint offsets
    {
        let shared_buffer = ResourceId::new(76);
        let mut pass = Pass::new_render(PassId::new(2), "mixed_render_pass_read_write");
        let draw0 = Draw::new(
            0,
            1,
            3,
            0,
            vec![ResourceUse::buffer_storage_read(
                shared_buffer,
                DataVersion::INITIAL,
                Some(0),
                Some(256),
            )],
        );
        let draw1 = Draw::new(
            1,
            1,
            3,
            0,
            vec![ResourceUse::buffer_storage_write(
                shared_buffer,
                DataVersion::INITIAL,
                Some(256),
                Some(1024),
            )],
        );
        pass.draws.push(draw0);
        pass.draws.push(draw1);

        let err = validate_pass_hazards(&pass)
            .expect_err("mixed storage read + storage write in render pass must be rejected");
        match err {
            HazardError::WholeBufferConflict { buffer_id, .. } => {
                assert_eq!(buffer_id, shared_buffer.get());
            }
            other => panic!("expected WholeBufferConflict, got: {other:?}"),
        }
    }

    // C. Compute pass with StorageBufferRead + StorageBufferWrite in the SAME dispatch at disjoint offsets
    {
        let shared_buffer = ResourceId::new(77);
        let mut pass = Pass::new_compute(PassId::new(3), "mixed_compute_same_dispatch_read_write");
        let dispatch = Dispatch::new(
            0,
            500,
            [64, 1, 1],
            vec![
                ResourceUse::buffer_storage_read(
                    shared_buffer,
                    DataVersion::INITIAL,
                    Some(0),
                    Some(256),
                ),
                ResourceUse::buffer_storage_write(
                    shared_buffer,
                    DataVersion::INITIAL,
                    Some(256),
                    Some(1024),
                ),
            ],
        );
        pass.dispatches.push(dispatch);

        let err = validate_pass_hazards(&pass)
            .expect_err("mixed storage read + storage write in same dispatch must be rejected");
        match err {
            HazardError::WholeBufferConflict { buffer_id, .. } => {
                assert_eq!(buffer_id, shared_buffer.get());
            }
            other => panic!("expected WholeBufferConflict, got: {other:?}"),
        }
    }

    // D. Compute pass with UniformBuffer + StorageBufferWrite in the SAME dispatch at disjoint offsets
    {
        let shared_buffer = ResourceId::new(78);
        let mut pass = Pass::new_compute(PassId::new(4), "mixed_compute_same_dispatch_uniform_write");
        let dispatch = Dispatch::new(
            0,
            500,
            [64, 1, 1],
            vec![
                ResourceUse::buffer_uniform(
                    shared_buffer,
                    DataVersion::INITIAL,
                    Some(0),
                    Some(256),
                ),
                ResourceUse::buffer_storage_write(
                    shared_buffer,
                    DataVersion::INITIAL,
                    Some(256),
                    Some(1024),
                ),
            ],
        );
        pass.dispatches.push(dispatch);

        let err = validate_pass_hazards(&pass)
            .expect_err("mixed uniform + storage write in same dispatch must be rejected");
        match err {
            HazardError::WholeBufferConflict { buffer_id, .. } => {
                assert_eq!(buffer_id, shared_buffer.get());
            }
            other => panic!("expected WholeBufferConflict, got: {other:?}"),
        }
    }
}

#[test]
fn positive_compute_separate_dispatches_mixed_roles_accepted() {
    // In WebGPU (§8.5, [S51]), compute usage scope is each dispatch.
    // Separate dispatches in the same compute pass can use a buffer in different roles (e.g. uniform read, then storage write).
    let shared_buffer = ResourceId::new(79);
    let mut pass = Pass::new_compute(PassId::new(5), "separate_dispatches_compute_pass");
    let dispatch0 = Dispatch::new(
        0,
        500,
        [64, 1, 1],
        vec![ResourceUse::buffer_uniform(
            shared_buffer,
            DataVersion::INITIAL,
            Some(0),
            Some(256),
        )],
    );
    let dispatch1 = Dispatch::new(
        1,
        500,
        [64, 1, 1],
        vec![ResourceUse::buffer_storage_write(
            shared_buffer,
            DataVersion::INITIAL,
            Some(256),
            Some(1024),
        )],
    );
    pass.dispatches.push(dispatch0);
    pass.dispatches.push(dispatch1);

    validate_pass_hazards(&pass)
        .expect("separate dispatches with mixed buffer roles in same compute pass must be accepted");
}

#[test]
fn positive_uniform_and_storage_read_coexistence_accepted() {
    // In WebGPU (§8.5, [S51]), UniformBuffer and StorageBufferRead are both read-only buffer usages
    // and ARE compatible on the same buffer within the same pass.

    // A. Render pass
    {
        let shared_buffer = ResourceId::new(80);
        let mut pass = Pass::new_render(PassId::new(1), "uniform_storage_read_render_pass");
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
        let draw1 = Draw::new(
            1,
            1,
            3,
            0,
            vec![ResourceUse::buffer_storage_read(
                shared_buffer,
                DataVersion::INITIAL,
                Some(256),
                Some(1024),
            )],
        );
        pass.draws.push(draw0);
        pass.draws.push(draw1);

        validate_pass_hazards(&pass)
            .expect("uniform and storage-read coexistence in render pass must be accepted");
    }

    // B. Compute pass with both read-only uses in ONE dispatch
    {
        let shared_buffer = ResourceId::new(81);
        let mut pass = Pass::new_compute(PassId::new(2), "uniform_storage_read_compute_pass");
        let dispatch = Dispatch::new(
            0,
            500,
            [64, 1, 1],
            vec![
                ResourceUse::buffer_uniform(
                    shared_buffer,
                    DataVersion::INITIAL,
                    Some(0),
                    Some(256),
                ),
                ResourceUse::buffer_storage_read(
                    shared_buffer,
                    DataVersion::INITIAL,
                    Some(256),
                    Some(1024),
                ),
            ],
        );
        pass.dispatches.push(dispatch);

        validate_pass_hazards(&pass)
            .expect("uniform and storage-read coexistence in single compute dispatch must be accepted");
    }
}

#[test]
fn test_resource_access_compatibility_matrix() {
    // WebGPU spec (§8.5, Compatible Usage List):
    // 1. Only StorageBufferWrite + StorageBufferWrite is the storage exception
    assert!(ResourceAccess::StorageBufferWrite.is_compatible_with(&ResourceAccess::StorageBufferWrite));

    // 2. All-read-only combinations are compatible
    assert!(ResourceAccess::UniformBuffer.is_compatible_with(&ResourceAccess::StorageBufferRead));
    assert!(ResourceAccess::StorageBufferRead.is_compatible_with(&ResourceAccess::UniformBuffer));
    assert!(ResourceAccess::VertexBuffer.is_compatible_with(&ResourceAccess::StorageBufferRead));
    assert!(ResourceAccess::StorageBufferRead.is_compatible_with(&ResourceAccess::StorageBufferRead));

    // 3. Mixed read/write is whole-buffer incompatible
    assert!(!ResourceAccess::StorageBufferRead.is_compatible_with(&ResourceAccess::StorageBufferWrite));
    assert!(!ResourceAccess::StorageBufferWrite.is_compatible_with(&ResourceAccess::StorageBufferRead));
    assert!(!ResourceAccess::UniformBuffer.is_compatible_with(&ResourceAccess::StorageBufferWrite));
    assert!(!ResourceAccess::StorageBufferWrite.is_compatible_with(&ResourceAccess::UniformBuffer));
    assert!(!ResourceAccess::VertexBuffer.is_compatible_with(&ResourceAccess::StorageBufferWrite));
    assert!(!ResourceAccess::StorageBufferWrite.is_compatible_with(&ResourceAccess::VertexBuffer));
}

#[test]
fn test_buffer_range_bounds_via_validate_pass_hazards() {
    // Unknown (None) or overflow ranges are conservatively treated as potentially overlapping
    let cases: &[(Option<u64>, Option<u64>, Option<u64>, Option<u64>, bool)] = &[
        // (offset_a, size_a, offset_b, size_b, is_legal)
        (Some(0), Some(1024), Some(1024), Some(1024), true),    // exact adjacent -> legal
        (Some(0), Some(1024), Some(512), Some(1024), false),    // real overlap -> alias
        (None, Some(1024), Some(1024), Some(1024), false),      // unknown offset -> alias
        (Some(0), None, Some(1024), Some(1024), false),          // unknown size -> alias
        (Some(u64::MAX - 10), Some(20), Some(0), Some(1024), false), // overflow -> alias
    ];

    for (i, &(oa, sa, ob, sb, is_legal)) in cases.iter().enumerate() {
        let buf = ResourceId::new(90 + i as u32);
        let mut pass = Pass::new_compute(PassId::new(1), "bounds_check_pass");
        let dispatch = Dispatch::new(
            0,
            500,
            [64, 1, 1],
            vec![
                ResourceUse::buffer_storage_write(buf, DataVersion::INITIAL, oa, sa),
                ResourceUse::buffer_storage_write(buf, DataVersion::INITIAL, ob, sb),
            ],
        );
        pass.dispatches.push(dispatch);

        if is_legal {
            validate_pass_hazards(&pass).expect("disjoint ranges must pass");
        } else {
            let err = validate_pass_hazards(&pass).expect_err("overlapping/unknown/overflow must fail");
            assert!(matches!(err, HazardError::ComputeWritableAlias { .. }));
        }
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
    tracker.end_frame_interval(canvas_id, cached_epoch).expect("end frame 1 interval");

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

    // 4. Repeated accesses remain valid until the host ends the interval.
    tracker.validate_canvas_access(canvas_id, output.epoch).expect("first access");
    tracker.validate_canvas_access(canvas_id, output.epoch).expect("second access");
    tracker.end_frame_interval(canvas_id, output.epoch).expect("end interval");
    let err = tracker
        .validate_canvas_access(canvas_id, output.epoch)
        .expect_err("expired canvas cannot be reused");
    assert_eq!(
        err,
        CanvasError::CanvasIntervalEnded {
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

    // End c1's interval only.
    tracker.end_frame_interval(c1, out1.epoch).unwrap();

    // c1 is expired, c2 is still valid in its interval.
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
fn re_register_canvas_with_changed_config_advances_epoch_and_invalidates_acquired_interval() {
    let canvas_id = CanvasId::new(7);
    let res1 = ResourceId::new(101);
    let res2 = ResourceId::new(102);
    let mut tracker = CanvasEpochTracker::new();

    // 1. Initial registration and acquire
    tracker.register_canvas(canvas_id, res1, 800, 600, CanvasFormat::Bgra8Unorm);
    let out1 = tracker
        .begin_frame_acquire(canvas_id)
        .expect("initial frame acquire");
    assert_eq!(out1.epoch, Epoch::new(1));
    assert_eq!(out1.width, 800);
    assert_eq!(out1.height, 600);
    assert_eq!(out1.resource_id, res1);
    assert_eq!(out1.format, CanvasFormat::Bgra8Unorm);

    // Prior to re-registration, output access is valid
    tracker
        .validate_canvas_access(canvas_id, out1.epoch)
        .expect("access must be valid during active acquisition");

    // 2. Re-register with SAME configuration: explicit no-op on epoch and active interval
    tracker.register_canvas(canvas_id, res1, 800, 600, CanvasFormat::Bgra8Unorm);
    assert_eq!(tracker.epoch_of(canvas_id), Some(Epoch::new(1)));
    tracker
        .validate_canvas_access(canvas_id, out1.epoch)
        .expect("same-config re-registration must preserve active acquisition interval and epoch");

    // 3. Re-register with CHANGED dimensions: closes interval and advances epoch
    tracker.register_canvas(canvas_id, res1, 1024, 768, CanvasFormat::Bgra8Unorm);
    assert_eq!(
        tracker.epoch_of(canvas_id),
        Some(Epoch::new(2)),
        "re-registration with changed dimensions must advance epoch"
    );

    // Old output is now stale: interval closed and epoch advanced
    assert!(
        tracker.validate_canvas_access(canvas_id, out1.epoch).is_err(),
        "old output from epoch 1 must be rejected after re-registration"
    );

    // 4. Re-acquire with new dimensions: advances to epoch 3
    let out2 = tracker
        .begin_frame_acquire(canvas_id)
        .expect("second frame acquire");
    assert_eq!(out2.epoch, Epoch::new(3));
    assert_eq!(out2.width, 1024);
    assert_eq!(out2.height, 768);
    tracker
        .validate_canvas_access(canvas_id, out2.epoch)
        .expect("newly acquired output must be valid");

    // Old out1 remains stale
    let err_old = tracker
        .validate_canvas_access(canvas_id, out1.epoch)
        .expect_err("old output must remain rejected across epochs");
    assert_eq!(
        err_old,
        CanvasError::CanvasCachedAcrossEpochs {
            canvas_id: 7,
            cached_epoch: 1,
            current_epoch: 3,
        }
    );

    // 5. Re-register with CHANGED resource_id: advances epoch and invalidates interval
    tracker.register_canvas(canvas_id, res2, 1024, 768, CanvasFormat::Bgra8Unorm);
    assert_eq!(tracker.epoch_of(canvas_id), Some(Epoch::new(4)));
    assert_eq!(tracker.find_by_resource(res1), None);
    assert_eq!(tracker.find_by_resource(res2), Some(canvas_id));
    assert!(tracker.validate_canvas_access(canvas_id, out2.epoch).is_err());

    // 6. Re-register with CHANGED format: advances epoch and invalidates interval
    let out3 = tracker.begin_frame_acquire(canvas_id).expect("acquire 3");
    assert_eq!(out3.epoch, Epoch::new(5));
    tracker.register_canvas(canvas_id, res2, 1024, 768, CanvasFormat::Rgba8Unorm);
    assert_eq!(tracker.epoch_of(canvas_id), Some(Epoch::new(6)));
    assert!(tracker.validate_canvas_access(canvas_id, out3.epoch).is_err());
}

#[test]
fn re_register_canvas_at_u64_max_handles_overflow_without_wrapping() {
    let canvas_id = CanvasId::new(99);
    let res = ResourceId::new(500);
    let mut tracker = CanvasEpochTracker::new();
    tracker.register_canvas(canvas_id, res, 800, 600, CanvasFormat::Bgra8Unorm);

    // Seed epoch at u64::MAX
    tracker.seed_epoch_for_test(canvas_id, Epoch::new(u64::MAX));
    assert_eq!(tracker.epoch_of(canvas_id), Some(Epoch::new(u64::MAX)));

    // Re-register with changed dimensions: must handle overflow by saturating at u64::MAX without wrapping to 0
    tracker.register_canvas(canvas_id, res, 1280, 720, CanvasFormat::Bgra8Unorm);
    assert_eq!(
        tracker.epoch_of(canvas_id),
        Some(Epoch::new(u64::MAX)),
        "epoch at u64::MAX must not wrap to 0"
    );

    // Attempting to acquire after saturation fails with EpochOverflow
    let err = tracker
        .begin_frame_acquire(canvas_id)
        .expect_err("acquiring after epoch saturation must fail with EpochOverflow");
    assert_eq!(
        err,
        CanvasError::EpochOverflow {
            canvas_id: 99,
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

#[test]
fn bundle_inside_copy_pass_rejected() {
    let mut pass = Pass::new_copy(PassId::new(10), "copy_with_bundle");
    let bundle_draw = Draw::new_bundle(0, 42, 100, vec![]);
    pass.draws.push(bundle_draw);

    // Direct hazard check
    let err = validate_pass_hazards(&pass).expect_err("bundle inside copy pass must be rejected");
    assert_eq!(err, HazardError::BundleInCopyPass { pass_id: 10 });

    // Through PassGraph compilation
    let mut graph = PassGraph::new();
    graph.add_pass(pass).expect("add pass");
    let compile_err = graph
        .compile(None)
        .expect_err("compilation of copy pass with bundle must fail");
    assert_eq!(
        compile_err,
        GraphError::Hazard(HazardError::BundleInCopyPass { pass_id: 10 })
    );
}

#[test]
fn bundle_then_direct_draw_requires_rebind() {
    let target_tex = ResourceId::new(10);
    let mut pass = Pass::new_render(PassId::new(1), "bundle_pass");
    pass.color_attachments
        .push(ColorAttachment::new_clear(target_tex, [0.0, 0.0, 0.0, 1.0]));

    // Draw 0: Render bundle execution
    let bundle_draw = Draw::new_bundle(0, 42, 100, vec![]);
    pass.draws.push(bundle_draw);

    // Draw 1: Direct draw assuming warm state (omitting rebind) -> must fail!
    let direct_warm = Draw::new(1, 100, 3, 0, vec![]).with_assumes_warm_state(true);
    pass.draws.push(direct_warm);

    let err = validate_pass_hazards(&pass)
        .expect_err("direct draw assuming warm state after bundle must fail");
    assert_eq!(
        err,
        HazardError::BundleDirectDrawRequiresRebind {
            pass_id: 1,
            draw_id: 1,
        }
    );

    // Compilation through PassGraph must also fail with exact hazard
    let mut bad_graph = PassGraph::new();
    bad_graph.add_pass(pass.clone()).expect("add pass");
    let bad_compile = bad_graph.compile(None).expect_err("compile must fail");
    assert_eq!(
        bad_compile,
        GraphError::Hazard(HazardError::BundleDirectDrawRequiresRebind {
            pass_id: 1,
            draw_id: 1,
        })
    );

    // Now update Draw 1 to explicitly rebind (requires_rebind = true / assumes_warm_state = false)
    pass.draws[1] = Draw::new(1, 100, 3, 0, vec![]).with_rebind_required(true);

    assert!(validate_pass_hazards(&pass).is_ok());

    let mut good_graph = PassGraph::new();
    good_graph.add_pass(pass).expect("add pass");
    let plan = good_graph.compile(None).expect("compile succeeds");

    // Assert plan and segment bundle accessors
    assert!(plan.has_bundles());
    assert_eq!(plan.total_bundle_count(), 1);
    assert_eq!(plan.segment_count(), 1);

    let seg = &plan.segments()[0];
    assert!(seg.has_bundles());
    assert_eq!(seg.bundle_count(), 1);
    assert_eq!(seg.bundle_boundaries(), vec![0]);
    assert_eq!(seg.bundle_ids(), vec![42]);

    // Draw 1 directly follows bundle Draw 0, so it crosses bundle boundary
    assert!(seg.draw_crosses_bundle_boundary(1));
    // Draw 0 does not follow a bundle
    assert!(!seg.draw_crosses_bundle_boundary(0));

    // Inspect individual draws
    assert!(seg.draws()[0].is_bundle());
    assert_eq!(seg.draws()[0].bundle_id(), Some(42));
    assert_eq!(seg.draws()[0].kind(), DrawKind::Bundle { bundle_id: 42 });

    assert!(seg.draws()[1].is_direct());
    assert_eq!(seg.draws()[1].bundle_id(), None);
    assert_eq!(seg.draws()[1].kind(), DrawKind::Direct);
    assert!(seg.draws()[1].rebind_required());
    assert!(!seg.draws()[1].assumes_warm_state());
}

#[test]
fn bundle_state_reset_interleaved_warm_state_lifecycle() {
    let target_tex = ResourceId::new(10);
    let mut pass = Pass::new_render(PassId::new(1), "interleaved_bundle_pass");
    pass.color_attachments
        .push(ColorAttachment::new_clear(target_tex, [0.0, 0.0, 0.0, 1.0]));

    // Pass helper: add bundle via RenderBundle
    let bundle = RenderBundle::new(50, 100, vec![]);
    pass = pass.with_bundle(bundle); // Draw 0: Bundle 50

    // Draw 1: Direct draw with explicit rebind
    let draw1 = Draw::new(1, 100, 3, 0, vec![]);
    pass = pass.with_draw(draw1);

    // Draw 2: Direct draw assuming warm state (inheriting Draw 1's state) -> LEGAL!
    let draw2 = Draw::new_warm_state(2, 100, 3, 0, vec![]);
    pass = pass.with_draw(draw2);

    // Validation must succeed because Draw 1 rebinded after Bundle 50
    assert!(validate_pass_hazards(&pass).is_ok());

    // Draw 3: Another bundle
    pass = pass.with_bundle_draw(51, 101, vec![]);

    // Draw 4: Direct draw assuming warm state after Bundle 51 -> ILLEGAL!
    let draw4 = Draw::new_warm_state(4, 101, 3, 0, vec![]);
    pass = pass.with_draw(draw4);

    let err = validate_pass_hazards(&pass).expect_err("Draw 4 must rebind after Bundle 51");
    assert_eq!(
        err,
        HazardError::BundleDirectDrawRequiresRebind {
            pass_id: 1,
            draw_id: 4,
        }
    );
}

// -----------------------------------------------------------------------------
// Seeded Deterministic Property Tests (fuzzing of plans, vqa.3 & 2v8.2)
// -----------------------------------------------------------------------------

/// Minimal 64-bit Linear Congruential Generator (LCG) for deterministic property testing.
///
/// Multiplier and increment are standard constants from Knuth / MMIX.
/// Provides a zero-dependency, reproducible pseudo-random stream across platforms.
#[derive(Clone, Copy, Debug)]
struct TestLcg {
    state: u64,
}

impl TestLcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_range(&mut self, min: u32, max: u32) -> u32 {
        if min >= max {
            return min;
        }
        min + (self.next_u32() % (max - min + 1))
    }

    fn next_bool(&mut self) -> bool {
        (self.next_u32() & 1) == 1
    }
}

#[test]
fn property_test_compile_never_panics_and_topological_order() {
    const SEED: u64 = 0x5EED_F3D9_9A55_0001;
    const ITERATIONS: usize = 1_000;
    let mut rng = TestLcg::new(SEED);

    for iter in 0..ITERATIONS {
        let mut graph = PassGraph::new();
        let pass_count = rng.next_range(1, 6) as usize;

        let mut expected_deps: Vec<(PassId, Vec<PassId>)> = Vec::new();
        let mut pass_ids: Vec<PassId> = Vec::new();

        for i in 0..pass_count {
            let pass_id = PassId::new((i as u32) + 1);
            pass_ids.push(pass_id);

            let kind_choice = rng.next_range(0, 2);
            let mut pass = match kind_choice {
                0 => {
                    let mut p = Pass::new_render(pass_id, format!("render_{i}"));
                    let target = ResourceId::new(rng.next_range(1, 8));
                    p = p.with_color_attachment(ColorAttachment::new_clear(
                        target,
                        [0.0, 0.0, 0.0, 1.0],
                    ));
                    let draw = Draw::new(0, rng.next_range(1, 5), 3, 0, vec![]);
                    p.with_draw(draw)
                }
                1 => {
                    let mut p = Pass::new_compute(pass_id, format!("compute_{i}"));
                    let dispatch = Dispatch::new(0, rng.next_range(1, 5), [1, 1, 1], vec![]);
                    p.with_dispatch(dispatch)
                }
                _ => {
                    let mut p = Pass::new_copy(pass_id, format!("copy_{i}"));
                    let src = ResourceId::new(rng.next_range(10, 15));
                    let dst = ResourceId::new(rng.next_range(16, 20));
                    let copy = CopyCommand::BufferToBuffer {
                        src,
                        src_offset: 0,
                        dst,
                        dst_offset: 0,
                        size: 64,
                    };
                    p.with_copy(copy)
                }
            };

            // Random dependencies to earlier passes to create valid DAGs
            let mut deps = Vec::new();
            if i > 0 && rng.next_bool() {
                let dep_idx = rng.next_range(0, (i - 1) as u32) as usize;
                let dep_id = pass_ids[dep_idx];
                pass = pass.with_dependency(dep_id);
                deps.push(dep_id);
            }

            expected_deps.push((pass_id, deps));
            let add_res = graph.add_pass(pass);
            assert!(
                add_res.is_ok(),
                "Failed to add pass with seed {SEED:#018x} at iter {iter}"
            );
        }

        // Randomly add additional dependencies or back-edges to test robustness
        if pass_count > 1 && rng.next_range(0, 3) == 0 {
            let from_idx = rng.next_range(0, (pass_count - 1) as u32) as usize;
            let to_idx = rng.next_range(0, (pass_count - 1) as u32) as usize;
            if from_idx != to_idx {
                let _ = graph.add_dependency(pass_ids[from_idx], pass_ids[to_idx]);
                expected_deps[from_idx].1.push(pass_ids[to_idx]);
            }
        }

        // Property: PassGraph::compile NEVER panics
        let result = graph.compile(None);

        // Property: If compilation succeeds, every accepted plan is a valid topological order
        if let Ok(plan) = result {
            let mut seg_order: std::collections::HashMap<PassId, usize> =
                std::collections::HashMap::new();
            for (seg_idx, seg) in plan.segments().iter().enumerate() {
                seg_order.insert(seg.pass_id(), seg_idx);
            }

            for (pass_id, deps) in &expected_deps {
                if let Some(&pass_idx) = seg_order.get(pass_id) {
                    for dep in deps {
                        if let Some(&dep_idx) = seg_order.get(dep) {
                            assert!(
                                dep_idx < pass_idx,
                                "Topological order invariant violated for seed {SEED:#018x} at iter {iter}: \
                                 pass {pass_id:?} (seg {pass_idx}) depends on {dep:?} (seg {dep_idx})"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn property_test_overlapping_read_write_subresources_rejected() {
    const SEED: u64 = 0x5EED_F3D9_9A55_0002;
    const ITERATIONS: usize = 1_000;
    let mut rng = TestLcg::new(SEED);

    for iter in 0..ITERATIONS {
        let scenario = rng.next_range(0, 2);
        match scenario {
            0 => {
                // Render pass: ColorAttachment write + Draw texture sampling on overlapping subresource
                let pass_id = PassId::new(rng.next_range(1, 500));
                let tex_id = ResourceId::new(rng.next_range(1, 100));
                let mut pass = Pass::new_render(pass_id, format!("render_hazard_{iter}"));

                let mip = rng.next_range(0, 3);
                let layer = rng.next_range(0, 5);
                let subresource = SubresourceRange::single_mip_layer(
                    mip,
                    layer,
                    TextureAspect::All,
                );

                let mut ca = ColorAttachment::new_clear(tex_id, [0.0, 0.0, 0.0, 1.0]);
                ca.view_subresource = subresource.clone();
                pass = pass.with_color_attachment(ca);

                let sample_use = ResourceUse {
                    resource_id: tex_id,
                    kind: ResourceKind::Texture,
                    version: DataVersion::INITIAL,
                    subresource: subresource.clone(),
                    access: ResourceAccess::SampledTexture,
                    byte_offset: None,
                    byte_size: None,
                    canvas_epoch: None,
                };

                pass = pass.with_draw(Draw::new(0, 10, 3, 0, vec![sample_use]));

                // Because this is the only color attachment, splitting cannot resolve it (feedback loop)
                let err = validate_pass_hazards(&pass).expect_err(&format!(
                    "Overlapping attachment and sampled texture must be rejected for seed {SEED:#018x} at iter {iter}"
                ));
                assert_eq!(
                    err,
                    HazardError::AttachmentSamplingConflict {
                        texture_id: tex_id.get(),
                        pass_id: pass_id.get(),
                        subresource: subresource.clone(),
                    },
                    "Seed {SEED:#018x} failed at iter {iter}"
                );

                // PassGraph compilation must also reject it
                let mut graph = PassGraph::new();
                graph.add_pass(pass).expect("add pass");
                let compile_err = graph.compile(None).expect_err(&format!(
                    "Compile must reject attachment sampling conflict for seed {SEED:#018x} at iter {iter}"
                ));
                assert_eq!(
                    compile_err,
                    GraphError::Hazard(HazardError::AttachmentSamplingConflict {
                        texture_id: tex_id.get(),
                        pass_id: pass_id.get(),
                        subresource,
                    }),
                    "Expected GraphError::Hazard(AttachmentSamplingConflict) for seed {SEED:#018x} at iter {iter}"
                );
            }
            1 => {
                // Compute pass: dispatch with writable alias (overlapping read+write or write+write)
                let pass_id = PassId::new(rng.next_range(1, 500));
                let buf_id = ResourceId::new(rng.next_range(1, 100));
                let mut pass = Pass::new_compute(pass_id, format!("compute_hazard_{iter}"));

                let write_use = ResourceUse::buffer_storage_write(buf_id, DataVersion::INITIAL, None, None);
                let read_use = ResourceUse::buffer_storage_read(buf_id, DataVersion::INITIAL, None, None);

                let dispatch = Dispatch::new(0, 10, [1, 1, 1], vec![write_use, read_use]);
                pass = pass.with_dispatch(dispatch);

                let err = validate_pass_hazards(&pass).expect_err(&format!(
                    "Compute writable alias must be rejected for seed {SEED:#018x} at iter {iter}"
                ));
                assert_eq!(
                    err,
                    HazardError::ComputeWritableAlias {
                        resource_id: buf_id.get(),
                        dispatch_id: 0,
                        subresource: SubresourceRange::WholeBuffer,
                    },
                    "Seed {SEED:#018x} failed at iter {iter}"
                );
            }
            _ => {
                // Copy pass: overlapping source and destination
                let pass_id = PassId::new(rng.next_range(1, 500));
                let buf_id = ResourceId::new(rng.next_range(1, 100));
                let mut pass = Pass::new_copy(pass_id, format!("copy_hazard_{iter}"));

                let copy = CopyCommand::BufferToBuffer {
                    src: buf_id,
                    src_offset: (rng.next_range(0, 10) * 64) as u64,
                    dst: buf_id,
                    dst_offset: (rng.next_range(0, 10) * 64) as u64,
                    size: 64,
                };
                pass = pass.with_copy(copy);

                let err = validate_pass_hazards(&pass).expect_err(&format!(
                    "Overlapping copy endpoints must be rejected for seed {SEED:#018x} at iter {iter}"
                ));
                assert_eq!(
                    err,
                    HazardError::OverlappingCopyEndpoints {
                        resource_id: buf_id.get(),
                    },
                    "Seed {SEED:#018x} failed at iter {iter}"
                );
            }
        }
    }
}

#[test]
fn property_test_disjoint_offset_buffer_writes_rejected_under_whole_buffer_rule() {
    const SEED: u64 = 0x5EED_F3D9_9A55_0003;
    const ITERATIONS: usize = 1_000;
    let mut rng = TestLcg::new(SEED);

    for iter in 0..ITERATIONS {
        let pass_id = PassId::new(rng.next_range(1, 500));
        let buf_id = ResourceId::new(rng.next_range(1, 100));
        let mut pass = Pass::new_render(pass_id, format!("whole_buf_{iter}"));

        // Generate strictly non-overlapping disjoint byte ranges in the same buffer
        let offset_a = (rng.next_range(0, 50) * 256) as u64;
        let size_a = (rng.next_range(1, 4) * 64) as u64;
        let gap = (rng.next_range(1, 10) * 256) as u64;
        let offset_b = offset_a + size_a + gap;
        let size_b = (rng.next_range(1, 4) * 64) as u64;

        // Invariant assertion: bytes are strictly disjoint
        assert!(
            offset_a + size_a <= offset_b,
            "Ranges must be disjoint for seed {SEED:#018x} at iter {iter}"
        );

        // One write (StorageBufferWrite), one read (UniformBuffer)
        let use_a = ResourceUse::buffer_storage_write(
            buf_id,
            DataVersion::INITIAL,
            Some(offset_a),
            Some(size_a),
        );
        let use_b = ResourceUse::buffer_uniform(
            buf_id,
            DataVersion::INITIAL,
            Some(offset_b),
            Some(size_b),
        );

        let draw0 = Draw::new(0, 10, 3, 0, vec![use_a]);
        let draw1 = Draw::new(1, 10, 3, 0, vec![use_b]);
        pass = pass.with_draw(draw0).with_draw(draw1);

        let hazard_err = validate_pass_hazards(&pass).expect_err(&format!(
            "Whole-buffer rule must reject disjoint offsets for seed {SEED:#018x} at iter {iter}"
        ));

        assert_eq!(
            hazard_err,
            HazardError::WholeBufferConflict {
                buffer_id: buf_id.get(),
                access_a: ResourceAccess::StorageBufferWrite,
                access_b: ResourceAccess::UniformBuffer,
                offset_a: Some(offset_a),
                offset_b: Some(offset_b),
            },
            "Expected WholeBufferConflict for seed {SEED:#018x} at iter {iter}"
        );

        // Through PassGraph compilation as well
        let mut graph = PassGraph::new();
        graph.add_pass(pass).expect("add pass");
        let compile_err = graph.compile(None).expect_err(&format!(
            "Compile must reject whole-buffer conflict for seed {SEED:#018x} at iter {iter}"
        ));
        assert_eq!(
            compile_err,
            GraphError::Hazard(HazardError::WholeBufferConflict {
                buffer_id: buf_id.get(),
                access_a: ResourceAccess::StorageBufferWrite,
                access_b: ResourceAccess::UniformBuffer,
                offset_a: Some(offset_a),
                offset_b: Some(offset_b),
            }),
            "Expected GraphError::Hazard(WholeBufferConflict) for seed {SEED:#018x} at iter {iter}"
        );
    }
}

#[test]
fn property_test_duplicate_pass_ids_rejected_before_mutation() {
    const SEED: u64 = 0x5EED_F3D9_9A55_0004;
    const ITERATIONS: usize = 1_000;
    let mut rng = TestLcg::new(SEED);

    for iter in 0..ITERATIONS {
        let mut graph = PassGraph::new();
        let initial_count = rng.next_range(1, 6) as usize;

        let mut existing_ids = Vec::new();
        for i in 0..initial_count {
            let pid = PassId::new((i as u32) + 1);
            existing_ids.push(pid);
            let mut pass = Pass::new_render(pid, format!("init_pass_{i}"));
            pass = pass.with_color_attachment(ColorAttachment::new_clear(
                ResourceId::new((i as u32) + 10),
                [0.0, 0.0, 0.0, 1.0],
            ));
            pass = pass.with_draw(Draw::new(0, 10, 3, 0, vec![]));
            graph.add_pass(pass).expect("add pass");
        }

        // Pick one of the existing IDs to duplicate
        let dup_idx = rng.next_range(0, (initial_count - 1) as u32) as usize;
        let dup_id = existing_ids[dup_idx];

        let duplicate_pass = Pass::new_render(dup_id, "attempted_duplicate");
        let err = graph.add_pass(duplicate_pass).expect_err(&format!(
            "Duplicate PassId {dup_id} must be rejected for seed {SEED:#018x} at iter {iter}"
        ));

        assert_eq!(
            err,
            GraphError::DuplicatePassId {
                pass_id: dup_id.get(),
            },
            "Expected DuplicatePassId for seed {SEED:#018x} at iter {iter}"
        );

        // Verification of "before mutation":
        // 1. Graph compiles successfully with EXACTLY initial_count segments
        let plan = graph.compile(None).expect(&format!(
            "Graph compile must succeed without corruption for seed {SEED:#018x} at iter {iter}"
        ));
        assert_eq!(
            plan.segment_count(),
            initial_count,
            "Segment count must remain initial_count (no mutation) for seed {SEED:#018x} at iter {iter}"
        );

        // 2. Fresh pass can still be added cleanly
        let fresh_id = PassId::new((initial_count as u32) + 100);
        let mut fresh_pass = Pass::new_render(fresh_id, "fresh_pass");
        fresh_pass = fresh_pass.with_color_attachment(ColorAttachment::new_clear(
            ResourceId::new(99),
            [0.0, 0.0, 0.0, 1.0],
        ));
        fresh_pass = fresh_pass.with_draw(Draw::new(0, 10, 3, 0, vec![]));
        assert!(
            graph.add_pass(fresh_pass).is_ok(),
            "Adding fresh pass must succeed after duplicate rejection for seed {SEED:#018x} at iter {iter}"
        );
    }
}

#[test]
fn property_test_dag_compilation_always_yields_valid_topological_order() {
    const SEED: u64 = 0x5EED_F3D9_9A55_0005;
    const ITERATIONS: usize = 1_000;
    let mut rng = TestLcg::new(SEED);

    for iter in 0..ITERATIONS {
        let mut graph = PassGraph::new();
        let pass_count = rng.next_range(2, 8) as usize;

        let mut pass_ids = Vec::new();
        let mut direct_deps: Vec<(PassId, Vec<PassId>)> = Vec::new();

        for i in 0..pass_count {
            let pid = PassId::new((i as u32) + 1);
            pass_ids.push(pid);

            let mut pass = Pass::new_render(pid, format!("dag_pass_{i}"));
            pass = pass.with_color_attachment(ColorAttachment::new_clear(
                ResourceId::new((i as u32) + 10),
                [0.0, 0.0, 0.0, 1.0],
            ));
            pass = pass.with_draw(Draw::new(0, 10, 3, 0, vec![]));

            // Add 0 to multiple dependencies to prior passes (guaranteed DAG)
            let mut deps = Vec::new();
            if i > 0 {
                let dep_count = rng.next_range(0, i as u32) as usize;
                for _ in 0..dep_count {
                    let dep_idx = rng.next_range(0, (i - 1) as u32) as usize;
                    let dep_id = pass_ids[dep_idx];
                    if !deps.contains(&dep_id) {
                        deps.push(dep_id);
                        pass = pass.with_dependency(dep_id);
                    }
                }
            }
            direct_deps.push((pid, deps));
            graph.add_pass(pass).expect("add pass");
        }

        let plan = graph.compile(None).expect(&format!(
            "DAG compilation must succeed for seed {SEED:#018x} at iter {iter}"
        ));

        assert_eq!(
            plan.segment_count(),
            pass_count,
            "Segment count must match pass count for seed {SEED:#018x} at iter {iter}"
        );

        let mut seg_index_map: std::collections::HashMap<PassId, usize> =
            std::collections::HashMap::new();
        for (idx, seg) in plan.segments().iter().enumerate() {
            seg_index_map.insert(seg.pass_id(), idx);
        }

        // Verify topological order invariant: for every dependency (A -> B, B depends on A),
        // A must be scheduled before B (seg_index(A) < seg_index(B))
        for (dependent_id, deps) in &direct_deps {
            let dependent_idx = seg_index_map[dependent_id];
            for dep_id in deps {
                let dep_idx = seg_index_map[dep_id];
                assert!(
                    dep_idx < dependent_idx,
                    "Topological ordering failed for seed {SEED:#018x} at iter {iter}: \
                     dependency {dep_id:?} at {dep_idx} scheduled after dependent {dependent_id:?} at {dependent_idx}"
                );
            }
        }
    }
}

#[test]
fn test_draw_viewport_and_scissor_accessors_and_compilation() {
    let mut graph = PassGraph::new();
    let pass_id = PassId::new(1);
    let mut pass = Pass::new_render(pass_id, "viewport_scissor_pass")
        .with_color_attachment(ColorAttachment::new_clear(ResourceId::new(10), [0.0, 0.0, 0.0, 1.0]));

    // 1. Default draw has None for viewport/scissor and disabled scissor test
    let default_draw = Draw::new(1, 100, 3, 0, vec![]);
    assert_eq!(default_draw.viewport(), None);
    assert_eq!(default_draw.scissor(), None);
    assert!(!default_draw.scissor_test_enabled());

    // 2. Explicit zero-size viewport (near-neighbor case: distinct from None default)
    let draw_explicit_zero = Draw::new(2, 100, 3, 0, vec![])
        .with_viewport_rect([0, 0, 0, 0]);
    assert_eq!(draw_explicit_zero.viewport(), Some([0, 0, 0, 0]));
    assert_ne!(draw_explicit_zero.viewport(), default_draw.viewport());

    // 3. Configure distinct non-zero viewport and scissor state
    let draw_configured = Draw::new(3, 101, 6, 256, vec![])
        .with_viewport_rect([0, 0, 1920, 1080])
        .with_scissor_rect([100, 200, 800, 600]);

    assert_eq!(draw_configured.viewport(), Some([0, 0, 1920, 1080]));
    assert_eq!(draw_configured.viewport_f32(), Some([0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0]));
    assert_eq!(draw_configured.scissor(), Some([100, 200, 800, 600]));
    assert!(draw_configured.scissor_test_enabled());

    // 4. Scissor-disable after enable: emit full-attachment scissor to clear clipping
    let draw_restore_scissor = Draw::new(4, 102, 3, 0, vec![])
        .with_full_attachment_scissor(1920, 1080);
    assert_eq!(draw_restore_scissor.scissor(), Some([0, 0, 1920, 1080]));
    assert!(!draw_restore_scissor.scissor_test_enabled());

    pass = pass
        .with_draw(default_draw)
        .with_draw(draw_explicit_zero)
        .with_draw(draw_configured)
        .with_draw(draw_restore_scissor);
    graph.add_pass(pass).expect("add pass");

    // 5. Compile and verify viewport/scissor preservation in execution plan
    let plan = graph.compile(None).expect("compile plan");
    assert_eq!(plan.segment_count(), 1);
    let draws = plan.segments()[0].draws();
    assert_eq!(draws.len(), 4);

    assert_eq!(draws[0].viewport(), None);
    assert_eq!(draws[0].scissor(), None);
    assert!(!draws[0].scissor_test_enabled());

    assert_eq!(draws[1].viewport(), Some([0, 0, 0, 0]));

    assert_eq!(draws[2].viewport(), Some([0, 0, 1920, 1080]));
    assert_eq!(draws[2].scissor(), Some([100, 200, 800, 600]));
    assert!(draws[2].scissor_test_enabled());

    assert_eq!(draws[3].scissor(), Some([0, 0, 1920, 1080]));
    assert!(!draws[3].scissor_test_enabled());
}

#[test]
fn test_draw_with_range_preserves_explicit_parameters() {
    let draw = Draw::new(1, 10, 0, 0, vec![])
        .with_range([3, 2, 3, 5]);

    assert_eq!(draw.vertex_count, 3);
    assert_eq!(draw.instance_count(), 2);
    assert_eq!(draw.first_vertex(), 3);
    assert_eq!(draw.first_instance(), 5);

    let draw_zero = Draw::new(2, 10, 0, 0, vec![])
        .with_range([6, 0, 0, 0]);

    assert_eq!(draw_zero.vertex_count, 6);
    assert_eq!(draw_zero.instance_count(), 0);
    assert_eq!(draw_zero.first_vertex(), 0);
    assert_eq!(draw_zero.first_instance(), 0);
}

#[test]
fn positive_readonly_depth_stencil_and_sampled_view_coexistence_accepted() {
    assert!(ResourceAccess::ReadOnlyDepthStencil.is_compatible_with(&ResourceAccess::SampledTexture));
    assert!(ResourceAccess::SampledTexture.is_compatible_with(&ResourceAccess::ReadOnlyDepthStencil));

    let tex_depth = ResourceId::new(200);
    let mut pass = Pass::new_render(PassId::new(1), "readonly_depth_sampling_pass");
    pass.depth_stencil_attachment = Some(DepthStencilAttachment {
        target_id: tex_depth,
        view_subresource: SubresourceRange::full_texture(),
        depth_load_op: None,
        depth_store_op: None,
        depth_clear_value: 1.0,
        depth_read_only: true,
        stencil_load_op: None,
        stencil_store_op: None,
        stencil_clear_value: 0,
        stencil_read_only: true,
    });
    pass.draws.push(Draw::new(
        0, 1, 3, 0,
        vec![ResourceUse::texture_sampled(
            tex_depth,
            DataVersion::INITIAL,
            SubresourceRange::full_texture(),
        )],
    ));

    assert!(validate_pass_hazards(&pass).is_ok());

    let mut graph = PassGraph::new();
    graph.add_pass(pass).expect("add pass");
    let plan = graph.compile(None).expect("compile plan");
    assert_eq!(plan.split_count(), 0);
    assert_eq!(plan.segment_count(), 1);
}

#[test]
fn negative_writable_depth_stencil_and_sampled_view_rejected() {
    assert!(!ResourceAccess::DepthStencilAttachment.is_compatible_with(&ResourceAccess::SampledTexture));
    assert!(!ResourceAccess::SampledTexture.is_compatible_with(&ResourceAccess::DepthStencilAttachment));

    let tex_depth = ResourceId::new(201);
    let mut pass = Pass::new_render(PassId::new(1), "writable_depth_sampling_pass");
    pass.depth_stencil_attachment = Some(DepthStencilAttachment {
        target_id: tex_depth,
        view_subresource: SubresourceRange::full_texture(),
        depth_load_op: Some(LoadOp::Clear),
        depth_store_op: Some(StoreOp::Store),
        depth_clear_value: 1.0,
        depth_read_only: false,
        stencil_load_op: None,
        stencil_store_op: None,
        stencil_clear_value: 0,
        stencil_read_only: true,
    });
    pass.draws.push(Draw::new(
        0, 1, 3, 0,
        vec![ResourceUse::texture_sampled(
            tex_depth,
            DataVersion::INITIAL,
            SubresourceRange::full_texture(),
        )],
    ));

    assert!(matches!(
        validate_pass_hazards(&pass),
        Err(HazardError::AttachmentSamplingConflict { texture_id, .. }) if texture_id == tex_depth.get()
    ));

    let mut graph = PassGraph::new();
    graph.add_pass(pass).expect("add pass");
    assert!(matches!(
        graph.compile(None),
        Err(GraphError::Hazard(HazardError::AttachmentSamplingConflict { texture_id, .. })) if texture_id == tex_depth.get()
    ));
}
