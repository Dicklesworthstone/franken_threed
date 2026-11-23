//! State at an introduced render-pass boundary must match the source draw.

use f3d_core::ownership::DataVersion;
use f3d_graph::{
    ColorAttachment, DepthStencilAttachment, Draw, LoadOp, Pass, PassGraph, PassId, ResourceId,
    ResourceUse, StoreOp, split_pass_on_hazard, validate_pass_hazards,
};

fn alternating_pass() -> Pass {
    let mut pass = Pass::new_render(PassId::new(1), "inherited_state").with_color_attachment(
        ColorAttachment::new_clear(ResourceId::new(100), [0.1, 0.2, 0.3, 1.0]),
    );
    for index in 0..12 {
        let version = DataVersion::new(index + 1);
        let shared = ResourceId::new(50);
        let use_buffer = if index / 2 % 2 == 0 {
            ResourceUse::buffer_uniform(shared, version, Some(index * 256), Some(256))
        } else {
            ResourceUse::buffer_storage_write(shared, version, Some(index * 256), Some(256))
        };
        let mut draw = Draw::new(
            index as u32,
            200 + index as u32,
            3,
            index as u32 * 256,
            vec![
                use_buffer,
                ResourceUse::buffer_vertex(
                    ResourceId::new(60),
                    version,
                    Some(index * 64),
                    Some(64),
                ),
            ],
        );
        draw.instance_count = 2 + index as u32;
        draw.first_vertex = index as u32;
        draw.first_instance = 4 + index as u32;
        pass.draws.push(draw);
    }
    pass
}

#[test]
fn split_boundaries_materialize_raster_state_and_rebind_complete_draw_records() {
    let mut pass = alternating_pass();
    pass.draws[0].viewport = Some([2, 3, 60, 50]);
    pass.draws[0].scissor = Some([4, 5, 20, 25]);
    pass.draws[0].scissor_test_enabled = true;
    pass.draws[2].assumes_warm_state = true;
    pass.draws[3].viewport = Some([6, 7, 30, 40]);
    // Bundles inherit raster state but contain their own bindings.
    pass.draws[4] = Draw::new_bundle(4, 42, 204, pass.draws[4].uses.clone());
    pass.draws[6] = pass.draws[6].clone().with_full_attachment_scissor(64, 64);
    pass.draws[7].assumes_warm_state = true;
    pass.draws[8].scissor_test_enabled = true; // Inherits the full attachment.
    pass.draws[9].viewport = Some([0, 0, 0, 0]);
    pass.draws[9].scissor = Some([10, 11, 0, 12]);
    pass.draws[9].scissor_test_enabled = true;
    pass.draws[10].assumes_warm_state = true;
    let original = pass.clone();

    let segments = split_pass_on_hazard(&pass, PassId::new(2)).unwrap();
    assert_eq!(
        pass, original,
        "compilation cannot mutate source declarations"
    );
    assert_eq!(segments.len(), 6);
    let first: Vec<_> = segments.iter().map(|segment| &segment.draws[0]).collect();
    assert_eq!(
        first.iter().map(|draw| draw.viewport).collect::<Vec<_>>(),
        vec![
            Some([2, 3, 60, 50]),
            Some([2, 3, 60, 50]),
            Some([6, 7, 30, 40]),
            Some([6, 7, 30, 40]),
            Some([6, 7, 30, 40]),
            Some([0, 0, 0, 0]),
        ]
    );
    for draw in &first[..3] {
        assert_eq!(draw.scissor, Some([4, 5, 20, 25]));
        assert!(draw.scissor_test_enabled);
    }
    assert_eq!(first[3].scissor, Some([0, 0, 64, 64]));
    assert!(!first[3].scissor_test_enabled);
    assert_eq!(
        first[4].scissor, None,
        "disabled clipping must not reappear"
    );
    assert_eq!(first[5].scissor, Some([10, 11, 0, 12]));
    assert!(
        first[5].scissor_test_enabled,
        "empty scissors are real state"
    );

    for segment in &segments {
        validate_pass_hazards(segment).expect("each resumed scope must be legal");
        assert_eq!(segment.draws.len(), 2);
        for (index, actual) in segment.draws.iter().enumerate() {
            let expected = &original.draws[actual.draw_id as usize];
            if index == 1 {
                assert_eq!(actual, expected, "interior draws keep their original state");
            }
            assert_eq!(actual.pipeline_id, expected.pipeline_id);
            assert_eq!(
                actual.uses, expected.uses,
                "versions, ranges and roles cannot change"
            );
            assert_eq!(
                actual.uniform_dynamic_offset,
                expected.uniform_dynamic_offset
            );
            assert_eq!(actual.kind, expected.kind);
            assert_eq!(actual.vertex_count, expected.vertex_count);
            assert_eq!(actual.instance_count, expected.instance_count);
            assert_eq!(actual.first_vertex, expected.first_vertex);
            assert_eq!(actual.first_instance, expected.first_instance);
        }
        assert!(!segment.draws[0].assumes_warm_state);
    }
    assert!(
        segments[3].draws[1].assumes_warm_state,
        "interior reuse stays legal"
    );
}

#[test]
fn boundary_overrides_win_over_inherited_raster_state() {
    let mut pass = alternating_pass();
    pass.draws.truncate(4);
    pass.draws[0] = pass.draws[0]
        .clone()
        .with_viewport_rect([1, 2, 30, 40])
        .with_scissor_rect([3, 4, 10, 20]);
    pass.draws[2] = pass.draws[2]
        .clone()
        .with_viewport_rect([5, 6, 20, 10])
        .with_scissor_rect([7, 8, 9, 10]);
    let segments = split_pass_on_hazard(&pass, PassId::new(2)).unwrap();
    assert_eq!(segments[1].draws[0], pass.draws[2]);
    assert_eq!(segments[1].draws[1], pass.draws[3]);
}

#[test]
fn resumed_raster_state_preserves_all_attachment_load_store_and_final_resolve_operations() {
    let mut pass = alternating_pass();
    pass.draws.truncate(6);
    pass.draws[0] = pass.draws[0]
        .clone()
        .with_viewport_rect([3, 4, 50, 40])
        .with_scissor_rect([5, 6, 20, 30]);
    pass.draws[2].assumes_warm_state = true;
    pass.draws[4].assumes_warm_state = true;
    pass.color_attachments[0].store_op = StoreOp::Discard;
    pass.color_attachments[0].resolve_target = Some(ResourceId::new(102));
    pass.color_attachments
        .push(ColorAttachment::new_load(ResourceId::new(101)));
    let mut depth = DepthStencilAttachment::new_depth_clear(ResourceId::new(103), 0.75);
    depth.depth_store_op = Some(StoreOp::Discard);
    depth.stencil_read_only = false;
    depth.stencil_load_op = Some(LoadOp::Clear);
    depth.stencil_store_op = Some(StoreOp::Discard);
    depth.stencil_clear_value = 9;
    pass.depth_stencil_attachment = Some(depth.clone());
    let mut graph = PassGraph::new();
    graph.add_pass(pass.clone()).unwrap();
    let plan = graph.compile(None).unwrap();
    assert_eq!(plan.split_count(), 2);
    for (index, segment) in plan.segments().iter().enumerate() {
        assert_eq!(segment.draws()[0].viewport, pass.draws[0].viewport);
        assert_eq!(segment.draws()[0].scissor, pass.draws[0].scissor);
        let mut expected_colors = pass.color_attachments.clone();
        let mut expected_depth = depth.clone();
        if index > 0 {
            for color in &mut expected_colors {
                color.load_op = LoadOp::Load;
            }
            expected_depth.depth_load_op = Some(LoadOp::Load);
            expected_depth.stencil_load_op = Some(LoadOp::Load);
        }
        if index < 2 {
            for color in &mut expected_colors {
                color.store_op = StoreOp::Store;
                color.resolve_target = None;
            }
            expected_depth.depth_store_op = Some(StoreOp::Store);
            expected_depth.stencil_store_op = Some(StoreOp::Store);
        }
        assert_eq!(segment.color_attachments(), expected_colors);
        assert_eq!(segment.depth_stencil_attachment(), Some(&expected_depth));
    }
}
