//! Texture storage roles must obey both pass scopes and individual draw aliases.

use f3d_core::ownership::DataVersion;
use f3d_graph::{
    ColorAttachment, DepthStencilAttachment, Draw, GraphError, HazardError, LoadOp, Pass,
    PassGraph, PassId, ResourceAccess, ResourceId, ResourceKind, ResourceUse, StoreOp,
    SubresourceRange, TextureAspect, can_split_pass, split_pass_on_hazard, validate_pass_hazards,
};

fn texture_use(access: ResourceAccess, mip: u32, layer: u32) -> ResourceUse {
    let mut usage = ResourceUse::texture_sampled(
        ResourceId::new(50),
        DataVersion::INITIAL,
        SubresourceRange::single_mip_layer(mip, layer, TextureAspect::Color),
    );
    usage.access = access;
    usage
}

fn render(uses: Vec<Vec<ResourceUse>>) -> Pass {
    let mut pass = Pass::new_render(PassId::new(1), "texture_roles").with_color_attachment(
        ColorAttachment::new_clear(ResourceId::new(100), [0.1, 0.2, 0.3, 1.0]),
    );
    for (index, uses) in uses.into_iter().enumerate() {
        pass.draws.push(Draw::new(index as u32, 1, 3, 0, uses));
    }
    pass
}

#[test]
fn storage_and_sampled_roles_require_separate_scopes_in_either_order() {
    for roles in [
        [
            ResourceAccess::StorageTextureWrite,
            ResourceAccess::SampledTexture,
        ],
        [
            ResourceAccess::SampledTexture,
            ResourceAccess::StorageTextureWrite,
        ],
    ] {
        let uses: Vec<_> = roles
            .into_iter()
            .map(|role| texture_use(role, 0, 0))
            .collect();
        let separate = render(uses.iter().cloned().map(|usage| vec![usage]).collect());
        assert!(matches!(
            validate_pass_hazards(&separate),
            Err(HazardError::TextureUsageConflict { texture_id: 50, .. })
        ));
        assert!(can_split_pass(&separate));
        let segments = split_pass_on_hazard(&separate, PassId::new(2)).unwrap();
        assert_eq!(segments.len(), 2);
        for segment in &segments {
            validate_pass_hazards(segment).unwrap();
        }

        let aliased = render(vec![uses]);
        assert!(validate_pass_hazards(&aliased).is_err());
        assert!(!can_split_pass(&aliased), "a draw is never divided");
        let mut graph = PassGraph::new();
        graph.add_pass(aliased).unwrap();
        assert!(matches!(graph.compile(None), Err(GraphError::Hazard(_))));
    }
}

#[test]
fn storage_only_draws_share_a_scope_but_writable_aliases_do_not() {
    let usage = texture_use(ResourceAccess::StorageTextureWrite, 0, 0);
    assert!(usage.access.is_compatible_with(&usage.access));
    let pass = render(vec![vec![usage.clone()], vec![usage.clone()]]);
    validate_pass_hazards(&pass).unwrap();
    let mut graph = PassGraph::new();
    graph.add_pass(pass).unwrap();
    let plan = graph.compile(None).unwrap();
    assert_eq!(plan.segment_count(), 1);
    assert_eq!(plan.split_count(), 0);

    let mut aliased = render(vec![vec![usage.clone(), usage.clone()]]);
    assert!(matches!(
        validate_pass_hazards(&aliased),
        Err(HazardError::DrawWritableAlias {
            resource_id: 50,
            draw_id: 0,
            ..
        })
    ));
    assert!(!can_split_pass(&aliased));
    // An opaque bundle record does not prove separate internal draw scopes.
    aliased.draws[0] = Draw::new_bundle(0, 7, 1, vec![usage.clone(), usage]);
    assert!(validate_pass_hazards(&aliased).is_err());
}

#[test]
fn writable_canvas_views_obey_the_same_draw_alias_rules() {
    for kinds in [
        [ResourceKind::CanvasOutput, ResourceKind::CanvasOutput],
        [ResourceKind::CanvasOutput, ResourceKind::Texture],
        [ResourceKind::Texture, ResourceKind::CanvasOutput],
    ] {
        let uses = kinds
            .into_iter()
            .map(|kind| {
                let mut usage = texture_use(ResourceAccess::StorageTextureWrite, 0, 0);
                usage.kind = kind;
                usage
            })
            .collect();
        let pass = render(vec![uses]);
        assert!(matches!(
            validate_pass_hazards(&pass),
            Err(HazardError::DrawWritableAlias {
                resource_id: 50,
                ..
            })
        ));
        assert!(!can_split_pass(&pass));
    }
}

#[test]
fn texture_and_canvas_tags_share_cross_draw_usage_scopes() {
    for kinds in [
        [ResourceKind::CanvasOutput, ResourceKind::Texture],
        [ResourceKind::Texture, ResourceKind::CanvasOutput],
    ] {
        for roles in [
            [
                ResourceAccess::StorageTextureWrite,
                ResourceAccess::SampledTexture,
            ],
            [
                ResourceAccess::SampledTexture,
                ResourceAccess::StorageTextureWrite,
            ],
        ] {
            let uses = kinds
                .into_iter()
                .zip(roles)
                .map(|(kind, role)| {
                    let mut usage = texture_use(role, 0, 0);
                    usage.kind = kind;
                    vec![usage]
                })
                .collect();
            let pass = render(uses);
            assert!(matches!(
                validate_pass_hazards(&pass),
                Err(HazardError::TextureUsageConflict { .. })
            ));
            assert!(can_split_pass(&pass));
            let segments = split_pass_on_hazard(&pass, PassId::new(2)).unwrap();
            assert_eq!(segments.len(), 2);
            for segment in &segments {
                validate_pass_hazards(segment).unwrap();
            }
        }
    }
}

#[test]
fn disjoint_texture_views_remain_legal_and_full_views_still_conflict() {
    let write = texture_use(ResourceAccess::StorageTextureWrite, 2, 3);
    for access in [
        ResourceAccess::SampledTexture,
        ResourceAccess::StorageTextureWrite,
    ] {
        for (mip, layer) in [(1, 3), (2, 2), (3, 4)] {
            let pass = render(vec![vec![write.clone(), texture_use(access, mip, layer)]]);
            validate_pass_hazards(&pass).unwrap();
            assert!(!can_split_pass(&pass));
        }
    }

    let mut full = texture_use(ResourceAccess::SampledTexture, 0, 0);
    full.subresource = SubresourceRange::full_texture();
    let pass = render(vec![vec![write], vec![full]]);
    assert!(matches!(
        validate_pass_hazards(&pass),
        Err(HazardError::TextureUsageConflict { .. })
    ));
    assert!(can_split_pass(&pass));
}

#[test]
fn texture_role_splits_preserve_outputs_versions_raster_state_and_dependencies() {
    let mut pass = render(
        (0..6)
            .map(|index| {
                let access = if index / 2 == 1 {
                    ResourceAccess::SampledTexture
                } else {
                    ResourceAccess::StorageTextureWrite
                };
                let mut usage = texture_use(access, 1, 2);
                usage.version = DataVersion::new(index + 1);
                vec![usage]
            })
            .collect(),
    );
    pass.draws[0] = pass.draws[0]
        .clone()
        .with_viewport_rect([2, 3, 40, 50])
        .with_scissor_rect([4, 5, 20, 30]);
    pass.draws[2].assumes_warm_state = true;
    pass.draws[4].assumes_warm_state = true;
    pass.color_attachments[0].store_op = StoreOp::Discard;
    pass.color_attachments[0].resolve_target = Some(ResourceId::new(101));
    let mut depth = DepthStencilAttachment::new_depth_clear(ResourceId::new(102), 0.75);
    depth.depth_store_op = Some(StoreOp::Discard);
    pass.depth_stencil_attachment = Some(depth);

    let mut graph = PassGraph::new();
    graph.add_pass(pass.clone()).unwrap();
    graph
        .add_pass(Pass::new_copy(PassId::new(2), "consumer").with_dependency(pass.id))
        .unwrap();
    let plan = graph.compile(None).unwrap();
    assert_eq!(plan.split_count(), 2);
    assert_eq!(plan.segment_count(), 4);
    assert_eq!(plan.segments()[3].pass_id(), PassId::new(2));
    for (index, segment) in plan.segments()[..3].iter().enumerate() {
        assert_eq!(segment.draw_count(), 2);
        assert_eq!(segment.draws()[0].viewport, pass.draws[0].viewport);
        assert_eq!(segment.draws()[0].scissor, pass.draws[0].scissor);
        assert!(!segment.draws()[0].assumes_warm_state);
        for (offset, draw) in segment.draws().iter().enumerate() {
            assert_eq!(draw.uses, pass.draws[index * 2 + offset].uses);
            assert_eq!(draw.draw_id, (index * 2 + offset) as u32);
        }
        let color = segment.primary_color_attachment().unwrap();
        assert_eq!(color.target_id, ResourceId::new(100));
        assert_eq!(
            color.load_op,
            if index == 0 {
                LoadOp::Clear
            } else {
                LoadOp::Load
            }
        );
        assert_eq!(
            color.store_op,
            if index == 2 {
                StoreOp::Discard
            } else {
                StoreOp::Store
            }
        );
        assert_eq!(
            color.resolve_target,
            if index == 2 {
                Some(ResourceId::new(101))
            } else {
                None
            }
        );
        let depth = segment.depth_stencil_attachment().unwrap();
        assert_eq!(
            depth.depth_load_op,
            Some(if index == 0 {
                LoadOp::Clear
            } else {
                LoadOp::Load
            })
        );
        assert_eq!(
            depth.depth_store_op,
            Some(if index == 2 {
                StoreOp::Discard
            } else {
                StoreOp::Store
            })
        );
    }
}

#[test]
fn fixed_attachments_cannot_be_removed_to_legalize_storage_access() {
    let storage = texture_use(ResourceAccess::StorageTextureWrite, 0, 0);
    let mut pass = render(vec![vec![storage]]);
    let mut depth = DepthStencilAttachment::new_depth_clear(ResourceId::new(50), 1.0);
    depth.depth_read_only = true;
    depth.depth_load_op = None;
    depth.depth_store_op = None;
    pass.depth_stencil_attachment = Some(depth);
    assert!(matches!(
        validate_pass_hazards(&pass),
        Err(HazardError::TextureUsageConflict { .. })
    ));
    assert!(!can_split_pass(&pass));

    pass.draws[0].uses[0].access = ResourceAccess::SampledTexture;
    validate_pass_hazards(&pass).expect("read-only depth can be sampled");
    pass.depth_stencil_attachment = None;
    pass.color_attachments[0].target_id = ResourceId::new(50);
    pass.draws[0].uses[0].access = ResourceAccess::StorageTextureWrite;
    assert!(matches!(
        validate_pass_hazards(&pass),
        Err(HazardError::MultipleWriters { .. })
    ));
    assert!(!can_split_pass(&pass));
}

#[test]
fn another_split_reason_cannot_hide_resolve_target_feedback() {
    let mut pass = render(vec![
        vec![texture_use(ResourceAccess::StorageTextureWrite, 0, 0)],
        vec![texture_use(ResourceAccess::SampledTexture, 0, 0)],
    ]);
    pass.color_attachments[0].resolve_target = Some(ResourceId::new(60));
    pass.draws[0].uses.push(ResourceUse::texture_sampled(
        ResourceId::new(60),
        DataVersion::INITIAL,
        SubresourceRange::full_texture(),
    ));
    assert!(matches!(
        validate_pass_hazards(&pass),
        Err(HazardError::AttachmentSamplingConflict { texture_id: 60, .. })
    ));
    assert!(!can_split_pass(&pass));
}
