//! `f3d-core`: Typed IDs, layouts, capability records, errors, and feature manifests.
//!
//! Position in dependency direction: bottom (`core <- everything`).
//! First-party semantic, numerical, compiler, scene, and resource code uses `#![forbid(unsafe_code)]`.
//! Core library code never prints casually to stdout/stderr.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod capability;
pub mod error;
pub mod handle;
pub mod layout;
pub mod manifest;
pub mod ownership;
#[cfg(feature = "test-support")]
pub mod test_evidence;

pub use capability::{AdapterInfo, CapabilityRecord, DeviceLimits};
pub use error::{F3dError, HandleError, SourceSpan};
pub use handle::{
    Arena, AttributeDomain, BundleDomain, DeviceGeneration, Domain, GeometryDomain, GpuHandle,
    Handle, LifecycleState, MaterialDomain, ObjectDomain, PipelineDomain, RegionDomain,
    RenderTargetDomain, TaskDomain, TextureDomain,
};
pub use layout::{
    AFFINE_ROWS_ALIGNMENT, AFFINE_ROWS_BYTES, AffineRows, COLOR_UNIFORM_BYTES,
    COPY_BYTES_PER_ROW_ALIGNMENT,
    DEFAULT_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT, DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT,
    DrawIndexedIndirectArgs, DrawIndirectArgs, GpuMatrixLayout, InstanceRecord, LAYOUT_TABLE,
    LayoutError, LayoutRow, LayoutTable, dump_layouts, layout_table,
    PROJECTIVE_MAT4_ALIGNMENT, PROJECTIVE_MAT4_BYTES, ProjectiveMat4, WGSL_MAT4X3_BYTES,
    VERTEX_POS_COLOR_ALIGNMENT, VERTEX_POS_COLOR_BYTES, VERTEX_POS_COLOR_STRIDE,
    VERTEX_POS_NORMAL_UV_ALIGNMENT, VERTEX_POS_NORMAL_UV_BYTES, VERTEX_POS_NORMAL_UV_STRIDE,
    VERTEX_POS_UV_BYTES, VERTEX_POS_UV_STRIDE, VertexPosColor, VertexPosNormalUv, VertexPosUv,
    WRITE_BUFFER_ALIGNMENT,
    aligned_bytes_per_row, aligned_copy_bytes_per_row,
    is_matrix4_affine, is_matrix4_f64_affine, validate_affine_target,
    validate_composite_storage_array_stride,
    validate_copy_bytes_per_row, validate_dynamic_storage_offset,
    validate_dynamic_uniform_offset, validate_storage_array_stride,
    validate_write_buffer_alignment,
};
pub use manifest::{
    FeatureEntry, FeatureManifest, FeatureRoute, FeatureStatus, PINNED_UPSTREAM_COMMIT,
    PINNED_UPSTREAM_RELEASE,
};
pub use ownership::{
    Author, BorrowScope, BorrowState, BorrowToken, CopyAccounting, DataVersion, Epoch, OwnerMode,
    OwnershipError, PerUseByteBuffer, PerUseSnapshotStore, RegionState, SnapshotEntry, UseRecord,
};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use core::num::NonZeroU32;

    #[test]
    fn handle_word_packing_and_unpacking() {
        let index = 42;
        let generation_val = NonZeroU32::new(7).unwrap();
        let handle: Handle<ObjectDomain> = Handle::new(index, generation_val);

        assert_eq!(handle.index(), 42);
        assert_eq!(handle.generation(), generation_val);

        let (w0, w1) = handle.to_words();
        assert_eq!(w0, 42);
        assert_eq!(w1, 7);

        let restored = Handle::<ObjectDomain>::from_words(w0, w1).expect("valid words");
        assert_eq!(handle, restored);

        let packed = handle.pack_u64();
        let unpacked = Handle::<ObjectDomain>::unpack_u64(packed).expect("valid u64");
        assert_eq!(handle, unpacked);
    }

    #[test]
    fn handle_invalid_generation_rejected() {
        let err = Handle::<ObjectDomain>::from_raw(10, 0).expect_err("generation 0 is invalid");
        assert_eq!(err, HandleError::InvalidGeneration { raw_generation: 0 });
    }

    #[test]
    fn arena_insert_get_remove_lifecycle() {
        let mut arena: Arena<ObjectDomain, String> = Arena::new();

        let h1 = arena.insert(String::from("mesh_1")).expect("insert ok");
        assert_eq!(h1.index(), 0);
        assert_eq!(h1.generation().get(), 1);

        assert_eq!(arena.get(h1).expect("lookup ok"), "mesh_1");
        assert_eq!(
            arena.get_lifecycle(h1).expect("lifecycle ok"),
            LifecycleState::CpuExistent
        );

        arena
            .set_lifecycle(h1, LifecycleState::GpuAllocated)
            .expect("set lifecycle ok");
        assert_eq!(
            arena.get_lifecycle(h1).expect("lifecycle ok"),
            LifecycleState::GpuAllocated
        );

        let removed = arena.remove(h1).expect("remove ok");
        assert_eq!(removed, "mesh_1");

        // Stale handle access is rejected
        let lookup_err = arena.get(h1).expect_err("stale handle must fail");
        assert_eq!(
            lookup_err,
            HandleError::GenerationMismatch {
                index: 0,
                expected_generation: 1,
                actual_generation: 2,
            }
        );

        // Reusing slot yields incremented generation
        let h2 = arena.insert(String::from("mesh_2")).expect("re-insert ok");
        assert_eq!(h2.index(), 0);
        assert_eq!(h2.generation().get(), 2);
        assert_eq!(arena.get(h2).expect("lookup ok"), "mesh_2");
    }

    #[test]
    fn arena_exhaustion_at_u32_max_prevents_slot_resurrection() {
        let mut arena: Arena<ObjectDomain, String> = Arena::new();

        let h0 = arena.insert(String::from("first")).expect("insert 0");
        assert_eq!(h0.index(), 0);

        // Force slot 0 generation to u32::MAX
        let max_generation = NonZeroU32::new(u32::MAX).unwrap();
        arena.force_set_slot_generation_for_test(0, max_generation);
        let h0_max: Handle<ObjectDomain> = Handle::new(0, max_generation);

        // Remove slot 0 at boundary: must become Exhausted rather than Retired
        let removed = arena.remove(h0_max).expect("remove boundary");
        assert_eq!(removed, "first");

        // Slot 0 is now exhausted; subsequent insert must allocate a NEW index (index 1), NOT slot 0
        let h1 = arena.insert(String::from("second")).expect("insert 1");
        assert_eq!(
            h1.index(),
            1,
            "Exhausted slot 0 must not be resurrected; arena must allocate index 1"
        );

        // Old handle is rejected
        let err = arena.get(h0_max).expect_err("exhausted slot lookup");
        assert_eq!(err, HandleError::SlotVacant { index: 0 });
    }

    #[test]
    fn gpu_handle_device_generation_validation() {
        let dev1 = DeviceGeneration::INITIAL;
        let dev2 = dev1.next().expect("next generation");
        assert_ne!(dev1, dev2);

        let h: Handle<TextureDomain> = Handle::new(5, NonZeroU32::new(1).unwrap());
        let gpu_handle = GpuHandle::new(h, dev1);

        assert_eq!(gpu_handle.validate_device(dev1).expect("dev1 matches"), h);

        let err = gpu_handle.validate_device(dev2).expect_err("dev2 mismatch");
        assert_eq!(
            err,
            HandleError::DeviceGenerationMismatch {
                expected_device: dev1.get(),
                current_device: dev2.get(),
            }
        );
    }

    #[test]
    fn manifest_blocking_status_logic() {
        assert!(FeatureStatus::Unclassified.is_blocking());
        assert!(FeatureStatus::Unimplemented.is_blocking());
        assert!(FeatureStatus::Untested.is_blocking());
        assert!(FeatureStatus::KnownRegression.is_blocking());
        assert!(FeatureStatus::Stub.is_blocking());
        assert!(FeatureStatus::NoOpSubstitute.is_blocking());
        assert!(FeatureStatus::CandidateRefusalOnValidSource.is_blocking());

        assert!(!FeatureStatus::Verified.is_blocking());
        assert!(!FeatureStatus::HostBlocked.is_blocking());

        let mut manifest = FeatureManifest::new_pinned();
        assert_eq!(manifest.upstream_release, PINNED_UPSTREAM_RELEASE);
        assert_eq!(manifest.upstream_commit, PINNED_UPSTREAM_COMMIT);
        assert!(!manifest.has_blocking_features());

        manifest.add_feature(FeatureEntry {
            symbol: String::from("MeshStandardMaterial"),
            category: String::from("material"),
            route: FeatureRoute::SpecializedWebGpu,
            status: FeatureStatus::Unimplemented,
            notes: None,
        });
        assert!(manifest.has_blocking_features());
        assert_eq!(manifest.count_blocking_features(), 1);
        assert_eq!(manifest.count_by_route(FeatureRoute::SpecializedWebGpu), 1);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn handle_serde_domain_mismatch_rejected() {
        let geom_handle: Handle<GeometryDomain> = Handle::from_raw(3, 10).unwrap();
        let json = serde_json::to_string(&geom_handle).expect("serialize geometry handle");

        // Deserializing geometry handle JSON as an ObjectDomain handle must fail
        let result: Result<Handle<ObjectDomain>, _> = serde_json::from_str(&json);
        assert!(result.is_err(), "Domain mismatch must fail deserialization");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("domain mismatch"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn capability_record_serde_roundtrip() {
        let cap = CapabilityRecord::unknown();
        let json = serde_json::to_string(&cap).expect("serialize");
        let restored: CapabilityRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cap, restored);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn manifest_serde_roundtrip() {
        let mut manifest = FeatureManifest::new_pinned();
        manifest.add_feature(FeatureEntry {
            symbol: String::from("BufferGeometry"),
            category: String::from("geometry"),
            route: FeatureRoute::SpecializedWebGpu,
            status: FeatureStatus::Verified,
            notes: Some(String::from("Interleaved and non-interleaved validated")),
        });

        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let restored: FeatureManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(manifest, restored);
    }
}
