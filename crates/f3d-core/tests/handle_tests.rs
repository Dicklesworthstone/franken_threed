use f3d_core::*;

#[test]
fn handle_abi_words_roundtrip() {
    let handle: Handle<GeometryDomain> = Handle::from_raw(1024, 88).expect("valid handle");
    let (w0, w1) = handle.to_words();
    assert_eq!(w0, 1024);
    assert_eq!(w1, 88);

    let reconstructed = Handle::<GeometryDomain>::from_words(w0, w1).expect("valid words");
    assert_eq!(handle, reconstructed);
    assert_eq!(handle.index(), 1024);
    assert_eq!(handle.generation().get(), 88);
}

#[test]
fn handle_pack_unpack_u64() {
    let handle: Handle<MaterialDomain> = Handle::from_raw(4096, 500).expect("valid handle");
    let packed = handle.pack_u64();
    let unpacked = Handle::<MaterialDomain>::unpack_u64(packed).expect("valid packed u64");
    assert_eq!(handle, unpacked);
}

#[test]
fn arena_multiple_inserts_and_reuse() {
    let mut arena: Arena<ObjectDomain, &'static str> = Arena::new();

    let h0 = arena.insert("node_0").expect("insert 0");
    let h1 = arena.insert("node_1").expect("insert 1");
    let h2 = arena.insert("node_2").expect("insert 2");

    assert_eq!(h0.index(), 0);
    assert_eq!(h1.index(), 1);
    assert_eq!(h2.index(), 2);

    assert_eq!(*arena.get(h0).expect("lookup"), "node_0");
    assert_eq!(*arena.get(h1).expect("lookup"), "node_1");
    assert_eq!(*arena.get(h2).expect("lookup"), "node_2");

    // Remove middle slot
    let removed = arena.remove(h1).expect("remove h1");
    assert_eq!(removed, "node_1");

    // Re-accessing h1 is generation mismatch
    let err = arena.get(h1).expect_err("h1 stale");
    assert_eq!(
        err,
        HandleError::GenerationMismatch {
            index: 1,
            expected_generation: 1,
            actual_generation: 2,
        }
    );

    // Insert new item should reuse index 1 with generation 2
    let h1_new = arena.insert("node_1_v2").expect("reuse slot 1");
    assert_eq!(h1_new.index(), 1);
    assert_eq!(h1_new.generation().get(), 2);
    assert_eq!(*arena.get(h1_new).expect("lookup"), "node_1_v2");
}

#[test]
fn gpu_handle_device_generation_lifecycle() {
    let dev1 = DeviceGeneration::INITIAL;
    let dev2 = dev1.next().expect("dev2");
    let dev3 = dev2.next().expect("dev3");

    let h: Handle<TextureDomain> = Handle::from_raw(3, 1).expect("handle");
    let gpu_handle = GpuHandle::new(h, dev2);

    // dev1 is stale
    let err1 = gpu_handle.validate_device(dev1).expect_err("dev1 mismatch");
    assert_eq!(
        err1,
        HandleError::DeviceGenerationMismatch {
            expected_device: 2,
            current_device: 1,
        }
    );

    // dev2 matches
    assert_eq!(gpu_handle.validate_device(dev2).expect("valid"), h);

    // dev3 is newer (device was lost and recreated)
    let err3 = gpu_handle.validate_device(dev3).expect_err("dev3 mismatch");
    assert_eq!(
        err3,
        HandleError::DeviceGenerationMismatch {
            expected_device: 2,
            current_device: 3,
        }
    );
}

#[test]
fn structured_error_formatting_and_spans() {
    let span = SourceSpan {
        file: String::from("src/render.ts"),
        start_line: 120,
        start_col: 5,
        end_line: 120,
        end_col: 30,
    };

    let err = F3dError::SpecializationRefusal {
        reason: String::from("aliased dynamic material property write"),
        span: Some(span.clone()),
    };

    let formatted = format!("{err}");
    assert!(formatted.contains("src/render.ts:120:5-120:30"));
    assert!(formatted.contains("aliased dynamic material property write"));
}

#[test]
fn capability_record_limits() {
    let record = CapabilityRecord::unknown();
    assert_eq!(record.host_environment, "unknown");
    assert!(!record.webgpu_supported);
    assert!(record.limits.is_none(), "unknown() must not fabricate limits");
    assert_eq!(record.preferred_canvas_format, "unknown");
    assert_eq!(record.color_space, "unknown", "unknown() must not fabricate color space");
}

#[test]
fn handle_serde_domain_isolation() {
    let attr_handle: Handle<AttributeDomain> = Handle::from_raw(7, 3).expect("handle");
    let json = serde_json::to_string(&attr_handle).expect("serialize attribute handle");

    // Deserializing attribute JSON as TextureDomain must fail
    let err = serde_json::from_str::<Handle<TextureDomain>>(&json)
        .expect_err("domain mismatch must be rejected");
    assert!(err.to_string().contains("domain mismatch"));
}

// -----------------------------------------------------------------------------
// Seeded deterministic property tests (vqa.3)
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
}

#[test]
fn property_test_handle_pack_unpack_and_words_roundtrip() {
    const SEED: u64 = 0x4841_4E44_0001_0001;

    // Explicit boundary cases
    let boundary_pairs: [(u32, u32); 12] = [
        (0, 1),
        (0, 2),
        (0, u32::MAX - 1),
        (0, u32::MAX),
        (1, 1),
        (1, u32::MAX),
        (u32::MAX - 1, 1),
        (u32::MAX - 1, u32::MAX),
        (u32::MAX, 1),
        (u32::MAX, 2),
        (u32::MAX, u32::MAX - 1),
        (u32::MAX, u32::MAX),
    ];
    for (idx, (index, generation)) in boundary_pairs.iter().copied().enumerate() {
        let handle: Handle<GeometryDomain> = Handle::from_raw(index, generation)
            .unwrap_or_else(|e| panic!("from_raw failed for boundary {idx} ({index}, {generation}): {e:?}"));
        assert_eq!(handle.index(), index, "index mismatch for boundary {idx}");
        assert_eq!(handle.generation().get(), generation, "generation mismatch for boundary {idx}");

        let (w0, w1) = handle.to_words();
        assert_eq!(w0, index, "to_words w0 mismatch for boundary {idx}");
        assert_eq!(w1, generation, "to_words w1 mismatch for boundary {idx}");

        let reconstructed = Handle::<GeometryDomain>::from_words(w0, w1)
            .unwrap_or_else(|e| panic!("from_words failed for boundary {idx}: {e:?}"));
        assert_eq!(handle, reconstructed, "reconstructed mismatch for boundary {idx}");

        let packed = handle.pack_u64();
        let expected_packed = ((generation as u64) << 32) | (index as u64);
        assert_eq!(packed, expected_packed, "pack_u64 mismatch for boundary {idx}");

        let unpacked = Handle::<GeometryDomain>::unpack_u64(packed)
            .unwrap_or_else(|e| panic!("unpack_u64 failed for boundary {idx}: {e:?}"));
        assert_eq!(handle, unpacked, "unpacked mismatch for boundary {idx}");
    }

    // Randomized round-trip fuzzing
    let mut rng = TestLcg::new(SEED);
    for i in 0..10_000 {
        let index = rng.next_u32();
        let mut generation = rng.next_u32();
        if generation == 0 {
            generation = 1;
        }

        let handle: Handle<MaterialDomain> = Handle::from_raw(index, generation)
            .unwrap_or_else(|e| panic!("from_raw failed for seed {SEED:#018x} at iter {i}: {e:?}"));

        assert_eq!(handle.index(), index, "index mismatch for seed {SEED:#018x} at iter {i}");
        assert_eq!(handle.generation().get(), generation, "generation mismatch for seed {SEED:#018x} at iter {i}");

        let (w0, w1) = handle.to_words();
        assert_eq!(w0, index, "w0 mismatch for seed {SEED:#018x} at iter {i}");
        assert_eq!(w1, generation, "w1 mismatch for seed {SEED:#018x} at iter {i}");

        let from_words = Handle::<MaterialDomain>::from_words(w0, w1)
            .unwrap_or_else(|e| panic!("from_words failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(handle, from_words, "from_words mismatch for seed {SEED:#018x} at iter {i}");

        let packed = handle.pack_u64();
        let expected_packed = ((generation as u64) << 32) | (index as u64);
        assert_eq!(packed, expected_packed, "pack_u64 mismatch for seed {SEED:#018x} at iter {i}");

        let unpacked = Handle::<MaterialDomain>::unpack_u64(packed)
            .unwrap_or_else(|e| panic!("unpack_u64 failed for seed {SEED:#018x} at iter {i}: {e:?}"));
        assert_eq!(handle, unpacked, "unpack_u64 mismatch for seed {SEED:#018x} at iter {i}");
    }
}

#[test]
fn property_test_handle_zero_generation_rejection() {
    const SEED: u64 = 0x4841_4E44_0002_0002;

    // Explicit boundary indices with generation 0
    let boundary_indices = [0, 1, 2, u32::MAX - 1, u32::MAX];
    for &index in &boundary_indices {
        let err_raw = Handle::<TextureDomain>::from_raw(index, 0)
            .expect_err("from_raw must reject zero generation");
        assert_eq!(
            err_raw,
            HandleError::InvalidGeneration { raw_generation: 0 },
            "expected InvalidGeneration for index {index}"
        );

        let err_words = Handle::<TextureDomain>::from_words(index, 0)
            .expect_err("from_words must reject zero generation");
        assert_eq!(
            err_words,
            HandleError::InvalidGeneration { raw_generation: 0 },
            "expected InvalidGeneration for index {index}"
        );

        let packed_zero_gen = index as u64; // upper 32 bits = 0
        let err_unpack = Handle::<TextureDomain>::unpack_u64(packed_zero_gen)
            .expect_err("unpack_u64 must reject zero generation");
        assert_eq!(
            err_unpack,
            HandleError::InvalidGeneration { raw_generation: 0 },
            "expected InvalidGeneration for packed {packed_zero_gen:#018x}"
        );
    }

    // Randomized fuzzing asserting zero generation is always rejected
    let mut rng = TestLcg::new(SEED);
    for i in 0..5_000 {
        let index = rng.next_u32();

        let err_raw = Handle::<AttributeDomain>::from_raw(index, 0)
            .expect_err("from_raw must reject 0");
        assert_eq!(
            err_raw,
            HandleError::InvalidGeneration { raw_generation: 0 },
            "from_raw failed for seed {SEED:#018x} at iter {i}"
        );

        let err_words = Handle::<AttributeDomain>::from_words(index, 0)
            .expect_err("from_words must reject 0");
        assert_eq!(
            err_words,
            HandleError::InvalidGeneration { raw_generation: 0 },
            "from_words failed for seed {SEED:#018x} at iter {i}"
        );

        let packed_zero_gen = index as u64;
        let err_unpack = Handle::<AttributeDomain>::unpack_u64(packed_zero_gen)
            .expect_err("unpack_u64 must reject 0");
        assert_eq!(
            err_unpack,
            HandleError::InvalidGeneration { raw_generation: 0 },
            "unpack_u64 failed for seed {SEED:#018x} at iter {i}"
        );
    }
}

