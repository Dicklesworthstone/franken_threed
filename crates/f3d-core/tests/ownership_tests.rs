//! Integration tests for single-writer epochs, authority transitions, and immutable per-use versions.
//!
//! NO-CLAIM: this state primitive is not an ECS, full scene ownership analysis, or GPU proof.

use core::num::NonZeroU32;
use f3d_core::handle::{Handle, MaterialDomain, RegionDomain};
use f3d_core::ownership::{
    Author, BorrowScope, BorrowState, BorrowToken, CopyAccounting, DataVersion, Epoch, OwnerMode,
    OwnershipError, PerUseByteBuffer, PerUseSnapshotStore, RegionState, UseRecord,
};

#[test]
fn epoch_and_data_version_words_abi() {
    // Roundtrip decomposition into two 32-bit words (high, low)
    let epoch = Epoch::new(0x1234_5678_9ABC_DEF0);
    let (hi, lo) = epoch.to_words();
    assert_eq!(hi, 0x1234_5678);
    assert_eq!(lo, 0x9ABC_DEF0);
    assert_eq!(epoch.high_u32(), 0x1234_5678);
    assert_eq!(epoch.low_u32(), 0x9ABC_DEF0);

    let restored_epoch = Epoch::from_words(hi, lo);
    assert_eq!(epoch, restored_epoch);
    assert_eq!(restored_epoch.get(), 0x1234_5678_9ABC_DEF0);

    // Roundtrip for DataVersion
    let version = DataVersion::new(0xFEED_FACE_CAFE_BEEF);
    let (v_hi, v_lo) = version.to_words();
    assert_eq!(v_hi, 0xFEED_FACE);
    assert_eq!(v_lo, 0xCAFE_BEEF);
    assert_eq!(version.high_u32(), 0xFEED_FACE);
    assert_eq!(version.low_u32(), 0xCAFE_BEEF);

    let restored_version = DataVersion::from_words(v_hi, v_lo);
    assert_eq!(version, restored_version);
    assert_eq!(restored_version.get(), 0xFEED_FACE_CAFE_BEEF);

    // Verify precision preservation beyond JavaScript Number safe integer (2^53 - 1)
    let large_val = (1u64 << 55) | 0x1337;
    let large_epoch = Epoch::new(large_val);
    let (l_hi, l_lo) = large_epoch.to_words();
    let reconstructed = Epoch::from_words(l_hi, l_lo);
    assert_eq!(large_epoch, reconstructed);
    assert_eq!(reconstructed.get(), large_val);
}

#[test]
fn legal_js_to_wasm_publish_and_transfer() {
    let handle = Handle::<RegionDomain>::new(1, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    // Initial state: default fallback is JS author
    assert_eq!(state.mode(), OwnerMode::Js);
    assert_eq!(state.author(), Author::Js);
    assert_eq!(state.current_epoch(), Epoch::ZERO);
    assert_eq!(state.current_version(), DataVersion::INITIAL);
    assert_eq!(state.unpublished_writes(), 0);

    // JS records a mutation
    let v2 = state.record_write(Author::Js).expect("JS write succeeds");
    assert_eq!(v2.get(), 2);
    assert_eq!(state.unpublished_writes(), 1);

    // JS publishes uncommitted writes at current epoch (0) -> advances epoch to 1
    let ep1 = state
        .publish(Author::Js, Epoch::ZERO)
        .expect("JS publish succeeds");
    assert_eq!(ep1.get(), 1);
    assert_eq!(state.unpublished_writes(), 0);

    // Legal transfer from JS to Wasm at epoch 1 -> advances epoch to 2
    let ep2 = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(1))
        .expect("transfer to Wasm succeeds");
    assert_eq!(ep2.get(), 2);
    assert_eq!(state.author(), Author::Wasm);
    assert_eq!(state.mode(), OwnerMode::Wasm);

    // Wasm records a mutation
    let v3 = state.record_write(Author::Wasm).expect("Wasm write succeeds");
    assert_eq!(v3.get(), 3);
    assert_eq!(state.unpublished_writes(), 1);

    // Wasm publishes at epoch 2 -> advances epoch to 3
    let ep3 = state
        .publish(Author::Wasm, Epoch::new(2))
        .expect("Wasm publish succeeds");
    assert_eq!(ep3.get(), 3);
    assert_eq!(state.unpublished_writes(), 0);
}

#[test]
fn legal_mirrored_mode_transitions() {
    let handle = Handle::<RegionDomain>::new(2, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::with_mode(
        handle,
        OwnerMode::Mirrored {
            author: Author::Js,
        },
    );

    assert!(state.mode().is_mirrored());
    assert_eq!(state.author(), Author::Js);

    // JS writes and publishes (epoch 0 -> 1)
    state.record_write(Author::Js).expect("JS write ok");
    state
        .publish(Author::Js, Epoch::ZERO)
        .expect("JS publish ok");

    // Transfer in mirrored mode changes author to Wasm (epoch 1 -> 2)
    let ep2 = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(1))
        .expect("transfer in mirrored mode ok");
    assert_eq!(ep2.get(), 2);
    assert_eq!(
        state.mode(),
        OwnerMode::Mirrored {
            author: Author::Wasm
        }
    );
    assert_eq!(state.author(), Author::Wasm);
}

#[test]
fn red_v1_and_blue_v2_survive_same_submission() {
    // Invariant: Two uses of one logical material after mutation must retain distinct
    // versions and immutable snapshots/buffer slices rather than overwrite a single current slot.
    let mat_handle = Handle::<MaterialDomain>::new(10, NonZeroU32::new(1).unwrap());
    let reg_handle = Handle::<RegionDomain>::new(10, NonZeroU32::new(1).unwrap());
    let mut region = RegionState::new(reg_handle);

    // Alignment 256 for WebGPU uniform buffer slices
    let mut store = PerUseSnapshotStore::<MaterialDomain, [f32; 4]>::new(256)
        .expect("valid alignment");

    let red_color = [1.0f32, 0.0, 0.0, 1.0];
    let blue_color = [0.0f32, 0.0, 1.0, 1.0];

    // Pass A uses Version 1 (Red)
    let v1 = region.current_version();
    let epoch0 = region.current_epoch();
    let use_a = store
        .record_use(mat_handle, v1, epoch0, red_color, 16)
        .expect("record Pass A Red use");
    assert_eq!(use_a.slice_id(), 0);
    assert_eq!(use_a.byte_offset(), 0);
    assert_eq!(use_a.byte_length(), 16);
    assert_eq!(use_a.version(), DataVersion::INITIAL);

    // Material is mutated to Blue between passes
    let v2 = region
        .record_write(Author::Js)
        .expect("record write for Blue");
    assert_eq!(v2.get(), 2);

    // Pass B uses Version 2 (Blue)
    let use_b = store
        .record_use(mat_handle, v2, epoch0, blue_color, 16)
        .expect("record Pass B Blue use");
    assert_eq!(use_b.slice_id(), 1);
    assert_eq!(use_b.byte_offset(), 256); // Correctly aligned to 256 bytes
    assert_eq!(use_b.byte_length(), 16);
    assert_eq!(use_b.version(), v2);

    // Assert that both slices coexist in the same submission schedule
    let slice_a = store.get_use(&use_a).expect("lookup slice A via use record");
    let slice_b = store.get_use(&use_b).expect("lookup slice B via use record");

    // Slice A is STILL Red, slice B is Blue - no overwrite occurred
    assert_eq!(*slice_a.data(), red_color);
    assert_eq!(slice_a.record().version(), v1);
    assert_eq!(*slice_b.data(), blue_color);
    assert_eq!(slice_b.record().version(), v2);

    assert_eq!(store.len(), 2);
    assert_eq!(store.total_bytes(), 272); // 256 aligned offset + 16 bytes
}

#[test]
fn per_use_byte_buffer_slices_immutable_and_aligned() {
    let mat_handle = Handle::<MaterialDomain>::new(20, NonZeroU32::new(1).unwrap());
    let mut byte_buf = PerUseByteBuffer::<MaterialDomain>::new(256).expect("alignment 256 ok");

    let red_bytes: [u8; 4] = [255, 0, 0, 255];
    let blue_bytes: [u8; 4] = [0, 0, 255, 255];

    let rec_a = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &red_bytes)
        .expect("append slice A");
    assert_eq!(rec_a.byte_offset(), 0);
    assert_eq!(rec_a.byte_length(), 4);

    let rec_b = byte_buf
        .append_slice(mat_handle, DataVersion::new(2), Epoch::ZERO, &blue_bytes)
        .expect("append slice B");
    assert_eq!(rec_b.byte_offset(), 256);
    assert_eq!(rec_b.byte_length(), 4);

    let slice_a = byte_buf.get_slice(&rec_a).expect("slice A lookup");
    let slice_b = byte_buf.get_slice(&rec_b).expect("slice B lookup");

    assert_eq!(slice_a, &red_bytes);
    assert_eq!(slice_b, &blue_bytes);

    let raw = byte_buf.as_bytes();
    assert_eq!(&raw[0..4], &red_bytes);
    assert_eq!(&raw[256..260], &blue_bytes);
}

#[test]
fn stale_after_reset_reads_fail_not_aba_blue() {
    // Defect regression: PerUseByteBuffer::get_slice must not trust public UseRecord offsets
    // after reset. After reset + append blue, an old red record MUST NOT read blue (ABA).
    let mat_handle = Handle::<MaterialDomain>::new(21, NonZeroU32::new(1).unwrap());
    let mut byte_buf = PerUseByteBuffer::<MaterialDomain>::new(256).expect("buf new ok");

    let red_bytes = [255u8, 0, 0, 255];
    let blue_bytes = [0u8, 0, 255, 255];

    let red_record = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &red_bytes)
        .expect("append red");
    assert_eq!(byte_buf.get_slice(&red_record).unwrap(), &red_bytes);

    // Frame/submission cycle resets the buffer, advancing the allocation generation
    let new_gen = byte_buf.reset().expect("reset ok");
    assert_eq!(new_gen, 2);

    // Pass B appends blue at byte_offset 0
    let blue_record = byte_buf
        .append_slice(mat_handle, DataVersion::new(2), Epoch::ZERO, &blue_bytes)
        .expect("append blue");
    assert_eq!(blue_record.byte_offset(), 0);
    assert_eq!(byte_buf.get_slice(&blue_record).unwrap(), &blue_bytes);

    // Invariant: The old red record has generation 1; buffer is at generation 2.
    // Querying with old red record must strictly FAIL and NEVER return blue!
    let err = byte_buf
        .get_slice(&red_record)
        .expect_err("stale red record must fail after reset");
    assert_eq!(
        err,
        OwnershipError::StaleSliceRecord {
            expected_generation: 2,
            actual_generation: 1,
            slice_id: red_record.slice_id(),
        }
    );
}

#[test]
fn foreign_store_record_rejected() {
    let mat_handle = Handle::<MaterialDomain>::new(22, NonZeroU32::new(1).unwrap());
    let mut store_a = PerUseByteBuffer::<MaterialDomain>::new(256).expect("store a");
    let store_b = PerUseByteBuffer::<MaterialDomain>::new(256).expect("store b");

    assert_ne!(store_a.store_id(), store_b.store_id());

    let rec_a = store_a
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &[1, 2, 3, 4])
        .expect("append a");

    // Passing record from store_a into store_b must be rejected
    let err = store_b
        .get_slice(&rec_a)
        .expect_err("foreign record must be rejected");
    assert_eq!(
        err,
        OwnershipError::ForeignSliceRecord {
            expected_store: store_b.store_id(),
            actual_store: store_a.store_id(),
        }
    );
}

#[test]
fn record_offset_tampering_rejected() {
    let mat_handle = Handle::<MaterialDomain>::new(23, NonZeroU32::new(1).unwrap());
    let mut byte_buf = PerUseByteBuffer::<MaterialDomain>::new(256).expect("buf ok");

    let rec = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &[10, 20, 30, 40])
        .expect("append ok");

    // Tamper with byte_offset
    let tampered = UseRecord::new(
        rec.resource(),
        rec.version(),
        rec.epoch(),
        rec.store_id(),
        rec.generation(),
        rec.slice_id(),
        rec.byte_offset() + 100, // Tampered offset!
        rec.byte_length(),
    );

    let err = byte_buf
        .get_slice(&tampered)
        .expect_err("tampered record must fail");
    assert_eq!(
        err,
        OwnershipError::SliceIdentityMismatch {
            slice_id: rec.slice_id(),
        }
    );
}

#[test]
fn align_up_overflow_does_not_panic() {
    let mat_handle = Handle::<MaterialDomain>::new(24, NonZeroU32::new(1).unwrap());
    let mut store = PerUseSnapshotStore::<MaterialDomain, u32>::new(256).expect("store ok");

    // Extreme byte length triggers overflow check cleanly without panic
    let err = store
        .record_use(mat_handle, DataVersion::new(1), Epoch::ZERO, 42, u64::MAX)
        .expect_err("u64::MAX byte length must overflow next offset");
    assert_eq!(
        err,
        OwnershipError::VersionOverflow { current: 0 }
    );
}

#[test]
fn aba_transfer_replay_prevented() {
    // Defect regression: transfer_authority must advance publication epoch.
    // Otherwise JS -> Wasm -> JS allows replaying the stale JS -> Wasm transfer command.
    let handle = Handle::<RegionDomain>::new(25, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    // Initial state: author = JS, epoch = 0
    assert_eq!(state.author(), Author::Js);
    assert_eq!(state.current_epoch(), Epoch::ZERO);

    // 1. JS transfers to Wasm at Epoch(0) -> advances epoch to 1
    let ep1 = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::ZERO)
        .expect("transfer to Wasm ok");
    assert_eq!(ep1, Epoch::new(1));
    assert_eq!(state.author(), Author::Wasm);

    // 2. Wasm transfers back to JS at Epoch(1) -> advances epoch to 2
    let ep2 = state
        .transfer_authority(Author::Wasm, Author::Js, Epoch::new(1))
        .expect("transfer back to JS ok");
    assert_eq!(ep2, Epoch::new(2));
    assert_eq!(state.author(), Author::Js);

    // 3. Stale transfer replay attempt: re-issuing the original transfer_authority(JS, Wasm, Epoch(0))
    // Because author is JS again, without epoch advance this would have succeeded (ABA).
    // With monotonic epoch advance, it is strictly REJECTED with StaleEpoch!
    let err_stale = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::ZERO)
        .expect_err("stale transfer replay must fail");
    assert_eq!(
        err_stale,
        OwnershipError::StaleEpoch {
            expected: Epoch::new(2),
            actual: Epoch::ZERO,
        }
    );
}

#[test]
fn transition_mode_author_guarded_and_advances_epoch() {
    let handle = Handle::<RegionDomain>::new(26, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    // Unauthorized caller cannot transition mode
    let err_unauth = state
        .transition_mode(Author::Wasm, OwnerMode::Wasm, Epoch::ZERO)
        .expect_err("unauthorized author cannot transition mode");
    assert_eq!(
        err_unauth,
        OwnershipError::UnauthorizedWriter {
            expected: Author::Js,
            actual: Author::Wasm,
        }
    );

    // Authorized transition advances epoch (0 -> 1)
    let ep1 = state
        .transition_mode(Author::Js, OwnerMode::Wasm, Epoch::ZERO)
        .expect("authorized transition ok");
    assert_eq!(ep1, Epoch::new(1));
    assert_eq!(state.author(), Author::Wasm);

    // Stale epoch transition rejected
    let err_stale = state
        .transition_mode(Author::Wasm, OwnerMode::Js, Epoch::ZERO)
        .expect_err("stale transition rejected");
    assert_eq!(
        err_stale,
        OwnershipError::StaleEpoch {
            expected: Epoch::new(1),
            actual: Epoch::ZERO,
        }
    );
}

#[test]
fn dedup_rejects_inconsistent_snapshot_metadata() {
    let mat_handle = Handle::<MaterialDomain>::new(27, NonZeroU32::new(1).unwrap());
    let mut byte_buf = PerUseByteBuffer::<MaterialDomain>::new(256).expect("buf ok");

    let rec1 = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &[1, 2, 3, 4])
        .expect("first record ok");

    // Exact match deduplicates idempotently
    let rec1_dup = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &[1, 2, 3, 4])
        .expect("exact match ok");
    assert_eq!(rec1, rec1_dup);

    // Inconsistent epoch rejected
    let err_epoch = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::new(1), &[1, 2, 3, 4])
        .expect_err("inconsistent epoch rejected");
    assert_eq!(
        err_epoch,
        OwnershipError::ImmutableSnapshotViolation {
            version: DataVersion::new(1),
            detail: "conflicting metadata or byte content provided for existing immutable version",
        }
    );

    // Inconsistent byte length rejected
    let err_len = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &[1, 2, 3, 4, 5])
        .expect_err("inconsistent length rejected");
    assert_eq!(
        err_len,
        OwnershipError::ImmutableSnapshotViolation {
            version: DataVersion::new(1),
            detail: "conflicting metadata or byte content provided for existing immutable version",
        }
    );

    // Inconsistent byte content rejected
    let err_data = byte_buf
        .append_slice(mat_handle, DataVersion::new(1), Epoch::ZERO, &[9, 9, 9, 9])
        .expect_err("inconsistent content rejected");
    assert_eq!(
        err_data,
        OwnershipError::ImmutableSnapshotViolation {
            version: DataVersion::new(1),
            detail: "conflicting metadata or byte content provided for existing immutable version",
        }
    );
}

#[test]
fn stale_transfer_rejected() {
    let handle = Handle::<RegionDomain>::new(3, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    // Published at epoch 0 -> epoch advances to 1
    state.publish(Author::Js, Epoch::ZERO).expect("publish ok");
    assert_eq!(state.current_epoch(), Epoch::new(1));

    // Stale epoch (attempting transfer at epoch 0 when current is 1)
    let err = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(0))
        .expect_err("stale epoch transfer must fail");
    assert_eq!(
        err,
        OwnershipError::StaleEpoch {
            expected: Epoch::new(1),
            actual: Epoch::new(0),
        }
    );

    // Future/mismatched epoch also rejected
    let err_future = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(5))
        .expect_err("future epoch transfer must fail");
    assert_eq!(
        err_future,
        OwnershipError::StaleEpoch {
            expected: Epoch::new(1),
            actual: Epoch::new(5),
        }
    );
}

#[test]
fn transfer_with_unpublished_writes_rejected() {
    let handle = Handle::<RegionDomain>::new(4, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    // JS records a mutation but does not publish
    state.record_write(Author::Js).expect("write ok");
    assert_eq!(state.unpublished_writes(), 1);

    // Authority transfer rejected due to uncommitted writes
    let err = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::ZERO)
        .expect_err("transfer with unpublished writes must fail");
    assert_eq!(
        err,
        OwnershipError::UnpublishedWritesPending {
            pending_count: 1,
            current_epoch: Epoch::ZERO,
        }
    );

    // Once published, transfer succeeds (epoch 0 -> 1 on publish, 1 -> 2 on transfer)
    state
        .publish(Author::Js, Epoch::ZERO)
        .expect("publish pending writes");
    let ep2 = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(1))
        .expect("transfer now succeeds");
    assert_eq!(ep2, Epoch::new(2));
    assert_eq!(state.author(), Author::Wasm);
}

#[test]
fn double_writer_unauthorized_mutation_rejected() {
    let handle = Handle::<RegionDomain>::new(5, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    // JS is authoritative author. Wasm attempt to write is rejected
    let err_write = state
        .record_write(Author::Wasm)
        .expect_err("unauthorized writer must fail");
    assert_eq!(
        err_write,
        OwnershipError::UnauthorizedWriter {
            expected: Author::Js,
            actual: Author::Wasm,
        }
    );

    // Wasm attempt to publish is also rejected
    let err_pub = state
        .publish(Author::Wasm, Epoch::ZERO)
        .expect_err("unauthorized publisher must fail");
    assert_eq!(
        err_pub,
        OwnershipError::UnauthorizedWriter {
            expected: Author::Js,
            actual: Author::Wasm,
        }
    );

    // Stale transfer attempt from Wasm when JS owns it
    let err_trans = state
        .transfer_authority(Author::Wasm, Author::Js, Epoch::ZERO)
        .expect_err("unauthorized transferor must fail");
    assert_eq!(
        err_trans,
        OwnershipError::UnauthorizedWriter {
            expected: Author::Js,
            actual: Author::Wasm,
        }
    );
}

#[test]
fn same_author_transfer_rejected() {
    let handle = Handle::<RegionDomain>::new(6, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    let err = state
        .transfer_authority(Author::Js, Author::Js, Epoch::ZERO)
        .expect_err("same author transfer must fail");
    assert_eq!(
        err,
        OwnershipError::SameAuthorTransfer { author: Author::Js }
    );
}

#[test]
fn counter_overflow_checked() {
    // Epoch overflow
    let max_epoch = Epoch::new(u64::MAX);
    let err_ep = max_epoch.checked_next().expect_err("overflow epoch");
    assert_eq!(
        err_ep,
        OwnershipError::EpochOverflow { current: u64::MAX }
    );

    // DataVersion overflow
    let max_version = DataVersion::new(u64::MAX);
    let err_ver = max_version.checked_next().expect_err("overflow version");
    assert_eq!(
        err_ver,
        OwnershipError::VersionOverflow { current: u64::MAX }
    );
}

#[test]
fn late_mutation_corrupting_first_snapshot_fails() {
    let mat_handle = Handle::<MaterialDomain>::new(30, NonZeroU32::new(1).unwrap());
    let mut store = PerUseSnapshotStore::<MaterialDomain, [f32; 4]>::new(16)
        .expect("valid alignment");

    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];

    // Version 1 recorded as Red
    let rec1 = store
        .record_use(mat_handle, DataVersion::new(1), Epoch::ZERO, red, 16)
        .expect("record v1 red ok");

    // Recording duplicate with IDENTICAL metadata and data is idempotent
    let rec1_dup = store
        .record_use(mat_handle, DataVersion::new(1), Epoch::ZERO, red, 16)
        .expect("idempotent record ok");
    assert_eq!(rec1, rec1_dup);

    // Attempting to overwrite or record DIFFERENT data (Green) for existing Version 1 fails
    let err = store
        .record_use(mat_handle, DataVersion::new(1), Epoch::ZERO, green, 16)
        .expect_err("corrupting existing snapshot version must fail");

    assert_eq!(
        err,
        OwnershipError::ImmutableSnapshotViolation {
            version: DataVersion::new(1),
            detail: "conflicting metadata or data provided for existing immutable version snapshot",
        }
    );

    // Verify first snapshot was not corrupted
    let slice0 = store.get_use(&rec1).expect("lookup slice");
    assert_eq!(*slice0.data(), red);
}

#[test]
fn invalid_alignment_rejected() {
    let err_zero = PerUseSnapshotStore::<MaterialDomain, u32>::new(0)
        .expect_err("alignment 0 must fail");
    assert_eq!(
        err_zero,
        OwnershipError::InvalidAlignment { alignment: 0 }
    );

    let err_non_pow2 = PerUseByteBuffer::<MaterialDomain>::new(15)
        .expect_err("non-power of two alignment must fail");
    assert_eq!(
        err_non_pow2,
        OwnershipError::InvalidAlignment { alignment: 15 }
    );
}

#[test]
fn store_id_overflow_near_max_fails_cleanly() {
    f3d_core::ownership::set_next_store_id_for_testing(u64::MAX);
    let err = PerUseByteBuffer::<MaterialDomain>::new(256)
        .expect_err("store ID at u64::MAX must overflow");
    assert_eq!(
        err,
        OwnershipError::StoreIdOverflow { current: u64::MAX }
    );
    // Reset store ID counter for other tests
    f3d_core::ownership::set_next_store_id_for_testing(100_000);
}

#[test]
fn borrow_scope_positive_lifecycle_and_view_accounting() {
    let mut scope = BorrowScope::new();
    assert_eq!(scope.state(), BorrowState::Idle);
    assert!(scope.is_idle());
    assert!(!scope.is_borrowed());
    assert!(!scope.is_growth_blocked());
    assert_eq!(scope.growth_generation(), 1);

    // Enter borrow scope
    let token = scope.enter().expect("enter borrow scope");
    assert_eq!(token.get(), 1);
    assert_eq!(scope.state(), BorrowState::Borrowed);
    assert!(scope.is_borrowed());
    assert!(!scope.is_idle());
    assert!(scope.is_growth_blocked());

    // Record zero-copy view bytes while borrowed
    scope.record_view_bytes(512).expect("record view bytes ok");
    assert_eq!(scope.accounting().bytes_view, 512);
    assert_eq!(scope.accounting().total_copied_bytes(), 0);
    assert_eq!(scope.accounting().total_transported_bytes(), 512);

    // Exit borrow scope with valid token
    scope.exit(token).expect("exit borrow scope");
    assert_eq!(scope.state(), BorrowState::Idle);
    assert!(scope.is_idle());
    assert!(!scope.is_borrowed());
    assert!(!scope.is_growth_blocked());

    // Growth generation remains stable across clean borrows without growth
    assert_eq!(scope.growth_generation(), 1);
}

#[test]
fn borrow_scope_reentry_strictly_rejected() {
    let mut scope = BorrowScope::new();
    let token1 = scope.enter().expect("initial enter ok");
    assert_eq!(token1.get(), 1);

    // Reentrant enter attempt fails with typed error containing active token
    let err = scope.enter().expect_err("reentrant enter must fail");
    assert_eq!(
        err,
        OwnershipError::BorrowScopeReentry {
            active_token: 1,
        }
    );

    // Scope remains borrowed
    assert_eq!(scope.state(), BorrowState::Borrowed);

    // Exit first borrow
    scope.exit(token1).expect("exit first borrow");
    assert!(scope.is_idle());

    // Next enter succeeds with incremented token ID
    let token2 = scope.enter().expect("second enter ok");
    assert_eq!(token2.get(), 2);
    scope.exit(token2).expect("exit second borrow");
}

#[test]
fn borrow_scope_growth_during_borrow_strictly_rejected() {
    let mut scope = BorrowScope::new();
    assert_eq!(scope.growth_generation(), 1);

    // When idle, linear memory growth succeeds and increments generation
    let growth_gen2 = scope.record_growth(4).expect("growth while idle ok");
    assert_eq!(growth_gen2, 2);
    assert_eq!(scope.growth_generation(), 2);

    // Enter borrow scope
    let token = scope.enter().expect("enter ok");

    // Attempting linear memory growth while borrowed is strictly rejected
    let err = scope.record_growth(1).expect_err("growth while borrowed must fail");
    assert_eq!(
        err,
        OwnershipError::LinearMemoryGrowthBlocked {
            state: BorrowState::Borrowed,
        }
    );

    // Generation did not advance
    assert_eq!(scope.growth_generation(), 2);

    // Exit borrow scope
    scope.exit(token).expect("exit ok");

    // Growth succeeds again after exit
    let growth_gen3 = scope.record_growth(2).expect("growth after exit ok");
    assert_eq!(growth_gen3, 3);
    assert_eq!(scope.growth_generation(), 3);
}

#[test]
fn borrow_scope_explicit_growth_blocked_state() {
    let mut scope = BorrowScope::new();
    scope.block_growth();
    assert_eq!(scope.state(), BorrowState::GrowthBlocked);
    assert!(scope.is_growth_blocked());
    assert!(!scope.is_idle());
    assert!(!scope.is_borrowed());

    // Linear memory growth rejected in GrowthBlocked state
    let err = scope.record_growth(1).expect_err("growth blocked");
    assert_eq!(
        err,
        OwnershipError::LinearMemoryGrowthBlocked {
            state: BorrowState::GrowthBlocked,
        }
    );

    // Borrows can still be taken while growth is blocked
    let token = scope.enter().expect("borrow while growth blocked ok");
    assert_eq!(scope.state(), BorrowState::Borrowed);
    assert!(scope.is_borrowed());

    // Exiting returns state to GrowthBlocked because explicit block is active
    scope.exit(token).expect("exit ok");
    assert_eq!(scope.state(), BorrowState::GrowthBlocked);

    // Unblocking restores Idle state and permits growth
    scope.unblock_growth();
    assert_eq!(scope.state(), BorrowState::Idle);
    assert!(scope.is_idle());
    let growth_gen = scope.record_growth(1).expect("growth unblocked");
    assert_eq!(growth_gen, 2);
}

#[test]
fn borrow_scope_exit_validation_and_token_mismatch() {
    let mut scope = BorrowScope::new();

    // Exiting while idle fails
    let err_idle = scope
        .exit(BorrowToken::new(1))
        .expect_err("exit while idle must fail");
    assert_eq!(err_idle, OwnershipError::BorrowScopeNotActive);

    // Enter scope -> gets token 1
    let token1 = scope.enter().expect("enter ok");
    assert_eq!(token1, BorrowToken::new(1));

    // Exiting with wrong token fails
    let err_mismatch = scope
        .exit(BorrowToken::new(999))
        .expect_err("exit with mismatched token must fail");
    assert_eq!(
        err_mismatch,
        OwnershipError::BorrowTokenMismatch {
            expected: 1,
            actual: 999,
        }
    );

    // State remains borrowed despite failed exit attempt
    assert_eq!(scope.state(), BorrowState::Borrowed);

    // Exiting with correct token succeeds
    scope.exit(token1).expect("exit with correct token ok");
    assert_eq!(scope.state(), BorrowState::Idle);
}

#[test]
fn borrow_scope_exact_copy_accounting_counters() {
    let mut scope = BorrowScope::new();

    // Recording view bytes outside borrow scope is rejected
    let err_view = scope
        .record_view_bytes(1024)
        .expect_err("view outside borrow scope must fail");
    assert_eq!(err_view, OwnershipError::BorrowScopeNotActive);

    // Enter borrow and record view bytes
    let token = scope.enter().expect("enter ok");
    scope.record_view_bytes(1024).expect("record view ok");
    scope.record_view_bytes(2048).expect("record second view ok");
    scope.exit(token).expect("exit ok");

    // Record copies (which can happen outside or inside borrows)
    scope.record_write_buffer_copy(4096);
    scope.record_staging_copy(512);

    let acct = scope.accounting();
    assert_eq!(acct.bytes_view, 3072);
    assert_eq!(acct.bytes_copied_write_buffer, 4096);
    assert_eq!(acct.bytes_copied_staging, 512);
    assert_eq!(acct.total_copied_bytes(), 4608);
    assert_eq!(acct.total_transported_bytes(), 7680);

    // Reset counters (e.g. at frame boundary)
    scope.accounting_mut().reset();
    assert_eq!(scope.accounting().bytes_view, 0);
    assert_eq!(scope.accounting().bytes_copied_write_buffer, 0);
    assert_eq!(scope.accounting().bytes_copied_staging, 0);
    assert_eq!(scope.accounting().total_copied_bytes(), 0);
    assert_eq!(scope.accounting().total_transported_bytes(), 0);
}

