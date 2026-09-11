//! Integration tests for single-writer epochs, authority transitions, and immutable per-use versions.
//!
//! NO-CLAIM: this state primitive is not an ECS, full scene ownership analysis, or GPU proof.

use core::num::NonZeroU32;
use f3d_core::handle::{Handle, MaterialDomain, RegionDomain};
use f3d_core::ownership::{
    Author, BorrowScope, BorrowState, BorrowToken, DataVersion, Epoch, OwnerMode,
    OwnershipError, PerUseByteBuffer, PerUseSnapshotStore, RegionAuthorshipDump, RegionState,
    UseRecord,
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
fn region_state_authorship_dump_reflects_transfer() {
    let handle = Handle::<RegionDomain>::new(42, NonZeroU32::new(1).unwrap());
    let mut state = RegionState::new(handle);

    // Initial state: Js author, zero epoch, no pending writes
    let d0 = state.authorship_dump();
    assert_eq!(
        d0,
        RegionAuthorshipDump {
            mode: OwnerMode::Js,
            author: Author::Js,
            current_epoch: Epoch::ZERO,
            published_epoch: Epoch::ZERO,
            unpublished_writes: 0,
        }
    );
    assert_eq!(
        format!("{d0}"),
        "mode=Js, author=Js, current_epoch=0, published_epoch=0, unpublished_writes=0"
    );

    // Mutation pending publication: unpublished_writes increments, epochs stay zero
    state.record_write(Author::Js).expect("JS write succeeds");
    let d1 = state.authorship_dump();
    assert_eq!(d1.unpublished_writes, 1);
    assert_eq!(d1.current_epoch, Epoch::ZERO);
    assert_eq!(d1.published_epoch, Epoch::ZERO);
    assert_eq!(
        format!("{d1}"),
        "mode=Js, author=Js, current_epoch=0, published_epoch=0, unpublished_writes=1"
    );

    // Publish advances current_epoch and published_epoch to 1, clears pending writes
    state
        .publish(Author::Js, Epoch::ZERO)
        .expect("JS publish succeeds");
    let d2 = state.authorship_dump();
    assert_eq!(d2.current_epoch, Epoch::new(1));
    assert_eq!(d2.published_epoch, Epoch::new(1));
    assert_eq!(d2.unpublished_writes, 0);
    assert_eq!(
        format!("{d2}"),
        "mode=Js, author=Js, current_epoch=1, published_epoch=1, unpublished_writes=0"
    );

    // Transfer authority from JS to Wasm at epoch 1 -> advances current_epoch to 2,
    // switches mode and author to Wasm, while published_epoch stays at 1
    let ep2 = state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(1))
        .expect("transfer to Wasm succeeds");
    assert_eq!(ep2.get(), 2);

    let d3 = state.authorship_dump();
    assert_eq!(d3.mode, OwnerMode::Wasm);
    assert_eq!(d3.author, Author::Wasm);
    assert_eq!(d3.current_epoch, Epoch::new(2));
    assert_eq!(d3.published_epoch, Epoch::new(1));
    assert_eq!(d3.unpublished_writes, 0);
    assert_eq!(
        format!("{d3}"),
        "mode=Wasm, author=Wasm, current_epoch=2, published_epoch=1, unpublished_writes=0"
    );

    // Transition to Mirrored mode with Author::Wasm -> advances current_epoch to 3
    state
        .transition_mode(
            Author::Wasm,
            OwnerMode::Mirrored {
                author: Author::Wasm,
            },
            Epoch::new(2),
        )
        .expect("transition to mirrored succeeds");
    let d4 = state.authorship_dump();
    assert_eq!(
        d4.mode,
        OwnerMode::Mirrored {
            author: Author::Wasm
        }
    );
    assert_eq!(d4.author, Author::Wasm);
    assert_eq!(d4.current_epoch, Epoch::new(3));
    assert_eq!(d4.published_epoch, Epoch::new(1));
    assert_eq!(
        format!("{d4}"),
        "mode=Mirrored(Wasm), author=Wasm, current_epoch=3, published_epoch=1, unpublished_writes=0"
    );

    // Mirrored transfer back to JS advances current_epoch to 4, published_epoch stays at 1
    state
        .transfer_authority(Author::Wasm, Author::Js, Epoch::new(3))
        .expect("transfer mirrored to JS succeeds");
    let d5 = state.authorship_dump();
    assert_eq!(
        d5.mode,
        OwnerMode::Mirrored {
            author: Author::Js
        }
    );
    assert_eq!(d5.author, Author::Js);
    assert_eq!(d5.current_epoch, Epoch::new(4));
    assert_eq!(d5.published_epoch, Epoch::new(1));
    assert_eq!(
        format!("{d5}"),
        "mode=Mirrored(Js), author=Js, current_epoch=4, published_epoch=1, unpublished_writes=0"
    );
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

// -----------------------------------------------------------------------------
// Seeded deterministic property tests (vqa.3)
// -----------------------------------------------------------------------------

/// Minimal 64-bit Linear Congruential Generator (LCG) for deterministic property testing.
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

#[derive(Clone, Copy, Debug)]
struct StepRecord {
    step: usize,
    action: &'static str,
    author: Author,
    epoch_words: (u32, u32),
    result_variant: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct TransitionRing {
    entries: [Option<StepRecord>; 16],
    cursor: usize,
    count: usize,
}

impl TransitionRing {
    const fn new() -> Self {
        Self {
            entries: [None; 16],
            cursor: 0,
            count: 0,
        }
    }

    fn push(&mut self, record: StepRecord) {
        self.entries[self.cursor] = Some(record);
        self.cursor = (self.cursor + 1) % 16;
        self.count += 1;
    }
}

impl core::fmt::Display for TransitionRing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let total = if self.count < 16 { self.count } else { 16 };
        if total == 0 {
            return write!(f, "    [no steps recorded]");
        }
        let start = if self.count < 16 { 0 } else { self.cursor };
        for i in 0..total {
            let idx = (start + i) % 16;
            if let Some(entry) = &self.entries[idx] {
                if i > 0 {
                    writeln!(f)?;
                }
                write!(
                    f,
                    "    [{:02}] step={}: action={}, author={}, epoch_words=({:#010x}, {:#010x}), result={}",
                    i,
                    entry.step,
                    entry.action,
                    entry.author,
                    entry.epoch_words.0,
                    entry.epoch_words.1,
                    entry.result_variant,
                )?;
            }
        }
        Ok(())
    }
}

fn result_variant_name<T>(res: &Result<T, OwnershipError>) -> &'static str {
    match res {
        Ok(_) => "Ok",
        Err(OwnershipError::UnauthorizedWriter { .. }) => "Err(UnauthorizedWriter)",
        Err(OwnershipError::UnpublishedWritesPending { .. }) => "Err(UnpublishedWritesPending)",
        Err(OwnershipError::StaleEpoch { .. }) => "Err(StaleEpoch)",
        Err(OwnershipError::SameAuthorTransfer { .. }) => "Err(SameAuthorTransfer)",
        Err(OwnershipError::EpochOverflow { .. }) => "Err(EpochOverflow)",
        Err(OwnershipError::VersionOverflow { .. }) => "Err(VersionOverflow)",
        Err(OwnershipError::StoreIdOverflow { .. }) => "Err(StoreIdOverflow)",
        Err(OwnershipError::ImmutableSnapshotViolation { .. }) => "Err(ImmutableSnapshotViolation)",
        Err(OwnershipError::StaleSliceRecord { .. }) => "Err(StaleSliceRecord)",
        Err(OwnershipError::ForeignSliceRecord { .. }) => "Err(ForeignSliceRecord)",
        Err(OwnershipError::SliceIdentityMismatch { .. }) => "Err(SliceIdentityMismatch)",
        Err(OwnershipError::SliceOutOfBounds { .. }) => "Err(SliceOutOfBounds)",
        Err(OwnershipError::SliceNotFound { .. }) => "Err(SliceNotFound)",
        Err(OwnershipError::InvalidSliceLength { .. }) => "Err(InvalidSliceLength)",
        Err(OwnershipError::InvalidAlignment { .. }) => "Err(InvalidAlignment)",
        Err(OwnershipError::BorrowScopeReentry { .. }) => "Err(BorrowScopeReentry)",
        Err(OwnershipError::BorrowScopeNotActive) => "Err(BorrowScopeNotActive)",
        Err(OwnershipError::BorrowTokenMismatch { .. }) => "Err(BorrowTokenMismatch)",
        Err(OwnershipError::LinearMemoryGrowthBlocked { .. }) => "Err(LinearMemoryGrowthBlocked)",
        Err(OwnershipError::Handle(_)) => "Err(Handle)",
    }
}

fn expect_err_with_ring<T: core::fmt::Debug>(
    res: Result<T, OwnershipError>,
    op_desc: &str,
    seed: u64,
    iter: usize,
    step: usize,
    ring: &TransitionRing,
) -> OwnershipError {
    match res {
        Err(e) => e,
        Ok(v) => panic!(
            "{op_desc} must fail but returned Ok({v:?}) for seed {seed:#018x} at iter {iter} step {step}\nrecent transitions:\n{ring}"
        ),
    }
}

fn unwrap_ok_with_ring<T>(
    res: Result<T, OwnershipError>,
    op_desc: &str,
    seed: u64,
    iter: usize,
    step: usize,
    ring: &TransitionRing,
) -> T {
    match res {
        Ok(v) => v,
        Err(e) => panic!(
            "{op_desc} failed with error {e:?} for seed {seed:#018x} at iter {iter} step {step}\nrecent transitions:\n{ring}"
        ),
    }
}

#[test]
fn property_test_region_state_transition_sequences() {
    const SEED: u64 = 0x0BE4_57A7_E001_0001;
    const ITERATIONS: usize = 2_000;
    const STEPS_PER_ITER: usize = 50;

    let mut rng = TestLcg::new(SEED);

    for i in 0..ITERATIONS {
        let handle_idx = rng.next_u32();
        let handle_generation = NonZeroU32::new((rng.next_u32() % 10_000) + 1).unwrap();
        let handle = Handle::<RegionDomain>::new(handle_idx, handle_generation);

        let initial_mode = match rng.next_u32() % 3 {
            0 => OwnerMode::Js,
            1 => OwnerMode::Wasm,
            _ => OwnerMode::Mirrored {
                author: if rng.next_u32() % 2 == 0 {
                    Author::Js
                } else {
                    Author::Wasm
                },
            },
        };
        let mut state = RegionState::with_mode(handle, initial_mode);
        let mut ring = TransitionRing::new();

        for step in 0..STEPS_PER_ITER {
            let prev_epoch = state.current_epoch();
            let prev_version = state.current_version();
            let prev_author = state.author();
            let prev_unpublished = state.unpublished_writes();

            // Generate an epoch candidate:
            // 0..=2: exactly matching current_epoch (fresh)
            // 3: stale epoch (prior to current_epoch)
            // 4: future/mismatched epoch
            let epoch_selector = rng.next_u32() % 5;
            let candidate_epoch = match epoch_selector {
                0..=2 => prev_epoch,
                3 => {
                    let stale_delta = (rng.next_u32() as u64 % 10) + 1;
                    Epoch::new(prev_epoch.get().saturating_sub(stale_delta))
                }
                _ => {
                    let future_delta = (rng.next_u32() as u64 % 10) + 1;
                    Epoch::new(prev_epoch.get().wrapping_add(future_delta))
                }
            };
            let is_stale_or_mismatched_epoch = candidate_epoch != prev_epoch;

            let action_type = rng.next_u32() % 4;
            match action_type {
                0 => {
                    // record_write
                    let writer = if rng.next_u32() % 2 == 0 {
                        Author::Js
                    } else {
                        Author::Wasm
                    };
                    let res = state.record_write(writer);
                    ring.push(StepRecord {
                        step,
                        action: "record_write",
                        author: writer,
                        epoch_words: candidate_epoch.to_words(),
                        result_variant: result_variant_name(&res),
                    });

                    if writer == prev_author {
                        let new_ver = unwrap_ok_with_ring(
                            res,
                            "record_write by authoritative author",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            new_ver,
                            prev_version.checked_next().unwrap(),
                            "version mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.unpublished_writes(),
                            prev_unpublished + 1,
                            "unpublished count mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.current_epoch(),
                            prev_epoch,
                            "epoch must not change on record_write for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else {
                        let err = expect_err_with_ring(
                            res,
                            "record_write by unauthoritative author",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::UnauthorizedWriter {
                                expected: prev_author,
                                actual: writer,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.unpublished_writes(),
                            prev_unpublished,
                            "state must not mutate on failed write for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    }
                }
                1 => {
                    // publish
                    let publisher = if rng.next_u32() % 2 == 0 {
                        Author::Js
                    } else {
                        Author::Wasm
                    };
                    let res = state.publish(publisher, candidate_epoch);
                    ring.push(StepRecord {
                        step,
                        action: "publish",
                        author: publisher,
                        epoch_words: candidate_epoch.to_words(),
                        result_variant: result_variant_name(&res),
                    });

                    // INVARIANT: State machine never accepts a stale epoch
                    if is_stale_or_mismatched_epoch && publisher == prev_author {
                        assert!(
                            res.is_err(),
                            "publish must never accept stale/mismatched epoch {candidate_epoch:?} (current {prev_epoch:?}) for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    }

                    if publisher != prev_author {
                        let err = expect_err_with_ring(
                            res,
                            "publish by unauthorized author",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::UnauthorizedWriter {
                                expected: prev_author,
                                actual: publisher,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else if is_stale_or_mismatched_epoch {
                        let err = expect_err_with_ring(
                            res,
                            "publish with stale epoch",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::StaleEpoch {
                                expected: prev_epoch,
                                actual: candidate_epoch,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else {
                        let new_epoch = unwrap_ok_with_ring(
                            res,
                            "authorized publish",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            new_epoch,
                            prev_epoch.checked_next().unwrap(),
                            "published epoch must advance by 1 for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.current_epoch(),
                            new_epoch,
                            "state current_epoch mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.unpublished_writes(),
                            0,
                            "unpublished_writes must be cleared on publish for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    }
                }
                2 => {
                    // transfer_authority
                    let from = if rng.next_u32() % 2 == 0 {
                        Author::Js
                    } else {
                        Author::Wasm
                    };
                    let to = if rng.next_u32() % 2 == 0 {
                        Author::Js
                    } else {
                        Author::Wasm
                    };
                    let res = state.transfer_authority(from, to, candidate_epoch);
                    ring.push(StepRecord {
                        step,
                        action: "transfer_authority",
                        author: from,
                        epoch_words: candidate_epoch.to_words(),
                        result_variant: result_variant_name(&res),
                    });

                    // INVARIANT: State machine never accepts a stale epoch
                    if is_stale_or_mismatched_epoch && from == prev_author && from != to {
                        assert!(
                            res.is_err(),
                            "transfer_authority must never accept stale/mismatched epoch {candidate_epoch:?} (current {prev_epoch:?}) for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    }

                    if from == to {
                        let err = expect_err_with_ring(
                            res,
                            "same author transfer",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::SameAuthorTransfer { author: from },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else if from != prev_author {
                        let err = expect_err_with_ring(
                            res,
                            "transfer from non-author",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::UnauthorizedWriter {
                                expected: prev_author,
                                actual: from,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else if is_stale_or_mismatched_epoch {
                        let err = expect_err_with_ring(
                            res,
                            "transfer with stale epoch",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::StaleEpoch {
                                expected: prev_epoch,
                                actual: candidate_epoch,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else if prev_unpublished > 0 {
                        let err = expect_err_with_ring(
                            res,
                            "transfer with unpublished writes",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::UnpublishedWritesPending {
                                pending_count: prev_unpublished,
                                current_epoch: prev_epoch,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else {
                        // INVARIANT: Every accepted transfer advances the epoch
                        let new_epoch = unwrap_ok_with_ring(
                            res,
                            "valid authority transfer",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            new_epoch,
                            prev_epoch.checked_next().unwrap(),
                            "accepted transfer must advance epoch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert!(
                            new_epoch.get() > prev_epoch.get(),
                            "accepted transfer epoch must be strictly greater for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.current_epoch(),
                            new_epoch,
                            "current_epoch mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.author(),
                            to,
                            "new author must be {to:?} for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.unpublished_writes(),
                            0,
                            "pending writes must be zero for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    }
                }
                _ => {
                    // transition_mode
                    let author = if rng.next_u32() % 2 == 0 {
                        Author::Js
                    } else {
                        Author::Wasm
                    };
                    let new_mode = match rng.next_u32() % 3 {
                        0 => OwnerMode::Js,
                        1 => OwnerMode::Wasm,
                        _ => OwnerMode::Mirrored {
                            author: if rng.next_u32() % 2 == 0 {
                                Author::Js
                            } else {
                                Author::Wasm
                            },
                        },
                    };
                    let res = state.transition_mode(author, new_mode, candidate_epoch);
                    ring.push(StepRecord {
                        step,
                        action: "transition_mode",
                        author,
                        epoch_words: candidate_epoch.to_words(),
                        result_variant: result_variant_name(&res),
                    });

                    // INVARIANT: State machine never accepts a stale epoch
                    if is_stale_or_mismatched_epoch && author == prev_author {
                        assert!(
                            res.is_err(),
                            "transition_mode must never accept stale/mismatched epoch {candidate_epoch:?} (current {prev_epoch:?}) for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    }

                    if author != prev_author {
                        let err = expect_err_with_ring(
                            res,
                            "transition_mode by unauthorized author",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::UnauthorizedWriter {
                                expected: prev_author,
                                actual: author,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else if is_stale_or_mismatched_epoch {
                        let err = expect_err_with_ring(
                            res,
                            "transition_mode with stale epoch",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::StaleEpoch {
                                expected: prev_epoch,
                                actual: candidate_epoch,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else if prev_unpublished > 0 {
                        let err = expect_err_with_ring(
                            res,
                            "transition_mode with unpublished writes",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            err,
                            OwnershipError::UnpublishedWritesPending {
                                pending_count: prev_unpublished,
                                current_epoch: prev_epoch,
                            },
                            "error mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    } else {
                        let new_epoch = unwrap_ok_with_ring(
                            res,
                            "valid transition_mode",
                            SEED,
                            i,
                            step,
                            &ring,
                        );
                        assert_eq!(
                            new_epoch,
                            prev_epoch.checked_next().unwrap(),
                            "transition_mode must advance epoch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.current_epoch(),
                            new_epoch,
                            "current_epoch mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.mode(),
                            new_mode,
                            "mode mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                        assert_eq!(
                            state.author(),
                            new_mode.author(),
                            "author mismatch for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
                        );
                    }
                }
            }

            // GLOBAL INVARIANTS AFTER EVERY STEP
            assert!(
                state.current_epoch().get() >= prev_epoch.get(),
                "epoch must never decrease for seed {SEED:#018x} at iter {i} step {step}\nrecent transitions:\n{ring}"
            );
        }
    }
}

#[test]
fn property_test_snapshot_store_and_borrow_scope_interleavings() {
    const SEED: u64 = 0x5070_5580_0001_0001;
    const ITERATIONS: usize = 2_000;
    const STEPS_PER_ITER: usize = 50;

    let mut rng = TestLcg::new(SEED);

    for iter in 0..ITERATIONS {
        let mut store: PerUseSnapshotStore<MaterialDomain, [f32; 4]> =
            PerUseSnapshotStore::new(16).unwrap_or_else(|e| {
                panic!("store init failed for seed {SEED:#018x} at iter {iter}: {e:?}")
            });
        let mut scope = BorrowScope::new();

        let mut current_records: Vec<(UseRecord<MaterialDomain>, [f32; 4])> = Vec::new();
        let mut stale_records: Vec<UseRecord<MaterialDomain>> = Vec::new();
        let mut active_token: Option<BorrowToken> = None;
        let mut explicit_growth_blocked: bool = false;

        let mut expected_view_bytes: u64 = 0;
        let mut expected_wb_copies: u64 = 0;
        let mut expected_staging_copies: u64 = 0;

        for step in 0..STEPS_PER_ITER {
            let action = rng.next_u32() % 9;
            match action {
                0 => {
                    // 1. Snapshot publish
                    let mat_index = rng.next_u32() % 8;
                    let handle = Handle::<MaterialDomain>::new(mat_index, NonZeroU32::new(1).unwrap());
                    let version_val = (rng.next_u32() % 4) as u64 + 1;
                    let version = DataVersion::new(version_val);
                    let epoch = Epoch::new(rng.next_u32() as u64 % 3);
                    let color = [mat_index as f32, version_val as f32, 0.0, 1.0];

                    let res = store.record_use(handle, version, epoch, color, 16);
                    match res {
                        Ok(rec) => {
                            assert_eq!(
                                rec.store_id,
                                store.store_id(),
                                "store_id mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                            assert_eq!(
                                rec.generation,
                                store.generation(),
                                "generation mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                            let entry = store.get_use(&rec).unwrap_or_else(|e| {
                                panic!("get_use failed for newly recorded slice for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                            });
                            assert_eq!(
                                entry.data,
                                color,
                                "data mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                            current_records.push((rec, color));
                        }
                        Err(OwnershipError::ImmutableSnapshotViolation { .. }) => {
                            // Conflicting metadata/data provided for already recorded version; sound rejection
                        }
                        Err(other) => {
                            panic!("unexpected error on record_use for seed {SEED:#018x} at iter {iter} step {step}: {other:?}");
                        }
                    }
                }
                1 => {
                    // 2. Reuse of a slot after release (store reset + immediate slot 0 reuse)
                    let prev_store_generation = store.generation();
                    let new_store_generation = store.reset().unwrap_or_else(|e| {
                        panic!("store.reset failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                    });
                    assert_eq!(
                        new_store_generation,
                        prev_store_generation + 1,
                        "generation must advance on reset for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    assert_eq!(
                        store.len(),
                        0,
                        "store must be empty after reset for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    assert_eq!(
                        store.total_bytes(),
                        0,
                        "total_bytes must be 0 after reset for seed {SEED:#018x} at iter {iter} step {step}"
                    );

                    // All records from prior generation are now stale
                    for (r, _) in current_records.drain(..) {
                        stale_records.push(r);
                    }

                    // Slot reuse: immediately allocate in new generation at slice_id 0
                    let reuse_handle = Handle::<MaterialDomain>::new(100, NonZeroU32::new(1).unwrap());
                    let reuse_color = [1.0, 1.0, 1.0, 1.0];
                    let new_rec = store
                        .record_use(reuse_handle, DataVersion::new(1), Epoch::ZERO, reuse_color, 16)
                        .unwrap_or_else(|e| {
                            panic!("slot reuse record_use failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                        });
                    assert_eq!(
                        new_rec.slice_id,
                        0,
                        "reused slot must occupy slice 0 for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    assert_eq!(
                        new_rec.generation,
                        new_store_generation,
                        "reused slot generation mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    current_records.push((new_rec, reuse_color));
                }
                2 => {
                    // 3. Read with a stale generation (ABA prevention invariant)
                    if stale_records.is_empty() {
                        // Populate a stale record by recording and resetting if empty
                        if current_records.is_empty() {
                            let init_h = Handle::<MaterialDomain>::new(1, NonZeroU32::new(1).unwrap());
                            let init_rec = store
                                .record_use(init_h, DataVersion::new(1), Epoch::ZERO, [0.5, 0.5, 0.5, 1.0], 16)
                                .unwrap_or_else(|e| {
                                    panic!("record_use failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                                });
                            current_records.push((init_rec, [0.5, 0.5, 0.5, 1.0]));
                        }
                        let _ = store.reset().unwrap_or_else(|e| {
                            panic!("reset failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                        });
                        for (r, _) in current_records.drain(..) {
                            stale_records.push(r);
                        }
                    }

                    let stale_idx = (rng.next_u32() as usize) % stale_records.len();
                    let stale_rec = &stale_records[stale_idx];
                    assert!(
                        stale_rec.generation < store.generation(),
                        "stale record generation must be strictly less than current store generation for seed {SEED:#018x} at iter {iter} step {step}"
                    );

                    // INVARIANT: A stale generation is NEVER readable after slot reuse (no ABA)
                    let lookup_res = store.get_use(stale_rec);
                    match lookup_res {
                        Err(OwnershipError::StaleSliceRecord {
                            expected_generation,
                            actual_generation,
                            slice_id,
                        }) => {
                            assert_eq!(
                                expected_generation,
                                store.generation(),
                                "expected_generation mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                            assert_eq!(
                                actual_generation,
                                stale_rec.generation,
                                "actual_generation mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                            assert_eq!(
                                slice_id,
                                stale_rec.slice_id,
                                "slice_id mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                        }
                        other => panic!(
                            "INVARIANT VIOLATION: stale generation read must return StaleSliceRecord, got {other:?} for seed {SEED:#018x} at iter {iter} step {step}"
                        ),
                    }
                }
                3 => {
                    // 4. Borrow enter
                    let enter_res = scope.enter();
                    if let Some(tok) = active_token {
                        // Already borrowed: MUST reject reentry
                        match enter_res {
                            Err(OwnershipError::BorrowScopeReentry { active_token: act }) => {
                                assert_eq!(
                                    act,
                                    tok.get(),
                                    "active token mismatch on reentry for seed {SEED:#018x} at iter {iter} step {step}"
                                );
                            }
                            other => panic!(
                                "INVARIANT VIOLATION: reentrant borrow enter must return BorrowScopeReentry, got {other:?} for seed {SEED:#018x} at iter {iter} step {step}"
                            ),
                        }
                        assert_eq!(
                            scope.state(),
                            BorrowState::Borrowed,
                            "scope must remain Borrowed on reentry attempt for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                    } else {
                        // Idle: MUST succeed
                        let tok = enter_res.unwrap_or_else(|e| {
                            panic!("borrow enter while idle failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                        });
                        assert_eq!(
                            scope.state(),
                            BorrowState::Borrowed,
                            "scope must transition to Borrowed for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert!(
                            scope.is_borrowed(),
                            "is_borrowed must be true for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        active_token = Some(tok);
                    }
                }
                4 => {
                    // 5. Growth attempt during borrow
                    let pages = (rng.next_u32() % 8) + 1;
                    let prev_growth_generation = scope.growth_generation();
                    let growth_res = scope.record_growth(pages);

                    if scope.is_borrowed() {
                        // INVARIANT: Growth during an open borrow is ALWAYS rejected
                        match growth_res {
                            Err(OwnershipError::LinearMemoryGrowthBlocked { state }) => {
                                assert_eq!(
                                    state,
                                    BorrowState::Borrowed,
                                    "state must be Borrowed on growth block for seed {SEED:#018x} at iter {iter} step {step}"
                                );
                            }
                            other => panic!(
                                "INVARIANT VIOLATION: growth during borrow must return LinearMemoryGrowthBlocked, got {other:?} for seed {SEED:#018x} at iter {iter} step {step}"
                            ),
                        }
                        assert_eq!(
                            scope.growth_generation(),
                            prev_growth_generation,
                            "growth_generation must not advance when blocked for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                    } else if scope.is_growth_blocked() {
                        match growth_res {
                            Err(OwnershipError::LinearMemoryGrowthBlocked { state }) => {
                                assert_eq!(
                                    state,
                                    BorrowState::GrowthBlocked,
                                    "state must be GrowthBlocked for seed {SEED:#018x} at iter {iter} step {step}"
                                );
                            }
                            other => panic!(
                                "INVARIANT VIOLATION: growth when GrowthBlocked must fail, got {other:?} for seed {SEED:#018x} at iter {iter} step {step}"
                            ),
                        }
                        assert_eq!(
                            scope.growth_generation(),
                            prev_growth_generation,
                            "growth_generation must not advance for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                    } else {
                        let next_growth_generation = growth_res.unwrap_or_else(|e| {
                            panic!("growth while idle failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                        });
                        assert_eq!(
                            next_growth_generation,
                            prev_growth_generation + 1,
                            "growth_generation must advance by 1 for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert_eq!(
                            scope.growth_generation(),
                            next_growth_generation,
                            "scope growth_generation mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                    }
                }
                5 => {
                    // 6. Borrow exit with matching and mismatching token
                    if let Some(tok) = active_token {
                        let test_mismatch = (rng.next_u32() & 1) == 1;
                        if test_mismatch {
                            // Exit with mismatched token
                            let bogus_token = BorrowToken::new(tok.get().wrapping_add(50_000));
                            let exit_res = scope.exit(bogus_token);
                            match exit_res {
                                Err(OwnershipError::BorrowTokenMismatch { expected, actual }) => {
                                    assert_eq!(
                                        expected,
                                        tok.get(),
                                        "expected token mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                                    );
                                    assert_eq!(
                                        actual,
                                        bogus_token.get(),
                                        "actual token mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                                    );
                                }
                                other => panic!(
                                    "INVARIANT VIOLATION: exit with mismatched token must return BorrowTokenMismatch, got {other:?} for seed {SEED:#018x} at iter {iter} step {step}"
                                ),
                            }
                            // INVARIANT: Exit with a mismatched token never releases
                            assert_eq!(
                                scope.state(),
                                BorrowState::Borrowed,
                                "scope must remain Borrowed on mismatched exit for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                            assert!(
                                scope.is_borrowed(),
                                "scope must remain is_borrowed on mismatched exit for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                        } else {
                            // Exit with matching token
                            scope.exit(tok).unwrap_or_else(|e| {
                                panic!("valid borrow exit failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                            });
                            if explicit_growth_blocked {
                                assert_eq!(
                                    scope.state(),
                                    BorrowState::GrowthBlocked,
                                    "scope must return to GrowthBlocked after valid exit when growth explicitly blocked for seed {SEED:#018x} at iter {iter} step {step}"
                                );
                                assert!(
                                    scope.is_growth_blocked(),
                                    "scope must be growth blocked after exit for seed {SEED:#018x} at iter {iter} step {step}"
                                );
                            } else {
                                assert_eq!(
                                    scope.state(),
                                    BorrowState::Idle,
                                    "scope must return to Idle after valid exit for seed {SEED:#018x} at iter {iter} step {step}"
                                );
                                assert!(
                                    scope.is_idle(),
                                    "scope must be idle after valid exit for seed {SEED:#018x} at iter {iter} step {step}"
                                );
                            }
                            active_token = None;
                        }
                    } else {
                        // Attempt exit while not borrowed (Idle or GrowthBlocked)
                        let bogus_token = BorrowToken::new((rng.next_u32() as u64) + 1);
                        let exit_res = scope.exit(bogus_token);
                        match exit_res {
                            Err(OwnershipError::BorrowScopeNotActive) => {}
                            other => panic!(
                                "exit while idle must return BorrowScopeNotActive, got {other:?} for seed {SEED:#018x} at iter {iter} step {step}"
                            ),
                        }
                    }
                }
                6 => {
                    // 7. Copy accounting increments
                    let view_bytes = ((rng.next_u32() % 1024) as u64) + 16;
                    let view_res = scope.record_view_bytes(view_bytes);
                    if scope.is_borrowed() {
                        view_res.unwrap_or_else(|e| {
                            panic!("record_view_bytes while borrowed failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                        });
                        expected_view_bytes += view_bytes;
                    } else {
                        match view_res {
                            Err(OwnershipError::BorrowScopeNotActive) => {}
                            other => panic!(
                                "record_view_bytes outside borrow must return BorrowScopeNotActive, got {other:?} for seed {SEED:#018x} at iter {iter} step {step}"
                            ),
                        }
                    }

                    let wb_bytes = ((rng.next_u32() % 2048) as u64) + 32;
                    scope.record_write_buffer_copy(wb_bytes);
                    expected_wb_copies += wb_bytes;

                    let stg_bytes = ((rng.next_u32() % 512) as u64) + 8;
                    scope.record_staging_copy(stg_bytes);
                    expected_staging_copies += stg_bytes;

                    // INVARIANT: CopyAccounting counters equal the exact number of accepted copies
                    let acct = scope.accounting();
                    assert_eq!(
                        acct.bytes_view,
                        expected_view_bytes,
                        "bytes_view counter mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    assert_eq!(
                        acct.bytes_copied_write_buffer,
                        expected_wb_copies,
                        "bytes_copied_write_buffer counter mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    assert_eq!(
                        acct.bytes_copied_staging,
                        expected_staging_copies,
                        "bytes_copied_staging counter mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    assert_eq!(
                        acct.total_copied_bytes(),
                        expected_wb_copies + expected_staging_copies,
                        "total_copied_bytes mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                    assert_eq!(
                        acct.total_transported_bytes(),
                        expected_view_bytes + expected_wb_copies + expected_staging_copies,
                        "total_transported_bytes mismatch for seed {SEED:#018x} at iter {iter} step {step}"
                    );
                }
                7 => {
                    // 8. Explicit block_growth
                    let was_borrowed = scope.is_borrowed();
                    scope.block_growth();
                    explicit_growth_blocked = true;

                    // INVARIANT: block_growth always marks is_growth_blocked() true
                    assert!(
                        scope.is_growth_blocked(),
                        "scope must report is_growth_blocked() true after block_growth for seed {SEED:#018x} at iter {iter} step {step}"
                    );

                    if was_borrowed {
                        // INVARIANT: block_growth during an active borrow preserves state as Borrowed
                        assert_eq!(
                            scope.state(),
                            BorrowState::Borrowed,
                            "block_growth during borrow must preserve Borrowed state for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert!(
                            scope.is_borrowed(),
                            "scope must remain borrowed after block_growth for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                    } else {
                        // INVARIANT: block_growth when not borrowed transitions state to GrowthBlocked
                        assert_eq!(
                            scope.state(),
                            BorrowState::GrowthBlocked,
                            "state must be GrowthBlocked after block_growth for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert!(
                            !scope.is_idle(),
                            "scope must not be idle when GrowthBlocked for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                    }
                }
                _ => {
                    // 9. Explicit unblock_growth
                    let was_borrowed = scope.is_borrowed();
                    let was_growth_blocked = scope.state() == BorrowState::GrowthBlocked;
                    scope.unblock_growth();
                    explicit_growth_blocked = false;

                    if was_borrowed {
                        // INVARIANT: unblock_growth during an active borrow preserves state as Borrowed,
                        // and growth remains blocked until the borrow exits.
                        assert_eq!(
                            scope.state(),
                            BorrowState::Borrowed,
                            "unblock_growth during borrow must preserve Borrowed state for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert!(
                            scope.is_borrowed(),
                            "scope must remain borrowed after unblock_growth for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert!(
                            scope.is_growth_blocked(),
                            "growth must remain blocked while borrow active for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                    } else {
                        // INVARIANT: unblock restores Idle state and unblocks growth
                        assert_eq!(
                            scope.state(),
                            BorrowState::Idle,
                            "state must be Idle after unblock_growth for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert!(
                            scope.is_idle(),
                            "scope must be idle after unblock_growth for seed {SEED:#018x} at iter {iter} step {step}"
                        );
                        assert!(
                            !scope.is_growth_blocked(),
                            "growth must be unblocked after unblock_growth for seed {SEED:#018x} at iter {iter} step {step}"
                        );

                        // INVARIANT: unblock restores growth (verify growth succeeds immediately if previously blocked)
                        if was_growth_blocked {
                            let prev_gen = scope.growth_generation();
                            let growth_res = scope.record_growth(1);
                            let next_gen = growth_res.unwrap_or_else(|e| {
                                panic!("growth after unblock failed for seed {SEED:#018x} at iter {iter} step {step}: {e:?}")
                            });
                            assert_eq!(
                                next_gen,
                                prev_gen + 1,
                                "growth_generation must advance after unblock for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                            assert_eq!(
                                scope.growth_generation(),
                                next_gen,
                                "scope growth_generation mismatch after unblock for seed {SEED:#018x} at iter {iter} step {step}"
                            );
                        }
                    }
                }
            }
        }

        // Clean up if a borrow is still open at the end of the iteration
        if let Some(tok) = active_token {
            scope.exit(tok).unwrap_or_else(|e| {
                panic!("final cleanup exit failed for seed {SEED:#018x} at iter {iter}: {e:?}")
            });
        }
    }
}



