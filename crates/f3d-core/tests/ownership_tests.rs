//! Integration tests for single-writer epochs, authority transitions, and immutable per-use versions.
//!
//! NO-CLAIM: this state primitive is not an ECS, full scene ownership analysis, or GPU proof.

use core::num::NonZeroU32;
use f3d_core::handle::{Handle, MaterialDomain, RegionDomain};
use f3d_core::ownership::{
    Author, DataVersion, Epoch, OwnerMode, OwnershipError, PerUseByteBuffer, PerUseSnapshotStore,
    RegionState,
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

    // JS publishes uncommitted writes at current epoch (0)
    let ep1 = state
        .publish(Author::Js, Epoch::ZERO)
        .expect("JS publish succeeds");
    assert_eq!(ep1.get(), 1);
    assert_eq!(state.unpublished_writes(), 0);

    // Legal transfer from JS to Wasm at epoch 1
    state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(1))
        .expect("transfer to Wasm succeeds");
    assert_eq!(state.author(), Author::Wasm);
    assert_eq!(state.mode(), OwnerMode::Wasm);

    // Wasm records a mutation
    let v3 = state.record_write(Author::Wasm).expect("Wasm write succeeds");
    assert_eq!(v3.get(), 3);
    assert_eq!(state.unpublished_writes(), 1);

    // Wasm publishes at epoch 1
    let ep2 = state
        .publish(Author::Wasm, Epoch::new(1))
        .expect("Wasm publish succeeds");
    assert_eq!(ep2.get(), 2);
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

    // JS writes and publishes
    state.record_write(Author::Js).expect("JS write ok");
    state
        .publish(Author::Js, Epoch::ZERO)
        .expect("JS publish ok");

    // Transfer in mirrored mode changes author to Wasm
    state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(1))
        .expect("transfer in mirrored mode ok");
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
    let slice_a = store.get_slice(use_a.slice_id()).expect("lookup slice A");
    let slice_b = store.get_slice(use_b.slice_id()).expect("lookup slice B");

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

    // Once published, transfer succeeds
    state
        .publish(Author::Js, Epoch::ZERO)
        .expect("publish pending writes");
    state
        .transfer_authority(Author::Js, Author::Wasm, Epoch::new(1))
        .expect("transfer now succeeds");
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

    // Recording duplicate with IDENTICAL data is idempotent and returns existing record
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
            detail: "conflicting data provided for existing immutable version snapshot",
        }
    );

    // Verify first snapshot was not corrupted
    let slice0 = store.get_slice(rec1.slice_id()).expect("lookup slice");
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
