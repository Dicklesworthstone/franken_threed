//! Single-writer epochs, ownership state transitions, and immutable per-use data versions.
//!
//! NO-CLAIM: this state primitive is not an ECS, full scene ownership analysis, or GPU proof.
//!
//! # Architecture
//!
//! Across the JavaScript and WebAssembly execution boundary, resources are assigned an
//! [`OwnerMode`] retaining exactly one authoritative [`Author`] at any point in time.
//! State transitions enforce that:
//!
//! 1. Only the current author may record writes and advance [`DataVersion`].
//! 2. Uncommitted writes must be explicitly published via [`Epoch`] progression before
//!    authority can be transferred.
//! 3. Authority transfer strictly advances the publication [`Epoch`] and rejects stale epochs
//!    or pending unpublished writes, preventing stale command replay across author roundtrips (ABA).
//! 4. Mode transitions via [`RegionState::transition_mode`] require explicit authorization from the
//!    current author, zero pending writes, and monotonic epoch advance.
//! 5. Versions and epochs are monotonic 64-bit counters decomposed into two explicit 32-bit words
//!    at host boundaries to prevent precision loss from JavaScript `Number` (IEEE 754 float64).
//! 6. Pass planning and bridge snapshots retain immutable per-use versions and buffer slices
//!    so that mutations to a shared resource between passes (e.g. Red in Pass A, Blue in Pass B)
//!    coexist in the same submission schedule without in-place slot overwrites.
//! 7. [`PerUseByteBuffer`] and [`PerUseSnapshotStore`] are authoritative, non-clonable allocation
//!    arenas. Store identifiers are allocated via checked CAS loop to prevent silent counter wraparound.
//!    Old records after store reset (ABA slice reuse) or records from foreign stores are strictly rejected.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::error::HandleError;
use crate::handle::{Domain, Handle, RegionDomain};

static NEXT_STORE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Allocates a globally unique store identifier, failing cleanly with typed overflow error
/// if the counter reaches `u64::MAX`.
fn allocate_store_id() -> Result<u64, OwnershipError> {
    let mut current = NEXT_STORE_COUNTER.load(Ordering::Relaxed);
    loop {
        let next = current
            .checked_add(1)
            .ok_or(OwnershipError::StoreIdOverflow { current })?;
        match NEXT_STORE_COUNTER.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(current),
            Err(actual) => current = actual,
        }
    }
}

/// Set the next store identifier counter for near-max overflow regression testing.
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn set_next_store_id_for_testing(id: u64) {
    NEXT_STORE_COUNTER.store(id, Ordering::Relaxed);
}

/// Authoritative writer identity across the host / runtime boundary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Author {
    /// JavaScript host environment is the authoritative writer.
    Js,
    /// WebAssembly core runtime is the authoritative writer.
    Wasm,
}

impl Author {
    /// Returns the other author.
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Js => Self::Wasm,
            Self::Wasm => Self::Js,
        }
    }
}

impl fmt::Display for Author {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Js => write!(f, "Js"),
            Self::Wasm => write!(f, "Wasm"),
        }
    }
}

/// Single-writer ownership mode for a resource or memory region.
///
/// Retains exactly one authoritative author at any point in time.
/// Default fallback is [`OwnerMode::Js`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OwnerMode {
    /// Solely owned and authored by JavaScript.
    Js,
    /// Solely owned and authored by WebAssembly.
    Wasm,
    /// Mirrored across the boundary, with an explicit authoritative author.
    Mirrored {
        /// Authoritative writer for the mirrored resource.
        author: Author,
    },
}

impl Default for OwnerMode {
    fn default() -> Self {
        Self::Js
    }
}

impl OwnerMode {
    /// Returns the authoritative writer for this mode.
    #[inline]
    pub const fn author(self) -> Author {
        match self {
            Self::Js => Author::Js,
            Self::Wasm => Author::Wasm,
            Self::Mirrored { author } => author,
        }
    }

    /// Returns `true` if this mode is mirrored across the boundary.
    #[inline]
    pub const fn is_mirrored(self) -> bool {
        matches!(self, Self::Mirrored { .. })
    }
}

impl fmt::Display for OwnerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Js => write!(f, "Js"),
            Self::Wasm => write!(f, "Wasm"),
            Self::Mirrored { author } => write!(f, "Mirrored({author})"),
        }
    }
}

/// Monotonically increasing publication epoch.
///
/// Decomposes into two 32-bit words at host ABI boundaries to prevent
/// precision loss from JavaScript `Number` (IEEE 754 float64).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct Epoch(u64);

impl Epoch {
    /// The initial zero epoch.
    pub const ZERO: Self = Self(0);

    /// Construct an epoch from a raw `u64`.
    #[inline]
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Extract the raw `u64` epoch counter.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Decompose into `(high_u32, low_u32)` for host ABI transport without JS `Number` conversion.
    #[inline]
    pub const fn to_words(self) -> (u32, u32) {
        ((self.0 >> 32) as u32, self.0 as u32)
    }

    /// Reconstruct an [`Epoch`] from `(high_u32, low_u32)` words.
    #[inline]
    pub const fn from_words(high: u32, low: u32) -> Self {
        Self(((high as u64) << 32) | (low as u64))
    }

    /// High 32 bits of the epoch counter.
    #[inline]
    pub const fn high_u32(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Low 32 bits of the epoch counter.
    #[inline]
    pub const fn low_u32(self) -> u32 {
        self.0 as u32
    }

    /// Monotonically advance to the next epoch, checking for `u64` overflow.
    #[inline]
    pub fn checked_next(self) -> Result<Self, OwnershipError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(OwnershipError::EpochOverflow { current: self.0 })
    }
}

impl fmt::Debug for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Epoch({})", self.0)
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Monotonically increasing data version for per-use immutability.
///
/// Preserves precise version identity across host ABI boundaries using
/// two 32-bit words.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct DataVersion(u64);

impl Default for DataVersion {
    fn default() -> Self {
        Self::INITIAL
    }
}

impl DataVersion {
    /// Initial version for a freshly created resource.
    pub const INITIAL: Self = Self(1);

    /// Construct a version from a raw `u64`.
    #[inline]
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Extract the raw `u64` version counter.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Decompose into `(high_u32, low_u32)` for host ABI transport without JS `Number` conversion.
    #[inline]
    pub const fn to_words(self) -> (u32, u32) {
        ((self.0 >> 32) as u32, self.0 as u32)
    }

    /// Reconstruct a [`DataVersion`] from `(high_u32, low_u32)` words.
    #[inline]
    pub const fn from_words(high: u32, low: u32) -> Self {
        Self(((high as u64) << 32) | (low as u64))
    }

    /// High 32 bits of the version counter.
    #[inline]
    pub const fn high_u32(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Low 32 bits of the version counter.
    #[inline]
    pub const fn low_u32(self) -> u32 {
        self.0 as u32
    }

    /// Monotonically advance to the next version, checking for `u64` overflow.
    #[inline]
    pub fn checked_next(self) -> Result<Self, OwnershipError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(OwnershipError::VersionOverflow { current: self.0 })
    }
}

impl fmt::Debug for DataVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataVersion({})", self.0)
    }
}

impl fmt::Display for DataVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Errors occurring during ownership transitions, epoch publications, or versioned snapshotting.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OwnershipError {
    /// An unauthorized writer attempted to mutate, publish, or transition state.
    UnauthorizedWriter {
        /// The author permitted to write.
        expected: Author,
        /// The author that attempted to write.
        actual: Author,
    },
    /// Authority transfer rejected because uncommitted writes are pending publication.
    UnpublishedWritesPending {
        /// Number of pending uncommitted writes.
        pending_count: u32,
        /// The current unpublished epoch.
        current_epoch: Epoch,
    },
    /// An operation was attempted against a stale or mismatched epoch.
    StaleEpoch {
        /// The current actual epoch.
        expected: Epoch,
        /// The stale epoch provided by caller.
        actual: Epoch,
    },
    /// Attempted to transfer authority to the existing author.
    SameAuthorTransfer {
        /// The author attempted as both source and destination.
        author: Author,
    },
    /// Epoch counter overflowed `u64::MAX`.
    EpochOverflow {
        /// Current epoch value that failed to increment.
        current: u64,
    },
    /// Version counter overflowed `u64::MAX`.
    VersionOverflow {
        /// Current version value that failed to increment.
        current: u64,
    },
    /// Globally unique store identifier counter overflowed `u64::MAX`.
    StoreIdOverflow {
        /// Current store counter value that failed to increment.
        current: u64,
    },
    /// Violation of snapshot immutability: attempted mutation or conflicting metadata for an existing version snapshot.
    ImmutableSnapshotViolation {
        /// Version that was attempted to be re-recorded or corrupted.
        version: DataVersion,
        /// Description of the attempted violation.
        detail: &'static str,
    },
    /// A use record from an earlier allocation generation was accessed after store reset (ABA slice reuse prevention).
    StaleSliceRecord {
        /// Current generation of the buffer/store.
        expected_generation: u64,
        /// Stale generation from the provided record.
        actual_generation: u64,
        /// Slice ID referenced by the stale record.
        slice_id: u32,
    },
    /// A use record from a different store instance was provided to this store.
    ForeignSliceRecord {
        /// Store ID of the receiving store.
        expected_store: u64,
        /// Foreign store ID found on the record.
        actual_store: u64,
    },
    /// The provided use record does not match the stored metadata for that slice (offset or size tampering).
    SliceIdentityMismatch {
        /// Slice ID where metadata mismatched.
        slice_id: u32,
    },
    /// Slice offset or length exceeds buffer bounds or overflows address calculations.
    SliceOutOfBounds {
        /// Slice starting byte offset.
        offset: u64,
        /// Slice byte length.
        length: u64,
        /// Current buffer length.
        buffer_len: usize,
    },
    /// Requested slice was not found in the per-use buffer.
    SliceNotFound {
        /// The requested slice identifier.
        slice_id: u32,
    },
    /// Slice byte length is invalid or cannot fit into host address space.
    InvalidSliceLength {
        /// The invalid byte length.
        length: u64,
    },
    /// Alignment requirement was not satisfied (e.g. zero or not a power of two).
    InvalidAlignment {
        /// The requested alignment that was invalid.
        alignment: u64,
    },
    /// Underlying handle validation failed.
    Handle(HandleError),
}

impl fmt::Display for OwnershipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnauthorizedWriter { expected, actual } => {
                write!(
                    f,
                    "unauthorized writer: author '{actual}' cannot write (authoritative author is '{expected}')"
                )
            }
            Self::UnpublishedWritesPending {
                pending_count,
                current_epoch,
            } => {
                write!(
                    f,
                    "authority transfer rejected: {pending_count} uncommitted write(s) pending at epoch {current_epoch}"
                )
            }
            Self::StaleEpoch { expected, actual } => {
                write!(
                    f,
                    "stale epoch: expected current epoch {expected}, got {actual}"
                )
            }
            Self::SameAuthorTransfer { author } => {
                write!(f, "invalid transfer: authority already held by '{author}'")
            }
            Self::EpochOverflow { current } => {
                write!(f, "epoch counter overflow at {current}")
            }
            Self::VersionOverflow { current } => {
                write!(f, "data version counter overflow at {current}")
            }
            Self::StoreIdOverflow { current } => {
                write!(f, "store ID allocation overflow at {current}")
            }
            Self::ImmutableSnapshotViolation { version, detail } => {
                write!(
                    f,
                    "immutable snapshot violation at version {version}: {detail}"
                )
            }
            Self::StaleSliceRecord {
                expected_generation,
                actual_generation,
                slice_id,
            } => {
                write!(
                    f,
                    "stale slice record for slice {slice_id}: buffer generation is {expected_generation}, record has stale generation {actual_generation}"
                )
            }
            Self::ForeignSliceRecord {
                expected_store,
                actual_store,
            } => {
                write!(
                    f,
                    "foreign slice record rejected: store ID {expected_store} cannot resolve record from store ID {actual_store}"
                )
            }
            Self::SliceIdentityMismatch { slice_id } => {
                write!(
                    f,
                    "slice identity mismatch for slice {slice_id}: record metadata does not match stored entry"
                )
            }
            Self::SliceOutOfBounds {
                offset,
                length,
                buffer_len,
            } => {
                write!(
                    f,
                    "slice out of bounds: range [{offset}..{offset}+{length}] exceeds buffer length {buffer_len}"
                )
            }
            Self::SliceNotFound { slice_id } => {
                write!(f, "per-use slice {slice_id} not found in buffer")
            }
            Self::InvalidSliceLength { length } => {
                write!(f, "invalid slice length {length}: exceeds addressable memory")
            }
            Self::InvalidAlignment { alignment } => {
                write!(
                    f,
                    "invalid buffer alignment {alignment}: must be a non-zero power of two"
                )
            }
            Self::Handle(e) => write!(f, "handle error in ownership: {e}"),
        }
    }
}

impl core::error::Error for OwnershipError {}

impl From<HandleError> for OwnershipError {
    fn from(err: HandleError) -> Self {
        Self::Handle(err)
    }
}

/// Tracks single-writer authority, epoch transitions, and write versioning for a resource or region.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegionState<D: Domain = RegionDomain> {
    handle: Handle<D>,
    mode: OwnerMode,
    current_epoch: Epoch,
    current_version: DataVersion,
    unpublished_writes: u32,
}

impl<D: Domain> RegionState<D> {
    /// Construct a new region state with default [`OwnerMode::Js`], zero epoch, and initial version.
    pub const fn new(handle: Handle<D>) -> Self {
        Self {
            handle,
            mode: OwnerMode::Js,
            current_epoch: Epoch::ZERO,
            current_version: DataVersion::INITIAL,
            unpublished_writes: 0,
        }
    }

    /// Construct a new region state with an explicit initial [`OwnerMode`].
    pub const fn with_mode(handle: Handle<D>, mode: OwnerMode) -> Self {
        Self {
            handle,
            mode,
            current_epoch: Epoch::ZERO,
            current_version: DataVersion::INITIAL,
            unpublished_writes: 0,
        }
    }

    /// The tracked resource or region handle.
    #[inline]
    pub const fn handle(&self) -> Handle<D> {
        self.handle
    }

    /// Current ownership mode.
    #[inline]
    pub const fn mode(&self) -> OwnerMode {
        self.mode
    }

    /// Current authoritative author.
    #[inline]
    pub const fn author(&self) -> Author {
        self.mode.author()
    }

    /// Current published epoch.
    #[inline]
    pub const fn current_epoch(&self) -> Epoch {
        self.current_epoch
    }

    /// Current data version.
    #[inline]
    pub const fn current_version(&self) -> DataVersion {
        self.current_version
    }

    /// Number of uncommitted writes pending publication.
    #[inline]
    pub const fn unpublished_writes(&self) -> u32 {
        self.unpublished_writes
    }

    /// Record a mutation from the specified author.
    ///
    /// # Errors
    ///
    /// - Returns [`OwnershipError::UnauthorizedWriter`] if `author` is not authoritative.
    /// - Returns [`OwnershipError::VersionOverflow`] if the version counter overflows.
    pub fn record_write(&mut self, author: Author) -> Result<DataVersion, OwnershipError> {
        let expected = self.author();
        if author != expected {
            return Err(OwnershipError::UnauthorizedWriter {
                expected,
                actual: author,
            });
        }

        let next_version = self.current_version.checked_next()?;
        self.unpublished_writes = self
            .unpublished_writes
            .checked_add(1)
            .ok_or(OwnershipError::VersionOverflow {
                current: self.unpublished_writes as u64,
            })?;
        self.current_version = next_version;
        Ok(self.current_version)
    }

    /// Publish pending writes, advancing the publication epoch and clearing unpublished writes.
    ///
    /// # Errors
    ///
    /// - Returns [`OwnershipError::UnauthorizedWriter`] if `author` is not authoritative.
    /// - Returns [`OwnershipError::StaleEpoch`] if `expected_epoch` does not match `self.current_epoch`.
    /// - Returns [`OwnershipError::EpochOverflow`] if the epoch counter overflows.
    pub fn publish(
        &mut self,
        author: Author,
        expected_epoch: Epoch,
    ) -> Result<Epoch, OwnershipError> {
        let expected = self.author();
        if author != expected {
            return Err(OwnershipError::UnauthorizedWriter {
                expected,
                actual: author,
            });
        }

        if expected_epoch != self.current_epoch {
            return Err(OwnershipError::StaleEpoch {
                expected: self.current_epoch,
                actual: expected_epoch,
            });
        }

        let next_epoch = self.current_epoch.checked_next()?;
        self.current_epoch = next_epoch;
        self.unpublished_writes = 0;
        Ok(self.current_epoch)
    }

    /// Transfer authority from `from` to `to`.
    ///
    /// Requires that:
    /// 1. `from != to`
    /// 2. `from` is the current authoritative author.
    /// 3. `at_epoch == self.current_epoch` (epoch is fresh).
    /// 4. `unpublished_writes == 0` (all writes have been published).
    ///
    /// Upon successful validation, this advances [`Epoch`] to prevent replay of stale transfer commands,
    /// updates authority, and returns the newly advanced [`Epoch`].
    ///
    /// # Errors
    ///
    /// - Returns [`OwnershipError::SameAuthorTransfer`] if `from == to`.
    /// - Returns [`OwnershipError::UnauthorizedWriter`] if `from` is not authoritative.
    /// - Returns [`OwnershipError::StaleEpoch`] if `at_epoch` is stale.
    /// - Returns [`OwnershipError::UnpublishedWritesPending`] if uncommitted writes exist.
    /// - Returns [`OwnershipError::EpochOverflow`] if advancing the epoch counter overflows.
    pub fn transfer_authority(
        &mut self,
        from: Author,
        to: Author,
        at_epoch: Epoch,
    ) -> Result<Epoch, OwnershipError> {
        if from == to {
            return Err(OwnershipError::SameAuthorTransfer { author: from });
        }

        let expected = self.author();
        if from != expected {
            return Err(OwnershipError::UnauthorizedWriter {
                expected,
                actual: from,
            });
        }

        if at_epoch != self.current_epoch {
            return Err(OwnershipError::StaleEpoch {
                expected: self.current_epoch,
                actual: at_epoch,
            });
        }

        if self.unpublished_writes > 0 {
            return Err(OwnershipError::UnpublishedWritesPending {
                pending_count: self.unpublished_writes,
                current_epoch: self.current_epoch,
            });
        }

        let next_epoch = self.current_epoch.checked_next()?;
        self.current_epoch = next_epoch;

        match self.mode {
            OwnerMode::Js => self.mode = OwnerMode::Wasm,
            OwnerMode::Wasm => self.mode = OwnerMode::Js,
            OwnerMode::Mirrored { .. } => self.mode = OwnerMode::Mirrored { author: to },
        }

        Ok(self.current_epoch)
    }

    /// Transition to a new ownership mode at the specified epoch with zero pending writes.
    ///
    /// Requires that `author` is the current authoritative author, `at_epoch` matches `self.current_epoch`,
    /// and `unpublished_writes == 0`. Advances the publication epoch to prevent replay attacks.
    ///
    /// # Errors
    ///
    /// - Returns [`OwnershipError::UnauthorizedWriter`] if `author` is not authoritative.
    /// - Returns [`OwnershipError::StaleEpoch`] if `at_epoch` is stale.
    /// - Returns [`OwnershipError::UnpublishedWritesPending`] if uncommitted writes exist.
    /// - Returns [`OwnershipError::EpochOverflow`] if advancing the epoch counter overflows.
    pub fn transition_mode(
        &mut self,
        author: Author,
        new_mode: OwnerMode,
        at_epoch: Epoch,
    ) -> Result<Epoch, OwnershipError> {
        let expected = self.author();
        if author != expected {
            return Err(OwnershipError::UnauthorizedWriter {
                expected,
                actual: author,
            });
        }

        if at_epoch != self.current_epoch {
            return Err(OwnershipError::StaleEpoch {
                expected: self.current_epoch,
                actual: at_epoch,
            });
        }

        if self.unpublished_writes > 0 {
            return Err(OwnershipError::UnpublishedWritesPending {
                pending_count: self.unpublished_writes,
                current_epoch: self.current_epoch,
            });
        }

        let next_epoch = self.current_epoch.checked_next()?;
        self.current_epoch = next_epoch;
        self.mode = new_mode;
        Ok(self.current_epoch)
    }
}

/// Immutable record of a specific version of a resource bound to a pass or draw use.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UseRecord<D: Domain> {
    resource: Handle<D>,
    version: DataVersion,
    epoch: Epoch,
    store_id: u64,
    generation: u64,
    slice_id: u32,
    byte_offset: u64,
    byte_length: u64,
}

impl<D: Domain> UseRecord<D> {
    /// Construct a new use record.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        resource: Handle<D>,
        version: DataVersion,
        epoch: Epoch,
        store_id: u64,
        generation: u64,
        slice_id: u32,
        byte_offset: u64,
        byte_length: u64,
    ) -> Self {
        Self {
            resource,
            version,
            epoch,
            store_id,
            generation,
            slice_id,
            byte_offset,
            byte_length,
        }
    }

    /// The tracked resource handle.
    #[inline]
    pub const fn resource(&self) -> Handle<D> {
        self.resource
    }

    /// Exact data version at the time of recording.
    #[inline]
    pub const fn version(&self) -> DataVersion {
        self.version
    }

    /// Publication epoch at the time of recording.
    #[inline]
    pub const fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Store instance identifier that allocated this slice.
    #[inline]
    pub const fn store_id(&self) -> u64 {
        self.store_id
    }

    /// Allocation generation of the store at the time this slice was allocated.
    #[inline]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Unique index of this slice within the store generation.
    #[inline]
    pub const fn slice_id(&self) -> u32 {
        self.slice_id
    }

    /// Starting byte offset of this slice within the combined buffer.
    #[inline]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Total byte length of this slice.
    #[inline]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }
}

/// An immutable snapshot entry pairing a [`UseRecord`] with typed snapshot data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotEntry<D: Domain, T> {
    record: UseRecord<D>,
    data: T,
}

impl<D: Domain, T> SnapshotEntry<D, T> {
    /// The immutable use record metadata.
    #[inline]
    pub const fn record(&self) -> &UseRecord<D> {
        &self.record
    }

    /// The snapshot data payload.
    #[inline]
    pub const fn data(&self) -> &T {
        &self.data
    }
}

/// Aligns `offset` up to the next multiple of `alignment`.
/// `alignment` must be a power of two.
///
/// Uses checked arithmetic to prevent panics on `u64` overflow.
#[inline]
const fn align_up(offset: u64, alignment: u64) -> Result<u64, OwnershipError> {
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return Err(OwnershipError::InvalidAlignment { alignment });
    }
    let mask = alignment - 1;
    let added = match offset.checked_add(mask) {
        Some(val) => val,
        None => return Err(OwnershipError::VersionOverflow { current: offset }),
    };
    let aligned = added & !mask;
    Ok(aligned)
}

/// Storage pool for immutable per-use snapshots of value types.
///
/// Admitted types `T` must be plain value types (`Copy + PartialEq + 'static`, such as uniform structs
/// or fixed arrays) with no interior mutability or aliased references.
///
/// Each store instance possesses authoritative, non-clonable allocation identity.
/// Ensures that multiple uses of a mutating resource across passes allocate distinct slices
/// and preserve historical versions without in-place overwrite.
#[derive(Debug)]
pub struct PerUseSnapshotStore<D: Domain, T: Copy + PartialEq + 'static> {
    store_id: u64,
    generation: u64,
    entries: Vec<SnapshotEntry<D, T>>,
    alignment: u64,
    current_offset: u64,
}

impl<D: Domain, T: Copy + PartialEq + 'static> PerUseSnapshotStore<D, T> {
    /// Create a new snapshot store with byte alignment for subsequent slices (e.g. 256 bytes for WebGPU UBOs).
    ///
    /// Allocates a globally unique store identifier via checked CAS increment.
    pub fn new(alignment: u64) -> Result<Self, OwnershipError> {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(OwnershipError::InvalidAlignment { alignment });
        }
        let store_id = allocate_store_id()?;
        Ok(Self {
            store_id,
            generation: 1,
            entries: Vec::new(),
            alignment,
            current_offset: 0,
        })
    }

    /// Construct a store with an explicit store ID for testing.
    #[doc(hidden)]
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_store_id_for_testing(store_id: u64, alignment: u64) -> Result<Self, OwnershipError> {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(OwnershipError::InvalidAlignment { alignment });
        }
        Ok(Self {
            store_id,
            generation: 1,
            entries: Vec::new(),
            alignment,
            current_offset: 0,
        })
    }

    /// The store instance identifier.
    #[inline]
    pub const fn store_id(&self) -> u64 {
        self.store_id
    }

    /// The current allocation generation of this store.
    #[inline]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Record a pass use of a resource with its snapshot data.
    ///
    /// If the resource at the given `version` was already recorded with identical metadata
    /// (`epoch`, `byte_length`) and `data`, returns the existing [`UseRecord`].
    /// If conflicting metadata or data is provided for an already recorded version,
    /// returns [`OwnershipError::ImmutableSnapshotViolation`].
    pub fn record_use(
        &mut self,
        handle: Handle<D>,
        version: DataVersion,
        epoch: Epoch,
        data: T,
        byte_length: u64,
    ) -> Result<UseRecord<D>, OwnershipError> {
        for entry in &self.entries {
            if entry.record.resource == handle && entry.record.version == version {
                if entry.record.epoch == epoch
                    && entry.record.byte_length == byte_length
                    && entry.data == data
                {
                    return Ok(entry.record);
                }
                return Err(OwnershipError::ImmutableSnapshotViolation {
                    version,
                    detail: "conflicting metadata or data provided for existing immutable version snapshot",
                });
            }
        }

        let aligned_offset = align_up(self.current_offset, self.alignment)?;
        let slice_end = aligned_offset
            .checked_add(byte_length)
            .ok_or(OwnershipError::VersionOverflow {
                current: aligned_offset,
            })?;

        // Ensure the end offset and the subsequent aligned offset do not overflow u64
        let _next_aligned = align_up(slice_end, self.alignment).map_err(|_| {
            OwnershipError::VersionOverflow {
                current: aligned_offset,
            }
        })?;

        let slice_id = u32::try_from(self.entries.len()).map_err(|_| {
            OwnershipError::VersionOverflow {
                current: self.entries.len() as u64,
            }
        })?;

        let record = UseRecord::new(
            handle,
            version,
            epoch,
            self.store_id,
            self.generation,
            slice_id,
            aligned_offset,
            byte_length,
        );

        self.entries.push(SnapshotEntry { record, data });
        self.current_offset = slice_end;
        Ok(record)
    }

    /// Look up a slice by its unique `slice_id`.
    pub fn get_slice(&self, slice_id: u32) -> Result<&SnapshotEntry<D, T>, OwnershipError> {
        self.entries
            .get(slice_id as usize)
            .ok_or(OwnershipError::SliceNotFound { slice_id })
    }

    /// Look up an entry by its [`UseRecord`], verifying store identity, generation, and record metadata.
    pub fn get_use(&self, record: &UseRecord<D>) -> Result<&SnapshotEntry<D, T>, OwnershipError> {
        if record.store_id != self.store_id {
            return Err(OwnershipError::ForeignSliceRecord {
                expected_store: self.store_id,
                actual_store: record.store_id,
            });
        }
        if record.generation != self.generation {
            return Err(OwnershipError::StaleSliceRecord {
                expected_generation: self.generation,
                actual_generation: record.generation,
                slice_id: record.slice_id,
            });
        }
        let entry = self
            .entries
            .get(record.slice_id as usize)
            .ok_or(OwnershipError::SliceNotFound {
                slice_id: record.slice_id,
            })?;
        if entry.record != *record {
            return Err(OwnershipError::SliceIdentityMismatch {
                slice_id: record.slice_id,
            });
        }
        Ok(entry)
    }

    /// Look up the first entry for a specific handle and version.
    pub fn get_by_version(
        &self,
        handle: Handle<D>,
        version: DataVersion,
    ) -> Option<&SnapshotEntry<D, T>> {
        self.entries
            .iter()
            .find(|e| e.record.resource == handle && e.record.version == version)
    }

    /// All recorded snapshot entries in this store.
    #[inline]
    pub fn entries(&self) -> &[SnapshotEntry<D, T>] {
        &self.entries
    }

    /// Number of recorded slices.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no slices have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes spanned by all allocated slices including internal alignment padding.
    #[inline]
    pub const fn total_bytes(&self) -> u64 {
        self.current_offset
    }

    /// Reset the store for a subsequent submission cycle or frame, advancing the allocation generation
    /// to invalidate old [`UseRecord`] handles (ABA prevention).
    pub fn reset(&mut self) -> Result<u64, OwnershipError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(OwnershipError::VersionOverflow {
                current: self.generation,
            })?;
        self.entries.clear();
        self.current_offset = 0;
        Ok(self.generation)
    }
}

/// Contiguous byte buffer for immutable per-use uniform and storage slices.
///
/// Ensures byte slices written for distinct versions of a resource are immutable,
/// contiguous, aligned, and never overwritten by subsequent mutations.
/// Each buffer instance is authoritative and non-clonable.
#[derive(Debug)]
pub struct PerUseByteBuffer<D: Domain> {
    store_id: u64,
    generation: u64,
    records: Vec<UseRecord<D>>,
    bytes: Vec<u8>,
    alignment: u64,
}

impl<D: Domain> PerUseByteBuffer<D> {
    /// Create a new byte buffer with specified slice alignment (e.g. 256 for WebGPU UBOs).
    ///
    /// Allocates a globally unique store identifier via checked CAS increment.
    pub fn new(alignment: u64) -> Result<Self, OwnershipError> {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(OwnershipError::InvalidAlignment { alignment });
        }
        let store_id = allocate_store_id()?;
        Ok(Self {
            store_id,
            generation: 1,
            records: Vec::new(),
            bytes: Vec::new(),
            alignment,
        })
    }

    /// Construct a byte buffer with an explicit store ID for testing.
    #[doc(hidden)]
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_store_id_for_testing(store_id: u64, alignment: u64) -> Result<Self, OwnershipError> {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(OwnershipError::InvalidAlignment { alignment });
        }
        Ok(Self {
            store_id,
            generation: 1,
            records: Vec::new(),
            bytes: Vec::new(),
            alignment,
        })
    }

    /// The store instance identifier.
    #[inline]
    pub const fn store_id(&self) -> u64 {
        self.store_id
    }

    /// The current allocation generation of this buffer.
    #[inline]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Append an immutable byte slice for a resource version and epoch.
    ///
    /// Validates full metadata (`epoch`, `byte_length`) and byte content for idempotent deduplication.
    /// Performs all offset and alignment calculations with checked bounds before mutating buffer memory.
    ///
    /// # Errors
    ///
    /// - Returns [`OwnershipError::ImmutableSnapshotViolation`] if attempting to re-record
    ///   an existing version for the same handle with differing content or metadata.
    /// - Returns [`OwnershipError::VersionOverflow`] if buffer offsets exceed `u64::MAX`.
    /// - Returns [`OwnershipError::InvalidSliceLength`] if slice cannot fit in addressable memory.
    pub fn append_slice(
        &mut self,
        handle: Handle<D>,
        version: DataVersion,
        epoch: Epoch,
        slice_bytes: &[u8],
    ) -> Result<UseRecord<D>, OwnershipError> {
        let byte_len = slice_bytes.len() as u64;

        for record in &self.records {
            if record.resource == handle && record.version == version {
                let start = match usize::try_from(record.byte_offset) {
                    Ok(s) => s,
                    Err(_) => {
                        return Err(OwnershipError::SliceOutOfBounds {
                            offset: record.byte_offset,
                            length: record.byte_length,
                            buffer_len: self.bytes.len(),
                        });
                    }
                };
                let len = match usize::try_from(record.byte_length) {
                    Ok(l) => l,
                    Err(_) => {
                        return Err(OwnershipError::SliceOutOfBounds {
                            offset: record.byte_offset,
                            length: record.byte_length,
                            buffer_len: self.bytes.len(),
                        });
                    }
                };
                let end = match start.checked_add(len) {
                    Some(e) => e,
                    None => {
                        return Err(OwnershipError::SliceOutOfBounds {
                            offset: record.byte_offset,
                            length: record.byte_length,
                            buffer_len: self.bytes.len(),
                        });
                    }
                };
                let existing = self.bytes.get(start..end).ok_or(OwnershipError::SliceOutOfBounds {
                    offset: record.byte_offset,
                    length: record.byte_length,
                    buffer_len: self.bytes.len(),
                })?;

                if record.epoch == epoch && record.byte_length == byte_len && existing == slice_bytes {
                    return Ok(*record);
                }
                return Err(OwnershipError::ImmutableSnapshotViolation {
                    version,
                    detail: "conflicting metadata or byte content provided for existing immutable version",
                });
            }
        }

        let current_len = self.bytes.len() as u64;
        let aligned_offset = align_up(current_len, self.alignment)?;
        let slice_end = aligned_offset
            .checked_add(byte_len)
            .ok_or(OwnershipError::VersionOverflow {
                current: aligned_offset,
            })?;

        // Ensure the end offset and the subsequent aligned offset do not overflow u64
        let _next_aligned = align_up(slice_end, self.alignment).map_err(|_| {
            OwnershipError::VersionOverflow {
                current: aligned_offset,
            }
        })?;

        let _ = usize::try_from(slice_end).map_err(|_| OwnershipError::InvalidSliceLength {
            length: slice_end,
        })?;

        let padding = (aligned_offset - current_len) as usize;
        self.bytes.resize(self.bytes.len() + padding, 0);
        let byte_offset = self.bytes.len() as u64;
        self.bytes.extend_from_slice(slice_bytes);

        let slice_id = u32::try_from(self.records.len()).map_err(|_| {
            OwnershipError::VersionOverflow {
                current: self.records.len() as u64,
            }
        })?;

        let record = UseRecord::new(
            handle,
            version,
            epoch,
            self.store_id,
            self.generation,
            slice_id,
            byte_offset,
            byte_len,
        );
        self.records.push(record);
        Ok(record)
    }

    /// Look up an immutable slice by its [`UseRecord`].
    ///
    /// Strictly verifies store instance ID, allocation generation, and stored record equality
    /// to prevent reading recycled slices (ABA) or trusting foreign/tampered records.
    /// Slices bytes with checked index calculations without panics.
    pub fn get_slice(&self, record: &UseRecord<D>) -> Result<&[u8], OwnershipError> {
        if record.store_id != self.store_id {
            return Err(OwnershipError::ForeignSliceRecord {
                expected_store: self.store_id,
                actual_store: record.store_id,
            });
        }
        if record.generation != self.generation {
            return Err(OwnershipError::StaleSliceRecord {
                expected_generation: self.generation,
                actual_generation: record.generation,
                slice_id: record.slice_id,
            });
        }

        let stored = self
            .records
            .get(record.slice_id as usize)
            .ok_or(OwnershipError::SliceNotFound {
                slice_id: record.slice_id,
            })?;

        if stored != record {
            return Err(OwnershipError::SliceIdentityMismatch {
                slice_id: record.slice_id,
            });
        }

        let start = usize::try_from(record.byte_offset).map_err(|_| {
            OwnershipError::SliceOutOfBounds {
                offset: record.byte_offset,
                length: record.byte_length,
                buffer_len: self.bytes.len(),
            }
        })?;
        let len = usize::try_from(record.byte_length).map_err(|_| {
            OwnershipError::SliceOutOfBounds {
                offset: record.byte_offset,
                length: record.byte_length,
                buffer_len: self.bytes.len(),
            }
        })?;
        let end = start
            .checked_add(len)
            .ok_or(OwnershipError::SliceOutOfBounds {
                offset: record.byte_offset,
                length: record.byte_length,
                buffer_len: self.bytes.len(),
            })?;

        if end > self.bytes.len() {
            return Err(OwnershipError::SliceOutOfBounds {
                offset: record.byte_offset,
                length: record.byte_length,
                buffer_len: self.bytes.len(),
            });
        }

        Ok(&self.bytes[start..end])
    }

    /// Look up an immutable slice and record by `slice_id`.
    pub fn get_slice_by_id(&self, slice_id: u32) -> Result<(&UseRecord<D>, &[u8]), OwnershipError> {
        let record = self
            .records
            .get(slice_id as usize)
            .ok_or(OwnershipError::SliceNotFound { slice_id })?;
        let slice = self.get_slice(record)?;
        Ok((record, slice))
    }

    /// Contiguous byte slice of the entire buffer for GPU upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// All recorded use records in allocation order.
    #[inline]
    pub fn records(&self) -> &[UseRecord<D>] {
        &self.records
    }

    /// Total byte length of the buffer.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Reset the buffer for a subsequent submission cycle or frame, advancing the allocation generation
    /// to invalidate old [`UseRecord`] handles (ABA prevention).
    pub fn reset(&mut self) -> Result<u64, OwnershipError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(OwnershipError::VersionOverflow {
                current: self.generation,
            })?;
        self.records.clear();
        self.bytes.clear();
        Ok(self.generation)
    }
}
