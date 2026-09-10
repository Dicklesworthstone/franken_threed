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
//! 3. Authority transfer with stale epoch or pending unpublished writes is strictly rejected.
//! 4. Versions and epochs are monotonic 64-bit counters decomposed into two explicit 32-bit words
//!    at host boundaries to prevent precision loss from JavaScript `Number` (IEEE 754 float64).
//! 5. Pass planning and bridge snapshots retain immutable per-use versions and buffer slices
//!    so that mutations to a shared resource between passes (e.g. Red in Pass A, Blue in Pass B)
//!    coexist in the same submission schedule without in-place slot overwrites.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use crate::error::HandleError;
use crate::handle::{Domain, Handle, RegionDomain};

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
    /// An unauthorized writer attempted to mutate or publish state.
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
    /// Violation of snapshot immutability: attempted mutation of an existing version snapshot.
    ImmutableSnapshotViolation {
        /// Version that was attempted to be re-recorded or corrupted.
        version: DataVersion,
        /// Description of the attempted violation.
        detail: &'static str,
    },
    /// Requested slice was not found in the per-use buffer.
    SliceNotFound {
        /// The requested slice identifier.
        slice_id: u32,
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
            Self::ImmutableSnapshotViolation { version, detail } => {
                write!(
                    f,
                    "immutable snapshot violation at version {version}: {detail}"
                )
            }
            Self::SliceNotFound { slice_id } => {
                write!(f, "per-use slice {slice_id} not found in buffer")
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
    /// # Errors
    ///
    /// - Returns [`OwnershipError::SameAuthorTransfer`] if `from == to`.
    /// - Returns [`OwnershipError::UnauthorizedWriter`] if `from` is not authoritative.
    /// - Returns [`OwnershipError::StaleEpoch`] if `at_epoch` is stale.
    /// - Returns [`OwnershipError::UnpublishedWritesPending`] if uncommitted writes exist.
    pub fn transfer_authority(
        &mut self,
        from: Author,
        to: Author,
        at_epoch: Epoch,
    ) -> Result<(), OwnershipError> {
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

        match self.mode {
            OwnerMode::Js => self.mode = OwnerMode::Wasm,
            OwnerMode::Wasm => self.mode = OwnerMode::Js,
            OwnerMode::Mirrored { .. } => self.mode = OwnerMode::Mirrored { author: to },
        }

        Ok(())
    }

    /// Transition to a new ownership mode at the specified epoch with zero pending writes.
    ///
    /// # Errors
    ///
    /// - Returns [`OwnershipError::StaleEpoch`] if `at_epoch` is stale.
    /// - Returns [`OwnershipError::UnpublishedWritesPending`] if uncommitted writes exist.
    pub fn transition_mode(
        &mut self,
        new_mode: OwnerMode,
        at_epoch: Epoch,
    ) -> Result<(), OwnershipError> {
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

        self.mode = new_mode;
        Ok(())
    }
}

/// Immutable record of a specific version of a resource bound to a pass or draw use.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UseRecord<D: Domain> {
    resource: Handle<D>,
    version: DataVersion,
    epoch: Epoch,
    slice_id: u32,
    byte_offset: u64,
    byte_length: u64,
}

impl<D: Domain> UseRecord<D> {
    /// Construct a new use record.
    pub const fn new(
        resource: Handle<D>,
        version: DataVersion,
        epoch: Epoch,
        slice_id: u32,
        byte_offset: u64,
        byte_length: u64,
    ) -> Self {
        Self {
            resource,
            version,
            epoch,
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

    /// Unique index of this slice within the per-use buffer.
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
#[inline]
const fn align_up(offset: u64, alignment: u64) -> Result<u64, OwnershipError> {
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return Err(OwnershipError::InvalidAlignment { alignment });
    }
    let mask = alignment - 1;
    let aligned = (offset + mask) & !mask;
    if aligned < offset {
        return Err(OwnershipError::VersionOverflow { current: offset });
    }
    Ok(aligned)
}

/// Storage pool for immutable per-use snapshots of typed data.
///
/// Ensures that multiple uses of a mutating resource across passes
/// allocate distinct slices and preserve historical versions without in-place overwrite.
#[derive(Clone, Debug)]
pub struct PerUseSnapshotStore<D: Domain, T: Clone> {
    entries: Vec<SnapshotEntry<D, T>>,
    alignment: u64,
    current_offset: u64,
}

impl<D: Domain, T: Clone> PerUseSnapshotStore<D, T> {
    /// Create a new snapshot store with byte alignment for subsequent slices (e.g. 256 bytes for WebGPU UBOs).
    pub fn new(alignment: u64) -> Result<Self, OwnershipError> {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(OwnershipError::InvalidAlignment { alignment });
        }
        Ok(Self {
            entries: Vec::new(),
            alignment,
            current_offset: 0,
        })
    }

    /// Record a pass use of a resource with its snapshot data.
    ///
    /// If the resource at the given `version` was already recorded with identical data,
    /// returns the existing [`UseRecord`]. If conflicting data is provided for an already recorded
    /// version, returns [`OwnershipError::ImmutableSnapshotViolation`].
    pub fn record_use(
        &mut self,
        handle: Handle<D>,
        version: DataVersion,
        epoch: Epoch,
        data: T,
        byte_length: u64,
    ) -> Result<UseRecord<D>, OwnershipError>
    where
        T: PartialEq,
    {
        for entry in &self.entries {
            if entry.record.resource == handle && entry.record.version == version {
                if entry.data == data {
                    return Ok(entry.record);
                }
                return Err(OwnershipError::ImmutableSnapshotViolation {
                    version,
                    detail: "conflicting data provided for existing immutable version snapshot",
                });
            }
        }

        let aligned_offset = align_up(self.current_offset, self.alignment)?;
        let next_offset = aligned_offset
            .checked_add(byte_length)
            .ok_or(OwnershipError::VersionOverflow {
                current: aligned_offset,
            })?;

        let slice_id = self.entries.len() as u32;
        let record = UseRecord::new(
            handle,
            version,
            epoch,
            slice_id,
            aligned_offset,
            byte_length,
        );

        self.entries.push(SnapshotEntry { record, data });
        self.current_offset = next_offset;
        Ok(record)
    }

    /// Look up a slice by its unique `slice_id`.
    pub fn get_slice(&self, slice_id: u32) -> Result<&SnapshotEntry<D, T>, OwnershipError> {
        self.entries
            .get(slice_id as usize)
            .ok_or(OwnershipError::SliceNotFound { slice_id })
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

    /// Reset the store for a subsequent submission cycle or frame.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.current_offset = 0;
    }
}

/// Contiguous byte buffer for immutable per-use uniform and storage slices.
///
/// Ensures byte slices written for distinct versions of a resource are immutable,
/// contiguous, aligned, and never overwritten by subsequent mutations.
#[derive(Clone, Debug)]
pub struct PerUseByteBuffer<D: Domain> {
    records: Vec<UseRecord<D>>,
    bytes: Vec<u8>,
    alignment: u64,
}

impl<D: Domain> PerUseByteBuffer<D> {
    /// Create a new byte buffer with specified slice alignment (e.g. 256 for WebGPU UBOs).
    pub fn new(alignment: u64) -> Result<Self, OwnershipError> {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(OwnershipError::InvalidAlignment { alignment });
        }
        Ok(Self {
            records: Vec::new(),
            bytes: Vec::new(),
            alignment,
        })
    }

    /// Append an immutable byte slice for a resource version and epoch.
    ///
    /// # Errors
    ///
    /// - Returns [`OwnershipError::ImmutableSnapshotViolation`] if attempting to re-record
    ///   an existing version for the same handle with differing content.
    /// - Returns [`OwnershipError::VersionOverflow`] if buffer offsets exceed `u64::MAX`.
    pub fn append_slice(
        &mut self,
        handle: Handle<D>,
        version: DataVersion,
        epoch: Epoch,
        slice_bytes: &[u8],
    ) -> Result<UseRecord<D>, OwnershipError> {
        for record in &self.records {
            if record.resource == handle && record.version == version {
                let start = record.byte_offset as usize;
                let end = start + record.byte_length as usize;
                if let Some(existing) = self.bytes.get(start..end) {
                    if existing == slice_bytes {
                        return Ok(*record);
                    }
                }
                return Err(OwnershipError::ImmutableSnapshotViolation {
                    version,
                    detail: "conflicting byte content provided for existing immutable version",
                });
            }
        }

        let current_len = self.bytes.len() as u64;
        let aligned_offset = align_up(current_len, self.alignment)?;
        let padding = (aligned_offset - current_len) as usize;

        self.bytes.resize(self.bytes.len() + padding, 0);
        let byte_offset = self.bytes.len() as u64;
        self.bytes.extend_from_slice(slice_bytes);

        let slice_id = self.records.len() as u32;
        let record = UseRecord::new(
            handle,
            version,
            epoch,
            slice_id,
            byte_offset,
            slice_bytes.len() as u64,
        );
        self.records.push(record);
        Ok(record)
    }

    /// Look up an immutable slice by its [`UseRecord`].
    pub fn get_slice(&self, record: &UseRecord<D>) -> Result<&[u8], OwnershipError> {
        let start = record.byte_offset as usize;
        let end = start + record.byte_length as usize;
        self.bytes
            .get(start..end)
            .ok_or(OwnershipError::SliceNotFound {
                slice_id: record.slice_id,
            })
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

    /// Reset the buffer for a subsequent submission cycle or frame.
    pub fn reset(&mut self) {
        self.records.clear();
        self.bytes.clear();
    }
}
