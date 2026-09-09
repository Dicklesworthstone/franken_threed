//! Generation-checked typed handles and slot arenas.
//!
//! Invariants:
//! 1. Handles carry an index and a generation to prevent ABA handle collisions.
//! 2. GPU handles additionally carry a DeviceGeneration to detect device loss/recreation.
//! 3. Counter wrap retires the slot identity permanently rather than allowing collision.
//! 4. ABI transport passes handles as two explicit 32-bit words, never as a 64-bit float.
//! 5. Domain identity is preserved and validated across serialization boundaries.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::num::NonZeroU32;

use crate::error::HandleError;

/// Trait implemented by all identity domain marker types.
pub trait Domain: Copy + Clone + PartialEq + Eq + 'static {
    /// Human-readable name of the domain for diagnostics.
    const DOMAIN_NAME: &'static str;
}

/// Scene object domain (Mesh, Group, Camera, Light).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ObjectDomain;
impl Domain for ObjectDomain {
    const DOMAIN_NAME: &'static str = "object";
}

/// Geometry / BufferGeometry domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct GeometryDomain;
impl Domain for GeometryDomain {
    const DOMAIN_NAME: &'static str = "geometry";
}

/// Buffer attribute domain (positions, normals, uvs, colors).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct AttributeDomain;
impl Domain for AttributeDomain {
    const DOMAIN_NAME: &'static str = "attribute";
}

/// Material domain (MeshStandardMaterial, RawShaderMaterial, etc.).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MaterialDomain;
impl Domain for MaterialDomain {
    const DOMAIN_NAME: &'static str = "material";
}

/// Texture domain (2D, 3D, Cube, Video, Canvas).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TextureDomain;
impl Domain for TextureDomain {
    const DOMAIN_NAME: &'static str = "texture";
}

/// Render target / framebuffer domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RenderTargetDomain;
impl Domain for RenderTargetDomain {
    const DOMAIN_NAME: &'static str = "render_target";
}

/// Compiled pipeline / shader variant domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PipelineDomain;
impl Domain for PipelineDomain {
    const DOMAIN_NAME: &'static str = "pipeline";
}

/// Reusable WebGPU render bundle domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BundleDomain;
impl Domain for BundleDomain {
    const DOMAIN_NAME: &'static str = "bundle";
}

/// Memory or state region domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RegionDomain;
impl Domain for RegionDomain {
    const DOMAIN_NAME: &'static str = "region";
}

/// Async task publication domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TaskDomain;
impl Domain for TaskDomain {
    const DOMAIN_NAME: &'static str = "task";
}

/// A typed, generation-checked handle into an arena.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Handle<D: Domain> {
    index: u32,
    generation: NonZeroU32,
    _marker: PhantomData<D>,
}

impl<D: Domain> Copy for Handle<D> {}

impl<D: Domain> Clone for Handle<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: Domain> Handle<D> {
    /// Construct a new handle from index and generation.
    pub const fn new(index: u32, generation: NonZeroU32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// Try constructing a handle from raw u32 values.
    pub fn from_raw(index: u32, raw_generation: u32) -> Result<Self, HandleError> {
        let generation_val = NonZeroU32::new(raw_generation)
            .ok_or(HandleError::InvalidGeneration { raw_generation })?;
        Ok(Self::new(index, generation_val))
    }

    /// The 0-indexed slot index in the arena.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generational counter of the handle.
    pub const fn generation(self) -> NonZeroU32 {
        self.generation
    }

    /// Pack into a 64-bit integer: upper 32 bits = generation, lower 32 bits = index.
    pub const fn pack_u64(self) -> u64 {
        ((self.generation.get() as u64) << 32) | (self.index as u64)
    }

    /// Unpack from a 64-bit integer packed with `pack_u64`.
    pub fn unpack_u64(packed: u64) -> Result<Self, HandleError> {
        let index = packed as u32;
        let generation_raw = (packed >> 32) as u32;
        Self::from_raw(index, generation_raw)
    }

    /// Split into two 32-bit words for safe boundary transport (e.g. Wasm ABI).
    pub const fn to_words(self) -> (u32, u32) {
        (self.index, self.generation.get())
    }

    /// Reconstruct from two 32-bit words.
    pub fn from_words(index: u32, generation_word: u32) -> Result<Self, HandleError> {
        Self::from_raw(index, generation_word)
    }
}

impl<D: Domain> fmt::Debug for Handle<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Handle<{}>({}#{})",
            D::DOMAIN_NAME,
            self.index,
            self.generation
        )
    }
}

impl<D: Domain> fmt::Display for Handle<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}#{}", D::DOMAIN_NAME, self.index, self.generation)
    }
}

#[cfg(feature = "serde")]
impl<D: Domain> serde::Serialize for Handle<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Handle", 3)?;
        state.serialize_field("domain", D::DOMAIN_NAME)?;
        state.serialize_field("index", &self.index)?;
        state.serialize_field("generation", &self.generation.get())?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, D: Domain> serde::Deserialize<'de> for Handle<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct HandleData {
            domain: alloc::string::String,
            index: u32,
            generation: u32,
        }

        let data = HandleData::deserialize(deserializer)?;
        if data.domain != D::DOMAIN_NAME {
            return Err(serde::de::Error::custom(alloc::format!(
                "domain mismatch: expected '{}', got '{}'",
                D::DOMAIN_NAME,
                data.domain
            )));
        }
        Self::from_raw(data.index, data.generation).map_err(serde::de::Error::custom)
    }
}

/// Device generation to track GPU instance recreations and invalidate stale GPU handles.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceGeneration(pub NonZeroU32);

impl DeviceGeneration {
    /// Initial device generation (1).
    pub const INITIAL: Self = Self(match NonZeroU32::new(1) {
        Some(v) => v,
        None => unreachable!(),
    });

    /// Create from a raw u32 value.
    pub fn new(val: u32) -> Result<Self, HandleError> {
        let generation_nz = NonZeroU32::new(val)
            .ok_or(HandleError::InvalidGeneration { raw_generation: val })?;
        Ok(Self(generation_nz))
    }

    /// Advance to next generation (after device loss/recreation).
    pub fn next(self) -> Option<Self> {
        NonZeroU32::new(self.0.get().checked_add(1)?).map(Self)
    }

    /// Raw numeric value.
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// A handle to a GPU-resident resource, paired with its owning device generation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuHandle<D: Domain> {
    /// Base arena handle.
    pub handle: Handle<D>,
    /// Device generation that created this resource.
    pub device_generation: DeviceGeneration,
}

impl<D: Domain> GpuHandle<D> {
    /// Create a new GPU handle.
    pub const fn new(handle: Handle<D>, device_generation: DeviceGeneration) -> Self {
        Self {
            handle,
            device_generation,
        }
    }

    /// Verify that this GPU handle matches the current active device generation.
    pub fn validate_device(self, current: DeviceGeneration) -> Result<Handle<D>, HandleError> {
        if self.device_generation == current {
            Ok(self.handle)
        } else {
            Err(HandleError::DeviceGenerationMismatch {
                expected_device: self.device_generation.get(),
                current_device: current.get(),
            })
        }
    }
}

/// Explicit lifecycle states for resources (Section 6.5).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LifecycleState {
    /// Exists only on CPU; not yet allocated on GPU.
    CpuExistent,
    /// GPU memory has been reserved/allocated.
    GpuAllocated,
    /// Initial data has been written/uploaded to GPU.
    Initialized,
    /// Enqueued in a command encoder or submitted schedule.
    Submitted,
    /// Marked for deletion / freed; eligible for reuse with incremented generation.
    Retired,
    /// Generation counter wrapped or reached MAX; permanently exhausted and never reused.
    Exhausted,
}

/// Internal slot state in an `Arena`.
struct Slot<T> {
    generation: NonZeroU32,
    value: Option<T>,
    lifecycle: LifecycleState,
}

/// Generation-checked slot storage for items belonging to domain `D`.
pub struct Arena<D: Domain, T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
    _marker: PhantomData<D>,
}

impl<D: Domain, T> Default for Arena<D, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Domain, T> Arena<D, T> {
    /// Create an empty arena.
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            _marker: PhantomData,
        }
    }

    /// Number of total allocated slots (active + vacant).
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Insert an item into the arena, returning its typed generation-checked handle.
    pub fn insert(&mut self, value: T) -> Result<Handle<D>, HandleError> {
        if let Some(free_idx) = self.free_head {
            let slot = &mut self.slots[free_idx as usize];
            let current_generation = slot.generation;
            slot.value = Some(value);
            slot.lifecycle = LifecycleState::CpuExistent;
            // Advance free head to next available Retired slot
            self.free_head = self.find_first_retired();
            Ok(Handle::new(free_idx, current_generation))
        } else {
            let index = u32::try_from(self.slots.len())
                .map_err(|_| HandleError::IndexOutOfBounds {
                    index: u32::MAX,
                    capacity: u32::MAX,
                })?;
            let initial_generation = NonZeroU32::new(1).expect("1 is non-zero");
            self.slots.push(Slot {
                generation: initial_generation,
                value: Some(value),
                lifecycle: LifecycleState::CpuExistent,
            });
            Ok(Handle::new(index, initial_generation))
        }
    }

    /// Look up a reference to the value for a valid handle.
    pub fn get(&self, handle: Handle<D>) -> Result<&T, HandleError> {
        let slot = self.validate_slot(handle)?;
        slot.value.as_ref().ok_or(HandleError::SlotVacant { index: handle.index })
    }

    /// Look up a mutable reference to the value for a valid handle.
    pub fn get_mut(&mut self, handle: Handle<D>) -> Result<&mut T, HandleError> {
        let slot = self.validate_slot_mut(handle)?;
        slot.value.as_mut().ok_or(HandleError::SlotVacant { index: handle.index })
    }

    /// Query current lifecycle state of a slot.
    pub fn get_lifecycle(&self, handle: Handle<D>) -> Result<LifecycleState, HandleError> {
        let slot = self.validate_slot(handle)?;
        Ok(slot.lifecycle)
    }

    /// Update the lifecycle state of an active slot.
    ///
    /// Invariants:
    /// - `Retired` and `Exhausted` are terminal / free-list states managed exclusively by `remove()`.
    /// - Calling `set_lifecycle` on a vacant slot returns `HandleError::SlotVacant`.
    /// - Calling `set_lifecycle` to set `Retired` or `Exhausted` returns `HandleError::InvalidLifecycleTransition`.
    /// - Permanent exhaustion cannot be reversed through public setters.
    pub fn set_lifecycle(
        &mut self,
        handle: Handle<D>,
        state: LifecycleState,
    ) -> Result<(), HandleError> {
        let slot = self.validate_slot_mut(handle)?;
        if slot.value.is_none() || slot.lifecycle == LifecycleState::Exhausted {
            return Err(HandleError::SlotVacant { index: handle.index });
        }
        match state {
            LifecycleState::Retired | LifecycleState::Exhausted => {
                Err(HandleError::InvalidLifecycleTransition)
            }
            _ => {
                slot.lifecycle = state;
                Ok(())
            }
        }
    }

    /// Remove an item, freeing its slot and incrementing its generation.
    ///
    /// Counter-wrap invariant: if the generation counter would overflow `u32::MAX`,
    /// the slot transitions to `LifecycleState::Exhausted` and is never reused,
    /// preventing ABA collisions.
    pub fn remove(&mut self, handle: Handle<D>) -> Result<T, HandleError> {
        let slot = self.validate_slot_mut(handle)?;
        let val = slot.value.take().ok_or(HandleError::SlotVacant { index: handle.index })?;

        // Advance generation or retire permanently
        if let Some(next_generation) = slot.generation.get().checked_add(1).and_then(NonZeroU32::new) {
            slot.generation = next_generation;
            slot.lifecycle = LifecycleState::Retired;
            self.free_head = self.find_first_retired();
        } else {
            // Reached u32::MAX boundary: slot is permanently exhausted and cannot be reused
            slot.lifecycle = LifecycleState::Exhausted;
            self.free_head = self.find_first_retired();
        }

        Ok(val)
    }

    fn validate_slot(&self, handle: Handle<D>) -> Result<&Slot<T>, HandleError> {
        let idx = handle.index as usize;
        if idx >= self.slots.len() {
            return Err(HandleError::IndexOutOfBounds {
                index: handle.index,
                capacity: self.slots.len() as u32,
            });
        }
        let slot = &self.slots[idx];
        if slot.generation != handle.generation {
            return Err(HandleError::GenerationMismatch {
                index: handle.index,
                expected_generation: handle.generation.get(),
                actual_generation: slot.generation.get(),
            });
        }
        Ok(slot)
    }

    fn validate_slot_mut(&mut self, handle: Handle<D>) -> Result<&mut Slot<T>, HandleError> {
        let idx = handle.index as usize;
        if idx >= self.slots.len() {
            return Err(HandleError::IndexOutOfBounds {
                index: handle.index,
                capacity: self.slots.len() as u32,
            });
        }
        let slot = &mut self.slots[idx];
        if slot.generation != handle.generation {
            return Err(HandleError::GenerationMismatch {
                index: handle.index,
                expected_generation: handle.generation.get(),
                actual_generation: slot.generation.get(),
            });
        }
        Ok(slot)
    }

    fn find_first_retired(&self) -> Option<u32> {
        for (i, slot) in self.slots.iter().enumerate() {
            // Only Retired slots can be reused; Exhausted slots are permanently dead.
            if slot.value.is_none() && slot.lifecycle == LifecycleState::Retired {
                return Some(i as u32);
            }
        }
        None
    }

    /// Internal helper for testing boundary exhaustion behavior.
    #[cfg(test)]
    pub(crate) fn force_set_slot_generation_for_test(&mut self, index: u32, generation: NonZeroU32) {
        if let Some(slot) = self.slots.get_mut(index as usize) {
            slot.generation = generation;
        }
    }
}
