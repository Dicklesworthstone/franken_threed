//! Generation-checked publication state for deferred or cancellable tasks.
#![forbid(unsafe_code)]

/// Publication state tracking a monotonic generation and optional value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedState {
    generation: u32,
    value: Option<u32>,
}

impl PublishedState {
    /// Creates a new publication state with the given initial generation.
    #[must_use]
    pub fn new(generation: u32) -> Self {
        Self {
            generation,
            value: None,
        }
    }

    /// Returns the current active generation.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Returns the published value, if any.
    #[must_use]
    pub fn value(&self) -> Option<u32> {
        self.value
    }

    /// Increments the generation and returns the new one without touching value.
    pub fn replace(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    /// Attempts to publish a value under the given generation.
    ///
    /// If `generation` matches the current active generation, stores the value
    /// and returns `true`. On mismatch, leaves `value` untouched and returns `false`.
    pub fn try_publish(&mut self, generation: u32, value: u32) -> bool {
        if generation != self.generation {
            false
        } else {
            self.value = Some(value);
            true
        }
    }
}
