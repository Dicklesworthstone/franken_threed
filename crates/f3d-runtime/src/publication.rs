//! Generation-checked publication state for deferred or cancellable tasks.
#![forbid(unsafe_code)]

/// Publication state tracking a monotonic generation and optional value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedState {
    generation: Option<u32>,
    value: Option<u32>,
}

impl PublishedState {
    /// Creates a new publication state with the given initial generation.
    #[must_use]
    pub fn new(generation: u32) -> Self {
        Self {
            generation: Some(generation),
            value: None,
        }
    }

    /// Returns the current active generation, or `None` if exhausted.
    #[must_use]
    pub fn generation(&self) -> Option<u32> {
        self.generation
    }

    /// Returns the published value, if any.
    #[must_use]
    pub fn value(&self) -> Option<u32> {
        self.value
    }

    /// Increments the generation with checked arithmetic, returning the new one.
    ///
    /// Once the generation reaches `u32::MAX`, subsequent replacements yield `None`
    /// and transition the state to permanently exhausted without modifying any
    /// previously published value.
    pub fn replace(&mut self) -> Option<u32> {
        self.generation = self.generation.and_then(|g| g.checked_add(1));
        self.generation
    }

    /// Attempts to publish a value under the given generation.
    ///
    /// If `generation` matches the current active generation, stores the value
    /// and returns `true`. If exhausted or mismatched, leaves `value` untouched
    /// and returns `false`.
    pub fn try_publish(&mut self, generation: u32, value: u32) -> bool {
        if self.generation == Some(generation) {
            self.value = Some(value);
            true
        } else {
            false
        }
    }
}
