//! Pure Rust burst counter for tracking polls per pump turn.
//!
//! Tracks exact poll counts within each upstream pump turn identifier without
//! depending on host sampling granularity.

/// Counts task polls grouped by pump turn identifier.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BurstCounter {
    current_turn: Option<u32>,
    polls_this_turn: u32,
    max_polls_in_a_turn: u32,
    total_polls: u32,
}

impl BurstCounter {
    /// Creates a new empty burst counter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current_turn: None,
            polls_this_turn: 0,
            max_polls_in_a_turn: 0,
            total_polls: 0,
        }
    }

    /// Records a poll in the specified pump `turn`.
    ///
    /// Resets `polls_this_turn` to 0 when `turn` differs from `current_turn`,
    /// increments `polls_this_turn` and `total_polls`, updates `max_polls_in_a_turn`,
    /// and returns `polls_this_turn`.
    pub fn record(&mut self, turn: u32) -> u32 {
        if self.current_turn != Some(turn) {
            self.current_turn = Some(turn);
            self.polls_this_turn = 0;
        }
        self.polls_this_turn += 1;
        self.total_polls += 1;
        if self.polls_this_turn > self.max_polls_in_a_turn {
            self.max_polls_in_a_turn = self.polls_this_turn;
        }
        self.polls_this_turn
    }

    /// Returns the maximum number of polls observed in any single pump turn.
    #[must_use]
    pub const fn max_polls_in_a_turn(&self) -> u32 {
        self.max_polls_in_a_turn
    }

    /// Returns the total number of polls recorded across all turns.
    #[must_use]
    pub const fn total_polls(&self) -> u32 {
        self.total_polls
    }
}
