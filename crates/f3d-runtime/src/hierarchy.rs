//! Rust-owned, incremental world-matrix evaluation for a fixed-membership forest.
//!
//! Inputs are explicit local matrices, not an implicit application tick. Node
//! indices remain stable across reparenting. JavaScript/animation callers own
//! when TRS is composed and when a snapshot is submitted. Disabled automatic
//! world updates form authoritative world-matrix boundaries, not hidden local
//! transforms. Matrix arithmetic uses the existing f64 Three.js-order kernel.
//!
//! Dirty subtree roots are kept in preorder: a leaf edit visits one node, an
//! ancestor edit visits its descendants, and a clean solve visits none. No
//! recursive calls or per-node allocation occur during solving. Publication
//! requires the current, solved revision; edits invalidate old read receipts.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

use f3d_math::Matrix4;

/// Invalid input or an attempt to publish an obsolete world-matrix snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HierarchyError {
    /// An array does not have the required number of elements.
    Length { field: &'static str, expected: usize, actual: usize },
    /// Node identities must fit the signed parent-index transport.
    TooManyNodes,
    /// Parents must be -1 (root) or a different, existing node index.
    Parent { node: usize, parent: i32 },
    /// The proposed parent graph contains a cycle.
    Cycle,
    /// An update refers to a node outside this fixed-membership forest.
    Node { index: u32, count: usize },
    /// Boolean transport fields accept only zero and one.
    Flag { node: usize, value: u8 },
    /// External world matrices require automatic world updates to be disabled.
    AutomaticWorld { node: u32 },
    /// A revision cannot be incremented without wrapping.
    RevisionExhausted,
    /// A receipt belongs to a prior mutation revision.
    Stale { requested: u64, current: u64 },
    /// The current input revision has not been solved yet.
    Unsolved { current: u64 },
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { field, expected, actual } => write!(f, "{field}: expected {expected} elements, got {actual}"),
            Self::TooManyNodes => write!(f, "hierarchy exceeds signed 32-bit node indexing"),
            Self::Parent { node, parent } => write!(f, "invalid parent {parent} for node {node}"),
            Self::Cycle => write!(f, "hierarchy parent graph contains a cycle"),
            Self::Node { index, count } => write!(f, "node {index} is outside hierarchy of {count} nodes"),
            Self::Flag { node, value } => write!(f, "world-auto flag for node {node} must be 0 or 1, got {value}"),
            Self::AutomaticWorld { node } => write!(f, "disable automatic world updates before writing node {node}'s world matrix"),
            Self::RevisionExhausted => write!(f, "hierarchy mutation revision exhausted"),
            Self::Stale { requested, current } => write!(f, "stale hierarchy revision {requested}; current revision is {current}"),
            Self::Unsolved { current } => write!(f, "hierarchy revision {current} has not been solved"),
        }
    }
}

impl std::error::Error for HierarchyError {}

/// Actual work performed by one synchronous solve, not a timing estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HierarchySolveStats {
    /// Input revision represented by the result.
    pub revision: u64,
    /// Number of nodes in the union of dirty subtrees.
    pub visited: usize,
    /// Matrix products performed for automatically updated, non-root nodes.
    pub multiplied: usize,
    /// Automatically updated root matrices copied from their local matrices.
    pub copied_roots: usize,
    /// Visited externally owned world matrices preserved without recomputation.
    pub preserved_worlds: usize,
    /// Distinct world matrices changed since the preceding solve.
    pub changed: usize,
}

#[derive(Debug)]
struct Topology {
    parents: Vec<i32>,
    order: Vec<usize>,
    position: Vec<usize>,
    subtree_end: Vec<usize>,
}

impl Topology {
    fn new(parents: &[i32]) -> Result<Self, HierarchyError> {
        let count = parents.len();
        if count > i32::MAX as usize { return Err(HierarchyError::TooManyNodes); }
        let mut first_child = vec![None; count];
        let mut next_sibling = vec![None; count];
        for (node, &parent) in parents.iter().enumerate() {
            if parent < -1 || parent >= 0 && (parent as usize >= count || parent as usize == node) {
                return Err(HierarchyError::Parent { node, parent });
            }
            if parent >= 0 {
                let parent = parent as usize;
                next_sibling[node] = first_child[parent];
                first_child[parent] = Some(node);
            }
        }
        let mut order = Vec::with_capacity(count);
        let mut position = vec![0; count];
        let mut subtree_end = vec![0; count];
        let mut stack = Vec::with_capacity(count);
        for node in (0..count).rev() {
            if parents[node] == -1 { stack.push((node, false)); }
        }
        while let Some((node, exiting)) = stack.pop() {
            if exiting {
                subtree_end[node] = order.len();
            } else {
                position[node] = order.len();
                order.push(node);
                stack.push((node, true));
                // Linked siblings descend by node index, so the LIFO traversal
                // visits siblings in ascending index order deterministically.
                let mut child = first_child[node];
                while let Some(index) = child {
                    stack.push((index, false));
                    child = next_sibling[index];
                }
            }
        }
        // With exactly one parent per node, cycles are precisely components
        // unreachable from a root. Traversal never enters a cyclic component.
        if order.len() != count { return Err(HierarchyError::Cycle); }
        Ok(Self { parents: parents.to_vec(), order, position, subtree_end })
    }
}

/// Packed transform state. All input validation precedes mutation, including
/// validation of every entry in batched updates and proposed reparentings.
#[derive(Debug)]
pub struct TransformHierarchy {
    topology: Topology,
    local: Vec<Matrix4>,
    world: Vec<Matrix4>,
    automatic_world: Vec<bool>,
    dirty: BTreeSet<usize>,
    world_changed: Vec<bool>,
    changed: Vec<u32>,
    revision: u64,
    solved_revision: Option<u64>,
    last_stats: HierarchySolveStats,
}

fn length(field: &'static str, expected: usize, actual: usize) -> Result<(), HierarchyError> {
    if expected == actual { Ok(()) } else { Err(HierarchyError::Length { field, expected, actual }) }
}

fn matrix_count(count: usize) -> Result<usize, HierarchyError> {
    count.checked_mul(16).ok_or(HierarchyError::TooManyNodes)
}

fn matrices(raw: &[f64]) -> Vec<Matrix4> {
    raw.chunks_exact(16).map(|chunk| {
        let mut elements = [0.0; 16];
        elements.copy_from_slice(chunk);
        Matrix4::from_elements(elements)
    }).collect()
}

fn same_bits(matrix: &Matrix4, raw: &[f64]) -> bool {
    matrix.elements.iter().zip(raw).all(|(a, b)| a.to_bits() == b.to_bits())
}

impl TransformHierarchy {
    /// Construct a forest from column-major f64 local matrices and signed
    /// parents (-1 denotes a root). Omitted worlds start at identity; omitted
    /// world-auto flags default to enabled. Non-finite values and signed zeros
    /// are preserved, as with the existing scalar Matrix4 implementation.
    pub fn new(
        parents: &[i32], local_matrices: &[f64],
        world_matrices: Option<&[f64]>, world_auto: Option<&[u8]>,
    ) -> Result<Self, HierarchyError> {
        let topology = Topology::new(parents)?;
        let count = parents.len();
        let elements = matrix_count(count)?;
        length("local matrices", elements, local_matrices.len())?;
        if let Some(world) = world_matrices { length("world matrices", elements, world.len())?; }
        if let Some(flags) = world_auto {
            length("world-auto flags", count, flags.len())?;
            for (node, &value) in flags.iter().enumerate() {
                if value > 1 { return Err(HierarchyError::Flag { node, value }); }
            }
        }
        let mut result = Self {
            topology,
            local: matrices(local_matrices),
            world: world_matrices.map_or_else(|| vec![Matrix4::identity(); count], matrices),
            automatic_world: world_auto.map_or_else(|| vec![true; count], |flags| flags.iter().map(|&v| v == 1).collect()),
            dirty: BTreeSet::new(), world_changed: vec![false; count],
            changed: Vec::with_capacity(count),
            revision: 0, solved_revision: None, last_stats: HierarchySolveStats::default(),
        };
        result.dirty_all_roots();
        Ok(result)
    }

    /// Number of stable node slots in the forest.
    #[must_use]
    pub fn len(&self) -> usize { self.local.len() }
    /// Whether the forest is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.local.is_empty() }
    /// Monotonic mutation revision. No-op writes retain the revision.
    #[must_use]
    pub const fn revision(&self) -> u64 { self.revision }
    /// Statistics from the last solve, including a zero-work clean solve.
    #[must_use]
    pub const fn last_stats(&self) -> HierarchySolveStats { self.last_stats }

    fn next_revision(&self) -> Result<u64, HierarchyError> {
        self.revision.checked_add(1).ok_or(HierarchyError::RevisionExhausted)
    }

    fn validate_indices(&self, indices: &[u32]) -> Result<(), HierarchyError> {
        for &index in indices {
            if index as usize >= self.len() { return Err(HierarchyError::Node { index, count: self.len() }); }
        }
        Ok(())
    }

    fn dirty_node(&mut self, node: usize) { self.dirty.insert(self.topology.position[node]); }
    fn dirty_all_roots(&mut self) {
        self.dirty.clear();
        for (node, &parent) in self.topology.parents.iter().enumerate() {
            if parent == -1 { self.dirty.insert(self.topology.position[node]); }
        }
    }

    /// Atomically update selected local matrices. Duplicate node indices are
    /// allowed and applied in input order (last write wins).
    pub fn set_local_matrices(&mut self, indices: &[u32], values: &[f64]) -> Result<u64, HierarchyError> {
        self.set_matrices(indices, values, false)
    }

    /// Atomically update externally owned world matrices. Every selected node
    /// must have world-auto disabled; its descendants are then invalidated.
    pub fn set_world_matrices(&mut self, indices: &[u32], values: &[f64]) -> Result<u64, HierarchyError> {
        self.set_matrices(indices, values, true)
    }

    fn set_matrices(&mut self, indices: &[u32], values: &[f64], external: bool) -> Result<u64, HierarchyError> {
        self.validate_indices(indices)?;
        length("matrix update", matrix_count(indices.len())?, values.len())?;
        if external {
            for &node in indices {
                if self.automatic_world[node as usize] { return Err(HierarchyError::AutomaticWorld { node }); }
            }
        }
        let bank = if external { &self.world } else { &self.local };
        if !indices.iter().zip(values.chunks_exact(16)).any(|(&node, raw)| !same_bits(&bank[node as usize], raw)) {
            return Ok(self.revision);
        }
        let revision = self.next_revision()?;
        for (&node, raw) in indices.iter().zip(values.chunks_exact(16)) {
            let matrix = if external { &mut self.world[node as usize] } else { &mut self.local[node as usize] };
            if !same_bits(matrix, raw) {
                matrix.elements.copy_from_slice(raw);
                if external { self.world_changed[node as usize] = true; }
                self.dirty_node(node as usize);
            }
        }
        self.revision = revision;
        Ok(revision)
    }

    /// Change automatic-world ownership, preserving the current world matrix
    /// when disabling it. Re-enabling schedules reconstruction from the parent.
    pub fn set_world_auto(&mut self, indices: &[u32], flags: &[u8]) -> Result<u64, HierarchyError> {
        self.validate_indices(indices)?;
        length("world-auto update", indices.len(), flags.len())?;
        for (&node, &value) in indices.iter().zip(flags) {
            if value > 1 { return Err(HierarchyError::Flag { node: node as usize, value }); }
        }
        if !indices.iter().zip(flags).any(|(&node, &v)| self.automatic_world[node as usize] != (v == 1)) {
            return Ok(self.revision);
        }
        let revision = self.next_revision()?;
        for (&node, &flag) in indices.iter().zip(flags) {
            if self.automatic_world[node as usize] != (flag == 1) {
                self.automatic_world[node as usize] = flag == 1;
                self.dirty_node(node as usize);
            }
        }
        self.revision = revision;
        Ok(revision)
    }

    /// Atomically replace the parent graph without changing node identities or
    /// local matrices. Cycles, self-parenting and unknown parents reject before
    /// any cached state is changed. This operation does not implement attach's
    /// keep-world-transform conversion; callers own that local-matrix decision.
    pub fn reparent(&mut self, parents: &[i32]) -> Result<u64, HierarchyError> {
        length("parents", self.len(), parents.len())?;
        if self.topology.parents == parents { return Ok(self.revision); }
        let topology = Topology::new(parents)?;
        let revision = self.next_revision()?;
        self.topology = topology;
        self.dirty_all_roots();
        self.revision = revision;
        Ok(revision)
    }

    /// Solve dirty subtrees synchronously, coalescing ancestor/descendant edits.
    /// Arbitrary (including projective) matrices use the scalar Matrix4 kernel;
    /// no premature affine assumption or f32 conversion is made here.
    pub fn solve(&mut self) -> HierarchySolveStats {
        let mut stats = HierarchySolveStats { revision: self.revision, ..HierarchySolveStats::default() };
        self.changed.clear();
        let mut covered_end = 0;
        for start in std::mem::take(&mut self.dirty) {
            if start < covered_end { continue; }
            covered_end = self.topology.subtree_end[self.topology.order[start]];
            for position in start..covered_end {
                let node = self.topology.order[position];
                stats.visited += 1;
                let external = std::mem::replace(&mut self.world_changed[node], false);
                if !self.automatic_world[node] {
                    if external { self.changed.push(node as u32); }
                    stats.preserved_worlds += 1;
                    continue;
                }
                let parent = self.topology.parents[node];
                let next = if parent == -1 {
                    stats.copied_roots += 1;
                    self.local[node]
                } else {
                    stats.multiplied += 1;
                    let mut matrix = Matrix4::zero();
                    matrix.multiply_matrices(&self.world[parent as usize], &self.local[node]);
                    matrix
                };
                if external || !same_bits(&self.world[node], &next.elements) {
                    self.changed.push(node as u32);
                }
                self.world[node] = next;
            }
        }
        // Every visited node appears at most once, even after an external
        // write followed by enabling world-auto. Reuse the capacity allocated
        // at construction rather than allocating a tree entry for each change.
        self.changed.sort_unstable();
        stats.changed = self.changed.len();
        self.solved_revision = Some(self.revision);
        self.last_stats = stats;
        stats
    }

    fn check_receipt(&self, revision: u64) -> Result<(), HierarchyError> {
        if revision != self.revision { return Err(HierarchyError::Stale { requested: revision, current: self.revision }); }
        if self.solved_revision != Some(revision) { return Err(HierarchyError::Unsolved { current: revision }); }
        Ok(())
    }

    /// Borrow the solved world bank in stable node-index order.
    pub fn world_matrices(&self, revision: u64) -> Result<&[Matrix4], HierarchyError> {
        self.check_receipt(revision)?;
        Ok(&self.world)
    }
    /// Stable node indices whose world matrices changed during the last solve.
    pub fn changed_indices(&self, revision: u64) -> Result<&[u32], HierarchyError> {
        self.check_receipt(revision)?;
        Ok(&self.changed)
    }
    /// Owned f64 transport copy, never a borrowed view into Wasm linear memory.
    pub fn copy_world_matrices(&self, revision: u64) -> Result<Vec<f64>, HierarchyError> {
        Ok(self.world_matrices(revision)?.iter().flat_map(|matrix| matrix.elements).collect())
    }
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
mod browser {
    use super::{HierarchyError, TransformHierarchy};
    use wasm_bindgen::prelude::*;

    fn js_error(error: HierarchyError) -> JsValue { JsValue::from_str(&error.to_string()) }

    /// Stateful Wasm boundary for packed transform snapshots; no JS math kernel.
    #[wasm_bindgen(js_name = F3dTransformHierarchy)]
    pub struct WasmTransformHierarchy { inner: TransformHierarchy }

    #[wasm_bindgen(js_class = F3dTransformHierarchy)]
    impl WasmTransformHierarchy {
        /// Empty optional banks select identity worlds and enabled world-auto.
        #[wasm_bindgen(constructor)]
        pub fn new(parents: &[i32], locals: &[f64], worlds: &[f64], world_auto: &[u8]) -> Result<WasmTransformHierarchy, JsValue> {
            let world = (!worlds.is_empty()).then_some(worlds);
            let flags = (!world_auto.is_empty()).then_some(world_auto);
            Ok(Self { inner: TransformHierarchy::new(parents, locals, world, flags).map_err(js_error)? })
        }
        /// Mutation revision transported losslessly as a JavaScript BigInt.
        #[wasm_bindgen(getter)]
        pub fn revision(&self) -> u64 { self.inner.revision() }
        /// Bulk local-matrix edits with all-or-nothing input validation.
        #[wasm_bindgen(js_name = setLocalMatrices)]
        pub fn set_local_matrices(&mut self, nodes: &[u32], matrices: &[f64]) -> Result<u64, JsValue> {
            self.inner.set_local_matrices(nodes, matrices).map_err(js_error)
        }
        /// External world-matrix edits for nodes with world-auto disabled.
        #[wasm_bindgen(js_name = setWorldMatrices)]
        pub fn set_world_matrices(&mut self, nodes: &[u32], matrices: &[f64]) -> Result<u64, JsValue> {
            self.inner.set_world_matrices(nodes, matrices).map_err(js_error)
        }
        /// Explicit world-matrix ownership flags.
        #[wasm_bindgen(js_name = setWorldAuto)]
        pub fn set_world_auto(&mut self, nodes: &[u32], flags: &[u8]) -> Result<u64, JsValue> {
            self.inner.set_world_auto(nodes, flags).map_err(js_error)
        }
        /// Validated fixed-membership reparenting.
        pub fn reparent(&mut self, parents: &[i32]) -> Result<u64, JsValue> {
            self.inner.reparent(parents).map_err(js_error)
        }
        /// Synchronously solve, returning the publication revision.
        pub fn solve(&mut self) -> u64 { self.inner.solve().revision }
        /// Owned column-major f64 output in stable node-index order.
        #[wasm_bindgen(js_name = worldMatrices)]
        pub fn world_matrices(&self, revision: u64) -> Result<Vec<f64>, JsValue> {
            self.inner.copy_world_matrices(revision).map_err(js_error)
        }
        /// Changed stable node indices from the last solve.
        #[wasm_bindgen(js_name = changedIndices)]
        pub fn changed_indices(&self, revision: u64) -> Result<Vec<u32>, JsValue> {
            self.inner.changed_indices(revision).map(<[u32]>::to_vec).map_err(js_error)
        }
        /// [visited, multiplied, copied roots, preserved worlds, changed].
        #[wasm_bindgen(js_name = solveStats)]
        pub fn solve_stats(&self) -> Vec<u32> {
            let s = self.inner.last_stats();
            vec![s.visited as u32, s.multiplied as u32, s.copied_roots as u32, s.preserved_worlds as u32, s.changed as u32]
        }
    }
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
pub use browser::WasmTransformHierarchy;

#[cfg(test)]
mod tests {
    use super::*;

    fn translation(x: f64, y: f64, z: f64) -> [f64; 16] {
        let mut m = Matrix4::identity().elements;
        m[12] = x; m[13] = y; m[14] = z;
        m
    }
    fn flat(values: &[[f64; 16]]) -> Vec<f64> { values.iter().flatten().copied().collect() }
    fn forest(parents: &[i32]) -> TransformHierarchy {
        let locals = flat(&vec![translation(1.0, 0.0, 0.0); parents.len()]);
        TransformHierarchy::new(parents, &locals, None, None).unwrap()
    }
    fn x(h: &TransformHierarchy, index: usize) -> f64 {
        h.world_matrices(h.revision()).unwrap()[index].elements[12]
    }

    #[test]
    fn arbitrary_node_order_and_multiple_roots() {
        let mut h = forest(&[2, -1, 1, -1, 0]);
        let stats = h.solve();
        assert_eq!(stats.visited, 5);
        assert_eq!(stats.multiplied, 3);
        assert_eq!(stats.copied_roots, 2);
        assert_eq!((x(&h, 0), x(&h, 1), x(&h, 2), x(&h, 3), x(&h, 4)), (3.0, 1.0, 2.0, 1.0, 4.0));
    }
    #[test]
    fn empty_forest_is_a_solved_empty_snapshot() {
        let mut h = forest(&[]);
        assert!(h.is_empty());
        assert_eq!(h.solve().visited, 0);
        assert_eq!(h.copy_world_matrices(0).unwrap(), Vec::<f64>::new());
    }
    #[test]
    fn leaf_edits_and_clean_frames_skip_unaffected_nodes() {
        let mut h = forest(&[-1, 0, 0, 1, 1, 2]);
        h.solve();
        assert_eq!(h.solve().visited, 0);
        h.set_local_matrices(&[4], &translation(10.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.solve().visited, 1);
        assert_eq!(x(&h, 4), 12.0);
        assert_eq!(x(&h, 5), 3.0);
        assert_eq!(h.changed_indices(h.revision()).unwrap(), &[4]);
    }
    #[test]
    fn overlapping_dirty_subtrees_are_solved_once() {
        let mut h = forest(&[-1, 0, 0, 1, 1, 2]);
        h.solve();
        h.set_local_matrices(&[4, 1, 0], &flat(&[translation(2.0, 0.0, 0.0); 3])).unwrap();
        let stats = h.solve();
        assert_eq!(stats.visited, 6);
        assert_eq!(stats.multiplied, 5);
        assert_eq!(x(&h, 4), 6.0);
    }
    #[test]
    fn disjoint_dirty_subtrees_do_not_force_a_full_forest_scan() {
        let mut h = forest(&[-1, 0, 0, 1, 2, -1, 5, 6]);
        h.solve();
        h.set_local_matrices(&[1, 6], &flat(&[translation(3.0, 0.0, 0.0); 2])).unwrap();
        assert_eq!(h.solve().visited, 4);
        assert_eq!(x(&h, 4), 3.0);
    }
    #[test]
    fn manual_worlds_are_boundaries_not_hidden_local_transforms() {
        let locals = flat(&[translation(1.0, 0.0, 0.0); 3]);
        let worlds = flat(&[translation(0.0, 0.0, 0.0), translation(20.0, 0.0, 0.0), translation(0.0, 0.0, 0.0)]);
        let mut h = TransformHierarchy::new(&[-1, 0, 1], &locals, Some(&worlds), Some(&[1, 0, 1])).unwrap();
        assert_eq!(h.solve().preserved_worlds, 1);
        assert_eq!(x(&h, 2), 21.0);
        h.set_local_matrices(&[0], &translation(100.0, 0.0, 0.0)).unwrap();
        h.solve();
        assert_eq!(x(&h, 2), 21.0);
        h.set_world_matrices(&[1], &translation(40.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.solve().visited, 2);
        assert_eq!(h.changed_indices(h.revision()).unwrap(), &[1, 2]);
        assert_eq!(x(&h, 2), 41.0);
        h.set_world_auto(&[1], &[1]).unwrap();
        h.solve();
        assert_eq!(x(&h, 2), 102.0);
    }
    #[test]
    fn manual_world_write_batch_rejects_before_mutating_any_node() {
        let mut h = forest(&[-1, 0]);
        h.solve();
        h.set_world_auto(&[0], &[0]).unwrap(); h.solve();
        let revision = h.revision();
        assert!(matches!(h.set_world_matrices(&[0, 1], &flat(&[translation(99.0, 0.0, 0.0); 2])), Err(HierarchyError::AutomaticWorld { node: 1 })));
        assert_eq!(h.revision(), revision);
        assert_eq!(x(&h, 0), 1.0);
    }
    #[test]
    fn invalid_last_index_does_not_partially_apply_a_batch() {
        let mut h = forest(&[-1, 0]); h.solve();
        assert!(h.set_local_matrices(&[0, 2], &flat(&[translation(99.0, 0.0, 0.0); 2])).is_err());
        assert_eq!(h.revision(), 0);
        assert_eq!(h.solve().visited, 0);
        assert_eq!(x(&h, 0), 1.0);
    }
    #[test]
    fn invalid_last_flag_does_not_partially_apply_a_batch() {
        let mut h = forest(&[-1, 0]); h.solve();
        assert!(h.set_world_auto(&[0, 1], &[0, 2]).is_err());
        assert_eq!(h.revision(), 0);
        assert!(h.set_world_matrices(&[0], &translation(9.0, 0.0, 0.0)).is_err());
    }
    #[test]
    fn cycle_rejection_preserves_last_valid_snapshot_and_node_ids() {
        let mut h = forest(&[-1, 0, 1]); h.solve();
        assert_eq!(h.reparent(&[2, 0, 1]), Err(HierarchyError::Cycle));
        assert_eq!(h.revision(), 0);
        assert_eq!(x(&h, 2), 3.0);
        h.reparent(&[2, -1, 1]).unwrap();
        h.solve();
        assert_eq!((x(&h, 0), x(&h, 1), x(&h, 2)), (3.0, 1.0, 2.0));
    }
    #[test]
    fn invalid_parents_and_shapes_are_rejected() {
        for parents in [&[-2][..], &[1][..], &[0][..], &[1, 0][..]] {
            assert!(TransformHierarchy::new(parents, &flat(&vec![translation(0.0, 0.0, 0.0); parents.len()]), None, None).is_err());
        }
        assert!(TransformHierarchy::new(&[-1], &[0.0; 15], None, None).is_err());
        assert!(TransformHierarchy::new(&[-1], &[0.0; 16], Some(&[]), None).is_err());
        assert!(TransformHierarchy::new(&[-1], &[0.0; 16], None, Some(&[2])).is_err());
    }
    #[test]
    fn revisions_gate_publication_before_and_after_solving() {
        let mut h = forest(&[-1]);
        assert!(matches!(h.world_matrices(0), Err(HierarchyError::Unsolved { current: 0 })));
        h.solve();
        let old = h.copy_world_matrices(0).unwrap();
        h.set_local_matrices(&[0], &translation(3.0, 0.0, 0.0)).unwrap();
        assert!(matches!(h.world_matrices(0), Err(HierarchyError::Stale { .. })));
        assert!(matches!(h.world_matrices(1), Err(HierarchyError::Unsolved { .. })));
        h.solve();
        assert!(h.changed_indices(0).is_err());
        assert_eq!(old[12], 1.0);
        assert_eq!(x(&h, 0), 3.0);
    }
    #[test]
    fn no_op_updates_preserve_receipts_but_signed_zero_changes_do_not() {
        let mut h = forest(&[-1]); h.solve();
        assert_eq!(h.set_local_matrices(&[0], &translation(1.0, 0.0, 0.0)).unwrap(), 0);
        assert_eq!(h.reparent(&[-1]).unwrap(), 0);
        assert_eq!(h.set_world_auto(&[0], &[1]).unwrap(), 0);
        let mut local = translation(1.0, 0.0, 0.0); local[1] = -0.0;
        assert_eq!(h.set_local_matrices(&[0], &local).unwrap(), 1);
        h.solve();
        assert_eq!(h.world_matrices(1).unwrap()[0].elements[1].to_bits(), (-0.0f64).to_bits());
    }
    #[test]
    fn nan_payloads_and_infinities_survive_root_copies() {
        let mut local = translation(0.0, 0.0, 0.0);
        local[0] = f64::from_bits(0x7ff8_0000_0000_0042); local[2] = f64::INFINITY;
        let mut h = TransformHierarchy::new(&[-1], &local, None, None).unwrap(); h.solve();
        assert_eq!(h.set_local_matrices(&[0], &local).unwrap(), 0);
        let world = &h.world_matrices(0).unwrap()[0].elements;
        assert_eq!(world[0].to_bits(), local[0].to_bits());
        assert_eq!(world[2], f64::INFINITY);
    }
    #[test]
    fn duplicate_updates_obey_source_order() {
        let mut h = forest(&[-1]); h.solve();
        h.set_local_matrices(&[0, 0], &flat(&[translation(2.0, 0.0, 0.0), translation(5.0, 0.0, 0.0)])).unwrap();
        assert_eq!(h.solve().visited, 1);
        assert_eq!(x(&h, 0), 5.0);
    }
    #[test]
    fn projective_noncommuting_transforms_use_parent_times_local() {
        let mut parent = Matrix4::identity(); parent.elements[0] = 2.0; parent.elements[3] = 0.25;
        let local = Matrix4::from_elements(translation(3.0, 4.0, 5.0));
        let mut expected = Matrix4::zero(); expected.multiply_matrices(&parent, &local);
        let mut h = TransformHierarchy::new(&[-1, 0], &flat(&[parent.elements, local.elements]), None, None).unwrap();
        h.solve();
        assert_eq!(h.world_matrices(0).unwrap()[1], expected);
        assert_eq!(expected.elements[12], 6.0);
        assert_eq!(expected.elements[15], 1.75);
    }
    #[test]
    fn deep_hierarchies_do_not_recurse_on_the_native_or_wasm_stack() {
        let count = 50_000;
        let parents: Vec<i32> = (0..count).map(|n| n as i32 - 1).collect();
        let mut h = forest(&parents);
        assert_eq!(h.solve().visited, count);
        assert_eq!(x(&h, count - 1), count as f64);
        h.set_local_matrices(&[(count - 1) as u32], &translation(2.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.solve().visited, 1);
    }
    #[test]
    fn change_tracking_reuses_capacity_and_deduplicates_external_to_auto_updates() {
        let mut h = forest(&[-1, 0, 1]); h.solve();
        let allocation = h.changed.as_ptr();
        let capacity = h.changed.capacity();
        h.set_world_auto(&[1], &[0]).unwrap();
        h.set_world_matrices(&[1], &translation(20.0, 0.0, 0.0)).unwrap();
        h.set_world_auto(&[1], &[1]).unwrap();
        assert_eq!(h.solve().visited, 2);
        assert_eq!(h.changed_indices(h.revision()).unwrap(), &[1]);
        assert_eq!(x(&h, 1), 2.0);
        assert_eq!(x(&h, 2), 3.0);
        assert_eq!(h.changed.as_ptr(), allocation);
        assert_eq!(h.changed.capacity(), capacity);
        assert!(h.world_changed.iter().all(|&changed| !changed));
        assert_eq!(h.solve().changed, 0);
        assert_eq!(h.changed.as_ptr(), allocation);
    }
    #[test]
    fn revision_exhaustion_does_not_modify_any_bank() {
        let mut h = forest(&[-1]); h.solve();
        h.revision = u64::MAX; h.solved_revision = Some(u64::MAX);
        assert_eq!(h.set_local_matrices(&[0], &translation(4.0, 0.0, 0.0)), Err(HierarchyError::RevisionExhausted));
        assert_eq!(x(&h, 0), 1.0);
        assert!(h.dirty.is_empty());
    }
}
