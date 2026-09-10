//! Pass graph definition, topological scheduling, cycle detection, and compilation into execution plans.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::canvas::{CanvasEpochTracker, CanvasId};
use crate::error::GraphError;
use crate::hazard::{can_split_pass, split_pass_on_hazard, validate_pass_hazards};
use crate::pass::{Pass, PassId};
use crate::plan::{ExecutionPlan, PlanSegment};
use crate::resource::ResourceKind;

/// Top-level graph of passes and dependencies (§6.1, §6.3).
#[derive(Default, Clone, Debug)]
pub struct PassGraph {
    passes: Vec<Pass>,
    next_id: u32,
}

impl PassGraph {
    /// Create an empty pass graph.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            passes: Vec::new(),
            next_id: 1,
        }
    }

    /// Allocate a fresh unique PassId with checked overflow.
    pub fn alloc_pass_id(&mut self) -> Result<PassId, GraphError> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(GraphError::PassIdOverflow)?;
        Ok(PassId::new(id))
    }

    /// Add a pass to the graph, returning its ID.
    ///
    /// Rejects duplicate PassId BEFORE mutating graph state.
    pub fn add_pass(&mut self, pass: Pass) -> Result<PassId, GraphError> {
        if self.passes.iter().any(|p| p.id == pass.id) {
            return Err(GraphError::DuplicatePassId {
                pass_id: pass.id.get(),
            });
        }
        if pass.id.get() >= self.next_id {
            self.next_id = pass
                .id
                .get()
                .checked_add(1)
                .ok_or(GraphError::PassIdOverflow)?;
        }
        let id = pass.id;
        self.passes.push(pass);
        Ok(id)
    }

    /// Add an explicit ordering constraint: `dependent` must execute after `dependency`.
    pub fn add_dependency(
        &mut self,
        dependent: PassId,
        dependency: PassId,
    ) -> Result<(), GraphError> {
        if !self.passes.iter().any(|p| p.id == dependency) {
            return Err(GraphError::MissingDependency {
                pass_id: dependent.get(),
                dependency_id: dependency.get(),
            });
        }
        let pass = self
            .passes
            .iter_mut()
            .find(|p| p.id == dependent)
            .ok_or(GraphError::PassNotFound {
                pass_id: dependent.get(),
            })?;

        if !pass.dependencies.contains(&dependency) {
            pass.dependencies.push(dependency);
        }
        Ok(())
    }

    /// Compiles the pass graph into an ordered, validated `ExecutionPlan`.
    ///
    /// Pipeline:
    /// 1. Reject empty graph.
    /// 2. Validate all dependency targets exist.
    /// 3. Usage-scope hazard validation and automatic pass splitting.
    /// 4. Rewire downstream consumers of split passes to depend on the final segment.
    /// 5. Validate canvas usage against the provided `CanvasEpochTracker` (cannot bypass guard).
    /// 6. Topological sort with cycle detection.
    /// 7. Materialize `PlanSegment` sequence.
    pub fn compile(
        &self,
        canvas_tracker: Option<&CanvasEpochTracker>,
    ) -> Result<ExecutionPlan, GraphError> {
        if self.passes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        // 1. Verify all explicit dependencies exist
        for pass in &self.passes {
            for &dep in &pass.dependencies {
                if !self.passes.iter().any(|p| p.id == dep) {
                    return Err(GraphError::MissingDependency {
                        pass_id: pass.id.get(),
                        dependency_id: dep.get(),
                    });
                }
            }
        }

        // 2. Perform hazard analysis and automatic pass splitting
        let mut resolved_passes = Vec::new();
        let mut split_count = 0;
        let mut split_reasons = Vec::new();
        let mut alloc_id = self.next_id;
        let mut split_rewires = Vec::new();

        for pass in &self.passes {
            match validate_pass_hazards(pass) {
                Ok(()) => {
                    resolved_passes.push(pass.clone());
                }
                Err(err) => {
                    if can_split_pass(pass) {
                        let next_pid = PassId::new(alloc_id);
                        alloc_id = alloc_id.checked_add(10).ok_or(GraphError::PassIdOverflow)?;
                        let splits = split_pass_on_hazard(pass, next_pid)?;
                        let final_split_id = splits.last().unwrap().id;
                        split_rewires.push((pass.id, final_split_id));

                        split_count += splits.len().saturating_sub(1);
                        split_reasons.push(alloc::format!(
                            "Split pass '{}' ({}) into {} segments preserving outputs and load/store state",
                            pass.name,
                            pass.id,
                            splits.len()
                        ));
                        resolved_passes.extend(splits);
                    } else {
                        return Err(GraphError::Hazard(err));
                    }
                }
            }
        }

        // 3. Rewire downstream consumers of split passes to depend on the final split segment
        for (original_id, final_id) in split_rewires {
            if original_id != final_id {
                for p in &mut resolved_passes {
                    // Do not rewire dependencies inside the split segments themselves
                    if p.id != original_id && p.id != final_id {
                        for dep in &mut p.dependencies {
                            if *dep == original_id {
                                *dep = final_id;
                            }
                        }
                    }
                }
            }
        }

        // 4. Validate canvas usage freshness; missing tracker on canvas use CANNOT bypass guard
        let mut captured_canvas_epoch = None;
        for pass in &resolved_passes {
            for u in pass.all_uses() {
                let detected_canvas = if let Some(tracker) = canvas_tracker {
                    tracker
                        .find_by_resource(u.resource_id)
                        .map(|cid| (cid, u.canvas_epoch))
                        .or_else(|| {
                            if u.kind == ResourceKind::CanvasOutput {
                                Some((CanvasId::new(u.resource_id.get()), u.canvas_epoch))
                            } else {
                                None
                            }
                        })
                } else if u.kind == ResourceKind::CanvasOutput {
                    Some((CanvasId::new(u.resource_id.get()), u.canvas_epoch))
                } else {
                    None
                };

                if let Some((canvas_id, epoch_opt)) = detected_canvas {
                    let tracker = canvas_tracker.ok_or_else(|| {
                        GraphError::Canvas(crate::error::CanvasError::CanvasNotAcquired {
                            canvas_id: canvas_id.get(),
                        })
                    })?;
                    let captured_epoch = epoch_opt.ok_or_else(|| {
                        GraphError::Canvas(crate::error::CanvasError::CanvasNotAcquired {
                            canvas_id: canvas_id.get(),
                        })
                    })?;
                    tracker.validate_canvas_access(canvas_id, captured_epoch)?;
                    captured_canvas_epoch = Some(captured_epoch);
                }
            }
        }

        // 5. Topological sort with cycle detection
        let ordered_passes = topological_sort(&resolved_passes)?;

        // 6. Build execution plan segments
        let segments: Vec<PlanSegment> = ordered_passes.iter().map(PlanSegment::from_pass).collect();
        let pass_count = segments.len();

        Ok(ExecutionPlan {
            segments,
            canvas_epoch: captured_canvas_epoch,
            pass_count,
            split_count,
            split_reasons,
        })
    }
}

/// Performs a deterministic topological sort of passes, returning `CycleDetected` if a cycle exists.
fn topological_sort(passes: &[Pass]) -> Result<Vec<Pass>, GraphError> {
    let mut pass_map = BTreeMap::new();
    let mut in_degrees = BTreeMap::new();
    let mut adj_list: BTreeMap<PassId, Vec<PassId>> = BTreeMap::new();

    for pass in passes {
        pass_map.insert(pass.id, pass.clone());
        in_degrees.insert(pass.id, 0);
        adj_list.insert(pass.id, Vec::new());
    }

    for pass in passes {
        for &dep in &pass.dependencies {
            if let Some(outgoing) = adj_list.get_mut(&dep) {
                outgoing.push(pass.id);
            }
            if let Some(deg) = in_degrees.get_mut(&pass.id) {
                *deg += 1;
            }
        }
    }

    // Worklist of passes with in-degree 0 (stable sorting by PassId)
    let mut zero_in_degree = Vec::new();
    for (&id, &deg) in &in_degrees {
        if deg == 0 {
            zero_in_degree.push(id);
        }
    }
    zero_in_degree.sort_by(|a, b| b.cmp(a));

    let mut sorted_passes = Vec::new();

    while let Some(current_id) = zero_in_degree.pop() {
        let pass = pass_map.get(&current_id).unwrap();
        sorted_passes.push(pass.clone());

        if let Some(neighbors) = adj_list.get(&current_id) {
            for &next_id in neighbors {
                let deg = in_degrees.get_mut(&next_id).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    zero_in_degree.push(next_id);
                    zero_in_degree.sort_by(|a, b| b.cmp(a)); // maintain deterministic lowest-first order
                }
            }
        }
    }

    // If not all passes were visited, there is a cycle
    if sorted_passes.len() < passes.len() {
        let mut cycle_nodes = Vec::new();
        for (&id, &deg) in &in_degrees {
            if deg > 0 {
                cycle_nodes.push(id.get());
            }
        }
        return Err(GraphError::CycleDetected { cycle: cycle_nodes });
    }

    Ok(sorted_passes)
}
