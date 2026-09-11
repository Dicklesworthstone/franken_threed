# CONTRACT.md: `f3d-graph`

> Crate contract for `f3d-graph` per AGENTS.md "Documentation and Contracts".

---

## 1. Purpose and Position in the Dependency Direction

`f3d-graph` implements the middle execution representation of FrankenThreeD (§6.1 "pass graph"):
recording render, compute, and copy passes, typed resource access, explicit attachment
load/store/clear/resolve semantics, required order constraints, and WebGPU usage-scope hazard tracking.

In the planned workspace crate dependency architecture:

```text
core <- math
core/math <- scene
core <- assets
core <- shader
core/scene/shader <- graph
core/graph <- gpu
scene/assets/gpu + Asupersync <- runtime
core/scene/shader/graph <- compiler
compiler/runtime <- cli
public crate APIs <- conformance
```

- Initial implementation requires only `f3d-core` (handle domains, versions, layouts, and epochs).
- It introduces zero placeholders or premature scene/shader dependencies.
- It introduces zero browser dependencies (`wasm-bindgen`, `web-sys`) and zero native driver APIs.
- `#![no_std]` capable with `alloc`.

---

## 2. Public Types and Semantics

| Type | Semantics |
|---|---|
| `PassGraph` | Top-level graph containing passes, explicit dependencies, and history references. |
| `Pass` | Individual scheduled execution unit of kind `Render`, `Compute`, or `Copy`. |
| `PassKind` | Execution category (`Render`, `Compute`, `Copy`). Copies are strictly segregated from render passes. |
| `PassId` | Strongly-typed 32-bit handle identifying a pass in the graph. |
| `ResourceId` | Strongly-typed 32-bit handle identifying a GPU-backed buffer, texture, or canvas output. |
| `ResourceKind` | Resource classifier: `Buffer`, `Texture`, `CanvasOutput`. |
| `ResourceUse` | Pairs a `ResourceId`, `DataVersion`, `SubresourceRange`, and `ResourceAccess`. |
| `SubresourceRange` | Subresource coverage: `WholeBuffer` for buffers; mip/layer/aspect ranges for textures. |
| `TextureAspect` | Texture aspect mask (`All`, `Color`, `DepthOnly`, `StencilOnly`) with overlap checking. |
| `ResourceAccess` | Typed usage mode distinguishing read-only access (uniform, vertex, index, sample, copy src) from writable access (storage write, color attachment, depth/stencil attachment, copy dst). |
| `AttachmentOp` / `ColorAttachment` / `DepthStencilAttachment` | Explicit load/store/clear/resolve attachment configuration for render passes. |
| `CanvasOutput` / `CanvasId` / `CanvasEpochTracker` | Short-lived canvas swapchain representation bound strictly to the "acquire → submit" frame interval. |
| `ExecutionPlan` / `PlanSegment` | Ordered sequence of validated pass segments compiled for bridge submission. |
| `HazardError` / `GraphError` / `CanvasError` | Strongly-typed non-panicking error hierarchy for usage-scope violations. |

---

## 3. Invariants

1. **Whole-Buffer Subresource Invariant**: For WebGPU usage-scope validation, a buffer is a single whole subresource. Assigning disjoint byte offsets or ranges in one buffer incompatible usages (e.g. storage write and uniform read in the same render pass) is strictly illegal and cannot legalize the shared buffer.
2. **Render-Pass Usage Scope Invariant**: Within a single render pass, all accesses to any texture subresource or buffer must be compatible. A texture subresource bound as a color or depth attachment cannot simultaneously be sampled in the same render pass.
3. **Pass Splitting Invariant**: When an attachment-versus-sampling hazard is detected within a render pass with separate legal destinations, the pass graph automatically splits the pass into sequential ordered segments, preserving all color destinations, depth/stencil state, and load/store semantics without dropping outputs. Unsupported feedback loops (e.g. sampling an attachment when no separate destination exists) are strictly rejected with `HazardError::AttachmentSamplingConflict`, NEVER legalized by dropping outputs.
4. **Downstream Consumer Rewiring Invariant**: When a pass is split into multiple sequential segments, explicit dependencies from downstream consumer passes are rewired to the final split segment, guaranteeing that downstream passes cannot execute prematurely between split segments.
5. **Compute Dispatch Writable Alias Invariant**: Within a single compute dispatch, writable aliases (multiple writable bindings pointing to the same buffer or overlapping texture subresource) are strictly forbidden and rejected.
6. **Separation of Copy Passes and Pitch Alignment**: A copy command cannot be recorded or executed inside a render pass. Copies reside exclusively in dedicated `Copy` passes. Texture-to-buffer copy rows must be 256-byte aligned (`COPY_BYTES_PER_ROW_ALIGNMENT`) with checked overflow.
7. **Canvas Freshness and Epoch Isolation**: Canvas swapchain textures are valid strictly inside a frame's acquire-to-submit interval. Plans carry captured output epochs that are validated against the `CanvasEpochTracker` at compilation and execution/publication boundaries. Using a canvas texture across output epochs returns `CanvasError::CanvasCachedAcrossEpochs`. Missing trackers on canvas use cannot bypass this guard.
8. **Multiple Canvases Epoch Independence**: Multiple canvases maintain separate output epochs. Advancing the epoch on canvas A never alters the validity of canvas B.
9. **Topological Order and Cycle Detection**: Passes execute in deterministic topological order. Dependency cycles are detected and rejected with `GraphError::CycleDetected`. Duplicate `PassId`s are rejected before mutating graph state, and `next_id` allocations use checked addition against `u32::MAX`.
10. **Zero Memory Barrier Fabrication**: The compiler and pass graph never invent explicit native memory barriers or sync primitives that browser WebGPU does not expose.
11. **Zero Dropped Passes**: Every admitted pass is either compiled into the execution plan or returns an explicit error; passes are never silently dropped.

---

## 4. Implementation Ownership Route

- **Route**: New Rust execution core (`crates/f3d-graph`).
- Pure computational graph analysis, hazard validation, pass splitting, and plan ordering.
- Zero retained JavaScript execution and zero exact-backend fallback.
- Portable across native targets and `wasm32-unknown-unknown`.

---

## 5. Error Model and Source-Error Preservation

- Errors are strictly typed via `GraphError`, `HazardError`, and `CanvasError`.
- No panics in normal or adversarial operation; all compilation and validation steps return `Result<T, GraphError>`.
- Error variants carry exact conflicting resource handles, subresource ranges, pass IDs, and byte offset annotations for debuggability.

---

## 6. Determinism and Numeric-Fidelity Class

- Deterministic topological ordering: ties in dependency order are resolved by stable, monotonic pass creation index.
- No floating-point decisions or nondeterministic hash iteration order.
- Plan generation is strictly bit-identical across runs given identical pass specifications.

---

## 7. Cancellation Behavior

- Pass graph compilation is synchronous, allocation-bounded, and deterministic.
- Graph instances can be dropped cleanly at any point without leaking external handles or leaving unmanaged background tasks.

---

## 8. Unsafe Boundary

- **`#![forbid(unsafe_code)]`** is declared at the crate root.
- No unsafe blocks, functions, or external pointers exist in `f3d-graph`.

---

## 9. Feature Flags

- `default = ["std", "serde"]`
- `std`: Standard library support.
- `serde`: Serialization and deserialization for plan debugging and diagnostic snapshots.
- `test-support`: Additional test fixtures and diagnostic verification helpers.

---

## 10. Conformance Tests

- Unit tests in `crates/f3d-graph/tests/graph_tests.rs`:
  - Positive: Legal independent read-only combinations and explicit topological ordering.
  - Negative: Disjoint byte offsets in ONE buffer cannot legalize incompatible usages (whole-buffer rule).
  - Positive & Splitting: Automatic pass splitting when attachment vs sampling conflicts occur in a render pass.
  - Negative: Compute per-dispatch writable binding aliases rejected.
  - Negative: Pass dependency cycle detection.
  - Negative: Canvas texture caching across output epochs rejected.
  - Positive: Bridge single-pass and two-target plans matching ChartreuseFern's bridge protocol.
  - Positive: Versioned buffer snapshot (Red-A / Blue-B) preserving distinct `DataVersion`s without overwrite.
  - Negative: Render bundle execution inside a copy pass is rejected (`HazardError::BundleInCopyPass`).
  - Negative & Positive: Direct draw following a bundle assuming warm state is rejected (`HazardError::BundleDirectDrawRequiresRebind`), while explicit rebind succeeds and correctly reports bundle boundaries and bundle IDs.
  - Positive: Interleaved bundle and direct draw lifecycle where draws following explicit rebind can inherit state, but a subsequent bundle immediately resets state again.
  - Property Test (LCG Fuzzing, 1,000 iterations): Random small pass graphs with random resource uses assert `PassGraph::compile` never panics and all accepted plans satisfy topological order.
  - Property Test (LCG Fuzzing, 1,000 iterations): Any pass reading and writing overlapping subresources is strictly rejected (`AttachmentSamplingConflict`, `ComputeWritableAlias`, `OverlappingCopyEndpoints`).
  - Property Test (LCG Fuzzing, 1,000 iterations): Disjoint-offset writes to one buffer are strictly rejected under the whole-buffer rule (`WholeBufferConflict`).
  - Property Test (LCG Fuzzing, 1,000 iterations): Duplicate PassIds are strictly rejected before mutating graph state (`DuplicatePassId`), leaving graph integrity intact.
  - Property Test (LCG Fuzzing, 1,000 iterations): Complex DAG structures with random multi-edge dependencies verify that every accepted plan strictly orders dependencies before dependents.

---

## 11. Stable Consumer Contract for Bridge Consumption (`gpu_host.rs`)

> **NO-CLAIM: Exposing typed plan accessors is a compositional software interface for bridge command serialization; it is not physical GPU command recording, driver execution, or measured frame acceleration.**

ChartreuseFern's `crates/f3d-runtime/src/gpu_host.rs` reads compiled `ExecutionPlan` instances to generate binary `GpuCommand` packets without inspecting graph internals or parsing unstructured tables.

### 11.1 ExecutionPlan Accessors

| Method | Return Type | Description |
|---|---|---|
| `plan.segments()` | `&[PlanSegment]` | Ordered slice of executable plan segments. |
| `plan.segment_count()` | `usize` | Total number of segments in the plan. |
| `plan.is_empty()` | `bool` | Returns `true` if the plan contains zero segments. |
| `plan.canvas_epoch()` | `Option<Epoch>` | Captured output epoch of any bound canvas swapchains. |
| `plan.pass_count()` | `usize` | Number of executed passes produced after compilation. |
| `plan.split_count()` | `usize` | Number of split passes introduced during hazard resolution. |
| `plan.split_reasons()` | `&[String]` | Diagnostic descriptions of any pass splits. |
| `plan.has_bundles()` | `bool` | Returns `true` if any segment executes a render bundle. |
| `plan.total_bundle_count()` | `usize` | Total number of render bundle executions across all segments. |
| `plan.validate_execution(tracker)` | `Result<(), GraphError>` | Validates canvas epoch freshness against active tracker intervals. |

### 11.2 PlanSegment Accessors

| Method | Return Type | Description |
|---|---|---|
| `seg.pass_id()` | `PassId` | Strongly typed identifier of the pass. |
| `seg.name()` | `&str` | Diagnostic pass name. |
| `seg.kind()` | `PassKind` | Pass kind: `Render`, `Compute`, or `Copy`. |
| `seg.is_render()` | `bool` | Returns `true` if this segment encodes a WebGPU Render pass. |
| `seg.is_compute()` | `bool` | Returns `true` if this segment encodes a WebGPU Compute pass. |
| `seg.is_copy()` | `bool` | Returns `true` if this segment encodes an isolated Copy pass. |
| `seg.color_attachments()` | `&[ColorAttachment]` | All color attachments bound to this segment. |
| `seg.primary_color_attachment()` | `Option<&ColorAttachment>` | First color attachment, if any. |
| `seg.depth_stencil_attachment()` | `Option<&DepthStencilAttachment>` | Optional depth/stencil attachment configuration. |
| `seg.draws()` | `&[Draw]` | Slice of draw call buckets in this segment (both direct draws and bundle executions). |
| `seg.draw_count()` | `usize` | Number of draw call buckets. |
| `seg.first_draw()` | `Option<&Draw>` | First draw call bucket, if any. |
| `seg.has_bundles()` | `bool` | Returns `true` if this segment contains any bundle executions. |
| `seg.bundle_count()` | `usize` | Number of render bundle executions in this segment. |
| `seg.bundle_boundaries()` | `Vec<usize>` | Draw indices of all bundle executions marking state reset points. |
| `seg.bundle_ids()` | `Vec<u32>` | List of all bundle IDs executed in this segment. |
| `seg.draw_crosses_bundle_boundary(draw_idx)` | `bool` | `true` if draw command at `draw_idx` directly follows a bundle execution. |
| `seg.dispatches()` | `&[Dispatch]` | Slice of compute dispatches in this segment. |
| `seg.dispatch_count()` | `usize` | Number of compute dispatches. |
| `seg.copies()` | `&[CopyCommand]` | Slice of copy commands in this segment. |
| `seg.copy_count()` | `usize` | Number of copy commands. |
| `seg.first_copy()` | `Option<&CopyCommand>` | First copy command, if any. |
| `seg.versioned_reads()` | `&[(ResourceId, DataVersion)]` | All versioned resource reads. |
| `seg.versioned_writes()` | `&[(ResourceId, DataVersion)]` | All versioned resource writes. |

### 11.3 Attachment, Draw, and Copy Accessors

| Type | Method | Return Type | Description |
|---|---|---|---|
| `ColorAttachment` | `target_id()` | `ResourceId` | Render target texture or canvas resource. |
| `ColorAttachment` | `load_op()` | `LoadOp` | Explicit load operation (`Clear`, `Load`, `DontCare`). |
| `ColorAttachment` | `store_op()` | `StoreOp` | Explicit store operation (`Store`, `Discard`). |
| `ColorAttachment` | `clear_color()` | `[f32; 4]` | Normalized `[r, g, b, a]` clear color. |
| `ColorAttachment` | `resolve_target()` | `Option<ResourceId>` | Optional MSAA resolve destination. |
| `ColorAttachment` | `canvas_epoch()` | `Option<Epoch>` | Captured swapchain epoch if targeting a canvas. |
| `ColorAttachment` | `is_canvas()` | `bool` | `true` if attachment is a canvas presentation surface. |
| `DepthStencilAttachment` | `target_id()` | `ResourceId` | Depth/stencil texture resource. |
| `DepthStencilAttachment` | `depth_load_op()` | `Option<LoadOp>` | Depth load operation. |
| `DepthStencilAttachment` | `depth_store_op()` | `Option<StoreOp>` | Depth store operation. |
| `DepthStencilAttachment` | `depth_clear_value()` | `f32` | Depth clear value (typically `1.0`). |
| `DepthStencilAttachment` | `depth_read_only()` | `bool` | Whether depth writes are disabled. |
| `Draw` | `kind()` | `DrawKind` | Command category (`Direct` or `Bundle { bundle_id }`). |
| `Draw` | `is_bundle()` | `bool` | `true` if this command executes a pre-recorded render bundle. |
| `Draw` | `is_direct()` | `bool` | `true` if this command is a direct draw. |
| `Draw` | `bundle_id()` | `Option<u32>` | Pre-recorded bundle identifier (`None` for direct draws). |
| `Draw` | `assumes_warm_state()` | `bool` | `true` if this draw assumes warm/inherited pipeline and binding state. |
| `Draw` | `rebind_required()` | `bool` | `true` if this draw explicitly rebinds its pipeline and resources. |
| `Draw` | `pipeline_id()` | `u32` | Bound shader/pipeline variant handle. |
| `Draw` | `vertex_count()` | `u32` | Number of vertices to draw. |
| `Draw` | `instance_count()` | `u32` | Number of instances to draw. |
| `Draw` | `uniform_dynamic_offset()` | `u32` | 256-byte aligned dynamic uniform buffer offset. |
| `Draw` | `vertex_buffer()` | `Option<ResourceId>` | Bound vertex buffer handle, if any. |
| `Draw` | `vertex_buffer_id()` | `u32` | Raw vertex buffer identifier (`0` if none). |
| `CopyCommand` | `is_texture_to_buffer()` | `bool` | `true` if command is a readback/staging copy. |
| `CopyCommand` | `bytes_per_row()` | `Option<u32>` | 256-byte aligned row pitch from core layout helper. |
| `CopyCommand` | `as_texture_to_buffer()` | `Option<(ResourceId, ResourceId, u32, u32, u32)>` | Tuple of `(texture_id, buffer_id, width, height, bytes_per_row)`. |

### 11.4 Pre-Compiled Bridge Plan Constructors

| Constructor | Signature / Parameters | Output Topology |
|---|---|---|
| `build_single_pass_bridge_plan` | `(target_id, pipeline_id, vertex_buf, vertex_count, version, canvas_epoch, canvas_tracker)` | 1 Render pass |
| `build_two_target_bridge_plan` | `(offscreen_tex, canvas_res, readback_buf, pipeline_offscreen, pipeline_canvas, vbuf, vcount, width, height, version, canvas_epoch, canvas_tracker)` | 2 Render passes + 1 Copy pass |
| `build_red_a_blue_b_plan` | `(shared_buf, target_a, target_b, readback_a, readback_b, red_version, blue_version, red_offset, blue_offset, pipeline_id)` | 2 Render passes (black clear `[0.0, 0.0, 0.0, 1.0]`) + 2 Copy passes (`aligned_bytes_per_row(64)`) |
| `build_red_a_blue_b_render_plan` | `(shared_buf, target_a, target_b, red_version, blue_version, red_offset, blue_offset, pipeline_id)` | 2 Render passes (black clear `[0.0, 0.0, 0.0, 1.0]`) without readback copies |

### 11.5 Render Bundle State-Reset Contract for Bridge Consumption (`gpu_host.rs`)

> **NO-CLAIM: Exposing render bundle boundaries and state-reset tracking is a compositional graph invariant; it is not physical GPU bundle execution, driver command recording, or measured frame acceleration.**

In WebGPU, pre-recorded `GPURenderBundle` objects are executed inside a render pass via `GPURenderPassEncoder.executeBundles([bundle])`.

#### Invariants & Hazard Rules:
1. **Pass Segregation**: Render bundles execute strictly within `PassKind::Render` passes. Placing a render bundle inside a `PassKind::Copy` pass is an illegal usage hazard and returns `HazardError::BundleInCopyPass { pass_id }`. Placing a bundle inside a `PassKind::Compute` pass returns `HazardError::BundleInNonRenderPass { pass_id, pass_kind: PassKind::Compute }`.
2. **WebGPU State Invalidation**: Per WebGPU specification, invoking `executeBundles` resets all pipeline state, vertex buffer bindings, index buffer bindings, and bind groups on the current render pass encoder to an empty/unbound state.
3. **Rebind Requirement (No Warm-State Assumption)**: Any direct draw that immediately follows a render bundle execution within the same render pass CANNOT assume warm state (`assumes_warm_state: true`). It must explicitly rebind its pipeline and bindings (`rebind_required: true` / `assumes_warm_state: false`). Violations are rejected during pass validation with `HazardError::BundleDirectDrawRequiresRebind { pass_id, draw_id }`.
4. **Subsequent Draw State Propagation**: Once a direct draw following a bundle has rebinded, subsequent direct draws may inherit that restored state until another bundle execution is encountered.

#### Consumer Guide for ChartreuseFern (`f3d-runtime/src/gpu_host.rs`):
When lowering a `PlanSegment` of kind `PassKind::Render` into `GpuCommand` entries:
- Iterate through `segment.draws()`.
- For each `draw`:
  - If `draw.is_bundle()` (or `if let Some(bundle_id) = draw.bundle_id()`):
    Emit `GpuCommand::ExecuteBundles { bundle_id }` (or corresponding packet opcode `OPCODE_EXECUTE_BUNDLES`).
  - If `draw.is_direct()`:
    If `segment.draw_crosses_bundle_boundary(draw_index)` or `draw.rebind_required()`:
    Ensure pipeline and bind groups are recorded fresh rather than skipped under any driver-level state cache.

---

## 12. No-Claim Boundaries

> **NO-CLAIM: static graph correctness is not browser rendering, whole H1 support, or measured optimization.**
>
> Validating usage scopes, splitting passes, and generating an ordered execution plan does not imply that WebGPU commands have executed on a physical GPU, that browser pixels have been rendered, or that any performance acceleration has been measured. Acceleration claims require live benchmark execution on a fingerprint-identified physical device.
