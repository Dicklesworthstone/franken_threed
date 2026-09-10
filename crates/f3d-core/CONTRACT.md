# CONTRACT.md: `f3d-core`

> Crate contract for `f3d-core` per AGENTS.md "Documentation and Contracts".

---

## 1. Purpose and Position in the Dependency Direction

`f3d-core` forms the foundational root of the FrankenThreeD workspace:

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

`f3d-core` has zero dependencies on any sibling FrankenThreeD crate, zero dependencies on browser bindings (`wasm-bindgen`, `web-sys`), and zero dependencies on native host or build tooling. It is strictly `#![no_std]` capable with `alloc`.

---

## 2. Public Types and Semantics

| Type | Semantics |
|---|---|
| `Handle<D>` | Generational handle generic over identity domain `D`. Stores `index: u32` and `generation: NonZeroU32`. Encodes/decodes to explicit 32-bit words for safe ABI boundary passing. |
| `DeviceGeneration` | Non-zero generational counter representing active GPU device residency. |
| `GpuHandle<D>` | Pairs a `Handle<D>` with `DeviceGeneration` to invalidate GPU handles upon device loss or context re-acquisition. |
| `Arena<D, T>` | Generational slot storage with ABA collision prevention, explicit lifecycle transitions (`CpuExistent` → `GpuAllocated` → `Initialized` → `Submitted` → `Retired`), and counter-wrap retirement. |
| `LifecycleState` | Explicit state tracking for CPU vs GPU resource residency. |
| `CapabilityRecord` | Structured hardware, browser, and WebGPU limits/features snapshot. Unknown values are explicitly `Unknown`. |
| `FeatureManifest` | Single feature compatibility registry tracking Three.js r186 features and assigned execution routes. |
| `FeatureRoute` | Route classification: `SpecializedWebGpu`, `GeneralWebGpu`, `RetainedJs`, `ExactBackend`. |
| `FeatureStatus` | Verification status enum enforcing the blocking state rules. |
| `F3dError` | Non-printing structured error model preserving source spans and diagnostic codes. |

### 2.1 GPU Wire Layout Types and Constants

| Type / Constant | Semantics |
|---|---|
| `AffineRows` | 48-byte packed affine transform record (3 rows of `vec4<f32>`, 16-byte aligned). Stores linear basis in `xyz` and translation in `w`. Serializes to exact little-endian `f32` without unsafe memory casts. |
| `ProjectiveMat4` | 64-byte retained full 4x4 matrix record (4 columns of `vec4<f32>`, 16-byte aligned) preserving arbitrary perspective and projection components when matrices are non-affine. |
| `GpuMatrixLayout` | Enumerates target memory layouts: `AffineRows48` (48B), `ProjectiveMat4x4` (64B), and WGSL native `WgslMat4x3Padded64` (64B, 16-byte column alignment). |
| `VertexPosUv` | 20-byte canonical vertex record (`position: [f32; 3]` [12B] at offset 0, `uv: [f32; 2]` [8B] at offset 12, 4-byte aligned). Array stride is `VERTEX_POS_UV_STRIDE` (20 bytes). |
| `InstanceRecord` | 64-byte instance transform record (`transform: AffineRows` [48B], `instance_id: u32` [4B], 12B padding to 16B alignment). |
| `DrawIndirectArgs` | 16-byte WebGPU `drawIndirect` parameter record (`vertex_count`, `instance_count`, `first_vertex`, `first_instance`). |
| `DrawIndexedIndirectArgs` | 20-byte WebGPU `drawIndexedIndirect` parameter record (`index_count`, `instance_count`, `first_index`, `base_vertex`, `first_instance`). |
| `LayoutError` | Strictly typed layout errors: `BufferTooSmall`, `UnalignedOffset`, `NonAffineMatrix`, `IncompatibleTargetLayout`, `UnalignedBytesPerRow`, `UnalignedWriteBuffer`, `CalculationOverflow`. |
| `AFFINE_ROWS_BYTES` (48) | Size in bytes of a packed `AffineRows` record (alignment: 16). |
| `PROJECTIVE_MAT4_BYTES` (64) | Size in bytes of a full `ProjectiveMat4` record (alignment: 16). |
| `COLOR_UNIFORM_BYTES` (16) | Size in bytes of an RGBA color uniform record (`vec4<f32>`, alignment: 16). |
| `COLOR_UNIFORM_ALIGNMENT` (16) | Byte alignment of a color uniform record under WGSL rules. |
| `VERTEX_POS_UV_BYTES` (20) | Size in bytes of a canonical position + UV vertex record (`VERTEX_POS_UV_STRIDE` = 20, alignment: 4). |
| `WRITE_BUFFER_ALIGNMENT` (4) | WebGPU `writeBuffer` offset and size alignment constraint. |
| `COPY_BYTES_PER_ROW_ALIGNMENT` (256) | WebGPU texture copy row pitch alignment constraint. |
| `DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT` (256) | Default WebGPU dynamic uniform buffer offset alignment limit. |
| `DEFAULT_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT` (256) | Default WebGPU dynamic storage buffer offset alignment limit. |

### 2.2 Ownership and Single-Writer Epoch Types

| Type / Component | Semantics |
|---|---|
| `Author` | Authoritative writer identity across the host / runtime boundary: `Author::Js` (JavaScript host) or `Author::Wasm` (WebAssembly runtime). `Author::opposite()` returns the alternate author. |
| `OwnerMode` | Single-writer ownership mode: `Js`, `Wasm`, or `Mirrored { author: Author }`. Fallback default across all state regions is `OwnerMode::Js`. |
| `Epoch(u64)` | Monotonically increasing 64-bit publication counter. Decomposes into `(high_u32, low_u32)` words via `to_words()` / `from_words()` for safe host ABI transport without JavaScript `Number` precision loss. Advances via `checked_next()`, returning `OwnershipError::EpochOverflow`. |
| `DataVersion(u64)` | Monotonically increasing 64-bit dependency version counter. Decomposes into `(high_u32, low_u32)` words for host transport without float64 precision loss. Advances on recorded writes. |
| `RegionState` | Authoritative region state machine tracking `mode: OwnerMode`, `author: Author`, `current_epoch: Epoch`, `published_epoch: Epoch`, `current_version: DataVersion`, and `unpublished_writes: u64`. |
| `UseRecord<D>` | Immutable record representing a resource snapshot consumed by a pass or bridge schedule. Carries `{ resource: Handle<D>, version: DataVersion, epoch: Epoch, store_id: u64, generation: u64, slice_id: u32, byte_offset: u64, byte_length: u64 }`. |
| `SnapshotEntry<D, T>` | Stores a typed immutable snapshot paired with its `UseRecord<D>`. |
| `PerUseSnapshotStore<D, T>` | Non-clonable arena maintaining immutable typed snapshots. Slices are allocated at aligned byte offsets; lookups verify store identity and generation. |
| `PerUseByteBuffer<D>` | Authoritative, non-clonable linear byte buffer arena for packed uniform and data slices. Enforces alignment (e.g. 256-byte WebGPU dynamic uniform offset alignment) and pre-allocation arithmetic bounds. |
| `OwnershipError` | Non-panicking typed errors: `UnauthorizedWriter`, `UnpublishedWritesPending`, `StaleEpoch`, `SameAuthorTransfer`, `EpochOverflow`, `VersionOverflow`, `StoreIdOverflow`, `ImmutableSnapshotViolation`, `StaleSliceRecord`, `ForeignSliceRecord`, `SliceIdentityMismatch`, `SliceOutOfBounds`, `SliceNotFound`, `InvalidSliceLength`, `InvalidAlignment`, `Handle`. |

---

## 3. Invariants

1. **Generation checking**: Every handle lookup validates both slot bounds and slot generation. Stale handles return `HandleError::GenerationMismatch`.
2. **Device generation isolation**: GPU handles validate device generation. Stale handles from a previous GPU device return `HandleError::DeviceGenerationMismatch`.
3. **Counter-wrap retirement**: If a slot generation counter reaches wrap (`u32::MAX`), the slot is permanently retired rather than producing an ABA collision.
4. **ABI word splitting**: Handle transport across host boundaries uses two explicit 32-bit words (`to_words` / `from_words`), never a 64-bit float.
5. **No casual printing**: No stdout/stderr printing exists in `f3d-core`. All diagnostics are carried in structured error types.
6. **No-cut rule enforcement**: Feature manifests distinguish blocking states (`Unclassified`, `Unimplemented`, `Untested`, `KnownRegression`, `Stub`, `NoOpSubstitute`, `CandidateRefusalOnValidSource`) from non-blocking states (`Verified`, `Retained`, `HostBlocked`).

### 3.1 GPU Layout Invariants

1. **Exact structural affine rule with source precision eligibility**:
   - `is_matrix4_affine` and `is_matrix4_f64_affine` enforce exact structural equality on row 3: `e[3] == 0.0 && e[7] == 0.0 && e[11] == 0.0 && e[15] == 1.0`.
   - **Zero epsilon permitted**: small perspective coefficients (e.g. $e[3] = 5\times 10^{-7}$) create severe perspective divide errors on large coordinates (e.g. $x = 10^7 \implies w = 6.0$, a $6\times$ geometric error). Discarding them is strictly forbidden.
   - Matrices originating in `f64` must verify affine structure at full `f64` precision before narrowing to `f32`; any non-affine matrix must be retained in `ProjectiveMat4`.
2. **Operation-scoped alignment validation**:
   - Dynamic uniform buffer offsets must be multiples of `minUniformBufferOffsetAlignment` (default 256): `validate_dynamic_uniform_offset`.
   - Dynamic storage buffer offsets must be multiples of `minStorageBufferOffsetAlignment` (default 256): `validate_dynamic_storage_offset`.
   - Storage buffer array stride must be $\ge$ element size and aligned to element alignment (`validate_storage_array_stride`) or 16 bytes for composite records (`validate_composite_storage_array_stride`).
   - WGSL native `mat4x3` 64-byte padded layout is rejected as an incompatible target for 48-byte `AffineRows` writes (`validate_affine_target`).
   - WebGPU `writeBuffer` offsets and sizes must be 4-byte aligned: `validate_write_buffer_alignment`.
   - Texture copy row pitch must be 256-byte aligned: `validate_copy_bytes_per_row`.
3. **Checked readback row pitch**:
   - `aligned_bytes_per_row(width)` calculates `(width * 4)` rounded up to the next multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` (256) with checked arithmetic overflow guards at every step, returning `Err(LayoutError::CalculationOverflow)` on overflow.
   - `aligned_copy_bytes_per_row(width, bytes_per_pixel)` provides general pixel-width checked alignment.
4. **Canonical WGSL declarations**:
   - Canonical string constant `WGSL_AFFINE_ROWS_DECLARATION` provides the standard WGSL `AffineRows` struct, `transform_affine_point`, `transform_affine_vector`, and `affine_to_mat4x4`.
   - `WGSL_PROJECTIVE_MAT4_DECLARATION` provides the standard `ProjectiveMat4` struct.
   - `generate_wgsl_declarations()` synthesizes standard declarations for inclusion in shader compilation pipelines.

### 3.2 Ownership and State Transition Invariants

1. **Single Author Authority & Guarded Writes**:
   - Only the active `Author` may record writes via `RegionState::record_write(author)`. Calls by the non-authoritative environment fail with `OwnershipError::UnauthorizedWriter`.
2. **Mandatory Publication Before Transfer**:
   - Uncommitted writes (`unpublished_writes > 0`) must be explicitly published via `RegionState::publish(author, at_epoch)` before authority can be transferred or modes changed. Attempting transfer with pending writes fails with `OwnershipError::UnpublishedWritesPending`.
3. **Monotonic Epoch Progression**:
   - `publish(author, at_epoch)` advances `published_epoch` and `current_epoch` monotonically. Stale epochs fail with `OwnershipError::StaleEpoch`.
4. **Replay-Safe Authority Transfer (ABA Prevention)**:
   - `RegionState::transfer_authority(from, to, at_epoch)` requires zero pending writes and matching current epoch. It switches the active author and mode while **strictly advancing the publication epoch**. This prevents replaying stale transfer commands across roundtrips (e.g. JS → Wasm → JS).
5. **Guarded Mode Transitions**:
   - `RegionState::transition_mode(author, new_mode, at_epoch)` requires authorization from the active author, zero pending writes, matching epoch, and advances the publication epoch.
6. **Unique Store Identity**:
   - Every `PerUseSnapshotStore` and `PerUseByteBuffer` is assigned a globally unique `store_id: u64` allocated via checked CAS loop (`allocate_store_id()`). Overflow at `u64::MAX` cleanly returns `OwnershipError::StoreIdOverflow`. Stores are non-clonable to prevent duplicating allocation authority.
7. **Generation Invalidation (ABA Slice Recycling Guard)**:
   - Calling `reset()` on a buffer or snapshot store advances `generation: u64` monotonically. Slices looked up with a stale `UseRecord` from a previous generation fail with `OwnershipError::StaleSliceRecord`.
8. **Foreign Record Rejection**:
   - Passing a `UseRecord` allocated by Store A into Store B fails with `OwnershipError::ForeignSliceRecord`. Tampered record lengths or offsets fail with `OwnershipError::SliceIdentityMismatch` or bounds errors.
9. **Immutable Snapshots (Red-A / Blue-B Coexistence)**:
   - Two uses of the same logical resource mutated between passes (e.g. Red v1 in Pass A, Blue v2 in Pass B) retain distinct `DataVersion` values and distinct buffer offsets within the same submission. Recording a duplicate with identical metadata and bytes is idempotent; conflicting data for an existing version returns `OwnershipError::ImmutableSnapshotViolation`.
10. **Pre-Allocation Arithmetic Bounds**:
    - When appending slices or recording uses, `aligned_offset`, `slice_end = aligned_offset.checked_add(len)`, and the subsequent aligned next offset are strictly checked against `u64::MAX` before allocating or resizing memory, preventing arithmetic overflow and returning `OwnershipError::VersionOverflow`.

---

## 4. Implementation Ownership Route

- **Ownership route**: New Rust core.
- Does not execute retained JavaScript or exact backend components.
- Portable across native platforms and `wasm32-unknown-unknown`.

---

## 5. Error Model and Source-Error Preservation

- Errors are strictly typed via `F3dError`, `HandleError`, `LayoutError`, and `OwnershipError`.
- Compiler and specialization refusal errors capture optional `SourceSpan` (`file:line:col-line:col`).
- No panics in normal operations; all lookups return `Result<T, E>`.

---

## 6. Determinism and Numeric-Fidelity Class

- Deterministic state indexing: slots are allocated in monotonic order and re-used via explicit free lists.
- Byte-level determinism: all GPU wire records (`AffineRows`, `ProjectiveMat4`, `VertexPosUv`, `InstanceRecord`) serialize via explicit little-endian byte operations (`to_le_bytes`).
- Exact structural checks: affine structural eligibility is evaluated with exact float equality checks (`== 0.0`, `== 1.0`) with zero rounding tolerance.

---

## 7. Cancellation Behavior

- Operations in `f3d-core` are synchronous, bounded, and allocation-only.
- Arena removal cleans up values synchronously without lingering background operations.

---

## 8. Unsafe Boundary

- **`#![forbid(unsafe_code)]`** is declared at the crate root.
- No unsafe blocks or functions are permitted in `f3d-core`.

---

## 9. Feature Flags

- `default = ["std", "serde"]`
- `std`: Enables standard library integration (`core::error::Error`).
- `serde`: Enables serialization and deserialization for `CapabilityRecord`, `FeatureManifest`, `Handle`, and `F3dError`.

---

## 10. Conformance Tests

- Unit tests in `src/lib.rs` verify handle packing, unpacking, generation mismatch rejection, arena lifecycle transitions, GPU handle device validation, and manifest blocking logic.
- Integration tests in `tests/` test serde roundtrip, layout packing, alignment rules, ownership state transitions, immutable per-use versions, and workspace lints.

---

## 11. No-Claim Boundaries

- `f3d-core` provides types and data structures; it does NOT claim acceleration or WebGPU rendering capability on its own.
- **CPU layout tests are not GPU execution proof**: Native unit and integration tests for `f3d-core::layout` verify byte serialization, memory strides, and mathematical constraints on the host. They do **not** prove GPU shader execution, WebGPU driver compliance, pipeline creation, or acceleration; those claims belong strictly to `f3d-gpu`, `f3d-runtime`, and browser E2E suites.
- **Ownership state primitives are not an ECS or GPU proof**: State primitives in `f3d-core::ownership` provide deterministic, typed single-writer state machines and immutable per-use snapshot tracking across the host/core boundary. They do **not** claim to be a general-purpose ECS, full scene graph ownership analysis, or GPU execution proof.
- Registration of a feature in `FeatureManifest` does not imply implementation; status must be `Verified` under its declared route with matching conformance tests.
