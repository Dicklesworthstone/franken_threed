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

---

## 3. Invariants

1. **Generation checking**: Every handle lookup validates both slot bounds and slot generation. Stale handles return `HandleError::GenerationMismatch`.
2. **Device generation isolation**: GPU handles validate device generation. Stale handles from a previous GPU device return `HandleError::DeviceGenerationMismatch`.
3. **Counter-wrap retirement**: If a slot generation counter reaches wrap (`u32::MAX`), the slot is permanently retired rather than producing an ABA collision.
4. **ABI word splitting**: Handle transport across host boundaries uses two explicit 32-bit words (`to_words` / `from_words`), never a 64-bit float.
5. **No casual printing**: No stdout/stderr printing exists in `f3d-core`. All diagnostics are carried in structured error types.
6. **No-cut rule enforcement**: Feature manifests distinguish blocking states (`Unclassified`, `Unimplemented`, `Untested`, `KnownRegression`, `Stub`, `NoOpSubstitute`, `CandidateRefusalOnValidSource`) from non-blocking states (`Verified`, `Retained`, `HostBlocked`).

---

## 4. Implementation Ownership Route

- **Ownership route**: New Rust core.
- Does not execute retained JavaScript or exact backend components.
- Portable across native platforms and `wasm32-unknown-unknown`.

---

## 5. Error Model and Source-Error Preservation

- Errors are strictly typed via `F3dError` and `HandleError`.
- Compiler and specialization refusal errors capture optional `SourceSpan` (`file:line:col-line:col`).
- No panics in normal operations; all lookups return `Result<T, HandleError>`.

---

## 6. Determinism and Numeric-Fidelity Class

- Deterministic state indexing: slots are allocated in monotonic order and re-used via explicit free lists.
- No floating-point math is performed in `f3d-core`.

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
- Integration tests in `tests/` test serde roundtrip and workspace lints.

---

## 11. No-Claim Boundaries

- `f3d-core` provides types and data structures; it does NOT claim acceleration or WebGPU rendering capability on its own.
- Registration of a feature in `FeatureManifest` does not imply implementation; status must be `Verified` under its declared route with matching conformance tests.
