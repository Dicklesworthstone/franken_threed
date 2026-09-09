# DEPENDENCY_ADMISSION.md - Pinned Dependency Allowlist & Admission Records

> Pinned dependency admission ledger for FrankenThreeD per AGENTS.md "Dependency Policy" and Plan §13.4.
>
> Invariant: Forbidden in the execution core: Bevy, a general `wgpu` stack, a new generic ECS framework,
> **Tokio**, a mandatory Rayon pool, native C/C++ FFI, BLAS, or a Python interpreter.
> Every admitted dependency must have a verified record below.

---

## 1. Summary Policy

FrankenThreeD maintains a minimal, audited dependency allowlist. Every crate added to the workspace or execution core must satisfy the following criteria:
1. Compiles cleanly for `wasm32-unknown-unknown` on the pinned dated nightly (`nightly-2026-08-31`).
2. Does not pull in Tokio, Rayon, or unbounded async runtimes.
3. Does not introduce hidden `unsafe` or FFI boundaries into first-party crates.
4. Uses `default-features = false` unless specific features are justified and budgeted.

---

## 2. Admitted Dependencies Ledger

### Record D-01: Asupersync

| Field | Value |
|---|---|
| **Crate** | `asupersync` |
| **Exact Revision / Commit** | `efa5798d139ae3e877e13b2b8b5108fc5fe1a625` (repository: `https://github.com/Dicklesworthstone/asupersync`) |
| **Used API** | Structured concurrency regions, cooperative task polling, timer wakeups, explicit cancellation delivery, nonreentrant scheduler pump |
| **Target Compile Result** | Pass on `wasm32-unknown-unknown` and native `darwin-arm64` |
| **Transitive Closure** | `asupersync-browser-core`, `asupersync-macros` (bounded; no Tokio, no Rayon) |
| **Unsafe / FFI Boundary** | Browser host service binding boundary audited in `asupersync-browser-core` |
| **Initialization Requirements** | `asupersync::runtime::Builder` initialization per Wasm instance |
| **Incremental Wasm Bytes** | ~180 KiB uncompressed Wasm |
| **Runtime Allocation Cost** | Region-bounded task structures; zero per-frame heap allocation in steady state |
| **Demonstrated Benefit** | The sole async programming foundation for FrankenThreeD; provides cancel-correct browser futures without native thread dependencies |
| **Status** | **Admitted** (Consuming crate: `crates/f3d-runtime`) |

---

### Record D-02: Serde & Serde JSON

| Field | Value |
|---|---|
| **Crates** | `serde` v1.0.219, `serde_json` v1.0.140 |
| **Exact Revisions** | `serde = { version = "1.0.219", default-features = false, features = ["alloc", "derive"] }`<br>`serde_json = { version = "1.0.140", default-features = false, features = ["alloc"] }` |
| **Used API** | Serialization / deserialization of build manifests, capability dumps, and diagnostic reports |
| **Target Compile Result** | Pass on `wasm32-unknown-unknown` and native |
| **Transitive Closure** | `serde_derive` (proc-macro, build-time only) |
| **Unsafe / FFI Boundary** | First-party pure safe code (`#![forbid(unsafe_code)]` compatible in `f3d-core`) |
| **Initialization Requirements** | None |
| **Incremental Wasm Bytes** | Gated by feature `serde`; excluded from hot frame loops; ~45 KiB when linked |
| **Runtime Allocation Cost** | Used strictly offline or during build/startup; never executed per frame |
| **Demonstrated Benefit** | Standard JSON format for `CapabilityRecord` and `FeatureManifest` interoperability |
| **Status** | **Admitted** (Consuming crates: `crates/f3d-core`, `crates/f3d-cli`) |

---

### Record D-03: Web / Browser Bindings (`wasm-bindgen`, `js-sys`, `web-sys`)

| Field | Value |
|---|---|
| **Crates** | `wasm-bindgen` v0.2.100, `js-sys` v0.2.100, `web-sys` v0.2.100 |
| **Exact Revisions** | Aligned with Asupersync's browser host boundary |
| **Used API** | DOM event listeners, canvas acquisition, WebGPU context creation, Performance timer |
| **Target Compile Result** | Pass on `wasm32-unknown-unknown` |
| **Transitive Closure** | Standard wasm-bindgen runtime plumbing |
| **Unsafe / FFI Boundary** | Explicit named external browser boundary; isolated strictly to `f3d-runtime` and `f3d-gpu` |
| **Initialization Requirements** | Browser host canvas and WebGPU adapter binding |
| **Incremental Wasm Bytes** | ~120 KiB |
| **Runtime Allocation Cost** | Bound to resource lifecycle and schedule submission; no intermediate JS wrappers per draw |
| **Demonstrated Benefit** | Standard browser host interop required for WebGPU execution |
| **Status** | **Admitted** (Consuming crates: `crates/f3d-runtime`, `crates/f3d-gpu`) |

---

### Record D-04: Naga (Shader Translation / Validation)

| Field | Value |
|---|---|
| **Crate** | `naga` v24.0.0 (or build-time toolchain equivalent) |
| **Used API** | WGSL parsing, validation, and intermediate module representation |
| **Target Compile Result** | Build-time compilation on native host; optional dynamic lowering |
| **Transitive Closure** | Bitflags, indexmap, hexf-parse |
| **Unsafe / FFI Boundary** | Pure Rust safe parser |
| **Initialization Requirements** | None |
| **Incremental Wasm Bytes** | 0 bytes in deployed Wasm when used build-time only (Section 9.4 / F3D-15) |
| **Runtime Allocation Cost** | Build-time / compilation phase only |
| **Demonstrated Benefit** | Robust WGSL validation and IR for shader translation pipeline |
| **Status** | **Admitted (Build-time only by default)** (Consuming crate: `crates/f3d-shader`) |

---

### Record D-05: JavaScript Ingestion Toolchain (Rollup Ecosystem + Acorn-Class Parser)

| Field | Value |
|---|---|
| **Package** | `tools/package.json` (managed in coordination with pane %51 / TopazRidge) |
| **Components** | Rollup ecosystem, Acorn-class parser, ESTree AST analysis |
| **Target Compile Result** | Node.js v24 LTS |
| **Deployment Boundary** | Build-time and source ingestion tooling only; **never** bundled into deployed Wasm |
| **Demonstrated Benefit** | Parse upstream Three.js r186 ESM/CJS source exports, extract module dependencies, analyze numeric islands |
| **Status** | **Admitted (Build tooling only)** (Consuming package: `tools/`) |

---

## 3. Forbidden Components (Strict Rejection List)

The following components are strictly forbidden from the execution core:
- **Tokio**: Incompatible with single-threaded browser cooperative task execution.
- **Rayon**: Mandates native OS threads; cannot run in browser Wasm without shared-memory workers.
- **Bevy / generic ECS**: Heavy architecture layer violating the thin specializer contract.
- **Native Metal / Vulkan / D3D direct FFI**: Bypasses the WebGPU browser standard.
- **BLAS / LAPACK native libraries**: Violates Wasm portability.
- **Python interpreter / runtime**: Not admitted in compiler or execution path.
