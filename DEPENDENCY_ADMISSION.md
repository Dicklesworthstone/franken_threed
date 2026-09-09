# DEPENDENCY_ADMISSION.md - Candidate Dependency Allowlist & Admission Records

> Candidate dependency admission ledger for FrankenThreeD per AGENTS.md "Dependency Policy" and Plan §13.4.
>
> Invariant: Forbidden in the execution core: Bevy, a general `wgpu` stack, a new generic ECS framework,
> **Tokio**, a mandatory Rayon pool, native C/C++ FFI, BLAS, or a Python interpreter.
> Every candidate dependency must have a verified, unembellished record below.

---

## 1. Candidate Dependencies Status Table

| Crate | Candidate Pinned Version / Commit | Used API (Planned) | Native Compile Result | wasm32 Compile Result | Status / Verification Notes |
|---|---|---|---|---|---|
| `serde` | `=1.0.229` (`default-features = false`, `features = ["alloc", "derive"]`) | Serialization/deserialization for manifests and reports | PASS (remote check on `hz4` via RCH) | Unmeasured (pending verification) | Candidate admitted for offline metadata only; not in hot frame loop |
| `serde_json` | `=1.0.151` (`default-features = false`, `features = ["alloc"]`) | JSON format serialization for capability dumps and reports | PASS (remote check on `hz4` via RCH) | Unmeasured (pending verification) | Candidate admitted for offline metadata only; not in hot frame loop |
| `asupersync` | Plan citation: `efa5798d139ae3e877e13b2b8b5108fc5fe1a625`<br>Sibling HEAD: `f84ac21289e2bf00b842f22aa4b26ad452590660` | Structured concurrency regions, cooperative task polling, timer wakeups, explicit cancellation | Not run | Not run | **No integration has been compiled.** Version divergence between plan citation and sibling repo HEAD must be resolved before workspace integration. |
| `wasm-bindgen` | Version to align with Asupersync host boundary | External browser boundary, JS exception handling, Wasm export plumbing | Not run | Not run | Unmeasured; exact revision to be pinned when Asupersync host integration is compiled |
| `js-sys` | Version to align with Asupersync host boundary | Direct JavaScript global object bindings (`Function`, `Uint8Array`, `Promise`) | Not run | Not run | Unmeasured; exact revision to be pinned when Asupersync host integration is compiled |
| `web-sys` | Version to align with Asupersync host boundary | WebGPU context, canvas acquisition, DOM event listener bindings | Not run | Not run | Unmeasured; exact revision to be pinned when Asupersync host integration is compiled |
| `naga` | Pinned revision TBD (normally build-time only) | WGSL validation, AST inspection, shader translation | Not run | Not run | Unmeasured; build-time dependency for shader translation pipeline (Plan §9.4) |
| Rollup / Acorn | Pinned via `tools/package-lock.json` | Parsing Three.js r186 ESM/CJS source, AST extraction, island analysis | Node.js host | N/A (Build-time JS) | Unmeasured; managed under `tools/` package; never bundled in deployed Wasm |

---

## 2. Dependency Admission & Audit Protocol

1. **No Faked or Fabricated Evidence**:
   - Compile results are recorded only after an actual verified build run on the target architecture.
   - Binary sizes (KiB) and memory allocation costs must be measured from real artifacts, never guessed or asserted without profiler output.
   - Only crates confirmed to exist in the upstream repository may be listed (no speculative sub-crates).
2. **Target Requirements**:
   - All runtime execution crates must compile cleanly on `wasm32-unknown-unknown` under the pinned dated nightly toolchain.
   - Pure safe Rust (`#![forbid(unsafe_code)]`) is mandatory for first-party semantic, numerical, compiler, scene, and resource crates.
   - The browser-binding layer (`wasm-bindgen`/`web-sys`) is a named external boundary strictly isolated to runtime host adapters.
3. **Asupersync Foundation Constraint**:
   - Asupersync is the sole async foundation. Two independently instantiated Asupersync versions or forked runtimes are forbidden.
   - Current status: Plan §13.4 / §23 cited `efa5798d139ae3e877e13b2b8b5108fc5fe1a625`, while sibling repository HEAD is `f84ac21289e2bf00b842f22aa4b26ad452590660`. Neither has been compiled within the FrankenThreeD workspace yet.

---

## 3. Strictly Forbidden Components

The following components are unconditionally forbidden from the execution core:
- **Tokio**: Incompatible with single-threaded browser cooperative execution; pulls in native thread pool.
- **Rayon**: Mandates native OS threads; cannot run in standard browser Wasm without shared-memory workers.
- **Bevy / Generic ECS Frameworks**: Unnecessary architectural abstraction violating the thin specializer contract.
- **Native Metal / Vulkan / D3D Direct FFI**: Bypasses WebGPU standard; breaks browser portability.
- **BLAS / LAPACK Native Libraries**: Breaks Wasm target compatibility.
- **Python Interpreter / Native Embeddings**: Not permitted in compiler or execution path.
