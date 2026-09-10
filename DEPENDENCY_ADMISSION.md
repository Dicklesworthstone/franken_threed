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
| `serde` | `=1.0.229` (`default-features = false`, `features = ["alloc", "derive"]`) | Serialization/deserialization for manifests and reports | PASS (remote check on `hz4` via RCH) | PASS (remote check on `hz4` at 2026-09-09T22:31:52Z via RCH, exit 0, `RCH_REQUIRE_REMOTE=1`) | Candidate admitted for offline metadata only; not in hot frame loop |
| `serde_json` | `=1.0.151` (`default-features = false`, `features = ["alloc"]`) | JSON format serialization for capability dumps and reports | PASS (remote check on `hz4` via RCH) | PASS (remote check on `hz4` at 2026-09-09T22:31:52Z via RCH, exit 0, `RCH_REQUIRE_REMOTE=1`) | Candidate admitted for offline metadata only; not in hot frame loop |
| `asupersync` | Path dependency at `/Users/jemanuel/dp/asupersync` (`default-features = false`); sibling HEAD `5008e16fb` (merge of origin/main over `869483616`, `e8eb45baf`, `096f5a433`, which committed the former 16-file browser-host working tree: E0277 local spawn aliases, typed browser_host_services, browser pump edits, net/tcp cfg allows, wasm32 clock work, three_lane admission and drain fix); working tree clean; development absolute path, not yet a portable immutable git dependency | Structured concurrency regions, cooperative task polling, timer wakeups, explicit cancellation | PASS (strict RCH hz2: `cargo check -p asupersync --lib` at 5008e16fb 18:11Z exit 0 `Checking asupersync v0.4.11 (/Users/jemanuel/dp/asupersync)`; f3d-runtime `probe_unit_tests` 2/2 at 942c9f2 18:33Z exit 0) | PASS (strict RCH hz2 `cargo build --locked -p f3d-runtime --features browser --target wasm32-unknown-unknown` at 942c9f2 18:23Z exit 0, log `rch_f3d_runtime_wasm_build_942c9f2.log` shows `Compiling asupersync v0.4.11 (/Users/jemanuel/dp/asupersync)`) | Direct path dependency of crates/f3d-runtime. Browser execution of the packaged 942c9f2 Wasm: Chrome 02.2/2026-09-10T18-42-26.068Z-55298 and Safari 02.2/2026-09-10T18-42-39.922Z-56996, nine probes pass, Rust per-pump-turn maximum 4 in both. |
| `wasm-bindgen` | `=0.2.126` (direct optional dependency in `crates/f3d-runtime` under `browser` feature) | External browser boundary, JS exception handling, Wasm export plumbing | N/A (wasm32 browser feature) | PASS on hz2 via strict RCH (RCH_REQUIRE_REMOTE=1), log rch_f3d_runtime_wasm_build13.log at 09:27:17Z exit 0 | Direct optional dependency of crates/f3d-runtime under browser feature with exact pin =0.2.126. wasm32 compile verified. |
| `js-sys` | `=0.3.103` (direct optional dependency in `crates/f3d-runtime` under `browser` feature) | Direct JavaScript global object bindings (`Function`, `Uint8Array`, `Promise`) | N/A (wasm32 browser feature) | PASS on hz2 via strict RCH (RCH_REQUIRE_REMOTE=1), log rch_f3d_runtime_wasm_build13.log at 09:27:17Z exit 0 | Direct optional dependency of crates/f3d-runtime under browser feature with exact pin =0.3.103. wasm32 compile verified. |
| `web-sys` | `=0.3.103` (direct optional dependency in `crates/f3d-runtime` under `browser` feature; features: `AbortController`, `AbortSignal`, `ReadableStream`, `ReadableStreamDefaultReader`, `RequestInit`, `Response`, `Window`) | WebGPU context, canvas acquisition, DOM event listener bindings | N/A (wasm32 browser feature) | Compiled on hz2 in build14 at 09:45Z (Compiling web-sys v0.3.103), consumer verification pending | Direct optional dependency of crates/f3d-runtime under browser feature with exact pin =0.3.103 and listed features. Compiled on wasm32; consumer verification pending (build failed later in f3d-runtime). |
| `wasm-bindgen-futures` | `=0.4.76` (direct optional dependency in `crates/f3d-runtime` under `browser` feature) | Bridge between Rust futures and JavaScript promises | N/A (wasm32 browser feature) | Compiled on hz2 in build14 at 09:45Z (Compiling wasm-bindgen-futures v0.4.76), consumer verification pending | Direct optional dependency of crates/f3d-runtime under browser feature with exact pin =0.4.76. Compiled on wasm32; consumer verification pending (build failed later in f3d-runtime). |
| `naga` | None claimed (absent from asupersync lockfile; version deferred to F3D-15) | WGSL validation, AST inspection, shader translation | Not run | Not run | Unmeasured; build-time dependency for shader translation pipeline (Plan §9.4); deferred to F3D-15 |
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
   - Current status: Selected workspace revision is `4232e302bfd971ffe4875dc44e29a31c32d7fa39` (controller default 2026-09-09T22:48Z; manifests identical to `f84ac21`; plan cited `efa5798d139ae3e877e13b2b8b5108fc5fe1a625`). Nothing has been compiled in this workspace yet.

---

## 3. Strictly Forbidden Components

The following components are unconditionally forbidden from the execution core:
- **Tokio**: Incompatible with single-threaded browser cooperative execution; pulls in native thread pool.
- **Rayon**: Mandates native OS threads; cannot run in standard browser Wasm without shared-memory workers.
- **Bevy / Generic ECS Frameworks**: Unnecessary architectural abstraction violating the thin specializer contract.
- **Native Metal / Vulkan / D3D Direct FFI**: Bypasses WebGPU standard; breaks browser portability.
- **BLAS / LAPACK Native Libraries**: Breaks Wasm target compatibility.
- **Python Interpreter / Native Embeddings**: Not permitted in compiler or execution path.
