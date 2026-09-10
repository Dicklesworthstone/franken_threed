# CONTRACT.md: `f3d-runtime`

> Crate contract for `f3d-runtime` per AGENTS.md "Documentation and Contracts" and Plan §11, §17.1.

---

## 1. Purpose and Position in the Dependency Direction

`f3d-runtime` is the async lifecycle, host services, and Asupersync execution bridge crate for FrankenThreeD:

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

- **Dependencies**: `f3d-core`, `asupersync` (pinned revision), and bounded browser-binding packages (`wasm-bindgen`, `js-sys`, `web-sys`).
- **Dependents**: `f3d-compiler` (route analysis runtime target), `f3d-cli` (orchestration), `f3d-conformance` (verification).
- Browser bindings are strictly isolated to the named host boundary. No browser binding leaks into numerical or compiler crates.

---

## 2. Public Types and Semantics

| Type | Semantics |
|---|---|
| `BrowserRuntimeConfig` | Configuration for browser execution: single-worker topology, microtask burst limits, ready handoff limits, and trace retention. |
| `BrowserRuntimeCore` | Encapsulates the Asupersync runtime, exposing threadless single-worker startup, `spawn_local`, and `browser_pump`. |
| `ProbeEvent` | Structured telemetry event for task lifecycle transitions (`spawn`, `poll`, `wake`, `complete`) with source tagging (`timer`, `host_turn`, `microtask`) and wall timestamps. |
| `ProbeSummary` | Aggregate execution verification summary per probe. |
| `ProbeKind` | Identifies the 5 required Phase 0 validation probes: `TimerSequence`, `ChannelJoinChain`, `HostTurnYield`, `ReentrancyGuard`, `BurstLimit`. |
| `BrowserExecutionHarness` | Orchestrator executing the 5 verification probes, validating assertions, and emitting JSON lines evidence. |

---

## 3. Invariants

1. **Sole Async Foundation**: Asupersync is the only async programming foundation. No Tokio, no Rayon, no unstructured promise pool.
2. **Single Wasm Instance**: All Asupersync core and host services link into one Wasm instance. No dual-memory or split runtime instantiation.
3. **No Native Threads in Core**: Browser execution runs strictly on a threadless single worker; `SharedArrayBuffer` and `Atomics.wait` are not required.
4. **Non-Reentrant Pump**: `BrowserWorkerPump` must never re-enter when a host callback or running task synchronously invokes the pump; re-entrancy returns `PumpDrainOutcome::ReentrantPrevented`.
5. **Bounded Microtask Burst**: The pump must drain at most the configured `microtask_burst_limit` before yielding to the browser macrotask event loop.
6. **No Hidden Synchronous Yields**: Originally synchronous application tasks cannot be split across browser turns.
7. **Structured Observable Logging**: Every probe emits structured JSON-lines events without casual stdout/stderr printing.

---

## 4. Implementation Ownership Route

- **Ownership route**: New Rust core + upstream Asupersync browser host services (`BrowserHostServices`).
- Retained components are limited to standard browser host APIs (event loop, `queueMicrotask`, `MessageChannel`, `setTimeout`).
- Does not use exact backend rendering components.

---

## 5. Error Model and Source-Error Preservation

- Errors during runtime initialization, task admission, and probe execution return structured error types (`RuntimeError`, `ProbeError`).
- Cancellation delivers `JoinError::Cancelled` cleanly to joined handles without panicking.
- Host communication errors are captured with typed error codes.

---

## 6. Determinism and Numeric-Fidelity Class

- Deterministic task scheduling order for ready tasks within priority bands.
- Timestamps recorded as discrete monotonic millisecond / microsecond offsets from probe inception.

---

## 7. Cancellation Behavior

- Cancellation is cooperative and prompt: cancelled local tasks transition to terminal cancel lanes.
- Draining before teardown ensures child tasks are cleaned up before regions retire.
- No late publication: dropped handles or cancelled regions reject subsequent task completion writes.

---

## 8. Unsafe Boundary

- **`#![forbid(unsafe_code)]`** is declared at the crate root.
- The browser binding boundary (`wasm-bindgen`) is an audited external FFI boundary. All first-party runtime code is pure safe Rust.

---

## 9. Feature Flags

- `default = ["std", "serde"]`
- `std`: Enables standard library integration and formatting.
- `serde`: Enables JSON-lines serialization for probe event logs and summaries.
- `browser`: Enables `wasm-bindgen`, `js-sys`, and `web-sys` host bindings for `wasm32-unknown-unknown`.

---

## 10. Conformance Tests

- `tests/probe_unit_tests.rs`: Tests pump non-reentrancy state machine, burst bound enforcement, timer sequence ordering, and channel/join chains using safe test mocks.
- `tests/fixtures/browser_execution/`: Checked-in Wasm browser test page (`index.html`), static shim (`shim.js`), and test harness (`test_harness.mjs`) for installed Safari and Chrome execution.

---

## 11. No-Claim Boundaries

- `f3d-runtime` implements the async execution pump and validation probes; it does **not** claim rendering acceleration, WebGPU scene compilation, or full application equivalence on its own.
- Passing unit or fixture tests does not constitute hardware benchmark evidence or 3× acceleration claims.
