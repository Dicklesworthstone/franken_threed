# Asupersync Browser Host Gap Analysis and Upstream Work Plan

> **Bead ID**: `f3d-02-asupersync-browser-execution-58j.1`  
> **Status**: In Progress / Deliverable Analysis  
> **Date**: 2026-09-09  
> **Author**: ChartreuseFern (Pane %50)  
> **Scope**: Target Asupersync browser host capabilities, version pinning, section 11.2 gap analysis, and upstream work plan charged against the 10,000-line allocation.

---

## 1. Fixed Revision Statement & Pinned Dependency Revision Set

### 1.1 Fixed Sibling Revision
- **Repository**: `/Users/jemanuel/dp/asupersync`
- **Pinned Commit**: `4232e302bfd971ffe4875dc44e29a31c32d7fa39`
- **Historical Anchor**: `f84ac21289e2bf00b842f22aa4b26ad452590660` (27 commits prior to `4232e30` measured via `git rev-list --count f84ac2128..4232e302b`; manifests and lockfiles are byte-for-byte identical across both commits).
- **Plan-Cited Commit Check**: Commit `efa5798d139ae3e877e13b2b8b5108fc5fe1a625` exists in git history (`git cat-file -t` returns `commit`); it is separated by 59 commits from `f84ac2128` and 86 commits from `4232e30`.

### 1.2 Toolchain Channel
- **File**: `rust-toolchain.toml:7`
- **Channel**: `channel = "nightly-2026-08-31"`
- **Components**: `["rustfmt", "clippy"]`

### 1.3 Binding Family & Serialization Versions (Cargo.lock Resolved)
All crates in FrankenThreeD interacting with the browser boundary or serialized manifests share this exact resolved version set from `/Users/jemanuel/dp/asupersync/Cargo.lock`:

| Crate | Declared (Workspace `Cargo.toml`) | Declared (`asupersync-browser-core/Cargo.toml`) | Resolved (`Cargo.lock`) | Lockfile Citation |
|---|---|---|---|---|
| `wasm-bindgen` | `"0.2"` (`Cargo.toml:568`) | `"0.2"` (`asupersync-browser-core/Cargo.toml:24`) | `0.2.126` | `Cargo.lock:4141-4142` |
| `wasm-bindgen-futures` | `"0.4"` (`Cargo.toml:569`) | `"0.4"` (`asupersync-browser-core/Cargo.toml:25`) | `0.4.76` | `Cargo.lock:4154-4155` |
| `js-sys` | `"0.3"` (`Cargo.toml:567`) | `"0.3"` (`asupersync-browser-core/Cargo.toml:26`) | `0.3.103` | `Cargo.lock:1926-1927` |
| `web-sys` | `version = "0.3"` (`Cargo.toml:570`) | `version = "0.3"` (`asupersync-browser-core/Cargo.toml:27`) | `0.3.103` | `Cargo.lock:4235-4236` |
| `serde` | `version = "1.0"` (`Cargo.toml:421`) | `version = "1"` (`asupersync-browser-core/Cargo.toml:48`) | `1.0.229` | `Cargo.lock:3138-3139` |
| `serde_json` | `"1.0"` (`Cargo.toml:422`) | `"1.0"` (`asupersync-browser-core/Cargo.toml:50`) | `1.0.151` | `Cargo.lock:3179-3180` |

### 1.4 Naga Audit
- **Lockfile Check**: `naga` does **not** appear anywhere in `/Users/jemanuel/dp/asupersync/Cargo.lock` (0 occurrences). It remains an isolated, build-time shader compiler for `f3d-shader` and is not imported by `asupersync`.

### 1.5 Single Wasm Instance Architectural Rule (Plan §11.1)
FrankenThreeD links Asupersync core + host integration directly into the single application Wasm module. It **must not** instantiate a separate `asupersync-browser-core` Wasm instance alongside a FrankenThreeD Wasm instance with disjoint linear memories and separate handle tables.

---

## 2. Gap Analysis Table: Plan Section 11.2 Foundation Requirements

Plan Section 11.2 mandates eight executable capabilities before building the rendering system. The table below maps each requirement to its concrete implementation state in `asupersync` at `4232e302b` with exact `file:line` evidence.

| # | Plan §11.2 Foundation Requirement | Status | What Exists at `4232e30` (Evidence Citations) | What Is Missing for Executable Rust Execution |
|---|---|---|---|---|
| **1** | **Rust futures actually being polled and producing observed results** | **MISSING** | • `BrowserRuntimeInner` (`src/runtime/builder.rs:1964-1969`), `BrowserRuntime` (`src/runtime/builder.rs:1977-2051`), and `BrowserRuntimeBuilder` (`src/runtime/builder.rs:2132-2216`) exist but expose **zero** methods accepting a `Future`, closure, or `Waker`.<br>• Fixture `tests/fixtures/rust-browser-consumer/crate/src/lib.rs:446-458` registers raw handles and pushes manually synthesized outcomes into the ledger without polling any future.<br>• `wasm_bindgen_futures::spawn_local` in `asupersync-browser-core/src/lib.rs:591-594` polls only an internal hardcoded fetch block.<br>• Canonical `RuntimeBuilder::build()` unconditionally fails closed with `ConfigError` on `target_arch = "wasm32"` (`src/runtime/builder.rs:3554-3566`).<br>• `Cx::spawn` on detached context fails closed with `SpawnError::RuntimeUnavailable` (`src/cx/cx.rs:4385-4390`). | Host services implementation to bootstrap a worker threadlessly on `wasm32-unknown-unknown`, wire wakers into the host event loop, and accept caller-supplied Rust futures (`LocalBoxFuture` / `Cx::spawn`). |
| **2** | **Timers and host-turn wakeups without native threads** | **PARTIAL** | • `BrowserMonotonicClock` exists with host sample ingestion and drift bounding (`src/time/driver.rs:111-145`).<br>• `BrowserReactor` exists and binds to `MessagePort` / `BroadcastChannel` (`src/runtime/reactor/browser.rs:91-104`, `105-110`).<br>• Required contract documented in `docs/wasm_browser_scheduler_semantics.md:20-28`. | Integration of `BrowserMonotonicClock` and `BrowserReactor` into a threadless `TimerDriverHandle`. Currently `DeadlineMonitorHostService` requires an OS thread (`src/runtime/builder.rs:287-290`). Threadless timer pump via JS `setTimeout` callbacks is missing. |
| **3** | **Nonreentrant scheduler pump and bounded microtask bursts** | **PARTIAL** | • `ThreeLaneWorker::run_once` exists to execute one task from the scheduler queues (`src/runtime/scheduler/three_lane.rs:6980-6994`).<br>• Nonreentrancy guard and microtask burst limit (default 32 steps) are normatively specified in `docs/wasm_browser_scheduler_semantics.md:104-124`, `200-240`. | Concrete Rust pump loop driving `ThreeLaneWorker::run_once` via `queueMicrotask`, enforcing `ReentrancyGuard`, and performing macro-task yield (`MessageChannel` / `setTimeout(0)`) when burst limit is reached. |
| **4** | **Cancellation delivered to running cooperative tasks** | **PARTIAL** | • Cancellation queue lane in `ThreeLaneWorker` (`src/runtime/scheduler/three_lane.rs:6997-7007`).<br>• Core cancellation tokens and checkpointing in `src/cx/` and `src/sync/cancellation.rs`.<br>• `BrowserRuntime` supports `WasmAbortPropagationMode` (`src/runtime/builder.rs:2151`), but only bridges JS `AbortController` to ledger handles. | Host event-loop delivery of cancellation into running Rust tasks, because workers are never started on browser `wasm32`. Once worker pump exists, cancellation lane will drain cooperative checkpoints. |
| **5** | **Real fetch cancellation and eventual cleanup** | **PARTIAL** | • WASM-bindgen boundary implementation is complete: `spawn_browser_fetch` wires `AbortController` (`asupersync-browser-core/src/lib.rs:580-596`).<br>• Cancellation calls `controller.abort()` (`asupersync-browser-core/src/lib.rs:944-947`).<br>• Cleans up inflight table `INFLIGHT_FETCHES` (`asupersync-browser-core/src/lib.rs:390-399`, `511-514`).<br>• Detects `AbortError` and maps to cancelled outcome (`asupersync-browser-core/src/lib.rs:568-570`). | Exposed only via JSON-string ABI calls (`fetch_request_impl`, `task_cancel_impl`), not as a typed native Rust async API (`AsyncRead` / `Response` stream) callable from `Cx` inside the application runtime. |
| **6** | **Child tasks drained before scene-region teardown** | **PARTIAL** | • Region hierarchy and scope tables exist (`src/runtime/region_table.rs:1-450`).<br>• `BrowserRuntime::close_scope` and `BrowserRuntime::close` drain child handles from the ledger (`src/runtime/builder.rs:2034-2050`).<br>• Native runtime has region close-to-quiescence progression (`close_region_command` at `src/runtime/state.rs:4317-4347`, `advance_region_state` at `src/runtime/state.rs:9152-9220`). | In browser context, drain only updates synchronous ledger handles; it does not await real async child task futures to quiescent completion before region destruction. |
| **7** | **No late publication into a cancelled/replaced scene** | **PARTIAL** | • `asupersync` core provides capability tokens and cancellation tokens (`src/cx/cx.rs:4380-4395`, `src/sync/cancellation.rs:1-120`).<br>• Dispatcher rejects task operations on closed scopes (`require_active_runtime_or_region_handle` at `src/types/wasm_abi.rs:1880-1908`, `scope_close` handle tree release at `src/types/wasm_abi.rs:1866-1878`, `2012-2041`, invoked via `BrowserRuntime::close_scope` at `src/runtime/builder.rs:2034-2042`). | F3D-specific `SceneEpoch` / `GenerationId` gating and GPU commit boundary protection (separating staging staging, `writeBuffer`, and `queue.submit()` per Plan §11.4) must be integrated into `f3d-runtime` over Asupersync capability tokens. |
| **8** | **Explicit behavior for unsupported host contexts** | **PRESENT** | • `BrowserRuntimeBuilder::inspect_execution_ladder()` produces truthful diagnostics (`src/runtime/builder.rs:2185-2190`).<br>• `BrowserRuntimeBuilder::build()` returns structured `BrowserRuntimeBuildError::Unsupported` with execution ladder (`src/runtime/builder.rs:2210-2213`).<br>• `RuntimeBuilder::build()` on `wasm32` fails closed with explicit message citing missing contract (`src/runtime/builder.rs:502-513`, `3554-3566`). | None. Host capability probing and structured fail-closed error reporting are fully implemented and verified. |

---

## 3. Upstream Work Plan (Charged to 10,000-Line Foundation Allocation)

Per Plan §11.2 and §13.2, all missing host-services infrastructure must be implemented upstream in `asupersync` rather than forking runtime semantics inside `franken_threed`.

### Work Item 1: Browser Host Services & Threadless Worker Bootstrap
- **Target Module**: `asupersync::runtime::builder` (`src/runtime/builder.rs`) and new `src/runtime/host/browser.rs`.
- **Semantic Contract**: Implement the `RuntimeHostServices` trait (`src/runtime/builder.rs:317-340`) for `wasm32-unknown-unknown` satisfying `BrowserHostServicesContract::V1` (`src/runtime/builder.rs:259-284`). Provide single-worker initialization without invoking `std::thread::spawn` or thread-joining primitives.
- **Proof Test**: Headless browser / `wasm-bindgen-test`: `RuntimeBuilder::new().current_thread().build()` returns `Ok(Runtime)` on `wasm32` instead of failing with `ConfigError`.
- **Line Estimate**: **1,500 lines** (implementation + unit/contract tests).

### Work Item 2: Nonreentrant Event-Loop Pump & Microtask Burst Driver
- **Target Module**: `asupersync::runtime::scheduler` (`src/runtime/scheduler/browser_pump.rs`).
- **Semantic Contract**: Implement the normative contract from `docs/wasm_browser_scheduler_semantics.md:104-124`. Step the single worker's `ThreeLaneWorker::run_once` inside a microtask pump triggered via `queueMicrotask`. Impose a `ReentrancyGuard` to prevent re-entrant execution during host callbacks. Limit microtask drain to `max_microtask_burst` (32 steps); yield to host macro-task queue (`MessageChannel` / `setTimeout(0)`) upon burst exhaustion.
- **Proof Test**: Schedule 100 CPU tasks; assert microtask burst caps at 32, host macro-task yield occurs, and browser animation/event loop remains unblocked.
- **Line Estimate**: **2,000 lines** (pump engine, reentrancy guards, yield hooks + integration tests).

### Work Item 3: Threadless Timer Driver & Host-Turn Wakeups
- **Target Module**: `asupersync::time::driver` (`src/time/driver.rs`) and `src/runtime/reactor/browser.rs`.
- **Semantic Contract**: Wire `BrowserMonotonicClock` (`src/time/driver.rs:111-145`) and `BrowserReactor` (`src/runtime/reactor/browser.rs:91-104`) to drive timers without threads. Replace native `DeadlineMonitorHostService` thread with host `setTimeout` wakeups scheduled for the earliest pending deadline in the timer wheel.
- **Proof Test**: `asupersync::time::sleep(Duration::from_millis(50)).await` resolves accurately in the browser via host timers without background threads.
- **Line Estimate**: **1,800 lines** (timer driver integration, reactor callbacks, deadline wheel dispatch + tests).

### Work Item 4: Wasm Future Adapter & Task Spawning Seam
- **Target Module**: `asupersync::runtime` (`src/runtime/wasm_spawn.rs`) and `src/cx/cx.rs`.
- **Semantic Contract**: Provide a browser-capable `spawn` path on `Cx` and `Runtime` that accepts `F: Future<Output = T> + 'static` (without requiring `Send` in single-threaded browser mode). Wire the task's `Waker` to schedule a pump step via `queueMicrotask`.
- **Proof Test**: Spawn an async block performing sequential `.await` yields; verify task runs to completion and returns the expected value.
- **Line Estimate**: **1,500 lines** (local task queueing, single-threaded waker bridge, `Cx` local spawn API + tests).

### Work Item 5: Region Quiescence & Asynchronous Child Task Drain
- **Target Module**: `asupersync::runtime::region_table` (`src/runtime/region_table.rs`) and `src/cx/child_region.rs`.
- **Semantic Contract**: Implement asynchronous region drain: `region.close_and_drain().await`. Transitions region to closing, propagates cancellation to child tasks across scheduler queues, pumps the event loop, and completes only when all child tasks reach terminal states (`Completed`, `Cancelled`, or `Panicked`).
- **Proof Test**: Spawn slow asynchronous child tasks in a sub-region; trigger `close_and_drain().await`; verify teardown blocks until all tasks reach terminal state and late spawns fail.
- **Line Estimate**: **1,200 lines** (drain future, cancellation propagation, quiescence barrier + tests).

### Work Item 6: Typed Asynchronous Fetch Stream Adapter
- **Target Module**: `asupersync-browser-core::fetch` (`asupersync-browser-core/src/fetch.rs`) and `asupersync::net::fetch`.
- **Semantic Contract**: Wrap the existing `AbortController`-based fetch boundary (`asupersync-browser-core/src/lib.rs:580-596`) into a typed Rust async function: `fetch(cx: &Cx, request: Request) -> Result<Response, FetchError>`. Bind cancellation to `Cx` cancellation token and expose response body as an `AsyncRead` stream.
- **Proof Test**: Trigger fetch, cancel task via `CancelToken`, assert `AbortSignal.aborted == true`, and verify immediate cleanup in `INFLIGHT_FETCHES`.
- **Line Estimate**: **1,000 lines** (stream wrapper, cancellation binding, error mapping + tests).

---

## 4. Allocation Envelope Accounting

| Work Item | Module | Purpose | Estimated Lines (Code + Tests) |
|---|---|---|:---:|
| **Item 1** | `asupersync::runtime::builder` | Browser Host Services & Threadless Bootstrap | 1,500 |
| **Item 2** | `asupersync::runtime::scheduler` | Nonreentrant Microtask Pump & Burst Limiting | 2,000 |
| **Item 3** | `asupersync::time::driver` | Threadless Timer Driver & Reactor Wakeups | 1,800 |
| **Item 4** | `asupersync::runtime` / `cx` | Wasm Future Spawning & Waker Adapter | 1,500 |
| **Item 5** | `asupersync::runtime::region` | Asynchronous Region Quiescence & Drain | 1,200 |
| **Item 6** | `asupersync-browser-core` | Typed Async Fetch Stream Adapter | 1,000 |
| **Total Estimated Upstream Work** | | | **9,000** |
| **Foundation Allocation Envelope** | | | **10,000** |
| **Contingency Margin Remaining** | | | **+1,000** |

### Envelope Verdict
- **Flag**: **WITHIN ALLOCATION** (9,000 lines <= 10,000 lines).
- **Status**: The required upstream work fits within the 10,000-line foundation budget without requiring project reserve allocation or scope expansion.

---

## 5. Exported Host Hooks Required from JavaScript

To support the single Wasm instance architecture without OS threads, the host environment must supply the following standard Web platform hooks to the Asupersync runtime:

1. **Microtask Pump Trigger**: `globalThis.queueMicrotask` for low-latency scheduler stepping.
2. **Macro-Task Yield Trigger**: `MessageChannel` (`port.postMessage(null)`) with fallback to `setTimeout(fn, 0)` for yielding when microtask burst limit (32) is reached.
3. **Monotonic Clock**: `performance.now()` for monotonic host time sampling into `BrowserMonotonicClock`.
4. **Deadline Timers**: `setTimeout` / `clearTimeout` for scheduling deadline wakeups into the threadless timer driver.
5. **Fetch Cancellation**: `AbortController` / `AbortSignal` for binding network cancellations to Asupersync task lifecycles.
