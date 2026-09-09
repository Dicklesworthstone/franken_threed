# AGENTS.md - FrankenThreeD

> Guidelines for AI coding agents working in this Three.js-compatibility,
> Rust/Wasm compiler, and WebGPU rendering codebase.

---

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU
MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE 0.1 - VALUE DELIVERY OVER PROCESS: NO PROCESS PORN

FrankenThreeD currently has **zero implementation**. There is a plan document
and nothing else. Every hour spent on meta-infrastructure is an hour not spent
on the Phase 0 foundation gate, which is the only thing that can tell us whether
this project should exist at all.

**Process is never the product unless the user explicitly asks for process
work.** Beads, plans, Agent Mail, audits, manifests, provenance, logging, CI,
test harnesses, dashboards, status reports, and agent coordination exist only to
support delivery. They must not become a self-perpetuating substitute for it.

### The value test

Before doing non-product work, answer all three questions:

1. What concrete capability, correctness defect, or immediate implementation
   blocker does this work address?
2. Is this the smallest direct action that addresses it?
3. Will its likely value exceed its implementation, maintenance, review, and
   delay cost right now?

If any answer is unclear, **do not do the work**. Return to implementing the
requested functionality. Speculative future usefulness, elegance, completeness,
or "more confidence" is not enough.

### Hard scope limits

- A review request authorizes reading the relevant code, reproducing concrete
  defects, making the smallest sound fixes, and adding focused regressions. It
  does **not** authorize redesigning adjacent infrastructure, inventing a new
  analyzer, exhaustively hardening hypothetical cases, or chasing unrelated
  pre-existing failures.
- Never turn a small review or repair into hundreds or thousands of lines of CI,
  harness, analyzer, schema, logging, provenance, or planning code without the
  user's explicit approval for that expansion.
- The feature manifest (Section 5.15 of the plan) is **one generated JSON data
  file inside existing tooling**. It is not a service, a policy framework, a
  dashboard, or a microservice. If you find yourself building infrastructure
  *around* the manifest rather than filling it in, stop.
- Do not build a validator for a validator, a harness for a harness, or an
  analyzer whose principal purpose is proving internal process artifacts unless
  that exact system is the requested deliverable.
- Do not chase a failure already present on `HEAD` unless it blocks the
  requested deliverable and the user authorizes broadening scope.
- Do not spawn an agent swarm for a narrow task. Delegate only concrete,
  independent implementation or bounded verification that materially shortens
  the path to the requested result.

### Implementation must dominate

Unless the user explicitly requests planning, governance, CI, or tooling:

- Spend the dominant share of effort on working product code and direct tests of
  that product code.
- Prefer an existing test seam over creating a new framework.
- Keep logging actionable and proportional.
- Treat Beads as concise execution bookkeeping. Once a Bead is clear enough to
  implement safely, implement it.
- Run the narrowest relevant checks while iterating. Run broad repository gates
  only when there is a coherent implementation ready for that proof.
- When choosing between missing Phase 0 functionality and optional
  meta-infrastructure, implement the Phase 0 functionality.

### Mandatory checkpoint and stop rule

At the first sign of scope expansion, pause and state in plain language: the
outcome being delivered; the files and approximate size of the proposed
expansion; why it is strictly necessary; and the smaller alternatives
considered. If that explanation cannot establish immediate net value in a few
sentences, **do not proceed**. Ask first.

When work has drifted into process porn, stop immediately. Do not add more
tests, proof layers, or cleanup to justify the sunk cost. Freeze the tree,
disclose the exact state candidly, and wait for direction.

### Concrete anti-pattern

Writing a 2,000-line compatibility-report schema, an evidence-provenance
pipeline, and a coverage dashboard before a single Rust future has been polled
in a browser is a canonical failure, not thoroughness. The plan's own first
session instruction is explicit: *"an honest measurement and a tiny working
application, not a forest of empty crates."*

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a
new file that you yourself created, such as a test file. You must always ask and
receive clear, written permission before deleting a file or folder of any kind.

---

## Irreversible Git & Filesystem Actions - DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`,
   `rm -rf`, or any command that can delete or overwrite code/data must never be
   run unless the user explicitly provides the exact command and states, in the
   same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might
   delete or overwrite, stop immediately and ask for specific approval.
3. **Safer alternatives first:** Use non-destructive inspection first:
   `git status`, `git diff`, backups, or explicit hand-written patches.
4. **Mandatory explicit plan:** Even after explicit authorization, restate the
   command verbatim, list exactly what will be affected, and wait for
   confirmation that your understanding is correct.
5. **Document the confirmation:** When running any approved destructive command,
   record the authorizing user text, the command actually run, and the execution
   time in your final response.

---

## Git Branch: ONLY Use `main`, NEVER `master`

The default branch is `main`.

- All work happens on `main`.
- Never create, switch to, or push feature branches unless the user explicitly
  overrides this file.
- Never reference `master` in code or docs. If you see it, treat it as a bug.

---

## RULE 2: NO GIT BRANCHES. NO GIT WORKTREES. EVER.

`main` is the one and only branch. There is no "temporary" branch, no per-agent
branch, no per-task branch, and no scratch worktree.

### FORBIDDEN

- `git branch <anything-other-than-main>`
- `git checkout -b <foo>` or `git switch -c <foo>`
- `git worktree add ...`
- Pushing non-main refs to `origin`
- Creating pull requests or draft PRs from feature branches
- Working in scratch clones at paths like `/tmp/franken_threed-*` or
  `~/projects/franken_threed-*` to isolate work
- Using any tool or harness that creates branches or worktrees as a side effect

### WHAT YOU DO INSTEAD

- Commit directly to `main` when the user asks for commits and the work is ready.
- Keep unfinished work in the working tree.
- Coordinate through Agent Mail reservations when multiple agents are active.
- Use Beads issue IDs and file reservations as the isolation mechanism.
- If another agent changed files, do not revert or stash their work.

---

## Project Truth Sources

This repository is currently **plan-first**. The authoritative design document
is:

- `COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENTHREED.md` (v2.0)

Read the relevant sections before broad design work. It defines the product
contract, compilation strategy, compatibility contract, execution architecture,
renderer design, testing program, performance gates, crate atlas, and phase
sequence. This `AGENTS.md` is the operating contract for agents; the plan is the
technical constitution.

Section numbers referenced throughout this file are the plan's sections. When
this file and the plan disagree on a technical rule, **the plan wins** and you
should flag the discrepancy.

---

## FrankenThreeD - This Project

FrankenThreeD is a **Three.js-compatible application specializer with a
Rust/Wasm + WebGPU execution core**.

Input: an ordinary Three.js project targeting the pinned release.
Output: a full-compatibility application package — self-contained HTML where the
code/resource graph closes, or a normal networked/module build where it does
not.

The central idea is **recovering information across ordinary library
boundaries**: which objects share a rendering program, which values actually
change, which changes can affect which passes, which computations have
observable intermediate results, and which object-oriented operations can become
one bulk operation. The compiled artifact contains the answer plus a small
invalidation condition, instead of rediscovering it 120 times per second.

The most important optimization is **removing a question from the frame loop** —
not replacing individual JavaScript functions with Rust functions.

### What this project is NOT

- Not a general JavaScript-to-Rust compiler.
- Not a new general-purpose game engine, editor, physics engine, or ECS.
- Not a `Scene.toJSON()` exporter.
- Not a native Metal/Vulkan/D3D backend and not a full `wgpu` stack.
- Not a WebGL emulation layer.
- Not a "fast supported subset" of Three.js.
- Not a renamed upstream renderer.

---

## THE FOUR PROMISES - NEVER CONFLATE THEM

This is the signature honesty rule of the project. Section 5.1 defines four
statuses that must be kept separate in code, tests, reports, commit messages,
and anything you say to the user:

| Promise | Meaning |
| --- | --- |
| **API compatibility** | Every pinned-release API contract has a working production path under its source-supported host prerequisites, whether or not upstream tests it |
| **Application equivalence** | The complete application feature/control surface is preserved |
| **Implementation ownership** | Which behavior executes in new Rust, retained JavaScript, or a browser adapter |
| **Acceleration** | The actual matched benchmark result on a named device/browser/workload |

Passing a test through retained JavaScript is legitimate compositional
compatibility. It is **not** evidence that a Rust implementation of that feature
exists. A fast renderer does **not** imply that every application callback was
accelerated.

### Execution routes and what each may legitimately claim

| Route | What it may claim |
| --- | --- |
| Specialized WebGPU | Verified accelerated Rust/Wasm/WebGPU execution |
| General WebGPU | Working WebGPU execution when specialization is unavailable; speed measured independently |
| Retained JS/host component | Full component functionality — **not** a Rust rewrite |
| Exact backend component (pinned upstream WebGLRenderer/backend) | Exact backend behavior and functional compatibility; **never** counted as the new renderer's acceleration |

Every test result records its implementation owner. Every accelerated test
asserts its designated rendering work went through the new backend. Every
exact-component test asserts it received actual native objects.

---

## THE NO-CUT RULE

An **optimization refusal is not an application-feature refusal.**

"Not in our demos," "not tested upstream," "uncommon," "dynamic," and "not yet
ported" are not release exemptions. A valid upstream operation must have a
working production implementation under the same applicable host prerequisites.

Scope is **all existing features and functionality of Three.js r186** — the
complete source surface, not the finite list of names in the plan document.
Sequencing is allowed. Permanent omission is not.

These manifest states are **blocking**, never a pass:

`unclassified` · `unimplemented` · `untested` · `known-regression` · `stub` ·
`no-op-substitute` · `candidate-refusal-on-valid-source`

`retained` describes *ownership*, not an automatic pass. `host-blocked` requires
the reference to have the same missing prerequisite.

Never average import coverage, behavior coverage, new-backend rendering
coverage, exact-component integration, and capable-host validation into a single
flattering "100%" number. Report them separately.

---

## Honesty Rules Specific To This Project

These are non-negotiable and they are what makes the project's output worth
anything:

1. **No speedup claim before a benchmark runs.** The plan states no speedup has
   been measured. Do not write "faster," "optimized," or a multiplier in docs,
   comments, or reports without a benchmark, machine fingerprint, reference
   configuration, and confidence interval.
2. **A symbol census is not an implementation.** Never label a plan, a static
   source inspection, or a name-matching census as implemented compatibility.
3. **Never fabricate a native handle.** Do not return a fake GL object, do not
   silently turn a synchronous result into a Promise, do not drop a pass, and do
   not infer source-visible capabilities from unrelated WebGPU limits.
4. **Never present the original renderer's logical draw count as the new
   backend's actual GPU work.** F3D telemetry lives in its own diagnostic
   namespace.
5. **Never regenerate golden images from the candidate.** A mismatch is a
   defect until proven otherwise.
6. **A candidate failure is a failed cell, not an excluded cell.** Crashes,
   unsupported paths, fidelity failures, and timeouts never shrink the
   benchmark denominator.
7. **Do not silently reduce quality to win.** Reduced resolution, shadow size,
   MSAA, anisotropy, sample counts, geometry detail, or temporal history are
   quality changes, not removal of abstraction overhead.
8. **Distinguish structural cost estimates from measured timings** everywhere
   they appear.

---

## Upstream Pin Discipline

- **Compatibility anchor:** Three.js **r186**, source commit
  `148ef33ecb6d2502ff796d4554abd1549c95d519`.
- The annotated tag object hash is **not** the source commit. Do not confuse
  them.
- Release updates are deliberate compatibility events, not automatic movements
  of the oracle. A new upstream version creates a new explicit target with its
  own source/assets/test/performance manifests. Retain old results.
- Do not infer compatibility from a semver string or a docs page. Track upstream
  through an immutable diff of exports, options, shader chunks, registrations,
  backends, addons, examples, and tests.
- A newer unverified release may only be used with its actual retained component
  and truthful unverified-version status.
- Removed APIs from releases before r186 are not invented as current features.
  Still-shipped deprecated aliases and their warning/error behavior remain in
  scope.

---

## Toolchain: Rust, Cargo, Wasm

- **Dated nightly toolchain**, pinned in `rust-toolchain.toml`. Never a moving
  `nightly` tag in the release contract. Test a new nightly in a separate lane
  before moving the anchor.
- Commit the toolchain file, lockfiles, and application compiler inputs.
- Primary target: `wasm32-unknown-unknown`. It has its own supported environment
  and instruction features. Browser Wasm cannot issue arbitrary native AVX-512
  or Apple-specific instructions.
- Portable SIMD (`simd128`) is allowed on the pinned nightly where its Wasm
  codegen is favorable. Keep a scalar reference and inspect the emitted code.
- Rust 2024 edition, Cargo workspace, eleven crates (see below).
- Do not introduce another package manager for the Rust side. The JavaScript
  ingestion toolchain is a separate, bounded, audited dependency.

### Unsafe Code

First-party semantic, numerical, compiler, scene, and resource code uses
`#![forbid(unsafe_code)]`.

- Browser binding machinery is a **named external boundary**, not evidence that
  the browser, drivers, std, or transitive dependencies contain no unsafe.
- A zero-copy JS view must come through an audited binding boundary whose
  lifetime contract is actually enforced. A desired speedup does not justify
  hand-written unsafe pointer casts or bypassing borrow rules in engine code.
- Use safe byte serialization and checked IDs rather than unchecked layout
  casts.
- No application callback, memory growth, or scheduler re-entry may occur while
  a Rust borrow of linear memory is live. Release the borrow before calling
  effectful host/user code, or copy.

### Dependency Policy

Small, pinned, audited allowlist (Section 13.4):

- `wasm-bindgen` / `js-sys` / `web-sys`, aligned with Asupersync's boundary
- shared serialization for build manifests and diagnostics — **not** per-frame
  JSON
- **Naga** as a shader compiler/validator component, normally build-time only
- the existing JavaScript bundler/parser toolchain for source ingestion
  (Rollup-ecosystem + Acorn-class parser, pinned in Phase 0)
- required runtime/build-time asset codecs selected by the application
  dependency closure
- version-pinned upstream Three.js public/host/tool/backend components required
  for full compatibility, with exact renderer execution strictly isolated and
  reported
- test/fuzz/browser-automation tools outside the shipping execution path

**Forbidden in the execution core:** Bevy, a general `wgpu` stack, a new generic
ECS framework, **Tokio**, a mandatory Rayon pool, native C/C++ FFI, BLAS, or a
Python interpreter.

For each candidate crate record: exact revision, used API, target compile
result, transitive closure, unsafe/FFI boundary, initialization requirements,
incremental Wasm bytes, runtime allocation cost, and demonstrated benefit.
Prefer `default-features = false`.

Sibling Franken libraries (`FrankenNumPy`, `FrankenSciPy`, `FrankenTorch`,
`FrankenLibC`) are **resources, not assumptions**. Reuse is decided per crate or
per kernel with target and dependency checks. `fnp-linalg` carries an
unconditional Rayon dependency; FrankenTorch is CPU-first; FrankenLibC targets
Linux/glibc, not the browser. Importing everything would undermine both the size
budget and the foundation requirement.

Do not transplant large source trees into this repository to hide their
dependency or line count.

---

## Asupersync Is The Only Async Foundation

Asupersync owns all FrankenThreeD async execution. Do **not** introduce Tokio,
an alternative executor, or an unstructured promise pool as a workaround for
browser integration.

- Prefer **one** application Wasm instance linking the admitted Asupersync
  core/host integration. Do not ship a separately instantiated browser-core
  runtime plus a second FrankenThreeD runtime with independent memories and
  ownership tables.
- Where the browser host-services seam is incomplete, **implement the missing
  host in Asupersync upstream** rather than forking its semantics inside
  FrankenThreeD. That work is charged to this project's budget (10,000 lines
  allocated).
- An ownership ledger closing is **not** proof that a browser callback stopped
  executing. Acceptance evidence is executable behavior: real futures polled,
  real wakeups, real cancellation delivered, real drain.
- Regions belong at session, scene/view, asset-preparation, shader/pipeline
  preparation, and worker-job granularity. **Never one task per object,
  triangle, uniform, or draw.**
- Cancellation is not rollback of issued effects. `writeBuffer`, `writeTexture`,
  and external-image copies are already queue effects before `submit()`.

---

## Planned Workspace Crates

Eleven crates. Not dozens of tiny architectural layers. No new crate without
explicit user approval.

| Crate | Responsibility | Planned new Rust lines (incl. tests) |
| --- | --- | ---: |
| `f3d-core` | Typed IDs/layouts, capabilities/epochs, errors, the single feature/route manifest | 10,000 |
| `f3d-math` | Fixed-size math, SIMD batches, robust geometry primitives | 12,000 |
| `f3d-scene` | Packed scene state, dirty propagation, animation, query structures | 25,000 |
| `f3d-assets` | Asset pack, safe decoders/preprocessing, resource metadata | 20,000 |
| `f3d-shader` | Standard material/TSL/ESSL semantics and dynamic lowering | 30,000 |
| `f3d-graph` | Pass semantics, hazards, lifetime planning, schedule specialization | 14,000 |
| `f3d-gpu` | Standard new-WebGPU resource/render/compute paths, backend boundary tests | 27,000 |
| `f3d-runtime` | Asupersync, lifecycle/host protocol, exact-component and tool integration | 12,000 |
| `f3d-compiler` | Route analysis, import closure, islands/guards/codegen | 24,000 |
| `f3d-cli` | `build` / `inspect` / `verify` / `bench` / `serve` orchestration | 7,000 |
| `f3d-conformance` | Feature/behavior closure, upstream/host/tool integration, performance tests | 24,000 |
| **Subtotal** | | **205,000** |
| Charged Asupersync foundation work | Browser-host completion/tests | 10,000 |
| **Planned** | | **215,000** |
| Contingency reserve | | 30,000 |
| **Ceiling** | | **245,000** |

Also budgeted: ~12,000 new JS/TS lines (source adapters, host bindings, harness)
and ~15,000 authored WGSL lines. Track them openly; do not hide an engine
outside the Rust budget.

A forecast above the ceiling triggers an implementation/reuse decision **before**
more infrastructure is built. It does **not** authorize dropping Three.js
features or their tests.

### Dependency Direction

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

- Browser bindings must not leak into numerical/compiler modules.
- Native asset/build tools must not become dependencies of the deployed Wasm.
- If you need to violate the direction, stop and redesign the API.

---

## Core Invariants

These are the rules that adversarial tests exist to enforce. Violating one is a
rendering defect, not a tuning choice.

### Queue ordering is not command-recording ordering

Two `writeBuffer` calls before a single submission can make **both** recorded
draws read the second value. This pattern is **always wrong**:

```text
write shared uniform = A; encode draw A
write shared uniform = B; encode draw B
submit both draws
```

Use immutable per-use buffer slices with distinct bindings/offsets,
source-indexed parameter tables, legal encoded copies between passes, or
separate submissions. The planner tracks **per-use data versions**, not a last
value per object per frame.

**Mandatory regression:** render red to target A, mutate the shared material to
blue, render to target B, submit a combined schedule. A must stay red, B blue.
Extend to compute uniforms, multiple cameras, nested callbacks, and two logical
frames in flight.

### Bundles reset render-pass state

After `executeBundles`, invalidate cached render-pass bindings and pipeline
state before subsequent direct draws — the API clears that state even for an
empty bundle sequence. **Mandatory regression:** bundle-then-direct-draw with
distinct bindings. A falsely warm state cache is a defect, not an optimization.

A bundle key includes device generation, pipeline/layout, attachment formats and
sample count, depth/stencil compatibility, bound resource identities, offsets,
and recorded draw parameters. Changing a recorded binding, dynamic offset,
geometry range, or draw count is not an in-place patch.

### CPU mutation and GPU-visible mutation are different histories

Keep the public CPU content version, the source's upload-request/version/range
behavior, and the last GPU-visible content **separate**. Coalescing uploads
across gaps must not expose CPU edits the reference would leave GPU-stale — fill
from the last GPU-visible shadow, prove the bytes equal, or keep the smaller
writes. The same applies to padding a sub-word update to alignment. **A scan
detecting a CPU mutation is not permission to upload it early.**

### Synchronous work cannot acquire invisible yield points

A cooperative runtime does not authorize splitting an originally synchronous
update across browser turns. Chunking with a host yield is allowed only for
source-asynchronous work, isolated worker jobs, or a transformation whose
interleaving equivalence is established. An effectful loop that throws after
several mutations cannot become an all-or-nothing bulk commit unless those
intermediate effects are unobservable.

### One authoritative owner per state region

Every region is JavaScript-owned, Wasm-owned, or mirrored under an explicit
epoch contract. **A mirror is not a license for two writers.** Authority
transfers are ordered, versioned, and tested. The general fallback is JavaScript
ownership, not an optimistic guess that an escaped object will not be modified.

### Numeric fidelity

Use `f64` or retained JavaScript for observable scalar operations; convert at
the same admitted boundaries as the reference. Preserve signed zero, exceptional
values, observable operation ordering, and rounding. JavaScript integer
coercions, shift counts, remainder, and `Math.min`/`Math.max` need explicit
lowering — Rust casts are not substitutes. Transcendental functions require
differential testing against the browser.

FP16, relaxed arithmetic, different reduction order, or lossy compressed data
belongs in a **separately labeled optional mode**, never the primary speedup
comparison.

### Layout is explicit, generated, and tested

`AffineRows` is three `vec4<f32>` rows with translation in each row's fourth
component: **48 bytes**, row dot-products against `(x,y,z,1)`. For a public
column-major matrix `e`, the rows are `[e0,e4,e8,e12]`, `[e1,e5,e9,e13]`,
`[e2,e6,e10,e14]`.

WGSL's `mat4x3<f32>` is four padded three-component columns and occupies **64
bytes**. Writing twelve tightly packed floats into that type is wrong. Rust
struct layout is not WGSL layout. Distinguish storage-array stride from
dynamic-uniform-offset alignment. Validate size/stride/round-trip independently
of the renderer.

### Resource identity and lifetime

Handles carry an index and a generation, plus a separate **device generation**
for GPU resources. Never let a delayed promise publish into a reused slot.
Distinguish CPU existence, GPU allocation, initialized content, submitted use,
and retirement. `dispose()` releases a *residency*, not necessarily the public
object — a legally reused material/texture/geometry gets fresh backend residency
without changing its JavaScript identity.

### The exact-backend boundary is irreversible

Choose a renderer route **before** binding its canvas/context/device or exposing
native handles. Treat it as an irreversible semantic decision for that renderer
lifetime unless the source itself supports creating/replacing the renderer at a
tested boundary. Revocable numeric-island guards are not enough to justify a
later WebGPU/WebGL switch on the same exposed canvas.

If a renderer or its canvas/native state can escape to opaque code that may call
GL-specific methods, preserve the exact component **from construction**.

### Error scopes are shared device state

Push scopes, issue the owned operations, and pop without an intervening await or
user callback that lets another task interleave unrelated scopes; only then
await the results. Concurrent Asupersync tasks do **not** get private error-scope
stacks.

### Canvas and external textures are not ordinary persistent resources

Acquire the current canvas texture for the actual rendering interval; never
cache its texture/view in a long-lived scene cache. Avoid an arbitrary yield
between acquisition and use. External video textures have source-dependent
lifetimes — refresh at supported media/render boundaries, retain `VideoFrame`
ownership until uses complete, close exactly once.

---

## Mandatory Counterexample Suite

Every one of these has a named regression test. They are part of the **measured
optimized path**, never disabled in benchmark mode:

| Counterexample | Guards against |
| --- | --- |
| Red-A / blue-B queue-write snapshot | Treating queue writes as per-draw snapshots |
| Bundle-then-direct-draw state reset | Falsely warm binding/pipeline caches |
| Upload-range stale-byte gap fill | Exposing CPU edits the reference leaves GPU-stale |
| Callback multiplicity under batching | Erasing or reordering per-object effects |
| Synchronous-yield interleaving | Splitting a run-to-completion update |
| `AffineRows` vs `mat4x3` layout | Silent 48/64-byte layout corruption |
| Guard failure without double-execution | Replaying already-performed side effects |
| Cancelled upload / device loss during compile, upload, submit | Stale publication into a replaced device |
| Delayed worker response after ID reuse | ABA handle collisions |

Each optimizer pass additionally needs a legality statement, a conservative
executable reference, counterexample tests, and an ablation measurement.

---

## Testing Policy

### Three production-route lanes

Run all three separately. They cannot substitute for each other:

1. **Full functional routing** — the ordinary production product.
2. **Forced new-backend execution** — all applicable rendering cases with
   retained renderer submission **disabled**.
3. **Exact native-backend integration** — real GL contexts, sync reads, native
   handles, context loss/restore.

Lane 2 exists specifically to prevent a compatibility-only wrapper from
masquerading as this project.

### Upstream suite

Redirect upstream source import paths to production compatibility exports
through a candidate module-resolution tree. **Keep the original test bodies,
expected values, and assertions intact.** Use a mapped working copy or resolver
— never edit the upstream oracle checkout.

- Every active upstream test must be imported and executed in its required
  environment. Syntax/import/setup errors and assertion-count shortfalls **fail**
  the inventory rather than becoming empty successful suites.
- A candidate-only skip is a failure. An upstream-existing skip stays visible as
  an unproven case, not a new pass.
- Preserve the E2E lane's own thresholds (per-pixel rule, 0.1% different-pixel
  limit). Never loosen a threshold to pass.
- Record the resolved implementation for every import. Validate which package,
  source revision, instance, and submission path actually executed.
- The upstream screenshot lane uses a software Vulkan ICD and special launch
  flags. It is a **regression oracle, never a performance lane.**

### Feature-completeness tests beyond upstream

Every callable/option/behavior family in the manifest needs an executable
positive case plus relevant invalid-input, mutation, serialization, and
lifecycle cases. Constructor existence or matching function names is
insufficient.

Mandatory family suites: all import paths/aliases; every loader/exporter/codec;
every material/map/state family; each TSL/node and shader entry; every
pass/effect; all controls and alternate renderers; audio/media; XR; physics
wrappers; Inspector/editor/tool workflows; all native-backend public APIs.

Required high-risk combinations: skinning+morphs+shadows+export;
clipping+stencil+transparency; instancing+negative scale+custom shader
indices+picking; transmission+color management+render targets; texture
update+multiple renderer residencies+disposal; runtime shader/node
registration+readback; CSS/SVG+camera changes+input; XR+skinning+layers+session
reset; editor undo/redo+shared materials+serialization.

### Adversarial and property tests

Aliasing, typed-array writes, subclass overrides, reflective access,
`needsUpdate`, callbacks that reenter rendering, mutable material keys,
matrix-read materialization, deoptimization after an assumption changes.

Geometry/math: empty data, negative scale, shear, singular matrices, degenerate
triangles, ray-hit ties, extreme coordinates, NaNs, infinities, integer overflow
boundaries.

Lifecycle: cancel at every supported await/publication boundary; lose the device
during compilation/upload/submission; dispose while work is pending; grow Wasm
memory; deliver delayed worker responses after ID reuse.

Fuzz: asset-pack parsing, shader preprocessing/translation, handle decoding,
buffer layout validation, serialized execution plans.

### Loaders, exporters, round trips

A loader is not complete because one bundled model renders. Exercise `load`,
`loadAsync`, `parse`, async parse, LoadingManager handlers and URL modifiers,
headers/credentials/CORS, response types, progress, cancellation, errors,
caching, relative dependencies.

Export must work **after** an accelerated update — materialize current
CPU-observable state at the writer's original boundaries. Never export the
initial captured scene while the user sees a newer one. Required:
import → edit/animate → export → re-import in both reference and candidate, plus
cross-loading.

---

## Performance Program

Performance claims must be reproducible and measurable. Do not write "fast"
without a benchmark, target, machine fingerprint, and acceptance band.

### What is measured

**Completed logical frames per wall-clock second** in a bounded steady-state
pipeline, including application update work, scene preparation, bridge work,
uploads, command preparation, GPU execution, and completion.

Do **not** substitute the time to call `queue.submit`, a GPU-only kernel timer,
or a matrix microbenchmark for end-to-end throughput. Two applications capped at
120 displayed FPS do not exhibit a 3× displayed-FPS improvement.

Measure cold and warm startup separately: document acquisition, decoding, Wasm
compilation, shader/pipeline compilation, asset preparation, first correct
frame, first interactive frame.

### References

- **R0** — as authored, with shared deterministic harness instrumentation only.
- **R1** — a competent Three.js configuration under a **frozen, bounded** recipe
  of documented precompilation, instancing/batching, render-bundle, and
  resource-management optimizations. Not an unlimited manual rewrite.
- **R2** — an equivalent current Three.js WebGPU implementation.

Select the strongest eligible configuration by **independent reference-only
calibration**, then freeze source and settings before scored paired runs. Do not
take the fastest noisy trial among many as the denominator. Publish R0 and every
eligible configuration, not only the selected one. Render bundles and
upload-range coalescing are **baseline capabilities, not F3D inventions**.

### Gates

| Gate | Requirement |
| --- | --- |
| Equivalence | Visual, state, control, and lifecycle profile passes before any speedup counts |
| Eligibility | All reference-valid H1–H8 cells stay required; candidate failures never shrink the denominator |
| Headline floor | ≥ **2×** median paired completed-frame throughput in **every** required cell |
| Stratum target | ≥ **3×** geometric mean across required headline cells per primary stratum |
| Confidence | Paired 95% lower bounds meet the 2× cell floor and exceed 2× for the stratum geomean |
| Modern coverage | H7 and H8 stay in the required score |
| Tail | Frozen p95/p99 frame-time, missed-deadline, input-response budgets |
| Controls | ≤ 5% sustained-throughput regression on valid control cases |
| Startup/memory | Frozen absolute device budgets; bounded caches; no OOM |
| Thermal | Phone floors hold in the sustained segment, not only the cool start |
| Deployment | Portable standalone passes independently; isolated/shared-memory results are additive only |
| New-backend ownership | Scored GPU work uses the new renderer, not retained submission |
| Size | Inside the 245k planned Rust ceiling |

Aggregate cell ratios in **log space** with equal preregistered weights.
Bootstrap at the paired run/block level — adjacent correlated frames are not
independent experiments. Freeze repeat count, stopping rule, aggregation
weights, and infrastructure-failure treatment **before** scoring. An unfavorable
valid run is not an infrastructure failure.

### Headline corpus (H1–H8)

| ID | Upstream example |
| --- | --- |
| H1 | `webgpu_performance_renderbundle` |
| H2 | `webgl_marchingcubes` |
| H3 | `webgl_animation_multiple` |
| H4 | `webgl_shadowmap_performance` |
| H5 | `webgl_instancing_dynamic` |
| H6 | `webgl_postprocessing_advanced` |
| H7 | `webgpu_skinning_instancing_individual` |
| H8 | `webgpu_compute_particles_fluid` |

H7 and H8 prevent the suite from consisting only of legacy object-submission
bottlenecks. They may be hard to accelerate. **That difficulty is part of the
go/no-go test, not a reason to remove them after disappointing results.**

After the manifest is frozen, a slow or difficult case stays in its declared
category. Any corpus revision creates a **new** benchmark version and retains
the old results. Fewer than six headline cases, or loss of either modern-compute
category, blocks a broad launch claim.

Also required and published (not part of the headline gate): a minimal
GPU-bound full-screen shader as a negative control, and a small ordinary
interactive scene as a fixed-overhead/startup control.

### Device matrix

| Stratum | Browser evidence |
| --- | --- |
| iPhone 17 Pro Max | Safari on a pinned OS/browser build; Chrome checked separately |
| M5 MacBook | Safari and Chrome, separate runs on the same machine |
| NVIDIA discrete GPU | Chrome on supported Windows; Linux as a separately recorded lane |
| AMD discrete GPU | Chrome on supported Windows; Linux as a separately recorded lane |

No fictional "Safari on Windows NVIDIA" target. A bundled automation WebKit
build or an iPhone simulator is a separate test environment, not interchangeable
hardware evidence. Product labels (M5, NVIDIA) do not determine exposed WebGPU
features or limits — probe at runtime and record unknowns as unknown.

### Benchmark hardware discipline

**GPU timing runs on a reserved, uncontended physical device.** The benchmark
machine must not be running unrelated agent builds, other swarm members, or
background GPU work. Remote build workers may compile artifacts; they may not
produce hardware timings. If you cannot guarantee an uncontended device, do not
report a number.

---

## Early Kill Gates

Do not spend the entire 215k planned lines before learning whether the idea
works.

1. **Foundation gate** — actual browser task execution and cleanup, plus a legal
   WebGPU bridge, on Safari and Chrome. No ledger-only "execution," no native
   thread dependency.
2. **Bridge gate** — measure direct JS WebGPU, simple Wasm→host calls, bulk
   command transfer, and generated host submission on matched workloads. The
   selected path must show an advantage opportunity **after its own overhead**.
3. **First product gate** — automatically transform complete H1 and H2, preserve
   all controls/backend modes/material variants, score H1's WebGPU dynamic
   workload and H2 through the new renderer, with real guards, materialization,
   bootstrap, and bridge costs. Require ≥ 2× on phone and M5 strata. If H1 is
   not viable against its competent reference, **do not substitute an easier
   toy** — reopen the project decision.
4. **Generalization gate** — repeat on held-out applications and at least one
   discrete-GPU stratum, with a measured end-to-end route for H7/H8.
   Microbenchmarks alone cannot pass.
5. **Final gate** — exhaustive functional and accelerated-rendering closure, all
   upstream/integration tests, all performance conditions.

---

## Implementation Sequence

- **Phase 0** — Full source/feature inventory and executable foundation. Pin
  r186. Reconcile every export/source/addon/registration/tool/example into the
  single manifest. Profile H1/H2/H7/H8 references. Freeze performance settings,
  strong-reference selection, and startup/memory/tail budgets. Execute real
  Asupersync Rust futures and cancellation in Safari and Chrome. Measure bridge
  variants and the queue/state counterexamples.
- **Phase 1** — Complete H1 and H2 as real applications with real acceleration.
- **Phase 2** — Generalize without losing public functionality.
- **Phase 3** — Full new-backend rendering and dynamic shader coverage.
- **Phase 4** — Feature, environment, and tool closure; demanding performance.
- **Phase 5** — Complete deployment and release validation.

**Build Phase 0 first.** Do not start with frontier or optional-research work.
The plan's three tiers — Solid, Frontier, Optional research — must stay
distinguishable in docs, feature gates, and status reports. Optional research
(geometric algebra, neural rendering, differentiable scene optimization, novel
ray tracing, general JS compilation) must never consume the compatibility
reserve before the core gates pass.

### Work breakdown

`F3D-01` … `F3D-24` in Section 19 are the proposed implementation units. Use
them as Bead titles when the tracker exists. Note the plan's own warning: F3D-15
(shader) starts as source analysis/normalization and WGSL generation, integrating
with F3D-07 as **one early vertical slice** — not a circular demand for finished
subsystems. F3D-18 and F3D-22 start small and accrue coverage continuously.

**No task defers the safety or semantic precondition of an earlier claimed
performance result.**

---

## Code Editing Discipline

### No script-based code changes

Do not run broad regex or script-based rewrites over source files. Make changes
manually with focused patches. Use `ast-grep` only when the pattern is genuinely
syntactic and the diff is reviewable.

### No file proliferation

Revise existing files in place unless a new file is genuinely new functionality
or a required contract/test artifact. Forbidden: `main_v2.rs`, `improved.rs`,
`new_version.rs`, `final_final.rs`, duplicate experimental module copies.

### No new crates without approval

Eleven crates is the plan. Adding a twelfth is an architecture decision, not an
implementation detail.

### Backwards compatibility

This project is pre-implementation. Prefer the correct design over compatibility
shims for **F3D's own** APIs. This does **not** apply to the Three.js surface,
where upstream compatibility is the entire product.

### Comments and docs

Document invariants, ownership routes, error models, and no-claim boundaries.
Avoid comments restating code. In GPU/shader/layout code, include enough detail
for the next agent to verify byte offsets, strides, coordinate conventions, and
ordering assumptions.

---

## Output Style

Core library code should not print casually to stdout/stderr.

- Use structured tracing or report events for observability.
- CLI output must be deterministic and documented, with stable JSON alongside
  human-readable text.
- Errors intended for agents should be structured and actionable.
- Instrumentation must be removable or bounded in production. The renderer must
  not allocate a JSON event or run a statistical controller per draw.
- Report precise source spans for specialization barriers — the developer needs
  to know *which* line prevented a fast path.

---

## Compiler Checks

After substantive Rust changes, verify the relevant checks pass. DSR is the
first choice for repo-level gates and release builds. RCH is for narrow ad hoc
Cargo probes or when DSR is unavailable.

```bash
dsr quality --tool franken_threed
dsr build franken_threed --target darwin/arm64
```

**If the workspace is still plan-only, do not invent checks.** State that no
Cargo workspace exists yet and validate markdown or file presence only.

### DSR - Required CI and Release Runner

GitHub Actions is not the CI source of truth here; the account is throttled.
Always prefer DSR for repo-level verification, release builds, and fallback
release work.

- Use `dsr` if on `PATH`; otherwise
  `/Users/jemanuel/projects/doodlestein_self_releaser/dsr`.
- `dsr repos info franken_threed` — registry wiring
- `dsr quality --tool franken_threed [--dry-run]`
- `dsr build franken_threed --target darwin/arm64`
- `dsr doctor`, `dsr health all`

Report the exact DSR command, pass/fail status, and any run log or artifact path.
If DSR is unavailable, report the exact blocker and label any local fallback
clearly.

### RCH - Remote Compilation Helper

```bash
rch exec -- env CARGO_TARGET_DIR="${RCH_TARGET_BASE:-${TMPDIR:-/tmp}}/rch_target_f3d_check" cargo check --all-targets
rch exec -- env CARGO_TARGET_DIR="${RCH_TARGET_BASE:-${TMPDIR:-/tmp}}/rch_target_f3d_test" cargo test --all-targets
```

**ALWAYS base new target dirs on `${RCH_TARGET_BASE:-${TMPDIR:-/tmp}}`** — never
bare `/tmp`. These dirs are 5–15G each and are never auto-cleaned. Prefer
reusing the two names above.

**Disk preflight:** before any heavy build/test lane on this Mac, run
`sbh check --need 20G`. A nonzero exit means the Data volume is under pressure —
STOP and reclaim first. A full disk silently corrupts shared SQLite writers
(`br`, cass, agent mail) mid-commit.

**Browser and GPU work does not belong on RCH.** Remote workers compile; they do
not run Safari, drive WebGPU, or produce timings.

---

## Documentation and Contracts

Each crate ships a `CONTRACT.md` before it becomes a dependency target, stating:

- purpose and position in the dependency direction
- public types and semantics
- invariants
- **implementation ownership route** (new Rust / general new backend / retained
  JS / exact backend)
- error model and source-error preservation
- determinism and numeric-fidelity class
- cancellation behavior
- unsafe boundary, if any
- feature flags
- conformance tests
- **no-claim boundaries**

Keep `COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENTHREED.md` high-level. Do not
turn it into a dumping ground for implementation notes that belong in crate docs,
contracts, or issue records.

---

## Agent Workflow

### Start of work

1. Read this file.
2. Read the plan sections relevant to the task.
3. Read `README.md`, the relevant crate `CONTRACT.md`, and the Beads issue if
   they exist.
4. Inspect the tree before editing.
5. Reserve files through Agent Mail if multiple agents are active.

### While working

- Keep changes tightly scoped.
- Do not disturb unrelated files.
- Do not revert changes you did not make.
- Keep every technical claim attached to a test, a contract, or explicit
  no-claim language.
- Name the implementation route for anything you build or test.

### End of work

1. Run applicable format/check/test lanes.
2. Report exactly what passed and what did not run.
3. If Beads is in use, update issue status and run `br sync --flush-only`.
4. Release Agent Mail file reservations.
5. Leave clear handoff notes for any blocker.

---

## MCP Agent Mail - Multi-Agent Coordination

When Agent Mail tools are available, use them for coordination and file
reservations.

1. Register identity for this project path.
2. Reserve files before editing.
3. Use the Beads issue ID or task name as the thread ID.
4. Send start/progress/completion messages for shared work.
5. Release reservations when finished.

Reservations are advisory, but in this project they are the isolation mechanism.
They replace branches and worktrees.

---

## Beads (`br`) - Issue Tracking

If `.beads/` exists, use Beads for task state and dependency tracking.

```bash
br ready
br list --status=open
br show <id>
br update <id> --status=in_progress
br close <id> --reason "Completed"
br sync --flush-only
```

- Use Beads IDs in Agent Mail thread IDs and commit messages.
- Do not run bare interactive tools in automated sessions when robot/non-TUI
  modes exist.
- `br sync --flush-only` does not commit. Stage and commit intentionally.

---

## `bv` - Graph-Aware Triage

If `bv` is available and `.beads/` exists, use robot modes only. Bare `bv`
launches an interactive TUI and can block the session.

```bash
bv --robot-triage
bv --robot-next
bv --robot-plan
bv --robot-insights
```

---

## `ubs` - Bug Scanner

Before committing code, run `ubs` on changed files:

```bash
ubs $(git diff --name-only --cached)
```

Fix true positives at the root cause and rerun on the affected files.

---

## Search Tools

- `rg` for fast targeted text search; `rg --files` to inspect file sets.
- `ast-grep` when syntax matters.
- AI search tools only for exploratory architecture questions, never for exact
  symbol lookups.

Do not use broad scripted rewrites where a hand patch is safer.

---

## Current Repository State

At the time this file was written, `/Users/jemanuel/projects/franken_threed`
contained only:

- `COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENTHREED.md`
- `AGENTS.md`
- `README.md`
- `LICENSE`, `.gitignore`, `.gitattributes`
- `.beads/` — the `br` issue tracker (prefix `f3d`), holding the plan converted
  into an execution graph: one epic per Section 19 work unit (F3D-01 … F3D-24,
  with F3D-12/15/18/21/22 split into an early slice and a closure epic exactly
  as Section 19 prescribes), child tasks under each, three standalone kill-gate
  beads (Foundation, Bridge, Generalization; gates 3 and 5 live inside F3D-11
  and F3D-24), and one deferred placeholder for the optional research tier.
  Epic-level `blocks` edges encode the Section 19 closure ordering; child-level
  edges encode actual start prerequisites. Start with `br ready --json`.

There is **no Cargo workspace, no pinned Three.js checkout, no test harness, and
no measurement**. When those are added, update this section or replace it with
the actual workspace map.

The plan document explicitly states that it does not create a repository,
issues, commits, or releases, and that no FrankenThreeD performance result has
been established.

---

<!-- bv-agent-instructions-v2 -->

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking and [beads_viewer](https://github.com/Dicklesworthstone/beads_viewer) (`bv`) for graph-aware triage. Issues are stored in `.beads/` and tracked in git.

### Using bv as an AI sidecar

bv is a graph-aware triage engine for Beads projects (.beads/beads.jsonl). Instead of parsing JSONL or hallucinating graph traversal, use robot flags for deterministic, dependency-aware outputs with precomputed metrics (PageRank, betweenness, critical path, cycles, HITS, eigenvector, k-core).

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). `br` handles creating, modifying, and closing beads.

**CRITICAL: Use ONLY --robot-* flags. Bare bv launches an interactive TUI that blocks your session.**

#### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns everything you need in one call:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command

# Token-optimized output (TOON) for lower LLM context usage:
bv --robot-triage --format toon
```

#### Other bv Commands

| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with unblocks lists |
| `--robot-priority` | Priority misalignment detection with confidence |
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions, cycle breaks |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |

#### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work (no blockers)
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank scores
```

### br Commands for Issue Management

```bash
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export DB to JSONL
```

### Workflow Pattern

1. **Triage**: Run `bv --robot-triage` to find the highest-impact actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Always run `br sync --flush-only` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads changes to JSONL
git commit -m "..."     # Commit everything
git push                # Push to remote
```

<!-- end-bv-agent-instructions -->

---

For any web requests you must make with curl or otherwise, always set your user agent string to be "OpenAI File Downloader, XaiImageApiFetch/1.0"
