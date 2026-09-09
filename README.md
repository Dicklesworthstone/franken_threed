# FrankenThreeD

<div align="center">

[![Status](https://img.shields.io/badge/status-plan--stage%20%C2%B7%20no%20implementation-orange)](#honest-status)
[![Upstream](https://img.shields.io/badge/three.js-r186%20pinned-000000)](#the-pin)
[![Core](https://img.shields.io/badge/core-Rust%20%2F%20Wasm%20%2B%20WebGPU-b7410e)](#architecture)
[![Async](https://img.shields.io/badge/async-asupersync%20only-0969da)](#asupersync-is-the-async-foundation)
[![Budget](https://img.shields.io/badge/rust%20ceiling-245k%20lines-8250df)](#workspace-and-budget)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20AI%20rider-yellow)](LICENSE)

</div>

**FrankenThreeD is a Three.js-compatible application specializer with a Rust/Wasm
and WebGPU execution core.**

You give it an ordinary Three.js project. It gives you back a full-compatibility
application package — a self-contained HTML file when the code and resource graph
closes, or a normal networked/module build when it does not — in which the
expensive, provably-closed parts of the frame loop have been compiled into
specialized Rust kernels, packed GPU state, and a persistent render schedule.

Existing authoring knowledge stays useful. You keep writing Three.js.

---

## Honest status

**Nothing is implemented.** This repository currently contains one document:
[`COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENTHREED.md`](COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENTHREED.md)
(v2.0), plus this README, `AGENTS.md`, and repository hygiene files.

| Question | Answer |
|---|---|
| Is there a Cargo workspace? | No |
| Is there a compiler, renderer, or CLI? | No |
| Has any speedup been measured? | **No.** Zero benchmarks have been run |
| Has the Three.js suite been executed against a candidate? | No |
| Has the exhaustive symbol/behavior census been executed? | No — the plan defines it, it has not been run |
| What has been done? | A source-informed architecture and implementation plan, with the pinned upstream source inspected |

The plan is deliberately written to be falsifiable. It specifies kill gates that
can stop this project early, and it says so explicitly. Treat every number below
as a **target or a budget**, never as a result.

---

## TL;DR

**The problem.** A general-purpose renderer must answer the same questions on
every frame: which objects share a program, which values changed, which changes
affect which passes, what can be batched. A Three.js application usually has one
stable answer to most of those questions, and pays to rediscover it 120 times a
second.

**The non-solution.** Translating `object.position.x = value` into an equally
fine-grained Wasm call just relocates the overhead. Language substitution is not
an optimization.

**The idea.** Recover information across ordinary library boundaries at build
time. Turn thousands of fine-grained property writes into one contiguous update
kernel, legally versioned bulk uploads, and a persistent GPU schedule. Ship the
answer plus a small invalidation condition instead of the question.

> The most important optimization is **removing a question from the frame loop.**

**The honesty constraint.** For a simplified serial critical path, if fraction
`p` is accelerated by factor `k` and new overhead is fraction `h`:

```text
speedup = 1 / ((1 - p) + p/k + h)
```

Even infinitely fast replacement of one part cannot produce 3× unless roughly
two-thirds of the relevant critical path is removable *before* new overhead. That
is why this project's gates are measured end to end, on real devices, against a
strong reference — and why they can fail.

---

## The pin

| | |
|---|---|
| Upstream release | **Three.js r186** (September 8, 2026) |
| Source commit | `148ef33ecb6d2502ff796d4554abd1549c95d519` |

The annotated tag object hash is *not* the source commit. Release updates are
deliberate compatibility events, not automatic movements of the oracle. A new
upstream version gets its own source, asset, test, and performance manifests, and
old results are retained.

---

## What full compatibility actually means here

The scope is **all existing features and functionality of r186** — the complete
source surface, not a convenient subset. That includes the ESM/CommonJS roots,
`three/webgpu`, `three/tsl`, `three/addons`, wildcard addon and source paths,
every loader and exporter, WebXR, WebAudio, CSS2D/CSS3D/SVG renderers, physics
wrappers, the editor, the Inspector, devtools, and TSL tooling.

### The no-cut rule

> An optimization refusal is not an application-feature refusal.

"Not in our demos," "not tested upstream," "uncommon," "dynamic," and "not yet
ported" are not release exemptions. Sequencing is allowed. Permanent omission is
not.

### Four promises, never conflated

The single most important discipline in the project. These four statuses are
tracked separately in code, tests, reports, and documentation:

| Promise | Meaning |
|---|---|
| **API compatibility** | Every pinned-release API contract has a working production path under its source-supported host prerequisites |
| **Application equivalence** | The complete application feature and control surface is preserved |
| **Implementation ownership** | Whether behavior executes in new Rust, retained JavaScript, or a browser adapter |
| **Acceleration** | The actual matched benchmark result on a named device, browser, and workload |

Passing a test through retained JavaScript is legitimate compositional
compatibility. It is **not** evidence that a Rust implementation exists. A fast
renderer does **not** imply that every application callback was accelerated.

### Four execution routes

| Route | May legitimately claim |
|---|---|
| **Specialized WebGPU** | Verified accelerated Rust/Wasm/WebGPU execution |
| **General WebGPU** | Working WebGPU execution when specialization is unavailable; speed measured separately |
| **Retained JS / host component** | Full component functionality — not a Rust rewrite |
| **Exact backend component** | Exact backend behavior and functional compatibility; **never** credited as acceleration |

The fourth route exists because some things cannot be faked. `getContext()`
returning a real WebGL context, shared-context `resetState()`, synchronous
render-target readback, native program handles, and construction-time XR
fallbacks are genuine features. A WebGPU-only imitation cannot promise their
identity or semantics — so the pinned upstream backend is retained, isolated,
version-pinned, and reported, rather than the feature being refused.

The mirror-image rule keeps that from becoming an excuse: **every portable
standard rendering family must also pass through the new WebGPU backend with
retained renderer submission disabled.** A release containing only a renamed
upstream renderer fails, no matter how good its compatibility numbers are.

---

## Architecture

### Pipeline

```text
Original application + pinned Three.js + declared assets
                         |
             module resolution / source analysis
                         |
            Three.js semantic and effect analysis
                         |
      +------------------+---------------------+
      |                                        |
retained JavaScript                    compilable numeric islands
DOM / callbacks / escapes               typed updates / geometry / animation
      |                                        |
      +------------ semantic scene IR ----------+
                         |
               render/compute pass graph
                         |
            resource and capability specialization
                         |
   application Wasm + WGSL + static host code + asset pack
                         |
                 one self-contained HTML
```

### Three representations, not a tower of frameworks

- **Semantic IR** — object identity, ownership, numeric types, update
  dependencies, effects, observations, source locations.
- **Pass graph** — resource reads/writes, attachment operations, required order,
  history dependencies, pass provenance.
- **Execution plan** — packed buffers, shader variants, resource assignments,
  draw/dispatch buckets, reusable bundles, a small per-frame command sequence.

Typed IDs into owned arrays. No heap-allocated trait-object graphs in the hot
path. A bounded worklist optimizer, not a theorem prover.

### The unit of specialization: a closed update island

An island is admitted only when the compiler can answer: who can mutate each
value including aliases; who can read intermediate state and when; which
operations are pure; which numeric semantics are required; and what invalidates
the specialization.

Observing a property stay constant for 1,000 frames is **not** proof that it is
constant. A trace can suggest an island; it cannot justify erasing a possible
behavior. Guards validate assumptions before affected work runs, and every
optimized kernel ships alongside its conservative production path.

Write barriers are instrumented at build time on provably covered writes — **not**
by wrapping the application in JavaScript Proxies, which would change observable
behavior and add overhead to exactly the paths we want to remove.

---

## Workspace and budget

Eleven crates. Not dozens of tiny architectural layers.

| Crate | Responsibility | Planned Rust lines |
|---|---|---:|
| `f3d-core` | Typed IDs/layouts, capabilities/epochs, errors, the feature/route manifest | 10,000 |
| `f3d-math` | Fixed-size math, SIMD batches, robust geometry primitives | 12,000 |
| `f3d-scene` | Packed scene state, dirty propagation, animation, query structures | 25,000 |
| `f3d-assets` | Asset pack, safe decoders/preprocessing, resource metadata | 20,000 |
| `f3d-shader` | Material/TSL/ESSL semantics and dynamic lowering | 30,000 |
| `f3d-graph` | Pass semantics, hazards, lifetime planning, schedule specialization | 14,000 |
| `f3d-gpu` | New-WebGPU resource/render/compute paths, backend boundary tests | 27,000 |
| `f3d-runtime` | Asupersync, lifecycle/host protocol, exact-component and tool integration | 12,000 |
| `f3d-compiler` | Route analysis, import closure, islands/guards/codegen | 24,000 |
| `f3d-cli` | build / inspect / verify / bench / serve orchestration | 7,000 |
| `f3d-conformance` | Feature closure, upstream/host/tool integration, performance tests | 24,000 |
| | **Subtotal** | **205,000** |
| | Charged Asupersync browser-host work | 10,000 |
| | **Planned** | **215,000** |
| | Contingency reserve | 30,000 |
| | **Ceiling** | **245,000** |

Plus roughly 12,000 JS/TS lines for adapters, host bindings, and harness
integration, and roughly 15,000 authored WGSL lines — tracked openly rather than
hidden outside the Rust budget.

These are **estimates, not measured implementation sizes.** A forecast above the
ceiling triggers a reuse decision, not a scope cut. Retained upstream modules and
third-party codec/compiler code are reported separately with size and ownership.

### Dependency direction

```text
core <- math
core/math <- scene
core <- assets
core <- shader
core/scene/shader <- graph
core/graph <- gpu
scene/assets/gpu + asupersync <- runtime
core/scene/shader/graph <- compiler
compiler/runtime <- cli
public crate APIs <- conformance
```

Browser bindings do not leak into numerical or compiler modules. Native
asset/build tooling never becomes a dependency of the deployed Wasm.

---

## Proposed command surface

**These commands do not exist yet.**

```bash
f3d inspect ./src/main.ts --format json
f3d build ./index.html --compat full --standalone --out ./dist/demo.html
f3d build ./src/main.ts --compat full --deployment networked --out ./dist
f3d verify ./index.html --candidate ./dist/demo.html
f3d verify-upstream --ref r186 --features all --tests all
f3d bench ./benchmarks/headline.toml --device local
f3d explain ./dist/demo.f3d-report.json
f3d serve ./dist --port 8080
```

`--compat full` is the **default**, not a premium tier or a later phase.
`--require-accelerated-webgpu` is a diagnostic strict mode that fails when an
acceleration requirement cannot be met; it does not redefine the product's
functional scope.

Every build emits a report, in human-readable text and stable JSON, covering: the
Three.js version and asset/module hashes; what was specialized, generally
executed, retained, or kept on an exact backend; functional gaps versus host
limitations versus optimization-only barriers, with no conversion between those
statuses; estimated structural costs clearly distinguished from measured timings;
precise source spans for specialization barriers; and what would unlock a fast
path without weakening behavior.

No LLM is needed in the deployed application or the compiler correctness path.

---

## asupersync is the async foundation

[asupersync](https://github.com/Dicklesworthstone/asupersync) is the sole async
programming foundation for FrankenThreeD-owned work. No Tokio, no parallel
executor, no unstructured promise pool.

The plan treats this as a **critical-path risk, not a checked box.** Before a
large rendering system gets built, a real Rust-authored browser program must
demonstrate: futures actually polled and producing observed results; timers and
host-turn wakeups without native threads; a nonreentrant scheduler pump with
bounded microtask bursts; cancellation delivered to running cooperative tasks;
real fetch cancellation and cleanup; child tasks drained before scene-region
teardown; no late publication into a cancelled scene; explicit behavior for
unsupported hosts.

Where the browser host-services seam is incomplete, the missing host is
implemented **upstream in asupersync**, not forked into FrankenThreeD. A closed
ownership ledger is not proof that a browser callback stopped executing.

---

## The correctness rules that make this hard

These are the failure modes the architecture is explicitly built to catch. Each
has a mandatory regression test that stays enabled in benchmark mode.

**Queue writes are not per-draw snapshots.** Two `writeBuffer` calls before one
submission can make both recorded draws read the second value. The regression:
render red to target A, mutate the shared material to blue, render to target B,
submit one combined schedule — A must stay red and B blue. The planner tracks
per-use data versions, not a last value per object per frame.

**Bundles reset render-pass state.** After `executeBundles`, cached bindings and
pipeline state must be invalidated before subsequent direct draws — even for an
empty bundle sequence. A falsely warm state cache is a rendering defect, not an
optimization.

**CPU mutation and GPU-visible mutation are different histories.** An attribute
can change in a CPU array while the source deliberately has not requested an
upload. Coalescing across gaps must not expose edits that the reference would
leave GPU-stale. A scan detecting a CPU mutation is not permission to upload it
early.

**Synchronous work cannot acquire invisible yield points.** A cooperative runtime
does not authorize splitting an originally synchronous update across browser
turns — that can expose half-updated state, alter Promise ordering, or deliver
input inside a computation that previously ran to completion.

**One authoritative owner per state region.** JavaScript-owned, Wasm-owned, or
mirrored under an explicit epoch contract. A mirror is not a license for two
writers.

**Layout is explicit.** `AffineRows` is three `vec4<f32>` rows — 48 bytes. WGSL's
`mat4x3<f32>` is four padded columns — 64 bytes. Writing twelve tightly packed
floats into that type is wrong. Rust struct layout is not WGSL layout.

**`dispose()` releases a residency, not necessarily the public object.** A legally
reused material, texture, or geometry gets fresh backend residency without
changing its JavaScript identity, while GPU handle generations prevent late work
from targeting a recycled allocation.

**The backend choice is irreversible.** A renderer route is selected before its
canvas, context, or device is bound. If the renderer or its native state can
escape to opaque code that may call GL-specific methods, the exact component is
preserved from construction.

**Numeric fidelity is not negotiable for the headline score.** `f64` or retained
JavaScript for observable scalar operations; signed zero, exceptional values, and
observable ordering preserved. FP16, relaxed arithmetic, and changed reduction
order go in a separately labeled optional mode.

---

## Demonstration corpus

Eight preregistered headline workloads, chosen from the pinned example inventory
*before* any candidate performance is known:

| ID | Upstream example | What it forces |
|---|---|---|
| H1 | `webgpu_performance_renderbundle` | Existing bundles, heterogeneous geometry, per-object material data, static/dynamic toggles, a genuine WebGL backend mode |
| H2 | `webgl_marchingcubes` | Rebuilt procedural geometry, normals/colors, four legacy `ShaderMaterial` variants |
| H3 | `webgl_animation_multiple` | Multiple animated assets, independent poses |
| H4 | `webgl_shadowmap_performance` | Repeated scene/shadow work, many objects |
| H5 | `webgl_instancing_dynamic` | Already-instanced geometry with dynamic instance data |
| H6 | `webgl_postprocessing_advanced` | Multiple effects, render targets, pass ordering |
| H7 | `webgpu_skinning_instancing_individual` | Compute skinning, per-instance skeleton/morph data |
| H8 | `webgpu_compute_particles_fluid` | MLS-MPM compute, atomics, indirect dispatch, runtime particle counts |

H1 already defaults to 4,000 objects with render bundles. H8 already implements
multi-kernel GPU compute with atomic cell data. **Render bundles, compute
shaders, instancing, and WebGPU are baseline techniques, not this project's
novelty.** The baseline is strong on purpose.

H7 and H8 exist specifically so the suite is not just legacy object-submission
bottlenecks. They may prove hard to accelerate. **That difficulty is part of the
go/no-go test, not a reason to remove them after disappointing results.**

Additional required fidelity and control cases — published, but scored
separately — include `webgpu_vxgi_sponza`, SSGI/TRAA temporal effects, glTF
transmission and dispersion, clipping/stencil, compute cloth, video textures, a
deliberately GPU-bound full-screen shader as a **negative control**, and a small
ordinary scene as a startup/fixed-overhead control.

---

## Performance gates

What is measured: **completed logical frames per wall-clock second** in a bounded
steady-state pipeline — application updates, scene preparation, bridge work,
uploads, command preparation, GPU execution, and completion. Not the time to call
`queue.submit`. Not a GPU-only kernel timer. Not a matrix microbenchmark.

| Gate | Requirement |
|---|---|
| Equivalence | Visual, state, control, and lifecycle profile passes before any speedup counts |
| Eligibility | All reference-valid H1–H8 cells stay required; candidate failures never shrink the denominator |
| Headline floor | ≥ **2×** median paired completed-frame throughput in **every** required cell |
| Stratum target | ≥ **3×** geometric mean across required headline cells per device/browser stratum |
| Confidence | Paired 95% lower bounds meet the 2× floor and exceed 2× for the stratum geomean |
| Modern coverage | H7 and H8 stay in the required score |
| Tail | Frozen p95/p99 frame-time, missed-deadline, and input-response budgets |
| Controls | ≤ 5% sustained-throughput regression on valid control cases |
| Startup & memory | Frozen absolute device budgets; bounded caches; no OOM |
| Thermal | Phone floors hold in the sustained segment, not only the cool start |
| Deployment | Portable standalone passes independently |
| Ownership | Scored GPU work goes through the new renderer, not retained submission |
| Functional closure | All feature gates pass independently of the benchmark corpus |

The reference is chosen to be **strong**: R0 as authored, R1 a competent Three.js
configuration under a frozen bounded optimization recipe, R2 an equivalent
current Three.js WebGPU implementation. Selection happens by independent
reference-only calibration and is frozen before scored runs, so the denominator
cannot be cherry-picked from noisy trials.

Difficult control cases, regressions, startup costs, thermal behavior, and
failures get published alongside the wins.

### Device matrix

| Stratum | Browser evidence |
|---|---|
| iPhone 17 Pro Max | Safari on a pinned OS/browser build; Chrome checked separately |
| M5 MacBook | Safari and Chrome, separate runs on the same machine |
| NVIDIA discrete GPU | Chrome on supported Windows; Linux as a separate lane |
| AMD discrete GPU | Chrome on supported Windows; Linux as a separate lane |

Physical devices, not emulation. No fictional "Safari on Windows NVIDIA" target.
The upstream screenshot lane selects a software Vulkan ICD and special launch
flags — it is a regression oracle and can never substantiate a hardware speedup
claim.

---

## Kill gates

The project is designed to be stoppable early, before the full budget is spent.

1. **Foundation** — real browser task execution, wakeups, and cancellation
   through asupersync on Safari and Chrome, plus a legal WebGPU bridge. No
   ledger-only "execution."
2. **Bridge** — measure direct JS WebGPU, simple Wasm→host calls, bulk command
   transfer, and generated host submission on matched workloads. The chosen path
   must show an advantage *after its own overhead*.
3. **First product** — automatically transform complete H1 and H2 with every
   control, backend mode, and material variant intact, and hit ≥ 2× on the phone
   and M5 strata against an independently selected strong reference. If H1 is not
   viable, the project decision is reopened rather than swapping in an easier
   demo.
4. **Generalization** — repeat on held-out applications and at least one
   discrete-GPU stratum, with a measured end-to-end route for H7 and H8.
5. **Final** — exhaustive functional and accelerated-rendering closure, all
   upstream and integration tests, all performance conditions.

> If these fail, the project does not launch as a performance replacement. A
> foundation gate can stop development long before full compatibility is
> implemented.

---

## Phases

| Phase | Goal |
|---|---|
| **0** | Full source/feature inventory and executable foundation. Pin r186, census every export and registration, profile the strong references, freeze budgets, prove real browser task execution and the bridge counterexamples |
| **1** | Complete H1 and H2 as real applications with real, independently validated acceleration |
| **2** | Generalize aliases, effects, guards, ownership, dirty propagation, animation, and the full loader/exporter/codec inventory without losing public functionality |
| **3** | Full new-backend rendering and dynamic shader coverage, including portable legacy GLSL and runtime node registration |
| **4** | Feature, environment, and tool closure — exporters, media, XR, physics wrappers, editor/Inspector/devtools — plus demanding performance |
| **5** | Deployment and release validation: strict standalone closure, CSP, cold startup, device loss, mobile thermals, held-out workloads |

Phase 0 first. Frontier work must earn its complexity; optional research
(geometric algebra, neural rendering, differentiable scene optimization, general
JavaScript compilation) must never consume the compatibility reserve before the
core gates pass.

---

## Scope boundaries

Not built here: a JavaScript VM, a browser or driver stack, a general
WebGL-emulation layer, a native Metal/Vulkan/D3D backend, a physics engine, a new
editor, a marketplace, a render farm, or an ML optimization service.

**These are reuse decisions, not feature exclusions.** Existing editor
operations, physics wrappers, and legacy backend APIs are expressly in scope —
they are preserved by reusing the appropriate upstream, browser, or external
component, with the same engine version, inputs, solver semantics, and public
wrapper behavior. An acceleration pass cannot replace Rapier/Ammo/Jolt simulation
with a visually plausible approximation.

There are no permanent "core only," "no exporters," "no XR," "static shaders
only," or "all the tests we happen to pass" release variants labeled full
compatibility.

---

## Dependencies

Small, pinned, audited. First-party semantic, numerical, compiler, scene, and
resource code uses `#![forbid(unsafe_code)]`; the browser binding family is a
named external boundary rather than a claim that nothing beneath it is unsafe.

**Allowed:** `wasm-bindgen`/`js-sys`/`web-sys` aligned with asupersync's
boundary; shared serialization for build manifests and diagnostics (never
per-frame JSON); [Naga](https://wgpu.rs/doc/naga/) as a normally build-time
shader compiler and validator; the existing JavaScript bundler/parser toolchain
for source ingestion; required asset codecs selected by dependency closure;
version-pinned upstream Three.js components for exact compatibility; and
test/fuzz/browser-automation tools outside the shipping path.

**Not in the execution core:** Bevy, a general `wgpu` stack, a new generic ECS,
Tokio, a mandatory Rayon pool, native C/C++ FFI, BLAS, or a Python interpreter.

Sibling Franken libraries — [FrankenNumPy](https://github.com/Dicklesworthstone/franken_numpy),
[FrankenSciPy](https://github.com/Dicklesworthstone/frankenscipy),
[FrankenTorch](https://github.com/Dicklesworthstone/frankentorch),
[FrankenLibC](https://github.com/Dicklesworthstone/frankenlibc) — are resources,
not assumptions. Reuse is decided per crate or per kernel with target and
dependency checks. Importing everything would undermine both the size budget and
the foundation requirement.

A note on the shader frontend: "run all Three.js GLSL through Naga" is not a
plan. Naga's GLSL frontend targets Vulkan GLSL, not a turnkey WebGL GLSL ES
1.00/3.00 entrance. The ESSL preprocessing, semantic normalization, stage
interface reconstruction, and uniformity work are first-party, and they are the
largest single uncertainty in the budget.

---

## Repository layout

```text
franken_threed/
├── COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENTHREED.md   # the technical constitution
├── AGENTS.md                                               # operating contract for AI coding agents
├── README.md
├── LICENSE                                                 # MIT + OpenAI/Anthropic rider
├── .gitignore
├── .gitattributes
└── .beads/                                                 # br issue graph: the plan as 212 beads (see AGENTS.md)
```

Everything else — the Cargo workspace, the pinned upstream checkout, the feature
manifest, the conformance harness, the benchmark corpus — is future work. The
work itself is tracked as beads: run `br ready --json` to see what is unblocked.

---

## Reading paths

- **Evaluating whether this is worth doing:** plan §1 (executive decision), §2
  (what source inspection changes), §16.7 (kill gates), §20 (risks).
- **Understanding the compiler:** §4 (compilation strategy), §6 (IR and execution
  architecture), §4.3–4.5 (islands, write barriers, guards).
- **Understanding the compatibility contract:** §5 in full, especially §5.1 (four
  promises), §5.11 (full-surface scope), §5.12 (feature-family map), §5.16
  (functional vs accelerated closure).
- **Understanding the renderer:** §8 (WebGPU renderer), §9 (materials, TSL,
  legacy shaders), §6.7 (queue ordering), §8.5 (hazards).
- **Understanding how claims get verified:** §14 (testing), §16 (performance
  methodology), §22 (definition of done).
- **Working in this repo as an agent:** [`AGENTS.md`](AGENTS.md).

---

## FAQ

**Can I use this today?** No. There is no code.

**Is this a fork of Three.js?** No. It is a compiler and execution core that
consumes an unmodified pinned Three.js, retains upstream components where exact
behavior demands them, and reports exactly what it retained.

**Why not just use the Three.js WebGPU renderer?** You should — it is one of the
required baselines. Three.js already ships a WebGPU renderer, render bundles,
compute-based fluid simulation, and compute-driven instanced skinning. This
project has to beat that, not rediscover it.

**Why not compile all of JavaScript to Rust?** Because that is a different,
larger, worse project. The compiler understands Three.js semantics and a
restricted set of numeric operations. Dynamic UI and general language features
stay in the browser's JavaScript engine, where they belong.

**Why is the plan so preoccupied with what it cannot claim?** Because the
difference between a real result and a demo is entirely in the parts that are
easy to leave out: the failed cells, the retained components, the startup cost,
the thermal segment, the untested API, the control case that got slower.

**Will it hit 3×?** Unknown. Nothing has been measured. The plan states the gates
and states that failing them means no performance-replacement launch, even if the
implementation is otherwise impressive.

---

## License

MIT with an OpenAI/Anthropic rider. See [`LICENSE`](LICENSE).
