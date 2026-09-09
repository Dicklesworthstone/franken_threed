# Comprehensive Plan for the Design of FrankenThreeD

**Version:** 2.0, full Three.js functionality and zero permanent feature exclusions  
**Date:** September 9, 2026  
**Review scope:** Full upstream functional surface, exact legacy/backend compatibility, all addons and browser/tool integrations, feature-level acceptance, and preservation of the v1.1 compiler/GPU correctness rules  
**Status:** Source-informed architecture and implementation plan; not an implementation or a benchmark report  
**Initiator:** Jeffrey Emanuel  
**Proposed project/repository name:** FrankenThreeD / `frankenthreed`  
**Upstream compatibility anchor:** Three.js r186, commit `148ef33ecb6d2502ff796d4554abd1549c95d519`  
**Implementation budget:** 215,000 planned new Rust lines, including tests and project-required foundation work; 30,000 reserve; 245,000 maximum planned total  
**Product thesis:** Preserve all existing functionality of the pinned Three.js release. Specialize eligible execution into data-oriented Rust/Wasm and WebGPU without making unsupported optimization a missing application feature.
**Functional scope:** Entire r186 source/package/addon surface and its official integration workflows, not only APIs exercised by tests or selected demonstrations.
**Release rule:** Full functional coverage, accelerated rendering coverage, and measured performance are separate mandatory gates; none substitutes for another.

> **Build a compiler-backed execution system, not another general-purpose game engine.**
>
> A successful FrankenThreeD preserves the pinned release's complete functionality, including dynamic behavior and integrations that cannot truthfully become WebGPU operations. Its optimized execution core is Rust/Wasm and WebGPU; necessary exact compatibility components are isolated, version-pinned, dependency-pruned, and explicitly reported. Self-contained applications remain a primary deliverable, alongside a full-compatibility package/networked mode for applications whose behavior inherently needs live resources or host services.
>
> **No permanent feature cuts.** An optimization refusal is not an application-feature refusal. “Not in our demos,” “not tested upstream,” “uncommon,” “dynamic,” and “not yet ported” are not release exemptions. A valid upstream operation must have a working production implementation under the same applicable host prerequisites. Retained compatibility execution is never misrepresented as native Rust implementation or as a FrankenThreeD WebGPU speedup.
>
> Version 2.0 deliberately replaces v1.1's product-wide no-WebGL rule and restricted-application escape clauses. The accelerated backend stays WebGPU-only; the *product* retains the exact backend behavior needed for complete Three.js compatibility. This avoids imposing WebGL restrictions on WebGPU execution while also avoiding missing features.

---

## Contents

1. [Executive decision](#1-executive-decision)
2. [What the source inspection changes](#2-what-the-source-inspection-changes)
3. [Product contract and developer experience](#3-product-contract-and-developer-experience)
4. [The compilation strategy](#4-the-compilation-strategy)
5. [The compatibility contract](#5-the-compatibility-contract)
   - [Full-surface scope](#511-full-surface-scope-and-no-cut-rule)
   - [Mandatory feature-family map](#512-required-feature-families-and-implementation-paths)
   - [Exhaustive feature manifest](#515-one-exhaustive-feature-manifest-not-an-expanding-bureaucracy)
   - [Functional and accelerated-rendering closure](#516-full-compatibility-must-not-conceal-unfinished-accelerated-rendering)
6. [Internal representation and execution architecture](#6-internal-representation-and-execution-architecture)
7. [CPU data structures and mathematical choices](#7-cpu-data-structures-and-mathematical-choices)
8. [The WebGPU renderer](#8-the-webgpu-renderer)
9. [Materials, TSL, and legacy shaders](#9-materials-tsl-and-legacy-shaders)
10. [Assets and self-contained HTML](#10-assets-and-self-contained-html)
11. [Asupersync integration and cancellation](#11-asupersync-integration-and-cancellation)
12. [Hardware and browser specialization](#12-hardware-and-browser-specialization)
13. [Dependency and library-reuse policy](#13-dependency-and-library-reuse-policy)
14. [Full upstream testing and stronger equivalence tests](#14-full-upstream-testing-and-stronger-equivalence-tests)
15. [The demanding demonstration corpus](#15-the-demanding-demonstration-corpus)
16. [Performance methodology and go/no-go gates](#16-performance-methodology-and-gono-go-gates)
17. [Workspace and size budget](#17-workspace-and-size-budget)
18. [Implementation sequence](#18-implementation-sequence)
19. [Execution-ready work breakdown](#19-execution-ready-work-breakdown)
20. [Principal risks and scope boundaries](#20-principal-risks-and-scope-boundaries)
21. [First implementation session](#21-first-implementation-session)
22. [Definition of done](#22-definition-of-done)
23. [Source register](#23-source-register)

---

## 1. Executive decision

### 1.1 The recommendation

FrankenThreeD should be a **Three.js-compatible application specializer with a Rust/WebGPU execution core**.

Its main innovation should be recovering information across ordinary library boundaries: which objects share a rendering program, which values actually change, which changes can affect which passes, which computations have observable intermediate results, and which object-oriented operations can become one bulk operation.

The distinction is decisive. A translation from `object.position.x = value` into an equally fine-grained Wasm call merely relocates overhead. A compiler that turns thousands of such operations into a contiguous update kernel, legally versioned bulk uploads, and a persistent GPU schedule can remove substantial work. One upload or one submission per frame is an optimization opportunity, not a correctness rule: an application can observe several different versions of a resource within a single frame.

The product should have three mutually reinforcing advantages:

- **Existing authoring knowledge remains useful.** Humans and coding agents continue using the complete pinned Three.js surface, including its objects, loaders, exporters, controls, materials, animation, TSL, browser integrations, and extension points.
- **Execution is specialized to the actual application.** The output does not carry a universal renderer's entire hot-path decision machinery when the application only needs a small, stable subset.
- **Equivalence and performance travel with the build.** The developer gets an actionable compatibility report, a description of what was specialized, and reproducible comparison commands.

These are design objectives. No speedup has been measured for FrankenThreeD at the time of this plan.

### 1.2 The central performance hypothesis

Many worthwhile Three.js applications repeatedly perform expensive work on a highly structured scene: traversing objects, rebuilding transforms, updating similar animation state, discovering the same material configuration, issuing repeated state changes, and submitting related geometry separately.

The hypothesis is that **application-level specialization plus better data movement** can remove enough of that work to achieve the required improvement. It must be tested against current Three.js, including its WebGPU renderer and existing optimizations.

For a simplified serial critical path, if fraction `p` is accelerated by factor `k`, and new overhead is fraction `h` of original time:

```text
speedup = 1 / ((1 - p) + p/k + h)
```

Even infinitely fast replacement of one part cannot produce 3× unless approximately two-thirds of the relevant critical path is removable, before new overhead. Real rendering overlaps CPU and GPU activity, so the actual experiment must measure the pipeline rather than simply add CPU and GPU timings.

An illustrative, unmeasured example: reducing CPU preparation from 18 ms to 4 ms can transform a scene whose GPU takes 5 ms. Reducing CPU preparation from 2 ms to 0.5 ms does little for a scene whose unchanged GPU work takes 18 ms.

**The project must therefore optimize work, not advertise a language substitution.**

### 1.3 Non-negotiable launch gates

The exact benchmark contract is in Section 16. Its essential requirements are:

1. Automatically transform the preregistered demanding applications, with the same input source on both sides except for explicitly shared harness instrumentation.
2. Pass the full feature/behavior inventory, complete upstream tests, and additional integration gates. Every scored application also passes image/state/interaction checks before timing results count; a green test suite alone does not establish full functionality.
3. Achieve at least **2× completed-frame throughput on every baseline-valid headline workload/device cell**, and at least **3× geometric-mean throughput within each primary device/browser stratum**, with the confidence rules in Section 16. A candidate crash, unsupported path, fidelity failure, or timeout is a failed cell, not a reason to remove it from the denominator.
4. Compare against the strongest eligible reference selected by independent calibration under a frozen, bounded optimization recipe. Reference eligibility requires the same application-observation contract, not merely similar pixels, and includes a competent Three.js WebGPU configuration when available.
5. Publish the difficult control cases, regressions, startup costs, thermal behavior, and failures as well as the wins.
6. Do not launch the project as a performance replacement if these requirements fail. A foundation gate can stop development long before full compatibility is implemented.

This is deliberately stricter than finding one convenient demo that improves by 3×. It is not a claim that every possible GPU shader can run three times faster on the same hardware.

### 1.4 Keep the scope concentrated

The project is not a replacement for JavaScript, the browser, native GPU drivers, or the third-party engines used by upstream integrations. It preserves the existing Three.js editor/tool workflows rather than building a new editor, and preserves existing physics wrappers rather than inventing new physics. It does not need a new entity-component-system framework, distributed database, reinforcement-learning controller, or universal symbolic optimizer. These are implementation-reuse choices, not exclusions of existing Three.js functionality.

The first-class objects are modest: **application modules, semantic scene state, update kernels, render/compute passes, resources, and compiled schedules**.

---

## 2. What the source inspection changes

### 2.1 Three.js is already more modern than the motivating critique suggests

The direction of the critique is useful: older graphics contracts and repeatedly interpreted abstractions can constrain performance. However, the implementation must not assume that Three.js cannot exploit WebGPU.

Three.js already exposes a WebGPU renderer, and its current examples include a WebGPU-only render-bundle performance example, compute-based fluid simulation, and compute-driven instanced skinning. The render-bundle example explicitly constructs `BundleGroup` objects; the fluid example already uses storage buffers, atomics, multiple compute kernels, indirect dispatch data, and asynchronous compilation. [S3], [S7], [S9], [S10]

Consequently, render bundles, compute shaders, instancing, and WebGPU support are **baseline techniques, not FrankenThreeD's novelty**. FrankenThreeD must combine them with program specialization and more efficient ownership/layout of application state.

For terminology, WebGL 2 is derived from OpenGL ES 3.0, rather than being literally the desktop OpenGL 3.3 API. That does not invalidate the broader concern about its programming model. [S4]

### 2.2 Pin an actual release, not an evolving branch

The inspected latest release is r186, published September 8, 2026. Its annotated tag resolves to commit:

```text
148ef33ecb6d2502ff796d4554abd1549c95d519
```

That is the initial compatibility and benchmark anchor. The tag-object hash is not the source commit. Release updates should be deliberate compatibility events, not automatic movements of the oracle. [S1], [S2]

### 2.3 Asupersync's browser contract needs an execution proof

The inspected browser documentation explicitly distinguishes the shipped JS/TS ownership and host-adapter boundary from full native-style Rust task execution. In particular, its documented ABI task-spawn operation allocates a task handle rather than accepting an arbitrary Rust future, and the Rust-authored browser bootstrap is characterized as a preview with a remaining host-services gap. [S14]

This is not a reason to replace Asupersync. It is a reason to put **real browser-host execution, wakeups, cancellation, and cleanup** on the critical path and contribute any necessary work to Asupersync itself.

Passing a `cargo check` for Wasm, or closing an ownership ledger, is not equivalent to proving that browser tasks actually executed and drained.

### 2.4 The full Three.js test suite is not a single renderer certification

The current package scripts distinguish core unit tests, addon unit tests, end-to-end screenshots, WebGPU screenshots, and tree-shaking checks. Unit tests import individual source files, including renderer-related internal utilities, directly. Redirecting only the package name `three` will not redirect those tests. [S5], [S27]

The E2E runner also excludes numerous difficult examples, including temporal effects and compute scenes. Its inspected launch configuration selects a software Vulkan ICD and uses special browser flags. That lane is useful as an upstream regression oracle, but cannot substantiate real-hardware speedup claims. [S6]

FrankenThreeD therefore needs both the **complete upstream test inventory** and a **separate real-device comparison harness**.

### 2.5 The sibling libraries are resources, not assumptions

The inspected `fnp-linalg` manifest has an unconditional Rayon dependency. The numerical libraries also reference older Asupersync versions than the inspected Asupersync root. FrankenTorch's documented implementation path is CPU-first. FrankenLibC targets a Linux/glibc ABI, not the browser. [S16], [S17], [S18], [S19], [S31]

Reuse must be decided at the level of individual crates or kernels, with target and dependency checks. Importing everything would undermine both the size budget and the programming-foundation requirement.

### 2.6 Do not freeze WebGPU assumptions in 2024

Current Chrome documentation describes immediates and subgroup-size control. These are useful optional specialization opportunities, not capabilities to assume on every browser or adapter. The system must query actual features, WGSL language features, and limits, and generate a legal plan for that environment. [S21], [S22]

The architecture should not depend on universal support for native-style multi-draw-indirect, arbitrary bindless resources, mesh shaders, ray-tracing hardware, or shared CPU/GPU memory.

### 2.7 The first examples already require nontrivial shader and host support

The original milestone order underestimated its own first demonstration. H1 uses `MeshToonNodeMaterial` and the Inspector, and its controls include a navigation-based backend selection. H2 starts with an environment-lit Standard material and exposes Lambert, Phong, texture/color, and four legacy `ShaderMaterial` choices. A basic unlit renderer cannot preserve those demonstrations and all their application controls. [S7], [S29], [S40]

The corrected sequence brings the required node/material slice, environment handling, finite ESSL shader corpus, and a minimal sound update-island implementation into the first product gate. Broader generalization still comes later. Version 2.0 also preserves H1's genuine WebGL backend choice through the exact compatibility component. The WebGPU setting is scored for acceleration; both backend settings and their navigation/control behavior are required functional cases. A performance configuration is not permission to remove a UI branch.

### 2.8 Compiling to Wasm does not remove the browser boundary

The browser still owns GPU validation, driver compilation, presentation, and host scheduling. The engineering goal is to remove redundant semantic work and unnecessary crossings, not to claim that WebGPU or Wasm has no abstraction cost. The compiled program also has residual JavaScript, synchronization, memory, and startup costs; every one belongs in the experiment.

### 2.9 The full-surface audit changes the compatibility architecture

Version 1.1 was not a complete Three.js-functionality plan: it tied too much scope to an admitted application/test corpus, allowed some dynamic shader/node/decoder paths to be diagnosed rather than implemented, and excluded genuine WebGL/backend modes. Passing the entire active upstream suite would not fix features that the suite does not exercise.

The pinned package exposes the root ESM/CommonJS entry, `three/webgpu`, `three/tsl`, the addon aggregate, wildcard addon paths, and source paths. The addon aggregate includes exporters, alternate renderers, physics wrappers, WebXR, controls, and many other categories; it is *not* the sole addon entry point. `Three.WebGPU.js` also exports backend classes, node builders, storage objects, render pipelines, and loaders. Those are part of the actual surface, not hypothetical third-party internals. [S5], [S52], [S53]

The repository contains an editor, devtools, TSL tooling, documentation/manual examples, and the test/examples trees. Existing editor behavior includes scripting, history, loading/storage, XR, renderer changes, and animation-related state. Reusing these tools is smaller than replacing them, but integration and behavioral verification are mandatory. [S54], [S57]

A fundamental distinction follows: `getContext()` returning a real WebGL context, shared-context `resetState()`, and synchronous render-target readback must not be faked by an unrelated WebGPU object or a Promise. Three.js's own XR integration includes a construction-time WebGL fallback. Complete compatibility therefore needs explicit backend-preserving execution, not a broader list of typed refusals. [S55], [S56]

Sections 5.11–5.17 define the full feature scope and closure rules. Sections 3.3 and 6.9 keep exact compatibility outside the specialized hot path. No capability, performance, or scope claim in this document overrides those rules.

---

## 3. Product contract and developer experience

### 3.1 Input and output

**Input:** a Three.js project, HTML/import-map application, module entry point, or library consumer targeting a supported pinned release. Support the package's existing ESM, CommonJS, source, addon, WebGPU, and TSL entry points in the environments in which upstream supports them. A build transformation must not turn an otherwise working cold numerical/server-side import into a mandatory browser/GPU initialization. [S5]

**Output:** a full-compatibility application package, with self-contained HTML as the primary deployment form for applications with a closable code/resource graph. Also provide normal networked/module output for live-data, external-service, runtime-import, and library use cases. A deployment limitation is reported as such, not relabeled as an unsupported Three.js feature.

The standalone HTML embeds all required reachable code, compatibility components, Wasm, WGSL, CSS/fonts, codecs, static assets, and notices. A proof of unreachability may omit a component; an unobserved branch in a captured trace may not. User-provided files and live media remain live inputs. Intentional document navigation is separately verified.

This is not a `Scene.toJSON()` exporter. Callbacks, dynamic imports permitted by the deployment, input, animation, loading, procedural geometry, export, editor-generated scripts, and runtime state changes remain executable. Full functionality does not require every feature to be included in every small static page; it requires the product to implement every feature and include every reachable dependency for that page.

### 3.2 Proposed command surface

These proposed commands do not exist yet:

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

`--compat full` is the default, not an optional premium or later phase. A developer may additionally request `--require-accelerated-webgpu` for a declared renderer session/workload. That diagnostic strict mode fails if its acceleration requirement cannot be met; it does not redefine the full product's functional scope. The full artifact remains the performance/deployment reference, including all reachable compatibility bytes.

An application profile selects inputs, settings, permissions, and performance workloads; it cannot delete valid upstream API contracts or reachable controls from the full-compatibility claim. Existing build tools are thin frontend adapters to the same compiler. Ordinary bundler, module, Node-side utility, and upstream editor/player integration are required delivery paths, not excuses to invent another language toolchain.

### 3.3 One optimized backend, complete compatibility routing

Maintain **one new accelerated GPU backend: Rust/Wasm plus WebGPU**. Do not build a second new general-purpose Rust renderer. Compose it with version-pinned production compatibility components so functional coverage is complete without forcing modern execution to obey WebGL's limitations.

| Execution route | Responsibility | What it can legitimately claim |
|---|---|---|
| Specialized WebGPU | Proven update islands, packed state, compiled schedules, efficient GPU execution | Verified accelerated Rust/Wasm/WebGPU execution |
| General WebGPU | The same new backend with conservative state synchronization, runtime variants, and unoptimized scheduling | Working WebGPU execution when specialization is unavailable; speed measured independently |
| Retained JS/host component | Existing public math/scene utilities, loaders/exporters, controls, DOM/SVG/audio, tooling, and appropriate shader-construction code | Full component functionality, not a Rust rewrite |
| Exact backend component | Pinned upstream WebGLRenderer/WebGLBackend, or pinned backend/renderer machinery required by genuinely opaque native-backend/custom-backend contracts | Exact backend behavior and functional compatibility; never counted as the new renderer's acceleration |

These routes exist in normal production builds when reachable, not only in tests. Backend selection happens at a sound source boundary, normally renderer construction, with the identity/resource rules in Section 6.9. The unit may be an entire connected renderer/resource group rather than a single draw. Calling an unfamiliar addon, generating a shader at runtime, or loading an unseen model cannot result in an omitted feature. Keep an adequate general execution path or the exact backend path selected early enough to preserve behavior.

The complete public function set is available on all routes applicable to that source backend. An unexpected call is not handled by fabricating a GL object, silently changing a synchronous result into a Promise, or dropping a pass. Native handles, synchronization, extension queries, custom backend injection, explicit backend selection, and external-context interoperability can require the exact route. Ordinary unimplemented materials, textures, TSL nodes, or passes are *engineering gaps*, not intrinsically backend-bound exceptions.

**Prevent a compatibility-only wrapper from masquerading as this project.** All upstream-standard rendering features with expressible WebGPU semantics must also have a working path through the new renderer by release. The H1–H8 scored workloads execute entirely through that new GPU backend, with a call-path check that rules out retained renderer submission. Retained cold helpers are allowed and charged. Exact-backend cases are separately reported and tested; they cannot satisfy the accelerated-rendering gate or rescue a failed speedup.

**No WebGPU prerequisite for unrelated functionality.** A CSS/SVG-only scene, numerical utility, loader/exporter, WebAudio component, or upstream-supported WebGL application must continue to function when WebGPU is absent. Select the source-compatible host route and avoid initializing unused GPU/Wasm services. The standalone accelerated profile still has to qualify on its named modern devices independently.

This is complete functionality through deliberate composition, not a lowest-common-denominator unified GPU API. Unused exact components are removed from provably closed builds. Where their potential use is unresolved, preserve them and report the real cost.

### 3.4 Bootstrap and synchronous API shape

WebGPU device acquisition and Wasm instantiation are asynchronous preparation; many public constructors and operations are synchronous. Preserve the original module graph, live bindings, import identity, constructor/return types, side effects, callback registration, and source-required host activation.

Use the source's existing asynchronous startup boundary for accelerated device preparation when available. A compiler-owned pre-entry bootstrap is allowed only for an application whose observations admit the delay. CPU objects, utilities, controls, and the intended canvas can exist before accelerated device readiness, but bounded deferred renders require correct per-use snapshots, not merely a mutable scene pointer.

**When deferral is not equivalent, use the exact compatible route from construction.** This includes pre-existing GL contexts, genuine native-handle observations, synchronous GPU readback, and startup whose observations cannot tolerate the new asynchronous dependency. These are no longer full-product admission failures. A compiler flag demanding exclusively accelerated WebGPU may report the unmet optimization requirement, separately from the functioning compatibility build.

Conservatively retain synchronous JS implementations for CommonJS/Node utility entry points until equivalent synchronous Wasm availability is established. Do not make `require('three')` return a Promise. Do not access `window`, `document`, or `navigator.gpu` just because a server imported `Vector3` or an exporter whose source path does not need them. Preserve the pinned module's actual environment requirements, not an imagined universal headless renderer.

Tests include cycles, top-level await, scripts before/after entry, `DOMContentLoaded`, immediate canvas access, first user input, and initialization errors. Media playback, pointer lock, fullscreen, XR session entry, and other activation-sensitive operations stay on their valid source host path. Replaying a captured event after awaiting initialization does not recreate transient activation. [S35]

Startup measurements include required compatibility code and all preparation. Runtime failures must preserve the source API's error/cleanup contract; they must not produce a partially initialized facade that silently drops valid operations.

### 3.5 Reports that make the system useful to coding agents

Every build should explain, in human-readable text and stable JSON:

- the selected Three.js version and asset/module hashes;
- which work was specialized, executed by the general backend, retained in JS, or kept on an exact backend;
- distinct functional gaps, host/source limitations, and optimization-only barriers, with no conversion of one status into another;
- estimated structural costs, clearly distinguished from measured timings;
- precise source spans and reasons for specialization barriers;
- changes needed to unlock a fast path without weakening behavior;
- the selected device profile and optional capability variants.

For example, a report might explain that a typed array escapes to an unknown plugin, so safe change detection requires a conservative copy; it should identify that escape and the copied byte count. It must not claim a measured improvement before a benchmark runs.

No LLM is necessary in the deployed application or compiler correctness path. Agents can interpret reports and propose source edits, but those edits still face the same verification gates.

### 3.6 What makes the project accretive

The reusable value is not just a renderer. It is a modest, testable compiler boundary between a popular authoring API and a specialized execution plan. An improvement to a write-barrier analysis, a resource-lifetime rule, a material specialization, or a bulk animation kernel can accelerate many independently authored applications without retraining their authors.

The user should receive a benefit from ordinary, idiomatic Three.js. Requiring a wholesale rewrite into a new DSL would sacrifice the main reason for building the project.

---

## 4. The compilation strategy

### 4.1 Do not write a general JavaScript-to-Rust compiler

The compiler understands Three.js-related semantics and a restricted, useful set of numeric operations. It does not attempt to reimplement all JavaScript language behavior in Rust.

Use the existing project's build process to normalize TypeScript, JSX, and module syntax when present. A narrow build-time JavaScript parser/helper supplies a structured module representation to the Rust compiler. Initial preference: the existing Rollup ecosystem plus an Acorn-class parser, pinned and audited during Phase 0. The exact parser is a bounded dependency decision, not a reason to build a new frontend language stack.

The optimizer, scene analysis, resource scheduling, and code generation belong in Rust. Generated Rust kernels are compiled into an application-specific Wasm module with the pinned nightly toolchain. Dynamic UI and language features remain in the browser's JavaScript engine.

### 4.2 The pipeline

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

### 4.3 The unit of specialization: a closed update island

An island is a set of objects and computations whose inputs, outputs, mutation paths, and observation boundaries are known. It may contain thousands of instances, a skeletal animation batch, or a procedural geometry kernel.

The compiler should admit an island only when it can answer:

1. Who can mutate each value, including aliases?
2. Who can read intermediate state, and when?
3. Which operations are pure numeric work, and which have effects?
4. Which numeric semantics are required?
5. What invalidates the specialization?

A source trace can suggest an island, but observing a property stay constant for 1,000 frames is not proof that it is constant. Profiling information can choose among already-correct plans; it cannot justify erasing a possible behavior.

### 4.4 Source-level write barriers, not universal Proxies

Instrument provably covered writes at build time. Coalesce the resulting field changes into a frame or observation-boundary batch. Account for aliases, compound assignments, evaluation order, setters, method overrides, and exceptions.

Do not wrap the entire application in JavaScript Proxies. That approach risks changing observable behavior and introduces overhead in precisely the paths we want to remove.

Unknown or escaped mutations select a conservative ownership mode. In particular, arbitrary writes through a retained typed-array alias cannot generally be discovered by instrumenting only the original variable. Respect Three.js's explicit upload-version contract where it applies; where additional synchronization is needed, pay for a scan/copy or retain the state in JavaScript and disclose the cost.

The compiler must not claim work proportional only to the number of changed fields unless its mutation coverage actually supports that claim.

### 4.5 Guarded specialization and safe deoptimization

A specialized island may depend on a material configuration, object shape, or callback identity. A guard validates these assumptions before the affected work executes. On guard failure, invalidate the relevant plan and use the compatibility execution path until a new specialization is ready.

Guard failure must not cause the application to run an update twice. Side effects already performed cannot be undone by replaying the callback. Therefore, pure kernels are guarded before execution, and effectful blocks remain ordered barriers unless a stronger transformation is justified.

Reconstruction of observable objects is permitted at a declared materialization boundary, but must preserve stable identity and existing references. Replacing a referenced `Vector3` with a newly allocated lookalike is not equivalent.

### 4.6 First compilation targets

Start with constructs that have a favorable benefit-to-complexity ratio:

| Pattern | Initial lowering |
|---|---|
| Many object transforms updated from time and numeric parameters | Bulk CPU Wasm kernel and packed transform upload |
| Repeated geometry/material with differing transforms or colors | Instance stream or equivalent grouped draw plan |
| Stable material/lighting combinations | Prebuilt shader variants and bindings |
| Known AnimationMixer/track paths | Batched interpolation and pose propagation, preserving public events |
| Known procedural geometry addon | Safe Rust kernel behind the same API contract |
| TSL graph with known node semantics | WGSL generation and persistent resource/pass plan |
| DOM controls and unusual effectful callbacks | Retain JavaScript and synchronize at explicit boundaries |

General numeric-loop lifting follows these recognized paths; it must not become a prerequisite for the first useful result.

### 4.7 Module identity and the minimum compiler needed before benchmarking

Resolve the actual module graph, including relative imports, import maps, addon-internal imports, dynamic imports with a finite closure, and the project's existing build transforms. Preserve ES-module singleton identity and live bindings. `three`, `three/webgpu`, `three/tsl`, and relevant `three/src/*` paths must resolve through a coherent versioned adapter set, not accidentally instantiate incompatible duplicate cores. Do not deduplicate two independently intended application instances merely because their source bytes happen to match.

Use exact pinned module/API fingerprints for recognized substitutions. A modified addon loses optimization recognition and remains executable through its ordinary JS/compatible backend path. An unrecognized Three.js revision requires its actual versioned compatibility component and a new parity target before a full-version support claim; a matching class name alone is insufficient. Do not reject a valid addon on the pinned version merely because it was not recognized by the optimizer. Keep class/prototype identities coherent across retained modules and the facade.

The first compiler slice must already cover the guards, alias checks, input packing, output materialization, and conservative path required by the H1 update loop. Those costs cannot be implemented after the first performance result. The general island optimizer is later work, but the first kernel must be derived from source semantics, not selected by an example filename, benchmark flag, or a memorized input trace.

### 4.8 Specialization is partial evaluation with an explicit environment

Offline evaluation is allowed only for inputs proven independent of unbound runtime state. A captured DOM, viewport, adapter, query string, locale, clock, random value, or successful network response is not automatically an application constant. Preserve random-source consumption and observable construction/ID ordering when baking work would otherwise advance them differently. Runtime-created objects and unobserved branches remain executable; a capture is not a complete application specification.

Pure islands need a concrete domain contract: ordinary guarded data properties, admitted number/array types, known call targets, and covered observations. A guard must not invoke a getter or Proxy trap an extra time. Prototype changes, replaced methods, reflection, unknown callbacks, `eval`, and escaped aliases may force JavaScript ownership. Do not replace public data properties with accessors when descriptors, enumeration, or external references can observe the difference.

Emit both the optimized kernel and its production conservative path. Define guard failure, materialization, and re-entry before emitting optimized code. The cost model includes guard scans and facade traffic, not just the inner Rust loop. Small or frequently observed islands should remain JavaScript when crossing/packing costs dominate.

### 4.9 Synchronous work cannot acquire invisible yield points

A cooperative runtime does not authorize splitting an originally synchronous update across browser turns. Doing so can expose half-updated state, alter Promise ordering, or deliver user input inside a computation that previously ran to completion. Chunking with a host yield is allowed for source-asynchronous work, isolated worker jobs, or a transformation whose interleaving equivalence is established; otherwise the admitted synchronous kernel runs to completion. [S35]

Cancellation checks in such a kernel cannot promise that a main-thread cancellation callback executes before the browser gets control again. Report this bound honestly. Preserve source exceptions and partial effects: an effectful loop that throws after several mutations cannot be replaced by an all-or-nothing bulk commit unless those intermediate effects are unobservable. Partial numeric work may use scratch output and commit only where that is actually equivalent.

### 4.10 The insight worth defending

The most important optimization is often **removing a question from the frame loop**.

If source analysis has already established a material variant, resource layout, object grouping, or dependency, the application should not rediscover it 120 times per second. The compiled artifact should contain the answer and a small invalidation condition.

That is a more substantial architectural difference than replacing individual JavaScript functions with Rust functions.

---

## 5. The compatibility contract

### 5.1 Four distinct promises

Keep four statuses separate:

| Promise | Meaning |
|---|---|
| API compatibility | Every existing pinned-release API contract has a working production path under its source-supported host prerequisites, whether or not it is tested upstream |
| Application equivalence | Preserve the complete application feature/control surface; use bounded observation contracts only to validate optimization and deployment, not to delete features |
| Implementation ownership | Which behavior executes in new Rust, retained JavaScript, or a browser adapter |
| Acceleration | The actual matched benchmark result on a named device/browser/workload |

Passing a test through a retained JavaScript implementation is legitimate compositional compatibility, but it is not evidence that a Rust implementation of that feature exists. Likewise, a fast renderer does not imply that every application callback was accelerated.

### 5.2 The JavaScript object surface

Compatibility must cover object identity, prototypes and `instanceof`, property descriptors and all existing property/method contracts, method overrides, stable nested references, cloning/serialization, event ordering, loading/error callbacks, disposal, and mutation visibility.

Preserve `position`, `quaternion`, `rotation`, `scale`, matrix update flags, geometry attributes, draw ranges, material groups, layers, visibility, and render ordering according to the pinned upstream behavior. Preserve the observable relationship between Euler rotation and quaternion changes, not just the final orientation in one frame.

Tests must include callbacks that mutate scene state during rendering and subsequent reads of world matrices. A cached renderer must not hide state updates that upstream makes observable.

Retain a pinned compatibility implementation for cold or highly reflective surfaces when that is smaller and safer than rewriting them. Any such modules are included in normal builds when required, carry their licenses, and appear in the ownership report.

### 5.3 One authoritative owner for each state region

Every region is assigned one of three explicit modes:

- **JavaScript-owned:** public objects are authoritative; the execution core receives snapshots or deltas at known boundaries.
- **Wasm-owned:** an admitted island has packed authoritative state; supported observations materialize or access its current values through generated bindings.
- **Mirrored with an epoch contract:** two representations exist, but only one may author a particular epoch, and publication is explicit.

A mirror is not a license for two writers. Authority transfers must be ordered, versioned, and tested. The general fallback is JavaScript ownership, not an optimistic guess that an escaped object will not be modified.

### 5.4 Numeric fidelity

Public Three.js numeric operations have JavaScript-number behavior, while buffer storage and GPU execution introduce their own precision boundaries. Do not replace all public arithmetic with `f32` merely because the GPU uses it.

Use `f64` or retained JavaScript for observable scalar operations; convert to GPU representations at the same admitted boundaries as the reference. Preserve signed zero, exceptional values, operation ordering where observable, and relevant rounding behavior. JavaScript integer coercions, shift counts, remainder, and `Math.min`/`Math.max` behavior need explicit lowering; Rust casts and similarly named arithmetic methods are not substitutes for those semantics. Unknown/non-numeric operands stay on the JavaScript path rather than introducing a Rust panic or an unintended coercion. [S47] Transcendental functions require differential testing: sharing an `f64` type alone does not prove agreement with a browser's math implementation.

GPU variants must preserve the source's precision and computation contract. FP16, relaxed arithmetic, different reduction order, or compressed data that changes values belongs in a separately labeled optional mode, not the primary speedup comparison. Conversely, do not promise CPU-style bit-exact results across GPU implementations where the source API permits implementation variation. Define the allowed GPU error/behavior envelope before optimization and require the source and candidate to stay inside it, including long-running temporal/simulation tests. [S36]

### 5.5 Rendering order is semantic

Transparency, stencil operations, depth behavior, clipping, feedback, and callbacks can make order observable. Grouping and pass scheduling must respect those dependencies.

Opaque batching is not automatically legal either. Equal-depth surfaces, `renderOrder`, custom depth behavior, side-effecting shaders, and callbacks can expose changes. Use conservative legality rules and test adversarial examples.

A fused shader must preserve required intermediate quantization and sampling boundaries. Two full-screen passes are not necessarily equivalent to their algebraic composition evaluated at higher precision.

### 5.6 Backend-bound behavior is retained, not refused

Preserve real WebGL integration: constructor `context`/`canvas` options, `getContext()`, `getContextAttributes()`, extensions and capabilities, native programs/resources exposed by diagnostic/state APIs, shared-context `resetState()`, GL buffer attributes, synchronous and asynchronous pixel reads, copy operations, context loss/restoration, and applicable canvas capture/preservation behavior. The source exposes genuine backend objects; a WebGPU-only imitation cannot promise their identity or semantics. [S55]

Execute such contracts through the exact production backend component when no genuinely equivalent transformation exists. Preserve its native canvas/context and source ordering. In particular, synchronous readback is not implemented as asynchronous WebGPU mapping disguised behind the same method name. Default full compatibility must work; a failure must preserve the reference's return/exception/event contract under the same input/host conditions. A separate compiler diagnostic is permitted for an explicitly requested incompatible *optimization-only* restriction; it is not a substitute for an upstream runtime error shape.

Backend-preserving execution also covers legitimate custom `Backend`/`Renderer` subclasses, source-path integrations, runtime shader/backend extensions, and WebXR configurations that require a GL-backed session. For native WebGPU contracts preserve actual device/resource objects through the appropriate component; do not assume arbitrary subclasses can be reproduced by a name-based shim. [S53], [S56]

The source/runtime and dependency analysis determines the safe selection *before* an irreversible canvas/device/native-handle binding. Unknown future native use through escaped references requires conservative exact selection. A late method call is not permission to swap canvases, lose state, replay side effects, or synchronously block on a browser Promise. Section 6.9 gives the boundary rules.

Full functional tests exercise these real paths. The new backend's separate rendering-coverage tests forbid using exact-backend execution to hide an ordinary WebGPU implementation gap.

### 5.7 Same functionality, truthful physical execution

Preserve all existing calls, controls, observable state, callback/error behavior, serialization, and lifecycle of the supported source release. Faster completion and permitted cross-GPU numeric variation do not imply identical wall-clock timing or identical private renderer machinery. An optimization may change private representation or physical work only when it preserves the application contract.

Public backend-sensitive observations are real features, not blanket “diagnostic exceptions.” If an application depends on native programs/contexts, exact backend capabilities, or physical draw statistics to control behavior, use an exact route or establish a specific correct transformation. Do not supply fake native handles or infer source-visible capabilities from unrelated WebGPU limits. An actual public reference counter must retain its contract; any separately derived logical accounting is explicitly labeled.

FrankenThreeD's own physical telemetry is exposed separately through its report/diagnostic namespace. Report actual draw/dispatch counts, route ownership, upload costs, and retained-component execution. Never present the original renderer's logical draw count as the new backend's actual GPU work.

Embedding may change physical URLs, while ordinary networked deployment retains network behavior. Preserve logical loader/resource identity and all retained data required for import/export. A source requiring HTTP observations that an offline page cannot supply uses the full networked mode; this does not remove the loader feature from the product.

### 5.8 CPU mutation and GPU-visible mutation are different histories

An attribute may have changed in a CPU array while the source deliberately has not requested a GPU upload. Keep the public CPU content version, the source's upload-request/version/range behavior, and the last GPU-visible content separate. Preserve the pinned renderer's effective upload ranges, mutations to those ranges, and `onUpload` callback ordering. The r186 WebGL attribute implementation explicitly tracks versions and performs its own range coalescing; do not substitute an idealized upload model for its actual behavior. [S41]

Coalescing new uploads across gaps must not expose additional CPU edits that the reference would leave GPU-stale. Fill such gaps from the last GPU-visible shadow, prove the bytes equal, or keep the smaller writes. The same rule applies to padding a sub-word update to WebGPU's alignment. A scan detecting a CPU mutation is not permission to upload it early.

Track these histories per device/renderer replica when a resource is shared across views or renderer instances. Public array identity, interleaving, normalized/integer interpretation, draw ranges, and source behavior on resizing remain intact.

### 5.9 Disposal releases a residency, not necessarily the public object

Separate logical object identity from backend allocation identity. Follow each pinned API's actual disposal semantics rather than treating every `dispose()` as an irreversible tombstone. For example, the inspected WebGL geometry manager removes backend bookkeeping on disposal and can register a subsequent use again. [S43]

A legal reused material, texture, or geometry receives fresh backend residency without changing its existing JavaScript identity. GPU handles still use generations so late work cannot target a recycled allocation. Shared CPU attributes may require independent GPU allocations when disposal/update lifetimes diverge; automatic deduplication must not merge observable ownership or events. Renderer shutdown, user-facing disposal, cache eviction, and device loss are distinct transitions.

### 5.10 State exposure governs offload

`AnimationMixer.update()`, explicit matrix updates, ray queries, and user callbacks can synchronously observe results before `renderer.render()` is called. Finish or materialize the results at those original boundaries. Do not defer an observable animation update until a convenient frame commit.

Retain synchronous raycasting on CPU-authoritative state. GPU picking may be an additional async API or a proven internal strategy, not a Promise substituted for a synchronous `Raycaster` result. A BVH must preserve custom `raycast` overrides, source traversal/tie ordering, and face/instance identities. Canonical triangle IDs stay separate from any private reordered acceleration structure.

### 5.11 Full-surface scope and no-cut rule

The scope is **all existing features and functionality of Three.js r186**, not “all features we choose to admit.” The initial source commit remains `148ef33ecb6d2502ff796d4554abd1549c95d519`; version updates follow Section 5.17. The complete source surface, not the finite list of names in this document, is authoritative.

Include the union of:

1. All package export paths and shipped public entry points, including ESM/CommonJS, `three/webgpu`, `three/tsl`, `three/addons`, wildcard addon/source paths, and actual packaged assets. Reconcile package metadata with built/package contents; a manifest entry is not proof its file exists.
2. Every existing export, constructor, public method/property, constant, registration/extension hook, option, serialization contract, and observable source-path utility in `src/` and `examples/jsm/`, recursively. A module absent from `Addons.js` is not excluded. Directly importable internal classes get their actual compatibility implementation without forcing the new renderer to use their architecture internally.
3. All official renderer backends and alternate renderers, controls, interaction, WebAudio/media, WebXR, physics wrappers, compression/codec support, and third-party integrations shipped or exercised by the pinned official examples. Preserve the referenced external component; do not claim it is a new engine implementation.
4. Existing editor/player, Inspector/devtools, TSL tools, examples, and relevant manual workflows as integration consumers. Preserve the current tools and file formats, not a newly invented subset or a new Rust editor.

The exact repo/package inventory reconciles these overlapping roots and records duplicates/aliases without dropping valid import paths. [S5], [S52], [S53], [S54]

An API never becomes out of scope because upstream has no test, because its module is large, because it requires permissions/another device, because it mutates at runtime, or because a portable WebGPU translation is inconvenient. Sequencing is allowed; permanent omission is not. A genuine source/host limitation remains the same limitation in the candidate, with a capable-environment lane wherever upstream has one.

Compatibility is evaluated over **source API and operation, arguments/state, source backend, and actual host capabilities/permissions**. Preserve every source-defined valid operation, including optional and runtime-selected branches, wherever the reference supports that combination. Full coverage is not a promise to synthesize browser APIs or permissions that the reference lacks, nor a promise that every feature can run on every device. It is a prohibition against additional candidate-caused feature loss. Underlying undefined or implementation-dependent GPU behavior retains its documented envelope; it is not a license to change defined application behavior.

Removed APIs from releases before r186 are not invented as current features; still-shipped deprecated aliases and warning/error behavior remain covered. The goal is the actual pinned release, not an incoherent union of incompatible historical versions. Future versions create explicit additional targets rather than silently changing this one.

### 5.12 Required feature families and implementation paths

This table is the mandatory family-level work map, **not a finite whitelist**. Every source entry discovered under Section 5.11 must map to a row or add a row before release. “Retained” means a production implementation wired to coherent public objects and tested end to end, not a stub, unexecuted file, or test-only alias.

| Family | Required scope | Default implementation and proof obligations |
|---|---|---|
| Package/module APIs | Root ESM/CommonJS; WebGPU/TSL/addon/source imports; re-exports, constants, side effects, module identity, supported Node/browser use | Retained public modules plus generated export routing; import every path and exercise source-required initialization/identity |
| Math and numeric helpers | Every vector, matrix, quaternion, Euler, color/color-management, bounds, ray, plane, triangle, interpolant, coordinate and MathUtils operation; addon OBB/capsule/noise/hull/octree/sampling/LUT utilities | Retained reference-compatible scalar surface; safe Rust bulk kernels when proven; edge cases, coercions, mutability, return identity |
| Scene graph and core data | Object3D/Scene/Group, hierarchy/layers/visibility, events, cloning, JSON, userData, manual matrix flags, traversal, disposal/reuse, CPU and GPU resource observations | JS semantic objects with versioned packed replicas; include subclass and reentry behavior |
| Geometry and attributes | Every core/addon geometry, indexed/nonindexed/interleaved/instanced/normalized/integer attributes, groups/draw ranges, bounds/normals/tangents, morphs, buffer update/copy/resize behavior | Native data path plus retained constructors/utilities; actual GLBufferAttribute use on exact GL route, not a dummy buffer |
| Curves, shapes and modifiers | All paths/curves, NURBS, shape holes/triangulation, extrusion/text/fonts, generators, subdivision/simplification/tessellation/curve modifiers and official CSG integrations | Retain complete algorithms first; accelerate measured kernels; preserve topology, attributes and synchronous results |
| Cameras, views and output | Perspective/orthographic/array/stereo/cube cameras; view offsets, projections, reversed/log depth, viewport/scissor/DPR, multiple canvases/targets, offscreen output | Native renderer plus exact native-context route where observed; compare public matrices and all views |
| Animation and deformation | Every clip/action/mixer/group/track/interpolant, property binding/mixing, blending/additive/crossfade/loop/events, skeleton/skinning/morphs, retargeting/IK and character utilities | Complete retained contract with batched Rust kernels; synchronous state and callback/event order; not just sampled poses |
| Materials and render state | All built-in and node materials, custom materials, every parameter/map and valid combination, alpha/hash/test/coverage, blending/depth/stencil/sides/clipping, shadow/depth overrides | Full standard material paths through the new backend; dynamic options and state transitions; genuine backend extensions retain exact execution |
| Physical shading and color | Pinned Standard/Physical equations and all extensions, including retroreflection, clearcoat, sheen, anisotropy, iridescence, transmission/dispersion/volume/IOR/specular; all tone/color spaces and premultiplication | Versioned shader semantics and property sweeps, with texture/animation/light combinations; no generic-PBR substitution |
| Lights, environments and shadows | Every core/addon light, probe, projector/IES/area/sun/clustered path, helpers/generators, environment/PMREM, cube maps, all shadow techniques, CSM, contact/progressive techniques | New GPU backend for expressible paths; retained construction/baking logic with complete invalidation and effect ordering |
| Textures and media sources | All texture/source/sampler classes, 2D/cube/3D/arrays, depth/storage/render-target/external/video/canvas/HTML, compressed formats/mips, partial updates, filters/anisotropy/color/orientation | Device-specific lawful formats without reducing source quality; preserve source capability/error behavior and live source lifecycle |
| WebGL renderer contract | All constructors/options/methods/properties, source-facing GL state/resources/extensions, sync/async reads, copying, framebuffer preservation/capture, context loss/reuse, explicit backend selection | Exact retained native GL where required; source-equivalent non-native rendering may specialize into WebGPU after sound analysis |
| Modern renderer/backend APIs | Full WebGPURenderer/Renderer/Backend and exported concrete backends, BundleGroup, QuadMesh, RenderPipeline/DirectRenderPipeline/PostProcessing, CanvasTarget/BlendMode, readbacks/storage and backend extension hooks | New WebGPU implementation of standard execution; exact component for native opaque subclass/handle contracts; no forced downgrade of modern features |
| TSL, nodes and compute | Every node export, NodeMaterial/loader/serialization/registration, Fn/flow/control/types/operators, shader builders and WGSL/GLSL function paths, all update frequencies, storage/atomics/indirect/queries and feature-gated operations | Complete general lowering and pinned construction components, dynamic graphs included; preserve native extension routes; test unseen runtime graphs and source-valid errors |
| Legacy and dynamic shaders | ShaderMaterial/RawShaderMaterial, ShaderChunk/ShaderLib/uniform libraries, defines/includes, onBeforeCompile/cache keys, string edits, shader diagnostics, every official shader | ESSL-to-WGSL and runtime variant handling for portable behavior; genuine GL-dependent contracts run on exact GL, never placeholder shaders |
| Postprocessing and special effects | Entire EffectComposer/Pass family and every official pass/shader, output conversion, masks/stencil, ping-pong/history, bloom/AO/SSR/SSGI/DOF/AA/upscaling, custom passes, stereo/anaglyph/ASCII/outlines | Execute complete passes conservatively before optimizing; same reset/disposal/resize/order; custom/native pass interaction retains exact resource group |
| Model, geometry and scientific loaders | All core/addon loaders, not just glTF: OBJ/MTL, FBX, Collada, STL/PLY, 3DM/3MF/3DS, USD, LDraw, AMF/VRML/VTK/VOX, point clouds/splats, volume/scientific data, fonts/SVG, animation and archive formats present in tree | Retain full loaders/runtime dependencies; load/parse/async/local-file/network variants, multi-file references, metadata and error paths |
| Texture/image and compression loaders | All existing HDR/EXR/UltraHDR, DDS/KTX/KTX2/PVR/TGA/TIFF and image/LUT paths; Draco/Basis/meshopt and other shipped decoders/transcoders | Include runtime codecs when future input needs them; browser/build-time preprocessing is an optimization, not the only supported path |
| glTF and material extensions | Every built-in registered extension plus shipped/example-registered plugins: instancing, compression, texture formats/transforms, material extensions, animation-pointer/variants/progressive/splat integrations as actually present | Inventory code registrations, not only documentation bullets; preserve parser/plugin callbacks, metadata and unknown/required-extension behavior |
| Exporters and serialization | All core JSON/object/material/geometry/node serialization and every addon exporter, including glTF/GLB, OBJ, PLY, STL, USDZ, Draco, EXR, KTX2 and any additional source entry | Retain production writers, runtime encoding dependencies and CPU materialization; semantic round trips and source-compatible output/options/errors |
| Controls and interaction | All control classes, input mappings, selection/picking, transform gizmos/snapping, HTMLMesh/InteractiveGroup, helper geometry and camera/scene utilities | Browser-owned input with real identity, focus, pointer capture/lock and event ordering; no rasterized substitute for interactive DOM |
| Alternate renderers | CSS2DRenderer, CSS3DRenderer/objects/sprites, SVGRenderer/SVGObject/Projector, and any other shipped output backend | Keep real DOM/CSS/vector output and documented limitations; no mandatory GPU request for non-GPU scenes |
| Audio and live media | Audio/AudioListener/PositionalAudio/AudioAnalyser/AudioContext, filters/nodes/playback/loop/rates, live sources and audio loaders; video/frame/canvas sources | Actual WebAudio/browser media graphs, not simulated values; activation, seeking, source replacement, permission and cleanup tests |
| WebXR/AR/VR | WebXRManager/XRManager, sessions/spaces/cameras/views/layers/framebuffers, reference-space resets, controllers/hands/haptics/buttons/models, hit testing/planes/depth/camera/lighting where upstream supports them, XR-specific fallback | Capability-qualified native WebGPU XR and exact GL-backed sessions; actual XRFrame lifecycle and real-device evidence, not normal rAF emulation |
| Miscellaneous addons and physics integrations | Every misc/helper/utility/worker object, GPUComputationRenderer and GPGPU, water/sky/reflectors/splats/volumes, progressive maps, sculpting/painters, official path-tracing integrations, Ammo/Jolt/Rapier wrappers and referenced official integrations | Retain existing algorithms/external engines where appropriate; optimize graphics without modifying the authoritative simulation or public adapter behavior |
| Inspector, tooling and editor workflows | Full Inspector hooks/panels, source-visible debug APIs, devtools integration, editor import/edit/undo-redo/serialize/script/player/export/publish/XR and TSL tools | Reuse existing UI/tools, adapt renderer services and truthful instrumentation; author/edit/reload/export/play actual projects |
| Deployment, workers and environment behavior | Existing worker/offscreen/library flows, MIME/URL/loading policies, credentials/errors, environment prerequisites, server-side utility imports and all official integration examples | Full networked/package mode plus independently valid standalone mode; host limitations match upstream, never a universal WebGPU bootstrap |

Examples in this table are verified from the pinned exports/source and the current official API descriptions; exact membership and options are derived from the source census rather than guessed from a list of familiar names. [S52], [S53], [S55], [S56], [S57], [S58], [S59], [S60], [S61], [S62]

### 5.13 Complete loading, extension and export behavior

A loader is not complete merely because one bundled model renders. Exercise `load`, `loadAsync`, `parse` and async parse where provided; loader configuration and LoadingManager handlers/URL modifiers; request headers/credentials/CORS; response types; progress, cancellation where offered, errors, caching, and relative dependencies. Preserve arrays, metadata, cameras, scenes, animations, extensions, and independent mutable object instances. Use the source's registration code as well as its documentation: the inspected GLTFLoader registers extensions beyond what a short documentation list can safely establish. [S58]

Runtime decoding and transcoding are mandatory whenever a caller can provide previously unseen compressed models, textures, archives, or local files. Include the needed existing codec or a verified replacement automatically. A build-time-only decoder cannot satisfy that contract. Assets embedded for performance tests keep every required runtime branch; shared libraries are imported only when reachable and remain separately attributed.

Export must work **after** an accelerated update. Materialize current CPU-observable transforms, geometry, skin/morph data, animation, textures and metadata at the writer's original boundaries. Never export the initial captured scene while the user sees a newer scene. Do not promise an exporter captures dynamic GPU-only state that upstream itself does not export; match its actual contract, warnings and errors.

Required checks include import → edit/animate → export → re-import in both reference and candidate, plus reference/candidate cross-loading. Compare semantics and consumer validity where formats permit nondeterministic byte layout; preserve byte-exact output wherever the source actually specifies it. For lossy codecs, use the same settings and tolerance; a different lossy encoding is not automatically equivalent. Preserve custom plugin registration/unregistration and callback identity/order for loaders and exporters.

### 5.14 Browser integrations and existing tools are first-class features

**DOM/CSS/SVG:** Preserve actual nodes, vector elements, style/layout behavior, event targets, focus, selection, visibility, parentage and removal. Mixed CSS/WebGPU views share the source's camera/transform observations. Flattening labels or SVG into the canvas is not equivalent. A page using only these renderers has no reason to require WebGPU. [S59], [S60]

**WebAudio/media:** Preserve native audio nodes, scheduled playback, filters, rate/loop/positional behavior, analyser data, source ownership and activation. Connect Asupersync only to operations it actually owns; do not move a user-activation-sensitive call behind a new await. Keep browser autoplay/origin restrictions and the upstream's actual unsupported-environment behavior. [S61]

**WebXR:** Preserve the native session/frame lifecycle, source reference spaces, view-dependent transforms, session animation loop, layers, input/hand/controller state, and supported AR integrations. Complete actual GL-backed sessions as well as native WebGPU XR when available. The existing XR fallback helper explicitly constructs and installs a new renderer; honor that contract instead of secretly switching an already-exposed canvas/device. A capable headset/browser lane is required for behavior that a phone cannot exercise. [S56], [S62]

**Editor/Inspector/devtools:** Reuse the upstream programs and their dependencies. Verify scene import, live material/geometry edits, selection/gizmos, undo/redo, scripts, animation, project persistence, player preview, exports/publishing, and applicable XR paths against the pinned original. Rust source maps and generated shader inspection augment rather than replace existing APIs. Preserve editor-authored script execution with its original host/security requirements; the engine's policy against *introducing* `eval` does not prohibit a source application's existing scripting feature. Build-time capture remains sandboxed. [S57]

**Physics and other external integrations:** Keep the same engine version, inputs, solver/time-step semantics, and public wrapper behavior. An acceleration pass cannot replace Rapier/Ammo/Jolt simulation with a visually plausible approximation. External modules referenced by official workflows are dependency compatibility obligations, not a requirement to rewrite those entire products in Rust.

### 5.15 One exhaustive feature manifest, not an expanding bureaucracy

Maintain one generated-and-reviewed manifest, provisionally `compatibility/three-r186.json`, with one record per importable symbol and its behavior families. Expand it from the full recursive source/package tree, AST exports and class/prototype surfaces, documentation signatures, registered plugins/nodes/codecs, and official examples/tools. The build must detect new or unmatched source entries. Resolve complete directory trees; truncated/paged results cannot establish the census.

Each record has: immutable source path/blob identity; export aliases; class/method/property/options and defaults; sync/async return contract; events/errors and state effects; serialization and extension behavior; owner route; supported host prerequisites; implementation and behavioral-test references; functional status; new-WebGPU-path status; and any explicitly intrinsic native-backend dependency. These are fields in one ordinary manifest, not separate services or a new policy framework.

`unclassified`, `unimplemented`, `untested`, `known-regression`, `stub`, `no-op-substitute`, and `candidate-refusal-on-valid-source` are blocking states. `retained` describes ownership, not an automatic pass. `host-blocked` requires the reference to have the same missing prerequisite, and does not establish successful behavior on a capable host. Existing upstream TODO/skips do not waive the corresponding feature's candidate test obligation.

Use separate coverage numbers for import/symbol closure, validated behavior families, new-backend rendering coverage, exact-component integration, and capable-host validation. Do not average them into a single flattering “100%” number. Every public contract must have an implementation and positive behavioral test; applicable validation/error paths and important interactions get additional tests. Function name coverage or one constructor assertion is not full functional coverage.

This plan does not claim the exhaustive symbol/behavior census has already been executed, or that a finite test set proves equivalence for every JavaScript program. It sets full intended scope and defines concrete release evidence without pretending planning is implementation.

### 5.16 Full compatibility must not conceal unfinished accelerated rendering

Maintain two independently enforced closure conditions:

**Functional closure:** every existing upstream feature has a working production path under its actual host conditions. No valid operation ends in an F3D-specific unsupported diagnostic in the default full product.

**Accelerated-rendering closure:** every standard upstream rendering/material/texture/node/pass/compute feature whose semantics are expressible in WebGPU runs through the new renderer, using retained construction/addon logic when appropriate. The full applicable visual/behavior corpus is also run with retained renderer submission disabled. This is broader than H1–H8. A portable material or pass is not marked “native-backend-bound” merely because implementing it is difficult. Numerical/backend semantics that genuinely require original GL are separately evidenced.

Exact legacy context interoperability, source-synchronous GPU reads, native diagnostics/handles and arbitrary opaque backend subclasses are legitimate separately reported categories. A blanket “all custom shaders/addons use upstream” rule is not acceptable accelerated closure. Standard dynamic GLSL/TSL programs, public subclass/registration mechanisms, and custom passes that use portable APIs require the generic new execution path, with original execution preserved for cases whose actual semantics need it.

A release containing only a renamed upstream renderer passes neither the accelerated-rendering nor performance gate, even if functional closure is excellent. Conversely, excellent benchmark results do not waive a missing exporter, audio operation, XR integration, or editor workflow.

### 5.17 Version drift and completeness maintenance

Support the actual pinned stable release, then track upstream changes through an immutable diff of exports, public options/properties, shader chunks, registrations, render/backends, addons, examples and tests. Do not infer compatibility from a semver/version string or a stale docs page. Check the latest stable release before a new FrankenThreeD release and state its tested Three.js version explicitly.

Preserve prior completed version targets while an updated target closes its new/changed contracts. New functionality remains a required parity item, not an automatic exception. A new target gets its own source/assets, test and performance manifests; retain old results and do not shift the benchmark oracle midway through an experiment. A newer unverified release may be used only with its actual retained component and truthful unverified-version status, never as an already-certified accelerated version.

---

## 6. Internal representation and execution architecture

### 6.1 Three representations are enough

Use three compact representations rather than a tower of independent frameworks.

**Semantic IR** records object identity, ownership, numeric types, update dependencies, effects, observations, and source locations. It is the bridge between the authoring program and executable state.

**Pass graph** records rendering and computation: resource reads/writes, attachment operations, required order, history dependencies, and the source of each pass.

**Execution plan** records a legal implementation for a selected device: packed buffers, shader variants, resource assignments, draw/dispatch buckets, reusable bundles, and a small sequence of per-frame commands.

The representations should use typed IDs into owned arrays. Avoid heap-allocated trait-object graphs in the hot path. A simple worklist optimizer with a bounded set of transformations is preferable to a general theorem prover or unrestricted equality-saturation engine.

### 6.2 The semantic dependency graph

Represent dependency categories separately: local/world transforms, geometry content and bounds, material program identity, material uniform values, texture content, light parameters, shadow inputs, camera state, and render-target topology.

A color change should not invalidate a mesh's world transform. A camera change should not cause a static geometry upload. A texture content update should not require recompiling its sampling code when type/layout are unchanged.

Each dependency has a version. A derived result records the versions it consumed. Recompute only when one changes. Use compact generation counters and dirty bitsets; avoid cryptographic hashing of live state every frame.

The compiler may precompute the affected dependency closure for common update classes. Dynamic insertion, removal, reparenting, and resource-layout changes rebuild only the affected structural plan.

### 6.3 The execution protocol follows source observation boundaries

There is no new universal application tick that owns the timing of every animation, callback, and render. The compiled program follows the source call/effect order. A browser presentation frame may contain zero, one, or many logical render/compute operations.

At each admitted synchronous operation:

```text
validate that operation's specialization guards
execute source-ordered updates and effects, exactly once
materialize the state that must be visible when this operation returns
```

At a render/compute boundary:

```text
consume source-visible state and upload requests at their proper versions
perform the reference-required matrix/node/callback updates in order
snapshot each draw/pass's required data version, including callback mutations
refresh only invalidated execution-plan segments
emit legal, ordered queue writes/copies and command-buffer work
submit at the required observation/presentation boundary
track completion and retire resources outside the synchronous hot path
```

A material change between two draws requires two draw-visible values even when both commands share one submission. Reentrant rendering uses nested logical render contexts and restores the outer camera, target, viewport/scissor, callback, and resource state at the same boundaries as the reference. It cannot overwrite an outer operation's scratch packet.

Asynchronous preparation publishes only at a boundary allowed by its source adapter. Loading callbacks that were observable before the next animation tick cannot be delayed indiscriminately to a frame boundary. Background jobs may yield between bounded pieces; ordinary synchronous source operations obey Section 4.9. No optimization coalesces away an observable render, update, clear, read, callback, or history transition.

### 6.4 Host boundary: bulk rather than chatty

The browser remains the authority exposing WebGPU. Wasm does not bypass its validation or call Metal directly. The host adapter should therefore accept coarse packets or execute statically generated host routines, not receive a Wasm-to-JavaScript call for every property and every tiny renderer decision.

Implement and measure two alternatives:

1. a compact command/data packet decoded by a small JavaScript adapter;
2. application-specialized JavaScript submission code generated at build time.

Keep the better measured variant for each workload class, using one semantic execution plan. Generate ordinary static JavaScript source: no production `eval` or `new Function` requirement.

Static draw portions become reusable render bundles where legal. Dynamic data stays in buffers updated in bulk. Command buffers themselves are freshly encoded for submission; a submitted command buffer is not a reusable render bundle.

### 6.5 Resource identity and lifetime

Use handles containing an index and generation, with a separate device generation for GPU resources. Never let a delayed promise publish a texture into a reused slot.

Distinguish CPU resource existence, GPU allocation, initialized content, submitted use, and eventual retirement. A logical disposal follows the source contract while backend allocation retirement remains separate; an in-flight use can outlive that residency's public release event. Pending, not-yet-submitted commands must not reference a destroyed or recycled resource.

Maintain bounded staging rings and a configurable small number of frames in flight. Exhaustion creates backpressure, not unbounded allocation. Pipeline variants, transient resources, and asset caches have explicit byte/count budgets and eviction policies.

### 6.6 Layout and safe transport

Use explicit wire layouts with checked arithmetic, byte ranges, alignment, and endian conversion. Rust structure layout is not automatically WGSL layout. Padding for vectors, matrices, array strides, uniforms, and storage buffers must be generated and tested from a shared schema.

A correctness-first bridge can copy into typed arrays. An optimized bridge must account for every remaining copy. Any borrowed Wasm memory view must obey its lifetime contract, must not survive unsafe growth/reentrancy, and must not be treated as an independently owned transferable buffer.

Pre-reserve reasonable memory and rebuild host views when the memory buffer changes. Do not disable safety checks by constructing unchecked pointers to shader or asset data. Keep private borrowed views inside the trusted generated adapter; no application callback, memory growth, or scheduler re-entry is allowed while the corresponding Rust borrow is live. Copy when that boundary cannot be enforced. Release the Rust borrow before calling effectful host/user code. Safe Rust in the core does not make arbitrary JavaScript access to its linear memory a sound ownership protocol.

### 6.7 Queue ordering is not command-recording ordering

WebGPU queue writes and command-encoder operations have different commit points. Recording a draw does not place it on the queue. In particular, two `writeBuffer` calls before a single submission can make both recorded draws read the second value; the GPUWeb project's own example of a multipass blur documents exactly this failure. [S33]

Never implement a packet like this for two different draw states:

```text
write shared uniform = A; encode draw A
write shared uniform = B; encode draw B
submit both draws
```

Instead use immutable per-use buffer slices with distinct bindings/offsets, source-indexed parameter tables, legal encoded copies between passes, or separate submissions in the necessary queue order. A copy cannot be inserted inside a render pass as though it were an ordinary draw; split the pass if the selected strategy requires it. Optional immediates are another admitted strategy, not a portability assumption.

Required regression: render red to target A, mutate the shared material to blue, render to target B, and submit a combined schedule. A must stay red and B blue. Extend it to compute uniforms, multiple cameras, nested callbacks, and two logical frames in flight. The planner tracks **per-use data versions**, not just a last value per object per frame.

### 6.8 The binary boundary has a small, checked ABI

Define protocol version, opcode and length bounds, device/scene generation, buffer offsets and lengths, and a bounded set of commands. Validate ranges before obtaining views or submitting work. Do not pass a Rust `u64` handle through a JavaScript Number when it may exceed exact integer range: use checked smaller IDs, two words, or a deliberate BigInt boundary. Counter wrap must retire/invalidate an identity domain rather than produce an ABA collision.

Alignment rules belong to the actual operation. `writeBuffer` uses four-byte copy alignment and its data offsets are in typed-array elements or bytes according to input type; encoded buffer-to-texture copies have a separate row-pitch constraint. Do not apply encoder padding rules blindly to `writeTexture` or assume mapped buffers may simultaneously participate in GPU work. [S34], [S45], [S46]

Account for CPU-to-host copies separately from GPU retirement. A source view whose bytes have been synchronously copied by the upload API need not remain pinned until GPU completion; a mapped staging allocation used by an encoded copy has a different lifetime. This distinction is both a safety rule and a performance opportunity.

### 6.9 Exact-backend boundaries and renderer/resource identity

Choose a renderer route before binding its canvas/context/device or exposing native handles. Treat it as an **irreversible semantic decision for that renderer lifetime**, unless the source itself supports creating/replacing the renderer at a tested boundary. Revocable numeric-island guards are not enough to justify a later switch between WebGPU and WebGL on the same exposed canvas.

Use conservative source/escape analysis. If a renderer or its canvas/native state can escape to opaque code that may call GL-specific or native-backend methods, preserve the corresponding exact component from construction. If the source closes those observations, its non-native work can use the new backend without carrying legacy decisions in the hot loop. If new behavior invalidates a *reversible* specialization, use the general path on the same compatible backend; do not assume unrelated GPU state can be migrated synchronously.

Preserve coherent module/class identity across retained code and optimized facade objects. Avoid loading a second accidental core copy, globally patching shared prototypes, or changing `instanceof`/constructor/prototype observations. Exact components use the actual pinned implementation and the same source-compatible object graph; generated optimized replicas remain private. Any source that genuinely intended independent copies keeps them independent.

GPU handles belong to their API, device, context and owner realm. No zero-copy interchange between `WebGLTexture` and `GPUTexture` is assumed. Source operations that share native contexts/resources form a connected compatibility group and stay on that exact backend. Independent views can coexist; an explicit supported transfer uses real copies, lawful synchronization and measured costs, not transparent per-draw ping-pong. CPU assets may have separate correctly versioned backend residencies.

The router is a small integration layer within `f3d-compiler`, `f3d-runtime`, and existing host code, not another renderer abstraction with per-draw backend dispatch. Report static route decisions and runtime executions. Every accelerated test asserts that its designated rendering work uses the new backend; every exact-component test asserts that it receives actual native objects and preserves the source's lifecycle.

---

## 7. CPU data structures and mathematical choices

### 7.1 The default mathematical representation

Use the conventional representations that match the operation:

| Operation | Representation |
|---|---|
| Local rigid rotation | Quaternion; unit-quaternion fast path only when its invariant is established |
| Local transform with ordinary scaling | Translation, quaternion, scale |
| General affine world transform | Packed 3×4 affine matrix, with a compatible public 4×4 view where required |
| Projection / general projective transform | 4×4 matrix |
| Normal transform | Correct inverse-transpose or proven equivalent specialization |
| Skeletal deformation | The reference application's matrix-based skinning by default |
| Broad-phase spatial work | AABBs, spheres, planes, and a purpose-built BVH |

This is a design choice, not a claim that one representation is universally fastest. The important optimizations are selecting the appropriate case, avoiding recomputation, and processing many objects together.

### 7.2 Why not make Clifford algebra the foundation?

A general Clifford or geometric-algebra representation is not needed to express the project's dominant workloads. Adding it to every transformation would increase the compatibility and optimization burden without an established application-level advantage.

Use geometric-algebra or exterior-algebra kernels only when a specific operation becomes demonstrably simpler, more robust, or faster, and retain the conventional implementation as a differential oracle. These are optional experiments, not architectural dependencies.

Dual-quaternion skinning similarly belongs in a separate opt-in path. It does not in general produce the same deformation as linear matrix blending, so substituting it by default would violate the core purpose of this project.

### 7.3 Data layout matters more than elegant notation

Separate hot numeric arrays from cold names, user data, event handlers, and debug metadata. Use structure-of-arrays or small array-of-structures-of-arrays blocks for large homogeneous batches. Preserve stable public identity through handles rather than forcing public objects to be the storage layout.

Batch transformations across objects so SIMD lanes have independent work. Select scalar and SIMD implementations based on measured crossover sizes; small scenes may be faster without packing overhead.

Use safe portable SIMD on the pinned nightly where its Wasm code generation is favorable, and keep a scalar reference. Inspect the emitted `simd128` code and benchmark it in the actual target browsers. Browser Wasm should not pretend it can issue arbitrary native AVX-512 or Apple-specific instructions. The `wasm32-unknown-unknown` target has its own supported environment and instruction features. [S23]

### 7.4 Transform correctness

Special-case rigid and uniform-scale transforms only after proving the necessary conditions. A hierarchy can produce shear even when individual nodes are authored as TRS. Negative scale changes orientation; singular transforms require explicit reference-compatible behavior.

Avoid recursively recomputing the whole tree each frame. Maintain a stable parent-before-child order, dirty roots, and compact affected intervals where the topology permits them. Reparenting invalidates the relevant topology segment. Public manual-update flags continue to control when updates are observable.

Do not normalize, repair, or canonicalize user-authored transforms merely to make optimization easier. A separately requested hardened mode may reject pathological inputs, but full compatibility follows the pinned upstream behavior.

### 7.5 Make the packed affine layout unambiguous

“3×4 matrix” is a mathematical description, not a wire layout. Define `AffineRows` as three `vec4<f32>` rows, with translation in each row's fourth component: 48 bytes for the GPU record, with row dot-products against `(x,y,z,1)`. For a public column-major Three.js matrix `e`, the rows are `[e0,e4,e8,e12]`, `[e1,e5,e9,e13]`, and `[e2,e6,e10,e14]`.

WGSL's `mat4x3<f32>` is four padded three-component columns and occupies 64 bytes under its layout rules; writing twelve tightly packed floats into that type is wrong. The schema must distinguish storage-array stride from the device's dynamic-uniform-offset alignment. Validate size/stride/round-trip tests independently of the renderer. [S36]

Keep a full projective representation for non-affine inputs. Do not discard the fourth row, normalize a non-unit authored quaternion, or use inverse-transpose shortcuts until the required preconditions are established.

### 7.6 Spatial queries

Implement a small graphics-oriented BVH for culling and ray queries. Begin with a robust CPU builder and traversal; add refit and a measured rebuild heuristic for dynamic scenes. Stable tie-breaking and reference-compatible hit ordering matter for picking.

A statistical or heuristic culling decision may choose a conservative bound, but must not discard potentially visible geometry. Deformed meshes need conservative bounds or correct refits. Objects outside the main camera frustum may still contribute to shadows, reflections, transmission, or other passes.

GPU culling is an optional execution strategy, not a different visibility contract. Occlusion based on previous frames requires conservative recovery for motion/disocclusion and cannot silently create missing geometry.

### 7.7 Animation and procedural geometry

For recognized animation paths, lower interpolation, pose propagation, and buffer packing together. Preserve interpolation type, track ordering, additive/blended behavior, event times, loop events, morph weights, and observable mixer state. Animation events remain ordered effects, not GPU-only state.

Cache pose work only when source time/weights and skeleton state agree. Share invariant data across instances without forcing all instances to share a pose.

Start procedural geometry with known addon implementations such as marching cubes. CPU Wasm execution offers an early route to improvements without changing CPU-visible geometry. GPU generation is valuable only where output order, precision, topology, bounds, and synchronous observations can still satisfy the contract.

Rendering optimization must not change a physics engine's authoritative time step, solver iterations, simulation precision, or state merely to improve the graphics benchmark.

---

## 8. The WebGPU renderer

### 8.1 One new accelerated backend, several legal plans

Implement one new WebGPU backend with conservative and specialized execution plans. Exact upstream renderer/backend components in Section 3.3 are separate, demand-loaded compatibility dependencies, not another newly implemented Rust GPU abstraction. There is no native Vulkan/Metal/D3D renderer and no complete `wgpu` stack beneath the new backend.

This keeps the implementation aligned with the product's actual browser target. The browser supplies platform translation; FrankenThreeD supplies scene specialization, resource planning, and efficient command/data preparation.

### 8.2 Solid optimizations to implement first

**Persistent material programs.** Specialize on geometry layout, material features, render state, and lighting configuration. Separate shader-program identity from mutable uniform values. Precompile common variants asynchronously and bound the variant cache.

**Persistent bindings and bundles.** Reuse bindings and legal render bundles until their structural dependencies change. Measure bundle rebuild cost and avoid bundles for highly unstable draw structure where they lose.

**Automatic instance formation.** Merge draw preparation only when the full draw-observation contract permits it, not merely when pipeline keys match. Preserve logical object, face, vertex, and instance IDs, including custom shader uses of instance/vertex indices. An originally noninstanced draw must not accidentally acquire a different logical instance ID after batching. Negative-determinant transforms may require a separate winding bucket. Per-object callbacks remain source-ordered effects; merging physical draws does not erase or reorder them. Unknown shader side effects or callback-dependent state prevent the affected batching.

**Incremental transforms and uploads.** Update only the derived fields and upload ranges whose input versions changed. Merge nearby ranges when that costs less than issuing many tiny uploads.

**Pass/resource planning.** Reuse compatible texture objects across disjoint lifetimes, eliminate genuinely dead passes, and preserve necessary history. This is resource-object reuse, not a claim that the browser exposes arbitrary native heap aliasing.

#### Bundle reuse has a structural contract

A bundle key includes the device generation, pipeline/layout, attachment formats and sample count, depth/stencil compatibility, bound resource identities, offsets, and recorded draw parameters. Changing data in an already bound buffer can be legal; changing a recorded binding, dynamic offset, geometry range, or draw count is not an arbitrary in-place patch to the bundle. Frame-ring variants must have matching recorded bindings or be rebuilt.

After `executeBundles`, invalidate the host adapter's cached render-pass bindings and pipeline state before issuing subsequent direct draws. The API clears that state, even for an empty bundle sequence. Add a bundle-then-direct-draw regression test with distinct bindings. A falsely warm state cache is a rendering defect, not an optimization. [S37]

### 8.3 Draw organization

Use a bounded set of buckets keyed by pipeline, geometry layout, binding strategy, and ordering constraints. Keep geometry/index buffers persistent. Represent per-object transforms and legal material parameters in packed tables.

For compatible repeated meshes, ordinary instancing is the first strategy. For heterogeneous geometry, use measured combinations of concatenated buffers, separate draws, and reusable bundles. Vertex pulling may help some workloads but is not assumed to beat ordinary vertex/index buffers.

For GPU-selected work, use portable indirect draw/dispatch commands and fixed slots or compacted instance streams as appropriate. Do not require a universally available indirect-count or multi-draw API. A zero-instance indirect slot is a conservative option, but its submission/validation cost still counts. Optional indirect-first-instance behavior must be capability-checked; a bucket-base index or equivalent legal representation supplies the fallback. Texture arrays or material tables require compatible formats/layouts and bounded bindings; do not assume arbitrary bindless texture access.

### 8.4 Many-light rendering

A clustered/Forward+ lighting path is worth implementing once representative many-light workloads justify it. It must include every light that can contribute under the source's model. Finite influence bounds require a valid bound; infinite-range lights need a separate path.

Overflow in a tile/cluster light list cannot drop lights. Use an overflow spill or exact slower path. Preserve the material's accumulation semantics within the admitted numeric tolerance, and keep the simple forward path for small light counts.

Clustered lighting already exists among current Three.js examples, so the novelty is not the name of the technique. The comparison must include that strong baseline where relevant. [S8]

### 8.5 Shadows, transparency, and render targets

Implement ordinary shadow types and material-specific shadow behavior before exotic lighting. Cache shadow work only when all shadow inputs are unchanged, including geometry deformation, visibility, light/camera state, alpha/displacement effects, and render-target contents. Execute any source-required callback or node update even when the GPU result can be reused; skip the entire operation only if its effects are also proven absent.

Preserve sorted transparency and custom blending. Weighted blended order-independent transparency is not an equivalent replacement for arbitrary sorted alpha blending and is not the default.

Support every applicable rendering contract in the full upstream feature inventory, not only those used by the current tests: cube/array/3D textures, render-target layers and mip levels, multiple render targets, depth/stencil, viewport/scissor, multisampling/resolves, clipping, and explicit clear/load/store behavior. Resource formats and usage flags are part of the plan's validation.

#### Resource hazards are defined by API usage scopes

Track resource versions and overlapping mip/layer/aspect ranges, not only texture names. For usage-scope validation a buffer is a whole subresource: assigning disjoint byte ranges incompatible roles does not make the shared buffer legal. The allocator must separate those physical buffers or their usage scopes. Also enforce the distinct per-draw/dispatch restrictions on writable binding aliases. A render pass has a usage scope whose restrictions can forbid an attachment/sampling combination even when a source scheduler places the corresponding draws in a seemingly safe order. A compute dispatch has its own usage-scope rules. Split passes, use a distinct versioned resource, or select another legal plan when necessary. Do not invent explicit native memory barriers that browser WebGPU does not expose. Copy commands have their own validation; they are not assumed to share the render-pass scope rules. [S51]

Resource reuse is allowed only after the old logical contents have no remaining consumers and the queued order is valid. Persistent histories, pending readbacks, aliased views, and source-visible render targets prevent inappropriate transient reuse. Mapping/staging credits have their own completion requirements. Attachment load/store/clear/resolve operations are explicit dependencies, never inferred solely from whether a later pass samples the image.

#### Canvas, media, and projection state are not ordinary persistent buffers

Acquire the current canvas texture for the actual rendering interval; do not keep its texture/view in a long-lived scene cache. Resize, canvas reconfiguration, and a changed device invalidate the affected attachment plan. Acquire after asynchronous preparation and avoid an arbitrary yield between acquisition and the work that uses it. A zero-sized or hidden canvas follows a defined pause/resize policy without creating illegal resources. Multiple canvases/renderer instances have separate output and resource epochs. [S48]

External video textures have a source-dependent lifetime and color conversion, not an indefinitely reusable ordinary texture binding. Refresh them at supported media/render boundaries, retain `VideoFrame` ownership until its uses are done, and close it exactly once. Cache only what remains valid; do not retain an expired external texture in a reusable bundle. Test seeking, pause/resume, source replacement, frame closure, and origin restrictions. [S39]

Keep public camera/matrix observations consistent with the source backend. A private clip-depth/projection conversion must not overwrite a public WebGL projection matrix with WebGPU conventions. Match reversed/logarithmic depth, cube-face orientation, viewport/scissor origins, alpha mode, color space, and render-target output conversion. Add tests for rendering the same camera to both a target and the canvas; a generic final tone-mapping pass is not automatically equivalent.

### 8.6 Mobile bandwidth and tile behavior

Reduce attachment traffic by avoiding unnecessary stores, clears, copies, and full-resolution intermediates that are provably unused. Optional transient-attachment support is a capability-specific path; do not depend on it universally. [S21]

Do not silently reduce render resolution, shadow size, MSAA, anisotropy, samples per pixel, geometry detail, or temporal history to obtain a performance win. Those are quality changes, not removal of abstraction overhead.

Postprocessing fusion requires a legality rule for sampling, precision, blending, history, derivatives, and texture-format boundaries. A fusion that increases shader register pressure or duplicate sampling may lose; profile it.

### 8.7 GPU-resident computation

Move work to compute only when doing so removes a demonstrated bottleneck and does not require costly or semantically invalid readback.

Good candidates include already-GPU-authored TSL kernels, culling, large animation batches with appropriate state authority, and geometry whose CPU observations are absent or satisfied independently. Keep a CPU path for small batches and synchronous public observations. Moving CPU-authored `f64` simulation to `f32` compute is not an exact optimization; it requires a separately admitted precision contract and cannot enter the primary score merely because the image initially looks similar.

A compute shader with global dependencies often needs multiple dispatches. Fusing them without a valid synchronization model is incorrect. Workgroup barriers do not synchronize an entire grid. Bounds checks and workgroup-tail handling preserve logical array domains, including overflow/empty cases; WebGPU robustness is not a substitute for generating the right computation. Do not use nondeterministic atomic ordering or a different reduction tree without satisfying the admitted numeric contract. [S36]

### 8.8 Explicit cost accounting

The debug/performance build records CPU preparation, bridge encoding, upload bytes/calls, binding changes, bundle rebuilds, draw/dispatch counts, pipeline creation, staging memory, and GPU timing where supported.

Instrumentation must be removable or bounded in production. The renderer should not allocate a detailed JSON event or run a statistical controller for every draw.

---

## 9. Materials, TSL, and legacy shaders

### 9.1 Three shader entrances

Support three entrances into the same shader/resource representation:

1. recognized built-in Three.js material semantics;
2. TSL/node-based shader graphs;
3. the complete legacy `ShaderMaterial` / `RawShaderMaterial` surface, including valid runtime-generated programs.

Portable programs require general new-backend execution; genuine native GL/opaque backend semantics retain the exact route. Unrecognized-but-valid shader code is not an omitted feature in the full product.

All entrances must carry source locations and readable diagnostics. The browser still performs final WebGPU validation and driver compilation; ahead-of-time generation of WGSL does not eliminate device-specific pipeline creation.

### 9.2 Built-in materials

Pin the exact upstream material and lighting behavior. Implement Basic, normal/depth/distance, Lambert/Phong/Toon/Matcap, Standard/Physical, line/point/sprite, and shadow-related behavior in the order the test/dependency census requires.

The physical-material matrix must cover the actual pinned features, including clearcoat, transmission, attenuation, iridescence, sheen, anisotropy, dispersion, and every newly introduced r186 property, including retroreflective controls, whether or not the current demo corpus exercises it. It is not enough to render a visually plausible generic PBR model.

Match color-space handling, texture interpretation, exposure/tone mapping, premultiplication, environment filtering, normal/tangent conventions, and output conversion. Use controlled reference scenes and material sweeps, including high-contrast and near-degenerate cases.

### 9.3 TSL construction and TSL execution have different lifetimes

Let ordinary JavaScript construct TSL graphs, then translate recognized nodes into typed shader IR. Retain dynamic uniforms as data; their changes do not by themselves require rebuilding the shader graph. Structural changes invalidate the affected program and pass dependencies. Do not imply that current Three.js necessarily rebuilds every graph each frame: the proposed advantage must be measured against its existing caches.

A compiled graph is not a pure static shader. Preserve `updateBefore`, `update`, and `updateAfter` at the source's FRAME, RENDER, or OBJECT frequency, with the correct renderer/camera/material/object/scene/compute context and update-reference identity. The pinned `NodeFrame` also treats a callback returning `false` specially when advancing update markers. A physical batch must not collapse multiple logically distinct updates or advance a marker before a failed update would have done so upstream. Reentrant rendering uses an explicit context stack. [S42]

Graph-construction callbacks remain JavaScript unless their semantics are admitted. All existing standard nodes and portable custom node/registration behavior need working general lowering. Opaque custom backend machinery retains exact source execution selected at a sound boundary; a missing custom-node adapter is not a full-product feature exemption. Never substitute a constant or silently retain the old graph. Retaining pinned, cold TSL construction helpers or build-time shader-generation components is permitted when smaller and more reliable than rewriting them. That reuse is independent of the separately permitted exact backend component. New-backend validation and scored frames still forbid retained renderer submission. Attribute ownership and cost honestly.

Static variants may be lowered at build time only under Section 4's environment rules. Genuinely dynamic graph construction requires the needed runtime lowering machinery. Every reachable material/control variant in a claimed standalone application must have a valid path: included ahead-of-time code or a disclosed runtime compiler. A shader seen only in a training trace is not the whole variant set.

### 9.4 Legacy GLSL is a major engineering task

A plan that says “run all Three.js GLSL through Naga” is incomplete. Naga's documented GLSL frontend targets Vulkan GLSL versions, not a complete turnkey WebGL GLSL ES 1.00/3.00 frontend. [S20]

Use a dedicated, bounded compatibility pipeline:

```text
Three shader source + includes/defines + material hooks
       -> ESSL preprocessing and semantic normalization
       -> explicit stage interfaces, resources, and builtins
       -> supported Naga IR/GLSL entrance
       -> validation and WGSL emission
       -> browser compilation and differential render tests
```

The normalization layer must handle actual language constructs, not regex-only syntax substitutions. The first shader spike includes macros, includes, samplers, precision declarations, derivatives, fragment depth, discard, loops, matrix/vector operations, and interface matching. Add combined-sampler splitting, integer/flat varyings, attribute normalization, explicit buffer layout, and source vertex/instance builtins.

ESSL-valid code may still require substantial transformation to satisfy WGSL's uniformity and resource rules. Implicit-derivative texture sampling under nonuniform control flow is an explicit test category. Silencing a uniformity diagnostic is not proof that behavior became portable; either establish an equivalent legal transformation or retain the genuinely required original backend. An unimplemented transformation of defined portable behavior remains a new-backend coverage gap, not an automatic intrinsic exception. A source whose results depend on undefined GPU behavior is not an exact cross-backend reference. [S36]

Coordinate-system differences require explicit treatment: clip-depth range, framebuffer conventions, winding, texture orientation, fragment coordinates, and interpolation. Point-size and line behavior need appropriate geometry/shader expansion when the source contract requires it.

### 9.5 Shader hooks and dynamic programs are required functionality

Preserve `onBeforeCompile`, mutable ShaderChunk/ShaderLib/uniform data, custom cache keys, runtime shader strings, program errors/diagnostics, node registrations and custom builders. Execute hooks at the original logical boundaries and retain exact ordering, current uniforms and effects. No blanket “static shaders only” product mode counts as full compatibility.

A finite proven variant set may be emitted ahead of time. Otherwise include the necessary runtime compiler/construction path automatically when reachable; its load/compile cost is real. Valid source-generated programs must continue to work on unseen inputs. Never keep rendering an old shader, a placeholder, a simpler material or a constant node while reporting success. New-backend tests generate graphs/shaders *after* startup and after prior variants have executed.

Naga remains the preferred narrow compiler dependency; the ESSL normalization and semantic adaptation remain first-party work with their existing budget. Keep it build-time-only in statically closed pages, but include the runtime path where source behavior needs it. A nonportable GL program or native shader-object callback uses the exact GL route under Section 6.9 rather than a failed feature. A portable program cannot be classified as permanently legacy-only just to avoid completing translation.

Previously unvisited variant compilation belongs in interaction-tail measurements and reachable-branch validation. Original GL error callbacks receive real native objects on the exact backend; WebGPU diagnostics and the F3D generated-source reports are separately truthful. The generic renderer must support the same public dynamic-material/node semantics without relying on a trace that happened to contain only a few variants.

### 9.6 Do not turn shader optimization into approximation

Allowed default transformations include constant propagation, dead code elimination, specialization of inactive material features, and equivalent resource access simplifications. Floating-point reassociation, reduced precision, changed filtering, changed sample counts, and numerical algorithm substitutions require separate justification.

Keep a conservative shader-generation path and compare every optimization against it. Diagnostic shaders for intermediate normals, depth, material channels, and world position make subtle mismatches much easier to locate than final-color screenshots alone.

---

## 10. Assets and self-contained HTML

### 10.1 Compile assets, but preserve the application

Resolve assets through a manifest containing logical URI, content hash, media type, source/license metadata, and dependencies. A small virtual-resource resolver preserves relative base paths, loader `setPath`/`setResourcePath`, URL modifiers, query/fragment behavior, and `new URL(..., import.meta.url)` as required by the source application. Code, CSS, worker, shader-include, and model-resource resolution use the same explicit closure policy. Rewriting every request to an unrelated blob URL without a logical identity map is insufficient.

Static model/texture data can be predecoded or reorganized when public observations remain correct. Preserve required source metadata and response shapes: a caller may request text/bytes, inspect a loader result, or construct a second instance independently. Deduplicating asset bytes must not accidentally share mutable scene objects, materials, animation state, or callback identities.

Preprocessing must not make an originally asynchronous loader callback unexpectedly synchronous. Preserve required causal ordering, progress/error contracts, cancellation ownership, and source-visible resource identity without promising identical wall-clock completion times. Some detailed HTTP/cache observations are incompatible with a fully embedded build; report those as admission boundaries, not fabricated network behavior.

### 10.2 All existing formats and runtime inputs

Support every loader/exporter/codec format in the Section 5.11 source closure. GLTF/GLB and the first demonstration formats are implementation order only, not the supported-format limit. Retain complete uncommon loaders and writers when this is smaller and safer than a Rust rewrite, with the same options, validation, data and callbacks.

Static Draco, Basis/KTX2, mesh compression and other data may be processed at build time, but all source-valid runtime loading, parsing, transcoding and export remain supported. A caller may select a new file or fetch an unseen compressed asset after deployment. Automatically include its required runtime codec/encoder or exact existing component; a “please preconvert your model” error is not full compatibility. Follow source-supplied external-engine/plugin dependencies without silently replacing their algorithms.

Do not substitute lossier textures or geometry. When upstream uses GPU-compressed content, preserve the appropriate format/content/filtering behavior or prove and measure an equivalent representation. Use capability-selected embedded variants, an included runtime transcoder, or a justified equivalent representation; one compressed format is not assumed to cover every target. Preserve mipmaps, block edges, orientation, premultiplication, color conversion and samplers.

A small statically closed page omits unused formats by proven dependency closure. An application offering arbitrary supported-format import/export includes the corresponding full runtime closure. Large resulting size is reported and optimized by packaging/reuse, not hidden through format omissions.

### 10.3 A single file is a packaging mode, not a universal deployment environment

The standalone HTML embeds all required static resources and contains no runtime dependency on a CDN. It initializes Wasm from embedded bytes and uses embedded or blob-backed resources as permitted by the browser. The precise guarantee is no undeclared code/asset/subresource fetch after each document load. An application-initiated navigation can load the same self-contained HTML again; that is a new document load, not evidence that the application can navigate without a server or browser cache.

Base64 increases encoded size, and decoding can create transient copies. The packer should use bounded chunks and release intermediate references promptly, but must not assume that the browser immediately frees embedded document text, script source, or decoded buffers. Account for simultaneous HTML/source strings, decoded bytes, Wasm memory, image/transcode staging, and GPU resources. Report both download bytes and peak startup memory. Do not pretend an embedded binary automatically gets the same streaming-compilation path as a separately served Wasm response.

Use HTTPS or a supported localhost serving context. A double-clicked `file://` page is not a universal WebGPU deployment promise. Shared-memory worker execution also needs isolation policy supplied by the host, not a magical HTML meta tag. [S24], [S32]

### 10.4 Complete functionality across deployment profiles

**Portable standalone:** one HTML file with every reachable dependency, no SharedArrayBuffer requirement, and the source-correct route on a main-thread or supported worker host. It may include exact compatibility components. Its accelerated workloads use capability-selected new WebGPU execution, while non-GPU/legacy features keep their proper host routes.

**Isolated accelerated deployment:** the same functionality with verified response headers and optional worker/shared-memory features. This is additive and separately measured; it cannot rescue the portable accelerated score.

**Full networked/module deployment:** preserve live fetch/HTTP behavior, external services, supported dynamic module resolution, media/permissions, and library/Node consumption. Required when those source behaviors are inherently non-embeddable; this is a first-release product mode, not an unsupported-feature workaround. A deployment selector cannot claim an offline snapshot preserves a live service.

Generate static JavaScript rather than runtime source evaluation. Serialize embedded data with a context-aware HTML/script encoder: a source string containing `</script>` or attacker-controlled markup must not terminate a generated element. Escaping data for JSON alone does not establish safe HTML embedding. Validate generated scripts by reparsing them.

Publish the required CSP allowances, including Wasm execution and any blob-worker/resource requirements. Test deployment in the intended iframe/permissions context as well as a top-level page; isolation and worker permissions are host conditions, not compiler promises. Normal module/multi-file deployment is supported for source behaviors requiring it, but must not replace the requested single-file deliverable for applications whose closure permits one.

### 10.5 Unknown assets and live applications

A strict standalone build fails when it cannot close the resource graph. It reports the unresolved URL expression or dynamic import rather than quietly retaining network dependencies.

The Sponza example, for instance, resolves a model through an external model index. The packer must lock and include the selected index/model dependencies, not leave a mutable branch URL in the output. [S11]

Applications whose essential behavior is live network data may use an explicitly networked build mode. They are not described as fully offline equivalents. An application-sized asset pack also needs limits; packaging a multi-gigabyte scene into one HTML file is not automatically a good experience even when technically possible.

### 10.6 Packaging verification

After delivering each document, deny network subresource/code/asset requests, clear incidental caches, and execute the complete application interaction script, including every reachable mode. When a control intentionally navigates, allow only the frozen HTML document request and verify its content identity; then reapply the no-subresource rule. Alternatively provide a proven same-document control transformation, but do not silently change navigation/history semantics. Test H1's count/bundle navigation as well as continuous animation. Its genuine-WebGL backend request must create the source-correct exact backend after navigation and remains a required functional test. Only the separately scored accelerated run fixes the WebGPU setting. [S7]

Exercise loader success/error, file input, resize, material switches, disposal, and reopening under the supported server context. Ordinary UI, DOM/CSS/SVG overlays, and WebAudio can remain browser-owned components; the packer must include their dependencies and preserve interaction/accessibility rather than flatten them into a screenshot. WebXR and permission-dependent behavior need capable, explicitly identified test environments; they are not silently credited on an ordinary phone/browser lane.

Verify packed hashes and required licenses. Bound recursive assets, compressed expansion, geometry dimensions, shader size, and allocations. Reject path traversal or resource references outside the declared project/allowlist. Dynamic capture is an isolated build step without developer credentials, default private-network access, or arbitrary install-script privileges. Captured application code is untrusted, not part of the compiler's authority.

Cache build artifacts using the complete semantic inputs: source/module and asset hashes, compiler and library revisions, target/toolchain/features, ABI/layout version, shader options, and admission profile. A stale cache hit must not resurrect a different material program or incompatible Wasm layout. Verify deterministic rebuilds where the build inputs admit determinism.

---

## 11. Asupersync integration and cancellation

### 11.1 Asupersync owns async execution

Asupersync is the sole async programming foundation for FrankenThreeD-owned work. Do not introduce Tokio, a parallel alternative executor, or an unstructured promise pool as a workaround for browser integration.

Browser promises remain host mechanisms. The Asupersync adapter must connect actual operations, completion, cancellation, and resource ownership to its task/region model. Prefer one application Wasm instance linking the admitted Asupersync core/host integration; do not accidentally ship a separately instantiated browser-core runtime and a second FrankenThreeD runtime with independent memories and ownership tables. Existing application codec/physics Wasm modules are separately disclosed components, not competing FrankenThreeD executors. The distinction is essential: an ownership record alone cannot prove that a browser callback stopped executing. [S12], [S14]

### 11.2 Foundation milestone

Before building a large rendering system, execute a real Rust-authored browser program through the chosen Asupersync path. It must demonstrate:

- Rust futures actually being polled and producing observed results;
- timers and host-turn wakeups without native threads;
- a nonreentrant scheduler pump and bounded microtask bursts;
- cancellation delivered to running cooperative tasks;
- real fetch cancellation and eventual cleanup;
- child tasks drained before scene-region teardown;
- no late publication into a cancelled/replaced scene;
- explicit behavior for unsupported host contexts.

Where the inspected host-services seam is incomplete, implement the missing host in Asupersync rather than forking its semantics inside FrankenThreeD. Use its existing browser scheduler contract as the starting point, but treat executable behavior, not the presence of a contract document, as the acceptance evidence. [S13], [S14]

The proposed 10,000-line foundation allocation is a planning envelope, not proof that the gap fits. Exceeding it requires using the reserve and reassessing project viability before building a second runtime accidentally.

### 11.3 Regions at useful granularity

Use regions for the application session, scene/view lifetime, asset preparation, shader/pipeline preparation, and long-running worker jobs. Do not create one task per object, triangle, uniform, or draw.

The per-frame numerical/render preparation path is predominantly synchronous, batched work. Budget checks or chunk yields obey Section 4.9: a check does not create permission to interleave an originally synchronous operation. Asupersync coordinates ownership and concurrent preparation around that path rather than imposing asynchronous machinery on each arithmetic operation.

A priority policy must preserve UI responsiveness and cancellation progress without starving ordinary asset work. It is an engineering policy with measured bounds, not a claim of real-time scheduling guarantees from a browser.

### 11.4 GPU effects have several commit boundaries

Command recording that has not issued queue operations can be abandoned and its private staging reservations reclaimed. However, `writeBuffer`, `writeTexture`, and external-image copies are already host/queue effects even before a later `queue.submit()`. A policy saying that everything before submission is rollback-safe is incorrect. Track preparation, each issued transfer, command submission, completion, and logical publication separately. [S33], [S34], [S46]

An issued update to a shared visible buffer cannot be undone merely by cancelling its task. Prefer private/versioned destinations until publication where the source contract allows it; otherwise preserve the source's partial-effect semantics. After submission the GPU work is not generally retractable. Cancellation prevents stale publication, retains necessary resources, and eventually records completion, failure, or device loss; it does not promise that a shader stopped instantly or earlier JavaScript callbacks were rolled back.

Track retirement in coarse submission epochs and use a bounded credit window. `queue.onSubmittedWorkDone()` acknowledges a queue high-water mark, not a reusable per-command-buffer fence object. Use that acknowledgement to release eligible credits without serially awaiting every normal frame. Loss/failure is a terminal cleanup outcome, not successful frame completion. [S49]

### 11.5 Device loss is not necessarily state recovery

Device loss invalidates every resource/pipeline of that device generation. Classify resource recovery explicitly:

| State class | Valid recovery action |
|---|---|
| Immutable packed assets or CPU-authoritative data | Recreate allocations and upload the current authoritative version |
| GPU state with a sufficient checkpoint and admitted replay | Restore/replay only under the specified numeric and input-history contract |
| GPU-only simulation, accumulated render history, or uncaptured output | Report state loss and follow an explicit reset/restart/error policy |

A snapshot of the initial scene cannot reconstruct the current fluid simulation. Deterministic clocks do not make arbitrary GPU replay bit-deterministic. Mandatory readback checkpoints have a real bandwidth/latency/memory cost and must be included in results; they are not a free recovery guarantee. A reset is visible application behavior and cannot be described as seamless state preservation.

Use a fresh adapter/device acquisition path, bounded retries, fresh device-generation handles, and explicit canvas/history reconfiguration. Do not repeatedly request replacement devices from an already consumed or expired adapter. [S51] A stale compile/decode/worker result cannot publish into the replacement. Do not automatically rerun already committed source callbacks. Tests distinguish reconstruction success, admitted reset, unrecoverable state loss, and deliberate teardown.

Test hide/show, page suspension, resize, scene replacement, and restoration of the source's timer policy. Avoid processing an unbounded obsolete-frame backlog, but do not silently rewrite a source simulation's time step to accomplish it. A terminated page or noncooperative foreign operation has no universal finalization-time guarantee.

### 11.6 Worker strategy

The first useful product must work without shared-memory threads. A dedicated worker may own supported CPU or GPU tasks if the actual browser/Asupersync lane permits it, but moving execution off the main thread must preserve input ordering and observation semantics. Keep the device, its resources, and submission in their supported owner realm; do not assume `GPUDevice` or arbitrary GPU handles can be posted as transferable objects. OffscreenCanvas ownership is selected and tested before relying on a main-thread fallback. A synchronous public read cannot become an asynchronous worker round trip without an admitted transformation.

Worker messages carry task/region identity, sequence, scene/device generation, and operation ownership. A cancelled job retains a drain obligation; force-terminating a worker is not proof that its externally committed effects were rolled back.

SharedArrayBuffer pools are a later, isolated-deployment option. They require a real scheduler/memory-safety design and target tests, not a `std::thread::spawn` call in an ordinary Wasm build. [S23], [S24]

### 11.7 GPU validation is asynchronous and error scopes are shared state

Associate operations with scene/device/resource generations and collect pipeline failures, validation errors, uncaptured errors, and device loss. A synchronously returned GPU object is not by itself evidence of a valid allocation or pipeline. A failed compilation must not become a published valid cache entry. Scope memory budgets conservatively because out-of-memory handling is not a dependable capacity-discovery algorithm. [S44]

Serialize access to a device's error-scope stack. Push scopes, issue the owned API operations, and pop the scopes without an intervening await or user callback that lets another task interleave unrelated scopes; only then await the returned error results. Concurrent Asupersync tasks do not acquire private GPU error-scope stacks. Keep operation-specific result promises and error provenance even when setup occurs concurrently. [S38]

These checks are part of the initial backend, not late release instrumentation. Include red-A/blue-B uniform snapshots, cancelled queued uploads, invalid pipeline creation, bundle/direct state transitions, and delayed errors after device replacement in the first integrated tests.

---

## 12. Hardware and browser specialization

### 12.1 Primary device matrix

The launch matrix should include physical devices, not only emulation:

| Stratum | Primary browser evidence |
|---|---|
| iPhone 17 Pro Max | Safari on a pinned OS/browser build; Chrome checked separately for actual engine and capabilities |
| M5 MacBook | Safari and Chrome, separate runs on the same machine |
| NVIDIA discrete GPU | Chrome on a supported Windows configuration; supported Linux configuration as a separately recorded lane |
| AMD discrete GPU | Chrome on a supported Windows configuration; supported Linux configuration as a separately recorded lane |

Safari results apply to platforms on which Safari actually runs. Do not create a fictional “Safari on Windows NVIDIA” target. Product labels such as M5 or NVIDIA do not determine all exposed WebGPU features or memory limits.

WebKit documents WebGPU support in Safari 26.0; the test matrix still needs exact current builds and actual runtime probes. Use the installed Safari on the physical target for Safari claims: a bundled automation WebKit build or an iPhone simulator is a separate test environment, not interchangeable hardware/browser evidence. [S26]

### 12.2 Capability record

Capture the browser/OS build, available adapter information, requested/enabled device features, WGSL language features, supported limits, selected canvas format/color space, and execution-host capabilities.

Record information unavailable for privacy or API reasons as unknown. Do not invent VRAM capacity from a maximum buffer size or assume an adapter name proves a particular native driver path.

Use the application-required WebGPU feature/limit profile for the accelerated backend; do not impose WebGL's lower denominator on it. Separately preserve all source-supported execution on lower-capability or WebGL-only environments through the full compatibility components. A missing accelerated profile may disable acceleration, not unrelated functionality. A source WebGPU-only operation still requires its actual GPU capabilities; no feature query is falsified to manufacture support.

Negotiate the application's minimum requirements before device creation, including storage-buffer/binding counts and workgroup/texture dimensions. Select optional variants only against the features and limits actually enabled on the created device, not merely the adapter's advertised maxima. A runtime variant requiring a larger limit or unrequested feature needs an explicit new-device strategy or a legal existing-device path; it cannot silently use the adapter limit. Feature names, WGSL directives, and binding APIs are versioned together. [S44]

### 12.3 A small specialization portfolio

Choose among a few measured implementations: CPU versus GPU culling, ordinary draws versus bundles, conservative workgroup sizes, a few legal data-layout choices, and simple versus clustered lighting.

Optional immediates can reduce some small dynamic binding overhead. Optional subgroup-size control can improve specific compute kernels. Query and request support as documented; do not assume a fixed subgroup width on another adapter. [S21], [S22]

Subgroup algorithms need a portable path and must handle partial groups correctly. FP16 is enabled only when the application's precision contract permits it, not simply because the feature exists.

### 12.4 Tuning without a research platform

Perform bounded calibration on representative kernels, outside the interactive critical path. Cache the selected plan against source/runtime/browser/capability identity. Persist source-level recipes and WGSL, not GPU handles or assumed portable driver binaries. Per-device pipelines are recreated/validated for each device generation. Use conservative defaults and invalidate stale profiles.

A tuning decision changes how equivalent work is executed, not how much visual or simulation work is required. Do not introduce an online ML optimizer or numerical-policy subsystem into every frame. Optional offline analysis may use the sibling scientific libraries, but the chosen policy should remain small and explainable.

### 12.5 Memory and power realism

Treat CPU/Wasm memory, upload staging, and GPU allocations as distinct budgets even on unified-memory hardware. Browser WebGPU does not make arbitrary Wasm memory directly usable as a GPU buffer. Bound individual allocations and total live reservations below the tested application/device budget; a large GPU buffer limit is not available RAM. Checked address arithmetic and chunking must respect the actual Wasm address space rather than assuming memory64 support.

Measure sustained mobile behavior, including throttling, not just the first few cool seconds. At a capped frame rate, reduced busy time can be useful headroom; it is not automatically a measured energy saving or a threefold increase in displayed FPS.

---

## 13. Dependency and library-reuse policy

### 13.1 First-party rules

Use a dated nightly toolchain, not a moving `nightly` tag in the release contract. Commit the toolchain, lockfiles, and application compiler inputs. Test a new nightly in a separate lane before updating the anchor.

First-party semantic, numerical, compiler, scene, and resource code uses `#![forbid(unsafe_code)]`. Browser binding machinery is a named external boundary, not evidence that the browser, drivers, standard library, or all transitive dependencies contain no unsafe implementation. A zero-copy JS view must be supplied through an audited binding boundary whose lifetime contract is actually enforced; a desired speedup does not justify hand-written unsafe pointer casts or bypassing borrow rules in engine code.

Use safe byte serialization and checked IDs rather than unchecked layout casts. Do not add native C/C++ FFI, BLAS, or a Python interpreter to the core. Existing third-party components of a source application remain separately identified application dependencies; retaining one is not permission to make it an undisclosed engine dependency.

### 13.2 Reuse decisions

| Library | Recommended role | Do not assume |
|---|---|---|
| Asupersync | Mandatory async foundation; browser host integration; native build/serve orchestration | That native lifecycle guarantees automatically transfer to arbitrary browser code |
| FrankenNumPy | Selected shape/stride utilities, reference numeric kernels, offline preprocessing where beneficial | That generic ndarray machinery is optimal for 3×4 transforms, or that Rayon-bearing crates are browser-ready |
| FrankenSciPy | Optional offline geometry/interpolation/optimization and test analysis | That a KDTree is a complete dynamic rendering BVH or a full workspace belongs in the page |
| FrankenTorch | Optional later offline analysis/differentiable experiments; selected CPU utilities only after justification | That its documented CPU path is an existing WebGPU renderer or GPU tensor backend |
| FrankenLibC | Normally no browser dependency; possibly an independently portable safe math kernel if needed | That a Linux glibc ABI belongs in `wasm32-unknown-unknown` |

These recommendations follow the inspected manifests/documentation rather than assuming readiness from repository names. [S12], [S14], [S15], [S16], [S17], [S18], [S19]

### 13.3 The dependency admission test

For each candidate crate, record its exact revision, used API, target compile result, transitive dependency closure, unsafe/FFI boundary, initialization requirements, incremental Wasm bytes, runtime allocation cost, and demonstrated benefit.

Prefer `default-features = false` with only required features. Make unwanted parallelism optional upstream rather than linking a second scheduler. Resolve Asupersync version skew before integration; two independently instantiated versions with different ownership types are not “one runtime.”

Do not transplant large source trees into another repository to hide their dependency or line count. Reusable improvements belong upstream where reasonable, and project-required new work is charged to this project's budget.

### 13.4 Small outside-dependency allowlist

Initial candidates, all subject to a pinned audit:

- the existing `wasm-bindgen`, `js-sys`, and `web-sys` boundary family, aligned with Asupersync;
- shared serialization support already used by the suite, chiefly for build manifests and diagnostics, not per-frame JSON;
- Naga as a shader compiler/validator component, normally build-time-only;
- the existing JavaScript bundler/parser toolchain for source ingestion;
- required existing runtime/build-time asset codecs and encoders, selected by the application dependency closure;
- version-pinned upstream Three.js public/host/tool/backend components needed for full compatibility, with exact renderer execution strictly isolated and reported;
- test/fuzz/browser-automation tools outside the shipping execution path.

The inspected Asupersync browser-core manifest already uses the Wasm binding family and serialization crates. Reusing that boundary is preferable to inventing another incompatible bridge. [S15]

No Bevy, general `wgpu` stack, new generic ECS framework, Tokio, or mandatory Rayon pool in the FrankenThreeD execution core. Retained upstream/external integrations keep their source host scheduling where required; they do not create a second FrankenThreeD-owned Rust executor. This is a focused browser specialization system, not an attempt to build every supporting technology from scratch.

### 13.5 Code ownership and licenses

Retained Three.js source, addon code, models, textures, and external decoder code each retain their applicable notices and provenance. The generated artifact reports them. A source-inspected implementation is not described as a legally “clean-room” rewrite merely because it is written in Rust.

The code-count report separately lists new Rust, new JavaScript, authored WGSL, generated code, retained upstream modules, tests/assets, and third-party dependencies. No category silently disappears from the report. Reusing a shader-construction helper, math utility or loader is not the same as retained renderer submission; reports and call-path tests distinguish them. The exact-backend route may intentionally retain a renderer, with no claim that this is pure-Rust/new-WebGPU execution. All code, safety boundaries, licenses and resulting bytes remain disclosed. Third-party codec/physics Wasm of C/C++ origin is an explicit inherited dependency, never described as memory-safe Rust or silently linked as native FFI.

---

## 14. Full upstream testing and stronger equivalence tests

### 14.1 Full feature inventory AND the complete upstream test suite

Release requires both Section 5.15's complete feature/behavior closure and the entire pinned test inventory. Neither a selected-example profile nor an existing upstream TODO/skip removes a feature from the product. Passing all current assertions can coexist with untested missing APIs; therefore it is necessary but insufficient.

Generate the source/package/symbol/option/registration inventory first, then attach core unit, addon unit, E2E, WebGPU, tree-shaking and other applicable upstream checks. Every source entry must be mapped, even if no upstream test imports it. Count active assertions, imported tests, TODO/skips, setup/import failures, host-blocked cases and newly written feature tests separately. Do not claim totals or a completion percentage before running the census.

Preserve original test assertions and thresholds. Add positive functional tests for previously untested features, their applicable error paths and source-observable transitions. Fully cover loaders/exporters and browser/tool integrations, not only rendering. Documentation examples and registration code are additional evidence for required behavior, not replacements for tests. [S5], [S52], [S53]

Run separate production-route lanes: full functional routing; forced new-backend execution for all applicable rendering cases; and exact native-backend integration. This prevents both hidden functionality gaps and a trivial all-upstream wrapper that appears accelerated only in a few artificial scenes.

### 14.2 Run unchanged assertions against the candidate

Create a candidate module-resolution tree that redirects the upstream source import paths to the production compatibility exports. Keep the original test bodies, expected values, and assertions intact. Use a mapped working copy/resolver without editing the upstream oracle checkout. Record the resolved implementation for every import. Retained modules and exact renderer components are allowed only as declared production owners in the full functional lane. Hidden/test-only delegation is forbidden; the forced new-backend lane and scored accelerated workloads reject retained renderer submission.

For internal utilities such as `WebGLRenderLists`, supply the same semantic utility surface when required by the test contract, without requiring the optimized renderer to use that utility as its hot-path representation. A compatible utility and a private packed render schedule can coexist. The inspected tests assert exact object structure and identity-related behavior, making this distinction necessary. [S28]

A declared exact-backend test can exercise the shipped pinned renderer component and count toward functional integration coverage, but not new-backend implementation or acceleration coverage. An accidental import of the untouched oracle renderer, or a hidden upstream implementation enabled only for tests, counts toward none of these. Validate which package, source revision, instance and submission path actually executed.

### 14.3 Ownership-aware results

Every result includes the implementation owner: new Rust/WebGPU, general new-backend execution, retained production JavaScript/host component, declared exact native-backend component, or unresolved. Retained modules are allowed for compositional compatibility and size control, but do not count as Rust feature coverage.

Every active upstream test must be imported and executed in its required environment. Syntax/import/setup errors, assertion-count shortfalls, and unhandled rejections fail the inventory rather than becoming empty successful suites. A candidate-only skip is a failure. An upstream-existing skip remains visible as an unproven upstream case, not a new pass.

An environment that cannot exercise a feature gets a separate blocked status with a required capable lane. It cannot silently shrink the global denominator. A genuine unresolved backend incompatibility leaves the full-suite gate closed. Full-suite compatibility does not establish complete feature coverage or that the specialized execution path was exercised: add paired production-profile runs and require each claimed optimizer pass to execute on designated positive tests as well as fall back correctly on adversarial ones.

### 14.4 Screenshot oracle versus hardware arena

Keep an upstream-shaped regression lane for reproducibility, including its fixed configuration where appropriate. Do not treat it as a performance lane: the inspected runner specifies a software Vulkan ICD and special launch flags. [S6]

Create a separate hardware arena with ordinary supported browser configurations, actual selected adapters, and no forced software renderer or experimental enabling flags in launch claims.

The baseline and candidate must use the same hardware, browser build, viewport, device pixel ratio, material settings, sample counts, assets, input sequence, and frame/time state.

### 14.5 More than a screenshot

Use deterministic application clocks and random sources supplied by a shared test harness. Keep the measurement clock separate and real: overriding the application's `performance.now()` or animation timestamp must not replace the wall clock used to compute throughput. Preserve seed consumption, timer/microtask behavior, and one update per logical frame. Capture a sequence of frames and associated state observations, not only the first frame after loading.

Test pointer/mouse/touch controls, wheel/pinch, keyboard inputs, resizing, control-panel changes, loading completion/error, animation transitions, picking, material changes, scene additions/removals, and disposal. Compare event/state traces and visual results at corresponding logical times. Specify required causal order rather than demanding identical wall-clock arrival times for independent asynchronous work. Diagnostic FPS/GPU counters may truthfully differ; any excluded diagnostic pixels are narrowly identified before testing, not broad masks that can hide scene defects.

For temporal renderers and simulations, preserve warmup, history, fixed-step policy, and random inputs. Chaotic dynamics require a predefined numerical/behavioral contract; do not excuse an arbitrary divergent trajectory because two individual frames look plausible.

### 14.6 Visual tolerances

For the upstream lane, preserve its existing expected assets and thresholds. The inspected E2E configuration uses its own per-pixel error rule and a 0.1% different-pixel limit; FrankenThreeD must not loosen that threshold to pass. [S6]

For additional cross-backend hardware comparisons, preregister tolerances from baseline repeatability and the actual intended visual contract before optimizing. Use color-space-aware comparisons, edge/silhouette checks, region checks, depth/normal diagnostic targets, and temporal consistency in addition to aggregate image metrics.

Do not silently regenerate golden images from the candidate. Exact pixel identity across different GPU backends is not assumed, but “looks similar to an observer” alone is too weak.

### 14.7 Adversarial and property tests

Add tests for aliasing, typed-array writes, subclass overrides, reflective access, `needsUpdate`, callbacks that reenter rendering, mutable material keys, matrix-read materialization, and deoptimization after an assumption changes.

Add geometry/math tests for empty data, negative scale, shear, singular matrices, degenerate triangles, ray-hit ties, extreme coordinates, NaNs, infinities, and integer overflow boundaries.

Add lifecycle tests that cancel at every supported await/publication boundary; lose the GPU device during compilation/upload/submission; dispose resources while work is pending; grow Wasm memory; and deliver delayed worker responses after ID reuse.

Fuzz asset-pack parsing, shader preprocessing/translation, handle decoding, buffer layout validation, and serialized execution plans. Shader input validation and resource/dispatch limits reduce exposure, but do not claim they prove every GPU program terminates within a hard deadline.

### 14.8 Optimization-specific evidence

Each optimizer pass needs a small legality statement, a conservative executable reference, counterexample tests, and an ablation measurement. Include the queue-write snapshot, upload-range stale-byte, callback multiplicity, synchronous-yield, affine-layout, and bundle-state counterexamples from this revision. Guards/materialization are part of the measured optimized path, not disabled in benchmark mode. The reference is available for development comparisons; production does not double-render every frame to prove itself continuously.

Use a few strong invariants rather than a large bureaucracy: one owner per mutable epoch, no stale publication, preserved observation order, legal resource usage, and no optimization result accepted before equivalence passes. A held-out source corpus and unseen input/control sequences must exercise the same compiler; selecting kernels by example path or replaying captured outputs is disallowed.

### 14.9 Feature-completeness tests beyond upstream assertions

From the single feature manifest, require an executable positive case for every existing callable/option/behavior family and the relevant invalid-input, mutation, serialization and lifecycle cases. Preserve source errors rather than demanding errors that upstream does not produce. Constructor existence, identical function names, or merely retaining source bytes is insufficient.

Add mandatory family-wide suites for: all import paths/aliases; every loader/exporter and codec; every standard material/map/state family; each TSL/node and shader entry; every pass/effect; all controls/alternate renderers; audio/media; XR; physics wrappers; Inspector/editor/tool workflows; and all native-backend public APIs. Include runtime inputs not present at build time.

Cover known high-risk combinations explicitly, plus generated pairwise combinations where applicable: skinning+morphs+shadows+export; clipping+stencil+transparency; instancing+negative scale+custom shader indices+picking; transmission+color management+render targets; texture update+multiple renderer residencies+disposal; runtime shader/node registration+readback; CSS/SVG+camera changes+input; XR+skinning+layers+session reset; editor undo/redo+shared materials+serialization. Pairwise tests supplement rather than replace known higher-order counterexamples.

Run original and transformed official examples with all actual controls, loaders and mode transitions exercised. Retain unchanged upstream skips in the inherited lane, but supply additional tests for their functional surface. Reconcile every otherwise unlisted example/manual/tool workflow into the same inventory. No single screenshot or demo substitutes for these families.

For genuine GL contracts test actual context identity, external context reuse, extension/state/native-handle access, sync pixels visible immediately on return, raw-GL+Three.js interleaving and reset, GLBufferAttribute behavior, context loss/restoration, and canvas capture. Test absence of WebGPU with working upstream GL/DOM/audio cases. For XR/media, record real capability and permission prerequisites and use actual supported devices; emulators/mocks are additional tests, not successful hardware feature evidence.

### 14.10 Release coverage invariants

The release verifier fails on an unmatched source/export path, feature without an implementation/behavioral test, candidate-specific valid-input refusal, placeholder/no-op substitution, unexplained extra skip, incomplete runtime codec/variant closure, or known functional regression. Missing upstream tests add work; they do not shrink the requirement.

All portable standard rendering cases also run through the new WebGPU backend with exact renderer submissions disabled. Any intrinsic native-backend exception names the concrete observable contract and its real compatible implementation; it cannot be a catch-all category for unfinished rendering. Exact-component presence alone is not functional completion until its integration tests pass.

A device-specific block is legitimate only when the reference cannot exercise the same operation with the same host permissions/capabilities. The feature remains globally required on a capable lane. Lack of access to a required headset, browser or other supported host is an outstanding verification obligation, not a passed gate or a feature exclusion. If the pinned upstream feature has no currently executable positive path even on its specified host prerequisites, document that upstream limitation, preserve the exact conditional implementation/error behavior, and identify that evidence class explicitly; do not count it as successful positive validation or use it to conceal a candidate implementation gap.

The final report states the pinned version, all inventory counts, new-backend versus exact ownership, completed and unavailable-host evidence, remaining known defects, and benchmark results separately. It must not label a plan, static source inspection, or symbol-name census as implemented “100% compatibility.”

---

## 15. The demanding demonstration corpus

### 15.1 Select before optimizing

The following are verified names in the pinned Three.js example inventory. Several have been inspected directly; the remaining paths are candidates whose full source/asset dependency closure must be inventoried in Phase 0. Workload classification below is a hypothesis, not a measured bottleneck diagnosis. [S7], [S8], [S9], [S10], [S11], [S29]

Keep the unmodified default example as a correctness case. Add stress variants using documented controls or one shared, clearly recorded harness parameterization applied identically to baseline and candidate. Never increase difficulty only on the baseline or lower quality only on FrankenThreeD.

### 15.2 Preregistered headline candidates

| ID | Upstream example | What it forces the project to handle | Candidate source of gain |
|---|---|---|---|
| H1 | `webgpu_performance_renderbundle` | Existing bundles, heterogeneous geometry, per-object material data, static/dynamic toggles | Data/program specialization beyond merely enabling bundles |
| H2 | `webgl_marchingcubes` | Rebuilt procedural geometry, normals/colors, changing field parameters | Bulk safe-Rust geometry kernel and reduced allocation/upload work |
| H3 | `webgl_animation_multiple` | Multiple animated assets, independent poses and scene updates | Batched animation, interpolation, transforms, shared invariant data |
| H4 | `webgl_shadowmap_performance` | Repeated scene/shadow work and many objects | Legal batching, dirty dependencies, shared work across passes |
| H5 | `webgl_instancing_dynamic` | Already-instanced geometry with dynamic instance data | Update-kernel compilation and upload efficiency rather than naïve instancing claims |
| H6 | `webgl_postprocessing_advanced` | Multiple effects, render targets, pass ordering | Legal graph specialization and resource/traffic reduction |
| H7 | `webgpu_skinning_instancing_individual` | Compute skinning, individual variation, morph data, shadows | Pose/update specialization and equivalent cross-pass data reuse |
| H8 | `webgpu_compute_particles_fluid` | GPU compute, atomics, grid/particle transfers, dynamic counts and interaction | A genuinely better equivalent compute/data/scheduling plan |

H7 and H8 intentionally prevent the headline suite from consisting only of legacy object-submission bottlenecks. They may be difficult to accelerate by the required amount. That difficulty is part of the project's go/no-go test, not a reason to remove them after disappointing results.

### 15.3 What direct inspection revealed

H1 already defaults to 4,000 objects, uses multiple geometry types and distinct toon-material colors, and supports render bundles and dynamic behavior. It is a particularly good test of whether the compiler can combine equivalent material programs and packed state without changing source-level identity. Both the enabled-bundle reference and a competent instanced/material-data reference belong in the comparison if they satisfy the full declared application-observation contract. Collapsing source objects or changing custom instance IDs just to accelerate the reference is not automatically eligible. [S7]

H2 calls its procedural update on each animation step and exposes resolution, blob count, isolation, and wall/floor controls. This is an early candidate for a known-addon Rust kernel rather than a general JavaScript compiler. Its default material and every material-selector branch, including the four legacy ShaderMaterial variants, belong in the first product slice; the UI cannot be restricted to an easy shader after the fact. [S40] [S29]

H8 already implements an MLS-MPM-style particle/grid computation with multiple kernels, atomic integer cell data, indirect dispatch attributes, and runtime particle-count changes. Its default parameter is 32,768 particles, with storage reserved for up to 131,072. Simply relocating JavaScript renderer code cannot be assumed to make it 3× faster. [S9]

H7 already computes instanced deformation through storage-buffer data and custom TSL, with per-instance skeleton/morph information. It is a serious compatibility and performance test, not a simple crowd rendered through separate naïve draw calls. [S10]

#### H1's complete backend and navigation contract

H1's full artifact supports both source backend settings, including static/dynamic updates, object counts, bundle settings, materials, camera controls, and resizing. Its WebGL choice must produce a genuine GL-backed renderer through the exact component; it no longer produces an F3D-specific unsupported-mode message. Selecting WebGPU uses the new renderer for the accelerated tests.

The source reloads the page for several settings. Verify query/navigation behavior and each newly loaded document, with every reachable backend component embedded for standalone use. Score the preregistered WebGPU workload for acceleration and report the exact GL mode separately. The performance setting chooses which valid behavior is measured; it does not remove a functional branch or its startup bytes. All H2–H8 controls/material/runtime branches remain required as well.

### 15.4 Advanced fidelity and control set

Require, at minimum, additional cases covering:

| Case | Purpose |
|---|---|
| `webgpu_vxgi_sponza` | Voxel cone-traced GI, multiple render targets, temporal antialiasing, large assets and first-person control |
| `webgpu_postprocessing_ssgi` and `webgpu_postprocessing_traa` | Temporal state, history invalidation, difficult visual equivalence |
| `webgl_loader_gltf_transmission` and `webgpu_loader_gltf_dispersion` | Physically based transmission/dispersion and asset fidelity |
| `webgl_clipping_stencil` and `webgpu_clipping_stencil` | Ordering, clipping, stencil correctness |
| `webgpu_compute_cloth` | Compute dependencies and interaction with simulation state |
| `webgl_materials_video` / relevant WebGPU video example | External media lifecycle and texture updates |
| A minimal full-screen, deliberately GPU-bound shader | Negative control against false CPU-only speedup claims |
| A small ordinary interactive scene | Fixed-overhead and startup regression control |

These cases are not omitted from the published performance report. They are separated from the headline throughput gate before optimization because they also test fidelity and detect where acceleration cannot arise from the proposed mechanism. Report an all-valid-cases aggregate in addition to the headline score, and never describe the headline score as a universal average across all Three.js applications.

The current Sponza example explicitly combines VXGI and temporal antialiasing and fetches its asset through an external model index, so both its render graph and asset closure are material parts of the test. [S11]

### 15.5 No single demo tests everything

Demanding does not mean comprehensive. A fluid demo may barely exercise animation events; a physical-material demo may not stress object updates; an instancing test may have simple transparency.

Maintain the exhaustive feature-to-example/test crosswalk required by Section 5.15. Fill every uncovered family with targeted tests, including loaders/exporters, host integrations, tools and source-level APIs absent from the current demos. Do not advertise one visually impressive scene as proof of complete Three.js support.

### 15.6 Selection cannot become cherry-picking

Phase 0 may correct an invalid or unavailable candidate before any candidate performance is known, with a written reason and an equivalent replacement. After the benchmark manifest is frozen, a slow or difficult case stays in its declared category. Any later corpus revision creates a new benchmark version and retains the old results.

Eligibility is established by the reference, not by whether the candidate succeeds. If a particular baseline cannot run at the preregistered settings, record the baseline capability/limit failure rather than an infinite speedup. If the baseline succeeds but FrankenThreeD fails to compile, exceeds memory, times out, omits a reachable branch, violates its required new-backend execution route, or fails fidelity, the cell is a candidate failure and blocks the gate. It is never deleted as “invalid.”

Require a sufficiently broad common reference-valid set on every primary stratum; fewer than six headline cases, or loss of either modern-compute headline category, blocks a broad launch claim. Freeze defaults and stress parameters before candidate results; report default-scene behavior even when the declared headline stress setting is more demanding.

## 16. Performance methodology and go/no-go gates

### 16.1 The reference must be strong and its selection unbiased

Use three reference descriptions where applicable:

**R0: As authored.** The pinned application with shared deterministic harness instrumentation only.

**R1: Competent Three.js configuration.** Apply a frozen, bounded recipe of documented precompilation, instancing/batching, render-bundle, and resource-management optimizations. Preserve the declared object/state/control contract, not only pixels. This is not an unlimited manual rewrite of the application, a replacement simulation, or a hand-written FrankenThreeD-equivalent compiler.

**R2: Equivalent current Three.js WebGPU implementation.** Preserve scene/materials, logical updates, effects, resolution, precision, and inputs. Where a genuinely equivalent migration is not available, record the missing reference rather than inventing one or comparing different simulations.

Select the strongest eligible configuration using independent reference-only calibration, then freeze its source/settings before the scored paired runs. Do not take the fastest noisy trial among many configurations as the denominator: that selection can bias the result. Publish R0 and all eligible reference configurations as well as the selected comparison. Improvement in reference code creates a new frozen comparison, not selective replacement of favorable runs.

Existing render bundles and upload-range coalescing are baseline capabilities, not presumed FrankenThreeD inventions. The selected reference must not leave applicable optimizations disabled merely to create a win. [S7], [S41]

### 16.2 What is measured

Measure completed logical frames per wall-clock second in a bounded steady-state pipeline, including application update work, scene preparation, bridge work, uploads, command preparation, GPU execution, and completion.

Do not substitute the time to call `queue.submit`, a GPU-only kernel timer, or a matrix microbenchmark for end-to-end throughput.

For on-screen applications, additionally measure frame pacing, missed presentation deadlines, input response, and long tasks under the original interaction model. Two applications capped at 120 displayed FPS do not exhibit a measured 3× displayed-FPS improvement.

### 16.3 A fair uncapped lane

Use an offscreen, fixed-logical-time lane when presentation caps obscure throughput. Both implementations render the same logical state sequence with equivalent target dimensions, formats, samples, output conversion, and effects. First validate the reference's offscreen harness against its original on-screen rendering: switching a Three.js scene to a render target can change output/tone-mapping or resolve behavior. A faster but incomplete offscreen pipeline is not an eligible reference or candidate.

Freeze a small work-in-flight limit, provisionally two logical frames, for matched comparisons. Do not obtain a large apparent score by recording an unbounded queue or by starving the browser event loop. Each logical frame includes all its source updates, render/compute passes, and required effects; persistent simulation/history state advances exactly once. Source-required synchronization still applies even when it reduces overlap.

Use completion acknowledgements on both APIs. The WebGPU adapter records submission high-water marks and resolves bounded completion credits with `onSubmittedWorkDone`. The WebGL 2 reference inserts `fenceSync`, flushes, and polls `clientWaitSync` with zero timeout while yielding appropriately to the host. Neither side busy-spins; do not compare per-frame `gl.finish()` on the reference with an asynchronously pipelined candidate. Include acknowledgement overhead and run a matched empty/light-work calibration to quantify harness overhead. [S49], [S50]

Start the real measurement clock before the first scored application update after an initial drain; stop after the last scored GPU work completes. Record the exact number of credited logical frames. A device loss, asynchronous validation error, or rejected completion invalidates the run as a failure, not successful fast completion. App time and measurement time remain distinct.

Capture selected frame/state checkpoints throughout separate equivalence runs and uniformly instrumented spot-check windows. Avoid an asymmetry where only the reference performs expensive pixel readback. Report **completed offscreen-frame throughput**, not displayed FPS, beside normal on-screen pacing and input-latency results. An offscreen win that violates the interaction budgets does not pass the product gate.

### 16.4 Timing and experimental design

Use independent paired runs with alternating/randomized order and adequate warmup. Pin browser builds, OS, driver where available, viewport, DPR, power mode, cache state, and background-load policy. Avoid simultaneous GPU workloads from other agents or build tasks.

Measure cold startup separately: document acquisition, decoding, Wasm compilation, shader/pipeline compilation, asset preparation, first correct frame, and first interactive frame. Also measure warm-cache startup. Do not hide expensive startup work outside the report because the steady-state score looks attractive.

Report median, p95, p99, paired speedup, peak memory where measurable, and software-accounted resource/upload totals. Use GPU timestamps only when supported, and account for timer precision; unavailable telemetry remains unavailable.

Use enough independent runs for stable confidence intervals. Bootstrap or otherwise analyze at the paired run/block level rather than treating adjacent correlated frames as independent experiments. Preserve raw runs and the analysis script. Freeze the repeat count, stopping/retest rule, aggregation weights, and treatment of infrastructure failures before scoring. An unfavorable valid run is not an infrastructure failure; a genuinely invalid hardware session is retained with its reason and rerun as a complete pair.

For a cell, compute throughput from completed frames divided by actual elapsed time, then the candidate/reference ratio within matched blocks. Aggregate cell ratios in log space with equal, preregistered cell weights within each stratum. Report uncertainty at both cell and stratum levels. Tuning data and hidden validation inputs stay separate; an inconclusive confidence interval requires a new preregistered batch, not repeated peeking until one favorable interval appears.

### 16.5 Mobile and thermal protocol

Run a sustained mobile session, provisionally 20 minutes, with paired cooldown/recovery conditions and a recorded power/charging policy. Report the final steady segment separately from the initial cool segment. The phone's scored performance floor must hold in the declared sustained segment; a brief cool-start win does not satisfy a sustained-use claim.

The benchmark device must not be running unrelated agent builds or background GPU work. A remote build worker may compile artifacts, but hardware timing uses a reserved device.

Report energy per frame only with an instrument or counter that supports that claim. Do not infer a precise power improvement from CPU timings or a noisy change in battery percentage.

### 16.6 Launch performance contract

Freeze the corpus, hardware strata, logical workloads, quality settings, equivalence thresholds, reference recipe, confidence method, and deployment profile before candidate performance is known. Startup/memory/tail budgets are explicit numbers derived from the target experience and reference/foundation measurements during Phase 0; an unset budget leaves the gate incomplete, not automatically passed.

| Gate | Required outcome |
|---|---|
| Equivalence | Every required candidate passes its visual, state, control, and lifecycle profile before its speedup can count |
| Eligibility | All reference-valid H1–H8 cells remain required; candidate failures never shrink the denominator |
| Headline hard floor | At least 2× median paired completed-frame throughput in every required cell; an interval consistent with less than 2× is inconclusive, not a confident pass |
| Stratum target | At least 3× geometric mean across required headline cells in each primary device/browser stratum |
| Confidence | Paired 95% lower bounds meet the 2× cell floor and exceed 2× for the stratum geometric mean; publish the method and all intervals |
| Modern coverage | H7 and H8, or equivalent categories fixed before optimization, remain in the required score |
| Tail behavior | Meet the frozen p95/p99 frame-time, missed-deadline, and input-response budgets; analysis of a regression does not waive a failed budget |
| Controls | No greater than 5% sustained-throughput regression on valid control cases, with measurement uncertainty reported; publish every control and the all-case aggregate |
| Startup and memory | Meet frozen absolute device budgets, report cold/warm costs and reachable-variant compilation hitches, and demonstrate bounded caches/no OOM |
| Thermal | Phone performance floors hold in the declared sustained segment, not only the initial cool segment |
| Deployment | The portable standalone profile passes independently; isolated/shared-memory results are additional and cannot rescue its failed launch score |
| Standalone validity | No undeclared subresources/isolation prerequisites; every reachable compatibility component and navigation mode is included and verified |
| New-backend ownership | Scored accelerated GPU work uses the new renderer, not retained renderer submission; the ordinary full artifact, including reachable compatibility code, is measured |
| Functional closure | All Section 5.11 features and Section 14 completeness gates pass independently of this benchmark corpus |
| Size | Project Rust, tests, generated engine source, and charged foundation changes remain inside the 245k planned ceiling |

The 5% control allowance permits a small fixed-cost tradeoff where little acceleration is available, never reduced quality. A candidate control that fails fidelity is not a 5%-budget case. A changed product decision or benchmark corpus requires a newly labeled plan/results version and cannot retroactively turn the old failed gate green.

Aim for 3× on individual headline cells as well. The 2× individual floor and 3× per-stratum target remain distinct. Neither finite corpus establishes a universal 3× guarantee for all Three.js applications.

### 16.7 Early kill gates

Do not spend the entire 215k planned lines before learning whether the idea works.

**Foundation gate:** actual browser task execution/cleanup and a legal WebGPU bridge work on Safari and Chrome. No fake ledger-only execution or native-thread dependency.

**Bridge gate:** measure direct JavaScript WebGPU, simple Wasm-to-host calls, bulk command transfer, and generated host submission for matched workloads. The selected path must demonstrate an advantage opportunity after its own overhead; no fixed numeric win is assumed before measurement.

**First product gate:** automatically transform the complete H1 and H2 applications, preserve all controls/backend modes and material variants, and separately score H1's WebGPU dynamic workload plus H2 through the new renderer. Both H1 backend modes must work; retained exact execution cannot count as the new-backend speedup. Include actual guards, materialization, bootstrap, and bridge costs. Require the >=2× floor against the independently selected strong references on the phone and M5 strata before broad compatibility implementation proceeds. If H1 is not viable against its competent reference, do not replace it with an easier toy without explicitly reopening the project decision.

**Generalization gate:** repeat the result on held-out applications and at least one discrete-GPU stratum, with a measured end-to-end route for H7/H8. Profile H7/H8 references during Phase 0 and test promising equivalent kernel/data-path changes early, rather than postponing the hardest viability question until the renderer is complete. Microbenchmark improvements alone cannot pass.

**Final gate:** exhaustive functional and accelerated-rendering closure, all upstream/integration tests, and all Section 16.6 conditions. Failing it means no performance-replacement launch claim, even if the implementation is otherwise impressive.

### 16.8 Attribution of gains

Run ablations that disable one optimization at a time: update compilation, dirty tracking, material-program merging, bundles, GPU compute selection, pass/resource optimization, and optional hardware features.

Publish where the gain comes from and where the bottleneck moved. Do not multiply isolated kernel speedups together. A 5× transform kernel and a 2× uploader do not imply a 10× application.

### 16.9 Build cost and developer experience also count

Measure clean and incremental conversion time, Rust/Wasm and shader compilation time, cache hit correctness, generated bytes, and the latency of changing a common material or update function. These are separate from the frame-throughput score but part of whether the compiler is useful. Rebuilding every generated kernel for a color-only edit is a design regression even if the deployed frame is fast.

Use bounded per-application specialization and content-addressed build units inside the existing compiler, not a new build platform. Source maps and reports must identify the original application span that caused a rejection or expensive synchronization boundary. Runtime tuning, lazy material compilation, and deoptimization pauses all remain visible in the corresponding cold/interaction measurements.

---

## 17. Workspace and size budget

### 17.1 Proposed workspace

Use eleven crates with clear responsibilities, not dozens of tiny architectural layers.

| Crate | Responsibility | Planned new Rust lines, including local tests |
|---|---|---:|
| `f3d-core` | Typed IDs/layouts, capabilities/epochs, errors, single feature/route manifest | 10,000 |
| `f3d-math` | Fixed-size math, SIMD batches, robust geometry primitives | 12,000 |
| `f3d-scene` | Packed scene state, dirty propagation, animation, query structures | 25,000 |
| `f3d-assets` | Asset pack, selected safe decoders/preprocessing, resource metadata | 20,000 |
| `f3d-shader` | Complete standard material/TSL/ESSL semantics and required dynamic lowering | 30,000 |
| `f3d-graph` | Complete pass semantics, hazards, lifetime planning and schedule specialization | 14,000 |
| `f3d-gpu` | Full standard new-WebGPU resource/render/compute paths and backend boundary tests | 27,000 |
| `f3d-runtime` | Asupersync, lifecycle/host protocol, exact-component and browser/tool integration | 12,000 |
| `f3d-compiler` | Conservative route analysis, full import closure, islands/guards/code generation | 24,000 |
| `f3d-cli` | Build/inspect/verify/bench/serve orchestration | 7,000 |
| `f3d-conformance` | Feature/behavior closure, full upstream/host/tool integration and performance tests | 24,000 |
| **FrankenThreeD subtotal** | | **205,000** |
| **Charged Asupersync foundation work** | Browser-host completion/tests required by this project | **10,000** |
| **Planned subtotal** | | **215,000** |
| **Contingency reserve** | Compatibility/compiler surprises | **30,000** |
| **Planned ceiling** | Remains below the user's 250k limit | **245,000** |

These are estimates, not measured implementation sizes. The shader frontend and semantic-compatibility work are the largest uncertainty. A forecast above the ceiling triggers an implementation/reuse decision before more infrastructure is built. It does not authorize dropping existing Three.js features or their tests. The full-surface plan depends on deliberate reuse, not a claim that rewriting every codec, editor, browser integration and backend fits this Rust budget.

### 17.2 Other code budgets

Target no more than roughly 12,000 new JavaScript/TypeScript lines for source adapters, compatibility glue, host bindings, and browser harness integration, and roughly 15,000 authored WGSL lines. Track these openly rather than hiding an engine outside the Rust budget.

Retained upstream compatibility modules and third-party shader/compiler/codec code are reported separately with size and ownership. All checked-in/generated engine Rust counts toward the main ceiling. Application-specific generated kernels are also reported per artifact; generators must not be used to disguise a giant checked-in engine. Use one pinned, consistently applied line-count method including test code; do not minify Rust or move new engine work into a sibling repository to evade the budget. Existing sibling code is a disclosed dependency, while new changes required for this project are charged by their actual contribution.

The budgets rely on retaining complete existing public modules, loaders/exporters, host/tool integrations and narrowly isolated exact renderer components, while concentrating new Rust on the accelerated core and integration. Retained upstream source is explicitly a dependency, not a counted-as-Rust reimplementation. All new glue, tests and modifications still count in their corresponding budgets.

Version 2.0 reallocates the same 205k crate subtotal toward shaders, GPU integration and conformance; the Asupersync allowance and 30k reserve remain unchanged. It adds no crates, new runtime, generic cross-API rendering abstraction or policy service. The full feature manifest is one build/test data file inside existing tooling. The forecast is not evidence of implemented size or parity.

### 17.3 Dependency direction

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

Browser bindings should not leak into numerical/compiler modules. Native asset/build tools should not become dependencies of the deployed Wasm module. Keep debug instrumentation and runtime shader translation dependency-pruned. A build feature can omit a component only when the source closure proves it unnecessary or the user explicitly requests a non-full diagnostic artifact; it cannot silently disable existing application functionality.

### 17.4 Size is a product property too

Set separate deployed-byte and peak-memory budgets during Phase 0 for a small closed scene, each full headline artifact, and a runtime-extensible shader/loader application. Use actual compiled artifacts to set those budgets; this plan does not invent a universal sub-megabyte target without measurement.

Tree-shake unused compatibility modules and shader features. Rebuild an application-specific kernel set instead of embedding the full scientific-computing stack in every page.

---

## 18. Implementation sequence

### Phase 0: Full source/feature inventory and executable foundation

**Purpose:** establish complete intended functionality and the actual optimization opportunity before growing the engine.

Pin the complete r186 repo/package and coherent library/toolchain revisions. Reconcile every export/source/addon/registration and official tool/example family into the single feature manifest, including APIs with no tests. Identify inherited dependencies, actual host prerequisites, and concrete native-backend requirements; no “not in the corpus” exclusions. Profile H1/H2/H7/H8 references and freeze initial performance settings/strong-reference selection and startup/memory/tail budgets.

Build the ordinary production compatibility routing and smoke it against package entry points, an actual GL context/sync read, CSS/SVG, a runtime loader/exporter, and existing editor/Inspector integration. The first compatibility component may retain source wholesale; its presence is not an acceleration result. Verify the sound boundary for selecting exact native execution before any canvas binding.

Execute actual Asupersync Rust futures/cancellation in Safari/Chrome and close required host work upstream. Measure bridge variants, queue snapshots, layouts and state-reset counterexamples. Spike the complete finite H2 shader/material controls and H1 node/Inspector path. Establish which metadata/semantics permit automatic new-backend selection without breaking future native observations.

**Deliverables:** immutable full inventory and explicit unknown/gap counts; compatibility-route smoke evidence; reference profiles; real browser task execution; shader/bridge feasibility; resource budgets.

**Exit:** foundation and feasibility pass with no ambiguity that all source features remain required. A compatibility-only wrapper is merely an early correctness baseline; it cannot pass the product gate.

### Phase 1: Complete first applications with real acceleration

**Purpose:** demonstrate automatic transformation of original live applications, with no missing controls or test-only renderer substitution.

Implement coherent module ingestion, semantic objects/replicas, guarded H1 update extraction, and the accurate new WebGPU backend with source-ordered snapshots. Include H1 toon/TSL/Inspector and H2's entire material/control closure, environment lighting, and legacy shader variants. Keep all H1 backend/navigation modes functional in the same deployable artifact through the production exact component where required.

Integrate marching cubes, legal batching/bundles, resource histories, standalone packaging and current-state export/read observations. Add targeted loader/export, DOM/input and native-API integration tests as these public surfaces interact with optimized state. Start mapped upstream tests immediately and accrue both full-functional and forced-new-backend coverage.

**Deliverables:** full H1/H2 output artifacts; every control/mode works; all guards, compatibility bytes and setup costs measured; independently validated new-backend phone/M5 speedups; no file-name/trace-specialized cheat.

**Exit:** first product performance gate and both applications' full functional tests pass. Exact H1 GL mode is not counted as accelerated; it is still implemented and tested.

### Phase 2: Generalize without losing public functionality

Generalize aliases/effects/guards, ownership/materialization, dirty propagation, dynamic topology, animation and procedural kernels. Preserve the complete scalar/public API via retained modules where that is best; shared state must remain correct for controls, raycasts, cloning, serialization and exporters immediately after optimized updates.

Concurrently wire the complete loader/exporter/codec and cold-addon inventory, CSS2D/CSS3D/SVG, audio/media and standard controls to production routes. Use unseen runtime assets/shaders and official custom extension examples. Preserve Node/CommonJS/browser import semantics, worker/offscreen boundaries, and script/permission behavior.

**Exit:** held-out applications and dynamic input sequences work without F3D-specific valid-input refusals. Tests prove both successful specialization and conservative exact/general execution. Every family has an owner and active implementation/tests; any unfinished family remains a release blocker.

### Phase 3: Full new-backend rendering and dynamic shader coverage

Finish every standard material/map/state/light/shadow/texture/camera/render-target path, every standard node/compute entry, portable legacy GLSL, all official pass/effect families, and dynamic source shader/node registration. Reuse construction algorithms; do not force arbitrary public JavaScript into Rust. Preserve standard custom pass/subclass contracts and keep truly native opaque behavior on exact components.

Run all applicable rendering cases with retained renderer submission disabled, beyond H1–H8. Close color/precision/history/diagnostic/resource-life combinations and port no feature by substituting a simpler visual effect. Complete required runtime translators/codecs in extensible applications.

**Exit:** the full standard portable rendering inventory runs through the new backend. Each exact/native exception is evidenced by an actual observable requirement, not a missing implementation. H7/H8 and all advanced cases have complete behavior and performance profiles.

### Phase 4: Feature, environment and tool closure; demanding performance

Close the entire feature manifest and all inherited/additional tests, including previously untested/TODO features, exact WebGL/WebGPU backend APIs, exporters and plugins, media/audio, real XR integrations, physics wrappers, and editor/player/Inspector/devtools workflows. Complete a capable-host matrix; device limitations match the reference and do not delete features.

Finish paired input/runtime-variant cross-feature tests and original-project import/edit/export/reload workflows. Apply measured GPU optimizations only after the conservative full behavior works. Keep exact compatibility independently tested while requiring the new backend for all accelerated coverage and scored rendering.

**Exit:** no unimplemented/unclassified/stubbed valid-source feature, full tests pass, portable-rendering closure passes, and modern workload targets satisfy the unchanged performance criteria. Good compatibility alone and good demo performance alone are both insufficient.

### Phase 5: Complete deployment and release validation

Verify normal package/networked/library delivery and strict standalone closure, including reachable native modes and runtime codec/shader paths. Verify host/CSP constraints, cold startup, memory pressure, device/context loss, mobile thermal behavior, integration permissions and interaction tails. Confirm small closed artifacts exclude unneeded compatibility code without removing reachable features.

Run held-out workloads and final full-feature/forced-new-backend/exact-native suites on appropriate reserved hardware. Reconcile latest stable upstream changes against the tested version; retain immutable earlier results. Publish source/feature ownership, required host matrix, complete reports and raw paired measurements.

**Exit:** all Section 22 gates hold. No full-feature launch with unresolved feature gaps, and no performance-replacement launch without the new-backend speedup gates.

### Solid, frontier, and optional research

**Solid:** bulk host boundary; persistent resources; accurate materials; conservative dirty tracking; recognized update kernels; straightforward instancing; exact lifecycle and compatibility tests.

**Frontier:** automatic closed-island extraction; guards/materialization without losing savings; aggressive legal pass specialization; hardware-specific equivalent compute improvement. These are part of the performance thesis and must earn their complexity.

**Optional research:** generic geometric-algebra optimization, neural rendering, differentiable scene optimization, novel ray tracing, and more general JavaScript compilation. None is needed to declare the primary product complete, and none should consume the compatibility reserve before the core gates pass.

---

## 19. Execution-ready work breakdown

These IDs are proposed implementation units, not claims that beads/issues exist. A task with an early slice and later generalization states that distinction explicitly. Dependencies below refer to the named early acceptance slice unless the row says “closure”; this prevents the first benchmark from depending on a compiler that is postponed until after that benchmark.

| ID | Task | Depends on | Acceptance evidence |
|---|---|---|---|
| F3D-01 | Pin complete upstream/package; census every feature, registration, import, workflow and test | — | Full source-to-feature reconciliation; unknown entries block; actual host/dependency matrix |
| F3D-02 | Execute Rust futures through Asupersync browser host | 01 | Actual task bodies, wakeups, cancellation/drain in Safari/Chrome |
| F3D-03 | Profile strong references and bridge alternatives | 01; bridge integrates 02 | Reference-only H1/H2/H7/H8 and matched completed-work data |
| F3D-04 | Full module/package routing and exact-component construction boundaries | 01 | ESM/CJS/addon/source identities; real GL/sync read; no mandatory GPU for CPU/DOM imports |
| F3D-05 | IDs, layouts, epochs and safe host/resource transport | 02, 03 | Growth/reentry/stale-handle/queue snapshots and separate native-context ownership |
| F3D-06 | Packed transforms with complete public state/materialization | 05 | Matrices/attributes, synchronous queries, clone/serialize/export after updates |
| F3D-07 | Initial accurate new WebGPU material/render slice | 05, 15 | Complete H1 toon/Inspector and H2 starting materials, no retained renderer submission |
| F3D-08 | Legal grouping, per-use uniform versions and bundles | 06, 07, 12 | Guards/callbacks/IDs, red-A/blue-B and bundle/direct tests |
| F3D-09 | Complete marching-cubes computation/attribute behavior | 04, 05 | H2 geometry/normals/colors and every parameter/material variant |
| F3D-10 | Full standalone plus networked/module packaging | 04, 07 | Reachable modes/codecs embedded; native H1 switch/reload; live mode not falsified as offline |
| F3D-11 | First complete-application and new-backend product gate | 08, 09, 10, 12, 15, 22 | Full H1/H2 behavior and >=2× named new-backend phone/M5 results with actual full artifact |
| F3D-12 | Early sound island extraction and irreversible route analysis, then generalize | 04, 05 | Guards/materialization before 11; escaped native access selected exactly before binding |
| F3D-13 | Complete animation/math/geometry scalar behavior and bulk kernels | 06, 11, 12 | All contracts via real owners; event/IK/retarget/query/export combinations and held-out gains |
| F3D-14 | Entire standard material/light/texture/renderer format surface | 07, 11 | Every inventoried family and options, not only example-selected features |
| F3D-15 | Early H1/H2 shader preparation, then full standard ESSL/runtime coverage | 04 | Finite spike before 11; unseen runtime shaders/hooks and correctly retained native semantics |
| F3D-16 | Complete TSL/node/backend extension and compute contracts | 14, 15 | Every standard node/general dynamic path; native custom backend routing; callback frequencies |
| F3D-17 | All pass/effect families, histories, resource reuse and copies/readbacks | 14, 16 | Complete new-backend applicable coverage plus exact sync/native integration |
| F3D-18 | Full feature gates and unchanged upstream tests, continuously | 04; closure uses 12–17 | Every source/behavior entry covered; positive tests beyond TODO/skips; no ownership laundering |
| F3D-19 | All loaders/exporters/registrations/runtime codecs and current-state round trips | 10, 14 | New inputs, every existing format/options, plugin contracts, import/edit/export/reimport |
| F3D-20 | Profile-justified new-backend GPU improvements | 13, 16, 17 | H7/H8/general corpus ablations; retained renderers cannot satisfy accelerated closure |
| F3D-21 | All host/tool integrations and complete official workflows | 04; closure uses 17–20 | Controls/CSS/SVG/audio/media/XR/physics/worker/Inspector/editor/player/devtools coverage |
| F3D-22 | First-backend fault injection through final context/device recovery | 02, 05; closure uses 17 | Queued effects/error scopes/loss, exact native lifetime and stale-result safety |
| F3D-23 | Complete feature-host and performance hardware validation | 18, 19, 20, 21, 22 | Capable-host parity plus unchanged paired performance/thermal/startup/memory gates |
| F3D-24 | Final full-functionality/accelerated-coverage/deployment/size/version audit | 23 | All Section 22 gates; no feature gap hidden by a narrowed manifest or exact renderer |


F3D-15's initial work is source analysis/normalization and WGSL generation, not dependence on an already complete renderer. Its rendered acceptance integrates with F3D-07; they form one early vertical slice rather than a circular demand for finished subsystems. F3D-18 and F3D-22 start with their small early prerequisites and continuously accrue coverage. No task defers the safety or semantic precondition of an earlier claimed performance result.

Use incremental direct-to-main work only when the repository owner requests that implementation workflow. This document does not create a repository, issues, commits, or releases. Build/noninteractive tests may use remote build tooling with isolated target directories; GPU measurements use reserved, noncontended physical sessions. Repository-owned commands remain the verification source rather than depending exclusively on hosted CI.

---

## 20. Principal risks and scope boundaries

| Risk | Earliest decisive test | Response if it fails |
|---|---|---|
| The fastest Three.js baseline already removes the available overhead | H1 and representative baseline decomposition | Stop or substantially revise the thesis; do not weaken the reference |
| Wasm/JS transfer and facade synchronization erase gains | Bridge probes plus real automatic conversion | Change granularity/ownership; abandon per-method Wasm wrapping |
| Mutation coverage cannot preserve dynamic JavaScript cheaply | Escaped aliases and callback stress tests | Keep affected state JavaScript-owned; report cost; broad performance gate may fail |
| Browser Asupersync execution is not sufficiently implemented | Real task-body and cancellation milestone | Complete the upstream host; no alternate runtime hidden in F3D |
| Legacy shader translation becomes a compiler project larger than budget | Full finite H2 corpus and H1 material spike before the first product gate | Reuse a narrow compiler/construction component; stop if required coverage cannot fit |
| Pure-WebGPU-only product would remove real upstream APIs | Complete source/native-contract inventory | Isolate exact backend components; retain full functionality without imposing their restrictions on new WebGPU execution |
| Modern compute demos remain GPU-bound and <2× | H7/H8 complete-pipeline profiles | Find an equivalent algorithm/data-path improvement or fail launch gate |
| Fidelity changes under batching/fusion/precision optimization | Intermediate-target and temporal differential tests | Remove invalid optimization, not the failing oracle |
| Single-file startup costs overwhelm user experience | Cold mobile memory/startup tests | Prune only provably unreachable code and reduce copies; preserve every reachable translator/codec and report its cost |
| Dependency reuse brings native threads, old runtimes, or excessive code | Per-crate Wasm closure and size audit | Refactor narrowly upstream or use a smaller first-party kernel |
| Thermal measurements misrepresent practical performance | Sustained paired phone runs | Base claims on sustained results and disclose startup advantage separately |
| Queue writes or reused bundles expose the wrong resource version | Red-A/blue-B and bundle-then-direct probes | Use immutable slices/encoded copies, correct queue ordering and state invalidation |
| Sync-to-async optimization changes observable interleavings | Reentry, Promise order, exceptions and mixed-render tests | Keep the operation synchronous or prove the specific transformation |
| GPU-only state is lost but recovery is falsely called seamless | Device loss during H8/temporal history | Restore a real checkpoint or expose admitted reset/error; charge checkpoint cost |
| Offline artifact still needs hidden resources or mishandles reloads | Cold-cache no-subresource control/navigation sweep | Close the actual asset graph or fail standalone admission |
| Benchmark hides candidate failures or changes the offscreen workload | Reference-valid denominator and reference on/offscreen comparison | Keep failures in the gate; repair harness before scoring |
| Plan exceeds 245k Rust lines | Continuous source/forecast report | Increase focused upstream reuse and remove new machinery, not existing features or correctness tests |
| Full-suite success hides untested missing features | Feature-to-source/test reconciliation | Add positive behavior tests and close the actual contract; names/stubs do not count |
| Exact components turn the project into a renamed upstream wrapper | Forced new-backend standard-rendering corpus and route attribution | Complete the new backend and meet unchanged speedups; compatibility alone cannot pass |
| Late backend choice breaks canvas identity or native resources | Escaped-context/sync-read and XR replacement tests | Choose exact group before binding, or follow an actually supported source replacement boundary |

### Scope boundaries are not feature exclusions

Do not implement a new JS VM, browser/driver stack, general WebGL-emulation layer, native Metal/Vulkan/D3D backend, physics engine, editor, marketplace, render farm or ML optimization service. **Preserve all existing Three.js functionality in these areas by reusing the appropriate upstream/browser/external component.** Existing editor operations, physics wrappers and legacy/backend APIs are expressly in scope.

No mandatory general Clifford algebra, new neural renderer, speculative hardware ray tracing or mesh-shader architecture is required. This excludes unrelated inventions, not existing nodes/shaders/addons that use capabilities actually supported upstream. The specialized backend remains independent of WebGL; the exact production compatibility component remains available where required.

Third-party products outside the pinned Three.js tree are not all reimplemented. Their use of existing public extension points must continue to work, and official shipped/example integrations are required tested dependencies. Preserve compatible custom JS behavior rather than rejecting it merely because it has not been optimized. Version-pin external showcase integrations and keep public hooks intact for other consumers.

The source's existing host/permission/security/undefined-behavior limits remain limits. A live-service application is supported through live deployment, not a fictitious offline equivalent. Unknown implementation status is not a host limitation; a candidate-specific failure remains a blocking defect. There are no permanent “core only,” “no exporters,” “no XR,” “static shaders only,” or “all tests we happen to pass” release variants labeled full compatibility.

---

## 21. First implementation session

The first session should produce an honest measurement and a tiny working application, not a forest of empty crates.

Begin by recording the pinned Three.js commit and inspecting the complete source/package/feature/test map, including exports, addons, registrations and tool integrations not covered by upstream tests. Establish the full production compatibility component and sound native-context route before replacing renderer construction. Identify the exact source, material/TSL/ESSL variants, Inspector/UI dependencies, navigation, and assets for H1 and H2; record and preserve both H1 backend modes while separately selecting its scored WebGPU workload. Resolve a dated nightly compatible with the selected Asupersync revision. Execute a real browser Rust future and cancellation path; address the host gap before adding renderer abstractions.

Then submit a simple, correctly rendered frame through the bulk bridge and compare it with direct JavaScript WebGPU using the same work. Include the two-draw red-A/blue-B snapshot test and bundle-then-direct state reset before treating the bridge as usable. Keep the public result small, but measure the costs that could invalidate the architecture: command transfer, typed-array copying, initialization, and per-frame synchronization.

Next connect the original H1 source to the candidate compatibility boundary, even before it is fast. Preserve object identity and its dynamic/control behavior. This exposes whether the intended ingestion/ownership model is viable much earlier than a hand-authored Rust demo.

The first useful proof is a self-contained automatically transformed page, its comparison trace, and a measured breakdown. A polished CLI, elaborate evidence schema, or a large status dashboard is not a substitute.

---

## 22. Definition of done

FrankenThreeD is ready for its full-feature performance-replacement release only when **all** of the following hold:

**Full functionality is real.** Every existing feature in the pinned source/package/addon/tool universe has a working production implementation under the same actual source-host prerequisites. No permanent feature cuts, stubs, placeholder shaders, unimplemented codecs/exporters, omitted runtime variants, missing browser/tool integrations, or candidate-specific valid-input refusals remain. A narrowed application profile cannot change this requirement.

**The product is real.** Original projects retain their complete controls, backend choices, dynamic code/assets, import/export and integration behavior. Closable applications build to self-contained HTML containing all reachable code/assets; live/networked/module uses receive their correct deployment without a false offline promise. H1's GL mode actually works, not an unsupported-mode dialog.

**Both compatibility and new-backend coverage are real.** The exhaustive feature manifest, entire active upstream suite, added tests for uncovered features, official workflows and capable-host integrations all pass. Retained production ownership is explicit. Every portable standard rendering family also passes through the new WebGPU backend with retained renderer submission disabled. Genuine native-backend contracts have separately validated exact implementations. A compatibility-only wrapper cannot pass.

**Performance is real.** Every required reference-valid H1–H8 accelerated cell meets the unchanged >=2× individual throughput/confidence floor and every primary stratum meets the >=3× geometric-mean target, along with tails, sustained thermals, controls, startup and memory gates. Measure the ordinary full artifact and include application work, guards, compatibility payload, bridge and GPU completion. Retained renderer execution is not credited as new-WebGPU acceleration; failures never shrink the denominator.

**The foundation is real.** Asupersync owns and drains actual F3D Rust/host tasks within proven cooperative boundaries. Retained source/browser operations keep their real lifecycle rather than inheriting fictitious guarantees. Cancellation is not rollback of issued effects. Resource snapshots, native contexts, error scopes, device loss, exact-component boundaries and delayed completions remain correct.

**The implementation remains focused.** First-party safe Rust, dated nightly, narrow disclosed dependencies, coherent module/runtime identity, eleven crates and the 245k planned Rust ceiling hold. Full feature scope is made feasible by deliberate retained-code reuse, not a hidden rewrite in JS or an erased scope row. Actual size and retained upstream bytes are reported separately.

**Completeness remains maintainable.** A full immutable source-to-feature-to-test map detects newly added or changed upstream contracts. Tested version targets and newer unverified source are distinguished. No test-only fallback, census-only “100%,” or unmeasured speedup is published as implementation evidence.

The product promise is **full Three.js functionality plus independently demonstrated acceleration**, not “a fast supported subset” and not “the same upstream renderer under a new name.” Complete functionality does not assert a physically impossible universal 3× improvement for every GPU-bound operation; the demanding performance contract stays explicit and mandatory.

The plan specifies that complete scope. It does not claim implementation, full test execution or speedup has already been demonstrated.

---

## 23. Source register

Sources below were inspected on September 9, 2026. Three.js code references use the pinned r186 source commit wherever applicable. Asupersync's detailed browser documentation was inspected at commit `efa5798d139ae3e877e13b2b8b5108fc5fe1a625`; other sibling manifests/READMEs were retrieved from their then-current default branches. Before implementation, resolve all chosen dependencies into one coherent set of immutable revisions rather than treating separately inspected files as a tested integration.

The initial source review and this fresh-eyes revision did **not** compile these libraries, run the Three.js suite, measure the target devices, or establish any FrankenThreeD performance result. Version 1.1 added source checks of material/control construction, upload/disposal/node-update behavior, browser timelines, GPU data ordering, and benchmark completion APIs. Version 2.0 additionally rechecked r186 release/package exports, the addon aggregate and wildcard scope, public WebGPU/backend exports, WebGL native APIs, GLTF registration, XR fallback, existing editor state/workflows, and the alternate-renderer/audio/XR API families. This is a full-scope design audit, not a completed executable symbol/behavior census. Design corrections are not execution evidence; the revised task gates identify what must still be demonstrated.

| Ref | Source and relevance |
|---|---|
| [S1] | Three.js r186 release: release identity and date |
| [S2] | Annotated r186 tag object: resolves the compatibility source commit |
| [S3] | Three.js WebGPURenderer documentation: existing backend support |
| [S4] | Khronos WebGL 2 specification: OpenGL ES 3.0 basis |
| [S5] | Pinned Three.js package manifest: test/build entrypoints and package exports |
| [S6] | Pinned E2E runner: exclusions, comparison thresholds and launch configuration |
| [S7] | Pinned render-bundle example: strong existing baseline and dynamic behavior |
| [S8] | Pinned complete example index: verified example paths and coverage families |
| [S9] | Pinned fluid-compute example: current multi-kernel GPU workload |
| [S10] | Pinned individual instanced-skinning example: current compute deformation path |
| [S11] | Pinned VXGI Sponza example: demanding render graph and external asset resolution |
| [S12] | Asupersync README: structured concurrency and qualified adapter guarantees |
| [S13] | Pinned Asupersync browser scheduler contract: required host-turn behavior |
| [S14] | Pinned Asupersync WASM documentation: ownership-ledger versus task-execution scope |
| [S15] | Asupersync browser-core manifest: existing binding/dependency boundary |
| [S16] | FrankenNumPy linalg manifest: actual Rayon dependency |
| [S17] | FrankenSciPy manifest/README: selectable numerical crates and runtime dependency |
| [S18] | FrankenTorch README: documented CPU-first architecture |
| [S19] | FrankenLibC README: Linux/glibc deployment target |
| [S20] | Naga GLSL frontend documentation: supported GLSL entrance and translation limitation |
| [S21] | Chrome 149–150 WebGPU update: immediates and transient-attachment validation |
| [S22] | Chrome 151–152 WebGPU update: optional subgroup-size control |
| [S23] | Rust target documentation: `wasm32-unknown-unknown` environment |
| [S24] | SharedArrayBuffer documentation: cross-origin isolation requirements |
| [S25] | GPUBuffer.mapAsync documentation: ordinary asynchronous mapping contract |
| [S26] | WebKit Safari 26.0 feature announcement: shipped WebGPU support |
| [S27] | Pinned Three.js core unit-test import registry: source-path test routing |
| [S28] | Pinned WebGLRenderLists tests: exact semantic utility assertions |
| [S29] | Pinned marching-cubes example: per-frame procedural work and controls |
| [S30] | FrankenGit comprehensive plan: organizational reference, not inherited project scope |
| [S31] | FrankenNumPy root manifest: workspace structure and runtime version |
| [S32] | Browser GPU API documentation: secure-context requirements |
| [S33] | GPUWeb maintainers on queue writes versus recorded passes: repeated uniform updates are not per-draw snapshots |
| [S34] | GPUQueue.writeBuffer API: copying, offsets, typed-array units and alignment |
| [S35] | WHATWG browser event-loop processing: task/microtask and run-to-completion constraints |
| [S36] | W3C WGSL specification, August 31, 2026 snapshot: types, layout, uniformity and execution rules |
| [S37] | GPURenderPassEncoder.executeBundles API: state reset after bundle execution |
| [S38] | GPUDevice.popErrorScope API: asynchronous error results and scope handling |
| [S39] | GPUDevice.importExternalTexture API: source-dependent snapshot lifetime and media color space |
| [S40] | Pinned marching-cubes material/control construction: default Standard material and legacy ShaderMaterial variants |
| [S41] | Pinned WebGLAttributes implementation: distinct CPU/upload versions, effective update ranges and callbacks |
| [S42] | Pinned NodeFrame implementation: FRAME/RENDER/OBJECT update and updateBefore/updateAfter semantics |
| [S43] | Pinned WebGLGeometries manager: disposal, backend bookkeeping and subsequent registration |
| [S44] | GPUWeb explainer: device model, resource/usage validation and operation timelines |
| [S45] | GPUCommandEncoder.copyBufferToTexture API: copy-specific row-pitch/format constraints |
| [S46] | GPUQueue.writeTexture API: queue texture upload and data-layout contract |
| [S47] | ECMAScript Number and Math specification: source numeric semantics are not Rust casting semantics |
| [S48] | GPUCanvasContext.getCurrentTexture API: acquisition of the current rendering output |
| [S49] | GPUQueue.onSubmittedWorkDone API: completion of previously submitted queue work |
| [S50] | WebGL2RenderingContext.clientWaitSync API: nonblocking polling of a synchronization object |
| [S51] | GPUWeb specification source, resource usages/synchronization/adapters: whole-buffer scopes, per-dispatch versus per-pass validation, and consumed adapters |
| [S52] | Pinned addon aggregate: exporters, controls, renderers, physics, XR and many other existing feature families; wildcard imports extend beyond this aggregate |
| [S53] | Pinned Three.WebGPU.js: public backend/classes, builders, storage, pipelines, node loaders and node exports |
| [S54] | Pinned repository root tree: source/package, examples, editor, devtools, TSL, tests and manual/tooling roots |
| [S55] | Official WebGLRenderer API: genuine native context/state/extension handles, sync/async pixel reads, context reuse and constructor options |
| [S56] | Pinned WebGLXRFallback helper: actual construction-time backend fallback and renderer/session installation |
| [S57] | Pinned editor core: script/player/XR/history/storage/renderer/animation integration state |
| [S58] | Pinned GLTFLoader imports and registrations: full data surface, built-in extensions, decoders and separately registered integrations |
| [S59] | Official CSS3DRenderer API: actual hierarchical DOM/CSS rendering, not interchangeable canvas output |
| [S60] | Official SVGRenderer API: vector output is distinct existing functionality |
| [S61] | Official Audio API: native audio graph, playback, source, filter and lifecycle behavior |
| [S62] | Official WebXRManager API: session/controller/reference-space and renderer integration |

[S1]: https://github.com/mrdoob/three.js/releases/tag/r186
[S2]: https://api.github.com/repos/mrdoob/three.js/git/tags/819fadd6b663b74d828c6af72a543024f74d3877
[S3]: https://threejs.org/docs/pages/WebGPURenderer.html
[S4]: https://registry.khronos.org/webgl/specs/latest/2.0/
[S5]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/package.json
[S6]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/test/e2e/puppeteer.js
[S7]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/webgpu_performance_renderbundle.html
[S8]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/files.json
[S9]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/webgpu_compute_particles_fluid.html
[S10]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/webgpu_skinning_instancing_individual.html
[S11]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/webgpu_vxgi_sponza.html
[S12]: https://github.com/Dicklesworthstone/asupersync/blob/main/README.md
[S13]: https://github.com/Dicklesworthstone/asupersync/blob/efa5798d139ae3e877e13b2b8b5108fc5fe1a625/docs/wasm_browser_scheduler_semantics.md
[S14]: https://github.com/Dicklesworthstone/asupersync/blob/efa5798d139ae3e877e13b2b8b5108fc5fe1a625/docs/WASM.md
[S15]: https://github.com/Dicklesworthstone/asupersync/blob/main/asupersync-browser-core/Cargo.toml
[S16]: https://github.com/Dicklesworthstone/franken_numpy/blob/main/crates/fnp-linalg/Cargo.toml
[S17]: https://github.com/Dicklesworthstone/frankenscipy/blob/main/Cargo.toml
[S18]: https://github.com/Dicklesworthstone/frankentorch/blob/main/README.md
[S19]: https://github.com/Dicklesworthstone/frankenlibc/blob/main/README.md
[S20]: https://wgpu.rs/doc/naga/front/glsl/index.html
[S21]: https://developer.chrome.com/blog/new-in-webgpu-149-150
[S22]: https://developer.chrome.com/blog/new-in-webgpu-151-152
[S23]: https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html
[S24]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer
[S25]: https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer/mapAsync
[S26]: https://webkit.org/blog/17333/webkit-features-in-safari-26-0/
[S27]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/test/unit/three.source.unit.js
[S28]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/test/unit/src/renderers/webgl/WebGLRenderLists.tests.js
[S29]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/webgl_marchingcubes.html
[S30]: https://github.com/Dicklesworthstone/frankengit/blob/main/COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENGIT.md
[S31]: https://github.com/Dicklesworthstone/franken_numpy/blob/main/Cargo.toml
[S32]: https://developer.mozilla.org/en-US/docs/Web/API/GPU

[S33]: https://github.com/gpuweb/gpuweb/discussions/2509
[S34]: https://developer.mozilla.org/en-US/docs/Web/API/GPUQueue/writeBuffer
[S35]: https://html.spec.whatwg.org/multipage/webappapis.html
[S36]: https://www.w3.org/TR/2026/CRD-WGSL-20260831/
[S37]: https://developer.mozilla.org/en-US/docs/Web/API/GPURenderPassEncoder/executeBundles
[S38]: https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/popErrorScope
[S39]: https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/importExternalTexture
[S40]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/webgl_marchingcubes.html#L35-L265
[S41]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/src/renderers/webgl/WebGLAttributes.js
[S42]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/src/nodes/core/NodeFrame.js
[S43]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/src/renderers/webgl/WebGLGeometries.js
[S44]: https://gpuweb.github.io/gpuweb/explainer/
[S45]: https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/copyBufferToTexture
[S46]: https://developer.mozilla.org/en-US/docs/Web/API/GPUQueue/writeTexture
[S47]: https://tc39.es/ecma262/multipage/numbers-and-dates.html
[S48]: https://developer.mozilla.org/en-US/docs/Web/API/GPUCanvasContext/getCurrentTexture
[S49]: https://developer.mozilla.org/en-US/docs/Web/API/GPUQueue/onSubmittedWorkDone
[S50]: https://developer.mozilla.org/en-US/docs/Web/API/WebGL2RenderingContext/clientWaitSync

[S51]: https://github.com/gpuweb/gpuweb/blob/main/spec/index.bs#L1000-L1350

[S52]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/Addons.js
[S53]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/src/Three.WebGPU.js
[S54]: https://api.github.com/repos/mrdoob/three.js/git/trees/148ef33ecb6d2502ff796d4554abd1549c95d519
[S55]: https://threejs.org/docs/pages/WebGLRenderer.html
[S56]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/webxr/WebGLXRFallback.js
[S57]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/editor/js/Editor.js
[S58]: https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/loaders/GLTFLoader.js
[S59]: https://threejs.org/docs/pages/CSS3DRenderer.html
[S60]: https://threejs.org/docs/pages/SVGRenderer.html
[S61]: https://threejs.org/docs/pages/Audio.html
[S62]: https://threejs.org/docs/pages/WebXRManager.html
