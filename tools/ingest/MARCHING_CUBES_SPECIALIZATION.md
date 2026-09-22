# Guarded MarchingCubes addon integration

`marchingCubesRollupPlugin()` recognizes the exact pinned r186 addon and emits a
branch around the numerical prefix of its original `update` method and the
field-building loops. It uses `compileMarchingCubesKernel`,
`compileMarchingCubesFields` and `instantiateNumericKernel`; it does not
introduce a second compiler, runtime, table copy, or rewritten Three.js class.
The existing `--specialize-numeric` application opt-in includes this pass:

```sh
node tools/ingest/cli.mjs --entry app/index.html --build-app dist \
  --specialize-numeric --package-root /path/to/pinned/three.js
```

`buildApplication(entry, outDir, {specializeNumeric: true})` and
`bundleWithRollup(entry, {specializeNumeric: true})` use the same path. An options
object forwards `maxMemoryPages` and `maxIterations` to both generic and library
kernels. Without the numeric opt-in, the addon and base modules are unchanged.
The library report is nested at `numericSpecialization.libraryKernels.marchingCubes`
in both the returned build manifest and emitted `f3d-numeric-specialization.json`.
Library counts do not inflate the generic inferred-function counts. Direct
Rollup users must place `marchingCubesRollupPlugin()` before generic URL resolvers.

The original methods, construction-time geometry and maxPolyCount closures,
field setup, exports and publication statements remain JavaScript.
Draw-range callbacks, attribute setters, warnings and exceptions execute outside
the native-attempt catch. Once the numerical outputs have been published, there
is no fallback replay. Native failure executes the entire original numerical
prefix once, including upstream's insufficient-capacity semantics.

## Ownership and execution

The verified base EventDispatcher and Color constructors register freshly
allocated identities in a compiler-owned WeakSet. Their pinned source modules
and the pinned `three.core.js` build are recognized. Color registration happens
before an overridden `set()` can return another object or Proxy. A proxy around a registered object is not
registered. Unknown receivers/materials are rejected without property probes;
known owners require own data fields and an own writable data `count`. Accessor
fields, unsupported shapes and borrowed receivers retain JavaScript. Materials
without an own data `flatShading` field also conservatively retain JavaScript.
This is an opt-in source transformation in a realm with trusted runtime
bootstrap, not a sandbox or admission of arbitrary caller-provided descriptors.

Each addon module has a lazy dispatch token. Native initialization happens only
on an admitted invocation, and a failed initialization is not retried every
frame. Changed WebAssembly bindings or memory-growth hooks retain JavaScript
without invoking replacement getters. Restoring platform bindings, capacity or
disjoint buffers permits later native calls. The generic host refuses aliased
writable storage, shared/resizable buffers, unsupported views, capacity failures,
memory limits and exhausted execution fuel before publication.

The ABI passes live edge and triangle tables, all three instance-private edge
lists, current field/cache/palette and output views, grid scalars and output flags.
It preserves native-to-JavaScript transitions and cross-instance cache isolation.
UV/color toggles and reinitialization use fresh views; no scene data is baked in.
Each kernel defaults to 2,048 Wasm pages (128 MiB) and 100 million loop-body entries.
Neither limit caps or truncates the retained application's computation.

## Field construction

The same application opt-in compiles `addBall`, `addPlaneX`, `addPlaneY`,
`addPlaneZ`, `blur` and `reset`. Their numerical source spans are lifted directly
from the independently hash-verified addon into the shared checked compiler.
The source's operations are not replaced with an algebraically similar formula.
Reset preserves the y/z normal-cache components; blur keeps the exact neighbor
order and progressively rounded scalar accumulator from upstream.

Color conversion, scalar setup, and blur's `field.slice()` execute once in the
original method before the native attempt. A refused attempt resumes at the
original loop, not the method entry. Color/owner accessors, proxy Colors,
coercible non-numeric arguments, aliases, short views and changed `Math.sqrt`
retain source execution without speculative property callbacks. Genuine Color
instances (including default, array, CSS and hex conversions) provide guarded
own numeric channels. Partial writes and exceptions from retained JavaScript
remain observable exactly as in the original method.

Each field method has an independent lazy, reusable kernel per addon module.
Initialization failure is cached; shape, alias, memory and fuel refusals permit
later recovery. Calls remain synchronous, publishing current field/palette/cache
storage before returning; no field operation uploads geometry or resets `count`.
The application report exposes `compiledFieldKernels` and source spans. Runtime
`marchingCubesDiagnostics(effect).fieldKernels[method]` separates per-instance
native/fallback calls from the existing polygonizer counters. No public Three.js
exports or function wrappers are added, and source-tree imports and split chunks
use the same compiler-owned ownership registry.

## Validation scope

Run `node --test tools/ingest/marching_cubes_adapter.test.mjs` after installing
`tools/package-lock.json`. The adapter suite executes the shared Wasm compiler
against an explicitly source-shaped fixture and synthetic topology tables. It
checks publication, callback/exception ordering, reentrancy, table edits,
private-state isolation, guards, rollback, recovery and fuel forwarding. Its
explicit fixture hashes are test anchors, not upstream-conformance evidence.

`marching_cubes_numeric.test.mjs` separately checks the actual pinned upstream
numerics (see `MARCHING_CUBES.md`). `marching_cubes_bundle.test.mjs` additionally builds and executes the unmodified
pinned addon and Three.js core through the real Rollup/application/CLI path,
compares all 256 cube cases in eight output modes, exercises real materials,
callbacks and fallback recovery, and checks HTML packing and report consistency.
`marching_cubes_fields.test.mjs` checks source-lifted field loops across six
resolutions, nonfinite data and rollback. The full-addon suite also verifies
actual native field calls, Color variants, accessor/proxy/coercion traces,
one-time blur copies, field budgets, source-tree imports and dynamic chunks.
The integration workflow runs all four suites with locked dependencies and
the exact upstream checkout; it does not substitute host doubles for build tests.
The plugin report and runtime diagnostics distinguish compiled source, actual
Wasm calls and fallback calls. A compiled addon is not a measured acceleration
claim. Full H2 browser/material/control and performance gates remain open.
