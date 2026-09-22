# Guarded MarchingCubes addon integration

`marchingCubesRollupPlugin()` recognizes the exact pinned r186 addon and emits a
branch around only the numerical prefix of its original `update` method. It
calls `compileMarchingCubesKernel` and `instantiateNumericKernel`; it does not
introduce a second compiler, runtime, table copy, or rewritten Three.js class.
Use the plugin before a generic module-URL resolver. The build API/CLI opt-in is
documented alongside the ingestion entry points when enabled there.

The original method, its construction-time geometry and maxPolyCount closures,
field-building functions, exports and publication statements remain JavaScript.
Draw-range callbacks, attribute setters, warnings and exceptions execute outside
the native-attempt catch. Once the numerical outputs have been published, there
is no fallback replay. Native failure executes the entire original numerical
prefix once, including upstream's insufficient-capacity semantics.

## Ownership and execution

The verified base EventDispatcher constructor registers its freshly allocated
identity in a compiler-owned WeakSet. The pinned source module and pinned
`three.core.js` build are recognized. A proxy around a registered object is not
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
The defaults are 2,048 Wasm pages (128 MiB) and 100 million loop-body entries.
Neither limit caps or truncates the retained application's computation.

## Validation scope

Run `node --test tools/ingest/marching_cubes_adapter.test.mjs` after installing
`tools/package-lock.json`. The adapter suite executes the shared Wasm compiler
against an explicitly source-shaped fixture and synthetic topology tables. It
checks publication, callback/exception ordering, reentrancy, table edits,
private-state isolation, guards, rollback, recovery and fuel forwarding. Its
explicit fixture hashes are test anchors, not upstream-conformance evidence.

`marching_cubes_numeric.test.mjs` separately checks the actual pinned upstream
numerics (see `MARCHING_CUBES.md`). The integration workflow runs both suites.
The plugin report and runtime diagnostics distinguish compiled source, actual
Wasm calls and fallback calls. A compiled addon is not a measured acceleration
claim. Full H2 browser/material/control and performance gates remain open.
