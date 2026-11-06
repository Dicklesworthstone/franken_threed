# Helper graphs inside application loop islands

Closed loops inside ordinary callbacks, closures and methods can call proven
immutable module helpers without returning to JavaScript between helper calls.
The helper may take numeric typed arrays, forward arguments, return a Number or
return no value, and contain its own bounded loops. The enclosing application
function does not have to be closed or rewritten as a numeric kernel.

```js
function integrate(out, velocity, dt) {
  for (let j = 0; j < out.length; j++) out[j] += velocity[j] * dt;
}
function damp(velocity, factor) {
  for (let j = 0; j < velocity.length; j++) velocity[j] *= factor;
}
function energy(velocity) {
  let total = 0;
  for (let j = 0; j < velocity.length; j++) total += velocity[j] * velocity[j];
  return total;
}
export function makeFrame(out, velocity, upload) {
  let total = 0;
  return function frame(dt, substeps) {
    for (let i = 0; i < substeps; i++) {
      integrate(out, velocity, dt);
      damp(velocity, 0.99);
      total += energy(velocity);
    }
    upload(out, total);
    return total;
  };
}
```

With the existing opt-in application build, an admitted outer loop and its three
helpers execute in one import-free Wasm transaction. `upload` stays at its
original JavaScript boundary. The callback, closures and exported identities
remain application-owned; no renderer or frame scheduler is installed.

Helpers may also use [guarded native Math calls](NUMERIC_GENERAL_CONTROL.md#native-math-inside-numeric-loops),
including `sin`, `cos`, `atan2` and `pow`. Those artifacts declare the retained
host Math boundary in `mathImports`; the helper graph and outer loop remain one
Wasm transaction, with the same fallback and publication contract.

```sh
node tools/ingest/cli.mjs --entry ./src/main.mjs \
  --build-app ./dist/compiled --specialize-numeric
```

The output directory must be fresh. The programmatic build accepts
`{ specializeNumeric: true }`; the standalone `specializeNumericModule` transform
requires `{ loopIslands: true }`. The build's legacy `crossModule:false` option
keeps loop islands off unless `loopIslands:true` is also explicit. Post-link
compilation sees helpers linked into the same ES chunk; unresolved imported
helpers across separate chunks retain their existing identity-dispatch route.

## One source-ordered transaction

A captured array can be used only through helper arguments; it need not be
subscripted by the outer loop. Argument-position discovery proposes storage
layouts, then the compiler independently proves every reachable function.
Existing native guards still check actual scalar types, typed-array layouts,
ownership and aliases. All nine numeric typed-array layouts are supported by the
helper ABI; automatic variant selection remains bounded by storage hints.

Same-type overlapping views share staged storage across every helper. Each view
retains its own length and each store performs the source element conversion.
Adjacent closed loop statements may share that transaction without reordering
or fusing their source operations. Mutable captured scalar outputs use a private
Float64 channel and publish at the original loop boundary.

Root and helper loops spend one invocation-wide work budget. A bounds failure or
exhausted budget discards the entire speculative transaction, including writes
from earlier helpers and scalar reductions. The original loop then executes in
place, not by replaying the surrounding callback. Its ordinary effects and
exceptions remain observable. A new native invocation starts with fresh credit;
the retained JavaScript is not a termination sandbox.

A larger closed region can absorb previously selected direct helper-call
rewrites. Edits are removed only after full region closure succeeds. Failed
closure or a depleted compilation budget leaves the smaller routes intact.
Standalone helper registrations remain lazy for other call sites and exports.
Reports distinguish `absorbedCalls` from remaining `rewrittenCalls`; an absorbed
call is not a separate host-to-Wasm transition.

## Live module bindings in helpers

Automatic application specialization also supports helpers that read module
Number bindings, including live imports and transitive helper calls:

```js
let amplitude = 1;
function wave(x) { return Math.sin(x) * amplitude; }
export function configure(value) { amplitude = value; }
export function update(positions, time) {
  for (let i = 0; i < positions.length; i++) positions[i] = wave(i + time);
}
```

The compiler passes the current `amplitude` through private numeric parameters.
It reads the binding in the helper's module environment after the original call
arguments have been evaluated, and refreshes it for every invocation. A caller's
same-named parameter or local variable cannot replace that binding. The original
helper and exported function objects, declarations and source text stay intact.
The same conversion applies to helper graphs inside extracted callback loops;
ordinary surrounding effects remain at their original boundaries.

Captured scalar bindings must hold primitive Numbers and remain read-only for
the selected region and all reachable helpers. Changes between invocations,
such as `configure` above, are supported. Writes inside the region, including a
module `var` loop header, retain JavaScript. Uninitialized or non-number captures
also retain JavaScript, preserving zero-trip behavior, partial exceptions and
per-iteration coercions. Guarded preparation rechecks the live values before
native execution so host reentry cannot publish a stale snapshot.

Helpers can also read and write live captured numeric typed arrays:

```js
let positions = new Float32Array(12000);
let velocities = new Float32Array(12000);
function integrate(i, dt) {
  positions[i] += velocities[i] * dt;
}
export function update(dt) {
  for (let i = 0; i < positions.length; i++) integrate(i, dt);
}
```

The captured views enter the same staged transaction as explicit array
arguments. Same-type overlapping views share storage, every view keeps its own
bounds, and stores retain their element conversions. Scalar reductions and
writes through other helpers observe those stores in source order. A failed
bounds or work check discards all speculative output before the original
function or loop executes once. Captured arrays may be replaced between calls;
rebinding a captured variable inside the admitted region remains unsupported.

All nine numeric typed-array layouts use the existing ABI and native guards.
Allocation hints from the original capture bindings select bounded variants,
including mixed Float32 geometry, Uint32 indices and clamped byte colors.
Unemitted layout combinations retain JavaScript. Proxies, ordinary arrays,
shared/resizable storage and incompatible overlapping views keep their existing
fallback behavior. Binding identity is checked again after Wasm preparation,
so initialization or memory-growth reentry cannot publish an obsolete view.

This conversion reads declarative bindings only. It does not read properties,
invoke getters, inspect proxies or bake initializer values. Reports identify
admitted bindings in `helperCaptures` and distinguish array/scalar slots with
`helperCaptureKinds` when arrays are present. The raw numeric compiler and
standalone kernel source API still require explicitly closed helper functions.

## Binding and fallback boundaries

Every root helper call must resolve to the proven immutable module declaration,
not a same-named callback parameter, catch binding, block function or body var.
The check includes disjoint inner declarations that could otherwise hide a
capture. Parameter-initializer and switch-discriminant scope rules are preserved.
The compiler separately resolves bindings inside the extracted loop and inside
all helpers. Mutable helper identities, recursion, writes to captured bindings,
unsupported captures, reference escape, arbitrary object access and unsupported
effects refuse the native attempt, not use of the application.

A helper's `Math` binding belongs to its module; the loop's `Math` can belong to
a different callback environment. Both used bindings must pass their own live
intrinsic guard. A helper-only use does not read an unrelated callback binding.
Resolvers are lazy, preserve lexical initialization behavior, and run only after
the existing global descriptor guard; a global accessor is not invoked merely
to attempt native execution.

## Executable verification

```sh
node --test tools/ingest/numeric_loop_helpers.test.mjs \
  tools/ingest/numeric_loop_helper_transactions.test.mjs \
  tools/ingest/numeric_loop_helpers.integration.test.mjs \
  tools/ingest/numeric_helper_captures.test.mjs \
  tools/ingest/numeric_helper_capture_runtime.test.mjs \
  tools/ingest/numeric_helper_array_captures.integration.test.mjs
```

The tests execute real Wasm in Node and compare against untouched source. They
cover all nine storage layouts, aliasing, coercion order, scalar publication,
shadowing, live Math environments, nested work/bounds rollback, smaller-route
preservation, actual Rollup/CLI builds, dynamic modules, relocated packages and
hosts without Wasm. Application tests inspect real native invocations and
compare upload snapshots after every frame. They do not establish browser/GPU
execution, full application equivalence, renderer compatibility or a speedup.
