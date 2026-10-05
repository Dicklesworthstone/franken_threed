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

## Binding and fallback boundaries

Every root helper call must resolve to the proven immutable module declaration,
not a same-named callback parameter, catch binding, block function or body var.
The check includes disjoint inner declarations that could otherwise hide a
capture. Parameter-initializer and switch-discriminant scope rules are preserved.
The compiler separately resolves bindings inside the extracted loop and inside
all helpers. Mutation, recursion, helper captures, reference escape, arbitrary
object access and unsupported effects refuse the native attempt, not use of the
application.

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
  tools/ingest/numeric_loop_helpers.integration.test.mjs
```

The tests execute real Wasm in Node and compare against untouched source. They
cover all nine storage layouts, aliasing, coercion order, scalar publication,
shadowing, live Math environments, nested work/bounds rollback, smaller-route
preservation, actual Rollup/CLI builds, dynamic modules, relocated packages and
hosts without Wasm. Application tests inspect real native invocations and
compare upload snapshots after every frame. They do not establish browser/GPU
execution, full application equivalence, renderer compatibility or a speedup.
