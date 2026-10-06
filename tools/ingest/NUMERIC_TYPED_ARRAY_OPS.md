# Guarded bulk typed-array operations

The numeric compiler now emits native Wasm for discarded-result `set`, `fill`
and `copyWithin` calls. Bulk-only functions do not need an artificial JavaScript
loop to qualify. Immutable helper graphs, local borrowed references and ordinary
application callbacks can use the same execution path.

```js
export const makeFrame = (positions, source, upload) => {
  let total = 0;
  return function frame(delta) {
    positions.set(source);
    for (let i = 0; i < positions.length; i++) {
      positions[i] += delta;
      total += positions[i];
    }
    positions.copyWithin(1, 0, -1);
    positions.fill(0, -1);
    upload(positions, total);
    return total;
  };
};
```

An admitted sequence executes in **one native transaction**. The original loop
order, per-store numeric conversions and any same-type overlapping views remain
observable inside that transaction. The `upload` call stays in JavaScript at its
original boundary; no renderer or animation scheduler is installed.

## Build integration

Use the existing opt-in application build with a fresh output directory:

```sh
node tools/ingest/cli.mjs --entry ./src/main.mjs \
  --build-app ./dist/compiled --specialize-numeric
```

The programmatic application option is `{ specializeNumeric: true }`.
`specializeNumericModule` discovers bulk-only function declarations for its
normal direct-call path. Its `{ loopIslands: true }` option also extracts bulk
statements from retained callbacks/methods and combines adjacent bulk statements
and closed loops. Arbitrary calls, declarations and suspension points remain
boundaries. Existing literal-loop kernels retain compilation-budget priority.

Post-link compilation sees immutable helpers merged into the same ES chunk.
Producer registration and imported calls continue to use original function
identities across chunks. The generated application needs only the existing
self-contained runtime and dispatcher assets; the bulk emitter is build-time
code, not another shipping interpreter or host adapter.

## Native semantics and limits

Receivers and arguments are captured in source evaluation order. Numeric range
arguments implement truncation, negative-relative indices, NaN-to-zero range
behavior, infinities, clamping and omitted defaults. `set` checks its destination
extent even for an empty source. Each reference retains its own current view
length, including after local or parameter rebinding.

Same-layout `set` and `copyWithin` use Wasm `memory.copy`: overlapping ranges
behave like a source snapshot, and float NaN payload/sign bits are not converted
through scalar arithmetic. Non-overlapping cross-layout `set` performs native
Number element conversion. All nine numeric layouts are supported, including
clamped bytes. Mixed-layout overlapping writable storage retains JavaScript
under the existing alias guard instead of guessing representation semantics.

Each bulk operation charges its affected element count to the same invocation
budget as root/helper loops. JavaScript loop body entries cost one; an operation
on zero elements costs zero. Exceeding the budget aborts the entire speculative
transaction, including earlier bulk writes and scalar reductions. The original
statements then run in place without replaying the enclosing callback. Original
JavaScript exceptions and preceding source-visible writes remain intact.

Reports retain the existing `loopIslands` aggregate, with distinct
`TypedArrayBulkStatement` and `NumericWorkSequence` candidate kinds. Admitted
regions declare `typedArrayMethods`. Control reports identify method work as
`typed-array-set`, `typed-array-fill` or `typed-array-copyWithin`; their
`loopCount` includes these work regions, not only explicit JavaScript loops.
Native-call counts and measured timing remain separate from static reports.

## Live method guards and fallback

The runtime validates native method descriptors without invoking getters. An
instance override, per-constructor prototype override, changed shared method,
subclass, non-native receiver, or unsupported storage causes original-JavaScript
fallback. Requirements are rechecked after preparation, including possible
memory growth. Math requirements retain their separate live lexical guards.

The embedded `f64-operator-order+guarded-array-methods-v1` semantic contract makes
older runtimes reject the artifact rather than silently omit these checks.
As with the existing Math/typed-array primordials, trusted runtime bootstrap is
required; this is not a sandbox for an already-compromised host realm.

Native admission currently requires direct, non-optional method names and
discarded results. Chained calls, escaping `fill`/`copyWithin` results, computed
names, spread arguments, unsupported method counts, coercive objects and
ordinary-array `set` sources retain JavaScript. Numeric arguments and source
views must satisfy the actual invocation guards; discovery alone proves none
of these properties. Array allocation, `subarray`, sorting and reductions are
not added by this change. A compilation refusal does not remove those features
from the application's retained JavaScript.

## Verification

```sh
node --test tools/ingest/numeric_typed_array_ops.test.mjs \
  tools/ingest/numeric_typed_array_ops.integration.test.mjs
```

Tests execute real Wasm in Node, covering all nine layouts, 81 conversion pairs,
overlap, bit preservation, exact range semantics, descriptor guards, helper
arguments, rollback, automatic source discovery, imported identities and actual
Rollup/CLI builds. Linked dynamic applications compare arrays and upload
snapshots after every frame. Relocated output is tested in fresh processes both
with and without Wasm. These are execution/equivalence tests for the exercised
cases, not browser/GPU validation, full Three.js compatibility or a speedup.
