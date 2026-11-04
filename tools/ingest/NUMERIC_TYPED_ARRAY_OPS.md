# Guarded typed-array operations and callbacks

The numeric compiler now emits native Wasm for discarded-result `set`, `fill`
and `copyWithin` calls. Bulk-only functions do not need an artificial JavaScript
loop to qualify. Immutable helper graphs, local borrowed references and ordinary
application callbacks can use the same execution path.
`reduce` and `reduceRight` also produce numeric results through inline callbacks
within closed functions, helpers and admitted loop regions.
Discarded-result `forEach` calls admit inline numeric callbacks that update
staged arrays and captured scalars, including ordinary animation closures.
`some`, `every`, `find`, `findIndex`, `findLast` and `findLastIndex` admit inline
predicates with source-ordered effects and early termination.

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
`typed-array-set`, `typed-array-fill`, `typed-array-copyWithin`,
`typed-array-reduce`, `typed-array-reduceRight`, `typed-array-forEach` and the
six `typed-array-<search-method>` names; their
`loopCount` includes these work regions, not only explicit JavaScript loops.
Native-call counts and measured timing remain separate from static reports.

## Ordered numeric reductions

Inline arrow expressions and anonymous callbacks with a single numeric `return`
can compute sums, weighted totals and other ordered scalar reductions:

```js
export function weightedSum(values, weights, seed) {
  return values.reduce((total, value, index) => total + value * weights[index], seed);
}
```

Callbacks accept up to four plain identifier parameters: accumulator, element,
index and the original receiver view. Numeric seeds and captured scalars/views
use the existing invocation guards. Callback names shadow enclosing bindings
only inside their own body, including nested reductions. The receiver and length
are captured before the initial value is evaluated; callback reads and helper
writes observe staged storage in exact left-to-right or right-to-left order.

An omitted seed uses the first or last element. An unseeded singleton invokes no
callback; an empty unseeded input takes the original path for its source `TypeError`.
Each callback invocation charges the existing shared work budget. A bounds or
budget failure rolls back the complete transaction before the original source
runs once, preserving preceding writes and errors.

Reduction results can appear in closed numeric expressions. Automatic extraction
keeps the existing statement/loop boundaries; it does not create islands around
arbitrary expressions. External callback identities, asynchronous callbacks,
escaping reference results and arbitrary callback bodies remain in JavaScript.

## Numeric forEach callbacks

Inline synchronous arrow functions and anonymous functions can update a buffer
and accumulate captured numeric state in one transaction:

```js
positions.forEach((position, index) => {
  positions[index] = position + velocities[index] * delta;
  total += positions[index];
});
```

Callbacks accept up to three plain identifier parameters: element, index and
receiver view. Numeric assignments, updates, lexical locals, closed helpers,
Math, admitted array operations, branches and loops reuse the existing compiler.
An expression body can perform one of these effects. A callback `return` ends
only the current invocation; supported effects in its argument still execute.
The outer function and subsequent elements continue normally.

The iteration counter is independent of the mutable element and index
parameters, which are initialized anew for each invocation. The original
receiver and its length are captured once, including when the callback rebinds
a different local reference. Reads observe earlier staged writes and overlapping
views in source order. Callback-local declarations shadow outer bindings only
within their lexical scope; captured totals publish with the same transaction
as array writes. Nested callbacks and explicit loops share the invocation's
work budget. Empty inputs invoke no callback and consume no iteration credit.

Automatic discovery follows captured array aliases into these bodies. The
island planner prefers a closed containing operation or sequence over its
callback's inner loops, yielding one native boundary without overlapping edits.
If the containing callback is not closed, independently closed inner regions
can still qualify under the existing source rules.

External callback identities, named or asynchronous callbacks, default/rest or
destructured parameters, `thisArg`, `this`, `arguments`, escaping reference
values and rebinding the third callback parameter remain in JavaScript. The
compiler does not introduce a generic higher-order call mechanism or allocate
callback objects in Wasm. A changed method receives the original literal
callback through the existing guarded source fallback.

## Short-circuit predicates and searches

Closed functions, helpers and admitted loop regions can search numeric views
without evaluating callbacks for unvisited elements:

```js
export function allInside(positions, bound) {
  return positions.every(position => position >= -bound && position <= bound);
}

export function firstActive(weights, minimum) {
  return weights.findIndex(weight => weight > minimum);
}
```

The six search methods accept one inline synchronous predicate with up to three
plain identifier parameters: element, index and receiver. A predicate may use
numeric truthiness, Boolean comparisons, closed Boolean or numeric helpers,
early returns, branches, nested loops and staged mutations. Falling through or
returning without a value is falsy. Each visited element spends one credit from
the shared invocation budget, in addition to any nested work. Early termination
does not charge for or read remaining elements.

`some` and `every` produce actual JavaScript Boolean results at the public
boundary and work directly in admitted predicates. Their internal f64 0/1
representation carries an explicit `resultType: "boolean"` manifest tag. The
runtime validates this result before publishing any writes and converts it to
a Boolean. Older readers reject the new tag; they cannot expose 0/1 as Numbers.
Closed helpers also distinguish Boolean returns from numeric returns, including
transitive Boolean helpers and all-Boolean logical/conditional results. Boolean
scalar locals, Boolean arguments, mixed Boolean/Number returns and Boolean
values used as numeric arithmetic or storage remain outside this profile.

`findIndex` and `findLastIndex` return numeric indices, including native `-1`
for a miss. `find` and `findLast` return the original loaded Number on a hit,
including NaN and signed zero. A miss aborts the private transaction and runs
the original source to produce exactly `undefined`. No sentinel substitutes for
that public value. The abort discards all earlier staged array and captured
scalar writes; original code performs its effects once in the existing source
region. Host calls surrounding the region are not replayed.

Forward and reverse searches snapshot their receiver and length once. The
private counter and original loaded element are independent of callback
parameter assignments and writes to the receiver. Later visits observe earlier
staged mutations, including overlapping aliases. The third callback parameter
keeps the original receiver even if another local reference is rebound. Nested
search/forEach returns exit only their own callback. General callback identity,
`thisArg`, reference-valued predicates and the other forEach exclusions above
continue to use source execution.

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

Native admission requires direct, non-optional method names; `set`, `fill`,
`copyWithin` and `forEach` require discarded results. Chained calls, escaping
`fill`/`copyWithin` results, computed names, spread arguments, unsupported method
counts, coercive objects and
ordinary-array `set` sources retain JavaScript. Numeric arguments and source
views must satisfy the actual invocation guards; discovery alone proves none
of these properties. Array allocation, `subarray` and sorting are
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
with and without Wasm. Reduction cases also cover both directions, seeds,
empty/singleton inputs, callback shadowing, nested/helper effects and source
callback identity under method overrides. The `forEach` tests additionally cover
callback-only returns, mutable parameters, lexical captures, nested-loop
selection, staged mutation, scalar publication and source exceptions after
partial work. Search tests cover all six methods and layouts, exact Boolean and undefined
results, both iteration directions, original-value snapshots, partial budgets,
helper predicates, captured-state/alias rollback, method overrides and relocated
dynamic builds with or without Wasm. These are execution/equivalence tests,
not browser/GPU validation, full Three.js compatibility or a speedup.
