# Native buffer references and ping-pong pipelines

Closed numeric kernels and helpers can keep typed-array references in initialized
`const` and `let` bindings, select between same-layout views, and rebind mutable
references. General-control roots can also rebind their array parameters. This
admits iterative double-buffer algorithms without making each pass a separate
JavaScript call or requiring a different application array API.

```js
function solve(a, b, passes) {
  let source = a, destination = b;
  for (let pass = 0; pass < passes; pass++) {
    for (let i = 0; i < source.length; i++) {
      destination[i] = source[i] / 3 + 1.5;
    }
    const previous = source;
    source = destination;
    destination = previous;
  }
  return source[0];
}
export const frame = (a, b, passes) => solve(a, b, passes);
```

The existing opt-in application build discovers this code automatically:

```sh
node tools/ingest/cli.mjs --entry ./src/main.mjs \
  --build-app ./dist/compiled --specialize-numeric
```

Use a fresh output directory. The programmatic build accepts
`{ specializeNumeric: true }`. The standalone source transform keeps its existing
defaults; `{ loopIslands: true }` enables closed statements inside retained
callbacks and methods. References declared inside those statements participate
in the same transaction, including calls to eligible immutable module helpers.

## Reference semantics, not array allocation

Each reference initialization or assignment snapshots both the current borrowed
pointer and that view's intrinsic length. A later rebind does not change a prior
`const` snapshot, the caller's variable, or another function's parameter binding.
Conditional initializers and assignments evaluate their selector once and only
the selected reference value. Helpers receive their own pointer/length arguments.
No reference initialization allocates, copies, resizes or publishes array data.

Array parameter rebindings in roots require `generalControl:true`; automatic
candidate compilation selects this route. Its source loop predicates re-read the
current view length rather than retaining a counted loop's old array bound.
Local references require the checked-index ABI. The host ABI layouts and the
candidate compiler's preference for earlier admitted routes remain unchanged.

Float64Array, Float32Array, Int8Array, Uint8Array, Uint8ClampedArray, Int16Array,
Uint16Array, Int32Array and Uint32Array retain their element conversion at every
store. A reference has one storage layout within a specialization. A branch or
assignment that would mix layouts refuses that variant; runtime guards select an
eligible variant or execute the original JavaScript. Automatic layout selection
remains bounded; supported ABI layouts do not imply every combination is emitted.

Same-type overlapping views share the caller's staged transaction storage. Each
access checks the selected view's own bounds. Array writes, scalar reductions,
helper calls and swaps remain in source order. The existing shared work budget
covers root and helper loops. Bounds failures and exhausted budgets publish no
speculative changes before the original function or loop executes as fallback.

## Whole reference graph and source discovery

Memory effects are resolved after compiling every assignment and helper call.
The compiler follows all potential reference owners, including cycles from
swaps and sources assigned syntactically after an earlier read. This ensures the
next loop iteration cannot access an unstaged or unpublished input. The owner
union is deliberately conservative and can copy unchanged views; it is not an
alias-independence proof or a measured reduction in copying.

Source discovery propagates possible array slots through local initializers,
assignments, conditional values and reordered/transitive helper arguments.
Selectors remain numeric control, not additional array candidates. Discovery
never authorizes a binding: the compiler separately checks lexical scope, TDZ,
constness, helper identity and closure, and the host guards actual values. The
finite local worklists and bounded helper graph terminate on reference cycles.

## Retained boundaries

References cannot escape as function results, enter scalar arithmetic or identity
comparisons, or expose properties other than supported element access and
intrinsic length. Array allocation, methods such as `subarray`, unsupported
object operations, uninitialized bindings and mixed scalar/reference assignment
retain JavaScript. Helpers with captured state, recursion or unsupported effects
keep their existing refusal behavior. Valid application code is not rejected
merely because this native route is unavailable.

Rebinding an array variable captured from outside an extracted loop still stays
in JavaScript: the loop's scalar publication channel is not used to publish a
pointer as an application object. References declared inside the region, and
array parameters/local references of a whole compiled function, are admitted.
Original exports, function identities, surrounding callback effects and emitted
runtime guards retain their existing contracts. This installs no renderer or
frame scheduler and makes no browser/GPU, compatibility or speedup claim.

## Executable tests

```sh
node --test tools/ingest/numeric_array_references.test.mjs \
  tools/ingest/numeric_array_references.integration.test.mjs
```

Tests run real Wasm in Node against untouched JavaScript, across all nine element
layouts, cyclic swaps, conditional selections, alias snapshots, overlapping and
unequal-length views, helper parameter rebinding, source scope/TDZ checks,
NaN payload preservation and complete bounds/work-limit rollback. Real linked
dynamic application builds compare arrays and upload snapshots on every frame.
CLI output is executed after relocation in fresh processes both with and without
Wasm. These are execution and equivalence checks for the tested cases, not
performance measurements or a full application-compatibility claim.
