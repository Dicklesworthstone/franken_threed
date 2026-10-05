# Native numeric helper loops

The numeric compiler can close an acyclic graph of scalar helper functions
containing `for`, `while`, and `do ... while` loops into the same import-free
WebAssembly module as its selected entry point. A root can update numeric arrays,
return a Number, or delegate all its iteration to scalar helpers.

This extends the existing numeric specialization route. It does not replace
source functions with new public identities, evaluate source at build time, or
claim whole-application closure, GPU execution, or a measured speedup.

## Source use

Ordinary module source can retain its existing calls:

```js
function refine(value, iterations) {
  let result = value;
  for (let j = 0; j < iterations; j++) {
    result = (result + value / result) / 2;
  }
  return result;
}

export function update(out, input, iterations) {
  for (let i = 0; i < out.length; i++) {
    out[i] = refine(input[i], iterations);
  }
}

export function frame(out, input, iterations) {
  return update(out, input, iterations);
}
```

`specializeNumericModule` discovers the update and closes `refine` into its
private native call graph. The existing dispatcher still validates the actual
callee identity and argument/storage ABI at every call. Float32 storage is
promoted to Number precision before arithmetic and rounded only at stores.
Integer storage uses the existing checked integer ABI and conversions.

Discovery also follows calls to immutable module function declarations. A scalar
wrapper such as `function root(n) { return count(n) * 2; }` can therefore become
an entry even when only `count` contains a loop. Previously eligible direct-loop
kernels retain priority when `maxKernels` is small. Uncalled delegating wrappers
are not added in local-call mode; exported roots can be registered through the
existing `crossModule` mode.

Discovery is not a closure proof. Mutable or shadowed helpers, captured values,
non-scalar helper parameters, recursion, and unsupported effects still prevent
whole-function native admission. The original application remains executable.

## Explicit compilation and packaging

`compileNumericKernel` requires `generalControl: true` for helper loops. Supply
`parameterTypes` for the entry and the existing `helperSources` map of immutable
scalar function declarations. The caller remains responsible for matching those
sources to the entry's actual lexical bindings. `compileNumericCandidate` tries
this route after existing legacy, checked, and structured routes; explicit
`generalControl: false`, `structuredLoops: false`, or `checkedIndexing: false`
retain their opt-out behavior.

`buildNumericKernel` supports the same graph in a function-only source module.
Use `functionName` to select the intended public entry when the file contains
multiple loop-bearing functions. The emitted Wasm contains the helper bodies;
its runtime does not need the parser, the source repository, host helper imports,
or a new asynchronous execution boundary. The package preserves the original
function-only module as its fallback.

## One budget per entry invocation

All executed entry and helper loop bodies share `maxIterations`. Calling a
helper does not reset its budget, and separate helpers do not get independent
budgets. Calls in initializers, predicates, updates, and final return expressions
consume the same credits. A false `for`/`while` pre-test consumes no body credit;
`do ... while` consumes one before its first test. A body that takes `continue`
still consumes its entry credit.

The counter is private Wasm state, not a host import or an exported global. The
entry prologue resets it on every native invocation, including after an earlier
trap. Helper-free general-control programs keep the prior local counter and
byte-for-byte emitted binaries. Other legacy routes retain their original ABI.

The existing `maxIterations` bound is between 1 and 1,000,000,000, inclusive,
with a default of 1,000,000. The reachable graph also shares the 64-static-loop
limit. The eight-level loop-nesting limit includes nesting through helper calls,
not just nesting within each individual source function. Existing helper graph,
source-size, statement, and call-depth bounds remain in force.

Budget exhaustion traps before array publication. Without a fallback, the
runtime raises `KERNEL_EXECUTION_FAILED`. With a fallback, the existing runtime
invokes the retained entry with the original arguments and receiver; private
partial array writes are never published. This is not a sandbox or a guarantee
that a nonterminating original JavaScript fallback will terminate.

## Control and numeric semantics

Helper loops preserve Number-valued induction, dynamic ranges and steps,
left-to-right evaluation, loop-carried state, lexical initialization and shadowing,
comma-expression updates, unlabeled `break`/`continue`, and numeric early returns.
A `continue` reaches the correct update or post-test; a `break` skips both.

Every admitted helper must return a Number on all conservatively reachable exit
paths. A return after an unconditional break does not establish a numeric result.
Loops are conservatively allowed to exit, so a loop-only helper generally needs
a final numeric return. Return completeness is not inferred from guessed trip
counts or termination.

Guarded Math intrinsics and exact integer/bitwise lowering remain available in
helper bodies. Their requirements propagate to the entry's existing runtime
checks. No arithmetic reassociation, f32 arithmetic substitution, or parallel
independence is inferred from the presence of loops.

Helper arrays, captured outer state, recursive functions, indirect calls,
`for ... of`, `for ... in`, labeled control, suspension, and arbitrary effects
remain outside this profile. This work does not silently lower them to host
calls inside an otherwise native-labeled kernel.

## Reporting and verification

The existing v8/v9 general-control contract remains in use. `loopCount` includes
reachable helper loops; `maxLoopDepth` includes nesting through calls. Individual
helper loop descriptors include `functionName`, and their source spans are
translated from the helper's own source slice, not the root's slice. A helper
loop descriptor's `depth` is lexical depth within that helper.

The added unit and integration tests execute real Wasm in Node. They cover
repeated Float32/Float64 updates, nested abrupt control, integer operations,
shared budgets and reset, rollback, aliasing, lexical refusals, source discovery,
source-map coordinates, imported identity dispatch, and relocated packages with
native Wasm and with Wasm disabled. They do not establish browser/GPU performance
or whole-application equivalence beyond the admitted numeric profile.
