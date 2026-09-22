# Budgeted general numeric control (ABI v8)

The numeric compiler can now lower `for`, `while`, and `do…while` loops in
closed numeric functions to import-free Wasm. This includes nonzero and reverse
ranges, dynamic steps, mutable induction variables, loops inside branches,
mixed nested loops, scalar-only functions, and zero-argument numeric functions.
These are native loops, not source unrolling or a fixed library of kernels.

## Application and package entry points

`specializeNumericModule` discovers these loops in top-level function
declarations, including loops under lexical blocks and `if` statements. Direct
calls in the source unit use the existing lazy dispatch and retain the original
function identity, exports, argument evaluation order, and fallback. Loops in a
nested callback do not make its enclosing function a candidate. Discovery is
not admission: the entire function and every reachable helper must compile.

`compileNumericCandidate` tries general control after the existing prefix,
checked-index, fixed-unrolling, and structured-loop routes. Successful older
routes retain their bytecode and priority. `generalControl: false`,
`checkedIndexing: false`, or `structuredLoops: false` prevents automatic general
control admission. `generalControl: true` forces compilation of the original
source using v8; explicitly disabling either prerequisite is an error.

A standalone package can be built without manually extracting a kernel:

```javascript
// updates.mjs
export function updateRange(out, input, start, end, step) {
  for (let i = start; i < end; i += step) out[i] += input[i] / 3;
}
```

```javascript
import { buildNumericKernel } from './tools/ingest/numeric_kernel_build.mjs';

buildNumericKernel('updates.mjs', 'generated/range-kernel', {
  parameterTypes: ['f32[]', 'f32[]', 'f64', 'f64', 'f64'],
  maxIterations: 100000,
});
```

Import `createKernel` from the generated `kernel.mjs`, create an instance once,
and call its synchronous `run(out, input, start, end, step)`. Function-only source
files can also contain immutable scalar helper declarations; specify
`functionName` when multiple loop entries make selection ambiguous. The emitted
package retains the original source and supports relocation and no-Wasm hosts.

## Execution contract

Number arithmetic, comparisons, tests, initializers, and updates preserve source
order. Induction variables remain f64 Numbers, rather than wrapping/truncating
at 32 bits. Each Float32 store rounds at that source store, not at an arbitrary
batch boundary. `continue` reaches the correct `for` update or `do…while` test;
`break` and early `return` skip them. Unlabeled loop exits are supported.

All v8 loop bodies in one invocation share `maxIterations` body-entry credits
(default 1,000,000; accepted range 1–1,000,000,000). Outer loops consume credits
as well as nested loops. A false pre-test spends none, and a `do…while` body
executes once before its first test. The counter resets for each invocation.
The executable enforces the cap; it is not merely a host metadata declaration.
Compilation admits at most 64 loops and eight nested levels.

`maxIterations` applies to v8, not successful legacy routes. Force
`generalControl: true` on an explicit compiler/package call to use v8 budgeting
for an otherwise legacy-compatible function. The older v7 nested-loop cap
remains `maxNestedIterations`.

Every executed array access checks its own view before address conversion.
Arguments are guarded and accessed views packed before execution. Only after a
successful whole call are declared output ranges published. A late bounds trap
or exhausted budget publishes nothing and invokes the original function once
when fallback is configured; without fallback, the runtime throws. This does
not impose a termination bound on the original JavaScript fallback, which can
itself be nonterminating. This mechanism is not a sandbox.

Automatic application dispatch uses the existing same-type alias-preserving
storage contract. Explicit runtime/package callers keep their existing strict
alias default. Mixed-type writable overlaps retain JavaScript. Shared,
resizable, detached, proxy, and unsupported typed-array inputs remain subject
to the existing guards; no new coercions or user getters are introduced.

## Scope and evidence

v8 has one appended intrinsic length per array parameter, in declaration order,
and no synthetic array trip-count/stride descriptors. Scalar-only functions
append no lengths. Manifests report the control contract, executable budget,
loop count, depth, and checked-view semantics. Specialization reports nested
loop spans in the original source unit's coordinates.

The admitted expressions and helpers remain deliberately bounded. Object
properties, arbitrary host calls, capturing closures, `var`, labeled control,
and unsupported expressions retain the original application. Numeric-returning
entries still require a final numeric return. Integer topology arrays remain
read-only. This is not arbitrary-JavaScript compilation, zero-copy execution,
or a measured-speedup claim.

Run the direct compiler/runtime and application/package integration tests:

```sh
node --test tools/ingest/numeric_general_control.test.mjs \
  tools/ingest/numeric_general_control_integration.test.mjs
```

The tests execute native Wasm, compare results to original JavaScript, exercise
rollback and fallback, verify standalone relocation/no-Wasm behavior, and retain
seven byte-for-byte v1–v7 compiler fixtures. Browser/GPU, full Rollup packaging,
and Rust workspace validation are separate gates.
