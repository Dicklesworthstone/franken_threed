# Typed-array helper graphs in native numeric kernels

The numeric specializer can compile a closed update whose work is split among
helpers taking typed arrays. A wrapper need not index its array arguments itself.
Array positions propagate through direct immutable helper calls, then the whole
function graph must pass the existing compiler proof and runtime guards.

```js
function integrate(out, velocity, dt) {
  for (let i = 0; i < out.length; i++) out[i] += velocity[i] * dt;
}
function damp(out, factor) {
  for (let i = 0; i < out.length; i++) out[i] *= factor;
}
function update(out, velocity, dt, factor) {
  integrate(out, velocity, dt);
  damp(out, factor);
}
export function frame(out, velocity, dt, factor) {
  update(out, velocity, dt, factor);
}
```

Build the ordinary application with the existing opt-in:

```sh
node tools/ingest/cli.mjs --entry ./src/main.mjs \
  --build-app ./dist/compiled --specialize-numeric
```

The output directory must be fresh. The programmatic equivalent is
`await buildApplication(entry, outDir, { specializeNumeric: true })`.
For the example, an admitted `update` runs both helpers inside one import-free
Wasm instance and one array transaction. No helper calls back into JavaScript,
allocates a second staging buffer, or publishes intermediate arrays. This does
not fuse/reorder the source loops or establish a measured speedup.

## Semantics and storage

Helpers may read/write indexed elements, read intrinsic `.length`, forward an
array parameter to another closed helper, return Numbers, or return no value.
Void calls are admitted only in discarded-value contexts; mixing numeric and
bare returns or using a void result in arithmetic remains a refusal. Existing
scalar locals, arithmetic, bitwise operations, guarded Math intrinsics, branches,
loops, early returns and unlabeled break/continue keep their existing contracts.

Each typed-array argument is passed as a private pointer and its own intrinsic
length. A helper gets a bounded native specialization per parameter-storage
signature. Float64Array, Float32Array, Int8Array, Uint8Array, Uint8ClampedArray,
Int16Array, Uint16Array, Int32Array and Uint32Array use the existing checked ABI.
Every store performs the source element conversion before a subsequent read;
Float32 storage does not narrow intermediate Number arithmetic.

Helpers share the caller's staged memory, including same-type overlapping views.
Read/write effects propagate through cached and transitive calls. Each access
checks its own view, never the size of the shared allocation. Fractional,
nonfinite and out-of-view subscripts abort the native attempt rather than being
truncated into another element. Numeric negative zero remains element zero.

Calls evaluate arguments once, left to right. A compound array update evaluates
its destination and reads the old value before RHS helper effects. General
root/helper loops share one invocation-wide work budget, including helper calls
in tests, updates and return expressions. Exhaustion or an index trap publishes
no speculative writes; the original JavaScript receives the original arguments
and receiver. The next native invocation starts with fresh work credit. A budget
is not a sandbox: retained JavaScript can itself be nonterminating.

## Admission boundaries

The direct compiler needs `checkedIndexing:true` for array helper arguments;
`generalControl:true` implies it and allows loops within helpers. The candidate
compiler tries these existing routes automatically unless explicitly disabled.
A direct runtime caller must request `preserveAliasing:true` to admit overlapping
same-type views. Automatic source dispatch already supplies that ordered-storage
contract. Mixed writable element-type aliases retain JavaScript.

Automatic layout enumeration remains bounded. Float layouts and existing
allocation/wrapper hints select variants, including integer inputs and outputs;
unlisted storage combinations retain the original source. Discovery is not a
type proof and never reads source object properties. Original function identities
and exports remain intact; native dispatch is guarded at rewritten direct calls
and registered imported calls. Source text containing rewritten calls may change,
as with the existing specialization contract.

Only proven immutable direct function declarations are helper bindings here.
Closures/captures, recursion, array reference assignment or escape, array-valued
local aliases, array methods, optional/spread helper calls, arbitrary object
access and unsupported effects still refuse compilation, not application use.
The source integration does not expand loop-island capture admission to arbitrary
helper bindings. Original callback/property/upload effects remain at their
existing JavaScript boundaries. No renderer or frame loop is installed.

## Verification

```sh
node --test tools/ingest/numeric_array_helpers.test.mjs \
  tools/ingest/numeric_array_helpers.integration.test.mjs
```

Tests execute real Wasm in Node, all nine element layouts, aliased views, per-view
bounds, effect ordering, work-limit fallback, source coercion/refusals, imported
identities, real Rollup/CLI builds and relocated packages with and without Wasm.
They do not establish browser/GPU execution, full application equivalence or a
performance improvement. Existing host shape/ownership and policy guards apply.
