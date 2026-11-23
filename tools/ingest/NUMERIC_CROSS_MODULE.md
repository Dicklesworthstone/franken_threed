# Numeric Wasm dispatch across ESM chunks

An admitted numeric function no longer needs its callers in the same output
chunk. Exported closed functions register their **original function identity**
with the emitted dispatcher. Direct calls through named/default imports,
static module namespaces and immutable awaited-import bindings look up that
identity and reuse the producer's lazy Wasm instance. This is not
function replacement, import rebinding, or runtime code generation.

## Application builds

The existing opt-in application build path enables cross-chunk dispatch:

```sh
node tools/ingest/cli.mjs --entry app/index.html \
  --build-app generated/application --specialize-numeric
```

`numericKernelRollupPlugin` now defaults to `crossModule: true`. To keep
within-chunk specialization only, pass `crossModule: false` in its options or
in `buildApplication`'s `specializeNumeric` options object. `maxIterations` is
also forwarded to the numeric compiler, alongside `maxKernels` and
`maxMemoryPages`. Native general-loop budget exhaustion still retains the whole
original call, not a partial computation. The budget does not bound JavaScript
fallback execution or successful legacy loop routes.

Direct users of `specializeNumericModule` retain its previous default:

```javascript
const result = specializeNumericModule(source, {
  crossModule: true,
  runtimeModule: './shared/numeric_dispatch.mjs',
  maxIterations: 100000,
});
```

All participating source units must resolve that runtime to the **same ESM
module instance**. The application plugin emits one content-addressed dispatcher
and one kernel runtime asset, with relative imports from every changed chunk.
Moving the entire emitted graph preserves those relationships. Independent
runtime instances have independent registries, not a global name-based cache.

## Identity and execution contract

Named/default import aliases, export aliases, and re-export chains work because
they refer to the same function object. A registry key is never a function name,
URL, export spelling, property lookup, or source-text comparison. Weak keys do
not keep an otherwise unreachable function alive. Creating a registration does
not instantiate Wasm or read the producer's live Math binding.

The original admitted kernel keeps its identity, name, arity, descriptors,
source, and exports. Generated call-site helpers evaluate the original callee
before arguments. A live alias rebound during argument evaluation therefore
still calls the previously read function; the next call sees the new binding.
Shadowed imports, uncompiled callees, proxies, and early calls in ESM cycles
retain the actual source callee and its undefined direct-call receiver. Late
registration permits subsequent calls to use Wasm without replaying module
initialization or changing top-level-await/dynamic-entry scheduling.

Namespace calls retain their module receiver as well:

```js
import * as kernels from './particle-kernels.mjs';
export function frame(positions, velocity, dt) {
  kernels.integrate(positions, velocity, dt);
}
```

The compiler proves that `kernels` resolves to the static namespace import at
that call site. It selects the live `integrate` export before evaluating the
original arguments, then carries that exact function and namespace through
dispatch. Every fallback retains the namespace as `this`, including an
uncompiled method, early module-cycle call, changed live export, unavailable
Wasm, guard failure or exhausted work budget. Receiver state is invocation-local;
reentry cannot overwrite it. An admitted native function cannot observe `this`,
so no receiver is added to the Wasm ABI.

The same path supports default exports and string-literal member names, such
as `kernels.default(...)` and `kernels['update positions'](...)`. It does not
read an export twice or invoke a callee's `.call`/`.apply` properties. Imports
shadowed by parameters, body vars, block declarations, catch bindings or loop
bindings stay on their original member-call path. Parameter initializers and
switch discriminants retain their own lexical lookup rules.

The same dispatch is available after a dynamic import:

```js
export async function loadSimulation(preview) {
  const kernels = await import(preview ? './preview.mjs' : './simulation.mjs');
  return (positions, velocity, dt) => kernels.integrate(positions, velocity, dt);
}
```

The receiver must be a lexical `const` initialized directly by `await import(...)`.
Top-level await, nested async functions, block/loop scopes and closures over that
binding are supported. Import operands stay in place and execute once, including
finite choices imported from configuration modules. Module evaluation, promise
resolution, suspension, rejection and binding initialization retain their source
order; registration alone does not instantiate Wasm. A call before the binding
initializes still throws its original temporal-dead-zone error before arguments.

A dynamic import can resolve to an arbitrary value when a module exports a
callable `then`. Dispatch therefore proves only the immutable receiver binding,
not namespace identity or property purity. The original property access still
executes once before arguments, including getters/proxy traps and their exact
exceptions. Native admission uses the resulting function identity. Registry
misses and every guard failure retain the selected function and actual receiver,
including when an argument replaces the receiver's method.

Immutable destructured imports can use the same shared producer:

```js
export async function loadSimulation(preview) {
  const { integrate: step, fallback = step } =
    await import(preview ? './preview.mjs' : './simulation.mjs');
  return (positions, velocity, dt) => step(positions, velocity, dt);
}
```

This accepts shallow `const` object patterns with fixed named or string-literal
keys, aliases and defaults. The original destructuring retains getter/default
evaluation, initialization order and exceptions, even for arbitrary then-export
values. Direct calls preserve the selected function and their undefined receiver;
they do not reread the module property. A later export/property change therefore
does not replace an already initialized destructured binding. Native function
identity, lexical shadowing and temporal-dead-zone checks remain mandatory.
The same path works in emitted module workers, including relocated output and
the original JavaScript fallback when Wasm is unavailable or a budget is exhausted.

Native type selection and all existing ownership, bounds, Math, alias, scalar,
and transaction guards still apply on every invocation. Same-type aliases use
ordered shared scratch storage. Late bounds failures and exhausted native work
budgets publish nothing before one original fallback. Concurrent integer and
clamped-buffer variants remain available when selected by producer-side AOT
storage hints. No source allocation or application callback is executed by the
compiler to obtain those hints.

## Immutable arrow and function-expression kernels

Top-level `const` bindings can be whole-function producers as well as helpers:

```js
export const integrate = (positions, velocity, dt) => {
  for (let i = 0; i < positions.length; i++) {
    positions[i] += velocity[i] * dt;
  }
};
```

Closed concise arrows, block-bodied arrows and function expressions use the same
numeric compiler, typed-array guards and transactional fallback as declarations.
Their original function objects, source text, inferred names, arity and exports
remain intact. Only code generation normalizes the expression into a declaration.
Native registration occurs after the root and all reachable immutable helper
initializers execute. Earlier calls retain JavaScript, including calls within the
same declaration; temporal-dead-zone errors and top-level-await ordering stay
source-owned.

The whole body must satisfy the existing numeric admission rules. Lexical
`this`/`arguments`, recursive private names, async/generator functions and mutable
bindings do not gain a native route from this syntax support. Imported direct
calls reuse the same original-identity registration across modules and emitted
worker graphs. Existing declaration kernels retain their compilation-budget
priority. This expands eligibility; it does not establish a speedup.

## Scope and costs

Only direct, nonoptional calls through named/default imports, proven static
namespace imports or the immutable awaited-import bindings above are rewritten.
Arbitrary object methods, dynamic member names, optional calls, constructors,
tagged calls and indirect callback invocation remain JavaScript. Mutable
awaited-import bindings, nested/rest/dynamic-key destructuring, import promise callbacks and
direct `(await import(...)).method(...)` expressions remain outside this proof.
Uninstrumented outside callers still call the original function normally.

The pass works on the source unit it receives. A bundler may normalize a static
namespace call into a named-import call first; that form uses the existing
direct-import route. Namespace execution tests therefore run actual separate
ESM modules through `specializeNumericModule`, independently of normal builds
that perform this normalization.

The whole producer and its reachable scalar helpers must remain closed under
the existing compiler rules. Cross-chunk dispatch does not compile imported
helpers into a producer or infer storage allocations across chunks. Producer-local
live module Number/typed-array [captures](NUMERIC_LOOP_HELPERS.md#live-module-bindings-in-helpers)
keep their existing guards. Mutable original function bindings and direct-eval source units remain
conservative. Existing locally called kernels take priority over new export-only
candidates under the per-unit compilation budget. AOT alternatives remain capped.

Consumer-only chunks can import functions that were not compiled. Those calls
pay a registry lookup and retain JavaScript on a miss. They may cause runtime
assets to be emitted even when a bundle has zero compiled kernels. The report
separates `compiledKernels`/`registeredKernels` from `importedCalls`; a rewritten
lookup is **not evidence of a native call or a speedup**. Runtime diagnostics
remain opt-in and add no application exports. Memory packing is not zero-copy.

Non-ES outputs remain unspecialized. ES builds can request external, inline or
hidden source maps while keeping native specialization. The transform emits
character-level mappings that Rollup composes with its original module maps and
other build plugins. Original statements, retained loop fallbacks and later
application code keep their original file/line/column locations; generated
runtime scaffolding does not claim an application source location. These are
JavaScript source maps, not Wasm instruction debugging metadata.

Reports retain explicit pre-specialization chunk coordinates and map preliminary
chunk names to final bundle names. Runtime imports and imported-binding metadata
are kept consistent for producer-only, consumer-only, and mixed chunks.

## Validation

```sh
node --test tools/ingest/numeric_const_roots.test.mjs \
  tools/ingest/numeric_cross_module.test.mjs \
  tools/ingest/numeric_cross_chunk.test.mjs \
  tools/ingest/numeric_namespace_runtime.test.mjs \
  tools/ingest/numeric_namespace_imports.integration.test.mjs
```

The module tests execute separate ESM graphs and native Wasm against original
JavaScript. The chunk tests invoke the production Rollup plugin hooks through
an explicit contract harness, then execute the emitted ESM/runtime files. They
cover shared instances, relocation, integer variants, live bindings, cycles,
reentrant initialization, transactional fallback, unavailable Wasm, budgets,
metadata, and report/asset collisions. The harness is not a full Rollup run;
full bundling/hash-substitution, browser/GPU, and Rust gates remain separate.
