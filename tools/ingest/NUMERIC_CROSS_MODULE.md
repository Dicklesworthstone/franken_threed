# Numeric Wasm dispatch across ESM chunks

An admitted numeric function no longer needs its callers in the same output
chunk. Exported closed functions register their **original function identity**
with the emitted dispatcher. Direct calls through named or default imports
look up that identity and reuse the producer's lazy Wasm instance. This is not
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

Native type selection and all existing ownership, bounds, Math, alias, scalar,
and transaction guards still apply on every invocation. Same-type aliases use
ordered shared scratch storage. Late bounds failures and exhausted native work
budgets publish nothing before one original fallback. Concurrent integer and
clamped-buffer variants remain available when selected by producer-side AOT
storage hints. No source allocation or application callback is executed by the
compiler to obtain those hints.

## Scope and costs

Only direct, nonoptional calls spelled through named/default imports are newly
rewritten. Namespace/method calls, optional calls, constructors, tagged calls,
and indirect callback invocation remain JavaScript. A dynamically loaded chunk
can use its static imports, but `namespace.update()` is not rewritten by this
pass. Uninstrumented outside callers still call the original function normally.

The whole producer and its reachable scalar helpers must remain closed under
the existing compiler rules. Cross-chunk dispatch does not compile imported
helpers into a producer, infer storage allocations across chunks, or admit
captures. Mutable original function bindings and direct-eval source units remain
conservative. Existing locally called kernels take priority over new export-only
candidates under the per-unit compilation budget. AOT alternatives remain capped.

Consumer-only chunks can import functions that were not compiled. Those calls
pay a registry lookup and retain JavaScript on a miss. They may cause runtime
assets to be emitted even when a bundle has zero compiled kernels. The report
separates `compiledKernels`/`registeredKernels` from `importedCalls`; a rewritten
lookup is **not evidence of a native call or a speedup**. Runtime diagnostics
remain opt-in and add no application exports. Memory packing is not zero-copy.

Non-ES outputs and outputs requesting source maps remain unspecialized. Reports
retain explicit pre-specialization source coordinates and map preliminary chunk
names to final bundle names. Runtime imports and imported-binding metadata are
kept consistent for producer-only, consumer-only, and mixed chunks.

## Validation

```sh
node --test tools/ingest/numeric_cross_module.test.mjs \
  tools/ingest/numeric_cross_chunk.test.mjs
```

The module tests execute separate ESM graphs and native Wasm against original
JavaScript. The chunk tests invoke the production Rollup plugin hooks through
an explicit contract harness, then execute the emitted ESM/runtime files. They
cover shared instances, relocation, integer variants, live bindings, cycles,
reentrant initialization, transactional fallback, unavailable Wasm, budgets,
metadata, and report/asset collisions. The harness is not a full Rollup run;
full bundling/hash-substitution, browser/GPU, and Rust gates remain separate.
