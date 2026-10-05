# Numeric islands inside live application code

The application specializer can compile closed numeric loops **where they are
written**: inside callbacks, closures, class/object methods, and retained async
or generator functions. Authors do not need to extract a top-level function or
annotate a typed-array ABI. Existing whole-function specialization keeps priority.

```bash
node tools/ingest/cli.mjs \
  --entry ./src/main.mjs \
  --build-app ./dist/compiled \
  --specialize-numeric
```

The output directory must be fresh. The same behavior is enabled by
`buildApplication(entry, outDir, { specializeNumeric: true })` and
`bundleWithRollup(entry, { specializeNumeric: true })`.

For example, the two adjacent loops below form one native island. Property
access, the upload request, the callback, and the enclosing function stay in JS:

```js
export function createUpdater(attribute, velocity, onFrame) {
  let frames = 0;
  return function frame(dt) {
    const positions = attribute.array;
    let squaredLength = 0;
    for (let i = 0; i < positions.length; i++) {
      positions[i] += velocity[i] * dt;
    }
    for (let i = 0; i < positions.length; i++) {
      squaredLength += positions[i] * positions[i];
    }
    attribute.needsUpdate = true;
    onFrame(++frames, squaredLength);
    return attribute;
  };
}
```

## Execution boundary

`for`, `while` and `do/while` may have nested numeric control, including local
`break` and `continue`. Contiguous closed loops share one Wasm call and private
array transaction. A host call, intervening declaration or other statement is a
boundary: it is not moved, suppressed or included by an optimistic purity guess.
When a larger region cannot be compiled, individually closed inner loops remain
eligible. This is source-ordered multi-pass execution, not loop reordering.

Captures are read afresh at each original loop boundary, including new closure
instances, rebound arrays, changed scalar inputs and the current lexical `Math`.
Only declarative binding reads qualify; a property access inside the loop is not
assumed pure. Ordinary property reads outside the island remain in their original
location, as in the `attribute.array` example.

Array type, shape, backing-storage, alias and intrinsic guards run on every
invocation. The existing native runtime keeps source-order writes and each
storage conversion, including f32 rounding and integer wrapping/clamping.
Same-type overlapping views share the staged storage; unsupported overlaps and
ordinary/coercing arrays use the original JS. Integer allocation hints only
choose additional ahead-of-time variants; they do not replace runtime guards.

Counters and reductions can update outer `let`, `var`, parameter and catch
bindings. The compiler uses private f64 scalar storage and publishes back to the
actual lexical bindings only after native success. Scalar values are not narrowed
to f32 when geometry is f32. Const/import/self-name outputs and scalar type
changes outside the numeric compiler's contract remain in JS. A nonnumeric
initial scalar retains the original code without an extra coercion.

The original statements remain in place as fallback. A missing Wasm host, early
ESM-cycle call, unavailable capture, guard miss, bounds trap or native iteration
budget failure publishes no native outputs before running those statements.
Earlier callback effects are **not replayed**. Exceptions after publication are
not caught and converted into a second execution. A zero-trip loop does not gain
an observable exception from an otherwise unobserved TDZ capture.

The transformation adds no scheduler, host yield, frame loop, GPU initialization
or runtime source compilation. Wasm is instantiated lazily and reused per emitted
island/type variant. Async/generator suspension outside an island stays outside;
escaping returns, suspension inside a candidate, effectful calls, unresolved
globals and escaping `var` declarations refuse that candidate.

## Options and inspection

Application specialization remains opt-in. Use
`specializeNumeric: { loopIslands: false }` to keep only existing function-call
specialization. Explicit legacy `crossModule: false` also leaves islands disabled
unless `loopIslands: true` is supplied independently. The direct
`specializeNumericModule(source, { loopIslands: true })` API opts into islands;
its default remains unchanged.

`maxKernels` is shared with whole-function kernels within each source unit.
`maxMemoryPages` limits each native arena. `maxIterations` is shared across all
loops in one native attempt; exhaustion falls back, it does not truncate the
application. These are not a whole-process memory budget or a termination
guarantee for the original JS fallback. Variant families are bounded at one
primary and at most sixteen alternatives.

`f3d-numeric-specialization.json` includes `compiledLoopIslands`, original unit
spans, captures, scalar outputs, loop counts, retained reasons and variant sizes.
A `LoopSequence` record identifies a fused region. For linked builds, source
coordinates refer to the rendered chunk before specialization/hash substitution,
not an invented source map. Runtime assets remain content-addressed and movable
with the emitted package; actual runtime imports appear in Rollup metadata.

## Verification and limits

```bash
node --test tools/ingest/numeric_loop*.test.mjs
```

The five suites cover native dispatch, source transformation, actual application
builds/CLI/dynamic chunks, multi-pass fusion, and scalar-state publication. They
compare executed native Wasm against original JS and count native calls; they do
not claim browser rendering, full Three.js equivalence or measured acceleration.
No GPU timings or physical-device benchmark is provided by this feature.

This is a restricted numerical transformation, not a general JS-to-Wasm compiler.
It does not preserve original function-source strings: applications that observe
source text need an appropriate build contract or the specialization opt-out,
as with other source-changing bundler transforms. Full application compatibility
and the renderer/performance release gates remain separate requirements.
