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

### Source-shaped Euler updates

The same application opt-in also recognizes an ordinary forward child loop
whose body selects one object, reads its per-object Euler input and updates its
rotation. This separate source-pinned library path uses constructor allocation
witnesses and exact method/callback identities; arbitrary property access is
still outside the generic numeric island's contract.

```js
export function rotate(root, count) {
  for (let i = 0; i < count; i++) {
    const object = root.children[i];
    const speed = object.userData.angularVelocity;
    object.rotation.set(
      object.rotation.x + speed.x,
      object.rotation.y + speed.y,
      object.rotation.z + speed.z
    );
  }
}
```

The unchanged first update loop in the pinned H1 application has this shape.
Names and source filenames do not select the implementation. Component
`+`, `-`, `*` and `/` expressions can use different source axes. Scalar counts,
the current child-array length, and own numeric control fields on witnessed
literal objects are guarded at each invocation. The initial domain requires
the pinned built core, its ordinary child arrays, three-argument `Euler.set`,
and source Euler values reached through witnessed data objects such as the
original `Object3D.userData`.

One Wasm call computes Euler results and the pinned `Quaternion.setFromEuler`
arithmetic for all six rotation orders. The original JavaScript engine's native
`Math.cos` and `Math.sin` calls remain in the packing pass, in source order.
The public Euler and Quaternion objects keep identity and remain authoritative
between calls. This is a copying transaction, not persistent packed ownership or
a measured performance result.

Duplicate output objects, any speed/output alias, changed methods or rotation
callbacks, getters, proxies, nonwritable fields, unsupported counts and exhausted
native budgets retain the original loop before any native publication. Reentrant
callbacks and exceptions therefore keep their source partial effects. Live child
membership, API counts and input values are rechecked every call. The report's
`libraryKernels.eulerUpdates` records source spans, operations, retained trig and
the original-loop fallback.

`node tools/ingest/euler_update.test.mjs` checks the unchanged 4,000-object H1
loop, other variable names and expressions, all six rotation orders, identity,
live inputs, aliases, observable guards, reentry and partial failures against
the pinned source. It does not establish complete H1 application or hardware
performance closure.

### Source-shaped instance-matrix updates

The same build option now extracts the second H1 dynamic loop as one Wasm
transaction, including source-shaped loops with different variable names:

```js
for (let i = 0; i < mesh.count; i++) {
  rotation.makeRotationFromEuler(speeds[i]);
  mesh.getMatrixAt(i, temporary);
  temporary.multiply(rotation);
  mesh.setMatrixAt(i, temporary);
}
mesh.instanceMatrix.needsUpdate = true;
```

The compiler derives rotation and multiplication arithmetic from the pinned
`Matrix4` source. All six Euler orders and `premultiply` are covered. Native
JavaScript `cos` and `sin` remain in source evaluation order; one Wasm call
computes the complete loop. Instance stores retain Float32 or Float64 conversion,
while both public scratch matrices retain their full-precision final values and
array identities. The upload request remains at its original source boundary.

Constructor and literal-array allocation witnesses precede descriptor-only
guards for methods, Euler accessors, counts, indexed inputs and writable output
cells. Changed methods, proxies/getters, shared, resizable or immutable buffers, scratch
aliases, unknown arrays and unsupported values retain the complete original
loop before any native publication. Lazy initialization rechecks inputs, and a
failed native budget publishes no speculative changes before source fallback.
This preserves callback reentry and source partial failures without replaying an
already committed update.

`libraryKernels.instancedMatrixUpdates` reports source spans and the copying
ownership route. `node tools/ingest/instanced_matrix_update.test.mjs` compares the
unchanged H1 loop and renamed structural variants against retained source,
including observable scratch state and zero per-instance multiply dispatches on
the admitted path. This is implementation evidence, not a measured speedup or
closure of every instanced animation pattern.

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
