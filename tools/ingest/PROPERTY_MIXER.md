# Native accumulation and application of animation properties

The existing `specializeNumeric` build recognizes the pinned r186
`PropertyMixer` and lowers its `accumulate`, `accumulateAdditive`, and `apply`
numeric operations to Wasm. Ordinary `AnimationMixer` number, vector, color,
and morph-weight tracks reach this path without application changes. It composes
with the existing source-derived interpolation kernels.

```sh
node tools/ingest/cli.mjs --entry app/index.html --build-app dist \
  --specialize-numeric --package-root /path/to/pinned/three.js
```

`buildApplication(entry, outDir, {specializeNumeric: true})` uses the same
route. The build report contains `libraryKernels.propertyMixers`. Recognition
requires the exact source-module or built-core fingerprint at Three.js commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`; changed implementations retain their
source methods. The three default kernels total 5,208 Wasm bytes before packaging.

## Execution and ownership

The compiler derives the complete numeric operation from the original method
body and its `_lerp`, `_lerpAdditive`, and numeric identity helpers. Accumulation
weights, normal blending, additive initialization/blending, original-value
weighting, and frame-to-frame change detection execute in Wasm. Each operation
packs the property regions it accesses; unused oversized buffer tails are not
copied. Arithmetic remains f64 in the same expression and loop order, including
overlapping regions whose earlier writes feed later iterations.

The public `PropertyMixer`, methods, and buffer retain their identities. Its
weight fields and buffer are materialized synchronously. When values change,
the emitted `apply` method executes the original `binding.setValue` expression
after publication. Method lookup, callback reentry, callback exceptions, and
their observations of both weight fields and the full buffer remain source
effects. The surrounding action iteration, group membership, crossfade control,
loop/finished events, and binding dispatch remain in JavaScript. A loop listener
can still stop a later action during the same mixer update.

Constructor witnesses allow descriptor-only guards on ordinary mixer instances;
proxies do not inherit a target's witness. Native typed-array brand checks admit
attached, fixed, writable Float64Array storage. Custom mix/identity methods,
accessors, coercive arguments, unsuitable ranges, shared/resizable storage,
changed platform/iterator hooks, and native resource failures resume the
original method before publishing native state. NaN results also retain source
execution so aliases observe the source's exact payload bytes. Exceptions and
partial writes on these retained paths are preserved. Hosts without WebAssembly
continue executing the original source implementation.

String and boolean properties keep their source buffers and selection methods.
Quaternion blend calls keep the existing flat-quaternion specialization;
copy-only accumulation and change detection can use the new transaction when
they do not invoke quaternion-specific helpers. This is a copying, guarded
numeric path inside the retained mixer, not whole-mixer batching, a Rust
implementation, or evidence of a measured speedup.

## Verification

`property_mixer.test.mjs` runs real emitted Wasm from both pinned source imports
and the built core. Its ten cases cover byte-exact ordinary and additive blends,
overlapping regions, public identity, synchronous binding state, custom effects,
proxy receivers, throwing stores, callback reentry/throws, unusual storage,
NaN payloads, iterator hooks, budget rollback, bounded packing, and application
builds with and without WebAssembly. A real grouped animation app combines
vector, morph, quaternion, boolean and string tracks, crossfades, group removal
and re-addition, live track edits, and loop/finished listeners. It compares full
property buffers, target values, action state, callback traces and event traces
after every update against the unchanged source, while requiring actual Wasm
calls for all three operations.

All ten focused cases and nine adjacent interpolant cases pass. Chromium
151.0.7922.34 also matches all twelve grouped-mixer update snapshots exactly,
with 98 native operations (54 accumulation, 12 additive accumulation, 32 apply),
55 retained calls in the mixed numeric/quaternion/string/boolean app, and zero
browser errors. This is CPU arithmetic and source-effect evidence; renderer
execution, whole-application compatibility and hardware performance remain
separate claims.
