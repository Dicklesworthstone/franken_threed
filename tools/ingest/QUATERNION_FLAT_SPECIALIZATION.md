# Native quaternion arithmetic in source animation

The existing numeric-specialization build option now recognizes the pinned
r186 `Quaternion.slerpFlat` and `Quaternion.multiplyQuaternionsFlat` methods.
Ordinary `QuaternionLinearInterpolant`, quaternion keyframe tracks and additive
`PropertyMixer` calls enter the guarded Wasm arithmetic through their original
static-method calls. Applications keep their source animation APIs.

```sh
node tools/ingest/cli.mjs --entry app/index.html --build-app dist \
  --specialize-numeric --package-root /path/to/pinned/three.js
```

`buildApplication(entry, outDir, {specializeNumeric: true})` uses the same
path. The existing report exposes
`numericSpecialization.libraryKernels.quaternionFlat`, separately from generic
loop counts. Without the numeric option, the source modules remain unchanged.

## Execution and source behavior

The compiler requires the exact pinned Quaternion source blob or pinned
`three.core.js` build. It derives the arithmetic from those method bodies;
modified modules retain their complete JavaScript implementation. The original
methods keep their names, argument counts and dynamic lookup positions.
Replacing a static method, interpolant or property-binding hook continues to
execute that replacement at its original call site.

Source `Math.acos` and `Math.sin` remain JavaScript operations. Their exact
results feed the import-free Wasm arithmetic, including source comparisons,
sign correction, interpolation, square-root normalization and output
calculation. Four values from each source are captured before any destination
write, preserving in-place operations and overlapping typed-array views.
Destination conversion and publication retain x/y/z/w store order.

Animation interval selection, action scheduling, crossfades, blending order,
events and property binding remain in JavaScript. A loop listener can stop a
later action within the same mixer update, and a property-binding callback
still observes each synchronous source publication. This is native arithmetic
inside retained mixer execution; no whole-mixer batching or speedup is claimed.

## Guards and fallback

The native path admits numeric typed arrays with fixed, attached, unshared
storage, sufficient extents and numeric offsets. It retains the original method
for ordinary arrays, proxies, BigInt views, coercive offsets, short views,
shared/resizable storage, immutable destinations or changed required
platform/Math descriptors. Guard inspection does not invoke application getters
or Proxy traps. Wasm unavailability or initialization failure also keeps the
source path working.

A staged NaN result uses the original method before any visible write, because
JavaScript and Wasm may produce different NaN payload bytes. Source exceptions
and preceding partial stores remain intact on fallback; no published native
write is replayed.

`quaternion_flat.test.mjs` covers source and built-core bundling, real
interpolants and mixers, crossfades/additive blending, callback/event order,
in-place and mixed-layout storage, destination conversions, hook replacement,
guard fallbacks, partial-store exceptions, application build/report emission
and execution without WebAssembly. These Node execution checks do not establish
browser-wide numerical parity or measured acceleration.
