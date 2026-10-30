# Native arithmetic in source animation tracks

The existing `specializeNumeric` application build lowers the pinned r186
`LinearInterpolant`, `CubicInterpolant` and `BezierInterpolant` numeric segments
to Wasm. Ordinary number, vector, color and morph tracks reach them through the
retained `AnimationMixer` and keyframe-track factories. Quaternion tracks keep
the separate existing flat-quaternion kernel.

```sh
node tools/ingest/cli.mjs --entry app/index.html --build-app dist \
  --specialize-numeric --package-root /path/to/pinned/three.js
```

`buildApplication(entry, outDir, {specializeNumeric: true})` uses the same route.
The existing numeric report includes `libraryKernels.animationInterpolants`.
Recognition requires the exact source-module or built-core fingerprint; changed
implementations retain their original source path.

## Execution and ownership

The kernels derive their arithmetic from the actual method bodies. Linear
weights, cubic polynomial evaluation and the Bézier Newton/helper graph execute
in Wasm without reassociation or f32 intermediate arithmetic. Three compact
kernels total 5,231 Wasm bytes with the default limits, before packaging. Only the active two or four
sample vectors and applicable tangent controls are packed, independent of the
complete track or public result-buffer length.

Interval searching and cache state, cubic ending settings, action scheduling,
crossfade control, loop/finish events and property binding remain in the original
JavaScript. Numeric property accumulation and blending have a separate
[PropertyMixer route](./PROPERTY_MIXER.md). Source property reads and coercive prefixes keep their
original order and execute once. A mixer listener may still stop a later action
within the same update. Public interpolants, methods and result buffers keep
their identities, names and synchronous behavior.

Native array-brand checks admit fixed attached numeric typed storage. Output
conversion occurs at the original element stores. Ordinary arrays, proxies,
shared/resizable or immutable output storage, insufficient extents, overlapping
sample/output views, changed platform/Math hooks and native budget failures
resume the unexecuted original suffix before any native result is published.
NaN results also use that source path to preserve observable typed-buffer payload
bytes. Hosts without WebAssembly continue to execute the source implementation.

This is arithmetic execution inside the retained mixer, with copying and guard
costs. It is not whole-mixer batching, a Rust implementation, or a measured
speedup. Bézier interpolation without tangent controls retains its original
linear fallback. Discrete interpolation and unrecognized custom interpolants
remain unchanged.

## Verification

`animation_interpolants.test.mjs` executes real emitted Wasm from both source
imports and the pinned built core. It compares real mixers, live samples and
tangents, interval seeks and endings, all numeric output conversions, property
and proxy observations, aliases, partial-store exceptions, Math/iterator hooks, NaN bytes,
bounded active-sample packing, rollback and no-Wasm application builds against
the untouched reference. All nine focused cases and eleven adjacent quaternion
cases pass. A separate Chromium 151.0.7922.34 execution compares 18 mixer/control
snapshots and 12 interpolation buffers exactly against the pinned source, with
62 linear, 29 cubic and 29 Bézier Wasm calls and zero native fallbacks or
unexpected browser errors. This is CPU arithmetic evidence; renderer execution,
whole-application equivalence and hardware performance remain separate.
