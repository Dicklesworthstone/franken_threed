# Bulk skeletal palette updates

The existing `specializeNumeric` build lowers the eligible pinned
`Skeleton.update()` bone loop into one synchronous Wasm call. This includes
the source `Matrix4.multiplyMatrices()` arithmetic and palette packing across
the complete skeleton, rather than a separate boundary crossing for each bone.
Ordinary animated scenes and GLTFLoader-created skeletons use the same path.

```sh
node tools/ingest/cli.mjs --entry app/index.html --build-app dist \
  --specialize-numeric --package-root /path/to/pinned/three.js
```

`buildApplication(entry, outDir, {specializeNumeric: true})` enables the same
transformation. The report exposes `libraryKernels.skeletonUpdates`. Exact
source-module and built-core fingerprints authorize the transformation at
Three.js commit `148ef33ecb6d2502ff796d4554abd1549c95d519`; modified methods retain
their source behavior. Float32 and Float64 palette variants total 11,716 Wasm
bytes before packaging.

## Source semantics and ownership

The compiler derives the loop and matrix expressions from the pinned source.
Arithmetic keeps the source f64 expression order, with Float32 rounding at the
original palette stores. Public skeleton, bone and array identities remain
unchanged. Palette values and the shared scratch matrix's final state are
materialized before the original `boneTexture.needsUpdate = true` statement.
Texture setter observations, callback reentry and exceptions therefore retain
their source ordering. Padding and unused palette tails are not overwritten.

Ordinary allocation witnesses make descriptor-only admission possible without
probing arbitrary proxies. Bone/inverse arrays, matrix storage and source methods
must satisfy the guards before native state is published. Unknown arrays,
accessors, custom matrix methods, scratch feedback aliases, shared/resizable
buffers, insufficient storage and resource failures retain the complete original
loop. That path preserves its partial mutations and exceptions. Exceptional NaN
results also retain source execution to preserve payload bits visible through
typed aliases. Hosts without WebAssembly continue executing the original code.

The mixer, action/event control flow, scene hierarchy and renderer remain their
existing owners. This is a guarded CPU palette transaction with input/output
copying. It is not a Rust Skeleton implementation, complete pose propagation,
whole-mixer batching or a measured speedup.

## Verification

`skeleton_update.test.mjs` passes twelve focused checks using real generated Wasm
from both source and built-core imports. They cover live poses and inverse binds,
Float32/Float64 output, sparse/null/duplicate bones, array species and getters,
custom effects, scratch aliases, texture callback reentry/throws, partial writes,
storage/resource guards, mixer crossfades, an ordinary application build and a
host without WebAssembly. Ten PropertyMixer, twelve Matrix4 and fifteen subtree
transform checks also pass, for 49 focused and affected checks.

`node tools/ingest/skeleton_update.browser.mjs` loads the actual pinned
Soldier.glb twice and animates each model independently. Chromium 151.0.7922.34
matches all twelve checkpoints and 589,824 RGBA bytes exactly, together with
complete palettes, public bone matrices, CPU-deformed vertices, mixer events,
callback traces, inverse-bind edits and texture replacement. The four skins have
49, 2, 49 and 2 bones. The candidate executes 48 whole-palette Wasm calls covering
1,224 bones, with zero fallbacks and zero browser/WebGL errors.

Both browser lanes use the retained pinned WebGL renderer to isolate skeletal
CPU equivalence. New-renderer ownership and physical-device performance remain
separate evidence requirements.
