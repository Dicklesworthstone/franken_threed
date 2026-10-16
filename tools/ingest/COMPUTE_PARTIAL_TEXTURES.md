# Dynamic CPU texture inputs for TSL compute

The r186 compute adapter supports the common texture pool's partial RGBA byte
rows and uncompressed DataArrayTexture layer uploads. These inputs work through
`compute`, `computeAsync` and `compileComputeAsync`, alongside native target and
StorageTexture bindings. GPU execution still belongs to F3D, not a retained
upstream renderer.

## Captured content

A partial update composes an immutable source version from the preceding retained
snapshot of the same Source and complete native descriptor. Only the requested
physical upload spans come from the current CPU array. Unmentioned edits in that
array do not leak into a dispatch. Before any full upload, unmentioned texels are
zero, matching newly allocated native texture storage. Changing Source, extent or
a descriptor field that changes residency (including sampler state) starts a new
baseline. An update without ranges/layers is still a full upload.

For example, after initially consuming a DataTexture:

```js
// All source row-range offsets/counts are component counts, not byte offsets.
texture.image.data.set(new Uint8Array([255, 0, 0, 255]), 4);
texture.addUpdateRange(4, 4);
texture.needsUpdate = true;
renderer.compute(kernelReadingTexture);
```

The row must fit within the inspector's admitted unflipped RGBA byte base image.
Coalescing, component-to-pixel rounding and UNPACK_ALIGNMENT offsets follow the
existing `three_textures.mjs` uploader. Public range order/objects are not sorted
or merged by capture. Array-layer requests use tightly packed whole layers and
preserve the underlying typed bits, including integer and half/float data. Other
format, device-feature and sampler restrictions still come from the source pool.

Queued A and B updates capture distinct immutable pixel versions even when the
renderer is blocked preparing a scene. B includes A's admitted changes, but not
later unmarked CPU writes. Source aliases with the same descriptor/version share
one payload and the existing source pool's native residency/upload callback.

## Admission and acknowledgment

Capturing all inputs and reserving a public compute FIFO slot form one synchronous
transaction. A rejected binding, invalid dispatch size or full FIFO cannot become
the baseline for a subsequent partial update. Compilation performs the same
transaction before its asynchronous pipeline preparation. Before/after source
callbacks remain outside capture transactions so nested source computations keep
their existing ordering.

An actual upload acknowledges the original row/layer request before `onUpdate`,
only when Source, texture/source versions, image/data identities, and the request
contents still match. An older queued upload cannot clear a newer request. Public
arrays, Source objects and source constructors/IDs are not replaced or allocated.

Old baselines remain charged until a replacement commits; both old and new images
must fit the snapshot byte/version budgets at peak. A failed replacement retains
its valid predecessor. Disposal invalidates queued leases and releases their
charged pixel copies. The renderer still reports asynchronous source-pool failures
through wait, readback and copy completion.

## Scope and validation

The adapter currently uploads a complete composed private image for each new
version. This enables partial-update **semantics**, not a reduced GPU bandwidth or
speedup claim. Generated mipmaps remain the native source pool's responsibility.
Partial compressed mip chains, 3D/cube/ authored-mip row updates, decoded images and
video are not added by this implementation.

```sh
node --experimental-vm-modules --test \
  tools/ingest/tests_three_compute_textures.mjs \
  tools/ingest/tests_three_compute_partial.mjs
```

The 22 new tests exercise the actual compute mixin and sampled-owner modules,
with recording GPU, pool and shader-builder boundaries. They cover sequential
partial snapshots, zero initialization, aligned/coalesced rows, six typed-array
layer profiles, aliases, callback acknowledgment, rollback, queue rejection,
compilation, nested after-update kernels and failure propagation. The 16 existing
sampled-owner regression tests are also run. No physical-GPU pixel, pinned WGSL
compiler, full repository or Rust-suite validation is implied by these tests.
