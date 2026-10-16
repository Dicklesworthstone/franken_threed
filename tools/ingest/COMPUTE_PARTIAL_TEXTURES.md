# Dynamic CPU texture inputs for TSL compute

The r186 compute adapter supports the common texture pool's partial RGBA byte
rows, DataArrayTexture layers, and CompressedArrayTexture layers across authored
mip chains. These inputs work through
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

History belongs to a Source/descriptor, not to its last uploading texture alias.
Disposing that alias preserves the latest composed pixels while another alias
still retains the same source residency. History-only payloads stay charged to
the byte budget and are released with the last alias. Recreating private native
residency from this history does not replay an acknowledged upload callback.

## Compressed array layers

After one complete mip-chain upload, CompressedArrayTexture `layerUpdates` applies
each selected layer to **every** authored level. Slice offsets come from the native
inspector's `sliceBytes` metadata, preserving BC/ETC/ASTC block sizes, small mip
tails, and the uncompressed RGBA8 fallback used by compressed-texture loaders.
No block decoding, transcoding, format substitutions or pixel resizing occurs.
Device compression-feature and source-format admission remain with the pool.

Each level keeps its own baseline even when incoming mip records share one byte
array. Unmentioned layers retain their earlier bytes; modifying a mip array before
an older queued upload finishes cannot clear the newer layer request. Partial
compressed updates without a retained complete baseline are refused: the adapter
does not invent format-dependent encodings for uninitialized compressed blocks.
A source/descriptor change therefore requires another complete compressed upload.

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
its valid predecessor. An exception from `onUpdate` is different from rejected
admission: the native upload has already happened. Its acknowledged version is
kept as the next baseline, and retry does not replay that callback. Disposal
invalidates queued leases and releases their charged pixel copies; shared history
lasts only as long as a remaining alias needs it. The renderer still reports asynchronous source-pool failures
through wait, readback and copy completion.

## Scope and validation

The adapter currently uploads a complete composed private image for each new
version. This enables partial-update **semantics**, not a reduced GPU bandwidth or
speedup claim. Generated mipmaps remain the native source pool's responsibility.
Partial 3D/cube/authored-mip row updates, decoded images and video are not added
by this implementation. Compressed partial updates require a complete baseline;
they do not enable arbitrary compressed subrectangles or writable sampled inputs.

```sh
node --experimental-vm-modules --test \
  tools/ingest/tests_three_compute_textures.mjs \
  tools/ingest/tests_three_compute_partial.mjs
```

The 39 new tests exercise the actual compute mixin and sampled-owner modules,
with recording GPU, pool and shader-builder boundaries. They cover sequential
partial snapshots, zero initialization, aligned/coalesced rows, six typed-array
layer profiles, aliases, callback acknowledgment, rollback, queue rejection,
compilation, nested after-update kernels, compressed mip histories, alias disposal,
callback exceptions, and failure propagation. The 16 existing
sampled-owner regression tests are also run. No physical-GPU pixel, pinned WGSL
compiler, full repository or Rust-suite validation is implied by these tests.
