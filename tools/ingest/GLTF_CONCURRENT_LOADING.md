# Bounded parallel glTF resource loading

`loadGltfAsset` loads independent external buffers concurrently, with a shared
per-asset queue that also bounds subsequent lazy image requests. The default is
four complete resource reads at once. `maxConcurrentRequests` accepts integers
from 1 through 32; use 1 for serial I/O admission. The owning scene loader forwards
this option through its existing `assets` settings:

```js
const model = await loadGpuGltfAnimationScene(device, modelURL, {
  signal: controller.signal,
  assets: {
    maxConcurrentRequests: 4,
    maxBytes: 128 * 1024 * 1024,
    maxResourceBytes: 64 * 1024 * 1024,
  },
  textures: { maxConcurrentImageRequests: 4 },
});
```

A permit is held until the entire response body has been consumed or its read
has failed. Receiving HTTP headers alone never admits another request. Scheduling
is FIFO, but completion order may differ; returned buffers retain their original
glTF indices. This is an I/O queue, not an animation clock or GPU scheduler.

## Resource identity and budgets

Canonical URLs, including query strings but excluding fragments, and identical
data URIs still share one cached promise and one byte charge. Duplicate buffer
declarations use the largest declared extent when checking their shared result.
A fetched root model file counts toward `maxResources`; all model bytes count
toward the encoded-byte budget.

All selected buffer URLs, BIN extents, unique resource counts and aggregate
minimum declared bytes are preflighted before dependency requests start. Meshopt
and Draco fallback pruning remains authoritative: skipped buffers are neither
fetched nor replaced with fabricated storage. Decoding still starts only after
all required encoded buffers are ready and retains its existing byte budgets.

Each accepted streamed chunk is charged synchronously against the same total
budget. Concurrent reads cannot each spend the same remaining bytes. Bytes from
partial or failed reads remain charged, and failed requests remain cached. The
limits bound admitted encoded bytes, not browser networking overhead, decoder
scratch memory, temporary copies or total process memory.

Images are still lazy. Merely declaring an unreachable image does not request it.
Concurrent calls to `asset.readImage(index)` use the shared queue. The texture
preparation stage now performs bounded lookahead instead of requesting each
image only after the previous upload finishes.

## Texture preparation overlaps encoded reads

`createGltfTextureResources` defaults to `maxConcurrentImageRequests: 4`, also
configurable from 1 through 32. The owning scene loader forwards the setting
through `textures`. Only unique image indices in its validated texture requests
are prefetched; unselected scene images remain untouched.

The window counts the current image and both pending and completed lookahead.
A slow first image or decoder therefore cannot cause all later files to load at
once. After the current image is uploaded, the next slot is admitted. With the
standard asset provider, the transport limit and texture lookahead apply together;
a larger window never overrides `assets.maxConcurrentRequests`.

Image decoding, KTX2 transcoding, texture-budget reservations and GPU uploads
remain serial in first-use order. One image still feeds all its material uses;
sRGB and linear uses keep separate mip domains. This change does not introduce
concurrent native decoder scratch allocations, alter pixel conversion, or relax
encoded, decoded-image, transcode or GPU texture budgets.

The optional provider signature is `readImage(index, {signal})`. Existing one-
argument callbacks continue to work. Providers must keep each outstanding image's
encoded bytes stable; use a window of 1 for a provider that is not concurrency-
safe. Provider caches and their retained bytes remain provider-owned. The window
bounds this stage's current/lookahead images, not all cached data in the asset.

A lookahead read failure cancels texture preparation, including a stalled current
decoder or mip-pipeline wait. Caller cancellation and device loss also interrupt
pending image reads, GPU error-scope checks and queue-completion waits. Native
decoder adapters still close eventual results, and earlier owned textures are
released. Cancellation never destroys the borrowed device,
codec, or provider. A provider that ignores the optional signal can finish its
already admitted reads; their late results cannot trigger additional uploads.
`loadGltfAsset.readImage` retains its asset-wide cancellation semantics and ignores
that per-read option: its original construction signal cancels network I/O. Thus
a texture-only failure or device loss does not forcibly abort a borrowed asset's
network requests. No more than the admitted lookahead can remain outstanding.

## Cancellation and failures

A construction-time resource failure cancels active siblings and rejects queued
work without starting it. The original failure remains the rejection reason.
After an asset has been returned, an individual image failure is isolated: other
images can continue. Caller cancellation always rejects active and queued work
and prevents later reads, including cache hits.

The supplied Fetch API receives a per-request AbortSignal, rather than requiring
identity with the caller's signal. Abort does not wait for an uncooperative fetch,
reader, or stream cancellation promise. Eventual unread responses are cancelled;
late promise rejections are observed. This revokes publication, not a guarantee
that arbitrary injected I/O or remote server work can be forcibly interrupted.
The queue keeps no abort listener on the caller's signal while idle.

## Checks

```sh
node --test tools/ingest/gltf_asset.test.mjs \
  tools/ingest/gltf_asset_concurrency.test.mjs \
  tools/ingest/gltf_textures.test.mjs \
  tools/ingest/gltf_texture_prefetch.test.mjs
```

The concurrency suite exercises the production loader and retained codec planning
with native Node Fetch API calls to a loopback HTTP server. It verifies full-body
permit lifetimes, out-of-order completion, identity, cancellation and failure
behavior. Adversarial injected streams cover stalled cleanup, late responses and
byte-budget races. The texture suite connects that native HTTP path to production
texture preparation with recording GPU interfaces and explicit decoder fixtures.
It checks bounded lookahead across slow reads/decoders, serial upload order,
image/color-space sharing, mixed PNG/KTX2 routes, and cancellation of stalled GPU
waits. These checks do not establish browser/GPU execution, actual codec output
pixels, decoded pixel equivalence, or a measured application-level speedup.
