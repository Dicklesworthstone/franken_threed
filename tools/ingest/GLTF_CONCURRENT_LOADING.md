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
The root model file counts toward `maxResources` and the encoded-byte budget.

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
Concurrent calls to `asset.readImage(index)` use the shared queue; this does not
by itself make a caller's serial image-consumption loop concurrent.

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
  tools/ingest/gltf_asset_concurrency.test.mjs
```

The concurrency suite exercises the production loader and retained codec planning
with native Node Fetch API calls to a loopback HTTP server. It verifies full-body
permit lifetimes, out-of-order completion, identity, cancellation and failure
behavior. Adversarial injected streams cover stalled cleanup, late responses and
byte-budget races. These checks do not establish browser/GPU execution, decoded
pixel equivalence, or a measured application-level speedup.
