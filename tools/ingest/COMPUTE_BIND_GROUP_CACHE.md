# Native compute bind-group reuse

The existing `createGpuCompute()` dispatch path caches bind groups across both
synchronous and asynchronously prepared pipelines. Source `renderer.compute()`
uses that owner already; this does not introduce an alternate compute executor,
change the renderer FIFO, or combine/reorder submissions.

## Identity and lifetime

A cache key includes the actual pipeline identity, group index, binding slots,
native buffer identities and offsets/sizes, and native texture-view/sampler
identities. A fresh JavaScript binding descriptor does not require a fresh GPU
bind group. Different buffer contents or upload versions do not change a native
binding: subsequent dispatches observe current GPU bytes, not a cached CPU image.
Reordering descriptor entries with the same binding slots also retains reuse.

Pipeline identities remain separate even when shader binding declarations match.
This preserves default-layout exclusivity. Buffer ranges are snapshotted before
GPU allocation and remain part of the key. Mutating a caller's descriptor cannot
retarget an older group. Omitted size remains distinct from an explicit size.

Texture views and samplers are cached only when their WebIDL brands identify
`GPUTextureView` or `GPUSampler`. `GPUExternalTexture`, direct `GPUTexture`
resources and unknown resource types follow the uncached path. In particular,
external-texture object identity alone is not treated as a lifetime guarantee.
Unknown resource types are still passed to WebGPU; this cache does not invent a
new texture admission profile or replace native validation.

This follows the WebGPU resource-binding and pipeline-layout contracts:
https://www.w3.org/TR/webgpu/#resource-binding
https://www.w3.org/TR/webgpu/#default-pipeline-layout

## Bounds and invalidation

`maxCachedBindGroups` defaults to 256 for the compute owner. Set it to zero to
disable group reuse. Positive capacities retain only the most recently used
groups; a cache hit refreshes recency. Pipeline-local layout handles are cached
separately and remain bounded by the existing pipeline and device group limits.
`cachedBindGroupCount` reports the number of retained group entries, not their
opaque driver memory cost.

The owner invalidates affected groups before releasing or replacing one of its
native buffers. An unchanged allocation survives full and partial host uploads.
Invalidating one buffer does not flush unrelated groups. Eviction drops host
references; it never destroys caller-owned views, samplers or textures. Owner
disposal, abort, device loss and terminal GPU validation failure clear the cache.
No per-dispatch queue fence, CPU readback or duplicate buffer allocation is added.

Complete group/entry shape validation precedes GPU binding allocation, including
sparse arrays, duplicate slots and later malformed groups. Buffer alignment,
shader-layout compatibility and resource-device validation remain WebGPU checks.
Cache hits still execute the owner's GPU error scope and normal dispatch path.

## Verification

Run the focused suite:

```sh
node --test tools/ingest/gpu_compute_binding_cache.test.mjs
```

The suite executes the production compute and graphics-publication owners using
a recording GPU and independent byte arrays. It checks repeated direct/indirect
dispatches, ping-pong histories, full/partial uploads, independent byte ranges,
source graphics borrowing, pipeline separation, view/sampler identity, volatile
resource bypass, LRU eviction, release/resize and terminal/error paths.

The stable-binding case performs 128 submissions with one bind-group creation
and one layout-handle lookup, versus 128 each in the original implementation.
That is a deterministic API-call count, not a hardware throughput or latency
measurement. The byte model is not a WGSL interpreter. These tests do not claim
full renderer integration, actual shader execution, browser pixel parity,
repository-wide validation or a measured GPU speedup.
