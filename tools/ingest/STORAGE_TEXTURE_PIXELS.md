# Explicit simulation texture pixel transfers

`writeStorageTexturePixelsAsync(texture, data, options)` is an F3D extension on
its WebGPU renderer. It seeds or patches a `StorageTexture`, `StorageArrayTexture`
or `Storage3DTexture` from exact raw bytes, without first running a seeding kernel.
The existing renderer-owned allocation is shared with compute, native texture
copies and compatible material bindings. It does not install CPU images on the
source texture, change source versions, or replace `Texture.needsUpdate` semantics.

```js
await renderer.init();
const state = new THREE.StorageTexture(width, height);
state.type = THREE.UnsignedIntType;
state.format = THREE.RedIntegerFormat;
state.generateMipmaps = false;
state.minFilter = state.magFilter = THREE.NearestFilter;
const seed = new Uint32Array(width * height);
seed.fill(0xffffffff);
await renderer.writeStorageTexturePixelsAsync(state, seed);
// Kernels may now read/update state; no integer-to-float conversion occurred.
```

Supported storage formats are the owner's RGBA8, RGBA16F and R/RG/RGBA32 float,
signed integer and unsigned integer formats. Inputs are fixed, unshared
ArrayBuffers or views, including DataView. Bytes are copied, not numerically
converted: half-floats must already be encoded as half-float bits. Format and
channel count must match the destination. This is not a DataTexture-to-Texture
copy API, image decoding path, color conversion, or compressed/depth upload path.

Options `x`, `y`, `z`, `width`, `height`, `depth`, and `mipLevel` select positive
integer texels of one destination mip. Defaults cover the remaining extent from
the origin. Array layers retain their depth at every mip; volume depth shrinks.
`offset` is bytes relative to the supplied view. `bytesPerRow` and `rowsPerImage`
allow padded source rows and images; defaults are tightly packed. Only logical
pixel bytes enter the queued snapshot. `flipY` reverses rows inside each copied
slice, not slice order. Origins use native WebGPU top-left coordinates.

Uploads capture bytes, regions and mip-update policy synchronously at the call,
even while earlier rendering is preparing. They occupy the renderer FIFO, so
write/copy/compute ordering is preserved without waiting for GPU completion
between calls. The returned Promise includes upload validation, automatic mip
work, and submitted-work completion. Queued snapshots are bounded (128 MiB and
64 incomplete transfers by default); source data may be changed or detached after
the call. Source disposal, resize, device loss and renderer cancellation reject
stale requests. Snapshot memory is released on rejection or submission.

Automatic float mip updates reuse the storage owner's existing GPU path.
Integer base uploads with automatic mip interpolation enabled are rejected;
disable generation or automatic updates and explicitly author integer mips.

## Read back a simulation snapshot

`readStorageTexturePixelsAsync(texture, options)` is the companion F3D extension.
It submits a native texture-to-buffer copy at that call's position in the same
FIFO. Later compute, copies or uploads may change the texture while mapping is
pending; the result still belongs to the earlier copy point. This explicit API
introduces a CPU readback only when called; ordinary compute/copies stay on GPU.

```js
const before = renderer.readStorageTexturePixelsAsync(state);
renderer.compute(updateState); // Copy already submitted or queued ahead of this.
const after = renderer.readStorageTexturePixelsAsync(state);
const [beforeBytes, afterBytes] = await Promise.all([before, after]);
const beforeIDs = new Uint32Array(beforeBytes.buffer);
const afterIDs = new Uint32Array(afterBytes.buffer);
```

Results are independent, tightly packed `Uint8Array` bytes with the texture's
native component representation. `x/y/z`, `width/height/depth`, `mipLevel`, and
`flipY` select/reorient the result. Upload-only `offset`, `bytesPerRow`, and
`rowsPerImage` options are rejected, not silently ignored. Mapped 256-byte row
padding and image strides are removed before returning the result.

Each pending read reserves its full padded native staging allocation AND packed
CPU output against a 128 MiB aggregate budget. The 64-operation incomplete-transfer
bound is shared with uploads. Staging is destroyed on success, failure, renderer
cancellation or device loss, including during an unfinished map. Resize/disposal
before copy submission invalidates the captured source generation; disposal after
submission does not revoke the independent buffer snapshot. `waitForGPU()` also
waits for pending pixel transfers and reports asynchronous transfer failures.

```sh
node --test tests/unit/three_storage_pixel_upload.test.mjs \
  tests/unit/three_storage_pixel_readback.test.mjs
```

`tests/e2e/webgpu_renderer/native_storage_pixels.html` exercises upload, native
copy, ordered readback and overwrite for all 11 formats in 2D/array/3D on a real
WebGPU device. It exposes `window.__f3dStoragePixels` and uses minimal source
texture objects, not the retained Three.js shader builder. Run it in a served
checkout with WebGPU enabled. It is an execution fixture, not evidence of a pass
unless actually run, and is not an upstream-renderer parity test.

The focused tests use the real storage owner and a byte-accurate GPU transfer
double. They verify source capture, native allocation identity, region layouts,
ordering, budgets and cancellation, not WGSL execution or hardware performance.
