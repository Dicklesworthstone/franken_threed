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

```sh
node --test tests/unit/three_storage_pixel_upload.test.mjs
```

The focused tests use the real storage owner and a byte-accurate GPU transfer
double. They verify source capture, native allocation identity, region layouts,
ordering, budgets and cancellation, not WGSL execution or hardware performance.
