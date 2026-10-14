# Shared array and 3D texture residency

`createGpuThreeTextures` owns source `DataArrayTexture` and `Data3DTexture`
residency alongside ordinary DataTextures, cubes, images and video. No separate
volume allocator, frame loop, source copy or source-image replacement is installed.
These are general WebGPU operations, not Rust specialization or a speedup claim.

```js
const textures = createGpuThreeTextures(device, {three: THREE});
const atlas = new THREE.DataArrayTexture(pixels, width, height, layers);
atlas.needsUpdate = true;
textures.prepare([atlas]);

// After changing just layer 2 in atlas.image.data:
atlas.addLayerUpdate(2);
atlas.needsUpdate = true;
textures.update([atlas]);
await textures.whenIdle();
const binding = textures.binding(atlas); // viewDimension: '2d-array'
```

A Data3DTexture uses the same methods and returns a `3d` view. Its native storage
is genuinely three-dimensional, not an array of disconnected 2D textures.

## Storage, versions and ownership

R, RG and RGBA support UnsignedByteType, HalfFloatType (Uint16 bit patterns) and
FloatType (Float32Array). Uploads copy bytes, including half/float bit patterns,
from the supplied view's byte offset. Fixed, unshared, non-detached storage is
required. Source images, arrays and Source identities remain caller-owned.

Full uploads honor `unpackAlignment` across every row and slice. `rowsPerImage`
is the source height; the last source row need not include trailing padding.
Array layer updates select whole layers, deduplicate through the source Set,
and clear that Set after successful submission, before `onUpdate`. Unselected
layers are preserved; an initial partial upload leaves them at native zero.
The partial-layer profile requires tightly packed rows because r186's WebGL
uploader slices layer data by its tightly packed byte size. Padded full uploads
remain admitted. Data3DTexture updates upload the entire volume.

CPU changes without a new texture/source upload version stay GPU-stale. Shared
Source objects with identical storage/sampler profiles share one residency;
2D, array, cube and 3D domains cannot alias. Changing depth, wrapping or filtering
requires `prepare()` and a replacement charged against the peak residency budget.
Callbacks retain the existing source-before-callback, texture-after-callback
acknowledgment order. Callback exceptions do not become native device failures
and do not replay successfully submitted source uploads.

`maxPixels` counts base voxels or all array layers, not just one slice.
`maxTextureBytes` includes every mip and all resident/replacement storage. Arrays
use `maxTextureDimension2D` and `maxTextureArrayLayers`; volumes use the independent
`maxTextureDimension3D` for all three axes. The metadata inspector exposes the
corresponding `maxDimension`, `maxLayers` and `maxDimension3D` limits. All requested
textures are admitted before any GPU allocation or upload begins.

## GPU mipmaps

With `generateMipmaps: true`, arrays build independent 2D mip chains for every
layer. Array layer count never shrinks. Volumes reduce width, height and depth
at each level, down to 1x1x1. Each volume mip slice is rendered through a 3D view
with an explicit color-attachment `depthSlice`. Its slice index travels through
`firstInstance`, avoiding per-slice uniform buffers or CPU mip construction.

Mip filtering samples destination voxel centers in the preceding mip. This is
bilinear (array) or trilinear (volume) reduction, including floor-halved odd
extents and one-dimensional tails; it is not an area-weighted box filter.
sRGB byte-RGBA uses sRGB sampling and attachment views. Programs/pipelines are
cached by dimensionality and format in the same owner. A complete preceding
volume mip is submitted before any dependent mip.

Linear float32 filtering and generated float32 mipmaps require the negotiated
`float32-filterable` feature. Without it, nearest float32 sampling remains full
precision. There is no automatic half-float conversion or nearest fallback.

## Explicit boundaries

Array/3D `flipY` and `premultiplyAlpha` must be false. Generic 2D `updateRanges`
are not accepted for layered data. Authored uncompressed array/3D mipmaps are
rejected: r186's WebGL array/3D upload branches consume only the base image, so
this owner does not invent a different meaning for their `mipmaps` array.
Compressed, integer and depth storage retain their separate admission boundaries.
No rendered pixel parity with an upstream renderer is established by this work.

This document describes the shared owner API. The program scene bridge's older
private volume/morph allocation path is a separate integration boundary; it is
not automatically replaced merely by adding support to this owner.

## Validation

```sh
node --unhandled-rejections=strict --test \
  tools/ingest/tests/three_texture_volume.test.mjs
```

The tests execute the production inspector, allocator, uploader, mip command
builder, versioning and lifetime code. Source-descriptor classes and the GPU are
explicit boundary doubles. Byte uploads are checked, and shader/pass descriptors
are recorded; WGSL is not executed. Tests cover every admitted component/storage
combination, slice padding, layer selection, NPOT/1D pyramids, independent device
limits, shared-source identity, peak budgets, callback ordering, disposal and
injected failures, plus ordinary 2D/cube regressions. Browser pixel equivalence
and measured performance require separate validation.
