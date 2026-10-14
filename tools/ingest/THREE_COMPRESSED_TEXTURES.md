# Compressed source textures on the shared scene path

`createGpuThreeTextures` and the source scene bridge now use the same residency
owner for `CompressedTexture`, `CompressedArrayTexture`, and
`CompressedCubeTexture`. A `CubeTexture` containing compressed faces uses this
path too. Ordinary material maps and `sampler2DArray` programs consume these
bindings; applications do not need to supply a second allocator or borrowed map.

The caller supplies already transcoded blocks and authored mip levels. This
change neither starts a decoder nor fetches URLs, changes source pixels, creates
workers, or adds a submission loop. It adds native upload/sampling support, not
Rust specialization or a measured acceleration claim.

```js
// device must have enabled the feature for the supplied compression family.
const owner = createGpuThreeTextures(device, {three: THREE});
const texture = new THREE.CompressedTexture(
  authoredMips, width, height, THREE.RGBA_BPTC_Format
);
texture.needsUpdate = true;
owner.prepare([texture]);
const {view, sampler} = owner.binding(texture);

// CompressedArrayTexture updates preserve every unselected layer, at every mip.
atlas.addLayerUpdate(2);
atlas.needsUpdate = true;
owner.update([atlas]); // atlas must already have been prepared by this owner.
await owner.whenIdle();
```

## Native formats and capabilities

The admitted format families are RGBA DXT1/3/5 (BC1/2/3), signed and unsigned
RGTC R/RG (BC4/5), signed and unsigned BPTC HDR RGB (BC6H), BPTC RGBA (BC7),
ETC1-compatible ETC2 RGB, ETC2 RGB/RGBA, EAC R11/RG11 signed/unsigned, and all
14 r186 ASTC block footprints from 4x4 through 12x12. RGBA8 authored-mip output
inside a CompressedTexture is also supported, for decoder fallback results.

BC, ETC/EAC and ASTC require their enabled `texture-compression-bc`,
`texture-compression-etc2` and `texture-compression-astc` device features,
respectively. Inspection accepts `features: device.features`; allocation uses
the actual device features automatically. The code never assumes that an adapter
feature was enabled on the supplied device.

NoColorSpace and LinearSRGBColorSpace use the native linear format. SRGBColorSpace
uses the corresponding sRGB sampling view without rewriting blocks. RGTC, BC6H
and EAC have no sRGB variant and reject that request. Base widths/heights must be
block aligned unless the device explicitly enabled
`texture-compression-unaligned`. The texture is never resized to make it legal.

## Authored mips, block rows and array layers

Mip arrays include level zero. Each following extent is floor-halved on both
axes, clamped to one. Cube faces must be square and have matching mip chains.
Array layer count stays constant across levels. Partial authored pyramids are
accepted; mip sampling clamps to their last authored level. A non-mip filter
clamps to level zero. No mip shader is run and no authored level is regenerated.

`bytesPerRow` counts block bytes and `rowsPerImage` counts block rows. Copy extents
round up to whole blocks, including small 1x1 and 2x2 mip tails; the allocated
texture retains its logical extent. Fixed, unshared Uint8Array views must contain
exactly the required blocks. Their byte offsets are preserved. Compression blocks
do not use pixel-row unpack padding. The RGBA8 fallback does honor unpackAlignment
and permits omitted trailing padding after its final row.

Each selected array layer is uploaded at every authored mip. Successful writes
clear `layerUpdates` before `onUpdate`; a failed partial upload leaves the request
unacknowledged and terminates the owner. Already-issued writes are not rolled back.

## Existing ownership and scene behavior

Compressed sources use the same texture/source version acknowledgments, compatible
Source sharing, peak replacement budgets, pruning, disposal and completion waits
as other textures. `maxTextureBytes` counts allocated compressed blocks across all
faces, layers and mips, not decoded RGBA pixels. Driver object overhead is not an
exactly measured part of that count. `maxPixels` still bounds logical base pixels
across all layers. No CPU edit becomes visible without the source upload request.

Loading program textures use dimension-matched defaults until preparation binds
the completed source. Source disposal or a storage/sampler change requires
preparation, using the renderer's existing preparation signal. Borrowed bindings
keep priority. Allocation-only `prepare(..., {initialize: true})` also works for
complete version-zero compressed sources without consuming pending layer updates.

## Boundaries and validation

This native path does not admit PVRTC, compressed 3D textures, arbitrary internal
format overrides, CPU block flipping/premultiplication, or generated compressed
mips. Legacy **RGB** DXT1 is deliberately not mapped to BC1 **RGBA**: their alpha
semantics differ for transparent codewords. RGBA DXT1 is supported. These are
remaining native implementation boundaries, not declarations of permanent product
feature exclusions. Validating every decoder and exact-backend route is separate.

The focused test uses actual pinned r186 texture and scene objects, production
scene traversal, program compilation, residency and render command construction:

```sh
F3D_THREE_ROOT=/path/to/pinned/three node --unhandled-rejections=strict --test \
  tools/ingest/tests/three_texture_compressed.test.mjs
```

The GPU boundary records bytes, views and commands. Coverage includes the admitted
format/color combinations, physical mip tails, array and cube uploads, layer
selection, shared sources, partial pyramids, limits, loading and disposal, native
failures and callback acknowledgment. Four scene cases cover material/program
consumption and loading transitions. The test does not decompress blocks or execute
WGSL. Browser pixel equivalence, end-to-end decoder integration, Rust/workspace
checks and performance measurements are not established by these tests.
