# Authored-asset GLB export

The runtime has two different export operations. `exportPoseGLB()` writes a
**static current pose** with world transforms baked into geometry. The new
`exportSourceGLB()` writes the **loaded authored asset** with its scenes, local
node hierarchy, skins, inverse-bind matrices, morph targets, animation clips,
cameras, lights and materials intact. It does not replace animation with the
frame currently being displayed or record later controller/pose edits.

## Owning loader

```js
import {loadGpuGltfAnimationScene} from './gltf_scene_loader.mjs';

const model = await loadGpuGltfAnimationScene(device, 'character.gltf', {
  sourceExport: {maxBytes: 64 * 1024 * 1024},
  exporting: true, // Independently opt in to the existing static-pose exporter.
});

model.update(0.5);
const animatedAsset = await model.exportSourceGLB();
const staticPose = await model.exportPoseGLB();
// Both results are independent ArrayBuffers, owned by this caller.
console.log(model.sourceExportBytes);
model.dispose();
```

`sourceExport` defaults to `false`. It accepts `true` or a limits object containing
`maxBytes`, `maxJsonBytes`, and `maxResources`. It is independent of `exporting`;
source export does not enable CPU reference deformation or posed export.

When enabled, construction closes **every source image**, including images in
inactive scenes and unused texture declarations, before allocating model textures
and the GPU scene. This can request more resources than normal selected-scene
rendering. Requests still use the existing asset loader's fetch, origin policy,
resource/byte limits and construction AbortSignal. Missing images fail the opted-in
construction; they are never omitted from a claimed self-contained asset.

Only the completed GLB snapshot is retained for source export. Subsequent
`exportSourceGLB({signal})` calls need no network request, image encoder, CPU
animation sampling or GPU readback. Each call returns a separate copy, so changing
or transferring one returned buffer cannot damage later exports. The per-call
signal is independent of the completed construction signal. Cancelling an export
does not dispose the model. A returned copy survives subsequent model disposal;
the internal snapshot is released along with the owned model resources.

`sourceExportEnabled` reports configuration. `sourceExportBytes` reports retained
file bytes and returns zero after disposal. These bytes have a separate budget;
they are not included in the scene's GPU `bufferBytes` or texture accounting.
Omitting the feature does not import the source packer, fetch extra images, or
retain an additional file snapshot. The sampling-only `buildAnimation` package
is unchanged; it does not include the owning asset loader.

## Standalone asset export

```js
import {loadGltfAsset} from './gltf_asset.mjs';
import {exportGltfAssetGLB} from './gltf_asset_export.mjs';

const asset = await loadGltfAsset('character.glb', assetOptions);
const glb = await exportGltfAssetGLB(asset, {
  signal: exportAbortController.signal,
  maxBytes: 128 * 1024 * 1024,
  maxJsonBytes: 16 * 1024 * 1024,
  maxResources: 4096,
});
```

This path needs no device, renderer, DOM, Node API or image codec. It accepts the
asset loader's normalized `json`, `buffers` and `readImage(index)` contract.
Use `asset.json`, **not** `asset.sourceJson`: successfully decoded meshopt/Draco
compression has already been projected into ordinary accessor storage.

All supplied buffer bytes and JSON are snapshotted before the first image
callback. URI images are resolved sequentially through `readImage`; identical URI
strings share a result. Embedded images are copied from their existing ranges.
PNG, JPEG and KTX2 bytes are retained without recompression or pixel conversion.
Image signatures and declared MIME types must agree. This is container/resource
packing, not a full image decoder or glTF semantic validator.

The writer merges buffers into one four-byte-aligned BIN chunk and relocates
bufferView offsets without changing accessor data or animation interpolation.
STEP, LINEAR and CUBICSPLINE keyframes, including signed quaternion values and
Hermite tangents, retain their original binary bytes. Core node, accessor,
texture, image and bufferView indices stay fixed during ordinary packing.
Original generator, copyright, names and extras remain data; JSON getters,
`toJSON`, cycles and nonfinite numbers are rejected rather than executed or
silently erased. Negative zero is preserved. When multiple named/annotated
buffers are merged, their metadata and offsets remain in the combined buffer's
`extras.f3dSourceBuffers` array.

## Limits and unsupported extensions

Defaults are 128 MiB for the completed GLB/binary staging, 16 MiB for source and
output JSON, and 4096 entries per buffer/image table. The bufferView table may
contain up to sixteen times `maxResources`, including newly embedded image views.
These limits bound each stage, not aggregate process/driver memory. Encoding,
staging, the retained file and a returned copy can coexist temporarily.

The current relocation profile recognizes `KHR_materials_unlit`,
`KHR_materials_emissive_strength`, `KHR_materials_clearcoat`,
`KHR_texture_transform`, `KHR_texture_basisu`, `KHR_lights_punctual`,
`KHR_mesh_quantization`, and `EXT_mesh_gpu_instancing`. Opaque extensions and
still-compressed bufferView records fail before image I/O: blindly copying an
unknown buffer reference or resource URI would corrupt the result or leave an
external dependency. Application `extras` are metadata, not loader resources.
The normal source route remains necessary for extensions outside this profile.

Tests cover binary preservation and loader ownership with explicit test
boundaries. They do not establish native GPU execution, universal exporter
compatibility, visual equivalence or a performance improvement.
