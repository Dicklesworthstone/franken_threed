# Authored-asset GLB export

The runtime distinguishes a static pose, the unchanged authored asset, and
explicitly supplied animation clips. `exportPoseGLB()` writes a **static current
pose** with world transforms baked into geometry.
`exportSourceGLB()` writes the **loaded authored asset** with its scenes, local
node hierarchy, skins, inverse-bind matrices, morph targets, animation clips,
cameras, lights and materials intact. It does not replace animation with the
frame currently being displayed or record later controller/pose edits.
`exportAnimationGLB(clips)` adds or replaces animation tracks on the authored
asset while preserving the rig and geometry; it does not record the live pose.

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
texture, image and bufferView indices stay fixed during ordinary packing. Codec
normalization may leave orphan declarations referring to skipped buffer slots.
Only those inaccessible orphan accessors/views are pruned, after tracing every
primitive, morph target, skin, animation sampler, image and GPU-instance accessor
reference across all scenes. The surviving accessor/view references are remapped;
node, skin, material, texture and image identities are not changed. A missing
resource that still has a user is an error, never fake zero-initialized geometry.
Backed unused resources are retained. Consumed, now-empty extension declaration
arrays are omitted rather than emitted as invalid glTF arrays.
Original generator, copyright, names and extras remain data; JSON getters,
`toJSON`, cycles and nonfinite numbers are rejected rather than executed or
silently erased. Negative zero is preserved. When multiple named/annotated
buffers are merged, their metadata and offsets remain in the combined buffer's
`extras.f3dSourceBuffers` array.

## Save generated or retargeted motion

`exportGltfAssetGLB(asset, {clips, animationMode})` accepts the `clips` records
used by `f3d-animation-v1` definitions. This connects clip baking to a reusable,
animated GLB, rather than a static-pose export or a separate JSON motion file.
For example, with previously loaded `sourceAsset` and `targetAsset`:

```js
import {decodeGltfAnimation} from './animation_gltf.mjs';
import {retargetAnimationClip, mapAnimationNodeNames} from './animation_retarget_clip.mjs';
import {exportGltfAssetGLB} from './gltf_asset_export.mjs';

const sourceDefinition = decodeGltfAnimation(sourceAsset.json, sourceAsset.buffers);
const targetDefinition = decodeGltfAnimation(targetAsset.json, targetAsset.buffers);
const mapping = mapAnimationNodeNames(sourceDefinition, targetDefinition, [
  {source: 'SourceHip', target: 'Hip'},
  {source: 'SourceArm', target: 'Arm'},
]);
const baked = retargetAnimationClip(sourceDefinition, targetDefinition, {
  mapping,
  rootMotion: {source: mapping[0].source, target: mapping[0].target, scale: 1.2},
  clip: 0,
  frameRate: 30,
  name: 'retargeted-walk',
});
const glb = await exportGltfAssetGLB(targetAsset, {
  clips: [baked.clip],
  animationMode: 'append', // Default: retain authored clips before new clips.
});
```

Bindings and channel `node` values must use the target asset's **original glTF
node order**. Do not supply renderer draw indices, joint-palette indices, or
synthetic nodes created by GPU-instance expansion. Destination meshes, skeletons,
inverse-bind matrices, morph targets, cameras, lights and materials retain their
identities. The same motion can target all mesh instances using that skeleton.

Each clip is `{name?, channels: [...]}`. Channels contain `node`, `path`, `times`,
`values`, and optional `interpolation`. Translation, rotation, scale and morph
weights support LINEAR, STEP and CUBICSPLINE. Times and values can be dense number
arrays or fixed, unshared typed arrays. Cubic records use the existing runtime's
incoming-tangent/value/outgoing-tangent order. Morph outputs are flattened SCALAR
accessors, not vectors; their width must match every primitive of the target mesh.

New data becomes little-endian glTF FLOAT storage. Numeric values round to
Float32; no resampling, time rebasing, quaternion sign flip, normalization or
tangent reconstruction is performed by the writer. Source animation bytes stay
unchanged when appending. Distinct times that collapse to the same Float32 value,
nonfinite/overflowing values, nonunit rotation keys, duplicate targets within a
clip, TRS tracks on matrix nodes, and invalid morph widths are rejected. A
`quantizedRotation` marker does not authorize repairing nonunit source rotations.
For nonrepresentable times, Float32 rounding can shift a STEP transition slightly;
this is separate from the baker's explicitly approximate fixed-rate resampling.

`animationMode: 'replace'` removes **all** authored animation records before
adding the supplied clips. It requires an explicit array; `clips: []` removes all
animations without flattening the rig. No name-based merge is implied, and clip
order and duplicate names are preserved. Existing backed storage is retained even
when a replaced animation stops using it. Only unavailable codec orphans are
pruned. New accessor IDs are allocated after that pruning, so target indices
cannot accidentally refer to discarded data.

All new clip descriptors and numeric data are captured before image callbacks.
The input asset and clip records remain unchanged after success or rejection.
Clip/record data getters and sparse numeric arrays are refused. Limits cover at
most 4,096 new clips, 1,048,576 keys per channel, 16,777,216 new numeric components,
and `maxResources * 8` new channels. New animation and image views share the
existing `maxResources * 16` view budget. The finished binary and JSON budgets
still apply; the writer is not an unrestricted animation recorder.

### Export from an already loaded owning model

Enable `sourceExport` when constructing the model, then supply the same explicit
clip records to the new method:

```js
const glb = await model.exportAnimationGLB([baked.clip], {
  animationMode: 'append',
  signal: exportAbortController.signal,
});
```

This reparses and repacks the private, self-contained source snapshot using the
existing GLB parser and writer. There is no network request, image callback,
transcode, GPU readback or live animation evaluation. A model built without
`sourceExport` rejects this method instead of reconstructing a partial asset.
`exportSourceGLB()` continues to return the original authored snapshot; generated
exports do not replace it, change `sourceExportBytes`, modify live pose arrays, or
install new clips into an already constructed player's immutable clip table.

Per-call `maxBytes`, `maxJsonBytes`, and `maxResources` may tighten construction
limits, never increase them. Configure enough `sourceExport.maxBytes` headroom
for appended motion when loading the model. Parsing can temporarily copy the
retained snapshot within its construction byte ceiling even when a lower output
limit is supplied. The parse copy, output JSON, binary staging and completed file
are separate bounded stages, not one aggregate memory ceiling.

Clip capture and embedded-resource packing finish before the returned promise
yields. Mutating clip arrays immediately after the call cannot change that
export. An already captured result survives model disposal; a new export after
disposal fails. The per-call signal is independent of the completed construction
signal. This synchronous CPU operation does not yield to receive later browser
abort events and does not add a background scheduler.

Tests round-trip generated channels through the production GLB parser, animation
accessor decoder, sampler, local/world transforms and per-mesh skin palettes.
They also exercise the actual retargeter and baker through standalone and owning
export, with separate source/output interpolation choices. Owning-loader tests
use explicit fetch/material/texture/GPU boundaries; geometry codecs are not
invoked by the parser-only path. These tests do not prove native rendering or
full Three.js exporter API equivalence.

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
