# Export the current animated pose to GLB

Loaded models can save their current geometry and core materials as a standalone
binary glTF file. Enable this explicitly; default render-only loads retain no
extra CPU geometry or encoded images for export.

```js
import {loadGpuGltfAnimationScene} from './gltf_scene_loader.mjs';

const model = await loadGpuGltfAnimationScene(device, modelURL, {
  exporting: true,
  // Existing assets, decode, textures, scene and picking options are unchanged.
});
model.controller.createAction(0).play();
model.update(deltaSeconds);

const bytes = await model.exportPoseGLB({maxBytes: 64 * 1024 * 1024});
const blob = new Blob([bytes], {type: 'model/gltf-binary'});
// The application owns saving, downloading or otherwise using this Blob.
model.dispose();
```

`exportPoseGLB()` returns a promise for an `ArrayBuffer`. It captures the current
pose, not the original rest mesh. Its result is deliberately a **static posed
mesh scene**: morphing and skinning are baked into vertices, and transforms are
baked into world coordinates. No rig, animation clips or morph targets remain to
apply deformation a second time. This is not an editable-rig exporter or complete
Three.js `GLTFExporter` replacement. Cameras, lights, arbitrary user metadata,
render-only draw/visibility/material overrides and shader displacement are not
exported. The file contains every drawable in the selected loaded model, not just
the last frame's draw list. Original node/mesh/primitive/material IDs are retained
in each output node's `extras.f3dSource`; the source asset copyright is retained.

## Materials and coordinate handling

The writer supports the model path's core metallic-roughness and unlit materials,
alpha modes, double-sidedness, RGB/RGBA vertex colors, and base-color,
metallic-roughness, normal and emissive maps. Each map receives its own UV channel,
with its local transform followed by the shared material transform baked into
that channel. Unsupported material representations such as Lambert are refused,
not silently converted to approximate PBR. Texture bytes are embedded without
re-encoding and sampler settings are retained. Compatible texture uses and shared
encoded image arrays are deduplicated without merging distinct sampler behavior.

Positions are written as little-endian Float32. Normals use the inverse-transpose
world transform; tangents are transformed and reorthogonalized, with unit vectors
required by glTF. Reflections reverse triangle winding and tangent handedness.
Baking avoids writing non-TRS sheared node matrices. Undefined normal/tangent
frames, invalid triangle indices and nonfinite/overflowing values fail rather
than producing a corrupt file. Reflected unindexed geometry gains explicit
indices. This preserves the posed geometry contract, not pixel-identical
interpolation, GPU arithmetic or every external viewer's material behavior.

## CPU models and caller-owned textures

`createCpuGltfAnimationModel(json, buffers, {exporting: true})` reuses its current
CPU deformer outputs. After direct `model.pose.sample()` or `blend()`, call
`model.update()` before export. GPU model factories take the same top-level
`exporting` option; after direct pose sampling, call `model.upload()` first.
Normal `model.update(...)` already performs this synchronization. Disabled
exports throw `ANIMATION_EXPORT_DISABLED`; stale poses throw an explicit error.
`exportingEnabled` reports the construction choice.

CPU models and GPU models created with caller-managed textures need encoded image
resolution when their materials contain maps:

```js
const bytes = await model.exportPoseGLB({
  resolveTexture: async ({view, sampler}, {signal}) => {
    const image = await applicationEncodedImageFor(view, signal);
    return {
      bytes: image.pngOrJpegBytes, // Uint8Array of encoded PNG or JPEG, not pixels.
      mimeType: image.mimeType,
      sampler: applicationGltfSamplerFor(sampler),
    };
  },
});
```

The owning URL/GLB loader supplies this automatically from images already loaded
for texture uploads. It retains those encoded bytes only when export is enabled;
export causes no additional network fetch, native image decode or GPU readback.
A caller may override its resolver. Missing glTF filters are exported with the
effective defaults used by this loader, including configured filter overrides.
The source image bytes are validated for their core MIME signature, not fully
decoded or certified by the exporter.

## Lifetime, snapshot and limits

The complete geometry/material snapshot is captured synchronously before image
resolution awaits. Live playback can then advance, and an already captured export
can finish after model disposal. Export cancellation uses its own `signal`, not
the construction signal. Aborting stops an export waiting on a resolver even if
the resolver ignores that signal; late rejection is observed, but the exporter
cannot forcibly cancel application-owned I/O. Failed exports do not destroy a
usable model. No downloads, file writes or other external mutations are performed.

Use `exporting: {maxComponents, maxBytes, maxVertices, copyright}` to set limits
instead of `true`. Defaults are 16,777,216 retained source components, a 128 MiB
output file, and 4,194,304 vertices. `maxComponents` limits aggregate GPU-model
source snapshots; existing CPU deformer storage is not copied. Private CPU
reference deformers are created lazily on the first GPU-model export and updated
only on later export calls after pose changes. Render/update does no extra CPU
export deformation. A per-call byte/vertex limit can lower, not raise, a configured
construction limit. Final byte accounting includes JSON, image bytes and alignment.
These are separate storage limits, not one hard process/driver memory ceiling.

The standalone writer is `exportAnimationPoseGLB(pose, entries, options)` from
`animation_pose_export.mjs`, where each entry contains `{deformer, drawable,
source}`. Model wrappers manage those inputs and ownership for normal application
use. Static world-space export uses the CPU reference deformation profile rather
than reading back GPU f32 outputs. Picking state, textures, devices and application
frame loops remain independent.

## Focused verification

```sh
node --test tools/ingest/animation_pose_export.test.mjs \
  tools/ingest/animation_model_export.test.mjs
```

Writer tests independently inspect GLB/accessor bytes and reload an embedded-image
export through the production asset loader without network access. Integration
runs the production CPU deformer, writer, material preparation and model wrappers;
fixtures provide explicit packed poses, while accessor/camera/picking, GPU-scene
and native asset/texture boundaries are test doubles. The tests include actual
32-influence CPU skinning and morph arithmetic, but do not establish animation
track interpolation, real GPU submission, native image decoding, external-viewer
pixels, full-workspace CI, editable-rig round trips or acceleration.
