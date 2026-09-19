# Instanced glTF models and playback packages

`EXT_mesh_gpu_instancing` TRS assets can use the existing animated-model path.
Both optional and required declarations are accepted. The loader expands each
mesh instance into a bounded ordinary drawable; it does **not** batch these into
hardware-instanced draw calls or claim a performance improvement.

## Load a model

The existing owning loader handles the asset and texture stages:

```js
import {loadGpuGltfAnimationScene} from './gltf_scene_loader.mjs';

const model = await loadGpuGltfAnimationScene(device, 'scene.glb', {
  decode: {maxInstances: 4096, maxPrimitives: 4096},
});

// The existing frame, material, controller and attachment APIs are unchanged.
// Select and play an authored clip when this asset contains one:
model.controller.createAction(0).play();
model.update(0.016);
model.render(frame);
await model.whenIdle();
model.dispose();
```

For already loaded JSON/buffers, use `createGpuGltfAnimationScene` with the same
limits under `decode`, or `createCpuGltfAnimationModel` with the limits directly
in its options. `prepareGltfAnimationModel` expands and snapshots before its
asynchronous texture-preparation boundary; every material is checked before
texture resolution. Repeated instances share texture requests by texture index
and color space, but have independent geometry/output arrays.

## Transforms, animation and identity

Each instance uses `nodeWorld * instanceTranslation * instanceRotation *
instanceScale`. Missing TRS properties use their core defaults. FLOAT attributes,
sparse/interleaved storage and signed normalized BYTE/SHORT rotations use the
existing accessor decoder. Quantized rotations are normalized when converted to
FLOAT node rotations; malformed non-unit rotations are rejected.

Original node indices stay fixed. Synthetic mesh nodes are appended, and the
original node keeps its transform, children, camera and light. Only its mesh is
instanced: neither its attached camera/light nor its children are multiplied.
Parent transform animation stays on that original node. Mesh morph rest weights
and weight-animation channels are propagated to every generated mesh node.

`source[drawIndex].node` is the **pose-node index**, as required by the existing
mesh, query and export contracts. For instanced assets, every model/prepared
result also exposes a frozen lookup:

```js
const poseNode = model.source[drawIndex].node;
const original = model.instanceOrigins?.[poseNode];
// original is {node: originalGltfNodeIndex, instance: zeroBasedInstanceIndex},
// or undefined for an ordinary, non-instanced mesh.
```

Pose arrays and explicit animation masks use the expanded node domain. The
lookup does not change public glTF source JSON or automatically annotate a posed
GLB export with original instance provenance. Export remains the existing static,
expanded-mesh representation, not a reconstruction of the instancing extension.

## Build relocatable playback

`buildAnimation(entry, destination, {maxInstances: 4096})` now reads referenced
instance-TRS buffers as well as animation/inverse-bind buffers. Geometry-only
buffers remain outside this pose builder. Local files, data URIs and GLB BIN
chunks retain the existing root, no-network and byte-budget checks.

Instanced packages export frozen `instanceOrigins` from `animation.mjs`,
`playback.mjs` and, with `webgpu: true`, `gpu_playback.mjs`. Their manifest records
`meshInstanceCount` and `instanceExecution: 'expanded-node-mesh'`; the existing
`instances` field still describes **skin palettes**, not these mesh instances.
Use geometry decoded from the same asset through the model path so appended
pose IDs match. No decoder or normalizer is needed in the emitted runtime.
Packages without instancing retain their prior exports and omit this metadata.

## Bounds and current limitations

`maxInstances` defaults to 4096 and bounds added nodes across the entire asset,
including batches outside the selected render scene. Original plus generated
pose nodes cannot exceed 65536. `maxComponents` separately bounds decoded
instance attributes and expanded transform/morph-track data; the geometry and
renderer stages retain their own aggregate limits. `maxPrimitives` counts all
selected-scene expanded drawables. Exhaustion rejects rather than dropping an
instance, and generated-package byte checks run before creating the destination.

Skinned instancing, custom per-instance shader attributes and unknown instance
extensions still require the retained source route. This implementation is
JavaScript loading/pose execution feeding the existing CPU/GPU deformation and
rendering APIs. Native WebGPU pixels, full Three.js equivalence and acceleration
are separate validation obligations, not established by model or package tests.
