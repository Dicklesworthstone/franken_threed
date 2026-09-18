# Current-pose triangle selection

`animation_raycast.mjs` provides synchronous world-space geometric queries over
existing CPU deformation outputs. It does not advance animation, own a device,
read GPU buffers, install input listeners, or replace Three.js Raycaster.

```js
import {
  createAnimationRaycaster, rayFromAnimationCamera,
} from './animation_raycast.mjs';

const picker = createAnimationRaycaster(pose, [{
  deformer, indices, texCoords, doubleSided: false,
  source: { node: deformer.node, mesh: 0, primitive: 0, material: 0 },
}]);
const ray = rayFromAnimationCamera(view.sample({ aspectRatio: width / height }), [x, y]);
const hits = picker.raycast(ray, { firstHitOnly: true });
picker.dispose(); // never disposes the borrowed pose or deformers
```

`x` and `y` are normalized device coordinates in [-1,1], with Y pointing up.
Perspective rays start at the camera center; orthographic rays start at the
selected point on its camera plane and remain parallel. This helper consumes the
existing camera sample rather than reconstructing a scene or reading a canvas.
Ray `origin` and `direction` can instead be supplied directly. Direction is
normalized; optional `near` (default 0) and `far` (default Infinity) are **world-ray
distances**, inclusive. Camera clipping planes are not implicitly applied.

## Results and selection semantics

The returned array and every hit are immutable snapshots, ordered by distance,
then original draw index, then source face index. A hit includes `drawIndex`,
`node`, `source`, `faceIndex`, `distance`, world `point`, `barycentric`,
`frontFacing`, and world geometric `normal`. The normal faces against the ray;
it is not a vertex-interpolated or normal-mapped shading normal. Optional `uv` is
interpolated from the supplied UV set without applying the material UV transform.
`source` preserves the original glTF node/mesh/primitive/material IDs when supplied.

Triangle indices, UVs, side flags and source IDs are snapshotted at construction.
The deformer's positions and world matrix are borrowed and must be treated as
read-only outputs. Every deformer must match the current pose version before a
query. Direct pose sampling must be followed by deformation updates. Detached,
shared, resizable, nonfinite or stale output storage is refused, never interpreted
as an empty scene. Query and disposal reentry is rejected.

The query honors source front-facing triangles and `doubleSided`, including
reflected world transforms. Nonuniform scale, shear and surviving triangles under
singular transforms are handled in world space; fully degenerate triangles miss.
It tests geometry, **not rendered coverage**: texture alpha tests, transparency,
normal maps, shader displacement, per-draw renderer overrides and application
visibility are not evaluated. `drawIndices` explicitly restricts a query to a
unique list of registered draw indices. Shared edges can return both source faces.
There is no claim of complete Three.js interaction/control compatibility.

## Acceleration and resource bounds

A median-split bounding-volume hierarchy (BVH) with at most eight triangles per
leaf is built on the first query of each selected mesh. Subsequent deformation
versions refit its world-space bounds without rebuilding its topology. Repeated
queries of unchanged outputs reuse both positions and bounds. Nearest-only
traversal visits nearer boxes first and prunes boxes beyond the best known hit.
A hierarchy built before extreme deformation can become less efficient, but is
still refitted rather than using stale bounds. This is a CPU data-structure path,
not a measured GPU or application-level speedup.

`maxTriangles` bounds the aggregate topology (default and hard maximum 1,048,576).
`maxBytes` defaults to 128 MiB and bounds the raycaster's owned typed arrays,
including temporary centroid storage required by construction. Borrowed CPU
geometry/pose arrays, JS objects and returned hits are not included; it is not a
hard process-memory ceiling. An all-hit query defaults to `maxHits: 4096` and
fails rather than silently truncating on overflow. `firstHitOnly: true` selects
the actual nearest hit, not the first triangle encountered in storage order.
`bufferBytes` reports retained typed storage, and `lastQuery` reports successful
query counts and the pose version. Failed queries leave prior hit snapshots and
statistics intact; a failed refit is retried on the next valid query.

## Focused checks

```sh
node --test tools/ingest/animation_raycast.test.mjs
```

Tests cover independently expected rays/triangles, transformations, source IDs,
UVs, near/far limits, reflection/sides, hierarchy reuse/refit/pruning, storage and
lifecycle failures. A deterministic 256-triangle, 100-ray comparison uses an
independent Moller-Trumbore brute-force oracle rather than the production
shear-edge intersection routine. The tests supply CPU deformation outputs; they
do not establish animation-factory integration, GPU execution or pixel parity.

## Pick loaded models

CPU models, decoded GPU models, and the owning URL/GLB loader now provide the
same `raycast(ray, queryOptions)` and `pick(ndc, cameraOptions, queryOptions)`
methods. Enable the feature at construction; it is **off by default** so existing
render-only loads do not acquire extra CPU geometry storage or picking limits.

```js
import { loadGpuGltfAnimationScene } from './gltf_scene_loader.mjs';

const model = await loadGpuGltfAnimationScene(device, modelURL, {
  picking: true,
  // Existing assets/decode/textures/scene options are unchanged.
  scene: { renderer: { format: 'rgba8unorm-srgb' } },
});

// Within the application's existing frame loop, advance animated models as usual.
model.update(deltaSeconds);
model.renderCamera({ colorView, depthView }, {
  cameraNode: model.cameras[0].node,
  aspectRatio: width / height,
});

// Within an application-owned pointer handler, for a viewport filling the canvas:
const rect = canvas.getBoundingClientRect();
const ndc = [
  2 * (event.clientX - rect.left) / rect.width - 1,
  1 - 2 * (event.clientY - rect.top) / rect.height,
];
const hits = model.pick(ndc, {
  cameraNode: model.cameras[0].node,
  aspectRatio: width / height,
}, { firstHitOnly: true });
const selected = hits[0]; // undefined on a miss
// selected.source identifies the original node/mesh/primitive/material.
// selected.point, normal, barycentric and optional uv are independent snapshots.

model.dispose(); // also releases the model's picking state
```

For letterboxed/sub-viewports, calculate NDC against the actual rendered viewport,
not the whole canvas. The library does not register events, capture pointers,
change focus, own selection state, or add an animation loop. A model without an
authored camera can use `raycast` with a ray from its application's external camera.
`pick` consumes the current `view.sample()` camera contract, including its explicit
camera-node selection and viewport-aspect requirements. Camera clip planes are
not converted to ray distance filters; use `raycast` with explicit `near`/`far`
when those are needed. Queries still test geometry rather than alpha-masked pixels.

The CPU factory takes the same opt-in at its top level:

```js
const cpuModel = createCpuGltfAnimationModel(json, buffers, { picking: true });
cpuModel.sample(timeSeconds, { clip: 0 });
const hits = cpuModel.raycast({ origin, direction }, { firstHitOnly: true });
```

`createGpuGltfAnimationScene(device, json, buffers, { picking: true, ... })` and
`createGpuDecodedAnimationScene(device, decoded, { picking: true, ... })` also
support it. `pickingEnabled` reports the construction choice; a disabled query
throws `ANIMATION_PICK_DISABLED`, not a misleading empty hit list. `pickingStats`
reports the last successful query's counts and pose version, or null before any
successful query and after disposal. Query failures do not overwrite those stats.
The existing fluent update/upload/render methods still return the model; queries
return hit arrays, including through the owning loader.

### CPU materialization and ownership

CPU models reuse their existing deformer outputs without evaluating them again.
After direct `cpuModel.pose.sample(...)` or `blend(...)`, call `cpuModel.update()`
before querying. GPU models require `model.upload()` after direct pose sampling;
queries against unuploaded CPU poses are refused, just like camera rendering.
Normal `model.update(...)` already samples and uploads through the existing path.

Enabled GPU models snapshot position-only source geometry, skin attributes, morph
position deltas, selected material UVs, indices and source IDs before asynchronous
scene initialization yields. The first query creates private CPU deformers using
the existing CPU reference implementation; subsequent queries update them only
when the pose version changes. Normals/tangents, textures and GPU buffers are not
copied for selection. Morph targets that affect only normals preserve their target
slot without adding a position delta. No CPU deformation is added to render/update
calls, and no synchronous or asynchronous GPU readback is performed.

This deliberately uses the CPU reference deformation profile, including Float32
published positions, **not a promise of bit-identical GPU f32 shader results**.
Application shader displacement, render-only world/side/index-range overrides and
alpha coverage remain outside this geometric selection path. Use explicit query
draw selection where the application renders only some registered meshes.

A picking limits object can replace `true`:

```js
picking: {
  maxComponents: 16 * 1024 * 1024,
  maxTriangles: 1024 * 1024,
  maxBytes: 128 * 1024 * 1024,
}
```

`maxComponents` bounds the aggregate retained source-copy components of an enabled
GPU model, including position/skin/morph/UV/index snapshots; it is not charged
against already-owned CPU model deformers. Each private CPU deformer retains its
existing separate component/output accounting. `maxTriangles` and `maxBytes`
belong to the BVH/topology layer described above; its allocations are validated
when that layer is created (the first query for GPU models). A query-time budget
failure leaves rendering usable. Source copies, CPU deformation scratch/output,
BVH storage, and JS objects are separate allocations, not one global memory cap.

Disposal releases the picker and its privately owned CPU deformers, never a
borrowed device, texture or application pose. Model-owned poses still follow the
existing model lifecycle. Terminal scene/completion failures and owning-loader
texture loss release picking state too. Recoverable camera/ray/budget errors do
not destroy the model; getter reentry cannot update or dispose it mid-query.

### Integrated verification

```sh
node --test tools/ingest/animation_raycast.test.mjs \
  tools/ingest/animation_model_pick.test.mjs \
  tools/ingest/animation_model.test.mjs
```

The model integration tests execute production glTF accessor/geometry/material
decoding, pose animation/blending, morph-before-skin CPU deformation, camera
sampling, BVH refits and triangle queries. GPU-scene construction/submission is an
explicit test boundary. Four owning-loader seam tests also replace unchanged
asset transport and native texture preparation. They verify forwarding and
ownership, not native HTTP/image/GPU execution or pixel parity. The original model
regression suite runs unchanged, including its 32-influence CPU skinning case.
These focused checks are not a full-workspace/browser certification.
