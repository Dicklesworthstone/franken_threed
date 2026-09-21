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
read-only outputs. Every **selected** deformer must match the current pose version
before a query. Excluded draws are not accessed, so they may remain stale until
selected again. Direct pose sampling must be followed by updates of the selected
borrowed deformers. Detached, shared, resizable, nonfinite or stale selected output
storage is refused, never interpreted as an empty scene. Query and disposal
reentry is rejected. Deformer versions must advance whenever their outputs change.

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

A second, scene-level BVH groups selected mesh bounds when there are more than
eight nonempty meshes. It visits nearer scene boxes first, enabling nearest-hit
pruning across meshes, not just within each mesh. Its leaves contain at most four
meshes. Unchanged queries reuse the hierarchy; changed mesh versions refit it;
changed selection membership rebuilds it in the same bounded typed storage.
Reordering the same `drawIndices` does not rebuild or renumber anything. Small
selections use the linear mesh route without allocating a scene hierarchy.
`sceneBvh: false` at raycaster construction retains linear scene traversal for
comparison or to avoid its additional storage reservation.

All selected mesh data is validated/refitted before spatial rejection. Invalid
new geometry cannot be hidden by stale scene bounds. This still entails checking
selected version stamps and preparing all newly selected or changed geometry;
the hierarchy does not make the first full-scene query constant-cost. Extreme
deformation can reduce hierarchy quality, but bounds remain current. This is a
CPU data-structure path, not a measured GPU or application-level speedup.

`maxTriangles` bounds the aggregate topology (default and hard maximum 1,048,576).
`maxBytes` defaults to 128 MiB and bounds the raycaster's owned typed arrays,
including temporary centroid storage required by construction and the maximum
scene-hierarchy reservation when enabled. Borrowed CPU geometry/pose arrays, JS
objects and returned hits are not included; it is not a hard process-memory
ceiling. An all-hit query defaults to `maxHits: 4096` and fails rather than silently
truncating on overflow. `firstHitOnly: true` selects the actual nearest hit, not
the first triangle encountered in storage order. Equal-distance candidates remain
eligible for source draw/face tie-breaking.

`bufferBytes` reports retained typed storage. `lastQuery` reports successful-query
counts and the pose version. `meshesTested` counts mesh-BVH visits; `boxesTested`
counts boxes inside mesh BVHs. `sceneBoxesTested`, `sceneRebuilds`, and `sceneRefits`
report the separate scene work. A rebuild includes a refit. Failed queries leave
prior hit snapshots and statistics intact; a failed refit is retried on the next
valid query.

## Focused checks

```sh
node --test tools/ingest/animation_raycast.test.mjs \
  tools/ingest/animation_raycast_scene.test.mjs
```

Tests cover independently expected rays/triangles, transformations, source IDs,
UVs, near/far limits, reflection/sides, hierarchy reuse/refit/pruning, storage and
lifecycle failures. A deterministic 256-triangle, 100-ray comparison uses an
independent Moller-Trumbore brute-force oracle rather than the production
shear-edge intersection routine. The scene tests compare complete hit snapshots
against linear traversal across 720 seeded queries and changing poses/subsets.
The 1,024-mesh grid and reverse-depth fixtures assert reduced mesh visits, not
wall-time speedups. These tests supply CPU deformation outputs and do not
establish animation-factory integration, GPU execution or pixel parity.

## Pick loaded models

CPU models, decoded GPU models, and the owning URL/GLB loader provide the same
`raycast(ray, queryOptions)` and `pick(ndc, cameraOptions, queryOptions)` methods.
Enable the feature at construction; it is **off by default** so existing
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
scene initialization yields. Queries materialize private CPU deformers **only for
selected draws**, using the existing reference implementation. A cached deformer
updates only when selected at a new pose version; inactive cached deformers remain
untouched. First-time selection after several pose advances uses the current pose,
not an intermediate frame. Once a private deformer owns its source snapshot, the
pick preparation layer releases its duplicate geometry copy. Selecting all draws
still materializes the complete scene; a subset does not remove the initial
all-draw source snapshot or topology reservation.

Normals/tangents, textures and GPU buffers are not copied for selection. Morph
targets that affect only normals preserve their target slot without adding a
position delta. No CPU deformation is added to render/update calls, and no
synchronous or asynchronous GPU readback is performed. `pickingStats` also exposes
`materializedMeshes` (currently retained private CPU deformers) and `updatedMeshes`
(private deformations performed by this successful query, including first-time
creation). Borrowed CPU-model outputs contribute zero to these two counters.
Private caches prepared before a later query failure may be reused; published
statistics still describe the last successful query only.

This deliberately uses the CPU reference deformation profile, including Float32
published positions, **not a promise of bit-identical GPU f32 shader results**.
Application shader displacement, render-only world/side/index-range overrides and
alpha coverage remain outside this geometric selection path. Use explicit query
draw selection where the application renders only some registered meshes.

### Reuse a submitted LOD selection

Model query options accept `lodSelection` as an alternative to `drawIndices`:

```js
model.renderCamera(attachments, cameraOptions);
const selection = model.lodStats;
const hits = model.pick(ndc, cameraOptions, {
  lodSelection: selection,
  firstHitOnly: true,
});
// External cameras can use the same subset with model.raycast(ray, options).
```

This is explicit, not a change to default picking. `lodSelection` must supply the
current `poseVersion` and a bounded unique `drawIndices` array; the indices are
captured before ray getters run. Null/absent submitted selections, stale pose
versions and simultaneous `drawIndices` are rejected. After pose advancement,
render again before reusing that frame's LOD snapshot. A successful explicit draw
frame leaves `model.lodStats` null, so use its explicit source indices instead.

This option does not own or advance LOD hysteresis, choose a camera, or certify
snapshot ownership. The caller must pass a selection from the same model's draw
mapping and the intended camera/view. Retaining same-pose snapshots for multiple
cameras is supported; the latest camera is not silently substituted. These are
pre-frustum LOD members, not raster-visible pixels or custom-draw transforms.
The lower-level `createAnimationRaycaster` continues to accept `drawIndices`,
not model-specific `lodSelection` objects.

A picking limits object can replace `true`:

```js
picking: {
  maxComponents: 16 * 1024 * 1024,
  maxTriangles: 1024 * 1024,
  maxBytes: 128 * 1024 * 1024,
  sceneBvh: true, // default; false retains linear scene traversal
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
  tools/ingest/animation_raycast_scene.test.mjs \
  tools/ingest/animation_model_pick_lazy.test.mjs

# Broader existing model suites (require the complete repository):
node --test tools/ingest/animation_model_pick.test.mjs \
  tools/ingest/animation_model.test.mjs
```

The lazy-picking tests run the real model picker, CPU morph-before-skin deformer
and raycaster with packed pose/camera fixtures and an error-class-only runtime
double. They verify selective preparation, current-pose updates, source snapshots,
LOD snapshot validation and lifecycle behavior. They do not execute the pose
sampler, glTF decoder, GPU or browser.

The broader existing model integration suites exercise production glTF decoding,
pose animation/blending, camera sampling and triangle queries. GPU-scene and
owning-loader transport/texture boundaries are explicit test substitutions.
Listing their commands is not evidence of a fresh run. These focused checks are
not a full-workspace/browser certification.
