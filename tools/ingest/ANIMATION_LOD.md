# Distance LOD in animated GPU scenes

`createGpuAnimationScene` now connects the existing `createAnimationLod`
selector to implicit color draws and automatic shadow draws. The same path is
available through `createGpuDecodedAnimationScene`, `createGpuGltfAnimationScene`
and generated `gpu_playback.mjs` packages. This is an explicit animated-scene
API, not a `THREE.LOD` compatibility adapter or an automatic glTF LOD-extension
loader.

## Scene setup

Group levels refer to indices in the supplied `drawables` array, not glTF mesh
IDs, geometry IDs, GPU handles, or the transparency-sorted draw list. A level can
include several primitives. Ungrouped drawables remain selected. Every drawable
can occur in at most one level across all groups.

```js
const scene = await createGpuAnimationScene(device, pose, drawables, {
  lod: {
    maxCameras: 8,
    groups: [{
      node: 0, // Current pose node whose world origin anchors the distance.
      levels: [
        {distance: 0, drawIndices: [0, 1]},
        {distance: 20, hysteresis: 0.15, drawIndices: [2]},
      ],
    }],
  },
  frustumCulling: true,
  shadow: {lightIndex: 0}, // Optional; frame lighting must supply this light.
});

scene.update(deltaSeconds);
scene.render({
  colorView,
  depthView,
  viewProjection,
  lighting,
  lodCamera: {position: cameraWorldPosition, key: 'main', zoom: 1},
});
```

The first distance must be zero and subsequent distances strictly increase.
Hysteresis belongs to each level and is a fraction in `[0, 1)`. Selection uses
world-space distance divided by positive camera zoom. For example, a level at
20 with hysteresis 0.15 stays selected down to distance 17 after that camera has
successfully submitted the coarse level. A different camera key has independent
history. No screen-space error or orthographic-size rule is implied.

Nested LOD configuration is bounded and snapshotted before asynchronous scene
initialization. Mesh count, GPU storage and existing material admission still
apply to the complete resident scene, not just its selected level.

## Ordering, culling and shadows

Implicit color rendering selects LOD membership first, then applies frustum
culling and opaque/masked/transparent ordering. `sortObjects:false` preserves
source order among selected draws. Suppressed levels do not undergo color
bounds tests or transparent-origin sorting.

Automatic shadows use the same LOD membership before shadow fitting and depth
submission. Their subset is NOT the color-frustum result: off-camera selected
casters remain present. Shadow caster exclusions and `blend:'skip'` are still
applied, with the original drawable-index mapping preserved. Fitting includes
selected receivers even when they are not casters.

An explicit `frame.draws` bypasses LOD and culling and preserves caller order.
For automatic-shadow scenes, that frame must also supply an explicit shadow map
or `shadow:null`, as before. It does not update any LOD camera history.

## Authored glTF cameras

Pass `lod` in the existing scene options; no new loader or animation clock is
needed. `model.renderCamera(attachments, {cameraNode, aspectRatio})` derives the
LOD position from the same current-pose camera sample as its view projection.
This includes orthographic cameras, whose lighting block does not contain a
camera position. Authored camera zoom is 1 for this distance policy.

Automatic history keys are `gltf:<cameraNode>`, using the instantiated pose node,
not a camera definition that several nodes may share. A supplied
`attachments.lodCamera` overrides the automatic LOD camera without changing the
render view. External `model.render(frame)` calls must supply their own
`lodCamera` for implicit LOD draws.

## Publication and lifetime

`scene.lodStats` / `model.lodStats` is the immutable selection associated with the
latest successful color submission. It includes `poseVersion`, `cameraKey`,
`cameraPosition`, `zoom`, `groups`, source-order `drawIndices`, `selectedDraws`
and `suppressedDraws`. These are pre-frustum counts; `cullingStats` reports the
subsequent color rejection. A successful explicit draw frame sets `lodStats` to
null without resetting camera histories.

Recoverable selection, sorting, shadow or color errors do not commit hysteresis
or replace the last published selection. An automatic shadow pass can already
have submitted when color validation rejects; GPU work is not rolled back.
Successful submission is not GPU completion: use `whenIdle()` to drain work.

`lodCameraCount` reports occupied history slots. `maxCameras` defaults to 8 and
is bounded by 64. Capacity is explicit, with no hidden eviction. Use
`resetLodCamera(key)` to release an unused slot; it returns the scene/model for
chaining. Resetting the latest reported camera clears its `lodStats`. Scene
terminal failure or disposal releases all history. `lodEnabled` reports whether
LOD was configured. Omitting `lod` leaves the prior implicit path intact.

## Scope and validation

All level geometry remains GPU resident, and every mesh's deformation is still
updated at the existing upload boundary. This reduces selected draw work, not
resident geometry storage or all-level deformation. Picking and pose export are
unchanged; they do not automatically hide inactive levels. This does not provide
progressive asset streaming, mesh simplification, screen-error selection or an
automatic extension/retained-renderer route.

Run:

```sh
node --test tools/ingest/animation_lod_scene.test.mjs tools/ingest/animation_lod_model.test.mjs
```

These regressions execute the real LOD selector, scene, ordering, shadow
orchestration, model wrapper and package builder. GPU resources, bounds/projection
and decoding are controlled doubles. They verify submission inputs, failure
publication, lifetime and relocated-module behavior, not GPU pixels, browser
fidelity, actual glTF decoding or a measured performance gain.
