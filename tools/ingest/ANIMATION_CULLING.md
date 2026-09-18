# Conservative animated frustum culling

The existing WebGPU animation scene can omit off-screen **implicit** draws using
current-pose bounds. Enable it explicitly; the default remains unculled rendering
with the existing transparent ordering:

```js
const scene = await createGpuAnimationScene(device, pose, drawables, {
  frustumCulling: true,
  maxBoundsBytes: 16 * 1024 * 1024,
  maxBoundsComponents: 16_777_216,
});
scene.update(deltaSeconds);
scene.render({ colorView, depthView, viewProjection, lighting });
console.log(scene.cullingStats, scene.boundsBytes);
scene.dispose();
```

GPU glTF model factories and the owning URL/GLB loader already forward these
options through their existing `scene` option. For example:

```js
const model = await loadGpuGltfAnimationScene(device, modelURL, {
  scene: { frustumCulling: true },
});
model.update(deltaSeconds);
model.renderCamera({ colorView, depthView }, { cameraNode, aspectRatio });
```

The generated `gpu_playback.mjs` package also includes the bounds dependency and
accepts the same scene options. CPU-only package output is unchanged. The
`cullingStats` and `boundsBytes` diagnostics belong to the direct scene API; model
wrappers do not currently forward those diagnostics.

## Behavior and ownership

All GPU mesh deformers still advance/upload on the original schedule. Culling
removes only draw submissions, not compute work, callbacks, animation steps or
pose publication. An explicit `frame.draws` array bypasses both implicit ordering
and culling, including per-draw transforms, duplicates and empty clears. This
provides the existing escape hatch for application-selected/custom draws.
`sortObjects: false` disables sorting without disabling culling; surviving meshes
then retain source order. Otherwise opaque/masked order and stable transparent
back-to-front order remain intact.

Bounds and geometry must represent the same uploaded pose. Direct pose changes
require `scene.upload()` before rendering, as before. Culling uses a fresh camera
snapshot on every render, so multiple views do not require another pose update.
The renderer receives the same camera matrix snapshot used by the culling test.
A failed camera validation or rejected render does not replace published culling
statistics. Statistics acknowledge a successful submission, not GPU completion.

Each successful direct-scene render publishes `poseVersion`, `testedMeshes`,
`culledMeshes` and `submittedDraws`. Explicit draws report zero automatic tests
and culled meshes. With culling disabled, `cullingStats` is null and no source
geometry is scanned or retained for bounds. Disposal and terminal scene failures
release the summaries without destroying the borrowed pose/device.

## Bounds computation and limits

Construction scans base positions, position morph deltas, and skin indices/weights
once. It retains six extrema per base/target plus the used joint indices, not
source vertex arrays. Updates combine signed morph intervals, then union the
mesh-local joint-transformed intervals and apply actual nonnegative f32 weight-sum
bounds. Update work scales with morph targets and used joints, not vertex count.
There is no CPU vertex skinning, GPU readback, or additional GPU buffer.

The hull is intentionally conservative and may be loose for large skeletons or
sparse/disjoint influences. Arithmetic expansion includes f32 rounding,
contraction/reassociation slack and optional subnormal flushing. Clip tests use
homogeneous WebGPU planes (-w <= x,y <= w, 0 <= z <= w), without inverse matrices
or division. Touching/crossing boxes remain visible; unknown or overflowing bounds
**fail open** and retain the draw. This is not occlusion culling or a tight
per-triangle test, and it does not account for arbitrary custom displacement
shaders. The bounds describe the existing animation deformation position profile.

`maxBoundsBytes` limits retained typed summary bytes across the scene, separately
from the GPU buffer budget. `maxBoundsComponents` limits the aggregate source scan.
They are not a hard JavaScript/driver heap limit. Source geometry and the pose must
remain stable until asynchronous scene initialization resolves, as in the existing
scene contract. Construction failures unwind previously created scene resources.

The lower-level `createAnimationBounds` and `animationBoundsVisible` exports live
in `animation_bounds.mjs`. Bounds snapshots contain immutable `min`, `max`,
`poseVersion` and `bounded` fields. This API owns only its numeric summaries.

## Focused checks

```sh
node --test tools/ingest/animation_bounds.test.mjs \
  tools/ingest/animation_culling.test.mjs \
  tools/ingest/animation_culling.package.test.mjs \
  tools/ingest/animation_draw_order.test.mjs \
  tools/ingest/animation_draw_order.scene.test.mjs
```

The numerical tests compare bounds with independently evaluated f32 morph/skin
vertices and homogeneous clipping. Scene tests execute the actual bounds,
ordering and scene code with supplied packed poses and explicit controller,
GPU-deformer and renderer doubles. Package checks execute the actual emitter and
relocated scene modules, with explicit accessor/runtime/GPU fixtures. These prove
numeric cases, submission selection, package closure and ownership—not native
shader execution, production animation interpolation or full Three.js parity.
The 1,000-mesh fixture submits ten visible meshes and omits 990 off-screen draws;
that is a functional count, not a frame-rate or speedup measurement. Native browser
execution was blocked by the environment's page-access policy during this work.
