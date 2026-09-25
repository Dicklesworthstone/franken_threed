# Live Three.js skin and morph deformation

The explicit r186 source-scene bridge accepts native `SkinnedMesh` and ordinary
`Mesh` with position/normal morph targets. It captures current source state and
feeds the existing registered WebGPU compute deformer. Rigid and instanced rigid
meshes keep their existing paths. No second animation clock, source geometry
rewrite, cloned scene, WebGL fallback, or new frame loop is introduced.

```js
import {createGpuThreeHdrCanvas} from './three_canvas.mjs';

const renderer = await createGpuThreeHdrCanvas(canvas, scene, {
  three: THREE,
  scene: {
    maxDeformedMeshes: 64,
    maxDeformationBytes: 128 * 1024 * 1024,
    deformation: {maxJoints: 256, maxMorphTargets: 64},
  },
  output: {toneMapping: 'aces-filmic', exposure: 1},
});

// Inside the application's existing frame callback:
mixer.update(deltaSeconds); // The caller still owns/advances AnimationMixer.
renderer.render(camera);    // Captures current bones/morph weights, then draws.

// After changing static geometry, skin attributes, or morph target data:
geometry.attributes.position.needsUpdate = true;
await renderer.prepare();

// At the end of the application's lifetime:
renderer.dispose(); // Source scene, skeleton, geometry and mixer remain owned by you.
```

`createGpuThreeCanvas` and `createGpuThreeScene` use the same deformation path.
The HDR factory still performs opaque whole-image tone mapping, not per-material
Three.js `toneMapped` parity. All existing material, texture, shadow, source-hook,
and scene admission restrictions remain in force.

## Source and GPU contracts

Position/normal morph targets may be relative deltas or absolute target values.
Absolute targets are converted to deltas at preparation. Live negative and
extrapolating weights remain valid within the finite f32 arithmetic profile.
Skinning follows morphing. Each joint's mesh-local palette matrix is
`bindMatrixInverse * bone.matrixWorld * boneInverse * bindMatrix`; this preserves
attached/detached source bind spaces without applying the mesh world transform
twice. Base normals and tangent directions use the same skin matrix with w=0;
tangent handedness is retained. The renderer performs the normal/world transforms.

Four source skin influences per vertex are supported. Weights must already be
nonnegative and sum to one within the existing core tolerance. Normalized native
integer attributes are decoded through their standard component accessors.
The bridge does not normalize weights silently. Position, normal, tangent,
index, UV0 and RGB/RGBA surface attributes are captured without rewriting the
source; interleaved native attributes are also supported. Color morphs, tangent
morphs, instanced skin/morph meshes, batched meshes, custom attribute readers and
custom upload callbacks remain explicit refusals.

Static snapshots track attribute/data/array identities and upload versions,
morph layout, skeleton/bone identities and inverse-bind identities. Edits marked
with `needsUpdate`, replacement geometry, or structural skeleton changes require
`prepare()` before drawing again. Bone matrices, bind matrix values and morph
influences are captured on each visible rendered frame without re-registration.
As with the source attribute API, in-place static edits without `needsUpdate`
are not detected by rescanning every vertex. The source scene updates world
matrices under its existing automatic-update rules; externally owned bones or
manually updated scenes must already have current matrices. Culling and sorting
use the source bounds; callers remain responsible for refreshing dynamic bounds.

Each animated source mesh has an independent GPU owner even when source geometry
is shared. Material groups, draw ranges and the two passes for double-sided
transparent materials consume that same owner's output, rather than deforming
once per material. All visible animated owners are captured before one core
batch, and that batch is submitted before its consuming scene draws. An invalid
later pose prevents the entire deformation batch from dispatching. Source errors
are retryable; native failures are terminal. Source state capture does not run a
CPU per-vertex deformation loop. The existing core factory does run its CPU
reference once during initial admission; no measured speedup is asserted here.

## Ownership, limits and explicit use

Scene defaults are 256 deformed meshes and a 128 MiB GPU deformation budget.
Per-mesh defaults are 1,048,576 vertices, 1,024 joints, 256 morph targets and
16,777,216 decoded components. `maxDeformationBytes` charges current plus pending
replacement deformation GPU buffers; renderer index/surface buffers and source
CPU snapshots are separate. Replacements are published only after registration
and source revalidation. Old consumers are drained before their owners retire.
An over-budget or invalid replacement leaves the previous owner retained.

Standalone users can import `createGpuThreeDeformation(device, mesh, {three})`.
Its `.deformer` is the actual core GPU owner accepted by
`createGpuAnimationRenderer.addMesh`; `.surface` carries captured indices, UVs and
RGBA colors. Call `.update()` after updating source world matrices, or use
`updateGpuThreeDeformations(owners)` for a single same-device batch. Submit all
consumers before the next update. `.whenIdle()` drains, and `.dispose()` releases
only owned deformation resources. The lower-level `createThreeDeformationBinding`
exposes the packed pose/geometry contract without creating GPU resources.

Canvas factories propagate their owned lifetime signal into source preparation.
For canvas use, provide a top-level `signal`, not an independent `scene.signal`.
The standalone scene and deformation factories also accept lifetime signals.
Abort/disposal ends owner waits; a late core owner is disposed on arrival. This
is not cancellation of already-issued native GPU work or an ability to interrupt
an underlying driver compilation synchronously.

`buildAnimation(..., {webgpu: true, threeScene: true})` includes both deformation
modules and their public factories in `gpu_playback.mjs`. Ordinary GPU and CPU
packages do not acquire this optional source bridge.

## Validation boundary

The host tests execute real binding, ownership, source traversal, preparation and
canvas wiring code with recorded source/native boundaries. They are not the
retained Three.js implementation and do not execute WGSL. The separate
`tests/e2e/three_deformation/index.html` probe uses the pinned retained module and
native GPU vertex readback; a missing adapter is blocked, never passed. Its
result is `window.__f3dThreeDeformationResult`. Browser execution, full renderer
compatibility, and acceleration require separate evidence.
