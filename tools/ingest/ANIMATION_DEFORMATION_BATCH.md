# Batched, input-sensitive GPU deformation

Animated scenes can submit changed compute meshes together instead of creating
one command encoder, error-scope pair, submission and acknowledgement per mesh.
This is an opt-in change to the existing WebGPU scene upload boundary, not a new
renderer, scheduler, animation sampler or Three.js compatibility route.

```js
const scene = await createGpuAnimationScene(device, pose, drawables, {
  deformationBatch: true,
  // Existing renderer, shadow, LOD, culling and rigidGeometry options still apply.
  renderer: { format: 'rgba8unorm-srgb' },
});

scene.update(deltaSeconds);
scene.render({ colorView, depthView, viewProjection, lighting });
const work = scene.deformationStats;
await scene.whenIdle();
```

For `createGpuGltfAnimationScene`, pass the flag under `scene`:

```js
const model = await createGpuGltfAnimationScene(device, json, buffers, {
  scene: { deformationBatch: true },
});
```

The decoded-model factory accepts the flag directly in its options. Those
factories already forward scene options; the new work counters described here
are exposed by the **raw scene**, not newly forwarded by the model/URL wrappers.
The flag defaults to **false**, preserving the existing individual-submit path.
Initial construction still uploads and evaluates each mesh individually.

## What work is removed

Every upload validates and snapshots each compute mesh's current world transform,
instance-local skin palette and morph weights. The palette and weights are
converted to the existing f32 execution profile and compared word-for-word with
the **last successfully submitted** values. Signed zero changes are preserved;
differences lost in the original f64-to-f32 conversion do not force a dispatch.

A mesh with unchanged shader inputs keeps its existing vertex-buffer contents.
Its world matrix, `poseVersion` and `version` still advance, so a moving parent,
camera-only pose change or paused deformation cannot leave render consumers with
stale transforms. If no shader inputs changed, the upload creates no encoder,
queue writes, compute submission or new queue acknowledgement. Outstanding errors
and completion promises from earlier work remain observable.

All changed compute meshes use one encoder and one queue submission. Each keeps
its own palette, morph-weight and output buffers: batching never overwrites a
shared parameter buffer for several dispatches. Each deformation has its own
compute pass; dynamic flat normals retain a separate ordered pass immediately
after their mesh's deformation. The WGSL shaders and vertex-buffer ABI are
unchanged. There is no CPU vertex evaluation on update and no GPU readback.

The input cache assumes **only the deformer writes its vertex buffer**. Consumers
may draw or copy from it, but external storage writes invalidate the cache's
premise. Use the individual update path when deliberately replacing GPU output.

## Render, LOD, culling and shadow behavior

Deformation still finishes submission at `update()` / `upload()`, never during
`render()`. Direct pose sampling still requires `upload()` before rendering.
Every resident mesh acknowledges the same current pose, including inactive LOD
levels and off-camera shadow casters. Bounds update only after deformation and
world snapshots are current. Two cameras can select different levels from one
uploaded pose without triggering more compute work.

This does **not** skip an animated mesh merely because its LOD level or color
frustum is inactive. Such a mesh is still dispatched if its shader inputs change;
that preserves alternate-camera and shadow use without guessing visibility.
All mesh palettes/morph weights are still inspected on upload. Geometry remains
resident. This is not streaming, visibility-driven deferred deformation,
GPU-driven culling or a measured application-level speedup.

`rigidGeometry:true` remains complementary. Shared rigid handles publish their
transforms separately and never enter the compute batch. A scene containing only
rigid handles needs no compute batch submission.

## Counters and memory

`scene.deformationBatchEnabled` reports the opt-in. `deformationStats` is null
before the first successful batched upload, in individual mode, and after
terminal release/disposal. A successful batch publishes an immutable snapshot:

| Field | Meaning |
| --- | --- |
| `poseVersion` | Current pose acknowledged by the complete scene upload. |
| `meshes` | Compute meshes checked, excluding pooled rigid handles. |
| `dispatchedMeshes` / `skippedMeshes` | Compute meshes with changed / unchanged submitted shader inputs. |
| `dispatches` | Compute dispatches, including separate flat-normal passes. |
| `submissions` | Zero or one compute queue submission for this upload. |
| `bufferWrites` / `uploadedBytes` | Palette and morph-weight queue writes for dispatched meshes. |
| `rigidMeshes` | Rigid handles updated outside the compute cohort. |

A changed mesh currently uploads both its palette and morph inputs when present;
this is mesh-level pruning, not partial-range upload coalescing. Counters exclude
initialization, color/shadow draws, and completion drains. A recoverable color
frame rejection does not erase or repeat an already successful upload.

The unchanged GPU byte budget still counts the same buffers. Per-deformer
`inputCacheBytes` reports additional CPU submitted-input shadows: four bytes per
skin-palette component and morph target. Unskinned, unmorphed geometry needs no
input shadow. Cache sizes are derived from already bounded GPU input buffers,
not unbounded caller lengths. `scene.deformationInputCacheBytes` sums these CPU
shadows separately from `bufferBytes`. They are released on terminal GPU failure
or disposal. Existing staging arrays and other pose/geometry allocations remain
separate; this is not a whole-process memory ceiling.

## Direct multi-model use

The lower-level API also batches distinct deformers belonging to independent
poses, provided they use the same device:

```js
import { updateGpuAnimationDeformers } from './animation_webgpu.mjs';

const work = updateGpuAnimationDeformers(deformers, { skipUnchanged: true });
await Promise.all(deformers.map(deformer => deformer.whenIdle()));
```

`skipUnchanged` defaults to false. A standalone `deformer.update()` still forces
its existing single-mesh submission. The batch accepts at most 4,096 distinct
owned deformers, with no duplicates, foreign handles or cross-device members.
An empty batch is a no-op. Submit any draws/copies consuming a pose before the
next update that overwrites its buffers; merely recording a draw is not enough.

All member inputs are prepared before GPU encoding or queue writes. Lower-level
preflight failure publishes nothing and is retryable. Encoding/queue failures
make the attempted cohort terminal; already enqueued work is not rolled back.
Async scope/completion errors propagate through every affected owner's
`whenIdle()`, and an earlier unresolved error cannot be hidden by a later batch
or unchanged-input skip. The scene retains its stricter existing rule: an upload
failure terminates and releases the whole scene, not a partially drawable pose.

## Validation

```sh
node --test tools/ingest/animation_webgpu_batch.test.mjs \
  tools/ingest/animation_scene_batch.test.mjs
```

The focused suite runs real GPU-deformer, CPU-admission and scene-orchestration
code. Packed poses, error-class-only runtime replacements and recording WebGPU
calls are explicit boundaries; scene controller/render/bounds/LOD/shadow/rigid
seams are controlled doubles. The tests cover exact submitted input words,
changed subsets, signed zero, retry/cache publication, independent poses,
flat-normal pass ordering, culling/shadow version inputs, scene lifetime and
cumulative asynchronous errors. They do not execute WGSL, decode actual glTF,
render pixels, certify native browsers or establish wall-time speedups.
