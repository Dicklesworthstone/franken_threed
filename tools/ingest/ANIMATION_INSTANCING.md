# Instanced draw submission for animated scenes

The existing material renderer accepts `instancing: true` to combine consecutive
compatible opaque/masked meshes into WebGPU `draw` / `drawIndexed` calls with
an instance count. It retains each logical mesh's world/normal matrices, color,
alpha cutoff, material factors and shared UV transform in its own draw packet.
This is native instanced command generation, not merely shared vertex storage.
Native shader/pixel validation and performance measurements are separate claims.

## Enable the complete rigid-mesh path

```js
import {loadGpuGltfAnimationScene} from './gltf_scene_loader.mjs';

const model = await loadGpuGltfAnimationScene(device, 'scene.glb', {
  scene: {
    rigidGeometry: true,
    renderer: {instancing: true},
    maxMeshes: 1024,
  },
});

// Continue using the application's existing frame loop and attachments.
model.update(deltaSeconds);
model.render(frame);
await model.whenIdle();
model.dispose();
```

The same nested `scene` settings work with `createGpuGltfAnimationScene`. Pass
`{rigidGeometry: true, renderer: {instancing: true}}` directly as the options of
`createGpuAnimationScene`. Both switches default to false. `rigidGeometry` shares
immutable vertex data and avoids rigid deformation compute; `instancing` shares
compatible material streams and combines their draw commands. Each remains a
separate selection: deforming meshes can use the instancing renderer, but distinct
vertex buffers cannot be combined into the same draw.

For standalone renderer use:

```js
const renderer = await createGpuAnimationRenderer(device, {
  instancing: true,
  maxDraws: 1024,
});
const a = await renderer.addMesh(firstGpuMesh, materialA);
const b = await renderer.addMesh(secondGpuMesh, materialB);
renderer.render({...frame, draws: [a, b]});
console.log(renderer.drawCount, renderer.drawCallCount);
await renderer.whenIdle();
renderer.dispose();
```

`drawCount` and `maxDraws` count **logical mesh instances**, not batches.
`drawCallCount` reports the number of native calls in the last successful render
submission; it is a renderer getter, not a new high-level model getter. Zero-count
logical draws retain a command, as on the ordinary path. Submission is not GPU
completion: use `whenIdle()` to observe cumulative validation/device errors.

## What may share a draw

Only adjacent draws with the same native vertex buffer, index buffer/format,
UV/color buffer, texture bindings, pipeline, and first/count range are combined.
Pipeline matching includes winding, culling, blend and attachment configuration.
Different per-instance transforms, colors, normal scale, occlusion strength,
metallic/roughness/emissive factors and shared UV transforms remain independent.
The existing five-map PBR, Lambert, unlit, depth-only, normal-frame, projected
shadow receiver and environment receiver shader variants use instance packets.

No sorting is added. A sequence A/B/A is not regrouped into A/A/B. `BLEND` draws
remain separate even with identical resources, preserving existing transparent
ordering. Explicit draw lists and world overrides keep their original order and
packet indices. Scene culling/sorting still happens before renderer submission;
this feature does not change which meshes the scene chooses to draw. Automatic
shadow casters retain their existing separate depth-renderer configuration; this
switch does not automatically batch that other renderer.

Separately registered meshes can share immutable index and packed UV/color streams
when all bytes match. Hashes narrow the search; a complete comparison establishes
equality, including signed zero. Matching borrowed texture-view/sampler identities
share a bind group. Different sampler/view objects are not assumed equivalent,
even when their descriptions appear similar. Source arrays and descriptors are
snapshotted before asynchronous registration and are never used as mutable cache
keys. Each mesh still has its own identity and disposal operation.

## Storage, ownership and limits

The opt-in renderer replaces its dynamic uniform binding with a read-only storage
arena. Each instance uses the same 256-byte draw packet, padded to the existing
alignment when necessary. The vertex shader reads `instance_index`; the fragment
shader receives a flat packet index. No extra vertex stream or per-vertex copy is
introduced. A completed host preflight precedes writes/submission, and successive
`render()` calls consume their snapshots immediately rather than deferring them.

The device must expose sufficient `maxStorageBuffersPerShaderStage`,
`maxStorageBufferBindingSize` for the entire draw arena, and at least 11
`maxInterStageShaderVariables`. Insufficient limits reject before allocation;
there is no silent capability downgrade. The disabled path has no new storage
requirements and retains its original shader/binding/upload representation.

`allocatedBytes` counts shared GPU streams once. The renderer retains one CPU
comparison copy per unique index/surface buffer, bounded by its GPU byte budget,
plus bounded incoming packed data during registration. Cached GPU streams are
reference-counted: the last mesh release destroys a stream, not the first one.
The renderer never destroys borrowed geometry buffers, textures, samplers or the
device. Renderer disposal/device loss retires its owned resources.

Only completed, validated registrations publish cache entries. Concurrent new
registrations may allocate private duplicates until validation completes; equal
successful entries are then merged and the redundant allocations retired. A
failed registration cannot lend an invalid buffer to another mesh. Consequently,
concurrent admission may require transient GPU space beyond the final shared
size. Register sequentially for the tightest allocation budget. Scene-level
material reservations remain conservative and can reject earlier than direct
renderer admission at an exact shared-buffer limit.

## Generated playback packages

Use `buildAnimation(entry, destination, {webgpu: true, rigidGeometry: true})` to
include the existing rigid pool, then enable `renderer: {instancing: true}` when
creating the emitted scene. Instancing is implemented inside the emitted material
renderer, so it needs no additional build flag or runtime dependency. Building or
importing a package does not enable it implicitly.

This path does not eliminate per-instance CPU pose/normal-matrix packing, source
geometry decoding, or draw-list traversal. It does not infer equivalence of
separate mutable deformation buffers or implement arbitrary Three.js instancing
contracts. Reduced allocation/command counts are directly testable; image parity,
whole-application correctness and throughput require their own native validation.
