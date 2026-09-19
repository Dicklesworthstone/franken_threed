# Shared rigid geometry in animated scenes

Enable `rigidGeometry: true` on the existing GPU scene to upload static mesh
vertices once and share identical GPU data across nodes. Rigid nodes still follow
animated parent/local transforms; their per-frame update only publishes their
world matrix. No deformation compute pipeline, palette/morph buffers, vertex
writes or compute submissions are needed for these nodes.

```js
import {loadGpuGltfAnimationScene} from './gltf_scene_loader.mjs';

const model = await loadGpuGltfAnimationScene(device, 'scene.glb', {
  scene: {rigidGeometry: true, maxMeshes: 1024},
});
// Use the existing controller, update/render and completion APIs.
model.update(deltaSeconds);
model.render(frame);
await model.whenIdle();
model.dispose();
```

For `createGpuGltfAnimationScene`, pass the same setting under `scene`. For
`createGpuAnimationScene(device, pose, drawables, options)`, pass it directly in
`options`. The default is false and retains the prior compute-deformation path.
Scene mesh/draw capacities still apply; this does not silently raise them.

## Execution and compatibility

The pool uses the existing 40-byte position/normal/tangent vertex layout. Sharing
requires identical Float32 words and attribute presence. Hash matches are checked
with a complete comparison, including signed zeros; different source Float64
values can share only when their packed Float32 representations match.

Every mesh keeps its own material, UV/index streams, draw identity, pose version
and world transform. Transparent sorting, explicit draw order/overrides, culling
and automatic shadow registration remain per mesh. Skins, morph bindings and
regenerated flat normals use the existing compute implementation, even in a
mixed scene. Geometry must remain stable during scene initialization, as before.

This is **not hardware-instanced draw batching**. Draw calls and their material
buffers remain separate. It also does not deduplicate source glTF decode arrays
or eliminate CPU pose propagation. Native pixel equivalence and performance
measurements remain separate from buffer-sharing tests.

## Ownership and budgets

`scene.bufferBytes` counts shared GPU vertex storage once. The direct scene also
exposes `rigidGeometryStats` with `meshes`, `uniqueGeometries`, `bufferBytes` and
`computeMeshes`. The higher-level model APIs retain their existing aggregate
`bufferBytes` getter; the statistics object is a direct-scene API.

`maxBytes` remains the aggregate scene GPU-buffer budget. Material allocations
are reserved before admitting geometry, and a duplicate can fit with zero extra
geometry bytes. The deformer `maxBytes` and `maxComponents` settings still bound
admission. The pool additionally retains one CPU comparison copy of each unique
buffer, bounded by its `maxBytes`, and at most one equally bounded incoming
snapshot. Per-mesh transform storage is bounded by `maxMeshes`.

A scene owns its pool. Disposing one mesh reference cannot destroy a sibling's
buffer; the last reference or pool disposal retires it. Device loss and native
allocation/completion errors release owned buffers and remain observable.
Pose, device, textures and attachments remain borrowed.

For direct use, `createGpuRigidGeometryPool(device, pose, options)` provides
`addMesh(geometry)`, returning the existing renderer-compatible mesh interface.
Its `bufferBytes` is zero because the pool owns the allocation;
`sharedBufferBytes` reports referenced storage. Call each mesh's `update()` after
changing the pose. Handle `whenIdle()` covers pool initialization, while the
pool's `whenIdle()` also waits for submitted queue work. The renderer remains
responsible for its draws' completion.

## Relocatable packages

Build with `buildAnimation(entry, destination, {webgpu: true, rigidGeometry: true})`
to include the pool and export `createGpuRigidGeometryPool` and
`canUseRigidAnimationGeometry` from `gpu_playback.mjs`. Enable the scene option
at runtime as well. This is an API option, not a new CLI flag.

The generated pool/scene modules remain local to the package and make no fetches
or GPU allocations on import. Omitting the build option leaves the pool out;
CPU-only output and ordinary GPU file lists do not acquire this dependency.
All emitted bytes are charged before creating the output directory.
