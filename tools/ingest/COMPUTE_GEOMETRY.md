# Compute-authored vertex and instance streams

The new WebGPU renderer can bind a compute-produced `StorageBufferAttribute`
straight into ordinary `BufferGeometry` vertex inputs. Its existing geometry
path lets `InstancedMesh` use `StorageInstancedBufferAttribute` for
`instanceMatrix` and `instanceColor`.
Compute and drawing share the same GPU allocation; there is no intermediate
copy, readback, CPU shadow, or geometry-side re-upload of these streams.
This is general WebGPU execution with retained source TSL preparation, not a
Rust specialization or a measured speedup.

```js
const position = new THREE.StorageBufferAttribute(initialPositions, 3);
const positions = storage(position, 'vec3', position.count);
const delta = uniform(0.01);
const update = Fn(() => {
  positions.element(instanceIndex).x.addAssign(delta);
})().compute(position.count);

await renderer.init();
await renderer.compileComputeAsync(update); // Publishes storage; no dispatch.
geometry.setAttribute('position', position);
mesh.frustumCulled = false;                  // Or provide conservative bounds.
await renderer.compileAsync(scene, camera);

renderer.compute(update);
renderer.render(scene, camera);             // Same queue and same GPU buffer.
```

Supported ordinary streams are Float32 position, normal, tangent, UV channels,
and RGB/RGBA color; existing CPU-authored indices and other CPU streams can
coexist with them. Padded `vec3` position/normal arrays use a 16-byte stride but
feed three components to the vertex shader. Instance matrices expose four
columns at a 64-byte stride; padded instance RGB uses xyz at a 16-byte stride.
Program-reflected Float32 attributes also borrow their storage at the declared
shader locations, with instance divisor one.

## Important boundaries

Compute or `compileComputeAsync` must upload an attribute on the drawing device
before its first geometry use. After setting a shared attribute's `needsUpdate`,
submit it through compute/precompilation before drawing. Geometry never consumes
that pending upload itself. Unversioned CPU edits do not change resident values.
GPU simulation does not update CPU arrays, bounds, raycasting data or export
results. Supply conservative bounds or disable source frustum culling.

Compute owns and charges the buffer. Geometry disposal only releases its borrow;
compute disposal/device loss revokes every associated borrow. Buffer replacement
requires geometry update before reuse and advances its native-buffer generation.
A source attribute cannot have two different compute owners on one borrowed
device: that ambiguous case fails explicitly instead of mixing renderer state.

Compute-authored indices, integer vertex fetch and repeated instance divisors
must not be inferred from Float32 vertex-buffer support. Shader storage reads
through authored node-material slots use the separate profile below.

## Native indirect geometry

The native renderer executes `BufferGeometry.setIndirect(attribute, offset)`
through `drawIndirect` or `drawIndexedIndirect`. `offset` is a byte offset or
an array of byte offsets, as in pinned r186. Ordinary built-in materials select
this native path automatically; an authored node slot is unnecessary.

```js
const args = new THREE.IndirectStorageBufferAttribute(
  new Uint32Array([3, 1, 0, 0, 0]), 1); // indexed count, instances, first, base vertex, first instance
geometry.setIndirect(args);
const words = TSL.storage(args, 'uint', args.count);
const update = TSL.Fn(() => {
  words.element(0).assign(6);
})().compute(1);
renderer.compute(update);
renderer.render(scene, camera);
```

The command borrows the compute owner's GPU allocation. It introduces no CPU
readback or copy of GPU-produced arguments. Ordinary unnormalized Uint32/Int32
`BufferAttribute` arguments use the same versioned upload owner. Storage-backed
three-component attributes retain the source's padded layout; ordinary buffer
attributes retain their packed words. Indexed commands need 20 in-bounds bytes;
nonindexed commands need 16. Wireframe uses its generated index buffer and the
indexed layout. Offsets and upload histories are captured independently for
queued renders, including partial source uploads.

Source geometry upload precedes ordinary node updates; offsets are selected
after those updates. Ordinary materials retain the pinned observer behavior:
an indirect-only `needsUpdate` does not itself force a full render-object
refresh. A preceding compute upload remains visible even when the material or
source bundle reuses its bindings. Bundle offsets and buffer identity remain
recorded until bundle invalidation, while GPU contents stay live.

Retained shader-storage bindings may alias the same argument allocation. Using
that allocation as a generated vertex input as well still has an explicit
preparation gap after a compute upload changes its host version; the retained
vertex proof rejects the stale version. That combination remains unfinished.

Source zero object-instance counts and negative effective draw ranges suppress
the call. A zero-length draw range still allows indirect commands, whose GPU
arguments supply their actual extent. Default infinite `InstancedBufferGeometry`
instance counts are admitted for indirect drawing. `BatchedMesh` retains its
source direct-subdraw precedence. Source object statistics remain separate from
the number of physical GPU calls. Argument contents, including optional nonzero
first-instance support, receive the device's normal WebGPU validation.

`three_renderer_indirect.test.mjs` checks the production facade, captured bytes,
resource sharing and bundle behavior with a recording device. The
`native_indirect_compute` and `native_indirect_host_upload` cases in the existing
native-primitives browser comparison exercise real compute and draw commands.
This is general WebGPU execution with retained source preparation, not a Rust
specialization or a measured performance claim.

## Native graphics storage bindings

The native program GPU owner also accepts borrowed `read-only-storage` bindings
alongside uniform blocks, samplers and textures. `captureComputeStorage(device,
attribute)` captures a requested source version and normalized layout without
copying or uploading CPU bytes. Resolve the lease in the renderer FIFO, after
the preceding compute operation has published that version. Vertex and fragment
bindings can alias the same allocation.

Resolution rejects a mismatched publication, changed owner or allocation,
disposal and device loss. Binding admission checks storage usage, alignment,
range and per-stage limits. Cached groups use buffer identity, offset and size,
so a new resource wrapper does not allocate another equivalent bind group.

Authored node materials can use `storage().element()` expressions in vertex and
fragment slots with Float32, Int32 and Uint32 scalar/vector arrays, standalone
Float32 `mat4` arrays, plus flat non-atomic structs containing scalar, vector and
matrix fields. A vec3 array uses
a padded four-word stride; structs retain the pinned source's member alignment
and matrix-column padding. This public route uses the compute owner's shared
upload history to initialize storage or apply marked host uploads at queued
render submission. No preceding compute call is required for host data, and
graphics compilation performs no upload. Unchanged host versions retain
compute-produced GPU contents.

Node updates may replace a compatible attribute, while alias/layout changes
require a new program. Ordinary `Mesh.count` and TSL `instanceIndex` also work
with these shader reads. Graphics storage writes, atomics, nested/boolean/atomic
struct members, standalone `mat2`/`mat3` arrays and PBO-backed storage remain
outside this profile.
See `THREE_NODE_PROGRAMS.md` for the graph, capture and browser-check contracts.

## Authored materials on storage-backed InstancedMesh

Authored node materials can also consume the source's built-in instance matrix
and color transforms directly from storage. Matrices require a Float32
`StorageInstancedBufferAttribute` with item size 16; optional colors use item size
3 or padded size 4. The source builder preserves matrix position transforms,
inverse-transpose normals and instance-color multiplication. Additional authored
storage expressions can use `instanceIndex`, and `positionNode` runs after the
instance transform.

```js
const material = new THREE.MeshBasicNodeMaterial();
material.colorNode = TSL.vec3(0.8, 0.6, 0.4);
const mesh = new THREE.InstancedMesh(geometry, material, 3);
mesh.setMatrixAt(1, new THREE.Matrix4().makeTranslation(2, 0, 0));
mesh.setMatrixAt(2, new THREE.Matrix4().makeTranslation(-2, 0, 0));
mesh.instanceMatrix = new THREE.StorageInstancedBufferAttribute(
  mesh.instanceMatrix.array, 16);
mesh.instanceColor = new THREE.StorageInstancedBufferAttribute(3, 3);
mesh.instanceColor.array.fill(1);
mesh.frustumCulled = false; // Or supply conservative bounds for GPU transforms.
scene.add(mesh);
renderer.render(scene, camera); // Host data can initialize storage directly.
```

Compute kernels can address these same attributes with `storage(..., 'mat4',
capacity)` and `storage(..., 'vec3', capacity)`. The graphics path consumes their
resident GPU values without an additional upload when the host version is
unchanged. Matrix/color attribute replacement or capacity changes rebuild the
graph; data, version and draw-count changes reuse it. Each queued draw keeps the
attributes and count captured before after-update callbacks. Its count must fit
the smaller matrix/color capacity, with zero still running source callbacks.

Culling continues to use CPU bounds, so GPU transforms require conservative
`mesh.boundingSphere` values or disabled frustum culling. Ordinary CPU instance
attributes, `InstancedBufferGeometry`, skinning and morphing remain outside this
authored-material profile.

## Checks

```sh
node --test tests/unit/gpu_compute_attributes.test.mjs tests/unit/gpu_shared_geometry.test.mjs
node tools/ingest/cli.mjs tests/e2e/webgpu_renderer/compute_geometry.html \
  --build-app /tmp/f3d-compute-geometry --route-webgpu-renderer
```

Serve the built application over localhost and inspect
`window.__f3dComputeGeometry`. A separate dependency-free native GPU fixture is
`tests/e2e/webgpu_renderer/native_compute_geometry.html`, served from the
repository root; inspect `window.__f3dNativeGeometry`. It exercises actual WGSL,
vertex/instance fetch and pixel readback, but uses minimal source data objects,
not the full r186 scene facade.

Unit checks use command-recording GPU doubles. Browser fixtures were syntax
checked, but their execution in the authoring host was blocked at navigation
by `ERR_BLOCKED_BY_ADMINISTRATOR`; no pixel-parity or hardware result is claimed.
