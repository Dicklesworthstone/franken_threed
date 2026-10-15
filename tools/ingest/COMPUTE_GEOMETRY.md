# Compute-authored vertex and instance streams

The new WebGPU renderer can bind a compute-produced `StorageBufferAttribute`
straight into ordinary `BufferGeometry` vertex inputs. `InstancedMesh` can use
`StorageInstancedBufferAttribute` for `instanceMatrix` and `instanceColor`.
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

Compute-authored indices, integer vertex fetch, repeated instance divisors, and
storage-node expressions in node-material slots are not added by this change.
Those operations must not be inferred from Float32 vertex-buffer support.

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
