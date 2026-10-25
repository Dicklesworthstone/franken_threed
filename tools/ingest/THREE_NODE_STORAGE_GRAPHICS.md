# GPU-authored storage images in native TSL graphics

Native graphics graphs can read a `StorageTexture` or `StorageArrayTexture`
through an ordinary `TextureNode`, and a `Storage3DTexture` through the pinned
`Texture3DNode`. Compute and graphics use the renderer's same storage owner and
the same native image. Sampling does not require copying to a RenderTarget,
creating a CPU pixel shadow, allocating an upload image, or adding a new
per-draw GPU fence.

```js
// simulation is an existing StorageTexture, and kernel writes that image.
// Use an ordinary texture() read in graphics, not a textureStore() node.
material.colorNode = THREE.TSL.texture(simulation);
await renderer.init();
renderer.compute(kernel);
renderer.render(scene, camera);
await renderer.waitForGPU();
```

For a compute-produced volume, use the source three-dimensional coordinates:

```js
const volume = new THREE.Storage3DTexture(width, height, depth);
volume.type = THREE.FloatType;
volume.format = THREE.RedFormat;
volume.minFilter = volume.magFilter = THREE.NearestFilter;
// uvw is the application's vec3 coordinate node; the producer writes volume.
material.colorNode = THREE.TSL.vec3(THREE.TSL.texture3D(volume, uvw).r);
```

The application still owns its source texture. The native renderer's existing
submission ordering determines which compute/upload results a draw observes.
The graph preparation layer captures texture identities and binding metadata,
not a copy of the GPU pixels. Replacing a node's texture with a compatible new
storage image can reuse the program, while already captured draws retain their
previous image. Reconfiguration or disposal invalidates accepted storage
bindings rather than retargeting them to another allocation.

## Admitted reads

The shared float-sample formats are linear RGBA8, RGBA16F, and R/RG/RGBA32F.
2D, 2D-array and 3D storage inputs are supported. Array reads use the source
TextureNode layer coordinate; layers are not collapsed into a 2D image. Volume
reads require the exact pinned Texture3DNode and a Storage3DTexture, preserving
its vec3 sampled coordinates or integer voxel coordinates for unsampled loads.
A volume's third coordinate is not an array-layer/depthNode argument. The source
builder retains authored vertex/fragment visibility, UVs and explicit levels.

A 2D storage image can also occupy an already admitted material map slot on the
native node route. Storage arrays and volumes are not implicitly accepted as 2D
material maps. This does not change the selection rule that ordinary built-in
materials without authored node slots keep their existing fixed rendering path.

Nearest float32 reads work without `float32-filterable`. Linear float32 sampling
still requires that feature on the actual GPUDevice. The shared storage owner's
sampler layout agrees with the program layout: non-filtering for unfilterable
float32 images, filtering when the feature makes their float samples filterable.
Integer compute bindings retain their existing non-filtering layout. Volume
samplers retain all three wrapping axes, including per-draw captured `wrapR`.
Invalid wrapping is rejected by the owner's sampler-capture boundary.

Storage dimensions, formats and mip controls remain subject to the authoritative
storage owner's admission, device limits and byte budgets. Volume limits apply
to width, height and depth; its byte budget includes every voxel and mip level.
Generated mip chains remain shared with compute; graphics does not regenerate
them independently. Volume mips use the existing trilinear reduction, shrinking
all three spatial dimensions rather than treating depth as an array-layer count.
Manual mip writes retain the owner's manual policy. Sampling allocates neither
a target UV replica nor a second copy of the storage image.

## Graph and lifetime boundaries

Program selection includes storage-vs-CPU ownership, array/volume dimension,
volume-node kind and the texture alias partition, in addition to existing source
format and sampling metadata. Switching ownership or dimension rebuilds the
graph; changing only GPU contents does not. A 2D/3D node-resource mismatch fails
explicitly instead of silently changing coordinate interpretation. The shared
sampled-input adapter supplies the actual layout, and missing or contradictory
layouts are rejected rather than inferred through the CPU uploader.

This is a read-only graphics entrance. `StorageTextureNode` writes, writable
bindings introduced during construction, integer graphics samples, CPU-backed
Data3DTexture inputs, 3D gathers, depth comparisons on volumes, and storage-backed
PMREM environments remain explicitly outside the profile. A storage resource
flag alone is not permission to write it in a draw. The existing owner rejects
overlapping sampled reads and storage writes in one dispatch. Disposal, renderer
cancellation, device loss and producer-validation errors keep their existing
storage-owner failure behavior.

## Verification

Run the self-contained focused suite:

```sh
node --test tools/ingest/three_node_storage_graphics.test.mjs
```

The suite executes the production graph preparation and storage owner, including
its real mip helper modules. Source objects and WGSL construction are structural
recording fixtures; unrelated attribute, shadow and render-pipeline imports are
isolated by an explicitly scoped ESM loader. GPU calls and submission order are
recorded, and independent byte arrays check producer/consumer image identity and
queued values. The fixture exercises the same storage layout/capture/resolve
methods used by the unified sampled-input adapter, but does not instantiate that
adapter or the full renderer.

The volume extension passes 130 tests with no failures, cancellations or skips.
The expanded suite fails 56 tests against the unchanged graph compiler from
5c6ea98af5c6e51c26c805f33477c179a0c4bdb2; its original 57-test suite passes before
the extension. Both modified JavaScript modules pass `node --check`. Existing
production dependencies were verified against their Git blob hashes before the
run. No storage owner, uploader, submission FIFO or mip implementation is changed
by this extension.

This is graph admission, resource-routing and lifecycle evidence, not real WGSL
compilation, compute execution, browser pixel parity, repository-wide validation
or a performance benchmark.
