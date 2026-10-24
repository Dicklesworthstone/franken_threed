# GPU-authored storage images in native TSL graphics

Native graphics graphs can now read a `StorageTexture` or `StorageArrayTexture`
through an ordinary `TextureNode`. Compute and graphics use the renderer's same
storage owner and the same native image. Sampling does not require copying to a
RenderTarget, creating a CPU pixel shadow, allocating an upload image, or adding
a new per-draw GPU fence.

```js
// simulation is an existing StorageTexture, and kernel writes that image.
// Use an ordinary texture() read in graphics, not a textureStore() node.
material.colorNode = THREE.TSL.texture(simulation);
await renderer.init();
renderer.compute(kernel);
renderer.render(scene, camera);
await renderer.waitForGPU();
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
Both 2D and 2D-array storage inputs are supported. Array reads use the source
TextureNode layer coordinate; layers are not collapsed into a 2D image. The
source builder retains the authored vertex/fragment visibility, UVs, explicit
levels and ordinary sampling/load semantics.

A 2D storage image can also occupy an already admitted material map slot on the
native node route. A storage array is not implicitly accepted as a 2D material
map. This does not change the selection rule that ordinary built-in materials
without authored node slots keep their existing fixed rendering path.

Nearest float32 reads work without `float32-filterable`. Linear float32 sampling
still requires that feature on the actual GPUDevice. The shared storage owner's
sampler layout now agrees with the program layout: non-filtering for unfilterable
float32 images, filtering when the feature makes their float samples filterable.
Integer compute bindings retain their existing non-filtering layout.

Storage dimensions, formats and mip controls remain subject to the authoritative
storage owner's admission, device limits and byte budgets. Generated mip chains
remain shared with compute; graphics does not regenerate them independently.
Manual mip writes retain the owner's manual policy. Sampling allocates neither
a target UV replica nor a second copy of the storage image.

## Graph and lifetime boundaries

Program selection includes storage-vs-CPU ownership, array dimension and the
texture alias partition, in addition to existing source format and sampling
metadata. Switching ownership or dimension rebuilds the graph; changing only
GPU contents does not. The shared sampled-input adapter supplies the actual
layout, and missing or contradictory layouts are rejected rather than inferred
through the CPU uploader.

This is a read-only graphics entrance. `StorageTextureNode` writes, writable
bindings introduced during construction, integer graphics samples, volume/3D
graphs and storage-backed PMREM environments remain explicitly outside the
profile. A storage resource flag alone is not permission to write it in a draw.
The existing owner rejects overlapping sampled reads and storage writes in one
dispatch. Disposal, renderer cancellation, device loss and producer-validation
errors keep their existing storage-owner failure behavior.

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

The final run passed 57 tests with no failures, cancellations or skips. The same
suite failed 42 tests against the unchanged graph preparation and storage owner.
This is graph admission, resource-routing and lifecycle evidence, not real WGSL
compilation, compute execution, browser pixel parity or a performance benchmark.
