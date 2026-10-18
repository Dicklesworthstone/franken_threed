# Integer storage and GPU texture copies

The new WebGPU renderer admits `IntType` and `UnsignedIntType` storage textures
in R, RG and RGBA 32-bit native formats, including the pinned r186 integer-format
aliases. `StorageTexture`, `StorageArrayTexture` and `Storage3DTexture` use the
same owned texture for compute writes, later reads and compatible typed bindings.
Values such as `0xffffffff` and `16777217` never pass through a float texture.
Integer data requires matching signed/unsigned shader inputs, not an ordinary
float material map. This is general WebGPU execution with retained TSL shader
preparation, not a Rust specialization or a measured performance improvement.

## Copy simulation state without downloading it

The public `copyTextureToTexture` method now supports two storage-texture
endpoints with the same native format:

```js
const current = new THREE.StorageTexture(width, height);
current.type = THREE.UnsignedIntType;
current.format = THREE.RedIntegerFormat;
current.generateMipmaps = false;
current.minFilter = current.magFilter = THREE.NearestFilter;
const history = current.clone();

await renderer.init();
await renderer.compileComputeAsync([produceCurrent, consumeHistory]);
renderer.compute(produceCurrent);
renderer.copyTextureToTexture(current, history);
renderer.compute(consumeHistory);
await renderer.waitForGPU();
```

Here the kernels refer to `current` and `history`, respectively. The copy is a
native texture-to-texture command: it does not allocate a staging buffer, map
texture data, invoke a conversion shader or change the source CPU objects.
First use allocates any missing destination storage under the shared byte budget.

The initialized call returns `undefined`, including when preceding render
preparation requires queueing. Its command occupies the renderer's existing
submission FIFO; later compute/render work cannot overtake it. The call captures
the texture generations, box, position, mip levels and destination mip-update
policy. Resize or disposal before submission rejects the stale queued copy.
Deferred errors surface on the next public call and through `waitForGPU()`.

```js
renderer.copyTextureToTexture(source, destination,
  new THREE.Box3(new THREE.Vector3(x, y, layer),
                 new THREE.Vector3(x + w, y + h, layer + layers)),
  new THREE.Vector3(dstX, dstY, dstLayer), sourceMip, destinationMip);
```

Box2 selects one source depth slice at z=0; Vector2 places the destination
origin at z=0. Box3/Vector3 address array layers or volume slices. Coordinates
are nonnegative integer texels of the selected mip,
not base-level pixels. Without a box, the whole source mip is copied. Volume
mips shrink depth; array mips retain their layer count. Bounds and both endpoint
allocation requirements are checked before either allocation or copy command.

Same-texture copies require disjoint native subresources: different mips or
nonoverlapping 2D array layers. Disjoint x/y rectangles in the same layer, and
different voxel slices in one 3D mip, do not meet WebGPU's copy rule. A 2D texture
and a 2D array can exchange selected layers; native 2D-to-3D copies and format
conversion are not admitted. Depth, compressed, ordinary source-data and
render-target endpoints are not added by this storage-to-storage path.

## Mip and filtering boundaries

Integer texture samplers require nearest filtering. Set `generateMipmaps=false`
for a single level, or `mipmapsAutoUpdate=false` to allocate levels for explicit
compute writes/copies. Automatic float-style integer mip interpolation is
rejected before a base-level write; no averaging/truncation policy is invented.
For existing float storage textures, copying into level zero regenerates mips
under the captured auto-update policy. Copying a higher level does not overwrite
manual levels. No extra copy or upload is needed before a subsequent texture read.

## Focused checks

```sh
node --test tests/unit/three_storage_integers.test.mjs \
  tests/unit/three_storage_texture_copy.test.mjs tests/unit/three_storage_textures.test.mjs
node tools/ingest/cli.mjs tests/e2e/webgpu_renderer/storage_integer_copy.html \
  --build-app /tmp/f3d-storage-copy --route-webgpu-renderer
```

The unit tests record GPU commands and use byte-accurate mock texture copies;
they do not execute WGSL. The routed fixture checks the public renderer and
pinned TSL builder. The dependency-free `native_storage_integer_copy.html` fixture
checks actual WGSL store/load, native copies and manual mip copies for all six
integer formats in 2D, arrays and volumes using minimal source texture objects.
Both expose `window.__f3dIntegerCopy` when served from the repository root or
built output. Neither fixture is a performance benchmark.
