# Storage texture arrays and 3D volumes

The compute texture owner now accepts pinned r186 `StorageArrayTexture` and
`Storage3DTexture`, in addition to `StorageTexture`. Storage writes, later
compute reads, and compatible dimensioned sampling bindings share one native
allocation. No buffer staging, CPU texture upload, or readback is needed between
producer and consumer. This is general WebGPU execution, not Rust specialization
or a measured performance claim.

Use the source TSL operations: `textureStore` with `storageTexture(t).depth(layer)`
for array writes, `textureLoad(t, xy).depth(layer)` for array reads, and
`textureStore(t, xyz, value)` / `texture3DLoad(t, xyz)` for volume writes/reads.
The existing renderer compute queue preserves the selected texture generation.
Texture dimension/format changes select different layouts and shader programs.

Array textures retain all layers in every mip. Volume textures halve width,
height and depth until every dimension reaches one; their mip count includes
the depth axis. Automatic reduction uses clamped, center-sampled bilinear taps
within each array layer and trilinear taps across volume voxels. Each dependent
mip is a separate compute pass with disjoint mip views; one submission contains
the full chain. Float32 inputs use `textureLoad` and explicit unfilterable
layouts, so no optional filtering feature or precision downgrade is needed.

The r186 array and volume constructors **do not set `mipmapsAutoUpdate`**.
That absence is preserved and disables automatic regeneration, matching the
source binding check. Set `generateMipmaps = true` and
`mipmapsAutoUpdate = true` to opt into generated levels. With auto-update disabled,
manual storage writes can select allocated mip levels without being overwritten.
Volume `wrapR` is captured in the sampler's W address mode.

Supported formats remain linear RGBA8, RGBA16F, and R/RG/RGBA32F. Linear float32
sampling still requires the actual `float32-filterable` device feature. Positive
integer dimensions, depth-aware memory budgets, 3D spatial limits, array-layer
limits, stale generations, and conflicting sampled/storage usage are checked.
This does not make a standard 2D material map accept a 3D texture: consumers must
use the matching source texture operation/binding. Cubes, compressed/integer
storage formats and general node-material shader execution remain separate work.

## Focused checks

```sh
node --test tests/unit/three_storage_textures.test.mjs \
  tests/unit/three_storage_arrays.test.mjs tests/unit/three_storage_volumes.test.mjs
node tools/ingest/cli.mjs tests/e2e/webgpu_renderer/storage_dimensions.html \
  --build-app /tmp/f3d-storage-dimensions --route-webgpu-renderer
```

The source fixture checks texture-store/load, float32 mip readback, and
volume-to-vertex-color rendering. `native_storage_dimensions.html`, served from
the repository root, checks actual WGSL and every mip against an independent
CPU interpolation reference, using minimal source-data objects rather than the
full r186 facade. Both expose `window.__f3dStorageDimensions`.

Authoring-session evidence: 36 command-recording tests passed. Fixture JavaScript
was syntax checked. The native browser attempt was blocked at localhost
navigation by `ERR_BLOCKED_BY_ADMINISTRATOR`, before GPU execution. No successful
browser, source parity, hardware or performance result is claimed.
