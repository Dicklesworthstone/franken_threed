# Native integer storage-texture loads

Native TSL graphics can consume compute-produced integer labels, identifiers,
bitfields and simulation state without routing the texels through floating-point
storage. The graphics compiler admits signed and unsigned 32-bit texel loads
from `StorageTexture`, `StorageArrayTexture` and `Storage3DTexture` through the
renderer's existing shared storage owner.

## Usage and formats

Use the pinned source's sampler-free load factories, not `texture()` sampling:

```js
const labels = new THREE.StorageTexture(width, height);
labels.type = THREE.UnsignedIntType;
labels.format = THREE.RedIntegerFormat;
labels.colorSpace = THREE.NoColorSpace;

// texelCoordinates is the application's integer-coordinate TSL node.
// colorFromId is the application's TSL graph mapping an integer ID to a color.
const id = THREE.TSL.textureLoad(labels, texelCoordinates);
material.colorNode = colorFromId(id);

renderer.compute(labelKernel); // The existing producer writes labels.
renderer.render(scene, camera);
```

Volumes use `THREE.TSL.texture3DLoad(volume, voxelCoordinates)`. Array loads
retain the source TextureNode's separate layer coordinate. The formats are
`r32sint`, `rg32sint`, `rgba32sint`, `r32uint`, `rg32uint` and `rgba32uint`.
Both ordinary R/RG/RGBA format names and their explicit IntegerFormat variants
are accepted by the shared storage owner with `IntType` or `UnsignedIntType`.

The retained r186 builder infers integer scalar/vector values from the texture
and emits typed `texture_2d`, `texture_2d_array` or `texture_3d` declarations.
The native graph descriptor uses the matching `sint` or `uint` sample type.
No sampler is required or admitted for these loads. Consequently a storage
texture's default linear min/mag filter metadata does not prevent a texel load:
those filters are not used by the shader. `float32-filterable` is irrelevant to
integer bindings. An application may explicitly map or convert the loaded
integer to a display color after its integer computation.

## Ownership, cache and lifetime

Graphics and compute use the same native image and allocation owner. This
change adds no CPU pixel shadow, integer-to-float upload, texture copy, alternate
residency pool, or per-draw GPU completion fence. A captured draw retains its
texture identity and observes preceding producers through the existing queue.
Compatible replacement textures can reuse the graph while older captured draws
retain their earlier image.

Signedness, source format, dimensionality and binding aliases remain structural.
An incompatible replacement cannot reuse a stale descriptor. Integer operations
are validated during admission, after source construction, during cache checks,
and after source updates. Integer-bearing captures also preflight the live
texture references before observable node callbacks. Unsigned depth textures
retain their separate depth sample type and are not treated as integer color.

The existing owner remains authoritative for format, shape, device and byte
limits, mip levels, feedback hazards, generations, disposal, renderer lifetime,
device loss and producer validation failures. Manual integer mip writes remain
supported by that owner; automatic floating-point mip reduction is not silently
applied to integer data.

## Explicit boundaries

This is an authored, read-only, integer storage entrance. Inputs require
`NoColorSpace` and must not be normalized. Sampled/interpolated operations,
comparison, bias, gradients, gather, builder-introduced sampler bindings and
graphics storage writes are rejected. Integer textures cannot occupy ordinary
floating-point material-map slots. CPU-backed integer images, integer render
attachments, cube integer textures and storage-backed PMREM inputs are not
added by this change.

The pinned source inspected for the type and load contract is Three.js commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`: `NodeUtils.getTextureType`,
`NodeBuilder.getComponentTypeFromTexture`, `TextureNode.generateSnippet` and
`generate`, and `WGSLNodeBuilder.isUnfilterable`, `getUniformFromNode`,
`getUniforms` and `generateTextureLoad`. The adapter does not monkey-patch these
methods or replace their integer type inference.

## Verification

```sh
node --check tools/ingest/three_node_program.mjs
node --check tools/ingest/three_node_integer_graphics.test.mjs
node --test tools/ingest/three_node_integer_graphics.test.mjs
```

The focused suite passes **206/206** tests on Node 22.16.0 with no failures,
cancellations or skips. Against the unchanged compiler from
`f2f3eed88f6c1f2dbfbb507475fbeb477ad3dbd8`, the same suite has 149 failures and
57 passes. The original compiler and unchanged storage dependency were
reconstructed and verified against their exact repository Git blob hashes.
Both changed JavaScript modules pass syntax checks.

The matrix covers all three dimensions, both signednesses, six source format
names and vertex/fragment/shared visibility. Additional cases cover independent
raw integer producer bytes (including values outside float32's exact integer
range), queued replacements, callback ordering, alias changes, typed layout
changes, forbidden operations, manual mip ownership, generation/budget/feedback
checks, producer failure, cancellation, device loss, and float/depth regressions.

These tests execute the production graph compiler and storage owner with
structural source objects, a recording builder/device and controlled producer
writes. A scoped loader isolates unrelated graph contexts, copy planning and
float-only mip generators. They do not instantiate the unified sampled-input
adapter or full renderer, compile/execute WGSL, prove browser pixel parity, run
repository-wide tests, or measure hardware performance. Source inspection and
recording tests are not a substitute for that end-to-end validation.
