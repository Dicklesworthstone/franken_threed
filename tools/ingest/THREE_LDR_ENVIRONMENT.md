# LDR panorama environment lighting

The existing source environment owner also accepts ready 2:1 equirectangular
RGBA unsigned-byte panoramas. This extends `THREE_SCENE_ENVIRONMENT.md` without
adding another scene renderer, loader, frame loop or environment-filter profile.
It is environment lighting, not a new background or per-material envMap path.
The native GGX/DFG filter is still not a claim of Three PMREM pixel equivalence.

```js
// The application loads and owns this pinned r186 Texture.
const panorama = await new THREE.TextureLoader().loadAsync('environment.jpg');
panorama.mapping = THREE.EquirectangularReflectionMapping;
panorama.colorSpace = THREE.SRGBColorSpace;
scene.environment = panorama;

// The native scene was constructed with { three: THREE, environment: {} }.
await nativeScene.prepare();
nativeScene.render(camera, { colorView, depthView });
```

## Sources and conversion

Decoded HTML images, HTML/Offscreen canvases, ImageBitmaps and sRGB ImageData
use the existing shared texture owner for upload, source orientation and mip
creation. An ImageBitmap requires `flipY=false` and unpremultiplied alpha;
choose its orientation at decode time. The application remains responsible
for image loading, decoding, origin-clean browser-copy permission and lifetime.
No canvas pixels are read on the CPU and no source bitmap is closed.

One native render pass converts the uploaded base level to a same-size
`rgba16float` panorama. The view uses the source's declared sRGB or linear
transfer; alpha is linear coverage. Integer texel addressing reverses V after
the uploader applies `flipY`, translating r186's `asin(y)/PI+.5` convention to
the filter's north-first `acos(y)/PI`. It neither mirrors U nor resizes the
image. The existing filter consumes that normalized panorama directly.

A `DataTexture` may instead hold exactly width * height * 4 components in a
fixed, unshared `Uint8Array` or `Uint8ClampedArray`, including a nonzero-offset
view. CPU capture converts bytes to linear binary16, preserving alpha, source
orientation and input storage. RGB is sRGB-decoded only for `SRGBColorSpace`;
`LinearSRGBColorSpace` and `NoColorSpace` are already-linear values. The same
bounded nearest/ties-to-even conversion used by HDR inputs produces the upload.
Existing half-float and float HDR sources retain their previous behavior.

## Preparation and lifetime

The source must be ready and explicitly versioned, use RGBA storage and
`EquirectangularReflectionMapping`, and remain 2:1 within pixel/device limits.
Render targets, live video, compressed sources, refraction mapping, upload
callbacks, partial updates, premultiplication and authored mip chains are not
part of this explicit panorama profile. Other application paths remain separate.
The metadata check does not scan image or data pixels during rendering.

For browser images, the byte budget includes the entire generated source mip
chain, the normalized half-float panorama, filter uniforms and filtered outputs.
For byte DataTextures it includes the normalized half-float input and filtering
allocations. No quality setting is reduced to fit. The surrounding scene owner
also charges the old environment until outstanding consumers have drained.

Conversion drains before source GPU residency is retired; filtering completes
before publication. Source versions, identities, dimensions or sampler changes
during asynchronous preparation reject stale results. Abort, disposal, device
loss and GPU errors retire temporary resources without disposing borrowed input.
Changes to scene intensity/rotation do not upload or filter again. Canvas paints
and in-place byte edits require `panorama.needsUpdate=true` and `prepare()`.

Generated packages use the existing `webgpu:true`, `threeScene:true`,
`environment:true` build options; no new runtime module is introduced. The
CPU-only `captureThreeEnvironmentPixels` helper admits byte DataTextures but
explicitly refuses browser images instead of reading them back.

## Regression evidence

`node --test tools/ingest/three_ldr_environment.test.mjs` executes production
admission, ownership, byte conversion and GPU command generation. It covers all
256 byte values against an independent half-value oracle, both row orientations,
color spaces, rectangular mip tails, exact budgets, stale preparation and
failure/cancellation cleanup. Source classes, browser images, the texture-pool
boundary, GGX filter and GPU are explicit fixtures. This is not browser-copy,
native WGSL pixel-parity, complete compatibility or hardware-performance evidence.
