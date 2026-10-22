# Source HDR environments in native scenes

`createGpuThreeScene(device, scene, { three, environment: {} })` connects a
ready `scene.environment` panorama or data/image-backed cubemap to the existing native
image-based lighting filter and material renderer. This is an opt-in extension of the source scene
bridge, not a replacement for the pinned Three renderer or a PMREM-equivalence
claim. The application still owns its Three module, scene, camera, device,
attachments, animation clock, image decoding and frame scheduling.

## Render a ready HDR panorama

Import from `tools/ingest/three_scene.mjs` in the repository, or from
`gpu_playback.mjs` in a generated package built with the options below.
`THREE` must be the application's pinned r186 module. `readyHdrTexture` must
contain decoded, linear-sRGB RGBA pixels, not a URL or an already filtered PMREM
render target.

```js
import { createGpuThreeScene } from './gpu_playback.mjs';

readyHdrTexture.mapping = THREE.EquirectangularReflectionMapping;
readyHdrTexture.colorSpace = THREE.LinearSRGBColorSpace;
readyHdrTexture.needsUpdate = true;
scene.environment = readyHdrTexture;

const nativeScene = await createGpuThreeScene(device, scene, {
  three: THREE,
  environment: {},
  // Add shadow: {} when the scene also uses supported projected shadows.
});

// Borrow matching color/depth attachment views from the application's target.
// Each render submits synchronously. No internal frame loop is started.
nativeScene.render(camera, { colorView, depthView });

scene.environmentIntensity = 1.5;
scene.environmentRotation.y = Math.PI / 4;
nativeScene.render(camera, { colorView, depthView });
```

Intensity and rotation are read every frame. They do not refilter the panorama,
upload its pixels again, or replace scene geometry/material registrations.
Rotation uses the inverse of the source Euler rotation to transform world
sampling directions into the environment. All six source Euler orders are
accepted. Intensity must be finite, nonnegative and representable as f32.

The environment illuminates `MeshStandardMaterial` draws through the core's
metallic-roughness shading profile. Basic, Lambert, Phong and Toon draws do not
acquire scene-environment lighting. The actual effective material controls this
selection, including an admitted scene material override. Explicit material
`envMap` overrides and `MeshPhysicalMaterial` remain outside this source bridge.

Rigid, instanced and skin/morph-deformed Standard meshes use the same prepared
map. With `shadow: {}`, direct projected shadows and indirect environment
lighting compose without repeating deformation. Only adjacent equal-receiver
draws are grouped into a pass; global opaque/transparent ordering is retained.
Later passes load existing color and depth, including caller-supplied MSAA
attachments. This may add passes; no single-pass or performance claim is made.

## Source and numerical profile

Supply an r186 `DataTexture` with a 2:1 equirectangular image, RGBA format,
`HalfFloatType` plus `Uint16Array`, or `FloatType` plus `Float32Array`. The source
must be ready, have a positive texture version, and contain exactly four
components per pixel in fixed, unshared, attached storage.

Half-float input preserves finite source bits. Float32 input is rounded to
binary16 using nearest/ties-to-even, including subnormals and signed zero.
NaN, infinity and magnitudes above 65504 reject before native allocation; HDR
energy is not silently clipped. Linear-sRGB and `NoColorSpace` inputs are
admitted; the latter is interpreted as already-linear sRGB, not an unknown
primaries conversion.

The source `flipY` flag is honored. The upload adapts r186's equirectangular
vertical coordinate to the core's north-first panorama convention. Preserve the
orientation chosen by the application's decoder; this owner does not modify the
source flag or pixel array. Texture UV transforms and ordinary surface sampler
settings do not redefine the core's explicit panorama sampling profile.

The existing core filter produces diffuse E/pi, GGX-prefiltered radiance mips,
and a matching split-sum DFG texture. Its sampling profile is not Three's
PMREM/CubeUV layout or a promise of identical pixels. Render-target
textures, sRGB-encoded HDR, premultiplied pixels, authored mip chains, partial
update ranges and upload callbacks are rejected by this owner. Texture
backgrounds, fog and per-material reflection/refraction maps are not added by
this feature. Use the application's retained renderer for unsupported scenes. This opt-in API
reports a refusal rather than silently downgrading or switching renderers.

## Ready data cubemaps

The same scene path also accepts a source `CubeTexture` containing exactly six
ready `DataTexture` faces, in +X, -X, +Y, -Y, +Z, -Z order. Every face must be
square, equally sized, and use the parent's RGBA storage type. Supported types
are `HalfFloatType` / `Uint16Array`, `FloatType` / `Float32Array`, and
`UnsignedByteType` / `Uint8Array` or `Uint8ClampedArray`.

```js
// Each entry is an already-decoded r186 DataTexture, not a URL or image element.
const environment = new THREE.CubeTexture([px, nx, py, ny, pz, nz]);
environment.type = THREE.HalfFloatType; // Must agree with all six faces.
environment.format = THREE.RGBAFormat;
environment.colorSpace = THREE.LinearSRGBColorSpace;
environment.needsUpdate = true;
scene.environment = environment;
await nativeScene.prepare();
```

The parent uses `CubeReflectionMapping`; no application image decoder, worker,
new frame loop, or intermediate panorama is created. All six faces are checked
and captured before allocating the native six-layer `rgba16float` source. The
existing cube filter then produces the usual diffuse, specular and DFG resources.
HDR conversion follows the panorama rules above. Byte inputs can be linear or
sRGB; sRGB RGB is converted to linear before filtering, while alpha stays linear
and unpremultiplied. No color conversion is inferred from a face's own color-space
field: the parent cube's declared color space is authoritative.

Source handedness is baked into the upload: pinned r186 `CubeTextureNode`'s
X-direction reflection swaps +/-X and mirrors every face horizontally. When
`flipY` is true, pinned `WebGPUTextureUtils` additionally swaps +/-Y and reverses
rows. Neither the source face order nor its pixel arrays are mutated. Runtime
rotation and intensity use the existing environment descriptor without filtering
again. The source rules are from Three.js commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`, not a guessed skybox convention.

`maxPixels` counts all six faces together, and `maxBytes` includes their complete
binary16 native source plus filtering uniforms and outputs. Source face identity,
storage, dimensions and explicit face/parent versions participate in stale-state
checks. In-place pixel edits still require `environment.needsUpdate = true` and
`prepare()`; rendering does not scan the pixel buffers. The parent cube is the
lifetime owner: disposing it invalidates the filtered maps without disposing its
borrowed faces.

Compressed cubes, raw `{data,width,height}` face descriptors, render-target
cubes, refraction mapping and authored face mips remain outside this owner. This adds environment lighting,
not cube backgrounds or per-material `envMap` support. The panorama-background
copy API explicitly rejects cube input rather than treating a cube face as a
panorama.

## Ready browser-image cubemaps

The same `scene.environment` path also admits ordinary `CubeTextureLoader`
results once all six images have loaded. The parent must use RGBA unsigned-byte
storage and `CubeReflectionMapping`. Ready HTML images, HTML/Offscreen canvases,
ImageBitmaps and sRGB ImageData are supported; faces must be equally sized
squares. Mixing data-backed and browser-image faces is not admitted.

```js
const environment = await new THREE.CubeTextureLoader().loadAsync([
  'px.png', 'nx.png', 'py.png', 'ny.png', 'pz.png', 'nz.png',
]);
environment.colorSpace = THREE.SRGBColorSpace;
scene.environment = environment;
await nativeScene.prepare();
```

Loading/decoding remains the application's responsibility. Preparation reuses
`createGpuThreeTextures`, including its browser-copy, source color-space, sampler
and generated-mipmap contracts. It does not read a canvas, close a bitmap,
change source versions, reduce mip quality or start another frame loop. The
ImageBitmap contract requires `flipY=false` and unpremultiplied alpha; choose
orientation during decoding. Wide-gamut ImageData and authored cube mip chains
remain outside this profile. See `THREE_TEXTURES.md` for external-copy semantics.

Six GPU texel blits convert the uploaded cube to the existing linear
`rgba16float` filter input. Compatible sRGB views decode RGB on the GPU and leave
alpha linear. The blits mirror U and exchange X faces; when `flipY=true`, they
also exchange Y faces, because the shared uploader reverses rows but preserves
layer identities. The original cube, all image objects and their order remain
unchanged. There is no intermediate panorama or CPU image readback.

`maxBytes` conservatively charges the full uploaded source mip chain, normalized
half-float cube, filter uniforms and filtered outputs together. Conversion drains
before source residency is retired; filter readiness is still required before
publication. Abort, source disposal and device loss promptly retire preparation
resources, and changed versions, face identities, dimensions or sampling state
reject stale publication. Subsequent intensity/rotation changes do not upload or
filter again. Canvas paint changes still need `environment.needsUpdate=true`
and a preparation boundary.

## Explicit updates, replacement and lifetime

Panorama or cubemap content is copied and filtered only during creation or `prepare()`.
For changed pixels, acknowledge the source upload and prepare before rendering:

```js
// Modify readyHdrTexture.image.data using the application's chosen workflow.
readyHdrTexture.needsUpdate = true;
await nativeScene.prepare();
nativeScene.render(camera, { colorView, depthView });

scene.environment = anotherReadyHdrTexture;
await nativeScene.prepare();

scene.environment = null;
await nativeScene.prepare();
```

Identity, dimensions, storage, mapping/orientation and acknowledged version
changes invalidate the prepared environment. Rendering stale state refuses
rather than filtering during a frame or using the wrong panorama. Unacknowledged
in-place pixel edits are not interpreted as uploads and are not scanned by each
render. Unrelated `prepare()` calls reuse an unchanged filtered map.

A replacement is prepared while the old map remains owned. Prior submitted
consumers are drained before the old map is retired. Failed preparation does
not publish partial state; restoring the previous valid source allows its
previous ready map to render. Source mutation during asynchronous filtering
rejects stale publication. Source disposal, explicit owner disposal, abort and
device loss invalidate owned native resources without disposing application
geometry, camera, device or source pixels.

`await nativeScene.whenIdle()` includes environment readiness together with
other scene consumers. `nativeScene.dispose()` stops ownership and cancels
outstanding preparation waits; it does not undo commands already submitted to
the GPU. Stop calling the owner after disposal. `diagnostics.environmentBytes`
reports completed owned maps, and `colorPasses` includes combined environment
and shadow receiver spans. An in-progress filter's private temporary allocation
is bounded by its budget but is not exposed as completed-map residency.

## Bounds and deployment

Environment options are `maxBytes` (default 128 MiB), `maxPixels` (16,777,216),
`size` (128), `diffuseSize` (32), `lutSize` (128), `samples` (1024), and
`maxSampleWork` (268,435,456). The source panorama or complete six-face cube, filter uniform allocation and
filtered outputs must fit the budget together. Replacements also charge the
still-owned old map. Sizes and sample counts are never reduced automatically
to satisfy a budget. The existing core planner enforces device limits and work
bounds. The scene's ordinary material-texture budget is separate.

Build a relocatable source-scene package with:

```js
buildAnimation(inputGltfOrGlb, freshOutputDirectory, {
  webgpu: true,
  threeScene: true,
  environment: true,
});
```

The combined options package `three_environment.mjs` and its core dependencies,
and export `createGpuThreeEnvironment` / `ThreeEnvironmentError` for explicit
standalone ownership as well as `createGpuThreeScene`. Runtime source ownership
still requires `environment: {}`. CPU-only, ordinary GPU, source-only and
IBL-only packages do not gain the source-environment module. The build performs
no HDR fetch or filtering; importing generated entries initializes no GPU.

Focused tests execute production source ownership, conversion, scene/shadow
composition and package emission/relocation. Three classes, lower GPU resources,
filter/plan boundaries and pose decoding are explicit fixtures. These tests do
not establish native WGSL execution, retained-Three rendering parity, browser
pixels or a measured speedup.

`node --test tools/ingest/three_environment.test.mjs tools/ingest/three_cube_environment.test.mjs`
also exercises the actual cube filter command generator, with a recording GPU
and fixture source classes. Cubemap tests independently sample off-axis directions
on all faces, inspect transfer conversion and emitted cube views/passes, and
cover whole-source budgets, stale publication, cancellation and cleanup. This is
command/ownership evidence, not native rendered-pixel equivalence.

`node --test tools/ingest/three_image_environment.test.mjs` exercises image
admission, exact face/view bindings, byte budgets, asynchronous publication and
failure cleanup through the production source owner and conversion command
generator. Browser classes, the shared texture pool and GGX filter are explicit
fixtures; this test does not establish browser image-copy or rendered-pixel parity.
