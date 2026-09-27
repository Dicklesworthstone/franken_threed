# Native backgrounds for source scenes

`createGpuThreeScene(..., { background: {} })` draws a ready source 2D
texture, CubeTexture or HDR panorama behind the scene. It adds a visible background to the separately
selected `environment: {}` lighting path; one source texture can serve both.
The background retains original-resolution pixels, not an undersized filtered
lighting cube. No source scene clone, renderer replacement, image decoding,
frame loop, PMREM filtering or GPU readback is introduced.

This opt-in background profile extends the default source-scene refusal of
texture backgrounds described in `THREE_SCENE.md`. It is not complete Three.js
background compatibility. Ordinary RGBA byte/image UV textures and six-face
byte/image CubeTextures use source-owned residency and sampling. HDR panoramas
retain their original profile. Camera and ownership contracts are explicit below.

## Live source scene

```js
import { createGpuThreeScene } from './three_scene.mjs';

// THREE is the application's pinned r186 module. The decoder and pixels belong
// to the application: RGBA Float32Array/FloatType or Uint16Array/HalfFloatType.
readyHdrTexture.mapping = THREE.EquirectangularReflectionMapping;
readyHdrTexture.colorSpace = THREE.LinearSRGBColorSpace;
readyHdrTexture.needsUpdate = true;
scene.background = readyHdrTexture;
scene.backgroundBlurriness = 0;
scene.environment = readyHdrTexture; // Optional lighting, independently selected.

const nativeScene = await createGpuThreeScene(device, scene, {
  three: THREE,
  background: { maxBytes: 128 * 1024 * 1024 },
  environment: {},
  shadow: {}, // Optional supported directional/spot shadows.
  renderer: { format: 'rgba16float', sampleCount: 1 },
});

// Supply matching linear-HDR color/depth attachments. The existing HDR canvas
// compositor can tone-map the completed image; this pass does not tone-map it.
nativeScene.render(perspectiveCamera, { colorView, depthView });

scene.backgroundIntensity = 1.5;
scene.backgroundRotation.y = Math.PI / 4;
nativeScene.render(perspectiveCamera, { colorView, depthView });
```

Camera projection, camera world orientation, `backgroundIntensity` and
`backgroundRotation` are read for every frame. Those changes do not reupload
pixels, refilter lighting, or replace geometry/material registrations. Camera
translation is removed before composing inverse projection with orientation, so
moving the camera does not move the panorama's center. All six Euler orders,
view offsets, aspect ratios and both source WebGL/WebGPU clip conventions are
admitted. The bridge updates camera/scene matrices using its existing rules;
application-disabled matrix updates remain the application's responsibility.

The panorama covers the entire color attachment. The background pass has no
depth attachment and never clears, tests or writes scene depth. Geometry then
loads background color while retaining the caller's original depth clear/load
policy. Later environment/shadow receiver spans load both existing buffers.
With 4x MSAA, the background stores multisample color without an early resolve;
the normal scene pass retains the caller's resolve target. This adds a pass,
not a claimed single-pass optimization or measured speedup.

Rigid, instanced, skinned and morphed meshes continue through their existing
paths. Shadow depth rendering precedes the background and color; deformation
is not repeated for the background. Mixed-material/transparent order is not
changed. Null and solid-color backgrounds retain their existing behavior and
allocate no panorama. Diagnostics include completed/pending `backgroundBytes`,
last-successful-frame `backgroundPasses`, and total `colorPasses`.

## HDR source and common output profile

Supply the same ready, fixed, unshared, attached 2:1 RGBA HDR `DataTexture`
profile used by `THREE_SCENE_ENVIRONMENT.md`. Half-float input preserves finite
source bits. Float32 input rounds to binary16 with nearest/ties-to-even,
including subnormals and signed zero. Nonfinite values or magnitudes above
65504 reject before GPU allocation. Input dimensions, versions, mapping and
color space are checked; source pixels and upload versions are never modified.

The source `flipY` flag is adapted to a north-first native panorama exactly as
in environment lighting. Background rendering uses original-resolution base
pixels with repeat-U/clamp-V linear sampling. Source UV transforms, authored
mipmaps, sampler preferences, partial uploads and upload hooks do not redefine
this profile. Unsupported hooks, mip chains and source formats refuse rather
than being ignored. HDR refraction mapping, float CubeTextures, byte panoramas,
video and PMREM/CubeUV textures remain outside this source adapter. Byte/image
screen and cube profiles are described below.

Only non-reversed perspective cameras and `backgroundBlurriness === 0` are
supported for source HDR backgrounds. Orthographic/array cameras and blurred
backgrounds refuse explicitly; null/solid-color source backgrounds remain
available with the scene bridge's other admitted cameras. Fog is not added.

Background RGB remains linear and is multiplied by source intensity. Sampled
alpha is preserved, with no blend or forced-opaque conversion. There is no
implicit tone mapping, gamma approximation or output-color-space choice. Use
an admitted sRGB output view for native transfer encoding, or a linear HDR
attachment and the existing whole-image output compositor for tone mapping.
An ordinary unorm attachment can clip HDR values; it is not an HDR substitute.
The background-enabled scene defaults to `rgba8unorm` and one sample unless
`renderer.format` / `renderer.sampleCount` select another admitted profile.

## Ordinary 2D textures and cubemaps

Use the same `background:{}` scene option and existing preparation boundary.
No new factory or shader option is required at the source-scene level:

```js
// A loaded RGBA image/canvas Texture, or an RGBA Uint8Array DataTexture.
// The application loads/decodes the image and requests its upload as usual.
scene.background = readyTexture; // UVMapping; not an equirectangular byte map.
await nativeScene.prepare();
nativeScene.render(camera, { colorView, depthView });

// Offset/repeat/rotation/center and intensity are live uniform changes.
readyTexture.offset.x = 0.25;
scene.backgroundIntensity = 0.8;
nativeScene.render(camera, { colorView, depthView });

// readyCube has six equal square byte/image faces in Three's face order.
scene.background = readyCube;
await nativeScene.prepare();
nativeScene.render(perspectiveCamera, { colorView, depthView });
```

Screen textures use the source UV matrix and ignore camera orientation and
`scene.backgroundRotation`. Both source perspective and orthographic cameras
work. With `matrixAutoUpdate:true`, the built-in texture `updateMatrix()` runs
at capture, as in the retained source renderer; otherwise the authored matrix
is used. Custom matrix/update hooks are rejected. `backgroundBlurriness` must
remain zero for both screen and directional backgrounds.

CubeTextures use `CubeReflectionMapping` or `CubeRefractionMapping` as direction
lookups, not material refraction. Their camera contract remains non-reversed
perspective. Inverse background rotation is followed by the source's X-axis
lookup reflection for uploaded cubemaps. Camera translation is removed; face
order is not rewritten. Render-target, compressed and float cubes are not
silently converted into this byte/image profile.

The shared source texture pool validates all six faces before allocation,
preserves requested `flipY`, wrapping and filters, and either uploads authored
mips or generates each face's mip chain. Uncompressed cube `mipmaps` contain
additional levels (excluding the base); ordinary DataTexture `mipmaps` include
the base, matching pinned r186 storage conventions. Generated mips use the
existing native downsample profile, not PMREM or a promised seam-aware filter.
Source sRGB textures select an sRGB sampling view; there is no guessed gamma
conversion. Implicit derivatives choose mip levels with the source sampler.

All image readiness, fixed/unshared byte storage, supported browser-image types,
color and sampler restrictions of `three_textures.mjs` still apply. Each cube
face must be a DataTexture or all must be decoded images/canvases/ImageData.
Upload callbacks, ImageBitmap decode-policy inference and partial cube uploads
are not admitted. Pixel/source/version/sampler changes require `prepare()`;
UV and intensity edits do not. The background owns separate native residency
from material textures, even when both refer to the same source object.

## Preparation, replacement and lifetime

Changed source pixels require an acknowledged upload and explicit preparation:

```js
// Edit readyHdrTexture.image.data using the application's own workflow.
readyHdrTexture.needsUpdate = true;
await nativeScene.prepare();
nativeScene.render(perspectiveCamera, { colorView, depthView });

scene.background = anotherReadyHdrTexture;
await nativeScene.prepare();

scene.background = null; // Or a source Color; retire the former HDR owner.
await nativeScene.prepare();
```

Unrelated preparation reuses an unchanged background. Identity or acknowledged
version changes refuse stale rendering instead of silently displaying the old
texture or starting an upload in a frame. Unacknowledged in-place edits are not
scanned in render. A candidate that changes during asynchronous creation or
retirement waits is discarded before publication. Failed replacement keeps the
previous owner; restoring its valid source selection allows it to render.

`background.maxBytes` defaults to 128 MiB and charges the full-resolution
RGBA16F panorama, or every byte/image face and mip, plus the 128-byte uniform
packet. For cubes, maxPixels counts all six base faces together. Replacements also charge the
still-owned previous background until preceding consumers drain. `maxPixels`
defaults to 16,777,216; device texture limits also apply. Sizes are not silently
reduced. Optional `label` names native resources. Material-texture and filtered
lighting budgets are separate: selecting the same source for background and
lighting still owns separate raw and filtered GPU outputs.

`whenIdle()` includes background GPU completion and validation. Source disposal,
scene disposal, construction-signal lifetime abort and device loss invalidate
owned resources and promptly end pending waits. They do not dispose the
application's texture pixels, camera, device or geometry. A failure after a
successful background submission is not rollbackable; the scene becomes
terminal rather than presenting a partially submitted frame as retryable.

## Standalone native backgrounds and deployment

`createGpuAnimationBackground(device, gpuTexture, options)` is independent of
Three classes. It supports explicit `mapping: 'panorama' | 'cube' | 'screen'`,
filterable single-sample color inputs, output sample count 1 or 4, a borrowed
sampler override, and a bounded 128-byte frame packet. Panoramas use north-first
2:1 pixels; cubes use six square native WebGPU faces; screen UV (0,0) is bottom
left before the explicit six-value `uvTransform`.

`viewFormat` optionally selects a compatible native sampling view; its format
must already be permitted by the borrowed GPUTexture's creation descriptor.
`autoLod:true` selects implicit derivatives and sampler-driven mip selection.
Do not supply `frame.mipLevel` in that mode. The default remains explicit LOD,
including byte-identical default shader generation for all three mappings.

Panorama/cube frames provide column-major `directionFromClip` mapping clip
`(x,y,0.5,1)` to environment-space homogeneous directions. Camera translation
must be removed before composition. Frame intensity, mip level and transforms
are copied and validated before queue writes. Standalone 4x rendering may use
`resolveTarget`; a prefix for later scene rendering should omit it. The source
GPU texture, sampler, device and nonaliasing render attachments stay borrowed.
`createGpuThreeBackground` separately exposes ready source texture ownership and
capture/render operations; the scene bridge normally owns that coordination.

```js
buildAnimation(inputGltfOrGlb, freshOutputDirectory, {
  webgpu: true,
  background: true,
  threeScene: true,  // Source texture background owner and live scene adapter.
  environment: true, // Optional, independent IBL filtering/receiver feature.
});
```

`background: true` requires `webgpu: true`. It emits the native background module
and exports `createGpuAnimationBackground` / `AnimationBackgroundError` from
`gpu_playback.mjs`. Together with `threeScene`, it emits/exports the source
background owner and includes the static dependencies of the shared HDR helper.
Those imports do not run GPU operations or filter anything. IBL exports and
receiver dependencies still require `environment: true`; HDR URL decoding still
requires its existing explicit `hdr` option. Packages without backgrounds do
not acquire background modules. Output budgets account for all emitted bytes
before publishing the output directory, and the package remains relocatable.

Validation executes production frame packing, HDR conversion, source ownership,
scene/background commands, receiver composition and package generation. Native
GPU services, lower mesh/filter operations, Three classes and pose decoding are
recorded or fixture boundaries. These host tests do not establish native WGSL
pixels, retained-Three renderer equivalence or a measured performance result.

The byte/cube extension has focused command/lifetime/numerical coverage in
`three_texture_cube.test.mjs` and `three_background_textures.test.mjs`. Run with
`node --test tools/ingest/animation_background.test.mjs tools/ingest/three_texture_cube.test.mjs tools/ingest/three_background_textures.test.mjs`.
The original native background suite is unchanged. Source classes and GPU calls
are fixtures; these tests do not run HDR conversion, retained Three, native GPU
pixels or the full source-scene/package suite. The existing HDR integration test
loader also resolves the new shared texture import without replacing assertions.
