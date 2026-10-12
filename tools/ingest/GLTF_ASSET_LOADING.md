# Runtime glTF / GLB asset loading

`gltf_asset.mjs` loads complete glTF JSON or GLB 2.0 files and their binary buffers
without importing Node, initializing a renderer, or running an application loop.
It is the byte-loading layer for the existing model factories, not a substitute
for their geometry/material validation or a complete Three.js GLTFLoader.

```js
import { loadGltfAsset } from './gltf_asset.mjs';
import { createCpuGltfAnimationModel } from './animation_model.mjs';

const asset = await loadGltfAsset('https://example.test/models/robot.glb', {
  signal: abortController.signal,
  maxBytes: 128 * 1024 * 1024,
  maxResourceBytes: 64 * 1024 * 1024,
});
// For an untextured model; textured models also need resolveTexture.
const model = createCpuGltfAnimationModel(asset.json, asset.buffers);
model.sample(0.5, { clip: 0 });
model.dispose();
```

The source can instead be an ArrayBuffer or a typed byte view containing a whole
JSON/GLB file. Supply an absolute HTTP(S) `baseURL` when those bytes refer to
external files. Root URLs may be relative when `baseURL` is given. Raw JSON
strings are not interpreted as documents; encode them as UTF-8 bytes.
`parseGltfAsset(bytes)` provides a synchronous parser returning `{ json, bin }`.
It snapshots input bytes, validates GLB headers/chunk ordering/extents and UTF-8,
and ignores unknown chunks without confusing them with JSON or BIN storage.

The loader supports GLB BIN data, base64/percent-encoded data URIs and external
HTTP(S) buffers. Buffer order remains the original glTF index order. Declared
buffer lengths and BIN padding are checked. Identical resource URLs share their
bytes; fragments do not duplicate requests and query strings remain significant.

Images are deliberately lazy: `await asset.readImage(imageIndex)` returns
`{ bytes, mimeType }` for a PNG/JPEG URI or an embedded buffer-view image. Images
unused by the selected model need not be fetched. Concurrent requests for an
image share the same load. Image metadata is snapshotted at load time. Image
bytes are not decoded, flipped, color-converted or uploaded by this module.
Do not modify the returned JSON/binary data during model construction.

## Limits and I/O ownership

Pass `fetch` to use an application's Fetch implementation. Otherwise the global
Fetch API is used only when loading is explicitly requested. Responses must
provide streaming bodies: there is no unbounded `arrayBuffer()` fallback.
`maxResourceBytes`, `maxBytes` and `maxResources` limit each encoded resource,
aggregate unique loaded bytes (including decoded data-URI bytes), and resource
counts. Streaming reads enforce actual byte limits even when Content-Length is
missing or incorrect, and cancel the reader on failure. These are input limits,
not hard process-memory limits: parsing/copying has overhead and decoded geometry,
images and GPU textures require their own allocation budgets.

Dependencies are same-origin by default. List extra exact origins with
`allowedOrigins: ['https://assets.example.test']` to authorize a CDN. All network
requests use `credentials: 'omit'` and `redirect: 'error'`; URL credentials,
non-HTTP(S) schemes and unexpected redirected responses are rejected. A supplied
Fetch implementation must honor these options. This is not an SSRF sandbox or a
filesystem access API. No cookies, authorization headers or implicit redirects
are added. The supplied AbortSignal remains effective for later lazy image loads.
No automatic retry, polling, background fetch or new task scheduler is introduced.

Tests use actual Node Fetch Response/ReadableStream primitives and injected
transport responses. They establish parser/loading behavior, not network-service
availability, native image decoding, GPU execution, pixel parity or acceleration.

## Load an animated WebGPU model directly

`gltf_scene_loader.mjs` connects this asset loader to model preflight, native image
uploads and the existing animated WebGPU scene. For supported models, application
code no longer has to load each binary buffer, decode images, create samplers,
generate mipmaps, or supply a texture-resolution callback.

```js
import { loadGpuGltfAnimationScene } from './gltf_scene_loader.mjs';

// device is an existing GPUDevice. This loader never creates or destroys it.
const model = await loadGpuGltfAnimationScene(device, modelURL, {
  signal: abortController.signal,
  assets: {
    maxBytes: 128 * 1024 * 1024,
    maxResourceBytes: 64 * 1024 * 1024,
    // allowedOrigins: ['https://assets.example.test'],
  },
  decode: { scene: 0 },
  textures: {
    maxTextureBytes: 256 * 1024 * 1024,
    maxImagePixels: 16 * 1024 * 1024,
  },
  scene: {
    renderer: { format: 'rgba8unorm-srgb', depthFormat: 'depth24plus' },
  },
});

// For a model containing an animation clip; static models start at rest.
model.controller.createAction(0).play();
// Within the application's existing frame loop:
model.update(deltaSeconds);
model.render({ colorView, depthView, viewProjection, lighting });
// When completion is required, not necessarily after every frame:
await model.whenIdle();
// During application teardown:
model.dispose();
```

Whole JSON/GLB bytes are also accepted; embedded GLB geometry and images need no
network requests. The loader preflights the selected geometry/material route
before requesting any images or creating textures. It decodes geometry once,
loads only the material images it needs, waits for texture work to complete, and
then initializes the existing scene. It adds no renderer, frame loop, clock,
background polling or independent task scheduler. Source node/mesh/primitive IDs
and the existing scene's controller, update/upload/render/whenIdle API are retained.
`assetBytes`, `textureBytes` and `bufferBytes` expose the separate storage counts.

The returned model owns its newly created pose, scene buffers, uploaded textures
and decoded camera/light metadata. It does not own the device, render attachments
or caller-supplied external cameras and lighting.
Initialization/upload failures and cooperative construction cancellation release
all owned resources. Cancellation during image decoding rejects promptly; a late
ImageBitmap or raw decoder result is closed when it arrives, without uploading.
Scene initialization remains cooperative: its late result is disposed before
construction rejects. Recoverable frame errors leave resources usable;
terminal scene/texture failures release them. Device loss immediately releases
textures and rejects a pending image decode; after publication it is surfaced on
the next model operation. `dispose()` is idempotent,
and a scene's rejection of reentrant disposal does not prematurely free textures.
The construction signal does not automatically dispose an already returned model.

## Render through an authored camera

Loaded GPU models expose `cameras`, `lights`, `view` and `renderCamera`:

```js
// For an asset with a selected-scene camera; choose its original node ID.
const cameraNode = model.cameras[0].node;
model.update(deltaSeconds);
model.renderCamera({ colorView, depthView }, {
  cameraNode,
  aspectRatio: width / height,
});
```

This uses the current animated camera pose and all selected-scene punctual lights
without manually constructing projection matrices or copying light transforms.
Perspective (finite/infinite far) and orthographic projections use WebGPU depth.
`aspectRatio` supplies a viewport ratio only when the camera does not author one;
match or letterbox the render target for fixed-aspect/orthographic cameras.
Models with multiple cameras require explicit `cameraNode` selection. Models
without cameras continue to use `render(frame)` with an external view. To use
imported lights with an external camera, call `model.view.sampleLights()` and
supply that array in the external frame's lighting descriptor.

`renderCamera` supplies both `viewProjection` and `lighting`, and rejects frames
that also specify either field rather than silently overriding them. After direct
`model.pose.sample(...)` or `blend(...)`, call `model.upload()` before rendering;
stale camera/mesh poses are refused. Source camera/light metadata is validated
before image requests or GPU allocation. The CPU model factory exposes the same
`view`, `cameras` and `lights` metadata without a GPU or render method. See
GLTF_SCENE_VIEW.md for the transformation convention, limits and test boundaries.

## Native and raw images, sampling and mipmaps

`createGltfTextureResources(device, requests, readImage, options)` is also available
from `gltf_textures.mjs`. It returns a synchronous `resolveTexture(request)` callback
and owns its uploads until `dispose()`. By default it accepts a `createImageBitmap`
override or uses the global browser decoder. An explicit `decodeImage` callback
instead supplies raw RGBA8 for hosts without browser image APIs; see below.
Untextured models do not need an image decoder or texture uploads.

PNG IHDR and JPEG frame dimensions are checked against `maxImagePixels` and the
device texture dimension limit **before native decoding**. All color-space copies
and mip levels are charged against `maxTextureBytes` before decoding that image.
This is RGBA8 texel accounting, not a hard browser/driver memory limit; decoder and
driver overhead are separate. The header reader is not a complete image validator.
The native decoder must agree with the checked dimensions.

Images request no EXIF orientation, no alpha premultiplication and no color-space
conversion from ImageBitmap; uploads use no Y flip and straight alpha. Base-color
and emissive maps use sRGB views; normal and metallic-roughness maps use linear
views. Image pixels are shared only within the same color space, while sampler
objects retain source wrap and filtering choices. An image used in both color
spaces is decoded once but has separate mip storage: linear-channel and sRGB
filtering must not accidentally share the same prefiltered chain.

All six glTF minification filter choices are realized. Nonmip samplers clamp LOD
to zero, including when they share image pixels with a mipmapped sampler. Missing
source filters use the explicit default policy of linear magnification and
trilinear minification; override `defaultMagFilter`/`defaultMinFilter` in texture
options to change only unspecified fields. Required mipmaps are generated with
successive bilinear WebGPU render passes, using sRGB views/attachments for sRGB
images so filtering occurs in linear space. This is not a claim of matching every
source backend's mip-generation algorithm, particularly for non-power-of-two sizes.
GPU error scopes are popped before awaiting: no scopes are held across image,
network or application callbacks on a shared borrowed device.

### Explicit PNG/JPEG decoders for headless hosts

Supply the callback as `textures.decodeImage` to the owning scene loader, or as
`options.decodeImage` to `createGltfTextureResources`. The scene loader snapshots
this option before its first I/O await and forwards it with the construction
signal; no global decoder registration or browser polyfill is needed.

```js
import { loadGpuGltfAnimationScene } from './gltf_scene_loader.mjs';

// decodePngOrJpegRgba is an application-owned codec adapter implementing the
// contract below. This library does not install a PNG/JPEG codec dependency.
const model = await loadGpuGltfAnimationScene(device, modelBytes, {
  signal: abortController.signal,
  textures: {
    decodeImage: decodePngOrJpegRgba,
    maxImagePixels: 16 * 1024 * 1024,
    maxTextureBytes: 256 * 1024 * 1024,
  },
});
```

The callback receives one frozen descriptor:

```js
{
  bytes,                 // borrowed encoded Uint8Array; do not mutate/detach
  mimeType,              // 'image/png' or 'image/jpeg'
  width, height,         // dimensions already checked against header/budgets
  signal,                // aborted by caller cancellation or device loss
  imageOrientation: 'none',
  premultiplyAlpha: 'none',
  colorSpaceConversion: 'none',
}
```

Return, synchronously or asynchronously, `{width, height, data, bytesPerRow?, close?}`.
Dimensions must be the actual decoded dimensions, matching the header. `data` must
be RGBA8 in a fixed, unshared, attached `Uint8Array` or `Uint8ClampedArray`, including
an offset view. RGB values remain in the file's encoded channel domain and alpha
is straight, not premultiplied. Do not apply EXIF orientation, flip rows, convert
RGB to linear light or perform display color management. The callback is
responsible for honoring those flags; the adapter cannot infer transformations
from pixel values. An explicitly reported `flipY:true`,
`premultipliedAlpha:true`, or non-`rgba8unorm` format is rejected.

`bytesPerRow` defaults to `width * 4`. Padded rows are accepted when the safe
integer stride is at least that width and the last row fits within the byte view.
The adapter copies only pixel bytes into a tightly packed owned snapshot, then
calls an optional synchronous `close()` exactly once. A codec may free or overwrite
its scratch storage during `close()`. Without `close()`, it must not reuse the
returned storage until construction completes. Shared, resizable, detached,
nonbyte and truncated storage is rejected before GPU allocation for that image.

Raw uploads use `queue.writeTexture`, not `copyExternalImageToTexture`, and do not
need `ImageBitmap`, `Blob`, a DOM, or a canvas. Unorm storage preserves encoded RGB
and alpha; the same material/mipmap view rules above select sRGB versus linear
interpretation. One source image is decoded once even when several samplers or
both color spaces use it. Snapshot storage and codec scratch storage can coexist;
`maxImagePixels` bounds the tight snapshot, not arbitrary codec process memory.

Supplying `decodeImage` is an explicit route choice. Its errors are preserved and
never retried with ImageBitmap; `null` or another nonfunction is not a decoder.
Caller abort or device loss rejects a pending decode promptly and releases prior
owned uploads. A noncooperative codec may keep working; its eventual result is
closed and never uploaded, and a late rejection is observed. The adapter does
not terminate the caller's workers or destroy its borrowed GPU device.

A configured `textures.ktx2Loader` can coexist with `textures.decodeImage`.
BasisU KTX2 sources still use the retained loader, never the PNG/JPEG callback;
authored compressed mip chains are not regenerated. Both routes contribute to
the aggregate texture budget, and a failure in either rolls back earlier uploads.
Browser decoding remains the default when `decodeImage` is omitted.

## Staged preparation and existing APIs

`prepareGltfAnimationModel(json, buffers, decodeOptions)` in `animation_model.mjs`
returns frozen unique `textureRequests` and a single-use-on-success
`resolveTextures(resolver)` method. It exposes no unresolved or fake drawables.
A failed resolver can be retried without rereading geometry; caller-owned resolver
effects are not rolled back. The original synchronous `decodeGltfAnimationModel`
and CPU factory use the same implementation and retain their original contracts.
`createGpuDecodedAnimationScene` in `animation_model_gpu.mjs` consumes that resolved
output without repeating the decode. That lower-level factory still borrows
textures; use `createGpuGltfAnimationScene` for existing caller-managed textures.

The new owning loader inherits the model/renderer route's explicit restrictions:
triangle geometry, supported core direct-light/unlit materials, one shared UV
set/transform across maps, source tangents for normal maps, and normals for
deforming geometry. Unsupported codecs/extensions and occlusion still require
the source loader/backend. Authored cameras and up to eight KHR_lights_punctual
instances are supported; see GLTF_SCENE_VIEW.md. Environment lighting, shadows,
tone mapping, automatic transparency sorting and complete glTF/Three.js parity
remain outside this route. HTTP loading still uses the credential-free origin policy above.

Run the focused host checks with:

```sh
node --test tools/ingest/gltf_asset.test.mjs \
  tools/ingest/gltf_image_decode.test.mjs \
  tools/ingest/gltf_textures.test.mjs \
  tools/ingest/gltf_raw_ktx2.test.mjs \
  tools/ingest/gltf_scene_loader.test.mjs \
  tools/ingest/animation_model.test.mjs
```

The integration tests execute actual asset/accessor/material decoding, pose
sampling and CPU deformation. GPU-scene construction, GPU devices and ImageBitmap
are explicit test doubles; assertions cover upload/mipmap commands, option mapping,
ownership and failures, not native decoding, compiled shader execution or pixels.
Raw/BasisU tests likewise use explicit codec/queue doubles and synthetic KTX2
containers; they verify routing, byte snapshots, budgets and lifetime contracts.
Native browser verification was blocked by the execution environment's browser
policy during this implementation. No GPU performance or image-parity claim is made.
