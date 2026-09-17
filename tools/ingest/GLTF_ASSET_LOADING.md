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

The returned model owns its newly created pose, scene buffers and uploaded
textures. It does not own the device, render attachments, camera or lighting.
Initialization/upload failures and cooperative construction cancellation release
all owned resources. A late ImageBitmap or scene initialization result is disposed
before cancellation rejects. Recoverable frame errors leave resources usable;
terminal scene/texture failures release them. Device loss immediately releases
textures and is surfaced on the next model operation. `dispose()` is idempotent,
and a scene's rejection of reentrant disposal does not prematurely free textures.
The construction signal does not automatically dispose an already returned model.

## Native images, sampling and mipmaps

`createGltfTextureResources(device, requests, readImage, options)` is also available
from `gltf_textures.mjs`. It returns a synchronous `resolveTexture(request)` callback
and owns its uploads until `dispose()`. It accepts a `createImageBitmap` override
for environments providing that API explicitly; otherwise it uses the global
native decoder. Untextured models do not need an image decoder or texture uploads.

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
the source loader/backend. It does not add environment lighting, camera/light
extraction, tone mapping, automatic transparency sorting or complete glTF/Three.js
compatibility. HTTP loading still uses the credential-free origin policy above.

Run the focused host checks with:

```sh
node --test tools/ingest/gltf_asset.test.mjs \
  tools/ingest/gltf_textures.test.mjs \
  tools/ingest/gltf_scene_loader.test.mjs \
  tools/ingest/animation_model.test.mjs
```

The integration tests execute actual asset/accessor/material decoding, pose
sampling and CPU deformation. GPU-scene construction, GPU devices and ImageBitmap
are explicit test doubles; assertions cover upload/mipmap commands, option mapping,
ownership and failures, not native decoding, compiled shader execution or pixels.
Native browser verification was blocked by the execution environment's browser
policy during this implementation. No GPU performance or image-parity claim is made.
