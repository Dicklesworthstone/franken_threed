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
