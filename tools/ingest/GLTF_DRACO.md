# Runtime Draco geometry preparation

`decodeDracoMeshes` in `gltf_draco.mjs` decodes `KHR_draco_mesh_compression`
primitives into ordinary glTF accessor storage for the existing model factories.
It accepts a configured, caller-owned Three r186 `DRACOLoader` via its
`decodeGeometry` interface. It does not implement a new codec, instantiate Wasm,
spawn workers, fetch decoder code, or start an animation loop.

```js
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { decodeDracoMeshes } from './gltf_draco.mjs';
import { createCpuGltfAnimationModel } from './animation_model.mjs';

const dracoLoader = new DRACOLoader();
// Configure local decoder-code paths as required by your deployment.
const prepared = await decodeDracoMeshes(json, buffers, {
  decoder: dracoLoader,
  signal,
  maxEncodedBytes: 64 * 1024 * 1024,
  maxDecodedBytes: 128 * 1024 * 1024,
  maxDecodedBufferBytes: 64 * 1024 * 1024,
});
const model = createCpuGltfAnimationModel(prepared.json, prepared.buffers);
// Textured models also supply the existing resolveTexture callback.
// Keep/reuse dracoLoader until all its jobs have drained, then dispose it.
```

## Direct URL/GLB loading

The same decoder is now accepted by `loadGltfAsset`, including its existing
streamed HTTP(S), data-URI and embedded GLB routes:

```js
import { loadGltfAsset } from './gltf_asset.mjs';
const asset = await loadGltfAsset(modelURL, {
  dracoDecoder: dracoLoader,
  signal,
  maxDecodedBytes: 128 * 1024 * 1024,
});
```

The owning GPU model loader forwards it through the existing `assets` settings:

```js
import { loadGpuGltfAnimationScene } from './gltf_scene_loader.mjs';
const model = await loadGpuGltfAnimationScene(device, modelURL, {
  assets: { dracoDecoder: dracoLoader },
  signal,
  picking: true,
  exporting: true,
});
```

No manual geometry/accessor assembly is required between loading and model
construction. Existing material, scene-view, deformation and rendering restrictions
still apply. The model does not own or dispose the borrowed Draco loader.

Meshopt and Draco can coexist, even when a meshopt view contains a Draco
bitstream. Meshopt view decoding precedes Draco geometry decoding, with metadata
rebound to the resolved views. `maxDecodedBytes` is ONE shared output budget, not
an independent allowance for each codec. All known allocations are checked before
dependency I/O; undeclared generated indices are checked before copying them.
`decodedBytes` reports the sum, `decodedBufferViews` retains its meshopt count,
and `decodedPrimitives` reports the Draco primitive count. `bytesLoaded` remains
the encoded-input count. `sourceJson` retains both original codec descriptions.

The loader skips proven-unused Draco fallback files when decoding, or the
compressed file when using optional fallback. Shared accessor, morph, skin,
animation, sparse and image references keep their storage live. Opaque extensions
conservatively disable Draco-specific pruning; meshopt owns its own buffer
selection. No declared source storage is synthesized. Skipped original buffer
slots are explicitly `null`, and unused original accessor descriptions may still
reference them. The normalized primitive references use real appended storage.

Existing credential-free asset fetching, origin allowlists, streaming input
limits, cancellation and lazy-image caching remain in force. **Decoder code and
worker requests belong to DRACOLoader's separate configuration**, not the asset
Fetch policy. Configure/package those resources for your deployment; this adapter
does not silently put decoder-code requests under an asset-origin guarantee.

## Data and ownership

The adapter passes unique Draco attribute IDs and the accessor's requested typed
array classes to the decoder. Generated internal attribute names avoid color
conversion or interpretation of arbitrary source-controlled property names.
Decoded attribute storage is read directly, including interleaved padded storage;
normalized getters are not used. The original accessor's normalization flag is
preserved for the existing accessor reader. Data is written explicitly little
endian, with four-byte vertex alignment and separately padded index storage.

Decoded attribute widths/counts and index counts/ranges must agree with glTF
metadata. Floating-point attributes must be finite. Foreign output type/extent
mismatches and invalid indices fail before publication. Node/mesh/primitive IDs,
materials, unrelated attributes, morph targets, source transforms and unrelated
extensions are not replaced. Unsupported material or other extension semantics
remain subject to the existing model's validation.

Every compressed primitive receives private appended accessors, bufferViews and
byte arrays. Only its references are remapped. Original accessors remain intact
for other primitives that may share them, and separate decode results cannot
alias a reused decoder output arena. Original JSON/bytes are unchanged;
`sourceJson` retains the original description. The normalized in-memory result
is not a newly packaged file: original buffer declarations and unused fallback
accessors remain present. Use the posed exporter for a self-contained snapshot.

Each returned temporary geometry is disposed after copying, including failures
and late cancellation results. The decoder itself, its workers and code resources
remain caller-owned. Cleanup errors do not replace the original decoding error.

## Limits and cancellation

`maxEncodedBytes` charges every copied compressed range; aliases count separately.
`maxDecodedBytes` bounds total owned output; `maxDecodedBufferBytes` bounds one
attribute/index allocation. Padding is charged. `maxPrimitives` bounds compressed
primitive count (default 4096). Known accessor allocations are checked before
calling the decoder. An undeclared index stream is checked once its count is
available, before copying. Third-party decoding can allocate additional internal
memory; these limits are not a hard process-memory or worker-memory sandbox.

All compressed input ranges and metadata are snapshotted before asynchronous
work. Decoding proceeds sequentially, with no new executor or unstructured task
pool. Cancellation rejects the wait and prevents publication, observing late
foreign rejection and disposing late geometry. It cannot interrupt a running
synchronous/native codec or terminate the borrowed worker pool. A failed later
primitive does not publish a partial asset or rewrite source data.

`prepareDracoMeshes` exposes a one-shot `decode(buffers, {signal})` plan for
loaders that need metadata/capability/budget preflight before resource I/O.
Its `skippedBuffers` identifies proven-unused source storage for loaders.
Its `decodedBytes` is the reservation for declared accessor output; generated
indices, where no source index accessor exists, are additional checked output.

## Scope and verification

This adapter consumes DRACOLoader's **triangle-list** output for source
`TRIANGLES` primitives. Compressed `TRIANGLE_STRIP` requires a topology-aware
adapter and is explicitly refused rather than guessed. Ordinary uncompressed
geometry retains its existing topology paths. Optional Draco without a decoder
uses backed or sparse source fallback accessors; point-cloud-like decoder
outputs without connectivity are refused. Required compression without a
decoder fails. No missing fallback is turned into zero-filled geometry.

The tests exercise the real adapter against an explicit recording decoder and
BufferGeometry-shaped outputs. They check typed requests, layouts, ownership,
shared-accessor isolation, failure and cancellation behavior. They do not decode
Draco bitstreams, run a native/Wasm decoder, render pixels, establish full
GLTFLoader parity or measure a speedup. Asset integration tests execute the
production asset, meshopt and Draco adapters with explicit codec doubles and
genuine Node Response/ReadableStream primitives; they do not contact real HTTP
services, decode bitstreams or construct a GPU scene. The unchanged asset-loader
regression suite runs alongside them.

```sh
node --test tools/ingest/gltf_draco*.test.mjs tools/ingest/gltf_asset.test.mjs
```

Interface reference: Three r186 DRACOLoader, source commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`, blob
`413798f963908b169422eb2d40ea528ce1eacee5`.
Format reference:
https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_draco_mesh_compression
