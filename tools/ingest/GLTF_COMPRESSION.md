# Quantized and compressed glTF inputs

## Quantized geometry

The existing `decodeGltfGeometry`, `decodeGltfAnimationModel`, CPU/GPU model
factories and owning asset-to-scene loader accept `KHR_mesh_quantization` for
positions, normals, tangents, UV sets and position/normal/tangent morph deltas.
The asset must declare the extension in `extensionsRequired`. Quantization does
not change the existing restrictions on material extensions or primitive types.

Positions and UVs accept signed/unsigned 8-bit and 16-bit components, normalized
or literal. Normal/tangent components must be signed and normalized. Position
morph deltas must be signed; normal/tangent deltas must also be normalized.
Core unsigned-normalized UVs, colors and skin weights remain valid without the
extension. Integer colors, joints and weights do not gain new formats implicitly.

The existing accessor reader supplies normalized Float64 values, including sparse
overlays and interleaved/strided storage. No new f32 rounding or source mutation
is introduced at decoding. Node transforms, inverse-bind matrices and
`KHR_texture_transform` already carry any required dequantization; the decoder
neither guesses scales from accessor bounds nor applies these transforms twice.
Quantized normals/tangents need not be exactly unit length; normalization remains
at the renderer's existing lighting boundary. Generated flat normals, independent
instances and all admitted skin influences keep their existing paths.

Encoded input size is not the expanded geometry budget. Existing accessor and
output component budgets continue to charge the decoded arrays and any triangle
expansion. Deformation and GPU allocation retain their separate limits. This is
JavaScript data decoding, not a retained renderer submission or a Rust rewrite.
UV/color morph animation is not added by this change; the existing unsupported
morph-field errors remain explicit. No additional quantization is applied.

Run the focused checks:

```sh
node --test tools/ingest/animation_quantization*.test.mjs
```

These tests exercise binary accessor decoding, geometry conversion, animation
metadata and material preflight. They are not native GPU pixel or performance
measurements. The format rules follow the Khronos KHR_mesh_quantization spec:
https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_mesh_quantization/README.md

## Runtime meshopt loading

`loadGltfAsset` handles `EXT_meshopt_compression` and `KHR_meshopt_compression`
when supplied with a compatible retained `MeshoptDecoder`. Both synchronous
`decodeGltfBuffer` and asynchronous `decodeGltfBufferAsync` implementations are
supported after `decoder.ready`. The Three r186 decoder is meshoptimizer 1.1.
Supply that component explicitly, as with Three's GLTFLoader; the F3D adapter
neither implements another codec nor silently fetches one from a CDN.

```js
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { loadGltfAsset } from './gltf_asset.mjs';
import { createCpuGltfAnimationModel } from './animation_model.mjs';

const asset = await loadGltfAsset(url, {
  signal,
  meshoptDecoder: MeshoptDecoder,
  maxDecodedBytes: 128 * 1024 * 1024,
  maxDecodedBufferBytes: 64 * 1024 * 1024,
});
// For an untextured model; textured CPU models still supply resolveTexture.
const model = createCpuGltfAnimationModel(asset.json, asset.buffers);
```

The existing owning GPU loader passes its `assets` settings through unchanged:

```js
const model = await loadGpuGltfAnimationScene(device, url, {
  signal,
  assets: {
    meshoptDecoder: MeshoptDecoder,
    maxDecodedBytes: 128 * 1024 * 1024,
    maxDecodedBufferBytes: 64 * 1024 * 1024,
  },
  // Existing scene, textures, picking and exporting options retain their APIs.
});
```

Previously unseen compressed files are decoded when loaded, not restricted to
build-time preconverted models. Uncompressed assets do not inspect decoder
readiness or invoke its methods. Applications that do not need compression can
omit the decoder import entirely. Retained decoder code/Wasm is separately owned;
this is not a Rust codec, an additional F3D task executor or a performance claim.
The adapter does not spawn workers or instantiate Wasm; the supplied decoder may
own those mechanisms under its own contract.

### Modes, filters and fallback behavior

Attribute, triangle-index and index-sequence modes preserve the decoder's exact
output order. EXT accepts attribute stream v0; KHR additionally accepts v1.
Both support NONE, OCTAHEDRAL, QUATERNION and EXPONENTIAL filters. KHR additionally
supports COLOR. Triangle/index modes require their specified v1 bitstreams and
2- or 4-byte elements; filtering is restricted to attribute mode. Invalid
count/stride/filter combinations, ranges, headers and incorrect decoder output
sizes fail rather than producing zero-filled geometry. Payload decoding and
bitstream validation beyond the header belong to the supplied decoder.

With a decoder, only needed compressed/core buffers are fetched; unused fallback
URIs and placeholder allocations are skipped. This also works for GLB BIN data.
Without a decoder, an optional meshopt extension uses the actual uncompressed
fallback. A required compressed view, unavailable required decoder, invalid
layout or excessive decoded allocation is rejected before dependency I/O.
Failure of an available decoder is an error, not silent substitution of a
possibly different fallback or a retry that replays callbacks.

### Decoded representation and memory

`asset.json` is a private, normalized in-memory glTF description. Decoded views
point at appended owned buffers; original accessor and bufferView indices,
attribute offsets, source nodes, scene IDs and material references do not change.
Handled compression declarations are removed from this description, while other
extensions and metadata remain. `asset.sourceJson` preserves the original
compression description. Original buffer indices are retained; intentionally
skipped slots in `asset.buffers` are `null`, not synthetic allocations. No view in
the normalized description uses those absent slots. These values are decoded
loader state, not a ready-to-serialize replacement source file; the original
buffer declarations may still describe skipped fallback/compressed resources.
Use the existing posed-GLB exporter when a self-contained snapshot is required.

Compressed views can supply geometry, morphs, skins, animation accessors or image
bufferViews. Lazy image reads use the normalized view table. The existing
quantization, material, camera, deformation, picking and export routes then consume
ordinary decoded bytes; their other restrictions are unchanged. Draco, KTX2 and
unrelated extensions are not implemented by this adapter.

`bytesLoaded` remains the encoded input count. `decodedBytes` and
`decodedBufferViews` separately report decompression output. `maxDecodedBytes`
bounds the sum of decoded view sizes; `maxDecodedBufferBytes` bounds one view.
Aliases count separately because each view owns independent output storage.
The asset loader's existing `maxBytes`, `maxResourceBytes`, resource count,
streaming-read limits and credential-free origin policy remain in force.
Source-range snapshots, output copying, parsing, retained decoder scratch/Wasm
and native allocations add overhead; these limits are not hard process-memory
ceilings. Later decoded-component, texture and GPU budgets still apply.

For already-loaded JSON and buffers, `decodeMeshoptBuffers(json, buffers, {
 decoder, signal, maxEncodedBytes, maxDecodedBytes, maxDecodedBufferBytes
})` in `gltf_meshopt.mjs` returns the same normalized fields without doing I/O.
`prepareMeshoptBuffers` exposes a one-shot compressed-input plan and
`skippedBuffers` for loaders that need the fetch decisions first.

### Cancellation and test boundaries

Decoding proceeds sequentially, without an unstructured Promise pool. All
compressed source ranges and metadata are snapshotted before codec readiness or
asynchronous decoding can yield. Decoder output is copied before publication,
so reuse of a foreign output arena cannot overwrite a prior view. Failed later
views publish no partial result and do not rewrite the source JSON or bytes.

Abort signals reject readiness/async-decoding waits and observe late decoder
rejections. Cancellation prevents publication; it cannot interrupt an already
running synchronous/native codec or roll back its external effects. The caller
still owns that codec's lifecycle. No timeout or hard termination guarantee is
implied.

```sh
node --test tools/ingest/gltf_meshopt*.test.mjs \
  tools/ingest/animation_quantization*.test.mjs
```

Codec tests use the unmodified, MIT-licensed meshoptimizer 1.1 **JavaScript
reference decoder** with upstream golden compressed/decoded vectors, not the
native/Wasm decoder. It is retained only in `tests/fixtures/meshopt`; production
imports never reach it. Its blob is `d0686d25d08b261975d652e9c6cb9087a4779bc8`;
test vectors originate from `6151837440c50ac37934675ea205b0d1d21d3c8d`.
Malformed-payload/foreign-failure tests use explicit decoder doubles rather than
exposing the educational reference implementation to untrusted streams.
Transport tests use genuine Node Response/ReadableStream bodies with injected
Fetch responses. They prove loading, byte ownership and decoding, not real
HTTP services, native image decoding, Wasm decoder execution, GPU pixels,
complete Three.js compatibility or acceleration.

Format references:
https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_meshopt_compression
https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_meshopt_compression
