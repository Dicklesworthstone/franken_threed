# Still WebP textures

`EXT_texture_webp` uses the existing glTF asset, material, image and WebGPU upload
pipeline. No WebP codec, browser, worker, GPU device or I/O is created at import.

```js
const model = await loadGpuGltfAnimationScene(device, modelURL, {
  decode: { webp: true },
  // Browser: createImageBitmap is used with no orientation, alpha or color
  // conversion. In Node, supply textures.decodeImage with WebP support.
  sourceExport: true,
});
```

`decode.webp` is an explicit boolean capability, default `false`. Direct CPU or
prepared-model callers use `{ webp: true }`. Enabling it does not probe native
codec support: the caller must supply a WebP-capable decoder. Optional WebP
textures use their authored PNG/JPEG fallback when disabled. Required WebP is
rejected without support; a texture without a usable source is never replaced
with fabricated pixels. If both BasisU and WebP are enabled for one texture,
BasisU takes precedence, then WebP, then core. Failure of the selected source
never retries a different image or decoder.

Prepared texture requests include a frozen `imageMimeTypes` array. Core requests
allow PNG/JPEG only, even when a URI omits MIME metadata. Extension requests carry
both the selected MIME and an allowlist. The upload stage intersects every use of
an image before fetching, including duplicate requests, so sharing cannot weaken
another material's source contract. Borrowed custom texture resolvers must honor
these constraints as well as the sampler and color-space request.

## Decoding and ownership

Lossy VP8, lossless VP8L, and extended still VP8X with alpha or XMP are supported.
RIFF extents, chunk padding, dimensions, alpha declarations, and canvas/bitstream
agreement are checked before decoding. Animation, ICC profiles and EXIF are
rejected rather than silently choosing a frame, orientation or color conversion.
This is a bounded header/profile check, not validation of entropy-coded pixels;
the actual decoder must still reject malformed bitstreams.

The existing `textures.decodeImage` callback receives encoded bytes, MIME,
header-checked dimensions, an AbortSignal and the no-conversion contract. It
returns straight-alpha, unflipped RGBA8 in the encoded channel domain, optionally
with padded `bytesPerRow` and `close()`. The adapter snapshots rows before cleanup.
The default ImageBitmap path retains its bitmap only until uploads finish.
Cancellation/device loss closes late decoder results. Native decoders and devices
remain borrowed. Shared images decode once; sRGB and linear uses retain separate
mipmap domains. Existing image-pixel, texture-byte and prefetch limits apply.

## Authored export

`exportGltfAssetGLB` and the owning loader's `sourceExport` path preserve WebP
bytes, image indices, extension references, samplers, and authored PNG/JPEG
fallbacks. Source export closes every source image, not only the selected scene's
images, and requires no image codec. A WebP-only texture retains its required
extension declaration; WebP is never written as a core texture source. Conflicting
MIME requirements, undeclared WebP and hidden extension resources fail explicitly.

Source export checks encoded signatures and resource closure, not pixel validity.
A successful repack is not a claim that a renderer supports every source image's
profile. The runtime upload profile above remains deliberately narrower.

## Posed export

`exportAnimationPoseGLB` and model `exportPoseGLB` accept encoded `image/webp`
from the existing texture resolver. They preserve those bytes, independent map
UVs, material alpha, normal scales and sampler settings. The GLB uses an
`EXT_texture_webp` image reference and declares the extension both used and
required; it does not fabricate a PNG fallback or write WebP as a core source.

The same still-image header/profile checks used before uploading apply before
embedding WebP. This path does not decode, re-encode, read back GPU pixels, or
validate the entropy-coded bitstream. Existing GLB byte budgets and cancellation
remain in force. Posed export captures static world-space geometry and materials
before image resolution; it is not an editable rig/animation export. Use authored
export above to preserve the full source asset and its alternative image sources.

## Tests

```sh
node --test tools/ingest/gltf_webp.test.mjs \
  tools/ingest/gltf_webp_pipeline.test.mjs \
  tools/ingest/animation_pose_webp_export.test.mjs
```

The fixtures include native-encoded lossy, lossless, alpha and XMP WebP with
independently decoded expected RGBA. Tests exercise image headers, route selection,
MIME constraints, actual HTTP streams, source GLB round trips and GPU command/
cleanup boundaries. Posed-export tests independently inspect real emitted GLB
chunks, extension references, attributes, image bytes and cancellation behavior.
Recorded GPU commands do not establish native rendered-pixel or performance
equivalence.
