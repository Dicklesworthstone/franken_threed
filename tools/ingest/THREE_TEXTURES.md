# Source texture residency

`createGpuThreeTextures(device, {three})` realizes source-owned r186 textures on
an explicitly borrowed WebGPU device. It does not fetch image URLs, decode files,
replace a source image or install a frame loop. The owner exposes synchronous
`prepare(textures)`, `update(textures)`, `binding(texture)`, and `retain(textures)`,
as well as `whenIdle()` and `dispose()`. Source decoding/loading must finish and
`texture.needsUpdate = true` must be set before first admission.

```js
const owner = createGpuThreeTextures(device, {three: THREE});
owner.prepare([texture]);
const {view, sampler} = owner.binding(texture);
// Bind view/sampler to the existing renderer. Later, before the next draw:
texture.image.data[0] = 128;
texture.needsUpdate = true;
owner.update([texture]);
// Submit draws before uploading the next source version.
await owner.whenIdle();
owner.dispose(); // Does not dispose texture, its image, or device.
```

## Data and sampling

Unsigned-byte DataTexture R, RG and RGBA storage maps to native unorm formats.
RGBA sRGB textures use an unorm upload texture and sRGB sampling/render views;
the encoded source bytes are not gamma-converted on the CPU. Linear-sRGB and
NoColorSpace are sampled as linear numeric channels. Rows honor unpackAlignment
and optional flipY without mutating source arrays. Premultiplied DataTexture
bytes are not admitted: their conversion requires a separate explicit profile.

Already decoded HTMLImageElement, HTMLCanvasElement, OffscreenCanvas and sRGB
ImageData use WebGPU external-image copies with the requested flip and alpha
premultiplication. This is the browser-managed **sRGB external-copy profile**,
not an assertion of raw embedded-profile equivalence with every WebGL image
upload. Wide-gamut/raw decode requirements can keep using caller-owned bindings.
ImageBitmap flags are fixed at bitmap creation in WebGL; this owner refuses
ImageBitmap rather than guessing those original options. Video, float, depth,
cube/array/3D, render-target, external and compressed textures also retain their
separate paths. These are implementation boundaries, not removed project goals.

All six source minification filters, nearest/linear magnification, clamp/repeat/
mirrored-repeat and admitted anisotropy are mapped to native samplers. Mipmapped
filters require a complete pyramid. A nonmip filter clamps LOD to zero. Unsupported
anisotropic nearest minification is rejected rather than silently changed.

Authored DataTexture mip arrays include level zero and upload verbatim; source
generation must already be disabled. Generated pyramids use GPU render passes,
sampling each previous level into the next. sRGB mip filtering occurs through
sRGB views. The downsample profile is the same linear fullscreen pass used by
the existing glTF texture path; this is not a promise of byte-identical native
WebGL generateMipmap output, which needs native conformance measurements.

## Source history and lifetime

Texture objects sharing a Source and complete upload/sampling description share
one native allocation. Different transfer/filter/upload domains never alias.
A source-only version bump does not force a previously acknowledged texture
version to upload; setting texture.needsUpdate follows the normal source API.
Unrequested CPU changes remain GPU-stale. Each upload-domain resource tracks its
own source generation, preventing accidental cross-domain acknowledgment.

RGBA partial updates execute the pinned uploader's component-to-pixel rounding,
in-place row-local merging and range clearing. Row-crossing/unsupported ranges
are rejected before writes. Only requested source pixels are published; an
initial partial upload leaves the other native pixels zero-initialized. Mips
are regenerated from GPU state after a partial update, not stale CPU neighbors.

Source generation is acknowledged before onUpdate(texture); texture version is
acknowledged after it. Callback exceptions retain successful writes and source
acknowledgment, not a fictitious rollback or repeated callback on retry. Reentrant
updates/disposal are refused. The stricter automatic scene integration rejects
custom upload callbacks before using this explicit owner.

Sampling/storage/source identity changes require prepare(). New allocations are
charged before old ones retire, so the peak budget is bounded. Defaults are
128 MiB total texel/mip storage, 256 texture identities and 16 million pixels per
image. Temporary flipped row copies are at most one admitted base/mip image.
Driver/native object overhead is not reported as exact bytes. Source dispose
releases that identity's reference; siblings survive, and prepare can recreate
the disposed identity. retain() prunes unused ownership without source mutation.

Allocation/view/sampler/write/mipmap errors, rejected scopes, queue completion
failure and device loss terminate residency and release owned native resources.
Invalid inputs/budgets are recoverable. whenIdle observes cumulative native
validation and completion. Disposal ends a pending owner wait without claiming
to cancel work already submitted to the device.

## Evidence

`F3D_THREE_ROOT=/path/to/pinned/three node --test tools/ingest/three_textures.test.mjs`
executes the actual source Texture classes and complete retained WebGLTextures
uploader from commit `148ef33ecb6d2502ff796d4554abd1549c95d519`. Eighty deterministic
update frames compare byte histories, source range mutations and callback traces.
Other tests cover pooling, sampling, mips, external-copy descriptors, exact
budgets, native errors, disposal and pending completion. The device boundary is
a byte/command recorder: mip shaders and external-image conversions are not
executed in those tests. No native pixel parity or measured speedup is claimed.
