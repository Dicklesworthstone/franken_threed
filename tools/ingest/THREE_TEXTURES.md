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
ImageBitmap flags are fixed at bitmap creation in WebGL (r186 skips the unpack
flip/premultiply state for bitmaps) while the r186 WebGPU backend applies them at
copy time. This owner admits ImageBitmap only with `flipY=false` and
`premultiplyAlpha=false`, the profile where both renderers agree (GLTFLoader's
ImageBitmapLoader path); other flags are refused rather than guessing which
renderer's behavior the application expected. Video, float, depth,
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

The native WebGPU profile follows a different pinned DataTexture rule: a
requested version uploads the complete 2D image and leaves WebGL `updateRanges`
untouched. Its ordinary texture copies use that same accepted upload history.

Source generation is acknowledged before onUpdate(texture); texture version is
acknowledged after it. Callback exceptions retain successful writes and source
acknowledgment, not a fictitious rollback or repeated callback on retry. Reentrant
updates/disposal are refused. The stricter automatic scene integration rejects
custom upload callbacks before using this explicit owner.

Automatic WebGL-authored scenes enable `webglVersioning: true` on this owner.
For ordinary textures and frame-callback video, that profile retains each
Texture's accepted source, storage and sampler description until its
Texture.version changes. Versionless wrap/filter, image, Source, dimensions or
mip edits remain GPU-stale across inspection, preparation and draws. A requested
`needsUpdate` revalidates the current description, and disposal followed by
preparation admits current source state again. Source aliases retain shared
pixels and independent sampler histories. Cached descriptions contain metadata
without retaining obsolete image or mip upload arrays.

A WebGL video using `requestVideoFrameCallback` remains on the automatic default
binding until its first source version request, from a frame callback or an
explicit `needsUpdate`. Subsequent unrequested media and sampler changes retain
the accepted frame. Cache reuse checks the current image and the built-in
update hook: replacement with fallback media still invokes the source's
existing version-requesting hook after admission, and switching back retains
the latest accepted fallback frame. The owner installs no media callbacks and
does not control playback. The default/native video profile is unchanged.

Without that profile, sampling/storage/source identity changes require
prepare(). New allocations are charged before old ones retire, so the peak
budget is bounded. Defaults are
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

## Native sampled texture copies

Native `renderer.copyTextureToTexture()` accepts ordinary sampled textures as
destinations for matching, uncompressed base-level 2D copies. DataTexture and
ready decoded-image destinations retain GPU patches without changing their CPU
pixels or public versions. The first write detaches shared Source storage once;
other Texture aliases keep their accepted pixels. Later patches reuse the
residency and encode only the requested rectangle, with no CPU readback or
per-copy submitted-work fence. Public `waitForGPU()` retains validation and
completion responsibility.

Ready ordinary 2D material maps select the existing native sampled owner from
their first draw, so a future copy preserves uploaded pixels even after
unrequested CPU edits or Source replacement. Scene companions share that
capture path; scenes without these maps keep their existing selection. Static
bundles observe accepted GPU writes while preserving their sampler bindings.
New sampler bindings observe current parameters; existing bindings update when
their Texture version requests it. Filter edits preserve a program's constructed
texture-load or sampling choice until material invalidation; incompatible graph
and layout changes still trigger the existing checks. Decoded-image capture
waits for immutable bitmap pixels before allocating or uploading storage.

Ordinary destinations with mip chains, compressed/layered/cube storage, media
lifetimes, or the WebGL-authored renderer surface remain outside this copy
profile. Run `node tests/e2e/webgpu_renderer/native_sampled_copy.mjs` for pinned
source comparisons covering data, images, aliases, bundles and prior uploads.

## Evidence

`F3D_THREE_ROOT=/path/to/pinned/three node --test tools/ingest/three_textures.test.mjs`
executes the actual source Texture classes and complete retained WebGLTextures
uploader from commit `148ef33ecb6d2502ff796d4554abd1549c95d519`. Eighty deterministic
update frames compare byte histories, source range mutations, callback traces
and requested/unrequested wrapping and magnification changes. Automatic binding
regressions also cover shared Source uploads, source/mip replacement, disposal
and callback acknowledgment. A second differential executes the actual pinned
VideoTexture and WebGLTextures to compare first-frame admission, upload and
sampler histories, explicit source replacement and fallback-media transitions.
Its media boundary records submission parameters; browser decoding and frame
pixels are not executed. Other tests cover pooling, sampling, mips, external-copy descriptors, exact
budgets, native errors, disposal and pending completion. The device boundary is
a byte/command recorder: mip shaders and external-image conversions are not
executed in those tests. No native pixel parity or measured speedup is claimed.
