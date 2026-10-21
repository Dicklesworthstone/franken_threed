# Image-based lighting for animated materials

The existing environment filter now connects to the existing animated material
renderer. Lambert materials receive diffuse environment light; metallic-roughness
materials also receive roughness-dependent reflections. No second renderer,
animation clock or per-frame environment filtering is introduced. Optional HDR
file loading is available without an application-supplied image decoder.

```js
import { createGpuAnimationEnvironment } from './animation_environment.mjs';
import { createGpuAnimationScene } from './animation_scene.mjs';

// Lend a linear-sRGB rgba16float/rgba8unorm GPUTexture: either a 2:1 panorama
// or six square cube faces. The source and device remain caller-owned.
const environment = await createGpuAnimationEnvironment(device, hdrTexture, {
  size: 128,
  diffuseSize: 32,
  lutSize: 128,
  samples: 1024,
  maxTextureBytes: 64 * 1024 * 1024,
  maxSampleWork: 256 * 1024 * 1024,
});
const scene = await createGpuAnimationScene(device, pose, drawables, {
  renderer: { format: 'rgba16float', environment: true },
});

// Within the application's existing frame loop:
scene.update(deltaSeconds);
scene.render({
  colorView, depthView, viewProjection,
  lighting: { cameraPosition, lights: [] },
  environment: { map: environment, intensity: 1 },
});
await scene.whenIdle();
scene.dispose();
environment.dispose(); // Release only after its consumers have stopped.
```

`createGpuAnimationRenderer` accepts the same `environment: true` option and
`frame.environment` descriptor. Existing glTF model/loader scene options forward
`renderer.environment`; `renderCamera(frame, settings)` accepts the environment
in `frame` while supplying the authored camera and punctual lights itself.
For example, use `{scene: {renderer: {environment: true}}}` with
`loadGpuGltfAnimationScene`. The low-level factory above borrows an existing GPU
texture; the HDR loader below handles file decoding and upload instead.

## Loading HDR and EXR files

```js
import { loadGpuAnimationEnvironment } from './animation_environment_loader.mjs';

const environment = await loadGpuAnimationEnvironment(device, './studio.hdr', {
  baseURL: import.meta.url,
  size: 128,
  samples: 1024,
  maxInputBytes: 64 * 1024 * 1024,
  maxDecodedBytes: 256 * 1024 * 1024,
  maxTextureBytes: 128 * 1024 * 1024,
});
// Use with the same environment-enabled scene/renderer shown above:
// scene.render({...frame, environment: {map: environment, intensity: 1}});
```

`source` can instead be an `ArrayBuffer` or `Uint8Array` containing a complete
Radiance RGBE or supported scanline OpenEXR file. The loader decodes, uploads once, runs the existing IBL
preparation, waits for filtering completion and destroys the temporary panorama
before returning the ready receiver map. No application decoder, canvas, manual
texture upload or second frame loop is required. The caller owns the returned map
and must dispose it after its consumers stop; the GPU device remains borrowed.
`sourceInfo` reports dimensions, input bytes, exposure metadata, clamped component
count, upload texture bytes and peak logical texture bytes.

`format: 'auto'` selects EXR by its file signature, not the URL extension or HTTP
Content-Type. Use `format: 'hdr'` or `format: 'exr'` to require a specific format.
A failed selected decoder does not retry another one or fabricate a texture.
Both routes require complete **2:1 panoramas**. XYZE and unrelated image formats
remain outside this loader.

For Radiance RGBE, the decoder supports raw pixels, modern planar RLE and
legacy repeat packets. All eight scan-axis/sign orientations normalize to top-down
`-Y +X`. Pixels decode directly to tightly packed linear-sRGB RGBA16F with
nearest-even rounding; component scaling follows Three r186 HDRLoader. Exposure
header records are reported, not reapplied to already stored radiance. This is an
explicit linear-sRGB profile, not general Radiance colorimetry: nonunit gamma,
color correction or pixel aspect, and unsupported primaries, reject. Values above
65504 reject by default; `overflow: 'clamp'` explicitly permits saturation and
reports affected components. Neither tone mapping nor automatic exposure is added.

The standalone `decodeAnimationHdr(bytes, options)` also exports the owned
RGBA16F pixels without any GPU work. Its `maxDecodedBytes` bounds the output plus
one scanline of scratch storage; encoded bytes and header bytes have separate
limits. It accepts non-panorama dimensions for callers using decoded pixels directly.

### OpenEXR scanline profile

```js
const environment = await loadGpuAnimationEnvironment(device, './studio.exr', {
  baseURL: import.meta.url,
  format: 'exr', // Optional; the default detects the same magic bytes.
  size: 128,
  samples: 1024,
  maxInputBytes: 64 * 1024 * 1024,
  maxDecodedBytes: 256 * 1024 * 1024,
});
```

The decoder reads real single-part scanline EXR pixels with NONE, RLE, ZIPS
(one-row ZIP), ZIP (16-row blocks), PIZ (32-row wavelet/Huffman blocks), or
PXR24 (16-row delta/deflate blocks) compression. RGB(A) and luminance Y(A)
channels may independently use HALF, FLOAT or UINT. Finite HALF bits are preserved;
FLOAT and UINT narrow once to RGBA16F with ties-to-even rounding. Signed radiance
and alpha are retained, missing alpha becomes one, and Y is replicated to RGB.
Overflow rejects by default; `overflow: 'clamp'` explicitly saturates signed values
at 65504 and reports the affected output-component count. NaN/Infinity always
reject. No exposure multiplication, alpha conversion, tone mapping or resampling
is performed. The existing environment filter retains its own sampling profile.

PIZ reconstructs all source word bits losslessly, including the full 16-bit range
LUT and independent word planes for FLOAT/UINT. Its canonical Huffman decoder
supports code lengths through 58 bits without JavaScript integer truncation.
PXR24 preserves HALF/UINT words; compressed FLOAT samples contain the 24-bit
precision already chosen by the file encoder. This loader does not recompress or
introduce an additional PXR24 quality reduction. Raw fallback blocks retain their
full original FLOAT precision. Both codecs then use the same final RGBA16F
conversion as the other modes. No caller-provided codec is required.

This is an explicit linear-sRGB/Rec.709 profile. Other declared chromaticities,
nonunit gamma/pixel aspect, color-adaptation metadata, layered/subsampled channels,
tiled/deep/multipart images, and other codecs (including B44/B44A and DWA) reject.
They still require their retained source loader; this is not full EXR compatibility.
The environment entry additionally rejects cube-map files and cropped data windows
that differ from the display window. It does not reinterpret missing pixels or
stretch a cropped image into a panorama. Data-window origins and physical chunk
ordering do not change the top-down output order.

`animation_exr.mjs` exports `inspectAnimationExr(bytes, options)` for synchronous
header/chunk-range preflight and `decodeAnimationExr(bytes, options)` for an owned
RGBA16F image without GPU work. The latter also supports non-panorama dimensions.
`maxDecodedBytes` bounds the output plus two worst-case decoded blocks and,
for PIZ, bounded typed Huffman/range tables;
`maxInputBytes` independently bounds the encoded snapshot taken before the first
asynchronous operation. JavaScript metadata and native decompressor working
memory are not exact byte-budget claims. ZIP and PXR24 use the browser's
`DecompressionStream('deflate')`; streamed output cannot exceed the declared block.
Cancellation/device loss aborts pending decode/upload/filter construction without
publishing a partial map. Synchronous parsing/raw/RLE/PIZ work is not preemptible.

EXR `sourceInfo` adds `format`, `compression`, `channels`, `dataWindow`,
`displayWindow` and `peakDecodedBytes`. Existing RGBE metadata is unchanged.
Exposure headers are metadata, not instructions to scale the stored radiance.

HTTP(S) loading accepts an injected `fetch` and requires streaming response bytes.
Relative URLs require `baseURL`; embedded credentials and redirects reject, and
requests use `credentials: 'omit'`. Both declared length and actual streamed bytes
are bounded, including when Content-Length is absent or inaccurate. Assembly uses
bounded growing storage rather than retaining a list of chunks; during growth,
old and new input buffers can briefly coexist (at most twice `maxInputBytes`).
This is not an SSRF sandbox for a caller-provided transport.

Unlike the low-level factory's output-only texture budget, the loader's
`maxTextureBytes` counts the **peak panorama plus all filtered maps**. The temporary
panorama stays live until filtering finishes. Filter uniforms are separately
checked against device limits. `signal` cancels construction across fetch, reads,
upload validation, compilation and completion; device loss also aborts pending
work. Late results are cleaned up. Synchronous decoding is not preemptible, and
already submitted GPU work is not rolled back. Aborting the construction signal
after a successful return does not revoke the map; disposal and device loss do.

## Frame controls and composition

A frame descriptor is `{map, intensity?, rotation?}`. `map` is the completed
`createGpuAnimationEnvironment` or `loadGpuAnimationEnvironment` result on the
**same GPU device**. Intensity is a
nonnegative finite number (default 1). Rotation defaults to identity and is an
explicit column-major, right-handed **world-to-environment 3x3** orthonormal
matrix. It is not an Euler-angle vector or an environment-object world matrix;
invert an authored environment rotation before supplying it. Rotation and
intensity are copied into the frame's uniform, never applied to source textures.

Lit frames still supply either `lighting.cameraPosition` or orthographic
`lighting.viewDirection`. The punctual `lights` array may be empty. A nonempty
array adds direct lighting independently. Emission is added independently too:
changing environment intensity does not scale emission or direct lights.

Projected shadows and environment lighting can be used together, including
scene-owned automatic shadows. Shadows attenuate only their selected direct
light, not the indirect environment. Base-color, metallic-roughness, normal and
emissive maps retain their own samplers and UV coordinates; authored tangents and
derivative normal frames both feed the final mapped normal to IBL. Existing
OPAQUE, MASK and BLEND behavior and input draw order remain unchanged.

Omitting `frame.environment`, or setting it to `null`, uses the original
non-environment material pipelines even when the renderer was enabled for IBL.
Unlit meshes never receive environment lighting. The default renderer option is
`false`: it does not import the receiver module, compile IBL pipelines or allocate
an environment uniform. A depth-only renderer cannot enable IBL.

## Explicit sampling profile

The preparation stage retains its established cubemap orientation: face order
+X, -X, +Y, -Y, +Z, -Z; panoramas use `atan2(z,x)` horizontally and `acos(y)`
vertically. No flip or color conversion is guessed. Source radiance is linear.

The diffuse cube stores irradiance divided by pi, so Lambert multiplies it by
base color without dividing by pi a second time. The specular cube is GGX
prefiltered at perceptual roughness `level / (mipLevelCount - 1)`. The receiver
samples at `roughness * (mipLevelCount - 1)`, with the existing material roughness
floor of 0.045. The DFG lookup uses **(roughness, NdotV)** coordinates, matching
the producer's axis order. All IBL samples use explicit LOD, including after
alpha discard and nonuniform lighting control flow.

For metallic-roughness, this is the single-scattering split-sum profile:
`F0 = mix(0.04, base, metallic)`, `Fss = F0 * A + B`, and diffuse weight
`max(0, 1 - Fss) * (1 - metallic) * base`. A fully metallic surface therefore has
no diffuse contribution. Numerical overshoot in the lookup never subtracts
diffuse light. This is not Three.js PMREM/CubeUV pixel equivalence or a claim of
full PBR parity. Multiscattering compensation, material occlusion maps,
transmission, environment backgrounds and automatic exposure are not included.
Use the existing HDR presentation stage for tone mapping and display conversion.

## Ownership, budgets and failure behavior

Preparation waits for filtering completion before returning a stable immutable
snapshot. `sample(device)` verifies device identity and liveness. `whenIdle()` is
available to consumers. An AbortSignal belongs only to construction: aborting it
after successful preparation does not revoke the environment. Explicit disposal
and device loss do revoke it and release its owned textures.

The renderer borrows the environment. It owns one extra **64-byte uniform** once
a lit material is registered; scene `maxBytes` reserves this before allocating
subsequent deformers. Environment texture storage is separately bounded and
reported by the preparation object's `textureBytes`. No placeholder textures are
created. Device texture/sampler/uniform limits count all material maps, IBL maps
and optional shadows together. The renderer retains at most one cached binding
for IBL alone and one for IBL plus shadows; switching maps does not accumulate an
unbounded cache. Reusing a map with different intensity/rotation needs no new
bindings, buffers or textures.

Malformed inputs, wrong-device/disposed maps, attachment feedback using the same
view, and map changes observed during draw preparation fail before receiver GPU
writes or submission. Driver validation catches aliases that use different views
of the same texture. GPU completion errors join the renderer/scene's existing
cumulative failure path. Scene cleanup does not dispose borrowed environments,
devices or the direct scene API's borrowed pose.

## Relocatable generated players

Opt into IBL when calling the existing package builder:

```js
const built = buildAnimation(modelPath, outputDirectory, {
  webgpu: true,
  environment: true,
});
```

The generated `gpu_playback.mjs` then also exports
`createGpuAnimationEnvironment`; the filter and lazily imported receiver are
copied into the package and charged to the exact pre-write output-byte budget.
The manifest records `gpuEnvironment: 'f3d-animation-environment-v1'`. Importing
the package still creates no GPU resources: application code prepares a map and
enables `renderer.environment` explicitly, as in the examples above. The source
HDR texture, geometry, device and render attachments remain caller-supplied.

`environment: true` requires `webgpu: true`. It is a build API option, not an
implicit addition to the default GPU package. Omitting it leaves both the
CPU-only and ordinary GPU emission paths unchanged by this package feature.
This keeps applications that do not use IBL from acquiring its modules. The
relocation test runs preparation, animated scene submission and disposal using
only the emitted environment/material modules, with the original toolkit path
unavailable. Unrelated decoder/pose/deformer/shadow boundaries in that test are
explicit substitutes; this is not an end-to-end binary asset or native GPU test.

### Packaging the HDR loader

```js
const built = buildAnimation(modelPath, outputDirectory, {
  webgpu: true,
  environment: true,
  hdr: true,
});
// In the deployed application:
// import {loadGpuAnimationEnvironment} from './gpu_playback.mjs';
```

`hdr: true` additionally exports `decodeAnimationHdr` and
`loadGpuAnimationEnvironment` from the public GPU entry and emits their runtime
modules. Static dependency closure also includes `animation_exr.mjs` and its
`animation_exr_piz.mjs` helper, so this
same loader can receive EXR files after deployment. The standalone EXR decoder
is importable from that emitted module, not an added public GPU-entry export.
It requires `environment: true` (which requires `webgpu: true`). All added
bytes participate in the existing exact pre-write output budget. No HDR URL is
fetched while building or importing the package; loading begins only when called.
Omitting `hdr` leaves CPU, ordinary GPU and existing GPU-IBL package output bytes
and file lists unchanged. HDR modules are not required for those default routes.

The HDR relocation tests execute the actual builder, decoder, loader and filter
from a deployed package after making its source toolkit unavailable. Unrelated
glTF pose/material/scene dependencies are explicit substitutes in those tests;
this is not a full binary glTF-to-rendered-scene or native GPU equivalence test.

## Focused validation

The EXR decoder and loader integration tests:

```sh
node --test tools/ingest/animation_exr.test.mjs \
  tools/ingest/animation_exr_loading.test.mjs
```

There are 28 decoder cases and 20 loading/closure cases (the latter's default
invocation runs a VM-linker child). Actual compressed EXR bytes and native ZIP
inflation execute, including complete native OpenEXR-generated reference files.
Eight additional HALF/FLOAT references across NONE/RLE/ZIPS/ZIP decoded byte-for-
byte identically to native OpenEXR output narrowed to RGBA16F. Loading tests use
recording GPU interfaces and explicit existing filter/RGBE boundaries; the
relocation test executes the real static-closure helper and actual EXR decoder.
These results do not establish native WebGPU pixels, full EXR support, full-suite
compatibility, or a performance improvement.


The PIZ/PXR24 codec and loader regressions:

```sh
node --test tools/ingest/animation_exr.test.mjs \
  tools/ingest/animation_exr_piz.test.mjs \
  tools/ingest/animation_exr_pxr24.test.mjs
```

These 63 cases include complete native-codec fixtures, exact 58-bit Huffman
codes, malformed/overexpanded streams, signed data, mixed channel types, memory
limits, immediate input transfer and cancellation. The loader case exercises
six native panorama byte/stream routes through production source selection and
upload code; GPU/filter/RGBE boundaries are explicit substitutes. Twenty
native-codec references (12 PIZ, eight PXR24) match every output byte;
the full-LUT PIZ FLOAT case also matches all 65,536 original 32-bit words before
half conversion. Native files and compressed bytes execute, not fabricated
codec outputs. These checks do not execute WebGPU shaders or establish full EXR
compatibility or a performance gain.

The HDR decoder, loading and packaging tests:

```sh
node --test tools/ingest/animation_hdr.test.mjs \
  tools/ingest/animation_environment_loader.test.mjs \
  tools/ingest/animation_hdr_package.test.mjs
```

These 76 tests exercise real HDR bytes, scan orientations and packets, bounded
streaming, exact memory budgets, upload/filter lifetimes, cancellation, device
loss and relocated package execution. Fetch and GPU boundaries are recording
interfaces; they do not execute native WGSL or establish rendered-pixel parity.

The existing environment receiver/material tests:

```sh
node --test tools/ingest/animation_environment_receiver.test.mjs \
  tools/ingest/animation_environment_render.test.mjs
```

These 24 host tests run production preparation ownership, receiver packing,
material pipelines and scene submission against recording WebGPU interfaces.
Scene tests substitute only unrelated controller/deformer/ordering boundaries.
They establish binding, budget, lifecycle and command behavior, not execution of
native WGSL, rendered-pixel equivalence or a performance improvement. The native
browser attempt in the implementation environment was blocked from opening the
local test origin by browser policy, so no native result is asserted here.
