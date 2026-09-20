# Pixel capture and PNG export

The owning glTF loader and managed presentation can capture a submitted frame
without rendering it again. Raw readback preserves HDR values; PNG export writes
a display image with explicit color-space and alpha handling.

## Capture a loaded model

```js
import {loadGpuGltfAnimationScene} from './gltf_scene_loader.mjs';
import {captureAnimationPNG} from './animation_png.mjs';

// For canvas display capture, include COPY_SRC when configuring the context.
context.configure({device, format: 'bgra8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  alphaMode: 'premultiplied'});

const model = await loadGpuGltfAnimationScene(device, 'character.glb', {
  output: {format: 'bgra8unorm', toneMapping: 'aces-filmic',
    readback: {maxBytes: 64 * 1024 * 1024, maxPending: 2}},
});
model.render({target: context.getCurrentTexture(), viewProjection});

// Call in the same turn as render: a canvas current texture can expire later.
// The copy submits before this function returns its promise.
const pending = captureAnimationPNG(model);
// Other frames may now render; they cannot replace the queued capture.
const image = await pending;
const blob = new Blob([image.bytes], {type: image.mimeType});
// The application owns saving/downloading the Blob and any object-URL lifetime.
console.log(image.width, image.height, image.presentationVersion);
```

`captureAnimationPNG` also accepts a `createGpuAnimationPresentation` instance.
It captures `source: 'output'` using the tone mapping, exposure and alpha settings
already used for that displayed frame. It does not advance animation, render a
second frame, change the camera, or choose a new tone curve. Optional `x`, `y`,
`width`, `height`, `flipY`, `signal`, and `maxBytes` configure the capture/file.
The display target must be a readable RGBA/BGRA 8-bit texture with `COPY_SRC`.
Float display targets are refused for PNG, never silently clipped.

An independently owned result includes `bytes` (`Uint8Array`), `mimeType`, width,
height, color space, straight-alpha convention and the captured presentation
version. Failure in the originating draw/output/readback cannot become a blank
successful image. The source is borrowed; cancellation does not dispose it.
After readback has completed, PNG compression does not access the model again.

## Raw HDR and display pixels

```js
const pendingHDR = model.readPixels(); // Resolved linear HDR before display conversion.
const hdr = await pendingHDR;           // Float32Array, including values above 1.

// Alternatively: last display target, including its output transfer/tone mapping.
const output = await model.readPixels({source: 'output', flipY: true});
```

The default `source: 'hdr'` selects the owned single-sample `rgba16float` resolve,
including when rendering with MSAA. It does not require `COPY_SRC` on the display
target. `source: 'output'` selects the actual last target and requires that usage.
Capture only reads the rendered base mip/layer; standalone texture readback below
can select other mips/layers. Coordinate origins follow WebGPU's top-left texture
convention. The default rectangle extends from x/y to the image edge; flipping
reverses returned row order, not the source rectangle.

Raw results include `data`, `width`, `height`, `sourceFormat`, `componentType`,
`bytesPerRow`, `origin`, and `mipLevel`. Presentation adds `source`,
`presentationVersion`, `colorSpace`, and `alpha`. Its `colorSpace` describes actual
samples; raw `srgb` only describes the texture format, and is not a substitute.
BGRA bytes become tightly packed RGBA; half-float values become Float32 without
clipping. No gamma, tone, or alpha conversion is performed by raw readback.

`output.readback` defaults to false. It accepts true or `{maxBytes,maxPending,label}`.
Disabled presentation keeps the prior texture usages, performs no staging
allocation and does not import the readback module. `readbackPending`,
`readbackBufferBytes` and `readbackReservedBytes` expose current ownership. The
output buffer total includes readback staging; reserved bytes additionally count
the prospective returned arrays. Aggregate reservations remain charged until
mapping and producer validation complete. Capacity exhaustion rejects immediately
rather than growing a queue. Defaults are 64 MiB and four pending reads (maximum
64). Image encoding has its own limit, not a combined CPU/GPU memory guarantee.

Abort, disposal and device loss reject pending captures and release staging.
Borrowed textures/devices are not destroyed. Resize can retire old attachments
after their capture copy has already submitted. Completed arrays/files survive
later model disposal. `whenIdle()` includes requests present at its invocation;
per-request read errors are observed through their individual promises.

## Standalone textures and pixel arrays

```js
import {createGpuAnimationReadback} from './animation_readback.mjs';
import {encodeAnimationPNG} from './animation_png.mjs';

const reader = createGpuAnimationReadback(device, {maxPending: 2});
const pixels = await reader.readPixels(texture, {mipLevel: 0, layer: 0});
// Declare actual sample semantics explicitly for a standalone capture.
const png = await encodeAnimationPNG({...pixels, colorSpace: 'srgb', alpha: 'straight'});
reader.dispose();
```

Standalone readback accepts single-sample 2D `rgba8unorm`, `rgba8unorm-srgb`,
`bgra8unorm`, `bgra8unorm-srgb`, `rgba16float`, and `rgba32float` textures. It
copies to 256-byte-aligned staging rows and strips padding into independent
output storage. Depth/stencil, compressed, multisample and integer formats are
not reinterpreted. `completion` may supply a producer-validation promise; managed
presentation provides its own. Standalone `whenIdle()` drains current requests,
not unrelated queue work.

The PNG encoder accepts fixed, unshared Uint8Array/Uint8ClampedArray RGBA storage
with explicit `colorSpace: 'srgb' | 'linear'` (the latter means linear-sRGB
primaries) and `alpha: 'straight' | 'premultiplied' | 'opaque'`. It writes color
metadata, CRC-protected chunks, Sub-filtered scanlines and native zlib compression
via `CompressionStream('deflate')`. It needs no DOM, canvas encoder or Node import.
Missing host compression is an explicit error. It snapshots every pixel before
awaiting compression, so later source mutation/transfer cannot change the file.

PNG requires unassociated alpha: premultiplied samples are divided by alpha in
their stored color space and rounded to RGBA8. Hidden RGB at zero premultiplied
alpha cannot be recovered and becomes zero. Straight-alpha input retains hidden
RGB; opaque input receives alpha 255. No tone mapping or gamma conversion is
performed. A raw HDR array must first use the existing display-output pass.
The encoder's `maxBytes` (64 MiB default) separately bounds scanline staging,
compressed payload and complete PNG, not total process/native-codec memory.
Native compressor allocations and concurrently encoded files have their own cost.

Tests execute the actual copy planning, unpacking, output orchestration and PNG
writer. PNG tests use real host compression and independent zlib decoding; native
GPU methods and scene drawing are recorded test boundaries. These tests do not
establish native GPU-rendered pixel equivalence, complete Three.js compatibility,
or a measured rendering speedup.
