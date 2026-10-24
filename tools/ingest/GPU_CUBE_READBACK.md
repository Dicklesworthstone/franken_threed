# Explicit cube-face readback and transfers

Native targets created with `createGpuRenderTarget(device, {cube: true, ...})`
can read color and sampled depth, upload color, and copy color/depth between
selected faces. This extends the core GPU target API, not the Three.js facade's
`readRenderTargetPixelsAsync`. The facade's cube readback/transfer restrictions
remain unchanged.

```js
import {createGpuRenderTarget} from './gpu_render_target.mjs';

const target = createGpuRenderTarget(device, {
  width: 256,
  height: 256,
  cube: true,
  format: 'rgba8unorm',
  depthFormat: 'depth24plus',
  sampleDepth: true,
  copyDestination: true,
});

// Render each face using target.withFrame(callback, {face}).
// callback must submit its GPU commands synchronously.
const color = await target.readPixels({face: 4});
const depth = await target.readDepthPixels({face: 4, x: 32, y: 48, width: 64, height: 64});
// color: packed Uint8 RGBA; depth: packed Float32 native depth, not distance.
```

Face indices use the native cube layer order: 0=+X, 1=-X, 2=+Y, 3=-Y,
4=+Z, 5=-Z. A cube read requires an explicit face; it never infers the last
rendered face. Non-cube reads retain the default face zero and reject nonzero
faces. Invalid/missing cube faces fail with `GPU_READBACK_FACE` before any
readback allocation, including empty reads.

Color is copied directly from one native array layer. Depth reads use the
existing compute readback pipeline with a single-layer `2d`, `depth-only` view.
No cube filtering, CPU projection, depth averaging, precision downgrade, color
mipmap generation or temporary color texture is introduced. RGBA8/BGRA8,
half-float and supported R/RG/RGBA32F color layouts retain the existing output
contract. Half floats remain Uint16 bit patterns. Stencil is not read or changed.

Each call submits its copy/compute work before returning its promise. Later
face renders, option edits or target resize cannot replace the copied data or
its captured dimensions. Old target storage survives resize until its submitted
work fence. Disposal, caller cancellation and device loss reject unfinished
reads and release their temporary buffers without writing caller output.

`x`, `y`, `width`, `height`, `output`, `flipY`, `signal`, `maxBytes` and
`maxInFlightBytes` retain their existing meanings. `flipY` converts both the
region origin and output row order; it does not rotate or reorient a cube face.
All faces share the same per-device in-flight byte/request budgets with ordinary
2D color/depth reads.

## Color uploads and GPU-only copies

Enable `copyDestination: true` on any destination. Uploads accept the same fixed,
unshared typed-array layouts as 2D targets, including byte offsets, row strides,
BGRA channel conversion, half-float bit patterns and full-precision float32.
They consume the source bytes during the call, never after an await.

```js
const destination = createGpuRenderTarget(device, {
  width: 256, height: 256, cube: true, format: 'rgba8unorm',
  depthFormat: 'depth24plus', sampleDepth: true, copyDestination: true,
});

target.writePixels(new Uint8Array(256 * 256 * 4), {face: 2});
destination.copyFrom(target, {sourceFace: 2, face: 5});
// Copy the already-rendered depth independently, without touching color/stencil.
destination.copyDepthFrom(target, {sourceFace: 2, face: 5});
await destination.whenIdle();
```

`face` selects the destination for writes/copies; `sourceFace` selects a copy
source. Each cube endpoint requires its own explicit index, even for an empty
operation. A 2D endpoint defaults to zero and rejects other values. Cube-to-cube,
cube-to-2D and 2D-to-cube transfers share the same owner and checked planning path.
Every copy addresses exactly one array layer. Both targets must belong to the
same device and be distinct owners; same-owner copies remain rejected even
between different faces.

Color copies use native texture-to-texture commands, not CPU readback or a
staging image. Formats must be copy-compatible; an sRGB-compatible pair copies
raw bits rather than converting color space. A 2D source may select an allocated
mip with `sourceMipLevel`, or supply its resolved MSAA color. Destination writes
still address mip zero. `flipY` transforms both copy origins using their respective
extents; unlike an upload/readback it does not reverse copied rows.

Depth copies require `sampleDepth: true` on the source. They bind one depth-only
2D source face and one destination attachment face, preserving all other faces,
color attachments and stencil. Different depth formats require explicit
`allowFormatConversion: true`. A four-sample 2D source requires an explicit
`sampleIndex` to enter a single-sample cube face; samples are never averaged.
The existing 32-byte per-copy uniform budget and device/profile pipeline cache
are shared across faces, not multiplied into independent face budgets.

Uploads and copies submit synchronously and return a boolean (`false` for an
empty region). `whenIdle()` covers native validation and GPU completion,
including source validation dependencies. Resizing retains used old generations
until their submission fences. An error does not roll back a write already
submitted to the queue.

The owner still admits only single-sample, mip-zero cube targets. Automatic cube
mipmaps, array/volume targets, and complete Three.js cube compatibility remain
outside this implementation.

## Verification

The helper, owner and transfer suites use recording GPU devices. They check
face selection, byte/layout preservation, independent depth-array reference
results, untouched regions/aspects, source mip/MSAA selection, synchronous
submission, resize generations, shared budgets, validation failures and loss.
They execute production JavaScript but do not execute WGSL on real hardware.

```sh
node --test tools/ingest/gpu_target_readback.test.mjs \
  tools/ingest/gpu_target_cube_readback.test.mjs \
  tools/ingest/gpu_render_target.test.mjs \
  tools/ingest/gpu_render_target_cube_readback.test.mjs \
  tools/ingest/gpu_cube_transfer.test.mjs \
  tools/ingest/tests/gpu_depth_transfer.test.mjs \
  tools/ingest/tests/gpu_cube_depth_transfer.test.mjs
```

WebGPU source basis: https://www.w3.org/TR/webgpu/#dom-gputextureviewdescriptor-basearraylayer
and https://www.w3.org/TR/webgpu/#dom-gpucommandencoder-copytexturetobuffer .
No browser pixel-parity or hardware performance result is claimed here.
