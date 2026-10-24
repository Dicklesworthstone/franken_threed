# Explicit cube-face readback

Native targets created with `createGpuRenderTarget(device, {cube: true, ...})`
can read color and sampled depth from any of their six faces. This extends the
core GPU target API, not the Three.js facade's `readRenderTargetPixelsAsync`.
The facade's cube readback/transfer restrictions remain unchanged.

```js
import {createGpuRenderTarget} from './gpu_render_target.mjs';

const target = createGpuRenderTarget(device, {
  width: 256,
  height: 256,
  cube: true,
  format: 'rgba8unorm',
  depthFormat: 'depth24plus',
  sampleDepth: true,
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
2D color/depth reads. The owner still admits only single-sample, mip-zero cube
targets. This work does not enable cube uploads, target copies, automatic mipmaps,
array/volume targets, or complete Three.js cube compatibility.

## Verification

The new helper and owner integration suites use recording GPU devices. They
check distinct face contents, row packing and component types, BGRA conversion,
subrectangles and flipped rows, output view boundaries, synchronous submission,
old/new resize generations, budget reuse, pipeline reuse, cancellation and loss.
They execute production JavaScript but do not execute WGSL on real hardware.

```sh
node --test tools/ingest/gpu_target_readback.test.mjs \
  tools/ingest/gpu_target_cube_readback.test.mjs \
  tools/ingest/gpu_render_target.test.mjs \
  tools/ingest/gpu_render_target_cube_readback.test.mjs
```

WebGPU source basis: https://www.w3.org/TR/webgpu/#dom-gputextureviewdescriptor-basearraylayer
and https://www.w3.org/TR/webgpu/#dom-gpucommandencoder-copytexturetobuffer .
No browser pixel-parity or hardware performance result is claimed here.
