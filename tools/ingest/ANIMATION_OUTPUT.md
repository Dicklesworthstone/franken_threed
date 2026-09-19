# Linear HDR scene output

The opt-in WebGPU output path keeps scene lighting/blending in `rgba16float`
until the final whole-image pass. It provides `none`, `linear`, `reinhard`,
`cineon`, `aces-filmic`, `agx`, and `neutral` tone operators from the pinned
Three.js r186 shader equations. Exposure and tone operator can change per frame
without rebuilding the pipeline. `none` ignores exposure and does not clip.

## Owning glTF loader

```js
import {loadGpuGltfAnimationScene} from './gltf_scene_loader.mjs';

const model = await loadGpuGltfAnimationScene(device, modelURL, {
  // Existing asset/meshopt/Draco/BasisU texture options continue to work here.
  scene: {renderer: {sampleCount: 4, depthFormat: 'depth24plus'}},
  output: {
    format: navigator.gpu.getPreferredCanvasFormat(),
    toneMapping: 'agx',
    exposure: 1,
    outputAlpha: 'premultiplied',
    maxTextureBytes: 256 * 1024 * 1024,
  },
});

// The caller still owns the clock and canvas configuration.
model.update(deltaSeconds);
model.renderCamera({
  target: context.getCurrentTexture(),
  clearColor: [0, 0, 0, 0],
  output: {exposure: 1.5},
}, {cameraNode, aspectRatio: canvas.width / canvas.height});
await model.whenIdle();
model.dispose();
```

Use `render({target,viewProjection,lighting,...})` for an external camera.
The output path forces the internal scene renderer to `rgba16float`;
`output.format` selects the final target format. It preserves the requested
sample count and depth format. Explicit, conflicting output/scene sample or
depth settings are rejected, not silently overridden.

Without `output`, the original `{colorView,depthView,resolveTarget,...}` API is
unchanged and allocates no output pass or intermediate textures. With `output`,
those views are managed and must not be supplied by the caller. Authored cameras,
sorting, culling, pose updates, synchronous picking and posed export continue to
use the existing model. Tone mapping affects display pixels, not exported data.

## Explicit composition

`createGpuAnimationPresentation(device, options)` returns `rendererOptions`
for constructing a compatible borrowed scene/renderer. Call
`presentation.render(scene, {target,...frame})`, or
`presentation.renderCamera(model, {target,...frame}, cameraSettings)`.
Wait for both the borrowed scene and presentation with `whenIdle()`.
Disposing a presentation does not dispose the borrowed scene or final target.

For an already resolved linear texture, `createGpuAnimationOutput(device, options)`
is the lower-level pass: `output.render({source,target,exposure,toneMapping})`.
It allocates one 16-byte uniform buffer and no textures. Sources are single-sample,
single-layer 2D `rgba16float`, `rgba8unorm`, or `bgra8unorm` textures with
`TEXTURE_BINDING`; targets need `RENDER_ATTACHMENT` and matching dimensions.
Encoded sRGB source textures, feedback, implicit scaling and multisampled inputs
are rejected. The presenter performs MSAA resolution before invoking this pass.

## Color, history and ownership

Inputs use linear-sRGB primaries. Negative radiance is clamped before mapping;
alpha is clamped to [0,1] and is never tone mapped. Managed rendering uses
premultiplied linear color; a supplied `clearColor` is straight linear RGBA and
is premultiplied before clearing. The output pass unpremultiplies before nonlinear
operations, applies the selected transfer, then produces straight, premultiplied,
or opaque display output. Premultiplication occurs in the output encoding.
For `-srgb` attachment views the shader compensates for the hardware transfer,
rather than double-encoding semi-transparent pixels. A non-sRGB base texture or
canvas must explicitly enable its sRGB view format when that variant is chosen.

Every render immediately submits scene work before output work. Intermediate
color/depth buffers are reused for same-sized targets and resized to the actual
destination extent, with no resolution reduction. First use and resize require
clear loads; later `loadOp:'load'` / `depthLoadOp:'load'` preserve the *linear HDR
history*, not canvas bytes. A rejected host-side resized render preserves the
previous history. Driver errors are terminal and surface through `whenIdle()`.

`maxTextureBytes` bounds intermediate texel storage, including old-plus-new live
allocations during resize. Depth24plus is accounted as four bytes per sample;
this is not a measurement of driver metadata or deferred physical retirement.
`outputTextureBytes` and `outputBufferBytes` report these owned resources on the
loader result separately from existing image textures and scene buffers.
The canvas, device, retained codecs and worker pools remain caller-owned.

## Validation boundary

Node tests exercise production construction/submission/ownership and loader
composition with explicit GPU and upstream-stage doubles. They do not execute
WGSL or establish pixel equivalence. The attempted browser probe in this session
was blocked by `ERR_BLOCKED_BY_ADMINISTRATOR`; native shader/pixel validation is
still required. No performance result or full Three.js parity is claimed.
This whole-image output stage does not implement per-material `toneMapped`
exclusions, Display-P3, environment lighting, shadows, or an effect composer.
