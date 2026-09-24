# Offscreen rendering and HDR canvas output

`createGpuRenderTarget` supplies owned offscreen color, optional depth and MSAA
attachments to the existing explicit WebGPU rendering APIs. Its resolved color
texture can be sampled or copied for readback. It does not implement Three.js's
RenderTarget class, framebuffer state, mipmap generation, layered targets or
implicit readback.

```js
import {createGpuRenderTarget} from './gpu_render_target.mjs';

const target = createGpuRenderTarget(device, {
  width: 1280, height: 720, format: 'rgba16float',
  depthFormat: 'depth32float', sampleCount: 4,
  maxBytes: 256 * 1024 * 1024,
});
// Create the existing renderer using target.rendererOptions.
// Submit both producer and consumers before the borrowed-use callback returns.
target.withFrame((attachments, resolvedColor) => {
  renderer.render({...frame, ...attachments});
  output.render({source: resolvedColor, target: destinationTexture});
});
await target.whenIdle();
target.dispose(); // Does not destroy the borrowed device or other renderers.
```

Supported color formats are rgba16float, rgba8unorm, bgra8unorm and their two
8-bit sRGB variants. Depth is depth24plus, depth32float or null; sample count is
one or four. The resolved texture always has one sample and RENDER_ATTACHMENT,
TEXTURE_BINDING and COPY_SRC usage. Native format/sample support is still subject
to device validation; a host recorder does not certify a GPU's capabilities.

`withTexture(texture => ...)` lends the current resolved texture for an immediate
submitted consumer without starting a render pass. Neither callback may retain
native handles across an await, resize or disposal. The owner cannot prevent a
caller from caching an opaque handle; it cannot make unsubmitted future work safe.
No callback runs at zero width or height. There are no queue-completion waits in
ordinary borrowed-use calls. Caller exceptions do not erase submitted work.

Resize replaces attachments, not their contents. Used old generations remain
owned until a queue fence covering preceding producer/consumer submissions
settles. `maxBytes` covers the old-plus-new payload peak, including unresolved
retirements; depth24plus is conservatively charged four bytes per texel. Driver
allocation overhead is not included. A budget failure allocates nothing and is
retryable after `whenIdle()`. Native validation, allocation and queue failures
are terminal. Disposal and device loss release all generations and end pending
waits without claiming to cancel already-issued native work.

## Whole-image HDR canvas composition

`createGpuHdrCanvasRenderer(canvas, factory, options)` reuses the existing canvas
session's adapter negotiation, device ownership, sizing and cancellation. Its
factory gets `(device, rendererOptions, {signal})` and supplies the same
`render(input, frame)`, `prepare?()`, `whenIdle()` and `dispose()` contract as the
direct canvas factory. The renderer options select **linear rgba16float** and
the scene depth/MSAA choices, not the final canvas format.

```js
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';

const session = await createGpuHdrCanvasRenderer(canvas, makeSceneRenderer, {
  device, // Borrow this device; omit it to negotiate and own one.
  renderTarget: {sampleCount: 4, depthFormat: 'depth32float'},
  output: {toneMapping: 'aces-filmic', exposure: 1},
});
session.render(camera);
session.render(camera, {output: {toneMapping: 'reinhard', exposure: 2}});
await session.whenIdle();
session.dispose();
```

One immediate scene submission is followed by the existing output pass, which
samples the resolved HDR texture with textureLoad and submits a full-screen
triangle to that turn's canvas texture. There is no CPU pixel readback, texture
copy, extra scene traversal or new frame loop. Same-size frames reuse HDR storage
and the output bind group. Exposure/operator changes update only the output
packet; they do not rebuild pipelines. The existing none, linear, Reinhard,
Cineon, ACES filmic, AgX and Neutral operators retain their original equations.
NoToneMapping ignores exposure as defined by the output module.

The profile is **opaque, whole-image tone mapping to sRGB**. It maps every pixel,
including backgrounds or content a source application would mark toneMapped:false.
It is not a per-material Three.js tone-mapping implementation. Render a complete
scene with clear color/depth loads; the default clear is opaque black and custom
clear alpha must be one. Transparent scene materials may blend into that opaque
image. Transparent canvas compositing, Display-P3 and HDR display output are not
implied. The HDR intermediate is not itself a claim of an HDR display.

Use `renderTarget` for scene sampleCount/depthFormat/maxBytes (256 MiB default).
`target` retains the direct canvas format/size options, but must be single-sample
and depthless; the composition does not allocate redundant canvas depth/MSAA.
The session's rendererOptions describes this final presentation target; its
factory's options describe the HDR scene target. Diagnostics separate HDR target
payload, the 16-byte output uniform, scene counters and canvas ownership.

Canvas resizing is immediate; HDR storage follows on the next nonsuspended
render. A resize budget failure can therefore occur at render, before the scene
submits. Returning to the old size is safe. Suspension retains the previous HDR
allocation until a later rendered resize or disposal and runs neither stage.

The session supplies a lifetime AbortSignal to renderer factories, even when no
caller signal was provided. Cancellation during compilation or scene preparation
ends the wait and releases partial resources. A late scene owner is disposed
without publication. Source admission errors remain retryable; native stage
failures are terminal and never cause a scene replay or backend switch.

The host tests execute the actual target/session/composition/output modules with
the existing native-call recorder. The source renderer in those tests is explicit
and recorded, not an executed Three.js renderer. Native shader pixels and browser
behavior require a separate WebGPU run; no acceleration is claimed.
