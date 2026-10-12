# Sampled depth attachments

Part of `ft-173` (generalized render targets). This is new WebGPU execution with
retained JavaScript resource ownership, not Rust specialization or a speedup claim.

## Native offscreen owner

```js
import { createGpuRenderTarget } from './gpu_render_target.mjs';

const target = createGpuRenderTarget(device, {
  width: 512, height: 256,
  format: 'rgba16float', depthFormat: 'depth32float',
  sampleCount: 1, sampleDepth: true,
});
target.withFrame((attachments) => {
  // Render into attachments and submit before returning.
});
target.withDepthTexture((depth) => {
  // This is the SAME depth texture used by attachments.depthView.
  // Create its depth-only view, encode consumers, and submit before returning.
});
await target.whenIdle();
target.dispose();
```

`sampleDepth` defaults to false. Enabling it adds TEXTURE_BINDING usage to the
existing depth attachment; it allocates no second texture, creates no shaders,
and issues no copies or resolves. The callback has the same synchronous,
submitted-use lifetime and reentry restrictions as `withTexture`. Use with no
depth attachment is rejected, including for suspended zero-sized targets.

Native depth formats are `depth16unorm`, `depth24plus`, and `depth32float`.
One- and four-sample attachments are supported. Four-sample depth remains a
multisampled texture: use `texture_depth_multisampled_2d` and an explicit sample
index in WGSL. This API does not choose a depth-resolve policy for the application.
The source renderer's ordinary single-sample depth route is a separate contract.

Depth borrowing neither generates nor acknowledges pending color mipmaps.
Color rendering, mip generation, resolved-color readback, and their existing
APIs remain independent. Peak resize budgets include retired generations; a
depth-only consumer counts as use and keeps its old attachment alive until the
queue fence completes. Disposal, partial allocation failures, validation errors,
and device loss follow the same terminal-state/resource-release rules as color.
Depth16 is charged at two bytes per texel; Depth24plus and Depth32 at four.
These are logical storage budgets, not measurements of driver allocation overhead.

## Validation

```sh
node --test tools/ingest/gpu_render_target.test.mjs \
  tools/ingest/gpu_render_target_mipmaps.test.mjs \
  tools/ingest/gpu_render_target_depth.test.mjs
```

These tests execute the production owner against native-call recorders. They
check attachment identity, all three depth formats, multisample descriptors,
color/depth independence, budgets, retirement, cancellation and failures. They
are not native shader execution, pixel-parity evidence or performance results.
