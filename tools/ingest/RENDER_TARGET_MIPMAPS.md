# GPU-generated render-target mipmaps

The native offscreen owner and the source RenderTarget residency owner now admit
an opt-in, full color mip pyramid. The renderer's existing target-binding path
uses it without another renderer, frame loop, image decoder or CPU readback.

## Source RenderTarget

```js
const target = new THREE.RenderTarget(512, 256, {
  generateMipmaps: true,
  minFilter: THREE.LinearMipmapLinearFilter,
  magFilter: THREE.LinearFilter,
  samples: 4,
});

renderer.setRenderTarget(target);
renderer.render(sourceScene, sourceCamera);
renderer.setRenderTarget(null);
// A material in displayScene samples target.texture, not its own destination.
renderer.render(displayScene, displayCamera);
```

The existing renderer initialization/preparation contract still applies. Supported
source target formats remain RGBA8 (linear or sRGB) and linear RGBA16F, with one
or four samples. This change does not admit cube targets, MRT, sampled depth,
stencil, manually authored target mips or rendering into a nonzero mip level.

All six minification modes are mapped independently to texel interpolation and
mip interpolation. NearestFilter/LinearFilter clamp LOD to zero even when the
texture has a full chain. Mipmapped filters can access the generated pyramid;
without generation they clamp to the sole allocated level. Magnification and
wrap modes remain independent. Anisotropy is bounded to 16 and is used only when
all three WebGPU filter modes are linear.

Source native attachments remain top-down, preserving the existing readback
contract. Before the target is sampled, the resolved native color pyramid is
refreshed, then each level is vertically flipped into the UV-oriented sampling
pyramid. The shader sees the entire latter pyramid. The two pyramids are both
included in the target owner's shared byte budget. Nothing is downloaded or
re-uploaded. Sampler-only changes do not regenerate either chain or replace
texture views; another submitted render invalidates the contents. Toggling
`texture.generateMipmaps`, resizing or disposing the source invalidates queued
uses of the old target generation and retires its native resources.

## Native offscreen target

```js
import { createGpuRenderTarget } from './gpu_render_target.mjs';

const target = createGpuRenderTarget(device, {
  width: 512, height: 256,
  format: 'rgba16float', depthFormat: 'depth24plus', sampleCount: 4,
  generateMipmaps: true,
  maxBytes: 64 * 1024 * 1024,
});

target.withFrame((attachments) => {
  // Encode the scene into these attachments and submit before returning.
});
target.withTexture((texture) => {
  // All mip generation is submitted before this callback. texture.createView()
  // includes the full chain. Submit consumers before returning; do not await.
});
await target.whenIdle();
target.dispose();
```

The native profile also admits BGRA8 (linear or sRGB). `generateMipmaps` defaults
to false. No new pipeline or mip generation runs on the default path. Rendering
and MSAA resolution always use a single-level view of level zero. Only the
resolved color gets additional mip levels; multisampled color and depth do not.
`mipLevelCount` reports the current chain length (zero when suspended/disposed);
`diagnostics.mipGenerations` counts submitted chain refreshes, not render calls.

Each dimension is floor-halved down to one, including non-power-of-two and 1xN
textures. Successive GPU bilinear passes use disjoint one-level source and target
views. sRGB views decode before filtering and encode on output. This is bilinear
reduction, not an area-box algorithm or a claim of pixel identity with every
source backend. GPU error scopes close synchronously; allocation, pipeline,
submission and asynchronous validation failures are observed through the owner.

Initial and peak resize budgets include all resolved mip texels, the separate
MSAA attachment and depth. Used native target generations remain charged until
their retirement fence drains. Driver allocation overhead is not measured.

## Validation

```sh
node --test tools/ingest/gpu_render_target.test.mjs \
  tools/ingest/gpu_render_target_mipmaps.test.mjs \
  tools/ingest/three_render_targets.test.mjs \
  tools/ingest/three_render_target_mipmaps.test.mjs
```

These suites exercise production resource owners with explicit GPU/source
boundary recorders. They validate allocation descriptors, non-overlapping mip
views, all sampler modes, orientation-pass routing, reuse, MSAA, budgets,
retirement, disposal and injected failures. They do not execute WGSL, certify
pixels or measure performance. The implementation-session Chromium attempt could
not load the local test page or initialize its graphics backend; native shader
and pixel validation remains outstanding.
