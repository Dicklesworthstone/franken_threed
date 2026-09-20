# HDR bloom and final-image composition

The explicit WebGPU route now has a reusable HDR bloom implementation and a
composition API connecting it to the existing tone-mapping/output pass. They
are optional source-module APIs, not a Three.js `EffectComposer` or
`UnrealBloomPass` compatibility adapter. No source imports are rerouted.

## Render an existing model with managed attachments

```js
import { createGpuGltfAnimationScene } from './animation_model_gpu.mjs';
import { createGpuAnimationPostprocessor } from './animation_postprocess.mjs';

const post = await createGpuAnimationPostprocessor(device, {
  output: { format: canvasFormat, toneMapping: 'aces-filmic', exposure: 1 },
  bloom: { levels: 5, threshold: 1, softKnee: 0.5, strength: 0.8 },
  maxBytes: 256 * 1024 * 1024,
});
const model = await createGpuGltfAnimationScene(device, json, buffers, {
  decode: { resolveTexture },
  scene: { renderer: {
    format: 'rgba16float', depthFormat: post.depthFormat, sampleCount: 1,
  } },
});

// Within the application's EXISTING frame loop; no new loop is installed.
model.update(deltaSeconds);
post.renderScene(model, {
  target: context.getCurrentTexture(),
  frame: { viewProjection, lighting },
});
await post.whenIdle(); // use where the application requires completion

// Alternatively, use an authored camera through the existing model view:
post.renderScene(model, {
  target: context.getCurrentTexture(),
  camera: { cameraNode, aspectRatio: width / height },
  bloom: { strength: 0.4 },
  output: { toneMapping: 'reinhard', exposure: 1.5 },
});

post.dispose(); // releases postprocessing storage, NOT the model or device
model.dispose();
```

`renderScene()` owns and reuses a linear `rgba16float` scene attachment, optional
`depth32float` depth attachment, and bloom output. It never advances a pose or
controller. The borrowed model must already be updated/uploaded, exactly as for
its ordinary `render()` or `renderCamera()` call. External camera matrices,
lighting, environment/shadow inputs, draw selection and other frame settings
remain the model renderer's responsibility.

Configure the borrowed renderer for **single-sample `rgba16float`** and the same
`depthFormat`. The default managed depth format is `depth32float`; `null` supports
renderers without depth. Other managed depth formats and MSAA are not admitted.
Managed attachments clear each frame, defaulting to opaque black. Caller-supplied
attachments, resolve targets, `load` accumulation, or nonopaque clear colors are
rejected. This is an opaque-scene presentation profile, not transparent-canvas
bloom coverage. Settings on a frame do not mutate construction defaults.

## Process an already rendered linear image

```js
post.render({
  source: hdrSceneTexture,
  target: context.getCurrentTexture(),
  bloom: { strength: 1.2, threshold: 2, softKnee: 0.5 },
  output: { toneMapping: 'agx', exposure: 1 },
});
```

The source must contain opaque, finite linear-sRGB radiance in a single-sample
`rgba16float`, `rgba8unorm`, or `bgra8unorm` texture with `TEXTURE_BINDING` usage.
Source and target must be distinct and have equal extents. The destination needs
`RENDER_ATTACHMENT` and must match the configured output format. For an `-srgb`
output view over a non-sRGB base texture, the caller must have enabled the matching
view format; native validation owns that opaque capability check.

Enabled composition submits bloom first and final output second. The existing
output pass alone applies exposure, the selected tone operator, and display
transfer, avoiding display clipping before bloom. All seven existing operators
remain available. Composition uses straight input and opaque final output.
With **`strength: 0`**, it submits only the original output pass: no bloom pyramid
or intermediate is allocated for a fresh external-image composition.

## Standalone bloom and numerical profile

`createGpuAnimationBloom(device, options)` from `animation_bloom.mjs` provides
`render({source,target,...settings})`, `whenIdle()` and `dispose()` independently.
Its target is an equally sized `rgba16float` texture. It preserves source alpha,
but does not spread alpha with glow or implement a transparent compositing policy.

Bright extraction uses nonnegative linear RGB and Rec.709 luminance. The soft
knee width is `threshold * softKnee`; `softKnee` is in [0,1]. Extraction precedes
2x2 downsampling to retain small highlights. One to six ceil-halved levels are
supported, stopping at 1x1; edge taps clamp to the last valid pixel. Each level
uses horizontal then vertical [1,4,6,4,1]/16 blur. Explicit bilinear reconstruction
combines equally weighted levels, normalized by the actual level count. No
filterable-float extension or sampler object is required. `strength` is in [0,64]
and `threshold` in [0,65504]. Half-float intermediates and f32 shader arithmetic
are explicit precision boundaries. Composite radiance saturates at 65504 rather
than storing infinity; that storage clamp is not a display tone operator.

All bloom passes share **one command-buffer submission**. No CPU pixel scan,
readback, per-frame pipeline compilation, timer, animation loop or device creation
is performed. Texture/binding storage is reused while sizes/source identities
remain unchanged. This architecture is not a measured frame-time improvement.

## Ownership, budgets and failures

Standalone bloom `allocatedBytes` counts its 16-byte uniform plus two half-float
textures per pyramid level. Composition additionally counts the output uniform,
managed scene/depth targets and full-size intermediate. Budgets check combined
payload and conservative old/new resize overlap **before allocation**. A resize
may therefore require more budget than its steady-state footprint. Native driver
overhead, in-flight allocations retained by the driver, borrowed textures/scene
resources and JS objects are outside this logical payload bound.

Texture/parameter/extent/known format errors are checked before postprocessing
submissions and leave previous successful statistics intact. Native GPU failures
are terminal and release owned resources. Initial compilation can be cancelled
with `signal`; it does not cancel an already submitted frame. Disposal cancels
completion waits, never destroys the borrowed device/source/destination/model,
and is idempotent. Reentrant render/dispose calls are rejected.

`version` and `lastRender` acknowledge successful synchronous submission, **not
GPU completion**. `whenIdle()` also observes scoped validation and completion of
the borrowed scene used by `renderScene()`. Scene rendering or bloom may already
have submitted before a later native failure; GPU work and animation time are not
rolled back. Use the caller's normal terminal-error/recreation policy.

## Verification

```sh
node --test tools/ingest/animation_bloom.test.mjs \
  tools/ingest/animation_postprocess.test.mjs
```

Host tests execute the production bloom and existing output modules, replacing
only native WebGPU calls with an explicit boundary fixture. Managed-model calls
use a borrowed-scene seam. They cover pass order/feedback, uniforms, reuse/resize,
combined budgets, zero-strength bypass, camera forwarding, failure cleanup and
completion/disposal. They do **not** execute WGSL or prove full glTF pixel parity.

Serve the repository root on a secure origin and open
`tests/e2e/animation_bloom/index.html` for nine real-device pixel checks. These
compile the production shaders and read back constant-image, soft-knee,
highlight-spread, odd-border, sequential-uniform and bloom-before-tone-mapping
results. A native clear-source checks managed HDR/depth attachment composition.
The fixture refuses host spies as native evidence; missing WebGPU/adapters report
BLOCKED. Native checks were not executed in the implementation environment because
browser navigation was blocked. No native GPU, Rust/Wasm, full-workspace,
Three.js compatibility or performance certification is claimed by the host tests.
