# Source render targets on the new backend

The renderer facade now connects `setRenderTarget()` to native offscreen
attachments, feeds `target.texture` into later material bindings, and exposes
queue-ordered `readRenderTargetPixelsAsync()`. No retained renderer draws these
passes. Ownership is retained-JavaScript preparation plus new WebGPU execution;
this is not Rust specialization or measured acceleration.

```js
const target = new THREE.RenderTarget(512, 512, {
  type: THREE.HalfFloatType,
  samples: 4,
});
renderer.setRenderTarget(target);
await renderer.compileAsync(sourceScene, sourceCamera);
renderer.render(sourceScene, sourceCamera);

screenMesh.material.map = target.texture;
renderer.setRenderTarget(null);
await renderer.compileAsync(screenScene, screenCamera);
renderer.render(screenScene, screenCamera);

// WebGPU surface: Uint8Array for RGBA8; raw Uint16 half-float bits for RGBA16F.
const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, 16, 16);
```

The WebGL-compatible surface takes a caller-provided typed array after `height`
and uses bottom-left readback coordinates/row order. Its synchronous readback
remains an exact-backend operation, not a disguised Promise.

## Implemented profile

Single-color 2D RGBA8 (linear or sRGB) and linear RGBA16F targets; optional private
depth storage; zero/one or four samples with color resolve; physical-pixel target
viewport/scissor; nearest/linear filtering and source wrap/anisotropy settings.
Canvas pixel ratio and tone mapping do not change offscreen attachments. Existing
scene-bridge material restrictions still apply to the producing offscreen scene.

Sampling uses a persistent GPU-only vertically oriented replica of resolved
color, refreshed lazily after writes. This costs an additional color image and
one fullscreen GPU pass per written-and-sampled version, not a CPU download or
upload. An unchanged version needs no extra pass. Native views stay stable across
content writes; resizing/disposal invalidates them and triggers consumer
re-preparation. Aggregate target storage, including replicas, is bounded.

Two different targets can be alternated for ping-pong rendering. Reading the same
target being drawn is rejected as feedback. Deferred target/canvas calls retain
FIFO order and captured renderer frame settings. `draw -> readback -> draw`
submits the copy before the second draw without waiting for buffer mapping.
Target reconfiguration/disposal invalidates queued uses; renderer disposal also
cancels pending readbacks. As before, source scene objects remain live during
asynchronous preparation: this is **not** a snapshot of all scene state.

## Remaining gaps and verification

Offscreen ShaderMaterial/ShaderLib programs still need destination-specific
program routing/output state. MRT, cube/array targets, mip rendering/generation,
sampled depth and stencil targets are not implemented by this profile. This is
not a claim of complete EffectComposer/postprocessing compatibility.

`node --test tools/ingest/three_render_targets.test.mjs tools/ingest/three_renderer_targets.test.mjs`
checks residency and public renderer ordering using recording devices and
scene/canvas factory doubles. `tools/ingest/three_renderer.test.mjs` additionally
exercises the actual bridge and pinned source objects when the upstream checkout
is available. GPU pixels, full upstream conformance and hardware performance
require separate validation; the recording tests do not establish them.
