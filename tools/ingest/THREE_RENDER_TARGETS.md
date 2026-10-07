# Source render targets on the new backend

The renderer facade connects `setRenderTarget()` to native offscreen attachments,
feeds `target.texture` into later material bindings, and exposes queue-ordered
`readRenderTargetPixelsAsync()`. The WebGL-compatible surface also executes
`ShaderMaterial`/`RawShaderMaterial` and admitted ShaderLib programs into these
targets, including fullscreen Mesh roots used by postprocessing passes. No
retained renderer draws these passes. Ownership is retained-JavaScript
preparation plus new WebGPU execution; this is not Rust specialization or
measured acceleration.

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
Canvas pixel ratio and tone mapping do not change offscreen attachments.

On the WebGL-compatible surface, each target owns independent shader-program
state. Intermediate passes disable renderer tone mapping and emit linear values;
an sRGB attachment applies its transfer function natively rather than encoding
twice in the shader. Fragment coordinates and point-sprite quad expansion use
target and viewport dimensions. The built-in PointsMaterial size/scale uniforms
retain the renderer's canvas pixel ratio and logical height, as r186 does. Target programs remain available when the canvas uses its
whole-image HDR fallback. Existing shader/compiler, material, shadow, environment
and geometry restrictions still apply; this does not admit arbitrary GLSL or TSL
on the WebGPU public surface.

`render(root, camera)` and `compileAsync(root, camera)` accept source Object3D
roots as well as Scenes. Ordinary roots are traversed and updated in place,
without reparenting. A non-Scene root uses empty scene-level effects while its
object callbacks receive the original root. Callback-free, small fullscreen
shader roots use the per-call capture described below instead of borrowing
mutable shader inputs through asynchronous preparation.

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
cancels pending readbacks.

## Reused fullscreen shader passes

On the WebGL-compatible surface, callback-free `Mesh` roots with a source
`ShaderMaterial` or `RawShaderMaterial`, no children/instancing/deformation, and
at most six vertices capture their call-time inputs. This includes the ordinary
fullscreen triangle shape used by `FullScreenQuad`. Reusing the same mesh or
changing its material/uniforms in the same turn no longer replaces earlier
queued pass values. `renderAsync()` enqueues before its first await, so mixed
synchronous draw, async draw and readback calls retain their submission order
even during initialization.

Captures copy uniform scalars, arrays, typed-array slices, plain nested structs
and source math values; shader text/defines and material raster fields; camera
and world matrices; geometry upload bytes and draw ranges. They install into
stable private execution views at serialized preparation/submission boundaries,
never by temporarily rewriting application-owned objects across an await.
`compileAsync(root, camera)` prepares that same execution view; uniform-only
steady-state draws do not need a new preparation or material identity. A queued
pass is prepared by itself, not together with future passes whose inputs have
not yet been installed.

Texture objects remain borrowed GPU-resource identities, not pixel snapshots.
Disposal, uploaded source-version changes, sampler/transform edits or a changed
sampled-target generation fail as `THREE_PASS_STALE` before that pass submits.
Normal GPU writes to a ping-pong target do not change its identity and remain
queue ordered. Shared-memory inputs, accessors, custom callable uniform values
and unsupported mutable classes fail explicitly rather than executing hidden
copy hooks. This capture path uses additional CPU copying and memory; it is not
a zero-copy or performance claim. The default capture limits are 2 MiB per call,
32 MiB total pending plus installed state, and 64 retained pass roots. Captures
are released on queue failure/disposal; the last installed state is charged until
replaced or the renderer is disposed.

General scenes, large meshes and callback-bearing roots keep the existing
live-source path. This is **not** a complete per-call scene-state snapshot. Shader
compiler and renderer admission restrictions also remain in force.

## Remaining gaps and verification

MRT, cube/array targets, mip rendering/generation, sampled depth and stencil
targets are not implemented by this profile. Shader programs retain their
explicit unsupported-feature errors. Mesh-root, offscreen shader and small-pass
capture support remove postprocessing blockers but do not establish complete
EffectComposer compatibility, effect parity or arbitrary scene snapshots.

`node --test tools/ingest/three_pass_snapshot.test.mjs tools/ingest/three_renderer_pass_snapshot.test.mjs`
checks per-call data capture and the actual facade/target/readback orchestration
with source-data, scene/compiler and GPU boundary doubles. These tests do not
execute WGSL or certify the pinned Three.js implementation or rasterized pixels.

`node --test tools/ingest/three_render_targets.test.mjs tools/ingest/three_renderer_targets.test.mjs`
checks residency and public renderer ordering using recording devices and
scene/canvas factory doubles. The additional
`three_renderer_program_targets.test.mjs`, `three_scene_program_targets.test.mjs`
and `three_scene_roots.test.mjs` exercise destination-specific program state,
the actual scene bridge, borrowed shader textures, uniforms, root identity,
callbacks and geometry cleanup with the compiler/GPU boundaries replaced.
`tools/ingest/three_renderer.test.mjs` additionally exercises the actual bridge
and pinned source objects when the upstream checkout is available. GPU pixels,
full upstream conformance and hardware performance require separate validation;
the recording tests do not establish them.
