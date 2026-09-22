# Persistent render bundles for dynamic playback

The explicit material renderer can reuse native WebGPU draw commands while
continuing to upload current geometry, camera, lighting and per-draw parameters.
This is an opt-in CPU command-encoding optimization, not fewer GPU draws or an
automatic replacement of an application's Three.js renderer.

```js
const renderer = await createGpuAnimationRenderer(device, {
  format: 'rgba8unorm', depthFormat: 'depth32float',
  renderBundles: true, maxRenderBundles: 4, maxDraws: 1024,
});
const mesh = await renderer.addMesh(gpuGeometry);
// Update the supplied deformer or BufferGeometry residency before rendering.
renderer.render({colorView, depthView, viewProjection, draws: [mesh]});
// Changed data in the same buffers does not require re-recording commands.
renderer.render({colorView, depthView, viewProjection: nextCamera, draws: [mesh]});
await renderer.whenIdle();
console.log(renderer.bundleDiagnostics);
```

`createGpuAnimationScene` accepts the same settings in its `renderer` option.
It exposes `renderBundlesEnabled`, `renderBundleStats`, and
`clearRenderBundles()`. The existing GPU playback package includes the helper;
CPU-only package contents and dependency edges are unchanged.

## Reuse and invalidation

Each renderer owns a bounded least-recently-used cache of complete draw
schedules. Reuse checks every recorded pipeline, all vertex/index buffers,
index format, material/light bind groups, draw order and range. The renderer's
device, attachment formats/sample count, arena binding, stride and instancing
mode are fixed for that cache. Uniform offsets and first-instance indices
remain tied to the original logical draw positions. No object IDs, hashes,
source version counters or CPU-data scans stand in for these comparisons.

Geometry bytes, transforms, colors, map transforms and lighting uniforms remain
current. Viewport/scissor, attachment views and load/store/clear operations stay
on the enclosing render pass. The renderer still validates every frame, writes
current uniforms, and submits immediately: no extra yield, deferred frame, or
replay is introduced. Submit version A before updating a borrowed buffer to B.
Encoding a command buffer alone does not snapshot that buffer's contents.

Source draw-range changes and new storage identities produce new schedules.
Returning to a retained structural variant reuses it. Mesh disposal clears
cached references before retiring its resources. Renderer disposal/device loss
retire the cache; a new device requires a new renderer. The cache owns no GPU
buffers. Recording, execution or validation errors use the renderer's existing
terminal-error path; they do not retry the frame as direct draws.

`frame.renderBundles: false` selects direct encoding for that frame. Re-enabling
it can reuse still-valid schedules. This is useful for rapidly changing draw
lists: the cache does not yet partition a frame into independently stable
segments or predict whether a new schedule will be used twice. The shared
encoder fully rebinds direct draws, including after `executeBundles([])`.
Instancing remains an independent option; transparent draws retain their order.

## Bounds and observations

`maxRenderBundles` defaults to 4 (range 1..64). Each schedule contains at most
`maxDraws` logical records. Comparison storage is thus bounded by their product.
Native bundle memory is implementation-owned and not measurable as an exact
byte count here; it is not disguised as part of the GPU-buffer `maxBytes` total.
`clearRenderBundles()` drops cached references without destroying borrowed
resources. Counters remain cumulative after clearing.

`bundleDiagnostics` reports completed host builds, reuse/execute requests,
evictions, encoded draw calls, and current cache occupancy. These are host
recording observations, not successful GPU completion. `drawCallCount` still
counts draw commands inside a submitted bundle, not `executeBundles` calls.
Use `whenIdle()` to observe cumulative validation and queue completion.

The focused tests exercise the real renderer, scene playback, residency and
relocated packages against a byte-accurate queue boundary. They include 1,000
changing logical draws, multiple submitted buffer versions, range/resource
invalidation, BLEND ordering, native error boundaries and direct/bundle toggles.
The existing browser geometry page also checks pixels and buffer readback with
bundles enabled and disabled. Those native checks are a separate execution
requirement; Node results alone do not establish native GPU or H1/H2 parity,
and no measured throughput improvement is claimed.
