# Mutable BufferGeometry on the explicit GPU renderer

`createGpuBufferGeometry(device, geometry)` connects CPU-authored geometry,
including compiled MarchingCubes output, to the existing material/draw renderer.
It is an explicit borrowed-device API, not a replacement for `WebGLRenderer`.
The ordinary `buildAnimation(..., {webgpu:true})` package also exports the factory
from `gpu_playback.mjs`; CPU-only output has no new dependency.

```js
import {createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';

// effect is an ordinary MarchingCubes. Its public methods and arrays remain
// authoritative, whether their numerical loops run in JavaScript or Wasm.
const geometry = createGpuBufferGeometry(device, effect.geometry);
const renderer = await createGpuAnimationRenderer(device, {
  format: 'rgba8unorm', depthFormat: 'depth32float',
});
const mesh = await renderer.addMesh(geometry, {
  shading: 'lambert', vertexColors: true, baseColor: [1, 1, 1, 1],
});

function draw(colorView, depthView, viewProjection, lighting) {
  effect.reset();
  effect.addBall(0.5, 0.5, 0.5, 1.2, 12);
  effect.update();                  // Publishes CPU arrays and needsUpdate.
  geometry.update();                // Queues only source-requested versions.
  effect.updateMatrixWorld();
  renderer.render({                 // Submits immediately, before the next update.
    colorView, depthView, viewProjection, lighting,
    draws: [{mesh, worldMatrix: effect.matrixWorld.elements}],
  });
}
// Drain at an explicit completion boundary, not necessarily after every frame.
await renderer.whenIdle();
// When no further draws use these resources:
mesh.dispose(); renderer.dispose(); geometry.dispose();
// Neither factory owns the device, effect, CPU arrays or attachment textures.
```

## Geometry and material contract

The admitted streams are non-normalized Float32 `position` (vec3), `normal`
(vec3), `tangent` (vec4), `uv` (vec2), and `color` (RGB or RGBA), plus optional
Uint16/Uint32 scalar indices. Position is required; empty geometry is legal.
Interleaved streams share an allocation and upload history. Vertex layouts use
source strides and offsets directly; no 40-byte deformation repacking, static
surface duplication, or compute dispatch is introduced. RGB colors acquire
alpha 1 in the shader; absent colors use white. Material `vertexColors:false`
disables color consumption without reallocating geometry.

The existing unlit, Lambert and metallic-roughness profiles, mapped materials,
projected-shadow receiving, environment lighting and optional instancing use
these layouts. Normal/UV/tangent requirements remain explicit. This is not a
claim of complete Three.js material equivalence or all H2 selector branches.
Alternative material index/UV arrays (`indices`, `texCoords`, `mapCoordinates`)
are refused for a mutable handle, rather than silently overriding its source.
Morph and instance attributes use their existing deformation/instancing paths.
Unknown, normalized, integer vertex and arbitrary custom attribute layouts are
not admitted by this factory. Unsupported input is an error, not a placeholder.

`update()` refreshes source attributes, index identity and counts. Replacing an
attribute with a same-layout attribute rebinds its new storage without another
material pipeline. Layout changes require registering the corresponding new
material layout; a stale registration fails before submission. Changes to
`geometry.drawRange` are read at render time, even without `update()`. Per-draw
`first`/`count` intersect that range; multi-material groups remain explicit draws,
not an implicit Three.js material-array traversal.

## Upload history and lifecycle

The pinned r186 WebGLAttributes algorithm is the source contract, not an
idealized dirty-array model. First use uploads all bytes and leaves ranges
intact. Subsequent use requires a larger source version. Its in-place sort,
coalescing (including the one-element gap), range clearing, zero-count remainder
behavior and callback order are preserved. Version acknowledgment follows the
upload callback, including version changes made by that callback. Unrequested
CPU edits remain GPU-stale. WebGPU four-byte padding comes from the previous
GPU-visible shadow, not adjacent current CPU values.

An upload callback that throws may leave queued writes and cleared ranges, but
not an acknowledged version; retry follows that history. Earlier callbacks can
update a later attribute's same-sized view before its upload. Reentrant updates
and callback-induced storage-shape changes are outside the admitted contract.
Independent adapters retain separate histories; consuming the public ranges in
one adapter affects what a later adapter sees, just as independent source
renderers do. No speculative getter/proxy admission is claimed for this explicit
API.

Source `geometry.dispose()` releases residency, not the public object. Calling
`handle.update()` again recreates it; an existing same-layout mesh registration
picks up the new buffers. `handle.release()` does the same without disposing the
source. `handle.dispose()` closes the adapter permanently. Native upload errors,
rejected error scopes, queue completion failure and device loss are terminal.
Invalid shapes/ranges/budgets are recoverable; no invalid shape uploads a partial
frame before admission. A resident attribute cannot resize on a version update:
replace the attribute instead. Replaced identities remain cached until release,
within `maxAttributes` (default 128) and `maxBytes` (default 64 MiB). CPU shadows
are separately bounded by the same byte budget. Diagnostics count physical
allocation and queue-write requests; they are not Three.js renderer.info.

Submit consumers of version A before queuing version B. Encoding commands alone
does not snapshot buffers. The existing renderer submits each render call;
`whenIdle()` observes completion and cumulative validation, not a frame yield
inserted into synchronous source computation. Borrowing this adapter directly
from another encoder leaves submission ordering with that caller.

## Validation and remaining boundary

Run with the exact pinned Three.js checkout and locked tools dependencies:

```sh
F3D_THREE_ROOT=./oracle-three node --test tools/ingest/gpu_buffer_geometry*.test.mjs
```

The residency suite compares byte histories and callbacks against the actual
pinned `WebGLAttributes`, including subword padding and failure recovery. The
integration suite executes the actual material renderer with a byte-accurate
queue boundary, and builds the unchanged MarchingCubes addon through Rollup.
Twelve compiled update sequences assert native Wasm call counts, exact geometry
bytes, draw counts and stable allocations. These CPU/queue-boundary results do
not claim native GPU or pixel equivalence.

`tests/e2e/animation_render/index.html` also runs native RGB/index/draw-range and
two-versions-in-flight pixel/readback checks. Missing GPU support is a failure,
not a passing skip. Native checks were not executed in the implementation
container because browser navigation is policy-blocked. Full H2 browser/control
and material parity, automatic new-renderer routing and measured performance
remain separate open work; no speedup is claimed here.
