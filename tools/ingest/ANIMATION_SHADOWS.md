# Animated scene shadows

`createGpuAnimationScene` can own and update one projected directional or spot
shadow map. The depth pass shares the scene's existing GPU deformers; it does not
sample animation twice, skin vertices on the CPU, or create another frame loop.

```js
const scene = await createGpuAnimationScene(device, pose, decodedDrawables, {
  renderer: { format: 'rgba8unorm-srgb' },
  shadow: {
    lightIndex: 0,
    width: 1024,
    // Explicit policy for BLEND materials: do not cast depth-only shadows.
    blend: 'skip',
  },
});

// Within the application's existing frame loop:
scene.update(deltaSeconds);
scene.render({ colorView, depthView, viewProjection, lighting });
await scene.whenIdle(); // When completion is required, not necessarily each frame.
scene.dispose();       // Does not dispose the borrowed pose or GPU device.
```

The default `shadow: null` keeps the previous path: no shadow map, no shadow
bounds, and no import of the orchestration module. Enabling automatic shadows
also enables the existing renderer's receiving variants. Explicitly disabling
`renderer.shadows` or selecting a depth-only renderer conflicts with this mode.

## Loaded glTF scenes and authored lights

The existing model factories forward scene options, including `shadow`. This
also works with `loadGpuGltfAnimationScene(device, source, { scene: { shadow: {
lightIndex: 0, width: 1024, blend: 'skip' } } })`. Call the model's ordinary
`render(frame)` or `renderCamera(attachments, cameraSettings)`. The latter supplies
current animated camera and punctual-light transforms to the same scene render
path. No new model loop or light-graph traversal is required by the application.

`lightIndex` selects an entry in **the frame's `lighting.lights` array**, not a
source node ID. With authored camera rendering, this is the selected-scene light
instance order exposed by `model.lights`. Choose an existing directional or spot
light. Point lights and missing selections fail explicitly.

## Casters, materials and fitting

OPAQUE and MASK drawables cast by default. MASK preserves base-color alpha,
vertex alpha, the base-color texture, its independent UV coordinates/transform,
draw indices, sidedness and the current deformed geometry. Other PBR maps do not
affect caster alpha and are not passed to the depth-only renderer.

BLEND has no implicit depth-only interpretation. The default `blend: 'reject'`
requires either `blend: 'skip'` or an explicit `casters` boolean array excluding
those drawables. `casters` has one entry per decoded drawable, including separate
mesh primitives. Selecting a BLEND caster still fails even with `blend: 'skip'`.
Transparent transmission and ordinary Three.js object shadow flags are not
inferred. Lit receivers use the selected light's existing PCF shadow variant;
unlit material appearance and other lights remain unchanged.

Every automatic render fits the map to all current scene bounds, including
receivers and **off-camera casters**. Receiving-camera frustum culling never
removes casters. Compact morph/skin bounds update with the uploaded pose; source
vertex arrays are scanned only during initialization, not again each frame.
Current mesh world transforms and animated light transforms are included.

Directional lights receive a fitted orthographic volume. Spot lights retain
their actual position, direction and full outer cone, with an explicit near
exclusion and range-limited far plane. The cone projection stays square even
when the map's pixel dimensions are rectangular. Matrices are column-major and
use WebGPU clip depth 0..1. Nonfinite/unbounded geometry, degenerate directions,
and a 90-degree spot half-angle fail rather than producing a silently clipped
shadow. This is not cascaded, texel-stabilized, or source-engine camera fitting.

Options include `padding` (default 0.05, fraction of fitted extents), `minNear`
(default 0.001 world units), `bias` (default 0.0005 clip-depth units), `normalBias`
(default 0 world units), and `strength` (default 1). These values are explicit
application choices, not a claim that one bias is correct for every scene.

## Submission, overrides and ownership

Depth submits immediately before color on every implicit render, including
repeated renders at the same pose: borrowed alpha textures and lights can change
without advancing `pose.version`. After sampling a borrowed pose directly, call
`scene.upload()` as before. Stale/inconsistent poses fail before automatic depth
submission. `whenIdle()` includes map, receiver and deformation completion.

`frame.shadow: null` skips automatic shadows for that frame. An explicit
`frame.shadow: { map, lightIndex, ... }` uses a caller-owned map instead. Custom
`frame.draws` **must** make one of these explicit choices; the scene does not
guess matching caster transforms for overridden worlds, ranges, or materials.

The depth and color submissions are separate. A recoverable color validation
error can occur after depth has submitted; neither is rolled back. The previous
`shadowStats` remains published until another color submission succeeds. Terminal
shadow failures participate in scene cleanup, including the owning model's pose
and texture cleanup. Borrowed devices, alpha textures and poses remain borrowed
at their respective API boundaries.

## Budgets and diagnostics

`shadow.maxBytes` (default 64 MiB) bounds the owned depth texture plus its depth
renderer buffers. `shadow.maxBoundsBytes` (default 16 MiB) and
`shadow.maxBoundsComponents` (default 16,777,216) separately limit retained CPU
summaries and initialization scans. Those budgets are additional to the scene's
existing color/deformation `maxBytes`; the receiver's extra 96-byte uniform is
charged to that existing scene budget. CPU heap/driver overhead is not a hard
process-memory guarantee.

The direct scene exposes `shadowEnabled`, `shadowBytes`, `shadowBoundsBytes`, and
`shadowStats`. Stats contain the uploaded pose version, light index, caster count,
map submission version and immutable fitted view/bounds. They acknowledge the
latest successful color **submission**, not GPU completion. An explicitly
shadow-disabled/overridden successful frame publishes `shadowStats: null`.

`buildAnimation(..., { webgpu: true })` emits the complete lazy shadow dependency
graph and exports `fitAnimationShadowView` / `animationShadowWorldBounds` for
manual integrations. CPU-only packages remain byte-identical.

## Validation boundary

```sh
node --test tools/ingest/animation_shadow_view.test.mjs \
  tools/ingest/animation_scene_shadow.test.mjs \
  tools/ingest/animation_culling.package.test.mjs
```

These 30 host tests execute production scene orchestration, shadow ownership,
fitting, bounds, draw ordering and package building. Recording renderer/deformer
boundaries and isolated pose/decoder fixtures make submission and lifecycle
behavior observable; they do not execute native WGSL or certify pixels. Tests
cover rotated/degenerate fits, morph-plus-skin bounds, off-camera casters, alpha
material snapshots, budgets, failures and relocated package execution. Full
workspace, native GPU, Three.js visual parity and performance remain separate.
