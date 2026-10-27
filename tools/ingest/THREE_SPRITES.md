# Native source sprites

The ordinary `WebGPURenderer` route draws pinned r186 `Sprite` objects with
`SpriteMaterial` or `SpriteNodeMaterial` through the new WebGPU backend.
Default materials retain the fixed billboard path; authored nodes select the
native TSL program owner described below. Command preparation remains retained
JavaScript; this is no speedup claim.

## Default materials

The default node position formula is lowered to the existing unlit draw matrix.
Each sprite faces the current camera, uses the source world-matrix column lengths
for scale, and applies `center` and material `rotation` in their original order.
Object rotation does not tilt the billboard. `sizeAttenuation: false` scales with
view depth only for perspective cameras. Geometry Z is ignored, as in the source
sprite shader. Public world/model-view matrices and CPU geometry remain source
objects; no clone or temporary rewrite is used for this transform.

Color/map/alphaMap, independent UV transforms, alpha testing, source blend and
depth state, fog and its per-material opt-out use the existing material path.
The pinned native node path selects the alpha map's red component: r186's
`MaterialNode.OPACITY` texture result becomes a scalar through `NodeBuilder.format`.
Visibility, layers, render order and the source sprite-frustum test remain live.
ArrayCamera evaluates each billboard against its layer-compatible child view.
This fixed bridge does not schedule sprite shadow draws. The modern native
source selector used by authored graphs has different behavior, described below.

The source `count` defaults to one. Zero and negative integer counts draw nothing.
Larger counts expand to ordered draws within the renderer's existing draw budget,
so repeated translucent instances keep their blending effect. This path does not
claim instanced batching. Authored position/rotation/scale nodes select the
general node-program path through the facade. Calling the fixed scene bridge
directly with authored nodes still fails explicitly.

Structural material/map edits retain the scene bridge's preparation boundary.
Ordinary scene state follows the facade's existing live-source deferred-render
contract; this change does not add general deferred scene snapshots.

## Authored node sprites

Authored `positionNode`, `rotationNode`, `scaleNode`, `colorNode` and
`opacityNode` graphs now use the exact pinned `SpriteNodeMaterial` and
`WGSLNodeBuilder`. Ordinary `SpriteMaterial` with authored slots uses the pinned
node-library conversion. The source builder owns billboarding, center offsets,
world-column scale, rotation and perspective-only size attenuation; the native
renderer owns the GPU resources and submission. Negative world scale does not
reverse sprite winding, matching the source's Mesh-only determinant rule.

`Sprite.count` becomes the native draw's instance count. A 200,000-instance
particle sprite takes one instanced draw per selected material side, without
spending 200,000 entries of the draw budget. Counts, center values and node
uniforms are captured before asynchronous preparation. FRAME, RENDER and OBJECT
node updates keep their source frequency, and zero counts still run updates
without submitting a draw. Counts are captured after current updates and before
after-updates, so later queued renders cannot replace accepted values.

Compute-produced positions can be used directly:

```js
const positions = TSL.instancedArray(particleCount, 'vec3');
// Populate positions through the existing renderer.compute() path.
const material = new THREE.SpriteNodeMaterial();
material.positionNode = positions.toAttribute();
material.scaleNode = TSL.uniform(0.05);
material.opacityNode = TSL.shapeCircle();
const particles = new THREE.Sprite(material);
particles.count = particleCount;
particles.frustumCulled = false;
scene.add(particles);
```

Storage attributes and `positions.element(TSL.instanceIndex)` share the existing
compute-owned GPU allocation. Drawing never uploads stale CPU storage over a
compute result or reads it back. Source sprite frustum tests and sorting still
use the sprite's world transform, independently of generated particle positions;
disabling frustum culling is appropriate when those positions exceed its source
billboard bounds.

In native shadow scenes, sprites follow the pinned modern renderer's
`castShadow` selection and shadow-material inheritance, including shadow
callbacks, source count and side selection. No separate billboard-shadow formula
is introduced. The shadow recording-device regression checks those commands;
it does not establish sprite shadow pixel parity. Native object callbacks and
overrides retain the same wholly native scene requirement as mesh graphs.
Ordinary native ArrayCamera views retain the source per-view billboard matrices,
instanced counts and logical node-update frequency. XR, multiview and custom
material subclasses remain outside the current native graph profile. The browser
pixel comparisons below use ordinary perspective and orthographic cameras.

Validation:

- `node tools/ingest/three_sprites.test.mjs` exercises source formulas, parent
  shear/negative scale, perspective and orthographic cameras, live updates,
  texture bindings, count/order/culling, ArrayCamera and preserved public state.
- Focused cases in `three_node_program.test.mjs`,
  `three_node_scene_capture.test.mjs`, `three_renderer_nodes.test.mjs`,
  `three_renderer_storage_attributes.test.mjs` and
  `three_renderer_native_shadows.test.mjs` check exact source WGSL, update
  frequencies, queued uniforms/counts, 200,000-instance shared storage and the
  modern shadow override lifecycle.
- The `native_sprites*` scenarios in `tests/e2e/webgpu_renderer/parity.mjs` compare
  unchanged source scenes with pinned upstream WebGPURenderer. Pixel evidence is
  a software-GPU correctness check, separate from physical-device performance.
- `node tests/e2e/webgpu_renderer/sprites.mjs` runs the authored sprite scenarios
  directly on the source renderer and F3D backend, independently of automatic
  construction routing. Perspective, orthographic, live-node updates and
  compute-driven instancing match exactly at seven RGBA8 readback checkpoints
  per scenario (128 by 96 pixels), with no GPU validation errors on Chromium
  156.0.8078.73 / SwiftShader. The particle scenario keeps GPU-produced positions
  authoritative despite unrequested CPU-array edits and submits one native draw.
  This establishes correctness for those cases, not hardware throughput.
