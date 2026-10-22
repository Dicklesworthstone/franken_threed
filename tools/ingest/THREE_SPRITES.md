# Native source sprites

The ordinary `WebGPURenderer` route draws pinned r186 `Sprite` objects with
`SpriteMaterial` or default `SpriteNodeMaterial` through the new WebGPU backend.
Command preparation remains retained JavaScript; this is no speedup claim.

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
`castShadow` does not turn a sprite into a caster, matching upstream.

The source `count` defaults to one. Zero and negative integer counts draw nothing.
Larger counts expand to ordered draws within the renderer's existing draw budget,
so repeated translucent instances keep their blending effect. This path does not
claim instanced batching. Custom position/rotation/scale nodes and render hooks
continue to fail explicitly; they require the general node-program path.

Structural material/map edits retain the scene bridge's preparation boundary.
Ordinary scene state follows the facade's existing live-source deferred-render
contract; this change does not add general deferred scene snapshots.

Validation:

- `node tools/ingest/three_sprites.test.mjs` exercises source formulas, parent
  shear/negative scale, perspective and orthographic cameras, live updates,
  texture bindings, count/order/culling, ArrayCamera and preserved public state.
- The `native_sprites*` scenarios in `tests/e2e/webgpu_renderer/parity.mjs` compare
  unchanged source scenes with pinned upstream WebGPURenderer. Pixel evidence is
  a software-GPU correctness check, separate from physical-device performance.
