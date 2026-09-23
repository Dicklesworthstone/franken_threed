# Source-facing render state and indirect lights

The explicit material renderer now exposes fixed state required to consume
ordinary source materials without silently changing their depth/culling rules.
These fields are also forwarded by `createGpuAnimationScene`:

```js
const mesh = await renderer.addMesh(geometry, {
  shading: 'phong',
  side: 'back',                 // 'front', 'back', 'double'
  depthTest: true,
  depthWrite: true,             // explicit even for BLEND
  colorWrite: true,
  alphaMode: 'MASK',
});
renderer.render({...frame, draws: [{mesh, alphaCutoff: 0.3}]});
```

`side` and the existing `doubleSided` shorthand are mutually exclusive. Negative
world determinants still reverse winding; back-side materials cull front faces,
not a fictitious reflected geometry. Depth testing defaults to less-equal when enabled.
Disabling it also disables depth writes, following WebGL's disabled-depth-test
semantics. `colorWrite:false` sets the attachment write mask to zero. Omitting
`depthWrite` preserves the existing API default: true for OPAQUE/MASK, false for
BLEND. A live `alphaCutoff` override requires MASK and occupies the existing
per-use packet. It does not turn a BLEND material into combined mask/blend.

Fixed state selects a distinct pipeline and therefore participates in instancing
and render-bundle identity. Changing only a cutoff does not re-record the native
schedule. Registration and late-draw errors reject before allocation or frame
queue effects, respectively. No additional per-material buffer is required.

`depthCompare` can select any native depth function: `never`, `always`, `less`,
`less-equal`, `equal`, `greater-equal`, `greater` or `not-equal`. It participates
in pipeline/bundle identity and is forwarded through scene registration.
`threeLights:true` selects r186 point/spot falloff for all material profiles,
including Lambert/PBR, and admits nonnegative per-light `decay` (default 2).
Equal inner/outer spot angles produce a hard cone. Neither option changes the
default renderer's shader bytes, packet sizes or light capacity.

## Ambient and hemisphere irradiance

Create the renderer with `indirectLights:true` to admit two additional frame
light kinds. They use the existing 544-byte light buffer and share the existing
eight-light capacity with directional, point and spot lights:

```js
const renderer = await createGpuAnimationRenderer(device, {indirectLights:true});
const lighting = {viewDirection:[0,0,1], lights:[
  {type:'ambient', color:[0.1,0.1,0.1], intensity:1},
  {type:'hemisphere', direction:[0,1,0], color:[0.8,0.9,1],
   groundColor:[0.1,0.05,0.02], intensity:1},
]};
```

`direction` points toward the hemisphere sky, in world space. Sky and ground
colors are linear RGB in [0,1], scaled by nonnegative intensity. Ambient has no
position/direction/range; hemisphere has no position/range/cone. Invalid or
inapplicable fields are errors. A projected shadow must still select a direct
directional/spot light, never an indirect light.

The shader accumulates ambient or normal-weighted sky/ground irradiance into
indirect diffuse, divided by pi. Phong/toon/Lambert use their diffuse color; the
existing metallic-roughness profile weights diffuse by one minus metallic.
Occlusion multiplies this indirect term, not emission or direct lights. The
normal-map/flat-shading result drives hemisphere weighting. Toon's indirect
irradiance is not quantized. Its implicit ramp samples remain in a uniform
light-kind branch before alpha discard. Disabling `indirectLights` preserves
existing shader bytes, bindings and numerical profiles.

This extends the explicit renderer's profiles; it does not establish complete
r186 material, light-probe, PBR multiscattering or native pixel equivalence.

## Composing geometry allocation budgets

`createGpuBufferGeometry` accepts `maxInitialBytes`, defaulting to `maxBytes`.
`handle.update({maxAdditionalBytes})` limits new allocations for that call;
omission retains the old per-handle behavior. A caller managing many residencies
can pass its remaining aggregate GPU-buffer budget. Same-storage uploads need
zero additional allocation budget. A rejected addition issues no writes and
leaves the source upload history unchanged. Existing `maxBytes`/`maxAttributes`
still bound retained identities. CPU byte shadows have the same separate bound.

## Evidence

The source-state suite executes production registration, encoding, queue-byte
snapshots and mutable geometry. Ambient/hemisphere packet values are compared
with the retained r186 `WebGLLights` setup, pinned at
`148ef33ecb6d2502ff796d4554abd1549c95d519`. Existing shader-byte regression cases
remain intact with indirect lights disabled. These are host observations, not
GPU shader arithmetic. Native browser navigation was blocked by the authoring
environment; no pixel pass or measured speedup is claimed.
