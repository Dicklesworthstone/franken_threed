# World-space clipping on the explicit WebGPU path

The new draw renderer and `createGpuThreeScene` now support global and
per-material clipping planes. This is new host/WGSL implementation, not a
retained Three.js renderer submission. The rest of the scene bridge's admission
profile still applies; this does not establish full renderer compatibility.

```js
const cut = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
material.clippingPlanes = [cut];
material.clipIntersection = false;
material.clipShadows = true;
const clipping = { planes: [], localClippingEnabled: true };
const bridge = await createGpuThreeScene(device, scene, {
  three: THREE,
  clipping,
  renderer: { maxClippingPlanes: 8, renderBundles: true },
  // shadow: {}, // Optional existing directional/spot shadow owner.
});

// At the application's existing render boundary, without prepare():
cut.constant = 0.5;
bridge.render(camera, { colorView, depthView });
// Disable material planes; global planes in clipping.planes remain active.
clipping.localClippingEnabled = false;
```

`clipping` is a borrowed live control object. Its only fields are `planes`
(default `[]`) and `localClippingEnabled` (default `false`). Assign a new planes
array or mutate its source `THREE.Plane` values at normal application boundaries.
The factory requires the caller's pinned r186 module. Source planes/normals must
be ordinary `Plane`/`Vector3` instances with finite numeric data properties;
getters, zero normals and components outside finite f32 are not admitted.
The implementation neither modifies source planes nor normalizes their sign.

A plane removes points for which `dot(worldPosition, normal) + constant < 0`.
Global planes always union the removed half-spaces. Material planes also union
by default; `clipIntersection:true` removes only their common negative volume.
The local intersection is independent of the global union. An empty local array
removes nothing. Local planes are ignored when local clipping is disabled.

All five admitted material profiles use the same clipping path. Material groups,
override material selection, transparent back/front draws, native source
instances, generated draw batching, and render bundles retain separate per-use
clipping data. Current world positions include the source instance transform or
the already-deformed input position. Backgrounds are not clipped.

Plane coefficients, plane count, intersection mode and the local-enable flag
are live data, not material/pipeline keys. They require no material version bump
or preparation. Render-bundle reuse does not freeze the plane contents. Invalid
frame clipping is rejected before geometry/texture uploads, deformation or
shadow/color submissions; a corrected frame can retry. Construction failures
release newly owned resources. As elsewhere in this bridge, this is not rollback
of source matrix/LOD updates or already issued native effects.

## Shadows

The existing directional and spot shadow owners apply material-local planes
only when both local clipping and that material's `clipShadows` are enabled.
Global renderer planes are excluded from shadow-map passes, matching the pinned
source clipping policy. `shadow.autoUpdate` and `shadow.needsUpdate` still decide
when a map is redrawn: changing a plane does not secretly refresh a frozen map.
Clipping does not add point-light, variance, custom-material or transparent
shadow modes that the existing shadow profile does not implement.

## Direct renderer API and storage

Choose `clipping:true` when creating `createGpuAnimationRenderer`, then supply
`frame.clippingPlanes` for global planes and `draw.clippingPlanes` plus
`draw.clipIntersection` for local planes. Direct planes are four-component arrays
`[nx, ny, nz, constant]`, not Three objects. `createGpuAnimationShadowMap` also
accepts the `clipping`/`maxClippingPlanes` factory options and local draw fields.
Its direct depth API has no global plane input.

`maxClippingPlanes` is a fixed capacity for the combined global/local count per
draw, defaults to eight, and accepts 1–64. The enabled packet appends a 16-byte
header and one 16-byte vec4 per capacity slot to the original 256-byte packet.
Eight slots therefore use 400 bytes, normally a 512-byte aligned arena stride.
Device binding/allocation limits and existing memory budgets include this cost.
Unused slots are cleared. Distinct logical draws never overwrite a shared plane
uniform before submission. Default-off renderers retain the original packet.

GPU animation packages include the clipping helper; `threeScene:true` packages
include the source adapter. Imports create no GPU services or frame loop. CPU-only
packages do not gain this dependency. The feature adds no executor or scheduler.

## Verification boundary

```bash
F3D_THREE_ROOT=/path/to/pinned/three.js node --test \
  tools/ingest/animation_clipping.test.mjs \
  tools/ingest/three_scene_clipping.test.mjs
```

These tests use real pinned source objects, production renderer/scene/shadow
modules, relocated package imports, and a recording GPU device. They verify
numeric packets, ordering, generated shader integration and lifecycle behavior.
They do not execute WGSL in a browser, compare pixels or measure acceleration.
Stencil clipping caps and alpha-to-coverage clipping are still separate missing
features; this hard clipping path does not claim them.
