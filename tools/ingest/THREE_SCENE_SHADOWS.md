# Projected shadows in live Three.js scenes

`createGpuThreeScene` now connects native source shadow cameras and mesh flags to
its existing GPU renderer. This is the richer source-scene path in
`tools/ingest/three_scene.mjs`, not the opaque MeshBasicMaterial CPU packet adapter
in `tools/compat`. Rigid meshes, native instances and admitted GPU skin/morph
geometry share their existing residency between depth and color rendering.

## Enable source ownership

The application supplies its pinned r186 module, existing scene, camera, device
and real render attachments. Enable `shadow: {}` explicitly at construction:

```js
import {createGpuThreeScene} from './three_scene.mjs';

light.castShadow = true; // One DirectionalLight or SpotLight in the scene.
light.shadow.mapSize.set(1024, 1024);
mesh.castShadow = true;
floor.receiveShadow = true;

const bridge = await createGpuThreeScene(device, scene, {
  three: THREE,
  shadow: {
    maxBytes: 64 * 1024 * 1024,
    blend: 'reject',
  },
  renderer: {format: 'rgba8unorm-srgb', depthFormat: 'depth32float'},
});

// Within the application's existing update/render boundary:
// mixer.update(deltaSeconds); // When the application is animating this scene.
bridge.render(camera, {colorView, depthView});
await bridge.whenIdle(); // Only when completion is needed, not every frame.
```

There is no additional clock, cloned scene, CPU shadow deformer or retained
renderer fallback. Without `shadow`, source cast/receive flags still refuse
rather than silently producing an unshadowed image. `renderer.shadows: false`
conflicts with source ownership. A scene with no shadow-casting light owns no
depth map, even when the profile is enabled.

## What participates in each frame

The scene collects casters independently of the viewing camera's frustum. A
mesh outside that camera can still cast onto a visible receiver. The light's
source frustum selects depth draws; source visibility, ancestor visibility,
camera layers, material visibility and the current LOD selection still apply.
`frustumCulled: false` bypasses the corresponding intersection test.

One fused GPU deformation update covers both color-visible and shadow-only
animated meshes before either pass consumes them. Shared geometry and native
instance streams are uploaded once per frame and borrowed by both renderers.
This does not promise that retained source bounds calculations are free, nor
change the source geometry's own bounds-invalidation policy.

OPAQUE and MASK casters preserve material groups, indices, drawRange, current
world matrices and alpha inputs. MASK uses the current material opacity,
alphaTest, vertex colors, base-color texture and shared UV transform. Depth
materials omit lighting-only maps. Texture versions use the existing source
texture uploader; no network fetch or image decode is introduced.

`material.shadowSide` selects depth sidedness. With no explicit shadowSide, this
profile reverses FrontSide/BackSide and preserves DoubleSide, following the
retained WebGL depth-map default. This is a fixed projected-depth profile, not a
claim that every retained backend/filter uses identical shadow materials.

Only `receiveShadow: true` draws receive the selected light's map. Adjacent draws
with equal receiver status share a color pass; later spans load preceding color
and depth. Global source draw order is not rearranged, including transparent
items. This composition can add color passes and is not a measured optimization.
The selected light index follows the current lighting packet, including ambient
lights and visibility/layer changes. Other lights and emission remain unaffected
by the selected projected shadow.

## Live controls and preparation

Mesh cast/receive flags, source light/camera transforms, bias, normalBias and
shadow intensity remain live. The owner calls the native source shadow camera's
`updateMatrices()` at an updating frame boundary. Source camera projection edits
retain their ordinary explicit `updateProjectionMatrix()` requirement where
applicable. The private projection converts WebGL clip depth to WebGPU 0..1;
the source projection and `light.shadow.map` are not replaced.

`light.shadow.autoUpdate = false` retains the last successfully submitted native
map, including across unrelated `bridge.prepare()` calls. Request an update
explicitly after changing the scene:

```js
// After at least one successful map submission:
light.shadow.autoUpdate = false;
bridge.render(camera, attachments); // Reuse the existing depths.
light.shadow.needsUpdate = true;
bridge.render(camera, attachments); // Update depths, then receive them.
```

A newly created manual map needs `needsUpdate: true` for its first submission.
The flag clears only after successful depth submission, not after allocation or
capture. Frozen maps deliberately keep old depths when a pose changes.

Call `prepare()` after changing the shadow-casting light identity or castShadow
selection, replacing its camera/shadow object, resizing the map, or changing
material shadowSide. Existing material, layout and texture-binding preparation
rules still apply. Replacing a map while manual updates are disabled also needs
an explicit first update; the old map is not misrepresented as the new one.

## Bounds, failure and unsupported cases

`shadow.maxBytes` bounds depth texture plus depth-renderer allocation, including
old-plus-new map overlap during replacement. Geometry, textures and the color
renderer retain their separate existing budgets. Depth draw capacity follows
`renderer.maxDraws` (default 1024); depth registrations use the same two-slot
preparation capacity per `maxBindings` as color. Registrations may be prepared
for currently non-casting OPAQUE/MASK materials so cast flags can change live.

`prepare()` drains submitted dependencies before retiring old maps or casters,
revalidates source structure after asynchronous work, and retires unpublished
new allocations on failure. A rejected preparation preserves old valid
registrations where possible; callers must restore incompatible source edits
before using them. `whenIdle()`, abort, device failure and disposal include depth
work. Source objects, devices, attachments and borrowed textures remain borrowed.

Depth and color are separate immediate submissions, not one rollbackable frame.
A color error may follow an already submitted depth update. A failure after an
earlier receiver span is terminal for the color owner rather than permitting a
retry over a partially submitted frame.

This profile supports **one reachable directional or spot shadow light** and
fixed radius-one 3x3 PCF. Point-light cubemaps, cascades, alternate filters/radii,
custom shadow hooks/materials, source shadow shader nodes and shadow-mode
`scene.overrideMaterial` are rejected explicitly. BLEND casters reject by
default; `shadow.blend: 'skip'` explicitly excludes them from depth while keeping
their ordinary color/receiver draws. There is no translucent-shadow model.
Owned shadow mode rejects `frame.shadow` overrides instead of mixing source and
caller ownership. Other source-scene material/geometry restrictions remain.

`bridge.diagnostics` exposes `shadowBytes`, `shadowStats` and `colorPasses`.
`shadowStats` contains lightIndex, caster draw count, mapVersion and whether the
map updated for the last successful color submission. These are submission
observations, not completion certificates or performance measurements.

## Deployment and validation

`buildAnimation(entry, out, {webgpu: true, threeScene: true})` ships the lazy
`three_shadows.mjs` dependency with the existing animation shadow core. Generated
`gpu_playback.mjs` exposes the same `createGpuThreeScene` option. CPU-only and
ordinary GPU packages do not gain source-shadow modules. Importing a generated
package initializes no GPU services; the application still lends its source
Three module and scene at factory invocation.

```sh
node --test tools/ingest/three_scene_shadows.test.mjs \
  tools/ingest/three_scene_shadows.package.test.mjs
```

The 30 host regressions execute production scene/shadow orchestration and package
building with explicit native-renderer/residency, decoder and source fixtures.
They cover independent animated casters, receiver order, masks, instances,
preparation, frozen maps, disposal, relocation, artifact hashes and byte bounds.
They do not execute native WGSL, compiled deformation or retained Three math;
full browser GPU images, retained-renderer parity and speed remain unverified by
these tests. The feature does not establish automatic H1/H2 application closure.
