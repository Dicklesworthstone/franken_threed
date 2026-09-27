# Live source-scene distance fog

`createGpuThreeScene` admits the caller's built-in r186 `Fog` and `FogExp2` with
an explicit `fog:{}` option. It uses the existing native linear/exp2 material
fog, not a fullscreen approximation, depth readback or new animation clock.

## Build and render

```js
buildAnimation('actor.gltf', 'dist/actor', {
  webgpu: true, threeScene: true, fog: true,
});
```

```js
import {createGpuThreeScene} from './dist/actor/gpu_playback.mjs';

// THREE, scene, camera, device and attachments belong to the application.
scene.fog = new THREE.Fog(0x8794a3, 5, 40);
const bridge = await createGpuThreeScene(device, scene, {
  three: THREE, fog: {},
  renderer: {format: 'rgba8unorm-srgb', depthFormat: 'depth32float'},
});
bridge.render(camera, {colorView, depthView});

// Each change is live at the next existing render boundary. No prepare(),
// material registration or source-object replacement is required.
scene.fog.near = 8;
mesh.material.fog = false;
bridge.render(camera, {colorView, depthView});
scene.fog = new THREE.FogExp2(0x8794a3, 0.02);
bridge.render(camera, {colorView, depthView});
scene.fog = null;
bridge.render(camera, {colorView, depthView});
```

Fog creation/removal, linear/exp2 replacement, color/scalar changes and boolean
`material.fog` changes remain live. Basic, Lambert, Phong, Toon and Standard
materials use their actual rendered material's flag, including overrideMaterial,
material groups and both passes of double-sided transparent items. Structural
geometry/material/texture changes still require the usual `prepare()` boundary.

The bridge owns fog input: `frame.fog` is rejected, even on a disabled bridge.
`fog:{}` enables the underlying renderer's fog pipeline; a conflicting explicit
`renderer.fog` option is rejected. `fog:null` or omission disables source fog
support and continues to reject non-null `scene.fog`, rather than ignore it.

## Camera and admission

The existing source bridge admits one non-reversed perspective or orthographic
camera with the supplied module's WebGL or WebGPU coordinate convention. The
fog descriptor uses that frame's updated projection and the same private clip
conversion as the draw path. It recovers positive view depth from homogeneous
vertex clip coordinates before division, not radial distance or fragment depth.
The adapter does not rewrite source projection matrices or add camera updates.
The standalone camera helper supports additional projection families; it does
not widen the source bridge's camera contract.

Source Color values are already in the application's linear working space.
Color/scalars must fit the native finite-f32 contract. Linear edges must remain
distinct and increasing after quantization; color and exp2 density must be
nonnegative. Only the exact built-in Fog/FogExp2 prototypes are interpreted;
custom fog subclasses and accessor-backed source fog/color fields reject.
Malformed fog is checked before scene resource allocation, and each live frame
captures fog and camera depth before geometry, instance, deformation, texture,
shadow or background queue effects. Snapshots own their arrays.

## Ordered material receivers and ownership

Adjacent equal receiver states share a color span. Mixed `material.fog` flags
split only adjacent spans, preserving the global opaque/transparent schedule.
Every logical draw occurs once. The first span retains the original attachment
clear/load policy; subsequent spans load both color and depth. Null fog merges
all receivers into one span; an empty frame still submits its requested clear.

Fog changes shaded linear RGB only. Alpha, masking, blend order, depth coverage,
background color/panorama and shadow-map depth rendering are not fogged. The
wrapper leaves shadow/environment flags for their existing receiver owners and
aggregates their color-pass diagnostics. Mixed receivers can INCREASE color
passes/submissions; no speedup or native timing claim is made. The existing
48-byte fog uniform and 256-byte draw packet are unchanged by this integration.
A failure after a submitted prefix is terminal; a partially rendered frame is
not exposed as safely retryable. Disposal and idle waits forward to the owners.

## Packaging and evidence

Only combined `threeScene:true,fog:true` builds emit `three_fog.mjs` and export
`ThreeFogError`, `inspectThreeFog` and `threeFogDescriptor`. The module and
manifest participate in existing hashes and output byte budgets. Ordinary GPU
fog packages do not gain source dependencies. Disabled source packages neither
emit nor import fog modules. Importing an enabled package starts no GPU work.

Run from the repository root:

```sh
node --test tools/ingest/three_fog.test.mjs \
  tools/ingest/three_scene_fog.test.mjs \
  tools/ingest/build_animation_three_fog.test.mjs \
  tools/ingest/animation_fog_camera.test.mjs \
  tools/ingest/build_animation_fog.test.mjs
```

The production source bridge, builder, fog receiver, camera adapter and packer
execute in these tests. Minimal source classes, asset decoder/player and GPU
renderer/residency/deformation/texture services are explicit fixtures, not the
actual Three runtime or GPU. Coverage includes live state, every receiver pattern
up to eight draws, projection identities, override/group/double-sided ordering,
preflight and partial failure, relocation, optional package combinations, hashes
and exact/one-byte-short budgets. Combined packages are imported with unrelated
runtime effects disabled. Native shadow/IBL/background composition, WGSL driver
compilation, actual pixels, full repository tests and performance are not
certified by these host tests.
