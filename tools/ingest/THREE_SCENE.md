# Live Three.js scenes on the explicit GPU renderer

`createGpuThreeScene` connects a retained, source-owned Three.js Scene to the
existing WebGPU geometry residency and material renderer. The application no
longer needs to assemble per-mesh GPU descriptors, world matrices, light packets
or camera clip conversions for the admitted rigid-mesh profile.

```js
import * as THREE from './pinned-three/build/three.core.js';
import {createGpuThreeScene} from './three_scene.mjs';

const scene = new THREE.Scene();
const material = new THREE.MeshPhongMaterial({color: 0x3876cc});
const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material);
scene.add(mesh, new THREE.AmbientLight(0xffffff, Math.PI));
const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100);
camera.position.z = 4;

const bridge = await createGpuThreeScene(device, scene, {
  three: THREE,
  renderer: {format: 'rgba8unorm-srgb', depthFormat: 'depth32float',
    instancing: true, renderBundles: true, maxDraws: 1024},
  maxGeometryBytes: 64 * 1024 * 1024,
});

// At the application's existing render boundary; no new frame loop or yield.
mesh.rotation.y += 0.01;
material.shininess = 70;
bridge.render(camera, {colorView, depthView});
await bridge.whenIdle();

// Structural edits have an explicit asynchronous preparation boundary.
material.flatShading = true;
await bridge.prepare();
bridge.render(camera, {colorView: nextColorView, depthView});
bridge.dispose(); // The source scene, materials, camera and device remain alive.
```

The required compatibility anchor is Three.js r186, source commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`. The caller supplies that exact retained
component; the runtime revision/identity checks alone do not prove its source
hash. The bridge does not construct a WebGLRenderer, clone source objects,
replace public constructors, change the camera's coordinate system, or claim
complete renderer compatibility. It must be chosen explicitly before renderer
ownership is assigned. There is no hidden retained-renderer submission fallback.

## Source data and preparation

Rigid Mesh/BufferGeometry objects can use MeshBasicMaterial, MeshLambertMaterial,
MeshPhongMaterial, MeshToonMaterial and the existing MeshStandardMaterial-style
metallic-roughness profile. This is not full MeshStandardMaterial multiscattering
parity. Material uniforms and source world/camera/light matrices are read again
at every render. Shared geometry has one GPU residency but independent per-draw
world/material packets. Source arrays, materials, mesh identities and hierarchy
remain owned by Three.js.

Color, opacity, emission, MASK alpha thresholds, Phong specular/shininess, PBR factors, shared map
transforms, transforms, draw ranges, groups, visibility, renderOrder, camera
layers, light values and LOD selection are live. Prepare after adding a new
geometry/material binding, replacing a material, changing its side/alpha/depth/
flat/texture-layout state, disposing a material, or replacing a native texture
view/sampler. Changing geometry buffer identities within the same supported
layout is handled by residency updates; changing its layout needs preparation.
Material `.version` alone is not a reason to compile another pipeline: it does
not hide an actual structural change, and scalar changes stay per-use values.

All reachable meshes are admitted/prepared, including hidden LOD levels. A
hidden unsupported renderable is not silently certified. Unsupported source
features throw `ThreeSceneError`; original geometry/renderer errors retain their
structured codes. Custom material/scene render hooks and effectful attribute
upload callbacks are rejected rather than erased. The explicit lower-level
geometry API remains available for its original upload-callback semantics.

Preparation snapshots descriptors before pipeline awaits, validates the current
source structure and geometry layouts again before publication, and does not
publish a stale handle after disposal. A rejected preparation retires its newly
created bindings, not every previous valid binding. GPU uploads that have
already happened are not rolled back. As with source renderers, callers should
not mutate the scene concurrently with preparation. Runtime source errors can
leave source-visible matrix/LOD/range changes; there is no all-or-nothing source
transaction or replay.

## Upload and submission semantics

The existing `createGpuBufferGeometry` uploader observes attribute versions and
update ranges, not arbitrary CPU edits. CPU edits without `needsUpdate` remain
GPU-stale. Padding/range gaps retain the prior GPU-visible contents. A frame
uploads each submitted geometry once and immediately submits the complete draw
list; encoding alone is not a snapshot of shared buffer contents.

Visibility, layers, retained frustum intersection and LOD methods select source
objects. Opaque and transparent lists preserve the pinned source sorting keys;
material groups intersect drawRange. `scene.overrideMaterial` applies after the
original material selects its list and only when its `allowOverride` is true.
Source modelViewMatrix and normalMatrix observables are updated at the draw
boundary. Double-sided transparent materials use consecutive back/front draws,
including the source material version transitions, unless forceSinglePass is
true. Blended draws are never instanced together or reordered for batching.

The source camera's projection matrix is never rewritten. WebGL clip depth is
converted in the private draw matrix to WebGPU's 0..1 range; source frustum and
sorting use the original projection convention. WebGPU-coordinate cameras use
their projection directly. Orthographic cameras supply parallel view rays.
Scene/camera matrixWorldAutoUpdate and parent authority follow the retained
update boundary. Color backgrounds clear the color attachment with alpha one.
Texture backgrounds, fog, environment backgrounds and output postprocessing are not guessed.

Both renderer `indirectLights` and `threeLights` profiles are enabled. Ambient,
hemisphere, directional, point and spot lights use current world-space values;
point/spot decay and zero-penumbra spotlights are supported. At most eight visible
lights are admitted. Ambient/hemisphere lighting contributes indirect diffuse,
including ambient occlusion, not toon quantization or a punctual shadow.
MeshStandardMaterial uses the explicitly documented nonmetal diffuse/GGX profile,
not a complete source physically based renderer.

## Native material state and textures

Source front/back/double side, depthTest/depthWrite/depthFunc and colorWrite map
to immutable native pipeline state. The lower-level renderer also exposes
`side`, `depthTest`, `depthWrite`, `depthCompare` and `colorWrite` for direct
users. Disabling depth testing disables depth writes, as in the source WebGL
pipeline. These fields do not enlarge the 256-byte draw packet or add passes.

Ready source byte/image/canvas textures now receive owned WebGPU residency by
default. The bridge uploads requested source versions before each submitted use,
realizes samplers and mipmaps, shares compatible Source data and prunes unused
texture ownership after preparation. Image URLs must already be loaded; the
bridge starts no fetch or decode operation. See `THREE_TEXTURES.md` for the exact
byte and browser-managed sRGB external-copy profiles and remaining texture types.

```js
const bridge = await createGpuThreeScene(device, scene, {
  three: THREE,
  texture: {maxTextureBytes: 128 * 1024 * 1024, maxTextures: 256},
});
// No GPU binding Map is needed for an admitted material.map/normalMap/etc.
sourceTexture.image.data[0] = 128;
sourceTexture.needsUpdate = true;
bridge.render(camera, attachments); // Current bytes; same view, sampler and bundle.
```

Sampling/storage changes and source texture disposal require `prepare()` before
another draw. A source-data version update alone requires no new material
registration. Custom onUpdate callbacks require the lower-level explicit texture
owner, rather than being removed or hoisted by this bridge. Texture initialization
and pending native validation participate in bridge disposal/failure handling.

A caller-provided binding Map overrides automatic ownership per texture, and
`autoTextures:false` preserves exclusively borrowed operation:

```js
const textures = new Map([[sourceTexture, {
  view: uploadedView,
  sampler: realizedSampler,
  version: sourceTexture.version,
  sourceVersion: sourceTexture.source.version,
}]]);
const bridge = await createGpuThreeScene(device, scene, {three: THREE, textures});
```

These acknowledgements assert that the application has actually realized the
source upload, orientation, format/color space, mipmaps and sampler state. They
are not permission to invent a native handle or mark an incomplete upload done.
The bridge checks both version numbers before using the binding and does not
own, decode, destroy or silently reupload those borrowed textures. Refresh acknowledgements
after legitimate data uploads. Updating bytes in the same borrowed GPU texture
can reuse a bundle; changing its native binding requires preparation.

The admitted texture subset is material-dependent: base color, tangent-space
normal, emission, occlusion, Phong specular and toon gradient maps. Metallic and
roughness maps must share the same packed texture. Mutable maps use UV0 and one
shared affine transform; independent transforms or UV sets are rejected. Normal
mapping currently requires equal X/Y scale. Toon gradient maps use lighting
angle, not geometry UVs. Unsupported combinations remain errors, not ignored
fields or proof of full material coverage.

## Bounds, failures and ownership

Defaults are 16,384 reachable nodes, 256 geometries, 1,024 material bindings and
128 MiB of aggregate geometry buffers. `renderer.maxBytes` separately bounds its
owned draw/light/material buffers. Geometry bytes include retained old attribute
identities and pending new residencies; replacement cannot pretend that old
storage has already been freed. CPU upload shadows are the same size as geometry
storage and are additional CPU memory, not extra GPU bytes. `texture.maxTextureBytes`
separately bounds owned texel/mip storage (128 MiB by default), including transient
old-plus-new allocations. `diagnostics.textures` reports current ownership and
physical upload/mipmap counters. Borrowed textures
and implementation-owned pipeline/bundle memory are not disguised as these exact
buffer byte counts.

The renderer reserves two mesh-registration slots per `maxBindings` to hold an
old and newly prepared configuration concurrently. Both are charged to its byte
budget. Geometry additions and reuploads pass an aggregate remaining-byte limit
to the uploader, which checks new allocation cost before writes or callbacks.
The underlying uploader exposes `maxInitialBytes` at construction and
`update({maxAdditionalBytes})` for other aggregate owners. A zero allowance
permits existing-buffer updates, but not new buffers; omission restores that
uploader's ordinary per-residency budget.

Source geometry disposal releases residency but preserves source identity;
subsequent rendering can recreate buffers and record a new bundle. Source
material disposal invalidates its GPU registrations and requires preparation.
Bridge disposal releases owned resources/listeners, including late preparation,
without disposing source objects or borrowed textures. It ends pending prepare
and completion waits even if native pipeline/queue work has stalled; this does
not assert cancellation or rollback of that already-issued driver work. Device loss is terminal;
a replacement device needs a new bridge. `whenIdle()` reports cumulative GPU
validation/completion errors. `diagnostics` separates source items, logical GPU
draws, GPU draw commands (including reused bundles), residency bytes and bundle counters; these
are not source `renderer.info` or a performance measurement.

## Deployment and evidence

Direct users import `three_scene.mjs`. Generated GPU playback packages opt in
with `buildAnimation(entry, out, {webgpu: true, threeScene: true})`, or the CLI
flags `--build-animation out --animation-webgpu --animation-three-scene`. This
only packages the bridge; it does not turn the input glTF into a source Three.js
scene. The application lends the same pinned source module and scene at runtime.
CPU-only and ordinary GPU packages do not acquire these optional modules.
Source-scene packages include three_textures.mjs and additionally export
createGpuThreeTextures for explicit texture ownership outside a scene.

Host regressions execute real pinned Three.js scenes and the actual new renderer
against a byte-accurate queue recorder. They cover state mutation, cameras,
layers/LOD, groups, overrides, transparent ordering, upload histories, disposal,
preparation races, allocation bounds and relocated packages. An integration case
runs the existing Wasm-specialized MarchingCubes application build, uploads its
current position/normal arrays, and submits Phong/toon draws without replacing
its public geometry. These are not native shader/pixel measurements. The current
container blocks browser navigation, so native GPU validation is still required.

SkinnedMesh/InstancedMesh/BatchedMesh adoption, lines/points/sprites, source
shadow/environment ownership, additional material/map families, fog, clipping,
stencil, polygon offset, custom shaders/hooks, render targets and complete
output color/tone-mapping workflows remain required work. The separate explicit
animation, instance, shadow and environment APIs still exist, but are not
silently substituted for source semantics here. H1/H2 automatic application
closure and performance gates are not satisfied by this bridge. No measured
speedup or complete Three.js material/renderer parity is claimed.

The automatic-texture suite also builds the actual pinned Wasm-specialized addon,
uses source Phong maps and a UV-free toon gradient without a binding Map, and
checks source bytes at each queued use. Native texture pixels, sRGB mip filtering,
flipY, canvas copies and two in-flight texture versions have a separate harness:
`tests/e2e/three_textures/index.html`. Host command/byte tests do not certify those
native conversions, which must be run on an available WebGPU adapter.
