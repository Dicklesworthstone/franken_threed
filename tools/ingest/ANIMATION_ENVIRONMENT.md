# Image-based lighting for animated materials

The existing environment filter now connects to the existing animated material
renderer. Lambert materials receive diffuse environment light; metallic-roughness
materials also receive roughness-dependent reflections. No second renderer,
animation clock, image decoder or per-frame environment filtering is introduced.

```js
import { createGpuAnimationEnvironment } from './animation_environment.mjs';
import { createGpuAnimationScene } from './animation_scene.mjs';

// Lend a linear-sRGB rgba16float/rgba8unorm GPUTexture: either a 2:1 panorama
// or six square cube faces. The source and device remain caller-owned.
const environment = await createGpuAnimationEnvironment(device, hdrTexture, {
  size: 128,
  diffuseSize: 32,
  lutSize: 128,
  samples: 1024,
  maxTextureBytes: 64 * 1024 * 1024,
  maxSampleWork: 256 * 1024 * 1024,
});
const scene = await createGpuAnimationScene(device, pose, drawables, {
  renderer: { format: 'rgba16float', environment: true },
});

// Within the application's existing frame loop:
scene.update(deltaSeconds);
scene.render({
  colorView, depthView, viewProjection,
  lighting: { cameraPosition, lights: [] },
  environment: { map: environment, intensity: 1 },
});
await scene.whenIdle();
scene.dispose();
environment.dispose(); // Release only after its consumers have stopped.
```

`createGpuAnimationRenderer` accepts the same `environment: true` option and
`frame.environment` descriptor. Existing glTF model/loader scene options forward
`renderer.environment`; `renderCamera(frame, settings)` accepts the environment
in `frame` while supplying the authored camera and punctual lights itself.
For example, use `{scene: {renderer: {environment: true}}}` with
`loadGpuGltfAnimationScene`. Image loading, HDR decoding and source texture upload
are still application responsibilities; this does not add an HDR URL loader.

## Frame controls and composition

A frame descriptor is `{map, intensity?, rotation?}`. `map` is the completed
`createGpuAnimationEnvironment` result on the **same GPU device**. Intensity is a
nonnegative finite number (default 1). Rotation defaults to identity and is an
explicit column-major, right-handed **world-to-environment 3x3** orthonormal
matrix. It is not an Euler-angle vector or an environment-object world matrix;
invert an authored environment rotation before supplying it. Rotation and
intensity are copied into the frame's uniform, never applied to source textures.

Lit frames still supply either `lighting.cameraPosition` or orthographic
`lighting.viewDirection`. The punctual `lights` array may be empty. A nonempty
array adds direct lighting independently. Emission is added independently too:
changing environment intensity does not scale emission or direct lights.

Projected shadows and environment lighting can be used together, including
scene-owned automatic shadows. Shadows attenuate only their selected direct
light, not the indirect environment. Base-color, metallic-roughness, normal and
emissive maps retain their own samplers and UV coordinates; authored tangents and
derivative normal frames both feed the final mapped normal to IBL. Existing
OPAQUE, MASK and BLEND behavior and input draw order remain unchanged.

Omitting `frame.environment`, or setting it to `null`, uses the original
non-environment material pipelines even when the renderer was enabled for IBL.
Unlit meshes never receive environment lighting. The default renderer option is
`false`: it does not import the receiver module, compile IBL pipelines or allocate
an environment uniform. A depth-only renderer cannot enable IBL.

## Explicit sampling profile

The preparation stage retains its established cubemap orientation: face order
+X, -X, +Y, -Y, +Z, -Z; panoramas use `atan2(z,x)` horizontally and `acos(y)`
vertically. No flip or color conversion is guessed. Source radiance is linear.

The diffuse cube stores irradiance divided by pi, so Lambert multiplies it by
base color without dividing by pi a second time. The specular cube is GGX
prefiltered at perceptual roughness `level / (mipLevelCount - 1)`. The receiver
samples at `roughness * (mipLevelCount - 1)`, with the existing material roughness
floor of 0.045. The DFG lookup uses **(roughness, NdotV)** coordinates, matching
the producer's axis order. All IBL samples use explicit LOD, including after
alpha discard and nonuniform lighting control flow.

For metallic-roughness, this is the single-scattering split-sum profile:
`F0 = mix(0.04, base, metallic)`, `Fss = F0 * A + B`, and diffuse weight
`max(0, 1 - Fss) * (1 - metallic) * base`. A fully metallic surface therefore has
no diffuse contribution. Numerical overshoot in the lookup never subtracts
diffuse light. This is not Three.js PMREM/CubeUV pixel equivalence or a claim of
full PBR parity. Multiscattering compensation, material occlusion maps,
transmission, environment backgrounds and automatic exposure are not included.
Use the existing HDR presentation stage for tone mapping and display conversion.

## Ownership, budgets and failure behavior

Preparation waits for filtering completion before returning a stable immutable
snapshot. `sample(device)` verifies device identity and liveness. `whenIdle()` is
available to consumers. An AbortSignal belongs only to construction: aborting it
after successful preparation does not revoke the environment. Explicit disposal
and device loss do revoke it and release its owned textures.

The renderer borrows the environment. It owns one extra **64-byte uniform** once
a lit material is registered; scene `maxBytes` reserves this before allocating
subsequent deformers. Environment texture storage is separately bounded and
reported by the preparation object's `textureBytes`. No placeholder textures are
created. Device texture/sampler/uniform limits count all material maps, IBL maps
and optional shadows together. The renderer retains at most one cached binding
for IBL alone and one for IBL plus shadows; switching maps does not accumulate an
unbounded cache. Reusing a map with different intensity/rotation needs no new
bindings, buffers or textures.

Malformed inputs, wrong-device/disposed maps, attachment feedback using the same
view, and map changes observed during draw preparation fail before receiver GPU
writes or submission. Driver validation catches aliases that use different views
of the same texture. GPU completion errors join the renderer/scene's existing
cumulative failure path. Scene cleanup does not dispose borrowed environments,
devices or the direct scene API's borrowed pose.

## Focused validation

```sh
node --test tools/ingest/animation_environment_receiver.test.mjs \
  tools/ingest/animation_environment_render.test.mjs
```

These 21 host tests run production preparation ownership, receiver packing,
material pipelines and scene submission against recording WebGPU interfaces.
Scene tests substitute only unrelated controller/deformer/ordering boundaries.
They establish binding, budget, lifecycle and command behavior, not execution of
native WGSL, rendered-pixel equivalence or a performance improvement. The native
browser attempt in the implementation environment was blocked from opening the
local test origin by browser policy, so no native result is asserted here.
