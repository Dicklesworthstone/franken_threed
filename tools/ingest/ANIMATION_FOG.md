# Native distance fog in generated GPU playback

The explicit material renderer supports linear and exponential-squared fog with
`createGpuAnimationRenderer(device, {fog: true, ...})`. Fog is applied to shaded
linear RGB before blending; it does not change alpha, alpha testing or depth
coverage. The native renderer owns one 48-byte frame uniform. There is no
fullscreen pass or depth readback, and the 256-byte draw packet is unchanged.

## Deployable package and camera conversion

Opt in at build time as well as renderer creation:

```js
buildAnimation('actor.gltf', 'dist/actor', {
  webgpu: true,
  fog: true,
});
```

```js
import {
  createGpuAnimationRenderer,
  snapshotAnimationCameraFog,
} from './dist/actor/gpu_playback.mjs';

const renderer = await createGpuAnimationRenderer(device, {
  format: 'rgba8unorm-srgb', depthFormat: 'depth32float', fog: true,
});

// At the application's existing render boundary, after updating its camera.
// projection is the camera's COLUMN-MAJOR projection, before multiplying view.
const fog = snapshotAnimationCameraFog({
  type: 'linear', color: [0.2, 0.25, 0.3], near: 5, far: 40,
}, projection, {clipSpace: 'webgl'});
renderer.render({colorView, depthView, viewProjection, draws, fog});
// viewProjection must independently use WebGPU's 0..1 clip-depth convention.
// The adapter does NOT mutate projection or fix an incorrectly supplied VP.
```

Use `clipSpace:'webgpu'` (the default) when the input projection already uses
native clip depth. `clipSpace:'webgl'` accounts for the private `(z+w)/2`
conversion; it does not rewrite the source camera. Orthographic, perspective,
off-axis, reverse-Z and infinite-far projections use the same bounded linear
solve. Singular matrices and depth rows that do not fit finite f32 are errors.

The row is derived from `-rowZ(inverse(nativeProjection))`. It is dotted with
homogeneous vertex clip coordinates BEFORE division and interpolated normally.
Fragment depth, divided clip Z and radial camera distance are not substitutes.
`animationFogDepthFromProjection(projection, options)` exposes this immutable
four-component row for callers constructing their own low-level descriptors.

For exponential-squared fog use
`{type:'exp2', color:[r,g,b], density:0.02}`. Colors are already linear; no color
space is inferred. Linear near/far must be distinct increasing finite f32 values;
density is nonnegative. The existing native packer validates these fields.
Snapshots own their arrays, so subsequent edits to fog colors or projection
storage cannot alter an already constructed descriptor. Call the adapter again
when camera/fog values change; this requires no pipeline recompilation.

Passing `fog:null` explicitly resets a fog-enabled frame. Importing the package
creates no camera, GPU device, renderer, animation clock or network request.

## Scope and dependency isolation

The build option emits `animation_fog.mjs` (the renderer's lazy dependency) and
`animation_fog_camera.mjs`, and exports the adapter, row helper and
`AnimationFogError` through `gpu_playback.mjs`. Both modules are hashed and
charged to the ordinary output byte budget. The manifest records `gpuFog`.

Default/`fog:false` packages retain their existing file list and entry bytes.
Calling a renderer with `fog:true` from such an isolated package still requires
rebuilding with `fog:true`; it does not fetch missing toolkit files. CPU-only
builds cannot enable fog. This is a build API option, not a new CLI flag.

With both `threeScene:true` and `fog:true`, the builder also emits `three_fog.mjs`
and records `gpuThreeFog`. Select `fog:{}` when creating the source scene bridge
to own live `scene.fog` and `material.fog` exclusions. Those values are captured
at its existing render boundary; the caller must not also pass `frame.fog`.
The explicit adapter above remains available independently of Three.js.
See `THREE_SCENE_FOG.md` for ordered receiver spans, source admission and limits.
Neither path changes shadow, environment/background or output-transform semantics.

## Evidence

`animation_fog_camera.test.mjs` exercises depth reconstruction for perspective,
orthographic, oblique, reverse-Z and infinite-far cameras; 500 deterministic
invertible matrices; perspective-correct interpolation; native f32 packing;
immutable snapshots and malformed inputs.

`build_animation_fog.test.mjs` runs the production builder and emitted fog
modules after relocation, checks enabled/disabled dependency isolation, combined
optional packages, artifact hashes and exact/one-byte-short output budgets.
It deliberately uses fixtures for asset decoding, pose creation and unrelated
renderer/controller services, including a narrow lazy-import renderer fixture.
These tests do not execute native WGSL, certify pixels, or measure performance.

Source descriptor, material receiver, scene integration and relocated source
package coverage is in `three_fog.test.mjs`, `three_scene_fog.test.mjs` and
`build_animation_three_fog.test.mjs`; their fixture boundaries are documented in
`THREE_SCENE_FOG.md`.
