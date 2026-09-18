# glTF model playback

The model factories connect parsed glTF JSON and supplied binary buffers to the
existing animation player, mesh deformers and explicit WebGPU scene. Applications
no longer have to decode vertex streams or assemble skin/morph drawables manually.
These are opt-in APIs, not a replacement for Three.js's complete GLTFLoader.

## CPU mesh outputs

```js
import { createCpuGltfAnimationModel } from './animation_model.mjs';

// json: parsed glTF 2.0 document. buffers: ArrayBuffers or typed views indexed
// by json.buffers, or a synchronous (index) => bytes provider.
const model = createCpuGltfAnimationModel(json, buffers, { scene: 0 });
model.sample(0.5, { clip: 0, loop: true });
for (let i = 0; i < model.deformers.length; i++) {
  const mesh = model.deformers[i];
  // mesh.positions / normals / tangents: stable, mesh-local Float32 arrays.
  // mesh.worldMatrix: matching world transform. Do not skin/morph a second time.
  // model.drawables[i]: indices, colors and material descriptors.
  // model.source[i]: original node / mesh / primitive / material indices.
}
model.dispose();
```

One pose drives all mesh instances, with distinct world transforms and skin
palettes. Morph deltas precede linear-blend skinning, including up to 32 influences
per vertex. Invalid pose samples preserve the previous outputs. A deformation
failure is terminal for the whole model, not a usable partially updated scene.
Directly sampling `model.pose` requires `model.update()` before consuming meshes.
For a static model with no clips, the construction-time rest pose is ready to use.

## WebGPU scene

```js
import { createGpuGltfAnimationScene } from './animation_model_gpu.mjs';

const model = await createGpuGltfAnimationScene(device, json, buffers, {
  decode: { scene: 0, resolveTexture },
  scene: { renderer: { format: 'rgba8unorm-srgb', depthFormat: 'depth24plus' } },
});
model.controller.createAction(0).play();
// Inside the application's existing frame loop:
model.update(deltaSeconds);
model.render({ colorView, depthView, viewProjection, lighting });
await model.whenIdle();
model.dispose();
```

This composes `createGpuAnimationScene`; it does not add another renderer,
frame loop, clock, scheduler or asset Promise pool. Its pose is factory-owned and
is released on initialization failure, terminal scene failure or disposal.
Device, textures, samplers, attachments, camera and lighting remain caller-owned.
Explicit WebGPU f32 deformation and CPU pose arithmetic retain their existing
numerical profiles. Implicit draws keep opaque/masked source order, followed by
blended meshes sorted back-to-front using each uploaded node origin and the current
view-projection matrix. Equal depths retain source order. `scene: {sortObjects:false}`
preserves the original implicit list. Explicit `render({draws:[...]})` always
preserves caller order, including per-draw overrides and empty draw lists. This
is object-origin ordering, not triangle sorting: intersecting or unusually offset
transparent meshes can still need an application-supplied draw list.

## Texture resolution

`resolveTexture(request)` must synchronously return a borrowed `{ view, sampler }`.
Asynchronously load/decode images and create textures/mipmaps before construction.
The resolver receives a frozen descriptor with `textureIndex`, `imageIndex`,
`image`, `sampler` and `colorSpace` (`'srgb'` or `'linear'`). URI images carry their
URI; embedded images carry buffer/view indices, byte offset, length and MIME type.
Sampler descriptors preserve wrap modes and explicit filtering choices; absent
filters are left unspecified for the application's existing loader policy.

Use sRGB views for base-color/emissive images and linear views for normal and
metallic-roughness images. Resolution is cached by **texture index and color
space**, not image index, so different samplers and linear/sRGB uses do not alias
incorrectly. The resolver must honor the request; native views are opaque and
cannot be inspected for correct color encoding here. No implicit flipY, image
conversion, mip generation, network fetch or placeholder resource is performed.
All material requirements are validated before the first resolver call. A resolver
failure does not roll back caller effects or destroy borrowed resources.

Core PBR factors, alpha modes, vertex colors and the four supported maps are
translated into renderer inputs. `KHR_materials_unlit` keeps its unlit behavior
and ignores its lighting-related fallback fields. `KHR_texture_transform`
supports scale/rotation/offset and the extension's texture-coordinate override.
Each map can retain its own UV set and transform. Tangentless normal maps use the
renderer's derivative frame, reported as `DERIVATIVE_NORMAL_FRAME_NOT_MIKKTSPACE`;
this is not generated MikkTSpace tangent data.

## Animated meshes without source normals

Missing `NORMAL` expands triangles and all associated attributes, indices, UVs,
morph deltas and skin influences consistently. Tangents are ignored as required
by the flat-normal path. Static meshes retain construction-time flat normals.
Skinned or position-morphed primitives instead carry `geometry.flatNormals:true`
and report `DYNAMIC_FLAT_NORMALS`, in addition to `GENERATED_FLAT_NORMALS`.

On the CPU, normals are rebuilt from final mesh-local Float32 positions after
morphing and skinning, and published atomically with positions, bounds, transform
and version. On WebGPU, a second ordered compute pass writes normals into the
existing 40-byte-per-vertex output buffer. No extra mesh buffer, GPU readback or
per-frame CPU vertex deformation is introduced. The scene's existing upload,
submission, completion and disposal contracts still apply.

Direct deformer callers can request `flatNormals:true` for independent contiguous
triangles. Vertex count must be divisible by three. Omit tangents and normal/tangent
morph deltas; position deltas remain supported. Normal storage is optional and is
charged to the component budget when generated. Degenerate triangles produce zero
normals; extremely small edges/areas can also become zero in the explicit GPU f32
profile. Authored smooth normals keep their original deformation semantics.

## Scope and validation boundaries

Geometry supports triangle lists, strips and fans; normalized, interleaved and
sparse attributes; multiple primitives/instances; morph deltas; and all eight
joint/weight sets supported by the deformers. Source geometry is copied, not
rewritten. Decoding uses independent per-stage component limits; GPU resource
budgets remain enforced by the existing scene/deformer/renderer.

The direct-light material route is not full glTF or Three.js parity. Unsupported
codecs/material extensions, occlusion and nontriangle topology fail explicitly;
retain the source loader/backend for those cases. Authored cameras and punctual
lights are available through the model view, but environment lighting and tone
mapping are not supplied by these factories. Image/GLB/network loading remains
separate from these parsed-JSON/supplied-buffer entry points.

Host tests exercise binary decoding, real CPU pose/morph/skin execution, generated
flat normals, GPU command order, resource ownership and failure cleanup. GPU host
spies do not execute shaders. The existing `tests/e2e/animation_webgpu/index.html`
also runs native flat-normal readback checks over multiple scales, workgroups and
same-frame poses. Those browser checks require a real WebGPU device; a blocked
browser or missing adapter is not a passed test. No acceleration claim is made.
