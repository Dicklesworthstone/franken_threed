# Clearcoat on animated glTF materials

`KHR_materials_clearcoat` is supported by the model decoder, animated material
renderer, GPU scene construction and static posed GLB exporter. Optional and
required extension declarations both work through the existing APIs. No separate
loader option is needed for authored coating materials:

```js
const model = await loadGpuGltfAnimationScene(device, 'coated-model.glb', {
  exporting: true,
  scene: {rigidGeometry: true, renderer: {instancing: true}},
});
model.render(frame);
const posedGLB = await model.exportPoseGLB();
await model.whenIdle();
model.dispose();
```

Attachments, camera and frame timing remain caller-owned. The owning loader keeps
encoded source images for opt-in export; borrowed-texture APIs instead need an
explicit export resolver supplying encoded image bytes. No GPU readback is added.

## Material representation

Direct renderer registrations and decoded drawables use `clearcoatFactor` and
`clearcoatRoughnessFactor`, each in `[0,1]` with default zero. The optional borrowed
maps are `clearcoatTexture` (linear red channel), `clearcoatRoughnessTexture`
(linear green channel), and `clearcoatNormalTexture` (linear RGB tangent-space
normal). `clearcoatNormalScale`, default one, scales the latter's XY components
and requires that map. It is independent of the base material's `normalScale`.
All maps support independent UV sets and `KHR_texture_transform` during import,
represented by the existing `mapCoordinates` contract during rendering.

The coating starts from the geometry normal, not the base normal map. It uses
an authored deformed tangent when available, otherwise the existing bounded
fragment-derivative frame with the coating map's UVs. This is not MikkTSpace
tangent generation. Reflections and double-sided normal orientation retain the
renderer conventions. Normal-scale values may be negative but must fit finite
Float32 storage; factor and roughness values remain bounded.

The layer uses fixed IOR 1.5 and Schlick Fresnel at the coating normal/view angle.
It reuses the renderer's unit-Fresnel GGX lobe and split-sum environment profile,
including the existing perceptual roughness floor of 0.045. The coating replaces
the Fresnel-weighted portion of the whole base result, including emission;
alpha, base-color factors and source image pixels are not changed. Occlusion
still affects only indirect lighting, and projected shadows still affect only
the selected punctual light. This does not add transmission, refraction, variable
coating IOR, multiple scattering or complete Three.js physical-material parity.

## Ownership, limits and instancing

Coating settings are captured at mesh registration. They are **not per-draw
clearcoat overrides**. The existing 256-byte draw packet is unchanged; a separate
16-byte immutable material uniform carries the coating parameters. Factor-only
materials do not allocate placeholder images or a UV/color buffer.

Instanced registrations with identical coating parameters, texture identities,
geometry and packed material streams may batch. Different coating parameters
split the material binding and therefore the batch. The last material owner
releases its coating uniform; textures and the device remain borrowed. Material
validation and device-limit checks reject unsupported input rather than silently
omitting a map. A material may now use eight maps, plus any shadow/environment
resources; the scene reserves coating-uniform bytes before allocating geometry.

## Posed export

Export retains all three coating textures, factor/roughness values, independent
UV transforms and normal scale through `KHR_materials_clearcoat`. UV transforms
are baked into separate core texture-coordinate accessors. Encoded images and
samplers use the existing deduplication and color-space checks. Clearcoat,
`KHR_materials_emissive_strength`, punctual lights and other unlit materials in
one scene retain their separate extension declarations. Combining clearcoat and
unlit on the **same** material is rejected.

The result remains a static current-pose GLB, not an editable rig or animation.
Geometry and material data are captured before awaiting encoded image providers,
so subsequent model updates or disposal cannot change an already captured export.
Native WGSL execution, pixel equivalence and measured GPU performance remain
separate validation requirements; Node material/serialization tests do not
establish those properties.
