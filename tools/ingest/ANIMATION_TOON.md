# Explicit toon material execution

`createGpuAnimationRenderer` and `createGpuAnimationScene` now accept
`shading: 'toon'`. Generated GPU playback packages include the same draw path.
This is an explicit new-backend profile, not automatic MeshToonMaterial adoption
or a complete Three.js material compatibility claim.

```js
const mesh = await renderer.addMesh(gpuGeometry, {
  shading: 'toon',
  baseColor: [0.6, 0.2, 0.1, 1],
  // Optional borrowed linear texture. Use nearest filtering for discrete bands.
  gradientTexture: {view: rampView, sampler: rampSampler},
});
renderer.render({
  colorView, depthView, viewProjection,
  lighting: {viewDirection: [0, 0, 1], lights: [
    {type: 'directional', direction: [0, 0, -1]},
  ]},
  draws: [mesh],
});
await renderer.whenIdle();
```

The lighting rule follows `gradientmap_pars_fragment.glsl.js` and
`lights_toon_pars_fragment.glsl.js` in the pinned r186 source commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`. The signed dot product maps -1..1 to
texture coordinate 0..1; the second coordinate is zero. Only the linear red
channel supplies irradiance. It multiplies light radiance and diffuse color/pi,
without a second Lambert cosine. Negative NdotL is not silently discarded.

Without a gradient texture, irradiance transitions from 0.7 to 1 around
coordinate 0.7, with half-fwidth smoothing. A zero-width footprint uses the
limiting hard step explicitly, avoiding smoothstep's equal-edge singularity.
Gradient filtering, mipmapping and addressing follow the borrowed sampler/view;
no texture is copied, synthesized, decoded or owned by the renderer.

Angle coordinates are not geometry UVs. A gradient-only material requires no UV
attribute or synthetic surface buffer, including mutable BufferGeometry. Other
maps retain their existing UV requirements and independent mapCoordinates.
`mapCoordinates.gradientTexture` is rejected, rather than pretending a UV
transform controls the lighting ramp. Gradient textures share an exclusive slot
with the PBR parameter/Phong specular maps, with no new per-draw uniform storage.

Toon ramp evaluation happens before alpha discard, including instanced draws
with different per-use alpha values. The light loop does not skip fragments
before derivative or implicit texture sampling. Normal maps and flatShading
feed the final shading normal into that ramp. Directional/point/spot light
attenuation matches the Phong profile's r186 falloff; shadows modulate direct
light, while prepared environments supply unquantized diffuse indirect light.
There is no specular lobe, metallic-roughness parameter or clearcoat layer.

The existing 256-byte packet, scene budgets, immediate submission, material
snapshots, direct/bundled instancing and transparent ordering are preserved.
Changing draw colors, emission, camera, lights or bytes of a borrowed texture
before the next submitted use does not require recording another bundle.
Changing a bound texture identity requires a new material registration. Mesh
retirement clears recorded references but never destroys borrowed textures.

## Validation boundary

Host regressions cover real renderer/scene/relocated-package code, UV-free
mutable geometry, exact scene budgets, per-use snapshots and bundle reuse.
They observe commands and queue bytes, not shader arithmetic.

`runPhongToonChecks(device)` in `animation_render_materials.browser.mjs` adds
native pixel comparisons for both new profiles to the existing animation draw
E2E page. It exercises four combinations of instancing/bundles, analytic Phong
angles and shininess, linear specular-map R, point/spot attenuation, signed toon
bands, UV-free gradient maps, flat normals, masking and blending. Missing
WebGPU is a failure, not a passing skip. These native checks were not executed
in the authoring container; browser navigation was blocked by environment
policy. No native parity or measured speedup is claimed here.

Automatic Three.js material conversion, complete light families, light/bump/
displacement maps, custom shader hooks and complete H2 workflows remain required
compatibility work. Phong raw reflection/refraction remains separate as well.
