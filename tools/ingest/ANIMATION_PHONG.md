# Explicit Phong material execution

`createGpuAnimationRenderer` and `createGpuAnimationScene` accept a `phong`
shading profile. The generated GPU playback package contains the same code.
This is a new-backend material implementation, not automatic conversion of a
Three.js scene or a complete MeshPhongMaterial compatibility claim.

```js
const mesh = await renderer.addMesh(gpuGeometry, {
  shading: 'phong',
  baseColor: [0.4, 0.2, 0.1, 1],
  specularColor: [0.1, 0.1, 0.1], // linear RGB, not sRGB bytes
  shininess: 30,
  flatShading: false,
});
renderer.render({
  colorView, depthView, viewProjection,
  lighting: {
    cameraPosition: [0, 0, 5],
    lights: [{type: 'directional', direction: [0, 0, -1]}],
  },
  draws: [{mesh, specularColor: [0.2, 0.1, 0.05], shininess: 80}],
});
await renderer.whenIdle();
```

The lobe follows `bsdfs.glsl.js`, `common.glsl.js` and
`lights_phong_pars_fragment.glsl.js` from the pinned Three.js r186 source commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`: Lambert diffuse plus normalized
Blinn-Phong, implicit geometry factor 1/4, and the source exponential Schlick
approximation. It does not substitute the metallic-roughness GGX model. The
specular default is `0x111111` converted from sRGB to linear; shininess defaults
to 30 and is clamped to at least 1e-4, matching the source uniform boundary.
Phong point/spot lights use a squared-distance floor of 0.01, squared finite
range window and smooth spot penumbra. The existing Lambert/PBR profiles keep
their original punctual-light formulas.

Base-color/vertex-color multiplication, alpha masking/blending, double-sided
state, normal maps, emission and occlusion retain the existing renderer paths.
`specularTexture: {view, sampler}` supplies a borrowed linear 2D texture; only R
scales the specular lobe. `mapCoordinates.specularTexture` selects independent
UVs and its local transform for deformer inputs. Mutable BufferGeometry keeps
its existing source-owned UV contract. Phong does not accept metallic-roughness
or clearcoat parameters. Unknown or invalid fields are errors, not silent skips.

`flatShading: true` is available to lit profiles. The fragment shader derives
current geometric normals before any discard, so animated and procedural
geometry does not reuse stale smooth normals. Normal mapping on a flat profile
uses the current derivative tangent frame. The existing normal-stream and
invertible-world-transform admission requirements still apply.

## Submission, layout and lifetime

The draw packet remains 256 bytes. Phong specular R occupies word 23 (unused
metallic factor for this profile), G/B occupy normal-matrix padding words 55
and 59, and shininess occupies word 63. Word 51 remains occlusion strength.
Each logical draw gets an independent packet, including instanced draws and
persistent bundles. Changed material values do not change recorded bindings;
changed shading/flat state requires a new mesh registration/pipeline variant.

Specular and PBR parameter textures are mutually exclusive and share the same
binding slots. No extra buffer, placeholder texture or texture ownership is
introduced. All descriptors/arrays are snapshotted before registration awaits;
per-draw admission completes before queue writes. Disposal and device-loss
behavior are inherited from the renderer. Transparent draws remain ordered.

## Evidence and remaining scope

Host tests exercise actual renderer/scene/package code against a queue-byte
recorder, including current values in multiple submitted frames, direct/bundled
instancing, independent maps, invalid late draws and flat-normal generation.
They do not execute shader arithmetic or establish native pixel parity.

Prepared `frame.environment` contributes diffuse irradiance to Phong; it is
not a raw reflection/refraction cube map. Ambient/hemisphere/light maps,
reflection/refraction, bump/displacement, custom shader hooks and automatic
Three.js material adoption are not completed by this profile. These remain
required compatibility work, not permanently excluded source features. No
H2 completeness, native GPU validation or measured performance claim is made.
