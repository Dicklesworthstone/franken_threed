# ESSL → WGSL compiler

`essl_preprocess.mjs`, `essl_parse.mjs` and `essl_wgsl.mjs` compile WebGL GLSL
ES 3.00 (and 1.00) programs, as r186 `WebGLProgram` hands them to the GL driver,
into WGSL for the new WebGPU backend. It is first-party JavaScript, so it also
runs at runtime for programs built dynamically after startup (plan §9.5).

**Route vs. plan §9.4.** The plan's diagram routes normalized ESSL through a
Naga entrance. Naga's GLSL frontend rejects ESSL (`#version 300 es`, combined
`sampler2D`, loose uniforms), so this compiler emits WGSL directly from its own
typed AST. Naga is used as an independent WGSL validator (native, build-time,
not shipped), and the browser's own WGSL compiler is the final check.

## Interface

`compileEsslProgram(vertex, fragment, {defines, clipDepth})` returns
`{vertex, fragment, reflection}`: two WGSL modules with entry points
`f3d_vertex` / `f3d_fragment` and one shared resource interface.

| Resource | Binding | Notes |
| --- | --- | --- |
| Uniform buffer | group 0, binding 0 | Every non-sampler uniform of both stages. `reflection.uniforms[*].node` gives each leaf's byte offset/stride. `bool` → `u32`; scalar/`vec2` arrays → `vec4` elements; structs `@align(16)`. Internal `f3d_target` (framebuffer width, height) at `reflection.targetOffset` when `gl_FragCoord` is read. |
| Textures | group 1, binding 2k / 2k+1 | Texture / sampler of the k-th sampler uniform (arrays expand per element; ESSL already requires constant indices). Shadow samplers → depth texture + comparison sampler. |

Attributes keep declaration order (`layout(location)` honored; matrices take one
location per column). Varyings link by name; arrays/matrices take one location
per element/column; integer varyings are flat.

## Semantics preserved

- Clip depth: `gl_Position.z` remapped from GL `[-w, w]` to WebGPU `[0, w]`
  (`clipDepth: 'webgpu'` disables it for WebGPU-convention projections).
- `rows: 'gl'` (render targets sampled later by GL-convention shaders): clip Y
  is mirrored so storage rows are GL's bottom-up rows; `gl_FragCoord` and `dFdy`
  then need no flip; the caller swaps the pipeline's front face.
- `gl_FragCoord` in GL window coordinates (bottom-left origin); `dFdy` keeps GL's
  upward sign; `texture()` outside fragment shaders samples level 0.
- Vector `==`/`!=` reduce with `all`/`any`; `mod` uses `x - y*floor(x/y)`;
  shift counts become `u32`; `inverse`, `matrixCompMult`, `outerProduct` are
  generated helpers.
- Global stage I/O becomes private variables copied by the entry points.
- `out`/`inout` arguments copy through temporaries in source evaluation order;
  side-effecting `?:`, `&&`, `||`, `++`/`--` and assignments inside expressions
  are lowered to statements; multi-component swizzle stores are split.
- Implicit-derivative sampling in non-uniform control flow keeps GL's undefined-
  in-divergent-quads behavior by disabling WGSL's `derivative_uniformity`
  diagnostic. That is the same contract ESSL gives, not an approximation.

## Explicit errors (not silently dropped)

`gl_PointCoord` outside point programs, `gl_ClipDistance`,
filtered sampling of integer textures, true `switch` fallthrough, struct stage
I/O, `gl_FragData`, token pasting, type mismatches (ESSL has no implicit
conversions).

## Evidence

- `essl_wgsl.test.mjs`: lowering and layout regressions.
- Corpus: every program r186 `WebGLRenderer` compiled across the 299 upstream
  `webgl_*` examples (578 unique stages, built-in materials included):
  570 translate and validate with Naga and compile in Chrome; the 8 others are
  the explicit `gl_PointCoord` / `gl_ClipDistance` errors above. Corpus
  validation proves well-formed WGSL, not equal rendering; rendered parity is
  measured per scenario in `tests/e2e/webgpu_renderer`.

No performance claim.

## ShaderMaterial render path

`three_program.mjs` assembles each ShaderMaterial / RawShaderMaterial program
exactly as r186 `WebGLProgram` does for the parameters it can reach (precision
block, SHADER_* and material defines, attribute/instancing/fog defines,
built-in declarations, ESSL 3.00 macros, live `ShaderChunk` includes,
`unroll_loop` expansion, tone-mapping and output-encoding functions), maps
material state with `WebGLState` semantics (blend table, BACK culling with
front-face flips for BackSide and negative determinants), and packs current
uniform values every frame (built-ins from camera/object, `material.uniforms`
otherwise; fog uniforms refreshed into the material as WebGLRenderer does).

The scene bridge draws programs in the same passes and source order as other
materials (`animation_program_mesh.mjs`): each logical draw gets its own
uniform-arena slice, attributes come from geometry by name at the reflected
locations (InstancedBufferGeometry / InstancedMesh instance attributes use
instance step mode; absent attributes read GL's constant `(0, 0, 0, 1)`).
Admitted on the WebGL-surface route without a renderer tone-mapping pass.

Explicit errors for now: `lights: true`, skinned/morphed/batched objects,
clipping planes, uniform groups, shadow casting, indexed point sprites,
integer attributes, 3D/array/shadow sampler textures, texture-uniform swaps
without `prepare()`.

Points: compiled with `points: true`, each vertex is an instanced quad of
`gl_PointSize` pixels (clamped to [1, 1024], a common ANGLE
`ALIASED_POINT_SIZE_RANGE`; the true GL maximum is device-dependent), a point
whose center is outside the clip volume is dropped whole, and `gl_PointCoord`
has GL's upper-left origin. Scenario `shader_points` matches upstream
WebGLRenderer (424 of 76,800 pixels differ by ~1 on disc edges).

Evidence: scenario `shader_material` matches upstream WebGLRenderer (total
absolute channel difference 2 over 320×240). Examples now running:
`webgl_custom_attributes`, `webgl_custom_attributes_lines`,
`webgl_buffergeometry_selective_draw`, `webgl_buffergeometry_attributes_none`,
`webgl_buffergeometry_instancing`, `webgl_buffergeometry_instancing_billboards`,
`webgl_custom_attributes_points`, `webgl_custom_attributes_points3`,
`webgl_buffergeometry_custom_attributes_particles`.

## Built-in materials through ShaderLib programs (WebGL surface)

`three_webgl_program.mjs` ports r186 `WebGLPrograms.getParameters` /
`getUniforms`, `WebGLProgram` source assembly, `WebGLLights` (setup /
setupView) and `WebGLMaterials` uniform refresh as **retained JavaScript**
construction components (MIT, three.js authors), taking the application's own
`THREE` module so runtime `ShaderChunk`/`ShaderLib` edits are honored. The
assembled programs are byte-identical to what r186 WebGLRenderer hands the
driver (verified against browser-captured sources); ShaderMaterial now uses
the same parameter path (fixes `USE_UV1..3`, which upstream derives from map
channels, not geometry).

On the WebGL surface the scene bridge draws built-in materials with their
ShaderLib program when the core material path rejects a feature, and always
for `PointsMaterial` (GL point sizes): e.g. cube `envMap`
reflection/refraction with every `combine` op, `lightMap`, `bumpMap`,
`displacementMap`. Lights are wired from the ported light state (no shadow
maps yet). ShaderMaterial `lights: true` works through the same wiring.

MeshStandard/Physical programs get r186's DFG LUT (the same 16×16 RG
half-float table, as a DataTexture), so Physical extensions the core path
lacks (clearcoat, sheen, iridescence, specular color/intensity, IOR) render
through ShaderLib.

PMREM environments (`three_program_pmrem.mjs`): MeshStandard/Physical (and
Lambert/Phong reading `scene.environment`) get r186's cube-UV map. The
application's own `PMREMGenerator` runs unchanged (retained JavaScript) against
a recording host; its CubemapToCubeUV / EquirectToCubeUV / GGX passes are
compiled with `rows: 'gl'` (render targets keep GL's bottom-up rows, so the
receiver's `textureCubeUV` reads them with GL's uv convention) and replayed in
order on rgba16float targets, each draw with its own viewport, scissor and
uniform slice. Cache semantics follow `WebGLEnvironments.getPMREM`: generated
once the source is complete (until then the program renders without the
environment, as r186 does), never regenerated on source edits, released on
the source's `dispose`. Scenario `shaderlib_pmrem` (cube and equirect sources,
roughness 0.05–0.8, Physical clearcoat) matches upstream WebGLRenderer (0.01%
of pixels differ; mean channel difference 0.01). This runs H2
`webgl_marchingcubes` "shiny".

Shadow maps (`three_program_shadows.mjs`): a retained-JavaScript port of r186
`WebGLShadowMap` (map allocation, shadow cameras and per-face frustums, caster
culling, `getDepthMaterial` with custom depth/distance materials and per-material
clones, side/shadowSide, onBeforeShadow/onAfterShadow in source order). Caster
draws are the MeshDepthMaterial / MeshDistanceMaterial ShaderLib programs,
compiled with `rows: 'gl'` into depth32float maps (point lights: one layer per
cube face); receivers bind them as `sampler2DShadow` / `samplerCubeShadow`
(PCF: less-equal comparison, linear) or unfilterable `sampler2D`/`samplerCube`
(Basic), from the light state by uniform name as `WebGLRenderer.setProgram`
does. The WebGL surface switches to this owner when the core's single projected
map cannot own the scene (more than one shadow light, point-light shadows, or a
program draw that casts/receives); then every lit material draws its ShaderLib
program. Scenario `shaderlib_shadows` (directional + spot + point cube map,
Phong/Lambert/Standard receivers) matches upstream WebGLRenderer exactly (0
differing pixels); `webgl_shadowmap_pointlight` runs. Maps are depth32float
where r186 allocates 24-bit depth. Explicit errors: VSM, line/point casters,
wireframe casters; a new light, map-size or type change is a `prepare()`
boundary.

Clipping planes: r186 `WebGLClipping.setState` per draw (`threeClippingState`):
renderer planes first, then material planes when `localClippingEnabled`, in the
draw camera's view space; `clipIntersection` sets UNION/intersection counts;
shadow passes drop global planes and apply local ones only with `clipShadows`.
ShaderMaterial gets the uniform only with `clipping: true`, as upstream.
`alphaToCoverage` enables the pipeline's alpha-to-coverage on multisampled
targets (GL ignores it on single-sampled ones). Scenario `shaderlib_clipping`
matches upstream WebGLRenderer exactly (0 differing pixels);
`webgl_clipping`, `webgl_clipping_advanced`, `webgl_clipping_intersection` run.

Not admitted yet (explicit errors): transmission (needs its render target),
render-target PMREM sources (`pmremVersion`), pre-filtered cube-UV textures
supplied directly, Sprite materials.

Textures: the texture owner now admits HalfFloat (`*16float`) and Float
(`*32float`) DataTextures with byte-exact rows. Float32 linear filtering
requires the device's `float32-filterable` feature (requested, like r186
WebGPUBackend, whenever the adapter has it); without it nearest-sampled float
textures bind as unfilterable and linear ones fail explicitly.

Evidence: scenarios `shaderlib_maps` (340 of 76,800 pixels differ slightly)
and `shaderlib_physical` (166 differ slightly) vs upstream WebGLRenderer;
`helpers_lines_points` now draws real point sizes.
