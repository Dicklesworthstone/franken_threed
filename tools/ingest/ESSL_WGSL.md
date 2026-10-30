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
- `gl_DepthRange` is the GL default `{near: 0, far: 1, diff: 1}`: r186 never
  calls `gl.depthRange`, and pipelines record viewports with depth 0..1.
- Vector `==`/`!=` reduce with `all`/`any`; `mod` uses `x - y*floor(x/y)`;
  shift counts become `u32`; `inverse`, `matrixCompMult`, `outerProduct` are
  generated helpers.
- Global stage I/O becomes private variables copied by the entry points.
- `out`/`inout` arguments copy through temporaries in source evaluation order;
  side-effecting `?:`, `&&`, `||`, `++`/`--` and assignments inside expressions
  are lowered to statements; multi-component swizzle stores are split.
- Nonempty switch fallthrough uses reachable source-case suffixes in native WGSL
  cases. The selector runs once; conditional breaks, defaults between labels,
  return/discard and continue targets retain source control flow. A nested loop
  or switch consumes its own breaks. Empty labels merge; terminating cases are
  emitted once. Additional duplicated source is bounded at 1 MiB per switch.
- Switch labels share their source declaration scope. Mutable storage is hoisted
  with unique names, while initializer effects stay inside the original case.
  Named compile-time constants remain available from later labels. Explicit
  blocks keep their own scope, and later declarations do not become visible
  before their source location.
- Implicit-derivative sampling in non-uniform control flow keeps GL's undefined-
  in-divergent-quads behavior by disabling WGSL's `derivative_uniformity`
  diagnostic. That is the same contract ESSL gives, not an approximation.

## Explicit errors (not silently dropped)

`gl_PointCoord` outside point programs, `gl_ClipDistance`,
filtered sampling of integer textures, struct stage I/O, `gl_FragData`, token pasting, type mismatches (ESSL has no implicit
conversions).

## Evidence

- `essl_wgsl.test.mjs`: lowering and layout regressions. Switch tests execute
  emitted scalar helper control flow through a small test-only JS syntax adapter
  and check loop targets and scope diagnostics. These checks do not validate
  WGSL or establish rendered parity; Naga/browser validation remains separate.
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
environment, as r186 does), ordinary textures are never regenerated on source edits, released on
the source's `dispose`. Scenario `shaderlib_pmrem` (cube and equirect sources,
roughness 0.05–0.8, Physical clearcoat) matches upstream WebGLRenderer (0.01%
of pixels differ; mean channel difference 0.01). This runs H2
`webgl_marchingcubes` "shiny".

GPU-authored equirectangular and cube render targets feed that filtering path without
CPU readback. Their `needsPMREMUpdate`/`pmremVersion` changes regenerate the same
cube-UV allocation and retain its sampler/view identity; ordinary texture upload
versions do not trigger a bake. Source-disposal handling retains r186's branch
asymmetry: an initial versioned target bake does not install the ordinary cache
disposal listener. An input binding is acquired once for the whole
generation, kept alive across pipeline preparation and the submitted queue
fence, and validated before submission and publication. Supplied cube-UV atlases
are sampled directly through the existing texture owner. The WebGL-authored
route retains its existing live-scene preparation contract; this does not add
immutable snapshots of arbitrary queued scenes.

`WebGLCubeRenderTarget` uses the existing native target owner, six color layers,
source face selectors and explicit face readback. Retained `CubeCamera` observes
`WebGLCoordinateSystem`, including its original face directions/up vectors; queued
cube renders retain their camera poses. GPU sampling flips every face and mip
into the existing UV replica and exposes a cube view to `samplerCube`, direct
environment maps and PMREM. Automatic mip generation follows each accepted
base-level render's source flag, so a cold CubeCamera capture filters after its
sixth face. The owner reserves the full immutable pyramid without exposing
unallocated source levels. WebGL resize uses target extents; native cube allocation
semantics remain separate. Cube storage and its replica retain the existing
budget, feedback checks and asynchronous program leases.

`three_renderer_webgl_cube.test.mjs` checks allocation, generation order, sampling,
resize, leases and PMREM reuse. Browser `program_cube_targets.mjs` compares twelve
source/candidate checkpoints (172,032 RGBA bytes) in Chromium 151/SwiftShader:
all six face readbacks, base cube sampling and resized sampling are byte-identical;
LOD 2 differs by at most one channel level; retained equirectangular conversion
by at most six. Each of two changing PMREM checkpoints has 8 of 6,144 pixels over
channel-12 tolerance (maximum 79/83), within the existing 2% pixel budget.
There is no hardware performance claim.

`three_program_pmrem_targets.test.mjs` exercises versioned reuse, public renderer
integration and asynchronous input lifetimes. `program_pmrem_targets.mjs` compares
five source/F3D checkpoints: changed GPU panorama with and without an invalidation
request, two requested regenerations and a supplied atlas. Chromium 151/SwiftShader
passes the existing WebGL channel-12/pixel-2% budget: 8 of 6,144 pixels exceed the
channel threshold in each panorama checkpoint; the direct atlas differs by at
most one channel level. No hardware performance claim.

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

Backgrounds: on the WebGL surface, texture backgrounds are r186
`WebGLBackground.addToRenderList` meshes drawn as programs (the plane with
`ShaderLib.background` for 2D textures; the camera-centred box with
`ShaderLib.backgroundCube` for cube and cube-UV textures, PMREM-blurred when
`backgroundBlurriness > 0`), unshifted ahead of the sorted opaque list, with
intensity/rotation/uvTransform and `toneMapped` from the texture's color space.
Scenarios `shaderlib_background_cube` / `shaderlib_background_texture` match
upstream WebGLRenderer (0.06% / 0% of pixels differ); `webgl_materials_envmaps`,
`webgl_materials_cubemap`, `webgl_materials_cubemap_refraction`,
`webgl_materials_blending`, `webgl_effects_stereo` run. Equirectangular
backgrounds without blur, and equirect envMaps on Basic/Lambert/Phong, use
r186 `WebGLEnvironments.getCube`: the application's own
`WebGLCubeRenderTarget.fromEquirectangularTexture` (CubeCamera's six face
renders of the CubemapFromEquirect program, recorded by the same host as PMREM,
GL rows, sRGB-encoded storage for 8-bit sRGB sources, render-pass box-filtered
mipmaps when requested). Scenario `shaderlib_equirect_cube` matches upstream
WebGLRenderer (0 differing pixels beyond tolerance; mean diff 0.01).
`webgl_loader_gltf` (HDR equirect background + environment, ACES) and
`webgl_materials_envmaps_exr` run; mipmap filtering is GPU box filtering where
GL's generateMipmap is implementation-defined.

Tone mapping (WebGL surface): r186 WebGLRenderer tone-maps inside each
`toneMapped` material's program, before blending and sRGB encoding, not with a
whole-image pass. With `renderer.toneMapping` set, the WebGL facade now keeps
its direct shader-encoded canvas and every draw is its ShaderLib program
(shadows, scene.environment through PMREM, backgrounds included); a scene with
a draw the program route cannot take (e.g. skinned or morphed meshes) raises
`THREE_SCENE_TONE_MAPPING` and the renderer falls back, from then on, to the
previous HDR target plus output pass (an approximation for blended and
`toneMapped: false` draws). Scenario `shaderlib_tone_mapping` (ACES, exposure,
transparent blend, `toneMapped: false`, shadows) matches upstream
WebGLRenderer exactly (0 differing pixels); `webgl_lights_physical`,
`webgl_loader_gltf_variants`, `webgl_loader_gltf_instancing`,
`webgl_loader_texture_ultrahdr` now run.

Renderer-route `PMREMGenerator` results: `fromScene` captures (e.g.
RoomEnvironment) and `fromCubemap` sources are generated by the application's
own `PMREMGenerator` at the bridge's preparation boundary. Scene captures render
through the offscreen host's scene path (projectObject culling, painter-sorted
lists, a capture-local WebGLLights state, depth24plus per face, instanced
meshes through program geometry residencies). On the WebGL surface programs
own these environments: scenario `room_environment` now matches upstream
WebGLRenderer (0 differing pixels; it was the known 4% approximation through the
core panorama owner). `webgl_materials_envmaps_hdr`, `webgl_pmrem_cubemap`,
`webgl_pmrem_equirectangular`, `webgl_loader_gltf_sheen`,
`webgl_loader_gltf_dispersion`, `webgl_lights_sunlight`, `webgl_furnace_test`
run. Captured scenes with shadows, textured backgrounds, environments or
texture maps other than the DFG LUT are explicit errors.

Skinning: skinned meshes draw their ShaderLib program with r186's bone texture
(`skeleton.computeBoneTexture()` on first use, texelFetch in the program),
`bindMatrix`/`bindMatrixInverse` from the object, and `skeleton.update()` once
per frame on the first visible or shadow-casting draw (WebGLObjects.update);
shadow casters use skinned depth programs. Programs take skinned meshes under
in-shader tone mapping (elsewhere the core deformation path still draws them).
Scenario `shaderlib_skinning` (AgX, program shadow map) matches upstream
WebGLRenderer exactly (0 differing pixels). Morph targets remain an explicit
error on programs only for InstancedMesh morph textures: WebGLMorphtargets is
ported (per-geometry Float32 DataArrayTexture of position/normal/color deltas,
`morphTargetBaseInfluence` / `morphTargetInfluences` / texture size per draw,
texelFetch by gl_VertexID), bound as an rgba32float 2D array; scenario
`shaderlib_morph` (absolute and relative targets, Neutral tone mapping) matches
upstream WebGLRenderer exactly. Parity now forbids the HDR fallback for program-route
scenarios (`F3D_NO_FALLBACK` test seam), so a silent fallback fails them.

Lights only the programs shade: `RectAreaLight` (WebGLLights picks the float LTC
tables when the device has float32-filterable, as r186 does with
OES_texture_float_linear, else the half tables; RectAreaLightUniformsLib.init()
is required, as upstream) and projected spot-light `map`s put every lit
material on the program route, as program shadow maps do. Scenario
`shaderlib_area_lights` matches upstream WebGLRenderer (mean diff 0.02);
`webgl_lights_rectarealight` and `webgl_lights_spotlight` run.

Shader hooks: `material.onBeforeCompile(parameters, renderer)` runs as in
WebGLRenderer.getProgram, once per program (keyed by the assembled pre-hook
program and `customProgramCacheKey()`), on parameters whose `uniforms` are the
material's uniforms object that is later refreshed and packed; added uniforms
and rewritten chunks reach the program. `material.onBeforeRender` runs at the
draw. Materials with either hook always draw their program. Scenario
`shaderlib_before_compile` matches upstream WebGLRenderer;
`webgl_materials_modified`, `webgl_shadowmap_csm` (CSM chunk rewrites) and
`webgl_materials_toon` run.

WebGL cube target admission currently requires base-level rendering, one sample,
no stencil and no array depth. Authored WebGL cube mip descriptors and sampling a mip-filtered
cube before source generation are explicit unsupported boundaries. Native cube
authored mip support retains its existing contract.

Textures: the texture owner now admits HalfFloat (`*16float`) and Float
(`*32float`) DataTextures with byte-exact rows. Float32 linear filtering
requires the device's `float32-filterable` feature (requested, like r186
WebGPUBackend, whenever the adapter has it); without it nearest-sampled float
textures bind as unfilterable and linear ones fail explicitly.

Evidence: scenarios `shaderlib_maps` (340 of 76,800 pixels differ slightly)
and `shaderlib_physical` (166 differ slightly) vs upstream WebGLRenderer;
`helpers_lines_points` now draws real point sizes.

## WebGL ArrayCamera views

The WebGL-authored renderer now admits ordinary `ArrayCamera` scenes through
the native scene and program owners. It builds and sorts the parent-camera
frustum/layer list once, clears the attachments once, and renders each child's
whole list in source order. Child viewports use physical bottom-left pixels,
including on a canvas with a nonunit device pixel ratio. Children without a
viewport inherit the preceding effective viewport, including the dimensions
used by program point-sprite expansion.

Object and material callbacks receive the original child camera. Object
model-view/normal matrices update before the material callback. Layer tests and
later child identity, projection and viewport reads occur at their source draw
boundaries, so earlier callbacks can affect later draws. Per-use program bytes
and fixed-draw world transforms remain immutable until their GPU submission.
Overlapping double-sided transparent views retain camera-major side order and
the source material-version changes. Empty arrays clear without a parent draw.
Explicit `ShaderMaterial.uniformsNeedUpdate` requests are consumed before the
object's after-render callback, preserving the source update flag lifecycle.

Twelve focused recording-device cases cover these contracts. The browser
comparison is `tests/e2e/webgpu_renderer/program_array_cameras.mjs` and checks
pixels, callback order, public matrices and side-version changes against the
pinned WebGLRenderer. Eight Chrome 151 SwiftShader checkpoints match all 224,256
RGBA bytes exactly, with identical callback traces, matrices and version deltas,
56 native draw calls and zero GPU/browser errors. Cases cover split/overlapping
views, child layers, parent culling, mixed projections, shared scissoring,
target resize and empty-array clearing. This is correctness evidence with no
performance claim. Native WebGPU ArrayCamera retains its existing eager
capture and object-major ordering.

WebGL ArrayCamera transmission and BatchedMesh are explicit boundaries pending
per-view transmission bindings and indirect-texture histories. XR/multiview and
the existing viewport/resource capacity restrictions also remain explicit.
