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
| Uniform buffer | group 0, binding 0 | Every non-sampler uniform of both stages. `reflection.uniforms[*].node` gives each leaf's byte offset/stride. `bool` → `u32`; scalar/`vec2` arrays → `vec4` elements; structs `@align(16)`. Two-row matrix arrays use aligned wrappers with their original 8-byte column stride. Internal `f3d_target` (framebuffer width, height) at `reflection.targetOffset` when `gl_FragCoord` is read. |
| Textures | group 1, binding 2k / 2k+1 | Texture / sampler of the k-th sampler uniform (arrays expand per element; ESSL already requires constant indices). Shadow samplers → depth texture + comparison sampler. |

Attributes keep declaration order (`layout(location)` honored; matrices take one
location per column). Varyings link by name; arrays/matrices take one location
per element/column; integer varyings are flat. ESSL 3.00 struct varyings flatten
numeric scalar/vector members and matrix columns into consecutive locations,
then reconstruct the original private struct before fragment code executes.
Generated slot names avoid collisions with application varying names, and field
access retains WGSL identifier escaping. Reflection keeps the source struct type
and first location. Smooth, flat and centroid interpolation survive this mapping;
centroid is also preserved for ordinary varying declarations.

Struct interface validation follows ESSL 3.00 sections 4.3.4 and 4.3.6: vertex
outputs and fragment inputs may be structs, while vertex attributes and fragment
outputs may not. Structs may contain numeric scalars, vectors and matrices, but
not arrays, nested structs, booleans or opaque members; arrays of structs are
also invalid. An integer member requires the whole varying to be flat. Matching
struct interfaces must agree on type and interpolation qualification. These
checks reject source-invalid declarations; ESSL 1.00 struct varyings are not
admitted. The source rules are in the
[Khronos ESSL 3.00 specification](https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf).

## Semantics preserved

- Clip depth: `gl_Position.z` remapped from GL `[-w, w]` to WebGPU `[0, w]`
  (`clipDepth: 'webgpu'` disables it for WebGPU-convention projections).
- `rows: 'gl'` (render targets sampled later by GL-convention shaders): clip Y
  is mirrored so storage rows are GL's bottom-up rows; `gl_FragCoord` and `dFdy`
  then need no flip; the caller swaps the pipeline's front face.
- `gl_FragCoord` in GL window coordinates (bottom-left origin); `dFdy` keeps GL's
  upward sign; `texture()` outside fragment shaders samples level 0.
- Floating-point 2D/3D projected sampling supports implicit LOD, bias, explicit
  LOD and explicit gradients, with or without texel offsets. Only the texture
  coordinates are divided by the final projective component; explicit gradients
  are already projected, and a 2D `vec4` coordinate ignores its third component.
  Offset sampling also supports 2D arrays without changing their layer index.
  Vertex implicit sampling uses level zero; bias is fragment-only. Named local
  constant offsets retain WGSL `const` bindings, including when other arguments
  have side effects. `texelFetchOffset` preserves float, signed and unsigned
  payloads for 2D, 3D and array textures. The legacy `texture2DProjGradEXT` alias
  uses the same projected-gradient lowering.
- ESSL 3.00 `sampler2DShadow` supports `textureProj` and `textureProjOffset`.
  The complete `xyz` coordinate, including the comparison reference, is divided
  by `w` exactly once. Fragment code uses native comparison sampling; vertex
  code compares at the base level without implicit derivatives. Constant texel
  offsets, sampler arrays and nested sampler helpers retain their source inputs.
  Comparison bindings carry a `depthReferenceClampOffset` control in a 16-byte
  uniform slot. Fixed-point source depth clamps the comparison reference to
  `[0, 1]`; floating depth leaves it unchanged, including when both use an
  oriented `depth32float` sampling image. The bound image owner supplies the
  flag, so a later public texture edit cannot change an accepted image's domain.
- ESSL 3.00 `sampler2DShadow` also supports `textureLod`,
  `textureLodOffset`, `textureProjLod` and `textureProjLodOffset` in
  vertex and fragment stages when the float LOD is proven constant zero.
  The shared lowering uses `textureSampleCompareLevel`, preserving the
  comparison sampler, independent min/mag filters, per-image reference domain,
  single coordinate evaluation and projected reference division. Signed zero,
  scalar float constructors and safe constant aliases are admitted; nonzero
  or runtime levels, arbitrary arithmetic and unsigned-derived values are
  refused. Offsets retain the existing constant integer-expression checks.
- Signed and unsigned integer 2D, 2D-array and 3D samplers implement nearest
  `texture`, `textureLod` and `textureGrad`, including their constant-offset
  variants, through typed `textureLoad`, preserving all 32 bits. Integer 2D/3D
  samplers also support all six projected forms with implicit LOD, explicit LOD
  or gradients, with or without offsets. Projective coordinates divide once;
  explicit gradients are already projected. Fragment `texture`, `textureOffset`,
  `textureProj` and `textureProjOffset` also accept a float bias proven within
  `[-2, 2]`: literals with signs/parentheses, scalar `float` constructors, safe
  `const` aliases, and explicit built-in `clamp(value, low, high)` with ordered
  closed constant bounds. The proof excludes arbitrary arithmetic and
  unsigned-derived values, and respects shadowed built-ins and mutable aliases.
  Bias is added once to derivative LOD after projection; coordinates and bias
  expressions retain single evaluation and source argument order. OpenGL ES 3.0
  table 6.28 guarantees this range; larger bias limits are implementation-dependent
  and have no matching WebGPU capability. Vertex implicit sampling uses level
  zero. Offsets apply in the selected mip's texel coordinates before wrapping,
  without changing an array layer. Helpers apply clamp/repeat/mirror addressing,
  rounded array layers, and explicit or derivative-based mip selection, following
  [ESSL 3.00 section 8.8](https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf)
  and [OpenGL ES 3.0 section 3.8.10](https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf).
  Sampler arrays and function
  parameters carry independent controls in 16-byte uniform slots at each
  texture's `samplingOffset`. The admitted binding's frozen sampler descriptor
  supplies those controls; retained direct draws keep their accepted bytes.
  Mip minification uses the floor alternative allowed by ES 3.0 equation 3.23
  footnote 4, matching the tested ANGLE path; the base-level half tie stays at
  zero. This does not close the automatic texture owner's pre-existing gap for
  public sampler-property edits made without `needsUpdate`.
- `gl_DepthRange` is the GL default `{near: 0, far: 1, diff: 1}`: r186 never
  calls `gl.depthRange`, and pipelines record viewports with depth 0..1.
- Vector `==`/`!=` reduce with `all`/`any`; `mod` uses `x - y*floor(x/y)`;
  shift counts become `u32`; `inverse`, `matrixCompMult`, `outerProduct` are
  generated helpers.
- Geometric functions retain ESSL's scalar and floating-vector signatures.
  Scalar `normalize` uses `sign`, including in constant expressions; scalar
  `faceforward`, `reflect` and `refract` use cached, typed value helpers because
  WGSL only provides vector overloads. Each argument runs once. Refraction
  returns zero before taking a square root when its discriminant is negative.
  Vector calls retain the corresponding native WGSL builtins. Integer inputs,
  mixed vector widths, non-`vec3` cross products and vector refraction ratios
  are rejected according to the source signatures.
- ESSL 3.00 whole-array and struct equality compare all corresponding elements
  and fields, including nested structs, arrays, matrices and booleans. Cached
  typed helpers take each operand by value once; matrix-producing expressions
  no longer repeat for every column. Arrays use loops rather than expanded
  comparisons. `!=` negates the same comparison, and surrounding `&&`/`||`
  retain their short-circuit behavior. Generated helper calls in global constant
  initializers use the existing entry initialization path, before `main`.
- ESSL 1.00 struct equality is supported when the structure has no array or
  opaque members. Array equality requires ESSL 3.00. These version rules follow
  section 5.7 of the respective Khronos specifications. Existing uniform layout
  conversion feeds comparisons; wrapped boolean-vector arrays now preserve all
  components for both whole-array and indexed reads.
- Uniform arrays of `mat2`, `mat3x2` and `mat4x2` align their first element to
  16 bytes and use 16-, 32- and 32-byte element strides respectively. Their
  matrix columns remain 8 bytes apart; padding never becomes matrix data.
  Indexed access, whole-array values, function arguments and nested uniform
  structs unwrap to the original GLSL matrix type. Reflection drives the
  ordinary uniform packer, including live edits without recompiling. The
  rectangular types also work through source `UniformsGroup` blocks supplied
  as typed std140 data, including conversion from 16-byte source columns to
  8-byte WGSL columns. Pinned r186 `WebGLUniforms` only supplies ordinary
  material upload setters for square matrix types.
- Global stage I/O becomes private variables copied by the entry points.
- Assignment and increment destinations capture every dynamic subscript once,
  before the right-hand expression. Their expression results read the same
  captured destination. Nested struct/array/matrix and swizzle destinations
  retain the selected original storage, including indexing a swizzled vector.
- User-function arguments evaluate once from left to right: `in` copies a value,
  `out` captures a destination without reading its old value, and `inout` copies
  both. Copy-back reuses the captured destination even if a later argument or
  the function body changes its index. `modf` applies the same destination
  rule. Hoisted builtin/constructor effects preserve earlier argument values
  and their order; constant aggregate indices remain constant expressions.
  Output-parameter copy-back order for aliased destinations is unspecified by
  [ESSL 3.00 section 6.1.1](https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf).
  Side-effecting `?:`, `&&`, `||`, `++`/`--` and assignments inside expressions
  lower to statements; multi-component swizzle stores split into component writes.
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
integer cube sampling and integer bias outside the proved portable range, `gl_FragData`, token pasting, type mismatches (ESSL has no implicit
conversions). Shadow sampling with bias, explicit gradients, or LOD outside the
proved constant-zero 2D profile remains unsupported; those forms are not
approximated with implicit sampling or an arbitrary level.

## Evidence

- `essl_shadow_sampling.test.mjs` and `essl_shadow_lod.test.mjs` cover
  comparison signatures, zero-LOD proof, projective reference division,
  constant offsets, single evaluation, helper/array propagation, mixed
  integer/depth metadata and public ShaderMaterial preparation. The focused
  compiler suites passed **29 tests**. `three_program_depth_sampling.test.mjs`
  separately exercises real target ownership and comparison bindings.
- `program_texture_sampling.mjs` passed **112 exact browser checkpoints**
  (**229,376 RGBA bytes**), including 48 existing cases and 64 new explicit-LOD
  cases: four forms, both shader stages, fixed/floating depth and four independent
  min/mag filter pairs. The new cases use interior comparison references;
  source assertions establish visible filtering and vertex offset effects.
  The run used pinned r186 `148ef33ecb6d2502ff796d4554abd1549c95d519`,
  Chromium 153.0.8010.0/SwiftShader and the `general-webgpu` route, with zero
  GPU or browser errors. This also executes the six earlier projected-shadow
  cases whose previous browser launch was blocked.
- A **separate endpoint diagnostic is not exact** on that source driver:
  with fixed-point stored depth 1 and reference 1.25, source implicit and
  explicit samples both return 0; candidate implicit and explicit samples
  both return 255 (512 differing red bytes). The existing candidate clamp is
  retained, consistent with [OpenGL ES 3.0 §3.8.15](https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf).
  The runner reports the discrepancy and checks agreement between implicit and
  explicit operations within each renderer; it does not include these bytes in
  the exact claim or relax the ordinary pixel comparisons. This pre-existing
  endpoint difference remains a compatibility gap. The new lowering follows
  [WGSL base-level comparison semantics](https://www.w3.org/TR/WGSL/#texturesamplecomparelevel).

`essl_geometric.test.mjs` covers the typed scalar/vector boundary, stage-local
helper reuse, constant normalization and one evaluation of helper arguments.
`node tests/e2e/webgpu_renderer/program_geometric.mjs` compares 53 checkpoints
against pinned r186 WebGL: scalar normalization, faceforward, reflection and
refraction in ESSL 1.00/3.00 vertex and fragment programs, total internal
reflection and its boundary, constants, side effects and vector regressions.
Chromium 153.0.8010.0 / SwiftShader matches all 27,136 RGBA bytes exactly with
zero GPU/browser errors. These are software GPU correctness checks.

- `essl_lvalues.test.mjs` has ten focused regressions for destination capture,
  function argument value/address order, nested swizzles, `modf`, stored versus
  computed aggregates, constant offset/switch expressions and invalid targets.
  `program_lvalues.mjs` exercises 23 programs in both vertex and fragment stages.
  On Chromium 153.0.8010.0 / SwiftShader, 40 checkpoints match pinned r186 WebGL
  exactly (5,120 RGBA bytes); all 46 candidate checkpoints match their specified
  expected values (5,888 bytes), with zero GPU/browser errors.
  Six checkpoints remain explicit source differences: a later argument mutating
  an earlier `inout` value gives source 73 versus specified 13; a `modf` output
  index mutating its earlier input gives whole/fraction 99/0 versus 14/0.25;
  `combine(x, ++x, x++)` gives 46 versus 26 for the fixture's initial value and
  function. Each occurs in both stages. Raw WebGL2, without Three's renderer,
  reproduces the source values. The compiler retains the once-only, left-to-right
  argument-copy rules of ESSL 3.00 section 6.1.1. The runner accepts either a
  conformant source result or exactly the observed driver alternative, records
  actual differences separately, and never counts them as source parity. This
  driver/source equivalence limitation remains open; these results do not claim
  complete ESSL parity or hardware performance.
- `essl_texture_sampling.test.mjs`: six focused tests cover projected/offset
  signatures, projective divisors, already-projected gradients, source constant
  offsets, argument side effects, vertex LOD and typed texel fetches. The new
  tests and existing ESSL/integer-sampling suites pass all 47 tests.
  `program_texture_sampling.mjs` compares 42 pinned WebGL/WebGPU checkpoints
  exactly (86,016 RGBA bytes), including bias-selected mips, 3D/array offsets,
  signed/unsigned texels, vertex sampling and the r186 legacy gradient alias.
  The candidate uses `general-webgpu`; Chromium 153 SwiftShader reports no
  browser or GPU errors. These are software GPU correctness checks, not a
  hardware performance result or complete ESSL closure.
- `essl_integer_bias.test.mjs`: five focused tests cover signed/unsigned 2D,
  3D and array bias families, projected/offset signatures, portable bounds,
  const aliases, built-in shadowing, uint wrap refusal, closed clamp bounds,
  vertex rejection and argument effects. This suite and three adjacent sampling
  suites pass all 23 tests.
- `essl_integer_projected_sampling.test.mjs`: six focused tests cover valid
  signed/unsigned overloads, typed payloads, projection and gradient dimensions,
  constant offsets, source argument order, sampler arrays/helpers and vertex LOD.
  The new suite and three adjacent compiler/sampling suites pass all 53 tests.
  `program_integer_projected_sampling.mjs` compares 220 pinned WebGL/WebGPU
  checkpoints exactly (450,560 RGBA bytes), including 2D/3D/array offsets,
  projected `vec3`/`vec4` coordinates, independent sampler controls, side effects,
  portable-bias literals, clamp limits, biased mip selection, bias argument
  effects and vertex sampling. Exact reflected texel boundaries and large
  finite positive/negative coordinates retain the observed source addressing,
  including floating-point loss of a small offset at large coordinates.
  Chromium 153.0.8010.0 / SwiftShader reports `general-webgpu` and zero GPU/browser
  errors. These are software GPU correctness checks, not complete ESSL closure.
- `three_program_integer_sampling.test.mjs`: six focused lowering, layout,
  signature and bound-sampler metadata regressions. The existing 32 compiler
  and 25 texture-owner tests also pass. `program_integer_sampling.mjs` compares
  all 50 source/candidate checkpoints exactly (307,200 RGBA bytes), including
  high-bit integer values, wraps, mip boundaries, explicit gradients, arrays,
  volume slices and vertex sampling. Both sides use pinned r186; the candidate
  reports `general-webgpu` with native draws. Re-running this suite alongside
  the projected/offset extension on Chromium 153.0.8010.0 / SwiftShader again
  matched all 50 checkpoints with no GPU/browser errors. This is software GPU correctness evidence,
  not a physical-device performance result or complete ESSL closure.
- `essl_wgsl.test.mjs`: lowering and layout regressions. Switch tests execute
  emitted scalar helper control flow through a small test-only JS syntax adapter
  and check loop targets and scope diagnostics. These checks do not validate
  WGSL or establish rendered parity; Naga/browser validation remains separate.
- Matrix-array regressions cover all nine matrix dimensions, neighboring scalar
  uniforms, nested structs, whole-array reads and single evaluation of dynamic
  indices. `three_program.test.mjs` checks global `mat2` arrays and live edits
  through the material packer. Its nested struct-array values are compiler/packer
  coverage only: this nested fixture has not established source pixel parity.
  A separate compiler-level packing
  test checks all six `mat3x2` components and its trailing padding. A production
  `UniformsGroup` regression exercises rectangular arrays from typed std140
  data and preserves the source's once-per-frame upload visibility.
  `tests/e2e/webgpu_renderer/program_matrix_arrays.mjs` compares the pinned
  WebGL renderer with the new WebGPU renderer for global `mat2[]` values and
  typed std140 `mat3x2[]` blocks. All sixteen Scene/fullscreen Mesh checkpoints,
  including live edits and queued group disposal/reuse, match exactly (6,144
  RGBA bytes) on Chromium 153 SwiftShader, with no browser or GPU errors.
  Fullscreen captures preserve linked ordinary-uniform dictionary identity and
  share the source block's once-per-frame upload history and disposal generation.
  Rectangular block reads use direct dynamic indexing; the
  local whole-array conversion remains covered by compiler/packer tests.
  This is software GPU correctness evidence, not hardware or speedup evidence.
- Struct-interface regressions execute the generated pack/unpack assignments
  with distinct scalar, vector and matrix-column values, and check locations,
  smooth/flat/centroid interpolation, identifier collisions, independent stage
  compilation and source/link errors. These tests do not execute a GPU or prove
  interpolation pixel parity.
- Aggregate-comparison regressions execute generated helper control flow and
  member/index accesses with a small test-local vector equality adapter. They
  check nested values, short-circuiting, operand call counts, exceptional scalar
  comparisons, constant initialization, uniform vector widths and bounded array
  code size. The focused `aggregate equality: live ShaderMaterial` test in
  `three_program.test.mjs` also compiles a real pinned ShaderMaterial, packs live
  struct/array/bool uniform edits and compiles an unseen runtime inequality
  variant. It does not execute the generated WGSL or establish GPU parity.
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
source `ShaderMaterial.uniforms` or a fresh ShaderLib clone. A replacement
`parameters.uniforms` dictionary becomes the renderer's active uniform owner
without replacing the public `material.uniforms`; added numeric values and
samplers reach the program. Selecting an older cached variant retains the latest
dictionary, matching the source's assignment boundary. Material disposal clears
the hook result and dictionary so later reuse recompiles the hook; renderer,
target and PMREM teardown release those caches and their listeners without
disposing application textures. Callback, source-assembly and translation errors
do not publish a replacement or cache a failed hook result.
`three_program.test.mjs` checks these ownership transitions against the unchanged
pinned `WebGLRenderer.getProgram` and disposal functions, with GL program
allocation recorded, and checks actual packed values and sampler identities.
Those checks do not execute a GPU or establish pixel parity.
`material.onBeforeRender` runs at the
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
