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

`gl_PointCoord` (point sprites need quad expansion), `gl_ClipDistance`,
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
