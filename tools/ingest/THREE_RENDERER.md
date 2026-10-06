# Drop-in `WebGPURenderer` on the new backend

`createWebGPURendererClass(THREE)` (`three_renderer.mjs`) returns a class with the
r186 `WebGPURenderer` public surface that ordinary applications use, executing
admitted frames through the source scene bridge (`three_scene.mjs`) on an owned
canvas. Route: **general-webgpu** — new WebGPU execution with retained JavaScript
command preparation. Not a Rust/Wasm specialization and not an acceleration claim.

## Building an application onto it

```sh
node tools/ingest/cli.mjs path/to/index.html --build-app out --route-webgpu-renderer
```

The build decides the route from the whole module graph with the existing route
decider. It substitutes only when every `WebGPURenderer` construction site routes
to general-webgpu, no exact-backend renderer or GL escape exists, and every import
of `build/three.webgpu.js` is the admitted pinned r186 build. The substituted
module re-exports the upstream namespace (every other class keeps its upstream
identity) and replaces only `WebGPURenderer`. Otherwise the build is unchanged
and the printed/returned `rendererRoute.reason` says why.

## Supported surface

`new WebGPURenderer({canvas, antialias, alpha, depth, samples, powerPreference,
requiredLimits, requiredFeatures, device})`, `init()`, `render()` (throws the source
error before `init()`), `renderAsync()`, `compileAsync()`, `setAnimationLoop()`,
`setSize()`, `setPixelRatio()`, `setDrawingBufferSize()`, size getters, clear
color/alpha/depth, `autoClear*`, `toneMapping` (fixed at `init()`),
`toneMappingExposure`, `shadowMap.enabled`, `info`, `dispose()`.

Scene content is whatever `three_scene.mjs` admits: Mesh/InstancedMesh/skinned and
morphed meshes; Line/LineSegments/Points (one-pixel, as upstream WebGPU; LineLoop is
reported and not drawn, as upstream); Basic/Lambert/Phong/Toon/Standard,
LineBasic and Points materials; all source blending modes, premultiplied alpha,
polygon offset; ambient/hemisphere/directional/point/spot lights (up to 8); fog;
one directional/spot shadow; HDR environment/background panoramas; clipping
planes; textures with transforms.

## Explicit differences

- Structural edits (new objects/materials, material structure, enabling fog,
  environment, background or shadows) need asynchronous preparation. `render()`
  then presents nothing (the canvas keeps its last image) and the pending
  requests are submitted in order when preparation completes. Callbacks are
  never skipped. Counted in `info.f3d.deferredRenders`; call
  `compileAsync(scene, camera)` to prepare ahead of the first frame.
- Admission errors found during deferred preparation throw from the next
  `render()`/`compileAsync()`.
- Not admitted (explicit `F3DRendererError`/`ThreeSceneError`): render targets,
  XR, custom tone mapping, stencil buffers, logarithmic/reversed depth, non-sRGB
  output, node materials, MeshPhysical/Normal/Matcap/Depth materials, sprites,
  transparent clears with tone mapping.
- Upstream passes non-one factors with Min/Max blend equations, which WebGPU
  rejects; this renderer keeps the GL meaning.

## Evidence

- `node --test tools/ingest/three_renderer.test.mjs tools/ingest/renderer_route_build.test.mjs`
  (recording device: command/queue behavior).
- `node tests/e2e/webgpu_renderer/parity.mjs` — pixel comparison with pinned
  upstream `WebGPURenderer` in headless Chromium on SwiftShader. In that
  SwiftShader build, MSAA resolves into a reinterpreted `-srgb` canvas view lose
  sRGB encoding even for raw WebGPU code, so parity runs single-sample.
