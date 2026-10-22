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
morphed meshes; Sprite; Line/LineSegments/Points (one-pixel, as upstream WebGPU; LineLoop is
reported and not drawn, as upstream); Basic/Lambert/Phong/Toon/Standard and
Physical with the specular and clearcoat profiles below,
LineBasic and Points materials; all source blending modes, premultiplied alpha,
polygon offset; ambient/hemisphere/directional/point/spot lights (up to 8); fog;
one directional/spot shadow; HDR environment/background panoramas; clipping
planes; textures with transforms.

Native `MeshMatcapMaterial` and default `MeshMatcapNodeMaterial` support generated
view-space Matcap coordinates, the source fallback ramp, independent color,
normal and alpha maps, and live Matcap texture transforms. Matcap textures do not
need a geometry UV attribute; their alpha and channel selector are ignored by the
pinned source path. See [THREE_MATCAP.md](./THREE_MATCAP.md).

Native sprites accept `SpriteMaterial` and default `SpriteNodeMaterial`, with
camera-facing transforms, live center/rotation, size attenuation, map/alpha-map
textures and ordered transparent repeats from `Sprite.count`. ArrayCamera views
receive their own billboard matrices. See [THREE_SPRITES.md](./THREE_SPRITES.md)
for the supported source behavior and explicit custom-node/hook limits.

Authored native TSL graphics graphs now execute through the same new GPU owner.
The initial profile accepts rigid meshes with `MeshBasicNodeMaterial` or unlit
`NodeMaterial`, including position/vertex, color/fragment, opacity, alpha-test and
output graphs. The pinned `WGSLNodeBuilder` supplies real native shaders and
uniform layouts; F3D owns geometry, pipelines, bind groups, draw ordering and
submission. Float attributes, scalar/vector/matrix uniforms and ordinary sampled
2D texture graphs are supported. Graphics and compute share captured texture
versions; native target sampling borrows the actual attachment storage.
The existing fixed shader path continues serving default node materials.

Each initialized native node render captures its program, visible draw selection,
raster state, ranges, uniforms and texture versions before asynchronous preparation. Source
NodeFrame before/current/after updates run once at that logical call boundary,
preserving frame/render/object frequencies and source object identities. Two
queued target renders can therefore reuse a material and change its uniforms
without rewriting the earlier render. Compilation does not invoke those updates.
See [THREE_NODE_PROGRAMS.md](./THREE_NODE_PROGRAMS.md) for the initial profile,
resource bounds and remaining texture-family/storage/lighting graph gaps.

Standard and Physical materials, including their default node-material forms,
accept independent `metalnessMap` and `roughnessMap` textures with their own
UV channels and live transforms. Either can be used alone. Shared textures use
the combined binding; replacing a map or changing its channel triggers the
ordinary preparation path. Metalness reads blue and roughness reads green,
matching pinned r186.

Native `MeshPhysicalMaterial` and `MeshPhysicalNodeMaterial` with null node slots
support IOR, specular intensity/color and their two independent maps, with
intensity from alpha and color from RGB. Factors remain live per draw; direct
lighting and ambient diffuse energy use the pinned r186 physical equations.
They also support clearcoat, clearcoat roughness and their three independent factor,
roughness and normal maps. Factors, map transforms and XY normal scale remain
live; changing a map binding/channel or enabling/disabling the coating takes the
ordinary preparation path. The native r186 roughness map reads its red channel.
The dielectric coating uses the pinned GGX/Fresnel layering and leaves emission
unattenuated. The existing environment receiver keeps its documented radiance
profile. Other physical extensions remain outside this implementation; see
[THREE_SCENE.md](./THREE_SCENE.md).

Ordinary `ArrayCamera` rendering is admitted on this native WebGPU route, with
at most 256 non-reversed perspective or orthographic child cameras. The parent
selects layers and sorting order from the union of child frustums; each object
then draws through its layer-compatible children in source order. Child
viewports use top-left coordinates, logical canvas units scaled by DPR, and
physical pixels on render targets. Color/depth/stencil clear once per render.
Texture backgrounds respect child viewports and layers; 2D background UVs span
the full target. Directional backgrounds retain their perspective-camera limit.

Each `render()`/`renderAsync()` captures the array's child identities, matrices,
layers and viewports at call time, so deferred preparation cannot replace an
earlier request's views with later camera edits. Public matrix and projection
updates follow the pinned native renderer. This does not admit nested arrays,
multiview/XR cameras, or ArrayCamera on the WebGL-authored route. Custom native
scene/object/material render hooks retain their existing explicit rejection.

## Explicit differences

- Structural edits (new objects/materials, material structure, enabling fog,
  environment, background or shadows) need asynchronous preparation. `render()`
  then presents nothing (the canvas keeps its last image) and the pending
  requests are submitted in order when preparation completes. Counted in
  `info.f3d.deferredRenders`; call
  `compileAsync(scene, camera)` to prepare ahead of the first frame.
- Admission errors found during deferred preparation throw from the next
  `render()`/`compileAsync()`.
- Not admitted (explicit `F3DRendererError`/`ThreeSceneError`):
  XR, custom tone mapping, logarithmic/reversed depth, non-sRGB
  output, node graphs outside the initial unlit graphics profile, the remaining physical-material extensions,
  MeshDepth materials,
  transparent clears with tone mapping.
- Upstream passes non-one factors with Min/Max blend equations, which WebGPU
  rejects; this renderer keeps the GL meaning.

## Evidence

- `node --test tools/ingest/three_renderer.test.mjs tools/ingest/renderer_route_build.test.mjs`
  (recording device: command/queue behavior).
- `node tools/ingest/three_array_camera.test.mjs` — pinned-source camera state,
  object/view ordering, layers, DPR/target viewports, deferred snapshots, fog,
  texture backgrounds and final MSAA resolve on a recording device.
- `node tools/ingest/three_physical_clearcoat.test.mjs` — native physical and
  node-material coating bindings, live uniforms/maps and preparation transitions.
- `node tests/e2e/webgpu_renderer/parity.mjs` — pixel comparison with pinned
  upstream `WebGPURenderer` in headless Chromium on SwiftShader. In that
  SwiftShader build, MSAA resolves into a reinterpreted `-srgb` canvas view lose
  sRGB encoding even for raw WebGPU code, so parity runs single-sample.
