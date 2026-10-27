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

Construction routing follows local renderer and canvas aliases before choosing
an owner. Exports, external stores, returned renderer factories, unknown consumers,
dynamic member selection, native renderer state and custom subclasses retain the
upstream component. Unknown constructor options or externally supplied canvas
identities also stay upstream. Closed renderer calls, local aliases, locally
created DOM canvases and ordinary DOM attachment remain eligible. General
imported helpers remain opaque.

The graph can prove a bounded retained `OrbitControls` contract by verifying the
exact addon, resolved Three build and core bytes. The returned controls instance
continues to carry the canvas reference through local aliases and `domElement`;
it is not an exemption for arbitrary consumers. Closed updates, camera-control
methods, configuration and disposal remain eligible. Exports, unknown consumers,
event callbacks, private hooks, method replacement, reflection and unproved
constructor/receiver origins retain the upstream owner. Changed source bytes,
redirected base classes and unresolved code invalidate the proof. The original
H2 application is consequently eligible with `--route-webgl-renderer`, retaining
its actual OrbitControls and camera identities while using the new GPU backend.

H1 still retains the exact renderer: its Inspector Settings component replaces
`WebGPURenderer.prototype.init` to select a real WebGL backend from persisted
settings. Closing its controls edge does not close that native contract.
Direct new-backend tests establish renderer functionality separately from
automatic application routing and do not establish an H1 acceleration claim.

## Supported surface

`new WebGPURenderer({canvas, antialias, alpha, depth, samples, powerPreference,
requiredLimits, requiredFeatures, device})`, `init()`, `render()` (throws the source
error before `init()`), `renderAsync()`, `compileAsync()`, `setAnimationLoop()`,
`setSize()`, `setPixelRatio()`, `setDrawingBufferSize()`, size getters, clear
color/alpha/depth, `autoClear*`, `toneMapping`,
`toneMappingExposure`, `shadowMap.enabled`, `info`, `dispose()`.

Native `setOutputRenderTarget(target)` / `getOutputRenderTarget()` select a
single-color 2D target for final output. Working-linear output renders directly;
tone mapping or color conversion first renders into the pinned source's private
linear framebuffer, then executes its retained TSL output graph on the same new
WebGPU owner. Target dimensions, viewport, scissor, attachment precision and MSAA
come from the selected target. An explicit unrelated `setRenderTarget()` takes
precedence. Queued output passes keep their own target generations, tone operator,
exposure and render-frequency updates; `compileAsync()` prepares both passes.

Selection follows the pinned source, including its asymmetries: a converted
render leaves the effective output target selected by `getRenderTarget()`; direct
working-linear output keeps a null active selection. With a null active selection,
`clear()` clears the private framebuffer and draws its output when conversion is
needed, but otherwise clears the canvas. An explicit target clear never performs
the output transform. Cube, array, multiview and multiple-color output targets,
and output targets on the WebGL-authored facade, remain unsupported. See
[THREE_RENDER_TARGETS.md](./THREE_RENDER_TARGETS.md) for target profiles and lifetime.

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
The profile accepts rigid meshes with `MeshBasicNodeMaterial`, unlit
`NodeMaterial`, or authored Toon, Lambert and Phong node materials, including
position/vertex, color/fragment, opacity, alpha-test and output graphs. Lit
materials support authored normal/emissive nodes and Phong specular/shininess
nodes under ambient, hemisphere, directional, point and spot lights. The pinned
`WGSLNodeBuilder` supplies real native shaders and
uniform layouts; F3D owns geometry, pipelines, bind groups, draw ordering and
submission. Float attributes, scalar/vector/matrix uniforms and ordinary sampled
2D texture graphs are supported. Graphics and compute share captured texture
versions; native target sampling borrows the actual attachment storage.
Ordinary materials with bump/displacement/light maps or Physical sheen, iridescence,
anisotropy and retroreflectivity use those source shaders without authored node
slots. Other default node materials continue using the existing fixed path.

Each initialized native node render captures its program, visible draw selection,
raster state, ranges, uniforms and texture versions before asynchronous preparation. Source
NodeFrame before/current/after updates run once at that logical call boundary,
preserving frame/render/object frequencies and source object identities. Two
queued target renders can therefore reuse a material and change its uniforms
without rewriting the earlier render. Compilation does not invoke those updates.
Light selection follows source visibility and camera layers, while retained
analytic nodes preserve FRAME/RENDER light updates across objects and program
rebuilds. See [THREE_NODE_PROGRAMS.md](./THREE_NODE_PROGRAMS.md) for the profile,
resource bounds and remaining texture-family, writable/struct storage and advanced lighting gaps.

Native scalar/vector TSL storage reads now share compute's authoritative buffer
and upload history. Queued full and partial host uploads retain their own render
order, while unchanged host versions consume GPU-produced data without a stale
CPU refresh. Multiple host versions of one attribute within a single logical
render remain an explicit unsupported case; separate queued renders work.

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
profile. Sheen, iridescence, anisotropy and retroreflectivity instead select the
retained native node builder, including their ordinary material properties and
maps. Live values and image versions are captured independently for queued native
draws. This also supports bump and displacement maps on admitted ordinary mesh
materials. Baked light maps select the same native source builder; ambient
occlusion maps and authored AO nodes compose with those native shaders. Plain
AO-only materials retain their existing fixed profile, including ArrayCamera.
Transmission and dispersion still need their scene-color protocol;
see [THREE_NODE_PROGRAMS.md](./THREE_NODE_PROGRAMS.md).

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
- Ordinary fixed-path material values and material-array selectors are captured
  per request, so queued fixed/native transitions preserve accepted colors,
  shading and raster state. Original materials remain cache/disposal identities.
  Geometry, texture objects and fixed-path image upload histories retain their
  existing live contract; this does not snapshot the entire scene.
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

Native authored graphs also support ordinary `Mesh.count` and TSL
`instanceIndex`. Counts are captured after source node updates and retained
across queued pipeline preparation. Zero instances preserve update callbacks
while omitting GPU draws and source draw-call increments. This connects
scalar/vector compute storage to instanced particle geometry. CPU/storage-backed
`InstancedMesh` and ordinary `InstancedBufferGeometry` have their own native paths.

Native `InstancedBufferGeometry` accepts separate or interleaved per-instance
attributes with ordinary built-in or authored node materials. Its explicit finite
`geometry.instanceCount` takes precedence over `Mesh.count` and is captured after
node updates for each accepted render. Zero-count draws still run source updates.
As in pinned r186 WebGPU, instance streams advance once per instance regardless
of `meshPerAttribute`; WebGL-authored programs retain their separate divisor
semantics. Consumed stream capacities are checked without silently reducing the
requested count. Attribute bytes retain the existing versioned upload contract,
including partial updates and GPU-stale bytes outside requested ranges.

Native `BatchedMesh` also selects this path with ordinary or authored node
materials. It executes indexed/nonindexed source subdraws with per-instance
matrices and colors, visibility, ID reuse, source culling/sorting and `customSort`.
Each accepted render retains its source indirect IDs, texture versions and draw
ranges. The source callback precedes shader construction and transparent side
expansion; node updates run once per physical material pass, rather than once
per subdraw. Native shadows share the same batch data and shadow owner. Direct
commands and render bundles use the source draw ordinal as WebGPU
`firstInstance`. Empty batches retain callbacks without issuing draws. Source
geometry draw ranges do not replace batch ranges, and expanded draws stay
within the configured scene budget. See `THREE_NODE_PROGRAMS.md` for allocation
changes and the bounded lifecycle profile.
