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
decider. It substitutes when at least one `WebGPURenderer` construction site routes
to general-webgpu and all remaining sites either share that route or explicitly
select the source WebGL backend. Other exact-backend requirements and GL escapes
remain conservative, and every import of `build/three.webgpu.js` must be the
admitted pinned r186 build. The substituted
module re-exports the upstream namespace (every other class keeps its upstream
identity) and replaces only `WebGPURenderer`. Otherwise the build is unchanged
and the printed/returned `rendererRoute.reason` says why.

On a host without a WebGPU `requestAdapter` API, each routed constructor instead
creates its genuine pinned source renderer before reading options or binding a
canvas. Source WebGL construction stays synchronous; source `WebGPURenderer`
keeps its own device, `forceWebGL` and initialization fallback behavior. Option
getters, source errors, canvas identity and `instanceof` are preserved. This is
exact-backend compatibility, not new-renderer execution. An injected GPU host
still selects the native route. If adapter/device initialization fails after
native construction, that error remains visible: the exposed renderer is never
silently replaced with another owner.

Mixed applications preserve each explicit `forceWebGL: true` construction through
its genuine upstream constructor. The compiler resolves the original callee once
before the untouched argument list; it does not pre-read, clone or validate source
options through the native facade. Option getters, extra arguments, constructor
errors and source prototypes retain their source behavior. Renamed constructors,
inline scripts and lazy modules use the same source module instance retained by
their routed export. `instanceof` that export recognizes actual native and source
instances without accepting flag-shaped objects or broadening subclass checks.

Per-site `rendererRoute.decisions` continue to identify source-selected calls as
`exact-backend`; `exactConstructionSites` records their original source spans.
Applications containing only explicit source selections remain unchanged.
Runtime-valued `forceWebGL` selections retain the existing facade dispatch.
Opaque renderer access, unknown extensions, native-context observations and
prototype mutation still prevent the mixed build route.

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
requiredLimits, requiredFeatures, device, outputType, outputBufferType})`, `init()`, `render()` (throws the source
error before `init()`), `renderAsync()`, `compileAsync()`, `setAnimationLoop()`,
`setSize()`, `setPixelRatio()`, `setDrawingBufferSize()`, size getters, clear
color/alpha/depth, `autoClear*`, `toneMapping`,
`toneMappingExposure`, `shadowMap.enabled`, `info`, `dispose()`.

Native `outputType: HalfFloatType` presents into `rgba16float` with the pinned
extended canvas tone-mapping mode. Omitting the option keeps the browser's
preferred format; explicit `UnsignedByteType` selects `bgra8unorm`, as r186 does.
The intermediate scene precision remains independently controlled by
`outputBufferType`. Direct working-linear draws support one or four samples,
including half-float resolves. Converted output renders the sampled scene first
and presents through a depthless, single-sample output pass.

Half-float presentation preserves signed working-linear RGB and values above
one. Its software output transfer follows the pinned native TSL alpha and tone
equations without inventing an sRGB texture view. Canvas configuration retains
the source's default sRGB color space. Queued color-space changes, canvas
selection, resize retirement, framebuffer-copy storage and source-visible
texture types use the existing owners. A copied half-float canvas may still have
public `FramebufferTexture.type === UnsignedByteType`, following the source's
separate public-type and GPU-format rules.

Run `node tests/e2e/webgpu_renderer/native_hdr_output.mjs` with the existing
Chromium/Playwright environment for raw half-float source/candidate comparisons
before browser compositing. Recording-device coverage also exercises signed
clears, MSAA, queued output transfers, independent scene precision and the
eight-byte-per-texel multisample budget. HDR display behavior and physical-device
performance require their own host evidence.

Chromium 151.0.7922.34 / SwiftShader matched all 6,144 raw half-float words across
twelve cases, including all seven tone operators, alpha, direct MSAA, signed
clears and working-linear output. The tests retained values up to 7.75 and a
negative clear of -0.5, with no GPU/browser errors. The canvas runner uses
ordinary browser process separation: the in-process GPU combination used by
older offscreen tests drops the WebGPU instance during canvas readback on this
host, including in a renderer-free reference probe.

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
updates follow the pinned native renderer. This does not admit nested arrays
or multiview/XR cameras. Custom native
scene/object/material render hooks retain their existing explicit rejection.

## Public render-list sorting

Both renderer surfaces implement `setOpaqueSort()` and `setTransparentSort()`.
Passing `null` restores the source default; `sortObjects = false` bypasses the
comparators. Callback items retain source `id`, `object`, `geometry`, `material`,
`groupOrder`, `renderOrder`, `z` and original `group` identities, plus WebGL's
`materialVariant` or the native clipping context. The transparent comparator
orders WebGL's transmissive and transparent lists separately, and native
transmission's back-face list before its complete transparent list.

Native custom-sorted scenes share one source draw capture, including ordinary
material companions. Comparators run before object/node updates, and the accepted
draw order survives asynchronous preparation without replaying callbacks. Each
queued WebGL call retains its comparator references and sorting flag; comparator
execution follows that route's existing prepared-bridge lifecycle. JavaScript
closures remain application-owned. `three_renderer_sorting.test.mjs` compares
the pinned render-list callback traces and recorded GPU draw order, including
material groups, canvas/target queues, reset, disabled sorting and errors.
These are source/command correctness checks, not pixel or performance results.

## WebGL-authored batched ArrayCamera draws

The WebGL-authored route supports `BatchedMesh` through each child camera's
source culling and transparent sort order. The existing camera-major draw order,
bottom-left viewports, layer checks and source callbacks remain intact. Each
child can request different indirect-ID, matrix and color texture uploads.

GPU copies preserve those uploaded versions and replay them immediately before
the corresponding native draws. No CPU readback is involved, and versionless
CPU edits remain stale on the GPU as they do in the source renderer. A final
camera with no visible instances still leaves its requested texture uploads
resident. History buffers share the texture byte budget, are reused across
steady frames, and retire before preparation grows source residency or when
the scene owner is disposed.

`three_scene_array_batching.test.mjs` checks submitted texture bytes, callback
updates, stale CPU edits, empty final views, buffer reuse/disposal and combined
memory limits. `node tests/e2e/webgpu_renderer/program_array_batching.mjs`
matches eight source checkpoints exactly: opposite/reordered views, culling,
layers, callback updates, versionless edits, an empty final view and resize.
Chromium 151.0.7922.34 / SwiftShader matched all 210,432 RGBA bytes and callback
traces with 73 native draws and zero GPU/browser errors. ArrayCamera transmission
and the existing BatchedMesh shadow boundaries remain outside this addition.

## Logarithmic depth for WebGL-authored programs

`createWebGLRendererClass(THREE)` admits `logarithmicDepthBuffer: true` on
canvas and render-target draws. `renderer.capabilities.logarithmicDepthBuffer`
reports that constructor setting. The source ShaderLib programs execute their
original logarithmic-depth chunks through the ESSL-to-WGSL compiler, including
the fragment-depth output. Authored ShaderMaterial programs retain their own
chunk choices and uniform overrides.

The camera coefficient keeps r186's exact JavaScript expression,
`2 / (Math.log(camera.far + 1) / Math.LN2)`, and updates independently of program
compilation. Perspective shaders use the source logarithm; orthographic shaders
keep their source fragment-coordinate depth. Sampled depth attachments therefore
expose the same authored values to later effects. Cameras remain in the WebGL
coordinate system.

All affected materials select their source program, including the existing
deformation and shadow-program routes. The separate whole-image HDR fallback
cannot replace this route: it reports the existing tone-mapping admission error
if a source program is unavailable. Reversed depth on the WebGL-authored facade
remains outside this admission; native WebGPURenderer depth modes retain their
existing behavior.

`three_program_log_depth.test.mjs` verifies built-in/custom shader compilation,
camera coefficients, authored uniform precedence and actual facade submissions.
`node tests/e2e/webgpu_renderer/program_log_depth.mjs` compares unchanged r186
WebGL output and owned WebGPU color/depth attachments. Chromium 151.0.7922.34 /
SwiftShader passed 22 exact checkpoints and 540,672 RGBA bytes: a large depth
range where linear depth loses foreground separation, changed camera far planes,
authored shaders, orthographic cameras and sampled one-half/one-quarter depths.
Both enabled and disabled modes were compared, with zero GPU/browser errors.
These are correctness results, not physical-device performance measurements.

## Preserved WebGL-authored canvases

`createWebGLRendererClass(THREE)` admits `preserveDrawingBuffer: true` with
`autoClear: false`. Painting, motion trails and layered composition retain their
encoded color, depth and optional stencil contents across browser presentation
turns. One-sample and four-sample draws use the same owned history; multisample
color resolves into its single-sample presentation source after each draw.

The owner initializes fresh attachments to transparent black, depth one and
stencil zero before the first load. Each submitted frame copies the resulting
color texels into a newly acquired canvas texture. There is no CPU readback or
extra tone/color/alpha conversion in this presentation path. Scissored clears,
independent attachment clears and source alpha blending operate on retained
storage. Changed dimensions allocate fresh history; old attachments retire only
after their submitted work completes, and their overlap counts toward the
existing memory budget. Failed preparation before attachment use presents
nothing; already issued partial draws remain visible if a later callback fails.

Preserved canvases keep per-material shader tone mapping. A material requiring
the separate whole-image HDR fallback reports its existing
`THREE_SCENE_TONE_MAPPING` error before replacing the canvas owner: moving
encoded history into a linear output target would discard or reprocess earlier
pixels. This boundary also applies to clearing frames because scissors and
independent clear masks can retain history. Native WebGPURenderer output is
unaffected.

`node tests/e2e/webgpu_renderer/program_preserved_canvas.mjs` compares the pinned
WebGL renderer with actual new-backend canvas pixels after separate animation
frames. Chromium 151.0.7922.34 / SwiftShader passed 30 exact checkpoints and
123,904 RGBA bytes across one and four samples: first-load initialization,
opaque trails, alpha accumulation, retained depth, `clearDepth`, scissored clears,
empty loads, changed dimensions, same-size `setSize` and same-value pixel ratio.
There were zero GPU/browser errors. These are rendering-correctness results,
not physical-device performance measurements.

## WebGL-authored framebuffer snapshots

`renderer.copyFramebufferToTexture(texture, position = null, level = 0)` now
copies post-render color into an ordinary `THREE.FramebufferTexture` on the
WebGL-authored route. Later MeshBasicMaterial maps and ShaderMaterial samplers
read the same owned GPU image. Canvas and admitted render-target sources retain
their selected face/layer and mip; `level` selects the destination mip. Positions
use physical bottom-left pixels and only their `x`/`y` fields, while the copy
extent comes from the destination image at that mip.

Copies preserve the destination's public type and versions. The admitted RGBA
channel precision and storage color encoding must match the source: byte color,
half-float color or float color, with matching linear/sRGB storage. Native BGRA
canvas storage is copied as RGBA channels. Integer-coordinate
GPU texel reads preserve the source row convention and raw encoded values without
CPU readback, tone mapping or a second color conversion.

Producer draws, copies, requested mip initialization and consumers share the
existing submission queue. A queued consumer retains its original framebuffer
generation when later calls replace that allocation. The owner captures only
its private generations; it does not traverse application metadata to find them.
Explicit borrowed texture bindings retain precedence. Copies do not regenerate
lower mip levels; source-requested initialization generates them before its copy.

The destination's `onUpdate` runs synchronously during requested initialization.
It can change the selected source target or prevent the copy by throwing; copy
position and extent retain their earlier source reads. Acknowledged texture
versions follow the source, including a version change inside that callback.
Ordinary-camera draw callbacks can also copy the currently bound framebuffer.
The renderer submits the preceding draw prefix before initialization or copying,
then resumes later draws with color/depth/stencil load operations. A first-draw
copy observes the current clear. Earlier draws remain readable when initialization
throws; the extra F3D queue fence also reports that failed operation. Callbacks
execute once and no CPU readback is involved. Changing the active attachment
during this operation and ArrayCamera callback copies remain outside this slice.
WebGL depth snapshots, unmatched color
precision/encoding and the whole-image linear HDR fallback also retain explicit
admission errors.

`three_renderer_program_framebuffer_copy.test.mjs` has twelve focused cases for
ordering, generation lifetime, callback effects, source versions, mip timing and
format refusal. The affected framebuffer, canvas and target suites pass 174
additional checks. `node tests/e2e/webgpu_renderer/program_framebuffer_copy.mjs`
matches 19 source checkpoints and all 58,368 RGBA bytes exactly in Chromium
151.0.7922.34 / SwiftShader, with zero GPU/browser errors. This includes ordinary
material consumption, two producer-copy-consumer chains without intervening
waits, source resize, cube/array selection, matching half-float storage, raw sRGB
copies, initialization retargeting, before/after-draw copies, a clear-only prefix
and source-visible partial drawing after an initialization exception. No
hardware-performance claim is made.

## Callback direct draws and Lensflare

The WebGL-authored `renderBufferDirect(camera, scene, geometry, material, object,
group)` now executes ShaderMaterial and RawShaderMaterial draws inside an active
ordinary-camera callback. Programs and geometry first discovered inside the
callback register synchronously with the existing native owners. Every accepted
direct draw submits before the call returns, so later copies and callback
exceptions observe its actual pixels. Native validation scopes close before
returning to application code; delayed validation failures remain observable.

This enables the unchanged pinned `Lensflare` addon: save the scene color, draw
the depth-tested occlusion quad, copy the occlusion result, restore the saved
color, then draw each flare element. The new backend owns all GPU work; the
source addon and shader semantics remain intact. Cold first use requires neither
pre-running callbacks nor an explicit compilation call.

Direct draws retain their difference from normal object rendering: they do not
invoke object/material render hooks, update the caller's public model-view
matrices, or invent an extra pass for a DoubleSide material. Consecutive calls
keep custom uniforms and samplers GPU-stale until the source requests an update
or switches program, material or camera. Per-object matrix values retain their
source upload boundaries. Queued rendering also restores the accepted target,
face and mip while its callbacks run, then restores the caller's later selection.

Admission currently requires an unchanged active attachment, an ordinary camera
and shader materials. Standalone direct calls, ArrayCamera/BatchedMesh direct
interleaving and shadow/transmission callback contexts remain outside this
profile. After an exception following submitted work, the existing scene bridge
requires preparation before reuse; the target's already-drawn color remains
readable and the F3D queue fence reports the failed render.

The affected real-owner suite passes 170 uniform, point, stencil, index, restart
and framebuffer/direct cases; subsequent uniform-retention checks pass 45 cases
and all 15 public copy/direct cases pass after the queued-target regression was
added. Older mock-target and obsolete stencil-refusal assertions reproduce their
failures on unchanged main and were not changed by this feature.

`node tests/e2e/webgpu_renderer/program_lensflare.mjs` compares the actual addon
against pinned WebGL. Chromium 151.0.7922.34 / SwiftShader matches all nine
checkpoints and 135,168 RGBA bytes exactly, including cold hidden shaders,
full/partial occlusion, movement, offscreen/behind-camera culling, resize,
scissored history and a callback exception after direct drawing. All 17 callback
events, 14 framebuffer copies and 35 direct draws match, with zero GPU/browser
errors and the owned `general-webgpu` route. Hardware performance remains
unmeasured.

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
  XR, custom tone mapping, reversed depth on the WebGL-authored facade, non-sRGB
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

Native mesh materials also support wireframe rendering, including node-authored
deformation, instancing and batches. The renderer uses the pinned triangle-edge
generation rules and a line-list pipeline. Edge versions and draw ranges remain
independent across queued calls and across groups within one submission; source
node update order, index update requests and disposal retain their meaning.
Ordinary Basic/Toon/Lambert/Phong/Standard/Physical wireframe materials enter this
path automatically. Normal and Matcap source-library conversions also work in
native programs, preserving mixed scenes that share the native shadow owner.
Other existing material routes keep their prior selection.
