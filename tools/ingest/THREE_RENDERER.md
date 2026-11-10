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
Ordinary source selections dispatch before native option validation or getter
evaluation and forward every original constructor argument. Accessor selectors
and inherited/custom option containers retain the genuine source constructor;
opaque receivers remain under the compiler's conservative construction routing.
Opaque renderer access, unknown extensions, native-context observations and
prototype mutation still prevent the mixed build route.

Construction routing follows local renderer and canvas aliases before choosing
an owner. Exports, external stores, returned renderer factories, unknown consumers,
dynamic member selection, native renderer state and custom subclasses retain the
upstream component. Unknown constructor options or externally supplied canvas
identities also stay upstream. Closed renderer calls, local aliases, locally
created DOM canvases and ordinary DOM attachment remain eligible.

Ordinary static ESM factories, setup/frame helpers, aliases and reexport chains
can also retain closed renderer ownership across modules. The compiler links a
private analysis copy and applies the existing lexical ownership proof to it;
the original application modules, evaluation effects and construction spans
remain intact in production emission. Both WebGPU-authored and WebGL-authored
applications can use this route, including mixed explicit `forceWebGL: true`
sites that retain their genuine source constructor.

This proof requires one application entry, the exact pinned Three build bytes
and dependency identities, a closed static graph of at most 128 application
modules, and at most 2 MiB of application and linked analysis source each.
It never evaluates source. Exported factories or resources, unknown consumers,
native observations, mutable/indirect helper calls, dynamic imports, external
modules, cycles and unsupported linking shapes retain their original exact
route. Every original construction profile must survive linking; generated
positions never replace source diagnostics. The production-build regression
command is `node --max-old-space-size=768 --test tools/ingest/renderer_module_build.test.mjs`;
the smaller ownership regressions are in `renderer_module_ownership.test.mjs`.
This expands new-backend eligibility while retaining application JavaScript;
it is not a Rust rewrite or a measured acceleration result.

Private constant records can hold renderer constructors, renderer instances,
canvases and verified controls without forcing an exact backend. Static field
reads, constant aliases and simple destructuring retain the selected value's
ownership, so a local `toolkit.Renderer` and `state.renderer` can use the general
route. The original records and application statements remain intact. This proof
is limited to shallow, freshly created records with unique own data fields whose
uses all remain known. Exported records, unknown consumers, mutation, dynamic
keys, getters, spreads and reflection keep the exact component. Selecting a
native backend method from a private record also retains its existing exact-route
requirement; the record does not hide that observation.

The graph can prove bounded retained `OrbitControls`, `FirstPersonControls`,
`MapControls`, `TrackballControls` and `FlyControls` contracts by verifying the
exact addon, resolved Three build and core bytes. `MapControls` also requires its
actual resolved `OrbitControls` superclass to pass that proof. The returned
controls instance continues to carry the canvas reference through local aliases and `domElement`;
it is not an exemption for arbitrary consumers. Closed updates, camera-control
methods, configuration and disposal remain eligible. Exports, unknown consumers,
event callbacks, private hooks, method replacement, reflection and unproved
constructor/receiver origins retain the upstream owner. Changed source bytes,
redirected base classes and unresolved code invalidate the proof. The original
H2 application is consequently eligible with `--route-webgl-renderer`, retaining
its actual OrbitControls and camera identities while using the new GPU backend.

Map panning, trackball rotation/zoom/damping/multitouch and flight speed/roll/drag
settings keep their original source implementation. Named and namespace imports,
local factories and private records preserve the controls/camera/canvas identities.
Resize, reset, disconnect/reconnect and disposal remain source-owned operations;
callback or native-context exposure still selects the exact renderer. The bounded
source-proof regression command is
`node --max-old-space-size=256 --test tools/ingest/renderer_camera_controls_source.test.mjs`;
it verifies actual pinned source/import identities and analyzes the calling code.
The native DOM comparison is `node tools/ingest/renderer_camera_controls.browser.mjs`: it
builds source and routed applications, applies real pointer/keyboard/wheel input,
and compares camera state at each lifecycle boundary. This is compatibility and
new-backend routing validation, not a controls rewrite or a speedup measurement.

Applications using the pinned `RendererInspector` can retain its original class
while selecting the native WebGPU route. The graph verifies its exact source,
WebGPU/core dependencies and resolved TSL identity, then tracks attachment,
local aliases, `getRenderer`/`setRenderer`, frame observations and supported
settings. An emitted-application regression submits native draws and verifies
that the unchanged Inspector records the original scene and camera. Custom
hooks, private state, unknown renderer owners, escaped inspectors, changed
dependencies and open import graphs retain exact routing. This covers the
RendererInspector consumer; the full Settings/TSL Graph Inspector used by H1
retains its separate open-code and backend-mutation boundaries.

H1 still retains the exact renderer: its Inspector Settings component replaces
`WebGPURenderer.prototype.init` to select a real WebGL backend from persisted
settings, and the TSL Graph extension evaluates user-authored code with the full
Three namespace. Closing ordinary imported helpers does not close those native
and dynamic ownership boundaries.
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
2D, cube, array or 3D volume target for final output. Working-linear
output renders directly;
tone mapping or color conversion first renders into the pinned source's private
linear framebuffer, then executes its retained TSL output graph on the same new
WebGPU owner. Target dimensions, viewport, scissor, attachment precision and MSAA
come from the selected target. An explicit unrelated `setRenderTarget()` takes
precedence. Queued output passes keep their own target generations, tone operator,
exposure and render-frequency updates; `compileAsync()` prepares the scene and
output passes.

Multiple-color output targets support working-linear rendering with
`outputColorSpace = LinearSRGBColorSpace` and `toneMapping = NoToneMapping`.
Their attachments retain the ordinary target owner's formats, independent
precision, multisample resolves, array/volume layers and authored 2D mip levels.
When the public render-target selector is null, use an explicit
`TSL.outputStruct(...)` fragment graph to write each attachment. As in the
pinned source, renderer-level `setMRT()` output selection applies after an
explicit `setRenderTarget(outputTarget)`; it does not replace the fragment
graph for a null selection. The native compiler validates against the captured
physical attachment count while preserving that public null selector, including
queued target changes and program-cache reuse. Tone mapping or renderer color
conversion into MRT remains refused because the pinned private output pass
produces one color. Explicit clears still clear every selected attachment.

Direct layered output retains the target owner's face/layer/slice and mip
selection. Converted array and volume output renders the scene into the selected
private array layer, then converts every private layer into its matching output
layer/slice, as the pinned source does. Queued frames preserve each layer's scene
values and the output transform for each call. Source node updates still observe
face zero during the private scene, including deferred initialization. Converted
layered output requires mip zero; converted cube output additionally requires
face zero because the source private image has only one layer. Unsupported
converted selectors fail before target allocation or queued submission.

Selection follows the pinned source, including its asymmetries: a converted
render leaves the effective output target selected by `getRenderTarget()`; direct
working-linear output keeps a null active selection. With a null active selection,
`clear()` clears the private framebuffer and draws its output when conversion is
needed, but otherwise clears the canvas. An explicit target clear never performs
the output transform. Multiview output targets and output
targets on the WebGL-authored facade, remain unsupported. See
[THREE_RENDER_TARGETS.md](./THREE_RENDER_TARGETS.md) for target profiles and lifetime.

`node tests/e2e/webgpu_renderer/native_output_targets.mjs` checks twelve profiles
against the pinned source: six cube/array/volume direct and sRGB/Reinhard cases,
plus six MRT cases covering mixed precision, four-sample MSAA, named outputs,
arrays, volumes and authored mips. Chromium 151.0.7922.34 / SwiftShader matched
all 374,784 storage bytes, including queued layer/viewport changes, per-call
exposure, untouched cube faces, retained selection state and signed half-float
MRT values from -0.5 to 2, with no GPU or browser errors. This is native WebGPU
correctness evidence; it is not a hardware-performance measurement.

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
CPU refresh. Multiple host versions of one shader-storage attribute within a
logical render upload in source order to its one allocation. All draws observe
the accumulated contents at submission, matching pinned r186, including earlier
partial uploads and untouched GPU results. Generated scalar/vector vertex inputs
sharing that storage observe the final admitted version through immutable
geometry proofs, including signed integer inputs and retained bundle draws.
Vertex-only inputs keep the source's first upload per logical render. Preparation
and submission share the accepted history without replaying earlier CPU uploads;
separate queued renders retain their own histories. Indirect arguments can share
multiple host versions with read-only shader-storage bindings. Their first
geometry upload retains its source boundary, including after an earlier
shader-only object; later shader bindings can add full or partial uploads, and
every indirect draw consumes the final accumulated allocation at submission.
Indirect-only changes after that first use do not independently force uploads.
Untouched GPU-produced words and retained command bundles remain intact.
Multi-version histories combining generated vertex and indirect aliases,
generated index aliases and allocation-layout changes remain explicitly
unsupported; writable-versus-read-only storage hazards still apply.

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
or multiview/XR cameras.

## Native Scene lifecycle

Native `Scene.onBeforeRender` and `Scene.onAfterRender` now surround the retained
source object lifecycle. The before-hook runs after automatic world/camera
updates and raster selection, and before background, material and draw-list
capture. Position changes inside it need the same explicit matrix update as
pinned r186 to affect that render. The active render-object dispatcher is fixed
before this hook; changing it selects the next call's dispatcher.

Callbacks receive the original root, camera and selected working target. Public
target changes survive the callbacks, with output conversion selecting its final
target before the after-hook. A canvas change in the before-hook selects the
backend surface while retaining the earlier raster context. Nested before-hook
renders precede the parent; nested after-hook renders follow its output passes.
The after-hook sees restored render context and NodeFrame render ID and can
change or detach the accepted scene without rewriting its retained draws. An
after-hook exception preserves that already accepted work. Preparation never
replays either callback, including the asynchronous initialization path.

Native `compileAsync(root, camera, targetScene)` initializes and advances the
source frame before its before-hook, compiles the requested root, and uses a
distinct target scene for additional lights and background. Its hook and material
environment owner is the root when that root is a Scene, otherwise the target
Scene; the hook's scene argument remains the original root. Compilation does not call
the Scene after-hook or submit graphics draws. Source native material render
hooks remain inert, matching the pinned renderer.

Hooked scenes use the complete native draw capture and its existing resource
bounds. Ordinary rigid geometry shares its retained upload history with later
hooks. Attributes already owned by the separate fixed shadow, instancing,
deformation or mixed-list paths retain their explicit transition guard.
Bare non-renderable Object3D roots do not acquire Scene
callbacks. `three_renderer_scene_hooks.test.mjs` checks callback order, nested
work, selector edits, immutable GPU bindings and exception retention using actual
pinned source objects and a byte-accurate recording device; these are command
and lifecycle checks, not pixel or performance results.

## Queued ordinary native scene state

Shadowless rigid meshes, lines, points and sprites keep accepted world matrices
with their existing fixed geometry and material captures. Each queued render can
therefore use a different object transform or camera without a later update
rewriting earlier GPU inputs. This includes reflected winding, sprite centers,
rotations and counts, and native ArrayCamera child views. Submission does not
overwrite the objects' public matrix caches with an earlier queued transform.

Complete ordinary fixed scenes also retain their hierarchy, layer masks, group
and object render order, and standard light values. Removing a mesh after
`render()` leaves its accepted draw intact; adding one affects later calls.
Light colors, intensity, attenuation, world positions and target matrices survive
edits or removal while preparation is pending. These captures share the existing
pending-byte budget and keep the original objects as public/cache identities.
Geometry upload generations and fixed shaders retain their existing owners.

This extends the ordinary fixed capture; separate shadow, instancing,
deformation and mixed native/fixed owners retain their existing boundaries.
`three_renderer_fixed_snapshot.test.mjs` checks the actual queued GPU matrices,
light buffers, draw membership and fixed shader route with the recording device.
It does not establish additional pixel parity or a performance result.

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

## WebGL-authored ArrayCamera transmission

Ordinary WebGL `ArrayCamera` scenes support the existing physical-material
transmission profile through each child camera. Each child has a source
transmission target sized from its viewport and `transmissionResolutionScale`.
All child transmission passes precede all main child draws, following pinned
r186 order. Back-face and main draws sample their own child's resolved image;
sharing a material across views does not share the wrong sampler binding.

Target identity is shared across canvas and explicit-target scene bridges for
the same public renderer, source scene and camera. Different source scenes retain
separate maps, as `WebGLRenderStates` does. Destination-specific programs remain
local, and shared target storage is released after the last scene bridge.

Preparation finishes before custom sort callbacks and draw callbacks execute.
Comparators run once; offscreen callbacks receive the original child camera,
the selected transmission target, source clear/tone state and the temporary
viewport state. Object layers are checked at each draw. Renderer state, child
viewports and temporary material-side changes are restored after failures;
accepted callback effects are never retried. A callback that introduces an
unprepared child identity, target extent or back-face variant encounters an
explicit preparation boundary. Resizing between render calls is supported.
Empty arrays and scene overrides skip transmission passes.

`three_renderer_webgl_arrays.test.mjs` exercises distinct per-child GPU bindings,
source callback and custom-sort order, live layers, viewport resizing, target
identity across destinations, scene separation, front-face shader hooks and error
restoration using pinned source objects and the real scene bridge. These are
recorded-command and source integration checks; this addition has no browser
pixel or performance result. Existing shader/material boundaries remain in
force; BatchedMesh composition is described below.

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
traces with 73 native draws and zero GPU/browser errors.

## WebGL-authored BatchedMesh transmission and shadows

Batches now participate in the program route's transmission and shadow passes.
Opaque batches appear behind physical glass, and transmissive batches render
their required back faces before their main draws. Directional and point-light
shadow passes select batch instances for each shadow camera, including each
point-light face. Source callbacks run once per logical object/pass, with one
native draw and a captured `_gl_DrawID` per selected subdraw. Shadow draws retain
the source `onBeforeShadow`/`onAfterShadow` lifecycle without adding material
render callbacks.

Every pass captures uploaded indirect-ID, matrix and color textures through the
existing bounded GPU history buffers. Later cameras and callbacks cannot replace
the data consumed by earlier draws; versionless CPU edits retain their source
staleness. Copies occur between render passes, preserving accumulated depth and
color, and the latest uploaded texture state is restored for subsequent source
work. This composes with ArrayCamera transmission targets and ordinary main
draws. Source geometry/group admission is checked before batch expansion; an
admitted ordinary draw range does not clip individual batch subdraws.

`three_scene_array_batching.test.mjs` and
`tests/three_program_shadow_targets.test.mjs` check pass/face texture histories,
selected ranges, callbacks, indexed/nonindexed draws, material groups and empty
faces through the existing recording seams. The focused browser comparisons are
`tests/e2e/webgpu_renderer/program_batch_transmission.mjs` and
`tests/e2e/webgpu_renderer/program_batch_shadows.mjs`; they use the unchanged
WebGL parity envelope of 12 channel levels and at most 2% outlying pixels.
Chromium 151.0.7922.34 / SwiftShader matches all 294,912 shadow RGBA bytes exactly
across 12 checkpoints. The 12 transmission checkpoints pass with at most
25/6,144 outlying pixels (0.41%, below 2%; maximum channel difference 43).
Both comparisons match source callback/instance traces and report zero GPU or
browser errors against source commit `148ef33ecb6d2502ff796d4554abd1549c95d519`.
Existing shader, material and shadow-map profile limits remain in force. This is
retained-JavaScript program preparation with owned WebGPU rendering, without a
performance claim or a claim of complete physical-material/shadow coverage.

## Public PMREM generation for WebGL-authored applications

After `await renderer.init()`, the public
`PMREMGenerator.fromEquirectangular()` and `fromCubemap()` methods execute the
retained r186 conversion and GGX shaders through F3D-owned render requests. They
return real CubeUV `WebGLRenderTarget` objects; a compatible explicit destination
retains its identity and can be sampled or read back. Source pass state, filter
uniforms and ordinary input pixels are captured before the call returns, so
queued bakes retain independent accepted inputs. Disposing an input or the
generator after acceptance preserves those reads; private filter snapshots and
destination bridges retire after submission drains without retiring the returned
output.

A panorama or cube first uploaded through this path retains the same accepted
history when later used as an ordinary map or background, including Source-sharing
aliases and unmarked CPU edits. An input already resident in the separate
automatic WebGL scene texture pool refuses instead of reconstructing its prior
GPU bytes. Custom input upload callbacks and ordinary GPU-written copy
destinations are outside this detached-input profile; render-target inputs use
the existing GPU target owner and its lifetime guards. Texture-input calls before
initialization retain their previous delayed behavior.

`fromScene()` executes the retained source's six camera views,
sigma pre-blur and GGX filtering into a real CubeUV target. A single immutable
scene generation serves all six captured cameras, preserving accepted transforms,
material values, lights, shader uniforms and CPU instance matrices/counts while
later edits and disposal proceed. Actual pinned `RoomEnvironment` and `Sky`
inputs are covered. The WebGL helper ignores `options.renderTarget` for this
method, so the returned output remains a newly allocated source target. Calls
before initialization return that same real target synchronously and capture all
six views, sigma filtering and scene state before device acquisition completes.
They enter the renderer's existing FIFO. Geometry histories bind to the real
device before ordinary renderer uploads; a conflicting batch releases all newly
acquired claims. Disposal cancels pending work and releases captured payloads.

The admitted scene profile comprises rigid and skinned meshes, CPU instances,
ordinary source materials and texture-free shaders, and unshadowed lights, with
a solid or absent background. Skinned captures retain call-time bone worlds,
inverse binds, bind mode and bind matrices, including shared skeletons and
detached bones outside the captured hierarchy. Each accepted cube face preserves
the source's synchronous skeleton update and bone-texture creation behavior;
queued draws use a private palette through the existing ShaderLib or GPU
deformation route. Later animation, source disposal and generator retirement do
not change an accepted pose. Private palette residency is reused across captures
and participates in the existing capture budgets and retirement lifecycle.
Initialized scene bakes also capture ordinary Basic,
Lambert, Phong, Toon, Standard, Normal, Matcap and depth material maps. Source
shader assembly/reflection and material uniform refresh select the actual
sampled inputs; hidden draws and inactive map properties cannot establish a
texture history during later preparation. A normal map suppresses the source's
bump sampler, and Toon gradients are sampled only when a direct light invokes
their source helper. The existing sampled-texture owner
captures immutable pixels, sampler history and decoded-image readiness. UV
offset/repeat/center/rotation/matrix values are retained independently of upload
versions, so unmarked UV changes remain visible while unmarked pixel edits stay
GPU-stale. Disposing an input after acceptance preserves the bake. One prepared
descriptor remains with each shared captured image until its last owner retires,
preventing re-upload between queued uses of the same accepted image.
Scene admission uses the sampled owner's existing synchronous transaction: a
later map's rejection restores earlier upload histories, and original Sources
are registered for ordinary consumers only after the complete capture succeeds.

One Texture per Source is admitted within each scene bake. Aliases across
successive bakes and ordinary consumers share the existing uploaded history;
simultaneous aliases need per-face upload ordering and remain explicit. Texture
inputs before device initialization, ShaderMaterial sampler uniforms, Physical
material accessors, environments/overrides, custom render or shader hooks, custom
sorting, shadows, transmission, morphed/batched meshes, LOD, wireframes,
lines, points and sprites also retain explicit boundaries. Skinning requires
ordinary source skeletons and four-component skin attributes. Custom skeleton
methods, altered bone-texture uploads, and palettes aliased as material uniforms,
maps or geometry inputs require per-face source effects and remain explicit
boundaries. These restrictions
keep unsupported mutable state from being silently read after acceptance.

Captured geometry follows WebGL first-upload, version and partial-range history.
Only meshes selected by visible ancestry, layers and at least one cube frustum
advance those histories; hidden or culled inputs retain pending updates. Index
uploads retain source material/draw selection. Unmarked edits preserve the last
uploaded bytes, while source geometry/instance disposal starts a new history.
An attribute already resident in a separate ordinary WebGL scene owner requires
disposal before moving between owners. These histories persist across generator
retirement like source renderer attributes, and remain bounded by the snapshot
owner's budgets.

The new output's normal depth storage is initialized to the source value of one
before the six views load it with `autoClear = false`. Private background boxes,
scene bridges and ping-pong targets retire only after all active GPU waits release
them, including concurrent application waits and generator disposal. Returned
output targets retain their independent lifetime.

`compileCubemapShader()` and `compileEquirectangularShader()` now retain r186's
conversion-material construction and capture translated shaders before
asynchronous initialization. They return `undefined`, as the source methods do,
and validate modules on the owned device without allocating input textures,
draw geometry or render pipelines, or submitting a render. Later conversion
draws reuse the exact warmed GPU modules through `gpu_program_modules.mjs`.
Its per-device LRU retains at most 256 entries and 16 MiB of source strings;
failed modules are evicted and device loss clears residency. Generator disposal
preserves accepted warmup; renderer disposal cancels pending work. Validation
failures remain observable through the existing renderer error boundary.

The native renderer also exposes the source `compile` accessor, which returns
`compileAsync`. Native compilation now honors source frustum and layer
selection. Its pinned PMREM helpers use empty, culled geometry, so both public
warmup methods complete without compiling GPU programs, matching r186. Native
cube/array loading states are also admitted through the shared sampled-texture
owner. The native loading and subsequent real cube/panorama atlas comparisons
are described in [THREE_NODE_PROGRAMS.md](THREE_NODE_PROGRAMS.md).

The new module-cache and public-warmup suites pass all ten tests, covering
device isolation, eviction, retry, cancellation, pre-initialization capture,
actual conversion-module reuse and the native compile alias. The public PMREM
browser comparison below now also verifies that all four warmed modules are
consumed by subsequent cube and panorama pipelines.

Ten recording-device regressions cover source filter uniforms, destination
reuse, queued versions, Source aliases, accepted disposal, callback isolation and
repeated generator cleanup. Twelve generators can reuse one live output while
each disposed generator leaves zero private snapshot roots or dispatcher entries.
The adjacent snapshot/PMREM run passes 40 of 42 tests. Its two native clear-count
assertions, `native queued Color clears retain each accepted value` and
`pre-init native rendering captures background selection with deferred node updates`,
also fail with all five changed production modules replaced by their pre-change
versions; they remain existing failures.

`program_public_pmrem_targets.mjs` compares nine real source/candidate checkpoints
(221,184 RGBA bytes) for panorama and cube inputs, original-panorama background
reuse, marked updates, explicit output reuse and GPU input targets. Three direct
background/map history checkpoints are byte exact. Chromium 153 software-GPU
execution reports zero browser/GPU errors; at most 6 of 6,144 pixels in any
checkpoint exceed channel delta 12 (0.0977%, within the existing 2% gate). The
largest individual channel difference is 34. This is tolerance-based rendering
evidence, not a hardware-performance measurement.

The focused scene PMREM, existing public PMREM, shader warmup and pass snapshot
suites pass **49 tests** (10, 10, 6 and 23 respectively). The
`three_renderer_pmrem_scene.test.mjs` cases exercise actual source
RoomEnvironment/Sky capture, source attribute-range oracles, visibility/layer/
frustum history, disposal-based owner transfer, pre-allocation budget rejection,
and concurrent GPU waits across size changes and generator disposal. Startup
regressions cover adapter-gated scene acceptance, queued edits/disposal, cancellation
and atomic device-history claim rollback.

`tests/e2e/webgpu_renderer/program_public_pmrem_scene.mjs` compares five
source captures against both initialized and pre-initialization candidates
(**10 checkpoints, 245,760 RGBA bytes**): RoomEnvironment with sigma
0 and 0.35, a queued room with changed instances/light/material/capture position,
and two Sky uniform/sigma states. Source assertions establish that those edits
visibly affect the result. The existing channel-delta-12 / 2% pixel gate is
unchanged; both startup paths have over-threshold counts of 9, 5, 54, 1 and 0
of 6,144 pixels, at most
**0.879%**. The run used Chromium 153.0.8010.0/SwiftShader, the pinned r186 inputs
and `general-webgpu`, with zero browser/GPU errors. This is tolerance-based
rendering evidence, not exact equality or a hardware performance result.

`three_renderer_pmrem_textures.test.mjs` adds ten recording-device regressions
for accepted map versions and partial updates, UV controls without `needsUpdate`,
hidden/layer/frustum selection, eight material families, cross-bake Source
aliases and ordinary consumers, disposal/reuse, and explicit ownership/callback
boundaries, inactive-to-active sampler transitions, rejected asynchronous image
preparation, and atomic rollback when a later map fails admission. All ten and the ten
existing public texture-PMREM tests pass. The
older sampled-owner suite passes 24 of 25 tests; its unsupported-source assertion
at `tests_three_compute_textures.mjs:341` fails unchanged against the previous
implementation.

`tests/e2e/webgpu_renderer/program_public_pmrem_textures.mjs` compares eight
textured source/candidate atlases (**344,064 RGBA bytes**) in Chromium
153.0.8010.0/SwiftShader, using the same r186 pin and `general-webgpu` route.
Basic and Phong/normal maps, unmarked pixels, live UV transforms, Source aliases,
marked uploads and two queued CanvasTexture paintings are exercised. Five
checkpoints are byte exact; the other three each differ in one channel by one.
No pixel exceeds the unchanged delta-12 gate, and there are zero GPU/browser
errors. This establishes correctness in the tested software-GPU environment,
without a hardware performance claim.
The same runner also matches actual source texture-upload activity in **21
material/light cases**, including opaque/cutout alpha, depth, normal versus bump,
ambient/direct/hemisphere lighting and Toon gradients. The source still uploads
several mathematically dead output samplers in that driver; those histories are
preserved rather than inferred from their apparent contribution to the image.

## Custom tone mapping for WebGL-authored programs

The WebGL-authored facade admits `THREE.CustomToneMapping` through the source
material programs. The application-defined `CustomToneMapping(color)` function
in the live `ShaderChunk.tonemapping_pars_fragment` compiles through the runtime
GLSL-to-WGSL path, with the original `toneMappingExposure` uniform and material
shader hooks. It executes before output encoding and blending. Materials with
`toneMapped = false` bypass it, and ordinary render-target draws keep the
source's no-tone-mapping, linear-output behavior.

Queued requests retain their tone operator and exposure. Canvas requests in the
existing callback-free material/pass capture profile also retain an ordinary
string-valued `tonemapping_pars_fragment`, so queued custom curves can change
A-to-B-to-A without overwriting earlier draws. Both source cache keys and final
compilation use that private value; application ShaderChunk state is never
replaced. Use `material.needsUpdate` when changing the source.

Object/material shader hooks, custom sorting and accessor-backed tone chunks
keep their existing live evaluation boundary. If a captured scene gains hooks
before preparation or submission, its private chunk override is discarded.
Other ShaderChunk/ShaderLib and ordinary material shader sources also remain
live. Finish the earlier
`renderAsync()` before editing those sources when its earlier behavior must be
retained; this is not a snapshot of all application shader state.

Custom operators cannot use the fixed whole-image HDR output pass. Invalid or
unsupported custom GLSL fails explicitly without replacing the canvas owner.
A renderer that has already entered the existing whole-image HDR fallback also
rejects a switch to custom tone mapping, preserving its framebuffer history.
Native WebGPU custom output operators remain unsupported.

`three_program_custom_tone.test.mjs` compares compiled programs with the pinned
`WebGLProgram` source and checks actual submitted shader/uniform records for
source edits, shader hooks, bypasses and queued transitions. These are compiler
and recording-device checks, not rasterized-pixel or performance measurements.

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
if a source program is unavailable. Native WebGPURenderer depth modes retain
their existing behavior.

`three_program_log_depth.test.mjs` verifies built-in/custom shader compilation,
camera coefficients, authored uniform precedence and actual facade submissions.
`node tests/e2e/webgpu_renderer/program_log_depth.mjs` compares unchanged r186
WebGL output and owned WebGPU color/depth attachments. Chromium 151.0.7922.34 /
SwiftShader passed 22 exact checkpoints and 540,672 RGBA bytes: a large depth
range where linear depth loses foreground separation, changed camera far planes,
authored shaders, orthographic cameras and sampled one-half/one-quarter depths.
Both enabled and disabled modes were compared, with zero GPU/browser errors.
These are correctness results, not physical-device performance measurements.

## Reversed depth for WebGL-authored programs

`createWebGLRendererClass(THREE)` admits `reversedDepthBuffer: true` on canvas
and render-target draws. Both `capabilities.reversedDepthBuffer` and
`state.buffers.depth.getReversed()` report the owned mode. Source programs use
the zero-to-one clip volume selected by r186's `EXT_clip_control` path, preserving
their projection values and sampled depth without an additional GL-to-WebGPU
depth conversion. Raw point programs use the same near-plane clipping rule.

The source camera enters reversed mode when its program uniforms are packed,
after the object's before-render callback and before its after-render callback.
Fullscreen and cube-face snapshots preserve captured camera values, lens and
view offsets while publishing that same one-way transition to the original
camera. Program cache identity includes
the clip-depth convention even when raw shader text is unchanged. Raster state
uses all eight entries of r186's reversed comparison table, including its
Never/Always and Equal/NotEqual swaps. Polygon-offset slope factors reverse sign
while offset units retain the source value. Canvas, target, shadow and transmission
depth attachments use the matching clear value. PCF shadow samplers use the
source greater-equal comparison; Basic and VSM retain their source shader paths.

`three_program_reversed_depth.test.mjs` covers the comparison table, source/raw
shader cache separation, point clipping, actual canvas/target submissions and
cold/warm fullscreen and six-face cube camera snapshots. `program_reversed_depth.mjs` compares
ordinary and reversed modes against unchanged r186 WebGL on a host with
`EXT_clip_control`: large depth ranges and changed far planes, built-in and
authored shaders, orthographic cameras, all depth functions, point clipping,
positive/negative polygon-offset slopes, all cube faces and their public camera
projections, directional Basic/PCF/VSM shadows and point Basic/PCF shadows.

Chromium 151.0.7922.34 / SwiftShader passes all 76 checkpoints, comparing
1,228,800 RGBA bytes and 36,864 sampled depth values. All sampled depths and
camera observations match exactly. Seventy-five checkpoints are exact; the
reversed directional VSM checkpoint differs in 48 channel bytes by at most six,
with no pixels outside the existing tolerance. GPU and browser errors are zero.
This verifies the direct facade and this source host; automatic application
routing, hosts without the source extension and physical-device performance
are not established by this comparison.

`program_batch_transmission.mjs --reversed-depth` additionally compares
opaque geometry, glass back faces and transmission sampling in two ArrayCamera
views, including live callback uploads and versionless edits. All 12 checkpoints
and 294,912 RGBA bytes pass the existing channel-12 / two-percent pixel budget;
the worst outlier fraction is 20 of 6,144 pixels (0.33%), with maximum channel
difference 43. Callback/instance order matches exactly, with 160 native draws
and zero GPU/browser errors on the same host.

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

## Straight-alpha WebGL-authored canvases

`premultipliedAlpha: false` now admits the WebGPU program route. Source draws,
blending, clears and framebuffer copies use the original plain 8-bit encoded
drawing buffer. A final fullscreen pass multiplies RGB by alpha into the
premultiplied page image without changing the owned source image. It performs
no color transfer, tone mapping, filtering or CPU readback. The clear-color
path preserves alpha and follows the pinned `WebGLBackground`/`WebGLState`
construction option, including when the renderer's `alpha` option is false.

One and four samples, scissored/layered rendering, deferred animation tokens,
`preserveDrawingBuffer`, framebuffer copies and resize share the existing
attachment owner. Direct calls identify the current browser drawing buffer
through weak texture-identity tokens, so calls before presentation retain their
contents and a new browser buffer initializes transparent black/depth one.
Swapchain handles are still reacquired for each use; none are retained across
host turns. The extra owned color image counts against the attachment budget.
Preparation that never borrows attachments leaves the page image untouched.

This path keeps per-material shader tone mapping. Incompatible whole-image HDR
fallback remains an explicit error before replacing the owner, as it does for
preserved canvases. Ordinary native WebGPURenderer output is unchanged.

The focused canvas/facade/session/framebuffer suites pass 119 tests. New checks
execute the actual pinned clear-color code and inspect recorded render, copy,
resolve and presentation commands, including both byte formats and lifetime
boundaries. These checks do not execute WGSL or establish browser pixel parity;
that requires a capable browser lane.

## WebGL-authored output color spaces

The WebGL-authored route accepts linear sRGB and registered Display-P3 /
linear Display-P3 output in addition to sRGB. Register the pinned addon color
definitions with `ColorManagement.define` and set `renderer.outputColorSpace` as
in the source application. Non-sRGB canvas draws use their actual ShaderLib or
authored shader programs, including source primary conversion and transfer,
before blending into a plain UNORM attachment. Clear colors and solid backgrounds
use the source `Color.getRGB` output conversion before premultiplication. Public
render targets retain their independent working-linear shader output.

Each accepted draw retains its output selection during deferred preparation.
The native canvas uses the source definition's `drawingBufferColorSpace`.
Changing that gamut resets drawing-buffer color/depth/stencil history, including
an A-to-B-to-A change with no draw in B, as required by the
[WebGL drawing-buffer contract](https://registry.khronos.org/webgl/specs/latest/1.0/#DOM-WebGLRenderingContext-drawingBufferColorSpace).
Changing only the transfer within one gamut preserves existing attachments and
`preserveDrawingBuffer` history. These operations share the existing render FIFO.

Synchronous output-space mutation inside draw, shadow, sorting, shader-compile or
program-cache-key callbacks explicitly requires exact execution; it cannot queue
a delayed reset behind the draw whose attachment it changes. Unregistered color
spaces preserve the source lookup error. Non-sRGB output also requires the source
program path rather than a previously selected whole-image HDR fallback.

`node --test tools/ingest/three_renderer_webgl_color_spaces.test.mjs` compares
generated programs and clears with the unchanged pinned `WebGLProgram` and
`WebGLBackground`, and records queued output, display configuration, history
reset, offscreen behavior and callback boundaries. Canvas-owner regressions cover
the actual configuration descriptors. These are source/recording checks;
rasterized browser parity and hardware performance remain separate gates.

## Deferred multi-viewport WebGL animation frames

The WebGL facade retains one drawing buffer for each accepted
`setAnimationLoop` callback, including when `preserveDrawingBuffer` is false.
Cold shader/scene preparation can cross several browser presentation turns;
all canvas renders and framebuffer copies accepted by that callback keep the
same private frame token. Each presentation copies the accumulated owned color
into a freshly acquired swapchain texture. A later callback starts transparent
black, depth one and stencil zero, so this does not turn the default canvas into
a permanently preserved buffer. Explicit `preserveDrawingBuffer: true` retains
its existing cross-frame behavior. Manual render calls outside these callbacks
keep their direct presentation lifetime; straight-alpha canvases use the owned
encoded buffer and final alpha conversion described above.

The existing canvas owner allocates history color lazily on the first actual
attachment use. Warm callbacks reuse that storage; resize, in-flight old
attachments and MSAA resolve storage remain charged to the existing byte budget.
Stopping/replacing an animation loop does not invalidate work already accepted
with its token. Deferred draw callbacks and public framebuffer copies inherit
the accepted frame, without retaining a swapchain texture across host turns.

The legacy program raster also matches GL's disabled depth test: fragments do
not write depth when `depthTest` is false, even if the material's depth mask is
true. The private mask state retains that mask for later clears and draws.
This prevents fullscreen backgrounds from occluding later model passes.

The extended `program_preserved_canvas.mjs` passes all 42 checkpoints and
175,104 RGBA bytes exactly on Chromium 151.0.7922.34 / SwiftShader, including
12 default-canvas animation checks at one/four samples. One cold callback spans
five actual presentation textures while retaining four viewports and depth;
warm/new-shader frames, a partial next frame, resize and queued work after loop
stop also match source. The existing 30 explicitly preserved-canvas checks stay
exact, with zero GPU/browser errors throughout.

The unchanged original H6 `webgl_postprocessing_advanced` application now builds
and renders all four quadrants through owned WebGPU with its real assets; the
model remains visible across three controlled frames with no missing assets or
GPU/browser errors. This is functional application evidence, not complete
numerical closure: full-page comparison still has differences concentrated in
FilmPass grain, and hardware performance remains unmeasured.

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

## Standard composer stencil masks

The pinned r186 EffectComposer, MaskPass/ClearMaskPass and BloomPass use a
compiler-private stencil adapter on the owned WebGPU renderer. Dependency and
source fingerprints authorize only their finite state expressions; original
classes, exports and `instanceof MaskPass` identities remain intact. The H6
`webgl_postprocessing_advanced` graph now selects `general-webgpu`. Changed
sources, custom or mutated passes, open import graphs and arbitrary native
context/state consumers retain exact routing. Ordinary `getContext()` still
refuses on the owned renderer; no GL context is fabricated.

Per-renderer state captures explicit pass changes around accepted renders and
resolves material changes in actual draw order. Color/depth locks preserve
contents during mask clears. Stencil locks suppress test/write-mask setters,
while comparison/reference/operation setters remain live, including those from
stencil-writing materials. Composer's NOTEQUAL copy preserves the outside of
each pingpong image, and ClearMaskPass restores ordinary full-frame effects.
Pre-activation stencil draws, queued frames, failed requests and cancellation
retain ordered state without replaying source callbacks.

Mask raster variants reuse prepared shaders, geometry and texture bindings.
The existing program-mesh owner caches immutable bindings, charges them against
its capacity, and keeps earlier queued draws alive. Ordinary renderers do not
allocate pass capture tokens until a verified pass activates the adapter.

`program_composer_masks.mjs` compares both stored composer buffers with pinned
source across 12 checkpoints and 299,008 RGBA bytes: normal/inverse masks, two
masked swaps, depth-failing mask geometry, color/depth preservation, material
stencil inheritance, ClearMaskPass, resize and queued controls. Nineteen of 24
images match byte-for-byte; the other five differ by at most one channel unit,
with zero outliers under the existing WebGL tolerance and zero GPU/browser
errors on Chromium 151.0.7922.34 / SwiftShader. Full H6 application and hardware
performance closure are separate gates. Locked partial stencil clears and
locked masks combined with shadow/transmission prepasses are explicit runtime
boundaries.

## Bloom, depth of field, temporal accumulation and supersampling

The pinned `UnrealBloomPass`, `BokehPass`, `AfterimagePass`, `SSAARenderPass` and
`TAARenderPass` now execute through the owned WebGPU renderer. Their original
classes, shaders, blur chains, accumulation weights and pingpong targets remain
source JavaScript. Complete dependency fingerprints authorize these consumers;
changed shaders, custom subclasses, replaced methods and escaped instances
retain exact routing.

SSAA and TAA temporarily jitter a camera and restore it before returning. Their
verified scene-render expressions now enter a private capture scope so every
accepted draw keeps its own projection. Ordinary callback-free perspective and
orthographic cameras also retain their accepted matrices. Public reversed-depth
transitions still update the original camera, including its projection inverse.
Custom projection methods and callback-bearing scenes keep their established
ordinary rendering path; required supersampling capture explicitly refuses
them. Shadows, custom sorting, instancing, skinning and batching remain outside
this supersampling capture profile.

The verified `EffectComposer.setSize()` body runs inside the existing target
owner's resize scope. Queued fullscreen draws retain their sampled target
generations, including bloom's basic-material screen copy, while preparation
and submission bind the same accepted images. Two queued resizes therefore do
not replace earlier color or temporal-history inputs. Retired generations stay
charged to the existing resource budget until their leases are released. This
does not copy GPU images to the CPU or introduce a second target owner.
Explicit disposal still invalidates accepted work; direct `pass.setSize()`
outside the verified composer scope requires draining pending work first.

`program_advanced_passes.mjs` compares the unchanged source passes with pinned
WebGL across 39 checkpoints and 476,672 RGBA bytes. It covers cold shaders,
queued controls, normal/inverse bloom masks, TAA reset and complete 32-sample
accumulation, plus two queued resizes for every family. Chromium 151.0.7922.34 /
SwiftShader differs by at most one channel value, with zero pixels outside the
existing WebGL tolerance and zero GPU/browser errors. Recording-facade tests
also inspect distinct SSAA/TAA projections, accumulation weights, blur direction,
depth-material selection, history identity, sampled dimensions and explicit
disposal. These checks establish these source-pass paths; full application
coverage and physical-device performance remain separate gates.

## SSAO and GTAO normal buffers, occlusion and denoising

The pinned `SSAOPass` and `GTAOPass` now select `general-webgpu` with their
original sampling kernels, noise textures, blur/Poisson denoise shaders and
output modes. Their complete pass, shader and noise dependencies are verified;
modified sources or shaders, replaced methods and open consumers retain exact
routing. Public quality controls and queued composer resizes use the existing
program and target owners.

WebGL `MeshNormalMaterial` now uses its source ShaderLib program. The native
node-material path applies a color conversion that the WebGL normal buffer
does not perform; sharing that path previously corrupted the AO inputs. The
correction preserves native node-material behavior and fixes the WebGL normal
display and sampled normal-buffer values together.

Both source passes temporarily hide points and lines for their normal/depth
draw, then restore the public objects. The private verified-pass scope now
retains that exact exclusion list and its descendants through deferred
preparation and drawing. It neither rewrites public visibility during
submission nor freezes ordinary scene visibility or LOD selection. Exclusions
use the existing material capture limits and release with the accepted request.
The required capture profile remains ordinary callback-free scenes/cameras
without shadows or custom sorting; unsupported capture fails before retaining
additional targets.

`program_ambient_occlusion.mjs` compares 27 source/candidate checkpoints and
330,752 RGBA bytes on Chromium 151.0.7922.34 / SwiftShader, with zero GPU/browser
errors. It covers raw AO, blur/denoise, depth, normal and diffuse outputs,
temporary point/line exclusion, queued quality changes and two queued resizes.
Twenty-five checkpoints differ by at most one channel value. Each 80-by-56
resize image has 34 pixels above the 12-channel threshold (0.759%, within the
existing 2% budget), with maximum differences of 124 for SSAO and 112 for GTAO.
This is tolerance-based equivalence, not byte-exact parity or a hardware
performance result. `SAOPass` remains on the exact route: its sinusoidal random
hash amplifies fullscreen interpolation differences at some image sizes, and
its numerical parity has not been established.

## Selection outlines and queued mask inputs

The pinned `OutlinePass` now executes its original selected-object depth/mask
draws, visible/hidden edge classification, downsampling, separable blur and glow
composition through the owned WebGPU renderer. Source and dependency
fingerprints authorize its finite visibility/stencil operations; original
classes and shaders remain intact, and changed or escaped consumers retain
exact routing.

Each verified scene draw retains its temporary visibility exclusions and
background selection. A shader override uses the existing pass snapshot owner
to retain its uniform values and sampled target generations. That keeps the
mask camera's near/far values and depth-comparison texture matrix associated
with the accepted draw, even when the application queues another selection,
camera pose or composer resize. Stable private material views reuse the
existing program caches and snapshot budgets. Ordinary unscoped materials keep
their existing live-uniform behavior.

`program_outline_pass.mjs` compares 17 source/candidate checkpoints and
201,728 RGBA bytes on Chromium 151.0.7922.34 / SwiftShader, with zero GPU/browser
errors. It includes intermediate masks/depth/edges, cold and warm camera
movement, queued selections and camera/frustum/control changes, two queued
resizes and empty selection. Cold output, mask and edges, the moved camera,
the first queued frame, queued resizes and empty selection are byte-exact.
Packed depth differs by at most two channel values and the third queued frame
by at most one. A multi-selection frame has five differing silhouette pixels
in its mask; blur/glow expands these to 53 output pixels above the 12-channel
threshold (1.73%, within the existing 2% budget; maximum difference 126).
These are the established lane's tolerance results, not byte-exact family
closure or a physical-device performance measurement. Required captured scene
draws retain the callback-free ordinary scene/camera profile described above.

## GPUComputationRenderer texture simulations

The pinned `GPUComputationRenderer` addon now runs its floating-point texture
simulations through the owned WebGPU program and render-target paths. Its
original class, synchronous `init()` and `compute()` methods, variable objects,
dependency lists and alternating targets remain retained source JavaScript.
Each variable reads the previous step's inputs while writing its own next
target. Queued calls retain their accepted uniforms and texture identities;
the resulting textures can feed ordinary vertex and fragment shaders without
an intermediate CPU copy.

The addon originally checks `renderer.capabilities.maxVertexTextures`. The
compiler verifies the exact r186 addon, resolved `Pass.js` and core/build bytes,
then lowers only that guard to a private vertex-sampling availability query.
Every admitted WebGPU device supports this operation. The native renderer is
identified by a private registry; public flags cannot establish this identity.
Actual source renderers still perform the original strict-zero comparison and
capability access. Public GL limits are not fabricated or exposed by this
adapter. Modified dependencies, open extensions, escaped instances and replaced
methods retain exact ownership.

`gpu_computation_renderer.mjs` builds the unchanged addon through production
routing and compares it with source WebGL. Ten checkpoints cover float32 and
float16 targets, two mutually dependent variables, cold initialization, queued
uniform changes, current/alternate histories, explicit texture resets, manual
filtering and vertex/fragment texture consumers. Chromium 151.0.7922.34 /
SwiftShader matched all **5,120 state components and 122,880 RGBA bytes
exactly**, with zero browser or GPU errors. The browser fixture retains the
existing WebGL image tolerance of 12 channel values over at most 2% of pixels;
this run required none of that allowance. Six source-proof and ownership test
groups also pass. These checks establish this addon execution path, not full
simulation-family closure or hardware performance.

## Native multiple-render-target mip chains

Native 2D targets can render and sample mip chains on multiple
color attachments. Each attachment retains its own format, allocated mip count,
sampler and generation policy. Rendering an authored level binds that level on
every color and requires it to exist on every attachment; sampling, copies and
readback select the requested attachment's own levels. Four-sample targets render
at level zero and resolve before generating each selected color's lower levels;
authored nonzero-level rendering requires one sample.

Source `generateMipmaps` flags are captured per accepted render or texture copy.
Only the selected colors regenerate after that operation. Clear-only calls keep
lower levels intact, and changing a generation flag does not replace initialized
storage or erase other attachments. Copies into a generated level-zero color
complete its pyramid before subsequent consumers. Per-level depth attachments,
queued generation ownership and allocation overlap remain within the existing
resource budgets.

Public `readRenderTargetMipPixelsAsync` admission uses the selected color's
captured mip count and mip-local extent, including a secondary chain longer or
shorter than the primary attachment's chain. Invalid levels reject before
retaining a target or allocating native textures/staging buffers, even for an
empty rectangle. Legacy owners with only a primary mip count remain supported;
depth readback retains its separate single-level contract.

`native_mrt_mips.mjs` compares mixed sRGB-byte, linear-byte and half-float targets
with pinned r186, including authored overwrites, independent queued generation,
clear history, explicit LOD sampling and unequal color mip counts. Chromium
151.0.7922.34 / SwiftShader matches the stored and sampled texels exactly with
zero GPU/browser errors. That browser comparison covers ordinary 2D MRT.

Native 2D-array MRT now admits independent authored/generated color mip chains,
including per-layer generation, one/four-sample resolves and separate history
images. Native volume MRT admits independent authored chains: render levels must
exist in every attachment, while sampling, uploads, copies and readback can
select a longer chain on one color. Volume mip extents shrink in XYZ; depth
attachments retain their independent 2D/array allocation. The added production
owner/compute recording tests cover these paths, budgets and resize lifetimes;
they do not extend the browser pixel evidence above. Automatic volume generation,
volume history, cube MRT and WebGL-authored MRT mip pyramids remain excluded.
See [THREE_RENDER_TARGETS.md](./THREE_RENDER_TARGETS.md) for the current profiles.

## Native mipmapped pass histories

Native ordinary 2D PassNode histories keep a complete mip chain for each
physical current/previous color image, including independently selected MRT
attachments. History swaps change the selected image without moving texels or
discarding either pyramid. Dirty levels and cached filtering steps belong to
the physical texture; each accepted render captures its generation choices.
One- and four-sample histories resolve before filtering, and clear-only calls
preserve lower levels. Late history activation preserves the existing current
image. Public texture copies can consume either history role at an allocated
mip level without requiring that image to be filterable.

The existing target owner charges both complete resolved chains, base-level
MSAA storage and per-level depth attachments. Queued generations retain their
storage and byte charges across resize; changing generation flags does not
replace initialized images. The history borrow covers drawing, resolve and
selected filtering as one synchronous operation.

`native_history_mips.mjs` compares mixed half-float/byte histories with pinned
r186, including current/previous explicit-LOD sampling, queued complementary
generation flags, late attachment activation, authored lower levels and resize
while readbacks remain pending. All 14 checkpoints, 254 images, 33,784 components
and 54,600 stored bytes match exactly on Chromium 151.0.7922.34 / SwiftShader,
with zero GPU/browser errors. Hardware performance is unmeasured.

Native 2D-array histories retain all layers of each physical current/previous
image. Public PassNode subclasses can render several layers, sample their own
previous-frame layers during rendering, and expose them through the source
`getTextureNode().depth(layer)` and `getPreviousTextureNode().depth(layer)` API.
The subclass must preserve its layer count in `setSize()` and select each layer
when rendering; ordinary source PassNode sizing resets the target depth to one.
Independent MRT histories preserve their formats and late activation. Four-sample
histories retain a separate multisample image per layer and physical color;
ordinary or per-layer depth stays shared between the two temporal colors.

Single-color arrays also retain complete generated or authored mip chains for
both histories. A selected render view includes its physical image, layer and
mip level, while filtering updates every layer of only the selected image.
Recognized automatic subclass resizes preserve accepted older generations with
the same layer count until queued consumers and GPU work complete. Layer-count
changes and explicit disposal invalidate those generations. All array layers,
MSAA images, mip chains and private mip depth remain charged to the existing
target budget.

`native_history_layers.mjs` compares actual temporal accumulation in three layers
against pinned r186, with mixed half-float/byte MRT, four samples, generated mips,
late auxiliary-history activation and queued renders/readbacks across resize.
All 15 checkpoints, 261 images, 67,320 components and 114,672 stored bytes match
exactly on Chromium 151.0.7922.34 / SwiftShader, with zero GPU/browser errors.
The added layered MRT mip coverage described above is verified by owner
recording tests and is not part of this browser comparison. Sampled multisampled
array depth, cube histories and volume histories remain outside this profile.

## Registered native output color spaces

The native `WebGPURenderer` route accepts output spaces registered through the
pinned `ColorManagement.define()` API, including the optional `ColorSpaces.js`
addon's Display-P3, linear Display-P3 and extended-sRGB definitions. The working
space remains linear-sRGB. Output follows the source transfer and matrix
construction after tone mapping and before alpha premultiplication. Disabling
color management skips both the primaries conversion and transfer.

Queued draws and clears retain their accepted output profiles and exposure.
Changing only the output conversion replaces the final output program while
preserving linear scene color, depth and stencil history. Independent canvas
targets retain their own sessions, and native framebuffer copies continue to
read the same linear scene storage. Output matrices are captured before
asynchronous preparation and remain stable across explicit device recovery.

This preserves the pinned backend's canvas configuration: its effective display
color space remains sRGB. Selecting Display-P3 changes the source-authored
output transform; it does not establish physical wide-gamut display accuracy.
`native_color_spaces.mjs` compares actual canvas storage with unchanged r186:
10 cases and 16 presentations match all 15,872 bytes exactly in Chromium
141.0.7390.37 and 151.0.7922.34 / SwiftShader, with zero GPU/browser errors. Cases include byte and
half-float output, ordinary and node materials, signed/HDR values, tone mapping,
alpha, MSAA, queued changes and retained drawing. Hardware performance remains
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
  XR, custom tone mapping on the native output path,
  node graphs outside the initial unlit graphics profile, the remaining physical-material extensions,
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
