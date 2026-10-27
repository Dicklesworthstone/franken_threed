# Native TSL graphics programs

The native `WebGPURenderer` facade can draw authored TSL graphs on ordinary
meshes, CPU- or storage-backed `InstancedMesh` objects, `InstancedBufferGeometry`
and `BatchedMesh`. It retains the exact r186
`WGSLNodeBuilder`, node classes, uniform packing
and `NodeFrame`; FrankenThreeD owns the WebGPU pipelines, geometry residency,
uniform buffers, command encoding and submission. The path does not construct an
upstream renderer, translate through ESSL reflection, or claim measured
acceleration or a Rust implementation of node evaluation.

## Implemented graph surface

Built-in `MeshBasicMaterial`, `MeshToonMaterial`, `MeshLambertMaterial`,
`MeshPhongMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`,
`MeshNormalMaterial` and `MeshMatcapMaterial` can also
carry authored node slots. Their exact pinned `StandardNodeLibrary` conversions
supply the native graph while the builder, callbacks and disposal retain the
original source material identity. Ordinary materials also enter this path when
they use bump, displacement or baked light maps, or Physical sheen, iridescence, anisotropy,
retroreflectivity, transmission or dispersion and their maps. They require no authored node slot or
application rewrite. Other materials keep their existing fixed rendering path.
When one material group on a mesh needs native shading, its other groups use the
same source library conversion and geometry owner. All selected groups must pass
admission before any node update runs. Custom material subclasses and overridden
conversion methods remain outside this entrance.

Source material maps now lower through the pinned material accessors: base
color, alpha, emissive, normal, bump, displacement, AO, light, specular, roughness,
metalness, gradient and the admitted Physical lobe maps. Their UV channel,
transforms, color space and normal-map mode remain source behavior. Shared
metalness/roughness textures retain one sampled binding; changing a map's
sharing relationship with another map or authored texture rebuilds the binding
layout. Compatible image replacement retains the compiled program and each
queued draw's source pixel version. AO and baked light maps use the pinned
material lighting nodes, including independent UV channels, live intensities and
source diffuse/specular occlusion. Authored `aoNode` graphs use the same lighting
context. Basic light maps keep their source normalization and do not start
consuming scene analytic lights. Ordinary AO-only materials retain their existing
fixed path; AO also composes with native graphs and Physical features. Standard
and Physical material environment maps compose with these native graphs through
the pinned source environment nodes and PMREM owner.

Automatically selected native materials capture active lobe values and map image
versions independently for queued renders. Ordinary fixed-path requests now also
capture material selectors and callback-free scalar, color, vector and plane
values before asynchronous preparation. A fixed-to-native-to-fixed sequence can
remain queued while each accepted render keeps its own material state. Material
arrays, overrides, raster state, clipping and fog flags use those captured values;
cache and disposal ownership remain keyed by the original source materials.

Snapshot budgets are released after submission, failed preparation or disposal.
Native packets retain their existing lifecycle. Geometry, texture objects and
fixed-path image upload histories remain live; metadata and custom value objects
retain their source identities. Exotic material-array selectors and custom
accessor-backed values preserve their prior live behavior. This is a material
snapshot, not a deep snapshot of the entire mutable scene.

Use `MeshBasicNodeMaterial`, or an unlit base `NodeMaterial`, with `colorNode`,
`positionNode`, `vertexNode`, `fragmentNode`, `opacityNode`, `alphaTestNode` and
`outputNode` and `depthNode`. Ordinary source node arithmetic, functions and custom scalar/vector
float attributes compile through the pinned builder. Native scalar, integer,
boolean, vector and matrix uniforms retain their source group ordering and
padding. Clip-space vertex and fragment graphs with no uniform groups also work.

Authored `TSL.buffer()` arrays use native uniform-buffer bindings. The current
raw-array profile accepts Float32 `vec3`, `color`, `vec4` and `mat2`/`mat3`/`mat4`,
plus matching Int32/Uint32 three- and four-component vectors. Callers retain the
pinned source array layout: `vec3` elements have four-word strides and `mat3`
columns are padded. Final unused padding may be absent from the source view.
Managed `uniformArray()` inputs and raw scalar, two-component or boolean arrays
still need an additional profile.

Each source group may contain several uniform buffers alongside textures and
read-only storage. Every source binding keeps its index, visibility and exact
size, including separate vertex/fragment bindings for the same authored buffer.
Per-binding device limits remain enforced when the whole aligned group packet
exceeds one buffer's binding-size limit. Pipeline dynamic-binding and shader-stage
limits are checked before native allocation.

Default object-group arrays capture current typed-array values on every logical
draw, including edits without `needsUpdate` and compatible `.value` replacement.
Partial updates preserve bytes outside the requested ranges; a first partial
upload leaves untouched allocation bytes zero. Custom uniform groups retain
their source version gating. Queued draws retain independent packets, and render
bundle reuse includes every dynamic uniform offset.

The direct-module `native_tsl_buffer_arrays` browser comparison produced an exact
320×240 match against pinned r186: zero different pixels, mean difference zero,
maximum difference zero. It exercised two meshes sharing a material, four raw
array bindings across the shader stages, and three queued targets with in-place
edits and compatible replacements, repeated at frame 8 before poisoning the
source arrays. Chrome 151 software Vulkan completed the existing 30-frame gate
with no page, console, GPU or request errors. This is shader/rendering evidence;
it does not claim a fresh bundled-build check or hardware performance.

Authored `MeshToonNodeMaterial`, `MeshLambertNodeMaterial` and
`MeshPhongNodeMaterial` additionally use the pinned lighting models with ambient,
hemisphere, directional, point and spot lights. They accept `normalNode` and
`emissiveNode`; Phong also accepts `shininessNode` and `specularNode`. These slots
compose with ordinary TSL arithmetic and the sampled texture graphs below.

Source `LightProbe` irradiance is also available to ordinary and Node Toon,
Lambert, Phong, Standard and Physical materials. The exact pinned `LightProbeNode`
packs all nine spherical-harmonic coefficients with source intensity and FRAME
updates. Queued draws retain separate coefficient bytes without rebuilding a
compatible shader. Visible probes on the camera's layers select this route
automatically; disabled or excluded probes retain each request's inactive state.
Unlit meshes, sprites, lines and points keep their fixed path in mixed scenes,
and probe children remain independently traversable. Probe-lit shadow receivers
and casters use the existing shared native shadow owner; skipping a probe in the
fixed path does not shift another light's shadow index. This does not implement
light-probe grids or additional custom light models.

Source `RectAreaLight` now uses the pinned `RectAreaLightNode` and LTC lookup
textures on ordinary and Node Standard/Physical materials. Applications retain
the source initialization step:

```js
THREE.RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
```

The unchanged addon supplies the source half-float or filterable-float tables;
the existing texture owner uploads and shares them. Camera-space position,
orientation, half-width, half-height, color and intensity retain the source
RENDER update frequency. Queued draws preserve independent values, and removing
the light restores ordinary fixed rendering where no other native feature is
needed. Lit Toon/Lambert/Phong materials retain the source setup lifecycle and
its initialization prerequisite without inventing an area-light contribution.
Unlit materials, sprites, lines, points and independent light children keep
their existing paths. Selection honors visibility, camera layers and the
renderer lighting flag, and area-driven materials share native shadow ownership.

Thirteen program tests cover both source LTC formats, source shader bodies,
updates, initialization errors and registration identity. Two capture cases and
six complete facade cases cover routing, actual lookup-table upload bytes,
queued dimensions/radiance, mixed fixed objects, inactive ArrayCamera scenes and
shadow ownership. An additional facade test combines ordinary Physical
transmission with area lighting. These are source and recording-device checks;
browser pixels and performance remain unverified.

Explicit `renderer.library.addLight(NodeClass, LightClass)` registrations can
supply direct analytic illumination, including the pinned H7 `SunLight` and
`SunLightNode` addons. The light must directly extend the source `Light`, and
its registered node must directly extend `AnalyticLightNode` while retaining
the source setup/build/update lifecycle. The source node computes SunLight's
world-position-to-origin direction and color/intensity; FRAME updates stay
shared across objects and recompilations. Queued draws retain their captured
direction and radiance.

Constructor, registration and direct-light graph changes participate in program
selection. Changing the registered node class of a previously used light is
refused because the source retains its analytic node by light identity; a fresh
light can use the new registration. Projected maps and custom light color
nodes remain explicitly refused. The registered SunLight also retains its source
two-cascade shadow atlas through the native shadow owner described below.

Unassigned slots retain their source material color, emissive, specular and
shininess behavior. Both material arrays and scene overrides supply the effective
material to the builder's context.

Authored `MeshStandardNodeMaterial` and `MeshPhysicalNodeMaterial` use the same
direct-light path with the source GGX model and `metalnessNode`/`roughnessNode`.
Physical also accepts clearcoat strength, roughness and normal; sheen color and
roughness; iridescence strength, IOR and thickness; anisotropy; retroreflectivity;
IOR, specular color and specular intensity; transmission, thickness, attenuation
color/distance and dispersion nodes. Unassigned slots read their
ordinary material properties. Enabling or disabling a Physical feature selects
the corresponding source shader branch; changing its live value keeps the
program. The source's shared 16×16 RG16F DFG lookup texture goes through the same
versioned texture owner as authored textures, preserving its aliasing across
materials and clearcoat samples.

### Native transmission and viewport snapshots

Ordinary and Node Physical materials execute the pinned transmission, dispersion
and volume equations, including transmission and thickness maps. The native
`ViewportTextureNode` lifecycle records a framebuffer copy at the physical draw
which requested it. GPU submission draws the preceding color span, copies its
resolved color and generates mipmaps in one encoder, then resumes color, depth
and stencil with load operations. The resumed pass retains viewport and scissor.
All transmissive `DoubleSide` back faces precede the complete transparent/front
list; ordinary alpha double-sided draws retain their consecutive side order.
Shared front/back viewport nodes retain the source RENDER-frequency deduplication,
including a front snapshot first requested by an earlier alpha material.

The framebuffer owner preserves captured extents, texture generations, sampler
settings and temporary mip flags across queued renders. Canvas refraction reads
the linear scene target before output conversion, with the requested byte,
half-float or float precision and resolved MSAA. Nodes observe the pinned internal
source RenderTarget identity, type and size; application target selection is
restored after capture. Linear canvas output without tone mapping keeps its direct
path. Existing linear targets now preserve explicit color/depth/stencil loads.

Public `copyFramebufferToTexture(texture, rectangle)` also works outside node
updates after renderer initialization. It captures the selected source target,
floored Vector2/Vector4 rectangle and destination generation, then submits in the
same FIFO as renders and compute. Prepared copies submit synchronously; a copy
behind a cold render waits for that producer. A later material can sample the
copied image through the same shared GPU owner.

Public copies read resolved primary color, including MSAA targets and the
persistent linear canvas intermediate, without rendering the scene again.
Copy and destination mip generation share one command encoder. Destination
resize/version changes preserve accepted older generations, and the initialized
public texture type remains stable until its version advances. Disposed sources
or destinations reject and release their queued leases; an invalid destination
does not acquire or replace presentation. Ordinary public source-target resize
retains the existing stale-use rejection contract.

This path uses the existing owned WebGPU renderer and texture owners, with no
readback or retained renderer submission. Depth framebuffer copies, the
WebGL-authored copy route remain unimplemented. Native ArrayCamera copies preserve
their source draw boundaries; physical transmission pixel parity across views
remains unverified.
Distinct scene overrides retain both the original transmissive back/front phases
and the effective override's alpha side expansion, including four physical draws.
These are remaining compatibility tasks, not permanent exclusions.

Fourteen public-copy facade cases cover source/destination selection, all three
canvas precisions, direct linear output, MSAA resolves, FIFO ordering, mips,
version/type transitions and failure cleanup. These run alongside the existing
framebuffer-owner and transmission regressions.

Thirteen framebuffer/output tests, ten program tests, nine capture-order tests and
twelve complete facade integration tests cover this implementation. Program WGSL
matches the pinned builder byte for byte. Recording-device tests establish actual
copy/mip/draw ordering, queued uniforms and texture extents, mixed fixed/native
draws, empty ranges, compilation without copies, and output precision; they do not
establish browser pixel parity or performance.

Native Standard and Physical graphs also use ready 2D equirectangular, source
CubeUV or six-face CPU-data/decoded-image cube textures as `scene.environment`
or material `envMap`. A material map overrides the scene environment and uses its own
`envMapIntensity` and `envMapRotation`; scene fallback retains the scene controls.
Changing this ownership selects a fresh graph even when both properties share
the same texture. Stale captures reject before authored node callbacks run.
The pinned native `PMREMGenerator` supplies
conversion/filter graphs and atlas layout; FrankenThreeD captures and submits
those ordinary node draws through its own renderer. Generated PMREM targets and
the source DFG texture are borrowed through the same versioned texture owner.
The retained renderer/backend does not submit commands. Live environment
intensity and rotation remain source uniforms; replacing the environment selects
its corresponding atlas and preserves already queued draw bindings. A wholly
native scene bypasses the separate fixed-material environment filter.

Standard and Physical materials also accept authored `envNode` graphs. Procedural
irradiance, explicit `pmremTexture()` inputs, direct texture nodes and material
references use the pinned `EnvironmentNode` radiance, roughness, clearcoat and
rotation contexts. An authored graph takes precedence over material and scene
maps. Direct texture/reference replacements rebuild their source PMREM setup;
context-wrapped texture expressions retain ordinary source graph semantics.
Per-object updates and queued uniform values remain independent. Replacing or
removing the environment graph rejects stale captures before callbacks, then
allows a fresh graph to restore the source fallback.

Scene `environmentNode` graphs now use the same pinned lighting context. Material
`envNode` and `envMap` retain precedence; a valid scene node overrides the ordinary
scene texture. Procedural irradiance, PMREM expressions, texture and material
reference inputs preserve scene intensity, rotation and source update frequencies.
Ordinary Standard/Physical materials automatically select native shading when
they consume the graph. Fixed unlit draws in a mixed scene retain the captured
override decision without loading the unused scene texture. Graph replacement,
removal and source-invalidated structural changes select fresh programs; accepted
uniforms and texture leases keep their per-request values.

The public native `PMREMGenerator.fromScene()`, `fromEquirectangular()` and
`fromCubemap()` methods execute the pinned helper through F3D-owned render requests and return
real CubeUV render targets. Scene generation supports the source capture size,
position and near/far controls, optional Gaussian pre-blur, and GGX filtering.
Each of the six face renders retains its own camera matrices. The helper's
temporary null background stays null for those accepted renders after it
restores the application scene.

Immediate generator disposal and later generator resizing retain private filter
resources until accepted work drains. Repeated scene generations use separate
private solid-background boxes, preserving each generation's color. The public
output target remains caller-owned and sampleable after generator disposal.
`fromCubemap()` accepts the data and decoded-image cube profiles below, runs the source cube
conversion and GGX filtering, and returns an ordinary CubeUV atlas for native
PBR environments. The WebGL/core placeholder route retains its existing behavior.

A same-size equirectangular or cube input can set `needsPMREMUpdate = true` between
native renders and regenerate its existing atlas. The cached graph must place
its exact source PMREM before-updates at the first draw boundary. A private
native sky is always first; a scene without that sky must have no selected
fixed-material draw candidates. The generated filter passes enter the renderer
FIFO after earlier accepted renders and before the new consuming render, outside
the whole-render texture/storage transaction. Mark CPU pixel uploads with the
source `needsUpdate` flag as usual.

Refresh after an authored callback, refresh first needed by a later draw,
simultaneous graph recompilation, and unsafe mixed fixed/native foreground
ordering fail with `THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH` before the unsafe
work runs. Atlas resizing still requires a fresh PMREM node. Replacing the source
texture remains admitted; directly supplied CubeUV textures keep their ordinary
upload-version behavior. The compatibility adapter's placeholder PMREM textures
remain outside this native profile.

Material-environment regressions compare complete vertex and fragment WGSL with
the pinned builder for ordinary and Node Standard/Physical materials. Recording
GPU tests execute automatic cube conversion, shared atlas use, six-face upload
versions, replacement and refresh ordering through the F3D renderer. Authored
environment tests also compare exact source WGSL and submit an ordinary material
whose sole native feature is `envNode`, preserving two queued irradiance values.
Scene-node coverage adds twelve source compiler cases and ten facade integration
cases, including mixed fixed/native draws, disabled lighting, compilation without
draw updates, shared shadows and real PMREM generation/refresh. The focused
compiler/environment/capture checkpoint passes all 232 cases without skips.
These checks
establish source construction, capture and submission behavior; browser pixel
parity and physical-device performance for these combinations remain unverified.

Native panoramic backgrounds use the pinned source sphere, camera projection,
far-plane placement and PMREM sampling graph. Equirectangular backgrounds with
positive blurriness and direct CubeUV backgrounds are captured before ordinary
scene draws, including scenes whose user materials follow the fixed route.
Blur, intensity, rotation and camera uniforms retain their values for each queued
call. The private sky mesh does not enter the user hierarchy and stays first
after mixed native/fixed sorting, including when sorting is disabled.

Backgrounds keep a separate source PMREM node for each panorama identity.
Queued A-to-B-to-A replacements can therefore use differently sized atlases
without retiring A's filtering targets. Authored `pmremTexture()` graphs also use
the native path. Replacing one authored node's source with a different required
atlas size fails before callbacks; use a new PMREM node for that replacement.
Ordinary same-size replacements remain supported.

Authored `scene.backgroundNode` graphs use the same private sky draw, including
H7's `screenUV.y.mix(color(...), color(...))` gradient. They take precedence over
`scene.background`, including Color and unsupported texture fallbacks. Their
ordinary admitted uniforms, sampled textures and update nodes retain the original
scene/camera context and their values for each queued render. Changing the
background graph retains earlier queued programs. The renderer owns its private
sky material and geometry; disposing it does not dispose the application's
background node. Unsupported graph resources fail before application callbacks
or submission.

Sharp equirectangular backgrounds continue through the fixed background pass,
with their selector, camera, intensity and rotation captured for each accepted
native render. Queued sharp-to-blurred-to-sharp transitions now preserve all
three states, including later edits to the public scene. Mutable Color
backgrounds likewise retain each call's clear value. These capabilities do not
establish full H7 or H8 compatibility.

Authored `cubeTexture()` graphs sample source CubeTextures containing exactly
six CPU DataTexture or decoded platform-image faces. The decoded profile accepts
ImageData, ImageBitmap, HTMLImageElement, HTMLCanvasElement and OffscreenCanvas
through the shared immutable-image owner. The compiler retains the pinned
direction, reflection/refraction mapping and explicit-level semantics and
declares cube views in source binding order. Compatible replacements reuse the
program; alias-partition or mapping changes select a fresh source graph.

All six faces and authored mip levels are captured for each accepted upload
version. Mutable images retain call-time pixels before asynchronous upload;
shared image identities are cloned once, and only private bitmap snapshots are
closed. The native upload profile follows the pinned face swap, flipY and
premultiply behavior, including authored/generated mipmaps and the source CPU
base-level flip order. The default WebGL texture pool keeps its prior behavior.
Float filtering retains the existing device-feature checks.

Ordinary Standard/Physical material `envMap` and scene cube environments select
the native environment graph automatically. The source PMREM helper performs
conversion and filtering through F3D-owned draws before the consumer. Versioned
refresh preserves earlier accepted faces. Fixed sprites, lines, points and Basic
meshes can share that scene; disabled lighting does not upload an unused cube or
send it through the fixed panorama owner.

Render-target color cubes and unsampled cube loads remain outside this profile.
Cube textures cannot occupy ordinary 2D material map slots. Eighteen decoded-cube
regressions compare exact pinned WGSL and source upload descriptors, exercise
ordinary PBR/PMREM submission and verify image ownership. Host/GPU recording
doubles do not establish actual browser decoding, color conversion or pixel
parity.

Authored `texture()` graphs sample ordinary 2D data, decoded image/canvas and
resolved color render-target textures. The pinned builder owns UV channel and
matrix selection, explicit/generated UVs, color-space interpretation, explicit
levels and vertex/fragment sampling. Bind groups preserve the source's exact
resource order: a uniform block can follow samplers/textures, and groups can
contain resources without a uniform block. Filterable float formats are admitted;
float32 inputs require the actual `float32-filterable` device feature.

Authored `storage(attribute, type, count).element(index)` reads in vertex and
fragment graphs share the renderer's compute allocation. Float32, Int32 and
Uint32 scalar/vector `StorageBufferAttribute` and
`StorageInstancedBufferAttribute` arrays are admitted, including source vec3
padding. Standalone Float32 `mat4` storage arrays use a 16-word stride. Flat
non-atomic `struct()` records also work with scalar/vector 32-bit
members and `mat2`, `mat3` or `mat4` fields. The pinned `StructTypeNode` defines
their padded stride; mixed integer/float backing bytes and matrix-column padding
remain unchanged. The H8 fluid particle record (`position: vec3`,
`velocity: vec3`, `C: mat3`) occupies 80 bytes and can be written by compute and
read directly in both vertex and fragment graphs.

A compute-writable source node can be read by graphics; the pinned builder
supplies read-only graphics bindings. Member, matrix-column and whole-record
writes are rejected, while explicit local copies remain writable. Struct names,
member order, names and types participate in the program signature, so even
same-size layout changes require a new descriptor before node callbacks run.
Splitting or merging unsized storage aliases, including nodes created inside
`Fn`, invalidates the compiled binding layout. There is no readback, second
graphics allocation, or CPU upload merely because a graphics draw consumes
compute-produced values. Nested, boolean and atomic struct members are not yet
admitted; a three-word struct is also refused because the source storage uploader
treats that stride as a padded vec3.

Ordinary meshes support `Mesh.count` with TSL `instanceIndex`, including
compute-driven scalar/vector and structured particle positions. Each logical
draw captures the count after before/current node updates and resource capture,
before after-updates. Queued preparation cannot replace it with a later live
count. Zero counts still run source node callbacks but emit no physical draw;
direct and render-bundle paths retain the captured count. Negative source counts
clamp to zero; admitted counts must fit an unsigned 32-bit integer.

`InstancedMesh` can use the same authored materials with ordinary Float32
`InstancedBufferAttribute` matrices and optional RGB colors. The matrix item size
is 16 and the color item size is 3, with instance divisor one. The pinned builder
chooses a matrix uniform buffer when it fits the actual device binding limit;
larger arrays use four `vec4` columns sharing one interleaved instance stream.
Small matrix arrays capture current values on each draw without requiring
`needsUpdate`. Large matrices and CPU colors preserve the source attribute's
version, partial-update ranges and `DynamicDrawUsage` behavior.

Float32 `StorageInstancedBufferAttribute` matrices and colors remain admitted,
including the source's padded four-word RGB storage. CPU and storage inputs can
be mixed. Compute-backed inputs borrow their existing GPU allocation; they never
copy stale CPU arrays into separate graphics storage. The retained source
builder applies each instance matrix to positions, transforms normals with its
inverse transpose, and multiplies instance RGB into the material color.
Authored `positionNode` runs after the instance transform, and its graph can read
other storage through `instanceIndex`.

Source-generated `BufferAttributeNode` scalar/vector streams from raw Float32
arrays also use private captured geometry views. Ordinary geometry attributes
remain live and can be replaced without being trapped in a stale private view.
Generated streams require fixed, unshared arrays, unnormalized float input and
the default upload callback. Public BufferAttribute/InterleavedBuffer inputs and
compute-backed attribute conversion need additional shared-residency profiles;
they are refused rather than allowing ordinary and generated consumers to
diverge after partial updates. The pinned instance nodes' private derived
matrix/color owners are admitted.

The program records the smaller matrix/color capacity, or the matrix capacity
when colors are absent. After node updates, the captured count must fit that
capacity; an excessive count is rejected before submission rather than silently
reduced. Zero instances retain the same callback behavior described above.

```js
const tint = TSL.uniform(new THREE.Color(0.2, 0.5, 0.8));
const shift = TSL.uniform(0);
const material = new THREE.MeshBasicNodeMaterial();
material.colorNode = tint;
material.positionNode = TSL.positionLocal.add(TSL.vec3(shift, 0, 0));
const mesh = new THREE.Mesh(new THREE.PlaneGeometry(), material);
scene.add(mesh);

renderer.render(scene, camera);
shift.value = 0.5;
tint.value.setRGB(0.8, 0.3, 0.1);
renderer.render(scene, camera);
// The first queued render retains its own program, draw selection and uniforms.
```

Graphs emit linear color into the native attachment. The existing native output
path owns tone mapping and output transfer. Blending, depth, stencil, bias,
winding and color writes use the same raster adapter as other native meshes.
Default node materials with no authored slots continue through the existing
fixed material path. Ordinary fixed meshes and authored node meshes can share a
scene; a single material array must contain only authored node materials when
that mesh takes the node path.

## Call-time semantics and ownership

Each initialized logical render synchronously selects native draws and captures owned
uniform byte packets before asynchronous pipeline preparation. Selection follows
source visibility, layers, LOD, geometry groups, frustum culling, render order and
WebGPU depth sorting. Active callback-free bindings are admitted before node
updates; object callbacks and overrides construct their effective programs at
the source draw boundary. Hidden objects remain inactive. The source camera receives its WebGPU
projection convention before graph construction. Source object and material
identities reach the builder and callbacks.

`NodeFrame` preserves FRAME, RENDER and OBJECT frequencies, shared update
references, before/current/after ordering and the source retry behavior when an
update returns `false`. A logical draw updates once even if preparation retries.
Frame time and render identity remain fixed for the captured call; nested calls
and thrown callbacks restore the surrounding context. `compileAsync()` prepares
bindings without running node updates or advancing animation/render counters.
Calls before initialization prepare at the source's asynchronous boundary, when
device capabilities are available.
The native frame clock starts during renderer initialization and continues on
host animation frames even when the user animation callback is null. Setting
`setAnimationLoop(null)` stops application callbacks; renderer disposal stops
the owned clock. This lets FRAME nodes, including analytic lights, update in
applications that call `render()` directly.

Lit graphs select actual source lights through scene visibility, LOD and camera
layers, including during compilation. The selected light identities and the
renderer lighting switch participate in program selection; live light values
remain uniforms. Source analytic light nodes share their retained identity across
objects and program rebuilds, preserving FRAME updates for radiance/falloff and
RENDER updates for view-space light positions and targets. Queued draws retain
the light uniform bytes captured for their logical call. The profile admits up
to eight selected lights and validates unsupported light kinds or shadow/projector
state before any application node update.

Graphics and compute share the renderer's existing sampled-texture snapshot
adapter and upload pool. Each draw leases its post-update texture identities,
sampler state and requested CPU pixel versions before after-update callbacks.
Later source edits cannot replace older queued contents. Image readiness joins
the renderer FIFO; the adapter keeps immutable private images until use finishes.
Render-target inputs borrow their native attachment views through submission and
observe preceding GPU writes without readback, re-upload or a UV replica. Sampling
the current destination is rejected before submission. Texture capture capacity,
retirement and disposal use the shared owner's existing bounds.

Storage reads capture the post-update attribute identity, requested upload
version and layout before after-update callbacks. Initial host data and marked
full or partial uploads share compute's upload history; rendering can initialize
storage without a preceding compute call. Queued renders submit their captured
versions in FIFO order, including calls targeting different render targets.
`compute(kernel); render(scene, camera)` consumes that dispatch's allocation and
GPU contents without re-uploading an unchanged host version. Unmarked CPU edits
cannot overwrite resident GPU results. Vertex and fragment aliases borrow one
allocation, and `compileAsync()` performs no storage upload or node update.

Storage capture has its own transaction, nested with texture capture when both
are used. Registered source texture producers may run compute prerequisites
before the first draw or texture read in every active parent capture. Completed
child passes already precede their parent in the FIFO. Ordinary callbacks and
compute after an earlier parent draw still require a separate submission boundary.
Leases release their captured bytes on success and failure; the compute
owner retains GPU lifetime. Conflicting host versions of one storage attribute
within a single logical render are rejected before any storage writes. They
require additional per-draw storage scheduling; separate logical renders already
retain each version.

For `InstancedMesh`, replacing a matrix/color attribute, changing its capacity,
or adding/removing instance colors selects a new graph. Data, upload-version and
draw-count changes reuse the graph, including the source RGB padding transition.
Replacement during before/current node updates is explicitly refused because
the source-generated instance nodes retain their compiled attribute identities.
After-update changes cannot replace the current draw's captured resources or
count; a later render compiles against the new instance attributes. Objects
sharing one material retain their own instance bindings. CPU capacity or backing
array changes require replacing the attribute object: the pinned instance nodes
cache their original CPU arrays. Replacing `.array` on an already captured CPU
instance attribute fails explicitly instead of retaining a stale source closure.

Generated vertex streams capture after source before-updates and before ordinary
node updates, matching source geometry-upload order. A first upload captures the
whole array without consuming its update ranges; later versioned or dynamic
updates preserve unrequested bytes and consume the ranges. Each logical render
retains immutable data even if later calls mutate or poison the public arrays.
Source histories survive program replacement and retire on geometry/renderer
disposal. `compileAsync()` establishes the initial upload history for visible
generated streams at preparation without running node update callbacks; hidden
objects do not acquire that history until their first visible use. Replacing a
program waits for submitted work before retiring its old GPU geometry owner.

Instance culling still uses the source mesh's CPU `boundingSphere` before node
updates. GPU writes do not refresh CPU matrices, bounds or raycasting data.
Applications with compute-driven transforms must supply conservative bounds or
set `frustumCulled = false`; the renderer does not infer GPU bounds or change that
source setting.

Captured graph descriptors, raster state, source list properties, draw ranges and
uniform bytes survive later source edits. Material/geometry disposal invalidates
the compiler cache, and subsequent graph or structural changes select a new
program. The facade also captures ordinary CPU geometry through the existing
immutable geometry owner at the source upload boundary. Shared geometry uploads
once per logical render; unmarked edits and untouched partial ranges stay stale,
including when a render callback is added after initial compilation. This is not
a deep snapshot of an entire mutable scene.
GPU geometry uses the existing owner, while native uniform groups receive
independent aligned slices. Later arena growth leaves earlier groups and bytes
valid. Disposed or lost owners cannot publish pending pipelines, including
programs with no uniform groups. Pending captured bytes share the facade's
bounded budget and are released when the request finishes.

Native object callbacks and scene overrides execute the pinned
`Renderer.renderObject` lifecycle with an owned draw handler. `onBeforeRender`
runs before effective shader construction, both alpha sides receive their node
updates, and `onAfterRender` runs once after that pair. Source identities,
material groups, `allowOverride`, original visibility and list partitioning stay
intact. Position, alpha, displacement and transparency inheritance uses source
code, including its successful restoration and intentionally retained alpha and
transparency fields. A callback changing `scene.overrideMaterial` affects later
list items. `compileAsync()` runs object hooks without node updates; warm bundles
do not replay object hooks or override inheritance. Material `onBeforeRender`
remains inert, as it is in the pinned WebGPU renderer.

This callback path currently requires a source Scene with wholly native mesh
draws. Mixed fixed/native hook lists, callback replacement of selected geometry,
custom BatchedMesh hooks and recursive GPU writes still require additional
capture boundaries and fail explicitly. Callback exceptions retain earlier
source mutations and do not replay updates or run later after-callbacks.

## Logarithmic and reversed depth

Native perspective and orthographic mesh scenes accept `logarithmicDepthBuffer`
and `reversedDepthBuffer`. The pinned node builder supplies fragment-depth
expressions and custom `depthNode` precedence. Source camera projection changes,
the public clear-depth accessor, all eight reversed comparison functions,
frustum conventions, shadows and scene-pass depth sampling retain their source
behavior. Queued calls keep their accepted projection, near/far, clear value and
depth uniforms. Existing perspective panorama/cube background rays also accept
reversed projection. Ordinary native ArrayCamera views share these conventions.
XR and depth modes on the WebGL-authored renderer facade retain their existing
explicit boundaries.

`three_renderer_depth_modes.test.mjs` checks source camera/raster oracles and
actual captured GPU commands. `tests/e2e/webgpu_renderer/depth_modes.mjs` compares
four queued frames per case against an independent pinned source run and source
repeat, including full RGBA bytes and float32 depth words. Software Vulkan
correctness does not establish hardware performance.

## Native sprites

Source SpriteNodeMaterial and authored SpriteMaterial graphs render on both
Sprite and Mesh objects with the pinned billboard vertex shader, including
positionNode, scaleNode, rotationNode, center, live rotation, size attenuation,
maps and fog. A Mesh using ordinary SpriteMaterial selects this native path
without authored slots. Sprite sorting uses the source object's origin, and
negative sprite scale retains the source billboard winding.

Sprite.count, Mesh.count and InstancedBufferGeometry.instanceCount remain
instanced GPU draw counts. Queued renders retain independent uniforms, instance
counts and marked instanced attribute histories. Structural attenuation and
anchor-presence changes rebuild the source graph. Ten source-builder and
recording-device regressions cover these behaviors; the four object/material
combinations produce exact source vertex and fragment WGSL. These tests do not
establish browser pixel parity or a performance gain.

## Native source shadows

Directional and spot VSM shadows now run the pinned ShadowNode depth pass,
vertical moment blur and horizontal moment blur through the native command
owner. The two moment targets use native RG16F storage, and receivers sample the
actual horizontal output allocation. Ordinary materials select the same native
shadow owner automatically. VSM includes receive-only objects in its source
shadow list and preserves each material's side unless shadowSide overrides it.

Private source map resizes retain all accepted depth and moment generations.
Queued frames keep their own map dimensions, radius and blurSamples; returning
to an earlier size creates another generation. Retired maps are released after
their readers drain. Explicit application disposal remains an invalidation.
Source autoUpdate, needsUpdate, shadow callbacks and compile-without-rendering
semantics remain in force.

Point lights under global VSM keep the source six-face cube comparison filter
and do not create moment passes. Registered SunLight keeps the source two-tile
PCF atlas fallback and warning; its source vsmPass is empty. This does not add
variance filtering to the SunLight atlas. Transmitted shadows and custom shadow
filter/replacement nodes remain outside the admitted profile.

Seven focused recording-device cases cover directional and spot moments,
queued resize/uniform history, explicit disposal, ordinary-material routing,
side overrides and the Point/Sun fallbacks. Native VSM browser pixels have not
been compared with the pinned reference.

## Native ArrayCamera views

Ordinary native WebGPU `ArrayCamera` scenes accept up to 256 perspective or
orthographic child views. The source parent selects layers, union-frustum
visibility and list order. Each physical material side then draws its matching
children in source order. Object callbacks receive the original parent, and
`NodeFrame` before/current/after updates keep their logical FRAME, RENDER and
OBJECT frequencies; adding views does not repeat updates.

The retained builder emits the source camera matrix arrays. Every submitted
view owns a camera-index uniform packet, while queued renders retain accepted
matrix bytes, child-layer selection and viewport values. Viewport inheritance,
canvas DPR, physical render-target coordinates, depth ranges and reversed
projection remain intact. An empty camera array retains the pinned non-culled
parent-camera fallback; a nonempty array with no matching child still runs its
source node lifecycle without submitting a geometry draw. View-count changes
construct a new uniform layout without invalidating earlier accepted packets.

The same route supports native `PassNode` scene output, ordinary/native mixed
lists, and native instanced sprites. Framebuffer copy effects remain before or
after the complete group of child views belonging to their source draw. Native
array `BundleGroup` and `BatchedMesh` command ordering still require separate
work and fail explicitly, as do XR, multiview and nested camera arrays.
WebGL-authored ArrayCamera rendering keeps its existing boundary.

`three_renderer_native_arrays.test.mjs` uses the pinned `renderObject` and
`WebGPUBackend.draw` as an ordering oracle and inspects actual submitted uniform
bytes and commands. These checks establish the tested execution behavior, not
browser rasterization parity or a performance result.

## InstancedBufferGeometry

Ordinary meshes using `InstancedBufferGeometry` select native programs even with
an ordinary built-in material. Authored TSL attributes accept separate instance
streams and `InstancedInterleavedBuffer` layouts. The source's finite explicit
`geometry.instanceCount` wins over `Mesh.count`; it is captured after node updates
and before after-updates, including zero-count requests. Later queued renders
cannot replace that accepted count. Each shader-consumed stream must cover it;
the renderer does not infer a count or silently clamp it to an attribute capacity.

The native r186 backend uses instance step mode without applying the WebGL
`meshPerAttribute` divisor. This path preserves that behavior; the WebGL-authored
program path retains its divisor expansion. Existing geometry upload versions
and partial ranges preserve GPU-stale bytes outside requested updates. Native CPU
geometry snapshots preserve those accepted histories across queued calls.

## BatchedMesh

Pinned `BatchedMesh` objects select native programs with ordinary built-in or
authored node materials. Indexed and nonindexed batches execute their source
subdraw ranges, including heterogeneous geometry. The source callback owns
instance visibility, deleted/reused IDs, per-instance frustum culling, opaque
and transparent sorting, and `customSort`. The source batch shader reads matrix,
optional color and unsigned-integer indirect-index textures. Each WebGPU
`firstInstance` is the subdraw ordinal; the indirect texture resolves the actual
instance ID. An ordinary geometry draw range does not clip these source ranges.

One source material pass captures uniforms, textures and draw ranges once, then
shares that immutable resource history across its subdraws. Queued renders keep
independent matrix/color uploads, indirect IDs and ranges. Ordinary transparent
`DoubleSide` materials run node updates for each physical side, while the object
selection callback runs once before that split. Compilation follows the source
callback, so first colors and changed material sides from `customSort` reach
the shader and raster state. `compileAsync` runs source selection without running
ordinary node updates. Shadow casters use the same native shadow owner and the
pinned shadow callback sequence.

Texture allocation identities participate in program keys: introducing the
first color or resizing instance capacity rebuilds the source batching graph.
Replacing those allocations during node updates refuses the capture before GPU
publication. Empty batches and zero object counts retain source updates without
issuing GPU draws. Subdraw expansion, texture layouts and ranges remain bounded
by the existing admission and pending-capture budgets. Native direct commands
and render bundles both retain the draw ordinal. Custom BatchedMesh subclasses
and replaced object lifecycle hooks remain outside this profile; `customSort`
itself is supported. This supplies the pinned WebGPU subdraw loop, without a
claim of hardware multidraw acceleration or measured speedup.

## Native wireframes

Native mesh programs support `material.wireframe`, including ordinary Basic,
Toon, Lambert, Phong, Standard and Physical materials, authored vertex graphs,
instanced geometry, `InstancedMesh` and `BatchedMesh`. The retained source shader
runs over a native line-list pipeline and the pinned derived triangle-edge
index. Source range/group bounds are doubled before intersecting the captured
edge extent. Batched ranges already include their source multiplier and are
preserved directly. Polygon offset remains disabled for line primitives.

Edge histories follow the source index identity and version, or position
identity and version for nonindexed geometry. Rebuilding reads the entire CPU
array, independently of original index upload ranges, and leaves those ranges
untouched. Capture occurs after before-updates and before ordinary node updates.
Queued renders and multiple groups in one submission retain independent edge
generations; unchanged versions reuse their resident view. Generated instance
attributes compose with those views. `compileAsync` establishes initial edge
history, while source disposal invalidates it. Geometry layout or wireframe
topology changes rebuild the program; changes during node updates that invalidate
the accepted layout fail before publication. This profile uses source-width
WebGPU lines and makes no line-width emulation or performance claim.

Normal and Matcap materials have native source-library conversions so mixed
wireframe/shadow scenes retain their other material families. The exact pinned
builders supply normal mapping, authored `normalNode` overrides, Matcap texture
coordinates, texture color conversion and the source Matcap fallback. Adding or
removing a Matcap map rebuilds the graph; compatible replacements preserve each
captured texture binding. Fog checks use the converted source defaults without
mutating the original material.

## Authored scene fog

Assign a source TSL graph to `scene.fogNode` to compose fog with native material
output. The authored graph takes precedence over `scene.fog`; clearing it
restores ordinary `Fog` / `FogExp2` selection. Materials with `fog: false` leave
the graph inactive. Ordinary admitted mesh materials enter native rendering
automatically, without adding a material node slot.

Range, density and height fog, world/view-position arithmetic, and sampled
texture colors use the pinned source builder and the existing native resource
owners. Graph replacement and structural changes rebuild the program; accepted
queued draws retain their shader, uniforms and texture selections. Fog composes
with native lighting and shadows, while shadow-material passes leave fog
inactive. Unsupported nodes or resource layouts still fail admission.

Use the source render group for fog values that must change on otherwise
unchanged ordinary materials:

```js
const fogColor = TSL.uniform(new THREE.Color(0x9ab8c8)).setGroup(TSL.renderGroup);
const near = TSL.uniform(2).setGroup(TSL.renderGroup);
const far = TSL.uniform(12).setGroup(TSL.renderGroup);
const height = TSL.positionWorld.y.add(1.5).mul(0.35).clamp(0.15, 1);
scene.fogNode = TSL.fog(fogColor, TSL.rangeFogFactor(near, far).mul(height));

fogColor.value.set(0xb8a8ca);
far.value = 9;
```

The pinned source observer treats scene nodes differently from authored
material slots: an unchanged ordinary material receives a `SHARED` refresh.
Node callbacks still run, but its default object-group uniform bytes and
texture/sampler binding identities remain at their preceding full refresh.
Shared groups continue updating. An observed material, geometry or object
change triggers a full refresh; authored material slots retain the source
full-refresh policy. The facade preserves this behavior rather than forcing
extra uploads. If another full draw uploads the same sampled image, shared fog
bindings see those accepted pixels within that render, even when the uploader
sorts later. Later queued renders cannot replace those captured pixels.

Focused source-builder tests compare complete WGSL and callback frequencies.
Facade recording-device tests cover ordinary material selection, graph
replacement/removal, shared versus object uniforms, inactive fog, source shadow
ownership and texture versions across queued renders. Sampled-owner tests also
check shared image histories, sampler domains and disposal identity.

## Explicit remaining boundaries

This is a bounded native graphics entrance, not full TSL coverage. Writable or
atomic graphics storage, standalone `mat2`/`mat3` storage arrays, nested/boolean/atomic
struct members, PBO storage, managed uniform arrays, compute nodes inside
graphics graphs and multiple-render-target graphics graphs are refused.
Light-probe grids and other custom light models, transmitted shadows,
custom shadow filter/replacement nodes, projected light maps and custom
`lightsNode` graphs remain missing.
Depth/comparison, array, volume, compressed, storage, video and unfilterable
or integer texture graphs beyond the admitted 32-bit data/storage profiles still
require additional native profiles. Sampled
texture capture transactions reject reentrant texture capture from node updates
or upload callbacks.
The profile excludes shader context extensions and unregistered shadow lifecycles,
alpha hashing and alpha-to-coverage; skinned or morph meshes,
including per-instance morph textures; XR/multiview and native ArrayCamera bundles or BatchedMesh;
object render hooks outside the source lifecycle profile above and effectful
attribute upload callbacks.
Ordinary transparent `DoubleSide` graphs execute consecutive back-side and
front-side draws, matching the pinned renderer's source order. Each side owns a
retained program and captured uniforms; object updates run for each draw while
frame/render updates retain the source `NodeFrame` frequency. `forceSinglePass`
keeps its source meaning and selects one draw with culling disabled. Native
transmission uses the separate scene-color and scene-wide back-face protocol
described above.

There is a remaining parity difference when the pinned reference explicitly
precompiles a transparent `DoubleSide` normal-dependent graph. Its
`_createObjectPipeline` queue retains each material after the source restores
`DoubleSide`, so both queued side programs can compile with double-sided normals.
The candidate retains separate back/front shader variants. Direct rendering
without precompilation agrees on the side-specific normals. This is a documented
precompilation compatibility gap, not complete parity for every call sequence.

Unsupported graph slots, material subclasses, bindings and attribute layouts
produce explicit errors rather than silently selecting another renderer.

These boundaries remain missing compatibility work. They are neither release
exemptions nor evidence of complete source compatibility.

## Source basis and verification

The implementation follows Three.js r186 source commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`, particularly `WGSLNodeBuilder`,
`NodeMaterial`, `NodeFrame`, `NodeUniformsGroup`, common `Renderer` draw selection
and `RenderList` ordering. Native opaque sorting uses depth and object identity,
without WebGL's material-id sort key.

Native AO/light-map regressions compare complete vertex and fragment WGSL with
the pinned builder for Basic, Toon, Lambert, Phong, Standard and Physical, both
ordinary and node-material forms. They cover separate UV channels, source light
map normalization, AO composition, queued intensity values and compatible texture
replacement. These execute the real source builder and capture code in an
isolated JavaScript runtime; they do not establish browser pixel parity for the
new light-map combinations.

Material snapshot regressions exercise the actual facade, scene bridge and
command owner with the existing recording device. They verify queued fixed /
native transitions, source disposal, array group selection, double-sided source
state preservation, failure cleanup and ArrayCamera/fog behavior. Snapshot-helper
checks cover value copies, clipping/override state, accessors, capacity limits and
borrowed metadata. These are command and ownership checks in the available
JavaScript runtime, not browser pixel or performance measurements.

The ordinary-material browser scenarios `native_material_lobes`,
`native_material_lobe_updates` and `native_material_relief_maps` produced zero
mismatched pixels and zero mean RGB difference against the pin on Chrome 151
software Vulkan, using the unchanged channel tolerance 12 and pixel budget 2%.
They cover default Physical lobe properties, live updates, and Phong bump /
Standard displacement maps without authored nodes. The relief check reused its
completed bundled build after the shared test machine ran out of disk space.
These are correctness results, not hardware performance measurements.

`native_tsl_double_side_transparency` keeps the usual precompile step and checks
constant-color overlaps, varied opacity and negative scale.
`native_tsl_double_side_normals` explicitly renders without precompilation to
check side-specific normals. The precompilation difference above remains
reproducible with:

```sh
node tests/e2e/webgpu_renderer/parity.mjs --force-precompile native_tsl_double_side_normals
```

That call sequence differed on 13.39% of pixels in the pinned reference
comparison; it is a known parity gap, not an adjusted tolerance or excluded
candidate failure. Other scenarios retain precompilation by default.

The owner/capture test files cover actual
pinned shader generation and packing, callback contexts and frequencies,
independent byte history, call-time selection/ranges, admission and budgets,
pipeline failures, device loss and disposal:

```sh
node tools/ingest/three_material_snapshot.test.mjs
node tools/ingest/three_node_program.test.mjs
node tools/ingest/three_node_cube.test.mjs
node tools/ingest/three_node_attributes.test.mjs
node tools/ingest/three_node_program_gpu.test.mjs
node tools/ingest/three_node_scene_capture.test.mjs
```

Core, facade and packaged-application tests exercise the integrated route. Real
WebGPU parity scenes `native_tsl_uniforms`, `native_tsl_fragment` and
`native_tsl_object_targets` compare directly with the pinned renderer. All three
passed with zero mismatched pixels under the existing parity threshold; the
fragment case's mean channel difference was 0.01, and the other two were zero.
They cover shared uniform materials, position displacement, custom attributes,
translucent fragment graphs, nonidentity transforms, object-dependent updates
and distinct uniform values in queued offscreen renders.

The texture suites exercise source binding indices and stage visibility, sampler
and pixel-version history, compatible image replacement, target generation and
feedback checks, resource-only groups and bounded bind-group reuse. Browser
scenarios `native_tsl_textures`, `native_tsl_texture_updates` and
`native_tsl_target_texture` compare those generated programs with the pinned
renderer at the existing tolerance.

Facade regressions check source light update frequencies across objects and
program rebuilds, queued byte history, material arrays and overrides, and unused
versus unsupported active lights. They also verify shared RG16F DFG residency,
queued Physical feature changes and independent PBR uniform values.

The real WebGPU scenarios `native_tsl_lit_materials`, `native_tsl_lit_updates`
and `native_tsl_lit_textures` pass against the pinned renderer at the existing
12-channel/2% pixel tolerance. All three have zero pixels exceeding that
threshold; mean differences are 0, 0.01 and 0 respectively. They cover the three
non-PBR lighting models and five ordinary light types, light identity/layer/
visibility and live-value changes, authored texture-driven normals/emission,
and Phong specular/shininess. These are Chrome 151 software-Vulkan correctness
results, not hardware performance measurements.

`native_tsl_pbr_materials`, `native_tsl_pbr_updates` and
`native_tsl_pbr_textures` also pass in that environment with zero mismatched
pixels and zero mean difference. They exercise Standard roughness/metalness,
all admitted Physical lobes, feature activation and live value updates, and
authored textures including distinct base and clearcoat normal maps.

Storage tests cover all twelve scalar/vector types, source stage aliases,
post-update identities, queued compute/render versions and shared buffer
identity. They also cover unsized alias changes inside `Fn`, read-only admission,
publication-free graphics compilation and capture reentry. Facade regressions
verify queued full and partial uploads, unchanged CPU versus GPU state,
compile-only behavior and borrowed compute residency.

Browser scenes `native_tsl_storage`, `native_tsl_storage_compute` and
`native_tsl_storage_targets` matched the pinned reference exactly in RGB on
Chromium 151 with SwiftShader. They execute vertex/fragment reads, actual compute
kernels and separate queued render-target histories. This is correctness
evidence for that software-GPU environment, not a hardware performance result.

The `native_tsl_storage_fifo` scenario exercises compute-produced
positions, colors and signed/unsigned indices, a partial final workgroup, two
queued target renders and a later upload-version reset. A rerun of the current
storage integration on Chrome 151 software Vulkan had zero pixels outside the
existing 12-channel tolerance, mean difference 0.01859375 and maximum channel
difference 1 against the pinned renderer. The pixel budget remained 2%.

A direct-module browser comparison additionally queued compute-backed
`Mesh.count` values 3, 0 and 5 into independent targets, then changed the live
count before readback. All three images matched the pinned reference exactly in
RGB, including the empty target; CPU storage remained unchanged. Recording
regressions cover post-update capture, callbacks on zero draws, indexed and
nonindexed commands, bundle reuse/invalidation and source draw-call counts.
The `native_tsl_storage_structs` scenario uses the exact H8 particle schema,
including its 80-byte stride and padded mat3 columns. Its bounded comparison
matched the pinned reference exactly for three queued images and all 100
read-back float words, including a second compute update and unmarked CPU edits.
This does not establish compatibility for the complete H8 fluid application.

The storage-backed `InstancedMesh` extension passes the 60-test compiler suite
and four focused facade regressions. These check source matrix/normal/color
generation, cache invalidation, queued compute and host versions, zero counts,
attribute replacement, shared-material isolation and capacity refusal before
submission. The browser differential scenario `native_tsl_storage_instanced_mesh`
exercises compute-generated rotations and nonuniform scale on lit meshes,
authored position and color storage reads, and independent targets with changing
counts. Its Chrome 151 software-Vulkan comparison passed the unchanged 2% pixel
budget: 3 of 76,800 pixels (0.00390625%) exceeded the channel threshold of 12,
with mean difference 0.004609375 and maximum channel difference 154 at three
object-boundary pixels. Both implementations visibly retained panel counts
`[6, 3, 0, 5]`, including the empty target.

The instanced-mesh and FIFO runs imported the current renderer modules directly
and used the unchanged runner and scenarios for 30 frames. They validate native
GPU execution against pinned r186 in that software-GPU environment. They do not
include a fresh bundled-application build or a hardware performance measurement.

The ordinary CPU `InstancedMesh` regressions cover small matrix uniform arrays,
large four-column matrix streams, RGB colors, mixed CPU/storage inputs, inverse-
transpose normals, dynamic and partial uploads, queued counts and immutable
attribute snapshots. They also cover visible versus hidden compilation history,
backing-array refusal, graph replacement while submitted GPU work is pending,
and direct/bundle instance-capacity checks.

`native_tsl_cpu_instanced_mesh` compares four lit targets against pinned r186,
with final instance counts `[5, 3, 3, 2]`. Two targets use the small matrix-UBO
path; two allocate one more matrix than the actual uniform binding limit permits
and use generated interleaved columns. Both paths include nonuniform and mirrored
transforms, frame-8 color/matrix updates, distinct queued target values and later
CPU-array poisoning. Its Chrome 151 software-Vulkan direct-module comparison
matched all 76,800 RGB pixels exactly: zero mismatches, zero mean difference and
zero maximum difference, with 33 distinct colors in each image. The existing
30-frame gate and 12-channel/2% thresholds were unchanged. There were no page,
console, shader/pipeline or request errors. This is rendering correctness
evidence, not a fresh bundled build or hardware performance measurement.
The pixel capture includes the instance lifecycle fixes and the mapped-material
merge at `7610a9a`. The later concurrent SunLight/PMREM-refresh merge at
`11231f7d` passed the affected compiler, facade, scene-capture and environment
suites. Its additional browser rerun could not load the application because the
shared workspace exhausted disk/memory resources; that merged GPU result is not
claimed.

The environment and PMREM-background recording-device regressions cover initial
source filtering, shared PBR/sky atlases, queued control values, mixed draw order,
differently sized texture replacement, and rejection before callbacks or
submission when an authored PMREM node would retire queued resources.
`native_tsl_environment` and `native_tsl_background_blur` are registered browser
comparisons; their new pixel-parity results remain unverified. The first actual
environment run exposed the corrected color-only scissor-clear depth bug.
A subsequent candidate run crashed during compilation under exhausted workspace
memory/storage before producing its baseline image.



Authored-sky regressions cover the H7 gradient, original callback identities,
queued uniform and graph replacement, background-node precedence over Color or
unsupported textures, load preservation with `autoClear = false`, and private
resource ownership. The registered `native_tsl_background_nodes` browser scenario
renders two queued gradient skies into distinct targets and mutates their live
controls afterward. Its GPU pixel comparison has not yet been run.


The built-in source-material extension passes 75 compiler tests and 29 renderer
facade tests. The H7 regression uses the model's Physical map combination with
compute-authored position/normal storage and `Mesh.count`, checking source
identities, sRGB views, shared bindings, queued pixel/count histories and shared
compute residency. Other cases cover all six converted material classes,
material arrays, UV/normal mode changes, callback ordering and alias changes
between material maps and authored texture graphs. The registered
`native_tsl_source_materials` browser comparison covers mapped built-in
materials with authored deformation; its GPU pixel comparison is unmeasured.

Exact pinned SunLight addon tests cover transformed-parent direction semantics,
shared FRAME updates across objects, graph rebuilds and renderer owners,
queued radiance/position history, registration changes and lifecycle admission.
The direct-analytic implementation passed 82 compiler and 31 facade tests before
integration with concurrent uniform-buffer support. Shadow requests and
unsupported generated storage writes are refused before draw callbacks or GPU
acquisition. No new SunLight browser pixel-parity result is claimed.

Ordered PMREM refresh regressions cover one atlas across queued old/new source
pixel versions, original callback context/frequency, unsafe before-update and
mixed-foreground refusal, stale graph recompilation, and accepted child filters
surviving an outer callback failure. A private native background permits safe
refresh with both fixed and native scene draws. These are command/resource
ordering checks; refreshed-environment pixel parity is not established.

Public PMREM regressions exercise actual RoomEnvironment scene capture, optional
sigma blur and GGX filtering, six distinct submitted face matrices, temporary
background restoration, independent generation colors, resize/disposal lifetime,
and failed-wait cleanup. Native camera/background tests also cover immutable
Color clears, sharp/blurred transitions, original callback camera identity and
coherent pre-initialization selection. The registered
`native_tsl_scene_environment` browser scenario bakes an asymmetric room, disposes
the helper immediately, and lights mapped-source-compatible Standard/Physical
surfaces while also displaying the generated sky. Its recording-device path is
tested; GPU pixel parity for the new scene bake is not yet measured.

CPU cube regressions compare generated WGSL with the pinned builder, check cube
binding dimensions, compatible replacements, alias/mapping changes and invalid
source-view refusals. Recorded draws retain separate six-face and authored-mip
upload versions. Public fromCubemap tests check retained source method identity,
13 conversion/filter renders, HalfFloat source bytes surviving later edits,
immediate generator disposal and a sampled output atlas. The registered
`native_tsl_cube_textures` scenario combines six asymmetric direction panels
with Standard/Physical environment consumers; its recording-device path is
tested. Browser pixel parity for this cube extension is unmeasured.

## Native render-list controls

`renderer.opaque` and `renderer.transparent` select the corresponding native
and fixed draw lists using the pinned renderer's strict `=== true` checks.
Opaque selection is accepted before opaque node updates; transparent selection
is accepted afterward, so an opaque update can enable or disable the later
transparent phase. Disabled native phases do not compile their programs,
capture uniforms, run node updates or copy the framebuffer. Queued calls retain
their accepted selections. Filtering a parent mesh does not hide its children,
and fixed ArrayCamera views apply the same controls in each viewport.

Shader and texture backgrounds belong to the opaque phase. Color clears remain
independent. The final canvas output quad has its own opaque selection after
transparent updates, matching the source's output-pass ordering. Native tone
mapping or output color conversion uses the linear scene target from the first
render, preserving working-linear RGB above one until output conversion. The
canvas is acquired only after successful scene preparation; initialization waits
for resource validation without fencing unrelated GPU work.

`compileAsync` prepares only the enabled native phases and does not run draw
updates. The unchanged pinned `PassNode` can set `opaque = false` or
`transparent = false` to render a filtered scene and then restore its caller's
renderer controls. These controls are covered by recording-device tests of the
actual facade and pinned PassNode, including callback-driven selection changes,
queued captures, backgrounds and output conversion. They do not establish new
browser pixel parity. Native ArrayCamera execution has its separate checks above.

## Native scene passes and previous-frame color

The pinned `PassNode` and `RenderPipeline` can retain current and previous color
outputs through `getTextureNode(name)` and `getPreviousTextureNode(name)`:

```js
const scenePass = TSL.pass(scene, camera);
const current = scenePass.getTextureNode();
const previous = scenePass.getPreviousTextureNode();
const pipeline = new THREE.RenderPipeline(renderer, TSL.mix(previous, current, 0.8));
pipeline.render();
```

This blends two consecutive scene images. For recursive accumulation, a material
inside the scene pass can also sample its previous color output. The source
`NodeFrame` schedules the scene pass once per logical frame, including when its
own material references that pass. Sampling its active color or depth attachment
remains invalid feedback.

Named MRT outputs can each retain a history, including mixed RGBA8 and RGBA16F
attachments. Current and previous colors have distinct persistent GPU images;
four-sample targets also have distinct multisample attachments. Depth and colors
without a history remain shared. History swaps update bindings without rerunning
authored `Fn` setup. Late history activation preserves the already rendered
current image.

Automatic pass resizing creates new images while previously accepted draws,
copies and readbacks retain their original generations. The new previous image
starts with WebGPU's initialized contents. Selected MRT uploads, copies,
readbacks and external GPU color producers retain their captured history image
and validation dependencies. Allocations, including detached generations, count
against the renderer's existing target budgets.

The admitted history surface is 2D mip-zero color with one or four samples.
Create the source current/previous texture-node pair; calling only
`getPreviousTexture()` does not establish the source's complete swap lifecycle.
This PassNode swap surface remains color-only; depth, cube/mip and XR histories
are outside it. The separate SSGI and TRAA addon paths are described below;
TRAA copies its own depth history rather than using `getPreviousTexture()`.
The focused owner and renderer
tests cover swaps, feedback, queued resize, late activation, stable graph setup
and transfer ordering. Browser scenarios are `native_pipeline_history`,
`native_pipeline_history_resize` and `native_pipeline_history_accumulation`;
the accumulation scenario uses four samples.

## Native Bloom and Gaussian blur

The pinned `bloom` and `gaussianBlur` addons execute their original TSL graphs
and source `updateBefore` passes through the native renderer. Bloom retains its
luminance filter, five horizontal/vertical blur pairs and composite; Gaussian
blur retains its two separable passes, direction, kernel and alpha options.
They compose with scene passes, named MRT color outputs, ordinary node materials
and a `RenderPipeline` targeting either the canvas or a native default output
target. For example:

```js
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { gaussianBlur } from 'three/addons/tsl/display/GaussianBlurNode.js';

const scenePass = TSL.pass(scene, camera);
const color = scenePass.getTextureNode();
const softened = gaussianBlur(color);
const glow = bloom(softened, 0.8, 0.3, 1);
const pipeline = new THREE.RenderPipeline(renderer, color.add(glow));
pipeline.render();
```

The shared producer adapter discovers the source node's finite private target
set and accepts shared material contexts constructed by the retained builder.
It does not select shaders by effect name. Source `NodeFrame` frequency controls
remain authoritative, so repeated renders in one logical frame reuse the effect
output. Each accepted pass captures its uniforms and GPU texture dependencies;
intermediate colors stay on the GPU. Source-driven size changes preserve earlier
queued generations while retaining the existing target budget. Explicit effect
disposal invalidates queued dependents, and failed updates restore the effect's
temporary renderer controls.

Small Bloom outputs can reduce their last private level to a logical zero size.
Like pinned `Textures.getSize()`, native 2D targets retain one backing texel in a
zero dimension while preserving the source target dimensions and zero viewport.
That backing allocation is charged to the budget, and queued resizes through
zero retain the same generation-lifetime rules.

Gaussian blur preserves the source input's byte, half-float or Float32 precision.
Float32 inputs require the device's `float32-filterable` feature; both blur
intermediates then remain RGBA32F. For a Float32 scene pass, construct the renderer
with `outputBufferType: THREE.FloatType`, which is the type selected by the pinned
`PassNode.setup()`. The validated floating-point binding layout can survive an
effect's first output-type selection; unfilterable Float32, integer and depth
layouts remain distinct. Missing Float32 filtering is rejected explicitly.

This surface requires ordinary frame-update effects with directly owned 2D
targets and stable attachment formats during queued resizes. It does not admit
arbitrary application-created material contexts or view-offset histories, and
does not establish full temporal addon support. The actual pinned addon tests
cover pass order, composition, frame reuse, shared-material uniform snapshots,
queued controls and resize, GPU-only intermediates, disposal and failure cleanup.

Chromium 151/SwiftShader differentials matched all 4,096 RGBA16F channel words
exactly across four asymmetric 16x16 HDR cases: scene-pass Gaussian, scene-pass
Bloom, original color plus Bloom(Gaussian), and direct-texture Gaussian with
premultiplied alpha, half resolution, custom direction and sigma. Both renderers
used the same identity-swizzle adapter; there were no GPU validation errors and
no numeric tolerance. The comparison explicitly unpacked the pinned source's
padded readback rows; this establishes image-content parity, not equality of the
returned readback array layout. Native readbacks remain tightly packed.

A separate Float32 differential matched all 3,072 channel values exactly for
scene-pass Gaussian, original color plus Bloom(Gaussian), and premultiplied
Gaussian with half resolution and a custom direction. It verified RGBA32F
horizontal/vertical intermediates and Float32 readbacks, with zero validation
errors on the same Chromium/SwiftShader environment. Focused recording tests
also cover queued Float32 resize/parameter snapshots and capability refusal.

## Native screen-space global illumination

The pinned `SSGINode` executes its original hemisphere sampling, occlusion
bitfields, depth reconstruction and indirect-light shader through the native
renderer. Its two outputs retain their source formats: AO in `r8unorm` and GI
in packed `rg11b10ufloat`. The latter requires the enabled device feature
`rg11b10ufloat-renderable`; there is no substitute format or reduced-quality
shader. Normals may come from a scene MRT output or be reconstructed from depth.

```js
import { ssgi } from 'three/addons/tsl/display/SSGINode.js';

const scenePass = TSL.pass(scene, camera);
scenePass.setMRT(TSL.mrt({ output: TSL.output, normal: TSL.normalView }));
const color = scenePass.getTextureNode();
const indirect = ssgi(color, scenePass.getTextureNode('depth'),
  scenePass.getTextureNode('normal'), camera);
const result = TSL.vec4(color.rgb.mul(indirect.r).add(indirect.getGINode().rgb), 1);
const pipeline = new THREE.RenderPipeline(renderer, result);
```

The source effect owns its sample count, radius, thickness, intensity and
temporal sampling controls. FRAME updates run once per logical frame. Queued
passes retain their parameter values and both AO/GI generations across an
automatic resize; intermediate images stay on the GPU. Source temporal sampling
changes the sample pattern. It does not itself accumulate history: the addon
expects a temporal resolve when enabled, or a separate denoiser when disabled.

Chromium 151/SwiftShader matched the pinned source exactly for eight frames of
an HDR room scene, using both MRT normals and depth-derived normals, moving
geometry, changed radius/intensity and a resize. The comparison covered 6,848
AO bytes, 6,848 packed GI words, 27,392 final RGBA16F channel words and 27,392
scene-color channel words. AO and GI were spatially varied and final RGB exceeded
one. There were no GPU, page or console errors and no numerical tolerance.
Both runs used the same identity-swizzle adapter and logical frame clock;
source readback padding was excluded. This is rendering correctness evidence,
not a performance result or a full temporal-effects closure claim.

## Native temporal reprojection antialiasing

The pinned `TRAANode` now executes its original reprojection, velocity handling,
depth rejection, variance clipping and subpixel correction through the native
renderer. Its source pipeline callbacks apply camera jitter and clear the view
offset after rendering. The original builder records the effect's ownership of
that pipeline state; subsequent captures require the same registered texture
producer. Rebuilding the source pipeline establishes a fresh context without
duplicating its callbacks.

```js
import { traa } from 'three/addons/tsl/display/TRAANode.js';

const scenePass = TSL.pass(scene, camera);
scenePass.setMRT(TSL.mrt({ output: TSL.output, velocity: TSL.velocity }));
const result = traa(scenePass.getTextureNode(), scenePass.getTextureNode('depth'),
  scenePass.getTextureNode('velocity'), camera);
const pipeline = new THREE.RenderPipeline(renderer, result);
```

Use this source effect with MSAA disabled. It currently follows the native
ordinary-camera, non-reversed-depth scene-pass surface; XR remains outside it.
The source first samples a standalone, zero-initialized `DepthTexture`. Its
later depth history belongs to a target with `depthBuffer: false`: storage is
allocated for copies and sampling, while color rendering has no depth attachment
and cannot clear or overwrite that history. Half-float color history and depth
copies stay on the GPU.

History sizing follows the source beauty target at the time of the effect
update. For the checked scene-pass graph this preserves the initial 1×1 resolve
and the one-frame history-size delay after resizing. Depth copies occur only
when history and drawing-buffer extents agree. Queued draws, copies and reads
retain their accepted generations through submission and completion, including
when several logical frames and resizes are queued before preparation finishes.
Explicit target disposal still invalidates pending uses.

A paired Chromium 151/SwiftShader differential matched all 27,392 RGBA16F
channel words over eight frames of warmup, moving geometry, resizing and a
camera cut. Jitter index, camera projection, restored view state, history size
and previous-depth identity also matched on every frame. A second eight-frame
run composed SSGI AO/GI with the beauty output, passed that expression through
the source `convertToTexture()`/`RTTNode`, and applied TRAA. All 27,392 final
channel words matched again, with an observable SSGI contribution. Both
comparisons had zero GPU, page or console errors and used the same
identity-swizzle adapter and frame clock as the SSGI comparison above. These
are general WebGPU correctness checks, not hardware performance measurements
or closure of every temporal effect.

## Native voxel global illumination and material lighting contexts

The pinned `VXGINode` and `VXGIVolume` now execute their original voxelization,
light injection, opacity/radiance mip generation, cached bounce and cone-tracing
graphs through the native renderer. Storage buffers and 3D textures remain on
the shared compute/graphics owner. Source effect updates enqueue their compute
prerequisites before the consuming draw, with no readback or replacement shader.

Renderer, material and scene-pass contexts admit bounded source `ContextNode`
chains containing `getAO` and `getGI` hooks. The retained builder invokes these
hooks inside material lighting. In particular, `builtinGIContext` preserves its
AO clamp, GI normalization, material-AO composition and transparent-material
bypass. Ordinary Basic and lit mesh materials select the native source builder
when they consume such a context; authored dummy node slots are unnecessary.
Material context values retain precedence over renderer values, and scene passes
retain the source versioned merged-context cache. Hook replacement or context
version changes require a fresh graph; uniform values remain live per draw.
Other custom context fields and custom traversal implementations remain refused.

```js
import { vxgi } from 'three/addons/lighting/vxgi/VXGINode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';

const prePass = TSL.pass(scene, camera);
prePass.transparent = false;
prePass.setMRT(TSL.mrt({
  output: TSL.packNormalToRGB(TSL.normalView), velocity: TSL.velocity,
}));
prePass.getTexture('output').type = THREE.UnsignedByteType;
const normals = TSL.sample(uv => TSL.unpackRGBToNormal(prePass.getTextureNode().sample(uv)));
const indirect = vxgi(prePass.getTextureNode('depth'), normals, scene, camera, 16);
const beauty = TSL.pass(scene, camera);
beauty.contextNode = TSL.builtinGIContext(
  indirect.getAONode().sample(TSL.screenUV).r,
  indirect.getGINode().sample(TSL.screenUV).rgb,
);
const result = traa(beauty, prePass.getTextureNode('depth'), prePass.getTextureNode('velocity'), camera);
const pipeline = new THREE.RenderPipeline(renderer, result);
```

Use MSAA-disabled ordinary cameras for this checked composition. The example's
resolution is deliberately small; it does not establish the performance of the
full Sponza scene or a large volume. Preserve the source invalidation controls:
set `indirect.volume.lightingNeedsUpdate = true` after changing injected lights
and `indirect.volume.needsUpdate = true` when geometry requires revoxelization.
Without temporal resolve, set `indirect.useTemporalFiltering = false` for a fixed
sample pattern. Scene-pass context changes follow the source cache: advance
`beauty.needsUpdate` when replacing its context graph.

Compute admission requires all still-active parent passes to have no accepted
draws or texture reads. A mixed fixed/native pass cannot prove this prefix and
remains refused. Earlier accepted work survives a later parent failure, while
temporary effect/pass context and renderer controls restore normally. This does
not allow arbitrary compute callbacks inside a partly captured draw.

`tests/e2e/webgpu_renderer/vxgi.mjs` compares the source and candidate using exact
HDR readback, independently checking source repeatability first. Four frames of
spatially separated triangle surfaces cover changed lights and geometry. On
Chromium 151/SwiftShader, VXGI and VXGI-plus-TRAA each matched all 16,384 final
RGBA16F channel words, 16,384 GI words and 4,096 AO bytes, with zero GPU, page or
console errors. GI was nonzero and exceeded one; TRAA jitter indices matched.
This is correctness evidence, not a hardware-performance claim. Intersecting
room geometry also executes, but the pinned source's competing voxel triangle-ID
writes make its GI output non-bitwise-repeatable; it is not an exact GI oracle.

## Native clipping groups

Source `ClippingGroup` planes apply to native built-in and node-material meshes,
including nested union/intersection groups, group enable/layer selection and
`clipShadows`. Planes are projected into camera space with the pinned source
math, and the pinned `ClippingNode` supplies the shader equations. Each queued
draw retains its captured planes; moving a plane updates uniforms without
rebuilding the graph. Background draws remain outside scene clipping.

Source `alphaToCoverage` selects multisample coverage for four-sample targets.
`native_clipping_groups` and `native_clipping_msaa` exercise these paths. Legacy
renderer/material `clippingPlanes` are a separate API and are not enabled by
this native ClippingGroup implementation.
