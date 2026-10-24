# Native TSL graphics programs

The native `WebGPURenderer` facade can draw authored TSL graphs on ordinary
meshes and CPU- or storage-backed `InstancedMesh` objects. It retains the exact r186
`WGSLNodeBuilder`, node classes, uniform packing
and `NodeFrame`; FrankenThreeD owns the WebGPU pipelines, geometry residency,
uniform buffers, command encoding and submission. The path does not construct an
upstream renderer, translate through ESSL reflection, or claim measured
acceleration or a Rust implementation of node evaluation.

## Implemented graph surface

Built-in `MeshBasicMaterial`, `MeshToonMaterial`, `MeshLambertMaterial`,
`MeshPhongMaterial`, `MeshStandardMaterial` and `MeshPhysicalMaterial` can also
carry authored node slots. Their exact pinned `StandardNodeLibrary` conversions
supply the native graph while the builder, callbacks and disposal retain the
original source material identity. Materials without authored slots keep their
existing fixed rendering path. Custom material subclasses and overridden
conversion methods remain outside this entrance.

Source material maps now lower through the pinned material accessors: base
color, alpha, emissive, normal, bump, displacement, specular, roughness,
metalness, gradient and the admitted Physical lobe maps. Their UV channel,
transforms, color space and normal-map mode remain source behavior. Shared
metalness/roughness textures retain one sampled binding; changing a map's
sharing relationship with another map or authored texture rebuilds the binding
layout. Compatible image replacement retains the compiled program and each
queued draw's source pixel version. AO, light maps and material environment maps
still require additional lighting adapters.

Use `MeshBasicNodeMaterial`, or an unlit base `NodeMaterial`, with `colorNode`,
`positionNode`, `vertexNode`, `fragmentNode`, `opacityNode`, `alphaTestNode` and
`outputNode`. Ordinary source node arithmetic, functions and custom scalar/vector
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
light can use the new registration. Shadows, projected maps and custom light
color nodes remain explicitly refused. In particular, this enables H7's direct
SunLight illumination without claiming its cascaded-shadow path is complete.

Unassigned slots retain their source material color, emissive, specular and
shininess behavior. Both material arrays and scene overrides supply the effective
material to the builder's context.

Authored `MeshStandardNodeMaterial` and `MeshPhysicalNodeMaterial` use the same
direct-light path with the source GGX model and `metalnessNode`/`roughnessNode`.
Physical also accepts clearcoat strength, roughness and normal; sheen color and
roughness; iridescence strength, IOR and thickness; anisotropy; retroreflectivity;
and IOR, specular color and specular intensity nodes. Unassigned slots read their
ordinary material properties. Enabling or disabling a Physical feature selects
the corresponding source shader branch; changing its live value keeps the
program. The source's shared 16×16 RG16F DFG lookup texture goes through the same
versioned texture owner as authored textures, preserving its aliasing across
materials and clearcoat samples.

Native Standard and Physical graphs also use ready 2D equirectangular or source
CubeUV `scene.environment` textures. The pinned native `PMREMGenerator` supplies
conversion/filter graphs and atlas layout; FrankenThreeD captures and submits
those ordinary node draws through its own renderer. Generated PMREM targets and
the source DFG texture are borrowed through the same versioned texture owner.
The retained renderer/backend does not submit commands. Live environment
intensity and rotation remain source uniforms; replacing the environment selects
its corresponding atlas and preserves already queued draw bindings. A wholly
native scene bypasses the separate fixed-material environment filter.

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
`fromCubemap()` accepts the CPU-data cube profile below, runs the source cube
conversion and GGX filtering, and returns an ordinary CubeUV atlas for native
PBR environments. The WebGL/core placeholder route retains its existing behavior.

A same-size equirectangular input can set `needsPMREMUpdate = true` between
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
upload-version behavior. Direct cube environment inputs, custom environment nodes and
the compatibility adapter's placeholder PMREM textures remain outside this
native profile.

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
six CPU DataTexture faces. The compiler retains the pinned direction, reflection/
refraction mapping and explicit-level semantics and declares cube views in the
source binding order. Compatible cube replacements reuse the program; changes
to the texture alias partition or cube mapping select a new source graph.
The existing texture owner captures all six faces and authored mip levels for
each accepted upload version, preserving earlier draws after later CPU edits.
Float filtering retains the existing device-feature checks.

Decoded-image cubes, render-target cubes and unsampled cube loads remain outside
this profile. Cube textures cannot occupy ordinary material map slots or be
assigned directly as native scene environments; use the public generator's
CubeUV result for environment lighting.

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
WebGPU depth sorting. All reachable node bindings, including hidden objects, are
admitted before any node update callback. The source camera receives its WebGPU
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
are used, so a node callback cannot insert compute into a partially captured
render. Leases release their captured bytes on success and failure; the compute
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
generated streams at preparation without running application callbacks; hidden
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
program. Ordinary geometry arrays and attribute versions retain the existing
bridge's live upload contract; generated streams use the captured histories
above. This is not a deep snapshot of an entire mutable scene.
GPU geometry uses the existing owner, while native uniform groups receive
independent aligned slices. Later arena growth leaves earlier groups and bytes
valid. Disposed or lost owners cannot publish pending pipelines, including
programs with no uniform groups. Pending captured bytes share the facade's
bounded budget and are released when the request finishes.

Scene overrides respect each original material's `allowOverride`. The first
profile admits an override only when the source-inherited position node, alpha,
displacement and transparency state already matches the override. Other cases
fail explicitly before callbacks, without temporarily changing a shared source
material. Original material visibility and render-list partitioning still apply.

## Explicit remaining boundaries

This is a bounded native graphics entrance, not full TSL coverage. Writable or
atomic graphics storage, standalone `mat2`/`mat3` storage arrays, nested/boolean/atomic
struct members, PBO storage, managed uniform arrays, compute nodes inside
graphics graphs and multiple-render-target graphics graphs are refused. Authored
PBR transmission/dispersion, direct cube/custom environment lighting, area/probe and other custom light models,
shadows, projected light maps and custom `lightsNode` graphs remain missing.
Depth/comparison, decoded-image cube, array, volume, compressed, storage, video and unfilterable
or integer texture graphs still require additional native profiles. Sampled
texture capture transactions reject reentrant texture capture from node updates
or upload callbacks.
The profile excludes shader context extensions, fog/clipping/shadows, AO/light/material-environment
maps, wireframe, alpha hashing and alpha-to-coverage; `InstancedBufferGeometry`, skinned, batched or morph meshes,
including per-instance morph textures; ArrayCamera, reversed depth and XR views;
custom object/material render hooks and effectful attribute upload callbacks. A
transparent DoubleSide graph currently requires `forceSinglePass: true`.
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

The owner/capture test files cover actual
pinned shader generation and packing, callback contexts and frequencies,
independent byte history, call-time selection/ranges, admission and budgets,
pipeline failures, device loss and disposal:

```sh
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
Depth history, cube/mip histories, XR/view-offset histories and complete
TRAA/SSGI addon effects are outside this surface. The focused owner and renderer
tests cover swaps, feedback, queued resize, late activation, stable graph setup
and transfer ordering. Browser scenarios are `native_pipeline_history`,
`native_pipeline_history_resize` and `native_pipeline_history_accumulation`;
the accumulation scenario uses four samples.

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
