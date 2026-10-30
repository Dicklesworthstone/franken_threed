# Native TSL graphics programs

The native `WebGPURenderer` facade can draw authored TSL graphs on ordinary
meshes, authored `Sprite`, `Points`, `Line` and `LineSegments` objects,
CPU- or storage-backed `InstancedMesh` objects,
`InstancedBufferGeometry` and `BatchedMesh`. It retains the exact r186
`WGSLNodeBuilder`, node classes, uniform packing
and `NodeFrame`; FrankenThreeD owns the WebGPU pipelines, geometry residency,
uniform buffers, command encoding and submission. The path does not construct an
upstream renderer, translate through ESSL reflection, or claim measured
acceleration or a Rust implementation of node evaluation.

## Implemented graph surface

### Render-triggered geometry compute

`NodeMaterial.geometryNode` now admits the pinned `ComputeNode` lifecycle,
including a lazy `TSL.Fn(({ renderer, geometry }) => ...)()` constructor that
creates named storage attributes and returns a compute kernel. The official
`webgpu_compute_geometry.html` jelly pattern can initialize storage in
`kernel.onInit()`, update it through `geometryNode`, and read the same allocation
through `positionNode = TSL.attribute('storagePosition')`. Background nodes and
ordinary companion meshes share the native scene route without an application
rewrite. Hidden or layer-excluded constructors remain inactive.

The retained source builder constructs the graphics output and compute body.
Compute-only setup nodes are checked by the compute owner, so legal kernel
assignments do not become rejected graphics writes. Nodes generated into the
vertex or fragment shader still receive full graphics validation; returning or
sharing a write into an actual graphics flow does not bypass that boundary.
Custom ComputeNode subclasses or replaced construction/update methods remain
outside this profile.

The original `NodeFrame` drives before/current/after callbacks. A shared kernel
with the default `FRAME` frequency runs once across multiple objects and repeated
renders in that frame. `onInit()` dispatches precede the first update kernel;
subsequent draws use that same GPU buffer. Named storage attributes, ordinary
attributes promoted by `TSL.storage()`, source vec3 padding and generated program
cache keys retain one source constructor/kernel identity. Queued renders keep
their accepted compute uniforms and source upload versions, and unmarked CPU
arrays do not overwrite GPU-authored results. Compute before/after callbacks see
the kernel and null scene/object/camera/material; current compute callbacks also
see a null `compute`, as in `NodeManager.updateForCompute`. Nested work restores
the enclosing graphics callback context.

Public `compileAsync()` follows the geometry program's source lifecycle: advance
the shared frame, preserve its previous render ID, run before updates, capture
geometry before current updates, prepare that object's pipeline, then run its
after updates before advancing to the next geometry object. The precompilation
flag is restored before each await, so source shadow nodes keep their skip
behavior. Accepted initialization/update kernels drain before public compilation
resolves, including when a later source callback rejects the parent. Internal
shader construction and ordinary pipeline preparation do not independently
execute geometry kernels.

There is a remaining public-compilation construction boundary: accepted source
Fn graphs are built eagerly before the per-object update/pipeline sequence. A
later object's constructor that depends on side effects from an earlier object's
compile callbacks is not yet source-compatible. This profile supports the
ordinary lazy constructor and shared-kernel patterns; it does not establish
complete general `compileAsync()` parity or change the existing compilation
lifecycle of unrelated material graphs.

Geometry kernels use the existing render prerequisite FIFO. Disjoint earlier
native draws are admitted; a background or companion mesh does not itself block
the update. Live compute storage, indirect and texture bindings are checked
before upload capture against earlier parent readers and pending attachments.
Overlapping resources, untracked fixed draws and arbitrary object-hook compute
remain explicit refusals. This overlap check is conservative: pinned WebGPU
submits compute through an independent encoder before the unfinished parent
render encoder. This change does not introduce interleaved draw/compute passes
or claim support for every source-valid aliasing combination. Accepted child
kernels survive a later parent capture failure, with no parent draw submitted.

Sixteen `three_renderer_geometry_compute.test.mjs` cases cover exact pinned
vertex/fragment and compute WGSL, constructor/cache stability, shared-frame
updates, queued uniforms and GPU-buffer identity, compilation callback ordering
and IDs, pre-init compilation through a tone-mapped output target, compute
context restoration, late failures and buffer/texture overlap refusals. The
record device verifies commands and bytes; it does not execute WGSL. Adjacent
storage-attribute and storage-texture suites pass all 32 tests after reconciliation
with concurrent matrix-storage and HDR-output changes (48 focused cases total). The native-shadow
suite passes 25 of 29; its four target-generation-count and malformed hidden-skin
fixture failures also reproduce with unchanged main product modules. They remain
pre-existing failures, rather than a claim that every adjacent suite is green.

Both `native_geometry_compute.mjs` source/candidate browser cases pass exactly:
the official-style Fn constructor and a shared kernel used by two meshes match
688,128 RGBA bytes at fourteen checkpoints in Chrome 151.0.7922.34 / SwiftShader,
with zero GPU or browser errors and the owned general-WebGPU route. GPU position
and per-vertex counter readbacks verify source vec3 padding, initialization once,
stable storage identity, one kernel update per logical frame, untouched CPU arrays
and visible live changes. These additions retain the source JavaScript builder
and callback runtime; hardware performance and Rust node evaluation remain
unmeasured.

### Native occlusion queries

Objects with `occlusionTest = true` now issue GPU occlusion queries through the
owned native renderer. `renderer.isOccluded(object)` preserves the source API:
`null` outside an active render, `undefined` before results arrive, then a
boolean during object or node callbacks. Finishing the next render with the
same source context starts mapping the previous query frame. `waitForGPU()`
waits for mappings already started; it does not publish the latest frame early.

Contexts follow the pinned attachment signature, MRT identity and nesting depth.
Different scenes, cameras and equally configured targets share history as they
do in r186. Material groups and BatchedMesh subdraws share their object's query.
Internal command bundles execute inside the appropriate query spans, keeping
separate objects' results even when their drawing state is compatible. Queued
requests retain the accepted flags and object identities. Disabling the last
query retains its object's callback path and completed result history. Compilation
runs its source callbacks without submitting or advancing query history.

The source ArrayCamera branch does not encode object queries, and source
BundleGroup draws retain its warning and query omission. Requested empty frames
still resolve unused query slots so their empty result keeps a real GPU mapping
boundary. Disposal, device loss and failed mapping retire owned buffers/query
sets and prevent delayed result publication. Nested object-hook GPU writes keep
their existing separate-pass requirement; admitted nested node passes restore
the enclosing context. Copies inside an active query remain an unverified
source combination.

The three `native_occlusion.mjs` browser cases compare actual visible/hidden
geometry, grouped and batched objects, moving occluders, shared target history
and ArrayCamera's unused-query behavior. Observed callback results are also
rendered into comparison pixels. Chrome 151.0.7922.34 / SwiftShader matches all
1,032,192 RGBA bytes at twenty-one checkpoints with zero GPU errors. Hardware
performance remains unmeasured.

### Native framebuffer depth snapshots

`renderer.copyFramebufferToTexture(depthTexture)` and the retained
`viewportDepthTexture()` node now copy actual native framebuffer depth. Public
copies stay in the render FIFO; node copies execute after their preceding draws
and before their consuming draw, including ArrayCamera view spans. Resumed passes
load the existing color, depth and stencil attachments. Canvas, HDR scene and
ordinary render-target copies borrow the original depth allocation, including
private target depth that has no public `target.depthTexture`.

Copies and samples share the existing detached-depth owner and its captured
texture generations. The first allocation follows the source context's sample
count; same-version uses retain that allocation. One- and four-sample snapshots
preserve every sample, with source-generated multisampled depth loads. Public
depth type/format remains authored. Matching `depth24plus`, `depth32float` and
`depth24plus-stencil8` copies are admitted; mismatched native formats retain the
source diagnostic and omit the copy. Raw copies cover complete equal-sized
subresources, use all aspects and require equal sample counts.

This profile covers base-level ordinary 2D attachments. Cube, layered, volume
and nonzero-mip depth snapshots remain outside it. Disposal rejects pending
copies, while submitted storage survives its completion fence. Lazy canvas
depth borrows retain that storage without acquiring a presentation texture.
Deferred captures retain their accepted mip/face selectors independently of
later public selection changes.

The depth owner, complete framebuffer-copy facade and canvas suites pass 66
tests; adjacent canvas-session, target and color-framebuffer suites pass 56.
Four `native_depth_snapshots.mjs` source/candidate browser cases cover viewport
depth, MSAA, ArrayCamera and public float-depth copies. All 1,376,256 RGBA bytes
at twenty-eight checkpoints match in Chromium 151.0.7922.34 / SwiftShader with
zero GPU errors. These comparisons use the native F3D backend. Hardware
performance remains unmeasured.

### Materials and node graphs

Built-in `MeshBasicMaterial`, `MeshToonMaterial`, `MeshLambertMaterial`,
`MeshPhongMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`,
`MeshNormalMaterial`, `MeshMatcapMaterial`, `SpriteMaterial`, `PointsMaterial`,
`LineBasicMaterial` and `LineDashedMaterial` can also
carry authored node slots. Their exact pinned `StandardNodeLibrary` conversions
supply the native graph while the builder, callbacks and disposal retain the
original source material identity. Ordinary materials also enter this path when
they use bump, displacement or baked light maps, Basic/Lambert/Phong material
`envMap`, or Physical sheen, iridescence, anisotropy, retroreflectivity,
transmission or dispersion and their maps. They require no authored node slot or
application rewrite. Other materials keep their existing fixed rendering path.
When one material group on a mesh needs native shading, its other groups use the
same source library conversion and geometry owner. All selected groups must pass
admission before any node update runs. Custom material subclasses and overridden
conversion methods remain outside this entrance.

Native `PointsNodeMaterial`, `LineBasicNodeMaterial` and
`LineDashedNodeMaterial` retain their source graphs and object-selected WebGPU
topology. Points can use compute-produced storage directly, and point materials
on sprites or meshes retain source pixel sizing, DPR and viewport semantics.
Dashed lines use the source distance attribute and four authored dash slots.
See [THREE_NATIVE_PRIMITIVES.md](./THREE_NATIVE_PRIMITIVES.md) for the material
matrix, source limits and validation commands.

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
fixed path does not shift another light's shadow index.

The pinned `LightProbeGrid` addon also retains its registered analytic light
node, packed 3D spherical-harmonic atlas and source FRAME updates. Visible grids
automatically select native shading for ordinary and Node lit materials. Bounds,
resolution, intensity and positive falloff changes stay live; atlas replacement,
first bake and the zero/positive falloff branch rebuild the source graph. Inactive
and unlit grids do not require registration or hide their ordinary children.

The addon's unmodified `bake(renderer, scene, options)` runs through the owned
cube, float render-target, 3D slice and texture-copy paths. CubeCamera captures,
GPU SH projection, padded atlas packing, ranged baking and indirect-pass
snapshots retain their original order without CPU texture readback. Renderer
targets, viewport, visibility, XR and scene update controls are restored by the
source bake. The default 2×2×2 bake issues 84 render operations, so the bounded
renderer FIFO now admits 1024 pending operations by default. Larger synchronous
batches can use `createWebGPURendererClass(THREE, {maxPendingRenders: ...})` or the
source ranged-bake API; independent captured-byte budgets remain enforced.
`three_node_light_probe_grid.test.mjs` checks source shader equivalence, FRAME
uniform capture, structural changes, full/ranged GPU command ownership, inactive
selection and explicit queue-limit refusal. This profile does not implement
the separate `LightProbeGridWebGL` addon or arbitrary custom lighting models.

The `native_light_probe_grid.mjs` browser comparison exercises complete and
ranged bakes, indirect-pass atlas snapshots, live grid controls and a later
rebake with changed incident light. Both scenarios exactly matched pinned r186
at frames 1, 7, 8, 10, 12, 13 and 30: 344,064 RGBA bytes each, with zero GPU
validation errors on Chrome 151 software Vulkan. A zero-intensity checkpoint
verifies that baked irradiance visibly contributes to the final scene. This is
correctness evidence for the native path, not a hardware performance claim.

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

Source `SpotLight.map` projections now select native lighting automatically for
ordinary lit materials as well as node materials. The pinned `SpotLightNode`
owns projection coordinates, frustum selection, attenuation and render-time
shadow-matrix updates. The existing sampled owner captures ready 2D map inputs,
including compressed textures and their color-space views. Queued draws retain
their matrices, accepted map versions and replacement-map bindings; versionless
CPU edits do not change an already accepted upload. Projected maps also compose
with the existing PCF shadow owner.

Authored light colors use the distinct source contracts: spotlight `colorNode`
is a callable projection with a source cache key, such as
`TSL.Fn(([uv]) => TSL.texture(projectedTexture, uv.xy).rgb)`, while other analytic
lights consume a source color node graph directly. Spotlight color projections
take precedence over `map`, as upstream does. Texture and graph replacements
participate in program selection, while uniforms keep source update frequencies
and per-draw captured values. Analytic-node reuse remains controlled by the
pinned source light cache.

Visibility, camera layers and renderer lighting enablement govern this route;
inactive projected lights do not force native draws or inspect invalid maps.
Fifteen `three_renderer_light_projection.test.mjs` cases cover ordinary
Lambert/Standard routing, source WGSL, compressed maps, queued matrices and
resources, PCF integration, spotlight color projections, directional/point/
ambient color graphs and admission failures. These are recording-device checks,
not browser pixel or performance measurements.

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
light can use the new registration. Projected maps require the source
`SpotLightNode`; registered analytic lights can use source color graphs.
The registered SunLight also retains its source
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

### Native Basic/Lambert/Phong environments

`MeshBasicMaterial`, `MeshLambertMaterial`, `MeshPhongMaterial` and their node
material forms now render material-local environment reflections and refractions
through the native source graph. Ready cube maps and equirectangular images/data
are supported, including the `ReflectionColor` environment assigned by the
retained FBX loader. Ordinary materials select the native path directly; an
authored color slot is unnecessary. Once an environment material has entered
that path, removing and restoring its environment retains the same source
lighting and deformation owner, including for animated skinned meshes.

The pinned `BasicEnvironmentNode`, `CubeMapNode` and `BasicLightingModel` retain
Multiply, Mix and Add composition, reflectivity, specular-map strength,
`refractionRatio`, environment rotation and source normal handling. Lambert and
Phong retain their direct/baked lighting around that same environment finish.
These materials use reflectivity as their environment-strength control;
`envMapIntensity` remains a PBR control and does not acquire a new meaning.

A texture or material-reference `envNode` takes precedence over `envMap`.
Basic/Lambert/Phong do not fall back to `scene.environment` or
`scene.environmentNode` when their material environment is absent. Source
material rotation accessors still determine the rotation of an authored cube
node. Disabling source lighting or supplying a complete `fragmentNode` leaves
unused material environment maps inactive.

Equirectangular conversion executes the retained source method through six
F3D-owned cube render-target faces. The converted image is sampled through its
native cube view, without a CPU replica or an additional PMREM filter. Repeated
uses retain the source conversion cache; another panorama identity selects its
own conversion. Reflection/refraction mapping changes preserve the source cube
orientation and cached conversion. Compatible uniform changes reuse the graph.
At a source rebuild boundary, combine mode, input mapping, source identity and
graph changes select the corresponding variant. Callback-time structural
changes reject a stale draw before its incompatible resources can be accepted.

The pinned renderer has two relevant cache limitations: its material cache
reduces numeric properties other than `side` to zero/nonzero, which aliases
`MixOperation` and `AddOperation`, and a texture node's graph cache key excludes
the texture's mapping. `material.needsUpdate = true` alone does not invalidate
those source shaders. A retained Mix/Add transition therefore requires an
explicit material-disposal or environment-graph replacement boundary; changing
a texture `envNode`'s mapping requires replacing that node or disposing its
material. Native preparation reports `THREE_NODE_PROGRAM_ENVIRONMENT_CACHE`
for these unsupported live transitions instead of silently building a shader
that the pinned renderer would not build. The browser fixtures use these actual
source invalidation boundaries. General emulation of source builder-cache
collisions across otherwise-identical distinct materials is outside this slice.

Queued draws retain their environment textures, sampler/upload histories and
uniform bytes through the existing owners. A later material replacement or live
control edit cannot rewrite an earlier accepted draw. The active profile requires
ready cube/equirectangular inputs and the pinned texture/material-reference
`envNode` shapes. Arbitrary procedural environment nodes are not source-valid in
this BasicEnvironmentNode path: the pinned CubeMapNode does not produce a cube
texture for them. Standard/Physical procedural environment graphs keep their
existing separate source semantics.

Twenty-five focused `three_renderer_native_basic_environment.test.mjs` cases pass,
including complete source WGSL equality for all six material forms in reflection
and refraction modes, source precedence, combine variants, live controls,
callback guards, pinned cache-key/refusal/recovery behavior, queued native draws,
environment removal/restoration and actual six-face conversion commands.
Seventeen directly affected PBR material/environment regression cases also pass.
These record-device checks verify shader/resource behavior. The separate
`native_basic_environment.mjs` browser comparisons execute WGSL for mixed
Basic/Lambert/Phong cube materials, panorama conversion with an authored node,
and an animated skinned Phong mesh. All three cases match the pinned source
exactly at frames 1, 7, 8, 10, 12, 13 and 30: 21 checkpoints and 1,032,192 RGBA
bytes. They verify visible output, live changes, the owned `general-webgpu`
route and zero GPU/browser errors on Chromium 151.0.7922.34 with SwiftShader.
The skinning case removes and restores its environment during animation; the
other cases use the explicit source invalidation boundaries described above.
The source JavaScript graph builder remains retained; no Rust implementation or
hardware speedup is claimed.

### Native subsurface scattering materials

The pinned `MeshSSSNodeMaterial` now selects native graphics through its source
material graph. Its `SSSLightingModel` adds the original subsurface term to
direct diffuse lighting and retains the inherited physical shading. All six
source controls work: `thicknessColorNode`, `thicknessDistortionNode`,
`thicknessAmbientNode`, `thicknessAttenuationNode`, `thicknessPowerNode` and
`thicknessScaleNode`. Thickness can come from sampled textures or authored TSL
graphs; uniforms keep their source FRAME, RENDER and OBJECT update frequencies.

A null `thicknessColorNode` retains the source physical fallback. Inactive
lighting paths do not construct or update unused scattering graphs. The pinned
SSS lighting constructor does not enable the newer retroreflection lobe, even
when the inherited material property is set; native rendering preserves that
source behavior. Clearcoat, sheen, iridescence, anisotropy, transmission and
dispersion keep their source physical paths. Material environments retain
precedence over scene environments, and native shadows use the shared depth
target owner. Transmission uses the existing ordered opaque framebuffer copies.

Changing any thickness graph requires a new source program. A stale descriptor
rejects before node callbacks; mutations during construction or before/current
node updates cannot publish a stale draw. Uniform value changes retain the
program and each accepted render keeps independent bytes. Inherited physical
feature switches and transmission graphs retain their existing cache guards.

`three_renderer_native_sss_material.test.mjs` contains fifteen focused tests for
default routing, exact source vertex/fragment WGSL, textured directional/point/
spot scattering, physical lobes, queued state, graph invalidation, inactive
branches, environment precedence, transmission copies, spot shadows and generated
storage-write rejection. The tests execute the real renderer with a recording
device; they do not establish raster pixels or hardware performance. The retained
source builder and JavaScript callbacks still own graph construction and updates;
FrankenThreeD owns native resources and GPU submission.

The `native_sss_material.mjs` browser lane compares textured thickness controls,
inherited transmission/dispersion/clearcoat, and spot shadows with environment
overrides. All three match pinned source RGBA exactly at seven 128x96 checkpoints
each in Chrome 151.0.7922.34 / SwiftShader, with visible nonconstant output, live
changes and zero GPU/browser errors. These software-GPU comparisons establish
the tested rendering behavior; hardware performance remains unmeasured.

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
readback or retained renderer submission. Depth framebuffer copies use the native
snapshot profile above; the WebGL-authored copy route remains unimplemented. Native ArrayCamera copies preserve
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
padding. Standalone Float32 `mat2`, `mat3` and `mat4` storage arrays use 4-, 12-
and 16-word strides. A `mat3` requires three authored padded columns; packed
nine-word matrices are not silently repacked. Compute-produced matrices share
the same GPU allocation with vertex and fragment consumers, including queued
partial host uploads. Flat
non-atomic `struct()` records also work with scalar/vector 32-bit
members and `mat2`, `mat3` or `mat4` fields. The pinned `StructTypeNode` defines
their padded stride; mixed integer/float backing bytes and matrix-column padding
remain unchanged. The H8 fluid particle record (`position: vec3`,
`velocity: vec3`, `C: mat3`) occupies 80 bytes and can be written by compute and
read directly in both vertex and fragment graphs.

Source storage nodes marked with `.setPBO(true)` use that same native allocation.
The flag enables a separate WebGL fallback in the pinned cloth, birds and
skinning examples; `StorageArrayElementNode` only observes it when the builder
has no storage-buffer support. Native WGSL therefore retains ordinary storage
bindings, and changing the hint does not rebuild its graphics program or restore
old CPU values. Source vec3 padding, generated `.toAttribute()` vertex inputs,
fragment reads and queued partial uploads keep their existing owners. No PBO
texture emulation or CPU readback is introduced into the native path, and the
exact WebGL component is unchanged. Five new recording regressions in
`three_renderer_storage_attributes.test.mjs` compare source shaders, queued
bytes and allocation identity; all 26 storage-attribute cases pass.
`native_pbo_storage.mjs` also runs GPU spring/vertex cloth kernels and generated
instanced vertex/fragment consumers against the pinned renderer. Both cases
match all seven RGBA checkpoints exactly (688,128 compared bytes total), with
real storage readbacks, live hint toggles, native ownership and zero GPU/browser
errors on Chromium 151.0.7922.34/SwiftShader. These are bounded compute/graphics
workloads, not full official-example or hardware-performance certification.

A non-atomic compute-writable source node can be read by graphics; the pinned builder
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

Scalar Int32 and Uint32 storage also supports source `.toAtomic()` graphs with
TSL atomic operations in the fragment stage. The pinned builder supplies the
atomic WGSL, and native bind groups retain its writable storage access. Fragment
producers, compute dispatches and asynchronous readback share one GPU allocation
in queue order. A later requested partial CPU upload changes only its selected
components, preserving untouched GPU-produced values. A render pass cannot mix
writable and read-only/vertex uses of the same attribute, and a draw cannot bind
overlapping writable aliases. Float, vector, struct and vertex-stage atomics
remain refused before native uploads. Ten focused fragment-atomic tests cover
source WGSL, binding limits, aliases, captured uploads and FIFO ownership using
explicit recorded producer effects; they do not execute WGSL or prove GPU pixel
parity.

Ordinary meshes and native sprites support source `count` with TSL `instanceIndex`, including
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
the light uniform bytes captured for their logical call. Native forward graphs
include every selected light within the existing generated-node, shader-byte,
uniform and device-binding budgets. Unsupported light kinds or shadow/projector
state fail before any application node update.

Scenes with more than eight visible, layer-selected lights select native
rendering for the complete material list, including unlit companions and passes
with lighting disabled. This avoids the fixed shader's eight-entry light block
and preserves one shadow owner and the source light indices. Smaller scenes
retain their existing route. A queued scene that later removes lights keeps its
accepted light set and uniform values. Eleven `three_many_lights.test.mjs` cases
check nine and 32 mixed lights across Lambert, Phong, Toon, Standard and Physical
materials against pinned source shader bodies and exact uniform bytes, plus
queued 24-to-eight transitions, layers, inherited visibility, disabled lighting,
late-index shadows and real capacity failures. This uses the source forward
algorithm; it adds no clustered-light algorithm or performance claim.

Authored `material.lightsNode = TSL.lights([...])` compositions now select their
own source lights and lighting contributors. Lists can contain ordinary lights,
analytic light nodes, irradiance, ambient-occlusion and environment nodes. They
retain source precedence over scene membership, visibility/layers, the renderer
lighting switch and the material's ordinary light-collection flag. An explicit
off-scene shadow light uses the same native shadow prerequisite as scene lights.

The private source light list receives bounded graph validation and structural
cache keys. List edits and replacement graphs produce the appropriate program;
uniform-only changes retain the program and capture new per-draw values. Nine
`three_renderer_lights_graph.test.mjs` cases compare source WGSL and queued
lighting packets, explicit selection and disable flags, shadows and rejection
of generated unsupported writes. The root composition must retain the pinned
`LightsNode` methods; custom lighting-manager subclasses remain separate work.
These command/resource checks do not establish pixel parity.

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
allocation. Ordinary graphics-only `compileAsync()` does not update storage
bindings or run their node callbacks; the geometry-compute lifecycle described
above explicitly runs its source prerequisites.

Storage capture has its own transaction, nested with texture capture when both
are used. Registered source texture producers may run compute prerequisites
before the first draw or texture read in every active parent capture. Completed
child passes already precede their parent in the FIFO. Render-triggered geometry
compute may also follow earlier native draws when its live resources are disjoint
from their captured readers and the pending attachments. Unproven overlap and
ordinary callback compute still require separate public submissions.
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
generated streams at preparation. Geometry-compute programs execute their source
node lifecycle at that boundary as described above; other programs keep their
existing callback-free preparation. Hidden objects do not acquire that history
until their first visible use. Replacing a
program waits for submitted work before retiring its old GPU geometry owner.

Instance culling still uses the source mesh's CPU `boundingSphere` before node
updates. GPU writes do not refresh CPU matrices, bounds or raycasting data.
Applications with compute-driven transforms must supply conservative bounds or
set `frustumCulled = false`; the renderer does not infer GPU bounds or change that
source setting.

Captured graph descriptors, raster state, source list properties, draw ranges and
uniform bytes survive later source edits. Material/geometry disposal invalidates
the compiler cache, and subsequent graph or structural changes select a new
program. Native programs also capture ordinary CPU geometry through the existing
immutable geometry owner at the source upload boundary. Shared geometry uploads
once per logical render; unmarked edits and untouched partial ranges stay stale,
including when a render callback is added after initial compilation. This is not
a deep snapshot of an entire mutable scene.
GPU geometry uses the existing owner, while native uniform groups receive
independent aligned slices. Later arena growth leaves earlier groups and bytes
valid. Disposed or lost owners cannot publish pending pipelines, including
programs with no uniform groups. Pending captured bytes share the facade's
bounded budget and are released when the request finishes.

Geometry upload capture now follows the compiled shader's selected inputs, as
pinned `RenderObject.getAttributes()` and `Geometries.updateAttributes()` do.
Unused separate attributes acquire no upload history and consume no pending
update ranges. Enabling vertex colors after an earlier render or
`compileAsync()` therefore takes the color attribute's current bytes on its
first actual use. An active interleaved view still captures its complete backing
owner, so unversioned edits to another view of that same owner remain stale.
Source geometry disposal invalidates current shared attributes even when that
geometry's own shader never selected them; accepted older packets retain their
bytes until released.

Shaders driven by `vertexIndex` can omit position entirely. Their captured
geometry retains the source vertex count separately, preserving nonindexed draw
counts and callback/indirect range checks without uploading unused positions.
The complete geometry-history owner suite passes thirteen cases, and eighteen
focused facade/callback cases pass, including queued color first-use after
render and compilation and an object-hook vertex-index draw. These checks use
the recording device and do not add a browser-pixel or performance result.

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

Object lifecycle hooks now select this path for ordinary built-in materials as
well as authored node materials. All active companions use the same native
capture owner, preserving one source-ordered render list without requiring an
application-provided comparator. Grouped materials, live overrides, transparent
double-sided draws, visibility, camera layers and opaque/transparent phase
controls keep the source lifecycle above. Queued calls retain each draw's
material values and requested geometry uploads before preparation can await.
Removing the hooks retains the native owner and its upload history, including
when another scene shares the geometry or individual attributes.

This path requires a source Scene. Install ordinary object hooks before their
geometry attributes enter a fixed-path render or `compileAsync()`: adding them
after a fixed owner has accepted those attributes, even through another scene,
still fails before invoking any hook. Transferring the fixed owner's existing
upload history, including queued work and unversioned CPU edits, remains
unimplemented. Hooks added to an already native scene retain
its existing upload history. Callback replacement of selected geometry, custom
BatchedMesh hooks, effectful attribute upload hooks and recursive GPU writes also
require additional capture boundaries and fail explicitly. Callback exceptions
retain earlier source mutations and do not replay updates or run later
after-callbacks. These are remaining compatibility gaps, not source-invalid uses.

Fifteen focused cases in `three_node_scene_capture.test.mjs` and
`three_renderer_nodes.test.mjs` verify the ordinary/mixed lifecycle, source group
and material identities, compile-only effects, exceptions, queued draw bytes,
hook removal, cross-scene attribute history and the existing-owner boundary. The
recording device verifies commands and uploaded bytes; this change has no new
browser-pixel or performance measurement.

## Custom object dispatch

`setRenderObjectFunction()` now selects the native capture path for application
dispatchers, including scenes made entirely from ordinary built-in materials.
The function runs synchronously with the renderer as `this` and the original
source object, scene, camera, geometry, material and group. Its remaining
arguments include the scene's real source `Lighting` owner's `LightsNode`, the
selected clipping context and the source pass ID. The dispatcher is selected
once per render; calling the setter inside it affects subsequent renders.

A dispatcher may skip a selected item, forward it through
`renderer.renderObject()`, or forward several passes with alternate materials
and pass IDs. Each forwarded call runs the pinned object lifecycle and captures
its own material values, light selection and requested geometry uploads before
preparation can await. Changes to the supplied LightsNode between material
passes are observed in those passes. Successful renders restore its previous
light list through the source Lighting owner; exceptions retain completed source
effects while releasing capture resources and the nesting slot. Preparation
does not replay dispatchers or object callbacks. Warm bundles execute their
recorded passes without invoking a newly installed dispatcher until the bundle
is invalidated. `compileAsync()` uses the source default object lifecycle and
does not invoke the application dispatcher.

The pinned Basic lighting model still constructs selected analytic light nodes.
A selected RectAreaLight therefore retains the source LTC initialization
prerequisite even in an unlit pass; unused LTC tables are not uploaded.

Install the dispatcher before the selected attributes acquire a fixed-path
owner; the existing explicit history-transfer boundary above still applies.
Forwarded calls must retain the selected object, scene, camera, geometry and
group, along with the supplied light-node and clipping-context identities.
Replacing these selections or issuing nested GPU writes requires another capture
boundary and is refused. Removing the dispatcher retains the native upload
history. Registered shadow selectors remain distinct: only the exact source
function paired with its source override material receives shadow-pass access.

Seven focused facade cases cover skipped and repeated draws, alternate
materials, queued snapshots, dispatcher replacement, compilation, live lights,
retained bundles, exceptions and context boundaries. All thirty-seven selected
dispatch, lifecycle, lighting, shadow and bundle regressions pass with the
recording device. These checks establish command and upload behavior, without a
new browser-pixel or performance result.

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

Compute-produced positions through `instancedArray(...).toAttribute()` and
read-only storage bindings share the compute-owned GPU allocation without CPU
readback or stale uploads. A recording-device case renders 200,000 instances in
one draw. Converted SpriteMaterial views retain the source version after property
copying, so size-attenuation setters do not spuriously invalidate queued programs.

The separate `tests/e2e/webgpu_renderer/sprites.mjs` comparison establishes exact
RGBA8 pixels against the pinned renderer for perspective, orthographic, live-node
updates and compute-driven instancing at seven checkpoints per case. Chromium
156.0.8078.73 / SwiftShader reported no GPU validation errors. This checks the
direct native renderer route; it does not establish automatic construction routing,
ArrayCamera or shadow pixel parity, or physical-device performance. See
[THREE_SPRITES.md](THREE_SPRITES.md) for source culling semantics and route boundaries.

## Native wide lines

The pinned `Line2NodeMaterial` and official WebGPU `Line2`/`LineSegments2`
addons now select the native graphics path without requiring an authored node
slot. They retain the source expansion into indexed, instanced triangles,
including screen-space and world-unit widths, round caps, perspective near-plane
trimming, dash distances, endpoint colors and alpha-to-coverage. Their shared
interleaved endpoint/color/distance attributes use the existing geometry owner.

The source `onBeforeRender` callback still updates the line's viewport resolution
for public raycasting. Width, DPR, viewport, color and authored dash uniforms are
captured at the logical render boundary, so subsequent edits do not alter queued
draws. World-unit, dash, endpoint-color and transparency changes rebuild the
program; stale structural captures fail before authored node updates. Transparent
lines preserve the source opaque-framebuffer sampling and mip generation through
the existing ordered copy owner.

Visible wide-line scenes also capture compatible ordinary material companions
through the native path, preserving the addon's callback and opaque/transparent
draw order in mixed scenes. The shader cache retains the source coverage branch
for logical sample counts zero versus one, even though both use a single-sample
GPU attachment. Native binding readiness is checked before borrowing framebuffer,
storage or target resources, so a new shader can request preparation without
turning that recoverable boundary into a terminal framebuffer failure. Failures
after submitted GPU work keep their existing resource failure semantics.

`three_renderer_native_wide_lines.test.mjs` covers both official addons, the four
screen/world and solid/dashed shader combinations, queued value/count mutations,
source callbacks, MSAA transitions, stale-graph rejection and transparent copies.
The shader comparison normalizes generated uniform names and independent uniform
block declaration order while retaining each binding, member layout and shader
operation. These recording-device tests establish command/data behavior;
physical-device performance and the wider material-family closure remain open.

`tests/e2e/webgpu_renderer/native_wide_lines.mjs` compares screen-space connected
lines, world-unit dashed segments and transparent authored dashes against pinned
r186 at seven 128x96 RGBA checkpoints each. All three match exactly in Chromium
151.0.7922.34 / SwiftShader, including dynamic width/DPR/camera/count/visibility
changes and the source addon callback. The lane asserts the owned
`general-webgpu` submission route and reports zero GPU/browser errors. Shader
construction and addon behavior remain retained JavaScript; these software-GPU
results establish correctness for the tested scenes, not hardware acceleration.

`three_renderer_wide_lines.test.mjs` adds thirteen regressions for mixed ordinary
meshes, interleaved partial uploads, queued viewport/DPR controls, sample-count
changes and framebuffer-dependent shader transitions. The `native_wide_lines`
and `native_wide_lines_msaa` cases in `native_primitives.mjs` match all 688,128 RGBA
bytes at fourteen source/candidate checkpoints on Chromium 151.0.7922.34 /
SwiftShader, including four-sample coverage, transparency, near-camera trimming
and dynamic mode changes, with zero GPU validation errors.

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
variance filtering to the SunLight atlas.

The pinned `CSMShadowNode` addon now executes its cascade depth and VSM moment
passes through the same native owner. Actual source builder ancestry authorizes
its private Object3D casters under the selected analytic light's shadow graph;
type flags do not grant that access. Captured caster/shadow identities and the
pinned camera, material and caster-selection helpers are checked again before
updates. Source ShadowBaseNode compositions retain their graph construction and
ordinary node admission; arbitrary backend or replacement caster hooks remain
outside the admitted profile.

CSM supports uniform, logarithmic, practical and custom splits, faded/standard
graphs, perspective/orthographic cameras, and Basic/PCF/VSM filtering. Ordinary
materials select native rendering for the whole CSM scene. The retained root
fits cascades before its generated ShadowNodes run; public compilation reuses
the existing source precompilation lifecycle without rendering shadow passes.
Per-cascade update/freeze flags, callbacks and queued resize generations remain
with the existing shadow/resource owners.

Ten CSM recording cases in `three_renderer_native_shadows.test.mjs` cover these
contracts and reject invented ancestry, changed identities and replaced camera
helpers. `native_csm_shadows.mjs` compares three actual source/candidate browser
cases: ordinary PCF, faded VSM and orthographic Basic. All 21 checkpoints and
1,032,192 RGBA bytes match exactly in Chromium 151/SwiftShader, with zero GPU or
browser errors. Intensity toggles assert visible shadow changes. Browser map
dimensions stay fixed because the pinned source itself submits a destroyed
depth image after a live CSM mapSize change in this setup; queued resize
ownership has separate recording coverage. Construction and camera fitting
remain source JavaScript; this is native GPU execution without a speedup claim.

`renderer.shadowMap.transmitted = true` now routes ordinary materials through
the same native shadow owner. Directional and spot receivers sample the source
RGBA shadow map alongside its depth map; point receivers sample the source cube
color/depth pair after all six caster faces. Source `castShadowNode` color and
alpha, transparent and transmission caster handling, and `NoBlending` remain in
the retained shadow material construction. This does not infer colored optical
transmission beyond the source's own shadow graph.

Queued transmitted shadows retain their captured tint, alpha and matching color
and depth generations across map resizes. Point-shadow compilation registers
cube target metadata before sampling without allocating GPU images. Changes to
transmitted mode or `shadow.mapType` follow the source cached-graph lifecycle;
the source light must reset before those constructed resources change. Nine
focused `three_renderer_transmitted_shadows.test.mjs` cases check these paths,
compile-only registration, material restoration and rejection of generated
storage writes. These are resource/command checks, not GPU pixel comparisons.

Authored `LightShadow.filterNode` functions now construct their source filter
graphs on the native path. Both TSL `Fn` filters and ordinary callable filters
use the pinned shadow inputs, samplers, pass ordering and update lifecycle.
Filter-generated resources go through the same graph admission as material
nodes, so unsupported storage writes still fail before GPU submission. Queued
receiver draws retain their own filter-uniform values.

The pinned `ShadowNode` caches its constructed output. Assigning a new filter
alone does not replace that output; the source shadow lifecycle must reset it
(for example, `light.dispose()` followed by material invalidation). Native
program keys include filter identity without overriding that source behavior.
Five focused `three_renderer_shadow_filters.test.mjs` cases exercise directional
and spot filters, Basic/PCF comparison sampler defaults, queued uniforms,
replacement, callable admission and rejection of generated writes.

Seven focused recording-device cases cover directional and spot moments,
queued resize/uniform history, explicit disposal, ordinary-material routing,
side overrides and the Point/Sun fallbacks. The CSM VSM browser comparison above
does not establish separate spot-light VSM pixel coverage.

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
`BatchedMesh` and authored `BundleGroup` ordering are supported as described
below. XR, multiview and nested camera arrays still fail explicitly.
WebGL-authored ArrayCamera rendering keeps its existing boundary.

`three_renderer_native_arrays.test.mjs` uses the pinned `renderObject` and
`WebGPUBackend.draw` as an ordering oracle and inspects actual submitted uniform
bytes and commands. These checks establish the tested execution behavior, not
browser rasterization parity or a performance result.

### Authored ArrayCamera bundles

Source `BundleGroup` records now support ordinary native `ArrayCamera` targets
whose selected children have no viewport when the group is recorded. The pinned
r186 backend records these groups into a single `GPURenderBundleEncoder`, which
cannot set viewports. The native route preserves this source limitation and
refuses a cold selected child with an explicit viewport. Viewport-less commands
inherit the active pass viewport when the bundle executes, after ordinary draws;
ordinary companions may therefore establish that viewport after recording.

Warm groups keep the recorded child indices and layer selection, including an
empty selection. Later object visibility, child layers and child viewport edits
do not change those commands until `group.needsUpdate` records the group again.
Shared camera uniforms still advance, while local material/object values follow
the source static/dynamic refresh policy. Accepted queued frames own independent
camera-index packets and matrix bytes. Object callbacks run at recording, and
node updates retain their source refresh frequencies.

Bundled `BatchedMesh` draws use the same route for single and array cameras.
The cache stores one logical capture per source material-side invocation,
retaining its batch ranges and selected camera indices. Each frame expands those
commands in material-side, child-view, then subdraw order. Warm refresh therefore
does not multiply node updates or resource leases by the number of views or
instances, and batch sorting/selection runs only when the source rerecords.

Four focused cases in `three_node_scene_bundles.test.mjs` cover camera refresh,
static/empty selections, queued viewport inheritance, callbacks and retained
batch ordinals. Two `native_bundle_views.mjs` browser scenes additionally compare
mixed static/dynamic groups, changed layers/visibility, explicit invalidation,
ordinary viewport inheritance and translucent DoubleSide batches. All 688,128
RGBA bytes at fourteen checkpoints match pinned r186 under Chromium
151.0.7922.34 SwiftShader, with zero GPU validation errors. These establish the
tested ordinary-target behavior; layered/XR targets and hardware performance
remain outside this evidence.

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

Native `ArrayCamera` batches preserve the source view-first order: each selected
child executes every batch subdraw before the next child, including overlapping
transparent viewports and both material sides. The source callback still selects
and sorts against the parent camera once, and node updates run once per material
side. Child layers, camera-index uniforms and viewports are captured after those
updates. Queued renders keep independent camera and batch-texture histories;
canvas viewports use DPR while render-target viewports use physical pixels.
Empty child selections preserve callbacks and framebuffer copy boundaries without
issuing geometry commands. Expanded view/subdraw counts use the existing budget.
Authored ArrayCamera bundles use the viewport-less recording profile above.

The 19-case batch suite and 50-case scene-capture suite pass. Two browser scenes
cover indexed and nonindexed batches, mixed ordinary companions, overlapping
transparent views, custom sorting, camera/layer/visibility changes, and MSAA4.
All 688,128 RGBA bytes at fourteen checkpoints match pinned r186 under Chromium
151 SwiftShader, with zero GPU validation errors. This is correctness evidence,
not a hardware performance measurement.

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

## Alpha-hashed transparency

Ordinary admitted mesh materials with `alphaHash: true` now enter the native
source shader route automatically. The pinned `NodeMaterial` supplies its
derivative-based threshold and fragment discard, using the deformed local
position. Opacity, base-color/alpha maps, authored opacity nodes, alpha testing,
clipping and alpha-to-coverage compose through that same source graph. Hashing
does not force blending, change the sample count, or add transparent passes.

Changing `alphaHash` rebuilds the native shader; queued enabled/disabled/enabled
renders keep their accepted programs and material values. Source shadow
materials retain their own alpha behavior. Explicit fragment graphs and the
source Normal material's diffuse override keep their source bypass semantics.

Focused tests compare complete vertex/fragment WGSL with r186 and exercise live
opacity, map upload versions, queued route changes, construction/update mutation
checks and shadow ownership through the actual native facade. These checks do
not establish browser pixel parity or hardware performance.

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

## Native compressed texture inputs

Source `CompressedTexture`, `CompressedArrayTexture` and
`CompressedCubeTexture` inputs now enter native node graphics through the
pinned `TextureNode` and `CubeTextureNode` builders. Ordinary cube textures
containing compressed faces use the same path. Array samples retain their
source layer coordinates. Ordinary material maps routed through native node
materials share these bindings.

The existing sampled-texture owner validates and captures authored mip blocks,
color-space views, device compression features and sampler state. It supplies
the binding layout rather than inferring compressed HDR filterability from the
source upload type. This preserves BC, ETC/EAC, ASTC and RGBA8 decoder fallback
profiles without another uploader or block conversion. Source-generated WGSL
and the pinned builder's filtering behavior remain authoritative.

Queued draws keep their acknowledged upload versions and immutable block bytes;
compressed-array partial updates preserve unselected layers at every mip.
Texture family, dimension and shader-relevant type changes invalidate program
reuse. Capability, malformed mip and node/view dimension failures remain
pre-submission admission errors. See `THREE_COMPRESSED_TEXTURES.md` for the
shared owner's format and upload boundaries. The 16 focused cases in
`three_renderer_native_compressed.test.mjs` record generated WGSL, layouts,
uploads and draw bindings; they do not execute compressed-texture shaders or
establish browser pixel parity.

## Native media texture inputs

Source `VideoTexture` and `VideoFrameTexture` inputs now use the existing shared
native sampled-texture owner. Ordinary material map slots select this path
automatically; authored TSL texture nodes use the same bindings. The pinned
WebGPU builder emits ordinary `texture_2d<f32>` sampling for both classes, and
uploads use `copyExternalImageToTexture`. Public source classes, image identity,
texture versions and upload callbacks remain intact.

Each accepted upload version captures an immutable frame at the render call.
Caller-owned `VideoFrame` inputs are cloned; HTML video uses `VideoFrame`
capture when available and call-time `createImageBitmap(video)` otherwise.
Bitmap readiness joins the existing render FIFO. Later video playback, public
frame closure and sampler changes cannot replace an accepted frame. A
temporarily starved stream keeps its previously captured pixels, including
while an earlier bitmap capture is still preparing. Private captures are
charged to the existing snapshot budget and closed on release, rollback or
disposal; caller frames and playback state remain application-owned.

Native media retains the pinned WebGPU upload-version behavior. It does not
call the legacy `VideoTexture.update()` fallback, add frame notifications or
control playback. Version-zero textures receive the source's unuploaded 1x1
default allocation. Compilation inspects layouts without capturing frames or
uploading media. The profile retains single-level RGBA8 sampling and the
existing source format, sampler and origin checks.

Cloned media textures can share a source while keeping independent accepted
frames and upload callbacks. Updating one alias cannot advance another alias
whose texture version is unchanged. Shared node refreshes consult only the
same texture's accepted uploads.

Eighteen `three_renderer_native_media.test.mjs` cases exercise source WGSL, ordinary map and
alpha-map routing, queued versions, frame closure, private capture disposal,
alias isolation, bitmap readiness, default storage, origin failures and upload metadata. The
recording host/GPU checks do not establish browser decoder or color-pixel
parity.

## Native 3D texture inputs

Ordinary source `Data3DTexture` inputs now enter native graphics through
`TSL.texture3D` and `TSL.texture3DLoad`. The existing shared texture uploader owns
the allocation and upload history; shader generation stays with the pinned
`Texture3DNode` and `WGSLNodeBuilder`. Supported single-mip profiles include
normalized bytes, half floats and floats. Float32 linear filtering requires
the native `float32-filterable` capability.

Sampler-free loads also preserve signed and unsigned 32-bit integer volume
texels. They use source `IntType`/`UnsignedIntType` and integer texture formats,
with matching `texture_3d<i32>`/`texture_3d<u32>` declarations. Source 8/16-bit
integer graphics typing remains outside this profile.

Queued consumers retain acknowledged upload versions and immutable captured
bytes. CPU edits without `needsUpdate` do not upload early. Typed-array offsets,
padded rows, compatible image replacements and captured linear wrapR samplers
use the same ownership rules as existing 2D inputs. Node/texture dimension
mismatches fail before allocation. Volume dimensions and wrapping participate
in the program signature.

Target-owned `RenderTarget3D` and registered `WebGL3DRenderTarget` color textures
use their existing native 3D attachment as the sampled input in graphics and
compute. They are admitted only when the target owner supplies a matching 3D
layout; no CPU image, second uploader or UV-oriented replica is created.
Authored target mip levels remain available for explicit LOD sampling.

The focused `three_renderer_native_volumes.test.mjs` and
`three_renderer_volume_targets.test.mjs` suites compare generated source WGSL,
binding layouts, queued upload bytes and native ownership. They do not execute
shaders. Ordinary authored volume mip arrays and partial volume uploads remain
outside this entrance. The retained builder's module-global capability and
nearest-neighbor wrapR helper caches are unchanged; the tests disclose those
source constraints rather than presenting them as fixes.

## Native raymarched volume materials

The pinned `VolumeNodeMaterial` now selects native graphics automatically and
retains `VolumetricLightingModel` construction. Its loop, front/back ray
direction, Beer attenuation, source point/spot/rectangular-light scattering and
emission execute as source-generated WGSL through the owned WebGPU submission
path. `scatteringNode` and `scatteringEmissiveNode` retain callable JavaScript or
TSL `Fn` construction with the source `positionRay` context; generated bindings
still undergo ordinary native graph admission.

Live `steps`, offset and density/emission uniforms use the source NodeFrame
frequencies. Graph identity changes require a new program, and mutations during
updates cannot publish an old graph with new dependencies. Sampled 3D density
uses the shared texture owner. `depthNode` can borrow a preceding scene target's
depth attachment, and source spot shadows share the existing shadow owner with
ordinary opaque scene companions. These paths introduce no CPU pixel readback
or substitute scattering shader.

The source only constructs the volumetric loop when its selected lighting scope
is nonempty (or a source material lighting contribution activates it). Empty or
disabled lighting therefore leaves scattering/emission callbacks inert. An
emissive volume can use a zero-intensity source light to activate the lighting
model without adding direct scattering. The native path preserves this source
behavior. Analytic light color/intensity updates remain FRAME updates even when
multiple logical renders occur in one frame.

Twelve `three_renderer_native_volume_material.test.mjs` cases cover exact source
shader text, live and queued inputs, callable graphs, source update frequencies,
structural rejection, inactive lighting, actual depth-target borrowing and spot
shadows. The new suite plus existing sampled-volume and area-light renderer
coverage passes 36 tests. Shader construction and material graph callbacks remain
retained JavaScript; rendering uses the `general-webgpu` owner. Full material
family closure and physical-device acceleration remain separate open work.

`tests/e2e/webgpu_renderer/native_volume_material.mjs` compares emission, sampled
3D density with point/spot scattering, and actual target-depth clipping against
pinned r186. All three match exact RGBA at seven 128x96 checkpoints each under
Chromium 151.0.7922.34 / SwiftShader, with visible nonconstant output, live-change
checks and zero GPU/browser errors. The fixture advances the pinned NodeFrame
identically for each logical main frame, retaining one epoch for nested depth
renders; independent background animation timing cannot choose different FRAME
light values. These software-GPU results establish correctness for the tested
scenes, not hardware performance.

## Native read-only storage texture views

Authored `TSL.storageTexture()` texel reads now retain the pinned builder's
storage bindings in both vertex and fragment graphs. In r186 these graphics
bindings are read-only even when the node's compute access defaults to
write-only. The native layout preserves the exact WGSL storage format,
dimension, binding index and stage visibility, with no sampler. The shared
compute texture owner supplies the GPU allocation and single-mip view, so a
compute producer and its graphics consumers use the same storage without CPU
pixel uploads or readback.

The admitted formats are linear RGBA8 normalized, RGBA16 float, and
R/RG/RGBA32 float, signed integer and unsigned integer. Native layouts retain
2D, array and 3D storage view dimensions. Ordinary 2D and array texel reads are
covered by source/candidate browser execution; the pinned ordinary
`StorageTextureNode` 3D coordinate generator still has a source shader
limitation, and this change does not replace it or establish 3D read parity.

`setMipLevel()` selects the authored mip when the source binding is compiled.
Changing it afterward retains that compiled selection until material
invalidation or a new graph rebuilds the binding, matching r186. Compatible
texture replacements reuse the program while queued draws retain their own
texture identity and compiled mip. Separate stage bindings may share the same
native view. Compute and graphics requests keep FIFO order; disposed or
reconfigured queued textures fail before drawing. Compilation prepares shader
and binding layouts without dispatching a producer or allocating storage
pixels, except for explicit geometry-compute prerequisites described above.
Per-stage storage texture limits are enforced before native pipeline
allocation. Graphics texture stores, comparison, bias, gradient and gather
variants remain outside this read-only profile.

Eleven `three_renderer_storage_textures.test.mjs` cases check exact source WGSL,
formats, stage aliases, mip capture, queued replacements, shared compute
residency, limits, compilation and stale texture rejection. The direct GPU
suite also checks layout snapshotting and invalid access, dimension and format
rejection. The `native_storage_reads` browser lane compares RGBA8 2D, RGBA32F
array and RGBA32Uint 2D compute-authored mips 0/1/2 in both shader stages at
seven checkpoints per scene. All three matched pinned r186 exactly under
Chromium 151 software Vulkan, including updates and hide/show transitions,
with no GPU or browser errors. These results establish rendering correctness
for those scenes; they do not measure hardware acceleration.

## Explicit remaining boundaries

This is a bounded native graphics entrance, not full TSL coverage. Non-atomic
graphics writes, atomics outside the scalar integer fragment profile above,
matrix storage outside the padded Float32 profile, nested/boolean/atomic
struct members and managed uniform arrays remain refused.
Render-triggered compute beyond the source geometry profile and resource ordering
boundaries described above is unsupported; generated graphics writes still
require an explicitly admitted graphics access profile.
Other custom light models, custom shadow replacement nodes
and custom lighting-manager subclasses remain missing.
Media, compressed, depth/comparison, array, volume, storage,
unfilterable or integer profiles beyond those admitted above still require
additional native profiles. Sampled
texture capture transactions reject reentrant texture capture from node updates
or upload callbacks.
The profile excludes shader context extensions and unregistered shadow lifecycles,
skinned or morph meshes,
including per-instance morph textures; XR/multiview and layered ArrayCamera bundles;
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

## Native stereo scene passes

The pinned `stereoPass`, `anaglyphPass` and `parallaxBarrierPass` addons now
execute their scene and composite draws through the native renderer. The source
`StereoCamera` retains eye separation, projection updates and camera identities.
Side-by-side output preserves the source half-width viewport/scissor commands;
anaglyph and parallax output sample their independent left/right eye targets
using the unchanged source composite graphs.

Scene-pass admission now allows a bounded `PassNode` subclass prototype chain
while retaining the source named-texture/history methods and ordinary FRAME
lifecycle. Subclasses keep their construction, sizing and update methods. All
targets declared in their own fields or finite arrays use the existing native
effect generation owner, so resizing a later frame cannot replace the eye or
output images already accepted by earlier queued frames. Unsupported resource
operations still pass through the normal renderer admission; this does not
claim arbitrary pass subclasses or XR presentation are complete.

Seven `three_renderer_stereo_passes.test.mjs` cases compare composite WGSL with
the pinned builder and check actual eye cameras, viewports/scissors, source
target identities, queued colors and resized generations, plus restoration
after a child callback throws. These are command/resource tests, without a
stereo GPU pixel comparison or hardware performance claim.

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
