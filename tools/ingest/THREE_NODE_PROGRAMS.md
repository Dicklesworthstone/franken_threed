# Native TSL graphics programs

The native `WebGPURenderer` facade can draw authored TSL graphs on ordinary
meshes. It retains the exact r186 `WGSLNodeBuilder`, node classes, uniform packing
and `NodeFrame`; FrankenThreeD owns the WebGPU pipelines, geometry residency,
uniform buffers, command encoding and submission. The path does not construct an
upstream renderer, translate through ESSL reflection, or claim measured
acceleration or a Rust implementation of node evaluation.

## Implemented graph surface

Use `MeshBasicNodeMaterial`, or an unlit base `NodeMaterial`, with `colorNode`,
`positionNode`, `vertexNode`, `fragmentNode`, `opacityNode`, `alphaTestNode` and
`outputNode`. Ordinary source node arithmetic, functions and custom scalar/vector
float attributes compile through the pinned builder. Native scalar, integer,
boolean, vector and matrix uniforms retain their source group ordering and
padding. Clip-space vertex and fragment graphs with no uniform groups also work.

Authored `MeshToonNodeMaterial`, `MeshLambertNodeMaterial` and
`MeshPhongNodeMaterial` additionally use the pinned lighting models with ambient,
hemisphere, directional, point and spot lights. They accept `normalNode` and
`emissiveNode`; Phong also accepts `shininessNode` and `specularNode`. These slots
compose with ordinary TSL arithmetic and the sampled texture graphs below.
Unassigned slots retain their source material color, emissive, specular and
shininess behavior. Both material arrays and scene overrides supply the effective
material to the builder's context.

Authored `texture()` graphs sample ordinary 2D data, decoded image/canvas and
resolved color render-target textures. The pinned builder owns UV channel and
matrix selection, explicit/generated UVs, color-space interpretation, explicit
levels and vertex/fragment sampling. Bind groups preserve the source's exact
resource order: a uniform block can follow samplers/textures, and groups can
contain resources without a uniform block. Filterable float formats are admitted;
float32 inputs require the actual `float32-filterable` device feature.

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

Captured graph descriptors, raster state, source list properties, draw ranges and
uniform bytes survive later source edits. Material/geometry disposal invalidates
the compiler cache, and subsequent graph or structural changes select a new
program. Geometry arrays and attribute versions retain the existing bridge's
live upload contract; this is not a deep snapshot of an entire mutable scene.
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

This is a bounded native graphics entrance, not full TSL coverage. Storage/buffer,
compute and multiple-render-target graphics graphs are refused. Authored
Standard/Physical lighting, environment lighting, area/probe/custom lights,
shadows, projected light maps and custom `lightsNode` graphs remain missing.
Depth/comparison, cube, array, volume, compressed, storage, video and unfilterable
or integer texture graphs still require additional native profiles. Sampled
texture capture transactions reject reentrant texture capture from node updates
or upload callbacks.
The profile excludes shader context extensions, fog/clipping/shadows, material
textures, wireframe, alpha hashing and alpha-to-coverage; instanced, skinned,
batched or morph meshes; ArrayCamera, reversed depth and XR views; custom
object/material render hooks and effectful attribute upload callbacks. A
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

The three owner/capture test files cover actual
pinned shader generation and packing, callback contexts and frequencies,
independent byte history, call-time selection/ranges, admission and budgets,
pipeline failures, device loss and disposal:

```sh
node tools/ingest/three_node_program.test.mjs
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
versus unsupported active lights. Source shader construction and capture checks
passed before the execution environment became unavailable. Real WebGPU lighting
comparisons remain to be completed; the texture parity results above do not
establish lit-graph pixel equivalence.
