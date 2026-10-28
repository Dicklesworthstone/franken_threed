# Native source points and lines

The `WebGPURenderer` facade supports pinned r186 point and line node materials
through the same source `WGSLNodeBuilder` used by native mesh and sprite graphs.
FrankenThreeD owns GPU allocation, pipeline creation, encoding and submission.
There is no upstream renderer submission fallback.

## Materials and topology

| Source object | Materials | Native topology |
| --- | --- | --- |
| `Points` | `PointsMaterial`, `PointsNodeMaterial` | `point-list` |
| `Sprite` or `Mesh` | `PointsMaterial`, `PointsNodeMaterial` | `triangle-list` |
| `Line` | `LineBasicMaterial`, `LineBasicNodeMaterial`, `LineDashedMaterial`, `LineDashedNodeMaterial` | `line-strip` |
| `LineSegments` | The same line materials | `line-list` |

These are the validated material/object combinations. Topology follows the
source object, rather than the material name. Mesh wireframe retains its
existing line-list lowering. Indexed strips select the source index format:
`Uint16Array` and `Uint32Array` require distinct pipeline states. Switching
between indexed formats or to an unindexed strip invalidates the old
registration before drawing.

Ordinary materials with authored slots use the exact pinned node-library
conversion. Default point/line node materials and ordinary dashed lines select
the native route without a dummy node slot. Ordinary point materials on sprites
or meshes, or with color/alpha maps, also select it automatically. Ordinary
untextured points and basic lines retain their existing fixed route when they
have no native feature.

Common `positionNode`, `colorNode`, `opacityNode`, vertex/fragment and output
graphs compose with the existing node program surface. Dashed materials also
accept `offsetNode`, `dashScaleNode`, `dashSizeNode` and `gapSizeNode`. The source
shader reads `lineDistance` and applies the original dash/gap discard expression;
applications retain their normal `computeLineDistances()` or explicit distance
attribute setup.

## Point primitives and sized particle sprites

WebGPU point primitives have a size of one physical pixel. As in pinned r186,
`PointsNodeMaterial.sizeNode` does not resize an object of type `Points`, and
nodes used exclusively for sprite sizing do not receive updates on that path.
For sized particles, use the material on `Sprite` or suitable mesh geometry:

```js
const positions = TSL.instancedArray(particleCount, 'vec3');
const material = new THREE.PointsNodeMaterial({sizeAttenuation: false});
material.positionNode = positions.element(TSL.instanceIndex);
material.sizeNode = TSL.uniform(12);
material.opacityNode = TSL.shapeCircle();
const particles = new THREE.Sprite(material);
particles.count = particleCount;
particles.frustumCulled = false;
scene.add(particles);
```

The source shader applies logical pixel size, screen DPR, active viewport,
`scaleNode` and `rotationNode`. Perspective cameras additionally apply the
source logical canvas-height/depth attenuation when enabled. Orthographic
cameras retain the source nonattenuated branch. These are source point-material
equations, distinct from `SpriteNodeMaterial` world-sized billboards.

A point cloud can read the same GPU storage per vertex:

```js
material.positionNode = positions.element(TSL.vertexIndex);
const cloud = new THREE.Points(geometryWithParticleCount, material);
cloud.frustumCulled = false;
```

Storage remains the compute-owned GPU allocation. Rendering does not read back
positions or overwrite a compute result with CPU array edits lacking an upload
request. Source frustum tests still use source geometry/object bounds, so GPU
positions beyond those bounds require appropriate bounds or disabled culling.

## Accepted render state

Source object counts, draw ranges and uniform packets are captured at the
logical render boundary, before asynchronous preparation. Later queued edits
retain separate counts and uniforms. FRAME, RENDER and OBJECT updates follow
the source `NodeFrame` frequency; zero-instance draws still run their selected
node updates. Source ArrayCamera views retain distinct camera matrices without
repeating logical object updates. Negative world scale flips winding only for
meshes, matching the pinned renderer.

`LineLoop` preserves the source renderer's report-and-skip behavior, including
when it carries an authored graph. It is not silently rendered as an open strip.
Wide-line addons, custom material subclasses, XR and multiview remain outside
this material profile. Existing graph resource, shadow and bundle admission
limits still apply; this feature does not establish new shadow pixel or
hardware-performance claims.

## Validation

`node tools/ingest/three_renderer_native_primitive_graphs.test.mjs` checks all fourteen
material/object combinations against the exact source WGSL, native topology,
indexed-strip format changes, captured draw ranges, queued size/DPR/viewport and
count state, compute-storage ownership, ArrayCamera updates and LineLoop
reporting. Its GPU device records commands and resource bytes.

`node tests/e2e/webgpu_renderer/native_primitives.mjs` compares source and F3D
RGBA8 output for authored solid/dashed lines, compute-driven point clouds and
sized point sprites using perspective and orthographic cameras. It is the
separate real-WebGPU rasterization check; the recording tests alone do not
establish pixel parity or physical-device throughput.

The four browser scenarios matched the pinned source byte for byte at all seven
readback checkpoints per scenario, with no GPU validation errors on Chromium
156.0.8078.73 / SwiftShader. The mixed line scene submits four native draws, the
compute-produced point cloud one, and each sized-point scene six. This is
software-GPU correctness evidence, separate from hardware performance.
