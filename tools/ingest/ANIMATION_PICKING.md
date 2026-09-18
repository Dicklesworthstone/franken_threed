# Current-pose triangle selection

`animation_raycast.mjs` provides synchronous world-space geometric queries over
existing CPU deformation outputs. It does not advance animation, own a device,
read GPU buffers, install input listeners, or replace Three.js Raycaster.

```js
import {
  createAnimationRaycaster, rayFromAnimationCamera,
} from './animation_raycast.mjs';

const picker = createAnimationRaycaster(pose, [{
  deformer, indices, texCoords, doubleSided: false,
  source: { node: deformer.node, mesh: 0, primitive: 0, material: 0 },
}]);
const ray = rayFromAnimationCamera(view.sample({ aspectRatio: width / height }), [x, y]);
const hits = picker.raycast(ray, { firstHitOnly: true });
picker.dispose(); // never disposes the borrowed pose or deformers
```

`x` and `y` are normalized device coordinates in [-1,1], with Y pointing up.
Perspective rays start at the camera center; orthographic rays start at the
selected point on its camera plane and remain parallel. This helper consumes the
existing camera sample rather than reconstructing a scene or reading a canvas.
Ray `origin` and `direction` can instead be supplied directly. Direction is
normalized; optional `near` (default 0) and `far` (default Infinity) are **world-ray
distances**, inclusive. Camera clipping planes are not implicitly applied.

## Results and selection semantics

The returned array and every hit are immutable snapshots, ordered by distance,
then original draw index, then source face index. A hit includes `drawIndex`,
`node`, `source`, `faceIndex`, `distance`, world `point`, `barycentric`,
`frontFacing`, and world geometric `normal`. The normal faces against the ray;
it is not a vertex-interpolated or normal-mapped shading normal. Optional `uv` is
interpolated from the supplied UV set without applying the material UV transform.
`source` preserves the original glTF node/mesh/primitive/material IDs when supplied.

Triangle indices, UVs, side flags and source IDs are snapshotted at construction.
The deformer's positions and world matrix are borrowed and must be treated as
read-only outputs. Every deformer must match the current pose version before a
query. Direct pose sampling must be followed by deformation updates. Detached,
shared, resizable, nonfinite or stale output storage is refused, never interpreted
as an empty scene. Query and disposal reentry is rejected.

The query honors source front-facing triangles and `doubleSided`, including
reflected world transforms. Nonuniform scale, shear and surviving triangles under
singular transforms are handled in world space; fully degenerate triangles miss.
It tests geometry, **not rendered coverage**: texture alpha tests, transparency,
normal maps, shader displacement, per-draw renderer overrides and application
visibility are not evaluated. `drawIndices` explicitly restricts a query to a
unique list of registered draw indices. Shared edges can return both source faces.
There is no claim of complete Three.js interaction/control compatibility.

## Acceleration and resource bounds

A median-split bounding-volume hierarchy (BVH) with at most eight triangles per
leaf is built on the first query of each selected mesh. Subsequent deformation
versions refit its world-space bounds without rebuilding its topology. Repeated
queries of unchanged outputs reuse both positions and bounds. Nearest-only
traversal visits nearer boxes first and prunes boxes beyond the best known hit.
A hierarchy built before extreme deformation can become less efficient, but is
still refitted rather than using stale bounds. This is a CPU data-structure path,
not a measured GPU or application-level speedup.

`maxTriangles` bounds the aggregate topology (default and hard maximum 1,048,576).
`maxBytes` defaults to 128 MiB and bounds the raycaster's owned typed arrays,
including temporary centroid storage required by construction. Borrowed CPU
geometry/pose arrays, JS objects and returned hits are not included; it is not a
hard process-memory ceiling. An all-hit query defaults to `maxHits: 4096` and
fails rather than silently truncating on overflow. `firstHitOnly: true` selects
the actual nearest hit, not the first triangle encountered in storage order.
`bufferBytes` reports retained typed storage, and `lastQuery` reports successful
query counts and the pose version. Failed queries leave prior hit snapshots and
statistics intact; a failed refit is retried on the next valid query.

## Focused checks

```sh
node --test tools/ingest/animation_raycast.test.mjs
```

Tests cover independently expected rays/triangles, transformations, source IDs,
UVs, near/far limits, reflection/sides, hierarchy reuse/refit/pruning, storage and
lifecycle failures. A deterministic 256-triangle, 100-ray comparison uses an
independent Moller-Trumbore brute-force oracle rather than the production
shear-edge intersection routine. The tests supply CPU deformation outputs; they
do not establish animation-factory integration, GPU execution or pixel parity.
