# Live local-pose editing

`createAnimationPlayer()` now accepts explicit, transactional edits to its last
committed local pose. This enables procedural joints, interactive placement and
morph controls without writing stale world matrices or mismatched skin palettes.
It remains an explicit CPU pose API, not a Three.js Object3D/AnimationMixer facade.

```js
const pose = model.pose;
model.controller.update(deltaSeconds); // Optional: evaluate authored animation.
pose.edit([
  {node: jointNode, rotation: [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)]},
  {node: meshNode, weights: [smile, blink]},
]);
model.upload(); // Update existing deformers after ALL pose operations.
// Render using the model's existing render/renderCamera API.
```

`edit([{node, translation?, rotation?, scale?, weights?, matrix?}], options)`
uses **absolute local values**, not additive deltas. Omitted properties preserve
the previous committed values. One successful call, including `edit([])`,
increments `version` once, sets `mode` to `edit`, and rebuilds all world matrices
and every per-mesh skin palette. Stable public array identities are preserved.
`time` and `clip` retain the preceding sampler's provenance, not a new clock.

Edits target original/expanded pose node indices. Each node may appear once.
Matrix-authored nodes accept an affine `matrix` and/or morph `weights`; TRS nodes
accept TRS and/or weights. Representation changes and implicit shear decomposition
are not performed. Rotations must be unit quaternions within the existing input
tolerance and are normalized before use. Morph weights need the node's exact
width and may be negative or exceed one, as with the existing pose sampler.

The previous root transform is retained by default. `edit(changes,
{rootMatrix: matrix})` replaces it; explicit `null` clears it. Root transforms and
matrix edits are copied. Invalid edits, singular mesh transforms and numeric
failures do not publish a partial pose or advance its version. Inputs are read
and copied before evaluation. External side effects of caller getters (including
detaching an output buffer) are not rolled back; reentry is rejected.

`sample()`, `blend()` and `reset()` preserve their existing semantics: they replace
local state from authored tracks/rest and **do not retain procedural edits**.
Apply procedural edits after sampling/blending each frame, then update the
existing deformers. Picking/posed export must consume that same updated version.
The rig-preserving source exporter still saves the originally loaded asset,
not these later pose edits or newly recorded animation tracks.

## Solver / editor snapshots

`snapshotLocalPose()` returns a frozen descriptor containing independent,
caller-owned arrays: `translations`, `rotations`, `scales`, `morphWeights`,
`parents`, `morphOffsets`, `restRotations`, `rootMatrix` (or null), and `matrices`
(`[{node, matrix}]` for matrix-authored nodes). The format is
`f3d-local-pose-v1`; `nodeCount` and `version` identify the sampled state.
Changing or transferring snapshot storage cannot damage the live player.

Snapshots and subsequent edits use **private committed local state**, never
caller mutations of the public output arrays or scratch left by a failed sample.
Publication uses captured typed-array intrinsics and validates all output storage
before copying, so replacing public `.set`, `.buffer` or `.length` properties
does not inject callbacks into a partial publication. Detached output buffers or
disposed players reject operations. This is not a sandbox for replaced platform
intrinsics before module loading.

The player owns one additional local TRS/morph buffer set (80 bytes per node plus
8 bytes per morph weight). Successful evaluation swaps the private local buffers;
it does not add another full-array copy to every sample or allocate GPU resources.
Explicit snapshots and edit inputs allocate their own bounded copies. These are
implementation properties, not measured performance or visual-equivalence claims.
