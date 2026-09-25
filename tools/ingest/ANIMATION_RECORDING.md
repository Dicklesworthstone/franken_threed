# Record live poses as reusable animation

`createAnimationRecorder(pose, options)` turns the committed local state of an
existing packed animation player into a reusable `f3d-animation-v1` clip. Capture
**after** controller blending, retargeting, pose edits or inverse kinematics. It
reads `snapshotLocalPose()`, not writable public output arrays or deformed mesh
vertices. The recorder does not advance the player or install a clock.

```js
import {createAnimationRecorder} from './animation_recorder.mjs';

// model was loaded with sourceExport:true to enable rig-preserving GLB export.
// These must be the original destination glTF node indices, not draw/palette IDs.
const recorder = createAnimationRecorder(model.pose, {
  name: 'edited-take',
  tracks: [
    {node: hipNode, path: 'translation'},
    {node: hipNode, path: 'rotation'},
    {node: armNode, path: 'rotation'},
    {node: faceNode, path: 'weights'},
  ],
  interpolation: 'LINEAR',
  maxFrames: 3600,
  maxComponents: 1_000_000,
});

// In the caller's existing frame loop, after all pose changes:
model.update(deltaSeconds);
// Apply retargeting/IK/pose edits here, as appropriate for the application.
recorder.capture(elapsedSeconds);

// On an explicit stop action, outside that loop:
const take = recorder.finish();
const glb = await model.exportAnimationGLB([take.clip]);
recorder.dispose();
```

The same clip can be appended to a decoded definition's `clips` array before
creating a new player, or passed to `exportGltfAssetGLB(asset, {clips:[take.clip]})`.
Recording and exporting do not automatically change a running player's clip
table. To install the take into the existing model without replacing its pose,
use the explicit live-clip API:

```js
const [clipIndex] = model.pose.addClips([take.clip]);
const replay = model.controller.createAction(clipIndex);
// Explicitly play it, or crossfade from the application's current action.
model.controller.crossFade(currentAction, replay, 0.2);
```

Existing actions keep contributing until explicitly stopped or crossfaded. A
live recorder can continue capturing after other clips are installed; registry
changes alone do not advance the pose version. `model.pose.snapshotClips([clipIndex])`
returns independent installed keyframes for later reuse/export even after the
original `take.clip` arrays have been discarded. See `ANIMATION_LIVE_CLIPS.md` for
stable IDs, cumulative limits, selected snapshots and source-node boundaries.
A model's `exportSourceGLB()` still returns its original authored snapshot.

## Track and time semantics

Tracks are an explicit nonempty list of unique `{node, path}` bindings. Supported
paths are `translation`, `rotation`, `scale` and `weights`. Weight widths come
from the committed snapshot's morph offsets. Nodes using matrix transforms may
have weight tracks, but cannot be falsely represented as TRS tracks. An unknown
path, duplicate binding, missing morph weights or out-of-range node is an error.
The default maximum is 4096 tracks and 65,536 pose nodes.

`capture(time, {signal})` accepts finite, strictly increasing caller times,
including negative absolute times. The first **successful** capture establishes
the origin, and the output starts at zero. The recorder immediately rounds keys
and values to Float32 so it cannot accept distinct keys that the GLB exporter
would later collapse, or values that would overflow FLOAT storage. Signed zero
is retained. Quaternion signs are made continuous relative to the preceding
accepted frame without changing the represented orientation. Nonunit quaternion
keys are rejected rather than silently normalized.

Output interpolation is `LINEAR` (default) or `STEP`. No cubic tangents or motion
between recorded frames are recovered. Float32 rounding can shift a transition
slightly; fixed or irregular capture spacing is an approximation, not exact
preservation of the original controllers or solvers. Metadata reports
`approximate:true` and `accelerationClaim:false`.

Only selected **local** properties are recorded. External `rootMatrix` placement,
animated matrix nodes, camera/material parameters, events, visibility and solver
state are not captured. Record a root node's local translation/rotation for
portable root motion. Include every affected local property needed for a complete
replay; omitted properties use the destination player's normal rest/clip rules.
No world-transform decomposition or mesh baking is performed. In particular,
synthetic GPU-instance nodes must not be confused with original glTF node IDs.

## Ownership, transactions and limits

`capture()` returns the recorder for chaining. It stages all selected values and
checks the pose version and layout before accepting any part of a frame. A failed
capture does not consume time, frame count or component budget. It is legal to
record an unchanged pose at a later time. The recorder never changes the borrowed
pose's version, arrays, controller or lifetime.

`finish({signal})` requires at least one accepted frame. It returns caller-owned,
ordinary number arrays in `clip`, with `frameCount`, `components`, `duration`,
`sourceRange` (original first/last capture times) and `interpolation`. It seals the
recorder and releases its retained frame storage and pose reference. The result
remains usable after recorder or pose disposal. A pose disposed before finishing
prevents new captures but does not prevent finishing previously owned frames.
Finishing an empty recorder is a recoverable error. `dispose()` is idempotent,
releases unfinished frames and never disposes the borrowed pose.

Defaults are 6000 frames and 16,777,216 emitted numeric components. Components
include the repeated time array of **each** channel, not just stored values.
`componentsPerFrame`, `components`, `frameCount`, `duration`, `finished` and
`disposed` expose the recorder's state. Frame and component limits are checked
before requesting another full-pose snapshot. Snapshot storage is bounded by the
admitted pose layout (up to 1,048,576 total morph weights); retained frames and
final output arrays are separately bounded stages, not an aggregate process
memory budget. Finishing can temporarily hold both frames and output arrays.

Signals are per capture/finish operation. An aborted operation leaves prior
accepted frames available. Calls are synchronous and do not yield for later
browser abort events. No GPU work, worker or background scheduler is introduced.

The regression suite executes the real packed pose runtime, blend/edit paths,
local/world transforms and per-instance skin palettes. Deliberately malformed
snapshot providers exercise transaction failures. These CPU results do not
establish native GPU execution, pixels, automatic Three.js renderer routing or
performance improvements.
