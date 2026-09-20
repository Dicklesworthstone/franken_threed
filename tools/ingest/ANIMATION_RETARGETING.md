# Motion transfer between animated rigs

The packed pose path supports explicit reference-relative skeletal retargeting
and reusable destination clip baking. These utilities use the existing pose
sampler, transactional editor and skin-palette publication. They do not introduce
a renderer, frame scheduler or a second animation clock.

## Transfer a live pose

```js
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createAnimationRetargeter} from './animation_retarget.mjs';
import {mapAnimationNodeNames} from './animation_retarget_clip.mjs';

// Definitions are f3d-animation-v1, e.g. decodedModel.definition. Construct the
// binding while both players are in their intended reference stance.
const sourcePose = createAnimationPlayer(sourceDefinition);
const mapping = mapAnimationNodeNames(sourceDefinition, targetDefinition, [
  {source: 'Hip', target: 'Pelvis'},
  {source: 'LeftArm', target: 'UpperArm_L'},
]);
const transfer = createAnimationRetargeter(sourcePose, targetModel.pose, {
  mapping,
  // Optional reference-relative displacement with explicit unit/stride scaling:
  rootMotion: {source: mapping[0].source, target: mapping[0].target, scale: 1.2},
});

// In the application's existing frame loop:
sourcePose.sample(timeSeconds, {clip: 0, loop: true});
transfer.apply();
targetModel.upload(); // GPU model: upload the complete edited pose before render.
targetModel.render(frame);

// Release helpers separately. Disposing the transfer never disposes either pose.
transfer.dispose();
sourcePose.dispose();
```

For a CPU model, call `targetModel.update()` after transfer to recompute its
existing CPU deformers. Do not immediately call a destination animation sampler
unless replacing the transferred pose is intended. The next sample/blend/reset
retains its existing semantics and replaces procedural edits.

Bindings may instead be explicit `{source: nodeIndex, target: nodeIndex,
weight: 1}` records. Input order need not be hierarchical. A source node may
serve multiple destinations; a destination may occur only once. Named mapping
is an explicit alias list: exact, unique names are required, with no prefix
stripping, fuzzy matching or silent missing-bone selection. This avoids confusing
mesh names or similarly named joints with the intended skeleton.

Reference stances default to the players' committed local poses at binding
construction, not inverse-bind matrices, guessed T-poses or animation frame zero.
To bind already animated models, pass saved `sourceReference` and
`targetReference` values from `pose.snapshotLocalPose()`. References and options
are copied; changing the caller's snapshots later cannot alter the binding.

## Rotation, placement and blending

For each mapped bone, the desired destination model-space orientation is:

```
alignment * sourceCurrent * inverse(sourceReference)
          * inverse(alignment) * targetReference
```

Each quantity is a unit quaternion; reference/current orientations include node
ancestors but exclude the pose's external `rootMatrix`. The destination's local
rotation is recovered against its fully retargeted parent before local blending.
Different reference
bone axes, extra unmapped helper nodes and different node-index order are handled
without copying the source's local bone offsets or stretching destination bones.

`alignment` is an optional unit quaternion mapping source-model axes into the
destination model's axes. Root-motion displacement uses the same alignment and
an explicit nonnegative scale. It is then converted back through the current
destination parent. Other destination translations, scale, morph weights and
unmapped local channels stay current. External root placement remains untouched;
placing two characters at different world locations does not transfer that scene
placement from one to the other.

`apply({weight: 0.5})` blends from the **current destination local pose**. Per-map
weights multiply this weight. Repeated partial applies consequently approach the
source solution; they do not implicitly reset the destination. Sample the desired
base destination animation first when using this as a per-frame overlay. Zero
weight does not publish an edit. A successful nonempty transfer publishes once,
rebuilding all world transforms and every mesh instance's skin palette together.
The result reports source/destination versions and updated node IDs.

## Bake reusable clips

```js
import {retargetAnimationClip} from './animation_retarget_clip.mjs';

const baked = retargetAnimationClip(sourceDefinition, targetDefinition, {
  mapping,
  rootMotion: {source: mapping[0].source, target: mapping[0].target, scale: 1.2},
  clip: 0,
  start: 0.25,
  end: 2.0,
  frameRate: 30,
  name: 'retargeted-walk',
});
const definition = {
  ...targetDefinition,
  clips: [...(targetDefinition.clips ?? []), baked.clip],
};
const player = createAnimationPlayer(definition);
player.sample(0.5, {clip: definition.clips.length - 1});
```

The baker owns private players and never samples/edits a live model. It starts
from destination rest on every sample, so partial mapping weights do not drift
while baking. It includes both source-range endpoints, rebases output times to
zero and emits ordinary JSON-serializable rotation/root-translation tracks. The
result can be replayed without the source rig or live retargeter. It does not
rewrite an already constructed player's immutable clip table or turn a loaded
model's authored-source export snapshot into the new clip.

Source STEP, LINEAR and CUBICSPLINE tracks are evaluated by the existing sampler.
The output is explicitly fixed-rate **resampling** with LINEAR or STEP
interpolation. It is not an exact transformation of cubic tangents or arbitrary
step boundaries; increase sampling density or choose STEP output as appropriate.
`approximate: true` remains explicit. No visual/error tolerance is inferred from
a frame rate, and facial morph/scale/non-root position animation is not retargeted.

## Limits and unsupported transforms

The live binding defaults to 65,536 combined source/destination nodes and 4,096
rotation mappings. Active ancestors must have positive uniform scale, either TRS
or an affine matrix; matrix shear/reflection/nonuniform scale is refused instead
of guessing a decomposition. Mapped destination nodes must use TRS. Unrelated
nodes may retain their original transforms. A final skin-palette overflow or
detached pose output cannot publish a partial edit. A later GPU upload is a
separate boundary; a GPU failure does not roll back the committed CPU pose.

Baking additionally defaults to 6,000 frames, 16,777,216 emitted time/value
components and 16,777,216 combined node-frame evaluations. Each limit is checked
before the sampling loop. These are separate bounds, not a total process-memory
or wall-time guarantee; source-player construction retains its existing limits.
Baking is synchronous. A signal is checked before construction and between
samples, but it does not create background work or yield to a browser event loop.

This is an explicit packed-pose CPU utility, not the full Three.js SkeletonUtils
API, IK/foot-contact correction, automatic humanoid characterization or a native
GPU retargeting implementation. Tests execute the real pose sampler/editor and
skin palettes; they do not establish Three.js equivalence or rendered-image parity.
