# Translation root motion and in-place locomotion

A walk or run clip often moves the character's skeleton across its model space.
Simply repeating that clip resets this translation at every loop. The packed
playback path can now separate selected root translation axes into an explicit
motion track, play the remaining skeletal animation in place, and advance the
character's external placement without that repeat seam.

This is translation extraction and playback, not automatic humanoid recognition,
yaw extraction, physics, foot locking, or a second animation clock.

## Extract, install, and move

Generated `playback.mjs` and `gpu_playback.mjs` entries export the root-motion
helpers. In repository code, import them from `animation_root_motion.mjs` and
use the existing `createAnimationController` with a packed player.

```js
import {
  createPlayer, createAnimationController, extractAnimationRootMotion,
} from './playback.mjs';

const pose = createPlayer();
const take = extractAnimationRootMotion(pose, {
  clip: 0,
  node: rootNodeIndex, // explicit forest-root TRS node, not a guessed bone
  axes: [true, false, true], // X/Z locomotion; retain Y bob in the pose
});
const [clip] = pose.addClips([take.clip]);
const controller = createAnimationController(pose);
const locomotion = controller.createAction(clip, {
  rootMotion: take.rootMotion,
  loop: 'repeat',
}).play();

let placement = null; // null means identity, or supply an affine scene matrix
function updateAnimation(deltaSeconds) {
  controller.update(deltaSeconds, {
    rootMotionAction: locomotion,
    rootMatrix: placement,
  });
  placement = controller.rootMotionMatrix;
  // Refresh existing CPU deformers or upload this pose on the existing GPU path.
  // Do not advance this controller a second time for the same frame.
}
```

**Pass the last successful placement back on the next update.** The controller
never guesses whether the application's matrix already includes previous motion.
Omitting `rootMatrix` starts that update's composition from identity; it does not
implicitly accumulate the previous placement. The application may replace the
matrix to reposition, rotate, scale, or teleport its character.

For an existing CPU model, construct a controller over `cpuModel.pose`, perform
that controller update, then call `cpuModel.update()` to refresh its deformers.
That CPU model method refreshes geometry; it does not advance an action clock.
A scene/model that already owns a controller should keep that controller rather
than create a competing one. Use its existing refresh/upload boundary after a
direct controller update, instead of an additional clock-advancing model update.

The placement and in-place pose are passed to **one** existing pose blend. World
transforms and every mesh-local skin palette are rebuilt together, without an
extra edit or an intermediate published pose. A later deformation, GPU upload,
rendering, or external application effect is a separate boundary; those failures
do not undo an already committed CPU controller update.

## What extraction preserves

`extractAnimationRootMotion(pose, {clip, node, axes, name})` reads independent
snapshots from the live player. It neither samples that player, edits its current
pose, appends a clip, nor changes existing action indices. `clip` defaults to zero;
`node` is required. `axes` is three booleans, defaults to `[true,false,true]`, and
must select at least one axis. The optional output name is bounded to 4,096
characters; otherwise the original name receives an `-in-place` suffix.

The selected node must have no parent and must use TRS, not a matrix. Its local
translation is consequently in model coordinates. A child hip/joint under a
transform hierarchy is refused instead of incorrectly treating that local space
as model space. The chosen clip must actually animate that root's translation.

For selected axes, every in-place position key becomes the channel's first
position value. Selected cubic tangents become zero. Nonselected axes retain
their original values and tangents, so vertical motion can remain in the pose.
Other translation, rotation, scale and morph channels stay intact. Key times,
STEP/LINEAR/CUBICSPLINE interpolation, clip duration and endpoint holds are
preserved; this is not fixed-rate resampling or linearization of cubic motion.

The reference is the clip's first translation value, **not** the current animated
pose, the imported rest pose, or a guessed bind stance. Track positions themselves
remain absolute authored values; playback computes differences. Starting or
seeking to a different clip phase does not invent the displacement between that
phase and the reference. Supply an appropriate initial placement when needed.

The returned envelope contains `sourceClip`, `node`, frozen `axes`, a reusable
`clip`, and a `rootMotion` descriptor with format `f3d-root-translation-v1`. The
clip and descriptor arrays belong to the caller and are JSON-serializable. Install
the clip explicitly with `pose.addClips([take.clip])`; attach the motion descriptor
to that new clip's action. This pairing matters: attaching motion to the original
traveling clip and also applying external displacement would double the travel.
Extraction does not silently install anything or mark an authored clip as replaced.

## Action motion and placement ownership

`createAction(clip, {rootMotion})` validates and copies the optional descriptor.
Its duration must equal the installed clip's duration. `action.rootMotionTrack`
is a frozen metadata snapshot, including frozen time/value arrays. Mutating the
caller's original arrays later cannot change playback.

`action.setRootMotion(descriptor)` replaces the track atomically and returns the
action. `null` disables extraction for that action. A failed replacement retains
the previous track and storage charge. Replacement never moves its playhead or
synthesizes motion. Existing actions remain valid when more clips are installed.

`action.rootMotionDelta` is a frozen `[x,y,z]` model-space displacement from the
last **successful** controller update. It is not an accumulating queue or total
position. Before the first update it is zero. An inactive or stationary action
publishes zero on the next successful update. Configuration changes leave the
previous successful snapshot intact until another update commits, just like the
controller's event snapshots. Consume a result once, not again after an error.

Motion is deliberately **unweighted**. Pose weight, additive mode and node masks
do not scale it. A zero-weight action with a running clock still produces motion,
just as it can still produce loop and marker events. During a crossfade, both
running actions expose independent deltas. This avoids implying that pose-blend
weights define a universally correct locomotion or collision policy.

`update(delta, {rootMotionAction, rootMatrix})` applies only the explicitly chosen
live action. An action belonging to another controller, a disposed action, or an
action without a motion track is refused. No "best" action is selected from fade
weights. The immutable successful result appears as `controller.rootMotionMatrix`.
A successful update without a motion driver leaves that getter `null` and retains
the original pose `rootMatrix` semantics; per-action deltas are still available.

Applications with their own controller/physics policy may omit `rootMotionAction`
and consume each action's delta themselves. `applyAnimationRootMotion(matrix,
delta)` independently returns a frozen affine placement calculated as
`matrix * Translation(delta)`. Existing rotation, scale or shear transforms the
model-space displacement; the linear part is unchanged. There is no inferred
heading and no source-root rotation extraction. The application owns collision
response, action switching and any deliberate motion blending. Such external
placement is not part of the controller's atomic publication unless supplied to
the controller before its blend.

The external matrix places the **whole** packed model, including all forest
roots, not just the extracted skeleton. Select a model/clip whose movement has
that intended meaning; this is not multi-character motion separation inside one
shared packed pose.

## Playback boundaries

Root motion consumes the same local travel segments as action playback. For a
repeat segment, displacement is the sampled end minus sampled start, plus the
signed authored end-to-start stride for every actual repeat seam. The final
arrival of a finite action is not a wrap into a nonexistent next traversal.
Reverse playback uses the opposite signed stride. Large loop counts are handled
arithmetically, not by iterating through every traversal.

Ping-pong reflections do not add repeat strides. Full outward/return legs cancel,
and the remaining displacement is the difference between their local endpoint
samples. STEP discontinuities and cubic values are evaluated using their original
interpolation, including channels starting after zero or ending before the clip.
A rate warp that reverses inside one update contributes both travel segments. A
finite action finishing before the turn has no nonexistent returning motion.

Scheduled actions remain motionless until their deadline. Late starts use the
existing catch-up distance, even on a host `update(0)`. Paused, stopped and
zero-speed actions do not travel. Fade-to-stop truncates travel at its deadline.
Finished/clamped actions do not repeatedly emit motion. Zero-duration clips have
no translation displacement, even though they may emit a completion event.

Play/reset/seek/sync are controls, not implied travel. They do not move external
placement. Existing clamp/stop/rest behavior for the skeletal pose remains
unchanged. A partial-weight in-place clip can still blend against a different
imported rest translation; extraction does not rewrite the player's rest pose.

## Limits, rollback and persistence

A motion track accepts at most 1,048,576 keys and requires finite values and
strictly increasing times inside its duration. Cubic tracks require at least two
keys. The controller has a shared limit of **16,777,216 copied time/value
components** across all live action motion tracks, independently of the player's
clip budget. `controller.rootMotionComponents` reports the current charge.
Replacement/disposal releases that action's charge; rejected installation consumes
no action slot or component budget. Admission checks occur before copying values.

The standalone track factory also accepts `{maxComponents}` from zero through
16,777,216. These are component/admission limits, not total-process memory,
retained-snapshot, wall-time or GPU memory guarantees. Extraction makes independent
clip/local-pose snapshots under the player's existing limits. Replacement may
briefly retain both old and new copies, and callers can retain old snapshots.

Nonfinite displacement or placement is refused. A failed motion calculation,
marker-budget overflow, invalid matrix, detached pose output or failed real pose
blend leaves every committed action clock, fade, warp, schedule, pose byte/version,
event snapshot, motion delta and automatic placement result unchanged. Retry
explicitly with corrected options; do not apply a retained old delta again.
Caller getter effects are not a rollback sandbox. Existing reentrancy and
lifetime guards remain in effect.

The extracted in-place clip can use ordinary clip persistence/export paths.
The associated motion descriptor is separate action metadata: it is not added to
`pose.snapshotClips()`, automatically inferred from glTF extras, or written into a
GLB by extraction. Save `take.rootMotion` alongside the corresponding clip and
restore the pair together.

## Validation

```sh
node --test tools/ingest/animation_root_motion.test.mjs \
  tools/ingest/animation_controller_root_motion.test.mjs \
  tools/ingest/build_animation_root_motion.test.mjs
```

Tests cover analytic translation/cubic motion, 10,000 periodic-path comparisons,
real morph and distinct skin-palette publication, loop/reversal/schedule/fade
boundaries, partitioned updates, explicit crossfade leadership, failure rollback,
exact storage ceilings and released budgets. Production package tests build real
glTF data, relocate the generated directory, delete source assets, and import and
execute each interpolation mode with the production pose and CPU deformer. They
also verify emitted artifact hashes and output-budget refusal. These are CPU
semantics and package-consumer tests, not native GPU pixels, full-repository
coverage, Three.js equivalence or performance benchmarks.
