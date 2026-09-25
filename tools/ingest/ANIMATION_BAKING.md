# Bake procedural animation into reusable clips

`bakeAnimationClip(definition, options)` evaluates a bounded time interval on a
**new private animation player** and records selected local properties. It turns
sampled animation, blended actions, inverse kinematics and procedural edits into
an ordinary clip that no longer needs those controllers or solvers for playback.
It never rewinds a live model or samples deformed mesh vertices.

For recording an already running model at caller-selected times, use
`createAnimationRecorder` in `ANIMATION_RECORDING.md` instead.

## Existing clip resampling

```js
import {bakeAnimationClip} from './animation_bake.mjs';

const baked = bakeAnimationClip(definition, {
  clip: 0,
  tracks: [
    {node: hipNode, path: 'translation'},
    {node: armNode, path: 'rotation'},
    {node: faceNode, path: 'weights'},
  ],
  start: 0,
  end: 2,
  frameRate: 30,
  interpolation: 'LINEAR',
  name: 'resampled-take',
});
```

Without a custom evaluator, `clip` defaults to zero and `end` defaults to that
clip's duration. The production pose sampler evaluates the source interpolation,
including CUBICSPLINE. Sampling is non-looping: an explicitly extended range uses
the sampler's normal endpoint-clamping behavior. The output is LINEAR or STEP;
it does not reconstruct source tangents or promise exact motion between frames.

## Controllers, blending and inverse kinematics

`createEvaluator(pose)` constructs a synchronous evaluation session over the
private player. It returns `{sample(frame), dispose?}`. `clip` and
`createEvaluator` are mutually exclusive; custom evaluation needs an explicit
`end` time, and can operate on a definition with no authored clips at all.

```js
import {bakeAnimationClip} from './animation_bake.mjs';
import {createAnimationController} from './animation_controller.mjs';
import {solveAnimationIK} from './animation_ik.mjs';

const baked = bakeAnimationClip(targetDefinition, {
  start: 0,
  end: 2,
  frameRate: 60,
  name: 'walk-and-reach',
  tracks: [
    {node: hipNode, path: 'translation'},
    {node: hipNode, path: 'rotation'},
    {node: shoulderNode, path: 'rotation'},
    {node: elbowNode, path: 'rotation'},
  ],
  createEvaluator(pose) {
    const controller = createAnimationController(pose);
    controller.createAction(0, {loop: 'repeat'}).play();
    return {
      sample({time, delta}) {
        controller.update(delta);
        solveAnimationIK(pose, {
          effector: handNode,
          links: [{node: elbowNode}, {node: shoulderNode}],
          target: targetAtTime(time),
          iterations: 32,
        });
        // Any additional pose.edit(...) calls belong here, before capture.
      },
      dispose() { controller.dispose(); },
    };
  },
});

// model must have been constructed with sourceExport:true and must correspond
// to targetDefinition's original glTF node order, not synthetic instance nodes.
const glb = await model.exportAnimationGLB([baked.clip]);
// Or: await exportGltfAssetGLB(targetAsset, {clips: [baked.clip]});
```

An IK solve can be limited or unreachable. The evaluator should inspect its
returned `converged`/`distance` result and throw when its own quality threshold is
not met. The baker records the resulting committed pose; it does not reinterpret
solver success, automatically add missing tracks, or change chain constraints.
Include **all** affected local bindings needed for complete replay. Omitting a
joint's rotation or a morph channel is an explicit partial-track bake.

The returned evaluator is owned for the bake's duration. Its sample and cleanup
methods run with the evaluator as `this`. Cleanup is called once before the
private player is disposed, on success, sample failure or cancellation. A cleanup
failure prevents a successful return, but does not replace an earlier sampling
or cancellation error. A factory that throws before returning must release its
own partially constructed resources; the baker still disposes its private pose.

## Time grid and state

Each frozen frame contains `time` (absolute source time), `delta`, `index`,
`frameCount`, `relativeTime` and `progress`. The first frame's delta is zero;
subsequent deltas reflect the actual interval, including a shorter last step.
The private player starts in its imported rest pose. The baker does not reset it
between custom samples, so stateful controllers and simulations can advance in
order. For a nonzero `start`, a stateful evaluator must initialize/seek its own
state for that starting time; no unrequested warm-up simulation is performed.

Both endpoints are included exactly once. A zero-duration interval contains one
frame; a shorter-than-one-frame interval contains its two endpoints. The full
time grid is admitted before creating the custom evaluator, including precision
checks at large absolute times and after rebasing to Float32. Unrepresentable or
collapsing keys fail rather than silently dropping a frame.

The recorder converts times and values to Float32, rebases keys to zero, and
keeps quaternion signs continuous. `sourceRange` preserves the original first
and last source times. `duration` is the stored, rounded last key;
`requestedDuration` is the original `end - start`. `frameRate`, `frameCount`,
`components`, `interpolation` and `work` describe the bake. Metadata explicitly
reports `approximate:true`, `execution:'javascript-cpu-animation-bake'` and
`accelerationClaim:false`. No adaptive sampling or performance claim is implied.

Local TRS and morph weights are the recording profile. External `rootMatrix`
placement, matrix-node motion, visibility, controller events, solver internals,
materials and camera projection parameters are not serialized. A portable root
motion track must be a local root translation/rotation. The source definition
and any separately running model/player are not mutated by the baker itself.

## Admission, failure and execution boundaries

The recorder's `maxFrames` (6000), `maxComponents` (16,777,216), `maxNodes` (65,536)
and `maxTracks` (4096) options apply. Components include repeated per-track time
arrays. The complete output extent is checked before the first evaluator call.
`maxWork` defaults to 16,777,216 **node-frame** units and bounds scheduled pose
frames, not the internal work of arbitrary callbacks, skin palettes or IK solves.
The existing player enforces its own input/skin/clip storage limits. Private pose,
time grid, recorder snapshots/frames and completed arrays are separate bounded
stages, not one aggregate process-memory limit.

The optional `signal` is checked before setup, between frames, before capture and
before completion. A cancelled or failed bake returns no partial clip. Factory,
sample and cleanup callbacks must be synchronous; thenables are rejected and
late promise rejections are observed. They are trusted application code, not a
sandbox: external side effects cannot be rolled back, asynchronous external
resource creation cannot be reclaimed, and callback runtime is not preempted.
The synchronous bake does not yield to receive later browser abort events.

The focused tests execute the exact production sampler, recorder and CCD IK
solver. They compare baked replay's local/world transforms, morphs and separate
skin palettes for mesh instances sharing a skeleton, including moving targets,
LINEAR/STEP output, all three source interpolation modes, frame precision,
budget admission, cancellation and cleanup ordering. They do not establish GPU
rendering, pixel equivalence, native Three.js controller parity or a speedup.

These are direct runtime-module APIs. They do not change the default sampling-only
`buildAnimation` package or add an implicit authoring/recording loop to rendering.
