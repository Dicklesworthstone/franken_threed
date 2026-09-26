# Pole-controlled limb IK

`solveAnimationTwoBoneIK` adds an analytic positional/orientation solve alongside
`solveAnimationIK` (the existing bounded CCD/hinge solver). It can bend a perfectly
straight arm or leg toward an inward target, including the collinear case where
CCD has no angular descent direction. A world-space pole point selects the elbow
or knee half-plane. This is an explicit pose operation, not a physics, contact,
terrain-query, gait, or Three.js solver replacement.

```js
import {solveAnimationTwoBoneIK} from './animation_ik.mjs';

// The application's existing animation clock remains authoritative.
controller.update(deltaSeconds);
const result = solveAnimationTwoBoneIK(pose, {
  root: hipNode,
  joint: kneeNode,
  effector: ankleNode,
  target: desiredFootPosition, // world-space [x,y,z]
  pole: kneeGuidePosition,    // world-space POINT, not a direction
  weight: 1,
  endRotation: desiredFootQuaternion, // optional world-space [x,y,z,w]
  orientationWeight: 1,
});
// Consume the same edited pose through the existing CPU/GPU path.
scene.upload();
```

The root, joint and effector must be distinct, directly parented nodes in that
order. The application maps its actual rig to those indices; no bone-name,
retargeting, foot-contact or anatomical inference happens here. The solver changes
only local rotations, never translations, scales, bone lengths, morph weights,
clip installation, the controller, or the player's current time/clip. Sampling,
blending or resetting the player later replaces procedural edits as usual.

The solver takes one private local-pose snapshot, computes world transforms in
bounded scratch, and publishes at most one `pose.edit`. It never treats writable
public output arrays as source data. The player remains the sole evaluator of
world transforms and mesh-local skin palettes; a rejected final palette evaluation
cannot expose intermediate joint rotations. Public output-array identities remain
stable. It owns no GPU resources and does not update or upload deformers implicitly.

## Position, pole and orientation

The target is clamped to the reachable interval `abs(upperLength-lowerLength)`
through `upperLength+lowerLength`, without stretching. Fully folded equal-length
limbs and antiparallel targets are admitted. A pole on the root-target axis has no
unique bend plane: the current joint's projected bend is used first, then a
deterministic axis from the root's current basis. `poleFallback` reports `none`,
`current`, or `axis`; there is no hidden previous-frame state. Supply a continuous,
non-collinear pole for animation that needs continuous bend direction.

`weight` defaults to one and lies in [0,1]. It blends the root/joint local
quaternions toward the complete analytic solution using shortest-path SLERP.
Partial influence is not guaranteed to reach the target. Zero positional weight
does not publish normalization-only root/joint edits.

Optional `endRotation` is a unit world quaternion. `orientationWeight` defaults
to the positional weight, also in [0,1], and requires an orientation target when
specified. Orientation blends from the end's world orientation **after** the
positional solve toward the requested rotation, then converts back to a local
end rotation under the solved parent. Thus zero orientation influence leaves the
end's local rotation unchanged. Setting positional weight zero and orientation
weight one permits orientation-only hand/foot placement. Without `endRotation`,
the effector's local rotation remains unchanged, not its world orientation.

`tolerance` defaults to 1e-4 world units; `orientationTolerance` defaults to 1e-4
radians and lies in [0,pi]. The immutable result contains `distance` and
`initialDistance`, `targetDistance` (root to target), `solvedDistance` (clamped
root-target distance), `reachable`, `positionConverged`, `orientationError`
(null without an orientation target), `orientationConverged`, `converged`,
`poseVersion`, and immutable `changedNodes`. Convergence measures the final,
weighted scratch pose against the requested targets, not an unweighted ideal.
Geometric reachability and tolerance-based convergence are distinct.

## Admission and failure boundaries

Rotated nodes require positive uniform local scales and orientation-preserving,
uniform-scale world bases. Shear, reflections, anisotropy and matrix-authored
rotated nodes fail explicitly. Fixed matrix-authored ancestors and a matrix-authored
position-only effector are supported when the active world transforms satisfy
those requirements. The imported root matrix remains authoritative and unchanged.

There are at most 65,536 pose nodes and 256 nodes on the effector-to-root path.
Both bones must have nonzero finite world lengths, with shorter/longer ratio at
least 1e-12. The law of cosines uses normalized lengths instead of squaring world
lengths, but this is floating-point geometry, not exact arithmetic for arbitrarily
ill-conditioned rigs. Collapsed or overflowing positions fail rather than claim
a solution. There are no CCD iterations, hinge/Euler limits, twist constraints,
bone stretching, multiple-effector optimization or automatic collision avoidance.
Use the existing CCD solver for its separately documented hinge constraints.

Unknown options, malformed vectors/quaternions, excessive ancestry, nonadjacent
chains, invalid transforms and non-finite intermediates fail before publication.
Input arrays are read by bounded indices, without trusting their custom iterators.
Same-pose recursive limb solves are rejected. A source getter that changes or
disposes the pose is detected before publication; its own external side effects
are not rolled back. Existing pose errors retain their codes; solver admission
errors use `ANIMATION_IK_*` codes.

## Validation

```sh
node --test tools/ingest/animation_two_bone_ik.test.mjs
```

The tests execute the actual production pose sampler, transactional editor and
skin-palette evaluator, not a pose fixture. They include the straight-chain CCD
regression, opposite/degenerate poles, full folding, unreachable targets, world
transforms, end orientations, blending, final-publication failure, and randomized
analytic endpoint/length/pole checks. CPU pose tests are not native GPU pixel or
performance evidence, and do not establish parity with Three.js CCDIKSolver.
