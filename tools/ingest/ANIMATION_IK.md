# Procedural inverse kinematics

`solveAnimationIK()` adds synchronous positional CCD solving to the existing
packed pose player. Use it for a hand/foot/tool target after authored sampling
and before updating deformation. It changes local joint rotations, propagates
the same node hierarchy, and rebuilds the same per-mesh skin palettes. There is
no second animation clock, live intermediate pose, GPU allocation or readback.

```js
import {solveAnimationIK} from './animation_ik.mjs';

model.controller.update(deltaSeconds);
const result = solveAnimationIK(model.pose, {
  effector: handNode,
  target: [worldX, worldY, worldZ],
  links: [
    {node: elbowNode, hinge: {axis: [0, 0, 1], min: 0, max: 2.5}},
    {node: shoulderNode},
  ],
  iterations: 24,
  tolerance: 0.001,
  maxAngle: Math.PI / 4,
  weight: 1,
});
model.upload();
// Render with the model's usual render()/renderCamera() attachment contract.
console.log(result.converged, result.distance);
```

The target is a **world-space position**, including the pose's current root
transform. Effector and link indices are pose node indices, not skin palette
indices. List links nearest the effector first, then outward. Each must be a
strict ancestor of the preceding entry; fixed intermediate nodes can be omitted.
`enabled:false` keeps a link fixed. Neither bone translations nor scales change;
unreachable goals never stretch the skeleton.

## Constraints and results

Defaults are 16 iterations, tolerance `1e-4`, per-link-step `maxAngle=pi/2`, and
weight one. Work is bounded to 32 links, 64 iterations and 256 ancestors. Weight
must be in `[0,1]`. A partial weight blends the solved local rotations with the
entry pose, rather than scaling elapsed animation time.

A `hinge` supplies a nonzero local axis, an angle interval satisfying
`-pi <= min <= max <= pi`, and optionally a `referenceRotation` unit quaternion.
The default reference is the **imported rest rotation**, not the previously solved
frame. Angles stay relative to that fixed reference across calls. Entry rotation
must already be a pure hinge twist inside the interval; swing or out-of-range
poses are refused, not silently projected. A blended hinge interpolates its scalar
angle within the allowed interval, avoiding a quaternion shortest path through a
forbidden wrap interval. Other links use shortest-path quaternion blending.

The frozen result contains `converged`, `distance`, `initialDistance`, `iterations`,
`poseVersion`, and `changedNodes`. Distance describes the final **weighted** pose.
A constrained, unreachable or iteration-limited target can return
`converged:false`; CCD does not guarantee a global optimum. Exactly antiparallel
vectors have a deterministic rotation axis. Degenerate zero-length vectors do not
produce NaNs. Already-satisfied targets and zero-weight solves do not publish a
new pose version.

The solver uses the player's private local snapshot, so accidental writes to
public output arrays do not become solver input. It performs all iterations in
private scratch and calls `pose.edit()` at most once. Failed validation, stale
input snapshots, or a failing final world/palette evaluation leave the committed
pose unchanged. External side effects of input getters are not rolled back.
After a successful solve, explicitly update/upload the existing deformers before
rendering, picking or posed export. A later sample/blend/reset replaces procedural
rotations; run IK again after sampling on the next frame. Authored-asset source
export continues to preserve the original asset, not these live modifications.

## Supported transform profile

Active links must be TRS-authored and have positive, uniform local scale. Their
world transforms must also be orientation-preserving uniform-scale transforms.
Shear, anisotropic scale and reflections on active links are rejected before live
publication. Matrix-authored fixed ancestors are allowed when those conditions
remain satisfied. The effector may have its own transform; the solver targets its
origin, not an arbitrary offset or orientation.

This is an explicit CPU positional solver, **not** a drop-in Three.js CCDIKSolver,
Euler rotation-box implementation, multi-effector optimizer, retargeter or physics
constraint system. It introduces no automatic source routing or acceleration
claim. Tests execute the real pose runtime, solver and CPU deformer, including
independent palettes, morph streams, bounds, normal/tangent outputs and rollback.
They do not establish GPU pixels, universal rig compatibility or a speedup.
