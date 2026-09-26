/** Bounded, synchronous CCD inverse kinematics over the existing pose player.
 * A solve reads private local-pose snapshots and publishes at most one edit.
 * No clocks, GPU resources, animation resampling, or intermediate live poses.
 * This is an explicit positional solver, not Three.js CCDIKSolver API parity.
 * Active links require orientation-preserving uniform-scale world transforms.
 * Hinges use rest-relative local axes/angle intervals, not Euler box limits.
 */
import { AnimationPoseError } from "./animation_runtime.mjs";

const fail = (code, message) => {
  throw new AnimationPoseError("ANIMATION_IK_" + code, message);
};
const finite = (v, label) => {
  if (typeof v !== "number" || !Number.isFinite(v)) fail("VALUE", `${label} must be finite`);
  return v;
};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
function record(v, keys, label) {
  if (
    !v ||
    typeof v !== "object" ||
    Array.isArray(v) ||
    Object.keys(v).some((k) => !keys.includes(k))
  )
    fail("OPTIONS", `Invalid ${label}`);
}
function vector(v, n, label) {
  if ((!Array.isArray(v) && !ArrayBuffer.isView(v)) || v.length !== n)
    fail("VALUE", `Invalid ${label}`);
  const result = Array.from(v, (x) => finite(x, label));
  if (result.length !== n) fail("VALUE", `Invalid ${label} iterator extent`);
  return result;
}
function unit(v) {
  const scale = Math.max(...v.map(Math.abs));
  if (!scale) return null;
  const scaled = v.map((x) => x / scale),
    length = Math.hypot(...scaled);
  if (!Number.isFinite(scale) || !Number.isFinite(length)) fail("VALUE", "Vector overflow");
  return scaled.map((x) => x / length);
}
function quaternion(q) {
  const value = vector(q, 4, "quaternion"),
    norm = Math.hypot(...value);
  if (Math.abs(norm - 1) > 1e-3) fail("ROTATION", "Expected unit quaternion");
  return value.map((x) => x / norm);
}
function product(a, b) {
  const [x, y, z, w] = a,
    [X, Y, Z, W] = b;
  return [
    x * W + w * X + y * Z - z * Y,
    y * W + w * Y + z * X - x * Z,
    z * W + w * Z + x * Y - y * X,
    w * W - x * X - y * Y - z * Z,
  ];
}
const axisAngle = (axis, angle) => {
  const s = Math.sin(angle / 2);
  return [...axis.map((v) => v * s), Math.cos(angle / 2)];
};
function slerp(a, b, t) {
  let d = a.reduce((s, x, i) => s + x * b[i], 0);
  const sign = d < 0 ? -1 : 1;
  d = clamp(Math.abs(d), 0, 1);
  let x = 1 - t,
    y = t;
  if (1 - d > Number.EPSILON) {
    const angle = Math.acos(d),
      s = Math.sin(angle);
    x = Math.sin((1 - t) * angle) / s;
    y = Math.sin(t * angle) / s;
  }
  return quaternion(a.map((v, i) => x * v + y * sign * b[i]));
}
function compose(t, q, s) {
  const [x, y, z, w] = q,
    x2 = x + x,
    y2 = y + y,
    z2 = z + z,
    xx = x * x2,
    xy = x * y2,
    xz = x * z2,
    yy = y * y2,
    yz = y * z2,
    zz = z * z2,
    wx = w * x2,
    wy = w * y2,
    wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0],
    (xy + wz) * s[0],
    (xz - wy) * s[0],
    0,
    (xy - wz) * s[1],
    (1 - (xx + zz)) * s[1],
    (yz + wx) * s[1],
    0,
    (xz + wy) * s[2],
    (yz - wx) * s[2],
    (1 - (xx + yy)) * s[2],
    0,
    ...t,
    1,
  ];
}
function multiply(a, b, out) {
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      out[c * 4 + r] = finite(
        a[r] * b[c * 4] +
          a[4 + r] * b[c * 4 + 1] +
          a[8 + r] * b[c * 4 + 2] +
          a[12 + r] * b[c * 4 + 3],
        "World matrix",
      );
}
// A normalized basis, not a lossy matrix decomposition: anisotropy, shear and
// reflections are refused before they can corrupt local rotation updates.
function basis(m) {
  const columns = [0, 4, 8].map((o) => [m[o], m[o + 1], m[o + 2]]),
    lengths = columns.map((c) => Math.hypot(...c));
  if (
    lengths.some((x) => !Number.isFinite(x) || x <= 0) ||
    lengths.some((x) => Math.abs(x / lengths[0] - 1) > 1e-8)
  )
    fail("TRANSFORM", "IK needs positive uniform-scale world transforms");
  const axes = columns.map((c, i) => c.map((v) => v / lengths[i]));
  if (
    Math.abs(dot(axes[0], axes[1])) > 1e-8 ||
    Math.abs(dot(axes[0], axes[2])) > 1e-8 ||
    Math.abs(dot(axes[1], axes[2])) > 1e-8 ||
    dot(cross(axes[0], axes[1]), axes[2]) < 1 - 1e-8
  )
    fail("TRANSFORM", "IK does not admit shear or reflections");
  return axes;
}
function rotationBetween(a, b, maxAngle) {
  let axis = cross(a, b);
  const length = Math.hypot(...axis),
    d = clamp(dot(a, b), -1, 1);
  let angle = Math.atan2(length, d);
  if (!angle) return null;
  if (length < 1e-14) {
    if (d >= 0) return null;
    // Deterministic antiparallel direction, including exactly straight chains.
    let at = 0;
    for (let i = 1; i < 3; i++) if (Math.abs(a[i]) < Math.abs(a[at])) at = i;
    const perpendicular = [0, 0, 0];
    perpendicular[at] = 1;
    axis = unit(cross(a, perpendicular));
  } else axis = axis.map((x) => x / length);
  angle = Math.min(angle, maxAngle);
  return axisAngle(axis, angle);
}
function hingeState(input, initial, rest) {
  record(input, ["axis", "min", "max", "referenceRotation"], "hinge");
  const axis = unit(vector(input.axis, 3, "hinge axis"));
  if (!axis) fail("HINGE", "Hinge axis must be nonzero");
  const min = finite(input.min ?? -Math.PI, "hinge min"),
    max = finite(input.max ?? Math.PI, "hinge max");
  if (min < -Math.PI || max > Math.PI || min > max)
    fail("HINGE", "Hinge interval must satisfy -pi <= min <= max <= pi");
  const reference = quaternion(input.referenceRotation ?? rest),
    inverse = reference.map((v, i) => (i === 3 ? v : -v));
  let relative = quaternion(product(inverse, initial));
  if (relative[3] < 0) relative = relative.map((v) => -v);
  const twist = dot(relative, axis),
    swing = relative.slice(0, 3).map((v, i) => v - axis[i] * twist);
  if (Math.hypot(...swing) > 1e-7)
    fail("HINGE", "Entry rotation is not a twist about the reference hinge axis");
  const angle = 2 * Math.atan2(twist, relative[3]);
  if (angle < min - 1e-8 || angle > max + 1e-8)
    fail("HINGE", "Entry rotation is outside hinge limits");
  return {
    axis,
    min,
    max,
    reference,
    angle: clamp(angle, min, max),
    initialAngle: clamp(angle, min, max),
  };
}

/** solveAnimationIK(pose, {effector, target:[worldX,worldY,worldZ],
 * links:[{node,enabled=true,hinge?}], iterations=16, tolerance=1e-4,
 * maxAngle=pi/2, weight=1}) -> {converged,distance,initialDistance,iterations,
 * poseVersion,changedNodes}. Links run nearest-effector first and must be strict
 * ancestors in that order (intermediate fixed nodes may be omitted).
 * At most 32 links, 64 iterations and 256 nodes from effector to scene root.
 * Apply after controller/sample/blend, then update/upload existing deformers.
 * Unreachable/limited targets return converged:false, not a fabricated success.
 */
export function solveAnimationIK(pose, options) {
  record(
    options,
    ["effector", "target", "links", "iterations", "tolerance", "maxAngle", "weight"],
    "solver options",
  );
  if (typeof pose?.snapshotLocalPose !== "function" || typeof pose.edit !== "function")
    fail("POSE", "Expected an editable animation player");
  const snapshot = pose.snapshotLocalPose(),
    n = snapshot.nodeCount;
  if (snapshot.format !== "f3d-local-pose-v1" || !Number.isSafeInteger(n) || n < 1 || n > 65536)
    fail("POSE", "Invalid pose snapshot");
  const nodeIndex = (value) => {
    if (!Number.isInteger(value) || value < 0 || value >= n) fail("CHAIN", "Invalid pose node");
    return value;
  };
  const effector = nodeIndex(options.effector),
    target = vector(options.target, 3, "world target");
  const iterations = options.iterations ?? 16,
    tolerance = finite(options.tolerance ?? 1e-4, "tolerance");
  const maxAngle = finite(options.maxAngle ?? Math.PI / 2, "maxAngle"),
    weight = finite(options.weight ?? 1, "weight");
  if (
    !Number.isInteger(iterations) ||
    iterations < 1 ||
    iterations > 64 ||
    tolerance < 0 ||
    maxAngle <= 0 ||
    maxAngle > Math.PI ||
    weight < 0 ||
    weight > 1
  )
    fail("OPTIONS", "Invalid solve bounds");
  const inputLinks = options.links;
  if (!Array.isArray(inputLinks) || !inputLinks.length || inputLinks.length > 32)
    fail("CHAIN", "Expected 1..32 links");
  const matrixNodes = new Map(snapshot.matrices.map(({ node, matrix }) => [node, matrix]));
  const chain = [],
    visited = new Set();
  let cursor = effector;
  while (cursor !== -1) {
    nodeIndex(cursor);
    if (visited.has(cursor) || chain.length >= 256)
      fail("CHAIN", "Cyclic or excessive ancestor chain");
    visited.add(cursor);
    chain.push(cursor);
    cursor = snapshot.parents[cursor];
  }
  chain.reverse();
  const slots = new Map(chain.map((node, i) => [node, i]));
  const path = chain.map((node) => {
    const translation = vector(
      snapshot.translations.subarray(node * 3, node * 3 + 3),
      3,
      "translation",
    );
    const rotation = vector(snapshot.rotations.subarray(node * 4, node * 4 + 4), 4, "rotation");
    const scale = vector(snapshot.scales.subarray(node * 3, node * 3 + 3), 3, "scale");
    return {
      node,
      translation,
      rotation,
      scale,
      local: matrixNodes.get(node) ?? compose(translation, rotation, scale),
      world: new Float64Array(16),
    };
  });
  let previous = path.length - 1;
  const links = inputLinks.map((input) => {
    record(input, ["node", "enabled", "hinge"], "link");
    const node = nodeIndex(input.node),
      slot = slots.get(node);
    if (slot === undefined || slot >= previous)
      fail("CHAIN", "Links must be strict ancestors, nearest first");
    previous = slot;
    const enabled = input.enabled ?? true;
    if (typeof enabled !== "boolean") fail("OPTIONS", "enabled must be boolean");
    if (enabled && matrixNodes.has(node)) fail("TRANSFORM", "Active IK links require TRS nodes");
    const scale = path[slot].scale;
    if (
      enabled &&
      (scale.some((v) => v <= 0) || scale.some((v) => Math.abs(v / scale[0] - 1) > 1e-8))
    )
      fail("TRANSFORM", "Active link local scale must be positive and uniform");
    const initial = quaternion(path[slot].rotation),
      hingeInput = input.hinge;
    const hinge =
      hingeInput === undefined
        ? null
        : hingeState(hingeInput, initial, snapshot.restRotations.subarray(node * 4, node * 4 + 4));
    return { node, slot, enabled, initial, hinge };
  });
  function propagate(start = 0) {
    for (let i = start; i < path.length; i++) {
      const entry = path[i],
        parent = i ? path[i - 1].world : snapshot.rootMatrix;
      if (parent) multiply(parent, entry.local, entry.world);
      else for (let k = 0; k < 16; k++) entry.world[k] = finite(entry.local[k], "World matrix");
    }
  }
  function setRotation(link, q) {
    const entry = path[link.slot];
    entry.rotation = quaternion(q);
    entry.local = compose(entry.translation, entry.rotation, entry.scale);
    propagate(link.slot);
  }
  const end = path[path.length - 1].world;
  const distance = () =>
    finite(
      Math.hypot(target[0] - end[12], target[1] - end[13], target[2] - end[14]),
      "Target distance",
    );
  const live = () => {
    if (pose.disposed || pose.version !== snapshot.version)
      fail("STALE", "Pose changed during solver input evaluation");
  };
  propagate();
  for (const link of links) if (link.enabled) basis(path[link.slot].world);
  const initialDistance = distance();
  let performed = 0,
    current = initialDistance,
    moved = false;
  live();
  if (weight > 0)
    for (let i = 0; i < iterations && current > tolerance; i++) {
      let changed = false;
      performed++;
      for (const link of links) {
        if (!link.enabled) continue;
        const entry = path[link.slot],
          m = entry.world,
          axes = basis(m);
        const a = unit([end[12] - m[12], end[13] - m[13], end[14] - m[14]]),
          b = unit(target.map((v, j) => v - m[12 + j]));
        if (!a || !b) continue;
        const localA = unit(axes.map((axis) => dot(axis, a))),
          localB = unit(axes.map((axis) => dot(axis, b)));
        let q;
        if (link.hinge) {
          const h = link.hinge,
            pa = unit(localA.map((v, j) => v - h.axis[j] * dot(localA, h.axis))),
            pb = unit(localB.map((v, j) => v - h.axis[j] * dot(localB, h.axis)));
          if (!pa || !pb) continue;
          const step = clamp(
            Math.atan2(dot(h.axis, cross(pa, pb)), dot(pa, pb)),
            -maxAngle,
            maxAngle,
          );
          const next = clamp(h.angle + step, h.min, h.max);
          if (next === h.angle) continue;
          h.angle = next;
          q = product(h.reference, axisAngle(h.axis, next));
        } else {
          const delta = rotationBetween(localA, localB, maxAngle);
          if (!delta) continue;
          q = product(entry.rotation, delta);
        }
        setRotation(link, q);
        changed = true;
        moved = true;
        current = distance();
        if (current <= tolerance) break;
      }
      if (!changed) break;
    }
  if (moved && weight > 0 && weight < 1)
    for (const link of links)
      if (link.enabled) {
        const q = link.hinge
          ? product(
              link.hinge.reference,
              axisAngle(
                link.hinge.axis,
                link.hinge.initialAngle + weight * (link.hinge.angle - link.hinge.initialAngle),
              ),
            )
          : slerp(link.initial, quaternion(path[link.slot].rotation), weight);
        setRotation(link, q);
      }
  const edits =
    !moved || weight === 0
      ? []
      : links
          .filter(
            (link) =>
              link.enabled &&
              path[link.slot].rotation.some((v, i) => v !== snapshot.rotations[link.node * 4 + i]),
          )
          .map((link) => ({ node: link.node, rotation: path[link.slot].rotation }));
  // The existing player is the only publisher. A failed final palette evaluation
  // cannot expose any of the intermediate solver rotations to live consumers.
  const finalDistance = distance();
  live();
  if (edits.length) pose.edit(edits);
  return Object.freeze({
    converged: finalDistance <= tolerance,
    distance: finalDistance,
    initialDistance,
    iterations: performed,
    poseVersion: pose.version,
    changedNodes: Object.freeze(edits.map((edit) => edit.node)),
  });
}

// Analytic limbs share the CCD module's quaternion and transform admission.
// They do not change the CCD solver's convergence or hinge-limit contract.
const limbOperations = new WeakSet();
const subtract = (a, b) => a.map((v, i) => finite(v - b[i], "Position difference"));
const positionOf = (entry) => Array.from(entry.world.subarray(12, 15));
const inverseQuaternion = (q) => q.map((v, i) => i === 3 ? v : -v);
function limbVector(input, width, label) {
  if ((!Array.isArray(input) && !ArrayBuffer.isView(input)) || input.length !== width)
    fail("VALUE", `Invalid ${label}`);
  // Bounded indexed reads: an array's custom iterator cannot enlarge the work.
  return Array.from({ length: width }, (_, i) => finite(input[i], label));
}
function basisQuaternion(axes) {
  const [a, b, c] = axes, trace = a[0] + b[1] + c[2];
  let q, s;
  if (trace > 0) {
    s = 2 * Math.sqrt(trace + 1);
    q = [(b[2] - c[1]) / s, (c[0] - a[2]) / s, (a[1] - b[0]) / s, s / 4];
  } else if (a[0] > b[1] && a[0] > c[2]) {
    s = 2 * Math.sqrt(1 + a[0] - b[1] - c[2]);
    q = [s / 4, (b[0] + a[1]) / s, (c[0] + a[2]) / s, (b[2] - c[1]) / s];
  } else if (b[1] > c[2]) {
    s = 2 * Math.sqrt(1 + b[1] - a[0] - c[2]);
    q = [(b[0] + a[1]) / s, s / 4, (c[1] + b[2]) / s, (c[0] - a[2]) / s];
  } else {
    s = 2 * Math.sqrt(1 + c[2] - a[0] - b[1]);
    q = [(c[0] + a[2]) / s, (c[1] + b[2]) / s, s / 4, (a[1] - b[0]) / s];
  }
  return quaternion(q);
}
function limbSnapshot(pose) {
  if (!pose || typeof pose.snapshotLocalPose !== "function" || typeof pose.edit !== "function")
    fail("POSE", "Expected an editable animation player");
  const snapshot = pose.snapshotLocalPose(), n = snapshot.nodeCount;
  if (snapshot.format !== "f3d-local-pose-v1" || !Number.isSafeInteger(n) || n < 1 || n > 65536)
    fail("POSE", "Invalid pose snapshot");
  return snapshot;
}
function limbLive(pose, snapshot) {
  if (pose.disposed || pose.version !== snapshot.version)
    fail("STALE", "Pose changed during limb input evaluation");
}
function stageTwoBoneIK(snapshot, input) {
  record(input, ["root", "joint", "effector", "target", "pole", "weight", "tolerance",
    "endRotation", "orientationWeight", "orientationTolerance"], "two-bone solver options");
  const options = { ...input }, n = snapshot.nodeCount;
  const index = (v) => {
    if (!Number.isInteger(v) || v < 0 || v >= n) fail("CHAIN", "Invalid limb node");
    return v;
  };
  const root = index(options.root), joint = index(options.joint), effector = index(options.effector);
  if (root === joint || joint === effector || root === effector ||
      snapshot.parents[joint] !== root || snapshot.parents[effector] !== joint)
    fail("CHAIN", "A two-bone limb requires direct root -> joint -> effector parentage");
  const target = limbVector(options.target, 3, "world target"),
    pole = limbVector(options.pole, 3, "world pole");
  const weight = finite(options.weight ?? 1, "weight"),
    tolerance = finite(options.tolerance ?? 1e-4, "tolerance"),
    orientationWeight = finite(options.orientationWeight ?? weight, "orientationWeight"),
    orientationTolerance = finite(options.orientationTolerance ?? 1e-4, "orientationTolerance");
  const endRotation = options.endRotation === undefined ? null :
    quaternion(limbVector(options.endRotation, 4, "world end rotation"));
  if (weight < 0 || weight > 1 || orientationWeight < 0 || orientationWeight > 1 ||
      tolerance < 0 || orientationTolerance < 0 || orientationTolerance > Math.PI ||
      (!endRotation && options.orientationWeight !== undefined))
    fail("OPTIONS", "Invalid limb weights, tolerances or missing orientation target");
  const matrixNodes = new Map(snapshot.matrices.map(({ node, matrix }) => [node, matrix]));
  const chain = [], seen = new Set();
  for (let node = effector; node !== -1; node = snapshot.parents[node]) {
    index(node);
    if (seen.has(node) || chain.length >= 256) fail("CHAIN", "Cyclic or excessive limb ancestry");
    seen.add(node); chain.push(node);
  }
  chain.reverse();
  const path = chain.map((node) => {
    const translation = limbVector(snapshot.translations.subarray(node * 3, node * 3 + 3), 3, "translation"),
      rotation = limbVector(snapshot.rotations.subarray(node * 4, node * 4 + 4), 4, "rotation"),
      scale = limbVector(snapshot.scales.subarray(node * 3, node * 3 + 3), 3, "scale");
    return { node, translation, rotation, scale,
      local: matrixNodes.get(node) ?? compose(translation, rotation, scale), world: new Float64Array(16) };
  });
  const rootSlot = path.length - 3, a = path[rootSlot], b = path[rootSlot + 1], c = path[rootSlot + 2];
  const rotatable = endRotation ? [a, b, c] : [a, b];
  for (const entry of rotatable) {
    if (matrixNodes.has(entry.node) || entry.scale.some((v) => v <= 0 ||
        Math.abs(v / entry.scale[0] - 1) > 1e-8))
      fail("TRANSFORM", "Rotated limb nodes require positive uniform-scale TRS");
    quaternion(entry.rotation); // Validate without changing the committed baseline.
  }
  function propagate(start = 0) {
    for (let i = start; i < path.length; i++) {
      const entry = path[i], parent = i ? path[i - 1].world : snapshot.rootMatrix;
      if (parent) multiply(parent, entry.local, entry.world);
      else for (let k = 0; k < 16; k++) entry.world[k] = finite(entry.local[k], "World matrix");
    }
  }
  function setRotation(slot, q) {
    const entry = path[slot];
    entry.rotation = quaternion(q);
    entry.local = compose(entry.translation, entry.rotation, entry.scale);
    propagate(slot);
  }
  function align(slot, from, to) {
    const axes = basis(path[slot].world), u = unit(from), v = unit(to);
    if (!u || !v) fail("LENGTH", "Limb direction collapsed at working precision");
    const delta = rotationBetween(axes.map((axis) => dot(axis, u)),
      axes.map((axis) => dot(axis, v)), Math.PI);
    if (delta) setRotation(slot, product(path[slot].rotation, delta));
  }
  propagate();
  const rootAxes = basis(a.world);
  basis(b.world);
  if (endRotation) basis(c.world);
  const origin = positionOf(a), upper = subtract(positionOf(b), origin),
    lower = subtract(positionOf(c), positionOf(b)), toTarget = subtract(target, origin);
  const l1 = finite(Math.hypot(...upper), "Upper bone length"),
    l2 = finite(Math.hypot(...lower), "Lower bone length"),
    targetDistance = finite(Math.hypot(...toTarget), "Root-target distance"),
    initialDistance = finite(Math.hypot(...subtract(target, positionOf(c))), "Initial target distance");
  const scale = Math.max(l1, l2);
  if (!l1 || !l2 || Math.min(l1, l2) / scale < 1e-12)
    fail("LENGTH", "Two nonzero bones with a length ratio of at least 1e-12 are required");
  const minReach = Math.abs(l1 - l2), maxReach = finite(l1 + l2, "Limb reach"),
    reach = clamp(targetDistance, minReach, maxReach);
  // At the root target, preserve the current root-effector axis when it exists.
  const direction = unit(toTarget) ?? unit(subtract(positionOf(c), origin)) ?? unit(upper);
  const perpendicular = (v) => {
    const u = unit(v);
    if (!u) return null;
    const projected = u.map((x, i) => x - direction[i] * dot(u, direction));
    return Math.hypot(...projected) > 1e-10 ? unit(projected) : null;
  };
  let bend = perpendicular(subtract(pole, origin)), poleFallback = "none";
  if (!bend) { bend = perpendicular(upper); poleFallback = "current"; }
  if (!bend) {
    const axis = rootAxes.reduce((best, next) =>
      Math.abs(dot(next, direction)) < Math.abs(dot(best, direction)) ? next : best);
    bend = perpendicular(axis); poleFallback = "axis";
  }
  // Normalized law of cosines avoids squaring world-scale lengths. Equal
  // lengths also avoid dividing by a vanishing distance at a fully folded limb.
  let cosine;
  if (l1 === l2) cosine = (reach / l1) * 0.5;
  else {
    const u = l1 / scale, v = l2 / scale, d = reach / scale;
    cosine = 0.5 * (d + ((u - v) * (u + v)) / d) / u;
  }
  cosine = clamp(finite(cosine, "Limb bend cosine"), -1, 1);
  const sine = Math.sqrt(Math.max(0, (1 - cosine) * (1 + cosine))),
    desiredUpper = direction.map((v, i) => v * cosine + bend[i] * sine),
    endpoint = origin.map((v, i) => finite(v + direction[i] * reach, "Reachable endpoint"));
  const initialRoot = a.rotation.slice(), initialJoint = b.rotation.slice();
  if (weight > 0) {
    align(rootSlot, upper, desiredUpper);
    align(rootSlot + 1, subtract(positionOf(c), positionOf(b)), subtract(endpoint, positionOf(b)));
    if (weight < 1) {
      const solvedRoot = a.rotation.slice(), solvedJoint = b.rotation.slice();
      setRotation(rootSlot, slerp(initialRoot, solvedRoot, weight));
      setRotation(rootSlot + 1, slerp(initialJoint, solvedJoint, weight));
    }
  }
  if (endRotation && orientationWeight > 0) {
    // Blend from the world orientation AFTER the positional solve. Thus zero
    // orientation influence leaves the end's local rotation alone continuously.
    const current = basisQuaternion(basis(c.world)), parent = basisQuaternion(basis(b.world));
    setRotation(rootSlot + 2, product(inverseQuaternion(parent), slerp(current, endRotation, orientationWeight)));
  }
  const distance = finite(Math.hypot(...subtract(target, positionOf(c))), "Final target distance");
  let orientationError = null;
  if (endRotation) {
    const delta = product(inverseQuaternion(endRotation), basisQuaternion(basis(c.world)));
    orientationError = 2 * Math.atan2(Math.hypot(...delta.slice(0, 3)), Math.abs(delta[3]));
  }
  const edits = rotatable.filter((entry) => entry.rotation.some((v, i) =>
    v !== snapshot.rotations[entry.node * 4 + i])).map(({ node, rotation }) => ({ node, rotation }));
  // Weight-zero requests must not publish normalization-only changes.
  const selected = edits.filter(({ node }) => node === effector ? orientationWeight > 0 : weight > 0);
  return { edits: selected, chain, nodes: [root, joint, effector], result: {
    converged: distance <= tolerance && (orientationError === null || orientationError <= orientationTolerance),
    positionConverged: distance <= tolerance,
    orientationConverged: orientationError === null || orientationError <= orientationTolerance,
    reachable: targetDistance >= minReach && targetDistance <= maxReach,
    distance, initialDistance, targetDistance, solvedDistance: reach,
    orientationError, poleFallback,
  } };
}

/** Analytic root -> joint -> effector IK, with a world-space pole POINT.
 * Solve straight/folded limbs, clamp unreachable targets without stretching,
 * blend local rotations by weight, optionally align endRotation in world space.
 * See ANIMATION_LIMB_IK.md. This adds no clock, renderer, or contact detection.
 */
export function solveAnimationTwoBoneIK(pose, options) {
  if (!pose || (typeof pose !== "object" && typeof pose !== "function"))
    fail("POSE", "Expected an editable animation player");
  if (limbOperations.has(pose)) fail("REENTRANT", "Limb solve cannot be reentered");
  limbOperations.add(pose);
  try {
    const snapshot = limbSnapshot(pose), staged = stageTwoBoneIK(snapshot, options);
    limbLive(pose, snapshot);
    if (staged.edits.length) pose.edit(staged.edits);
    return Object.freeze({ ...staged.result, poseVersion: pose.version,
      changedNodes: Object.freeze(staged.edits.map(({ node }) => node)) });
  } finally { limbOperations.delete(pose); }
}

/** Solve up to 32 independent limbs from ONE committed snapshot and publish
 * ONE pose edit. Shared fixed ancestors are allowed; overlapping limbs or a
 * limb containing another limb's ancestor are refused, independent of weights.
 * This is not an order-dependent sequential or whole-body optimization solver.
 */
export function solveAnimationLimbIK(pose, limbs) {
  if (!pose || (typeof pose !== "object" && typeof pose !== "function"))
    fail("POSE", "Expected an editable animation player");
  if (limbOperations.has(pose)) fail("REENTRANT", "Limb solve cannot be reentered");
  limbOperations.add(pose);
  try {
    const snapshot = limbSnapshot(pose), count = limbs?.length;
    if (!Array.isArray(limbs) || !Number.isSafeInteger(count) || count < 0 || count > 32)
      fail("LIMIT", "Expected at most 32 independent limb requests");
    const staged = [], owners = new Map();
    for (let i = 0; i < count; i++) {
      const limb = stageTwoBoneIK(snapshot, limbs[i]);
      limbLive(pose, snapshot);
      for (const node of limb.nodes) {
        if (owners.has(node)) fail("OVERLAP", "Batch limb nodes must be disjoint");
        owners.set(node, i);
      }
      staged.push(limb);
    }
    for (let i = 0; i < staged.length; i++)
      for (const node of staged[i].chain)
        if (owners.has(node) && owners.get(node) !== i)
          fail("DEPENDENCY", "One batch limb cannot be an ancestor of another");
    const edits = staged.flatMap((limb) => limb.edits);
    limbLive(pose, snapshot);
    if (edits.length) pose.edit(edits);
    const poseVersion = pose.version;
    return Object.freeze({
      poseVersion,
      converged: staged.every((limb) => limb.result.converged),
      changedNodes: Object.freeze(edits.map(({ node }) => node)),
      limbs: Object.freeze(staged.map((limb) => Object.freeze({ ...limb.result, poseVersion,
        nodes: Object.freeze(limb.nodes),
        changedNodes: Object.freeze(limb.edits.map(({ node }) => node)) }))),
    });
  } finally { limbOperations.delete(pose); }
}
