/** Translation root motion for packed animation actions. This module samples
 * explicit clip-local endpoints, never a wall clock. It owns no pose or device.
 * Rotation/yaw extraction and automatic blend-weight policies are not implied.
 */
export class AnimationRootMotionError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationRootMotionError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationRootMotionError("ANIMATION_ROOT_MOTION_" + code, message);
};
const finite = (value, label) => {
  if (typeof value !== "number" || !Number.isFinite(value)) fail("VALUE", `Invalid ${label}`);
  return value;
};
function fields(value, allowed, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      Object.keys(value).some(key => !allowed.includes(key))) fail("OPTIONS", `Invalid ${label}`);
}
function numbers(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length)
    fail("SHAPE", `Invalid ${label}`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable)
      fail("STORAGE", `${label} must use fixed unshared storage`);
    try { new Uint8Array(value.buffer, 0, 0); } catch { fail("STORAGE", `Detached ${label}`); }
  }
  const result = new Array(length);
  for (let i = 0; i < length; i++) result[i] = finite(value[i], label);
  return result;
}
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Copy an explicit f3d-root-translation-v1 track. Duration is the associated
 * installed clip's duration, not necessarily its last translation key. Keys
 * before/after the requested local time hold their endpoint, as in the sampler.
 * advance() consumes actual monotone action travel, including repeat seams;
 * ping-pong and once use zero wraps. Work is logarithmic in keys, not loop count.
 */
export function createAnimationRootMotionTrack(input, duration, options = {}) {
  fields(options, ["maxComponents"], "root-motion limits");
  const { maxComponents = 16777216 } = options;
  if (!Number.isSafeInteger(maxComponents) || maxComponents < 0 || maxComponents > 16777216)
    fail("LIMIT", "Invalid root-motion component budget");
  fields(input, ["format", "duration", "interpolation", "times", "values"], "root-motion track");
  finite(duration, "clip duration");
  if (duration < 0 || input.format !== "f3d-root-translation-v1" || input.duration !== duration)
    fail("CLIP", "Root motion must match the associated clip duration");
  const interpolation = input.interpolation ?? "LINEAR";
  if (!["STEP", "LINEAR", "CUBICSPLINE"].includes(interpolation)) fail("INTERPOLATION", "Unknown interpolation");
  const cubic = interpolation === "CUBICSPLINE", count = input.times?.length;
  if (!Number.isSafeInteger(count) || count < (cubic ? 2 : 1) || count > 1048576)
    fail("LIMIT", "Invalid root-motion key count");
  const components = count * (cubic ? 10 : 4);
  if (components > maxComponents) fail("LIMIT", "Root-motion component budget exceeded");
  const times = numbers(input.times, count, "key times"),
    values = numbers(input.values, count * (cubic ? 9 : 3), "key values");
  for (let i = 0; i < count; i++)
    if (times[i] < 0 || times[i] > duration || (i && times[i] <= times[i - 1]))
      fail("KEYS", "Root-motion times must increase within the clip duration");
  const definition = Object.freeze({ format: input.format, duration, interpolation,
    times: Object.freeze(times), values: Object.freeze(values) });
  function sample(time) {
    finite(time, "local time");
    if (time < 0 || time > duration) fail("RANGE", "Local time exceeds clip duration");
    let lo = 0, hi = count - 1;
    if (time >= times[hi]) lo = hi;
    else while (lo + 1 < hi) {
      const mid = lo + Math.floor((hi - lo) / 2);
      if (times[mid] <= time) lo = mid; else hi = mid;
    }
    const stride = cubic ? 9 : 3, left = lo * stride + (cubic ? 3 : 0);
    if (interpolation === "STEP" || lo === count - 1 || time <= times[0] || time === times[lo])
      return values.slice(left, left + 3);
    const span = times[lo + 1] - times[lo], t = (time - times[lo]) / span, right = left + stride;
    const result = new Array(3);
    if (cubic) {
      const t2 = t * t, t3 = t2 * t, h00 = 2 * t3 - 3 * t2 + 1,
        h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
      for (let i = 0; i < 3; i++) result[i] = finite(
        h00 * values[left + i] + h10 * span * values[left + 3 + i] +
        h01 * values[right + i] + h11 * span * values[right - 3 + i], "sampled translation");
    } else for (let i = 0; i < 3; i++)
      result[i] = finite((1 - t) * values[left + i] + t * values[right + i], "sampled translation");
    return result;
  }
  const first = sample(0), last = sample(duration);
  function advance(start, end, wraps = 0, direction = 1) {
    if (!Number.isSafeInteger(wraps) || wraps < 0 || (direction !== 1 && direction !== -1))
      fail("ADVANCE", "Invalid repeat-seam count or direction");
    const a = sample(start), b = sample(end), result = new Array(3);
    for (let i = 0; i < 3; i++) {
      // Do not form a potentially overflowing stride for a non-wrapping segment.
      const seam = wraps === 0 ? 0 : finite((last[i] - first[i]) * wraps * direction, "repeat displacement");
      result[i] = finite(b[i] - a[i] + seam, "root displacement");
    }
    return result;
  }
  return Object.freeze({ definition, components, advance });
}

/** Separate one root's selected translation axes from a live player's copied
 * clip. The player, clip table, controls and current pose are unchanged. Install
 * result.clip with pose.addClips(), then attach result.rootMotion to its action.
 * Only a forest-root TRS node is admitted: local translation is then model-space
 * motion. Child joints, matrix nodes and rotation extraction require other paths.
 */
export function extractAnimationRootMotion(pose, options = {}) {
  fields(options, ["clip", "node", "axes", "name"], "extraction options");
  if (!pose || pose.disposed || typeof pose.snapshotClips !== "function" ||
      typeof pose.snapshotLocalPose !== "function") fail("POSE", "Expected a live animation player");
  const { clip: sourceClip = 0, node, axes: inputAxes = [true, false, true], name } = options;
  if (!Number.isSafeInteger(node) || node < 0 || node >= pose.nodeCount) fail("NODE", "Invalid root node");
  if (!Array.isArray(inputAxes) || inputAxes.length !== 3) fail("AXES", "Expected three boolean translation axes");
  const axes = [inputAxes[0], inputAxes[1], inputAxes[2]];
  if (axes.some(value => typeof value !== "boolean") || !axes.some(Boolean))
    fail("AXES", "Select at least one translation axis");
  if (name !== undefined && (typeof name !== "string" || name.length > 4096)) fail("VALUE", "Invalid clip name");
  const local = pose.snapshotLocalPose();
  if (local.format !== "f3d-local-pose-v1" || local.nodeCount !== pose.nodeCount ||
      local.parents[node] !== -1 || local.matrices.some(entry => entry.node === node))
    fail("NODE", "Root motion requires a forest-root TRS node");
  const snapshot = pose.snapshotClips([sourceClip]);
  const clip = snapshot.clips[0];
  const channel = clip.channels.find(item => item.node === node && item.path === "translation");
  if (!channel) fail("CHANNEL", "Selected clip has no translation channel on this root");
  const duration = pose.clips[sourceClip].duration;
  const cubic = channel.interpolation === "CUBICSPLINE", stride = cubic ? 9 : 3;
  const motionValues = channel.values.slice(), reference = channel.values.slice(cubic ? 3 : 0, cubic ? 6 : 3);
  for (let key = 0; key < channel.times.length; key++) {
    for (let slot = 0; slot < (cubic ? 3 : 1); slot++) for (let axis = 0; axis < 3; axis++) {
      const index = key * stride + slot * 3 + axis;
      if (axes[axis]) channel.values[index] = !cubic || slot === 1 ? reference[axis] : 0;
      else motionValues[index] = 0;
    }
  }
  clip.name = name ?? `${clip.name}-in-place`;
  const rootMotion = { format: "f3d-root-translation-v1", duration,
    interpolation: channel.interpolation, times: channel.times.slice(), values: motionValues };
  // Admit with the exact same track contract used by the action before returning
  // anything. Return caller-owned arrays, not the internal frozen validation copy.
  createAnimationRootMotionTrack(rootMotion, duration);
  return Object.freeze({ format: "f3d-root-motion-clip-v1", sourceClip, node,
    axes: Object.freeze(axes), clip, rootMotion });
}

/** Post-compose model-space displacement with an affine scene placement. The
 * linear part (rotation/scale/shear) stays unchanged; no heading is inferred.
 * Returns a frozen independent ordinary array suitable for pose rootMatrix.
 */
export function applyAnimationRootMotion(rootMatrix, displacement) {
  const matrix = rootMatrix == null ? identity() : numbers(rootMatrix, 16, "root matrix");
  const delta = numbers(displacement, 3, "root displacement");
  if (matrix[3] !== 0 || matrix[7] !== 0 || matrix[11] !== 0 || matrix[15] !== 1)
    fail("MATRIX", "Root matrix must be affine");
  for (let row = 0; row < 3; row++) matrix[12 + row] = finite(
    matrix[12 + row] + matrix[row] * delta[0] + matrix[4 + row] * delta[1] + matrix[8 + row] * delta[2],
    "placed translation");
  return Object.freeze(matrix);
}
