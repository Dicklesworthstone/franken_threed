/** Explicit, bounded recording of committed LOCAL poses into reusable clips.
 * Capture after the caller's controller/retarget/IK updates. No clock, sampling,
 * GPU readback or mutations of the borrowed pose. Tracks use pose node IDs.
 */
export class AnimationRecordingError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationRecordingError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationRecordingError("ANIMATION_RECORD_" + code, message);
};
function fields(value, allowed, label) {
  if (!value || typeof value !== "object" || Array.isArray(value))
    fail("OPTIONS", `Expected ${label} object`);
  for (const key of Object.keys(value))
    if (!allowed.includes(key)) fail("OPTIONS", `Unknown ${label} field: ${key}`);
}
function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    fail("LIMIT", `Invalid ${label}`);
  return value;
}
function finite(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value))
    fail("VALUE", `Expected finite ${label}`);
  return value;
}
function abort(signal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
}
function storage(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) ||
      value instanceof DataView || value.length !== length)
    fail("SHAPE", `Invalid snapshot ${label}`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable)
      fail("STORAGE", "Snapshot arrays must have fixed unshared storage");
    try { new Uint8Array(value.buffer, 0, 0); }
    catch { fail("STORAGE", "Detached snapshot storage"); }
  }
  return value;
}
const paths = ["translation", "rotation", "scale", "weights"];
const poseFields = ["translations", "rotations", "scales", "morphWeights"];

/** tracks: nonempty explicit [{node, path}], with no duplicate bindings.
 * capture(time, {signal}) uses a caller's monotonically increasing clock; its
 * first successful capture becomes time zero. Times/values round to Float32
 * immediately, so time collapse/overflow fail before accepting a frame.
 * finish() seals the recorder and returns {clip, frameCount, components, ...}.
 * The returned clip owns ordinary number arrays, suitable for the player or
 * exportAnimationGLB([clip]). Captured frames can be finished after pose disposal.
 * Only selected local TRS/morph tracks are recorded. Matrix animation, external
 * rootMatrix placement, events and effects between captures are NOT recorded.
 * LINEAR/STEP output approximates motion between samples; no cubic tangents are
 * inferred. Quaternion signs stay continuous, without normalizing bad inputs.
 * maxComponents counts emitted numbers, including repeated per-channel times.
 * Snapshot, retained frames and final arrays are separately bounded stages, not
 * one aggregate memory limit. Calls are synchronous; no abort-event yield occurs.
 */
export function createAnimationRecorder(pose, options = {}) {
  fields(options, ["tracks", "name", "interpolation", "maxFrames", "maxComponents",
    "maxNodes", "maxTracks"], "recorder options");
  const {tracks: inputTracks, name = "recording", interpolation = "LINEAR",
    maxFrames = 6000, maxComponents = 16777216, maxNodes = 65536, maxTracks = 4096} = options;
  integer(maxFrames, 1, 1048576, "frame limit");
  integer(maxComponents, 1, 16777216, "component limit");
  integer(maxNodes, 1, 65536, "node limit");
  integer(maxTracks, 1, 32768, "track limit");
  if (typeof name !== "string" || name.length > 4096)
    fail("OPTIONS", "Name must be text of at most 4096 characters");
  if (!["LINEAR", "STEP"].includes(interpolation))
    fail("OPTIONS", "Recording interpolation must be LINEAR or STEP");
  if (!Array.isArray(inputTracks) || !inputTracks.length || inputTracks.length > maxTracks)
    fail("LIMIT", "Supply a bounded nonempty track list");
  let busy = false, disposed = false, finished = false, frames = [];
  let frameCount = 0, firstTime = null, lastTime = null, duration = 0;
  function livePose() {
    if (!pose || pose.disposed || typeof pose.snapshotLocalPose !== "function")
      fail("POSE", "Expected a live snapshot-capable pose");
  }
  livePose();
  const nodeCount = integer(pose.nodeCount, 1, maxNodes, "pose node count");
  function snapshot() {
    livePose();
    const version = integer(pose.version, 0, Number.MAX_SAFE_INTEGER, "pose version");
    const result = pose.snapshotLocalPose();
    livePose();
    if (pose.version !== version || result?.version !== version || pose.nodeCount !== nodeCount)
      fail("STALE", "Pose changed during capture");
    return result;
  }
  function layout(state) {
    if (state?.format !== "f3d-local-pose-v1" || state.nodeCount !== nodeCount)
      fail("SHAPE", "Expected a matching local-pose snapshot");
    const parents = storage(state.parents, nodeCount, "parents");
    const offsets = storage(state.morphOffsets, nodeCount + 1, "morph offsets");
    if (offsets[0] !== 0) fail("SHAPE", "Morph offsets must start at zero");
    for (let i = 0; i < nodeCount; i++) {
      integer(parents[i], -1, nodeCount - 1, "parent");
      if (parents[i] === i) fail("SHAPE", "Self-parented pose node");
      integer(offsets[i + 1], offsets[i], Math.min(offsets[i] + 4096, 1048576), "morph extent");
    }
    for (let i = 0; i < 4; i++)
      storage(state[poseFields[i]], i === 3 ? offsets[nodeCount] : nodeCount * (i === 1 ? 4 : 3), poseFields[i]);
    if (!Array.isArray(state.matrices) || state.matrices.length > nodeCount)
      fail("SHAPE", "Invalid matrix-node table");
    const matrices = new Set();
    for (const entry of state.matrices) {
      const node = integer(entry?.node, 0, nodeCount - 1, "matrix node");
      if (matrices.has(node)) fail("SHAPE", "Duplicate matrix node");
      storage(entry.matrix, 16, "matrix");
      matrices.add(node);
    }
    return {parents, offsets, matrices};
  }
  const initial = layout(snapshot());
  // Retain only topology, not the initial pose's potentially large value arrays.
  let parents = Array.from(initial.parents), offsets = Array.from(initial.offsets);
  let matrixNodes = initial.matrices;
  const seen = new Set();
  let width = 0;
  const tracks = Array.from(inputTracks, (input) => {
    fields(input, ["node", "path"], "track");
    const node = integer(input.node, 0, nodeCount - 1, "track node"), path = input.path;
    const kind = paths.indexOf(path), key = `${node}:${path}`;
    if (kind < 0 || seen.has(key)) fail("TRACK", "Unknown or duplicate track binding");
    seen.add(key);
    if (path !== "weights" && matrixNodes.has(node))
      fail("TRACK", "Matrix nodes cannot be recorded as TRS channels");
    const count = path === "weights" ? offsets[node + 1] - offsets[node] : path === "rotation" ? 4 : 3;
    if (!count) fail("TRACK", "Weight track targets a node without morph weights");
    const result = {node, path, field: poseFields[kind], width: count,
      source: kind === 3 ? offsets[node] : node * count, offset: width};
    width += count;
    return result;
  });
  const perFrame = width + tracks.length;
  if (perFrame > maxComponents) fail("LIMIT", "A single frame exceeds the component limit");
  function run(operation) {
    if (disposed) fail("DISPOSED", "Recorder is disposed");
    if (finished) fail("FINISHED", "Recorder is already finished");
    if (busy) fail("REENTRANT", "Recorder operations cannot be reentered");
    busy = true;
    try { return operation(); } finally { busy = false; }
  }
  function release() {
    frames = [];
    pose = parents = offsets = matrixNodes = null;
  }
  const recorder = Object.freeze({
    capture(time, settings = {}) {
      return run(() => {
        fields(settings, ["signal"], "capture settings");
        const signal = settings.signal;
        abort(signal);
        finite(time, "capture time");
        const relative = Math.fround(firstTime === null ? 0 : time - firstTime);
        if ((lastTime !== null && time <= lastTime) || !Number.isFinite(relative) ||
            (frameCount && relative <= duration))
          fail("TIME", "Capture times must remain strictly increasing as rebased Float32 keys");
        if (frameCount >= maxFrames || (frameCount + 1) * perFrame > maxComponents)
          fail("LIMIT", "Recording frame/component budget exceeded");
        const state = snapshot(), current = layout(state);
        if (parents.some((p, i) => p !== current.parents[i]) ||
            offsets.some((v, i) => v !== current.offsets[i]) ||
            matrixNodes.size !== current.matrices.size ||
            [...matrixNodes].some((node) => !current.matrices.has(node)))
          fail("RIG", "Pose hierarchy or channel layout changed");
        const values = new Float32Array(width);
        for (const track of tracks) {
          for (let i = 0; i < track.width; i++) {
            const value = Math.fround(finite(state[track.field][track.source + i], track.path));
            if (!Number.isFinite(value)) fail("VALUE", "Recorded value overflows Float32");
            values[track.offset + i] = value;
          }
          if (track.path === "rotation") {
            const o = track.offset;
            if (Math.abs(Math.hypot(values[o], values[o + 1], values[o + 2], values[o + 3]) - 1) > 1e-3)
              fail("ROTATION", "Recorded rotations must be unit quaternions");
            if (frameCount) {
              const previous = frames[frameCount - 1].values;
              let dot = 0;
              for (let i = 0; i < 4; i++) dot += previous[o + i] * values[o + i];
              if (dot < 0) for (let i = 0; i < 4; i++) values[o + i] = -values[o + i];
            }
          }
        }
        abort(signal);
        livePose();
        if (pose.version !== state.version || pose.nodeCount !== nodeCount)
          fail("STALE", "Pose changed before frame publication");
        // A failed later track/time/budget check cannot leave a partial frame.
        frames.push({time: relative, values});
        firstTime ??= time;
        lastTime = time;
        duration = relative;
        frameCount++;
        return recorder;
      });
    },
    finish(settings = {}) {
      return run(() => {
        fields(settings, ["signal"], "finish settings");
        const signal = settings.signal;
        abort(signal);
        if (!frameCount) fail("EMPTY", "Capture at least one frame before finishing");
        const channels = tracks.map((track) => {
          const times = new Array(frameCount), values = new Array(frameCount * track.width);
          for (let frame = 0; frame < frameCount; frame++) {
            times[frame] = frames[frame].time;
            for (let i = 0; i < track.width; i++)
              values[frame * track.width + i] = frames[frame].values[track.offset + i];
          }
          return {node: track.node, path: track.path, interpolation, times, values};
        });
        abort(signal);
        const result = Object.freeze({clip: {name, channels}, frameCount,
          components: frameCount * perFrame, duration,
          sourceRange: Object.freeze([firstTime, lastTime]), interpolation,
          approximate: true, execution: "javascript-cpu-pose-recording", accelerationClaim: false});
        finished = true;
        release();
        return result;
      });
    },
    get frameCount() { return frameCount; },
    get components() { return frameCount * perFrame; },
    get componentsPerFrame() { return perFrame; },
    get duration() { return duration; },
    get finished() { return finished; },
    get disposed() { return disposed; },
    dispose() {
      if (busy) fail("REENTRANT", "Cannot dispose during a recorder operation");
      disposed = true;
      release();
    },
  });
  return recorder;
}
