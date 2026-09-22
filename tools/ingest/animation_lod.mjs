/** Distance-based selection of whole drawable groups at the current pose.
 * This is the explicit scene route, not an Object3D/THREE.LOD adapter. No clock,
 * geometry generation, asset loading, scene mutation or GPU work is performed.
 * Each camera key owns independent, bounded hysteresis state. prepare() is pure
 * with respect to that state; commit() follows successful scene submission.
 */
export class AnimationLodError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationLodError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationLodError(`ANIMATION_LOD_${code}`, message);
};
const object = (v, label) => {
  if (!v || typeof v !== "object" || Array.isArray(v)) fail("OPTIONS", `Expected ${label}`);
  return v;
};
const keys = (v, allowed) => {
  for (const key of Object.keys(v))
    if (!allowed.includes(key)) fail("OPTIONS", `Unknown option: ${key}`);
};
const integer = (v, min, max, label) => {
  if (!Number.isSafeInteger(v) || v < min || v > max) fail("LIMIT", `Invalid ${label}`);
  return v;
};
const finite = (v, label) => {
  if (typeof v !== "number" || !Number.isFinite(v)) fail("VALUE", `${label} must be finite`);
  return v;
};
function storage(value, length, label) {
  if (
    !ArrayBuffer.isView(value) ||
    value instanceof DataView ||
    value.length !== length ||
    !(value.buffer instanceof ArrayBuffer) ||
    value.buffer.resizable
  )
    fail("STORAGE", `Invalid ${label} storage`);
  try {
    new Uint8Array(value.buffer, 0, 0);
  } catch {
    fail("STORAGE", `${label} is detached`);
  }
  return value;
}
function cameraKey(value) {
  if (
    (typeof value === "string" && value.length > 0 && value.length <= 128) ||
    (typeof value === "number" && Number.isSafeInteger(value))
  )
    return value;
  fail("CAMERA", "Camera key must be a nonempty bounded string or safe integer");
}

/**
 * groups: [{node, levels:[{distance:0,drawIndices:[...]}, ...]}]
 * Levels are explicitly ordered by strictly increasing nonnegative distances.
 * Every level contains one or more draws; a draw belongs to at most one level
 * across ALL groups. Ungrouped draws remain selected. The group node's current
 * world translation is the distance origin, not a guessed primitive centroid.
 * Per-level hysteresis is a fraction in [0,1): the current coarse level persists
 * down to distance * (1-hysteresis). Camera zoom divides world-space distance.
 * Similar threshold equations to pinned Three.js r186 LOD.update, but camera
 * histories are deliberately isolated and Object3D visibility is never mutated.
 * maxCameras bounds retained histories (default 8, hard maximum 64); call
 * resetCamera(key) before reusing capacity. There is no hidden eviction.
 */
export function createAnimationLod(pose, drawCount, options) {
  integer(drawCount, 1, 4096, "draw count");
  const nodeCount = integer(pose?.nodeCount, 1, 65536, "pose node count");
  object(options, "LOD options");
  keys(options, ["groups", "maxCameras"]);
  const maxCameras = integer(options.maxCameras ?? 8, 1, 64, "maxCameras");
  const input = options.groups;
  if (!Array.isArray(input) || input.length < 1 || input.length > drawCount)
    fail("GROUP", "Expected a bounded nonempty group list");
  const membership = new Int32Array(drawCount).fill(-1);
  let levelCount = 0,
    groupedDraws = 0;
  const groups = input.map((source, groupIndex) => {
    object(source, "LOD group");
    keys(source, ["node", "levels"]);
    const node = integer(source.node, 0, nodeCount - 1, "group node"),
      levels = source.levels;
    if (!Array.isArray(levels) || !levels.length || levels.length > drawCount - levelCount)
      fail("GROUP", "Invalid aggregate level count");
    levelCount += levels.length;
    let previous = -1;
    const copied = levels.map((level, at) => {
      object(level, "LOD level");
      keys(level, ["distance", "hysteresis", "drawIndices"]);
      const distance = finite(level.distance, "Level distance"),
        hysteresis = finite(level.hysteresis ?? 0, "Hysteresis");
      if (
        distance < 0 ||
        distance <= previous ||
        (at === 0 && distance !== 0) ||
        hysteresis < 0 ||
        hysteresis >= 1
      )
        fail(
          "LEVEL",
          "Levels require distance zero first, increasing distances and hysteresis in [0,1)",
        );
      previous = distance;
      const indices = level.drawIndices;
      if (!Array.isArray(indices) || !indices.length || indices.length > drawCount - groupedDraws)
        fail("DRAW", "Expected bounded nonempty draw indices");
      const drawIndices = indices.map((index) => {
        integer(index, 0, drawCount - 1, "draw index");
        if (membership[index] !== -1)
          fail("DRAW", "A draw cannot belong to multiple levels or groups");
        membership[index] = groupIndex;
        groupedDraws++;
        return index;
      });
      return Object.freeze({ distance, hysteresis, drawIndices: Object.freeze(drawIndices) });
    });
    return Object.freeze({ node, levels: Object.freeze(copied) });
  });
  Object.freeze(groups);
  const histories = new Map(),
    pending = new WeakMap();
  let disposed = false,
    busy = false,
    epoch = {},
    lastSelection = null;
  function live() {
    if (disposed) fail("DISPOSED", "LOD selection has been disposed");
    if (
      pose.disposed ||
      pose.nodeCount !== nodeCount ||
      !Number.isSafeInteger(pose.version) ||
      pose.version < 0
    )
      fail("POSE", "Expected a live unchanged pose layout");
  }
  function exclusive(operation) {
    if (busy) fail("REENTRANT", "LOD operations cannot be reentered");
    busy = true;
    try {
      live();
      return operation();
    } finally {
      busy = false;
    }
  }
  // Check initial storage/layout without retaining a second copy of pose data.
  live();
  storage(pose.worldMatrices, nodeCount * 16, "world matrix");
  const result = Object.freeze({
    groups,
    maxCameras,
    get cameraCount() {
      return histories.size;
    },
    get lastSelection() {
      return lastSelection;
    },
    get disposed() {
      return disposed;
    },
    prepare(camera) {
      return exclusive(() => {
        const poseVersion = pose.version;
        object(camera, "LOD camera");
        keys(camera, ["position", "zoom", "key"]);
        const key = cameraKey(camera.key ?? "default"),
          zoom = finite(camera.zoom ?? 1, "Camera zoom");
        if (zoom <= 0) fail("CAMERA", "Camera zoom must be positive");
        const inputPosition = camera.position;
        if (
          (!Array.isArray(inputPosition) && !ArrayBuffer.isView(inputPosition)) ||
          inputPosition.length !== 3
        )
          fail("CAMERA", "Camera position requires XYZ");
        if (ArrayBuffer.isView(inputPosition)) storage(inputPosition, 3, "camera position");
        const position = Array.from(inputPosition, (v) => finite(v, "Camera position"));
        const previous = histories.get(key);
        if (!previous && histories.size === maxCameras)
          fail("CAMERA_LIMIT", "Reset an unused camera before adding another LOD history");
        const world = storage(pose.worldMatrices, nodeCount * 16, "world matrix");
        const state = new Uint32Array(groups.length),
          selected = [],
          active = new Uint8Array(drawCount);
        for (let i = 0; i < drawCount; i++) if (membership[i] === -1) active[i] = 1;
        for (let i = 0; i < groups.length; i++) {
          const group = groups[i],
            offset = group.node * 16;
          for (let j = 0; j < 16; j++) finite(world[offset + j], "Group world matrix");
          if (
            world[offset + 3] !== 0 ||
            world[offset + 7] !== 0 ||
            world[offset + 11] !== 0 ||
            world[offset + 15] !== 1
          )
            fail("MATRIX", "Group world matrix must be affine");
          const distance = finite(
            Math.hypot(
              position[0] - world[offset + 12],
              position[1] - world[offset + 13],
              position[2] - world[offset + 14],
            ) / zoom,
            "LOD distance",
          );
          let level = 0;
          for (let j = 1; j < group.levels.length; j++) {
            const candidate = group.levels[j];
            const threshold =
              candidate.distance * (previous?.[i] === j ? 1 - candidate.hysteresis : 1);
            if (distance < threshold) break;
            level = j;
          }
          state[i] = level;
          for (const index of group.levels[level].drawIndices) active[index] = 1;
          selected.push(Object.freeze({ group: i, node: group.node, level, distance }));
        }
        live();
        if (pose.version !== poseVersion || pose.worldMatrices !== world)
          fail("CHANGED", "Pose changed while selecting levels");
        const drawIndices = Object.freeze(Array.from(active.keys()).filter((i) => active[i] !== 0));
        const selection = Object.freeze({
          poseVersion,
          cameraKey: key,
          cameraPosition: Object.freeze(position),
          zoom,
          groups: Object.freeze(selected),
          drawIndices,
          selectedDraws: drawIndices.length,
          suppressedDraws: drawCount - drawIndices.length,
        });
        pending.set(selection, { epoch, state, key, poseVersion });
        return selection;
      });
    },
    commit(selection) {
      return exclusive(() => {
        const candidate = pending.get(selection);
        if (!candidate || candidate.epoch !== epoch)
          fail("TRANSACTION", "Selection is foreign or no longer current");
        if (candidate.poseVersion !== pose.version)
          fail("CHANGED", "Upload/select the new pose before committing LOD");
        histories.set(candidate.key, candidate.state);
        lastSelection = selection;
        epoch = {};
        pending.delete(selection);
        return result;
      });
    },
    resetCamera(key = "default") {
      return exclusive(() => {
        key = cameraKey(key);
        histories.delete(key);
        epoch = {};
        return result;
      });
    },
    dispose() {
      if (busy) fail("REENTRANT", "Cannot dispose during LOD selection");
      if (!disposed) {
        disposed = true;
        histories.clear();
        lastSelection = null;
        epoch = {};
      }
    },
  });
  return result;
}
