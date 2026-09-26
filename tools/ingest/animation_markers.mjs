/** Timed cue traversal for the packed controller. This module never advances a
 * clock or samples a pose. It consumes the controller's already-computed local
 * endpoints and traversal count, including a separate segment for each reversal.
 * No user callbacks, I/O, renderer initialization or mutable input retention.
 */
export class AnimationMarkerError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationMarkerError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationMarkerError("ANIMATION_MARKER_" + code, message);
};
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const count = (v) => Number.isSafeInteger(v) && v >= 0;

export function animationMarkerEventLimit(value) {
  if (!count(value) || value < 1 || value > 65536)
    fail("LIMIT", "maxMarkerEvents must be an integer in [1,65536]");
  return value;
}

/** Controller support API. Copy and sort explicit {name,time} records. Times
 * are local clip seconds in [0,duration]. Ties preserve input order in BOTH
 * playback directions; duplicate names/times denote distinct cues, not a set.
 * append() is an internal bounded traversal operation, not a second player.
 */
export function createAnimationMarkerTrack(input, duration) {
  if (!finite(duration) || duration < 0) fail("VALUE", "Invalid clip duration");
  if (!Array.isArray(input) || input.length > 4096)
    fail("LIMIT", "Expected at most 4096 markers per action");
  // Indexed reads (not the caller's iterator/map method) also reject sparse lists.
  const copied = [];
  const length = input.length;
  for (let i = 0; i < length; i++) {
    const item = input[i];
    if (!item || typeof item !== "object" || Array.isArray(item) ||
        Object.keys(item).some((key) => key !== "name" && key !== "time"))
      fail("SHAPE", "Expected a marker with name and time only");
    const name = item.name, time = item.time;
    if (typeof name !== "string" || name.length < 1 || name.length > 256)
      fail("VALUE", "Marker names must contain 1..256 characters");
    if (!finite(time) || time < 0 || time > duration)
      fail("VALUE", "Marker time must be within the clip duration");
    copied.push(Object.freeze({ name, time }));
  }
  // Sorting never mutates input; stable ties are the original declaration order.
  const forward = Object.freeze(copied.slice().sort((a, b) => a.time - b.time));
  const backward = Object.freeze(copied.slice().sort((a, b) => b.time - a.time));
  function bound(list, time, direction, inclusive) {
    let lo = 0, hi = list.length;
    while (lo < hi) {
      const mid = lo + Math.floor((hi - lo) / 2);
      const difference = direction * (list[mid].time - time);
      if (difference < 0 || (inclusive && difference === 0)) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  const atZero = bound(forward, 0, 1, true);
  const atEnd = forward.length - bound(forward, duration, 1, false);
  function range(start, end, direction, includeStart = false) {
    const list = direction > 0 ? forward : backward;
    return [list, bound(list, start, direction, !includeStart), bound(list, end, direction, true)];
  }
  const size = (r) => Math.max(0, r[2] - r[1]);
  function append(output, action, movement, maxEvents) {
    animationMarkerEventLimit(maxEvents);
    if (!Array.isArray(output) || output.length > maxEvents)
      fail("LIMIT", "Invalid pending marker-event storage");
    const { start, end, direction, loop, crossings, finished, traversal } = movement;
    if (!finite(start) || !finite(end) || start < 0 || start > duration || end < 0 || end > duration ||
        (direction !== 1 && direction !== -1) || !["repeat", "pingpong", "once"].includes(loop) ||
        !count(crossings) || !count(traversal) || !count(traversal + crossings) ||
        typeof finished !== "boolean" || (finished && crossings === 0))
      fail("ADVANCE", "Invalid controller traversal");
    if (!forward.length) return;
    const remaining = maxEvents - output.length;
    const admit = (n) => {
      if (!count(n) || n > remaining)
        fail("LIMIT", "Marker-event budget exceeded; retry a smaller update or raise maxMarkerEvents");
    };
    function emit(r, leg, dir) {
      for (let i = r[1]; i < r[2]; i++) {
        const marker = r[0][i];
        output.push(Object.freeze({
          type: "marker", action, name: marker.name, time: marker.time,
          direction: dir, traversal: leg,
        }));
      }
    }
    if (duration === 0) {
      // A zero-duration action's one natural completion is its only cue point.
      if (!finished) return;
      admit(forward.length);
      emit([forward, 0, forward.length], traversal, direction);
      return;
    }
    if (crossings === 0) {
      if (direction * (end - start) < 0) fail("ADVANCE", "Local segment reverses without a crossing");
      const partial = range(start, end, direction);
      admit(size(partial));
      emit(partial, traversal, direction);
      return;
    }
    const pingpong = loop === "pingpong";
    const exit = direction > 0 ? duration : 0;
    const first = range(start, exit, direction);
    const middle = crossings - 1;
    const finalDirection = pingpong && crossings % 2 ? -direction : direction;
    const entry = finalDirection > 0 ? 0 : duration;
    const last = finished ? null : range(entry, end, finalDirection, !pingpong);
    let total = size(first) + (last ? size(last) : 0);
    if (pingpong) {
      const positive = forward.length - atZero;
      const negative = forward.length - atEnd;
      total += Math.floor(middle / 2) * (positive + negative) +
        (middle % 2 ? (direction > 0 ? negative : positive) : 0);
    } else total += middle * forward.length;
    // Preflight is arithmetic, not one iteration per crossed loop. Huge jumps
    // cannot allocate an unbounded event list or stall on empty ping-pong legs.
    admit(total);
    emit(first, traversal, direction);
    let dir = direction;
    for (let leg = 1; leg <= middle; leg++) {
      if (pingpong) dir = -dir;
      const full = range(dir > 0 ? 0 : duration, dir > 0 ? duration : 0, dir, !pingpong);
      emit(full, traversal + leg, dir);
    }
    if (last) emit(last, traversal + crossings, finalDirection);
  }
  return Object.freeze({ markers: forward, append });
}
