/**
 * Explicit playback control for packed animation poses. Borrows a player;
 * controller.dispose() does not dispose that player. No timers or callbacks.
 *
 * createAction(clip, options) makes an independent action, even for the same
 * clip. play/stop/seek/configuration take effect on the next update(delta).
 * delta is finite nonnegative wall time in seconds; a signed action timeScale
 * controls playback direction. pause freezes its playhead, not its fade.
 * Fades change the action weight; play/stop/reset do not restore that weight.
 *
 * repetitions counts traversals (one ping-pong leg is one traversal), not
 * round trips. seek resets that count and the ping-pong orientation. A finite
 * action stops at its last endpoint; clampWhenFinished keeps contributing it.
 * Zero-duration actions finish once on their first nonzero playback advance.
 *
 * events is the immutable event list for the last SUCCESSFUL update. Loop
 * records aggregate multiple crossings, in action creation order, followed by
 * that action's finish record. Loop direction is the timeScale sign; finished
 * direction is the final local clip direction. Consume events each update.
 * They are not
 * callbacks and are not an unbounded queue. Failed updates change neither
 * clocks, fades, events nor the published pose. No per-action pose is copied.
 *
 * This is not a Three.js AnimationMixer adapter: it controls imported TRS and
 * morph tracks, not arbitrary PropertyBindings or preconverted additive clips.
 * Additive layers use player.blend's imported-rest-relative convention.
 */
import { AnimationPoseError } from "./animation_runtime.mjs";

const fail = (code, message) => {
  throw new AnimationPoseError(code, message);
};
const finite = (value, label) => {
  if (typeof value !== "number" || !Number.isFinite(value))
    fail("ANIMATION_ACTION_VALUE", `${label} must be finite`);
  return value;
};
const nonnegative = (value, label) => {
  finite(value, label);
  if (value < 0) fail("ANIMATION_ACTION_VALUE", `${label} must be nonnegative`);
  return value;
};
const boolean = (value, label) => {
  if (typeof value !== "boolean") fail("ANIMATION_ACTION_VALUE", `${label} must be boolean`);
  return value;
};
const blendMode = (value) => {
  if (value !== "normal" && value !== "additive")
    fail("ANIMATION_BLEND_MODE", "Unknown blend mode");
  return value;
};
function loopOptions(loop, repetitions) {
  if (!["once", "repeat", "pingpong"].includes(loop))
    fail("ANIMATION_ACTION_LOOP", "Unknown loop mode");
  if (repetitions !== Infinity && (!Number.isSafeInteger(repetitions) || repetitions < 1))
    fail("ANIMATION_ACTION_LOOP", "Repetitions must be a positive safe integer or Infinity");
  return { loop, repetitions: loop === "once" ? 1 : repetitions };
}
const EMPTY = Object.freeze([]);

export function createAnimationController(pose) {
  if (
    !pose ||
    typeof pose.blend !== "function" ||
    !Array.isArray(pose.clips) ||
    !Number.isSafeInteger(pose.nodeCount) ||
    pose.nodeCount < 0
  ) {
    fail("ANIMATION_ACTION_PLAYER", "Expected an animation pose player");
  }
  const records = [],
    owned = new WeakMap(),
    layers = [],
    pendingEvents = [];
  let time = 0,
    events = EMPTY,
    busy = false,
    disposed = false;
  function exclusive(operation, ...args) {
    if (disposed) fail("ANIMATION_CONTROLLER_DISPOSED", "Animation controller has been disposed");
    if (busy) fail("ANIMATION_REENTRANT", "Animation control cannot be reentered");
    if (pose.disposed) fail("ANIMATION_DISPOSED", "Animation player has been disposed");
    busy = true;
    try {
      return operation(...args);
    } finally {
      busy = false;
    }
  }
  function get(action) {
    const record = owned.get(action);
    if (!record || record.disposed)
      fail("ANIMATION_ACTION_DISPOSED", "Action is not live in this controller");
    return record;
  }
  function copyMask(mask) {
    if (mask === null) return null;
    if ((!Array.isArray(mask) && !ArrayBuffer.isView(mask)) || mask.length !== pose.nodeCount)
      fail("ANIMATION_MASK", "Mask must have nodeCount weights");
    const copied = new Float64Array(pose.nodeCount);
    for (let i = 0; i < copied.length; i++) {
      const value = nonnegative(mask[i], "Mask weight");
      if (value > 1) fail("ANIMATION_MASK", "Mask weights must be in [0,1]");
      copied[i] = value;
    }
    return copied;
  }
  const active = (state) => state.playing || (state.finished && state.clamp);
  function resetState(state) {
    state.time = state.speed < 0 ? state.duration : 0;
    state.completed = 0;
    state.orientation = 1;
    state.finished = false;
    state.paused = false;
    state.fadeDuration = 0;
    state.stopAfterFade = false;
  }
  function start(state) {
    if (state.finished) resetState(state);
    state.playing = true;
    state.paused = false;
  }
  function stop(state) {
    resetState(state);
    state.playing = false;
  }
  function fade(state, target, duration, stopWhenDone) {
    state.fadeFrom = state.weight;
    state.fadeTo = target;
    state.fadeDuration = duration;
    state.fadeElapsed = 0;
    state.stopAfterFade = stopWhenDone;
    if (duration === 0) {
      state.weight = target;
      if (stopWhenDone) stop(state);
    }
  }
  function createAction(clip, options = {}) {
    if (!Number.isInteger(clip) || clip < 0 || clip >= pose.clips.length)
      fail("ANIMATION_INDEX", "Invalid action clip");
    if (records.length >= 256)
      fail("ANIMATION_ACTION_LIMIT", "At most 256 live actions are supported");
    const {
      loop = "repeat",
      repetitions = Infinity,
      weight = 1,
      timeScale = 1,
      clampWhenFinished = false,
      mode = "normal",
      mask = null,
    } = options;
    const timing = loopOptions(loop, repetitions),
      duration = nonnegative(pose.clips[clip].duration, "Clip duration");
    const state = {
      duration,
      ...timing,
      weight: nonnegative(weight, "Weight"),
      speed: finite(timeScale, "Time scale"),
      clamp: boolean(clampWhenFinished, "clampWhenFinished"),
      time: 0,
      orientation: 1,
      completed: 0,
      playing: false,
      paused: false,
      finished: false,
      fadeFrom: 0,
      fadeTo: 0,
      fadeDuration: 0,
      fadeElapsed: 0,
      stopAfterFade: false,
    };
    resetState(state);
    const layer = {
      clip,
      time: 0,
      weight: 0,
      loop: false,
      mode: blendMode(mode),
      mask: copyMask(mask),
    };
    const record = { state, next: { ...state }, layer, disposed: false, action: null };
    const mutate = (operation) =>
      exclusive(() => {
        get(action);
        operation();
        return action;
      });
    const action = Object.freeze({
      clip,
      play() {
        return mutate(() => start(state));
      },
      stop() {
        return mutate(() => stop(state));
      },
      pause() {
        return mutate(() => {
          state.paused = true;
        });
      },
      reset() {
        return mutate(() => {
          resetState(state);
        });
      },
      seek(value) {
        return mutate(() => {
          nonnegative(value, "Seek time");
          if (value > duration) fail("ANIMATION_ACTION_VALUE", "Seek time exceeds clip duration");
          state.time = value;
          state.completed = 0;
          state.orientation = 1;
          state.finished = false;
        });
      },
      setWeight(value) {
        return mutate(() => {
          state.weight = nonnegative(value, "Weight");
          state.fadeDuration = 0;
          state.stopAfterFade = false;
        });
      },
      setTimeScale(value) {
        return mutate(() => {
          state.speed = finite(value, "Time scale");
        });
      },
      setLoop(value, count = Infinity) {
        return mutate(() => {
          const next = loopOptions(value, count);
          Object.assign(state, next);
          state.completed = 0;
          state.orientation = 1;
        });
      },
      setClampWhenFinished(value) {
        return mutate(() => {
          state.clamp = boolean(value, "clampWhenFinished");
        });
      },
      setBlendMode(value) {
        return mutate(() => {
          layer.mode = blendMode(value);
        });
      },
      setMask(value) {
        return mutate(() => {
          layer.mask = copyMask(value);
        });
      },
      fadeTo(target, seconds, options = {}) {
        return mutate(() => {
          const { stopWhenDone = false } = options;
          nonnegative(target, "Fade weight");
          nonnegative(seconds, "Fade duration");
          boolean(stopWhenDone, "stopWhenDone");
          fade(state, target, seconds, stopWhenDone);
        });
      },
      dispose() {
        return mutate(() => {
          record.disposed = true;
          records.splice(records.indexOf(record), 1);
        });
      },
      get time() {
        return state.time;
      },
      get weight() {
        return state.weight;
      },
      get timeScale() {
        return state.speed;
      },
      get playing() {
        return state.playing;
      },
      get paused() {
        return state.paused;
      },
      get finished() {
        return state.finished;
      },
      get completedTraversals() {
        return state.completed;
      },
      get loop() {
        return state.loop;
      },
      get repetitions() {
        return state.repetitions;
      },
      get effectiveWeight() {
        return active(state) ? state.weight : 0;
      },
      get disposed() {
        return record.disposed || disposed;
      },
    });
    record.action = action;
    owned.set(action, record);
    records.push(record);
    return action;
  }
  function advance(state, delta) {
    state.loopDelta = 0;
    state.endEvent = false;
    state.eventDirection = 1;
    if (!state.playing || state.paused || delta === 0 || state.speed === 0) return;
    const direction = Math.sign(state.speed) * state.orientation;
    state.eventDirection = direction;
    const finish = () => {
      state.playing = false;
      state.finished = true;
      state.endEvent = true;
    };
    if (state.duration === 0) {
      state.completed = 1;
      finish();
      return;
    }
    const distance = nonnegative(delta * Math.abs(state.speed), "Playback advance");
    if (distance === 0) return;
    const first = direction > 0 ? state.duration - state.time : state.time;
    if (distance < first) {
      state.time += direction * distance;
      return;
    }
    const remaining = distance - first,
      available = state.repetitions - state.completed;
    const crossings = 1 + Math.floor(remaining / state.duration);
    // A huge jump may be clamped by a finite action before it needs an exact
    // infinite loop count. Never wrap counts through a bitwise integer cast.
    const count = Math.min(crossings, available);
    if (!Number.isSafeInteger(count) || !Number.isSafeInteger(state.completed + count))
      fail("ANIMATION_ACTION_LIMIT", "Traversal count exceeds safe integer precision");
    state.completed += count;
    const odd = count % 2 === 1;
    const end = state.loop === "pingpong" ? (direction > 0 ? odd : !odd) : direction > 0;
    state.time = end ? state.duration : 0;
    if (state.loop === "pingpong") {
      state.eventDirection = odd ? direction : -direction;
      if (odd) state.orientation = -state.orientation;
    }
    if (count === available) {
      state.loopDelta = count - 1;
      finish();
      return;
    }
    state.loopDelta = count;
    const remainder = remaining % state.duration;
    if (state.loop === "pingpong") state.time += (odd ? -direction : direction) * remainder;
    else state.time = (direction > 0 ? 0 : state.duration) + direction * remainder;
  }
  function update(delta, options) {
    nonnegative(delta, "Update delta");
    const nextTime = finite(time + delta, "Controller time");
    layers.length = 0;
    pendingEvents.length = 0;
    for (const record of records) {
      const state = record.state,
        next = record.next;
      Object.assign(next, state);
      next.loopDelta = 0;
      next.endEvent = false;
      if (active(state)) {
        let clockDelta = delta,
          fadeFinished = false;
        if (state.fadeDuration > 0) {
          const left = state.fadeDuration - state.fadeElapsed;
          if (state.stopAfterFade) clockDelta = Math.min(delta, left);
          next.fadeElapsed = state.fadeElapsed + Math.min(delta, left);
          const fraction = next.fadeElapsed / state.fadeDuration;
          next.weight = (1 - fraction) * state.fadeFrom + fraction * state.fadeTo;
          fadeFinished = next.fadeElapsed >= state.fadeDuration;
          if (fadeFinished) {
            next.weight = state.fadeTo;
            next.fadeDuration = 0;
          }
        }
        advance(next, clockDelta);
        if (fadeFinished && state.stopAfterFade) stop(next);
        if (next.loopDelta > 0)
          pendingEvents.push(
            Object.freeze({
              type: "loop",
              action: record.action,
              count: next.loopDelta,
              direction: Math.sign(state.speed),
            }),
          );
        if (next.endEvent)
          pendingEvents.push(
            Object.freeze({
              type: "finished",
              action: record.action,
              direction: next.eventDirection,
            }),
          );
        if (active(next) && next.weight > 0) {
          record.layer.time = next.time;
          record.layer.weight = next.weight;
          layers.push(record.layer);
        }
      }
    }
    const nextEvents = pendingEvents.length ? Object.freeze(pendingEvents.slice()) : EMPTY;
    // Pose publication is the commit point. Nothing below invokes caller code.
    pose.blend(layers, options);
    for (const record of records) Object.assign(record.state, record.next);
    time = nextTime;
    events = nextEvents;
    return controller;
  }
  function crossFade(from, to, seconds) {
    const a = get(from).state,
      b = get(to).state;
    nonnegative(seconds, "Crossfade duration");
    if (a === b || !active(a))
      fail("ANIMATION_ACTION_FADE", "Crossfade needs distinct actions and an active source");
    if (!active(b)) {
      start(b);
      b.weight = 0;
    } else start(b);
    fade(a, 0, seconds, true);
    fade(b, 1, seconds, false);
    return controller;
  }
  const controller = Object.freeze({
    pose,
    createAction(clip, options) {
      return exclusive(createAction, clip, options);
    },
    update(delta, options) {
      return exclusive(update, delta, options);
    },
    crossFade(from, to, seconds) {
      return exclusive(crossFade, from, to, seconds);
    },
    get time() {
      return time;
    },
    get events() {
      return events;
    },
    get actionCount() {
      return records.length;
    },
    get disposed() {
      return disposed;
    },
    dispose() {
      if (busy) fail("ANIMATION_REENTRANT", "Cannot dispose during update");
      disposed = true;
      records.length = 0;
      events = EMPTY;
    },
  });
  return controller;
}
