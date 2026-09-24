/**
 * Explicit playback control for packed animation poses. Borrows a player;
 * controller.dispose() does not dispose that player. No timers or callbacks.
 *
 * createAction(clip, options) makes an independent action, even for the same
 * clip. play/stop/seek/configuration take effect on the next update(delta).
 * delta is finite nonnegative wall time in seconds; a signed action timeScale
 * controls playback direction. pause freezes its playhead, not fades or warps.
 * warp/warpTo integrate a linear speed ramp in wall time, including reversals;
 * halt ramps to zero while retaining the pose. Warps continue after natural
 * completion, but never restart a finished action. setTimeScale/stopWarping cancel
 * the ramp at the current speed. reset/stop cancel ramps without restoring speed.
 * Fades change the action weight; play/stop/reset do not restore that weight.
 * startAt(controllerTime).play() arms an absolute start. Before that time the
 * action contributes nothing and its playhead, fade and warp remain frozen.
 * A late start catches up from its deadline. cancelStart resumes without that
 * catch-up; reset/stop cancel pending starts. Only scheduled starts emit a
 * started event, before that action's loop/finish events, on successful update.
 * setDuration sets seconds per traversal without moving the playhead. syncWith
 * copies normalized phase and local speed once, including the ping-pong leg;
 * it does not copy activation, schedules, fades or masks. Crossfades optionally
 * align phase (sync) and match normalized playback rates (warp). See
 * ANIMATION_PLAYBACK_CONTROLS.md for timing and compatibility boundaries.
 *
 * repetitions counts traversals (one ping-pong leg is one traversal), not
 * round trips. seek resets that count and the ping-pong orientation. A finite
 * action stops at its last endpoint; clampWhenFinished keeps contributing it.
 * Zero-duration actions finish once on their first nonzero playback advance.
 *
 * events is the immutable event list for the last SUCCESSFUL update. Loop
 * records aggregate consecutive same-direction crossings, in action creation
 * order, followed by that action's finish record. A reversing warp can produce
 * two loop records. Loop direction is the timeScale sign; finished
 * direction is the final local clip direction. Consume events each update.
 * They are not
 * callbacks and are not an unbounded queue. Failed updates change neither
 * clocks, fades, warps, events nor the published pose. No per-action pose is copied.
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
  const active = (state) => state.startTime === null && (state.playing || (state.finished && state.clamp));
  function resetState(state) {
    state.time = state.speed < 0 ? state.duration : 0;
    state.completed = 0;
    state.orientation = 1;
    state.finished = false;
    state.paused = false;
    state.fadeDuration = 0;
    state.stopAfterFade = false;
    state.warpDuration = 0;
    state.warpElapsed = 0;
    state.startTime = null;
  }
  function start(state) {
    if (state.finished) {
      const scheduled = state.startTime;
      resetState(state);
      state.startTime = scheduled;
    }
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
  function warp(state, from, to, seconds) {
    // Validate everything before changing the currently published controls.
    finite(from, "Warp start time scale");
    finite(to, "Warp end time scale");
    nonnegative(seconds, "Warp duration");
    state.warpFrom = from;
    state.warpTo = to;
    state.warpDuration = seconds;
    state.warpElapsed = 0;
    state.speed = seconds === 0 ? to : from;
  }
  function positiveDuration(value, label) {
    nonnegative(value, label);
    if (value === 0) fail("ANIMATION_ACTION_DURATION", `${label} must be positive`);
    return value;
  }
  function rescale(value, fromDuration, toDuration) {
    // Try alternative evaluation orders when an intermediate ratio overflows
    // or underflows even though the final value is representable.
    if (value === 0) return value;
    let result = value * (toDuration / fromDuration);
    if (!Number.isFinite(result) || result === 0) result = (value / fromDuration) * toDuration;
    if (!Number.isFinite(result) || result === 0) result = (value * toDuration) / fromDuration;
    return result;
  }
  function scaledRate(speed, fromDuration, toDuration) {
    const result = rescale(speed, fromDuration, toDuration);
    if (!Number.isFinite(result) || (result === 0 && speed !== 0))
      fail("ANIMATION_ACTION_DURATION", "Duration-scaled time scale is not representable");
    return result;
  }
  function phaseTime(target, source) {
    // Local phase is bounded; clamp only a last-bit overshoot at the endpoint.
    return Math.min(target.duration, finite(rescale(source.time, source.duration, target.duration), "Synchronized time"));
  }
  function synchronizedClock(target, source) {
    positiveDuration(target.duration, "Target clip duration");
    positiveDuration(source.duration, "Source clip duration");
    const orientation = target.loop === "pingpong" ? source.orientation : 1;
    return {
      time: phaseTime(target, source),
      speed: scaledRate(source.speed, source.duration, target.duration) * source.orientation / orientation,
      orientation,
      completed: 0,
      finished: false,
      warpDuration: 0,
      warpElapsed: 0,
    };
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
      warpFrom: 0,
      warpTo: 0,
      warpDuration: 0,
      warpElapsed: 0,
      startTime: null,
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
      startAt(when) {
        return mutate(() => {
          state.startTime = nonnegative(when, "Scheduled start time");
        });
      },
      cancelStart() {
        return mutate(() => {
          state.startTime = null;
        });
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
          state.warpDuration = 0;
          state.warpElapsed = 0;
        });
      },
      setDuration(seconds) {
        return mutate(() => {
          positiveDuration(seconds, "Playback duration");
          positiveDuration(duration, "Clip duration");
          const speed = scaledRate(state.speed < 0 ? -1 : 1, seconds, duration);
          state.speed = speed;
          state.warpDuration = 0;
          state.warpElapsed = 0;
        });
      },
      syncWith(other) {
        return mutate(() => {
          const source = get(other).state;
          if (source !== state) Object.assign(state, synchronizedClock(state, source));
        });
      },
      warp(from, to, seconds) {
        return mutate(() => warp(state, from, to, seconds));
      },
      warpTo(to, seconds) {
        return mutate(() => warp(state, state.speed, to, seconds));
      },
      halt(seconds) {
        return mutate(() => warp(state, state.speed, 0, seconds));
      },
      stopWarping() {
        return mutate(() => {
          state.warpDuration = 0;
          state.warpElapsed = 0;
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
      get duration() {
        return duration;
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
      get warping() {
        return state.warpDuration > 0;
      },
      get effectiveTimeScale() {
        return state.playing && !state.paused && state.startTime === null ? state.speed : 0;
      },
      get scheduled() {
        return state.startTime !== null;
      },
      get startTime() {
        return state.startTime;
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
  function advance(state, distance, clockDirection) {
    state.loopDelta = 0;
    state.endEvent = false;
    state.eventDirection = 1;
    if (!state.playing || state.paused || distance === 0 || clockDirection === 0) return;
    const direction = clockDirection * state.orientation;
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
  function integratedDistance(seconds, from, to) {
    const high = Math.max(Math.abs(from), Math.abs(to));
    if (high === 0 || seconds === 0) return 0;
    const factor = (from / high + to / high) * 0.5,
      product = seconds * high;
    return finite(Number.isFinite(product) ? product * factor : (high * factor) * seconds, "Playback advance");
  }
  function advanceSegment(record, state, seconds, from, to = from) {
    if (!state.playing || state.paused || seconds === 0 || (from === 0 && to === 0)) return;
    const distance = state.duration === 0 ? 1 : Math.abs(integratedDistance(seconds, from, to));
    advanceDistance(record, state, distance, Math.sign(from || to));
  }
  function advanceDistance(record, state, distance, direction) {
    advance(state, distance, direction);
    if (state.loopDelta > 0) {
      const previous = pendingEvents.at(-1);
      const merge = previous?.type === "loop" && previous.action === record.action && previous.direction === direction;
      const event = Object.freeze({
        type: "loop",
        action: record.action,
        count: state.loopDelta + (merge ? previous.count : 0),
        direction,
      });
      if (merge) pendingEvents[pendingEvents.length - 1] = event;
      else pendingEvents.push(event);
    }
    if (state.endEvent)
      pendingEvents.push(Object.freeze({ type: "finished", action: record.action, direction: state.eventDirection }));
  }
  function advanceRamp(record, state, seconds, from, to) {
    if (from !== 0 && to !== 0 && Math.sign(from) !== Math.sign(to)) {
      // Do not integrate a reversing ramp into one signed displacement: loops
      // or a finite-action finish before the turning point would disappear.
      // Ratio form also avoids overflow when |from| + |to| is not finite.
      const a = Math.abs(from), b = Math.abs(to);
      const fraction = a >= b ? 1 / (1 + b / a) : (a / b) / (1 + a / b);
      const first = seconds * fraction;
      if (!state.playing || state.paused || seconds === 0) return;
      if (state.duration === 0) {
        advanceDistance(record, state, 1, Math.sign(from));
        return;
      }
      const outward = Math.abs(integratedDistance(first, from, 0));
      advanceDistance(record, state, outward, Math.sign(from));
      if (!state.playing) return;
      // Preserve the signed integral across the turn when that subtraction is
      // well-conditioned. For a tiny return leg use its own triangle instead:
      // subtracting two nearly equal outward/net areas can erase it or go negative.
      const returnFraction = a >= b ? (b / a) / (1 + b / a) : 1 / (1 + a / b);
      const triangle = Math.abs(integratedDistance(seconds * returnFraction, 0, to));
      const returning = triangle <= 8 * Number.EPSILON * outward ? triangle :
        nonnegative(outward - Math.sign(from) * integratedDistance(seconds, from, to), "Returning advance");
      advanceDistance(record, state, returning, Math.sign(to));
    } else advanceSegment(record, state, seconds, from, to);
  }
  function advanceClock(record, state, delta) {
    if (state.warpDuration === 0) {
      advanceSegment(record, state, delta, state.speed);
      return;
    }
    const step = Math.min(delta, state.warpDuration - state.warpElapsed),
      elapsed = state.warpElapsed + step,
      finished = elapsed >= state.warpDuration,
      fraction = elapsed / state.warpDuration,
      speed = finished ? state.warpTo : (1 - fraction) * state.warpFrom + fraction * state.warpTo;
    advanceRamp(record, state, step, state.speed, speed);
    state.speed = speed;
    state.warpElapsed = elapsed;
    if (finished) {
      state.warpDuration = 0;
      advanceSegment(record, state, delta - step, speed);
    }
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
      let actionDelta = delta;
      if (next.startTime !== null) {
        // Merely assigning a deadline does not activate an action. A paused
        // but playing action does start; pause still freezes only its playhead.
        if (!next.playing || nextTime < next.startTime) continue;
        actionDelta = nextTime - next.startTime;
        pendingEvents.push(Object.freeze({ type: "started", action: record.action, time: next.startTime }));
        next.startTime = null;
      }
      if (active(next)) {
        let clockDelta = actionDelta,
          fadeFinished = false;
        if (state.fadeDuration > 0) {
          const left = state.fadeDuration - state.fadeElapsed;
          if (state.stopAfterFade) clockDelta = Math.min(actionDelta, left);
          next.fadeElapsed = state.fadeElapsed + Math.min(actionDelta, left);
          const fraction = next.fadeElapsed / state.fadeDuration;
          next.weight = (1 - fraction) * state.fadeFrom + fraction * state.fadeTo;
          fadeFinished = next.fadeElapsed >= state.fadeDuration;
          if (fadeFinished) {
            next.weight = state.fadeTo;
            next.fadeDuration = 0;
          }
        }
        advanceClock(record, next, clockDelta);
        if (fadeFinished && state.stopAfterFade) stop(next);
        if (active(next) && next.weight > 0) {
          record.layer.time = next.time;
          record.layer.weight = next.weight;
          layers.push(record.layer);
        }
      } else if (state.finished && state.warpDuration > 0) {
        // Finish is not stop(): a wall-time ramp still reaches its target even
        // when an unclamped action completes between two host updates.
        advanceClock(record, next, actionDelta);
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
  function crossFade(from, to, seconds, options = {}) {
    const a = get(from).state,
      b = get(to).state;
    nonnegative(seconds, "Crossfade duration");
    if (a === b || !active(a))
      fail("ANIMATION_ACTION_FADE", "Crossfade needs distinct actions and an active source");
    if (!options || typeof options !== "object" || Array.isArray(options) ||
        Object.keys(options).some((key) => key !== "sync" && key !== "warp"))
      fail("ANIMATION_ACTION_FADE", "Expected crossfade sync/warp options");
    const { sync = false, warp: matchRates = false } = options;
    boolean(sync, "Crossfade sync");
    boolean(matchRates, "Crossfade warp");
    if (sync || matchRates) {
      positiveDuration(a.duration, "Source clip duration");
      positiveDuration(b.duration, "Target clip duration");
      if (a.loop !== b.loop)
        fail("ANIMATION_ACTION_FADE", "Synchronized crossfades require matching loop modes");
    }
    if (matchRates && (!a.playing || a.paused))
      fail("ANIMATION_ACTION_FADE", "Warped crossfades require a running source");
    // Configure copies so invalid ratios/options cannot half-start a target or
    // discard an existing schedule, fade or warp on either live action.
    const nextA = { ...a }, nextB = { ...b };
    const targetActive = active(b);
    nextB.startTime = null; // A crossfade is immediate, not a scheduled start.
    start(nextB);
    if (!targetActive) nextB.weight = 0;
    if (sync) {
      nextB.time = phaseTime(b, a);
      nextB.orientation = a.orientation;
      nextB.completed = 0;
      nextB.finished = false;
    }
    if (matchRates) {
      const startRate = scaledRate(a.speed, a.duration, b.duration) * a.orientation / nextB.orientation;
      const endRate = scaledRate(nextB.speed, b.duration, a.duration) * nextB.orientation / a.orientation;
      warp(nextA, a.speed, endRate, seconds);
      warp(nextB, startRate, nextB.speed, seconds);
    }
    fade(nextA, 0, seconds, true);
    fade(nextB, 1, seconds, false);
    Object.assign(a, nextA);
    Object.assign(b, nextB);
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
    crossFade(from, to, seconds, options) {
      return exclusive(crossFade, from, to, seconds, options);
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
