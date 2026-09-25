/** Bake a clip or a synchronous procedural evaluator on an isolated pose.
 * Reuses the real sampler/editor and recorder; never rewinds a live model.
 */
import {createAnimationPlayer} from "./animation_runtime.mjs";
import {createAnimationRecorder} from "./animation_recorder.mjs";

export class AnimationBakeError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationBakeError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationBakeError("ANIMATION_BAKE_" + code, message);
};
function finite(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value))
    fail("VALUE", `Expected finite ${label}`);
  return value;
}
function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    fail("LIMIT", `Invalid ${label}`);
  return value;
}
function abort(signal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
}
function synchronous(value, label) {
  if (value != null && (typeof value === "object" || typeof value === "function") &&
      typeof value.then === "function") {
    // Observe a mistaken async callback's eventual rejection, including attempts
    // to touch the private pose after the bake has failed and disposed it.
    Promise.resolve(value).catch(() => {});
    fail("ASYNC", `${label} must be synchronous`);
  }
  return value;
}

/** Default: sample clip (default 0) with the existing non-looping sampler.
 * Or createEvaluator(pose) -> {sample(frame), dispose?} supplies controller/IK/
 * procedural evaluation on a NEW private player. clip and createEvaluator are
 * mutually exclusive. A custom evaluator requires an explicit end time.
 * frame={time,delta,index,frameCount,relativeTime,progress} is immutable.
 * The first delta is zero; the initial private pose is rest. No implicit reset
 * occurs between samples, so stateful controllers can advance using delta.
 * Factories must clean up their own partial construction if they throw.
 * Successfully returned evaluators are disposed once, before the player, on
 * success, cancellation or failure. Cleanup errors do not mask a prior error.
 * All callbacks are trusted synchronous application code; their side effects,
 * internal work, determinism and external resources are not sandboxed/rolled back.
 * Both endpoints are sampled once. Zero duration has one frame. Output times
 * rebase to zero and round to Float32; output LINEAR/STEP is an approximation,
 * not preservation of cubic extrema, events or procedural behavior between keys.
 * Frame/component/node-frame limits and the entire time grid are admitted before
 * creating the custom evaluator. No renderer, device, clock or worker is owned.
 */
export function bakeAnimationClip(definition, options = {}) {
  if (!options || typeof options !== "object" || Array.isArray(options))
    fail("OPTIONS", "Expected bake options");
  for (const key of Object.keys(options))
    if (!["tracks", "name", "interpolation", "clip", "createEvaluator", "start", "end",
      "frameRate", "maxFrames", "maxComponents", "maxNodes", "maxTracks", "maxWork", "signal"].includes(key))
      fail("OPTIONS", `Unknown bake option: ${key}`);
  const {tracks, name = "baked", interpolation = "LINEAR", clip = 0, createEvaluator,
    start = 0, end: requestedEnd, frameRate = 30, maxFrames = 6000,
    maxComponents = 16777216, maxNodes = 65536, maxTracks = 4096,
    maxWork = 16777216, signal} = options;
  integer(maxFrames, 1, 1048576, "frame limit");
  integer(maxNodes, 1, 65536, "node limit");
  integer(maxWork, 1, Number.MAX_SAFE_INTEGER, "node-frame limit");
  finite(start, "start time");
  if (finite(frameRate, "frame rate") <= 0 || frameRate > 1000)
    fail("OPTIONS", "Frame rate must be in (0,1000]");
  if (createEvaluator !== undefined && (typeof createEvaluator !== "function" ||
      options.clip !== undefined || requestedEnd === undefined))
    fail("OPTIONS", "Custom evaluation needs a factory and end time, without clip selection");
  if (definition?.format !== "f3d-animation-v1" || !Array.isArray(definition.nodes))
    fail("SHAPE", "Expected a decoded animation definition");
  integer(definition.nodes.length, 1, maxNodes, "node count");
  abort(signal);
  let pose, recorder, evaluator, cleanup, failed = false;
  try {
    pose = createAnimationPlayer(definition);
    const end = requestedEnd === undefined
      ? pose.clips[integer(clip, 0, pose.clips.length - 1, "clip")].duration
      : finite(requestedEnd, "end time");
    if (createEvaluator === undefined) integer(clip, 0, pose.clips.length - 1, "clip");
    if (end < start) fail("RANGE", "End must not precede start");
    const span = end - start, scaled = span * frameRate;
    if (!Number.isFinite(span) || !Number.isFinite(scaled)) fail("LIMIT", "Bake range overflows");
    const steps = span === 0 ? 0 : Math.max(1,
      Math.ceil(scaled - 8 * Number.EPSILON * Math.max(1, scaled)));
    const frameCount = integer(steps + 1, 1, maxFrames, "frame count");
    const work = pose.nodeCount * frameCount;
    if (!Number.isSafeInteger(work) || work > maxWork) fail("LIMIT", "Node-frame budget exceeded");
    recorder = createAnimationRecorder(pose, {tracks, name, interpolation,
      maxFrames, maxComponents, maxNodes, maxTracks});
    if (recorder.componentsPerFrame * frameCount > maxComponents)
      fail("LIMIT", "Complete baked clip exceeds the component budget");
    const times = new Array(frameCount);
    let previousKey = -1;
    for (let index = 0; index < frameCount; index++) {
      const time = index === steps ? end : start + index / frameRate;
      const key = Math.fround(time - start);
      if (!Number.isFinite(time) || !Number.isFinite(key) || key <= previousKey ||
          time > end || (index && time <= times[index - 1]))
        fail("TIME", "Bake spacing is below source/Float32 time precision");
      times[index] = time;
      previousKey = key;
    }
    abort(signal);
    evaluator = createEvaluator === undefined
      ? {sample: ({time}) => pose.sample(time, {clip})}
      : synchronous(createEvaluator(pose), "Evaluator factory");
    if (evaluator && typeof evaluator === "object" && !Array.isArray(evaluator))
      cleanup = evaluator.dispose;
    if (!evaluator || typeof evaluator !== "object" || Array.isArray(evaluator) ||
        typeof evaluator.sample !== "function" || (cleanup !== undefined && typeof cleanup !== "function"))
      fail("EVALUATOR", "Factory must return {sample(frame), dispose?}");
    const sample = evaluator.sample;
    for (let index = 0; index < frameCount; index++) {
      abort(signal);
      const time = times[index];
      const frame = Object.freeze({time, delta: index ? time - times[index - 1] : 0,
        index, frameCount, relativeTime: time - start, progress: span === 0 ? 0 : (time - start) / span});
      synchronous(Reflect.apply(sample, evaluator, [frame]), "Evaluator sample");
      recorder.capture(time, {signal});
    }
    abort(signal);
    const recorded = recorder.finish({signal});
    return Object.freeze({...recorded, frameRate, requestedDuration: span, work,
      execution: "javascript-cpu-animation-bake"});
  } catch (error) {
    failed = true;
    throw error;
  } finally {
    try {
      if (typeof cleanup === "function") synchronous(Reflect.apply(cleanup, evaluator, []), "Evaluator disposal");
    } catch (error) {
      if (!failed) throw error;
    } finally {
      try { recorder?.dispose(); } finally { pose?.dispose(); }
    }
  }
}
