/** Bake reusable destination clips with the existing sampler and retargeter.
 * Sampling is explicit and bounded, with no live-player mutation or new clock.
 * This is fixed-rate resampling, not exact preservation of source interpolation.
 */

import { AnimationRetargetError, createAnimationRetargeter } from "./animation_retarget.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

const fail = (code, message) => {
  throw new AnimationRetargetError("ANIMATION_RETARGET_" + code, message);
};
function fields(v, allowed, label) {
  if (!v || typeof v !== "object" || Array.isArray(v)) fail("OPTIONS", `Expected ${label}`);
  for (const key of Object.keys(v))
    if (!allowed.includes(key)) fail("OPTIONS", `Unknown ${label} field: ${key}`);
}
function integer(v, min, max, label) {
  if (!Number.isSafeInteger(v) || v < min || v > max) fail("LIMIT", `Invalid ${label}`);
  return v;
}
function finite(v, label) {
  if (typeof v !== "number" || !Number.isFinite(v)) fail("VALUE", `Invalid ${label}`);
  return v;
}
function nodes(definition) {
  if (
    definition?.format !== "f3d-animation-v1" ||
    !Array.isArray(definition.nodes) ||
    !definition.nodes.length ||
    definition.nodes.length > 65536
  )
    fail("SHAPE", "Expected a decoded animation definition");
  return definition.nodes;
}
function abort(signal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
}

/** Resolve an explicit list of {source:'bone name',target:'bone name',weight?}
 * into node-index bindings. Only exact unique names are accepted. No namespace
 * stripping, partial match, default mesh/bone guesses or silent missing joints.
 * Return values can be used by either live retargeting or clip baking.
 */
export function mapAnimationNodeNames(sourceDefinition, targetDefinition, pairs) {
  const source = nodes(sourceDefinition),
    target = nodes(targetDefinition);
  if (!Array.isArray(pairs) || !pairs.length || pairs.length > 4096)
    fail("LIMIT", "Expected 1..4096 named bindings");
  function names(list) {
    const result = new Map();
    for (let i = 0; i < list.length; i++) {
      const name = list[i]?.name;
      if (typeof name !== "string" || !name.length) continue;
      result.set(name, result.has(name) ? -1 : i);
    }
    return result;
  }
  const from = names(source),
    to = names(target),
    used = new Set();
  return Object.freeze(
    pairs.map((pair) => {
      fields(pair, ["source", "target", "weight"], "named binding");
      const s = from.get(pair.source),
        t = to.get(pair.target),
        weight = finite(pair.weight ?? 1, "binding weight");
      if (s === undefined || t === undefined || s < 0 || t < 0)
        fail("MAPPING", "Every selected name must identify exactly one node");
      if (used.has(t) || weight < 0 || weight > 1)
        fail("MAPPING", "Duplicate target or invalid weight");
      used.add(t);
      return Object.freeze({ source: s, target: t, weight });
    }),
  );
}

/** Output .clip can be appended to a destination f3d-animation-v1 definition.
 * Channels are ordinary JSON-serializable number arrays, with times rebased to
 * zero. Both endpoints are included; a zero-duration range has one sample.
 * All source STEP/LINEAR/CUBICSPLINE evaluation uses the existing pose sampler.
 * Output interpolation is explicitly LINEAR (default) or STEP: motion between
 * samples, especially discontinuities/cubic extrema, is an approximation.
 * References are the definitions' imported rest poses. Each frame starts from
 * target rest so partial mapping weights cannot accumulate across samples.
 * maxComponents bounds emitted time/value numbers (including repeated times),
 * maxFrames bounds samples, and maxWork bounds combined node-frame evaluations.
 * These are not total process-memory/native-runtime limits. This function is
 * synchronous; a signal is checked before setup and between samples.
 */
export function retargetAnimationClip(sourceDefinition, targetDefinition, options = {}) {
  fields(
    options,
    [
      "mapping",
      "rootMotion",
      "alignment",
      "clip",
      "start",
      "end",
      "frameRate",
      "interpolation",
      "name",
      "maxFrames",
      "maxComponents",
      "maxWork",
      "maxNodes",
      "maxMappings",
      "signal",
    ],
    "clip options",
  );
  const {
    mapping,
    rootMotion: rootInput = null,
    alignment,
    clip = 0,
    start = 0,
    end: requestedEnd,
    frameRate = 30,
    interpolation = "LINEAR",
    name,
    maxFrames = 6000,
    maxComponents = 16777216,
    maxWork = 16777216,
    maxNodes = 65536,
    maxMappings = 4096,
    signal,
  } = options;
  integer(maxFrames, 1, 1048576, "frame limit");
  integer(maxComponents, 1, 16777216, "component limit");
  integer(maxWork, 1, Number.MAX_SAFE_INTEGER, "node-frame limit");
  integer(maxNodes, 2, 131072, "node budget");
  integer(maxMappings, 1, 65536, "mapping budget");
  const rootMotion = rootInput === null ? null : { ...rootInput };
  const count = nodes(sourceDefinition).length + nodes(targetDefinition).length;
  if (count > maxNodes) fail("LIMIT", "Combined rig node budget exceeded");
  if (!Array.isArray(mapping) || mapping.length > maxMappings)
    fail("LIMIT", "Expected bounded mapping array");
  if (
    finite(frameRate, "frame rate") <= 0 ||
    frameRate > 1000 ||
    !["LINEAR", "STEP"].includes(interpolation)
  )
    fail("OPTIONS", "Unsupported frame rate or output interpolation");
  if (name !== undefined && (typeof name !== "string" || name.length > 4096))
    fail("OPTIONS", "Clip name must be bounded text");
  if (signal !== undefined && (!signal || typeof signal.aborted !== "boolean"))
    fail("OPTIONS", "Expected an AbortSignal");
  abort(signal);
  let source, target, retarget;
  try {
    source = createAnimationPlayer(sourceDefinition);
    target = createAnimationPlayer(targetDefinition);
    integer(clip, 0, source.clips.length - 1, "source clip");
    const end =
      requestedEnd === undefined ? source.clips[clip].duration : finite(requestedEnd, "end time");
    finite(start, "start time");
    if (start < 0 || end < start || end > source.clips[clip].duration)
      fail("RANGE", "Require 0 <= start <= end <= source duration");
    const span = end - start,
      scaled = span * frameRate;
    // Remove only rounding noise at integral frame boundaries, not a final
    // authored interval. The exact end is still sampled and emitted once.
    const steps =
      span === 0 ? 0 : Math.max(1, Math.ceil(scaled - 8 * Number.EPSILON * Math.max(1, scaled)));
    const frameCount = integer(steps + 1, 1, maxFrames, "frame count");
    if (count * frameCount > maxWork) fail("LIMIT", "Node-frame workload budget exceeded");
    retarget = createAnimationRetargeter(source, target, {
      mapping,
      rootMotion,
      alignment,
      maxNodes,
      maxMappings,
    });
    const rootTarget = rootMotion === null ? null : rootMotion.target;
    const tracks = retarget.mapping
      .filter((b) => b.weight > 0)
      .map((b) => ({ node: b.target, path: "rotation", width: 4 }));
    if (rootTarget !== null) tracks.push({ node: rootTarget, path: "translation", width: 3 });
    if (!tracks.length) fail("MAPPING", "No active retargeted channels");
    const components = frameCount * tracks.reduce((total, t) => total + t.width + 1, 0);
    if (!Number.isSafeInteger(components) || components > maxComponents)
      fail("LIMIT", "Baked clip component budget exceeded");
    const times = new Array(frameCount),
      samples = new Array(frameCount);
    for (let i = 0; i < frameCount; i++) {
      const time = i === steps ? end : start + i / frameRate;
      samples[i] = time;
      times[i] = time - start;
      if (i && (time <= samples[i - 1] || time > end))
        fail("RANGE", "Frame spacing is below source time precision");
    }
    const channels = tracks.map((t) => ({
      node: t.node,
      path: t.path,
      interpolation,
      times: times.slice(),
      values: new Array(frameCount * t.width),
    }));
    for (let i = 0; i < frameCount; i++) {
      abort(signal);
      source.sample(samples[i], { clip });
      target.reset();
      retarget.apply();
      for (let k = 0; k < tracks.length; k++) {
        const track = tracks[k],
          channel = channels[k],
          values = track.path === "rotation" ? target.rotations : target.translations;
        const offset = track.node * track.width,
          at = i * track.width;
        // Keep quaternion signs continuous for LINEAR reuse and downstream tools.
        // A sign flip changes no rotation and does not modify the live sampler.
        let sign = 1;
        if (i && track.path === "rotation") {
          let dot = 0;
          for (let c = 0; c < 4; c++) dot += channel.values[at - 4 + c] * values[offset + c];
          if (dot < 0) sign = -1;
        }
        for (let c = 0; c < track.width; c++) channel.values[at + c] = sign * values[offset + c];
      }
    }
    abort(signal);
    return Object.freeze({
      clip: { name: name ?? source.clips[clip].name + "_retargeted", channels },
      frameCount,
      components,
      duration: span,
      sourceRange: Object.freeze([start, end]),
      frameRate,
      interpolation,
      approximate: true,
      execution: "javascript-cpu-retarget-bake",
      accelerationClaim: false,
    });
  } finally {
    retarget?.dispose();
    source?.dispose();
    target?.dispose();
  }
}
