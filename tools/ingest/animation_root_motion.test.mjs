import test from "node:test";
import assert from "node:assert/strict";
import { AnimationRootMotionError, createAnimationRootMotionTrack, extractAnimationRootMotion,
  applyAnimationRootMotion } from "./animation_root_motion.mjs";
const track = (extra = {}) => ({ format: "f3d-root-translation-v1", duration: 2,
  times: [0, 2], values: [10, 4, -3, 18, 6, 1], ...extra });
const close = (actual, expected, epsilon = 1e-10) => actual.forEach((value, i) =>
  assert.ok(Math.abs(value - expected[i]) <= epsilon, `${value} != ${expected[i]}`));
const error = code => e => e instanceof AnimationRootMotionError && e.code === `ANIMATION_ROOT_MOTION_${code}`;

test("linear displacement is reference-relative and counts forward and reverse seams", () => {
  const motion = createAnimationRootMotionTrack(track(), 2);
  assert.deepEqual(motion.advance(0, 0.5), [2, 0.5, 1]);
  assert.deepEqual(motion.advance(1.75, 0.25, 1), [2, 0.5, 1]);
  assert.deepEqual(motion.advance(0.25, 1.75, 1, -1), [-2, -0.5, -1]);
  assert.deepEqual(motion.advance(0, 0, 1000000000), [8000000000, 2000000000, 4000000000]);
});

test("zero-wrap differences telescope across ping-pong reflections", () => {
  const motion = createAnimationRootMotionTrack(track(), 2);
  const outward = motion.advance(0.25, 2), returning = motion.advance(2, 0.75);
  assert.deepEqual(outward.map((value, i) => value + returning[i]), motion.advance(0.25, 0.75));
  assert.deepEqual(motion.advance(0, 0), [0, 0, 0]);
});

test("STEP arrival and reverse departure use the authored discontinuity", () => {
  const motion = createAnimationRootMotionTrack(track({ interpolation: "STEP",
    times: [0, 0.5, 1, 2], values: [0, 0, 0, 5, 1, 0, 7, 2, 0, 10, 3, 0] }), 2);
  assert.deepEqual(motion.advance(0, 0.5), [5, 1, 0]);
  assert.deepEqual(motion.advance(0.5, 0.9), [0, 0, 0]);
  assert.deepEqual(motion.advance(0.5, 0.49), [-5, -1, 0]);
  assert.deepEqual(motion.advance(1.9, 0, 1), [3, 1, 0]);
});

test("nonzero first key and early last key hold to full clip boundaries", () => {
  const motion = createAnimationRootMotionTrack(track({ times: [0.5, 1.5] }), 2);
  assert.deepEqual(motion.advance(0, 0.25), [0, 0, 0]);
  assert.deepEqual(motion.advance(1.5, 2), [0, 0, 0]);
  assert.deepEqual(motion.advance(1.75, 0.25, 1), [0, 0, 0]);
  assert.deepEqual(motion.advance(0, 0, 2), [16, 4, 8]);
});

test("cubic endpoint samples use Hermite tangents, not linear interpolation", () => {
  // x(t) = t^3 and y(t) = t^2 over [0,2], with endpoint derivatives.
  const motion = createAnimationRootMotionTrack(track({ interpolation: "CUBICSPLINE",
    values: [0,0,0, 0,0,7, 0,0,0, 12,4,0, 8,4,7, 12,4,0] }), 2);
  for (let i = 0; i <= 32; i++) {
    const time = i / 16;
    close(motion.advance(0, time), [time ** 3, time ** 2, 0]);
  }
  close(motion.advance(1.5, 0.5, 2), [12.75, 6, 0]);
});

test("single-key and zero-duration tracks have no invented displacement", () => {
  for (const duration of [0, 2]) {
    const motion = createAnimationRootMotionTrack(track({ duration, times: [0], values: [2,3,4] }), duration);
    assert.deepEqual(motion.advance(0, duration, 1000000000), [0,0,0]);
  }
});

test("tracks snapshot caller arrays and expose frozen metadata", () => {
  const input = track(), motion = createAnimationRootMotionTrack(input, 2);
  input.times[1] = 1; input.values.fill(0);
  assert.deepEqual(motion.advance(0, 2), [8,2,4]);
  assert.ok(Object.isFrozen(motion)); assert.ok(Object.isFrozen(motion.definition));
  assert.ok(Object.isFrozen(motion.definition.times)); assert.ok(Object.isFrozen(motion.definition.values));
  assert.throws(() => { motion.definition.values[0] = 0; }, TypeError);
  const delta = motion.advance(0, 1); delta.fill(99);
  assert.deepEqual(motion.advance(0, 1), [4,1,2]);
});

test("typed inputs use only their declared subviews and reject detached/shared storage", () => {
  const values = Float64Array.from([99, ...track().values, 99]);
  const motion = createAnimationRootMotionTrack(track({ values: values.subarray(1, 7) }), 2);
  assert.deepEqual(motion.advance(0, 2), [8,2,4]);
  const detached = new Float64Array(6);
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  for (const values of [detached, new Float64Array(new SharedArrayBuffer(48)), new DataView(new ArrayBuffer(48))])
    assert.throws(() => createAnimationRootMotionTrack(track({ values }), 2), AnimationRootMotionError);
});

test("malformed tracks and traversal requests fail explicitly", () => {
  for (const input of [null, [], {}, track({ format: "wrong" }), track({ duration: 1 }),
    track({ interpolation: "SLERP" }), track({ extra: true }), track({ times: [] }),
    track({ times: [0,0] }), track({ times: [-1,2] }), track({ times: [0,3] }),
    track({ times: [0,NaN] }), track({ times: Array(1048577) }), track({ values: Array(6) }),
    track({ values: [0,0,0,NaN,0,0] }), track({ interpolation: "CUBICSPLINE", times: [0], values: Array(9).fill(0) })])
    assert.throws(() => createAnimationRootMotionTrack(input, 2), AnimationRootMotionError);
  const motion = createAnimationRootMotionTrack(track(), 2);
  for (const args of [[-1,0], [0,3], [NaN,0], [0,0,-1], [0,0,1.5], [0,0,Infinity], [0,0,1,0]])
    assert.throws(() => motion.advance(...args), AnimationRootMotionError);
});

test("non-finite sampled or accumulated motion is refused without corrupting the track", () => {
  const motion = createAnimationRootMotionTrack(track({ values: [0,0,0,Number.MAX_VALUE,0,0] }), 2);
  assert.throws(() => motion.advance(0,0,2), error("VALUE"));
  assert.deepEqual(motion.advance(0,1), [Number.MAX_VALUE / 2,0,0]);
  const cubic = createAnimationRootMotionTrack(track({ interpolation: "CUBICSPLINE", values: [
    0,0,0, 0,0,0, Number.MAX_VALUE,0,0, -Number.MAX_VALUE,0,0, 0,0,0, 0,0,0] }), 2);
  // Large tangents do not corrupt subsequent endpoint queries, regardless of
  // whether a particular representable intermediate sample is accepted.
  assert.deepEqual(cubic.advance(0,2), [0,0,0]);
});

test("scene placement post-composes local motion through existing rotation and scale", () => {
  const matrix = [0,0,-2,0, 0,3,0,0, 4,0,0,0, 10,20,30,1], before = matrix.slice();
  const result = applyAnimationRootMotion(matrix, [1,2,3]);
  assert.deepEqual(result.slice(12,15), [22,26,28]);
  assert.deepEqual(result.slice(0,12), matrix.slice(0,12));
  assert.deepEqual(matrix, before); assert.ok(Object.isFrozen(result));
  assert.deepEqual(applyAnimationRootMotion(null, [1,2,3]).slice(12,15), [1,2,3]);
});

test("placement validation and overflow cannot mutate input", () => {
  const valid = applyAnimationRootMotion(null, [0,0,0]);
  for (const matrix of [[], Array(16).fill(0), valid.map((x,i) => i === 3 ? 1 : x), valid.map((x,i) => i === 0 ? NaN : x)])
    assert.throws(() => applyAnimationRootMotion(matrix, [1,2,3]), AnimationRootMotionError);
  for (const delta of [[], [0,0,NaN], [0,Infinity,0]])
    assert.throws(() => applyAnimationRootMotion(valid, delta), AnimationRootMotionError);
  const extreme = valid.slice(); extreme[0] = Number.MAX_VALUE;
  const before = extreme.slice();
  assert.throws(() => applyAnimationRootMotion(extreme, [2,0,0]), error("VALUE"));
  assert.deepEqual(extreme, before);
});

// A snapshot-provider fixture isolates extraction from animation evaluation.
// Production sampler/controller round trips are covered by the integration suite.
function fixture({ cubic = false, parents = [-1,0], matrices = [], channels } = {}) {
  const original = { name: "walk", channels: channels ?? [
    { node: 0, path: "translation", interpolation: cubic ? "CUBICSPLINE" : "LINEAR", times: [0,2],
      values: cubic ? [1,2,3, 10,4,-3, 4,5,6, 7,8,9, 18,6,1, 10,11,12] : [10,4,-3,18,6,1] },
    { node: 1, path: "rotation", interpolation: "LINEAR", times: [0,2], values: [0,0,0,1,0,0,1,0] },
  ] };
  let reads = 0;
  const pose = { nodeCount: 2, clips: [{ duration: 2 }],
    snapshotLocalPose() { return { format: "f3d-local-pose-v1", nodeCount: 2, parents, matrices }; },
    snapshotClips(indices) { assert.deepEqual(indices, [0]); reads++; return { clips: [structuredClone(original)] }; } };
  return { pose, original, get reads() { return reads; } };
}

test("in-place extraction keeps vertical bob, other channels and source snapshots intact", () => {
  const source = fixture(), before = structuredClone(source.original);
  const result = extractAnimationRootMotion(source.pose, { node: 0 });
  assert.deepEqual(result.clip.channels[0].values, [10,4,-3,10,6,-3]);
  assert.deepEqual(result.rootMotion.values, [10,0,-3,18,0,1]);
  assert.deepEqual(result.clip.channels[1], before.channels[1]);
  assert.deepEqual(source.original, before); assert.equal(source.reads, 1);
  assert.equal(result.clip.name, "walk-in-place"); assert.deepEqual(result.axes, [true,false,true]);
  assert.equal(JSON.parse(JSON.stringify(result)).rootMotion.format, "f3d-root-translation-v1");
  const motion = createAnimationRootMotionTrack(result.rootMotion, 2);
  assert.deepEqual(motion.advance(0,2), [8,0,4]);
});

test("cubic extraction zeros only selected axis tangents, preserving original interpolation", () => {
  const source = fixture({ cubic: true });
  const result = extractAnimationRootMotion(source.pose, { node: 0, axes: [true,false,true], name: "stationary" });
  assert.deepEqual(result.clip.channels[0].values, [0,2,0, 10,4,-3, 0,5,0, 0,8,0, 10,6,-3, 0,11,0]);
  assert.deepEqual(result.rootMotion.values, [1,0,3, 10,0,-3, 4,0,6, 7,0,9, 18,0,1, 10,0,12]);
  assert.equal(result.clip.channels[0].interpolation, "CUBICSPLINE");
  assert.equal(result.clip.name, "stationary");
  result.clip.channels[0].values[0] = 999;
  assert.equal(source.original.channels[0].values[0], 1);
});

test("all-axis extraction removes only translation and rejects ambiguous root definitions", () => {
  const source = fixture();
  assert.deepEqual(extractAnimationRootMotion(source.pose, { node: 0, axes: [true,true,true] }).clip.channels[0].values,
    [10,4,-3,10,4,-3]);
  for (const options of [{}, {node:1}, {node:2}, {node:0,axes:[]}, {node:0,axes:[1,0,1]},
    {node:0,axes:[false,false,false]}, {node:0,name:2}, {node:0,name:"x".repeat(4097)}, {node:0,yaw:true}])
    assert.throws(() => extractAnimationRootMotion(source.pose, options), AnimationRootMotionError);
  assert.throws(() => extractAnimationRootMotion(fixture({matrices:[{node:0}]}).pose, {node:0}), error("NODE"));
  assert.throws(() => extractAnimationRootMotion(fixture({channels:[]}).pose, {node:0}), error("CHANNEL"));
  source.pose.disposed = true;
  assert.throws(() => extractAnimationRootMotion(source.pose, {node:0}), error("POSE"));
});

test("10000 deterministic periodic paths agree with unwrapped linear motion", () => {
  const motion = createAnimationRootMotionTrack(track(), 2);
  let seed = 12345;
  const random = () => { seed = (Math.imul(seed,1664525)+1013904223) >>> 0; return seed; };
  for (let i = 0; i < 10000; i++) {
    const start = (random() % 256) / 128, distance = (random() % 65536) / 128, dir = random() % 2 ? 1 : -1;
    const unwrapped = start + dir * distance, end = ((unwrapped % 2) + 2) % 2;
    const signedWraps = Math.round((unwrapped - end) / 2);
    close(motion.advance(start, end, Math.abs(signedWraps), signedWraps < 0 ? -1 : 1),
      [4 * distance * dir, distance * dir, 2 * distance * dir]);
  }
});
