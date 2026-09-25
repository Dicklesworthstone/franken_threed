import test from "node:test";
import assert from "node:assert/strict";
import {bakeAnimationClip, AnimationBakeError} from "./animation_bake.mjs";
import {createAnimationPlayer} from "./animation_runtime.mjs";
import {solveAnimationIK} from "./animation_ik.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function rig(interpolation = "LINEAR") {
  const ibm1 = identity(), ibm2 = identity(); ibm1[12] = -1; ibm2[12] = -2;
  const channel = (node, path, frames) => {
    const values = interpolation === "CUBICSPLINE"
      ? frames.flatMap(frame => [...frame.map(() => 0), ...frame, ...frame.map(() => 0)])
      : frames.flat();
    return {node, path, interpolation, times: [0, 0.5, 1], values};
  };
  return {format: "f3d-animation-v1", nodes: [
    {name: "Hip", parent: -1}, {name: "Elbow", parent: 0, translation: [1, 0, 0]},
    {name: "Hand", parent: 1, translation: [1, 0, 0]},
    {name: "MeshA", parent: -1, translation: [2, 0, 0], weights: [0, 0.5]},
    {name: "MeshB", parent: -1, translation: [-3, 0, 0]},
  ], skins: [{joints: [0, 1, 2], inverseBindMatrices: [...identity(), ...ibm1, ...ibm2]}],
  instances: [{node: 3, skin: 0}, {node: 4, skin: 0}], clips: [{name: "source", channels: [
    channel(0, "translation", [[0, 0, 0], [0.125, 0, 0], [0.25, 0, 0]]),
    channel(1, "rotation", [[0, 0, 0, 1], [0, 0, 0.6, 0.8], [0, 0, 1, 0]]),
    channel(3, "weights", [[0, 0.5], [0.25, 0.75], [1, -0.25]]),
  ]}]};
}
const tracks = () => [{node: 0, path: "translation"}, {node: 0, path: "rotation"},
  {node: 1, path: "rotation"}, {node: 3, path: "weights"}, {node: 3, path: "scale"}];
function close(actual, expected, tolerance = 2e-6) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++)
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance * Math.max(1, Math.abs(expected[i])),
      `component ${i}: ${actual[i]} vs ${expected[i]}`);
}
function comparePose(a, b) {
  for (const field of ["translations", "scales", "morphWeights", "worldMatrices", "jointMatrices"])
    close(a[field], b[field]); // Quaternions can differ by a sign without changing the pose.
}
const code = (name) => ({code: "ANIMATION_BAKE_" + name});

for (const sourceMode of ["LINEAR", "STEP", "CUBICSPLINE"]) for (const interpolation of ["LINEAR", "STEP"]) {
  test(`default bake samples real ${sourceMode} channels into replayable ${interpolation} tracks`, () => {
    const definition = rig(sourceMode), before = structuredClone(definition);
    const result = bakeAnimationClip(definition, {tracks: tracks(), frameRate: 8, interpolation});
    assert.equal(result.frameCount, 9); assert.equal(result.components, 189); assert.equal(result.work, 45);
    assert.equal(result.approximate, true); assert.equal(result.requestedDuration, 1);
    assert.deepEqual(result.sourceRange, [0, 1]); assert.equal(result.execution, "javascript-cpu-animation-bake");
    const original = createAnimationPlayer(definition), replay = createAnimationPlayer({...definition, clips: [result.clip]});
    for (const time of result.clip.channels[0].times) {
      original.sample(time); replay.sample(time); comparePose(replay, original);
    }
    assert.deepEqual(definition, before); original.dispose(); replay.dispose();
  });
}

for (const interpolation of ["LINEAR", "STEP"]) test(`procedural IK ${interpolation} bake replays solved transforms and per-mesh palettes`, () => {
  const definition = rig(), original = structuredClone(definition), expected = [];
  const live = createAnimationPlayer(definition); live.sample(0.37);
  const liveBefore = live.snapshotLocalPose(), liveVersion = live.version;
  let privatePose, cleanups = 0, solves = 0;
  const result = bakeAnimationClip(definition, {tracks: tracks(), start: 0, end: 1,
    frameRate: 8, interpolation, name: "reach", createEvaluator(pose) {
      privatePose = pose;
      assert.notEqual(pose, live);
      return {sample({time}) {
        pose.sample(time);
        pose.edit([{node: 3, scale: [1 + time / 4, 1, 1]}]);
        const target = [1 + time / 4, 0.8 + 0.2 * time, 0.2 * Math.sin(time)];
        const solved = solveAnimationIK(pose, {effector: 2, target, links: [{node: 1}, {node: 0}],
          iterations: 64, tolerance: 1e-5});
        assert.ok(solved.distance < 0.001); solves++;
        expected.push(Object.fromEntries(["translations", "scales", "morphWeights", "worldMatrices", "jointMatrices"]
          .map(field => [field, pose[field].slice()])));
      }, dispose() {assert.equal(pose.disposed, false); cleanups++;}};
    }});
  assert.equal(solves, 9); assert.equal(cleanups, 1); assert.equal(privatePose.disposed, true);
  const replay = createAnimationPlayer({...definition, clips: [result.clip]});
  for (let i = 0; i < result.frameCount; i++) {
    replay.sample(i / 8); comparePose(replay, expected[i]);
    assert.notDeepEqual(replay.jointMatrices.slice(0, 48), replay.jointMatrices.slice(48, 96));
  }
  assert.deepEqual(live.snapshotLocalPose(), liveBefore); assert.equal(live.version, liveVersion);
  assert.deepEqual(definition, original); replay.dispose(); live.dispose();
});

test("custom evaluators retain state, receive immutable ordered frames and are bound to their object", () => {
  let evaluator, privatePose;
  const received = [];
  const result = bakeAnimationClip(rig(), {tracks: [{node: 0, path: "translation"}],
    start: 10, end: 10.625, frameRate: 4, createEvaluator(pose) {
      privatePose = pose;
      evaluator = {sum: 0, calls: 0, disposed: false, sample(frame) {
        assert.equal(this, evaluator); assert.ok(Object.isFrozen(frame)); received.push(frame);
        this.sum += frame.delta; this.calls++;
        // The previous edit remains committed; the baker does not reset per frame.
        const before = pose.snapshotLocalPose().translations[0];
        pose.edit([{node: 0, translation: [before + 1, this.sum, 0]}]);
      }, dispose() {assert.equal(this, evaluator); this.disposed = true;}};
      return evaluator;
    }});
  assert.deepEqual(received.map(f => f.time), [10, 10.25, 10.5, 10.625]);
  assert.deepEqual(received.map(f => f.delta), [0, 0.25, 0.25, 0.125]);
  assert.deepEqual(received.map(f => f.index), [0, 1, 2, 3]);
  assert.deepEqual(received.map(f => f.frameCount), [4, 4, 4, 4]);
  assert.deepEqual(received.map(f => f.progress), [0, 0.4, 0.8, 1]);
  assert.deepEqual(result.clip.channels[0].times, [0, 0.25, 0.5, 0.625]);
  assert.deepEqual(result.clip.channels[0].values, [1, 0, 0, 2, 0.25, 0, 3, 0.5, 0, 4, 0.625, 0]);
  assert.equal(evaluator.disposed, true); assert.equal(privatePose.disposed, true);
});

test("sub-frame ranges and zero-duration ranges include both endpoints without duplicate samples", () => {
  const definition = rig();
  const one = bakeAnimationClip(definition, {tracks: tracks(), start: 0.5, end: 0.5, maxFrames: 1});
  assert.equal(one.frameCount, 1); assert.deepEqual(one.clip.channels[0].times, [0]);
  assert.deepEqual(one.sourceRange, [0.5, 0.5]);
  const tiny = bakeAnimationClip(definition, {tracks: tracks(), start: 0.25, end: 0.3, frameRate: 4});
  assert.equal(tiny.frameCount, 2); assert.deepEqual(tiny.clip.channels[0].times, [0, Math.fround(0.05)]);
  const rounded = bakeAnimationClip(definition, {tracks: tracks(), end: 0.1, frameRate: 30});
  assert.equal(rounded.frameCount, 4); assert.equal(rounded.duration, Math.fround(0.1));
  assert.equal(rounded.requestedDuration, 0.1);
});

test("capture selection and source data are independent of mutation after the bake", () => {
  const definition = rig(), selected = tracks();
  const first = bakeAnimationClip(definition, {tracks: selected, frameRate: 4});
  const second = bakeAnimationClip(definition, {tracks: selected, frameRate: 4});
  assert.deepEqual(first, second); selected.length = 0;
  definition.nodes[0].translation = [999, 999, 999];
  first.clip.channels[0].values.fill(777);
  assert.equal(second.clip.channels[0].values[0], 0);
});

for (const options of [
  {frameRate: 0}, {frameRate: 1001}, {frameRate: Infinity}, {start: NaN}, {end: Infinity},
  {start: 2, end: 1}, {maxFrames: 0}, {maxFrames: 4}, {maxComponents: 104}, {maxWork: 24},
  {maxNodes: 4}, {maxTracks: 4}, {interpolation: "CUBICSPLINE"}, {unknown: true},
  {tracks: [{node: 1, path: "bogus"}]}, {clip: 0},
]) test("invalid schedule, track or limits are refused before evaluator construction", () => {
  let calls = 0;
  assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), end: 1, frameRate: 4,
    createEvaluator() {calls++; return {sample() {}};}, ...options}));
  assert.equal(calls, 0);
});

test("exact frame/component/work limits succeed together", () => {
  let calls = 0;
  const result = bakeAnimationClip(rig(), {tracks: tracks(), end: 1, frameRate: 4,
    maxFrames: 5, maxComponents: 105, maxWork: 25,
    createEvaluator(pose) {return {sample({time}) {calls++; pose.sample(time);}};}});
  assert.equal(calls, 5); assert.equal(result.components, 105); assert.equal(result.work, 25);
});

for (const range of [{start: 1e16, end: 1e16 + 4, frameRate: 4},
  {start: 0, end: 1e40, frameRate: 1e-40}, {start: 0, end: 1e-50, frameRate: 1}])
  test("unrepresentable schedules fail before callbacks or a partial bake", () => {
    let calls = 0;
    assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), ...range,
      createEvaluator() {calls++; return {sample() {}};}}), code("TIME"));
    assert.equal(calls, 0);
  });

test("source clip selection is validated and custom factories need an explicit range", () => {
  for (const options of [{clip: 1}, {clip: -1}, {clip: "0"}, {createEvaluator() {}},
    {end: 1, createEvaluator: true}])
    assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), ...options}), AnimationBakeError);
});

test("cancellation before setup invokes no factory", () => {
  const controller = new AbortController(), reason = Error("cancel before setup"); controller.abort(reason);
  let calls = 0;
  assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), end: 1, signal: controller.signal,
    createEvaluator() {calls++;}}), e => e === reason);
  assert.equal(calls, 0);
});

for (const cancelAt of ["factory", 0, 2]) test(`cancellation at ${cancelAt} cleans the evaluator before its private pose`, () => {
  const controller = new AbortController(), reason = Error("cancel midway");
  let privatePose, calls = 0, cleanups = 0;
  assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), end: 1, frameRate: 4, signal: controller.signal,
    createEvaluator(pose) {
      privatePose = pose; if (cancelAt === "factory") controller.abort(reason);
      return {sample({index}) {calls++; if (index === cancelAt) controller.abort(reason);},
        dispose() {assert.equal(pose.disposed, false); cleanups++;}};
    }}), e => e === reason);
  assert.equal(calls, cancelAt === "factory" ? 0 : cancelAt + 1);
  assert.equal(cleanups, 1); assert.equal(privatePose.disposed, true);
});

test("failed sample preserves its original error even if cleanup fails", () => {
  const original = Error("sampling failed"); let privatePose, cleanups = 0;
  assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), end: 1,
    createEvaluator(pose) {privatePose = pose; return {
      sample() {throw original;}, dispose() {cleanups++; throw Error("cleanup failed");}};}}), e => e === original);
  assert.equal(cleanups, 1); assert.equal(privatePose.disposed, true);
});

test("cleanup failures prevent a successful result but still dispose the player", () => {
  const reason = Error("cleanup"); let privatePose;
  assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), end: 0,
    createEvaluator(pose) {privatePose = pose; return {sample() {}, dispose() {throw reason;}};}}), e => e === reason);
  assert.equal(privatePose.disposed, true);
});

test("factory failures and invalid evaluators always dispose the private pose", () => {
  for (const kind of ["throw", "null", "sample", "dispose"]) {
    let privatePose, cleanups = 0;
    assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), end: 0,
      createEvaluator(pose) {
        privatePose = pose;
        if (kind === "throw") throw Error("factory");
        if (kind === "null") return null;
        return kind === "sample" ? {sample: 2, dispose() {cleanups++;}} : {sample() {}, dispose: 4};
      }}));
    assert.equal(privatePose.disposed, true); assert.equal(cleanups, kind === "sample" ? 1 : 0);
  }
});

for (const phase of ["factory", "sample", "dispose"]) test(`async ${phase} is rejected and its late rejection is observed`, async () => {
  let privatePose, reject, cleanups = 0;
  const pending = new Promise((_, r) => {reject = r;});
  assert.throws(() => bakeAnimationClip(rig(), {tracks: tracks(), end: 0,
    createEvaluator(pose) {
      privatePose = pose;
      if (phase === "factory") return pending;
      return {sample() {if (phase === "sample") return pending;},
        dispose() {cleanups++; if (phase === "dispose") return pending;}};
    }}), code("ASYNC"));
  reject(Error("late failure")); await new Promise(resolve => setImmediate(resolve));
  assert.equal(privatePose.disposed, true); assert.equal(cleanups, phase === "factory" ? 0 : 1);
});

test("a factory can bake a rig with no authored clips using only procedural edits", () => {
  const definition = {format: "f3d-animation-v1", nodes: [{}]};
  const result = bakeAnimationClip(definition, {tracks: [{node: 0, path: "translation"}], end: 1, frameRate: 2,
    createEvaluator(pose) {return {sample({time}) {pose.edit([{node: 0, translation: [time, time * time, 0]}]);}};}});
  assert.deepEqual(result.clip.channels[0].values, [0, 0, 0, 0.5, 0.25, 0, 1, 1, 0]);
  const replay = createAnimationPlayer({...definition, clips: [result.clip]}); replay.sample(0.5);
  assert.deepEqual([...replay.translations], [0.5, 0.25, 0]); replay.dispose();
});
