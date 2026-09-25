import test from "node:test";
import assert from "node:assert/strict";
import {createAnimationPlayer} from "./animation_runtime.mjs";
import {createAnimationRecorder, AnimationRecordingError} from "./animation_recorder.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function rig() {
  const inverse = identity(); inverse[13] = -1;
  return {format: "f3d-animation-v1", nodes: [
    {name: "Hip", parent: -1}, {name: "Arm", parent: 0, translation: [0, 1, 0]},
    {name: "MeshA", parent: -1, translation: [2, 0, 0], weights: [0, 0.5]},
    {name: "MeshB", parent: -1, translation: [-3, 0, 0]},
  ], skins: [{joints: [0, 1], inverseBindMatrices: [...identity(), ...inverse]}],
  instances: [{node: 2, skin: 0}, {node: 3, skin: 0}], clips: [
    {name: "walk", channels: [
      {node: 0, path: "translation", times: [0, 1], values: [0, 0, 0, 2, 0, 1]},
      {node: 1, path: "rotation", times: [0, 1], values: [0, 0, 0, 1, 0, 0, 0.6, 0.8]},
      {node: 2, path: "weights", times: [0, 1], values: [0, 0.5, 1, -0.25]},
    ]},
    {name: "gesture", channels: [
      {node: 0, path: "translation", times: [0, 1], values: [0, 0, 0, 0, 1, 0]},
      {node: 1, path: "rotation", times: [0, 1], values: [0, 0, 0, 1, 0.6, 0, 0, 0.8]},
    ]},
  ]};
}
const tracks = () => [
  {node: 0, path: "translation"}, {node: 1, path: "rotation"},
  {node: 2, path: "weights"}, {node: 2, path: "scale"},
];
function close(actual, expected, tolerance = 2e-6) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++)
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance * Math.max(1, Math.abs(expected[i])),
      `component ${i}: ${actual[i]} vs ${expected[i]}`);
}
const code = (name) => ({code: "ANIMATION_RECORD_" + name});
function boundary(pose, hook) {
  return {get nodeCount() { return pose.nodeCount; }, get version() { return pose.version; },
    get disposed() { return pose.disposed; }, snapshotLocalPose() {
      const state = structuredClone(pose.snapshotLocalPose());
      hook(state);
      return state;
    }};
}

for (const interpolation of ["LINEAR", "STEP"]) {
  test(`capture blended and edited ${interpolation} motion, then replay separate mesh skin palettes`, () => {
    const definition = rig(), untouched = structuredClone(definition), pose = createAnimationPlayer(definition);
    const recorder = createAnimationRecorder(pose, {tracks: tracks(), interpolation, name: "authored-live"});
    const expected = [];
    for (let i = 0; i < 5; i++) {
      const time = i / 4;
      pose.blend([{clip: 0, time, weight: 0.7}, {clip: 1, time, weight: 0.3}]);
      pose.edit([{node: 2, scale: [1 + time / 2, 1, 1]}]);
      const version = pose.version, before = pose.snapshotLocalPose();
      assert.equal(recorder.capture(10 + time), recorder);
      assert.equal(pose.version, version);
      assert.deepEqual(pose.snapshotLocalPose(), before);
      expected.push({world: pose.worldMatrices.slice(), palettes: pose.jointMatrices.slice(),
        morph: pose.morphWeights.slice(), translation: pose.translations.slice()});
    }
    const result = recorder.finish();
    assert.equal(result.frameCount, 5); assert.equal(result.components, 80);
    assert.deepEqual(result.sourceRange, [10, 11]); assert.equal(result.duration, 1);
    assert.equal(result.approximate, true); assert.equal(result.accelerationClaim, false);
    assert.equal(recorder.finished, true);
    assert.deepEqual(result.clip.channels.map(c => c.times), Array(4).fill([0, 0.25, 0.5, 0.75, 1]));
    const replay = createAnimationPlayer({...definition, clips: [result.clip]});
    for (let i = 0; i < 5; i++) {
      replay.sample(i / 4);
      close(replay.worldMatrices, expected[i].world);
      close(replay.jointMatrices, expected[i].palettes);
      close(replay.morphWeights, expected[i].morph);
      close(replay.translations, expected[i].translation);
      assert.notDeepEqual(replay.jointMatrices.slice(0, 32), replay.jointMatrices.slice(32, 64));
    }
    assert.deepEqual(definition, untouched);
    replay.dispose(); recorder.dispose(); assert.equal(pose.disposed, false); pose.dispose();
  });
}

test("committed snapshots, not poisoned public output arrays, are the recording source", () => {
  const pose = createAnimationPlayer(rig()), recorder = createAnimationRecorder(pose, {tracks: tracks()});
  pose.sample(0.5); const snapshot = pose.snapshotLocalPose();
  pose.translations.fill(999); pose.rotations.fill(999); pose.scales.fill(999); pose.morphWeights.fill(999);
  recorder.capture(0);
  const clip = recorder.finish().clip;
  assert.deepEqual(clip.channels[0].values, Array.from(snapshot.translations.slice(0, 3), Math.fround));
  assert.deepEqual(clip.channels[2].values, Array.from(snapshot.morphWeights, Math.fround));
  assert.equal(pose.translations[0], 999); // Reading did not repair or mutate public state.
  pose.dispose();
});

test("only selected local channels are recorded, not external world placement", () => {
  const pose = createAnimationPlayer(rig()), root = identity(); root[12] = 100;
  pose.sample(1, {rootMatrix: root});
  const r = createAnimationRecorder(pose, {tracks: [{node: 0, path: "translation"}]});
  r.capture(-50); const result = r.finish();
  assert.deepEqual(result.clip.channels[0].values, [2, 0, 1]);
  assert.deepEqual(result.clip.channels[0].times, [0]);
  assert.deepEqual(result.sourceRange, [-50, -50]);
  assert.equal(pose.worldMatrices[12], 102); pose.dispose();
});

test("track descriptors are copied and captured data survives later edits and pose disposal", () => {
  const pose = createAnimationPlayer(rig()), selected = tracks();
  const r = createAnimationRecorder(pose, {tracks: selected}); selected[0].node = 3; selected.length = 0;
  pose.sample(0.5); r.capture(0); pose.sample(1); pose.dispose();
  const result = r.finish(); assert.equal(result.clip.channels[0].node, 0);
  assert.deepEqual(result.clip.channels[0].values, [1, 0, 0.5]);
  assert.notEqual(result.clip.channels[0].times, result.clip.channels[1].times);
  r.dispose(); result.clip.channels[0].values[0] = 8;
  assert.equal(result.clip.channels[0].values[0], 8);
});

test("quaternion signs remain continuous across antipodal edited poses", () => {
  const pose = createAnimationPlayer(rig());
  const r = createAnimationRecorder(pose, {tracks: [{node: 1, path: "rotation"}]});
  for (let i = 0; i < 4; i++) {
    pose.edit([{node: 1, rotation: [0, 0, (i % 2 ? -1 : 1) * 0.6, (i % 2 ? -1 : 1) * 0.8]}]);
    r.capture(i);
  }
  const values = r.finish().clip.channels[0].values;
  for (let i = 0; i < 4; i++) close(values.slice(i * 4, i * 4 + 4), [0, 0, 0.6, 0.8]);
  pose.dispose();
});

test("Float32 values and rebased times are committed once, with negative zero retained", () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: [{node: 0, path: "translation"}]});
  pose.edit([{node: 0, translation: [-0, 0.1, 0.2]}]); r.capture(100); r.capture(100.1);
  const c = r.finish().clip.channels[0];
  assert.deepEqual(c.times, [0, Math.fround(0.1)]);
  assert.deepEqual(c.values.slice(0, 3), [-0, Math.fround(0.1), Math.fround(0.2)]); pose.dispose();
});

test("a held pose can be recorded repeatedly without changing its version", () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: tracks()});
  const version = pose.version; r.capture(0).capture(1);
  assert.equal(pose.version, version); assert.equal(r.finish().frameCount, 2); pose.dispose();
});

for (const bad of [NaN, Infinity, "1", null]) test(`invalid capture time ${String(bad)} is recoverable`, () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: tracks()});
  assert.throws(() => r.capture(bad), AnimationRecordingError); assert.equal(r.frameCount, 0);
  r.capture(2); assert.deepEqual(r.finish().sourceRange, [2, 2]); pose.dispose();
});

test("duplicate, reversed, overflowing and Float32-collapsed times never consume a frame", () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: tracks()});
  r.capture(0).capture(1);
  for (const time of [1, -1, 1 + 1e-10, 1e40]) assert.throws(() => r.capture(time), code("TIME"));
  assert.equal(r.frameCount, 2); r.capture(2); assert.equal(r.finish().frameCount, 3); pose.dispose();
});

for (const limits of [{maxFrames: 1}, {maxComponents: 16}]) test("frame and emitted-number limits admit the exact boundary", () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: tracks(), ...limits});
  assert.equal(r.componentsPerFrame, 16); r.capture(0);
  assert.throws(() => r.capture(1), code("LIMIT")); assert.equal(r.frameCount, 1); assert.equal(r.components, 16);
  assert.equal(r.finish().components, 16); pose.dispose();
});

test("failure on a later track discards the whole frame, including its proposed time origin", () => {
  const pose = createAnimationPlayer(rig()); let broken = false;
  const wrapped = boundary(pose, s => {if (broken) s.scales[6] = 1e40;});
  const r = createAnimationRecorder(wrapped, {tracks: tracks()}); broken = true;
  assert.throws(() => r.capture(100), code("VALUE")); assert.equal(r.frameCount, 0);
  broken = false; r.capture(200); assert.deepEqual(r.finish().sourceRange, [200, 200]); pose.dispose();
});

for (const mutate of [
  s => {s.parents[1] = -1;}, s => {s.morphOffsets[2] = 1;},
  s => {s.matrices.push({node: 1, matrix: identity()});},
]) test("rig/layout changes are rejected without corrupting already accepted frames", () => {
  const pose = createAnimationPlayer(rig()); let broken = false;
  const r = createAnimationRecorder(boundary(pose, s => {if (broken) mutate(s);}), {tracks: tracks()});
  r.capture(0); broken = true;
  assert.throws(() => r.capture(1), AnimationRecordingError); assert.equal(r.frameCount, 1);
  broken = false; r.capture(1); assert.equal(r.finish().frameCount, 2); pose.dispose();
});

for (const mutate of [
  s => {s.rotations[7] = 0;}, s => {s.translations[0] = NaN;}, s => {s.scales[6] = Infinity;},
  s => {s.morphWeights = Array.from(s.morphWeights); s.morphWeights[0] = "1";}, s => {s.translations = new DataView(new ArrayBuffer(96));},
  s => {s.translations = new Float64Array(new SharedArrayBuffer(96));},
  s => {s.translations = new Float64Array(new ArrayBuffer(96, {maxByteLength: 192}));},
  s => {structuredClone(s.translations.buffer, {transfer: [s.translations.buffer]});},
]) test("invalid selected values and storage cannot publish a partial frame", () => {
  const pose = createAnimationPlayer(rig()); let broken = false;
  const r = createAnimationRecorder(boundary(pose, s => {if (broken) mutate(s);}), {tracks: tracks()});
  r.capture(0); broken = true;
  assert.throws(() => r.capture(1), AnimationRecordingError); assert.equal(r.frameCount, 1);
  assert.equal(r.finish().frameCount, 1); pose.dispose();
});

test("a version change inside snapshot acquisition rejects the stale frame", () => {
  const pose = createAnimationPlayer(rig()); let change = false;
  const r = createAnimationRecorder(boundary(pose, () => {if (change) pose.sample(1);}), {tracks: tracks()});
  change = true; assert.throws(() => r.capture(0), code("STALE")); assert.equal(r.frameCount, 0);
  change = false; r.capture(0); assert.equal(r.finish().frameCount, 1); pose.dispose();
});

test("cancellation is per operation; accepted frames remain finishable", () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: tracks()});
  const controller = new AbortController(), reason = new Error("cancel capture"); controller.abort(reason);
  assert.throws(() => r.capture(0, {signal: controller.signal}), e => e === reason);
  r.capture(1);
  assert.throws(() => r.finish({signal: controller.signal}), e => e === reason);
  assert.equal(r.finished, false); assert.equal(r.finish().frameCount, 1); pose.dispose();
});

test("abort after snapshot rejects before publication and before accepting its origin", () => {
  const pose = createAnimationPlayer(rig()), controller = new AbortController(); let cancel = false;
  const r = createAnimationRecorder(boundary(pose, () => {if (cancel) controller.abort();}), {tracks: tracks()});
  cancel = true; assert.throws(() => r.capture(5, {signal: controller.signal}), {name: "AbortError"});
  cancel = false; r.capture(8); assert.deepEqual(r.finish().sourceRange, [8, 8]); pose.dispose();
});

test("snapshot callbacks cannot reenter capture, finish or disposal", () => {
  const pose = createAnimationPlayer(rig()); let hook = () => {};
  const r = createAnimationRecorder(boundary(pose, () => hook()), {tracks: tracks()});
  for (const operation of [() => r.capture(1), () => r.finish(), () => r.dispose()]) {
    hook = operation; assert.throws(() => r.capture(0), code("REENTRANT")); assert.equal(r.frameCount, 0);
  }
  hook = () => {}; r.capture(0); r.finish(); pose.dispose();
});

test("empty finish is recoverable; sealed/disposed recorders never accept more work", () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: tracks()});
  assert.throws(() => r.finish(), code("EMPTY")); r.capture(0); r.finish();
  assert.throws(() => r.capture(1), code("FINISHED")); assert.throws(() => r.finish(), code("FINISHED"));
  r.dispose(); r.dispose(); assert.equal(pose.disposed, false);
  assert.throws(() => r.capture(1), code("DISPOSED")); pose.dispose();
});

test("a disposed pose prevents further captures but not finishing owned frames", () => {
  const pose = createAnimationPlayer(rig()), r = createAnimationRecorder(pose, {tracks: tracks()});
  r.capture(0); pose.dispose(); assert.throws(() => r.capture(1), code("POSE"));
  assert.equal(r.finish().frameCount, 1);
});

for (const options of [
  {tracks: []}, {tracks: [undefined]}, {tracks: Array(1)}, {tracks: [{node: 4, path: "rotation"}]},
  {tracks: [{node: -1, path: "rotation"}]}, {tracks: [{node: 0, path: "bogus"}]},
  {tracks: [{node: 0, path: "weights"}]}, {tracks: [{node: 1, path: "rotation"}, {node: 1, path: "rotation"}]},
  {interpolation: "CUBICSPLINE"}, {name: 1}, {name: "x".repeat(4097)}, {maxFrames: 0},
  {maxFrames: 1048577}, {maxComponents: 15}, {maxNodes: 3}, {maxTracks: 3}, {unknown: true},
]) test("invalid configuration fails before recording", () => {
  const pose = createAnimationPlayer(rig()), version = pose.version;
  assert.throws(() => createAnimationRecorder(pose, {tracks: tracks(), ...options}), AnimationRecordingError);
  assert.equal(pose.version, version); pose.dispose();
});

test("matrix nodes permit explicit morph recording, but never pretend to provide TRS tracks", () => {
  const pose = createAnimationPlayer({format: "f3d-animation-v1", nodes: [{matrix: identity(), weights: [0.25]}]});
  assert.throws(() => createAnimationRecorder(pose, {tracks: [{node: 0, path: "rotation"}]}), code("TRACK"));
  const r = createAnimationRecorder(pose, {tracks: [{node: 0, path: "weights"}]});
  pose.edit([{node: 0, weights: [0.75]}]); r.capture(0);
  assert.deepEqual(r.finish().clip.channels[0].values, [0.75]); pose.dispose();
});
