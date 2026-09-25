import test from "node:test";
import assert from "node:assert/strict";
import { createAnimationPlayer, AnimationPoseError } from "./animation_runtime.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const fields = ["translations", "rotations", "scales", "morphWeights", "worldMatrices", "jointMatrices"];
const translation = (x = 4) => ({node: 0, path: "translation", times: [0, 1], values: [0, 0, 0, x, 2, 0]});
const clip = (x = 4) => ({name: "move", channels: [translation(x)]});
function rig(clips = [clip()]) {
  const inverse = identity(); inverse[13] = -1;
  return {format: "f3d-animation-v1", nodes: [
    {parent: -1}, {parent: 0, translation: [0, 1, 0]},
    {parent: -1, translation: [2, 0, 0], weights: [0, .1, .2, .3, .4, .5, .6]},
    {parent: -1, translation: [-3, 0, 0]},
    {parent: 1, matrix: identity(), weights: [.25, .75]},
  ], skins: [{joints: [0, 1], inverseBindMatrices: [...identity(), ...inverse]}],
  instances: [{node: 2, skin: 0}, {node: 3, skin: 0}], clips};
}
function newClip(interpolation = "LINEAR") {
  const channel = (node, path, frames) => ({node, path, interpolation, times: [0, .5, 1.25, 2],
    values: interpolation === "CUBICSPLINE"
      ? frames.flatMap(v => [...v.map(() => 0), ...v, ...v.map(() => 0)]) : frames.flat()});
  return {name: "new motion", channels: [
    channel(1, "rotation", [[0,0,0,1], [0,0,.6,.8], [0,0,1,0], [0,0,0,-1]]),
    channel(0, "translation", [[0,0,0], [2,0,0], [3,2,0], [4,2,0]]),
    channel(0, "scale", [[1,1,1], [1,2,1], [2,1,2], [1,1,1]]),
    channel(2, "weights", [0, .25, .75, 1].map(t => Array.from({length: 7}, (_, i) => (i + 1) * t))),
    channel(4, "weights", [[.25,.75], [.5,0], [1,-1], [0,1]]),
  ]};
}
function state(p) {
  return {version: p.version, clip: p.clip, time: p.time, mode: p.mode,
    local: p.snapshotLocalPose(), arrays: fields.map(f => p[f].slice())};
}
function samePose(a, b) {
  for (const field of fields) assert.deepEqual(a[field], b[field], field);
}
function use(t, definition = rig()) {
  const p = createAnimationPlayer(definition); t.after(() => p.dispose()); return p;
}
const error = code => ({code: "ANIMATION_" + code});

test("installing clips preserves every live pose buffer, committed edit, root and playback marker", t => {
  const p = use(t), root = identity(); root[12] = 100;
  p.sample(.3, {rootMatrix: root});
  const matrix = identity(); matrix[14] = 7;
  p.edit([{node: 4, matrix}, {node: 2, weights: [1,2,3,4,5,6,7]}]);
  const before = state(p), references = fields.map(f => p[f]);
  const oldMetadata = p.clips, oldInstances = p.instances, oldOffsets = p.morphOffsets;
  assert.deepEqual(p.addClips([newClip(), clip(9)]), [1,2]);
  assert.deepEqual(state(p), before);
  for (let i = 0; i < fields.length; i++) assert.equal(p[fields[i]], references[i]);
  assert.equal(p.instances, oldInstances); assert.equal(p.morphOffsets, oldOffsets);
  assert.equal(p.clipVersion, 1); assert.equal(oldMetadata.length, 1);
  assert.equal(p.clips[0], oldMetadata[0]); assert.notEqual(p.clips, oldMetadata);
  p.edit([{node: 1, translation: [0,2,0]}]);
  assert.equal(p.worldMatrices[12], 101.2); // Preserved external root and sampled translation.
  assert.equal(p.snapshotLocalPose().matrices[0].matrix[14], 7);
});

for (const interpolation of ["LINEAR", "STEP", "CUBICSPLINE"]) {
  test(`late ${interpolation} channels reproduce initially imported motion and separate skin palettes exactly`, t => {
    const addition = newClip(interpolation), p = use(t), all = use(t, rig([clip(), addition]));
    const [index] = p.addClips([addition]);
    for (const loop of [false, true]) for (const time of [-2.5, 0, .2, .5, 1.75, 2, 6.75, .51, 1.25]) {
      p.sample(time, {clip: index, loop}); all.sample(time, {clip: index, loop}); samePose(p, all);
      assert.notDeepEqual(p.jointMatrices.slice(0,32), p.jointMatrices.slice(32,64));
    }
    p.reset(); all.reset(); samePose(p, all);
    p.sample(.75, {clip: 0}); all.sample(.75, {clip: 0}); samePose(p, all);
  });
  for (const mode of ["normal", "additive"]) test(`late ${interpolation} ${mode} layers share original binding and morph accumulation`, t => {
    const addition = newClip(interpolation), p = use(t), all = use(t, rig([clip(), addition]));
    p.blend([{clip: 0, time: .2, weight: .8}]); // Populate scratch before new bindings.
    p.addClips([addition]);
    for (const mask of [null, [1,.25,.5,0,.75]]) for (const weight of [0,.2,.8,1,2]) {
      const layers = [{clip: 0, time: .8, weight: .6}, {clip: 1, time: 1.75, weight, mode, mask}];
      p.blend(layers); all.blend(layers); samePose(p, all);
      p.blend(layers.toReversed()); all.blend(layers.toReversed()); samePose(p, all);
    }
  });
}

test("a clipless player admits first motion, new property bindings and wider morph scratch", t => {
  const p = use(t, rig([]));
  assert.equal(p.clips.length, 0);
  assert.deepEqual(p.addClips([newClip()]), [0]);
  p.blend([{clip: 0, time: 2, weight: .5}]);
  assert.deepEqual([...p.translations.slice(0,3)], [2,1,0]);
  assert.deepEqual([...p.morphWeights.slice(0,7)], [0.5,1.05,1.6,2.15,2.7,3.25,3.8]);
  assert.deepEqual([...p.morphWeights.slice(7)], [.125,.875]);
});

test("repeated batches append stable indices and immutable metadata snapshots", t => {
  const p = use(t), first = p.clips;
  const ids = p.addClips([{channels: []}, {channels: []}]), second = p.clips;
  assert.deepEqual(ids, [1,2]); assert.ok(Object.isFrozen(ids));
  assert.deepEqual(second.slice(1), [{name: "animation_1", duration: 0}, {name: "animation_2", duration: 0}]);
  assert.deepEqual(p.addClips([{name: "move", channels: []}]), [3]);
  assert.equal(first.length, 1); assert.equal(second.length, 3); assert.equal(p.clips.length, 4);
  assert.equal(p.clips[1], second[1]); assert.equal(p.clipVersion, 2);
  assert.ok(Object.isFrozen(p)); assert.ok(Object.isFrozen(p.clips));
  for (const info of p.clips) assert.ok(Object.isFrozen(info));
  assert.throws(() => {p.clips[0].duration = 9;}, TypeError);
  assert.throws(() => {ids[0] = 9;}, TypeError);
  p.sample(100, {clip: 1}); assert.equal(p.clip, 1); // Empty clips are legal rest-only motion.
});

test("empty additions do not invalidate metadata or either version", t => {
  const p = use(t), meta = p.clips, before = state(p);
  assert.deepEqual(p.addClips([]), []); assert.equal(p.clips, meta);
  assert.equal(p.clipVersion, 0); assert.deepEqual(state(p), before);
});

test("track arrays and names are copied, not live inputs or executable source", t => {
  const p = use(t, rig([])), input = clip();
  input.name = "'\n</script> globalThis.untrustedClipExecuted=true;";
  input.channels[0].times = new Float64Array([0,1]);
  input.channels[0].values = new Float64Array([-0,.1,.2, 4,2,0]);
  const name = input.name;
  p.addClips([input]);
  input.name = "mutated";
  input.channels[0].values.fill(999);
  structuredClone(input.channels[0].times.buffer, {transfer: [input.channels[0].times.buffer]});
  input.channels.length = 0;
  p.sample(0); assert.deepEqual([...p.translations.slice(0,3)], [-0,.1,.2]);
  p.sample(1); assert.deepEqual([...p.translations.slice(0,3)], [4,2,0]);
  assert.equal(p.clips[0].name, name); assert.equal(globalThis.untrustedClipExecuted, undefined);
});

test("input values may borrow pose output storage without changing the live frame", t => {
  const p = use(t), before = state(p);
  p.addClips([{channels: [{node: 0, path: "translation", times: [0], values: p.translations.subarray(0,3)}]}]);
  assert.deepEqual(state(p), before);
  p.translations[0] = 800;
  p.sample(0, {clip: 1}); assert.equal(p.translations[0], 0);
});

test("installation is independent across players created from the same definition", t => {
  const definition = rig(), before = structuredClone(definition), a = use(t, definition), b = use(t, definition);
  a.addClips([newClip()]); assert.equal(b.clips.length, 1); assert.equal(b.clipVersion, 0);
  assert.deepEqual(definition, before);
});

const invalidBatches = [
  undefined, null, {}, "clip", Array(1), [null], [{channels: Array(1)}],
  [{channels: [null]}], [{channels: "channels"}],
  [{channels: [ {...translation(), node: 99} ]}],
  [{channels: [ {...translation(), node: -1} ]}],
  [{channels: [ {...translation(), path: "visibility"} ]}],
  [{channels: [ {...translation(), node: 4} ]}],
  [{channels: [ {...translation(), path: "weights"} ]}],
  [{channels: [translation(), translation()]}],
  [{channels: [ {...translation(), interpolation: "UNKNOWN"} ]}],
  [{channels: [ {...translation(), times: [0,0]} ]}],
  [{channels: [ {...translation(), times: [-1,0]} ]}],
  [{channels: [ {...translation(), times: [0,Infinity]} ]}],
  [{channels: [ {...translation(), values: [0,0,0,NaN,0,0]} ]}],
  [{channels: [ {...translation(), values: [0,0]} ]}],
  [{channels: [ {...translation(), quantizedRotation: "yes"} ]}],
  [{channels: [ {node: 1, path: "rotation", times: [0], values: [0,0,0,0]} ]}],
  [{channels: [ {...translation(), interpolation: "CUBICSPLINE", times: [0]} ]}],
];
for (let i = 0; i < invalidBatches.length; i++) test(`invalid batch ${i} is atomic and does not consume clip IDs`, t => {
  const p = use(t); p.sample(.2); const meta = p.clips, before = state(p);
  const invalid = invalidBatches[i];
  assert.throws(() => p.addClips(Array.isArray(invalid) ? [newClip(), ...invalid] : invalid), AnimationPoseError);
  assert.equal(p.clips, meta); assert.equal(p.clipVersion, 0); assert.deepEqual(state(p), before);
  assert.deepEqual(p.addClips([clip(8)]), [1]); p.sample(.5, {clip: 1}); assert.equal(p.translations[0], 4);
});

test("late failing clip name conversion preserves all already installed clips and bindings", t => {
  const p = use(t); p.addClips([newClip()]); p.blend([{clip: 1, time: .7, weight: .5}]);
  const before = state(p), meta = p.clips, reason = Error("name failed");
  const input = newClip("CUBICSPLINE"); input.name = {toString() {throw reason;}};
  assert.throws(() => p.addClips([clip(30), input]), e => e === reason);
  assert.deepEqual(state(p), before); assert.equal(p.clips, meta); assert.equal(p.clipVersion, 1);
  assert.deepEqual(p.addClips([clip(10)]), [2]);
  p.sample(.5, {clip: 2}); assert.equal(p.translations[0], 5);
});

for (const operation of [p => p.addClips([]), p => p.sample(0), p => p.blend([]),
  p => p.edit([]), p => p.snapshotLocalPose(), p => p.reset(), p => p.dispose()]) {
  test("clip input callbacks cannot reenter live pose operations", t => {
    const p = use(t), before = state(p), meta = p.clips;
    assert.throws(() => p.addClips([{get channels() {operation(p); return [];}}]), error("REENTRANT"));
    assert.equal(p.clips, meta); assert.deepEqual(state(p), before); assert.equal(p.disposed, false);
    assert.deepEqual(p.addClips([clip()]), [1]);
  });
}

test("pose evaluation callbacks cannot install clips while a frame is staged", t => {
  const p = use(t), before = state(p), meta = p.clips;
  assert.throws(() => p.blend([{get clip() {p.addClips([clip()]); return 0;}, time: 0}]), error("REENTRANT"));
  assert.equal(p.clips, meta); assert.deepEqual(state(p), before);
});

for (const stage of ["before", "during"]) test(`detached pose outputs ${stage} installation prevent registry publication`, t => {
  const p = use(t), meta = p.clips;
  const detach = () => structuredClone(p.worldMatrices.buffer, {transfer: [p.worldMatrices.buffer]});
  if (stage === "before") detach();
  const input = clip();
  if (stage === "during") input.name = {toString() {detach(); return "detached";}};
  assert.throws(() => p.addClips([input]), error("OUTPUT_STORAGE"));
  assert.equal(p.clips, meta); assert.equal(p.clipVersion, 0); assert.equal(p.version, 0);
});

test("disposed players refuse even empty installations without touching input", () => {
  const p = createAnimationPlayer(rig()); p.dispose();
  assert.throws(() => p.addClips([]), error("DISPOSED"));
  assert.throws(() => p.addClips([{get channels() {throw Error("must not read");}}]), error("DISPOSED"));
});

test("clip count limit is cumulative across construction and successive batches", t => {
  const p = use(t, rig(Array.from({length: 4094}, () => ({channels: []}))));
  assert.deepEqual(p.addClips([{channels: []}]), [4094]);
  const meta = p.clips;
  assert.throws(() => p.addClips([{channels: []}, {channels: []}]), error("LIMIT"));
  assert.equal(p.clips, meta);
  assert.deepEqual(p.addClips([{channels: []}]), [4095]);
  assert.throws(() => p.addClips([{channels: []}]), error("LIMIT"));
  assert.deepEqual(p.addClips([]), []); assert.equal(p.clipVersion, 2);
});

test("keyframe component budget is cumulative and failed batches refund all staged components", t => {
  // Each filler contributes exactly 2^22 numbers including its time array.
  // Reuse caller-owned Float32 input; the player must own independent Float64 copies.
  const times = Float32Array.from({length: 1024}, (_, i) => i);
  const values = new Float32Array(1024 * 4095);
  const filler = {channels: [{node: 0, path: "weights", times, values}]};
  const p = use(t, {format: "f3d-animation-v1", nodes: [{weights: Array(4095).fill(0)}], clips: [filler, filler]});
  assert.deepEqual(p.addClips([filler]), [2]);
  const before = p.clips;
  const extra = {channels: [{node: 0, path: "translation", times: [0], values: [0,0,0]}]};
  assert.throws(() => p.addClips([filler, extra]), error("LIMIT"));
  assert.equal(p.clips, before); assert.equal(p.clipVersion, 1);
  assert.deepEqual(p.addClips([filler]), [3]); // Exact 2^24 total boundary.
  assert.throws(() => p.addClips([extra]), error("LIMIT"));
  assert.deepEqual(p.addClips([{channels: []}]), [4]);
  p.sample(.5, {clip: 3}); assert.equal(p.morphWeights.length, 4095);
});

test("malformed custom numeric iterators cannot bypass shape admission", t => {
  const p = use(t), input = clip();
  input.channels[0].values[Symbol.iterator] = function* () {yield 0;};
  assert.throws(() => p.addClips([input]), error("SHAPE"));
  assert.equal(p.clips.length, 1); assert.equal(p.clipVersion, 0);
});

test("added runtime-singular motion fails evaluation transactionally, not installation or later recovery", t => {
  const p = use(t), before = state(p);
  const [index] = p.addClips([{channels: [{node: 2, path: "scale", times: [0], values: [0,0,0]}]}]);
  assert.throws(() => p.sample(0, {clip: index}), error("SINGULAR_MESH"));
  assert.deepEqual(state(p), before); assert.equal(p.clipVersion, 1);
  p.sample(.5); assert.equal(p.translations[0], 2);
});
