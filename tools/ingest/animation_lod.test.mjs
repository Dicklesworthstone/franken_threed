import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationLod} from './animation_lod.mjs';
const code = name => ({code: `ANIMATION_LOD_${name}`});
const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
function poseAt(...positions) {
  const pose = {nodeCount: positions.length, version: 0, disposed: false, worldMatrices: new Float64Array(positions.length * 16)};
  positions.forEach((p, i) => {pose.worldMatrices.set(identity(), i * 16); pose.worldMatrices.set(p, i * 16 + 12);});
  return pose;
}
const config = (hysteresis = 0) => ({groups: [{node: 0, levels: [
  {distance: 0, drawIndices: [0, 2]}, {distance: 10, hysteresis, drawIndices: [1]}, {distance: 20, hysteresis, drawIndices: [3]},
]}]});
const camera = (z, extra = {}) => ({position: [0,0,z], ...extra});
function select(lod, z, extra) {const selection = lod.prepare(camera(z, extra)); lod.commit(selection); return selection;}

test('whole primitive groups switch together; ungrouped draws retain source order', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config());
  assert.deepEqual(select(lod, 0).drawIndices, [0,2,4]);
  assert.deepEqual(select(lod, 10).drawIndices, [1,4]);
  assert.deepEqual(select(lod, 20).drawIndices, [3,4]);
  assert.deepEqual(select(lod, 100).drawIndices, [3,4]);
  assert.equal(lod.lastSelection.suppressedDraws, 3); assert.equal(lod.lastSelection.selectedDraws, 2);
});

test('distance is Euclidean from current world translation, including animated parents', () => {
  const pose = poseAt([3,4,0]), lod = createAnimationLod(pose, 5, config());
  assert.equal(select(lod, 0).groups[0].distance, 5);
  pose.worldMatrices.set([6,8,0], 12); pose.version++;
  const result = select(lod, 0); assert.equal(result.groups[0].distance, 10); assert.equal(result.groups[0].level, 1);
  assert.equal(result.poseVersion, 1);
});

test('nonuniform scale, shear and reflection do not scale the world-distance thresholds', () => {
  const pose = poseAt([0,0,10]); pose.worldMatrices[0] = -20; pose.worldMatrices[4] = 7; pose.worldMatrices[5] = 0.01;
  const lod = createAnimationLod(pose, 5, config()); assert.equal(select(lod, 0).groups[0].level, 1);
});

test('camera zoom divides distance for perspective and orthographic camera callers', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config());
  assert.equal(select(lod, 15, {zoom: 2}).groups[0].level, 0);
  assert.equal(select(lod, 15, {zoom: 0.5}).groups[0].level, 2);
});

test('hysteresis avoids rapid switches when approaching the active coarse threshold', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config(0.2));
  assert.equal(select(lod, 10).groups[0].level, 1);
  assert.equal(select(lod, 9).groups[0].level, 1);
  assert.equal(select(lod, 8).groups[0].level, 1);
  assert.equal(select(lod, 7.99).groups[0].level, 0);
  assert.equal(select(lod, 9).groups[0].level, 0);
});

test('each camera keeps independent history even when calls alternate', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config(0.2));
  select(lod, 12, {key: 'main'}); select(lod, 3, {key: 'map'});
  assert.equal(select(lod, 9, {key: 'main'}).groups[0].level, 1);
  assert.equal(select(lod, 9, {key: 'map'}).groups[0].level, 0);
  assert.equal(lod.cameraCount, 2);
});

test('uncommitted render candidates neither create history nor advance hysteresis', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config(0.2));
  lod.prepare(camera(12));
  assert.equal(lod.cameraCount, 0); assert.equal(lod.lastSelection, null);
  assert.equal(select(lod, 9).groups[0].level, 0);
  const before = lod.lastSelection; lod.prepare(camera(12)); assert.equal(lod.lastSelection, before);
});

test('commit refuses foreign, repeated and superseded transactions', () => {
  const pose = poseAt([0,0,0]), a = createAnimationLod(pose, 5, config()), b = createAnimationLod(pose, 5, config());
  const first = a.prepare(camera(0)), second = a.prepare(camera(11));
  assert.throws(() => b.commit(first), code('TRANSACTION'));
  a.commit(second); assert.throws(() => a.commit(first), code('TRANSACTION')); assert.throws(() => a.commit(second), code('TRANSACTION'));
  assert.equal(a.lastSelection, second); assert.equal(b.cameraCount, 0);
});

test('pose changes invalidate a prepared selection without changing acknowledged state', () => {
  const pose = poseAt([0,0,0]), lod = createAnimationLod(pose, 5, config());
  const old = select(lod, 0), pending = lod.prepare(camera(12)); pose.version++;
  assert.throws(() => lod.commit(pending), code('CHANGED')); assert.equal(lod.lastSelection, old);
});

test('camera capacity is explicit and reset releases a history without altering last-frame evidence', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, {...config(0.2), maxCameras: 1});
  const previous = select(lod, 12, {key: 0});
  assert.throws(() => select(lod, 9, {key: 1}), code('CAMERA_LIMIT'));
  assert.equal(lod.lastSelection, previous); lod.resetCamera(0); assert.equal(lod.cameraCount, 0);
  assert.equal(lod.lastSelection, previous); assert.equal(select(lod, 9, {key: 1}).groups[0].level, 0);
});

test('configuration and returned state are independent immutable snapshots', () => {
  const input = config(), lod = createAnimationLod(poseAt([0,0,0]), 5, input);
  input.groups[0].levels[1].distance = 100; input.groups[0].levels[0].drawIndices[0] = 4;
  const result = select(lod, 11); assert.equal(result.groups[0].level, 1); assert.deepEqual(result.drawIndices, [1,4]);
  for (const value of [result, result.groups, result.groups[0], result.drawIndices, result.cameraPosition, lod.groups, lod.groups[0].levels[0].drawIndices]) assert.ok(Object.isFrozen(value));
  assert.throws(() => {result.groups[0].level = 0;});
  select(lod, 0); assert.equal(result.groups[0].level, 1);
});

test('multiple groups select independently using their own animated origin nodes', () => {
  const lod = createAnimationLod(poseAt([0,0,0], [0,0,100]), 5, {groups: [
    {node: 0, levels: [{distance: 0, drawIndices: [2]}, {distance: 10, drawIndices: [0]}]},
    {node: 1, levels: [{distance: 0, drawIndices: [1]}, {distance: 10, drawIndices: [3]}]},
  ]});
  const result = select(lod, 0); assert.deepEqual(result.drawIndices, [2,3,4]);
  assert.deepEqual(result.groups.map(g => g.level), [0,1]);
});

test('one-level groups work without inventing a missing lower-detail alternative', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 2, {groups: [{node: 0, levels: [{distance: 0, drawIndices: [1]}]}]});
  assert.deepEqual(select(lod, 1e100).drawIndices, [0,1]);
});

test('invalid, overlapping and unbounded group descriptions fail construction', () => {
  const pose = poseAt([0,0,0]);
  const invalid = [null, {}, {groups: []}, {groups: Array(6).fill({})}, {...config(), maxCameras: 0}, {...config(), maxCameras: 65},
    {...config(), unknown: true}, {groups: [{node: 1, levels: []}]}, {groups: [{node: 0, levels: []}]}];
  for (const input of invalid) assert.throws(() => createAnimationLod(pose, 5, input));
  for (const edit of [
    c => {c.groups[0].levels[0].distance = 1;}, c => {c.groups[0].levels[1].distance = 0;},
    c => {c.groups[0].levels[1].distance = Infinity;}, c => {c.groups[0].levels[1].hysteresis = 1;},
    c => {c.groups[0].levels[1].hysteresis = -1;}, c => {c.groups[0].levels[1].drawIndices = [0];},
    c => {c.groups[0].levels[1].drawIndices = [5];}, c => {c.groups[0].levels[1].drawIndices = [];},
    c => {c.groups[0].levels[0].drawIndices = [0,0];},
    c => {c.groups.push({node: 0, levels: [{distance: 0, drawIndices: [0]}]});},
  ]) {const input = config(); edit(input); assert.throws(() => createAnimationLod(pose, 5, input));}
});

test('invalid camera parameters leave the previous successful selection intact', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config()), before = select(lod, 0);
  for (const value of [null, {}, camera(0, {zoom: 0}), camera(0, {zoom: Infinity}), camera(0, {key: {}}), camera(0, {key: ''}),
    camera(0, {key: 'x'.repeat(129)}), {position: [0,0,NaN]}, {position: [0,0]}, {...camera(0), typo: true}]) assert.throws(() => lod.prepare(value));
  assert.equal(lod.lastSelection, before);
});

test('world matrix storage and finite affine transforms are validated on every selection', () => {
  const pose = poseAt([0,0,0]), lod = createAnimationLod(pose, 5, config()), before = select(lod, 0);
  for (const [at, value] of [[3,1], [15,0], [0,NaN], [12,Infinity]]) {
    const old = pose.worldMatrices[at]; pose.worldMatrices[at] = value; assert.throws(() => lod.prepare(camera(0))); pose.worldMatrices[at] = old;
  }
  const old = pose.worldMatrices; pose.worldMatrices = new Float64Array(2); assert.throws(() => lod.prepare(camera(0)), code('STORAGE')); pose.worldMatrices = old;
  pose.worldMatrices[12] = -Number.MAX_VALUE; assert.throws(() => lod.prepare({position: [Number.MAX_VALUE,0,0]}), code('VALUE'));
  assert.equal(lod.lastSelection, before);
});

test('shared, resizable, detached and DataView camera storage are refused', () => {
  const pose = poseAt([0,0,0]), lod = createAnimationLod(pose, 5, config());
  const detached = new Float64Array(3); structuredClone(detached.buffer, {transfer: [detached.buffer]});
  const arrays = [new Float64Array(new SharedArrayBuffer(24)), new Float64Array(new ArrayBuffer(24, {maxByteLength: 48})), detached, new DataView(new ArrayBuffer(24))];
  for (const position of arrays) assert.throws(() => lod.prepare({position}));
  structuredClone(pose.worldMatrices.buffer, {transfer: [pose.worldMatrices.buffer]});
  assert.throws(() => lod.prepare(camera(0)), code('STORAGE'));
});

test('camera getters cannot recursively select, commit, reset or dispose', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config()), pending = lod.prepare(camera(0));
  for (const operation of [() => lod.prepare(camera(0)), () => lod.commit(pending), () => lod.resetCamera(), () => lod.dispose()]) {
    assert.throws(() => lod.prepare({get position() {operation(); return [0,0,0];}}), code('REENTRANT'));
  }
  assert.equal(lod.cameraCount, 0); lod.commit(pending); assert.equal(lod.cameraCount, 1);
});

test('reset invalidates pending candidates, and disposed/changed poses are rejected', () => {
  const pose = poseAt([0,0,0]), lod = createAnimationLod(pose, 5, config());
  const pending = lod.prepare(camera(0)); lod.resetCamera(); assert.throws(() => lod.commit(pending), code('TRANSACTION'));
  pose.disposed = true; assert.throws(() => lod.prepare(camera(0)), code('POSE')); pose.disposed = false;
  pose.nodeCount++; assert.throws(() => lod.prepare(camera(0)), code('POSE')); pose.nodeCount--;
  lod.dispose(); lod.dispose(); assert.equal(lod.lastSelection, null); assert.equal(lod.cameraCount, 0);
  assert.throws(() => lod.prepare(camera(0)), code('DISPOSED')); assert.equal(pose.disposed, false);
});

test('deterministic camera trajectories agree with a separate visibility-threshold oracle', () => {
  const lod = createAnimationLod(poseAt([0,0,0]), 5, config(0.2)), states = new Map();
  let seed = 123456789;
  for (let n = 0; n < 1000; n++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const key = n % 4, distance = (seed % 3200) / 100, old = states.get(key) ?? [true,false,false];
    const adjusted = [0, 10 * (old[1] ? 0.8 : 1), 20 * (old[2] ? 0.8 : 1)];
    const expected = distance < adjusted[1] ? 0 : distance < adjusted[2] ? 1 : 2;
    assert.equal(select(lod, distance, {key}).groups[0].level, expected);
    states.set(key, [0,1,2].map(i => i === expected));
  }
});

test('pose edits triggered by camera getters cannot silently select an unacknowledged version', () => {
  const pose = poseAt([0,0,0]), lod = createAnimationLod(pose, 5, config());
  assert.throws(() => lod.prepare({get position() {pose.version++; return [0,0,0];}}), code('CHANGED'));
  assert.equal(lod.cameraCount, 0); assert.equal(lod.lastSelection, null);
});
