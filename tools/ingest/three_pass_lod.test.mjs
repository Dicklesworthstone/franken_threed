/** Pinned LOD.update is the oracle for selection and hysteresis. These tests
 * exercise the actual material/instance capture owners, without a GPU. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {captureThreeMaterialState, captureThreeLodSource} from './three_material_snapshot.mjs';
import {createThreePassInstances} from './three_pass_instances.mjs';

const retainedLodSource = captureThreeLodSource(T);
const options = (camera, lodSource = retainedLodSource) => ({camera, selectionOnly: true,
  allowInstances: true, allowDeformations: true, allowLod: true, lodSource});
function fixture(t) {
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(60, 1, .1, 100);
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicMaterial(), lod = new T.LOD();
  const meshes = [1, 2, 3].map(count => {
    const mesh = new T.InstancedMesh(geometry, material, count); mesh.frustumCulled = false; return mesh;
  });
  meshes.forEach((mesh, index) => lod.addLevel(mesh, index * 10, .2)); scene.add(lod);
  const owner = createThreePassInstances(T), frames = [];
  t.after(() => { for (const frame of frames) frame.instanceFrame?.release(); owner.dispose();
    for (const mesh of meshes) mesh.dispose(); geometry.dispose(); material.dispose(); });
  return {scene, camera, lod, meshes, owner, frames, lodSource: captureThreeLodSource(T)};
}
function capture(f, exclusions = null) {
  const frame = captureThreeMaterialState(T, f.scene, {...options(f.camera, f.lodSource), exclusions});
  assert.ok(frame?.cameraSafe(), 'ordinary source inputs admit a required pass');
  f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  frame.instanceFrame = f.owner.capture(f.scene, f.camera, frame);
  assert.ok(frame.instanceFrame); f.frames.push(frame); frame.instanceFrame.publishLod(); return frame;
}
const visible = lod => lod.levels.map(level => level.object.visible);
const selected = (frame, objects) => objects.map(object => !!frame.objects.get(frame.instanceFrame.objects.get(object)));

test('accepted automatic LOD follows source hysteresis and camera zoom while retaining every instance packet', t => {
  const sourceUpdate = T.LOD.prototype.update;
  const updates = [];
  T.LOD.prototype.update = function (camera) { updates.push(Object.getPrototypeOf(this)); return sourceUpdate.call(this, camera); };
  t.after(() => { T.LOD.prototype.update = sourceUpdate; });
  const f = fixture(t), reference = fixture(t);
  const expected = [];
  // 12 -> 9 stays on level one through hysteresis; 7 returns to level zero.
  // The zoom change at the same position also changes the effective distance.
  for (const [index, [distance, zoom]] of [[4, 1], [12, 1], [9, 1], [7, 1], [22, 1], [17, 1], [15, 1], [15, 2]].entries()) {
    for (const current of [f, reference]) {
      current.camera.position.z = distance; current.camera.zoom = zoom; current.camera.updateProjectionMatrix();
      current.scene.updateMatrixWorld(); current.camera.updateMatrixWorld();
    }
    sourceUpdate.call(reference.lod, reference.camera);
    for (const [level, mesh] of f.meshes.entries()) {
      mesh.setMatrixAt(0, new T.Matrix4().makeTranslation(index + level * 10, 0, 0)); mesh.instanceMatrix.needsUpdate = true;
      mesh.count = index % 2 ? 1 : level + 1;
    }
    const frame = capture(f), level = reference.lod.getCurrentLevel();
    assert.deepEqual(visible(f.lod), visible(reference.lod));
    assert.equal(f.lod.getCurrentLevel(), level);
    assert.deepEqual(selected(frame, f.meshes), visible(reference.lod));
    expected.push({frame, level, count: f.meshes[level].count, translation: index + level * 10});
  }
  assert.deepEqual(expected.map(value => value.level), [0, 1, 1, 0, 2, 2, 1, 0]);
  assert.equal(updates.length, expected.length, 'one source LOD update for each accepted call');
  f.lod.levels.reverse(); f.lod.remove(f.meshes[0]); f.lod._currentLevel = 77;
  for (const mesh of f.meshes) { mesh.visible = true; mesh.count = 0; mesh.instanceMatrix.array.fill(99); }
  f.camera.position.z = 99;
  for (const {frame, level, count, translation} of expected) {
    assert.equal(frame.cameraSafe(), true, 'ordinary later level and topology edits cannot retarget the accepted pass');
    frame.instanceFrame.install();
    const view = frame.instanceFrame.objects.get(f.meshes[level]);
    assert.equal(view.count, count); assert.equal(view.instanceMatrix.array[12], translation);
    assert.deepEqual(selected(frame, f.meshes), f.meshes.map((_, index) => index === level));
  }
  assert.equal(updates.length, expected.length, 'installing queued packets never replays LOD.update');
  assert.equal(f.lod.getCurrentLevel(), 77); assert.deepEqual(f.meshes.map(mesh => mesh.visible), [true, true, true]);
});

test('nested automatic LOD matches source projection order, inherited visibility and independent child layers', t => {
  const sourceUpdate = T.LOD.prototype.update, updates = [];
  T.LOD.prototype.update = function (camera) { updates.push(Object.getPrototypeOf(this)); return sourceUpdate.call(this, camera); };
  t.after(() => { T.LOD.prototype.update = sourceUpdate; });
  const f = fixture(t), outer = f.lod, inner = new T.LOD();
  outer.remove(f.meshes[0]); outer.levels[0].object = inner; outer.add(inner);
  inner.addLevel(f.meshes[0], 0); const alternate = new T.InstancedMesh(f.meshes[0].geometry, f.meshes[0].material, 1);
  alternate.frustumCulled = false; inner.addLevel(alternate, 3); t.after(() => alternate.dispose());
  f.camera.position.z = 5; let frame = capture(f);
  assert.deepEqual(updates, [outer, inner]);
  assert.deepEqual(selected(frame, [f.meshes[0], alternate, f.meshes[1], f.meshes[2]]), [false, true, false, false]);
  updates.length = 0; f.camera.position.z = 12; frame = capture(f);
  assert.deepEqual(updates, [outer], 'the selected outer level hides the nested LOD before its traversal');
  assert.deepEqual(selected(frame, [alternate, f.meshes[1]]), [false, true]);
  updates.length = 0; outer.layers.set(1); inner.visible = true; f.meshes[1].visible = false;
  f.camera.position.z = 1; frame = capture(f);
  assert.deepEqual(updates, [inner], 'a masked parent LOD still visits independently layered children');
  assert.deepEqual(selected(frame, [f.meshes[0], alternate, f.meshes[1]]), [true, false, false]);
  updates.length = 0; f.scene.visible = false; frame = capture(f);
  assert.deepEqual(updates, []); assert.deepEqual(selected(frame, [...f.meshes, alternate]), [false, false, false, false]);
  f.scene.visible = true; frame = capture(f, [inner]);
  assert.deepEqual(updates, [], 'an excluded subtree does not acquire source visibility effects');
  assert.deepEqual(selected(frame, [f.meshes[0], alternate]), [false, false]);
});

test('automatic LOD capture refuses accessor-backed and external levels before application effects', t => {
  const f = fixture(t); let reads = 0;
  assert.equal(captureThreeMaterialState(T, f.scene, {...options(f.camera), allowLod: false}), null,
    'ordinary instanced rendering retains its live LOD projection path');
  const level = f.lod.levels[1];
  Object.defineProperty(level, 'distance', {get() { reads++; return 10; }, configurable: true});
  assert.equal(captureThreeMaterialState(T, f.scene, options(f.camera)), null); assert.equal(reads, 0);
  Object.defineProperty(level, 'distance', {value: 10, writable: true, configurable: true});
  level.object = new T.Object3D();
  assert.equal(captureThreeMaterialState(T, f.scene, options(f.camera)), null);
  level.object = f.meshes[1]; f.lod.update = () => { reads++; };
  assert.equal(captureThreeMaterialState(T, f.scene, options(f.camera)), null); assert.equal(reads, 0);
  f.lod.update = T.LOD.prototype.update;
  Object.defineProperty(f.meshes[1], 'visible', {get() { reads++; return true; }, configurable: true});
  assert.equal(captureThreeMaterialState(T, f.scene, options(f.camera)), null); assert.equal(reads, 0);
});

test('LOD preparation publishes no visibility or hysteresis state when packet budgets refuse or a packet is released', t => {
  const f = fixture(t); f.camera.position.z = 12; f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const before = {level: f.lod.getCurrentLevel(), visibility: visible(f.lod)};
  const limited = createThreePassInstances(T, {maxBytes: 1}); t.after(() => limited.dispose());
  for (const [owner, limits] of [[f.owner, {maxValues: 1}], [limited, {}]]) {
    const frame = captureThreeMaterialState(T, f.scene, options(f.camera)); assert.ok(frame?.cameraSafe());
    assert.throws(() => owner.capture(f.scene, f.camera, frame, limits), {code: 'F3D_RENDERER_UNSUPPORTED'});
    assert.deepEqual({level: f.lod.getCurrentLevel(), visibility: visible(f.lod)}, before);
    assert.equal(owner.pendingBytes, 0);
  }
  const frame = captureThreeMaterialState(T, f.scene, options(f.camera));
  const packet = f.owner.capture(f.scene, f.camera, frame); packet.release(); packet.publishLod();
  assert.deepEqual({level: f.lod.getCurrentLevel(), visibility: visible(f.lod)}, before);
  assert.equal(f.owner.pendingBytes, 0);
});

test('LOD source dependencies and numeric inputs reject callbacks without invoking them', t => {
  const f = fixture(t); f.camera.position.z = 12; f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const frame = captureThreeMaterialState(T, f.scene, options(f.camera)); let calls = 0;
  for (const [object, key] of [[T.Vector3.prototype, 'setFromMatrixPosition'], [T.Vector3.prototype, 'distanceTo'],
    [T.Vector3.prototype, 'distanceToSquared'], [Math, 'sqrt']]) {
    const original = object[key]; object[key] = () => { calls++; };
    try {
      assert.equal(frame.cameraSafe(), false);
      assert.throws(() => f.owner.capture(f.scene, f.camera, frame, {maxValues: 1}), {code: 'F3D_RENDERER_UNSUPPORTED'});
    } finally { object[key] = original; }
  }
  const math = Object.getOwnPropertyDescriptor(globalThis, 'Math'); let safe;
  Object.defineProperty(globalThis, 'Math', {get() { calls++; return math.value; }, configurable: true});
  try { safe = retainedLodSource.safe(); } finally { Object.defineProperty(globalThis, 'Math', math); }
  assert.equal(safe, false); assert.equal(calls, 0);
  const elements = f.lod.matrixWorld.elements, original = Object.getOwnPropertyDescriptor(elements, '12');
  Object.defineProperty(elements, '12', {get() { calls++; return 0; }, configurable: true});
  try { assert.equal(captureThreeMaterialState(T, f.scene, options(f.camera)), null); }
  finally { Object.defineProperty(elements, '12', original); }
  f.camera.zoom = {valueOf() { calls++; return 1; }};
  assert.equal(captureThreeMaterialState(T, f.scene, options(f.camera)), null); f.camera.zoom = 1;
  assert.equal(calls, 0); assert.equal(f.lod.getCurrentLevel(), 0); assert.deepEqual(visible(f.lod), [true, true, true]);
});

test('private LOD invocation does not observe an application property on the retained method', t => {
  const f = fixture(t); f.camera.position.z = 12; let calls = 0;
  const update = retainedLodSource.update, descriptor = Object.getOwnPropertyDescriptor(update, 'call');
  Object.defineProperty(update, 'call', {value() { calls++; }, configurable: true});
  try { capture(f); }
  finally { if (descriptor) Object.defineProperty(update, 'call', descriptor); else delete update.call; }
  assert.equal(calls, 0); assert.equal(f.lod.getCurrentLevel(), 1); assert.deepEqual(visible(f.lod), [false, true, false]);
});

test('zero and one level LODs preserve the source no-op visibility and current level', t => {
  for (const count of [0, 1]) {
    const f = fixture(t); f.lod.levels.length = count; f.lod._currentLevel = 7; f.meshes[0].visible = false;
    f.camera.position.z = 12; const frame = capture(f);
    frame.instanceFrame.publishLod(); frame.instanceFrame.install(); frame.instanceFrame.publishLod();
    assert.equal(f.lod.getCurrentLevel(), 7); assert.deepEqual(f.meshes.map(mesh => mesh.visible), [false, true, true]);
    assert.deepEqual(selected(frame, f.meshes), [false, true, true]);
  }
});

test('source LOD can reveal an excluded parent while independently excluded children stay hidden', t => {
  const f = fixture(t), group = new T.Group(), child = new T.InstancedMesh(f.meshes[0].geometry, f.meshes[0].material, 1);
  child.frustumCulled = false; t.after(() => child.dispose());
  f.lod.remove(f.meshes[0]); f.lod.levels[0].object = group; f.lod.add(group); group.add(f.meshes[0], child);
  group.visible = child.visible = false; f.camera.position.z = 5;
  const frame = capture(f, [group, child]);
  assert.equal(group.visible, true); assert.equal(child.visible, false);
  assert.deepEqual(selected(frame, [f.meshes[0], child, f.meshes[1], f.meshes[2]]), [true, false, false, false]);
  assert.equal(frame.excludedObjects.has(group), false); assert.equal(frame.excludedObjects.has(f.meshes[0]), false);
  assert.equal(frame.excludedObjects.has(child), true);
});
