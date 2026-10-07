// Actual scene bridge; source objects, program compilation and GPU are doubles.
import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import {T, fixture} from './fixtures/scene_program_fixture.mjs';
register('./fixtures/scene_program_loader.mjs', import.meta.url);
const {createGpuThreeScene} = await import('./three_scene.mjs');
const make = (f, root) => createGpuThreeScene(f.device, root, {three: T, program: f.program});

test('fullscreen mesh roots render without cloning or changing their source parent', async () => {
  const f = fixture(), parent = new T.Object3D(); parent.children.push(f.mesh); f.mesh.parent = parent;
  const geometry = f.mesh.geometry, material = f.mesh.material;
  const bridge = await make(f, f.mesh);
  assert.equal(bridge.scene, f.mesh); bridge.render(f.camera, {colorView: {}});
  assert.equal(f.device.frames[0].frame.draws.length, 1);
  assert.equal(f.mesh.parent, parent); assert.deepEqual(parent.children, [f.mesh]);
  assert.equal(f.mesh.geometry, geometry); assert.equal(f.mesh.material, material);
  bridge.dispose(); assert.equal(f.mesh.parent, parent); assert.equal(material.listeners.get('dispose').size, 0);
});
test('non-Scene roots use empty scene effects while preserving child traversal', async () => {
  const f = fixture(), group = new T.Object3D(); group.children.push(f.mesh);
  // WebGLRenderer.getProgram uses its empty Scene for non-Scene root inputs.
  group.background = {isColor: true, r: 1, g: 0, b: 0}; group.fog = {unsupported: true}; group.environment = {};
  group.overrideMaterial = {unsupported: true};
  const bridge = await make(f, group); bridge.render(f.camera, {colorView: {}, clearColor: [0, 0, 0, 0]});
  assert.equal(f.device.frames[0].frame.draws.length, 1); assert.deepEqual(f.device.frames[0].frame.clearColor, [0, 0, 0, 0]);
  bridge.dispose();
});
test('root world-matrix update boundary is honored exactly once and never during preparation', async () => {
  const f = fixture(); let calls = 0;
  f.mesh.matrixWorldAutoUpdate = true; f.mesh.updateMatrixWorld = () => { calls++; };
  const bridge = await make(f, f.mesh); assert.equal(calls, 0);
  bridge.render(f.camera, {colorView: {}}); assert.equal(calls, 1);
  f.mesh.matrixWorldAutoUpdate = false; bridge.render(f.camera, {colorView: {}}); assert.equal(calls, 1);
  bridge.dispose();
});
test('program callbacks receive the original root, camera, geometry and material', async () => {
  const f = fixture(), group = new T.Object3D(), seen = []; group.children.push(f.mesh);
  const renderer = {}; f.program.state = () => ({toneMapping: 0, renderer});
  for (const name of ['onBeforeRender', 'onAfterRender']) f.mesh[name] = function(r, root, camera, geometry, material) {
    seen.push(name); assert.equal(this, f.mesh); assert.equal(r, renderer); assert.equal(root, group);
    assert.equal(camera, f.camera); assert.equal(geometry, f.mesh.geometry); assert.equal(material, f.mesh.material);
  };
  const bridge = await make(f, group); assert.deepEqual(seen, []);
  bridge.render(f.camera, {colorView: {}}); assert.deepEqual(seen, ['onBeforeRender', 'onAfterRender']); bridge.dispose();
});
test('root visibility and layer membership keep their source meaning', async () => {
  const f = fixture(), group = new T.Object3D(); group.children.push(f.mesh);
  const bridge = await make(f, group); group.visible = false; bridge.render(f.camera, {colorView: {}});
  assert.equal(f.device.frames.at(-1).frame.draws.length, 0);
  group.visible = true; f.mesh.layers.test = () => false; bridge.render(f.camera, {colorView: {}});
  assert.equal(f.device.frames.at(-1).frame.draws.length, 0);
  f.mesh.layers.test = () => true; bridge.render(f.camera, {colorView: {}});
  assert.equal(f.device.frames.at(-1).frame.draws.length, 1); bridge.dispose();
});
test('replacing a fullscreen material reuses original mesh identity and re-prepares its binding', async () => {
  const f = fixture(), bridge = await make(f, f.mesh), old = f.mesh.material;
  f.mesh.material = new T.Material(); f.mesh.material.uniforms.value.value = 9;
  assert.throws(() => bridge.render(f.camera, {colorView: {}}), {code: 'THREE_SCENE_PREPARE'});
  await bridge.prepare(); bridge.render(f.camera, {colorView: {}});
  assert.equal(new DataView(f.device.frames.at(-1).frame.draws[0].programUniforms.buffer).getFloat32(0, true), 9);
  assert.equal(bridge.scene, f.mesh); assert.equal(old.listeners.get('dispose').size, 0); bridge.dispose();
});
test('foreign or missing roots fail before renderer allocations', async () => {
  const f = fixture(); for (const root of [null, {}, {isScene: true}])
    await assert.rejects(make(f, root), {code: 'THREE_SCENE_SOURCE'});
  assert.equal(f.device.registrations.length, 0);
});
