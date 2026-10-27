/** Native source renderObject inheritance, including Inspector's overdraw material.
 * The source renderer supplies the synchronous lifecycle oracle. Recorded GPU
 * commands establish per-draw capture; browser checks separately execute WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import WebGPUPipelineUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUPipelineUtils.js';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const inherited = ['positionNode', 'alphaTest', 'alphaMap', 'displacementMap',
  'displacementScale', 'displacementBias', 'transparent', 'side'];
const state = material => Object.fromEntries(inherited.map(key => [key, material[key]]));

function fixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 32, height: 24, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 24], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  function mesh(material, parent = scene) {
    const geometry = new T.PlaneGeometry(), object = new T.Mesh(geometry, material);
    object.frustumCulled = false; parent.add(object);
    t.after(() => { geometry.dispose(); material.dispose(); }); return object;
  }
  return {renderer, device, scene, camera, captures, mesh};
}

test('Inspector overdraw inherits Toon source fields and keeps the original shared material identity', async t => {
  const f = fixture(t), original = new T.MeshToonNodeMaterial({side: T.DoubleSide});
  const object = f.mesh(original), override = new T.MeshBasicNodeMaterial({
    colorNode: N.vec3(.25), blending: T.AdditiveBlending, depthTest: true, depthWrite: true});
  t.after(() => override.dispose());
  f.scene.overrideMaterial = override;
  const before = state(override), expected = [], source = {_currentSourceMaterial: null,
    _handleObjectFunction(_object, material) { expected.push(state(material)); }};
  T.Renderer.prototype.renderObject.call(source, object, f.scene, f.camera, object.geometry, original, null, null);
  const after = state(override);
  Object.assign(override, before);
  const seen = []; override.colorNode = N.uniform(.25).onObjectUpdate(frame => {
    seen.push({material: frame.material, source: f.renderer._currentSourceMaterial, state: state(frame.material)}); return .25;
  });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(seen.map(value => value.state), expected);
  assert.ok(seen.every(value => value.material === override && value.source === original));
  assert.deepEqual(state(override), after, 'restoration and inherited alpha/transparent residue match source');
  assert.equal(object.material, original); assert.equal(f.scene.overrideMaterial, override);
  assert.equal(f.captures[0].draws.length, 1); assert.equal(f.renderer.info.f3d.nativeDrawCalls, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._pending.length, 0);
});

test('shared overrides inherit each material group independently and keep accepted uniform snapshots', async t => {
  const f = fixture(t), a = new T.MeshBasicNodeMaterial({alphaTest: .25}), b = new T.MeshBasicNodeMaterial({alphaTest: .75});
  a.positionNode = N.positionLocal.add(N.vec3(.125, 0, 0));
  b.positionNode = N.positionLocal.add(N.vec3(.375, 0, 0));
  const object = f.mesh(a); t.after(() => b.dispose()); object.material = [a, b];
  object.geometry.clearGroups(); object.geometry.addGroup(0, 3, 0); object.geometry.addGroup(3, 3, 1);
  const observed = [], override = new T.MeshBasicNodeMaterial(); t.after(() => override.dispose());
  override.colorNode = N.uniform(.5).onObjectUpdate(frame => {
    observed.push({material: frame.material, source: f.renderer._currentSourceMaterial,
      alphaTest: frame.material.alphaTest, positionNode: frame.material.positionNode}); return .5;
  });
  f.scene.overrideMaterial = override;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const first = f.captures.at(-1), firstBytes = first.draws.map(draw => draw.nodeUniforms.map(bytes => bytes.slice()));
  a.alphaTest = .5; b.alphaTest = .875; f.renderer.render(f.scene, f.camera);
  const second = f.captures.at(-1);
  assert.deepEqual(observed.map(value => value.alphaTest), [.25, .75, .5, .875]);
  assert.deepEqual(observed.map(value => value.source), [a, b, a, b]);
  assert.deepEqual(observed.map(value => value.positionNode), [a.positionNode, b.positionNode, a.positionNode, b.positionNode]);
  assert.ok(observed.every(value => value.material === override));
  assert.equal(override.positionNode, null, 'the shared override returns to its own position graph');
  assert.equal(override.alphaTest, .875, 'source deliberately retains the last inherited alpha threshold');
  assert.deepEqual(first.draws.map(draw => draw.nodeUniforms), firstBytes);
  assert.notDeepEqual(second.draws.map(draw => draw.nodeUniforms), firstBytes);
  assert.deepEqual(first.draws.map(draw => [draw.first, draw.count]), [[0, 3], [3, 3]]);
  await f.renderer.waitForGPU(); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native override selection preserves allowOverride false and original visibility', async t => {
  const f = fixture(t), original = new T.MeshToonNodeMaterial(), excluded = new T.MeshBasicNodeMaterial();
  excluded.colorNode = N.vec3(.5);
  excluded.allowOverride = false;
  const a = f.mesh(original), b = f.mesh(excluded), hidden = f.mesh(new T.MeshToonNodeMaterial()); hidden.material.visible = false;
  const override = new T.MeshBasicNodeMaterial({colorNode: N.vec3(.25)}); override.visible = false;
  t.after(() => override.dispose()); f.scene.overrideMaterial = override;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(f.captures[0].draws.map(draw => [draw.object, draw.material]), [[a, override], [b, excluded]]);
  assert.equal(override.visible, false, 'the source list uses original visibility');
  assert.equal(b.material, excluded); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('opaque overrides retain the source nullable transparency and native blending for both alpha conventions', async t => {
  for (const premultipliedAlpha of [false, true]) {
    const f = fixture(t), original = new T.MeshToonNodeMaterial(); f.mesh(original);
    const override = new T.MeshBasicNodeMaterial({colorNode: N.vec3(.25), opacity: .5, premultipliedAlpha});
    t.after(() => override.dispose()); f.scene.overrideMaterial = override;
    await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    assert.equal(override.transparent, null);
    assert.deepEqual(f.captures[0].draws[0].binding.raster.blend, new WebGPUPipelineUtils({})._getBlending(override));
  }
});

test('cold and warm override bundles follow the source refresh path without replaying object callbacks', async t => {
  const f = fixture(t), group = new T.BundleGroup(); f.scene.add(group);
  const original = new T.MeshToonNodeMaterial(), object = f.mesh(original, group);
  const override = new T.MeshBasicNodeMaterial({colorNode: N.vec3(.25), blending: T.AdditiveBlending});
  t.after(() => override.dispose()); f.scene.overrideMaterial = override;
  let before = 0, after = 0, updates = 0;
  object.onBeforeRender = () => { before++; }; object.onAfterRender = () => { after++; };
  override.colorNode = N.uniform(.25).onObjectUpdate(() => { updates++; return .25; });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual([before, after, updates], [1, 1, 2]);
  assert.equal(f.captures[1].draws[0].bundleWarm, true);
  assert.equal(f.captures[1].draws[0].material, override);
  group.needsUpdate = true; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual([before, after, updates], [2, 2, 3]);
  assert.equal(object.material, original); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
