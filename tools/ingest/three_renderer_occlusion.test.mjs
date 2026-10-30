/** Recorded native occlusion commands. Browser execution separately compares
 * actual depth-query results with pinned WebGPURenderer. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t, {bundles = false} = {}) {
  const device = geometryDevice(), querySets = [], maps = [], queryEvents = [];
  device.limits.maxTextureDimension2D = 8192;
  device.features = new Set(); device.samples = [0n, 7n];
  device.createTexture = descriptor => ({...descriptor, width: descriptor.size[0], height: descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() {this.destroyed = true;}});
  device.createSampler = value => value;
  device.createQuerySet = descriptor => {
    const set = {...descriptor, values: device.samples.slice(), destroy() {this.destroyed = true;}};
    querySets.push(set); return set;
  };
  const allocate = device.createBuffer;
  device.createBuffer = descriptor => {
    const buffer = allocate(descriptor);
    if (descriptor.usage & 1) buffer.mapAsync = async () => { maps.push(buffer); if (device.mapError) throw device.mapError; };
    return buffer;
  };
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = encode();
    return {...encoder, beginRenderPass(descriptor) {
      const pass = encoder.beginRenderPass(descriptor), events = []; queryEvents.push(events);
      return {...pass, beginOcclusionQuery(index) { assert.ok(descriptor.occlusionQuerySet); events.push(['begin', index]); },
        endOcclusionQuery() { events.push(['end']); }};
    }, resolveQuerySet(set, first, count, buffer, offset) {
      assert.equal(first, 0); assert.equal(offset, 0);
      new BigUint64Array(buffer.data).set(Array.from({length: count}, (_, index) => set.values[index] ?? 1n));
    }, copyBufferToBuffer(source, from, target, to, size) {
      new Uint8Array(target.data, to, size).set(new Uint8Array(source.data, from, size));
    }};
  };
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [32, 32], format: configured.format, usage: 16})})};
  const Renderer = createWebGPURendererClass(T, {renderBundles: bundles});
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicMaterial();
  const objects = [new T.Mesh(geometry, material), new T.Mesh(geometry, material)];
  for (const [index, object] of objects.entries()) {
    object.frustumCulled = false; object.renderOrder = index; object.occlusionTest = true; scene.add(object);
  }
  t.after(() => {renderer.dispose(); geometry.dispose(); material.dispose();});
  return {renderer, device, scene, camera, objects, querySets, maps, queryEvents};
}
async function render(f) { f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU(); }

test('native occlusion preserves null/undefined and the previous-render publication boundary', async t => {
  const f = fixture(t), seen = [];
  f.objects[0].onBeforeRender = renderer => seen.push(f.objects.map(object => renderer.isOccluded(object)));
  assert.equal(f.renderer.isOccluded(f.objects[0]), null);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.querySets.length, 0); assert.equal(f.maps.length, 0); assert.deepEqual(seen, [[undefined, undefined]]);
  seen.length = 0; // Source compilation runs object hooks, without submitting queries.
  await render(f); assert.deepEqual(seen, [[undefined, undefined]]); assert.equal(f.maps.length, 0);
  await render(f); assert.deepEqual(seen[1], [undefined, undefined]); assert.equal(f.maps.length, 1);
  await render(f); assert.deepEqual(seen[2], [true, false]); assert.equal(f.maps.length, 2);
  assert.equal(f.renderer.isOccluded(f.objects[0]), null);
  assert.deepEqual(f.queryEvents.filter(events => events.length), Array(3).fill([['begin', 0], ['end'], ['begin', 1], ['end']]));
  assert.ok(f.querySets.slice(0, 2).every(set => set.destroyed));
  assert.equal(f.renderer.info.f3d.route, 'general-webgpu');
});

test('queries wrap individual source objects even when internal command bundles are enabled', async t => {
  const f = fixture(t, {bundles: true}); await f.renderer.init();
  await render(f); await render(f);
  assert.equal(f.querySets.length, 2); assert.equal(f.querySets[0].count, 2);
  assert.ok(f.device.passes.some(pass => pass.bundles.length));
  const sequences = f.queryEvents.filter(events => events.length);
  assert.deepEqual(sequences, Array(2).fill([['begin', 0], ['end'], ['begin', 1], ['end']]));
});

test('consecutive material groups share one object query and retained live flags stay per request', async t => {
  const f = fixture(t); f.scene.remove(f.objects[1]);
  const object = f.objects[0]; object.material = [object.material, object.material];
  object.geometry.clearGroups(); object.geometry.addGroup(0, 3, 0); object.geometry.addGroup(3, 3, 1);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  object.occlusionTest = false; f.renderer.render(f.scene, f.camera);
  object.occlusionTest = true; await f.renderer.waitForGPU();
  assert.equal(f.querySets.length, 1); assert.equal(f.querySets[0].count, 1);
  assert.deepEqual(f.queryEvents.filter(events => events.length), [[['begin', 0], ['end']]]);
  assert.equal(f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main').length, 4);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('disabling the last query retains source callbacks and the completed result history', async t => {
  const f = fixture(t), seen = []; f.scene.remove(f.objects[1]);
  const object = f.objects[0];
  object.onBeforeRender = renderer => seen.push(renderer.isOccluded(object));
  await f.renderer.init(); await render(f); await render(f);
  object.occlusionTest = false; await render(f);
  assert.deepEqual(seen, [undefined, undefined, true]);
  assert.equal(f.querySets.length, 2);
});

test('same-shaped targets and different scenes share source context history, differing target formats do not', async t => {
  const f = fixture(t), seen = [];
  const a = new T.RenderTarget(32, 32), b = new T.RenderTarget(16, 16), c = new T.RenderTarget(16, 16, {type: T.HalfFloatType});
  t.after(() => {a.dispose(); b.dispose(); c.dispose();});
  f.objects[0].onBeforeRender = renderer => seen.push(renderer.isOccluded(f.objects[0]));
  await f.renderer.init();
  f.renderer.setRenderTarget(a); await render(f);
  f.renderer.setRenderTarget(b); await render(f);
  const otherScene = new T.Scene(); otherScene.add(...f.objects); f.scene = otherScene;
  f.renderer.setRenderTarget(a); await render(f);
  assert.deepEqual(seen, [undefined, undefined, true]);
  f.renderer.setRenderTarget(c); await render(f); assert.equal(seen.at(-1), undefined);
});

test('ArrayCamera preserves the source branch without issuing object queries', async t => {
  const f = fixture(t), seen = [];
  const child = f.camera; child.viewport = new T.Vector4(0, 0, 32, 32);
  f.camera = new T.ArrayCamera([child]);
  f.objects[0].onBeforeRender = renderer => seen.push(renderer.isOccluded(f.objects[0]));
  await f.renderer.init(); await render(f); await render(f); await render(f);
  assert.ok(f.querySets.length > 0); assert.ok(f.maps.length > 0);
  assert.ok(f.querySets.every(set => set.count === 1));
  assert.ok(f.queryEvents.every(events => events.length === 0), 'source allocates unused query slots without querying ArrayCamera draws');
  assert.deepEqual(seen, [undefined, undefined, false]);
});

test('nested node-render passes restore the enclosing source query context', async t => {
  const f = fixture(t), seen = [], inner = new T.Scene(); f.scene.remove(f.objects[1]);
  const outerObject = f.objects[0], innerObject = f.objects[1]; inner.add(innerObject);
  innerObject.onBeforeRender = renderer => seen.push(['inner', renderer.isOccluded(innerObject), renderer.isOccluded(outerObject)]);
  outerObject.material = new T.MeshBasicNodeMaterial(); t.after(() => outerObject.material.dispose());
  outerObject.material.colorNode = T.TSL.uniform(new T.Color(1, 1, 1)).onObjectUpdate(({renderer}) => {
    renderer.render(inner, f.camera);
    seen.push(['outer', renderer.isOccluded(outerObject), renderer.isOccluded(innerObject)]);
    return new T.Color(1, 1, 1);
  });
  await f.renderer.init(); await render(f); await render(f); await render(f);
  assert.deepEqual(seen, [
    ['inner', undefined, undefined], ['outer', undefined, undefined],
    ['inner', undefined, undefined], ['outer', undefined, undefined],
    ['inner', true, false], ['outer', true, false],
  ]);
  assert.equal(f.renderer.isOccluded(outerObject), null);
});

test('source BundleGroup flags warn and do not open their own GPU queries', async t => {
  const f = fixture(t), group = new T.BundleGroup(), warnings = [];
  group.add(...f.objects); f.scene.add(group);
  const originalWarn = console.warn; console.warn = (...args) => warnings.push(args.join(' '));
  try { await f.renderer.init(); await render(f); await render(f); }
  finally { console.warn = originalWarn; }
  assert.equal(f.querySets.length, 0);
  assert.ok(warnings.some(message => message.includes('Occlusion queries can not be recorded into render bundles')));
  assert.ok(f.device.passes.some(pass => pass.draws.length));
});

test('a failed asynchronous query map rejects waitForGPU and releases owned query resources', async t => {
  const f = fixture(t); await f.renderer.init(); await render(f);
  f.device.mapError = new Error('mapping failed'); f.renderer.render(f.scene, f.camera);
  await assert.rejects(f.renderer.waitForGPU(), /mapping failed/);
  assert.ok(f.querySets.every(set => set.destroyed));
  assert.ok(f.device.buffers.filter(buffer => buffer.label?.startsWith('f3d-occlusion')).every(buffer => buffer.destroyed));
});
