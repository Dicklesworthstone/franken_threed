/** Command recordings prove that Inspector-visible backend methods perform
 * native work; the browser lane exercises the unchanged pinned Timeline UI. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {RendererInspector} from '../../upstream/three.js/examples/jsm/inspector/RendererInspector.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t, materialFactory = () => new T.MeshBasicNodeMaterial()) {
  const device = geometryDevice(); device.features = new Set();
  device.limits.maxTextureDimension2D = 8192;
  device.createTexture = descriptor => ({...descriptor, width: descriptor.size[0], height: descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() {this.destroyed = true;}});
  device.createSampler = value => value;
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout() {return {};}});
  device.createComputePipeline = device.createRenderPipeline;
  device.createComputePipelineAsync = async descriptor => device.createComputePipeline(descriptor);
  device.computePasses = [];
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = encode();
    return {...encoder, beginComputePass(descriptor) {
      const pass = {descriptor, dispatches: [], ended: false}; device.computePasses.push(pass);
      return {setPipeline(value) {pass.pipeline = value;}, setBindGroup() {},
        dispatchWorkgroups(...groups) {pass.dispatches.push(groups);},
        dispatchWorkgroupsIndirect(buffer, offset) {pass.dispatches.push({buffer, offset});},
        end() {pass.ended = true;}};
    }};
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) {configuration = value;},
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [32, 32], format: configuration.format, usage: 16})})};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const geometry = new T.PlaneGeometry(), material = materialFactory(), mesh = new T.Mesh(geometry, material);
  scene.add(mesh);
  if (material.isMeshToonNodeMaterial) scene.add(new T.DirectionalLight(0xffffff, 2), new T.AmbientLight(0xffffff, .2));
  const inspector = renderer.inspector;
  inspector.currentFrame = inspector.currentRender = inspector.currentCompute = inspector.currentContext = null;
  inspector.beginRender = (uid, source, selectedCamera) => {
    inspector.currentRender = {uid, name: source.name || 'Scene', scene: source, camera: selectedCamera};
    inspector.currentContext = inspector.currentRender;
  };
  inspector.finishRender = () => {inspector.currentRender = inspector.currentContext = null;};
  inspector.beginCompute = (uid, group) => {inspector.currentCompute = {uid, name: 'Compute Group', group};};
  inspector.finishCompute = () => {inspector.currentCompute = null;};
  t.after(() => {renderer.dispose(); geometry.dispose(); material.dispose();});
  return {renderer, device, scene, camera, geometry, material, mesh, inspector};
}

test('native backend exposes prototype operations that the pinned Inspector Timeline can wrap', t => {
  const {renderer} = fixture(t);
  const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(renderer.backend));
  for (const method of ['beginRender', 'draw', 'finishRender', 'beginCompute', 'compute', 'finishCompute',
    'beginBundle', 'finishBundle', 'addBundle']) assert.ok(methods.includes(method), method);
});

function observe(f) {
  const backend = f.renderer.backend, originals = new Map(), events = [];
  for (const name of Object.getOwnPropertyNames(Object.getPrototypeOf(backend))) {
    if (name === 'constructor' || name.startsWith('_') || typeof backend[name] !== 'function') continue;
    const original = backend[name]; originals.set(name, original);
    backend[name] = function (...args) {
      events.push({name, target: args[0], node: name === 'compute' ? args[1] : null,
        dispatch: name === 'compute' ? args[4] : null,
        parameters: name === 'draw' ? args[0].getDrawParameters() : null,
        frame: f.renderer.info.frame, render: f.inspector.currentRender, compute: f.inspector.currentCompute});
      return original.apply(this, args);
    };
  }
  return {events, stop() {for (const [name, original] of originals) backend[name] = original;}};
}
const flush = async renderer => {while (renderer._drain) await renderer._drain;};
const deferred = () => {let resolve; const promise = new Promise(yes => {resolve = yes;}); return {promise, resolve};};

test('Inspector compute records and timestamp lookups retain the exact node or named group owner', async t => {
  const f = fixture(t), N = T.TSL;
  const values = new T.StorageBufferAttribute(new Uint32Array(8), 1), storage = N.storage(values, 'uint', 8);
  const kernel = N.Fn(() => {storage.element(N.instanceIndex).assign(N.instanceIndex);})().compute(8, [4]);
  kernel.name = 'Update particles'; t.after(() => kernel.dispose());
  const inspector = new RendererInspector(); inspector.resolveTimestamp = () => {};
  f.renderer.inspector = inspector;
  const calls = [], begin = inspector.beginCompute, finish = inspector.finishCompute;
  inspector.beginCompute = function (uid, owner) { calls.push({kind: 'begin', uid, owner}); return begin.call(this, uid, owner); };
  inspector.finishCompute = function (uid) { calls.push({kind: 'finish', uid}); return finish.call(this, uid); };
  await f.renderer.init(); await f.renderer.compileComputeAsync(kernel);
  f.renderer.info.frame = 3; inspector.begin();
  const single = [kernel], group = [kernel, kernel];
  single.name = 'One step'; group.name = 'Two steps';
  for (const owner of [kernel, single, group]) {
    f.renderer.compute(owner);
    const record = inspector.currentFrame.computes.at(-1);
    assert.equal(record.computeNode, owner, 'the unchanged Inspector must see the original source owner');
    assert.equal(record.name, owner.name);
    const [start, end] = calls.slice(-2);
    assert.equal(start.owner, owner); assert.equal(end.uid, start.uid);
    assert.equal(f.renderer.backend.getTimestampUID(owner), start.uid);
  }
  inspector.finish();
  await flush(f.renderer);
  assert.equal(f.device.computePasses.length, 4);
});

test('Inspector render completion receives the UID of the accepted render', async t => {
  const f = fixture(t), calls = [];
  const begin = f.inspector.beginRender, finish = f.inspector.finishRender;
  f.inspector.beginRender = function (uid, ...args) { calls.push(['begin', uid]); return begin.call(this, uid, ...args); };
  f.inspector.finishRender = function (uid) { calls.push(['finish', uid]); return finish.call(this, uid); };
  await f.renderer.init(); f.renderer.info.frame = 4;
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(calls.map(([kind]) => kind), ['begin', 'finish']);
  assert.equal(calls[1][1], calls[0][1]); assert.ok(calls[1][1].endsWith(':f4'));
  await flush(f.renderer);
});

for (const [kind, material] of [
  ['ordinary MeshToon', () => new T.MeshToonNodeMaterial()],
  ['authored node', () => new T.MeshBasicNodeMaterial({colorNode: T.TSL.vec3(.2, .4, .6)})],
]) test(`queued ${kind} passes expose captured source identities, draw ranges and Inspector frame context`, async t => {
  const f = fixture(t, material), watched = observe(f), gate = deferred(), started = deferred();
  await f.renderer.init();
  const pipeline = f.device.createRenderPipelineAsync.bind(f.device);
  f.device.createRenderPipelineAsync = async descriptor => {started.resolve(); await gate.promise; return pipeline(descriptor);};
  t.after(() => gate.resolve());
  f.renderer.info.frame = 3; f.geometry.setDrawRange(0, 3); f.renderer.render(f.scene, f.camera);
  await started.promise;
  f.renderer.info.frame = 4; f.geometry.setDrawRange(3, 3); f.renderer.render(f.scene, f.camera);
  f.renderer.info.frame = 99; f.geometry.setDrawRange(0, 6);
  assert.equal(f.inspector.currentRender, null); assert.equal(watched.events.length, 0);
  gate.resolve(); await flush(f.renderer);
  const draws = watched.events.filter(event => event.name === 'draw');
  assert.equal(draws.length, 2); assert.equal(f.device.encodedDrawCalls, 2);
  assert.deepEqual(draws.map(event => [event.frame, event.parameters.firstVertex, event.parameters.vertexCount]), [[3, 0, 3], [4, 3, 3]]);
  for (const event of draws) {
    assert.equal(event.target.object, f.mesh); assert.equal(event.target.material, f.material);
    assert.equal(event.target.geometry, f.geometry); assert.equal(event.render.camera, f.camera);
    assert.ok(event.render.uid.endsWith(`:f${event.frame}`));
  }
  assert.deepEqual(watched.events.filter(event => !event.name.includes('Bundle')).map(event => event.name),
    ['beginRender', 'draw', 'finishRender', 'beginRender', 'draw', 'finishRender']);
  assert.equal(f.renderer.info.frame, 99); assert.equal(f.inspector.currentRender, null);
  assert.equal(f.inspector.currentContext, null);
  const recorded = watched.events.length;
  watched.stop();
  assert.equal(f.renderer.backend._drawObject({}, {}), null, 'stopping inspection removes per-draw diagnostic allocation');
  f.renderer.render(f.scene, f.camera); await flush(f.renderer);
  assert.equal(watched.events.length, recorded); assert.equal(f.device.encodedDrawCalls, 3);
});

test('operation context restores absent fields on failure and never enters a replacement Inspector', t => {
  const f = fixture(t), marker = {name: 'accepted'};
  f.renderer.info.frame = 8; f.inspector.currentRender = marker;
  const scope = f.renderer.backend._capture({camera: f.camera});
  delete f.inspector.currentRender; f.renderer.info.frame = 12;
  const error = new Error('encoding failed');
  assert.throws(() => scope.run(() => {
    assert.equal(f.renderer.info.frame, 8); assert.equal(f.inspector.currentRender, marker); throw error;
  }), value => value === error);
  assert.equal(Object.hasOwn(f.inspector, 'currentRender'), false);
  assert.equal(f.renderer.info.frame, 12);
  const replacement = new T.InspectorBase(); f.renderer.inspector = replacement;
  const fields = Object.getOwnPropertyDescriptors(replacement);
  assert.equal(scope.run(() => {
    assert.equal(f.renderer.inspector, replacement); assert.equal(f.renderer.info.frame, 12); return 42;
  }), 42);
  assert.deepEqual(Object.getOwnPropertyDescriptors(replacement), fields);
});

test('inspecting fixed materials preserves physical instancing and exposes every grouped source', async t => {
  const f = fixture(t, () => new T.MeshToonNodeMaterial()), watched = observe(f);
  const second = new T.Mesh(f.geometry, f.material); second.position.x = .5; f.scene.add(second);
  await f.renderer.renderAsync(f.scene, f.camera);
  const draws = watched.events.filter(event => event.name === 'draw');
  assert.equal(draws.length, 1); assert.equal(f.device.encodedDrawCalls, 1);
  assert.equal(draws[0].parameters.instanceCount, 2);
  assert.deepEqual(draws[0].target.sourceDraws.map(draw => draw.object), [f.mesh, second]);
  assert.equal(f.device.snapshots.flat()[0].args[1], 2, 'inspection preserves the real grouped GPU instance count');
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(f.device.encodedDrawCalls, 1, 'warm execution reuses the original physical bundle');
  assert.equal(watched.events.filter(event => event.name === 'draw').length, 1);
  assert.equal(watched.events.filter(event => event.name === 'addBundle').length, 2);
});

test('queued compute reports the captured source dispatch and restores the surrounding Inspector context', async t => {
  const f = fixture(t), watched = observe(f), N = T.TSL;
  const values = new T.StorageBufferAttribute(new Uint32Array(8), 1), storage = N.storage(values, 'uint', 8);
  const kernel = N.Fn(() => {storage.element(N.instanceIndex).assign(N.instanceIndex);})().compute(8, [4]);
  await f.renderer.init(); await f.renderer.compileComputeAsync(kernel);
  // A render preparation keeps the following compute in the ordinary shared FIFO.
  const gate = deferred(), started = deferred(), pipeline = f.device.createRenderPipelineAsync.bind(f.device);
  f.device.createRenderPipelineAsync = async descriptor => {started.resolve(); await gate.promise; return pipeline(descriptor);};
  t.after(() => {gate.resolve(); kernel.dispose();});
  f.renderer.info.frame = 6; f.renderer.render(f.scene, f.camera); await started.promise;
  f.renderer.info.frame = 7; const dispatch = [2, 1, 1], group = [kernel]; group.name = 'Queued step';
  f.renderer.compute(group, dispatch);
  const acceptedUid = f.renderer.backend.getTimestampUID(group);
  group.length = 0; dispatch[0] = 1; f.renderer.info.frame = 12;
  assert.equal(f.device.computePasses.length, 0); assert.equal(f.inspector.currentCompute, null);
  gate.resolve(); await flush(f.renderer);
  const events = watched.events.filter(event => event.name.includes('Compute') || event.name === 'compute');
  assert.deepEqual(events.map(event => event.name), ['beginCompute', 'compute', 'finishCompute']);
  assert.deepEqual(events.map(event => event.frame), [7, 7, 7]);
  assert.ok(events.every(event => event.target === group), 'queued operations keep the original group identity');
  assert.ok(events.every(event => event.compute.group === group));
  assert.ok(events.every(event => event.compute.uid === acceptedUid));
  assert.equal(events[1].node, kernel); assert.deepEqual(events[1].dispatch, [2, 1, 1]);
  assert.ok(events[1].compute.uid.endsWith(':f7'));
  assert.deepEqual(f.device.computePasses[0].dispatches, [[2, 1, 1]]); assert.equal(f.device.computePasses[0].ended, true);
  assert.equal(f.renderer.info.frame, 12); assert.equal(f.inspector.currentCompute, null);
});

test('Timeline-visible bundle methods record real draws once and observe every warm execution', async t => {
  const f = fixture(t), group = new T.BundleGroup(), watched = observe(f);
  group.add(f.mesh); f.scene.add(group);
  await f.renderer.init(); f.renderer.info.frame = 1;
  f.renderer.render(f.scene, f.camera); await flush(f.renderer);
  f.renderer.info.frame = 2; f.renderer.render(f.scene, f.camera); await flush(f.renderer);
  assert.equal(f.device.bundleEncoders.length, 1, 'the second frame reuses the actual native bundle');
  assert.equal(watched.events.filter(event => event.name === 'draw').length, 1);
  assert.equal(watched.events.filter(event => event.name === 'beginBundle').length, 1);
  assert.equal(watched.events.filter(event => event.name === 'finishBundle').length, 1);
  assert.deepEqual(watched.events.filter(event => event.name === 'addBundle').map(event => event.frame), [1, 2]);
  assert.equal(f.device.snapshots.flat().length, 2, 'both frames execute the recorded native draw');
  assert.equal(f.inspector.currentRender, null); assert.equal(f.renderer.info.frame, 2);
});
