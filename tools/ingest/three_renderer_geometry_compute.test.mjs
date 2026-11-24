/** Source render-triggered compute through the native graphics/storage owners.
 * The device records dispatches and queue-visible bytes; it does not run WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceNodeManager from '../../upstream/three.js/src/renderers/common/nodes/NodeManager.js';
import SourceRenderObjects from '../../upstream/three.js/src/renderers/common/RenderObjects.js';
import SourceAttributeUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUAttributeUtils.js';
import SourceAttributes from '../../upstream/three.js/src/renderers/common/Attributes.js';
import SourceRenderer from '../../upstream/three.js/src/renderers/common/Renderer.js';
import SourceWebGPUBackend from '../../upstream/three.js/src/renderers/webgpu/WebGPUBackend.js';
import {AttributeType} from '../../upstream/three.js/src/renderers/common/Constants.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const storage = pass => [...pass.groups.values()].flatMap(group => group.entries)
  .map(entry => entry.resource?.buffer).filter(buffer => buffer?.usage & 128);
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
const packets = f => f.captures.flatMap(frame => frame.draws ?? []).filter(draw => draw.object === f.object);
function fixture(t, {background = true} = {}) {
  const device = geometryDevice(), events = [], captured = [], computes = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = desc => ({...desc, getBindGroupLayout: index => ({index})});
  device.createComputePipeline = desc => ({...desc, getBindGroupLayout: index => ({index})});
  device.createComputePipelineAsync = desc => Promise.resolve(device.createComputePipeline(desc));
  const create = device.createCommandEncoder, submit = device.queue.submit;
  device.createCommandEncoder = () => {
    const encoder = create(), passes = [];
    return {...encoder, beginComputePass(desc) {
      const pass = {desc, draws: [], groups: new Map(), dispatches: []}; passes.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.groups.set(index, group); },
        dispatchWorkgroups(...size) { pass.dispatches.push(size); }, end() {}};
    }, finish() { return [...encoder.finish(), ...passes]; }};
  };
  device.queue.submit = commands => {
    for (const list of commands) for (const pass of list) {
      if (pass.dispatches) {
        pass.bytes = [...pass.groups.values()].flatMap(group => group.entries).map(entry => entry.resource?.buffer)
          .filter(Boolean).map(buffer => new Uint8Array(buffer.data).slice());
        computes.push(pass); events.push(pass.pipeline.compute.module.label);
        device.onCompute?.(pass);
      } else if (pass.draws.length) events.push('render');
    }
    submit(commands);
  };
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [32, 32], format: configured.format, usage: configured.usage})})};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.BoxGeometry(), material);
  object.frustumCulled = false; scene.add(object);
  if (background) scene.backgroundNode = N.vec3(.0625, .125, .25);
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) { const frame = capture.call(this, request); if (frame) captured.push(frame); return frame; };
  t.after(() => { renderer.dispose(); object.geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, camera, material, object, events, computes, captures: captured};
}
function jelly(f) {
  const state = {builds: 0, initializations: 0, kernels: [], attributes: [], step: N.uniform(.125).setName('geometryStep')};
  f.material.geometryNode = N.Fn(({renderer, geometry}) => {
    state.builds++;
    const count = geometry.attributes.position.count, attribute = new T.StorageBufferAttribute(count, 3);
    geometry.setAttribute('storagePosition', attribute); state.attributes.push(attribute);
    const positions = N.storage(attribute, 'vec3', count), base = N.storage(geometry.attributes.position, 'vec3', count);
    const init = N.Fn(() => { positions.element(N.instanceIndex).assign(base.element(N.instanceIndex)); })().compute(count).setName('geometry-init');
    const update = N.Fn(() => { positions.element(N.instanceIndex).addAssign(N.vec3(state.step, 0, 0)); })().compute(count).setName('geometry-step');
    update.onInit(() => { state.initializations++; renderer.compute(init); }); state.kernels.push(update);
    return update;
  })();
  f.material.positionNode = N.attribute('storagePosition');
  return state;
}
function sourceCode(f) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = N.lights([]); builder.build(); return builder;
}

test('before-render compute produces the current draw storage at the source callback boundary', async t => {
  const f = fixture(t, {background: false}), attribute = new T.StorageBufferAttribute(1, 4);
  const values = N.storage(attribute, 'vec4', 1), callbacks = [];
  const kernel = N.Fn(() => { values.element(0).addAssign(N.vec4(.25, 0, 0, 0)); })().compute(1).setName('callback-step');
  t.after(() => kernel.dispose()); f.material.colorNode = values.element(0).xyz;
  f.device.onCompute = pass => { new Float32Array(storage(pass)[0].data)[0] += .25; };
  f.object.onBeforeRender = (renderer, scene, camera, geometry, material) => {
    assert.equal(renderer, f.renderer); assert.equal(scene, f.scene); assert.equal(camera, f.camera);
    assert.equal(geometry, f.object.geometry); assert.equal(material, f.material);
    callbacks.push('before'); renderer.compute(kernel); callbacks.push('computed');
  };
  f.object.onAfterRender = () => callbacks.push('after');
  await f.renderer.init();
  for (let i = 0; i < 2; i++) { f.renderer.render(f.scene, f.camera); await drain(f.renderer); }
  const gpu = storage(f.computes[0])[0];
  assert.deepEqual(draws(f).map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]), [.25, .5]);
  assert.deepEqual(f.events, ['callback-step', 'render', 'callback-step', 'render']);
  assert.deepEqual(callbacks, ['before', 'computed', 'after', 'before', 'computed', 'after']);
  assert.ok(attribute.array.every(value => value === 0), 'GPU results never overwrite the public CPU array');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const hook of ['onBeforeRender', 'onAfterRender'])
test(`disjoint ${hook} compute follows earlier compute and precedes the pending parent render`, async t => {
  const f = fixture(t, {background: false}), first = N.instancedArray(1, 'vec4'), second = N.instancedArray(1, 'vec4');
  const initial = N.Fn(() => { first.element(0).assign(N.vec4(.25)); })().compute(1).setName('earlier-callback');
  const later = N.Fn(() => { second.element(0).assign(N.vec4(.75)); })().compute(1).setName('later-callback');
  const material = new T.MeshBasicNodeMaterial(), reader = new T.Mesh(f.object.geometry, material);
  reader.frustumCulled = false; reader.renderOrder = -1; material.colorNode = first.element(0).xyz;
  reader.onBeforeRender = () => f.renderer.compute(initial); f.scene.add(reader);
  // A later callback can run after the first draw has consumed another storage
  // allocation. Its compute still submits before both pending parent draws.
  f.material.colorNode = N.vec3(.125); f.object[hook] = () => f.renderer.compute(later);
  t.after(() => { initial.dispose(); later.dispose(); material.dispose(); });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(f.events, ['earlier-callback', 'later-callback', 'render']);
  assert.equal(draws(f).length, 2); assert.notEqual(storage(f.computes[0])[0], storage(f.computes[1])[0]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued callback kernels retain per-call uniforms and storage consumers in the renderer FIFO', async t => {
  const f = fixture(t, {background: false}), values = N.instancedArray(1, 'vec4');
  const step = N.uniform(.125).setName('callbackStep'); let dispatch = 0;
  const kernel = N.Fn(() => { values.element(0).assign(N.vec4(step)); })().compute(1).setName('queued-callback');
  f.material.colorNode = values.element(0).xyz; f.object.onBeforeRender = () => f.renderer.compute(kernel);
  f.device.onCompute = pass => { new Float32Array(storage(pass)[0].data).fill(++dispatch * .25); };
  t.after(() => kernel.dispose()); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); step.value = .625; f.renderer.render(f.scene, f.camera);
  step.value = 99; await drain(f.renderer);
  const uniforms = pass => pass.bytes.flatMap(bytes => [...new Float32Array(bytes.buffer)]);
  assert.equal(f.computes.length, 2);
  assert.ok(uniforms(f.computes[0]).includes(.125)); assert.ok(uniforms(f.computes[1]).includes(.625));
  assert.ok(!uniforms(f.computes[1]).includes(99));
  const gpu = storage(f.computes[0])[0];
  assert.deepEqual(draws(f).map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]), [.25, .5]);
  assert.deepEqual(f.events, ['queued-callback', 'render', 'queued-callback', 'render']);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('initialized callback compute keeps the source undefined return value while submission is queued', async t => {
  const f = fixture(t, {background: false}), values = N.instancedArray(1, 'vec4'), results = [];
  const kernel = N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1).setName('sync-callback-result');
  f.material.colorNode = values.element(0).xyz;
  f.object.onBeforeRender = () => results.push(f.renderer.compute(kernel));
  t.after(() => kernel.dispose()); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.deepEqual(results, [undefined, undefined], 'internal preparation cannot change source callback control flow');
  assert.equal(f.computes.length, 2); assert.equal(draws(f).length, 2);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('computeAsync keeps public overrides and waits for its submission without joining later rendering', async t => {
  const f = fixture(t, {background: false}), values = N.instancedArray(1, 'vec4');
  const kernel = N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1).setName('async-call');
  f.material.colorNode = N.vec3(.5);
  const laterScene = new T.Scene(), laterMaterial = new T.MeshBasicNodeMaterial();
  laterMaterial.colorNode = N.vec3(.25, .5, .75);
  const laterObject = new T.Mesh(f.object.geometry, laterMaterial); laterObject.frustumCulled = false; laterScene.add(laterObject);
  let releaseEarlier, enteredEarlier, release, entered, completed = false, calls = 0;
  const earlierHeld = new Promise(resolve => { releaseEarlier = resolve; });
  const earlierPreparing = new Promise(resolve => { enteredEarlier = resolve; });
  const held = new Promise(resolve => { release = resolve; }), preparing = new Promise(resolve => { entered = resolve; });
  t.after(() => { releaseEarlier(); release(); kernel.dispose(); laterMaterial.dispose(); });
  await f.renderer.init();
  const prepare = f.renderer._prepare;
  f.renderer._prepare = function(destination, request) {
    if (request?.scene === f.scene) {
      enteredEarlier(); return earlierHeld.then(() => prepare.call(this, destination, request));
    }
    if (request?.scene === laterScene) {
      entered(); return held.then(() => prepare.call(this, destination, request));
    }
    return prepare.call(this, destination, request);
  };
  const compute = f.renderer.compute;
  f.renderer.compute = function(node, size) {
    calls++; assert.equal(this, f.renderer); assert.equal(node, kernel);
    const result = compute.call(this, node, size);
    assert.equal(result, undefined); return result;
  };
  f.renderer.render(f.scene, f.camera);
  const task = f.renderer.computeAsync(kernel); task.then(() => { completed = true; });
  f.renderer.render(laterScene, f.camera);
  await earlierPreparing; await new Promise(resolve => setImmediate(resolve));
  assert.equal(completed, false, 'computeAsync must wait for its own queued submission');
  assert.equal(f.computes.length, 0); releaseEarlier();
  await preparing; await new Promise(resolve => setImmediate(resolve));
  assert.equal(completed, true, 'later rendering preparation cannot extend this compute call');
  await task; assert.equal(calls, 1); assert.equal(f.computes.length, 1);
  assert.deepEqual(f.events, ['render', 'async-call']);
  release(); await drain(f.renderer);
  assert.deepEqual(f.events, ['render', 'async-call', 'render']);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('disposing initialized kernels after compute retains source dispatch order across a pending render', async t => {
  const f = fixture(t, {background: false}), reference = fixture(t, {background: false});
  const values = N.instancedArray(1, 'vec4'), amount = N.uniform(.125).setName('retiredAmount');
  const kernel = N.Fn(() => { values.element(0).addAssign(N.vec4(amount)); })().compute(1).setName('first-generation');
  let initializations = 0; kernel.onInit(() => initializations++);

  // The pinned public method and backend submit both generations before each
  // initialized compute() returns. Cache disposal cannot retract that work.
  const backend = new SourceWebGPUBackend(); backend.device = reference.device;
  const compiled = new Map(), sourceKernel = N.Fn(() => {})().compute(1).setName('first-generation');
  let sourceInitializations = 0; sourceKernel.onInit(() => sourceInitializations++);
  const source = {_initialized: true, _isDeviceLost: false, backend, info: reference.renderer.info,
    inspector: {beginCompute() {}, finishCompute() {}},
    _nodes: {nodeFrame: {renderId: 0}, updateBeforeForCompute() {}, updateForCompute() {}, updateAfterForCompute() {}, delete() {}},
    _bindings: {updateForCompute() {}, getForCompute: () => [], deleteForCompute() {}},
    _pipelines: {has: node => compiled.has(node), delete: node => compiled.delete(node), getForCompute(node) {
      if (!compiled.has(node)) {
        const pipeline = {}; compiled.set(node, pipeline);
        backend.get(pipeline).pipeline = {compute: {module: {label: node.name}}};
      }
      return compiled.get(node);
    }}};
  backend.renderer = source;
  assert.equal(SourceRenderer.prototype.compute.call(source, sourceKernel), undefined);
  assert.deepEqual(reference.events, ['first-generation']);
  sourceKernel.dispose(); sourceKernel.setName('second-generation');
  assert.equal(SourceRenderer.prototype.compute.call(source, sourceKernel, [3]), undefined);
  sourceKernel.dispose();

  let release, entered;
  const held = new Promise(resolve => { release = resolve; }), preparing = new Promise(resolve => { entered = resolve; });
  t.after(() => { release(); kernel.dispose(); sourceKernel.dispose(); });
  await f.renderer.init();
  const prepare = f.renderer._prepare;
  f.renderer._prepare = function(destination, request) {
    entered(); return held.then(() => prepare.call(this, destination, request));
  };
  f.renderer.render(f.scene, f.camera); await preparing;
  assert.equal(f.renderer.compute(kernel), undefined);
  kernel.dispose(); amount.value = .625; kernel.setName('second-generation');
  const second = f.renderer.computeAsync(kernel, [3]);
  second.catch(() => {});
  kernel.dispose(); amount.value = 99; kernel.setName('future-generation');
  assert.equal(f.computes.length, 0, 'accepted compute waits behind render preparation');
  release(); await second; await drain(f.renderer);
  assert.deepEqual(f.events, ['render', ...reference.events]);
  assert.deepEqual(f.computes.map(pass => pass.dispatches), reference.computes.map(pass => pass.dispatches));
  assert.equal(initializations, sourceInitializations); assert.equal(initializations, 2);
  const uniforms = pass => [...pass.groups.values()].flatMap(group => group.entries)
    .map(entry => entry.resource?.buffer).filter(buffer => buffer?.usage & 64);
  const capturedWords = pass => pass.bytes.flatMap(bytes => [...new Float32Array(bytes.buffer)]);
  assert.ok(capturedWords(f.computes[0]).includes(.125));
  assert.ok(capturedWords(f.computes[1]).includes(.625));
  assert.ok(f.computes.every(pass => !capturedWords(pass).includes(99)));
  assert.notEqual(uniforms(f.computes[0])[0], uniforms(f.computes[1])[0]);
  assert.ok(f.computes.flatMap(uniforms).every(buffer => buffer.destroyed), 'retired uniform generations release after submission');
  const buffer = storage(f.computes[0])[0];
  assert.equal(storage(f.computes[1])[0], buffer); assert.equal(buffer.destroyed, false);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  f.renderer.dispose(); assert.equal(buffer.destroyed, true);
});

test('pre-init compute retains its source Promise and completes its accepted dispatch', async t => {
  const f = fixture(t, {background: false}), values = N.instancedArray(1, 'vec4');
  const kernel = N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1).setName('before-init');
  t.after(() => kernel.dispose());
  const pending = f.renderer.compute(kernel);
  assert.ok(pending instanceof Promise); assert.equal(f.computes.length, 0);
  await pending; await drain(f.renderer);
  assert.equal(f.renderer._initialized, true); assert.deepEqual(f.events, ['before-init']);
});

test('nested async compute and a caught capture failure retain the outer submission completion', async t => {
  const f = fixture(t, {background: false}); f.material.colorNode = N.vec3(.5);
  const create = name => {
    const values = N.instancedArray(1, 'vec4');
    return N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1).setName(name);
  };
  const child = create('nested-child'), invalid = create('nested-invalid'), outer = create('nested-outer');
  let childTask, failedTask;
  outer.onInit(() => {
    failedTask = f.renderer.computeAsync(invalid, -1); failedTask.catch(() => {});
    childTask = f.renderer.computeAsync(child);
  });
  t.after(() => { child.dispose(); invalid.dispose(); outer.dispose(); }); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  await f.renderer.computeAsync(outer);
  assert.deepEqual(f.events, ['render', 'nested-child', 'nested-outer']);
  await childTask; await assert.rejects(failedTask, {code: 'THREE_COMPUTE_DISPATCH'});
  assert.equal(f.renderer.compute(child), undefined); await drain(f.renderer);
  assert.equal(f.events.at(-1), 'nested-child'); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('accepted callback compute remains submitted when a later object callback throws', async t => {
  const f = fixture(t, {background: false}), values = N.instancedArray(1, 'vec4');
  const kernel = N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1).setName('accepted-callback');
  f.material.colorNode = values.element(0).xyz;
  f.object.onBeforeRender = () => { f.renderer.compute(kernel); throw Error('application callback failed'); };
  t.after(() => kernel.dispose()); await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), /application callback failed/); await drain(f.renderer);
  assert.deepEqual(f.events, ['accepted-callback']); assert.equal(draws(f).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const resource of ['storage', 'indirect', 'attachment'])
test(`callback compute refuses a pending parent ${resource} alias before uploading or dispatching`, async t => {
  const f = fixture(t, {background: false}), values = N.instancedArray(1, 'vec4');
  let kernel, indirect = null;
  if (resource === 'attachment') {
    const target = new T.RenderTarget(16, 16); f.renderer.setRenderTarget(target);
    kernel = N.Fn(() => { values.element(0).assign(N.textureLoad(target.texture, N.ivec2(0))); })().compute(1);
    t.after(() => target.dispose()); f.material.colorNode = N.vec3(.5);
  } else if (resource === 'indirect') {
    indirect = new T.IndirectStorageBufferAttribute(new Uint32Array([1, 1, 1]), 1);
    f.material.colorNode = N.vec3(N.storage(indirect, 'uint', 3).element(0).toFloat());
    kernel = N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1);
  } else {
    f.material.colorNode = values.element(0).xyz;
    kernel = N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1);
  }
  f.object.onAfterRender = () => f.renderer.compute(kernel, indirect);
  t.after(() => kernel.dispose()); await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0);
  assert.equal(f.device.writes.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('compute compilation from an object callback keeps its deferred resource boundary', async t => {
  const f = fixture(t, {background: false}), values = N.instancedArray(1, 'vec4');
  const kernel = N.Fn(() => { values.element(0).assign(N.vec4(.25)); })().compute(1);
  f.material.colorNode = N.vec3(.5);
  let attempt;
  f.object.onBeforeRender = () => { attempt = f.renderer.compileComputeAsync(kernel); attempt.catch(() => {}); };
  t.after(() => kernel.dispose()); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  await assert.rejects(attempt, {code: 'THREE_NODE_PROGRAM_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('geometry constructors run once, public compilation computes, and named storage retains native residency', async t => {
  const f = fixture(t), state = jelly(f);
  const companionMaterial = new T.MeshBasicMaterial({color: 0x336699}), companion = new T.Mesh(new T.PlaneGeometry(), companionMaterial);
  companion.renderOrder = -1; companion.frustumCulled = false; f.scene.add(companion);
  t.after(() => { companion.geometry.dispose(); companionMaterial.dispose(); });
  await f.renderer.init();
  const raw = f.renderer._makeOutputRequests(f.scene, f.camera, null, false);
  assert.equal(state.builds, 0, 'capturing compilation selections does not construct later geometry graphs');
  assert.equal(state.initializations, 0); assert.equal(f.computes.length, 0);
  for (const request of raw) f.renderer._releaseRequest(request);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(state.builds, 1); assert.equal(state.initializations, 1); assert.equal(f.computes.length, 2);
  assert.equal(draws(f).length, 0, 'compilation prepares resources without drawing');
  for (let frame = 0; frame < 3; frame++) {
    f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  }
  assert.equal(state.builds, 1); assert.equal(state.initializations, 1);
  assert.equal(state.kernels.length, 1); assert.equal(state.attributes.length, 1);
  assert.deepEqual(f.events.filter(event => event !== 'render'), ['geometry-init', 'geometry-step', 'geometry-step', 'geometry-step', 'geometry-step']);
  assert.ok(f.events.indexOf('geometry-step') < f.events.indexOf('render'));
  assert.equal(state.attributes[0].itemSize, 4); assert.equal(state.attributes[0].array.length, state.attributes[0].count * 4);
  const gpu = storage(f.computes.find(pass => pass.pipeline.compute.module.label === 'geometry-step'))[0];
  assert.ok(draws(f).some(draw => [...draw.streams.values()].includes(gpu)));
  assert.ok(packets(f).every(packet => packet.binding.nodeProgram === packets(f)[0].binding.nodeProgram));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('geometry compute emits the exact source graphics output and kernel WGSL', async t => {
  const reference = fixture(t, {background: false}), oracle = jelly(reference); await reference.renderer.init();
  const expected = sourceCode(reference), kernel = new T.WGSLNodeBuilder(null, reference.renderer);
  kernel.compute = oracle.kernels[0]; kernel.build();
  const f = fixture(t, {background: false}); jelly(f); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(packets(f)[0].binding.nodeProgram.vertex), canonical(expected.vertexShader));
  assert.equal(canonical(packets(f)[0].binding.nodeProgram.fragment), canonical(expected.fragmentShader));
  const update = f.computes.find(pass => pass.pipeline.compute.module.label === 'geometry-step');
  assert.equal(canonical(update.pipeline.compute.module.code), canonical(kernel.computeShader));
});

function sharedKernel(f, name = 'shared-geometry') {
  const attribute = new T.StorageBufferAttribute(1, 4), values = N.storage(attribute, 'vec4', 1);
  const kernel = N.Fn(() => { values.element(0).addAssign(N.vec4(.125, 0, 0, 0)); })().compute(1).setName(name);
  f.material.geometryNode = kernel; f.material.positionNode = N.positionLocal.add(values.element(0).xyz);
  return {attribute, values, kernel};
}

function inheritedKernel(f, frequency = 'frame') {
  const attribute = new T.StorageBufferAttribute(1, 4), values = N.storage(attribute, 'vec4', 1);
  const state = {attribute, values, step: N.uniform(.125), nextStep: .125, initializations: 0, observe: () => {}};
  class SimulationStep extends T.ComputeNode {
    constructor() {
      super(N.Fn(() => { values.element(0).addAssign(N.vec4(state.step, 0, 0, 0)); })(), [1]);
      this.count = 1; this.name = 'inherited-geometry-step';
      this.updateBeforeType = this.updateType = this.updateAfterType = frequency;
      this.onInit(() => { state.initializations++; });
    }
    updateBefore(frame) {
      state.observe('before', frame);
      if (state.retryBefore) { state.retryBefore = false; return false; }
      return super.updateBefore(frame);
    }
    update(frame) { state.observe('current', frame); state.step.value = state.nextStep; }
    updateAfter(frame) { state.observe('after', frame); }
  }
  class ApplicationStep extends SimulationStep {}
  state.kernel = new ApplicationStep();
  f.material.geometryNode = state.kernel; f.material.positionNode = N.positionLocal.add(values.element(0).xyz);
  return state;
}

function customSetupKernel(f, setup = () => {}) {
  const state = inheritedKernel(f); state.setups = [];
  state.kernel.setup = function (builder) {
    state.setups.push(builder.compute === this ? 'compute' : builder.object.name);
    setup(builder, state);
    return T.ComputeNode.prototype.setup.call(this, builder);
  };
  return state;
}
function sourceGeometryPrograms(t, f) {
  f.renderer.debug.onNodeBuilderCreated = null;
  const nodes = new SourceNodeManager(f.renderer,
    {createNodeBuilder: object => new T.WGSLNodeBuilder(object, f.renderer)});
  const objects = new SourceRenderObjects(f.renderer, nodes, {}, {delete() {}}, {deleteForRender() {}}, {});
  const lights = N.lights([]), context = {id: 0};
  t.after(() => objects.dispose());
  return {nodes, use(object = f.object, selectAttributes = true) {
    const owner = objects.get(object, object.material, f.scene, f.camera, lights, context, null);
    return {owner, state: owner.getNodeBuilderState(), attributes: selectAttributes ? owner.getAttributes() : null};
  }};
}
function compileGeometry(f, object = f.object) {
  f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const programs = f.renderer._nodeProgramSupport ??= createThreeNodePrograms({three: T, renderer: f.renderer});
  return programs.compile(object.material, object,
    {object, material: object.material, scene: f.scene, camera: f.camera, lights: [],
      frameId: f.renderer._nodes.nodeFrame.frameId, renderId: f.renderer.info.calls});
}

test('custom geometry setup follows source shared keys, final-owner release and geometry disposal', async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const expected = customSetupKernel(reference), actual = customSetupKernel(candidate);
  await reference.renderer.init(); await candidate.renderer.init();
  const source = sourceGeometryPrograms(t, reference);
  const groups = [reference, candidate].map(f => {
    f.object.name = 'A';
    const other = new T.Mesh(f.object.geometry, f.material); other.name = 'B'; f.scene.add(other);
    return {f, a: f.object, b: other};
  });
  const first = source.use(groups[0].a), next = source.use(groups[0].b);
  const firstCandidate = compileGeometry(candidate, groups[1].a), secondCandidate = compileGeometry(candidate, groups[1].b);
  assert.equal(first.state, next.state); assert.notEqual(firstCandidate, secondCandidate);
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  for (const program of [firstCandidate, secondCandidate]) {
    assert.equal(canonical(program.vertex), canonical(first.state.vertexShader));
    assert.equal(canonical(program.fragment), canonical(first.state.fragmentShader));
  }
  assert.deepEqual(actual.setups, expected.setups); assert.deepEqual(actual.setups, ['A']);
  for (const {f} of groups) { f.material.needsUpdate = true; f.object.geometry.dispose(); f.renderer.info.calls++; }
  assert.equal(source.use().state, first.state);
  assert.equal(compileGeometry(candidate).vertex, firstCandidate.vertex);
  for (const group of groups) {
    group.a.dispatchEvent({type: 'dispose'});
    group.c = new T.Mesh(group.a.geometry, group.f.material); group.c.name = 'C'; group.f.scene.add(group.c);
  }
  assert.equal(source.use(groups[0].c).state, first.state); compileGeometry(candidate, groups[1].c);
  assert.deepEqual(actual.setups, ['A']);
  for (const group of groups) {
    group.b.dispatchEvent({type: 'dispose'}); group.c.dispatchEvent({type: 'dispose'});
    group.d = new T.Mesh(group.a.geometry, group.f.material); group.d.name = 'D'; group.f.scene.add(group.d);
  }
  const finalSource = source.use(groups[0].d).state, finalCandidate = compileGeometry(candidate, groups[1].d);
  assert.notEqual(finalSource, first.state);
  assert.equal(canonical(finalCandidate.vertex), canonical(finalSource.vertexShader));
  assert.equal(canonical(finalCandidate.fragment), canonical(finalSource.fragmentShader));
  assert.deepEqual(actual.setups, expected.setups); assert.deepEqual(actual.setups, ['A', 'D']);
});

test('lazy float attributes retain pre-setup keys and refresh source geometry without setup replay', async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const setup = builder => {
    if (builder.object && !builder.geometry.hasAttribute('setupOffset')) builder.geometry.setAttribute('setupOffset',
      new T.Float32BufferAttribute(new Float32Array(builder.geometry.attributes.position.count * 3), 3));
  };
  const expected = customSetupKernel(reference, setup), actual = customSetupKernel(candidate, setup);
  for (const [f, state] of [[reference, expected], [candidate, actual]]) {
    f.object.name = 'A'; f.material.positionNode = N.attribute('setupOffset').add(state.values.element(0).xyz);
    await f.renderer.init();
  }
  const source = sourceGeometryPrograms(t, reference), first = source.use(), firstCandidate = compileGeometry(candidate);
  const others = [reference, candidate].map(f => ['B', 'C'].map(name => {
    const object = new T.Mesh(f.object.geometry, f.material); object.name = name; f.scene.add(object); return object;
  }));
  const second = source.use(others[0][0]), third = source.use(others[0][1]);
  const secondCandidate = compileGeometry(candidate, others[1][0]), thirdCandidate = compileGeometry(candidate, others[1][1]);
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  for (const [actual, expected] of [[firstCandidate, first], [secondCandidate, second], [thirdCandidate, third]]) {
    assert.equal(canonical(actual.vertex), canonical(expected.state.vertexShader));
    assert.equal(canonical(actual.fragment), canonical(expected.state.fragmentShader));
  }
  assert.notEqual(first.owner.initialCacheKey, second.owner.initialCacheKey);
  assert.notEqual(first.state, second.state); assert.equal(second.state, third.state);
  assert.deepEqual(actual.setups, expected.setups); assert.deepEqual(actual.setups, ['A', 'B']);
  for (const f of [reference, candidate]) { f.material.needsUpdate = true; f.renderer.info.calls++; }
  assert.equal(source.use().state, second.state);
  const moved = compileGeometry(candidate); assert.notEqual(moved, firstCandidate);
  for (const f of [reference, candidate]) {
    f.object.geometry.setAttribute('setupOffset', new T.Float32BufferAttribute(
      new Float32Array(f.object.geometry.attributes.position.count * 3).fill(.125), 3));
    const geometry = f.object.geometry.clone(); t.after(() => geometry.dispose()); f.object.geometry = geometry;
  }
  const changed = source.use(), changedCandidate = compileGeometry(candidate);
  assert.equal(changed.state, second.state);
  assert.ok(changed.attributes.includes(reference.object.geometry.attributes.setupOffset));
  assert.equal(candidate.renderer._nodeProgramSupport.geometry(changedCandidate).attributes.setupOffset,
    candidate.object.geometry.attributes.setupOffset);
  assert.equal(canonical(changedCandidate.vertex), canonical(changed.state.vertexShader));
  assert.equal(canonical(changedCandidate.fragment), canonical(changed.state.fragmentShader));
  assert.deepEqual(actual.setups, expected.setups); assert.deepEqual(actual.setups, ['A', 'B']);
});

test('custom setup separates source graphics and compute versions through public compilation and queued draws', async t => {
  const f = fixture(t, {background: false}), state = customSetupKernel(f);
  const reference = fixture(t, {background: false}), expected = customSetupKernel(reference);
  reference.object.name = 'A'; await reference.renderer.init();
  const source = sourceGeometryPrograms(t, reference), sourceGraphics = source.use().state,
    sourceCompute = source.nodes.getForCompute(expected.kernel);
  f.object.name = 'A';
  const other = new T.Mesh(f.object.geometry, f.material); other.name = 'B'; other.frustumCulled = false; f.scene.add(other);
  let dispatches = 0;
  f.device.onCompute = pass => new Float32Array(storage(pass)[0].data).fill(++dispatches * .25);
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(state.setups, expected.setups); assert.deepEqual(state.setups, ['A', 'compute']);
  assert.equal(state.initializations, 1);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const first = packets(f)[0].binding.nodeProgram;
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(first.vertex), canonical(sourceGraphics.vertexShader));
  assert.equal(canonical(first.fragment), canonical(sourceGraphics.fragmentShader));
  assert.equal(canonical(f.computes[0].pipeline.compute.module.code), canonical(sourceCompute.computeShader));
  state.kernel.needsUpdate = true; state.nextStep = .625;
  expected.kernel.needsUpdate = true; reference.renderer.info.calls++;
  assert.equal(source.use().state, sourceGraphics);
  assert.notEqual(source.nodes.getForCompute(expected.kernel), sourceCompute);
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  state.nextStep = 99; await drain(f.renderer);
  assert.deepEqual(state.setups, expected.setups); assert.deepEqual(state.setups, ['A', 'compute', 'compute']);
  assert.equal(packets(f)[1].binding.nodeProgram, first, 'kernel invalidation retains the graphics builder');
  assert.equal(state.initializations, 1); assert.equal(f.computes.length, 2);
  const values = pass => pass.bytes.flatMap(bytes => [...new Float32Array(bytes.buffer)]);
  assert.ok(values(f.computes[1]).includes(.625)); assert.ok(!values(f.computes[1]).includes(99));
  const gpu = storage(f.computes[0])[0];
  assert.deepEqual(draws(f).map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]), [.25, .25, .5, .5]);
  reference.material.needsUpdate = true; reference.renderer.info.calls++;
  assert.notEqual(source.use().state, sourceGraphics);
  f.material.needsUpdate = true; f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(state.setups, expected.setups); assert.deepEqual(state.setups, ['A', 'compute', 'compute', 'A']);
  assert.notEqual(packets(f)[2].binding.nodeProgram, first); assert.equal(state.initializations, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('custom ComputeNode subclasses add source named storage attributes and dispatch once across owners', async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const authored = [reference, candidate].map(f => {
    const count = f.object.geometry.attributes.position.count, attribute = new T.StorageBufferAttribute(count, 3);
    const values = N.storage(attribute, 'vec3', count), setups = [];
    class GeometryStep extends T.ComputeNode {
      setup(builder) {
        if (builder.object) { setups.push(builder.object.name); builder.geometry.setAttribute('storagePosition', attribute); }
        return super.setup(builder);
      }
    }
    const kernel = new GeometryStep(N.Fn(() => { values.element(N.instanceIndex).addAssign(N.vec3(.125, 0, 0)); })(), [64]);
    kernel.count = count; kernel.name = 'class-geometry-step';
    f.object.name = 'A'; f.material.geometryNode = kernel; f.material.positionNode = N.attribute('storagePosition');
    const other = new T.Mesh(f.object.geometry, f.material); other.name = 'B'; other.frustumCulled = false; f.scene.add(other);
    return {kernel, attribute, setups, other};
  });
  await reference.renderer.init(); await candidate.renderer.init();
  const source = sourceGeometryPrograms(t, reference), first = source.use().state;
  const compute = source.nodes.getForCompute(authored[0].kernel), records = new WeakMap();
  new SourceAttributeUtils({device: reference.device, get(attribute) {
    let record = records.get(attribute); if (!record) records.set(attribute, record = {}); return record;
  }}).createAttribute(authored[0].attribute, 160);
  const second = source.use(authored[0].other).state;
  candidate.device.onCompute = pass => new Float32Array(storage(pass)[0].data).fill(.375);
  candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
  assert.deepEqual(authored[1].setups, authored[0].setups); assert.deepEqual(authored[1].setups, ['A', 'B']);
  assert.equal(candidate.computes.length, 1);
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  const programs = candidate.captures.flatMap(frame => frame.draws).map(draw => draw.binding.nodeProgram);
  for (const [program, expected] of [[programs[0], first], [programs[1], second]]) {
    assert.equal(canonical(program.vertex), canonical(expected.vertexShader));
    assert.equal(canonical(program.fragment), canonical(expected.fragmentShader));
  }
  assert.equal(canonical(candidate.computes[0].pipeline.compute.module.code), canonical(compute.computeShader));
  const gpu = storage(candidate.computes[0])[0];
  assert.ok(draws(candidate).every(draw => [...draw.streams.values()].includes(gpu)));
  assert.deepEqual(draws(candidate).map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]), [.375, .375]);
});

test('adding custom setup to an existing per-object graph waits for source material disposal', async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const expected = inheritedKernel(reference), actual = inheritedKernel(candidate);
  await reference.renderer.init(); await candidate.renderer.init();
  const source = sourceGeometryPrograms(t, reference), first = source.use().state;
  compileGeometry(candidate);
  for (const [f, state] of [[reference, expected], [candidate, actual]]) {
    f.object.name = 'A'; state.setups = [];
    state.kernel.setup = function (builder) {
      state.setups.push(builder.object.name); return T.ComputeNode.prototype.setup.call(this, builder);
    };
  }
  assert.equal(source.use().state, first);
  assert.throws(() => compileGeometry(candidate), {code: 'THREE_NODE_PROGRAM_MATERIAL_CACHE'});
  assert.deepEqual(actual.setups, expected.setups); assert.deepEqual(actual.setups, []);
  for (const f of [reference, candidate]) f.material.dispose();
  const rebuilt = source.use().state, program = compileGeometry(candidate);
  assert.notEqual(rebuilt, first);
  assert.deepEqual(actual.setups, expected.setups); assert.deepEqual(actual.setups, ['A']);
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(program.vertex), canonical(rebuilt.vertexShader));
  assert.equal(canonical(program.fragment), canonical(rebuilt.fragmentShader));
});

test('lazy jelly constructors share post-padding source keys and refresh named attributes without setup replay', async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const expected = jelly(reference), actual = jelly(candidate);
  await reference.renderer.init(); await candidate.renderer.init();
  const source = sourceGeometryPrograms(t, reference), records = new WeakMap();
  const upload = new SourceAttributeUtils({device: reference.device, get(attribute) {
    let record = records.get(attribute); if (!record) records.set(attribute, record = {}); return record;
  }});
  const owners = [reference, candidate].map(f => [f.object, ...['B', 'C'].map(name => {
    const object = new T.Mesh(f.object.geometry, f.material); object.name = name; f.scene.add(object); return object;
  })]);
  const states = [], programs = [];
  for (let index = 0; index < 3; index++) {
    states.push(source.use(owners[0][index]));
    upload.createAttribute(reference.object.geometry.attributes.storagePosition, 160);
    programs.push(compileGeometry(candidate, owners[1][index]));
    // Use the same pinned upload operation between owners: source vec3 storage
    // padding changes the next RenderObject key without rebuilding the first.
    upload.createAttribute(candidate.object.geometry.attributes.storagePosition, 160);
  }
  assert.notEqual(states[0].owner.initialCacheKey, states[1].owner.initialCacheKey);
  assert.equal(states[1].owner.initialCacheKey, states[2].owner.initialCacheKey);
  assert.notEqual(states[0].state, states[1].state); assert.equal(states[1].state, states[2].state);
  assert.equal(actual.builds, expected.builds); assert.equal(actual.builds, 2);
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  for (let index = 0; index < 3; index++) {
    assert.equal(canonical(programs[index].vertex), canonical(states[index].state.vertexShader));
    assert.equal(canonical(programs[index].fragment), canonical(states[index].state.fragmentShader));
  }
  const oldView = candidate.renderer._nodeProgramSupport.geometry(programs[0]);
  assert.equal(oldView.attributes.storagePosition.id, actual.attributes[0].id);
  for (const f of [reference, candidate]) f.renderer.info.calls++;
  const changed = source.use(), changedProgram = compileGeometry(candidate);
  assert.equal(changed.state, states[0].state); assert.notEqual(changedProgram, programs[0]);
  assert.ok(changed.attributes.includes(expected.attributes[1]));
  assert.equal(candidate.renderer._nodeProgramSupport.geometry(changedProgram).attributes.storagePosition.id,
    actual.attributes[1].id);
  assert.equal(oldView.attributes.storagePosition.id, actual.attributes[0].id, 'accepted metadata keeps the original named buffer');
  for (const f of [reference, candidate]) { f.object.geometry.dispose(); f.renderer.info.calls++; }
  assert.equal(source.use().state, states[0].state); compileGeometry(candidate);
  assert.equal(actual.builds, expected.builds); assert.equal(actual.builds, 2);
});

test('shared lazy jelly draws retain queued named-storage selections through later owner refresh', async t => {
  const f = fixture(t, {background: false}), state = jelly(f);
  const others = ['B', 'C'].map(name => {
    const object = new T.Mesh(f.object.geometry, f.material); object.name = name; object.frustumCulled = false;
    f.scene.add(object); return object;
  });
  let steps = 0;
  f.device.onCompute = pass => {
    if (pass.pipeline.compute.module.label === 'geometry-step') new Float32Array(storage(pass)[0].data).fill(++steps * .25);
  };
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  f.renderer.render(f.scene, f.camera);
  for (const object of others) object.visible = false;
  const replacement = new T.StorageBufferAttribute(f.object.geometry.attributes.position.count, 3);
  replacement.array.fill(.875); f.object.geometry.setAttribute('storagePosition', replacement);
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  replacement.array.fill(99); await drain(f.renderer);
  assert.equal(state.builds, 2); assert.equal(state.initializations, 2);
  assert.equal(steps, 3); assert.equal(f.computes.length, 5);
  const first = packets(f)[0].binding.nodeProgram, next = packets(f)[1].binding.nodeProgram;
  assert.notEqual(first, next); assert.equal(first.vertex, next.vertex);
  const gpu = f.computes.filter(pass => pass.pipeline.compute.module.label === 'geometry-step').map(pass => storage(pass)[0]);
  const location = next.attributes.find(attribute => attribute.name === 'storagePosition').location;
  const selected = draws(f).map(draw => draw.streams.get(draw.pipeline.vertex.buffers.findIndex(layout =>
    layout.attributes.some(attribute => attribute.shaderLocation === location))));
  assert.deepEqual(selected.slice(0, 3), [gpu[1], gpu[1], gpu[1]]);
  assert.notEqual(selected[3], gpu[0]); assert.notEqual(selected[3], gpu[1]);
  assert.deepEqual(draws(f).map((draw, index) => new Float32Array(draw.contents.get(selected[index]).buffer)[0]),
    [.5, .5, .5, .875]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const gpu of [false, true])
test(`shared ${gpu ? 'storage' : 'Float32'} geometry selects named inputs at source cold, warm and disposal before-update boundaries`, async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const cases = [reference, candidate].map(f => {
    const kernel = inheritedKernel(f, 'object'), geometry = f.object.geometry, count = geometry.attributes.position.count;
    const named = Array.from({length: 8}, (_, index) => {
      const attribute = gpu ? new T.StorageBufferAttribute(count, 3) : new T.Float32BufferAttribute(new Float32Array(count * 3), 3);
      attribute.array.fill((index + 1) * .125); return attribute;
    });
    const pinned = new T.StorageBufferAttribute(count, 3); pinned.array.fill(.0625);
    geometry.setAttribute('selectedPosition', named[0]);
    f.material.geometryNode = N.Fn(() => kernel.kernel)();
    f.material.positionNode = N.attribute('selectedPosition', 'vec3').add(N.bufferAttribute(pinned, 'vec3'));
    const replacement = new T.BoxGeometry(); replacement.setAttribute('selectedPosition', named[5]);
    t.after(() => replacement.dispose());
    const state = {f, geometry, replacement, named, pinned, before: () => {}};
    kernel.observe = stage => { if (stage === 'before') state.before(); };
    return state;
  });
  await reference.renderer.init(); await candidate.renderer.init();
  reference.renderer.compute = () => {};
  const source = sourceGeometryPrograms(t, reference), selected = [], states = [], pinnedSelections = [];
  const actions = [
    item => () => item.geometry.setAttribute('selectedPosition', item.named[1]),
    item => () => item.geometry.setAttribute('selectedPosition', item.named[2]),
    item => () => item.geometry.setAttribute('selectedPosition', item.named[3]),
    item => () => item.geometry.setAttribute('selectedPosition', item.named[4]),
    item => () => { item.f.object.geometry = item.replacement; },
    () => () => {},
    item => () => item.replacement.setAttribute('selectedPosition', item.named[6]),
    () => () => {},
  ];
  for (let index = 0; index < actions.length; index++) {
    for (const item of cases) {
      if (index === 3) item.geometry.dispose();
      if (index === 6) item.replacement.dispose();
      item.before = actions[index](item);
    }
    reference.renderer.info.calls++;
    const current = source.use(reference.object, false); states.push(current.state);
    source.nodes.updateBefore(current.owner);
    const attributes = current.owner.getAttributes();
    selected.push(cases[0].named.findIndex(attribute => attributes.includes(attribute)));
    assert.ok(attributes.includes(cases[0].pinned));
    candidate.renderer._nodes.nodeFrame.update(); candidate.renderer.render(candidate.scene, candidate.camera);
    await drain(candidate.renderer);
    const packet = packets(candidate).at(-1), view = packet.binding.programGeometry;
    assert.equal(candidate.renderer._nodeProgramSupport.geometry(packet.binding.nodeProgram).attributes.selectedPosition.id,
      cases[1].named[selected.at(-1)].id);
    assert.equal(view.attributes.selectedPosition.array[0], cases[1].named[selected.at(-1)].array[0]);
    const namedNode = packet.binding.nodeProgram.attributes.find(attribute => attribute.name !== 'position' && attribute.name !== 'selectedPosition');
    pinnedSelections.push(view.attributes[namedNode.name].id);
    const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
    assert.equal(canonical(packet.binding.nodeProgram.vertex), canonical(current.state.vertexShader));
    assert.equal(canonical(packet.binding.nodeProgram.fragment), canonical(current.state.fragmentShader));
  }
  assert.deepEqual(selected, [1, 1, 3, 4, 4, 5, 5, 6]);
  assert.ok(states.every(state => state === states[0]));
  assert.ok(pinnedSelections.every(id => id === cases[1].pinned.id));
  assert.equal(candidate.renderer._nodeCaptureBytes, 0);
});

test('a cold before update can promote a named Float32 input to native storage', async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const attributes = [reference, candidate].map(f => {
    const geometry = f.object.geometry, count = geometry.attributes.position.count;
    geometry.setAttribute('selectedPosition', new T.Float32BufferAttribute(new Float32Array(count * 3), 3));
    const attribute = new T.StorageBufferAttribute(count, 3); attribute.array.fill(.375);
    const kernel = N.Fn(() => {})().compute(1);
    kernel.updateBeforeType = 'object'; kernel.updateBefore = () => geometry.setAttribute('selectedPosition', attribute);
    f.material.geometryNode = N.Fn(() => kernel)(); f.material.positionNode = N.attribute('selectedPosition');
    return attribute;
  });
  await reference.renderer.init(); await candidate.renderer.init();
  const source = sourceGeometryPrograms(t, reference); reference.renderer.info.calls++;
  const expected = source.use(reference.object, false); source.nodes.updateBefore(expected.owner);
  assert.ok(expected.owner.getAttributes().includes(attributes[0]));
  candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
  const program = packets(candidate)[0].binding.nodeProgram;
  assert.equal(program.vertex, expected.state.vertexShader); assert.equal(program.fragment, expected.state.fragmentShader);
  const draw = draws(candidate)[0], location = program.attributes.find(attribute => attribute.name === 'selectedPosition').location;
  const gpu = draw.streams.get(draw.pipeline.vertex.buffers.findIndex(layout =>
    layout.attributes.some(attribute => attribute.shaderLocation === location)));
  assert.ok(gpu.usage & 128); assert.equal(new Float32Array(draw.contents.get(gpu).buffer)[0], .375);
  assert.equal(candidate.computes.length, 0); assert.equal(candidate.renderer._nodeCaptureBytes, 0);
});

test('inherited compute callbacks compile, share dispatches, and retain source shaders and native storage', async t => {
  const reference = fixture(t, {background: false}), expectedState = inheritedKernel(reference);
  await reference.renderer.init();
  const graphics = sourceCode(reference), compute = new T.WGSLNodeBuilder(null, reference.renderer);
  compute.compute = expectedState.kernel; compute.build();
  const f = fixture(t, {background: false}), state = inheritedKernel(f), other = new T.Mesh(f.object.geometry, f.material);
  other.frustumCulled = false; other.position.x = 1; f.scene.add(other);
  let dispatches = 0;
  f.device.onCompute = pass => new Float32Array(storage(pass)[0].data).fill(++dispatches * .25);
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.computes.length, 1); assert.equal(state.initializations, 1); assert.equal(draws(f).length, 0);
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(f.computes[0].pipeline.compute.module.code), canonical(compute.computeShader));
  f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera);
  state.nextStep = .625; f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  state.nextStep = 99; await drain(f.renderer);
  assert.equal(canonical(packets(f)[0].binding.nodeProgram.vertex), canonical(graphics.vertexShader));
  assert.equal(canonical(packets(f)[0].binding.nodeProgram.fragment), canonical(graphics.fragmentShader));
  assert.equal(f.computes.length, 2, 'one shared kernel runs once per source frame across objects and renders');
  assert.equal(state.initializations, 1);
  const uniforms = pass => pass.bytes.flatMap(bytes => [...new Float32Array(bytes.buffer)]);
  assert.ok(uniforms(f.computes[0]).includes(.125)); assert.ok(uniforms(f.computes[1]).includes(.625));
  assert.ok(!uniforms(f.computes[1]).includes(99));
  const gpu = storage(f.computes[0])[0], submitted = draws(f);
  assert.equal(submitted.length, 6);
  assert.deepEqual(submitted.map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]), [.25, .25, .25, .25, .5, .5]);
  assert.ok(state.attribute.array.every(value => value === 0));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const frequency of ['frame', 'render', 'object'])
test(`inherited compute ${frequency} callbacks match source NodeManager contexts and update markers`, async t => {
  const candidate = fixture(t, {background: false}), reference = fixture(t, {background: false});
  const actual = inheritedKernel(candidate, frequency), expected = inheritedKernel(reference, frequency);
  await candidate.renderer.init(); await reference.renderer.init();
  const traces = [[], []];
  for (const [index, f, state] of [[0, candidate, actual], [1, reference, expected]]) state.observe = (stage, frame) => {
    assert.equal(frame.renderer, f.renderer);
    if (frame.object !== null) {
      assert.equal(frame.object, f.object); assert.equal(frame.scene, f.scene);
      assert.equal(frame.camera, f.camera); assert.equal(frame.material, f.material); assert.equal(frame.compute, null);
    } else {
      assert.equal(frame.scene, null); assert.equal(frame.camera, null); assert.equal(frame.material, null);
      assert.equal(frame.compute, stage === 'current' ? null : state.kernel);
    }
    traces[index].push([stage, frame.object === null ? 'compute' : 'draw']);
  };
  reference.renderer.debug.onNodeBuilderCreated = null;
  const nodes = new SourceNodeManager(reference.renderer,
    {createNodeBuilder: object => new T.WGSLNodeBuilder(object, reference.renderer)});
  nodes.nodeFrame = reference.renderer._nodes.nodeFrame;
  const graphics = sourceCode(reference), renderObject = {...reference, getNodeBuilderState: () => graphics};
  // Use the pinned NodeManager for all callback ordering and frame contexts;
  // this source side needs no command submission to observe its node lifecycle.
  reference.renderer.compute = node => {
    const frame = nodes.nodeFrame, previous = frame.renderId;
    frame.renderId = ++reference.renderer.info.calls;
    try { nodes.updateBeforeForCompute(node); nodes.updateForCompute(node); nodes.updateAfterForCompute(node); }
    finally { frame.renderId = previous; }
  };
  for (const nextFrame of [false, false, true]) {
    if (nextFrame) { candidate.renderer._nodes.nodeFrame.update(); nodes.nodeFrame.update(); }
    candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
    nodes.nodeFrame.renderId = ++reference.renderer.info.calls;
    nodes.updateBefore(renderObject); nodes.updateForRender(renderObject); nodes.updateAfter(renderObject);
  }
  assert.deepEqual(traces[0], traces[1]);
  assert.equal(candidate.computes.length, frequency === 'frame' ? 2 : 3);
  assert.equal(candidate.renderer._nodeCaptureBytes, 0);
});

test('inherited compute false before callbacks retry within the same source frame', async t => {
  const f = fixture(t, {background: false}), state = inheritedKernel(f), stages = [];
  state.retryBefore = true; state.observe = stage => stages.push(stage);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.computes.length, 0); assert.equal(state.initializations, 0);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.computes.length, 1); assert.equal(state.initializations, 1);
  assert.deepEqual(stages, ['before', 'current', 'after', 'before']);
});

for (const method of ['setup', 'generate'])
test(`render-triggered compute retains its ${method === 'setup' ? 'shared setup profile' : 'pinned generate'} boundary`, async t => {
  const f = fixture(t, {background: false}), state = inheritedKernel(f);
  state.kernel[method] = function (...args) { return T.ComputeNode.prototype[method].apply(this, args); };
  if (method === 'setup') f.material.transparent = true;
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0);
});

test('compute lifecycle accessor rejection does not invoke application getters', async t => {
  const f = fixture(t, {background: false}), state = inheritedKernel(f); let reads = 0;
  Object.defineProperty(Object.getPrototypeOf(state.kernel), 'update', {
    get() { reads++; return () => {}; },
  });
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(reads, 0); assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0);
});

for (const attachment of [false, true])
test(`inherited compute refuses ${attachment ? 'the pending parent attachment' : 'a changed earlier draw storage upload'}`, async t => {
  const f = fixture(t, {background: false}), state = inheritedKernel(f);
  if (attachment) {
    const target = new T.RenderTarget(16, 16); t.after(() => target.dispose());
    state.kernel.computeNode = N.Fn(() => { state.values.element(0).assign(N.textureLoad(target.texture, N.ivec2(0))); })();
    f.renderer.setRenderTarget(target);
  } else {
    const material = new T.MeshBasicNodeMaterial(), reader = new T.Mesh(f.object.geometry, material);
    material.colorNode = state.values.element(0).xyz; reader.renderOrder = -1; reader.frustumCulled = false; f.scene.add(reader);
    reader.onAfterRender = () => { state.attribute.needsUpdate = true; };
    t.after(() => material.dispose());
  }
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const inherited of [false, true])
test(`${inherited ? 'inherited' : 'ordinary'} geometry compute updates unchanged storage already read by a parent draw`, async t => {
  const f = fixture(t, {background: false}), state = inherited ? inheritedKernel(f) : sharedKernel(f);
  const material = new T.MeshBasicNodeMaterial(), reader = new T.Mesh(f.object.geometry, material);
  material.colorNode = state.values.element(0).xyz; reader.renderOrder = -1; reader.frustumCulled = false; f.scene.add(reader);
  t.after(() => material.dispose());
  let dispatches = 0;
  f.device.onCompute = pass => new Float32Array(storage(pass)[0].data).fill(++dispatches * .25);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  state.attribute.array.fill(99);
  await drain(f.renderer);
  assert.equal(f.computes.length, 2);
  assert.deepEqual(f.events, [state.kernel.name, 'render', state.kernel.name, 'render']);
  const gpu = storage(f.computes[0])[0];
  assert.equal(storage(f.computes[1])[0], gpu);
  assert.deepEqual(draws(f).map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]), [.25, .25, .5, .5]);
  assert.equal(f.device.writes.filter(write => write.buffer === gpu).length, 1,
    'earlier captured draws must not replay their CPU payload over the computed allocation');
  assert.ok(state.attribute.array.every(value => value === 99), 'compute never publishes GPU bytes into the public array');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('overlapping geometry compute matches pinned source upload, callback and submit boundaries', async t => {
  const candidate = fixture(t, {background: false}), reference = fixture(t, {background: false});
  const states = [candidate, reference].map(f => {
    const state = sharedKernel(f), material = new T.MeshBasicNodeMaterial();
    const reader = new T.Mesh(f.object.geometry, material);
    material.colorNode = state.values.element(0).xyz; reader.renderOrder = -1; reader.frustumCulled = false; f.scene.add(reader);
    t.after(() => material.dispose());
    // The recording device models the same deterministic write on each side.
    // The source renderer/NodeFrame/backend below decide when it executes;
    // this checks queue-visible bytes, not WGSL arithmetic or driver validity.
    f.device.onCompute = pass => {
      const words = new Float32Array(storage(pass)[0].data); words[0] += .125;
    };
    return {...state, reader};
  });
  await candidate.renderer.init(); await reference.renderer.init();
  const actual = states[0], expected = states[1], source = sourceGeometryPrograms(t, reference);
  const backend = new SourceWebGPUBackend(); backend.device = reference.device;
  const attributes = new SourceAttributes({
    createStorageAttribute: attribute => backend.attributeUtils.createAttribute(attribute, 128 | 32 | 8 | 4),
    updateAttribute: attribute => backend.attributeUtils.updateAttribute(attribute),
  }, {createStorageAttribute() {}});
  const upload = () => attributes.update(expected.attribute, AttributeType.STORAGE);
  const resourceGroup = {}, compiled = new Set(), renderContext = {camera: reference.camera,
    textures: null, depthTexture: null, occlusionQueryCount: 0};
  source.nodes.nodeFrame = reference.renderer._nodes.nodeFrame;
  source.nodes.needsRefresh = () => T.RenderObjectRefreshType.FULL;
  const renderer = {_initialized: true, _isDeviceLost: false, _currentRenderBundle: null,
    _currentRenderContext: renderContext, backend, info: reference.renderer.info, inspector: {beginCompute() {}, finishCompute() {}},
    _nodes: source.nodes, _objects: {get: object => source.use(object).owner}, _geometries: {updateForRender() {}},
    _bindings: {updateForRender: upload, updateForCompute() {
      upload(); backend.get(resourceGroup).group = {entries: [{binding: 0, resource: {buffer: backend.get(expected.attribute).buffer}}]};
    }, getForCompute: () => [resourceGroup], deleteForCompute() {}},
    _pipelines: {has: node => compiled.has(node), delete: node => compiled.delete(node), updateForRender() {}, isReady: () => true,
      getForCompute(node) {
        const state = source.nodes.getForCompute(node); compiled.add(node);
        backend.get(state).pipeline = {compute: {module: {label: node.name, code: state.computeShader}}};
        return state;
      }}};
  backend.renderer = renderer;
  reference.renderer.compute = (...args) => SourceRenderer.prototype.compute.apply(renderer, args);
  const sourceDrawValues = [], sourceShaders = [];
  for (let turn = 0; turn < 2; turn++) {
    candidate.renderer._nodes.nodeFrame.update(); reference.renderer._nodes.nodeFrame.update();
    candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
    reference.renderer.info.calls++;
    const encoder = reference.device.createCommandEncoder(), pass = encoder.beginRenderPass({label: 'source-parent'});
    Object.assign(backend.get(renderContext), {encoder, currentPass: pass, renderBundles: []});
    backend.draw = () => {
      pass.setPipeline({vertex: {entryPoint: 'main', buffers: []}});
      pass.setBindGroup(0, {entries: [{binding: 0, resource: {buffer: backend.get(expected.attribute).buffer}}]});
      pass.draw(3);
    };
    for (const object of [expected.reader, reference.object]) {
      SourceRenderer.prototype._renderObjectDirect.call(renderer, object, object.material, reference.scene,
        reference.camera, N.lights([]), null, null, null);
      sourceShaders.push(source.use(object).state);
    }
    backend.finishRender(renderContext);
    const gpu = backend.get(expected.attribute).buffer;
    sourceDrawValues.push(...reference.device.snapshots.at(-1).map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]));
  }
  const gpu = storage(candidate.computes[0])[0], canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.deepEqual(draws(candidate).map(draw => new Float32Array(draw.contents.get(gpu).buffer)[0]), sourceDrawValues);
  assert.deepEqual(sourceDrawValues, [.125, .125, .25, .25]);
  assert.deepEqual(candidate.events, reference.events);
  assert.equal(canonical(candidate.computes[0].pipeline.compute.module.code), canonical(reference.computes[0].pipeline.compute.module.code));
  const programs = candidate.captures.flatMap(frame => frame.draws).map(draw => draw.binding.nodeProgram);
  for (let i = 0; i < programs.length; i++) {
    assert.equal(canonical(programs[i].vertex), canonical(sourceShaders[i].vertexShader));
    assert.equal(canonical(programs[i].fragment), canonical(sourceShaders[i].fragmentShader));
  }
  assert.deepEqual(actual.attribute.array, expected.attribute.array);
  assert.ok(actual.attribute.array.every(value => value === 0));
  assert.equal(candidate.renderer._nodeCaptureBytes, 0);
});

for (const mutation of ['dynamic', 'layout', 'multiple-versions'])
test(`geometry compute retains its overlap refusal for ${mutation} storage histories`, async t => {
  const f = fixture(t, {background: false}), state = sharedKernel(f), material = new T.MeshBasicNodeMaterial();
  const reader = new T.Mesh(f.object.geometry, material);
  material.colorNode = state.values.element(0).xyz; reader.renderOrder = -2; reader.frustumCulled = false; f.scene.add(reader);
  t.after(() => material.dispose());
  if (mutation === 'dynamic') state.attribute.setUsage(T.DynamicDrawUsage);
  else if (mutation === 'layout') reader.onAfterRender = () => { state.attribute.array = new Float32Array(8); state.attribute.count = 2; };
  else {
    reader.onAfterRender = () => { state.attribute.needsUpdate = true; };
    const later = new T.Mesh(f.object.geometry, material); later.renderOrder = -1; later.frustumCulled = false; f.scene.add(later);
  }
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.device.writes.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('geometry compute rechecks the storage identity selected by a late uniform getter', async t => {
  const f = fixture(t, {background: false}), earlier = new T.StorageBufferAttribute(1, 4);
  const original = new T.StorageBufferAttribute(1, 4), values = N.storage(original, 'vec4', 1);
  const step = N.uniform(.125).setGroup(N.uniformGroup('lateResourceSelection', -1));
  const material = new T.MeshBasicNodeMaterial(), reader = new T.Mesh(f.object.geometry, material);
  material.colorNode = N.storage(earlier, 'vec4', 1).element(0).xyz;
  reader.renderOrder = -1; reader.frustumCulled = false; f.scene.add(reader);
  let computeBuilder;
  const kernel = N.Fn(builder => {
    computeBuilder = builder;
    const amount = step.toVar();
    values.element(0).addAssign(N.vec4(amount, 0, 0, 0));
  })().compute(1);
  f.material.geometryNode = kernel;
  t.after(() => material.dispose());
  await f.renderer.init(); await f.renderer.compileComputeAsync(kernel);
  const writes = f.device.writes.length;
  const uniforms = computeBuilder.getBindings().flatMap(group => group.bindings).find(binding => binding.isUniformBuffer);
  const update = uniforms.update;
  let changes = 0, readingBinding = false;
  uniforms.update = function () {
    readingBinding = true;
    try { return update.call(this); } finally { readingBinding = false; }
  };
  Object.defineProperty(step, 'value', {configurable: true, get() {
    if (readingBinding && changes === 0) {
      changes++; values.value = earlier; earlier.needsUpdate = true;
    }
    return .125;
  }});
  // The warmed graph enumerates the original storage before binding.update()
  // reads the uniform; the actual captured storage now names the earlier draw.
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(changes, 1); assert.equal(values.value, earlier);
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0);
  assert.equal(f.device.writes.length, writes, 'late binding changes are refused before any queue write');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a shared ComputeNode updates once per source frame across objects and repeated renders', async t => {
  const f = fixture(t), {kernel} = sharedKernel(f); let initializations = 0;
  kernel.onInit(() => { initializations++; });
  const other = new T.Mesh(f.object.geometry, f.material); other.frustumCulled = false; other.position.x = 1; f.scene.add(other);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.computes.length, 1); assert.equal(initializations, 1);
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.computes.length, 2); assert.equal(initializations, 1);
  assert.ok(f.captures.every(frame => frame.draws.filter(draw => draw.object === f.object || draw.object === other).length === 2));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued geometry kernels retain uniforms and draw the shared buffer at each FIFO boundary', async t => {
  const f = fixture(t, {background: false}), state = jelly(f); let dispatch = 0;
  f.device.onCompute = pass => {
    if (pass.pipeline.compute.module.label === 'geometry-step') new Float32Array(storage(pass)[0].data).fill(++dispatch * .25);
  };
  await f.renderer.init(); f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  state.step.value = .625; f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  state.step.value = 99; await drain(f.renderer);
  const kernels = f.computes.filter(pass => pass.pipeline.compute.module.label === 'geometry-step');
  assert.equal(kernels.length, 2);
  const uniforms = pass => pass.bytes.flatMap(bytes => [...new Float32Array(bytes.buffer)]);
  assert.ok(uniforms(kernels[0]).includes(.125)); assert.ok(uniforms(kernels[1]).includes(.625));
  assert.ok(!uniforms(kernels[1]).includes(99));
  const gpu = storage(kernels[0])[0], meshDraws = draws(f);
  assert.equal(meshDraws.length, 2);
  assert.equal(new Float32Array(meshDraws[0].contents.get(gpu).buffer)[0], .25);
  assert.equal(new Float32Array(meshDraws[1].contents.get(gpu).buffer)[0], .5);
  assert.ok(state.attributes[0].array.every(value => value === 0), 'GPU writes do not publish into CPU storage');
  assert.deepEqual(f.events, ['geometry-init', 'geometry-step', 'render', 'geometry-step', 'render']);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const vertex of [false, true])
test(`geometry compute refuses changed history of an earlier ${vertex ? 'vertex stream' : 'shader storage'} before compute uploads`, async t => {
  const f = fixture(t, {background: false}), {attribute, values} = sharedKernel(f);
  const material = new T.MeshBasicNodeMaterial(), geometry = new T.PlaneGeometry(), reader = new T.Mesh(geometry, material);
  reader.frustumCulled = false; reader.renderOrder = -1; f.scene.add(reader);
  t.after(() => { geometry.dispose(); material.dispose(); });
  if (vertex) {
    const position = new T.StorageBufferAttribute(geometry.attributes.position.count, 3), positions = N.storage(position, 'vec3', position.count);
    geometry.setAttribute('position', position);
    f.material.geometryNode = N.Fn(() => { positions.element(N.instanceIndex).assign(N.vec3(0)); })().compute(position.count);
    reader.onAfterRender = () => { position.needsUpdate = true; };
  } else {
    material.colorNode = values.element(0).xyz;
    reader.onAfterRender = () => { attribute.needsUpdate = true; };
  }
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.device.writes.length, 0);
  assert.ok(attribute.array.every(value => value === 0)); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('geometry compute shares an earlier generated vertex allocation without replaying its upload', async t => {
  const f = fixture(t, {background: false}), geometry = new T.PlaneGeometry(), material = new T.MeshBasicNodeMaterial();
  const attribute = new T.StorageBufferAttribute(geometry.attributes.position.count, 3);
  const values = N.storage(attribute, 'vec3', attribute.count);
  geometry.setAttribute('position', attribute);
  const reader = new T.Mesh(geometry, material); reader.frustumCulled = false; reader.renderOrder = -1; f.scene.add(reader);
  f.material.geometryNode = N.Fn(() => { values.element(N.instanceIndex).assign(N.vec3(.25)); })().compute(attribute.count).setName('shared-vertices');
  t.after(() => { geometry.dispose(); material.dispose(); });
  f.device.onCompute = pass => new Float32Array(storage(pass)[0].data).fill(.25);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const gpu = storage(f.computes[0])[0];
  const first = draws(f).find(draw => [...draw.streams.values()].includes(gpu));
  assert.ok(first, 'the earlier draw borrows the exact compute-written vertex buffer');
  assert.equal(new Float32Array(first.contents.get(gpu).buffer)[0], .25);
  assert.deepEqual(f.events, ['shared-vertices', 'render']);
  assert.equal(f.device.writes.filter(write => write.buffer === gpu).length, 1);
  assert.equal(attribute.itemSize, 4); assert.ok(attribute.array.every(value => value === 0));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('geometry compute checks final sampled texture identities against earlier draw reads', async t => {
  const f = fixture(t, {background: false});
  const image = new T.DataTexture(new Uint8Array([128, 64, 32, 255]), 1, 1); image.needsUpdate = true;
  const material = new T.MeshBasicNodeMaterial(), geometry = new T.PlaneGeometry(), reader = new T.Mesh(geometry, material);
  material.colorNode = N.texture(image).rgb; reader.renderOrder = -1; reader.frustumCulled = false; f.scene.add(reader);
  const output = N.instancedArray(1, 'vec4');
  f.material.geometryNode = N.Fn(() => { output.element(0).assign(N.textureLoad(image, N.ivec2(0))); })().compute(1);
  t.after(() => { image.dispose(); geometry.dispose(); material.dispose(); });
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

function sharedTextureKernel(t, f, {dimension = '2d', mipmaps = false} = {}) {
  const texture = dimension === '3d' ? new T.Storage3DTexture(4, 4, 2)
    : dimension === '2d-array' ? new T.StorageArrayTexture(4, 4, 2) : new T.StorageTexture(4, 4);
  Object.assign(texture, {generateMipmaps: mipmaps, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  const sample = dimension === '3d' ? N.texture3D(texture, N.vec3(.5)) : N.texture(texture, N.vec2(.5));
  const output = N.storageTexture(texture).toWriteOnly();
  if (dimension === '2d-array') { sample.depthNode = N.int(1); output.depthNode = N.int(1); }
  const step = N.uniform(.25).setName('sharedTextureStep');
  const kernel = N.Fn(() => {
    N.textureStore(output, dimension === '3d' ? N.uvec3(0, 0, 1) : N.uvec2(0), N.vec4(step, 0, .5, 1));
  })().compute(1).setName('shared-storage-texture');
  f.material.geometryNode = kernel; f.material.colorNode = sample.rgb;
  const material = new T.MeshBasicNodeMaterial(), reader = new T.Mesh(f.object.geometry, material);
  material.colorNode = sample.rgb; reader.frustumCulled = false; reader.renderOrder = -1; f.scene.add(reader);
  t.after(() => { kernel.dispose(); texture.dispose(); material.dispose(); });
  return {texture, sample, output, step, kernel, reader};
}
const drawTexture = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .find(entry => entry.resource?.texture)?.resource.texture;
const computeTexture = pass => [...pass.groups.values()].flatMap(group => group.entries)
  .find(entry => entry.resource?.texture)?.resource.texture;

for (const dimension of ['2d', '2d-array', '3d'])
test(`geometry compute shares one ${dimension} storage texture with earlier pending sampled draws`, async t => {
  const f = fixture(t, {background: false}), state = sharedTextureKernel(t, f, {dimension}), observed = [];
  let updates = 0;
  f.device.queue.writeTexture = () => { throw Error('GPU-authored storage cannot acquire a CPU upload'); };
  f.device.onCompute = pass => { computeTexture(pass).recordedValue = ++updates * .25; };
  const submit = f.device.queue.submit;
  f.device.queue.submit = commands => {
    submit(commands);
    for (const passes of commands) for (const pass of passes) for (const draw of pass.draws)
      if (draw.pipeline.vertex.entryPoint === 'main') observed.push(drawTexture(draw).recordedValue);
  };
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); f.renderer._nodes.nodeFrame.update();
  state.step.value = .5; f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(f.events, ['shared-storage-texture', 'render', 'shared-storage-texture', 'render']);
  assert.deepEqual(observed, [.25, .25, .5, .5], 'both pending draws see compute before their parent submission');
  const native = computeTexture(f.computes[0]);
  assert.equal(computeTexture(f.computes[1]), native);
  assert.ok(draws(f).every(draw => drawTexture(draw) === native));
  assert.equal(native.dimension, dimension === '3d' ? '3d' : '2d');
  assert.equal(native.depthOrArrayLayers, dimension === '2d' ? 1 : 2);
  assert.equal(native.mipLevelCount, 1);
  const source = sourceCode(f), compute = new T.WGSLNodeBuilder(null, f.renderer);
  compute.compute = state.kernel; compute.build();
  const program = packets(f)[0].binding.nodeProgram;
  assert.equal(program.vertex, source.vertexShader); assert.equal(program.fragment, source.fragmentShader);
  assert.equal(f.computes[0].pipeline.compute.module.code, compute.computeShader);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('warm native bundles retain the sampled storage residency after their material map is replaced', async t => {
  const f = fixture(t, {background: false}), state = sharedTextureKernel(t, f);
  const replacement = new T.StorageTexture(4, 4), bundle = new T.BundleGroup();
  Object.assign(replacement, {generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  state.reader.material.colorNode = null; state.reader.material.map = state.texture;
  bundle.renderOrder = -1; bundle.add(state.reader); f.scene.add(bundle);
  t.after(() => replacement.dispose());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const native = computeTexture(f.computes[0]);
  assert.ok(draws(f).every(draw => drawTexture(draw) === native));
  state.reader.material.map = replacement; state.step.value = .5; f.renderer._nodes.nodeFrame.update();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.captures.at(-1).draws[0].bundleWarm, true);
  assert.ok(draws(f).every(draw => drawTexture(draw) === native), 'warm texture bindings keep their captured allocation');
  assert.equal(computeTexture(f.computes[1]), native);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const mutation of ['mip-chain', 'resize', 'version', 'source', 'dispose', 'recreate'])
test(`shared geometry texture compute refuses ${mutation} before writing a pending parent residency`, async t => {
  const f = fixture(t, {background: false}), state = sharedTextureKernel(t, f, {mipmaps: mutation === 'mip-chain'});
  let identity, retained;
  state.reader.onAfterRender = () => {
    const owner = f.renderer._getSampledTextureInputs();
    identity = owner.storageIdentityOf(state.texture);
    retained = owner.captureBinding({texture: state.texture, isSampledTexture: true});
    if (mutation === 'resize') state.texture.image.width = 8;
    else if (mutation === 'version') state.texture.needsUpdate = true;
    else if (mutation === 'source') state.texture.source = new T.TextureSource({...state.texture.image});
    else if (mutation === 'dispose' || mutation === 'recreate') {
      state.texture.dispose();
      if (mutation === 'recreate') {
        const owner = f.renderer._getSampledTextureInputs();
        const lease = owner.captureBinding({texture: state.texture, isSampledTexture: true});
        assert.notEqual(owner.capturedStorageIdentity(lease)?.identity, identity);
        owner.release(lease);
      }
    }
  };
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.device.writes.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  if (mutation !== 'dispose' && mutation !== 'recreate')
    assert.equal(retained.entry.alive, true, 'rejected preparation cannot retire the earlier reader residency');
  f.renderer._getSampledTextureInputs().release(retained);
  if (mutation === 'mip-chain') assert.equal(identity, null, 'mip generation retains its source binding-order boundary');
});

test('shared storage texture permission does not allow compute from arbitrary object callbacks', async t => {
  const f = fixture(t, {background: false}), state = sharedTextureKernel(t, f);
  state.reader.onAfterRender = () => f.renderer.compute(state.kernel);
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('shared texture compute rechecks late binding selection before retiring an earlier draw residency', async t => {
  const f = fixture(t, {background: false}), state = sharedTextureKernel(t, f);
  const other = new T.StorageTexture(4, 4); t.after(() => other.dispose());
  state.output.value = other;
  const step = N.uniform(.125).setGroup(N.uniformGroup('lateTextureSelection', -1));
  let computeBuilder;
  const kernel = N.Fn(builder => {
    computeBuilder = builder;
    const value = step.toVar();
    N.textureStore(state.output, N.uvec2(0), N.vec4(value));
  })().compute(1);
  f.material.geometryNode = kernel; t.after(() => kernel.dispose());
  await f.renderer.init(); await f.renderer.compileComputeAsync(kernel);
  const group = computeBuilder.getBindings().flatMap(group => group.bindings).find(binding => binding.isUniformBuffer);
  const update = group.update;
  let reading = false, changes = 0;
  group.update = function() { reading = true; try { return update.call(this); } finally { reading = false; } };
  Object.defineProperty(step, 'value', {configurable: true, get() {
    if (reading && changes === 0) {
      changes++; state.output.value = state.texture; state.texture.image.width = 8;
    }
    return .125;
  }});
  const writes = f.device.writes.length;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(changes, 1); assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0);
  assert.equal(f.device.writes.length, writes); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('geometry compute conservatively refuses sampling its pending parent attachment', async t => {
  const f = fixture(t, {background: false}), target = new T.RenderTarget(16, 16), output = N.instancedArray(1, 'vec4');
  f.material.geometryNode = N.Fn(() => { output.element(0).assign(N.textureLoad(target.texture, N.ivec2(0))); })().compute(1);
  t.after(() => target.dispose()); await f.renderer.init(); f.renderer.setRenderTarget(target);
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('accepted geometry initialization survives a later draw callback failure', async t => {
  const f = fixture(t, {background: false}), state = jelly(f);
  f.material.colorNode = N.uniform(new T.Color(.5, .25, .125)).onObjectUpdate(() => { throw Error('late color update'); });
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), /late color update/); await drain(f.renderer);
  assert.equal(state.initializations, 1); assert.deepEqual(f.events, ['geometry-init', 'geometry-step']);
  assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('nonshared geometry graph replacement rejects stale descriptors before dispatch', async t => {
  const f = fixture(t, {background: false}); f.material.transparent = true; jelly(f); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const before = packets(f)[0].binding.nodeProgram, count = f.computes.length;
  f.material.geometryNode = N.Fn(() => {})();
  assert.throws(() => f.renderer._nodeProgramSupport.captureDraw(before,
    {scene: f.scene, camera: f.camera, object: f.object, material: f.material, renderId: 2, frameId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(f.computes.length, count);
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.notEqual(packets(f)[1].binding.nodeProgram, before); assert.equal(f.computes.length, count);
});

for (const removal of [false, true])
test(`shared geometry graph ${removal ? 'removal' : 'replacement'} waits for source material invalidation`, async t => {
  const reference = fixture(t, {background: false}), candidate = fixture(t, {background: false});
  const expected = jelly(reference); jelly(candidate); await reference.renderer.init(); await candidate.renderer.init();
  const source = sourceGeometryPrograms(t, reference), first = source.use().state, records = new WeakMap();
  const upload = new SourceAttributeUtils({device: reference.device, get(attribute) {
    let record = records.get(attribute); if (!record) records.set(attribute, record = {}); return record;
  }});
  reference.renderer.compute = kernel => {
    for (const group of source.nodes.getForCompute(kernel).bindings) for (const binding of group.bindings)
      if (binding.isStorageBuffer) upload.createAttribute(binding.attribute, 160);
  };
  expected.kernels[0].onInitFunction();
  candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
  const before = packets(candidate)[0].binding.nodeProgram, count = candidate.computes.length;
  const originalGraphs = [reference.material.geometryNode, candidate.material.geometryNode];
  for (const f of [reference, candidate]) f.material.geometryNode = removal ? null : N.Fn(() => {})();
  reference.renderer.info.calls++; assert.equal(source.use().state, first);
  candidate.renderer._nodes.nodeFrame.update(); candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
  assert.equal(packets(candidate)[1].binding.nodeProgram, before); assert.equal(candidate.computes.length, count + 1);
  for (const f of [reference, candidate]) f.material.needsUpdate = true;
  reference.renderer.info.calls++; const changed = source.use().state; assert.notEqual(changed, first);
  candidate.renderer._nodes.nodeFrame.update(); candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
  const next = packets(candidate)[2].binding.nodeProgram;
  assert.notEqual(next, before); assert.equal(candidate.computes.length, count + 1);
  assert.equal(next.vertex, changed.vertexShader); assert.equal(next.fragment, changed.fragmentShader);
  if (removal) {
    reference.material.geometryNode = originalGraphs[0]; candidate.material.geometryNode = originalGraphs[1];
    reference.renderer.info.calls++; assert.equal(source.use().state, changed);
    candidate.renderer._nodes.nodeFrame.update(); candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
    assert.equal(packets(candidate).at(-1).binding.nodeProgram, next); assert.equal(candidate.computes.length, count + 1);
    for (const f of [reference, candidate]) f.material.needsUpdate = true;
    reference.renderer.info.calls++; const restored = source.use().state; assert.notEqual(restored, changed);
    candidate.renderer._nodes.nodeFrame.update(); candidate.renderer.render(candidate.scene, candidate.camera); await drain(candidate.renderer);
    const program = packets(candidate).at(-1).binding.nodeProgram;
    assert.notEqual(program, next); assert.equal(candidate.computes.length, count + 3);
    assert.equal(program.vertex, restored.vertexShader); assert.equal(program.fragment, restored.fragmentShader);
  }
});

test('hidden geometry constructors stay inert until selected for a source render', async t => {
  const f = fixture(t), state = jelly(f); f.object.visible = false; await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(state.builds, 0); assert.equal(state.initializations, 0); assert.equal(f.computes.length, 0);
  f.object.visible = true; f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(state.builds, 1); assert.equal(state.initializations, 1); assert.equal(f.computes.length, 2);
});

for (const inherited of [false, true])
test(`${inherited ? 'inherited' : 'ordinary'} compute cannot admit a generated graphics storage write`, async t => {
  const f = fixture(t, {background: false}), output = N.instancedArray(1, 'float');
  f.material.geometryNode = N.Fn(() => { output.element(0).assign(1); })().compute(1);
  let updates = 0;
  if (inherited) {
    class CustomStep extends T.ComputeNode { update() { updates++; } }
    f.material.geometryNode = new CustomStep(f.material.geometryNode.computeNode, [1]);
    f.material.geometryNode.count = 1; f.material.geometryNode.updateType = 'object';
  }
  f.material.colorNode = N.Fn(() => { output.element(0).assign(2); return N.vec3(output.element(0)); })();
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(updates, 0);
});


test('geometry compute uses source compute callback contexts and restores its parent draw', async t => {
  const f = fixture(t, {background: false}), calls = [], output = N.instancedArray(1, 'float');
  const observe = (stage, frame) => calls.push({stage, renderer: frame.renderer, scene: frame.scene,
    object: frame.object, camera: frame.camera, material: frame.material, compute: frame.compute});
  const gain = N.uniform(.5).onObjectUpdate(frame => { observe('current', frame); return .5; });
  gain.updateBeforeType = gain.updateAfterType = 'object';
  gain.updateBefore = frame => observe('before', frame); gain.updateAfter = frame => observe('after', frame);
  const kernel = N.Fn(() => { output.element(0).assign(gain); })().compute(1);
  f.material.geometryNode = kernel; f.material.colorNode = N.vec3(gain);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  for (const stage of ['before', 'current', 'after']) {
    const compute = calls.filter(call => call.stage === stage && call.object === null);
    assert.equal(compute.length, 1);
    assert.deepEqual(compute[0], {stage, renderer: f.renderer, scene: null, object: null,
      camera: null, material: null, compute: stage === 'current' ? null : kernel});
    assert.ok(calls.some(call => call.stage === stage && call.scene === f.scene && call.object === f.object &&
      call.camera === f.camera && call.material === f.material && call.compute === null));
  }
  assert.equal(f.renderer._nodes.nodeFrame.object, null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('public geometry compile preserves sequential pipeline callbacks, live frame, and restored render ID', async t => {
  const f = fixture(t, {background: false}), order = [], contexts = [], constructed = [];
  let stage = 0, prepared = false;
  const make = (object, name) => {
    const values = N.instancedArray(1, 'float');
    object.material.geometryNode = N.Fn(() => {
      order.push(name + ':construct'); constructed.push([name, stage, prepared]);
      const kernel = N.Fn(() => { values.element(0).addAssign(.125); })().compute(1).setName(name);
      kernel.updateBeforeType = 'render'; return kernel;
    })();
    const gain = N.uniform(.5).onObjectUpdate(frame => {
      order.push(name + ':current'); contexts.push([frame.renderId, frame.frameId]);
      if (name === 'first') stage = 2; return .5;
    });
    gain.updateBeforeType = gain.updateAfterType = 'object';
    gain.updateBefore = frame => {
      assert.equal(frame.renderer._isPreCompiling, true); order.push(name + ':before');
      if (name === 'first') stage = 1;
    };
    gain.updateAfter = frame => {
      assert.equal(frame.renderer._isPreCompiling, true); order.push(name + ':after');
      if (name === 'first') stage = 3;
    };
    object.material.colorNode = N.vec3(gain);
  };
  make(f.object, 'first'); f.object.renderOrder = -1;
  const material = new T.MeshBasicNodeMaterial(), other = new T.Mesh(new T.PlaneGeometry(), material);
  other.frustumCulled = false; f.scene.add(other); make(other, 'second');
  t.after(() => { material.dispose(); other.geometry.dispose(); });
  const pipeline = f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync = async descriptor => {
    assert.notEqual(f.renderer._isPreCompiling, true, 'precompilation state does not escape across awaits');
    order.push('pipeline'); const result = await pipeline(descriptor); prepared = true; return result;
  };
  await f.renderer.init();
  f.renderer.info.calls = 91; f.renderer._nodes.nodeFrame.renderId = 7;
  const initialFrame = f.renderer._nodes.nodeFrame.frameId;
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(order.filter(value => value !== 'pipeline'),
    ['first:construct', 'first:before', 'first:current', 'first:after',
      'second:construct', 'second:before', 'second:current', 'second:after']);
  assert.deepEqual(constructed, [['first', 0, false], ['second', 3, true]],
    'the later lazy constructor observes the earlier current/after and completed pipeline effects');
  assert.ok(order.indexOf('pipeline') > order.indexOf('first:current'));
  assert.ok(order.indexOf('pipeline') < order.indexOf('first:after'));
  assert.ok(contexts.every(([renderId, frameId]) => renderId === 7 && frameId > initialFrame));
  assert.equal(f.computes.length, 2); assert.equal(f.renderer._nodes.nodeFrame.renderId, 7);
  assert.notEqual(f.renderer._isPreCompiling, true); assert.equal(draws(f).length, 0);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.computes.length, 2, 'RENDER kernels retain the source restored render epoch across compiles');
  assert.equal(constructed.length, 2, 'cached graphs do not repeat their lazy constructors');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('later geometry constructor failure retains completed compile effects and accepted kernels', async t => {
  const f = fixture(t, {background: false}), state = jelly(f), seen = [];
  const gain = N.uniform(.5); gain.updateAfterType = 'object';
  gain.updateAfter = () => { seen.push('first:after'); };
  f.material.colorNode = N.vec3(gain);
  const material = new T.MeshBasicNodeMaterial(), other = new T.Mesh(new T.PlaneGeometry(), material);
  other.frustumCulled = false; f.scene.add(other);
  t.after(() => { material.dispose(); other.geometry.dispose(); });
  material.geometryNode = N.Fn(() => {
    seen.push('second:construct'); assert.deepEqual(seen, ['first:after', 'second:construct']);
    throw Error('later geometry constructor');
  })();
  await f.renderer.init();
  await assert.rejects(f.renderer.compileAsync(f.scene, f.camera), /later geometry constructor/);
  assert.deepEqual(seen, ['first:after', 'second:construct']);
  assert.equal(state.builds, 1); assert.equal(state.initializations, 1);
  assert.deepEqual(f.events, ['geometry-init', 'geometry-step']);
  assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.notEqual(f.renderer._isPreCompiling, true);
});

test('failed geometry pipeline leaves later lazy constructors inactive', async t => {
  const f = fixture(t, {background: false}), state = jelly(f); let laterBuilds = 0, after = 0;
  const gain = N.uniform(.5); gain.updateAfterType = 'object'; gain.updateAfter = () => { after++; };
  f.material.colorNode = N.vec3(gain);
  const material = new T.MeshBasicNodeMaterial(), other = new T.Mesh(new T.PlaneGeometry(), material);
  other.frustumCulled = false; f.scene.add(other);
  t.after(() => { material.dispose(); other.geometry.dispose(); });
  material.geometryNode = N.Fn(() => { laterBuilds++; return N.Fn(() => {})().compute(1); })();
  await f.renderer.init();
  f.device.createRenderPipelineAsync = async () => { throw Error('geometry pipeline rejected'); };
  await assert.rejects(f.renderer.compileAsync(f.scene, f.camera), /geometry pipeline rejected/);
  assert.equal(state.builds, 1); assert.equal(state.initializations, 1);
  assert.equal(laterBuilds, 0); assert.equal(after, 0);
  assert.deepEqual(f.events, ['geometry-init', 'geometry-step']);
  assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.notEqual(f.renderer._isPreCompiling, true);
});

test('public geometry compile drains accepted kernels after a later source callback rejects', async t => {
  const f = fixture(t, {background: false}), state = jelly(f);
  f.material.colorNode = N.uniform(new T.Color(.5, .25, .125)).onObjectUpdate(() => { throw Error('late compile update'); });
  await f.renderer.init();
  await assert.rejects(f.renderer.compileAsync(f.scene, f.camera), /late compile update/);
  assert.equal(state.initializations, 1); assert.deepEqual(f.events, ['geometry-init', 'geometry-step']);
  assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.notEqual(f.renderer._isPreCompiling, true);
});


test('pre-init geometry compilation advances the source frame once across scene and output requests', async t => {
  const f = fixture(t, {background: false}), state = jelly(f), target = new T.RenderTarget(16, 16);
  f.renderer.setOutputRenderTarget(target); f.renderer.toneMapping = T.ReinhardToneMapping;
  t.after(() => target.dispose());
  const frame = f.renderer._nodes.nodeFrame, seen = [];
  state.step.onFrameUpdate(value => { seen.push(value.frameId); return .125; });
  const init = f.renderer.init; let initializedFrame = null;
  f.renderer.init = async function (...args) {
    const result = await init.apply(this, args); initializedFrame ??= frame.frameId; return result;
  };
  assert.equal(f.renderer.initialized, false);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(frame.frameId, initializedFrame + 1);
  assert.deepEqual(seen, [frame.frameId]); assert.equal(state.builds, 1); assert.equal(state.initializations, 1);
  assert.deepEqual(f.events, ['geometry-init', 'geometry-step']);
  assert.equal(f.captures.length, 2, 'scene and conversion output retain one compilation clock');
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(draws(f).length, 0);
  assert.notEqual(f.renderer._isPreCompiling, true);
});
