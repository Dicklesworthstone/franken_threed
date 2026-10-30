/** Source render-triggered compute through the native graphics/storage owners.
 * The device records dispatches and queue-visible bytes; it does not run WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
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

test('geometry constructors run once, public compilation computes, and named storage retains native residency', async t => {
  const f = fixture(t), state = jelly(f);
  const companionMaterial = new T.MeshBasicMaterial({color: 0x336699}), companion = new T.Mesh(new T.PlaneGeometry(), companionMaterial);
  companion.renderOrder = -1; companion.frustumCulled = false; f.scene.add(companion);
  t.after(() => { companion.geometry.dispose(); companionMaterial.dispose(); });
  await f.renderer.init();
  const raw = f.renderer._makeOutputRequests(f.scene, f.camera, null, false);
  assert.equal(state.builds, 1); assert.equal(state.initializations, 0); assert.equal(f.computes.length, 0);
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
test(`geometry compute refuses overlap with an earlier ${vertex ? 'vertex stream' : 'shader storage'} before compute uploads`, async t => {
  const f = fixture(t, {background: false}), {attribute, values} = sharedKernel(f);
  const material = new T.MeshBasicNodeMaterial(), geometry = new T.PlaneGeometry(), reader = new T.Mesh(geometry, material);
  reader.frustumCulled = false; reader.renderOrder = -1; f.scene.add(reader);
  t.after(() => { geometry.dispose(); material.dispose(); });
  if (vertex) {
    const position = new T.StorageBufferAttribute(geometry.attributes.position.count, 3), positions = N.storage(position, 'vec3', position.count);
    geometry.setAttribute('position', position);
    f.material.geometryNode = N.Fn(() => { positions.element(N.instanceIndex).assign(N.vec3(0)); })().compute(position.count);
  } else material.colorNode = values.element(0).xyz;
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_REENTRY'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.device.writes.length, 0);
  assert.ok(attribute.array.every(value => value === 0)); assert.equal(f.renderer._nodeCaptureBytes, 0);
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

test('geometry graph replacement rejects stale descriptors before dispatch and rebuilds the source variant', async t => {
  const f = fixture(t, {background: false}); jelly(f); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const before = packets(f)[0].binding.nodeProgram, count = f.computes.length;
  f.material.geometryNode = N.Fn(() => {})();
  assert.throws(() => f.renderer._nodeProgramSupport.captureDraw(before,
    {scene: f.scene, camera: f.camera, object: f.object, material: f.material, renderId: 2, frameId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(f.computes.length, count);
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.notEqual(packets(f)[1].binding.nodeProgram, before); assert.equal(f.computes.length, count);
});

test('hidden geometry constructors stay inert until selected for a source render', async t => {
  const f = fixture(t), state = jelly(f); f.object.visible = false; await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(state.builds, 0); assert.equal(state.initializations, 0); assert.equal(f.computes.length, 0);
  f.object.visible = true; f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(state.builds, 1); assert.equal(state.initializations, 1); assert.equal(f.computes.length, 2);
});

test('compute-only assignment exemption cannot admit a generated graphics storage write', async t => {
  const f = fixture(t, {background: false}), output = N.instancedArray(1, 'float');
  f.material.geometryNode = N.Fn(() => { output.element(0).assign(1); })().compute(1);
  f.material.colorNode = N.Fn(() => { output.element(0).assign(2); return N.vec3(output.element(0)); })();
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(f.computes.length, 0); assert.equal(draws(f).length, 0);
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
  const f = fixture(t, {background: false}), order = [], contexts = [];
  const make = (object, name) => {
    const values = N.instancedArray(1, 'float');
    object.material.geometryNode = N.Fn(() => { values.element(0).addAssign(.125); })().compute(1).setName(name);
    object.material.geometryNode.updateBeforeType = 'render';
    const gain = N.uniform(.5).onObjectUpdate(frame => { order.push(name + ':current'); contexts.push([frame.renderId, frame.frameId]); return .5; });
    gain.updateBeforeType = gain.updateAfterType = 'object';
    gain.updateBefore = frame => { assert.equal(frame.renderer._isPreCompiling, true); order.push(name + ':before'); };
    gain.updateAfter = frame => { assert.equal(frame.renderer._isPreCompiling, true); order.push(name + ':after'); };
    object.material.colorNode = N.vec3(gain);
  };
  make(f.object, 'first'); f.object.renderOrder = -1;
  const material = new T.MeshBasicNodeMaterial(), other = new T.Mesh(new T.PlaneGeometry(), material);
  other.frustumCulled = false; f.scene.add(other); make(other, 'second');
  t.after(() => { material.dispose(); other.geometry.dispose(); });
  const pipeline = f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync = async descriptor => {
    assert.notEqual(f.renderer._isPreCompiling, true, 'precompilation state does not escape across awaits');
    order.push('pipeline'); return pipeline(descriptor);
  };
  await f.renderer.init();
  f.renderer.info.calls = 91; f.renderer._nodes.nodeFrame.renderId = 7;
  const initialFrame = f.renderer._nodes.nodeFrame.frameId;
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(order.filter(value => value !== 'pipeline'),
    ['first:before', 'first:current', 'first:after', 'second:before', 'second:current', 'second:after']);
  assert.ok(order.indexOf('pipeline') > order.indexOf('first:current'));
  assert.ok(order.indexOf('pipeline') < order.indexOf('first:after'));
  assert.ok(contexts.every(([renderId, frameId]) => renderId === 7 && frameId > initialFrame));
  assert.equal(f.computes.length, 2); assert.equal(f.renderer._nodes.nodeFrame.renderId, 7);
  assert.notEqual(f.renderer._isPreCompiling, true); assert.equal(draws(f).length, 0);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.computes.length, 2, 'RENDER kernels retain the source restored render epoch across compiles');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
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
