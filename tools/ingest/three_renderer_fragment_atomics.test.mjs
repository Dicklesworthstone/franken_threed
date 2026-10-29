/** Source fragment atomics and shared GPU buffer lifetimes. The recording
 * device supplies explicit producer effects; it does not execute WGSL or prove
 * fragment coverage, atomic arithmetic, or browser pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t) {
  const device = geometryDevice(), captures = [], events = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000,
    maxComputeWorkgroupsPerDimension: 65535});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc});
  const buffer = device.createBuffer;
  device.createBuffer = desc => ({...buffer(desc), mapAsync: async () => {}});
  device.createComputePipeline = desc => ({...desc, getBindGroupLayout: index => ({index})});
  device.createComputePipelineAsync = async desc => device.createComputePipeline(desc);
  const encode = device.createCommandEncoder, submit = device.queue.submit;
  device.createCommandEncoder = desc => {
    const encoder = encode(desc), extra = [];
    return {...encoder,
      beginComputePass() {
        const pass = {compute: true, draws: [], groups: new Map()}; extra.push(pass);
        return {setPipeline(pipeline) { pass.pipeline = pipeline; },
          setBindGroup(index, group) { pass.groups.set(index, group); }, dispatchWorkgroups() {}, end() {}};
      },
      copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
        extra.push({draws: [], copy: {source, sourceOffset, target, targetOffset, size}});
      }, finish() { return [...encoder.finish(), ...extra]; }};
  };
  device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes) {
      if (pass.copy) {
        const {source, sourceOffset, target, targetOffset, size} = pass.copy;
        new Uint8Array(target.data, targetOffset, size).set(new Uint8Array(source.data, sourceOffset, size));
        events.push('readback');
      } else if (pass.compute) { events.push('compute'); device.onCompute?.(pass); }
      else for (const draw of pass.draws) if (draw.pipeline.vertex.entryPoint === 'main') {
        events.push('render'); device.onDraw?.(draw);
      }
    }
    submit(commands);
  };
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [32, 32], format: configured.format, usage: 16})})};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new T.MeshBasicNodeMaterial(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  mesh.frustumCulled = false;
  const scene = new T.Scene(); scene.add(mesh);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) { const frame = capture.call(this, request); if (frame) captures.push(frame); return frame; };
  t.after(() => { renderer.dispose(); material.dispose(); mesh.geometry.dispose(); });
  return {device, renderer, material, mesh, scene, camera, captures, events};
}
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const storageEntries = draw => entries(draw).filter(entry => entry.resource?.buffer?.usage & 128);
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const storageWrites = f => f.device.writes.filter(write => write.buffer.label === 'f3d-compute-buffer' && write.buffer.usage & 128);
const counter = (type = 'uint', values = [7, 11, 13, 17]) => N.storage(new T.StorageBufferAttribute(
  new (type === 'int' ? Int32Array : Uint32Array)(values), 1), type, values.length).toAtomic();
const graph = (storage, value = 1) => N.Fn(() => N.vec3(N.atomicAdd(storage.element(0),
  storage.nodeType === 'int' ? N.int(value) : N.uint(value)).toFloat().mul(.01)))();

for (const type of ['uint', 'int']) test(`${type} fragment atomic graphs retain source WGSL, writable layout and cache transitions`, async t => {
  const f = fixture(t), storage = counter(type); f.material.colorNode = graph(storage, type === 'int' ? -3 : 3);
  await f.renderer.compileAsync(f.scene, f.camera); assert.equal(storageWrites(f).length, 0);
  await f.renderer.renderAsync(f.scene, f.camera);
  const descriptor = f.captures.find(frame => frame.draws.length).draws[0].binding.nodeProgram;
  const source = new T.WGSLNodeBuilder(f.mesh, f.renderer); source.scene = f.scene; source.camera = f.camera; source.build();
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
  assert.match(descriptor.fragment, new RegExp(`array< atomic<${type === 'int' ? 'i32' : 'u32'}>`));
  assert.match(descriptor.fragment, /var<storage, read_write>/); assert.match(descriptor.fragment, /atomicAdd\(/);
  assert.doesNotMatch(descriptor.vertex, /atomic</);
  const bindings = descriptor.groups.flatMap(group => group.entries).filter(entry => entry.buffer?.type === 'storage');
  assert.equal(bindings.length, 1); assert.equal(bindings[0].visibility, 2); assert.equal(bindings[0].buffer.minBindingSize, 16);
  assert.equal(storageEntries(draws(f)[0]).length, 1);
  storage.setAtomic(false); f.material.colorNode = storage.element(0).toFloat();
  await f.renderer.renderAsync(f.scene, f.camera);
  const next = f.captures.at(-1).draws[0].binding.nodeProgram;
  assert.notEqual(next, descriptor); assert.match(next.fragment, /var<storage, read>/); assert.doesNotMatch(next.fragment, /atomic</);
  assert.equal(storageWrites(f).length, 1, 'changing shader access does not overwrite unchanged GPU storage');
});

test('fragment producers, compute and readback retain FIFO GPU contents without CPU restoration', async t => {
  const f = fixture(t), storage = counter('int', [-7]), observed = [], buffers = [];
  f.material.colorNode = graph(storage);
  f.device.onDraw = draw => {
    const buffer = storageEntries(draw)[0].resource.buffer, words = new Int32Array(buffer.data);
    observed.push(words[0]); buffers.push(buffer); words[0] = -31;
  };
  f.device.onCompute = pass => {
    const buffer = [...pass.groups.values()].flatMap(group => group.entries).find(entry => entry.resource.buffer.usage & 128).resource.buffer;
    assert.equal(new Int32Array(buffer.data)[0], -31); buffers.push(buffer); new Int32Array(buffer.data)[0] = 91;
  };
  const compute = N.Fn(() => { N.atomicAdd(storage.element(0), N.int(3)); })().compute(1);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const first = f.renderer.getArrayBufferAsync(storage.value), computed = f.renderer.compute(compute);
  const second = f.renderer.getArrayBufferAsync(storage.value); f.renderer.render(f.scene, f.camera);
  const last = f.renderer.getArrayBufferAsync(storage.value); storage.value.array.fill(999);
  assert.deepEqual([...new Int32Array(await first)], [-31]); await computed;
  assert.deepEqual([...new Int32Array(await second)], [91]); assert.deepEqual([...new Int32Array(await last)], [-31]);
  await f.renderer.waitForGPU();
  assert.deepEqual(observed, [-7, 91]); assert.ok(buffers.every(buffer => buffer === buffers[0]));
  assert.deepEqual(f.events, ['render', 'readback', 'compute', 'readback', 'render', 'readback']);
  assert.equal(storageWrites(f).length, 1); assert.deepEqual([...storage.value.array], [999]);
});

test('requested CPU versions and partial uploads preserve fragment-produced untouched components', async t => {
  const f = fixture(t), storage = counter(), values = [];
  f.material.colorNode = graph(storage);
  f.device.onDraw = draw => {
    const words = new Uint32Array(storageEntries(draw)[0].resource.buffer.data); values.push([...words]); words[0] = 73;
  };
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  storage.value.array[1] = 41; storage.value.addUpdateRange(1, 1); storage.value.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); storage.value.array.fill(999);
  await f.renderer.waitForGPU();
  assert.deepEqual(values, [[7, 11, 13, 17], [73, 41, 13, 17]]);
  assert.equal(storageWrites(f).length, 2); assert.equal(storageWrites(f)[1].offset, 4);
  assert.equal(storageWrites(f)[1].input.byteLength, 4); assert.deepEqual(storage.value.updateRanges, []);
});

for (const conflict of ['read-only shader', 'generated vertex', 'duplicate writer'])
test(`fragment atomic ${conflict} aliases reject before GPU uploads`, async t => {
  const f = fixture(t), storage = counter(); f.material.colorNode = graph(storage);
  if (conflict === 'read-only shader') {
    const material = new T.MeshBasicNodeMaterial(); material.colorNode = N.storage(storage.value, 'uint', 4).element(0).toFloat();
    const second = new T.Mesh(f.mesh.geometry, material); second.frustumCulled = false; f.scene.add(second); t.after(() => material.dispose());
  } else if (conflict === 'generated vertex') f.material.positionNode = N.positionLocal.add(storage.toAttribute().toFloat().mul(.01));
  else {
    const alias = N.storage(storage.value, 'uint', 4).toAtomic();
    f.material.colorNode = N.Fn(() => N.vec3(N.atomicAdd(storage.element(0), N.uint(1)).add(N.atomicAdd(alias.element(0), N.uint(1))).toFloat()))();
  }
  await f.renderer.init();
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), error => /STORAGE|CONFLICT/.test(error.code));
  assert.equal(storageWrites(f).length, 0); assert.equal(draws(f).length, 0);
});

for (const invalid of ['float atomic', 'vertex atomic', 'stage limit']) test(`${invalid} refuses before native effects`, async t => {
  const f = fixture(t), storage = invalid === 'float atomic'
    ? N.storage(new T.StorageBufferAttribute(4, 1), 'float', 4).toAtomic() : counter();
  if (invalid === 'vertex atomic') {
    f.material.positionNode = N.Fn(() => N.positionLocal.add(N.atomicLoad(storage.element(0)).toFloat()))();
    f.material.colorNode = N.vec3(1);
  } else f.material.colorNode = graph(storage);
  if (invalid === 'stage limit') f.device.limits.maxStorageBuffersPerShaderStage = 0;
  await f.renderer.init();
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), error => /STORAGE|LIMIT/.test(error.code));
  assert.equal(storageWrites(f).length, 0); assert.equal(draws(f).length, 0);
});
