/** Source TSL storage attributes through the native renderer and shared compute
 * allocation. The recording device checks queue-visible bytes and ownership;
 * it deliberately does not evaluate WGSL or claim image parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceAttributes from '../../upstream/three.js/src/renderers/common/Attributes.js';
import SourceGeometries from '../../upstream/three.js/src/renderers/common/Geometries.js';
import {AttributeType} from '../../upstream/three.js/src/renderers/common/Constants.js';
import SourceWebGPUAttributeUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUAttributeUtils.js';
import SourceWebGPUBackend from '../../upstream/three.js/src/renderers/webgpu/WebGPUBackend.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {captureComputeStorage, captureComputeAttributeView, computeAttributeSnapshot} from './gpu_compute_attributes.mjs';
const Renderer = createWebGPURendererClass(T), N = T.TSL;
function fixture(t) {
  const device = geometryDevice();
  device.limits.maxTextureDimension2D = 8192; device.limits.maxBindingsPerBindGroup = 1000;
  device.createTexture = desc => ({dimension:'2d', depthOrArrayLayers:1, sampleCount:1, ...desc,
    width:desc.size.width ?? desc.size[0], height:desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture:this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc});
  let configured;
  const canvas = {width:64, height:64, style:{}, getContext:() => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture:() => device.createTexture({size:[64,64], format:configured.format, usage:16})})};
  device.computePasses = [];
  device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout:index => ({index})});
  device.createComputePipelineAsync = descriptor => Promise.resolve(device.createComputePipeline(descriptor));
  const create = device.createCommandEncoder.bind(device), submit = device.queue.submit.bind(device.queue);
  device.createCommandEncoder = descriptor => {
    const encoder = create(descriptor), passes = [];
    return {...encoder, beginComputePass(desc) {
      const pass = {desc, draws:[], groups:new Map(), dispatches:[]}; passes.push(pass); device.computePasses.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.groups.set(index, group); },
        dispatchWorkgroups(...size) { pass.dispatches.push(size); }, end() {}};
    }, finish() { return [...encoder.finish(), ...passes]; }};
  };
  device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes)
      if (pass.dispatches?.length) device.onCompute?.(pass);
    submit(commands);
  };
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50,1,.1,30); camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  mesh.frustumCulled = false; scene.add(mesh);
  return {renderer, device, scene, camera, material, mesh};
}
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const draws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const shaderStorage = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource?.buffer).filter(buffer => buffer?.usage & 128);
const computeStorage = pass => [...pass.groups.values()].flatMap(group => group.entries)
  .map(entry => entry.resource?.buffer).filter(buffer => buffer?.usage & 128);
const words = (draw, buffer, Type = Float32Array) => [...new Type(draw.contents.get(buffer).buffer)];
const kernel = storage => N.Fn(() => { storage.element(N.instanceIndex).addAssign(N.vec3(0)); })().compute(storage.value.count);

// Run the pinned upload implementation and its actual finishRender submission
// boundary. The recording device establishes queue-visible buffer history;
// it does not execute the reference shader or establish real-GPU pixel parity.
function sourceStorage(array, {vertex = false, shader = true, itemSize = 1, instanced = true} = {}) {
  const device = geometryDevice(), records = new WeakMap();
  const backend = {device, _isRenderCameraDepthArray: () => false, get(key) {
    let record = records.get(key);
    if (!record) { record = {}; records.set(key, record); }
    return record;
  }};
  const utils = new SourceWebGPUAttributeUtils(backend);
  backend.createStorageAttribute = attribute => utils.createAttribute(attribute, 128 | 32 | 8 | 4);
  backend.updateAttribute = attribute => utils.updateAttribute(attribute);
  const owner = new SourceAttributes(backend, {createStorageAttribute() {}});
  const attribute = new (instanced ? T.StorageInstancedBufferAttribute : T.StorageBufferAttribute)(array.slice(), itemSize);
  const info = {render: {calls: 0}}, geometries = new SourceGeometries(owner, info);
  const update = () => owner.update(attribute, AttributeType.STORAGE);
  return {device, attribute, update, get buffer() { return backend.get(attribute).buffer; }, render(updates) {
    info.render.calls++;
    const encoder = device.createCommandEncoder(), pass = encoder.beginRenderPass({});
    pass.setPipeline({vertex: {entryPoint: 'main', buffers: vertex ? [{}] : []}});
    for (const change of updates) {
      change(attribute);
      if (vertex) geometries.updateAttribute(attribute, AttributeType.STORAGE);
      if (shader) update();
      if (vertex) pass.setVertexBuffer(0, backend.get(attribute).buffer);
      pass.setBindGroup(0, {entries: shader ? [{binding: 0, resource: {buffer: backend.get(attribute).buffer}}] : []});
      pass.draw(3);
    }
    const context = {occlusionQueryCount: 0, textures: null};
    Object.assign(backend.get(context), {renderBundles: [], currentPass: pass, encoder});
    SourceWebGPUBackend.prototype.finishRender.call(backend, context);
  }};
}

test('one render accumulates shared shader-storage versions at the pinned submission boundary', async t => {
  const f = fixture(t), storage = N.instancedArray(new Float32Array(1), 'float');
  f.material.colorNode = N.vec3(storage.element(0));
  const other = new T.Mesh(f.mesh.geometry, f.material); other.frustumCulled = false; other.renderOrder = 1;
  f.scene.add(other);
  const callbacks = []; let frame = 0;
  const change = (attribute, index) => {
    attribute.array[0] = frame + (index ? .75 : .25); attribute.needsUpdate = true;
  };
  for (const [index, mesh] of [f.mesh, other].entries()) {
    mesh.onBeforeRender = () => { callbacks.push(`before:${frame}:${index}`); change(storage.value, index); };
    mesh.onAfterRender = () => { callbacks.push(`after:${frame}:${index}`); };
  }
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera); callbacks.length = 0;
  const source = sourceStorage(new Float32Array(1));
  for (frame = 0; frame < 2; frame++) {
    source.render([attribute => change(attribute, 0), attribute => change(attribute, 1)]);
    f.renderer.render(f.scene, f.camera);
  }
  storage.value.array[0] = 99; storage.value.needsUpdate = true;
  await drain(f.renderer);
  const selected = draws(f.device), buffer = shaderStorage(selected[0])[0];
  assert.deepEqual(selected.map(draw => words(draw, buffer)), draws(source.device).map(draw => words(draw, source.buffer)));
  assert.deepEqual(selected.map(draw => words(draw, buffer)[0]), [.75, .75, 1.75, 1.75]);
  assert.ok(selected.every(draw => shaderStorage(draw).every(value => value === buffer)));
  assert.equal(f.device.buffers.filter(value => value.label === 'f3d-compute-buffer' && value.usage & 128).length, 1);
  assert.deepEqual(callbacks, ['before:0:0', 'after:0:0', 'before:0:1', 'after:0:1',
    'before:1:0', 'after:1:0', 'before:1:1', 'after:1:1']);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const shader of [false, true])
test(`generated vertex ${shader ? 'and shader aliases observe final' : 'uploads retain first'} source versions within each queued render`, async t => {
  const f = fixture(t), storage = N.instancedArray(new Float32Array(3), 'float'), attribute = storage.value;
  f.material.positionNode = N.positionLocal.add(N.vec3(storage.toAttribute(), 0, 0));
  if (shader) f.material.colorNode = N.vec3(storage.element(0));
  const other = new T.Mesh(f.mesh.geometry, f.material); other.frustumCulled = false; other.renderOrder = 1;
  f.scene.add(other);
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  const source = sourceStorage(new Float32Array(3), {vertex: true, shader});
  let frame = 0;
  const changes = [0, 1].map(index => current => {
    current.array.fill(frame + (index ? .75 : .25)); current.needsUpdate = true;
  });
  f.mesh.onBeforeRender = () => changes[0](attribute); other.onBeforeRender = () => changes[1](attribute);
  for (frame = 0; frame < 2; frame++) {
    source.render(changes); f.renderer.render(f.scene, f.camera);
  }
  attribute.array.fill(99); attribute.needsUpdate = true;
  await drain(f.renderer);
  const selected = draws(f.device), buffer = [...selected[0].streams.values()].find(value => value.usage & 128);
  assert.deepEqual(selected.map(draw => words(draw, buffer)),
    draws(source.device).map(draw => words(draw, source.buffer)));
  assert.deepEqual(selected.map(draw => words(draw, buffer)[0]), shader ? [.75, .75, 1.75, 1.75] : [.25, .25, 1.25, 1.25]);
  assert.ok(selected.every(draw => [...draw.streams.values()].includes(buffer) && (!shader || shaderStorage(draw).includes(buffer))));
  assert.equal(f.device.buffers.filter(value => value.label === 'f3d-compute-buffer' && value.usage & 128).length, 1);
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, shader ? 5 : 3,
    'compile upload plus source-admitted vertex/shader versions, without future CPU data');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('signed per-vertex storage and shader reads share final uploads without changing integer vertex layout', async t => {
  const f = fixture(t), storage = N.attributeArray(4, 'ivec4'), attribute = storage.value;
  f.material.positionNode = N.positionLocal.add(storage.toAttribute().xyz.toFloat().mul(.01));
  f.material.colorNode = storage.element(0).xyz.toFloat().abs().mul(.01);
  const other = new T.Mesh(f.mesh.geometry, f.material); other.frustumCulled = false; other.renderOrder = 1;
  f.scene.add(other);
  const changes = [-3, -7].map(value => current => {
    for (let i = 0; i < 4; i++) current.array.set([value, 2, 3, 4], i * 4);
    current.needsUpdate = true;
  });
  f.mesh.onBeforeRender = () => changes[0](attribute); other.onBeforeRender = () => changes[1](attribute);
  const source = sourceStorage(new Int32Array(16), {vertex: true, itemSize: 4, instanced: false});
  await f.renderer.init(); source.render(changes); f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99);
  await drain(f.renderer);
  const selected = draws(f.device), buffer = shaderStorage(selected[0])[0];
  assert.deepEqual(selected.map(draw => words(draw, buffer, Int32Array)),
    draws(source.device).map(draw => words(draw, source.buffer, Int32Array)));
  for (const draw of selected) {
    const slot = [...draw.streams].find(([, value]) => value === buffer)[0];
    assert.equal(draw.pipeline.vertex.buffers[slot].stepMode, 'vertex');
    assert.equal(draw.pipeline.vertex.buffers[slot].arrayStride, 16);
    assert.equal(draw.pipeline.vertex.buffers[slot].attributes[0].format, 'sint32x4');
    assert.deepEqual(words(draw, buffer, Int32Array).slice(0, 4), [-7, 2, 3, 4]);
  }
  assert.equal(f.device.buffers.filter(value => value.label === 'f3d-compute-buffer' && value.usage & 128).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('retained bundles borrow later ordinary shader uploads through immutable generated vertex proofs', async t => {
  const f = fixture(t), storage = N.instancedArray(new Float32Array(3), 'float'), attribute = storage.value;
  f.material.positionNode = N.positionLocal.add(N.vec3(storage.toAttribute(), 0, 0));
  f.material.colorNode = N.vec3(storage.element(0));
  const group = new T.BundleGroup(); group.add(f.mesh); f.scene.add(group);
  const other = new T.Mesh(f.mesh.geometry, f.material); other.frustumCulled = false;
  f.scene.add(other);
  let frame = 0;
  f.mesh.onBeforeRender = () => { attribute.array.fill(.25); attribute.needsUpdate = true; };
  other.onBeforeRender = () => { attribute.array.fill(frame + .75); attribute.needsUpdate = true; };
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  for (frame = 0; frame < 2; frame++) f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99);
  await drain(f.renderer);
  other.onBeforeRender = () => {};
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  // Authored material nodes require FULL source refresh even in a static
  // bundle. Stabilize its captured vertex version after the final host write.
  const encoded = f.device.bundleEncoders.length;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const selected = draws(f.device), buffer = shaderStorage(selected[0])[0];
  assert.deepEqual(selected.map(draw => words(draw, buffer)[0]), [.75, .75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75]);
  assert.ok(selected.every(draw => [...draw.streams.values()].includes(buffer) && shaderStorage(draw).includes(buffer)));
  assert.equal(f.device.bundleEncoders.length, encoded, 'the unchanged final version reuses recorded GPU commands');
  assert.equal(f.device.buffers.filter(value => value.label === 'f3d-compute-buffer' && value.usage & 128).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const vertex of [false, true])
test(`shared ${vertex ? 'generated-vertex and ' : ''}shader-storage partial versions preserve previous uploads and untouched GPU results`, async t => {
  const f = fixture(t), storage = N.instancedArray(new Uint32Array(4), 'uint'), attribute = storage.value;
  f.material.colorNode = N.vec3(N.float(storage.element(0)).mul(.01));
  if (vertex) f.material.positionNode = N.positionLocal.add(N.vec3(N.float(storage.toAttribute()).mul(.01), 0, 0));
  const other = new T.Mesh(f.mesh.geometry, f.material); other.frustumCulled = false; other.renderOrder = 1;
  f.scene.add(other);
  const compute = N.Fn(() => { storage.element(N.instanceIndex).assign(7); })().compute(4);
  t.after(() => compute.dispose());
  f.device.onCompute = pass => new Uint32Array(computeStorage(pass)[0].data).fill(7);
  await f.renderer.init(); await f.renderer.computeAsync(compute);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  const source = sourceStorage(new Uint32Array(4), {vertex}); source.update(); new Uint32Array(source.buffer.data).fill(7);
  const changes = [[0, 11], [3, 22]].map(([index, value]) => current => {
    current.array.fill(99); current.array[index] = value;
    current.clearUpdateRanges(); current.addUpdateRange(index, 1); current.needsUpdate = true;
  });
  f.mesh.onBeforeRender = () => changes[0](attribute); other.onBeforeRender = () => changes[1](attribute);
  source.render(changes); f.renderer.render(f.scene, f.camera);
  attribute.array.fill(123);
  await drain(f.renderer);
  for (const mesh of [f.mesh, other]) mesh.onBeforeRender = () => {};
  source.render([() => {}, () => {}]); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const selected = draws(f.device);
  assert.deepEqual(selected.map(draw => words(draw, buffer, Uint32Array)),
    draws(source.device).map(draw => words(draw, source.buffer, Uint32Array)));
  assert.ok(selected.every(draw => words(draw, buffer, Uint32Array).join() === '11,7,7,22'));
  if (vertex) assert.ok(selected.every(draw => [...draw.streams.values()].includes(buffer)));
  assert.deepEqual(f.device.writes.filter(write => write.buffer === buffer).map(write => [write.offset, write.input.byteLength]),
    [[0, 16], [0, 4], [12, 4]]);
  assert.equal(f.device.buffers.filter(value => value.label === 'f3d-compute-buffer' && value.usage & 128).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('multiple storage versions with indirect aliases reject before storage writes', async t => {
  const f = fixture(t), storage = N.instancedArray(new Uint32Array([6, 1, 0, 0, 0]), 'uint');
  f.material.colorNode = N.vec3(N.float(storage.element(0)).mul(.01));
  f.mesh.geometry.indirect = storage.value;
  const other = new T.Mesh(f.mesh.geometry, f.material); other.frustumCulled = false; other.renderOrder = 1;
  f.scene.add(other);
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  for (const mesh of [f.mesh, other]) mesh.onBeforeRender = () => { storage.value.needsUpdate = true; };
  const writes = f.device.writes.filter(write => write.buffer.label === 'f3d-compute-buffer' && write.buffer.usage & 128).length;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_COMPUTE_READ_CONFLICT'});
  assert.equal(f.device.writes.filter(write => write.buffer.label === 'f3d-compute-buffer' && write.buffer.usage & 128).length, writes);
  assert.equal(draws(f.device).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('multiple storage versions cannot change allocation layout before submission', async t => {
  const f = fixture(t); await f.renderer.init();
  const owner = f.renderer._getStorageInputs(), attribute = new T.StorageBufferAttribute(new Float32Array(4), 1);
  const first = owner.capture(attribute);
  attribute.array = new Float32Array(8); attribute.count = 8; attribute.needsUpdate = true;
  const second = owner.capture(attribute);
  try {
    assert.throws(() => owner.withReads([first, second], () => assert.fail('invalid history must not issue')),
      {code: 'THREE_COMPUTE_READ_CONFLICT'});
    assert.equal(f.device.writes.length, 0);
  } finally { owner.release(first); owner.release(second); }
});

test('submission vertex proofs remain version-strict and never replay an already published storage history', async t => {
  const f = fixture(t); await f.renderer.init();
  const owner = f.renderer._getStorageInputs(), attribute = new T.StorageBufferAttribute(new Float32Array([.25]), 1);
  const first = owner.capture(attribute), view = Object.create(attribute);
  owner.vertexAttribute(first, view).install();
  const original = captureComputeStorage(f.device, attribute);
  attribute.array[0] = .75; attribute.needsUpdate = true;
  const second = owner.capture(attribute), history = owner.validate([first, second]);
  const submitted = history.refreshVertexAttribute(first, view);
  attribute.array[0] = 99; attribute.needsUpdate = true;
  let buffer;
  try {
    owner.withReads([first, second], () => {
      buffer = computeAttributeSnapshot(f.device, submitted).buffer;
      assert.equal(owner.resolve(first).buffer, buffer); assert.equal(owner.resolve(second).buffer, buffer);
      assert.deepEqual([...new Float32Array(buffer.data)], [.75]);
      assert.throws(() => original.resolve(), {code: 'COMPUTE_ATTRIBUTE_UNPREPARED'});
      assert.throws(() => computeAttributeSnapshot(f.device, view), {code: 'COMPUTE_ATTRIBUTE_UNPREPARED'});
      assert.throws(() => computeAttributeSnapshot(f.device, submitted, {index: true}), {code: 'COMPUTE_ATTRIBUTE_FORMAT'});
      assert.throws(() => computeAttributeSnapshot(geometryDevice(), submitted), {code: 'COMPUTE_ATTRIBUTE_OWNER'});
    });
    // A later GPU producer can alter contents without changing the host upload
    // version. Reentering the accepted read interval must retain that result.
    new Float32Array(buffer.data)[0] = .875;
    const writes = f.device.writes.length;
    owner.withReads([first, second], () => assert.deepEqual([...new Float32Array(buffer.data)], [.875]));
    assert.equal(f.device.writes.length, writes);
  } finally { owner.release(first); owner.release(second); }
});

for (const [type, Type, stride] of [['float', Float32Array, 1], ['vec3', Float32Array, 3], ['uvec2', Uint32Array, 2]])
test(`PBO-marked ${type} storage keeps the exact source WebGPU graph and ignores fallback-only toggles`, async t => {
  const f = fixture(t), storage = N.instancedArray(new Type(2 * stride).fill(1), type).setPBO(true);
  const element = storage.element(N.instanceIndex), value = N.float(stride === 1 ? element : element.x);
  f.material.positionNode = N.positionLocal.add(N.vec3(value.mul(.01), 0, 0));
  f.material.colorNode = N.vec3(value.mul(.01), .25, .5);
  await f.renderer.init();
  const context = {scene: f.scene, camera: f.camera, lights: []};
  const source = () => {
    const builder = new T.WGSLNodeBuilder(f.mesh, f.renderer);
    builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = N.lights([]);
    builder.build(); return builder;
  };
  const programs = createThreeNodePrograms({three: T, renderer: f.renderer}); t.after(() => programs.dispose());
  const expected = source(), descriptor = programs.compile(f.material, f.mesh, context);
  assert.equal(descriptor.vertex, expected.vertexShader); assert.equal(descriptor.fragment, expected.fragmentShader);
  assert.match(descriptor.vertex, /var<storage, read>/); assert.match(descriptor.fragment, /var<storage, read>/);
  storage.setPBO(false);
  const ordinary = source();
  assert.equal(ordinary.vertexShader, expected.vertexShader); assert.equal(ordinary.fragmentShader, expected.fragmentShader);
  assert.equal(programs.compile(f.material, f.mesh, context), descriptor);
  storage.setPBO(true);
  assert.equal(programs.compile(f.material, f.mesh, context), descriptor);
});

test('PBO compute, generated vertex inputs and fragment storage share one allocation across queued flag changes', async t => {
  const f = fixture(t), positions = N.instancedArray(3, 'vec3').setPBO(true), attribute = positions.value;
  Object.defineProperty(attribute, 'onUploadCallback', {get() { assert.fail('native compute and graphics leave WebGL upload callbacks inert'); }});
  f.mesh.count = 3; f.material.positionNode = N.positionLocal.add(positions.toAttribute());
  f.material.colorNode = positions.element(0).abs();
  const compute = kernel(positions); t.after(() => compute.dispose());
  f.device.onCompute = pass => new Float32Array(computeStorage(pass)[0].data).fill(.5);
  await f.renderer.init(); await f.renderer.computeAsync(compute);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  f.renderer.render(f.scene, f.camera);
  positions.setPBO(false); attribute.array.fill(99); // No upload request: keep the GPU-produced bytes.
  f.renderer.render(f.scene, f.camera);
  positions.setPBO(true); f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  const selected = draws(f.device); assert.equal(selected.length, 3);
  for (const draw of selected) {
    assert.ok([...draw.streams.values()].includes(buffer)); assert.ok(shaderStorage(draw).includes(buffer));
    assert.ok(words(draw, buffer).every(value => value === .5));
  }
  assert.equal(computeStorage(f.device.computePasses[0])[0], buffer);
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1);
  assert.equal(f.device.buffers.filter(value => value.label === 'f3d-compute-buffer' && value.usage & 128).length, 1);
  assert.equal(attribute.itemSize, 4, 'retain the pinned WebGPU vec3 storage padding');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  f.renderer.dispose(); assert.equal(buffer.destroyed, true);
});

test('PBO storage queued partial uploads preserve the GPU history outside each requested range', async t => {
  const f = fixture(t), storage = N.instancedArray(new Uint32Array(4), 'uint').setPBO(true), attribute = storage.value;
  f.material.colorNode = N.vec3(N.float(storage.element(0)).mul(.01), .25, .5);
  const compute = N.Fn(() => { storage.element(N.instanceIndex).assign(7); })().compute(4);
  t.after(() => compute.dispose());
  f.device.onCompute = pass => new Uint32Array(computeStorage(pass)[0].data).fill(7);
  await f.renderer.init(); await f.renderer.computeAsync(compute);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99); attribute.array[1] = 11; attribute.addUpdateRange(1, 1); attribute.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); attribute.clearUpdateRanges();
  storage.setPBO(false); attribute.array.fill(99); attribute.array[2] = 13;
  attribute.addUpdateRange(2, 1); attribute.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); attribute.array.fill(99);
  await drain(f.renderer);
  assert.deepEqual(draws(f.device).map(draw => words(draw, buffer, Uint32Array)),
    [[7, 7, 7, 7], [7, 11, 7, 7], [7, 11, 13, 7]]);
  assert.deepEqual(f.device.writes.filter(write => write.buffer === buffer).map(write => [write.offset, write.input.byteLength]),
    [[0, 16], [4, 4], [8, 4]]);
});

const matrixGraph = (storage, dimension) => {
  const matrix = storage.element(N.instanceIndex);
  if (dimension === 2) return N.vec3(matrix.mul(N.positionLocal.xy), N.positionLocal.z);
  if (dimension === 3) return matrix.mul(N.positionLocal);
  return matrix.mul(N.vec4(N.positionLocal, 1)).xyz;
};
const matrixColor = (storage, dimension) => {
  const matrix = storage.element(0);
  if (dimension === 2) return N.vec3(matrix.mul(N.vec2(.5, .25)), .5);
  return matrix.mul(dimension === 3 ? N.vec3(.5, .25, .5) : N.vec4(.5, .25, .5, 1)).xyz;
};

for (const dimension of [2, 3, 4]) test(`standalone mat${dimension} storage shares source-aligned compute and graphics bytes`, async t => {
  const f = fixture(t), stride = dimension === 3 ? 12 : dimension ** 2;
  const attribute = new T.StorageBufferAttribute(new Float32Array(2 * stride).fill(.125), stride);
  const storage = N.storage(attribute, `mat${dimension}`, 2), original = attribute.array;
  // Padding is storage too: keep its original bit pattern through capture and
  // reuse instead of rebuilding matrices from individual JavaScript numbers.
  const raw = new Uint32Array(original.buffer); if (dimension === 3) raw[3] = 0x7fc01234;
  const initial = raw.slice(), generated = raw.slice();
  new Float32Array(generated.buffer)[0] = .75;
  f.mesh.count = 2; f.material.positionNode = matrixGraph(storage, dimension);
  f.material.colorNode = matrixColor(storage, dimension);
  const compute = N.Fn(() => { storage.element(N.instanceIndex).assign(N[`mat${dimension}`](1)); })().compute(2, [2]);
  t.after(() => compute.dispose());
  f.device.onCompute = pass => { new Uint32Array(computeStorage(pass)[0].data).set(generated); };
  await f.renderer.init(); await f.renderer.computeAsync(compute);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  await f.renderer.renderAsync(f.scene, f.camera);
  const selected = draws(f.device); assert.equal(selected.length, 1);
  assert.ok(shaderStorage(selected[0]).includes(buffer));
  assert.deepEqual(words(selected[0], buffer, Uint32Array), [...generated]);
  assert.deepEqual([...raw], [...initial]);
  assert.equal(attribute.array, original); assert.equal(attribute.itemSize, stride); assert.equal(attribute.count, 2);
  assert.equal(buffer.size, stride * 2 * 4);
  const descriptor = selected[0].pipeline;
  assert.match(descriptor.vertex.module.code, new RegExp(`array< mat${dimension}x${dimension}<f32>`));
  assert.match(descriptor.fragment.module.code, new RegExp(`array< mat${dimension}x${dimension}<f32>`));
  assert.match(descriptor.fragment.module.code, /var<storage, read>/);
  assert.match(f.device.computePasses[0].pipeline.compute.module.code, /var<storage, read_write>/);
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(words(draws(f.device)[1], buffer, Uint32Array), [...generated]);
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1);
  f.renderer.dispose(); assert.equal(buffer.destroyed, true);
});

test('queued padded mat3 draws preserve partial host updates and untouched GPU-produced columns', async t => {
  const f = fixture(t), attribute = new T.StorageBufferAttribute(new Float32Array(24).fill(.125), 12);
  const storage = N.storage(attribute, 'mat3', 2);
  f.material.positionNode = matrixGraph(storage, 3); f.material.colorNode = matrixColor(storage, 3);
  const compute = N.Fn(() => { storage.element(N.instanceIndex).assign(N.mat3(1)); })().compute(2, [2]);
  t.after(() => compute.dispose());
  f.device.onCompute = pass => { new Float32Array(computeStorage(pass)[0].data).fill(.5); };
  await f.renderer.init(); await f.renderer.computeAsync(compute);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99); attribute.array[4] = .75; attribute.addUpdateRange(4, 1); attribute.needsUpdate = true;
  f.renderer.render(f.scene, f.camera);
  attribute.clearUpdateRanges();
  attribute.array.fill(99); attribute.array[12] = .25; attribute.addUpdateRange(12, 1); attribute.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); attribute.array.fill(99);
  await drain(f.renderer);
  const observed = draws(f.device).map(draw => words(draw, buffer));
  assert.equal(observed.length, 3);
  assert.deepEqual(observed.map(values => [values[0], values[4], values[12], values[15]]),
    [[.5, .5, .5, .5], [.5, .75, .5, .5], [.5, .75, .25, .5]]);
  const writes = f.device.writes.filter(write => write.buffer === buffer);
  assert.deepEqual(writes.map(write => [write.offset, write.input.byteLength]), [[0, 96], [16, 4], [48, 4]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const profile of ['packed mat3', 'integer matrix', 'atomic matrix']) test(`${profile} refuses before native storage effects`, async t => {
  const f = fixture(t), stride = profile === 'packed mat3' ? 9 : 12;
  const array = profile === 'integer matrix' ? new Uint32Array(24) : new Float32Array(stride * 2);
  const storage = N.storage(new T.StorageBufferAttribute(array, stride), 'mat3', 2);
  if (profile === 'atomic matrix') storage.toAtomic();
  f.material.positionNode = matrixGraph(storage, 3); f.material.colorNode = matrixColor(storage, 3);
  await f.renderer.init();
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(f.device.writes.filter(write => write.buffer.label === 'f3d-compute-buffer' && write.buffer.usage & 128).length, 0);
  assert.equal(draws(f.device).length, 0);
});

test('compute-authored sprites draw their full source count from one shared instanced allocation', async t => {
  const f = fixture(t), positions = N.instancedArray(200000, 'vec3');
  const material = new T.SpriteNodeMaterial();
  material.positionNode = positions.toAttribute(); material.colorNode = positions.element(N.instanceIndex).abs();
  material.scaleNode = N.uniform(.125); material.rotationNode = N.uniform(.25); material.opacityNode = N.shapeCircle();
  material.alphaToCoverage = true;
  const sprite = new T.Sprite(material); sprite.count = 200000; sprite.frustumCulled = false;
  f.scene.remove(f.mesh); f.scene.add(sprite);
  await f.renderer.init();
  f.device.onCompute = pass => {
    const buffer = computeStorage(pass)[0];
    new Float32Array(buffer.data).set([.25,.5,.75,0]);
  };
  await f.renderer.compute(kernel(positions));
  const buffer = captureComputeStorage(f.device, positions.value).resolve().buffer;
  f.renderer.render(f.scene, f.camera);
  sprite.count = 3; f.renderer.render(f.scene, f.camera);
  sprite.count = 0; f.renderer.render(f.scene, f.camera);
  // Preparation may run after the caller edits the object again. Neither its
  // count nor CPU zeros may replace the accepted native instance/storage state.
  sprite.count = NaN;
  await drain(f.renderer);
  const selected = draws(f.device); assert.equal(selected.length, 2);
  assert.deepEqual(selected.map(draw => draw.args.slice(0, 2)), [[6, 200000], [6, 3]]);
  for (const draw of selected) {
    assert.ok([...draw.streams.values()].includes(buffer)); assert.ok(shaderStorage(draw).includes(buffer));
    assert.deepEqual(words(draw, buffer).slice(0, 4), [.25, .5, .75, 0]);
    assert.equal(draw.pipeline.primitive.topology, 'triangle-list');
    const slot = [...draw.streams].find(([, value]) => value === buffer)[0];
    assert.equal(draw.pipeline.vertex.buffers[slot].stepMode, 'instance');
  }
  assert.ok(positions.value.array.every(value => value === 0));
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1);
  assert.equal(f.device.buffers.filter(candidate => candidate.label === 'f3d-compute-buffer' && candidate.usage & 128).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer.info.f3d.presentedRenders, 3);
});

test('padded instanced .toAttribute shares the compute and shader storage allocation after compileAsync', async t => {
  const f = fixture(t), positions = N.instancedArray(3, 'vec3'), attribute = positions.value;
  f.mesh.count = 3;
  f.material.positionNode = N.positionLocal.add(positions.toAttribute());
  f.material.colorNode = positions.element(0).abs();
  await f.renderer.compileAsync(f.scene, f.camera);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  assert.equal(attribute.itemSize, 4); assert.equal(buffer.size, 48);
  assert.equal(buffer.usage & (128 | 32), 128 | 32);
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1);
  f.device.onCompute = pass => {
    assert.equal(computeStorage(pass)[0], buffer);
    // Simulate the queued device producer, leaving the source CPU array zero.
    new Float32Array(buffer.data).set([.25,.5,.75,0,1,2,3,0,4,5,6,0]);
  };
  await f.renderer.compute(kernel(positions));
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const selected = draws(f.device); assert.equal(selected.length, 2);
  for (const draw of selected) {
    assert.ok([...draw.streams.values()].includes(buffer));
    assert.ok(shaderStorage(draw).includes(buffer));
    assert.deepEqual(words(draw, buffer), [.25,.5,.75,0,1,2,3,0,4,5,6,0]);
    const slot = [...draw.streams].find(([, value]) => value === buffer)[0];
    assert.equal(draw.pipeline.vertex.buffers[slot].arrayStride, 16);
    assert.equal(draw.pipeline.vertex.buffers[slot].stepMode, 'instance');
    assert.equal(draw.pipeline.vertex.buffers[slot].attributes[0].format, 'float32x3');
    assert.equal(draw.args[1], 3);
  }
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1,
    'compile, compute, and render retain one host upload and never restore CPU zeros');
  assert.equal(f.device.buffers.filter(candidate => candidate.label === 'f3d-compute-buffer' && candidate.usage & 128).length, 1);
  assert.ok(attribute.array.every(value => value === 0)); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a queued .toAttribute draw before compute retains producer order without pinning buffer bytes', async t => {
  const f = fixture(t), positions = N.instancedArray(2, 'vec3');
  positions.value.array.fill(.125); f.mesh.count = 2;
  f.material.positionNode = N.positionLocal.add(positions.toAttribute());
  f.material.colorNode = positions.element(0);
  f.device.onCompute = pass => new Float32Array(computeStorage(pass)[0].data).fill(.875);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  const computation = f.renderer.compute(kernel(positions));
  f.renderer.render(f.scene, f.camera);
  positions.value.array.fill(99);
  await Promise.all([computation, drain(f.renderer)]);
  const selected = draws(f.device); assert.equal(selected.length, 2);
  const buffer = captureComputeStorage(f.device, positions.value).resolve().buffer;
  assert.deepEqual(selected.map(draw => words(draw, buffer)[0]), [.125, .875]);
  assert.ok(selected.every(draw => [...draw.streams.values()].includes(buffer) && shaderStorage(draw).includes(buffer)));
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1);
  assert.equal(f.device.computePasses.length, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('integer per-vertex .toAttribute retains queued host versions and shared storage identity', async t => {
  const f = fixture(t), values = N.attributeArray(4, 'ivec4'), attribute = values.value;
  for (let i = 0; i < 4; i++) attribute.array.set([-3,4,5,6], i * 4);
  f.material.positionNode = N.positionLocal.add(values.toAttribute().xyz.toFloat().mul(.01));
  f.material.colorNode = values.element(0).xyz.toFloat().abs().mul(.1);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  for (let i = 0; i < 4; i++) attribute.array.set([-7,8,9,10], i * 4);
  attribute.needsUpdate = true; f.renderer.render(f.scene, f.camera); attribute.array.fill(99);
  await drain(f.renderer);
  const selected = draws(f.device); assert.equal(selected.length, 2);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  assert.deepEqual(selected.map(draw => words(draw, buffer, Int32Array).slice(0,4)), [[-3,4,5,6],[-7,8,9,10]]);
  for (const draw of selected) {
    const slot = [...draw.streams].find(([, value]) => value === buffer)[0];
    assert.equal(draw.pipeline.vertex.buffers[slot].stepMode, 'vertex');
    assert.equal(draw.pipeline.vertex.buffers[slot].attributes[0].format, 'sint32x4');
    assert.ok(shaderStorage(draw).includes(buffer));
  }
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 2);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('accepted storage vertex views keep their shape when the source is resized before preparation', async t => {
  const f = fixture(t), positions = N.instancedArray(2, 'vec3'), attribute = positions.value;
  attribute.array.set([1,2,3,4,5,6]); f.mesh.count = 2;
  f.material.positionNode = N.positionLocal.add(positions.toAttribute());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  attribute.array = new Float32Array(40).fill(99); attribute.count = 10; attribute.itemSize = 4; attribute.needsUpdate = true;
  f.mesh.count = 10;
  await drain(f.renderer);
  const selected = draws(f.device); assert.equal(selected.length, 1);
  const buffer = [...selected[0].streams.values()].find(value => value.usage & 128);
  assert.equal(buffer.size, 32); assert.deepEqual(words(selected[0], buffer), [1,2,3,0,4,5,6,0]);
  assert.equal(selected[0].args[1], 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('accepted storage vertex reads survive source disposal and refuse a foreign device', async t => {
  const f = fixture(t), positions = N.instancedArray(1, 'vec3');
  f.material.positionNode = N.positionLocal.add(positions.toAttribute());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  f.mesh.geometry.dispose(); f.material.dispose();
  await drain(f.renderer);
  assert.equal(draws(f.device).length, 1);
  const buffer = captureComputeStorage(f.device, positions.value).resolve().buffer;
  assert.equal(buffer.destroyed, false);
  const view = Object.create(positions.value);
  captureComputeAttributeView(f.device, positions.value, view).install();
  assert.equal(computeAttributeSnapshot(f.device, view).buffer, buffer);
  assert.throws(() => computeAttributeSnapshot(geometryDevice(), view), {code:'COMPUTE_ATTRIBUTE_OWNER'});
  f.renderer.dispose(); assert.equal(buffer.destroyed, true);
  assert.throws(() => computeAttributeSnapshot(f.device, view), /DISPOSED/);
});

test('renderer disposal cancels accepted storage vertex draws and preparation failure releases captured versions', async t => {
  const f = fixture(t), positions = N.instancedArray(2, 'vec3');
  f.material.positionNode = N.positionLocal.add(positions.toAttribute());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); f.renderer.dispose();
  await drain(f.renderer);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(draws(f.device).length, 0);
  assert.ok(f.device.buffers.filter(buffer => buffer.label === 'f3d-compute-buffer').every(buffer => buffer.destroyed));
  const g = fixture(t), other = N.instancedArray(2, 'vec3');
  g.material.positionNode = N.positionLocal.add(other.toAttribute());
  await g.renderer.init();
  g.device.createRenderPipelineAsync = () => Promise.reject(new Error('storage vertex pipeline rejected'));
  g.renderer.render(g.scene, g.camera);
  await assert.rejects(drain(g.renderer), /storage vertex pipeline rejected/);
  assert.equal(g.renderer._nodeCaptureBytes, 0); assert.equal(g.device.submissions.length, 0);
});


test('warm bundles retain storage-only vertex leases and observe later GPU producers', async t => {
  const f = fixture(t), positions = N.instancedArray(2, 'vec3'), group = new T.BundleGroup();
  positions.value.array.fill(.125); f.mesh.count = 2;
  f.material.positionNode = N.positionLocal.add(positions.toAttribute());
  group.add(f.mesh); f.scene.add(group);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const buffer = captureComputeStorage(f.device, positions.value).resolve().buffer;
  const encoded = f.device.bundleEncoders.length;
  assert.ok(encoded > 0);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  f.device.onCompute = pass => {
    assert.equal(computeStorage(pass)[0], buffer);
    new Float32Array(buffer.data).fill(.875);
  };
  await f.renderer.compute(kernel(positions));
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const selected = draws(f.device); assert.equal(selected.length, 3);
  assert.deepEqual(selected.map(draw => words(draw, buffer)[0]), [.125, .125, .875]);
  assert.ok(selected.every(draw => [...draw.streams.values()].includes(buffer)));
  assert.equal(f.device.bundleEncoders.length, encoded, 'warm source bundles reuse their GPU vertex commands');
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1,
    'warm bundles never overwrite computed vertices with their source CPU array');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('ordinary SHARED fog retains storage-only vertex leases and later queued GPU producers', async t => {
  const f = fixture(t), colors = N.attributeArray(4, 'vec3');
  const captures = [], capture = f.renderer._captureNodeFrame.bind(f.renderer);
  f.renderer._captureNodeFrame = request => {
    const packet = capture(request); if (request.scene === f.scene) captures.push(packet); return packet;
  };
  colors.value.array.fill(.125);
  f.scene.fogNode = N.fog(colors.toAttribute(), N.float(.5));
  assert.equal(f.material.colorNode, null); assert.equal(f.material.positionNode, null);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const first = captures[0], lease = first.storageUses[0];
  assert.equal(first.storageUses.length, 1);
  assert.doesNotThrow(() => first.storageOwner.validate([lease]), 'the observer retains the allocation after the first frame releases it');
  const buffer = captureComputeStorage(f.device, colors.value).resolve().buffer;
  f.device.onCompute = pass => {
    assert.equal(computeStorage(pass)[0], buffer);
    new Float32Array(buffer.data).fill(.875);
  };
  await f.renderer.compute(kernel(colors));
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(captures[1].storageUses, [lease]);
  // A new authored companion needs an asynchronous pipeline, so the third
  // SHARED draw is still queued when its material retires the observer.
  const companion = new T.MeshBasicNodeMaterial({fog: false}); companion.colorNode = N.vec3(.2, .3, .4);
  const extra = new T.Mesh(f.mesh.geometry, companion); extra.frustumCulled = false; f.scene.add(extra);
  f.renderer.render(f.scene, f.camera);
  assert.ok(f.renderer._drain, 'new pipeline preparation keeps the accepted draw queued');
  assert.deepEqual(captures[2].storageUses, [lease]);
  f.material.dispose();
  assert.doesNotThrow(() => first.storageOwner.validate([lease]), 'a queued SHARED draw survives observer retirement');
  await drain(f.renderer);
  const selected = draws(f.device).filter(draw => draw.pipeline.vertex.buffers.some((layout, slot) =>
    layout && draw.streams.get(slot) === buffer));
  assert.equal(selected.length, 3);
  assert.deepEqual(selected.map(draw => words(draw, buffer)[0]), [.125, .875, .875]);
  assert.ok(selected.every(draw => [...draw.streams.values()].includes(buffer)));
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1,
    'SHARED fog reuses the accepted GPU allocation without restoring its CPU array');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.throws(() => first.storageOwner.validate([lease]), {code: 'THREE_COMPUTE_STALE'},
    'retired observer and completed frames release the allocation lease');
  f.renderer.dispose(); assert.equal(buffer.destroyed, true);
});

for (const bundled of [false, true]) for (const vertex of [false, true])
test(`${bundled ? 'bundled' : 'ordinary SHARED'} ${vertex ? 'generated' : 'shader-only'} storage retains each preceding compute upload`, async t => {
  const f = fixture(t), colors = N.attributeArray(4, 'vec3'), attribute = colors.value;
  attribute.array.fill(.125);
  f.scene.fogNode = N.fog(vertex ? colors.toAttribute().add(colors.element(0)) : colors.element(0), N.float(.5));
  if (bundled) {
    const group = new T.BundleGroup(); group.add(f.mesh); f.scene.add(group);
  }
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const buffer = captureComputeStorage(f.device, attribute).resolve().buffer;
  const encoded = f.device.bundleEncoders.length;
  attribute.array.fill(.25); attribute.needsUpdate = true;
  f.renderer.compute(kernel(colors)); f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99); attribute.array[0] = .75; attribute.addUpdateRange(0, 1); attribute.needsUpdate = true;
  f.renderer.compute(kernel(colors)); f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99);
  await drain(f.renderer);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const selected = draws(f.device);
  assert.equal(selected.length, 4);
  assert.deepEqual(selected.map(draw => words(draw, buffer)[0]), [.125, .25, .75, .75]);
  assert.deepEqual(selected.map(draw => words(draw, buffer)[1]), [.125, .25, .25, .25], 'partial uploads leave unrequested words GPU-stale');
  assert.ok(selected.every(draw => (!vertex || [...draw.streams.values()].includes(buffer)) && shaderStorage(draw).includes(buffer)));
  if (bundled) assert.equal(f.device.bundleEncoders.length, encoded + (vertex ? 2 : 0), 'unchanged captured versions reuse their native command bundle');
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 3,
    'retained bindings never restore earlier snapshots or later unrequested CPU writes');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('repeated retained vertex refreshes retire old geometry while unchanged versions stay warm', async t => {
  const f = fixture(t), colors = N.attributeArray(4, 'vec3'), group = new T.BundleGroup();
  f.scene.fogNode = N.fog(colors.toAttribute(), N.float(.5));
  group.add(f.mesh); f.scene.add(group);
  const update = kernel(colors);
  await f.renderer.init(); await f.renderer.compileComputeAsync(update);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const alive = () => f.device.buffers.filter(buffer => !buffer.destroyed).length;
  const liveBuffers = alive(), geometryBytes = f.renderer.info.memory.attributesSize;
  for (let version = 1; version <= 64; version++) {
    colors.value.array[0] = version / 64; colors.value.needsUpdate = true;
    f.renderer.compute(update); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
    assert.equal(alive(), liveBuffers, 'replaced metadata retires the old native geometry buffers');
    assert.equal(f.renderer.info.memory.attributesSize, geometryBytes);
  }
  const encoded = f.device.bundleEncoders.length;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.device.bundleEncoders.length, encoded);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  f.renderer.dispose(); assert.equal(alive(), 0);
});
