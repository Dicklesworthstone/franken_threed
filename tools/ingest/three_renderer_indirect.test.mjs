/** Native source indirect geometry and shared compute residency. Recording
 * commands and submitted bytes proves ordering, not shader execution/pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {createAnimationRenderBundleCache} from './animation_render_bundles.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const draws = device => device.snapshots.flat().filter(draw => draw.indirect);
const argumentsOf = draw => Array.from(new Uint32Array(draw.contents.get(draw.indirect.buffer).buffer,
  draw.indirect.offset, draw.indexed ? 5 : 4));

async function fixture(t, {indexed = true, plain = false, wireframe = false, bundle = false} = {}) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = () => {};
  device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createComputePipelineAsync = async descriptor => device.createComputePipeline(descriptor);
  const encoder = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const result = encoder(), compute = [];
    return {...result, beginComputePass() {
      const groups = [];
      return {setPipeline() {}, setBindGroup(index, group) { groups[index] = group; },
        dispatchWorkgroups() { compute.push(groups); }, end() {}};
    }, finish() { const commands = result.finish(); commands.compute = compute; return commands; }};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const command of commands) for (const groups of command.compute ?? []) device.produce?.(groups);
    submit(commands);
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  renderer.outputColorSpace = T.LinearSRGBColorSpace; await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100); camera.position.z = 5;
  const source = new T.PlaneGeometry(), geometry = indexed ? source : source.toNonIndexed();
  const material = new T.MeshBasicMaterial({wireframe}), mesh = new T.Mesh(geometry, material);
  mesh.frustumCulled = false;
  const words = indexed || wireframe ? 5 : 4;
  const data = new Uint32Array(words * 2);
  data.set(words === 5 ? [3, 1, 0, 0, 0, 3, 1, 3, 0, 0] : [3, 1, 0, 0, 3, 1, 3, 0]);
  const attribute = plain ? new T.BufferAttribute(data, 1) : new T.IndirectStorageBufferAttribute(data, 1);
  geometry.setIndirect(attribute, [0, words * 4]);
  const group = bundle ? new T.BundleGroup() : scene;
  if (bundle) scene.add(group);
  group.add(mesh);
  return {renderer, device, scene, camera, mesh, geometry, material, attribute, words, group};
}

for (const indexed of [false, true]) for (const plain of [false, true])
  test(`${indexed ? 'indexed' : 'nonindexed'} ${plain ? 'ordinary' : 'storage'} source indirect arguments drive native commands`, async t => {
    const f = await fixture(t, {indexed, plain});
    f.geometry.setDrawRange(1, 1);
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    const submitted = draws(f.device);
    assert.equal(submitted.length, 2); assert.ok(submitted.every(draw => draw.indexed === indexed));
    assert.deepEqual(submitted.map(draw => draw.indirect.offset), [0, f.words * 4]);
    assert.deepEqual(submitted.map(argumentsOf), [Array.from(f.attribute.array.slice(0, f.words)), Array.from(f.attribute.array.slice(f.words))]);
    assert.equal(submitted[0].indirect.buffer, submitted[1].indirect.buffer);
    assert.equal(f.material.colorNode, undefined, 'ordinary materials select native execution automatically');
    assert.equal(f.renderer.info.render.drawCalls, 1, 'source object statistics remain separate from physical calls');
    assert.equal(f.renderer.info.f3d.nativeDrawCalls, 2);
    assert.equal(f.renderer._nodeCaptureBytes, 0);
  });

test('queued indirect offsets and marked uploads keep their accepted histories', async t => {
  const f = await fixture(t), offsets = [0]; f.geometry.indirectOffset = offsets;
  f.renderer.render(f.scene, f.camera);
  f.attribute.array[5] = 6; f.attribute.addUpdateRange(5, 1); f.attribute.needsUpdate = true;
  f.material.color.setRGB(.5, .75, 1); // The pinned ordinary-material observer selects FULL refresh.
  offsets[0] = 20; f.renderer.render(f.scene, f.camera);
  offsets[0] = 4; f.attribute.array.fill(99); f.geometry.indirect = null;
  await f.renderer.waitForGPU();
  const submitted = draws(f.device);
  assert.deepEqual(submitted.map(draw => draw.indirect.offset), [0, 20]);
  assert.deepEqual(submitted.map(argumentsOf), [[3, 1, 0, 0, 0], [6, 1, 3, 0, 0]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('compute and indirect drawing use one buffer without refreshing GPU-produced words from the CPU', async t => {
  const f = await fixture(t); f.geometry.indirectOffset = 0;
  const data = N.storage(f.attribute, 'uint', f.attribute.count);
  const kernel = N.Fn(() => { data.element(0).assign(6); })().compute(1);
  let produced;
  f.device.produce = groups => {
    produced = groups.flatMap(group => group?.entries ?? []).find(entry => entry.resource.buffer.usage & 256).resource.buffer;
    new Uint32Array(produced.data)[0] = 6; // Explicit recording-device producer.
  };
  f.renderer.compute(kernel);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const first = draws(f.device)[0];
  assert.ok(produced); assert.equal(first.indirect.buffer, produced);
  assert.equal(argumentsOf(first)[0], 6); assert.equal(f.attribute.array[0], 3);
  const uploads = f.device.writes.filter(write => write.buffer === produced).length;
  f.attribute.array[0] = 99; // Unrequested CPU edit cannot replace GPU state.
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(argumentsOf(draws(f.device).at(-1))[0], 6);
  assert.equal(f.device.writes.filter(write => write.buffer === produced).length, uploads);
});

for (const indexed of [false, true])
test(`${indexed ? 'indexed' : 'nonindexed'} indirect storage feeds vertex and fragment shaders from the computed argument allocation`, async t => {
  const f = await fixture(t, {indexed}); f.geometry.indirectOffset = 0;
  const material = new T.MeshBasicNodeMaterial(), data = N.storage(f.attribute, 'uint', f.attribute.count);
  t.after(() => material.dispose()); f.mesh.material = material;
  material.positionNode = N.positionLocal.add(N.vec3(data.element(0).toFloat().mul(.01), 0, 0));
  material.colorNode = N.vec3(data.element(1).toFloat().mul(.1), .25, .5);
  const kernel = N.Fn(() => { data.element(0).assign(6); data.element(1).assign(2); })().compute(1);
  t.after(() => kernel.dispose());
  let produced;
  f.device.produce = groups => {
    produced = groups.flatMap(group => group?.entries ?? []).find(entry => entry.resource.buffer.usage & 256).resource.buffer;
    new Uint32Array(produced.data).set([6, 2]); // Explicit recording-device producer, not WGSL execution.
  };
  f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const first = draws(f.device)[0], source = new T.WGSLNodeBuilder(f.mesh, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.build();
  const canonical = code => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(first.pipeline.vertex.module.code), canonical(source.vertexShader));
  assert.equal(canonical(first.pipeline.fragment.module.code), canonical(source.fragmentShader));
  for (const shader of [source.vertexShader, source.fragmentShader]) assert.match(shader, /var<storage, read>/);
  f.attribute.array.fill(99); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  for (const draw of draws(f.device)) {
    assert.equal(draw.indirect.buffer, produced); assert.deepEqual(argumentsOf(draw).slice(0, 2), [6, 2]);
    const bound = [...draw.groups.values()].flatMap(({group}) => group.entries)
      .filter(entry => entry.resource.buffer?.usage & 128);
    assert.equal(bound.length, 2, 'each shader stage retains its source storage binding');
    assert.ok(bound.every(entry => entry.resource.buffer === produced));
  }
  assert.equal(f.device.writes.filter(write => write.buffer === produced).length, 1);
  assert.equal(f.device.buffers.filter(buffer => buffer.label === 'f3d-compute-buffer' && buffer.usage & 128).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('indirect arguments cannot alias writable fragment storage in the same native pass', async t => {
  const f = await fixture(t), data = N.storage(f.attribute, 'uint', f.attribute.count).toAtomic();
  const material = new T.MeshBasicNodeMaterial(); t.after(() => material.dispose()); f.mesh.material = material;
  material.colorNode = N.Fn(() => N.vec3(N.atomicAdd(data.element(0), N.uint(1)).toFloat()))();
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'THREE_COMPUTE_STORAGE_CONFLICT'});
  assert.equal(draws(f.device).length, 0);
  assert.equal(f.device.writes.filter(write => write.buffer.label === 'f3d-compute-buffer' && write.buffer.usage & 128).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('source zero draw range still issues indirect arguments, zero object instances and negative ranges do not', async t => {
  const f = await fixture(t); f.geometry.indirectOffset = 0;
  f.geometry.setDrawRange(0, 0); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(draws(f.device).length, 1);
  f.mesh.count = 0; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(draws(f.device).length, 1);
  f.mesh.count = 1; f.geometry.setDrawRange(100, 1);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(draws(f.device).length, 1);
  assert.equal(f.renderer.info.render.drawCalls, 1, 'negative and zero-instance ranges do not count as source draws');
});

test('indirect upload precedes ordinary node updates while offsets are selected afterwards', async t => {
  const f = await fixture(t); f.geometry.indirectOffset = 0;
  const material = new T.MeshBasicNodeMaterial();
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => {
    f.attribute.array[5] = 6; f.attribute.needsUpdate = true; f.geometry.indirectOffset = 20;
    return new T.Color(1, 1, 1);
  });
  f.mesh.material = material;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(argumentsOf(draws(f.device)[0]), [3, 1, 3, 0, 0], 'the callback changed the offset but not already uploaded bytes');
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(argumentsOf(draws(f.device).at(-1)), [6, 1, 3, 0, 0]);
});

test('ordinary indirect-only host versions retain the pinned source observer upload boundary', async t => {
  const f = await fixture(t); f.geometry.indirectOffset = 0;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  f.attribute.array[0] = 6; f.attribute.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(argumentsOf(draws(f.device).at(-1))[0], 3, 'indirect-only changes do not force ordinary material FULL refresh');
  f.material.color.setRGB(.5, .75, 1);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(argumentsOf(draws(f.device).at(-1))[0], 6);
});

test('precompilation uploads indirect geometry without drawing or exposing later unversioned CPU edits', async t => {
  const f = await fixture(t); f.geometry.indirectOffset = 0;
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(draws(f.device).length, 0);
  f.attribute.array[0] = 99;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(argumentsOf(draws(f.device)[0])[0], 3);
});

test('GPU indirect arguments determine instance counts for default unbounded InstancedBufferGeometry', async t => {
  const f = await fixture(t), geometry = new T.InstancedBufferGeometry();
  geometry.setAttribute('position', f.geometry.attributes.position); geometry.setIndex(f.geometry.index);
  geometry.setIndirect(f.attribute, 0); f.mesh.geometry = geometry;
  assert.equal(geometry.instanceCount, Infinity);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(draws(f.device).length, 1); assert.equal(argumentsOf(draws(f.device)[0])[1], 1);
});

test('wireframe indirect draws use the generated index and five-word argument records', async t => {
  const f = await fixture(t, {indexed: false, wireframe: true});
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = draws(f.device);
  assert.equal(submitted.length, 2); assert.ok(submitted.every(draw => draw.indexed));
  assert.equal(submitted[0].pipeline.primitive.topology, 'line-list');
  assert.deepEqual(submitted.map(draw => draw.indirect.offset), [0, 20]);
});

test('ordinary and storage-backed three-word indirect attributes keep their distinct source layouts', async t => {
  for (const plain of [false, true]) {
    const f = await fixture(t), data = new Uint32Array([3, 1, 0, 0, 0, 0]);
    const attribute = plain ? new T.BufferAttribute(data, 3) : new T.IndirectStorageBufferAttribute(data, 3);
    f.geometry.setIndirect(attribute, 0);
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    const submitted = draws(f.device)[0];
    assert.equal(submitted.indirect.buffer.size, plain ? 24 : 32);
    assert.equal(attribute.itemSize, plain ? 3 : 4);
    assert.deepEqual(argumentsOf(submitted), [3, 1, 0, 0, 0]);
  }
});

test('invalid indirect offsets fail before submission and release captured snapshots', async t => {
  for (const offset of [-4, 2, 24, Number.MAX_SAFE_INTEGER, NaN]) {
    const f = await fixture(t); f.geometry.indirectOffset = offset;
    assert.throws(() => f.renderer.render(f.scene, f.camera), /INDIRECT/);
    assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});

test('empty source offset arrays issue no calls while ordinary draws recover after indirect removal', async t => {
  const f = await fixture(t); f.geometry.indirectOffset = [];
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(draws(f.device).length, 0);
  assert.equal(f.renderer.info.render.drawCalls, 1); assert.equal(f.renderer.info.f3d.nativeDrawCalls, 0);
  f.geometry.indirect = null; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.ok(f.device.snapshots.at(-1).some(draw => !draw.indirect && draw.args[0] === 6));
});

test('source BundleGroup retains its indirect command offsets until invalidated', async t => {
  const f = await fixture(t, {bundle: true});
  f.geometry.indirectOffset = 0; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  f.geometry.indirectOffset = 20; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(draws(f.device).map(draw => draw.indirect.offset), [0, 0]);
  f.group.needsUpdate = true; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(draws(f.device).map(draw => draw.indirect.offset), [0, 0, 20]);
});

for (const bundle of [false, true]) test(`${bundle ? 'bundled' : 'ordinary SHARED'} indirect draws observe preceding compute uploads, without later queued versions`, async t => {
  const f = await fixture(t, {bundle}); f.geometry.indirectOffset = 0;
  const data = N.storage(f.attribute, 'uint', f.attribute.count);
  const kernel = N.Fn(() => { data.element(0).addAssign(0); })().compute(1);
  await f.renderer.compileComputeAsync(kernel);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  f.attribute.array[0] = 6; f.attribute.needsUpdate = true;
  f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  f.attribute.array[0] = 0; f.attribute.needsUpdate = true;
  f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  assert.deepEqual(draws(f.device).map(draw => argumentsOf(draw)[0]), [3, 6, 0]);
  assert.deepEqual(draws(f.device).map(draw => draw.indirect.offset), [0, 0, 0]);
});

for (const bundle of [false, true]) for (const indirect of [false, true])
test(`${bundle ? 'bundled' : 'ordinary SHARED'} ${indirect ? 'indirect' : 'ordinary'} fog storage aliases follow the captured indirect upload`, async t => {
  const f = await fixture(t, {bundle});
  const attribute = indirect ? f.attribute : new T.StorageBufferAttribute(f.attribute.array.slice(), 1);
  f.geometry.setIndirect(attribute, 0);
  const data = N.storage(attribute, 'uint', attribute.count);
  f.scene.fogNode = N.fog(N.vec3(data.element(0).toFloat().div(6)), N.float(.5));
  const kernel = N.Fn(() => { data.element(0).addAssign(0); })().compute(1);
  await f.renderer.compileComputeAsync(kernel);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  attribute.array[0] = 6; attribute.needsUpdate = true;
  f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  attribute.array[0] = 0; attribute.needsUpdate = true;
  f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  attribute.array[0] = 99; // A retained binding must not admit an unversioned CPU edit.
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = draws(f.device);
  assert.deepEqual(submitted.map(draw => argumentsOf(draw)[0]), [3, 6, 0, 0]);
  for (const draw of submitted) {
    const bound = [...draw.groups.values()].flatMap(({group}) => group.entries)
      .filter(entry => entry.resource.buffer?.usage & 128).map(entry => entry.resource.buffer);
    assert.ok(bound.includes(draw.indirect.buffer), 'fog and indirect execution bind the same GPU allocation');
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const bundle of [false, true]) for (const indirect of [false, true])
test(`${bundle ? 'bundled' : 'ordinary SHARED'} ${indirect ? 'indirect' : 'ordinary'} generated vertex aliases follow captured indirect uploads`, async t => {
  const f = await fixture(t, {bundle});
  const attribute = indirect ? f.attribute : new T.StorageBufferAttribute(f.attribute.array.slice(), 1);
  // F3D's combined buffer usages admit this alias. The pinned source allocates
  // mutually exclusive INDIRECT/VERTEX usages, so this is no parity claim.
  f.geometry.setIndirect(attribute, 0);
  const data = N.storage(attribute, 'uint', attribute.count);
  f.scene.fogNode = N.fog(N.vec3(data.toAttribute().toFloat().div(6)), N.float(.5));
  const kernel = N.Fn(() => { data.element(0).addAssign(0); })().compute(1);
  await f.renderer.compileComputeAsync(kernel);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const encoded = f.device.bundleEncoders.length;
  attribute.array[0] = 6; attribute.needsUpdate = true;
  f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  attribute.array[0] = 0; attribute.needsUpdate = true;
  f.renderer.compute(kernel); f.renderer.render(f.scene, f.camera);
  attribute.array.fill(99);
  await f.renderer.waitForGPU();
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = draws(f.device);
  assert.deepEqual(submitted.map(draw => argumentsOf(draw)[0]), [3, 6, 0, 0]);
  for (const draw of submitted)
    assert.ok([...draw.streams.values()].includes(draw.indirect.buffer), 'vertex fetch and indirect arguments share their captured allocation');
  if (bundle) assert.equal(f.device.bundleEncoders.length, encoded + 2, 'unchanged captured versions reuse the refreshed geometry');
  assert.equal(f.device.writes.filter(write => write.buffer === submitted[0].indirect.buffer).length, 3);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native command bundle keys include indirect buffer identity and captured offsets', () => {
  const device = geometryDevice(), buffer = device.createBuffer({size: 40, usage: 256});
  const cache = createAnimationRenderBundleCache(device, {format: 'rgba8unorm', depthFormat: null,
    sampleCount: 1, bindGroup: null, stride: 256});
  const command = {record: {alphaMode: 'OPAQUE'}, pipeline: {vertex: {buffers: []}}, first: 0, count: 3,
    instanceCount: 1, nodeProgram: {groups: [{group: {}, offsets: []}]}, vertexBuffers: [],
    indexBuffer: null, nodeIndirect: {buffer, offsets: [0], skip: false}};
  const encoder = device.createCommandEncoder(), pass = encoder.beginRenderPass({});
  cache.execute(pass, [command], 1, {lightGroup: null});
  cache.execute(pass, [command], 1, {lightGroup: null});
  command.nodeIndirect = {buffer, offsets: [16], skip: false};
  cache.execute(pass, [command], 1, {lightGroup: null});
  assert.equal(device.bundleEncoders.length, 2);
  assert.deepEqual(device.bundleEncoders.map(bundle => bundle.draws[0].indirect.offset), [0, 16]);
  cache.dispose();
});
