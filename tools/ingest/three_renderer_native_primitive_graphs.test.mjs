/** Source point/line graphs and native commands. The device records commands;
 * browser comparisons supply rasterization evidence separately. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {captureComputeStorage} from './gpu_compute_attributes.mjs';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
import {createThreeNodeProgramMeshes} from './three_node_program_gpu.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
const floats = packets => packets.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const native = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);

function fixture(t, parameters = {}) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 80, height: 60, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device, ...parameters}); renderer.setSize(80, 60, false);
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20); camera.position.z = 4;
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) { const value = capture.call(this, request); if (value) captures.push(value); return value; };
  const add = object => { object.frustumCulled = false; scene.add(object); t.after(() => { object.geometry.dispose(); object.material.dispose(); }); return object; };
  return {device, canvas, renderer, scene, camera, captures, add};
}
function geometry(index = null) {
  const value = new T.BufferGeometry();
  value.setAttribute('position', new T.Float32BufferAttribute([-.6, -.2, 0, -.2, .3, 0, .2, -.3, 0, .6, .2, 0], 3));
  value.setAttribute('uv', new T.Float32BufferAttribute([0, 0, .25, .5, .75, .5, 1, 1], 2));
  value.setAttribute('lineDistance', new T.Float32BufferAttribute([0, .25, .5, .75], 1));
  if (index) value.setIndex(new T.BufferAttribute(new index([0, 1, 2, 3]), 1));
  return value;
}

for (const [Material, ObjectType, topology] of [
  [T.PointsMaterial, T.Points, 'point-list'], [T.PointsNodeMaterial, T.Points, 'point-list'],
  [T.PointsMaterial, T.Sprite, 'triangle-list'], [T.PointsNodeMaterial, T.Sprite, 'triangle-list'],
  [T.PointsMaterial, T.Mesh, 'triangle-list'], [T.PointsNodeMaterial, T.Mesh, 'triangle-list'],
  [T.LineBasicMaterial, T.Line, 'line-strip'], [T.LineBasicNodeMaterial, T.Line, 'line-strip'],
  [T.LineBasicMaterial, T.LineSegments, 'line-list'], [T.LineBasicNodeMaterial, T.LineSegments, 'line-list'],
  [T.LineDashedMaterial, T.Line, 'line-strip'], [T.LineDashedNodeMaterial, T.Line, 'line-strip'],
  [T.LineDashedMaterial, T.LineSegments, 'line-list'], [T.LineDashedNodeMaterial, T.LineSegments, 'line-list'],
]) test(`${Material.name} on ${ObjectType.name} retains source WGSL and ${topology} commands`, async t => {
  const f = fixture(t), material = new Material();
  const tint = N.uniform(new T.Color(.25, .5, .75)).setName('primitiveTint'); material.colorNode = tint;
  material.positionNode = N.positionLocal.add(N.vec3(.125, 0, 0));
  if (Material === T.PointsMaterial || Material === T.PointsNodeMaterial) {
    material.sizeNode = N.uniform(11).setName('pointSize'); material.scaleNode = N.vec2(.75, .5); material.rotationNode = N.float(.375);
  }
  if (Material === T.LineDashedMaterial || Material === T.LineDashedNodeMaterial) {
    material.offsetNode = N.uniform(.125).setName('dashOffset'); material.dashScaleNode = N.float(2);
    material.dashSizeNode = N.float(.375); material.gapSizeNode = N.float(.25);
  }
  const object = f.add(ObjectType === T.Sprite ? new ObjectType(material) :
    new ObjectType(ObjectType === T.Mesh ? new T.PlaneGeometry() : geometry(), material));
  object.scale.x = -1; object.count = 3;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const captured = f.captures.find(value => value.draws?.some(draw => draw.object === object)).draws.find(draw => draw.object === object);
  const descriptor = captured.binding.nodeProgram, source = new T.WGSLNodeBuilder(object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.build();
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
  const draws = native(f.device); assert.equal(draws.length, 1);
  assert.equal(draws[0].pipeline.primitive.topology, topology);
  assert.equal(draws[0].args[1], 3); assert.equal(captured.instanceCount, 3);
  assert.equal(captured.frontFaceCW, object.isMesh === true, 'only a Mesh reverses source winding');
  const old = captured.nodeUniforms.map(bytes => bytes.slice()); tint.value.setRGB(.875, .625, .125);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(captured.nodeUniforms, old); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('indexed source line strips choose uint16 and uint32 pipelines and keep captured ranges', async t => {
  const f = fixture(t), material = new T.LineBasicNodeMaterial(); material.colorNode = N.vec3(.2, .4, .6);
  const line = f.add(new T.Line(geometry(Uint16Array), material));
  line.geometry.setDrawRange(1, 2); await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  line.geometry.setDrawRange(0, 4); await drain(f.renderer);
  assert.deepEqual(native(f.device)[0].args.slice(0, 4), [2, 1, 1, 0]);
  assert.equal(native(f.device)[0].pipeline.primitive.stripIndexFormat, 'uint16');
  line.geometry.setIndex(new T.BufferAttribute(new Uint32Array([0, 1, 2, 3]), 1));
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const second = native(f.device).at(-1); assert.equal(second.pipeline.primitive.stripIndexFormat, 'uint32');
  assert.equal(second.args[0], 4); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('retained native line records invalidate uint16, uint32 and nonindexed strip pipelines', async t => {
  const f = fixture(t), material = new T.LineBasicNodeMaterial(), line = f.add(new T.Line(geometry(Uint16Array), material));
  await f.renderer.init(); f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const programs = createThreeNodePrograms({three: T, renderer: f.renderer});
  const context = {object: line, material, scene: f.scene, camera: f.camera, renderId: 1, frameId: 1};
  const nodeProgram = programs.compile(material, line, context), packet = programs.captureDraw(nodeProgram, context);
  const gpu = createGpuProgramGeometry(f.device, line.geometry, nodeProgram.attributes, {attributeTypes: 'webgpu'});
  const fail = (code, message) => { throw Object.assign(Error(message), {code}); };
  const owner = createThreeNodeProgramMeshes({device: f.device, format: 'rgba8unorm', depthFormat: 'depth24plus',
    sampleCount: 1, maxDraws: 8, label: 'retained-source-lines', fail,
    scoped: (_device, run) => ({value: run(), errors: Promise.resolve()})});
  t.after(() => { owner.dispose(); gpu.dispose(); programs.dispose(); });
  const raster = {blend: null, frontFace: 'ccw', cullMode: 'none', writeMask: 15, depthWriteEnabled: true, depthCompare: 'less-equal'};
  const add = () => owner.add(gpu, {nodeProgram, topology: nodeProgram.topology, raster});
  const stage = record => { const command = {}; owner.stage(record, {nodeUniforms: packet.uniforms}, command); return command; };
  let record = await add(); assert.equal(stage(record).pipeline.primitive.stripIndexFormat, 'uint16');
  for (const Index of [Uint32Array, null, Uint16Array]) {
    line.geometry.setIndex(Index ? new T.BufferAttribute(new Index([0, 1, 2, 3]), 1) : null); gpu.update();
    assert.throws(() => stage(record), {code: 'ANIMATION_RENDER_GEOMETRY'});
    record = await add(); const command = stage(record);
    assert.equal(command.indexFormat, Index === Uint32Array ? 'uint32' : Index === Uint16Array ? 'uint16' : null);
    assert.equal(command.pipeline.primitive.stripIndexFormat, command.indexFormat ?? undefined);
  }
});

test('point sprite pixels capture DPR, viewport, node size and logical update frequency before queued edits', async t => {
  const f = fixture(t), material = new T.PointsNodeMaterial({sizeAttenuation: true});
  const counts = {frame: 0, render: 0, object: 0};
  material.colorNode = N.uniform(new T.Color(.25, .5, .75)).onFrameUpdate(() => { counts.frame++; });
  const size = N.uniform(13).setName('sizedPoint').onRenderUpdate(() => { counts.render++; });
  material.sizeNode = size; material.rotationNode = N.uniform(.375).onObjectUpdate(() => { counts.object++; });
  const sprite = f.add(new T.Sprite(material)); sprite.count = 4;
  await f.renderer.init(); f.renderer.setPixelRatio(1.5); f.renderer.setViewport(3, 5, 42, 26);
  f.renderer.render(f.scene, f.camera);
  const first = f.captures.at(-1).draws[0], saved = first.nodeUniforms.map(bytes => bytes.slice());
  f.renderer.setPixelRatio(2); f.renderer.setSize(90, 70, false); size.value = 19; sprite.count = 0;
  f.renderer.render(f.scene, f.camera);
  size.value = 99; sprite.count = 99;
  const captures = f.captures.filter(value => value.draws?.[0]?.object === sprite);
  assert.deepEqual(captures.map(value => value.draws[0].instanceCount), [4, 0]);
  assert.deepEqual(counts, {frame: 1, render: 2, object: 2});
  assert.ok(floats(saved).includes(13)); assert.ok(floats(saved).includes(1.5));
  assert.ok(floats(saved).includes(63)); assert.ok(floats(saved).includes(39));
  assert.ok(floats(captures[1].draws[0].nodeUniforms).includes(19));
  assert.ok(floats(captures[1].draws[0].nodeUniforms).includes(2));
  assert.deepEqual(first.nodeUniforms, saved);
  await drain(f.renderer); assert.equal(native(f.device).length, 1); assert.equal(native(f.device)[0].args[1], 4);
});

test('point primitives ignore size-node updates and keep the source one-pixel shader', async t => {
  const f = fixture(t), material = new T.PointsNodeMaterial(); let sizes = 0;
  material.sizeNode = N.uniform(100).setName('unusedPointSize').onObjectUpdate(() => { sizes++; });
  f.add(new T.Points(geometry(), material)); await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(sizes, 0); assert.equal(native(f.device)[0].pipeline.primitive.topology, 'point-list');
  assert.doesNotMatch(native(f.device)[0].pipeline.vertex.module.code, /unusedPointSize/);
});

test('source point attenuation and primitive changes invalidate capture before callbacks', async t => {
  const f = fixture(t), material = new T.PointsMaterial(); let updates = 0;
  material.sizeNode = N.uniform(7).onObjectUpdate(() => { updates++; });
  const object = f.add(new T.Sprite(material)); await f.renderer.init(); f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const owner = createThreeNodePrograms({three: T, renderer: f.renderer}); t.after(() => owner.dispose());
  const context = {object, material, scene: f.scene, camera: f.camera, renderId: 1, frameId: 1};
  let descriptor = owner.compile(material, object, context); owner.capture(descriptor, context);
  material.sizeAttenuation = false; const before = updates;
  assert.throws(() => owner.capture(descriptor, context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, before);
  const next = owner.compile(material, object, context); assert.notEqual(next, descriptor); descriptor = next;
  object.isPoints = true;
  assert.throws(() => owner.capture(descriptor, context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, before);
});

test('native point clouds read the compute-owned storage allocation without a stale CPU upload', async t => {
  const f = fixture(t), positions = N.instancedArray(4, 'vec3'), material = new T.PointsNodeMaterial();
  material.positionNode = positions.element(N.vertexIndex); material.colorNode = N.vec3(.25, .5, .75);
  const points = f.add(new T.Points(geometry(), material));
  const computePasses = [], create = f.device.createCommandEncoder, submit = f.device.queue.submit;
  f.device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout: index => ({index})});
  f.device.createComputePipelineAsync = descriptor => Promise.resolve(f.device.createComputePipeline(descriptor));
  f.device.createCommandEncoder = descriptor => {
    const encoder = create(descriptor), passes = [];
    return {...encoder, beginComputePass(desc) {
      const pass = {desc, draws: [], groups: new Map(), dispatches: []}; passes.push(pass); computePasses.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.groups.set(index, group); },
        dispatchWorkgroups(...size) { pass.dispatches.push(size); }, end() {}};
    }, finish() { return [...encoder.finish(), ...passes]; }};
  };
  f.device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes) if (pass.dispatches?.length) {
      const buffer = [...pass.groups.values()].flatMap(group => group.entries).map(entry => entry.resource.buffer).find(buffer => buffer.usage & 128);
      new Float32Array(buffer.data).set([.25, .5, .75, 0]);
    }
    submit(commands);
  };
  const kernel = N.Fn(() => positions.element(N.instanceIndex).addAssign(N.vec3(0)))().compute(4);
  await f.renderer.init(); f.renderer.compute(kernel);
  const buffer = captureComputeStorage(f.device, positions.value).resolve().buffer;
  positions.value.array.fill(99); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draw = native(f.device)[0], storages = [...draw.groups.values()].flatMap(({group}) => group.entries)
    .map(entry => entry.resource?.buffer).filter(value => value?.usage & 128);
  assert.ok(storages.includes(buffer)); assert.equal(draw.pipeline.primitive.topology, 'point-list');
  assert.equal(draw.args[0], 4); assert.equal(draw.args[1], 1);
  assert.deepEqual([...new Float32Array(draw.contents.get(buffer).buffer)].slice(0, 4), [.25, .5, .75, 0]);
  assert.equal(f.device.writes.filter(write => write.buffer === buffer).length, 1);
  assert.equal(points.geometry.attributes.position.count, 4); assert.equal(f.renderer._nodeCaptureBytes, 0); kernel.dispose();
});

test('native lines and points use source ArrayCamera views without repeated node updates', async t => {
  const f = fixture(t), child = f.camera.clone(); child.position.x = .5; child.updateMatrixWorld(); f.camera.updateMatrixWorld();
  f.camera.viewport = new T.Vector4(0, 0, 40, 60); child.viewport = new T.Vector4(40, 0, 40, 60);
  const camera = new T.ArrayCamera([f.camera, child]); let updates = 0;
  const material = new T.LineBasicNodeMaterial(); material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; });
  f.add(new T.Line(geometry(), material)); f.add(new T.Points(geometry(), new T.PointsNodeMaterial()));
  await f.renderer.init(); f.renderer.render(f.scene, camera); await drain(f.renderer);
  assert.equal(updates, 1); assert.deepEqual(native(f.device).map(draw => draw.pipeline.primitive.topology),
    ['line-strip', 'line-strip', 'point-list', 'point-list']); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('source LineLoop reports and skips even with native materials while its sibling renders', async t => {
  const f = fixture(t), material = new T.LineBasicNodeMaterial(); let updates = 0;
  material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; });
  f.add(new T.LineLoop(geometry(), material)); f.add(new T.Points(geometry(), new T.PointsNodeMaterial()));
  const errors = [], previous = console.error; console.error = (...args) => errors.push(args.join(' '));
  try { await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer); }
  finally { console.error = previous; }
  assert.equal(updates, 0); assert.equal(native(f.device).length, 1);
  assert.equal(native(f.device)[0].pipeline.primitive.topology, 'point-list');
  assert.ok(errors.some(value => value.includes('THREE.LineLoop are not supported')));
});
