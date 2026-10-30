/** Native point/line graphs retain source shaders and primitive draw ownership. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T, {maxDraws: 16});
const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
function geometry(Index = null) {
  const result = new T.BufferGeometry();
  result.setAttribute('position', new T.Float32BufferAttribute([-.5, -.5, 0, .5, -.5, 0, -.5, .5, 0, .5, .5, 0], 3));
  if (Index) result.setIndex(new T.BufferAttribute(new Index([0, 1, 2, 3]), 1));
  return result;
}
function compiler(t, material, object) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})},
      capabilities: {getUniformBufferLimit: () => 65536}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null,
    getPixelRatio: () => 2, getSize: target => target.set(160, 90),
    getViewport: target => target.set(4, 8, 120, 60),
    currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  const owner = createThreeNodePrograms({three: T, renderer}); t.after(() => owner.dispose());
  const scene = new T.Scene(); scene.add(object); scene.updateMatrixWorld();
  const camera = new T.PerspectiveCamera(50, 1, .1, 100); camera.position.z = 4;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
  const context = {material, object, scene, camera, renderId: 1, frameId: 1, time: 0, deltaTime: 0};
  return {owner, renderer, scene, camera, context, compile: () => owner.compile(material, object, context)};
}
function assertSource(f, object, descriptor) {
  const source = new T.WGSLNodeBuilder(object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.build();
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
}
function fixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  return {renderer, device, scene, camera, captures};
}

for (const Type of [T.PointsNodeMaterial, T.PointsMaterial]) for (const sprite of [false, true])
test(`${Type.name} on ${sprite ? 'Sprite' : 'Points'} retains the source point-size and billboard graph`, t => {
  const material = new Type({size: 13}), tint = N.uniform(new T.Color(.25, .5, .75)).setName('primitiveTint');
  material.colorNode = tint; material.sizeNode = N.uniform(new T.Vector2(7, 9)).setName('primitiveSize');
  material.scaleNode = N.vec2(.5, .75); material.rotationNode = N.float(.375);
  material.positionNode = sprite ? N.vec3(.125, -.25, 0) : N.positionLocal.add(N.vec3(.125, 0, 0));
  const object = sprite ? new T.Sprite(material) : new T.Points(geometry(), material);
  const f = compiler(t, material, object), descriptor = f.compile(); assertSource(f, object, descriptor);
  assert.equal(descriptor.topology, sprite ? 'triangles' : 'points');
  if (!sprite) assert.doesNotMatch(descriptor.vertex, /primitiveSize|cos\(/, 'native points remain one-pixel primitives');
  if (sprite && material.isNodeMaterial) assert.match(descriptor.vertex, /primitiveSize/);
  const first = f.owner.capture(descriptor, f.context), saved = first.map(bytes => bytes.slice());
  tint.value.setRGB(.875, .625, .375); material.sizeNode.value.set(11, 15);
  assert.equal(f.compile(), descriptor);
  const second = f.owner.capture(descriptor, {...f.context, renderId: 2});
  assert.notDeepEqual(second, first); assert.deepEqual(first, saved);
});

for (const Type of [T.LineBasicNodeMaterial, T.LineBasicMaterial]) for (const segments of [false, true])
test(`${Type.name} on ${segments ? 'LineSegments' : 'Line'} retains exact source WGSL and topology`, t => {
  const material = new Type({linewidth: 8}); material.colorNode = N.vec3(.25, .5, .75);
  material.positionNode = N.positionLocal.add(N.vec3(.125, 0, 0));
  const object = new (segments ? T.LineSegments : T.Line)(geometry(Uint32Array), material);
  const f = compiler(t, material, object), descriptor = f.compile(); assertSource(f, object, descriptor);
  assert.equal(descriptor.topology, segments ? 'lines' : 'line-strip');
  assert.equal(descriptor.stripIndexFormat, segments ? null : 'uint32');
  assert.equal(descriptor.wireframe, false);
  f.owner.capture(descriptor, f.context);
});

test('point billboard attenuation changes rebuild before executing a draw callback', t => {
  const material = new T.PointsMaterial(); let updates = 0;
  material.colorNode = N.uniform(.25).onObjectUpdate(() => { updates++; return .25; });
  material.sizeNode = N.float(7);
  const object = new T.Sprite(material), f = compiler(t, material, object), descriptor = f.compile();
  f.owner.capture(descriptor, f.context); const before = updates;
  material.sizeAttenuation = false;
  assert.throws(() => f.owner.capture(descriptor, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, before);
  assert.notEqual(f.compile(), descriptor);
});

test('native line strip index-width changes rebuild before observable node updates', t => {
  const material = new T.LineBasicNodeMaterial(); let updates = 0;
  material.colorNode = N.uniform(.25).onObjectUpdate(() => { updates++; return .25; });
  const object = new T.Line(geometry(Uint16Array), material), f = compiler(t, material, object);
  const descriptor = f.compile(); f.owner.capture(descriptor, f.context); const before = updates;
  object.geometry.setIndex(new T.BufferAttribute(new Uint32Array([0, 1, 2, 3]), 1));
  assert.throws(() => f.owner.capture(descriptor, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, before);
  const next = f.compile(); assert.notEqual(next, descriptor); assert.equal(next.stripIndexFormat, 'uint32');
});

test('default node point and line materials submit native point-list, line-list and both indexed strip formats', async t => {
  const f = fixture(t), objects = [
    new T.Points(geometry(Uint16Array), new T.PointsNodeMaterial({size: 99})),
    new T.LineSegments(geometry(), new T.LineBasicNodeMaterial({linewidth: 9})),
    new T.Line(geometry(Uint16Array), new T.LineBasicNodeMaterial()),
    new T.Line(geometry(Uint32Array), new T.LineBasicNodeMaterial()),
  ];
  for (const object of objects) { object.frustumCulled = false; f.scene.add(object); }
  objects[0].geometry.setDrawRange(1, 3);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.captures[0].draws.length, 4);
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
  assert.equal(draws.length, 4, 'native primitives are not expanded into additional draws');
  assert.deepEqual(draws.map(draw => draw.pipeline.primitive.topology).sort(), ['line-list', 'line-strip', 'line-strip', 'point-list']);
  const points = draws.find(draw => draw.pipeline.primitive.topology === 'point-list');
  assert.equal(points.indexed, true); assert.deepEqual(points.args.slice(0, 3), [3, 1, 1]);
  assert.equal(points.pipeline.vertex.buffers[0].stepMode, 'vertex');
  assert.deepEqual(draws.filter(draw => draw.pipeline.primitive.topology === 'line-strip')
    .map(draw => [draw.index.format, draw.pipeline.primitive.stripIndexFormat]).sort(), [['uint16', 'uint16'], ['uint32', 'uint32']]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued point draws own immutable source ranges, instance counts and uniforms', async t => {
  const f = fixture(t), material = new T.PointsMaterial({size: 17}), tint = N.uniform(.25).setName('queuedPointTint');
  material.colorNode = tint;
  const object = new T.Points(geometry(), material); object.frustumCulled = false; f.scene.add(object);
  await f.renderer.init();
  object.count = 2; object.geometry.setDrawRange(0, 3); f.renderer.render(f.scene, f.camera);
  const first = f.captures.at(-1).draws[0], saved = first.nodeUniforms.map(bytes => bytes.slice());
  object.count = 4; tint.value = .75; object.geometry.setDrawRange(1, 2); f.renderer.render(f.scene, f.camera);
  object.count = 99; tint.value = 9; object.geometry.setDrawRange(3, 1);
  assert.deepEqual(first.nodeUniforms, saved);
  assert.deepEqual(f.captures.map(frame => frame.draws.map(draw => [draw.first, draw.count, draw.instanceCount])), [[[0, 3, 2]], [[1, 2, 4]]]);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.primitive.topology === 'point-list');
  assert.deepEqual(draws.map(draw => draw.args.slice(0, 3)), [[3, 2, 0], [2, 4, 1]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued indexed strips retain each captured index width and source upload bytes', async t => {
  const f = fixture(t), material = new T.LineBasicNodeMaterial();
  const object = new T.Line(geometry(Uint16Array), material); object.frustumCulled = false; f.scene.add(object);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  object.geometry.setIndex(new T.BufferAttribute(new Uint32Array([3, 2, 1, 0]), 1));
  f.renderer.render(f.scene, f.camera);
  object.geometry.index.array.fill(0);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.primitive.topology === 'line-strip');
  assert.deepEqual(draws.map(draw => draw.pipeline.primitive.stripIndexFormat), ['uint16', 'uint32']);
  assert.deepEqual(draws.map(draw => {
    const data = draw.contents.get(draw.index.buffer), Type = draw.index.format === 'uint16' ? Uint16Array : Uint32Array;
    return [...new Type(data.buffer, data.byteOffset, 4)];
  }), [[0, 1, 2, 3], [3, 2, 1, 0]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued byte-indexed strips widen without losing captured versions or native vertex 255', async t => {
  const f = fixture(t), material = new T.LineBasicNodeMaterial(), source = geometry(Uint8Array);
  source.setAttribute('position', new T.BufferAttribute(new Float32Array(256 * 3), 3));
  source.index.array.set([0, 1, 255, 3]);
  const object = new T.Line(source, material); object.frustumCulled = false; f.scene.add(object);
  const bytes = source.index.array;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  bytes.set([3, 255, 1, 0]); source.index.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  source.setIndex(new T.BufferAttribute(new Uint16Array([1, 0, 3, 2]), 1)); f.renderer.render(f.scene, f.camera);
  bytes.fill(73); source.index.array.fill(73); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.primitive.topology === 'line-strip');
  assert.deepEqual(draws.map(draw => [draw.index.format, draw.pipeline.primitive.stripIndexFormat]),
    [['uint32', 'uint32'], ['uint32', 'uint32'], ['uint16', 'uint16']]);
  assert.deepEqual(draws.map(draw => {
    const data = draw.contents.get(draw.index.buffer), Type = draw.index.format === 'uint16' ? Uint16Array : Uint32Array;
    return [...new Type(data.buffer, data.byteOffset, 4)];
  }), [[0, 1, 255, 3], [3, 255, 1, 0], [1, 0, 3, 2]]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const Material of [T.MeshBasicMaterial, T.MeshLambertMaterial, T.MeshPhongMaterial])
test(`ordinary ${Material.name} byte indices use native vertex 255 and immutable queued uploads`, async t => {
  const f = fixture(t), source = geometry();
  source.setAttribute('position', new T.BufferAttribute(new Float32Array(256 * 3), 3));
  source.setIndex(new T.BufferAttribute(new Uint8Array([0, 1, 255]), 1));
  const object = new T.Mesh(source, new Material()); object.frustumCulled = false; f.scene.add(object);
  const bytes = source.index.array;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.ok(f.captures.at(-1).draws.find(draw => draw.object === object)?.binding.nodeProgram,
    'ordinary byte-index materials must reach the source WebGPU owner');
  bytes.set([255, 1, 0]); source.index.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  bytes.fill(73); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.index && draw.streams.size);
  assert.deepEqual(draws.map(draw => draw.index.format), ['uint32', 'uint32']);
  assert.deepEqual(draws.map(draw => {
    const data = draw.contents.get(draw.index.buffer);
    return [...new Uint32Array(data.buffer, data.byteOffset, 3)];
  }), [[0, 1, 255], [255, 1, 0]]);
  assert.equal(source.index.array, bytes); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('sized point sprites retain source instancing and live viewport, pixel ratio and size uniforms', async t => {
  const f = fixture(t), material = new T.PointsNodeMaterial();
  material.sizeNode = N.uniform(new T.Vector2(7, 11)).setName('spritePointSize');
  material.positionNode = N.vec3(N.instanceIndex.mul(.25), 0, 0);
  const object = new T.Sprite(material); object.count = 3; object.frustumCulled = false; f.scene.add(object);
  await f.renderer.init(); f.renderer.setPixelRatio(2); f.renderer.setSize(80, 60);
  f.renderer.render(f.scene, f.camera); const first = f.captures.at(-1).draws[0];
  const saved = first.nodeUniforms.map(bytes => bytes.slice());
  f.renderer.setViewport(2, 4, 40, 30); material.sizeNode.value.set(13, 17);
  f.renderer.render(f.scene, f.camera);
  assert.equal(f.captures.at(-1).draws[0].binding.nodeProgram, first.binding.nodeProgram);
  assert.notDeepEqual(f.captures.at(-1).draws[0].nodeUniforms, first.nodeUniforms);
  assert.deepEqual(first.nodeUniforms, saved);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.module.code.includes('spritePointSize'));
  assert.equal(draws.length, 2); assert.ok(draws.every(draw => draw.pipeline.primitive.topology === 'triangle-list' && draw.args[1] === 3));
});

test('LineLoop remains outside the source native primitive family', t => {
  const material = new T.LineBasicNodeMaterial(); material.colorNode = N.vec3(.25);
  const object = new T.LineLoop(geometry(), material), f = compiler(t, material, object);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_OBJECT'});
});
