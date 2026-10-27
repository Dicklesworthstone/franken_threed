/** Native source sprite graphs retain the pinned billboard shader and draw state. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T, {maxDraws: 16});
function compiler(t, material, object) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})},
      capabilities: {getUniformBufferLimit: () => 65536}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null,
    currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  const owner = createThreeNodePrograms({three: T, renderer}); t.after(() => owner.dispose());
  const scene = new T.Scene(); scene.add(object); scene.updateMatrixWorld();
  const camera = new T.PerspectiveCamera(50, 1, .1, 100); camera.position.z = 4;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
  const context = {material, object, scene, camera, renderId: 1, frameId: 1, time: 0, deltaTime: 0};
  return {owner, renderer, scene, camera, context, compile: () => owner.compile(material, object, context)};
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
function graphs(material) {
  const position = N.uniform(new T.Vector3(.125, -.25, 0)).setName('spritePosition');
  const scale = N.uniform(new T.Vector2(.5, .75)).setName('spriteScale');
  const rotation = N.uniform(.375).setName('spriteRotation');
  material.positionNode = position.add(N.vec3(N.instanceIndex.mul(.5), 0, 0));
  material.scaleNode = scale; material.rotationNode = rotation;
  return {position, scale, rotation};
}
const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');

for (const Type of [T.SpriteNodeMaterial, T.SpriteMaterial]) for (const mesh of [false, true])
test(`${Type.name} on ${mesh ? 'Mesh' : 'Sprite'} retains exact source position, scale and rotation WGSL`, t => {
  const material = new Type({opacity: .6}), nodes = graphs(material);
  const object = mesh ? new T.Mesh(new T.PlaneGeometry(), material) : new T.Sprite(material);
  object.count = 3; object.scale.set(-2, 3, .5);
  if (!mesh) object.center.set(.125, .875);
  const f = compiler(t, material, object), descriptor = f.compile();
  const source = new T.WGSLNodeBuilder(object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.build();
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
  assert.match(descriptor.vertex, /@builtin\( instance_index \)/);
  assert.match(descriptor.vertex, /spriteScale/); assert.match(descriptor.vertex, /spriteRotation/);
  const first = f.owner.capture(descriptor, f.context), saved = first.map(bytes => bytes.slice());
  nodes.position.value.x = .875; nodes.scale.value.set(.25, .125); nodes.rotation.value = .625;
  if (!mesh) object.center.set(.75, .25);
  assert.equal(f.compile(), descriptor, 'live billboard values retain their compiled program');
  const second = f.owner.capture(descriptor, {...f.context, renderId: 2});
  assert.notDeepEqual(second, first); assert.deepEqual(first, saved, 'accepted source uniforms retain earlier values');
});

test('sprite attenuation and anchor-presence branches rebuild before node callbacks', t => {
  const material = new T.SpriteMaterial(); let updates = 0;
  material.rotationNode = N.uniform(.25).onObjectUpdate(() => { updates++; return .25; });
  const object = new T.Sprite(material), f = compiler(t, material, object);
  let descriptor = f.compile(); f.owner.capture(descriptor, f.context);
  for (const change of [() => { material.sizeAttenuation = false; }, () => { object.center = null; }]) {
    const before = updates; change();
    assert.throws(() => f.owner.capture(descriptor, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    assert.equal(updates, before, 'stale billboard programs do not execute callbacks');
    const next = f.compile(); assert.notEqual(next, descriptor); assert.notEqual(next.vertex, descriptor.vertex);
    descriptor = next; f.owner.capture(descriptor, {...f.context, renderId: updates + 1});
  }
});

test('native Sprite.count is one source instanced draw and queued uniforms/counts stay immutable', async t => {
  const f = fixture(t), material = new T.SpriteNodeMaterial(), nodes = graphs(material);
  const object = new T.Sprite(material); object.frustumCulled = false; object.scale.x = -1; f.scene.add(object);
  let before = 0, after = 0;
  object.onBeforeRender = () => { before++; }; object.onAfterRender = () => { after++; };
  await f.renderer.init();
  object.count = 32; f.renderer.render(f.scene, f.camera);
  const first = f.captures.at(-1), saved = first.draws[0].nodeUniforms.map(bytes => bytes.slice());
  object.count = 0; nodes.rotation.value = .625; object.center.set(.25, .75); f.renderer.render(f.scene, f.camera);
  object.count = 5; nodes.scale.value.set(.125, .25); f.renderer.render(f.scene, f.camera);
  object.count = 99; nodes.rotation.value = 9;
  assert.deepEqual(f.captures.map(value => value.draws[0].instanceCount), [32, 0, 5]);
  assert.ok(f.captures.every(value => value.draws[0].frontFaceCW === false), 'source callbacks retain sprite winding');
  assert.deepEqual(first.draws[0].nodeUniforms, saved);
  assert.deepEqual([before, after], [3, 3], 'callbacks run per source render object, including zero instances');
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.module.code.includes('spriteRotation'));
  assert.deepEqual(draws.map(draw => draw.args[1]), [32, 5], 'native instances never expand into CPU sprite draws');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('source sprite sorting ignores geometry bounds and negative scale does not flip billboard winding', async t => {
  const f = fixture(t), near = new T.Sprite(new T.SpriteNodeMaterial()), far = new T.Sprite(new T.SpriteNodeMaterial());
  for (const object of [near, far]) { object.material.rotationNode = N.float(.25); object.frustumCulled = false; }
  near.scale.x = -2; far.position.z = -1;
  far.geometry = far.geometry.clone(); far.geometry.boundingSphere = new T.Sphere(new T.Vector3(0, 0, 1000), 1);
  f.scene.add(near, far); await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(f.captures[0].draws.map(draw => draw.object), [far, near]);
  assert.deepEqual(f.captures[0].draws.map(draw => draw.frontFaceCW), [false, false]);
  assert.ok(f.device.snapshots.flat().filter(draw => draw.streams.size).every(draw => draw.pipeline.primitive.frontFace === 'ccw'));
});

test('ordinary SpriteMaterial on Mesh selects source billboard shading and Mesh.count instancing', async t => {
  const f = fixture(t), material = new T.SpriteMaterial({rotation: .25}), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.count = 3; object.frustumCulled = false; f.scene.add(object);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.captures[0].draws[0].instanceCount, 3);
  assert.match(f.captures[0].draws[0].binding.nodeProgram.vertex, /cos\(/);
  assert.equal(f.renderer.info.f3d.nativeDrawCalls, 1);
});

test('native sprite maps, alpha test, opacity and exponential fog use source material composition', async t => {
  const f = fixture(t), material = new T.SpriteMaterial({opacity: .5, alphaTest: .25});
  material.rotationNode = N.float(.125);
  material.map = new T.DataTexture(new Uint8Array([255, 64, 32, 128]), 1, 1);
  material.alphaMap = new T.DataTexture(new Uint8Array([192, 64, 255, 255]), 1, 1);
  material.map.needsUpdate = material.alphaMap.needsUpdate = true;
  const object = new T.Sprite(material); f.scene.add(object); f.scene.fog = new T.FogExp2(0x123456, .125);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draw = f.captures[0].draws[0];
  assert.equal(draw.nodeTextures.length, 2); assert.match(draw.binding.nodeProgram.fragment, /discard/);
  assert.match(draw.binding.nodeProgram.fragment, /exp\(/);
  assert.equal(f.renderer.info.f3d.nativeDrawCalls, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});


test('sprite instanced position attributes retain GPU-visible history and geometry instance-count precedence', async t => {
  const f = fixture(t), material = new T.SpriteNodeMaterial();
  const values = new Float32Array([-.5, 0, 0, .5, 0, 0, 1, 0, 0]);
  const positions = new T.InstancedBufferAttribute(values, 3); material.positionNode = N.attribute('particlePosition', 'vec3');
  const geometry = new T.InstancedBufferGeometry().copy(new T.PlaneGeometry()); geometry.instanceCount = 2;
  geometry.setAttribute('particlePosition', positions);
  const object = new T.Mesh(geometry, material); object.count = 99; object.frustumCulled = false; f.scene.add(object);
  await f.renderer.init();
  for (let step = 0; step < 3; step++) {
    if (step === 1) values[0] = -.875;
    if (step === 2) positions.needsUpdate = true;
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  }
  assert.deepEqual(f.captures.map(value => value.draws[0].instanceCount), [2, 2, 2]);
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.buffers.some(buffer => buffer.stepMode === 'instance'));
  assert.equal(draws.length, 3);
  const bytes = draws.map(draw => {
    const slot = draw.pipeline.vertex.buffers.findIndex(buffer => buffer.stepMode === 'instance');
    return draw.contents.get(draw.streams.get(slot));
  });
  assert.deepEqual(bytes[1], bytes[0], 'CPU edits without needsUpdate stay stale on the GPU');
  assert.notDeepEqual(bytes[2], bytes[0], 'the source upload version publishes the new position');
  assert.equal(new Float32Array(bytes[2].buffer)[0], -.875);
});
