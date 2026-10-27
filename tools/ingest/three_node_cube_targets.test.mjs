/** Exact r186 construction and recorded native submission contracts. The device
 * records resources and ordering; these assertions do not claim GPU pixel parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {targetDevice} from './fixtures/target_residency_fixture.mjs';

const N = THREE.TSL, Renderer = createWebGPURendererClass(THREE, {maxDraws: 64, maxBindings: 64});
const sampled = texture => ({texture, isSampledTexture: true});
const sampler = texture => ({texture, isSampler: true});
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
const cubeView = draw => views(draw).find(view => view.dimension === 'cube');
const tagged = (device, name) => device.snapshots.flat().filter(draw => draw.pipeline.fragment.module.code.includes(name));

function ownerFixture(t, options = {}) {
  const f = targetDevice(); f.device.limits.maxTextureArrayLayers = 256;
  const targets = createThreeRenderTargets(THREE, {getDevice: () => f.device, ...options});
  const inputs = createThreeComputeTargets({targets, device: f.device});
  t.after(() => { inputs.dispose(); targets.dispose(); });
  return {...f, targets, inputs};
}
function fixture(t) {
  const device = geometryDevice(); device.textures = []; device.textureWrites = []; device.events = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createComputePipelineAsync = async descriptor => device.createComputePipeline(descriptor);
  device.queue.writeTexture = (destination, data) => device.textureWrites.push({texture: destination.texture,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  const createEncoder = device.createCommandEncoder, submit = device.queue.submit;
  device.createCommandEncoder = (...args) => {
    const encoder = createEncoder(...args), finish = encoder.finish, compute = [];
    return {...encoder, beginComputePass() {
      const pass = {compute: true, groups: new Map(), draws: []}; compute.push(pass);
      return {setPipeline(value) { pass.pipeline = value; }, setBindGroup(index, group) { pass.groups.set(index, {group}); },
        dispatchWorkgroups(...counts) { pass.counts = counts; }, end() {}};
    }, finish() { return [...finish(), ...compute]; }};
  };
  device.queue.submit = commands => {
    submit(commands);
    for (const command of commands) for (const pass of command) device.events.push(pass);
  };
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const material = new THREE.NodeMaterial(); material.lights = false;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(), material); scene.add(mesh);
  return {device, renderer, material, mesh, scene, camera};
}
function producer(target) {
  const camera = new THREE.CubeCamera(.1, 10, target), scene = new THREE.Scene();
  const material = new THREE.MeshBasicNodeMaterial({side: THREE.BackSide});
  material.colorNode = N.uniform(new THREE.Color(.25, .5, .75)).setName('cubeProducer');
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(4, 4, 4), material));
  return {camera, scene, material};
}

test('native cube color registration preserves the public owner pointer and borrows all six faces', async t => {
  const f = ownerFixture(t), target = new THREE.CubeRenderTarget(8);
  assert.equal(target.texture.renderTarget, null);
  assert.throws(() => f.inputs.captureBinding(sampled(target.texture)), {code: 'THREE_TARGET_SOURCE'});
  const entry = f.targets.capture(target), source = f.inputs.captureBinding(sampled(target.texture));
  assert.deepEqual(f.inputs.layout(source), {texture: {sampleType: 'float', viewDimension: 'cube', multisampled: false}});
  assert.equal(f.calls.textures.length, 0, 'capture and layout do not allocate GPU storage');
  f.targets.render(entry, () => {}, {face: 5});
  f.inputs.withDispatch([source], () => {
    const view = f.inputs.resolve(source);
    assert.deepEqual(view.descriptor, {dimension: 'cube', baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: 6});
    assert.equal(view.texture, f.calls.textures[0]);
    assert.equal(view.texture.size.depthOrArrayLayers, 6);
    assert.equal(entry.busy, true);
  });
  assert.equal(f.calls.textures.length, 2); assert.equal(entry.sample, null);
  assert.equal(target.texture.renderTarget, null); assert.equal(target.depth, 1);
  f.inputs.release(source); await f.inputs.whenIdle();
});

test('cube color and depth bindings share one allocation and preserve captured sampler states', async t => {
  const f = ownerFixture(t), target = new THREE.CubeRenderTarget(4);
  target.depthTexture = new THREE.CubeDepthTexture(4); target.depthTexture.compareFunction = THREE.LessEqualCompare;
  const entry = f.targets.capture(target);
  const color = f.inputs.captureBinding(sampled(target.texture)), depth = f.inputs.captureBinding(sampled(target.depthTexture));
  target.texture.minFilter = target.texture.magFilter = THREE.NearestFilter;
  const first = f.inputs.captureBinding(sampler(target.texture));
  target.texture.minFilter = target.texture.magFilter = THREE.LinearFilter;
  const second = f.inputs.captureBinding(sampler(target.texture));
  f.inputs.withDispatch([color, depth, first, second], () => {
    assert.equal(f.inputs.resolve(color).texture, f.calls.textures[0]);
    assert.equal(f.inputs.resolve(depth).texture, f.calls.textures[1]);
    assert.equal(f.inputs.resolve(depth).descriptor.aspect, 'depth-only');
    assert.equal(f.inputs.resolve(first).minFilter, 'nearest');
    assert.equal(f.inputs.resolve(second).minFilter, 'linear');
    assert.equal(entry.sample, null); assert.equal(entry.depthSample, null);
  });
  assert.equal(f.targets.allocatedBytes, 4 * 4 * 6 * 8);
  for (const source of [color, depth, first, second]) f.inputs.release(source);
  await f.inputs.whenIdle();
});

test('cube source disposal and replacement invalidate pending readers before any GPU allocation', t => {
  for (const change of [target => target.dispose(), target => target.texture.dispose(),
    target => { target.texture = new THREE.CubeRenderTarget(4).texture; }]) {
    const f = ownerFixture(t), target = new THREE.CubeRenderTarget(4);
    f.targets.capture(target); const source = f.inputs.captureBinding(sampled(target.texture));
    change(target);
    assert.throws(() => f.inputs.withDispatch([source], () => assert.fail('stale cube read')), {code: 'THREE_TARGET_STALE'});
    assert.equal(f.calls.textures.length, 0); f.inputs.release(source);
  }
});

test('cube color feedback, aggregate budgets and ambiguous target ownership fail before sampling', t => {
  const f = ownerFixture(t), target = new THREE.CubeRenderTarget(4), entry = f.targets.capture(target);
  const source = f.inputs.captureBinding(sampled(target.texture));
  f.targets.render(entry, () => {
    assert.throws(() => f.inputs.withDispatch([source], () => assert.fail('feedback read')), {code: 'THREE_TARGET_REENTRANT'});
  });
  assert.throws(() => f.targets.bindingsFor(target).get(target.texture), {code: 'THREE_TARGET_FEEDBACK'});
  const alias = new THREE.CubeRenderTarget(4); alias.texture = target.texture;
  assert.throws(() => f.targets.capture(alias), {code: 'THREE_TARGET_SOURCE'});
  f.inputs.release(source);
  const limited = ownerFixture(t, {maxBytes: 4 * 4 * 6 * 8 - 1}), other = new THREE.CubeRenderTarget(4);
  limited.targets.capture(other); const lease = limited.inputs.captureBinding(sampled(other.texture));
  assert.throws(() => limited.inputs.withDispatch([lease], () => assert.fail('over budget')), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(limited.calls.textures.length, 0); limited.inputs.release(lease);
});

test('CubeCamera output reaches exact native cube shaders without texture uploads or public descriptor edits', async t => {
  const f = fixture(t); await f.renderer.init();
  const target = new THREE.CubeRenderTarget(8), p = producer(target), faces = target.texture.image;
  p.camera.update(f.renderer, p.scene);
  f.material.colorNode = N.cubeTexture(target.texture, N.vec3(.25, .5, 1)).level(0)
    .mul(N.uniform(1).setName('targetCubeSurface'));
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.renderer.info.f3d.offscreenRenders, 6);
  const consumers = tagged(f.device, 'targetCubeSurface'); assert.equal(consumers.length, 1);
  const view = cubeView(consumers[0]); assert.ok(view);
  assert.equal(view.texture.depthOrArrayLayers, 6); assert.equal(view.texture.format, 'rgba8unorm');
  const producerPasses = f.device.passes.filter(pass => pass.desc.colorAttachments[0].view.texture === view.texture);
  assert.equal(producerPasses.length, 6);
  assert.deepEqual(producerPasses.map(pass => pass.desc.colorAttachments[0].view.baseArrayLayer), [0, 1, 2, 3, 4, 5]);
  assert.ok(producerPasses.every(pass => f.device.events.indexOf(pass) < f.device.events.findIndex(event => event.draws?.some(draw => cubeView(draw)?.texture === view.texture))));
  assert.equal(f.device.textureWrites.filter(write => write.texture === view.texture).length, 0);
  assert.equal(target.texture.renderTarget, null); assert.equal(target.texture.image, faces);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  const context = {scene: f.scene, camera: f.camera, object: f.mesh, material: f.material};
  const programs = createThreeNodePrograms({three: THREE, renderer: f.renderer}); t.after(() => programs.dispose());
  const descriptor = programs.compile(f.material, f.mesh, context);
  const builder = new THREE.WGSLNodeBuilder(f.mesh, f.renderer); builder.scene = f.scene; builder.camera = f.camera; builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
});

test('queued cube producers and consumers retain source submission order and reuse the native color image', async t => {
  const f = fixture(t); await f.renderer.init();
  const target = new THREE.CubeRenderTarget(8), p = producer(target);
  f.material.colorNode = N.cubeTexture(target.texture, N.vec3(1, 0, 0)).level(0)
    .mul(N.uniform(1).setName('orderedCubeConsumer'));
  p.camera.update(f.renderer, p.scene); f.renderer.render(f.scene, f.camera);
  p.material.colorNode.value.setRGB(.75, .5, .25);
  p.camera.update(f.renderer, p.scene); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const consumers = tagged(f.device, 'orderedCubeConsumer'); assert.equal(consumers.length, 2);
  assert.equal(cubeView(consumers[0]).texture, cubeView(consumers[1]).texture);
  const relevant = f.device.events.flatMap(pass => {
    if (pass.desc?.colorAttachments[0].view.texture === cubeView(consumers[0]).texture) return ['produce'];
    if (pass.draws?.some(draw => draw.pipeline.fragment.module.code.includes('orderedCubeConsumer'))) return ['consume'];
    return [];
  });
  assert.deepEqual(relevant, [...Array(6).fill('produce'), 'consume', ...Array(6).fill('produce'), 'consume']);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 12); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native cube environments filter rendered faces through retained PMREM before their PBR consumer', async t => {
  const f = fixture(t); await f.renderer.init();
  const target = new THREE.CubeRenderTarget(16), p = producer(target);
  p.camera.update(f.renderer, p.scene);
  const material = new THREE.MeshStandardNodeMaterial({roughness: .4, metalness: .25});
  material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).setName('renderedCubeEnvironment');
  f.mesh.material = material; f.scene.environment = target.texture;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const consumer = tagged(f.device, 'renderedCubeEnvironment'); assert.equal(consumer.length, 1);
  const conversions = f.device.snapshots.flat().filter(draw => draw.pipeline.fragment.module.code.includes('texture_cube<f32>'));
  assert.equal(conversions.length, 1);
  const source = cubeView(conversions[0]).texture;
  assert.equal(source.depthOrArrayLayers, 6); assert.equal(source.format, 'rgba8unorm');
  assert.ok(views(consumer[0]).some(view => view.texture.format === 'rgba16float' && view.texture.width === 336 && view.texture.height === 64));
  assert.equal(f.renderer.info.f3d.offscreenRenders, 19);
  assert.equal(f.device.textureWrites.filter(write => write.texture === source).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('public native cube sampling rejects destination feedback before a consumer enters the queue', async t => {
  const f = fixture(t); await f.renderer.init();
  const target = new THREE.CubeRenderTarget(8); f.renderer.setRenderTarget(target, 0);
  f.material.colorNode = N.cubeTexture(target.texture, N.vec3(1, 0, 0)).level(0);
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_FEEDBACK'});
  assert.equal(f.device.textures.filter(texture => texture.depthOrArrayLayers === 6).length, 0);
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('public native compute can read the same rendered cube through source WGSL texture bindings', async t => {
  const f = fixture(t); await f.renderer.init();
  const target = new THREE.CubeRenderTarget(8), p = producer(target);
  p.camera.update(f.renderer, p.scene);
  const readCube = N.wgslFn('fn readRenderedCube(image: texture_cube<f32>, imageSampler: sampler) -> vec4<f32> { return textureSampleLevel(image, imageSampler, vec3<f32>(1.0, 0.0, 0.0), 0.0); }');
  const output = N.storage(new THREE.StorageBufferAttribute(1, 4), 'vec4', 1);
  // A raw texture argument leaves coordinates to the authored WGSL function.
  // CubeTextureNode's material/environment rotation needs a render context and
  // is not part of this source compute graph.
  const image = N.texture(target.texture, N.vec2(0)).setUpdateMatrix(false);
  const kernel = N.Fn(() => { output.element(0).assign(readCube({image, imageSampler: N.sampler(image)})); })().compute(1);
  f.renderer.compute(kernel); await f.renderer.waitForGPU();
  const dispatches = f.device.events.filter(event => event.compute); assert.equal(dispatches.length, 1);
  const view = cubeView(dispatches[0]); assert.ok(view); assert.equal(view.texture.depthOrArrayLayers, 6);
  assert.match(dispatches[0].pipeline.compute.module.code, /texture_cube<f32>/);
  assert.equal(f.device.passes.filter(pass => pass.desc.colorAttachments[0].view.texture === view.texture).length, 6);
  assert.equal(f.device.textureWrites.filter(write => write.texture === view.texture).length, 0);
  assert.equal(target.texture.renderTarget, null);
});
