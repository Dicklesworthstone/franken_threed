/** Native storage texel reads with the pinned builder and a recording device.
 * These checks prove layouts, immutable selection and shared residency. Actual
 * shader execution is covered by the source/candidate native_storage_reads lane. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t, limits = {}) {
  const device = geometryDevice(), textures = [], captures = [], events = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000, maxStorageTexturesPerShaderStage: 4, ...limits});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = () => { throw new Error('Storage consumers must not upload CPU pixels'); };
  device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createComputePipelineAsync = async descriptor => device.createComputePipeline(descriptor);
  const encode = device.createCommandEncoder, submit = device.queue.submit;
  device.createCommandEncoder = () => {
    const encoder = encode(), compute = [];
    return {...encoder, beginComputePass() {
      const pass = {draws: [], compute: true, groups: []}; compute.push(pass);
      return {setPipeline() {}, setBindGroup(index, group) { pass.groups[index] = group; }, dispatchWorkgroups() {}, end() {}};
    }, finish() { return [...encoder.finish(), ...compute]; }};
  };
  device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes) {
      if (pass.compute) events.push({compute: pass.groups.flatMap(group => group.entries).filter(entry => entry.resource.texture)});
      else for (const draw of pass.draws) if (draw.pipeline.vertex.entryPoint === 'main') events.push({draw});
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
  return {renderer, device, material, mesh, scene, camera, textures, captures, events};
}
function storage(t, {array = false, type = T.UnsignedByteType, format = T.RGBAFormat} = {}) {
  const texture = array ? new T.StorageArrayTexture(4, 4, 2) : new T.StorageTexture(4, 4);
  Object.assign(texture, {type, format, generateMipmaps: true, mipmapsAutoUpdate: false,
    minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  t.after(() => texture.dispose()); return texture;
}
function graph(f, texture, mip = 1) {
  let node = N.storageTexture(texture, N.ivec2(0, 0)).setSampler(false).setMipLevel(mip);
  if (texture.isArrayTexture) node = node.depth(N.int(1));
  f.material.colorNode = N.vec3(node.xyz);
  f.material.positionNode = N.positionLocal.add(N.vec3(node.x.toFloat().mul(.01), 0, 0));
  return node;
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const layouts = draw => draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries);
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .filter(entry => entry.resource?.texture).map(entry => entry.resource);
const descriptor = f => f.captures.at(-1).draws[0].binding.nodeProgram;
function sourceShaders(f) {
  const builder = new T.WGSLNodeBuilder(f.mesh, f.renderer); builder.scene = f.scene; builder.camera = f.camera; builder.build();
  return builder;
}

for (const profile of [
  {name: 'byte RGBA', native: 'rgba8unorm'},
  {name: 'half RGBA', type: T.HalfFloatType, native: 'rgba16float'},
  {name: 'float RG', type: T.FloatType, format: T.RGFormat, native: 'rg32float'},
  {name: 'signed R', type: T.IntType, format: T.RedIntegerFormat, native: 'r32sint'},
  {name: 'unsigned RGBA array', array: true, type: T.UnsignedIntType, native: 'rgba32uint'},
]) test(`${profile.name} storage reads preserve source shaders, mip views and both graphics stages`, async t => {
  const f = fixture(t), texture = storage(t, profile); graph(f, texture);
  await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[0], program = descriptor(f), source = sourceShaders(f);
  assert.equal(program.vertex, source.vertexShader); assert.equal(program.fragment, source.fragmentShader);
  assert.match(program.vertex, /texture_storage_2d(?:_array)?<[^>]+, read>/);
  assert.match(program.fragment, /textureLoad\(/); assert.doesNotMatch(program.fragment, /textureSample/);
  const storageLayouts = layouts(draw).filter(entry => entry.storageTexture);
  assert.deepEqual(storageLayouts.map(entry => entry.visibility).sort(), [1, 2]);
  assert.ok(storageLayouts.every(entry => entry.storageTexture.access === 'read-only' &&
    entry.storageTexture.format === profile.native && entry.storageTexture.viewDimension === (profile.array ? '2d-array' : '2d')));
  assert.equal(layouts(draw).filter(entry => entry.sampler || entry.texture).length, 0);
  const [vertex, fragment] = views(draw);
  assert.equal(vertex, fragment); assert.equal(vertex.baseMipLevel, 1); assert.equal(vertex.mipLevelCount, 1);
  assert.equal(vertex.texture.format, profile.native); assert.ok(vertex.texture.usage & 8);
  assert.equal(vertex.texture.depthOrArrayLayers, profile.array ? 2 : 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued texture replacements retain storage identity while compatible source programs are reused', async t => {
  const f = fixture(t), first = storage(t), second = storage(t), node = graph(f, first);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  node.value = second; f.renderer.render(f.scene, f.camera);
  node.value = first; await f.renderer.waitForGPU();
  const rendered = draws(f), programs = f.captures.map(frame => frame.draws[0].binding.nodeProgram);
  assert.equal(rendered.length, 2); assert.equal(programs[0], programs[1]);
  assert.notEqual(views(rendered[0])[0].texture, views(rendered[1])[0].texture);
  assert.equal(views(rendered[0])[0], views(rendered[0])[1]);
  assert.equal(views(rendered[1])[0], views(rendered[1])[1]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('storage mip selection follows source compilation and needsUpdate boundaries', async t => {
  const f = fixture(t), texture = storage(t), node = graph(f, texture, 0);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  node.setMipLevel(2); f.renderer.render(f.scene, f.camera);
  f.material.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  node.setMipLevel(99); await f.renderer.waitForGPU();
  assert.deepEqual(draws(f).map(draw => views(draw)[0].baseMipLevel), [0, 0, 2]);
  const programs = f.captures.map(frame => frame.draws[0].binding.nodeProgram);
  assert.equal(programs[0], programs[1]); assert.notEqual(programs[1], programs[2]);
  assert.equal(views(draws(f)[0])[0].texture, views(draws(f)[2])[0].texture);
});

test('compute producers and graphics storage consumers share one allocation and FIFO order', async t => {
  const f = fixture(t), texture = storage(t); graph(f, texture);
  const output = N.storageTexture(texture).toWriteOnly().setMipLevel(1);
  const compute = N.Fn(() => { N.textureStore(output, N.uvec2(0), N.vec4(.25, .5, .75, 1)); })().compute(1);
  t.after(() => compute.dispose());
  await f.renderer.init(); f.renderer.compute(compute); f.renderer.render(f.scene, f.camera);
  f.renderer.compute(compute); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(f.events.map(event => event.compute ? 'compute' : 'draw'), ['compute', 'draw', 'compute', 'draw']);
  const produced = f.events.filter(event => event.compute).map(event => event.compute[0].resource);
  assert.ok(produced.every(view => view === views(draws(f)[0])[0]));
  assert.equal(views(draws(f)[0])[0], views(draws(f)[1])[0]);
  assert.equal(f.textures.filter(value => value.label === 'f3d-storage-texture').length, 1);
});

test('compilation prepares storage bindings without dispatching or acquiring storage pixels', async t => {
  const f = fixture(t), texture = storage(t); graph(f, texture);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.events.length, 0); assert.equal(f.textures.filter(value => value.label === 'f3d-storage-texture').length, 0);
  await f.renderer.renderAsync(f.scene, f.camera); assert.equal(draws(f).length, 1);
});

test('storage read limits and invalid source stores fail before GPU submission', async t => {
  const f = fixture(t, {maxStorageTexturesPerShaderStage: 0}), texture = storage(t); graph(f, texture);
  await f.renderer.init(); assert.throws(() => f.renderer.render(f.scene, f.camera), /storage.*limits/);
  assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  f.device.limits.maxStorageTexturesPerShaderStage = 4;
  f.material.colorNode = N.Fn(() => { N.textureStore(texture, N.uvec2(0), N.vec4(1)); return N.vec3(1); })();
  assert.throws(() => f.renderer.render(f.scene, f.camera), /read-only texel loads/);
  assert.equal(f.device.submissions.length, 0);
});

test('disposed or reconfigured storage textures cannot satisfy an already queued native read', async t => {
  for (const mutation of [texture => texture.dispose(), texture => { texture.image.width = 8; }]) {
    const f = fixture(t), texture = storage(t); graph(f, texture);
    await f.renderer.init(); f.renderer.render(f.scene, f.camera); mutation(texture);
    await assert.rejects(f.renderer.waitForGPU(), /resized|disposed|reconfigured/);
    assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});
