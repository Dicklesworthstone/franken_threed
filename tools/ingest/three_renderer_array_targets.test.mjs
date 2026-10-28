/** Native source array targets through the public renderer and owned GPU path.
 * These recording-device checks observe attachments, WGSL, bindings and copy
 * regions. They do not simulate rasterized pixels or claim browser parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  const textures = [], reads = [], copies = [], captures = [];
  device.createTexture = desc => {
    const texture = {dimension: '2d', sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
      createView(options = {}) {
        assert.ok((options.baseArrayLayer ?? 0) + (options.arrayLayerCount ?? 1) <= texture.depthOrArrayLayers);
        return {texture, ...options};
      }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  const createBuffer = device.createBuffer;
  device.createBuffer = desc => Object.assign(createBuffer(desc), {mapAsync: async () => {}});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = () => Object.assign(createEncoder(), {
    copyTextureToBuffer(source, destination, size) { reads.push({source, destination, size}); },
    copyTextureToTexture(source, destination, size) { copies.push({source, destination, size}); },
  });
  device.createRenderPipeline = desc => {
    const pipeline = {...desc, getBindGroupLayout: () => ({})}; device.pipelines.push(pipeline); return pipeline;
  };
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial(); material.colorNode = N.vec3(.125, .375, .625);
  scene.add(new T.Mesh(new T.PlaneGeometry(), material));
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  return {renderer, device, scene, camera, material, textures, reads, copies, captures};
}
function target(t, options = {}) {
  const value = new T.RenderTarget(8, 6, {depth: 3, ...options}); t.after(() => value.dispose()); return value;
}
const drawPasses = device => device.passes.filter(pass => pass.draws.length);
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const textureEntries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries).filter(entry => entry.resource?.texture);
const textureLayouts = draw => draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries).filter(entry => entry.texture);


test('queued native array renders retain selected layers and share the default 2D depth attachment', async t => {
  const f = fixture(t), array = target(t); await f.renderer.init();
  for (const layer of [2, 0, 1]) { f.renderer.setRenderTarget(array, layer); f.renderer.render(f.scene, f.camera); }
  f.renderer.setRenderTarget(array, 2);
  await f.renderer.waitForGPU();
  const passes = drawPasses(f.device), views = passes.map(pass => pass.desc.colorAttachments[0].view);
  assert.deepEqual(views.map(view => view.baseArrayLayer), [2, 0, 1]);
  assert.ok(views.every(view => view.dimension === '2d' && view.arrayLayerCount === 1));
  assert.equal(new Set(views.map(view => view.texture)).size, 1);
  assert.equal(views[0].texture.depthOrArrayLayers, 3);
  const depth = passes.map(pass => pass.desc.depthStencilAttachment.view);
  assert.equal(new Set(depth.map(view => view.texture)).size, 1);
  assert.ok(depth.every(view => view.texture.depthOrArrayLayers === 1 && (view.baseArrayLayer ?? 0) === 0));
  assert.equal(f.renderer.getActiveCubeFace(), 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('array layer and authored mip level select independent attachment subresources', async t => {
  const f = fixture(t), array = target(t); array.texture.mipmaps = [{}, {}];
  await f.renderer.init(); f.renderer.setRenderTarget(array, 2, 1);
  await f.renderer.renderAsync(f.scene, f.camera);
  const view = drawPasses(f.device)[0].desc.colorAttachments[0].view;
  assert.deepEqual([view.baseArrayLayer, view.arrayLayerCount, view.baseMipLevel, view.mipLevelCount], [2, 1, 1, 1]);
  assert.equal(f.renderer.getActiveMipmapLevel(), 1);
  assert.equal(view.texture.mipLevelCount, 2);
});

for (const Type of [T.RenderTarget, T.WebGLArrayRenderTarget])
test(`${Type.name} color sampling binds the whole array and preserves queued layer uniforms`, async t => {
  const f = fixture(t), array = Type === T.RenderTarget ? target(t) : new Type(8, 6, 3);
  if (Type !== T.RenderTarget) t.after(() => array.dispose());
  const sourceOwner = array.texture.renderTarget;
  await f.renderer.init(); f.renderer.setRenderTarget(array, 2);
  await f.renderer.renderAsync(f.scene, f.camera);
  const rendered = drawPasses(f.device)[0].desc.colorAttachments[0].view.texture;
  const layer = N.uniform(2, 'int').setName('selectedArrayLayer');
  f.material.colorNode = N.texture(array.texture, N.vec2(.25, .75)).depth(layer).rgb;
  f.renderer.setRenderTarget(null); f.renderer.render(f.scene, f.camera);
  const first = f.captures.at(-1).draws[0].nodeUniforms.map(bytes => bytes.slice());
  layer.value = 0; f.renderer.render(f.scene, f.camera);
  const second = f.captures.at(-1).draws[0].nodeUniforms.map(bytes => bytes.slice());
  layer.value = 1; await f.renderer.waitForGPU();
  assert.notDeepEqual(first, second);
  assert.deepEqual(f.captures.at(-2).draws[0].nodeUniforms, first);
  const draws = nativeDraws(f.device).filter(draw => /texture_2d_array/.test(draw.pipeline.fragment.module.code));
  assert.equal(draws.length, 2);
  for (const draw of draws) {
    assert.match(draw.pipeline.fragment.module.code, /selectedArrayLayer/);
    assert.equal(textureLayouts(draw)[0].texture.viewDimension, '2d-array');
    const view = textureEntries(draw)[0].resource;
    assert.equal(view.texture, rendered); assert.equal(view.dimension, '2d-array');
    assert.deepEqual([view.baseArrayLayer, view.arrayLayerCount], [0, 3]);
  }
  assert.equal(array.texture.renderTarget, sourceOwner, 'sampling preserves the source color owner pointer');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const [arrayDepth, mode] of [[false, 'sample'], [true, 'sample'], [true, 'compare'], ['image', 'compare']])
test(`${arrayDepth || 'shared'} depth ${mode} preserves source target dimension and sampling mode`, async t => {
  const f = fixture(t), array = target(t, {useArrayDepthTexture: arrayDepth === true});
  array.depthTexture = new T.DepthTexture(8, 6);
  if (arrayDepth === 'image') array.depthTexture.image.depth = 3;
  if (mode === 'compare') array.depthTexture.compareFunction = T.LessEqualCompare;
  const source = N.texture(array.depthTexture, N.vec2(.25, .75));
  const sampled = arrayDepth ? source.depth(2) : source;
  f.material.colorNode = N.vec3(mode === 'compare' ? sampled.compare(.5) : sampled.level(0));
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = nativeDraws(f.device)[0], dimension = arrayDepth ? '2d-array' : '2d';
  assert.match(draw.pipeline.fragment.module.code, arrayDepth ? /texture_depth_2d_array/ : /texture_depth_2d\s*;/);
  // Source nearest depth reads lower normalized UVs to sampler-free texel loads.
  assert.match(draw.pipeline.fragment.module.code, mode === 'compare' ? /textureSampleCompare/ : /textureLoad/);
  assert.deepEqual(textureLayouts(draw).map(entry => entry.texture), [{sampleType: 'depth', viewDimension: dimension, multisampled: false}]);
  const view = textureEntries(draw)[0].resource;
  assert.equal(view.dimension, dimension); assert.equal(view.aspect, 'depth-only');
  assert.equal(view.texture.depthOrArrayLayers, arrayDepth ? 3 : 1);
  if (arrayDepth) assert.deepEqual([view.baseArrayLayer, view.arrayLayerCount], [0, 3]);
  const samplers = draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries).filter(entry => entry.sampler);
  assert.deepEqual(samplers.map(entry => entry.sampler.type), mode === 'compare' ? ['comparison'] : []);
});

test('public array copy and source-shaped readback preserve selected layers across queued state edits', async t => {
  const f = fixture(t), source = target(t), destination = target(t);
  await f.renderer.init(); f.renderer.setRenderTarget(source, 2); f.renderer.render(f.scene, f.camera);
  const options = {sourceFace: 2, face: 1};
  const copy = f.renderer.copyRenderTargetPixelsAsync(source, destination, options);
  const read = f.renderer.readRenderTargetPixelsAsync(destination, 0, 0, 2, 2, 0, 1);
  options.sourceFace = 0; options.face = 0; f.renderer.setRenderTarget(source, 0);
  await copy; const output = await read; await f.renderer.waitForGPU();
  assert.equal(output.length, 16, 'readback exposes one selected color layer');
  assert.equal(f.copies.length, 1); assert.equal(f.reads.length, 1);
  assert.deepEqual([f.copies[0].source.origin.z, f.copies[0].destination.origin.z, f.reads[0].source.origin.z], [2, 1, 1]);
  assert.equal(f.reads[0].source.texture, f.copies[0].destination.texture);
  assert.equal(f.reads[0].size.depthOrArrayLayers, 1);
});

test('invalid array layers and unsupported layered output targets fail before native allocation', async t => {
  const f = fixture(t), array = target(t);
  for (const layer of [-1, 3, .5, NaN, Infinity]) {
    assert.throws(() => f.renderer.setRenderTarget(array, layer), error => error.code?.endsWith('_FACE'));
    await assert.rejects(f.renderer.readRenderTargetPixelsAsync(array, 0, 0, 0, 0, 0, layer), error => error.code?.endsWith('_FACE'));
  }
  assert.throws(() => f.renderer.setOutputRenderTarget(array), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(f.textures.length, 0); assert.equal(f.device.buffers.length, 0);
  assert.equal(f.renderer._pending.length, 0);
});

test('a single-layer WebGL array target samples ordinary depth even when array depth is requested', async t => {
  const f = fixture(t), array = new T.WebGLArrayRenderTarget(8, 6, 1, {useArrayDepthTexture: true});
  t.after(() => array.dispose()); array.depthTexture = new T.DepthTexture(8, 6);
  f.material.colorNode = N.vec3(N.texture(array.depthTexture, N.vec2(.25, .75)).level(0));
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = nativeDraws(f.device)[0];
  assert.match(draw.pipeline.fragment.module.code, /texture_depth_2d\s*;/);
  assert.equal(textureLayouts(draw)[0].texture.viewDimension, '2d');
  assert.equal(textureEntries(draw)[0].resource.texture.depthOrArrayLayers, 1);
  assert.equal(array.depthTexture.isArrayTexture, false);
});

test('the same DepthTexture rebuilds from array sampling to ordinary 2D sampling after a layer shrink', async t => {
  const f = fixture(t), array = target(t, {useArrayDepthTexture: true});
  const depth = array.depthTexture = new T.DepthTexture(8, 6);
  f.material.colorNode = N.vec3(N.texture(depth, N.vec2(.25, .75)).depth(2).level(0));
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const previous = nativeDraws(f.device)[0], oldTexture = textureEntries(previous)[0].resource.texture;
  assert.match(previous.pipeline.fragment.module.code, /texture_depth_2d_array/);
  assert.equal(depth.isArrayTexture, true);
  array.setSize(8, 6, 1);
  f.material.colorNode = N.vec3(N.texture(depth, N.vec2(.25, .75)).level(0));
  await f.renderer.renderAsync(f.scene, f.camera);
  const next = nativeDraws(f.device).at(-1), view = textureEntries(next)[0].resource;
  assert.equal(array.depthTexture, depth); assert.equal(depth.isArrayTexture, false);
  assert.match(next.pipeline.fragment.module.code, /texture_depth_2d\s*;/);
  assert.equal(textureLayouts(next)[0].texture.viewDimension, '2d');
  assert.equal(view.dimension, '2d'); assert.equal(view.texture.depthOrArrayLayers, 1);
  assert.notEqual(view.texture, oldTexture); assert.equal(oldTexture.destroyed, true);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});


function recordCompute(device) {
  const passes = [];
  device.createComputePipeline = descriptor => ({...descriptor,
    getBindGroupLayout: index => descriptor.layout.bindGroupLayouts[index]});
  device.createComputePipelineAsync = descriptor => Promise.resolve(device.createComputePipeline(descriptor));
  const create = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = create(), commands = [];
    return {...encoder, beginComputePass() {
      const pass = {draws: [], groups: new Map(), dispatches: []}; passes.push(pass); commands.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; },
        setBindGroup(index, group) { pass.groups.set(index, group); },
        dispatchWorkgroups(...size) { pass.dispatches.push(size); }, end() {}};
    }, finish() { return [...encoder.finish(), ...commands]; }};
  };
  return passes;
}

for (const layer of [2, null])
test(`first native compute depth-array load retains ${layer === null ? 'implicit layer zero' : 'explicit layer two'}`, async t => {
  const f = fixture(t), passes = recordCompute(f.device), array = target(t, {useArrayDepthTexture: true});
  array.depthTexture = new T.DepthTexture(8, 6);
  assert.equal(array.depthTexture.isArrayTexture, false, 'the source constructor has not initialized array depth');
  const output = N.storage(new T.StorageBufferAttribute(1, 1), 'float', 1);
  const sampled = N.textureLoad(array.depthTexture, N.ivec2(0));
  const value = layer === null ? sampled : sampled.depth(layer);
  const kernel = N.Fn(() => { output.element(0).assign(value); })().compute(1);
  await f.renderer.init(); await f.renderer.computeAsync(kernel); await f.renderer.waitForGPU();
  assert.equal(f.captures.length, 0, 'no graphics compilation initializes the depth texture first');
  assert.equal(passes.length, 1); assert.deepEqual(passes[0].dispatches, [[1, 1, 1]]);
  const {pipeline, groups} = passes[0], shader = pipeline.compute.module.code;
  assert.match(shader, /texture_depth_2d_array/);
  assert.match(shader, layer === null ? /textureLoad\([^;]+,\s*0,\s*u32\(/ : /textureLoad\([^;]+,\s*2,\s*u32\(/);
  const layouts = pipeline.layout.bindGroupLayouts.flatMap(group => group.entries).filter(entry => entry.texture);
  assert.deepEqual(layouts.map(entry => entry.texture), [{sampleType: 'depth', viewDimension: '2d-array', multisampled: false}]);
  const view = [...groups.values()].flatMap(group => group.entries).find(entry => entry.resource?.texture).resource;
  assert.deepEqual([view.dimension, view.baseArrayLayer, view.arrayLayerCount, view.aspect], ['2d-array', 0, 3, 'depth-only']);
});
