/** Pinned source WGSL and recording-device ownership checks, not rasterization
 * or hardware validation. Integer output and clear words never pass through f32. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t) {
  const device = geometryDevice(), textures = [], captures = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 64,
    maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => {
    const pipeline = {...descriptor, getBindGroupLayout: () => ({})}; device.pipelines.push(pipeline); return pipeline;
  };
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false; scene.add(object);
  t.after(() => {object.geometry.dispose(); material.dispose();});
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  return {device, textures, captures, renderer, scene, camera, material, object};
}
function target(t, signed = false, format = T.RGBAIntegerFormat, options = {}) {
  const value = new T.RenderTarget(8, 6, {type: signed ? T.IntType : T.UnsignedIntType,
    format, colorSpace: T.NoColorSpace, depthBuffer: false, ...options});
  t.after(() => value.dispose()); return value;
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const layouts = draw => draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries);
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
function assertSource(f, descriptor) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer); builder.scene = f.scene; builder.camera = f.camera; builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
}

for (const signed of [false, true]) for (const [format, prefix] of [
  [T.RedIntegerFormat, 'r'], [T.RGIntegerFormat, 'rg'], [T.RGBAIntegerFormat, 'rgba'],
]) test(`${prefix}32${signed ? 'sint' : 'uint'} retains source output and a later sampler-free load of the same image`, async t => {
  const f = fixture(t), output = target(t, signed, format), suffix = signed ? 'sint' : 'uint';
  f.material.fragmentNode = signed ? N.ivec4(-2147483648, 2147483647, -16777217, 16777217)
    : N.uvec4(16777217, 4294967295, 2147483648, 1);
  f.renderer.setRenderTarget(output); await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const first = draws(f)[0], descriptor = f.captures[0].draws[0].binding.nodeProgram;
  assertSource(f, descriptor);
  const scalar = signed ? 'i32' : 'u32', nodeType = prefix === 'r' ? signed ? 'int' : 'uint'
    : `${signed ? 'i' : 'u'}vec${prefix === 'rg' ? 2 : 4}`;
  assert.deepEqual(descriptor.colorTypes, [nodeType]);
  assert.ok(descriptor.fragment.includes(`color: ${prefix === 'r' ? scalar : `vec${prefix === 'rg' ? 2 : 4}<${scalar}>`}`));
  assert.match(descriptor.fragment, signed ? /-2147483648/ : /16777217u/);
  assert.equal(first.pipeline.fragment.targets[0].format, `${prefix}32${suffix}`);
  assert.equal(first.pipeline.fragment.targets[0].blend, undefined);
  assert.equal(first.pipeline.multisample.count, 1);
  assert.ok(f.device.pipelines.filter(p => p.fragment?.targets[0]?.format === `${prefix}32${suffix}`)
    .every(p => p.vertex.entryPoint === 'main'), 'no speculative float shader targets an integer image');
  const image = f.device.passes.find(pass => pass.draws.length).desc.colorAttachments[0].view.texture;
  const input = N.textureLoad(output.texture, N.ivec2(0));
  f.material.fragmentNode = N.vec4(input.x.equal(signed ? N.int(-2147483648) : N.uint(16777217)).select(1, 0));
  f.material.needsUpdate = true; f.renderer.setRenderTarget(null);
  await f.renderer.renderAsync(f.scene, f.camera);
  const second = draws(f)[1];
  assert.match(second.pipeline.fragment.module.code, signed ? /texture_2d<i32>/ : /texture_2d<u32>/);
  assert.match(second.pipeline.fragment.module.code, /textureLoad\(/);
  assert.deepEqual(layouts(second).filter(entry => entry.texture).map(entry => entry.texture),
    [{sampleType: suffix, viewDimension: '2d', multisampled: false}]);
  assert.equal(layouts(second).filter(entry => entry.sampler).length, 0);
  assert.equal(entries(second).find(entry => entry.resource?.texture === image)?.resource.texture, image);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('mixed float, unsigned and signed MRT outputs keep each source output type', async t => {
  const f = fixture(t), output = target(t, false, T.RGBAFormat, {count: 3, type: T.HalfFloatType});
  Object.assign(output.textures[1], {type: T.UnsignedIntType, format: T.RGBAIntegerFormat});
  Object.assign(output.textures[2], {type: T.IntType, format: T.RGBAIntegerFormat});
  f.material.fragmentNode = N.outputStruct(N.vec4(.125, .25, .5, 1), N.uvec4(4294967295), N.ivec4(-16777217));
  f.renderer.setRenderTarget(output); await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const descriptor = f.captures[0].draws[0].binding.nodeProgram;
  assertSource(f, descriptor); assert.deepEqual(descriptor.colorTypes, ['vec4', 'uvec4', 'ivec4']);
  assert.deepEqual(draws(f)[0].pipeline.fragment.targets.map(entry => entry.format), ['rgba16float', 'rgba32uint', 'rgba32sint']);
});

for (const kind of ['float output', 'opposite signedness', 'insufficient components', 'blending', 'sampled input'])
test(`invalid integer target ${kind} refuses before a native pipeline or draw submission`, async t => {
  const f = fixture(t), output = target(t);
  f.renderer.setRenderTarget(output); f.material.fragmentNode = N.uvec4(1);
  if (kind === 'float output') f.material.fragmentNode = N.outputStruct(N.vec4(1));
  if (kind === 'opposite signedness') f.material.fragmentNode = N.outputStruct(N.ivec4(1));
  if (kind === 'insufficient components') f.material.fragmentNode = N.outputStruct(N.uint(1));
  if (kind === 'blending') f.material.transparent = true;
  if (kind === 'sampled input') {
    f.material.fragmentNode = N.texture(output.texture); f.renderer.setRenderTarget(null);
  }
  await f.renderer.init();
  const pipelines = f.device.pipelines.length, submissions = f.device.submissions.length;
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), kind === 'sampled input'
    ? {code: 'THREE_NODE_PROGRAM_TEXTURE'} : {code: 'ANIMATION_RENDER_OPTIONS'});
  assert.equal(f.device.pipelines.length, pipelines); assert.equal(f.device.submissions.length, submissions);
});

test('integer cube texel reads refuse before building an invalid WGSL cube load', async t => {
  const f = fixture(t), output = new T.CubeRenderTarget(4, {type: T.UnsignedIntType, format: T.RGBAIntegerFormat});
  t.after(() => output.dispose());
  f.material.fragmentNode = N.cubeTexture(output.texture, N.vec3(1, 0, 0)).setSampler(false);
  await f.renderer.init(); const before = [f.device.pipelines.length, f.device.textures?.length ?? f.textures.length];
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.deepEqual([f.device.pipelines.length, f.textures.length], before);
});

test('fixed floating-point meshes refuse integer targets before mesh allocation', async t => {
  const f = fixture(t), owner = await createGpuAnimationRenderer(f.device, {format: 'rgba32uint', depthFormat: null});
  t.after(() => owner.dispose()); const before = [f.device.buffers.length, f.device.pipelines.length];
  await assert.rejects(owner.addMesh({}), /typed source fragment program/);
  assert.deepEqual([f.device.buffers.length, f.device.pipelines.length], before);
});

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
test('scissored mixed clears preserve all 32 bits and WebGPU integer truncation', async t => {
  const f = fixture(t), formats = ['rgba32uint', 'rgba32sint', 'rgba16float'];
  const owner = await createGpuAnimationRenderer(f.device, {format: formats[0], colorFormats: formats, depthFormat: null});
  t.after(() => owner.dispose());
  const views = [{}, {}, {}], unsigned = [4294967295, 16777217, 2147483648, 4294967295.75],
    signed = [-2147483648, -16777217, 2147483647.75, -1.75], floats = [-.125, .25, .5, 1];
  owner.render({colorView: views[0], colorViews: views, loadOp: 'clear', clearColor: unsigned,
    clearColors: [unsigned, signed, floats], viewProjection: identity(), draws: [], scissor: [1, 1, 2, 3]});
  await owner.whenIdle();
  const clear = f.device.snapshots.flat().find(draw => draw.pipeline.label.endsWith('/scissor-clear'));
  assert.match(clear.pipeline.fragment.module.code, /@location\(0\) color0: vec4<u32>/);
  assert.match(clear.pipeline.fragment.module.code, /@location\(1\) color1: vec4<i32>/);
  assert.match(clear.pipeline.fragment.module.code, /@location\(2\) color2: vec4<f32>/);
  const buffer = entries(clear)[0].resource.buffer, saved = clear.contents.get(buffer);
  assert.deepEqual([...new Uint32Array(saved.buffer, 0, 4)], unsigned.map(Math.trunc));
  assert.deepEqual([...new Int32Array(saved.buffer, 16, 4)], signed.map(Math.trunc));
  assert.deepEqual([...new Float32Array(saved.buffer, 32, 4)], floats);
});

for (const signed of [false, true]) test(`integer ${signed ? 'signed' : 'unsigned'} clears reject range overflow before queue effects`, async t => {
  const f = fixture(t), format = signed ? 'rgba32sint' : 'rgba32uint';
  const owner = await createGpuAnimationRenderer(f.device, {format, depthFormat: null}); t.after(() => owner.dispose());
  for (const value of signed ? [-2147483649, 2147483648] : [-1, 4294967296]) {
    const before = [f.device.writes.length, f.device.submissions.length, f.device.pipelines.length];
    assert.throws(() => owner.render({colorView: {}, viewProjection: identity(), draws: [],
      loadOp: 'clear', clearColor: [value, 0, 0, 1], scissor: [0, 0, 1, 1]}), {code: 'ANIMATION_RENDER_VALUE'});
    assert.deepEqual([f.device.writes.length, f.device.submissions.length, f.device.pipelines.length], before);
  }
});

for (const dimension of ['2d-array', '3d']) for (const signed of [false, true])
test(`${dimension}: rendered ${signed ? 'signed' : 'unsigned'} integer targets feed native typed layer loads`, async t => {
  const f = fixture(t), options = {type: signed ? T.IntType : T.UnsignedIntType,
    format: T.RGBAIntegerFormat, colorSpace: T.NoColorSpace, depthBuffer: false};
  const output = dimension === '3d' ? new T.RenderTarget3D(8, 6, 4, options)
    : new T.RenderTarget(8, 6, {...options, depth: 4});
  t.after(() => output.dispose());
  f.material.fragmentNode = signed ? N.ivec4(-16777217, -2147483648, 2147483647, 1)
    : N.uvec4(16777217, 4294967295, 2147483648, 1);
  f.renderer.setRenderTarget(output, 2); await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assertSource(f, f.captures[0].draws[0].binding.nodeProgram);
  const attachment = f.device.passes.find(pass => pass.draws.length).desc.colorAttachments[0];
  if (dimension === '3d') assert.equal(attachment.depthSlice, 2);
  else assert.equal(attachment.view.baseArrayLayer, 2);
  const image = attachment.view.texture;
  const value = dimension === '3d' ? N.texture3DLoad(output.texture, N.ivec3(1, 0, 2))
    : N.textureLoad(output.texture, N.ivec2(1, 0)).depth(N.int(2));
  f.material.fragmentNode = N.vec4(value.x.equal(signed ? N.int(-16777217) : N.uint(16777217)).select(1, 0));
  f.material.needsUpdate = true; f.renderer.setRenderTarget(null);
  await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[1], scalar = signed ? 'i32' : 'u32';
  assert.ok(draw.pipeline.fragment.module.code.includes(`texture_${dimension.replace('-', '_')}<${scalar}>`));
  assert.match(draw.pipeline.fragment.module.code, /textureLoad\(/);
  assert.deepEqual(layouts(draw).filter(entry => entry.texture).map(entry => entry.texture),
    [{sampleType: signed ? 'sint' : 'uint', viewDimension: dimension, multisampled: false}]);
  assert.equal(layouts(draw).filter(entry => entry.sampler).length, 0);
  assert.equal(entries(draw).find(entry => entry.resource?.texture === image)?.resource.texture, image);
  assertSource(f, f.captures[1].draws[0].binding.nodeProgram);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
