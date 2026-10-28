/** Data3DTexture graphics through the source node builder and shared upload owner.
 * A recording device verifies WGSL, bindings, captured bytes and source versions;
 * these checks do not execute WGSL or claim browser pixel parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t, {features = [], maxDimension3D = 256} = {}) {
  const device = geometryDevice(), textures = [], writes = [], captures = [];
  device.features = new Set(features);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: maxDimension3D,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => {
    const value = {dimension: '2d', sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; },
      destroy() { this.destroyed = true; }};
    textures.push(value); return value;
  };
  device.createSampler = desc => ({...desc});
  device.queue.writeTexture = (destination, data, layout, size) => {
    const bytes = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
    writes.push({destination, bytes, layout: {...layout}, size: [...size]});
  };
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false;
  const scene = new T.Scene(); scene.add(object);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  return {renderer, device, material, object, scene, camera, textures, writes, captures};
}
function volume(t, {width = 2, height = 2, depth = 2, channels = 4, Type = Uint8Array,
  type = T.UnsignedByteType, format = [null, T.RedFormat, T.RGFormat, null, T.RGBAFormat][channels],
  filter = T.NearestFilter, data = new Type(width * height * depth * channels).fill(1), upload = true} = {}) {
  const texture = new T.Data3DTexture(data, width, height, depth);
  Object.assign(texture, {type, format, minFilter: filter, magFilter: filter});
  if (upload) texture.needsUpdate = true;
  t.after(() => texture.dispose()); return texture;
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const volumeView = draw => entries(draw).find(entry => entry.resource?.dimension === '3d').resource;
const layouts = draw => draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries);
const uploaded = (f, texture) => f.writes.filter(write => write.destination.texture === texture);
function assertSource(f, descriptor) {
  const source = new T.WGSLNodeBuilder(f.object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
}

// The pinned builder caches float32Filterable globally on its first build.
// Exercise the enabled device first; nearest reads and admission refusals below
// remain valid on the disabled device without changing retained source code.
for (const profile of [
  {name: 'float RGBA linear with feature', Type: Float32Array, type: T.FloatType,
    filter: T.LinearFilter, formatName: 'rgba32float', features: ['float32-filterable']},
  {name: 'byte R nearest', channels: 1, formatName: 'r8unorm'},
  {name: 'byte RGBA linear', filter: T.LinearFilter, formatName: 'rgba8unorm'},
  {name: 'half RG linear', channels: 2, Type: Uint16Array, type: T.HalfFloatType,
    filter: T.LinearFilter, formatName: 'rg16float'},
  {name: 'float R nearest', channels: 1, Type: Float32Array, type: T.FloatType,
    formatName: 'r32float', sampleType: 'unfilterable-float'},
]) test(`ordinary Data3D ${profile.name} retains source shader and shared 3D residency`, async t => {
  const f = fixture(t, profile), texture = volume(t, profile);
  f.material.colorNode = N.texture3D(texture, N.vec3(.25, .5, .75)).rgb;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[0], view = volumeView(draw), descriptor = f.captures[0].draws[0].binding.nodeProgram;
  assertSource(f, descriptor);
  assert.match(draw.pipeline.fragment.module.code, /texture_3d<f32>/);
  assert.match(draw.pipeline.fragment.module.code, profile.filter === T.LinearFilter ? /textureSample\(/ : /textureLoad\(/);
  assert.deepEqual(layouts(draw).filter(entry => entry.texture).map(entry => entry.texture),
    [{viewDimension: '3d', sampleType: profile.sampleType ?? 'float', multisampled: false}]);
  assert.equal(layouts(draw).filter(entry => entry.sampler).length, Number(profile.filter === T.LinearFilter));
  assert.deepEqual([view.texture.dimension, view.texture.width, view.texture.height, view.texture.depthOrArrayLayers], ['3d', 2, 2, 2]);
  assert.equal(view.texture.format, profile.formatName);
  const [write] = uploaded(f, view.texture);
  assert.equal(uploaded(f, view.texture).length, 1); assert.deepEqual(write.size, [2, 2, 2]);
  assert.deepEqual(write.bytes, new Uint8Array(texture.image.data.buffer));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const signed of [false, true])
test(`texture3DLoad preserves ${signed ? 'signed' : 'unsigned'} 32-bit source texels without sampler conversion`, async t => {
  const f = fixture(t), Type = signed ? Int32Array : Uint32Array;
  const data = new Type(signed ? [-2147483648, 2147483647, -16777217, 16777217, -1, 0, 1, 2]
    : [0xffffffff, 0x80000000, 0x01000001, 0x7fffffff, 0, 1, 2, 3]);
  const texture = volume(t, {channels: 1, Type, data, type: signed ? T.IntType : T.UnsignedIntType, format: T.RedIntegerFormat});
  const sampled = N.texture3DLoad(texture, N.ivec3(1, 0, 1));
  f.material.colorNode = N.vec3(sampled.x.equal(signed ? N.int(-1) : N.uint(0xffffffff)).select(1, 0));
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[0], view = volumeView(draw), sampleType = signed ? 'sint' : 'uint';
  assertSource(f, f.captures[0].draws[0].binding.nodeProgram);
  assert.match(draw.pipeline.fragment.module.code, signed ? /texture_3d<i32>/ : /texture_3d<u32>/);
  assert.match(draw.pipeline.fragment.module.code, /textureLoad\(/);
  assert.doesNotMatch(draw.pipeline.fragment.module.code, /textureSample/);
  assert.deepEqual(layouts(draw).filter(entry => entry.texture).map(entry => entry.texture),
    [{viewDimension: '3d', sampleType, multisampled: false}]);
  assert.equal(layouts(draw).filter(entry => entry.sampler).length, 0);
  assert.equal(view.texture.format, `r32${sampleType}`);
  assert.deepEqual(uploaded(f, view.texture)[0].bytes, new Uint8Array(data.buffer));
});

test('queued volume consumers retain acknowledged upload versions and immutable source bytes', async t => {
  const f = fixture(t), texture = volume(t), updates = [], image = texture.image, source = texture.source;
  texture.onUpdate = value => updates.push(value);
  const sampled = N.texture3D(texture, N.vec3(.5)); f.material.colorNode = sampled.rgb;
  await f.renderer.init();
  texture.image.data.fill(17); f.renderer.render(f.scene, f.camera);
  texture.image.data.fill(53); f.renderer.render(f.scene, f.camera);
  texture.image.data.fill(91); texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  texture.image.data.fill(127);
  await f.renderer.waitForGPU();
  const rendered = draws(f), native = rendered.map(draw => volumeView(draw).texture);
  assert.equal(rendered.length, 3); assert.equal(native[0], native[1]); assert.notEqual(native[1], native[2]);
  assert.deepEqual(native.map(value => [...uploaded(f, value)[0].bytes]),
    [Array(32).fill(17), Array(32).fill(17), Array(32).fill(91)]);
  assert.equal(f.writes.filter(write => write.destination.texture.dimension === '3d').length, 2);
  assert.deepEqual(updates, [texture, texture]);
  assert.equal(texture.image, image); assert.equal(texture.source, source);
  assert.ok(f.captures.every(frame => frame.draws[0].binding.nodeProgram === f.captures[0].draws[0].binding.nodeProgram));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('padded volume rows and nonzero typed-array offsets reach the shared uploader intact', async t => {
  const f = fixture(t), data = new Uint8Array(new ArrayBuffer(32), 5, 15);
  data.set([1, 2, 3, 99, 4, 5, 6, 99, 7, 8, 9, 99, 10, 11, 12]);
  const texture = volume(t, {width: 3, height: 2, depth: 2, channels: 1, data}); texture.unpackAlignment = 4;
  f.material.colorNode = N.texture3DLoad(texture, N.ivec3(2, 1, 1)).rgb;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); const expected = data.slice(); data.fill(255);
  await f.renderer.waitForGPU();
  const [write] = uploaded(f, volumeView(draws(f)[0]).texture);
  assert.deepEqual(write.bytes, expected); assert.deepEqual(write.layout, {bytesPerRow: 4, rowsPerImage: 2});
  assert.deepEqual(write.size, [3, 2, 2]);
});

test('compatible volume replacements retain each queued texture while sharing the source program', async t => {
  const f = fixture(t), a = volume(t), b = volume(t, {width: 3, depth: 4});
  a.image.data.fill(13); b.image.data.fill(37);
  const sampled = N.texture3D(a, N.vec3(.5)); f.material.colorNode = sampled.rgb;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  sampled.value = b; f.renderer.render(f.scene, f.camera); b.image.data.fill(99);
  await f.renderer.waitForGPU();
  const rendered = draws(f), first = volumeView(rendered[0]).texture, second = volumeView(rendered[1]).texture;
  assert.notEqual(first, second); assert.deepEqual([first.width, first.depthOrArrayLayers, second.width, second.depthOrArrayLayers], [2, 2, 3, 4]);
  assert.ok(uploaded(f, first)[0].bytes.every(value => value === 13));
  assert.ok(uploaded(f, second)[0].bytes.every(value => value === 37));
  assert.equal(f.captures[0].draws[0].binding.nodeProgram, f.captures[1].draws[0].binding.nodeProgram);
});

test('linear volume wrapR changes retain each queued sampler W mode', async t => {
  const f = fixture(t), texture = volume(t, {filter: T.LinearFilter});
  f.material.colorNode = N.texture3D(texture, N.vec3(.5, .5, 1.5)).rgb;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  texture.wrapR = T.RepeatWrapping; f.renderer.render(f.scene, f.camera);
  texture.wrapR = T.MirroredRepeatWrapping;
  await f.renderer.waitForGPU();
  assert.deepEqual(draws(f).map(draw => entries(draw).find(entry => entry.resource?.addressModeW).resource.addressModeW), ['clamp-to-edge', 'repeat']);
  assert.notEqual(f.captures[0].draws[0].binding.nodeProgram, f.captures[1].draws[0].binding.nodeProgram);
});

test('changing a live 2D texture node to Data3D invalidates the cached family before submission', async t => {
  const f = fixture(t), image = new T.DataTexture(new Uint8Array(16), 2, 2); image.needsUpdate = true;
  t.after(() => image.dispose()); const sampled = N.texture(image); f.material.colorNode = sampled.rgb;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  sampled.value = volume(t); const allocations = f.textures.length;
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(f.textures.length, allocations); assert.equal(draws(f).length, 1);
});

for (const kind of ['float filtering without feature', 'sampled integer', '16-bit integer', 'forged volume', 'wrong node family', '3D device limit'])
test(`unsupported volume profile refuses ${kind} before GPU texture allocation`, async t => {
  const f = fixture(t, {maxDimension3D: 4}); let texture = volume(t), sampled, expected = 'THREE_NODE_PROGRAM_TEXTURE';
  if (kind === 'float filtering without feature') texture = volume(t, {Type: Float32Array, type: T.FloatType, filter: T.LinearFilter});
  if (kind === 'sampled integer') texture = volume(t, {Type: Uint32Array, type: T.UnsignedIntType, format: T.RGBAIntegerFormat});
  if (kind === '16-bit integer') texture = volume(t, {Type: Uint16Array, type: T.UnsignedShortType, format: T.RGBAIntegerFormat});
  if (kind === 'forged volume') {
    texture = new T.DataTexture(new Uint8Array(32), 2, 2); texture.isData3DTexture = true; texture.image.depth = 2;
    texture.needsUpdate = true; t.after(() => texture.dispose());
  }
  if (kind === '3D device limit') { texture = volume(t, {depth: 5}); expected = 'THREE_TEXTURE_VALUE'; }
  sampled = kind === 'wrong node family' ? N.texture(texture) : kind === '16-bit integer'
    ? N.texture3DLoad(texture, N.ivec3(0)) : N.texture3D(texture, N.vec3(.5));
  f.material.colorNode = sampled.rgb;
  await f.renderer.init(); const allocations = f.textures.length;
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: expected});
  assert.equal(f.textures.length, allocations); assert.equal(draws(f).length, 0);
});
