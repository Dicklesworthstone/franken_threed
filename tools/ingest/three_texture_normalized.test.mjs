/** Normalized source data through the owned texture and renderer paths.
 * Recording-device checks preserve bytes, source WGSL and queue histories;
 * they do not execute shaders or establish browser pixel parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceTextureUtils, {getFormat} from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUTextureUtils.js';
import {createGpuThreeTextures} from './three_textures.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {device as textureDevice} from './tests/volume_texture_test_fixture.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const feature = 'texture-formats-tier1', features = [feature], N = T.TSL;
const profiles = [
  {name: 'snorm8', type: T.ByteType, Kind: Int8Array, suffix: '8snorm', values: [-128, -1, 0, 127]},
  {name: 'snorm16', type: T.ShortType, Kind: Int16Array, suffix: '16snorm', values: [-32768, -1, 0, 32767]},
  {name: 'unorm16', type: T.UnsignedShortType, Kind: Uint16Array, suffix: '16unorm', values: [0, 1, 32768, 65535]},
];
const bytes = data => new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
function input(profile, family = '2d', channels = 1, width = 2, height = 2) {
  const depth = ['array', 'volume'].includes(family) ? 2 : 1;
  const data = new profile.Kind(width * height * depth * channels + 2).subarray(1, -1);
  data.set(Array.from({length: data.length}, (_, i) => profile.values[i % profile.values.length]));
  const format = channels === 1 ? T.RedFormat : channels === 2 ? T.RGFormat : T.RGBAFormat;
  const texture = family === 'array' ? new T.DataArrayTexture(data, width, height, depth)
    : family === 'volume' ? new T.Data3DTexture(data, width, height, depth)
      : new T.DataTexture(data, width, height, format, profile.type);
  Object.assign(texture, {format, type: profile.type, normalized: profile.Kind.BYTES_PER_ELEMENT === 2,
    minFilter: T.NearestFilter, magFilter: T.NearestFilter, unpackAlignment: 1});
  texture.needsUpdate = true; return texture;
}
function sourceUploads(texture, format) {
  const writes = [];
  const utils = new SourceTextureUtils({device: {queue: {writeTexture(destination, data, layout, size) {
    const count = size.height * (size.depthOrArrayLayers ?? 1) * layout.bytesPerRow;
    writes.push({layer: destination.origin.z, bytes: bytes(data).slice(layout.offset, layout.offset + count),
      bytesPerRow: layout.bytesPerRow});
  }}}, get: () => ({texture: {}, textureDescriptorGPU: {format}})});
  utils.updateTexture(texture, {image: texture.image}); return writes;
}

for (const profile of profiles) for (const channels of [1, 2, 4]) for (const family of ['2d', 'array', 'volume'])
test(`${profile.name} ${channels}-channel ${family} storage keeps exact source typed bytes and sample layout`, async t => {
  const device = textureDevice({features}), pool = createGpuThreeTextures(device, {three: T, nativeMedia: true});
  const texture = input(profile, family, channels), image = texture.image, before = bytes(image.data).slice();
  t.after(() => { pool.dispose(); texture.dispose(); });
  const expectedFormat = (channels === 1 ? 'r' : channels === 2 ? 'rg' : 'rgba') + profile.suffix;
  assert.equal(getFormat(texture, device), expectedFormat);
  const source = sourceUploads(texture, expectedFormat);
  assert.equal(source.length, family === '2d' ? 1 : texture.image.depth, 'source uploads every expected slice');
  pool.prepare([texture]); await pool.whenIdle();
  const gpu = pool.nativeTexture(texture), binding = pool.binding(texture);
  assert.equal(gpu.desc.format, expectedFormat);
  assert.equal(binding.sampleType, profile.Kind.BYTES_PER_ELEMENT === 2 ? 'unfilterable-float' : 'float');
  assert.equal(binding.view.dimension, family === 'volume' ? '3d' : family === 'array' ? '2d-array' : '2d');
  assert.deepEqual(gpu.levels[0], before); assert.equal(texture.image, image);
  assert.equal(pool.diagnostics.textureBytes, before.byteLength);
  for (const write of source) {
    const offset = write.layer * texture.image.width * texture.image.height * channels * profile.Kind.BYTES_PER_ELEMENT;
    assert.deepEqual(gpu.levels[0].subarray(offset, offset + write.bytes.length), write.bytes);
    assert.equal(device.writes[0].layout.bytesPerRow, write.bytesPerRow);
  }
  image.data.fill(1); pool.update([texture]); assert.deepEqual(gpu.levels[0], before);
  texture.needsUpdate = true; pool.update([texture]);
  assert.deepEqual(gpu.levels[0], bytes(image.data)); assert.equal(pool.binding(texture).view, binding.view);
});

test('normalized formats retain capability, typed-storage and source mip boundaries before allocation', t => {
  for (const profile of profiles) {
    const device = textureDevice(), pool = createGpuThreeTextures(device, {three: T, nativeMedia: true});
    const texture = input(profile); t.after(() => { pool.dispose(); texture.dispose(); });
    if (profile.Kind.BYTES_PER_ELEMENT === 2)
      assert.match(getFormat(texture, device), /16[su]int$/, 'source without the feature selects integer storage, not normalized samples');
    assert.throws(() => pool.prepare([texture]), /texture-formats-tier1/);
    assert.equal(device.textures.length, 0); assert.equal(device.writes.length, 0);
  }
  const device = textureDevice({features}), pool = createGpuThreeTextures(device, {three: T, nativeMedia: true});
  t.after(() => pool.dispose());
  for (const profile of profiles) for (const mutate of [
    image => { image.image.data = new Float32Array(image.image.data.length); },
    image => { image.format = T.RGBAFormat; image.colorSpace = T.SRGBColorSpace; },
  ]) {
    const image = input(profile); mutate(image); t.after(() => image.dispose());
    assert.throws(() => pool.prepare([image])); assert.equal(device.textures.length, 0);
  }
  for (const profile of profiles.slice(1)) {
    const texture = input(profile); t.after(() => texture.dispose());
    texture.normalized = false;
    assert.match(getFormat(texture, device), /16[su]int$/, 'source fallback has an integer sample type');
    assert.throws(() => pool.prepare([texture]), /normalized=true/);
    texture.normalized = true; texture.generateMipmaps = true;
    assert.throws(() => pool.prepare([texture]), /authored source levels/);
  }
  assert.equal(device.textures.length, 0); assert.equal(device.writes.length, 0);
});

test('normalized16 authored mip bytes stay unfilterable while native public filter controls remain intact', t => {
  for (const profile of profiles.slice(1)) {
    const device = textureDevice({features}), pool = createGpuThreeTextures(device, {three: T, nativeMedia: true});
    const texture = input(profile); t.after(() => { pool.dispose(); texture.dispose(); });
    texture.minFilter = T.LinearMipmapLinearFilter; texture.magFilter = T.LinearFilter;
    texture.mipmaps = [texture.image, {width: 1, height: 1, data: new profile.Kind([profile.values[0]])}];
    pool.prepare([texture]); const gpu = pool.nativeTexture(texture), binding = pool.binding(texture);
    assert.deepEqual(gpu.levels[1], bytes(texture.mipmaps[1].data)); assert.equal(device.passes.length, 0);
    assert.equal(binding.sampleType, 'unfilterable-float'); assert.equal(binding.filtering, false);
    assert.equal(texture.minFilter, T.LinearMipmapLinearFilter); assert.equal(texture.magFilter, T.LinearFilter);
    assert.throws(() => pool.writeTexture(texture, () => assert.fail('must not write'), {generateMipmaps: true}), /filterable/);
  }
});

test('signed8 cube faces and generated mips retain source formats on tier1 devices', t => {
  const device = textureDevice({features}), pool = createGpuThreeTextures(device, {three: T, nativeMedia: true, nativeCube: true});
  const faces = Array.from({length: 6}, () => input(profiles[0]));
  const cube = new T.CubeTexture(faces);
  Object.assign(cube, {format: T.RedFormat, type: T.ByteType, generateMipmaps: false,
    minFilter: T.LinearFilter, magFilter: T.LinearFilter, unpackAlignment: 1}); cube.needsUpdate = true;
  const texture = input(profiles[0]); texture.generateMipmaps = true; texture.minFilter = T.LinearMipmapLinearFilter;
  t.after(() => { pool.dispose(); cube.dispose(); texture.dispose(); for (const face of faces) face.dispose(); });
  pool.prepare([cube, texture]);
  const gpu = pool.nativeTexture(cube), source = sourceUploads(cube, getFormat(cube, device));
  assert.equal(source.length, 6);
  assert.equal(pool.binding(cube).view.dimension, 'cube'); assert.equal(gpu.desc.format, 'r8snorm');
  for (const write of source) assert.deepEqual(gpu.levels[0].subarray(write.layer * 4, write.layer * 4 + 4), write.bytes);
  assert.equal(device.passes.length, 1); assert.equal(device.pipelines[0].fragment.targets[0].format, 'r8snorm');
  assert.equal(pool.nativeTexture(texture).desc.usage & 16, 16);
});

function rendererFixture(t) {
  const device = geometryDevice(), writes = [], captures = []; device.features = new Set(features);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination, bytes: bytes(data).slice(), layout: {...layout}, size: [...size]});
  let configured;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [16, 16], format: configured.format, usage: 16})})};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); scene.add(object); camera.position.z = 4;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) { const value = capture.call(this, request); if (value) captures.push(value); return value; };
  t.after(() => { renderer.dispose(); object.geometry.dispose(); material.dispose(); });
  return {renderer, device, material, object, scene, camera, writes, captures};
}
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries).filter(entry =>
  /(?:8snorm|16snorm|16unorm)$/.test(entry.resource?.texture?.format)).map(entry => entry.resource);
const draws = f => f.device.snapshots.flat().filter(draw => views(draw).length);

for (const profile of profiles) for (const family of ['2d', 'array', 'volume'])
test(`${profile.name} ${family} public rendering preserves source WGSL and queued image versions`, async t => {
  const f = rendererFixture(t), texture = input(profile, family), original = bytes(texture.image.data).slice();
  t.after(() => texture.dispose());
  const node = family === 'volume' ? N.texture3D(texture, N.vec3(.25, .5, .75))
    : family === 'array' ? N.texture(texture, N.vec2(.25, .75)).depth(1) : N.texture(texture, N.vec2(.25, .75));
  f.material.colorNode = node.rgb;
  if (family === '2d') texture.minFilter = texture.magFilter = T.LinearFilter;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  texture.image.data.fill(profile.values.at(-1)); texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  const requested = bytes(texture.image.data).slice(); texture.image.data.fill(1); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const rendered = draws(f), gpu = rendered.map(draw => views(draw)[0].texture);
  assert.equal(rendered.length, 3); assert.notEqual(gpu[0], gpu[1]); assert.equal(gpu[1], gpu[2]);
  assert.deepEqual(f.writes.find(write => write.destination.texture === gpu[0]).bytes, original);
  assert.deepEqual(f.writes.find(write => write.destination.texture === gpu[1]).bytes, requested);
  const descriptor = f.captures[0].draws[0].binding.nodeProgram;
  const source = new T.WGSLNodeBuilder(f.object, f.renderer); source.scene = f.scene; source.camera = f.camera; source.build();
  assert.equal(descriptor.vertex, source.vertexShader); assert.equal(descriptor.fragment, source.fragmentShader);
  const layouts = rendered[0].pipeline.layout.bindGroupLayouts.flatMap(group => group.entries);
  const type = profile.Kind.BYTES_PER_ELEMENT === 2 ? 'unfilterable-float' : 'float';
  assert.deepEqual(layouts.filter(entry => entry.texture).map(entry => entry.texture.sampleType), [type]);
  if (type === 'unfilterable-float') assert.equal(layouts.filter(entry => entry.sampler).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const profile of profiles.slice(1))
test(`${profile.name} authored mip levels retain source software filtering without a sampler`, async t => {
  const f = rendererFixture(t), texture = input(profile); t.after(() => texture.dispose());
  texture.minFilter = T.LinearMipmapLinearFilter; texture.magFilter = T.LinearFilter;
  texture.mipmaps = [texture.image, {width: 1, height: 1, data: new profile.Kind([profile.values.at(-1)])}];
  f.material.colorNode = N.texture(texture, N.vec2(.25, .75)).level(1).rgb;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const descriptor = f.captures[0].draws[0].binding.nodeProgram;
  const source = new T.WGSLNodeBuilder(f.object, f.renderer); source.scene = f.scene; source.camera = f.camera; source.build();
  assert.equal(descriptor.fragment, source.fragmentShader); assert.match(descriptor.fragment, /tsl_biquadraticTexture/);
  const draw = draws(f)[0], gpu = views(draw)[0].texture;
  assert.deepEqual(f.writes.filter(write => write.destination.texture === gpu).map(write => write.bytes), texture.mipmaps.map(mip => bytes(mip.data)));
  assert.equal(draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries).filter(entry => entry.sampler).length, 0);
});

test('normalized16 source-invalid sampler paths refuse before accepting a native draw', async t => {
  for (const operation of ['cube', 'bias', 'grad', 'gather', 'filtered-volume-level']) {
    const f = rendererFixture(t), texture = input(profiles[1], operation === 'filtered-volume-level' ? 'volume' : '2d');
    t.after(() => texture.dispose());
    let node;
    if (operation === 'cube') {
      const cube = new T.CubeTexture(Array.from({length: 6}, () => texture));
      Object.assign(cube, {type: T.ShortType, normalized: true, format: T.RedFormat, generateMipmaps: false, minFilter: T.NearestFilter});
      cube.needsUpdate = true; t.after(() => cube.dispose()); node = N.cubeTexture(cube, N.vec3(1, 0, 0));
    } else if (operation === 'filtered-volume-level') {
      texture.minFilter = texture.magFilter = T.LinearFilter;
      node = N.texture3D(texture, N.vec3(.25)).level(0);
    } else {
      node = N.texture(texture, N.vec2(.25));
      node = operation === 'grad' ? node.grad(N.vec2(.25), N.vec2(.5)) : node[operation](1);
    }
    f.material.colorNode = node.rgb; await f.renderer.init();
    await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), /Normalized 16-bit|normalized 16-bit/);
    assert.equal(draws(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});
