/** Native texture upload contracts against the pinned source helpers.
 * The existing byte recorder executes uploads, not shaders or browser copies.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.core.js';
import SourceTextureUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUTextureUtils.js';
import {createGpuThreeTextures, inspectThreeTexture} from './three_textures.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';
import {device} from './tests/volume_texture_test_fixture.mjs';

const raw = value => new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
function nativePool(t, options = {}) {
  const gpu = device(), pool = createGpuThreeTextures(gpu, {three: T, nativeMedia: true, nativeCube: true, ...options});
  t.after(() => pool.dispose()); return {gpu, pool};
}
function source(texture, format = 'rgba8unorm') {
  const writes = [], copies = [], flips = [], samplers = [], records = new Map();
  const image = texture.image;
  records.set(texture, {texture: {}, textureDescriptorGPU: {format,
    size: {width: image.width, height: image.height, depthOrArrayLayers: image.depth ?? 1}}});
  const utils = new SourceTextureUtils({get(value) {
    if (!records.has(value)) records.set(value, {}); return records.get(value);
  }, device: {createSampler(desc) { const sampler = {...desc}; samplers.push(sampler); return sampler; }, queue: {
    writeTexture(destination, data, layout, size) {
      writes.push({level: destination.mipLevel, layer: destination.origin.z,
        layout: {...layout}, size: {...size}, bytes: raw(data).slice()});
    },
    copyExternalImageToTexture(input, output, size) {
      copies.push({input: {...input}, output: {...output}, size: {...size}});
    },
  }}});
  // Record the exact source flip boundary. GPU filtering is not simulated.
  utils._flipY = (texture, descriptor, layer) => flips.push(layer);
  return {writes, copies, flips, samplers,
    upload() { utils.updateTexture(texture, {image}); },
    sampler() { utils.updateSampler({texture, textureNode: {}}); return samplers.at(-1); }};
}

for (const kind of ['2d', 'array', 'volume']) for (const [Type, type, suffix] of [
  [Uint8Array, T.UnsignedByteType, '8unorm'], [Uint16Array, T.HalfFloatType, '16float'],
  [Float32Array, T.FloatType, '32float'],
]) test(`native ${kind} ${suffix} uploads use source tight rows and ignore data premultiplication`, t => {
  const {gpu, pool} = nativePool(t);
  for (const [channels, format, prefix] of [[1, T.RedFormat, 'r'], [2, T.RGFormat, 'rg'], [4, T.RGBAFormat, 'rgba']]) {
    const layers = kind === '2d' ? 1 : 3;
    const pixels = new Type(3 * 2 * layers * channels + 4).subarray(2, -2);
    pixels.set(Array.from({length: pixels.length}, (_, i) => i + 1));
    const texture = kind === '2d' ? new T.DataTexture(pixels, 3, 2, format, type)
      : new T[kind === 'array' ? 'DataArrayTexture' : 'Data3DTexture'](pixels, 3, 2, layers);
    Object.assign(texture, {format, type, unpackAlignment: 8, premultiplyAlpha: true}); texture.needsUpdate = true;
    const before = raw(pixels).slice(), oracle = source(texture, prefix + suffix); oracle.upload();
    pool.prepare([texture]);
    assert.deepEqual(pool.nativeTexture(texture).levels[0], before);
    assert.deepEqual(raw(pixels), before); assert.equal(texture.unpackAlignment, 8);
    assert.equal(texture.premultiplyAlpha, true);
    for (const write of oracle.writes) assert.equal(write.layout.bytesPerRow, 3 * channels * Type.BYTES_PER_ELEMENT);
    assert.equal(gpu.writes.at(-1).layout.bytesPerRow, oracle.writes[0].layout.bytesPerRow);
    assert.equal(pool.inspect(texture).alignment, 1);
  }
});

test('native ignored unpack and alpha metadata share an existing residency without another upload', t => {
  const {gpu, pool} = nativePool(t), texture = new T.DataTexture(new Uint8Array([1, 2, 3, 4, 5, 6]), 3, 2, T.RedFormat);
  texture.needsUpdate = true; pool.prepare([texture]); const allocation = pool.nativeTexture(texture);
  texture.unpackAlignment = 8; texture.premultiplyAlpha = true; pool.update([texture]);
  assert.equal(pool.nativeTexture(texture), allocation); assert.equal(gpu.writes.length, 1);
  assert.deepEqual([...allocation.levels[0]], [1, 2, 3, 4, 5, 6]);
});

test('native flipped array updates preserve selection order, unrequested layers and CPU bytes', t => {
  const {gpu, pool} = nativePool(t), pixels = Uint8Array.from({length: 18}, (_, i) => i + 1);
  const texture = new T.DataArrayTexture(pixels, 3, 2, 3);
  Object.assign(texture, {format: T.RedFormat, flipY: true, unpackAlignment: 8, premultiplyAlpha: true}); texture.needsUpdate = true;
  const oracle = source(texture, 'r8unorm'); oracle.upload(); pool.prepare([texture]);
  assert.deepEqual(oracle.flips, [0, 1, 2]);
  assert.deepEqual([...pool.nativeTexture(texture).levels[0]], [4,5,6,1,2,3,10,11,12,7,8,9,16,17,18,13,14,15]);
  const before = pool.nativeTexture(texture).levels[0].slice(); pixels.fill(41);
  texture.addLayerUpdate(2); texture.addLayerUpdate(0); texture.needsUpdate = true;
  const selected = source(texture, 'r8unorm'); selected.upload();
  assert.deepEqual(selected.writes.map(write => write.layer), [2, 0]);
  assert.deepEqual(selected.writes.map(write => write.layout.offset), [12, 0]);
  texture.addLayerUpdate(2); texture.addLayerUpdate(0); pool.update([texture]);
  assert.deepEqual(gpu.writes.slice(-2).map(write => write.dest.origin[2]), [2, 0]);
  assert.deepEqual(pool.nativeTexture(texture).levels[0].slice(6, 12), before.slice(6, 12));
  assert.ok(pixels.every(value => value === 41)); assert.equal(texture.layerUpdates.size, 0);
});

test('native partial array upload uses tight layer offsets without flipping', t => {
  const {gpu, pool} = nativePool(t), pixels = Uint8Array.from({length: 18}, (_, i) => i + 1);
  const texture = new T.DataArrayTexture(pixels, 3, 2, 3);
  texture.format = T.RedFormat; texture.unpackAlignment = 8; texture.addLayerUpdate(1); texture.needsUpdate = true;
  pool.prepare([texture]);
  assert.deepEqual([...pool.nativeTexture(texture).levels[0]], [0,0,0,0,0,0,7,8,9,10,11,12,0,0,0,0,0,0]);
  assert.deepEqual(gpu.writes[0].size, [3, 2, 1]); assert.equal(gpu.writes[0].layout.bytesPerRow, 3);
});

for (const count of [2, 3]) test(`native authored ${count}-level data flips only the source base, with matching parity`, t => {
  const {gpu, pool} = nativePool(t), pixels = Uint8Array.from({length: 16}, (_, i) => i + 1);
  const texture = new T.DataTexture(pixels, 4, 4, T.RedFormat);
  texture.mipmaps = [texture.image, {width: 2, height: 2, data: new Uint8Array([21,22,23,24])},
    {width: 1, height: 1, data: new Uint8Array([31])}].slice(0, count);
  Object.assign(texture, {flipY: true, generateMipmaps: true, minFilter: T.LinearMipmapLinearFilter}); texture.needsUpdate = true;
  const oracle = source(texture, 'r8unorm'); oracle.upload(); pool.prepare([texture]);
  assert.equal(oracle.flips.length, count); assert.ok(oracle.flips.every(layer => layer === 0));
  const native = pool.nativeTexture(texture);
  assert.deepEqual([...native.levels[0]], count === 2 ? [...pixels] : [13,14,15,16,9,10,11,12,5,6,7,8,1,2,3,4]);
  assert.deepEqual([...native.levels[1]], [21,22,23,24]);
  assert.equal(gpu.passes.length, 0); assert.equal(texture.generateMipmaps, true);
  assert.equal(native.levels.length, count);
});

for (const minFilter of [T.NearestFilter, T.LinearFilter, T.NearestMipmapNearestFilter,
  T.NearestMipmapLinearFilter, T.LinearMipmapNearestFilter, T.LinearMipmapLinearFilter])
  test(`native sampler ${minFilter} matches source LOD and anisotropy on an incomplete chain`, t => {
    const {pool} = nativePool(t), texture = new T.DataTexture(new Uint8Array(4 * 4 * 4), 4, 4);
    texture.mipmaps = [texture.image, {width: 2, height: 2, data: new Uint8Array(16)}];
    Object.assign(texture, {minFilter, magFilter: T.LinearFilter, anisotropy: 8}); texture.needsUpdate = true;
    const expected = source(texture).sampler(); pool.prepare([texture]); const actual = pool.binding(texture).sampler;
    for (const field of ['minFilter', 'magFilter', 'mipmapFilter', 'lodMinClamp', 'lodMaxClamp', 'maxAnisotropy'])
      assert.equal(actual[field], expected[field], field);
  });

test('native compressed RGBA mips also ignore unpack padding and use the source sampler profile', t => {
  const {pool} = nativePool(t), texture = new T.CompressedTexture([
    {width: 1, height: 2, data: new Uint8Array([1,2,3,4,5,6,7,8])},
    {width: 1, height: 1, data: new Uint8Array([9,10,11,12])},
  ], 1, 2, T.RGBAFormat);
  Object.assign(texture, {unpackAlignment: 8, minFilter: T.NearestMipmapLinearFilter, anisotropy: 4}); texture.needsUpdate = true;
  pool.prepare([texture]); assert.deepEqual([...pool.nativeTexture(texture).levels[0]], [1,2,3,4,5,6,7,8]);
  assert.equal(pool.binding(texture).sampler.maxAnisotropy, 1); assert.equal(pool.binding(texture).sampler.lodMaxClamp, 32);
});

test('native ImageBitmap copies preserve source orientation and alpha flags without owning the image', t => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'ImageBitmap');
  class Bitmap { constructor() { this.width = this.height = 2; } close() { assert.fail('caller bitmap must remain open'); } }
  Object.defineProperty(globalThis, 'ImageBitmap', {value: Bitmap, configurable: true});
  t.after(() => { if (previous) Object.defineProperty(globalThis, 'ImageBitmap', previous); else delete globalThis.ImageBitmap; });
  const {gpu, pool} = nativePool(t), copies = [];
  gpu.queue.copyExternalImageToTexture = (input, output, size) => copies.push({input, output, size});
  for (const flipY of [false, true]) for (const premultiplyAlpha of [false, true]) {
    const bitmap = new Bitmap(), texture = new T.Texture(bitmap);
    Object.assign(texture, {flipY, premultiplyAlpha, generateMipmaps: false, minFilter: T.LinearFilter}); texture.needsUpdate = true;
    const oracle = source(texture); oracle.upload(); pool.prepare([texture]);
    const actual = copies.at(-1), expected = oracle.copies[0];
    assert.equal(actual.input.source, bitmap); assert.equal(actual.input.flipY, expected.input.flipY);
    assert.equal(actual.output.premultipliedAlpha, expected.output.premultipliedAlpha);
    assert.equal(texture.image, bitmap);
  }
});

test('queued native sampled captures retain tight bytes, version history and selected array layers', t => {
  const {gpu, pool} = nativePool(t);
  const owner = createThreeComputeTextures({storage: {}, poolFactory: () => pool}); t.after(() => owner.dispose());
  const texture = new T.DataArrayTexture(Uint8Array.from({length: 12}, (_, i) => i + 1), 3, 2, 2);
  Object.assign(texture, {format: T.RedFormat, flipY: true, unpackAlignment: 8, premultiplyAlpha: true}); texture.needsUpdate = true;
  const binding = {texture, isSampledTexture: true}, first = owner.captureBinding(binding);
  const firstView = owner.resolve(first), before = firstView.texture.levels[0].slice();
  texture.image.data.fill(67); texture.addLayerUpdate(1); texture.needsUpdate = true;
  const second = owner.captureBinding(binding); texture.image.data.fill(99);
  const secondView = owner.resolve(second);
  assert.deepEqual(firstView.texture.levels[0], before);
  assert.deepEqual(secondView.texture.levels[0].slice(0, 6), before.slice(0, 6));
  assert.deepEqual([...secondView.texture.levels[0].slice(6)], Array(6).fill(67));
  assert.equal(texture.layerUpdates.size, 0); assert.ok(gpu.writes.length >= 3);
  owner.release(first); owner.release(second);
  assert.equal(owner.diagnostics.snapshotBytes, 12, 'the latest accepted version seeds later partial updates');
  owner.dispose(); assert.equal(owner.diagnostics.snapshotBytes, 0);
});

test('native invalid source flips and existing WebGL upload boundaries remain explicit', t => {
  const {gpu, pool} = nativePool(t), volume = new T.Data3DTexture(new Uint8Array(8), 2, 2, 2);
  volume.format = T.RedFormat; volume.flipY = true; volume.needsUpdate = true;
  assert.throws(() => pool.prepare([volume]), {code: 'THREE_TEXTURE_FORMAT'}); assert.equal(gpu.textures.length, 0);
  const legacy = new T.DataTexture(new Uint8Array([1,0,0,0,2]), 1, 2, T.RedFormat);
  legacy.unpackAlignment = 4; legacy.needsUpdate = true;
  assert.equal(inspectThreeTexture(legacy, T).alignment, 4);
  legacy.premultiplyAlpha = true;
  assert.throws(() => inspectThreeTexture(legacy, T), {code: 'THREE_TEXTURE_FORMAT'});
});
