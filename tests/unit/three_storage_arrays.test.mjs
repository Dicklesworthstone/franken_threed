import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeStorageTextures, inspectThreeStorageTexture} from '../../tools/ingest/three_storage_textures.mjs';
import {storageArrayMipShader} from '../../tools/ingest/gpu_storage_array_mips.mjs';
import {THREE, gpu} from './storage_texture_fixture.mjs';
const arrayTexture = (width = 7, height = 3, depth = 5) => {
  const texture = new THREE.Texture(width, height);
  texture.isArrayTexture = true; texture.image.depth = depth;
  // These are the actual r186 StorageArrayTexture defaults, not StorageTexture's.
  delete texture.mipmapsAutoUpdate;
  return texture;
};
const write = (texture, mipLevel = 0) => ({texture, isSampledTexture: true, isSampler: true, store: true, access: 'writeOnly', mipLevel});
const sample = texture => ({texture, isSampledTexture: true, isSampler: true});
const setup = (options = {}) => {
  const device = gpu(), owner = createThreeStorageTextures(THREE, {getDevice: () => device, ...options});
  device.limits.maxTextureArrayLayers = 256;
  return {owner, device};
};

test('array mip allocation keeps every layer and counts NPOT tails without shrinking depth', () => {
  const shape = inspectThreeStorageTexture(arrayTexture(), THREE);
  assert.equal(shape.viewDimension, '2d-array'); assert.equal(shape.depth, 5);
  assert.equal(shape.mipLevelCount, 3); assert.equal(shape.bytes, (7 * 3 + 3 * 1 + 1) * 5 * 4);
  assert.equal(inspectThreeStorageTexture(arrayTexture(1, 1, 256), THREE).mipLevelCount, 1);
});

test('one-layer arrays remain array textures in all native views and binding layouts', () => {
  const {owner, device} = setup(), texture = arrayTexture(4, 4, 1);
  const output = owner.captureBinding(write(texture)), input = owner.captureBinding(sample(texture));
  const view = owner.resolve(output), scene = owner.bindingsFor().get(texture);
  assert.equal(view.descriptor.dimension, '2d-array');
  assert.equal(scene.view.descriptor.dimension, '2d-array'); assert.equal(scene.viewDimension, '2d-array');
  assert.equal(owner.layout(output).storageTexture.viewDimension, '2d-array');
  assert.equal(owner.layout(input).texture.viewDimension, '2d-array');
  assert.equal(view.texture, scene.view.texture); assert.equal(device.textures.length, 1);
  assert.deepEqual(device.textures[0].descriptor.size, [4, 4, 1]);
  assert.equal(device.textures[0].descriptor.dimension, '2d'); owner.dispose();
});

test('r186 absent auto-update flag stays absent and does not generate unwanted mip work', () => {
  const {owner, device} = setup(), texture = arrayTexture();
  const output = owner.captureBinding(write(texture)); owner.resolve(output); owner.written([output]);
  owner.resolve(owner.captureBinding(sample(texture)));
  assert.equal(Object.hasOwn(texture, 'mipmapsAutoUpdate'), false);
  assert.equal(device.calls.some(c => c[0] === 'dispatch'), false); owner.dispose();
});

test('array auto-mips dispatch one full-layer grid per level in one GPU submission', () => {
  const {owner, device} = setup(), texture = arrayTexture(33, 17, 5); texture.mipmapsAutoUpdate = true;
  const output = owner.captureBinding(write(texture)); owner.resolve(output); owner.written([output]);
  assert.deepEqual(device.calls.filter(c => c[0] === 'dispatch').map(c => c.slice(1)),
    [[2, 1, 5], [1, 1, 5], [1, 1, 5], [1, 1, 5], [1, 1, 5]]);
  assert.equal(device.calls.filter(c => c[0] === 'submit').length, 1);
  assert.equal(device.calls.some(c => ['upload', 'mip'].includes(c[0])), false);
  assert.equal(device.textures.length, 1); assert.equal(device.buffers.length, 0);
  const groups = device.calls.filter(c => c[0] === 'bind').map(c => c[1]);
  groups.forEach((group, i) => {
    const [a, b] = group.entries.map(e => e.resource);
    assert.equal(a.texture, b.texture);
    assert.deepEqual(a.descriptor, {dimension: '2d-array', baseMipLevel: i, mipLevelCount: 1});
    assert.deepEqual(b.descriptor, {dimension: '2d-array', baseMipLevel: i + 1, mipLevelCount: 1});
  });
  owner.resolve(owner.captureBinding(sample(texture)));
  assert.equal(device.calls.filter(c => c[0] === 'submit').length, 1); owner.dispose();
});

test('array mip shader bounds checks all layers, loads the same layer, and never samples neighboring layers', () => {
  const code = storageArrayMipShader('rgba32float');
  assert.match(code, /textureNumLayers\(destination\)/);
  assert.equal((code.match(/layer, 0\)/g) ?? []).length, 4);
  assert.match(code, /texture_storage_2d_array<rgba32float, write>/);
  assert.doesNotMatch(code, /textureSample|layer\s*[+-]/);
  assert.throws(() => storageArrayMipShader('rgba8unorm-srgb'), /format/);
});

test('manual array mip writes bind one mip across all layers without triggering regeneration', () => {
  const {owner, device} = setup(), texture = arrayTexture();
  const output = owner.captureBinding(write(texture, 1));
  assert.deepEqual(owner.resolve(output).descriptor, {baseMipLevel: 1, mipLevelCount: 1, dimension: '2d-array'});
  owner.written([output]); owner.resolve(owner.captureBinding(sample(texture)));
  assert.equal(device.calls.some(c => c[0] === 'dispatch'), false); owner.dispose();
});

test('all admitted float formats keep full-precision storage and feature-free mip loads', () => {
  for (const [type, format, native] of [[THREE.UnsignedByteType, THREE.RGBAFormat, 'rgba8unorm'],
    [THREE.HalfFloatType, THREE.RGBAFormat, 'rgba16float'], [THREE.FloatType, THREE.RedFormat, 'r32float'],
    [THREE.FloatType, THREE.RGFormat, 'rg32float'], [THREE.FloatType, THREE.RGBAFormat, 'rgba32float']]) {
    const {owner, device} = setup(), texture = arrayTexture(4, 4, 2);
    Object.assign(texture, {type, format, mipmapsAutoUpdate: true, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter});
    const output = owner.captureBinding(write(texture)); owner.resolve(output); owner.written([output]);
    const layout = device.calls.find(c => c[0] === 'layout')[1];
    assert.equal(layout.entries[0].texture.sampleType, 'unfilterable-float');
    assert.equal(layout.entries[1].storageTexture.format, native);
    assert.equal(device.textures[0].descriptor.format, native);
    owner.dispose();
  }
});

test('array budgets charge layers and enforce device layer/dimension limits before allocation', () => {
  const {owner, device} = setup({maxBytes: 64});
  assert.throws(() => owner.ensure(owner.capture(arrayTexture(4, 4, 2))), /BUDGET/);
  device.limits.maxTextureArrayLayers = 1;
  assert.throws(() => owner.ensure(owner.capture(arrayTexture(1, 1, 2))), /LIMIT/);
  device.limits.maxTextureDimension2D = 2;
  assert.throws(() => owner.ensure(owner.capture(arrayTexture(4, 1, 1))), /LIMIT/);
  assert.equal(device.textures.length, 0); owner.dispose();
});

test('depth changes invalidate queued array generations, without retaining the old allocation', () => {
  const {owner, device} = setup(), texture = arrayTexture();
  const old = owner.captureBinding(write(texture)); owner.resolve(old);
  texture.image.depth++;
  assert.throws(() => owner.resolve(old), /STALE/);
  const current = owner.captureBinding(write(texture)); owner.resolve(current);
  assert.equal(device.textures[0].destroyed, 1);
  assert.equal(device.textures[1].descriptor.size[2], 6);
  texture.dispose(); assert.equal(owner.allocatedBytes, 0);
  assert.throws(() => owner.resolve(current), /STALE/); owner.dispose();
});

test('feedback validation treats sampled array views as covering all layers', () => {
  const {owner, device} = setup(), texture = arrayTexture();
  const output = owner.captureBinding(write(texture)), input = owner.captureBinding(sample(texture));
  assert.throws(() => owner.validateDispatch([output, input]), /FEEDBACK/);
  owner.validateDispatch([output, owner.captureBinding(sample(arrayTexture()))]);
  owner.validateDispatch([output, owner.captureBinding({...write(texture, 1), access: 'readOnly'})]);
  assert.equal(device.textures.length, 0); owner.dispose();
});

test('array shape admission rejects missing depth, ambiguous dimensions and non-boolean auto flags', () => {
  for (const patch of [{image: {width: 2, height: 2}}, {image: {width: 2, height: 2, depth: 0}},
    {image: {width: 2, height: 2, depth: 1.5}}, {is3DTexture: true}, {isCubeTexture: true}, {mipmapsAutoUpdate: 'true'}])
    assert.throws(() => inspectThreeStorageTexture(Object.assign(arrayTexture(), patch), THREE));
});

test('auto-mip selection is captured, and later toggles do not change queued write behavior', () => {
  const {owner, device} = setup(), texture = arrayTexture(); texture.mipmapsAutoUpdate = true;
  const output = owner.captureBinding(write(texture)); texture.mipmapsAutoUpdate = false;
  owner.resolve(output); owner.written([output]);
  assert.equal(device.calls.filter(c => c[0] === 'dispatch').length, 2); owner.dispose();
});
