import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeStorageTextures, inspectThreeStorageTexture} from '../../tools/ingest/three_storage_textures.mjs';
import {storageVolumeMipShader} from '../../tools/ingest/gpu_storage_volume_mips.mjs';
import {THREE, gpu, deferred} from './storage_texture_fixture.mjs';
const volume = (width = 7, height = 3, depth = 5) => {
  const texture = new THREE.Texture(width, height);
  Object.assign(texture, {is3DTexture: true, isArrayTexture: false, wrapR: THREE.ClampToEdgeWrapping});
  texture.image.depth = depth; delete texture.mipmapsAutoUpdate;
  return texture;
};
const write = (texture, mipLevel = 0) => ({texture, isSampledTexture: true, isSampler: true, store: true, access: 'writeOnly', mipLevel});
const sample = texture => ({texture, isSampledTexture: true, isSampler: true});
const setup = (options = {}) => {
  const device = gpu(options), owner = createThreeStorageTextures(THREE, {getDevice: () => device, ...options});
  device.limits.maxTextureDimension3D = 2048; device.limits.maxTextureArrayLayers = 256;
  return {owner, device};
};

test('volume allocation shrinks all spatial axes and includes depth in the mip count', () => {
  const shape = inspectThreeStorageTexture(volume(), THREE);
  assert.equal(shape.viewDimension, '3d'); assert.equal(shape.mipLevelCount, 3);
  assert.equal(shape.bytes, (7 * 3 * 5 + 3 * 1 * 2 + 1) * 4);
  const tall = inspectThreeStorageTexture(volume(1, 1, 9), THREE);
  assert.equal(tall.mipLevelCount, 4); assert.equal(tall.bytes, (9 + 4 + 2 + 1) * 4);
});

test('3D storage, sampled and material bindings share native volume allocation, even at depth one', () => {
  for (const depth of [1, 5]) {
    const {owner, device} = setup(), texture = volume(4, 4, depth);
    const output = owner.captureBinding(write(texture)), input = owner.captureBinding(sample(texture));
    const storage = owner.resolve(output), binding = owner.bindingsFor().get(texture);
    assert.equal(storage.texture, binding.view.texture);
    assert.equal(storage.descriptor.dimension, '3d'); assert.equal(binding.viewDimension, '3d');
    assert.equal(binding.sampler.descriptor.addressModeW, 'clamp-to-edge');
    assert.equal(owner.layout(input).texture.viewDimension, '3d');
    assert.equal(owner.layout(output).storageTexture.viewDimension, '3d');
    assert.deepEqual(device.textures[0].descriptor.size, [4, 4, depth]);
    assert.equal(device.textures[0].descriptor.dimension, '3d');
    assert.equal(device.textures[0].descriptor.usage & 16, 0);
    assert.equal(device.textures.length, 1); assert.equal(device.buffers.length, 0); owner.dispose();
  }
});

test('volume mip generation dispatches decreasing depth grids with nonoverlapping levels', () => {
  const {owner, device} = setup(), texture = volume(33, 17, 65); texture.mipmapsAutoUpdate = true;
  const output = owner.captureBinding(write(texture)); owner.resolve(output); owner.written([output]);
  assert.deepEqual(device.calls.filter(c => c[0] === 'dispatch').map(c => c.slice(1)),
    [[4, 2, 8], [2, 1, 4], [1, 1, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1]]);
  const groups = device.calls.filter(c => c[0] === 'bind').map(c => c[1]);
  groups.forEach((group, i) => {
    const [a, b] = group.entries.map(e => e.resource);
    assert.equal(a.texture, b.texture);
    assert.deepEqual(a.descriptor, {dimension: '3d', baseMipLevel: i, mipLevelCount: 1});
    assert.deepEqual(b.descriptor, {dimension: '3d', baseMipLevel: i + 1, mipLevelCount: 1});
  });
  assert.equal(device.calls.filter(c => c[0] === 'compute').length, groups.length);
  assert.equal(device.calls.filter(c => c[0] === 'submit').length, 1);
  assert.equal(device.calls.some(c => ['upload', 'mip'].includes(c[0])), false); owner.dispose();
});

test('volume mip shader uses eight clamped texel taps and interpolates through depth', () => {
  const code = storageVolumeMipShader('r32float');
  assert.equal((code.match(/textureLoad\(/g) ?? []).length, 8);
  assert.match(code, /mix\(front, back, f.z\)/); assert.match(code, /any\(id >= target\)/);
  assert.match(code, /texture_storage_3d<r32float, write>/);
  assert.doesNotMatch(code, /textureSample|textureNumLayers/);
  assert.throws(() => storageVolumeMipShader('depth32float'), /format/);
});

test('undefined and false auto-mips leave manually authored 3D mip levels untouched', () => {
  for (const setting of [undefined, false]) {
    const {owner, device} = setup(), texture = volume(1, 1, 9);
    if (setting !== undefined) texture.mipmapsAutoUpdate = setting;
    const output = owner.captureBinding(write(texture, 3));
    assert.deepEqual(owner.resolve(output).descriptor, {baseMipLevel: 3, mipLevelCount: 1, dimension: '3d'});
    owner.written([output]); owner.resolve(owner.captureBinding(sample(texture)));
    assert.equal(device.calls.some(c => c[0] === 'dispatch'), false);
    assert.equal(texture.mipmapsAutoUpdate, setting); owner.dispose();
  }
});

test('wrapR is captured per call, without changing volume identity or old sampler settings', () => {
  const {owner} = setup(), texture = volume();
  const first = owner.captureBinding({texture, isSampler: true});
  texture.wrapR = THREE.RepeatWrapping;
  const second = owner.captureBinding({texture, isSampler: true});
  assert.equal(first.entry, second.entry);
  assert.equal(owner.resolve(first).descriptor.addressModeW, 'clamp-to-edge');
  assert.equal(owner.resolve(second).descriptor.addressModeW, 'repeat');
  texture.wrapR = -1; assert.throws(() => owner.captureBinding({texture, isSampler: true}), /SAMPLER/); owner.dispose();
});

test('volume limits apply to all three spatial axes, not array-layer or 2D limits', () => {
  const {owner, device} = setup(); device.limits.maxTextureDimension3D = 8;
  for (const size of [[9, 1, 1], [1, 9, 1], [1, 1, 9]])
    assert.throws(() => owner.ensure(owner.capture(volume(...size))), /LIMIT/);
  assert.equal(device.textures.length, 0);
  device.limits.maxTextureArrayLayers = 1; device.limits.maxTextureDimension2D = 1;
  owner.ensure(owner.capture(volume(8, 8, 8)));
  assert.equal(device.textures.length, 1); owner.dispose();
});

test('volume and array profiles never share incompatible mip pipelines', () => {
  const {owner, device} = setup(), a = volume(4, 4, 4), b = volume(4, 4, 4);
  Object.assign(b, {is3DTexture: false, isArrayTexture: true});
  for (const texture of [a, b, a]) {
    texture.mipmapsAutoUpdate = true;
    const output = owner.captureBinding(write(texture)); owner.resolve(output); owner.written([output]);
  }
  const pipelines = device.calls.filter(c => c[0] === 'pipeline').map(c => c[1]);
  assert.equal(new Set(pipelines).size, 2);
  assert.match(pipelines[0].descriptor.compute.module.code, /texture_storage_3d/);
  assert.match(pipelines[2].descriptor.compute.module.code, /texture_storage_2d_array/); owner.dispose();
});

test('volume bytes include every depth slice and reject impossible extents before native allocation', () => {
  const {owner, device} = setup({maxBytes: 448});
  const entry = owner.capture(volume()); assert.equal(entry.shape.bytes, 448); owner.ensure(entry);
  assert.equal(owner.allocatedBytes, 448);
  assert.throws(() => owner.ensure(owner.capture(volume(1, 1, 1))), /BUDGET/);
  assert.equal(device.textures.length, 1); owner.dispose();
  assert.throws(() => inspectThreeStorageTexture(volume(2 ** 26, 2 ** 26, 2 ** 26), THREE), /BUDGET/);
});

test('float32 volume sampling keeps precision and requires real linear-filter support', () => {
  const {owner, device} = setup(), texture = volume(); texture.type = THREE.FloatType;
  assert.throws(() => owner.bindingsFor().get(texture), /float32-filterable/);
  assert.equal(device.textures.length, 0);
  texture.minFilter = texture.magFilter = THREE.NearestFilter;
  const binding = owner.bindingsFor().get(texture);
  assert.equal(binding.sampleType, 'unfilterable-float');
  assert.equal(binding.view.texture.descriptor.format, 'rgba32float'); owner.dispose();
});

test('queued volume views fail after depth resize or dimensional reconfiguration', () => {
  const {owner, device} = setup(), texture = volume();
  const old = owner.captureBinding(write(texture)); owner.resolve(old);
  texture.image.depth = 9; assert.throws(() => owner.resolve(old), /STALE/);
  const newer = owner.captureBinding(write(texture)); owner.resolve(newer);
  assert.equal(device.textures[0].destroyed, 1);
  Object.assign(texture, {is3DTexture: false, isArrayTexture: true});
  assert.throws(() => owner.resolve(newer), /STALE/);
  const array = owner.captureBinding(write(texture)); owner.resolve(array);
  assert.equal(device.textures[1].destroyed, 1); assert.equal(array.entry.shape.viewDimension, '2d-array'); owner.dispose();
});

test('device loss and cancellation revoke volume allocations and pending completion', async () => {
  const {owner, device} = setup(), texture = volume(); owner.ensure(owner.capture(texture));
  device.lose({message: 'lost volume device'}); await Promise.resolve();
  assert.throws(() => owner.capture(texture), /DEVICE_LOST/); assert.equal(owner.allocatedBytes, 0);
  assert.equal(device.textures[0].destroyed, 1);
  const completion = deferred(), controller = new AbortController(), other = setup({completion, signal: controller.signal});
  other.owner.ensure(other.owner.capture(volume()));
  const idle = other.owner.whenIdle(); controller.abort(new Error('cancel volume'));
  await assert.rejects(idle, /cancel volume/); assert.equal(other.owner.allocatedBytes, 0);
});
