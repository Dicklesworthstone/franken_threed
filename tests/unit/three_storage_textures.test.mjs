import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeStorageTextures, inspectThreeStorageTexture} from '../../tools/ingest/three_storage_textures.mjs';
import {THREE, gpu, deferred} from './storage_texture_fixture.mjs';
const setup = (options = {}) => {
  const device = gpu(options), owner = createThreeStorageTextures(THREE, {getDevice: () => device, ...options});
  const texture = new THREE.Texture();
  return {device, owner, texture};
};
const write = texture => ({texture, isSampler: true, isSampledTexture: true, store: true, access: 'writeOnly'});
const sample = texture => ({texture, isSampler: true, isSampledTexture: true});

test('storage and ordinary scene sampling share one native allocation', async () => {
  const {device, owner, texture} = setup();
  const output = owner.captureBinding(write(texture));
  assert.equal(device.textures.length, 0);
  const view = owner.resolve(output);
  owner.written([output]);
  const binding = owner.bindingsFor().get(texture);
  assert.equal(binding.view.texture, view.texture);
  assert.equal(device.textures.length, 1);
  assert.equal(view.texture.descriptor.usage & 12, 12);
  assert.equal(device.calls.some(c => c[0] === 'upload'), false);
  assert.equal(binding.sampleType, 'float');
  await owner.whenIdle(); owner.dispose();
});

test('default storage mipmaps are GPU-generated once per write and reused on reads', () => {
  const {device, owner, texture} = setup();
  texture.image = {width: 7, height: 3};
  const output = owner.captureBinding(write(texture));
  owner.resolve(output); owner.written([output]);
  assert.equal(device.calls.filter(c => c[0] === 'mip').length, 2);
  assert.equal(owner.allocatedBytes, (7 * 3 + 3 + 1) * 4);
  owner.resolve(owner.captureBinding(sample(texture))); owner.resolve(owner.captureBinding(sample(texture)));
  assert.equal(device.calls.filter(c => c[0] === 'mip').length, 2);
  owner.written([output]); assert.equal(device.calls.filter(c => c[0] === 'mip').length, 4);
  owner.dispose();
});

test('manual mip writes select exactly one level and are not overwritten', () => {
  const {device, owner, texture} = setup(); texture.mipmapsAutoUpdate = false;
  const output = owner.captureBinding({...write(texture), mipLevel: 1});
  const view = owner.resolve(output); owner.written([output]);
  owner.resolve(owner.captureBinding(sample(texture)));
  assert.deepEqual(view.descriptor, {baseMipLevel: 1, mipLevelCount: 1});
  assert.equal(device.calls.some(c => c[0] === 'mip'), false); owner.dispose();
});

test('float32 formats retain their full storage and explicit unfilterable layout', () => {
  for (const [format, native, bytes] of [[THREE.RedFormat, 'r32float', 4], [THREE.RGFormat, 'rg32float', 8], [THREE.RGBAFormat, 'rgba32float', 16]]) {
    const {device, owner, texture} = setup(); texture.type = THREE.FloatType; texture.format = format;
    texture.minFilter = texture.magFilter = THREE.NearestFilter;
    const output = owner.captureBinding(write(texture)), input = owner.captureBinding(sample(texture));
    owner.resolve(output); owner.written([output]);
    assert.equal(output.entry.shape.format, native); assert.equal(output.entry.shape.texelBytes, bytes);
    assert.equal(owner.layout(input).texture.sampleType, 'unfilterable-float');
    assert.equal(owner.bindingsFor().get(texture).sampler.descriptor.magFilter, 'nearest');
    assert.equal(device.textures.length, 1); owner.dispose();
  }
});

test('sampler changes are captured at call time without replacing storage', () => {
  const {owner, texture} = setup();
  const before = owner.captureBinding({texture, isSampler: true});
  texture.magFilter = THREE.NearestFilter; texture.wrapS = THREE.RepeatWrapping;
  const after = owner.captureBinding({texture, isSampler: true});
  assert.equal(before.entry, after.entry);
  assert.equal(owner.resolve(before).descriptor.magFilter, 'linear');
  assert.equal(owner.resolve(after).descriptor.magFilter, 'nearest');
  assert.equal(owner.resolve(after).descriptor.addressModeU, 'repeat'); owner.dispose();
});

test('float32 filtering requires the actual device feature, never a precision downgrade', () => {
  const {owner, texture} = setup(); texture.type = THREE.FloatType;
  assert.throws(() => owner.captureBinding({texture, isSampler: true}), /float32-filterable/); owner.dispose();
  const enabled = setup({features: ['float32-filterable']}); enabled.texture.type = THREE.FloatType;
  const binding = enabled.owner.bindingsFor().get(enabled.texture);
  assert.equal(binding.sampleType, 'float'); assert.equal(binding.view.texture.descriptor.format, 'rgba32float'); enabled.owner.dispose();
});

test('feedback hazards are rejected before allocation but distinct ping-pong textures work', () => {
  const {device, owner, texture} = setup();
  const output = owner.captureBinding(write(texture)), input = owner.captureBinding(sample(texture));
  assert.throws(() => owner.validateDispatch([input, output]), /FEEDBACK/);
  assert.equal(device.textures.length, 0);
  owner.validateDispatch([output, owner.captureBinding(sample(new THREE.Texture()))]);
  owner.validateDispatch([output, owner.captureBinding({...write(texture), mipLevel: 1, access: 'readOnly'})]);
  owner.validateDispatch([owner.captureBinding({...write(texture), access: 'readWrite'})]); owner.dispose();
});

test('resize and disposal retire the allocation and reject queued stale views', () => {
  const {device, owner, texture} = setup();
  const old = owner.captureBinding(write(texture)); owner.resolve(old);
  texture.setSize(8, 4);
  assert.equal(device.textures[0].destroyed, 1); assert.equal(owner.allocatedBytes, 0);
  assert.throws(() => owner.resolve(old), /STALE/);
  owner.resolve(owner.captureBinding(write(texture)));
  assert.equal(device.textures[1].descriptor.size[0], 8);
  texture.dispose(); owner.dispose(); assert.equal(device.textures[1].destroyed, 1);
  assert.equal(texture.listeners.size, 0);
});

test('shape changes without dispose are detected; needsUpdate does not erase GPU-produced data', () => {
  const {device, owner, texture} = setup(); const old = owner.captureBinding(write(texture)); owner.resolve(old);
  texture.version++; texture.source.version++;
  const same = owner.captureBinding(write(texture)); owner.resolve(same); assert.equal(same.entry, old.entry);
  texture.image.width = 8; assert.throws(() => owner.resolve(old), /STALE/);
  owner.resolve(owner.captureBinding(write(texture))); assert.equal(device.textures[0].destroyed, 1); owner.dispose();
});

test('byte/count/device bounds and unsupported source profiles fail before allocation', () => {
  const {device, owner, texture} = setup({maxBytes: 16, maxTextures: 1});
  const s = owner.captureBinding(write(texture)); assert.throws(() => owner.resolve(s), /BUDGET/);
  assert.throws(() => owner.capture(new THREE.Texture()), /BUDGET/);
  assert.equal(device.textures.length, 0); owner.dispose();
  for (const patch of [{colorSpace: THREE.SRGBColorSpace}, {is3DTexture: true}, {type: -1}, {generateMipmaps: 'true'}, {image: {width: 0, height: 2}}]) {
    const t = Object.assign(new THREE.Texture(), patch);
    assert.throws(() => inspectThreeStorageTexture(t, THREE));
  }
});

test('external binding maps are preserved and device ownership is enforced', () => {
  let current = gpu(); const owner = createThreeStorageTextures(THREE, {getDevice: () => current});
  const borrowed = {}, value = {}, map = owner.bindingsFor(new Map([[borrowed, value]]));
  assert.equal(map.get(borrowed), value); assert.equal(map.has(borrowed), true);
  owner.resolve(owner.captureBinding(write(new THREE.Texture())));
  current = gpu(); assert.throws(() => owner.captureBinding(write(new THREE.Texture())), /DEVICE/); owner.dispose();
});

test('device errors, loss and cancellation reject completion and release resources', async () => {
  const {device, owner, texture} = setup(); device.validationError('bad texture');
  owner.resolve(owner.captureBinding(write(texture)));
  await assert.rejects(owner.whenIdle(), /bad texture/); assert.equal(owner.allocatedBytes, 0);
  const lost = setup(); lost.owner.resolve(lost.owner.captureBinding(write(lost.texture)));
  lost.device.lose({message: 'removed'}); await Promise.resolve();
  assert.throws(() => lost.owner.capture(lost.texture), /DEVICE_LOST/); assert.equal(lost.owner.allocatedBytes, 0);
  const completion = deferred(), controller = new AbortController(), cancelled = setup({completion, signal: controller.signal});
  cancelled.owner.resolve(cancelled.owner.captureBinding(write(cancelled.texture)));
  const idle = cancelled.owner.whenIdle(); controller.abort(new Error('cancelled'));
  await assert.rejects(idle, /cancelled/); assert.equal(cancelled.owner.allocatedBytes, 0);
});
