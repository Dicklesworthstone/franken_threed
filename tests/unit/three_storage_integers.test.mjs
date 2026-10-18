import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeStorageTextures, inspectThreeStorageTexture} from '../../tools/ingest/three_storage_textures.mjs';
import {THREE as base, gpu} from './storage_texture_fixture.mjs';
// Pinned r186 constant values; the existing fixture's Texture identity is kept.
const THREE = {...base, IntType: 1013, UnsignedIntType: 1014,
  RedIntegerFormat: 1029, RGIntegerFormat: 1031, RGBAIntegerFormat: 1033};
const formats = [['RedFormat', 'RedIntegerFormat', 'r', 4],
  ['RGFormat', 'RGIntegerFormat', 'rg', 8], ['RGBAFormat', 'RGBAIntegerFormat', 'rgba', 16]];
function texture(type = THREE.UnsignedIntType, format = THREE.RedIntegerFormat, dimension = '2d') {
  const value = new THREE.Texture(4, 2);
  Object.assign(value, {type, format, generateMipmaps: false,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter});
  if (dimension !== '2d') value.image.depth = 3;
  if (dimension === '2d-array') value.isArrayTexture = true;
  if (dimension === '3d') { value.is3DTexture = true; value.wrapR = THREE.ClampToEdgeWrapping; }
  return value;
}
const write = t => ({texture: t, isSampledTexture: true, isSampler: true, store: true, access: 'writeOnly'});
const sample = t => ({texture: t, isSampledTexture: true, isSampler: true});
const ownerFor = device => createThreeStorageTextures(THREE, {getDevice: () => device});

for (const [type, sampleType, suffix] of [[THREE.IntType, 'sint', 'sint'], [THREE.UnsignedIntType, 'uint', 'uint']])
  for (const [ordinary, integer, prefix, bytes] of formats)
    for (const dimension of ['2d', '2d-array', '3d'])
      test(`${prefix}32${suffix} ${dimension} uses exact storage and typed read layouts`, async () => {
        const device = gpu(), owner = ownerFor(device), t = texture(type, THREE[integer], dimension);
        const output = owner.captureBinding(write(t)), input = owner.captureBinding(sample(t));
        const shape = inspectThreeStorageTexture(t, THREE);
        assert.equal(shape.format, `${prefix}32${suffix}`);
        assert.equal(shape.texelBytes, bytes);
        assert.equal(shape.bytes, 4 * 2 * (dimension === '2d' ? 1 : 3) * bytes);
        assert.deepEqual(owner.layout(input), {texture: {sampleType, viewDimension: dimension, multisampled: false}});
        assert.deepEqual(owner.layout(output), {storageTexture: {access: 'write-only', format: shape.format, viewDimension: dimension}});
        const view = owner.resolve(output); owner.written([output]);
        const binding = owner.bindingsFor().get(t);
        assert.equal(binding.view.texture, view.texture);
        assert.equal(binding.sampleType, sampleType);
        assert.equal(device.textures.length, 1);
        assert.equal(device.textures[0].descriptor.format, shape.format);
        assert.equal(device.calls.some(c => ['upload', 'mip', 'compute'].includes(c[0])), false);
        assert.equal(owner.layout(owner.captureBinding({texture: t, isSampler: true})).sampler.type, 'non-filtering');
        const alias = texture(type, THREE[ordinary], dimension);
        assert.equal(inspectThreeStorageTexture(alias, THREE).format, shape.format);
        await owner.whenIdle(); owner.dispose();
      });

test('manual integer mip levels remain untouched by float mip reducers', () => {
  for (const dimension of ['2d', '2d-array', '3d']) {
    const device = gpu(), owner = ownerFor(device), t = texture(THREE.IntType, THREE.RGBAIntegerFormat, dimension);
    t.generateMipmaps = true; t.mipmapsAutoUpdate = false;
    const output = owner.captureBinding({...write(t), mipLevel: 1});
    const selected = owner.resolve(output); owner.written([output]);
    owner.resolve(owner.captureBinding(sample(t)));
    assert.equal(selected.descriptor.baseMipLevel, 1);
    assert.equal(selected.descriptor.mipLevelCount, 1);
    assert.equal(device.calls.some(c => ['mip', 'compute', 'submit'].includes(c[0])), false);
    owner.dispose();
  }
});

test('integer mip interpolation and filtering fail before GPU allocation', () => {
  const device = gpu(), owner = ownerFor(device), t = texture();
  t.generateMipmaps = true;
  assert.throws(() => owner.captureBinding(write(t)), /MIPS/);
  t.mipmapsAutoUpdate = false;
  for (const field of ['minFilter', 'magFilter']) {
    t[field] = THREE.LinearFilter;
    assert.throws(() => owner.captureBinding({texture: t, isSampler: true}), /SAMPLER/);
    t[field] = THREE.NearestFilter;
  }
  t.minFilter = THREE.NearestMipmapLinearFilter;
  assert.throws(() => owner.captureBinding({texture: t, isSampler: true}), /SAMPLER/);
  assert.equal(device.textures.length, 0); owner.dispose();
});

test('format changes select new integer generations; stale queued reads never bind a different type', () => {
  const device = gpu(), owner = ownerFor(device), t = texture();
  const unsigned = owner.captureBinding(sample(t)), old = owner.resolve(unsigned);
  t.type = THREE.IntType;
  assert.throws(() => owner.resolve(unsigned), /STALE/);
  const signed = owner.captureBinding(sample(t));
  assert.equal(old.texture.destroyed, 1);
  assert.equal(owner.layout(signed).texture.sampleType, 'sint');
  assert.notEqual(owner.resolve(signed).texture, old.texture);
  t.dispose(); assert.throws(() => owner.resolve(signed), /STALE/);
  assert.equal(owner.allocatedBytes, 0); owner.dispose();
});

test('integer aliases obey storage hazards and allocation budgets', () => {
  const device = gpu(), owner = createThreeStorageTextures(THREE, {getDevice: () => device, maxBytes: 31}), t = texture();
  const output = owner.captureBinding(write(t)), input = owner.captureBinding(sample(t));
  assert.throws(() => owner.validateDispatch([output, input]), /FEEDBACK/);
  assert.throws(() => owner.resolve(output), /BUDGET/);
  assert.equal(device.textures.length, 0);
  owner.validateDispatch([owner.captureBinding({...write(t), access: 'readWrite'})]);
  owner.dispose();
});

test('integer color space and unsupported component widths do not silently convert', () => {
  const t = texture();
  t.colorSpace = THREE.SRGBColorSpace;
  assert.throws(() => inspectThreeStorageTexture(t, THREE), /FORMAT/);
  t.colorSpace = THREE.NoColorSpace; t.type = THREE.UnsignedByteType;
  assert.throws(() => inspectThreeStorageTexture(t, THREE), /FORMAT/);
  t.type = THREE.FloatType;
  assert.throws(() => inspectThreeStorageTexture(t, THREE), /FORMAT/);
});
