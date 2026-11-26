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

for (const [type, format, native, bytes] of [
  [THREE.UnsignedByteType, THREE.RedFormat, 'r8unorm', 1],
  [THREE.UnsignedByteType, THREE.RGFormat, 'rg8unorm', 2],
  [THREE.HalfFloatType, THREE.RedFormat, 'r16float', 2],
  [THREE.HalfFloatType, THREE.RGFormat, 'rg16float', 4],
  [THREE.ByteType, THREE.RedFormat, 'r8snorm', 1],
  [THREE.ByteType, THREE.RGFormat, 'rg8snorm', 2],
  [THREE.ByteType, THREE.RGBAFormat, 'rgba8snorm', 4],
]) {
  test(`compact ${native} storage shares exact 2D/array/volume allocations and mip pipelines`, () => {
    for (const dimension of ['2d', '2d-array', '3d']) {
      const {device, owner, texture} = setup({features: ['texture-formats-tier1']});
      Object.assign(texture, {type, format, image: {width: 7, height: 3}});
      if (dimension !== '2d') {
        texture.image.depth = 5; texture[dimension === '3d' ? 'is3DTexture' : 'isArrayTexture'] = true;
        texture.wrapR = THREE.ClampToEdgeWrapping;
      }
      const output = owner.captureBinding(write(texture)), shape = output.entry.shape;
      assert.equal(shape.format, native); assert.equal(shape.texelBytes, bytes);
      assert.equal(shape.bytes, bytes * (dimension === '3d' ? 7 * 3 * 5 + 3 * 1 * 2 + 1 :
        (7 * 3 + 3 * 1 + 1) * (dimension === '2d-array' ? 5 : 1)));
      const view = owner.resolve(output); owner.written([output]);
      const binding = owner.bindingsFor().get(texture);
      assert.equal(binding.view.texture, view.texture); assert.equal(binding.sampleType, 'float');
      assert.equal(binding.sampler.descriptor.magFilter, 'linear');
      assert.equal(owner.allocatedBytes, shape.bytes); assert.equal(device.textures.length, 1);
      if (dimension === '2d') assert.equal(device.calls.filter(call => call[0] === 'mip').length, 2);
      else {
        const pipelines = device.calls.filter(call => call[0] === 'pipeline').map(call => call[1]);
        assert.equal(pipelines.length, 2);
        assert.ok(pipelines.every(pipeline => pipeline.descriptor.compute.module.code.includes(
          `texture_storage_${dimension === '3d' ? '3d' : '2d_array'}<${native}, write>`)));
      }
      assert.equal(device.calls.some(call => call[0] === 'upload'), false);
      owner.dispose(); assert.equal(view.texture.destroyed, 1); assert.equal(owner.allocatedBytes, 0);
    }
  });

  test(`compact ${native} storage enforces device features and source-valid access before allocation`, () => {
    const missing = setup(); Object.assign(missing.texture, {type, format});
    assert.throws(() => missing.owner.captureBinding(write(missing.texture)), /texture-formats-tier1/);
    assert.throws(() => missing.owner.layoutOf(sample(missing.texture)), /texture-formats-tier1/);
    assert.throws(() => missing.owner.ensure(missing.owner.capture(missing.texture)), /texture-formats-tier1/);
    assert.equal(missing.device.textures.length, 0); missing.owner.dispose();
    for (const tier2 of [false, true]) {
      const current = setup({features: ['texture-formats-tier1', ...(tier2 ? ['texture-formats-tier2'] : [])]});
      const {device, owner, texture} = current; Object.assign(texture, {type, format});
      const readWrite = {...write(texture), access: 'readWrite'};
      if (type === THREE.ByteType || format === THREE.RGFormat) assert.throws(() => owner.captureBinding(readWrite), /ACCESS/);
      else if (!tier2) assert.throws(() => owner.captureBinding(readWrite), /texture-formats-tier2/);
      else assert.equal(owner.layout(owner.captureBinding(readWrite)).storageTexture.access, 'read-write');
      assert.equal(device.textures.length, 0);
      const output = owner.captureBinding(write(texture));
      owner.validateDispatch([output, owner.captureBinding({...write(texture), mipLevel: 1, access: 'readOnly'})]);
      assert.throws(() => owner.validateDispatch([output, owner.captureBinding(sample(texture))]), /FEEDBACK/);
      owner.dispose();
    }
  });
}

test('signed normalized storage features follow its dimension and actual native usage', () => {
  for (const dimension of ['2d', '2d-array', '3d']) for (const format of [THREE.RedFormat, THREE.RGFormat, THREE.RGBAFormat]) {
    const {device, owner, texture} = setup();
    Object.assign(texture, {type: THREE.ByteType, format});
    if (dimension !== '2d') {
      texture.image.depth = 3; texture[dimension === '3d' ? 'is3DTexture' : 'isArrayTexture'] = true;
      texture.wrapR = THREE.ClampToEdgeWrapping;
    }
    try {
      if (dimension === '2d' || format !== THREE.RGBAFormat) {
        assert.throws(() => owner.captureBinding(write(texture)), /texture-formats-tier1/);
        assert.throws(() => owner.layoutOf(sample(texture)), /texture-formats-tier1/);
        assert.throws(() => owner.ensure(owner.capture(texture)), /texture-formats-tier1/);
        assert.equal(device.textures.length, 0);
      } else {
        // Core rgba8snorm storage needs no optional render-attachment feature
        // for array/volume mip generation: those levels use compute writes.
        const output = owner.captureBinding(write(texture)), view = owner.resolve(output);
        assert.equal(view.texture.descriptor.format, 'rgba8snorm');
        assert.equal(view.texture.descriptor.usage & 8, 8);
        assert.equal(view.texture.descriptor.usage & 16, 0);
        owner.written([output]);
        assert.ok(device.calls.some(call => call[0] === 'dispatch'));
        assert.equal(owner.layout(owner.captureBinding(sample(texture))).texture.sampleType, 'float');
        assert.equal(owner.resolve(owner.captureBinding({...write(texture), access: 'readOnly'})), view);
      }
    } finally { owner.dispose(); }
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

for (const failedScope of [0, 1]) {
  test(`storage scope ${failedScope} failure settles validation and idle while the sibling and queue are pending`, async t => {
    const sibling = deferred(), completion = deferred();
    const {device, owner, texture} = setup({completion});
    t.after(() => owner.dispose());
    const reports = [sibling.promise, sibling.promise];
    reports[failedScope] = Promise.resolve({message: 'storage allocation invalid'});
    device.popErrorScope = () => reports.shift();
    owner.ensure(owner.capture(texture));
    const validation = owner.validation, idle = owner.whenIdle();
    const outcomes = await Promise.race([
      Promise.allSettled([validation, idle]),
      new Promise(resolve => setImmediate(() => resolve(null))),
    ]);
    assert.ok(outcomes, 'a reported error must not wait for another scope or queue completion');
    for (const outcome of outcomes) {
      assert.equal(outcome.status, 'rejected');
      assert.equal(outcome.reason.code, 'THREE_STORAGE_TEXTURE_GPU');
      assert.match(outcome.reason.message, /storage allocation invalid/);
    }
    assert.equal(owner.allocatedBytes, 0); assert.equal(owner.size, 0);
    assert.equal(device.textures[0].destroyed, 1); assert.equal(texture.listeners.size, 0);
    sibling.reject(new Error('late scope rejection')); completion.resolve();
    await new Promise(resolve => setImmediate(resolve));
    await assert.rejects(owner.whenIdle(), /storage allocation invalid/);
    owner.dispose(); assert.equal(device.textures[0].destroyed, 1);
  });
}

test('storage transfer failure does not wait for a pending producer validation', async t => {
  const {device, owner, texture} = setup(), producer = deferred();
  t.after(() => owner.dispose());
  texture.generateMipmaps = false;
  const entry = owner.ensure(owner.capture(texture));
  await owner.whenIdle();
  device.validationError('invalid destination transfer');
  let calls = 0;
  owner.withTexture(entry, () => { calls++; }, null, {validation: producer.promise});
  const outcome = await Promise.race([
    owner.validation.then(() => null, error => error),
    new Promise(resolve => setImmediate(() => resolve(null))),
  ]);
  assert.match(outcome?.message ?? '', /invalid destination transfer/);
  assert.equal(calls, 1); assert.equal(owner.allocatedBytes, 0);
  producer.reject(new Error('late producer rejection'));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(device.textures[0].destroyed, 1);
});

test('successful storage scopes still wait for producer validation and GPU completion', async t => {
  const completion = deferred(), producer = deferred();
  const {owner, texture} = setup({completion});
  t.after(() => owner.dispose());
  texture.generateMipmaps = false;
  const entry = owner.ensure(owner.capture(texture));
  owner.withTexture(entry, () => {}, null, {validation: producer.promise});
  let validated = false, idle = false;
  const validation = owner.validation.then(() => { validated = true; });
  const finished = owner.whenIdle().then(() => { idle = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(validated, false); assert.equal(idle, false);
  producer.resolve(); await validation;
  assert.equal(idle, false); assert.ok(owner.allocatedBytes > 0);
  completion.resolve(); await finished;
  assert.equal(idle, true);
});
