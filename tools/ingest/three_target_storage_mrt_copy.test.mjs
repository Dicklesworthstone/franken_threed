import test from 'node:test';
import assert from 'node:assert/strict';
import {captureTargetToStorageCopy} from './three_target_texture_copy.mjs';
import {fixture, T, target, setFormat, formats, payload, raw, box, position, deferred, flush} from './fixtures/target_storage_copy_fixture.mjs';
const storageTexture = (format = 'rgba8unorm', width = 4, height = 4, depth = 1) => setFormat(new T.StorageTexture(width, height, depth), format);
const capture = (f, ...args) => captureTargetToStorageCopy(f.renderer, ...args);
const seed = (f, source, data, colorIndex = 0) => {
  const entry = f.targets.capture(source); f.targets.writePixels(entry, data, {colorIndex}); return entry;
};
const destinationBytes = (f, texture, mip = 0) => f.storage.ensure(f.storage.capture(texture)).native.levels[mip].bytes;

for (const [format] of formats) for (const colorIndex of [0, 1, 2])
  test(`${format}: selected target attachment ${colorIndex} enters storage bit-for-bit without sampling`, async () => {
    const f = fixture(), colors = ['rgba8unorm', 'rgba16float', 'r32float']; colors[colorIndex] = format;
    const source = target({colors}), destination = storageTexture(format), data = payload(format);
    const entry = seed(f, source, data, colorIndex), before = f.calls.textures.length;
    const request = capture(f, source.textures[colorIndex], destination);
    assert.ok(Object.isFrozen(request)); assert.equal(request.issue(), true);
    assert.deepEqual(destinationBytes(f, destination), raw(data));
    assert.equal(f.calls.textures.length, before + 1, 'only the storage owner allocation is added');
    assert.equal(f.calls.buffers.length, 0); assert.equal(f.calls.samplers.length, 0);
    assert.equal(f.calls.passes.length, 0); assert.equal(f.calls.fences, 0, 'copy does not wait for GPU completion');
    let native;
    entry.storage.withTexture(texture => { native = texture; }, {colorIndex});
    assert.equal(f.calls.copies[0].source.texture, native);
    assert.equal(f.calls.copies[0].destination.texture, f.storage.capture(destination).native);
    const sampled = f.storage.captureBinding({texture: destination, isSampledTexture: true});
    assert.equal(f.storage.resolve(sampled).texture, f.storage.capture(destination).native);
    assert.equal(f.targets.allocatedBytes, entry.shape.bytes, 'no source UV image');
    await Promise.all([f.targets.whenIdle(), f.storage.whenIdle()]); f.dispose();
    assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
  });

test('2D source subrectangle enters one array layer with call-time region and submission-time contents', async () => {
  const f = fixture(), source = target(), destination = storageTexture('rgba8unorm', 6, 5, 3);
  const entry = seed(f, source, new Uint8Array(64).fill(11));
  const region = box(1, 1, 2, 2), at = position(2, 2, 1);
  const request = capture(f, source.texture, destination, region, at);
  region.min.x = 0; region.max.x = 4; at.z = 2; at.y = 0;
  const data = payload('rgba8unorm', 4, 4, 37); f.targets.writePixels(entry, data);
  request.issue();
  const expected = new Uint8Array(6 * 5 * 3 * 4);
  for (let row = 0; row < 2; row++) expected.set(data.subarray(((1 + row) * 4 + 1) * 4, ((1 + row) * 4 + 3) * 4), ((5 + 2 + row) * 6 + 2) * 4);
  assert.deepEqual(destinationBytes(f, destination), expected);
  f.targets.writePixels(entry, new Uint8Array(64).fill(99));
  assert.deepEqual(destinationBytes(f, destination), expected, 'later source uploads cannot overwrite the copy');
  await f.storage.whenIdle(); f.dispose();
});

for (const format of ['rgba8unorm', 'rgba16float']) test(`${format}: MSAA source uses the selected resolved image`, async () => {
  const f = fixture(), source = target({colors: [format, format], samples: 4}), destination = storageTexture(format);
  const entry = f.targets.ensure(f.targets.capture(source)), data = payload(format);
  let resolved, multisampled;
  entry.storage.withFrame(frame => { multisampled = frame.colorViews[1].texture; });
  entry.storage.withTexture(native => { resolved = native; native.levels[0].bytes.set(raw(data)); }, {colorIndex: 1});
  capture(f, source.textures[1], destination).issue();
  assert.equal(f.calls.copies[0].source.texture, resolved); assert.notEqual(resolved, multisampled);
  assert.equal(resolved.sampleCount, 1); assert.deepEqual(destinationBytes(f, destination), raw(data));
  await f.storage.whenIdle(); f.dispose();
});

test('authored source mip copies to an allocated storage mip without overwriting neighboring levels', async () => {
  const f = fixture(), source = target({width: 8, height: 4, mips: 4}), destination = storageTexture('rgba8unorm', 8, 4);
  destination.generateMipmaps = true;
  const entry = f.targets.ensure(f.targets.capture(source));
  entry.storage.withTexture(native => native.levels.forEach((level, i) => level.bytes.fill(27 + i)));
  capture(f, source.texture, destination, null, null, 2, 2).issue();
  const native = f.storage.capture(destination).native;
  assert.ok(native.levels[2].bytes.every(value => value === 29));
  for (const level of [0, 1, 3]) assert.ok(native.levels[level].bytes.every(value => value === 0));
  assert.equal(f.calls.copies[0].source.mipLevel, 2); assert.equal(f.calls.copies[0].destination.mipLevel, 2);
  assert.equal(f.calls.passes.length, 0); await f.storage.whenIdle(); f.dispose();
});

test('source and destination automatic mip passes surround the copy, not the opposite way round', async () => {
  const f = fixture(), source = target({generateMipmaps: true}), destination = storageTexture();
  destination.generateMipmaps = true; destination.mipmapsAutoUpdate = true;
  seed(f, source, payload('rgba8unorm'));
  capture(f, source.texture, destination).issue();
  const labels = f.calls.submissions.map(batch => batch[0].label);
  assert.match(labels[0], /mip chain/); assert.equal(labels[1], 'f3d-target-to-storage-copy');
  assert.equal(labels[2], 'f3d-storage-mips'); assert.equal(labels.length, 3);
  await f.storage.whenIdle(); f.dispose();
});

for (const change of ['reorder', 'replace', 'format', 'resize', 'dispose', 'texture-dispose'])
  test(`queued target ${change} cannot retarget an accepted storage copy`, () => {
    const f = fixture(), source = target({colors: ['rgba8unorm', 'rgba8unorm']}), destination = storageTexture();
    const request = capture(f, source.textures[1], destination);
    if (change === 'reorder') source.textures.reverse();
    if (change === 'replace') { const replacement = new T.Texture(); replacement.isRenderTargetTexture = true; replacement.renderTarget = source; source.textures[1] = replacement; }
    if (change === 'format') source.textures[1].type = T.HalfFloatType;
    if (change === 'resize') source.setSize(8, 8);
    if (change === 'dispose') source.dispose();
    if (change === 'texture-dispose') source.textures[1].dispose();
    assert.throws(() => request.issue(), /THREE_TARGET_STALE/);
    assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.submissions.length, 0); f.dispose();
  });

for (const change of ['resize', 'dispose', 'device', 'renderer']) test(`queued ${change} rejects without allocating copy endpoints`, () => {
  const f = fixture(), source = target(), destination = storageTexture(), request = capture(f, source.texture, destination);
  if (change === 'resize') destination.image.width = 8;
  if (change === 'dispose') destination.dispose();
  if (change === 'device') f.renderer._device = {};
  if (change === 'renderer') f.renderer._disposed = true;
  assert.throws(() => request.issue(), /STALE|DEVICE|DISPOSED/);
  assert.equal(f.calls.textures.length, 0); f.dispose();
});

for (const budget of ['target', 'storage']) test(`${budget} allocation budget fails before either owner allocates a prefix`, () => {
  const f = fixture(budget === 'target' ? {targetBudget: 127} : {storageBudget: 63});
  const source = target(), destination = storageTexture(), request = capture(f, source.texture, destination);
  assert.throws(() => request.issue(), /BUDGET/); assert.equal(f.calls.textures.length, 0); f.dispose();
});

test('limits, dimensions, formats, levels and regions are admitted before native work', () => {
  const f = fixture(), source = target();
  assert.throws(() => capture(f, source.texture, storageTexture('rgba16float')), /FORMAT/);
  assert.throws(() => capture(f, source.texture, storageTexture(), box(0, 0, 5, 4)), /BOUNDS/);
  assert.throws(() => capture(f, source.texture, storageTexture(), null, null, 1), /LEVEL/);
  const volume = storageTexture(); volume.is3DTexture = true;
  assert.doesNotThrow(() => capture(f, source.texture, volume));
  const request = capture(f, source.texture, storageTexture()); f.device.limits.maxTextureDimension2D = 2;
  assert.throws(() => request.issue(), /LIMIT/); assert.equal(f.calls.textures.length, 0); f.dispose();
});

for (const [sourceVolume, destinationVolume] of [[true, false], [false, true], [true, true]])
test(`layered target to storage copies every selected slice (${sourceVolume ? '3D' : 'array'} to ${destinationVolume ? '3D' : 'array'})`, async () => {
  const f = fixture(), source = target({depth: 3, volume: sourceVolume, colors: ['rgba8unorm', 'rgba8unorm']});
  const destination = storageTexture('rgba8unorm', 4, 4, 4);
  if (destinationVolume) { destination.isArrayTexture = false; destination.is3DTexture = true; }
  const entry = f.targets.capture(source);
  for (let face = 0; face < 3; face++) f.targets.writePixels(entry, new Uint8Array(64).fill(31 + face), {face, colorIndex: 1});
  capture(f, source.textures[1], destination, null, position(0, 0, 1)).issue();
  const expected = new Uint8Array(256);
  for (let layer = 1; layer < 4; layer++) expected.fill(30 + layer, layer * 64, (layer + 1) * 64);
  assert.deepEqual(destinationBytes(f, destination), expected);
  assert.deepEqual(f.calls.copies[0].size, [4, 4, 3]);
  assert.deepEqual(f.calls.copies[0].destination.origin, [0, 0, 1]);
  assert.equal(f.calls.samplers.length, 0); assert.equal(f.calls.passes.length, 0);
  await Promise.all([f.targets.whenIdle(), f.storage.whenIdle()]); f.dispose();
});

test('target volume source mips shrink Z before entering storage array layers', async () => {
  const f = fixture(), source = target({width: 8, height: 4, depth: 6, volume: true, mips: 4});
  const destination = storageTexture('rgba8unorm', 4, 2, 5), entry = f.targets.ensure(f.targets.capture(source));
  entry.storage.withTexture(native => {
    for (let slice = 0; slice < 3; slice++) native.levels[1].bytes.fill(71 + slice, slice * 32, (slice + 1) * 32);
  });
  capture(f, source.texture, destination, null, position(0, 0, 1), 1).issue();
  const expected = new Uint8Array(160);
  for (let layer = 1; layer < 4; layer++) expected.fill(70 + layer, layer * 32, (layer + 1) * 32);
  assert.deepEqual(destinationBytes(f, destination), expected);
  assert.deepEqual(f.calls.copies[0].size, [4, 2, 3]); assert.equal(f.calls.copies[0].source.mipLevel, 1);
  assert.throws(() => capture(f, source.texture, destination, box(0, 0, 4, 2, 3), null, 1), /BOUNDS/);
  assert.equal(f.calls.copies.length, 1); await f.storage.whenIdle(); f.dispose();
});

test('empty copy remains GPU-free but still validates captured generations', () => {
  const f = fixture(), source = target(), destination = storageTexture();
  const request = capture(f, source.texture, destination, box(4, 0, 0, 4));
  assert.equal(request.issue(), false); assert.equal(f.calls.textures.length, 0);
  source.dispose(); assert.throws(() => request.issue(), /STALE/); f.dispose();
});

test('both borrowed owners reject recursive operations without losing the submitted copy', async () => {
  const f = fixture(), source = target(), destination = storageTexture(); seed(f, source, payload('rgba8unorm'));
  const request = capture(f, source.texture, destination);
  f.controls.onCopy = () => {
    assert.throws(() => request.issue(), /REENTRANT/);
    assert.throws(() => f.targets.writePixels(f.targets.capture(source), new Uint8Array(64)), /REENTRANT/);
    assert.throws(() => f.storage.withTexture(f.storage.capture(destination), () => {}), /REENTRANT/);
  };
  request.issue(); assert.equal(f.calls.copies.length, 1);
  await f.storage.whenIdle(); f.dispose();
});

test('pending source producer validation follows the copy without delaying submission or inserting a fence', async () => {
  const f = fixture(), source = target(), destination = storageTexture(), gate = deferred();
  const entry = seed(f, source, payload('rgba8unorm')); await f.targets.whenIdle();
  entry.storage.withColorDestination(() => {}, {validation: gate.promise});
  const fences = f.calls.fences; capture(f, source.texture, destination).issue();
  const validation = f.storage.validation; let settled = false; validation.then(() => { settled = true; }, () => { settled = true; });
  await flush(); assert.equal(settled, false); assert.equal(f.calls.copies.length, 1); assert.equal(f.calls.fences, fences);
  gate.resolve(); await validation; assert.equal(settled, true); f.dispose();
});

test('source producer rejection invalidates the storage result even when the copy itself validates', async () => {
  const f = fixture(), source = target(), destination = storageTexture(), gate = deferred();
  const entry = seed(f, source, payload('rgba8unorm')); await f.targets.whenIdle();
  entry.storage.withColorDestination(() => {}, {validation: gate.promise});
  capture(f, source.texture, destination).issue();
  const result = f.storage.whenIdle(); gate.reject(new Error('source producer rejected'));
  await assert.rejects(result, /source producer rejected/);
  assert.throws(() => f.storage.capture(destination), /source producer rejected/);
  f.dispose(); assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});

test('copy validation rejection is observed and releases the destination owner', async () => {
  const f = fixture(), source = target(), destination = storageTexture(); seed(f, source, payload('rgba8unorm'));
  f.storage.ensure(f.storage.capture(destination)); await Promise.all([f.targets.whenIdle(), f.storage.whenIdle()]);
  f.controls.onCopy = () => { f.controls.validation = {message: 'copy validation rejected'}; };
  capture(f, source.texture, destination).issue();
  await assert.rejects(f.storage.whenIdle(), /copy validation rejected/);
  assert.equal(f.storage.allocatedBytes, 0); f.dispose();
});

test('validation snapshots exclude future work and settled dependencies do not retain pending tasks', async () => {
  const f = fixture(), texture = storageTexture(), entry = f.storage.capture(texture);
  const earlier = f.storage.validation, later = deferred();
  f.storage.withTexture(entry, () => {}, null, {validation: later.promise});
  await earlier; let settled = false; const active = f.storage.validation.then(() => { settled = true; });
  await flush(); assert.equal(settled, false); later.resolve(); await active;
  await flush(); await f.storage.validation; f.dispose();
});

test('device loss interrupts a copy waiting on source validation without owning the device', async () => {
  const f = fixture(), source = target(), destination = storageTexture(), gate = deferred();
  const entry = seed(f, source, payload('rgba8unorm')); entry.storage.withColorDestination(() => {}, {validation: gate.promise});
  capture(f, source.texture, destination).issue();
  const idle = f.storage.whenIdle(); f.loss.resolve({message: 'device lost during transfer'});
  await assert.rejects(idle, /device lost during transfer/); gate.resolve(); f.dispose();
  assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});
