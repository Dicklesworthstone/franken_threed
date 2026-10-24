import test from 'node:test';
import assert from 'node:assert/strict';
import {captureStorageToTargetCopy, captureTargetToStorageCopy} from './three_target_texture_copy.mjs';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';
import {fixture, T, target, setFormat, formats, payload, raw, box, position, deferred, flush} from './fixtures/target_storage_copy_fixture.mjs';
const storageTexture = (format = 'rgba8unorm', width = 4, height = 4, depth = 1) => setFormat(new T.StorageTexture(width, height, depth), format);
const capture = (f, ...args) => captureStorageToTargetCopy(f.renderer, ...args);
function seed(f, texture, data, {mipLevel = 0, layer = 0, validation} = {}) {
  const entry = f.storage.capture(texture), binding = f.storage.captureBinding({texture, isSampledTexture: true, store: true, access: 'writeOnly', mipLevel});
  f.storage.withTexture(entry, native => {
    const level = native.levels[mipLevel];
    f.device.queue.writeTexture({texture: native, mipLevel, origin: [0, 0, layer]}, data,
      {bytesPerRow: level.width * native.bpp, rowsPerImage: level.height}, [level.width, level.height, 1]);
  }, binding, {validation});
  return entry;
}
const read = (f, source, colorIndex = 0, options = {}) => f.targets.ensure(f.targets.capture(source)).storage.readPixels({colorIndex, ...options});

for (const [format] of formats) for (const colorIndex of [0, 1, 2])
  test(`${format}: storage reaches target attachment ${colorIndex} without touching sibling colors or depth`, async () => {
    const f = fixture(), colors = ['rgba8unorm', 'rgba16float', 'r32float']; colors[colorIndex] = format;
    const destination = target({colors}), source = storageTexture(format), data = payload(format);
    const entry = f.targets.ensure(f.targets.capture(destination));
    entry.storage.withAttachments((colors, depth) => {
      colors.forEach((native, i) => native.levels[0].bytes.fill(71 + i));
    });
    let depth;
    entry.storage.withFrame(frame => { depth = frame.depthView.texture; depth.levels[0].bytes.fill(211); });
    seed(f, source, data); const before = f.calls.textures.length;
    assert.equal(capture(f, source, destination.textures[colorIndex]).issue(), true);
    assert.equal(f.calls.textures.length, before); assert.equal(f.calls.buffers.length, 0);
    assert.equal(f.calls.samplers.length, 0); assert.equal(f.calls.passes.length, 0); assert.equal(f.calls.fences, 0);
    const copy = f.calls.copies[0]; assert.equal(copy.source.texture, f.storage.capture(source).native);
    entry.storage.withTexture(native => assert.equal(copy.destination.texture, native), {colorIndex});
    assert.deepEqual(raw(await read(f, destination, colorIndex)), raw(data));
    for (let i = 0; i < colors.length; i++) if (i !== colorIndex)
      assert.ok(raw(await read(f, destination, i)).every(value => value === 71 + i));
    assert.ok(depth.levels[0].bytes.every(value => value === 211));
    await f.targets.whenIdle(); f.dispose(); assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
  });

test('one storage array slice and captured subrectangle enter a target without changing untouched texels', async () => {
  const f = fixture(), source = storageTexture('rgba8unorm', 6, 5, 3), destination = target();
  const data = payload('rgba8unorm', 6, 5, 17); seed(f, source, data, {layer: 2});
  const region = box(2, 1, 2, 2, 2), at = position(1, 1), request = capture(f, source, destination.texture, region, at);
  region.min.z = 0; region.max.z = 1; at.x = 0;
  request.issue(); const expected = new Uint8Array(64);
  for (let row = 0; row < 2; row++) expected.set(data.subarray(((row + 1) * 6 + 2) * 4, ((row + 1) * 6 + 4) * 4), ((row + 1) * 4 + 1) * 4);
  assert.deepEqual(await read(f, destination), expected); f.dispose();
});

test('source mip selection uses native mip-local dimensions and preserves destination neighbors', async () => {
  const f = fixture(), source = storageTexture('rgba8unorm', 8, 8), destination = target(); source.generateMipmaps = true;
  seed(f, source, new Uint8Array(16).fill(49), {mipLevel: 2});
  capture(f, source, destination.texture, null, position(1, 1), 2).issue();
  const result = await read(f, destination), expected = new Uint8Array(64);
  expected.fill(49, 20, 28); expected.fill(49, 36, 44); assert.deepEqual(result, expected);
  assert.equal(f.calls.copies[0].source.mipLevel, 2); assert.equal(f.calls.copies[0].destination.mipLevel, 0);
  f.dispose();
});

test('target destination mipmaps are refreshed after the write and not before it', async () => {
  const f = fixture(), source = storageTexture(), destination = target({generateMipmaps: true}); seed(f, source, payload('rgba8unorm'));
  const entry = f.targets.ensure(f.targets.capture(destination));
  capture(f, source, destination.texture).issue();
  assert.equal(entry.storage.diagnostics.mipGenerations, 0); assert.equal(f.calls.passes.length, 0);
  entry.storage.withTexture(() => {}); assert.equal(entry.storage.diagnostics.mipGenerations, 1);
  assert.equal(f.calls.submissions[0][0].label, 'f3d-storage-to-target-copy');
  assert.match(f.calls.submissions[1][0].label, /mip chain/);
  entry.storage.withTexture(() => {}); assert.equal(entry.storage.diagnostics.mipGenerations, 1);
  await f.targets.whenIdle(); f.dispose();
});

test('only primary writes invalidate the existing primary color replica; depth remains unchanged', async () => {
  const f = fixture(), source = storageTexture(), destination = target({colors: ['rgba8unorm', 'rgba8unorm']});
  seed(f, source, payload('rgba8unorm')); const entry = f.targets.capture(destination);
  f.targets.bindingsFor().get(destination.texture); entry.depthDirty = false;
  const copies = entry.copies;
  capture(f, source, destination.textures[1]).issue(); assert.equal(entry.dirty, false); assert.equal(entry.depthDirty, false);
  f.targets.bindingsFor().get(destination.texture); assert.equal(entry.copies, copies);
  capture(f, source, destination.texture).issue(); assert.equal(entry.dirty, true); assert.equal(entry.depthDirty, false);
  f.targets.bindingsFor().get(destination.texture); assert.equal(entry.copies, copies + 1);
  await f.targets.whenIdle(); f.dispose();
});

test('accepted copies capture commands now, observe source at submission, and retain readback history across later writes', async () => {
  const f = fixture(), source = storageTexture(), destination = target();
  const first = capture(f, source, destination.texture); seed(f, source, new Uint8Array(64).fill(12)); first.issue();
  const before = read(f, destination);
  seed(f, source, new Uint8Array(64).fill(63)); capture(f, source, destination.texture).issue();
  const after = read(f, destination); seed(f, source, new Uint8Array(64).fill(99));
  assert.ok((await before).every(value => value === 12)); assert.ok((await after).every(value => value === 63)); f.dispose();
});

test('repeated bidirectional copies preserve data and form acyclic validation dependencies', {timeout: 2000}, async () => {
  const f = fixture(), source = storageTexture(), a = target(), b = target(); seed(f, source, payload('rgba8unorm'));
  for (let i = 0; i < 12; i++) {
    capture(f, source, a.texture).issue(); captureTargetToStorageCopy(f.renderer, a.texture, source).issue();
    capture(f, source, b.texture).issue(); captureTargetToStorageCopy(f.renderer, b.texture, source).issue();
  }
  assert.equal(f.calls.fences, 0); assert.equal(f.calls.buffers.length, 0); assert.equal(f.calls.copies.length, 48);
  assert.deepEqual(await read(f, b), payload('rgba8unorm'));
  await Promise.all([f.targets.whenIdle(), f.storage.whenIdle()]); f.dispose();
});

test('storage producer validation gates both native and source-session readback without blocking submission', async () => {
  const f = fixture(), source = storageTexture(), destination = target(), gate = deferred();
  seed(f, source, new Uint8Array(64).fill(89), {validation: gate.promise}); capture(f, source, destination.texture).issue();
  const entry = f.targets.capture(destination), nativeOutput = new Uint8Array(64).fill(233), sessionOutput = nativeOutput.slice();
  const a = read(f, destination, 0, {output: nativeOutput}), b = readGpuTargetPixels(entry.session, {output: sessionOutput});
  let resolved = false; a.then(() => { resolved = true; }, () => {});
  await flush(); assert.equal(resolved, false); assert.ok(nativeOutput.every(value => value === 233));
  assert.equal(f.calls.fences, 0); assert.equal(f.calls.copies.length, 1);
  gate.resolve(); assert.ok((await a).every(value => value === 89)); assert.ok((await b).every(value => value === 89)); f.dispose();
});

test('producer rejection leaves native and source-session caller output untouched', async () => {
  const f = fixture(), source = storageTexture(), destination = target(), gate = deferred();
  seed(f, source, new Uint8Array(64).fill(89), {validation: gate.promise}); capture(f, source, destination.texture).issue();
  const entry = f.targets.capture(destination), a = new Uint8Array(64).fill(201), b = a.slice();
  const checks = Promise.all([
    assert.rejects(read(f, destination, 0, {output: a}), /bad producer/),
    assert.rejects(readGpuTargetPixels(entry.session, {output: b}), /bad producer/),
  ]);
  gate.reject(new Error('bad producer')); await checks;
  assert.ok(a.every(value => value === 201)); assert.deepEqual(a, b); f.dispose();
  assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
});

test('source-owner disposal interrupts dependent validation instead of leaving a target read hanging', {timeout: 2000}, async () => {
  const f = fixture(), source = storageTexture(), destination = target(), gate = deferred();
  seed(f, source, new Uint8Array(64), {validation: gate.promise}); capture(f, source, destination.texture).issue();
  const output = new Uint8Array(64).fill(222), rejected = assert.rejects(read(f, destination, 0, {output}), /DISPOSED/);
  f.storage.dispose(); await rejected; assert.ok(output.every(value => value === 222)); gate.resolve(); f.dispose();
});

test('the last source borrow scope is included in target validation', async () => {
  const f = fixture(), source = storageTexture(), destination = target(); seed(f, source, payload('rgba8unorm'));
  f.targets.ensure(f.targets.capture(destination)); await Promise.all([f.storage.whenIdle(), f.targets.whenIdle()]);
  const pop = f.device.popErrorScope; let pops = 0;
  f.device.popErrorScope = () => { const result = pop(); return ++pops === 3 ? Promise.resolve({message: 'outer source scope rejected'}) : result; };
  capture(f, source, destination.texture).issue();
  await assert.rejects(read(f, destination), /outer source scope rejected/); f.dispose();
});

for (const action of ['cancel', 'dispose', 'loss']) test(`${action} stops target readback awaiting source validation`, async () => {
  const f = fixture(), source = storageTexture(), destination = target(), gate = deferred(), abort = new AbortController();
  seed(f, source, new Uint8Array(64), {validation: gate.promise}); capture(f, source, destination.texture).issue();
  const output = new Uint8Array(64).fill(184), reading = read(f, destination, 0, {signal: abort.signal, output});
  const rejected = assert.rejects(reading, /cancelled|DISPOSED|lost/);
  if (action === 'cancel') abort.abort(new Error('cancelled'));
  if (action === 'dispose') destination.dispose();
  if (action === 'loss') f.loss.resolve({message: 'lost'});
  await rejected; assert.ok(output.every(value => value === 184)); gate.resolve(); await flush(); f.dispose();
  assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
});

for (const action of ['reorder', 'replace', 'target-dispose', 'source-dispose', 'resize', 'format', 'device'])
  test(`captured ${action} cannot redirect a storage-to-target write`, () => {
    const f = fixture(), source = storageTexture(), destination = target({colors: ['rgba8unorm', 'rgba8unorm']});
    const request = capture(f, source, destination.textures[1]);
    if (action === 'reorder') destination.textures.reverse();
    if (action === 'replace') { const texture = new T.Texture(); texture.isRenderTargetTexture = true; texture.renderTarget = destination; destination.textures[1] = texture; }
    if (action === 'target-dispose') destination.dispose();
    if (action === 'source-dispose') source.dispose();
    if (action === 'resize') source.image.width = 8;
    if (action === 'format') destination.textures[1].type = T.HalfFloatType;
    if (action === 'device') f.renderer._device = {};
    assert.throws(() => request.issue(), /STALE|DEVICE/); assert.equal(f.calls.textures.length, 0); f.dispose();
  });

for (const budget of ['target', 'storage']) test(`${budget} budget fails before allocation of either copy endpoint`, () => {
  const f = fixture(budget === 'target' ? {targetBudget: 127} : {storageBudget: 63});
  const request = capture(f, storageTexture(), target().texture);
  assert.throws(() => request.issue(), /BUDGET/); assert.equal(f.calls.textures.length, 0); f.dispose();
});

test('MSAA destinations, higher target levels, conversions and volume dimensions remain explicit refusals', () => {
  const f = fixture(), source = storageTexture(), destination = target();
  assert.throws(() => capture(f, source, target({samples: 4}).texture), /SAMPLES/);
  assert.throws(() => capture(f, source, target({mips: 2}).texture, null, null, 0, 1), /LEVEL/);
  assert.throws(() => capture(f, source, target({colors: ['rgba16float']}).texture), /FORMAT/);
  const volume = storageTexture(); volume.is3DTexture = true;
  assert.throws(() => capture(f, volume, destination.texture), /DIMENSION/);
  assert.throws(() => capture(f, storageTexture('rgba8unorm', 4, 4, 2), destination.texture), /BOUNDS/);
  assert.equal(f.calls.textures.length, 0); f.dispose();
});

test('empty requests are allocation-free and callbacks cannot reenter either owner', async () => {
  const f = fixture(), source = storageTexture(), destination = target();
  assert.equal(capture(f, source, destination.texture, box(4, 0, 0, 4)).issue(), false);
  assert.equal(f.calls.textures.length, 0);
  const request = capture(f, source, destination.texture);
  f.controls.onCopy = () => assert.throws(() => request.issue(), /REENTRANT/);
  request.issue(); await f.targets.whenIdle(); f.dispose();
});
