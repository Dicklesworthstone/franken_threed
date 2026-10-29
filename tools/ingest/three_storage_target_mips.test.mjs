/** Execute the production planner, capture and FIFO with recording owners. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {captureStorageToTargetCopy} from './three_target_texture_copy.mjs';
import {submitTextureCopy} from './three_texture_copy_lifetime.mjs';
import {targetCopyFixture, box, point, failure} from './three_target_copy_fixtures.mjs';

function setup(dimension = '2d', options = {}) {
  const f = targetCopyFixture();
  const source = f.storageTexture({dimension, width: 8, height: 8, depth: dimension === '2d' ? 1 : 8, mips: 4, ...options});
  const destination = f.target({dimension, width: 8, height: 8, depth: dimension === '2d' ? 1 : 8, mips: 4, ...options});
  for (const color of destination.entry.shape.colors) color.generateMipmaps = false;
  const capture = (region = null, position = null, sourceLevel = 1, destinationLevel = 1) =>
    captureStorageToTargetCopy(f.renderer, source, destination.texture, region, position, sourceLevel, destinationLevel);
  return {...f, f, source, destination, capture, copy(...args) {
    return submitTextureCopy(f.renderer, capture(...args), f.rejected);
  }};
}

for (const dimension of ['2d', '2d-array', '3d']) {
  test(`storage copies address an authored ${dimension} target mip without a mip-zero write`, async () => {
    const {f, source, destination, copy} = setup(dimension);
    assert.equal(copy(), undefined);
    assert.equal(f.commands.length, 1);
    const command = f.commands[0];
    assert.equal(command.from.texture, source.entry.native);
    assert.equal(command.to.texture, destination.entry.native[0]);
    assert.equal(command.from.mipLevel, 1); assert.equal(command.to.mipLevel, 1);
    assert.deepEqual(command.size, [4, 4, dimension === '2d' ? 1 : dimension === '3d' ? 4 : 8]);
    assert.equal(f.references, 0); assert.equal(f.releases, 1);
    await Promise.all(f.validations);
  });
}

for (const format of ['r8unorm', 'rg11b10ufloat', 'rgba16float', 'rgba32float']) {
  test(`authored target mip copies retain ${format} native storage`, () => {
    const {f, copy} = setup('2d', {format});
    copy(); assert.equal(f.commands.length, 1);
    assert.deepEqual(f.commands[0].size, [4, 4, 1]);
    assert.ok(!f.events.includes('source-borrow'));
  });
}

test('independent mips and immutable XYZ ranges survive queued caller mutations', () => {
  const {f, destination, copy} = setup('3d'); f.renderer._preparing = true;
  const region = box(1, 0, 1, 1, 2, 1), position = point(0, 0, 1);
  copy(region, position, 1, 2);
  region.min.x = 99; region.max.z = 99; position.z = 99;
  destination.texture.generateMipmaps = true; // Public edits cannot retarget the captured shape.
  assert.equal(f.references, 1); assert.equal(f.commands.length, 0);
  f.flush(); assert.deepEqual(f.errors, []);
  const command = f.commands[0];
  assert.deepEqual(command.from.origin, [1, 0, 1]); assert.deepEqual(command.to.origin, [0, 0, 1]);
  assert.deepEqual(command.size, [1, 2, 1]);
  assert.equal(command.from.mipLevel, 1); assert.equal(command.to.mipLevel, 2);
  assert.equal(f.references, 0);
});

test('NPOT volume tails floor-halve all three destination dimensions independently', () => {
  const {f, copy} = setup('3d', {width: 7, height: 3, depth: 5, mips: 3});
  copy(null, null, 2, 2);
  assert.deepEqual(f.commands[0].size, [1, 1, 1]); assert.equal(f.commands[0].to.mipLevel, 2);
});

test('authored base writes remain admitted and select mip zero', () => {
  const {f, copy} = setup(); copy(null, null, 1, 0);
  assert.equal(f.commands[0].to.mipLevel, 0); assert.deepEqual(f.commands[0].size, [4, 4, 1]);
});

test('generated or unproven chains reject nonzero writes before FIFO admission', () => {
  for (const generateMipmaps of [true, undefined]) {
    const {f, destination, copy} = setup(); destination.entry.shape.colors[0].generateMipmaps = generateMipmaps;
    f.renderer._preparing = true;
    assert.throws(() => copy(), {code: 'THREE_STORAGE_COPY_LEVEL'});
    assert.equal(f.references, 0); assert.equal(f.renderer._pending.length, 0); assert.deepEqual(f.events, []);
  }
});

test('mip-local volume destination bounds reject a slice valid only at level zero', () => {
  const {f, copy} = setup('3d');
  assert.throws(() => copy(box(0, 0, 0, 1, 1, 1), point(0, 0, 2), 1, 2), {code: 'THREE_STORAGE_COPY_BOUNDS'});
  assert.equal(f.references, 0); assert.deepEqual(f.events, []);
});

test('invalid destination levels and mip-local rectangles do not allocate or leak', () => {
  for (const level of [-1, 0.5, NaN, Infinity, 4, Number.MAX_SAFE_INTEGER]) {
    const {f, copy} = setup();
    assert.throws(() => copy(null, null, 1, level), {code: 'THREE_STORAGE_COPY_LEVEL'});
    assert.equal(f.references, 0); assert.deepEqual(f.events, []);
  }
  const {f, copy} = setup();
  assert.throws(() => copy(null, null, 0, 1), {code: 'THREE_STORAGE_COPY_BOUNDS'});
  assert.equal(f.references, 0); assert.deepEqual(f.events, []);
});

test('empty authored-mip copies release without joining a blocked FIFO', () => {
  const {f, copy} = setup(); f.renderer._preparing = true;
  copy(box(0, 0, 0, 0, 4, 1));
  assert.equal(f.references, 0); assert.equal(f.renderer._pending.length, 0); assert.deepEqual(f.events, []);
});

test('whole-volume depth and array limits precede any storage borrow', () => {
  for (const change of [
    ({f, destination}) => { destination.entry.shape.depthFormat = 'depth32float'; f.device.limits.maxTextureDimension2D = 4; },
    ({f, destination}) => { destination.entry.shape.depthFormat = 'depth32float'; delete f.device.limits.maxTextureDimension2D; },
    ({f, destination}) => { Object.assign(destination.entry.shape, {depthFormat: 'depth32float', depthArray: true, depthLayers: 8}); f.device.limits.maxTextureArrayLayers = 4; },
    ({f}) => { f.device.limits.maxTextureDimension3D = 4; },
    ({f}) => { f.storage.budgetError = failure('STORAGE_BUDGET'); },
    ({f}) => { f.targets.budgetError = failure('TARGET_BUDGET'); },
  ]) {
    const setupResult = setup('3d'); change(setupResult);
    assert.throws(() => setupResult.copy(box(0, 0, 0, 1, 1, 1)));
    const {f} = setupResult;
    assert.equal(f.references, 0); assert.equal(f.commands.length, 0);
    assert.ok(!f.events.includes('storage-read')); assert.ok(!f.events.includes('target-write'));
  }
});

test('missing array-layer limits reject before native residency is borrowed', () => {
  const {f, copy} = setup('2d-array'); delete f.device.limits.maxTextureArrayLayers;
  assert.throws(() => copy(), {code: 'THREE_STORAGE_COPY_LIMIT'});
  assert.equal(f.references, 0); assert.ok(!f.events.includes('storage-read'));
});

test('queue rejection, target disposal and renderer cancellation release exactly once', () => {
  for (const cancel of [
    ({f}) => f.renderer._pending[0].reject(failure('EARLIER')),
    ({destination}) => destination.dispose(),
    ({f}) => f.renderer._targetLifetime.abort(failure('CANCELLED')),
  ]) {
    const value = setup(); value.f.renderer._preparing = true; value.copy(); cancel(value);
    assert.equal(value.f.references, 0); value.f.flush();
    assert.equal(value.f.releases, 1); assert.equal(value.f.commands.length, 0); assert.equal(value.f.errors.length, 1);
  }
});

test('queued replacement and device migration reject before target writes', () => {
  for (const mutate of [
    ({source}) => { source.shape = {...source.shape}; },
    ({destination}) => { destination.entry.target.epoch++; },
    ({destination}) => { destination.entry.shape.textures[0] = {}; },
    ({f}) => { f.renderer._device = {}; },
  ]) {
    const value = setup(); value.f.renderer._preparing = true; value.copy(); mutate(value); value.f.flush();
    assert.equal(value.f.references, 0); assert.equal(value.f.commands.length, 0); assert.equal(value.f.errors.length, 1);
    assert.ok(!value.f.events.includes('target-write'));
  }
});

test('producer validation is captured after its borrow and gates destination completion', async () => {
  const {f, copy} = setup();
  const original = f.storage.withTexture;
  const error = failure('PRODUCER_GPU');
  f.storage.withTexture = (...args) => {
    const result = original(...args);
    f.storage.completion = Promise.reject(error); f.storage.completion.catch(() => {}); return result;
  };
  copy(); assert.equal(f.commands.length, 1);
  await assert.rejects(Promise.all(f.validations), error);
  assert.equal(f.references, 0);
});

test('validation snapshots are not redirected to later producer work', async () => {
  const {f, copy} = setup();
  copy(); f.storage.completion = new Promise(() => {});
  await Promise.all(f.validations);
  assert.equal(f.references, 0);
});

test('encoder and submission errors release retained authored-mip destinations', () => {
  for (const stage of ['onFinish', 'onSubmit']) {
    const {f, copy} = setup(); f.device[stage] = () => { throw failure('GPU'); };
    assert.throws(() => copy(), {code: 'GPU'});
    assert.equal(f.references, 0); assert.equal(f.commands.length, 0);
  }
});

test('generation checks run after encoder completion and before submission', () => {
  const {f, destination, copy} = setup();
  f.device.onFinish = () => { destination.entry.target.epoch++; };
  assert.throws(() => copy(), {code: 'STALE_TARGET'});
  assert.equal(f.references, 0); assert.equal(f.commands.length, 0);
});

test('nested compute turns retain a mip write until its exact submission position', async () => {
  const {f, copy} = setup(); let issue;
  f.renderer._computeTurn = {submit(run) { issue = run; return Promise.resolve(); }};
  copy(); assert.equal(f.references, 1); assert.equal(f.commands.length, 0);
  issue(); assert.equal(f.references, 0); assert.equal(f.commands[0].to.mipLevel, 1);
});

test('MSAA and cube destinations remain explicit unsupported operations', () => {
  for (const options of [{samples: 4}, {dimension: 'cube'}]) {
    const {f, copy} = setup('2d', options);
    assert.throws(() => copy()); assert.equal(f.references, 0); assert.deepEqual(f.events, []);
  }
});
