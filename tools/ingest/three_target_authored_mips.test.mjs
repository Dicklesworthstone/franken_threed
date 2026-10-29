/** Production capture/planning/FIFO with recording owner/device contracts. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {captureTargetColorCopy} from './three_target_texture_copy.mjs';
import {submitTextureCopy} from './three_texture_copy_lifetime.mjs';
import {targetCopyFixture, box, point, failure} from './three_target_copy_fixtures.mjs';

function setup(sourceOptions = {}, destinationOptions = sourceOptions) {
  const f = targetCopyFixture();
  const source = f.target({width: 8, height: 8, mips: 4, ...sourceOptions});
  const destination = f.target({width: 8, height: 8, mips: 4, ...destinationOptions});
  destination.entry.shape.colors[0].generateMipmaps = false;
  const capture = (region = null, position = null, sourceLevel = 1, destinationLevel = 1) =>
    captureTargetColorCopy(f.renderer, source.texture, destination.texture, region, position, sourceLevel, destinationLevel);
  return {f, source, destination, capture, copy(...args) { return submitTextureCopy(f.renderer, capture(...args), f.rejected); }};
}

for (const dimension of ['2d', '2d-array', '3d']) {
  test(`target-to-target ${dimension} copies select an authored destination mip in one command`, async () => {
    const {f, source, destination, copy} = setup({dimension, depth: dimension === '2d' ? 1 : 8});
    assert.equal(copy(), undefined);
    assert.equal(f.commands.length, 1); const command = f.commands[0];
    assert.equal(command.from.texture, source.entry.native[0]); assert.equal(command.to.texture, destination.entry.native[0]);
    assert.equal(command.from.mipLevel, 1); assert.equal(command.to.mipLevel, 1);
    assert.deepEqual(command.size, [4, 4, dimension === '2d' ? 1 : dimension === '3d' ? 4 : 8]);
    assert.ok(!f.events.includes('target-copy')); assert.equal(f.references, 0); assert.equal(f.releases, 2);
    await Promise.all(f.validations);
  });
}

for (const dimension of ['2d-array', '3d']) {
  test(`cube faces copy into authored ${dimension} mip slices without sampling or CPU conversion`, () => {
    const {f, source, copy} = setup({dimension: 'cube', pointer: false}, {dimension, depth: 8});
    copy(box(0, 0, 3, 2, 2, 2), point(0, 0, 0), 2, 2);
    assert.equal(f.commands.length, 1); const command = f.commands[0];
    assert.equal(command.from.texture, source.entry.native[0]);
    assert.deepEqual(command.from.origin, [0, 0, 3]); assert.deepEqual(command.to.origin, [0, 0, 0]);
    assert.deepEqual(command.size, [2, 2, 2]); assert.equal(command.to.mipLevel, 2);
    assert.equal(f.references, 0);
  });
}

test('one cube face can feed a nonzero 2D target mip', () => {
  const {f, copy} = setup({dimension: 'cube'}, {dimension: '2d'});
  copy(box(0, 0, 5, 2, 2, 1), null, 2, 2);
  assert.deepEqual(f.commands[0].from.origin, [0, 0, 5]); assert.deepEqual(f.commands[0].size, [2, 2, 1]);
});

test('independent source/destination mips retain native sRGB-copy compatibility', () => {
  const {f, copy} = setup({format: 'rgba8unorm-srgb'}, {width: 16, height: 16, mips: 5});
  copy(null, null, 1, 2);
  assert.equal(f.commands[0].from.mipLevel, 1); assert.equal(f.commands[0].to.mipLevel, 2);
  assert.deepEqual(f.commands[0].size, [4, 4, 1]);
});

test('resolved multisample source can feed an authored single-sample target mip', () => {
  const {f, source, copy} = setup({samples: 4, mips: 1}, {width: 16, height: 16, mips: 5});
  copy(null, null, 0, 1);
  assert.equal(f.commands[0].from.texture, source.entry.native[0]); assert.equal(f.commands[0].to.mipLevel, 1);
});

test('raw HDR/float copies need no sampler or float32-filterable feature', () => {
  for (const format of ['rg11b10ufloat', 'r32float', 'rg32float', 'rgba32float']) {
    const {f, copy} = setup({format}); copy();
    assert.equal(f.commands[0].to.mipLevel, 1); assert.equal(f.references, 0);
  }
});

test('NPOT volumes use mip-local XYZ extents, including one-texel tails', () => {
  const {f, copy} = setup({dimension: '3d', width: 7, height: 3, depth: 5, mips: 3});
  copy(null, null, 2, 2); assert.deepEqual(f.commands[0].size, [1, 1, 1]);
});

test('destination volume Z bounds shrink while array layer counts remain fixed', () => {
  for (const dimension of ['3d', '2d-array']) {
    const {f, copy} = setup({dimension, depth: 8});
    const run = () => copy(box(0, 0, 0, 1, 1, 1), point(0, 0, 3), 1, 2);
    if (dimension === '3d') { assert.throws(run, {code: 'THREE_STORAGE_COPY_BOUNDS'}); assert.equal(f.commands.length, 0); }
    else { run(); assert.equal(f.commands[0].to.origin[2], 3); }
    assert.equal(f.references, 0);
  }
});

test('capture freezes bounds and attachments while GPU contents are read at submission', () => {
  const {f, source, copy} = setup({dimension: '2d-array', depth: 8}); f.renderer._preparing = true;
  const region = box(1, 1, 3, 2, 2, 2), position = point(0, 0, 1);
  copy(region, position); region.min.z = 7; region.max.x = 99; position.z = 7;
  source.entry.native[0].stamp = 42;
  f.device.onSubmit = () => assert.equal(source.entry.native[0].stamp, 42);
  assert.equal(f.references, 2); f.flush();
  assert.deepEqual(f.commands[0].from.origin, [1, 1, 3]); assert.deepEqual(f.commands[0].to.origin, [0, 0, 1]);
  assert.deepEqual(f.commands[0].size, [2, 2, 2]); assert.equal(f.references, 0); assert.deepEqual(f.errors, []);
});

test('generated and unknown chains, cube mips and multisample destinations reject before scheduling', () => {
  for (const change of [
    ({destination}) => { destination.entry.shape.colors[0].generateMipmaps = true; },
    ({destination}) => { delete destination.entry.shape.colors[0].generateMipmaps; },
    ({destination}) => { destination.entry.shape.cube = true; },
    ({destination}) => { destination.entry.shape.sampleCount = 4; },
  ]) {
    const value = setup(); change(value); value.f.renderer._preparing = true;
    assert.throws(() => value.copy()); assert.equal(value.f.references, 0); assert.deepEqual(value.f.events, []);
  }
});

test('malformed destination levels and incompatible formats are side-effect free', () => {
  for (const level of [-1, 0.5, NaN, Infinity, 4]) {
    const {f, copy} = setup();
    assert.throws(() => copy(null, null, 1, level), {code: 'THREE_STORAGE_COPY_LEVEL'});
    assert.equal(f.references, 0); assert.deepEqual(f.events, []);
  }
  const {f, copy} = setup({}, {format: 'rgba16float'});
  assert.throws(() => copy(), {code: 'THREE_STORAGE_COPY_FORMAT'}); assert.equal(f.references, 0);
});

test('in-place and history-alias copies stay rejected even for distinct authored mips', () => {
  const {f, source, destination, copy} = setup();
  assert.throws(() => captureTargetColorCopy(f.renderer, source.texture, source.texture, null, null, 1, 2), {code: 'THREE_STORAGE_COPY_FEEDBACK'});
  source.entry.historyGroup = destination.entry.historyGroup = {};
  assert.throws(() => copy(), {code: 'THREE_STORAGE_COPY_FEEDBACK'});
  assert.equal(f.references, 0); assert.deepEqual(f.events, []);
});

test('empty authored copies release both generations without waiting for the queue', () => {
  const {f, copy} = setup(); f.renderer._preparing = true;
  copy(box(0, 0, 0, 0, 4, 1));
  assert.equal(f.references, 0); assert.equal(f.releases, 2); assert.equal(f.renderer._pending.length, 0);
});

test('combined allocation budget is checked before either native texture is borrowed', () => {
  const {f, copy} = setup(); const validate = f.targets.validateNativeTextures;
  f.targets.validateNativeTextures = sources => { validate(sources); if (sources.length === 2) throw failure('COMBINED'); };
  assert.throws(() => copy(), {code: 'COMBINED'});
  assert.equal(f.references, 0); assert.ok(!f.events.includes('source-borrow')); assert.ok(!f.events.includes('target-write'));
});

test('per-slice transfer byte budgets apply to nonzero destination mips', () => {
  const {f, copy} = setup({width: 4096, height: 4096, format: 'rgba32float'},
    {width: 8192, height: 8192, format: 'rgba32float'});
  assert.throws(() => copy(null, null, 0, 1), {code: 'GPU_TRANSFER_BUDGET'});
  assert.equal(f.references, 0); assert.deepEqual(f.events, []);
});

test('whole volume/depth allocations must fit even when copying a tiny mip', () => {
  for (const endpoint of ['source', 'destination']) {
    const value = setup({dimension: '3d', depth: 8});
    value[endpoint].entry.shape.depthFormat = 'depth32float'; value.f.device.limits.maxTextureDimension2D = 4;
    assert.throws(() => value.copy(box(0, 0, 0, 1, 1, 1)), {code: 'THREE_STORAGE_COPY_LIMIT'});
    assert.equal(value.f.references, 0); assert.ok(!value.f.events.includes('source-borrow'));
  }
});

test('source GPU validation is captured after borrowing generated source mips', async () => {
  const {f, source, copy} = setup(); const error = failure('SOURCE_MIPS');
  const borrow = f.targets.withNativeTexture;
  f.targets.withNativeTexture = (...args) => {
    const result = borrow(...args);
    source.entry.storage.validation = Promise.reject(error); source.entry.storage.validation.catch(() => {}); return result;
  };
  copy(); await assert.rejects(Promise.all(f.validations), error);
  assert.equal(f.commands.length, 1); assert.equal(f.references, 0);
});

test('source entry validation gates destination completion independently of native validation', async () => {
  const {f, source, copy} = setup(); const error = failure('SOURCE_ENTRY');
  source.entry.validation = Promise.reject(error); source.entry.validation.catch(() => {});
  copy(); await assert.rejects(Promise.all(f.validations), error);
});

test('later reverse-copy validation cannot create a dependency cycle', async () => {
  const {f, source, copy} = setup(); copy();
  source.entry.storage.validation = Promise.all(f.validations);
  await Promise.all(f.validations);
});

test('FIFO failures and disposal release both retained generations exactly once', () => {
  for (const cancel of [
    ({f}) => f.renderer._pending[0].reject(failure('EARLIER')),
    ({source}) => source.dispose(), ({destination}) => destination.dispose(),
    ({f}) => f.renderer._targetLifetime.abort(failure('CANCELLED')),
  ]) {
    const value = setup(); value.f.renderer._preparing = true; value.copy(); cancel(value);
    assert.equal(value.f.references, 0); value.f.flush();
    assert.equal(value.f.releases, 2); assert.equal(value.f.commands.length, 0); assert.equal(value.f.errors.length, 1);
  }
});

test('encoder/submission failure and generation changes never leak a captured target', () => {
  for (const stage of ['onFinish', 'onSubmit']) for (const queued of [false, true]) {
    const {f, copy} = setup(); f.renderer._preparing = queued; f.device[stage] = () => { throw failure('GPU'); };
    if (queued) { copy(); f.flush(); assert.equal(f.errors[0].code, 'GPU'); }
    else assert.throws(() => copy(), {code: 'GPU'});
    assert.equal(f.commands.length, 0); assert.equal(f.references, 0); assert.equal(f.releases, 2);
  }
  const {f, destination, copy} = setup(); f.device.onFinish = () => { destination.entry.target.epoch++; };
  assert.throws(() => copy(), {code: 'STALE_TARGET'}); assert.equal(f.commands.length, 0); assert.equal(f.references, 0);
});

test('nested compute admission retains one captured range, not one request per layer', async () => {
  const {f, copy} = setup({dimension: '2d-array', depth: 8}); let issue, turns = 0;
  f.renderer._computeTurn = {submit(run) { turns++; issue = run; return Promise.resolve(); }};
  copy(); assert.equal(turns, 1); assert.equal(f.references, 2); assert.equal(f.commands.length, 0);
  issue(); assert.equal(f.commands.length, 1); assert.deepEqual(f.commands[0].size, [4, 4, 8]);
  assert.equal(f.references, 0); assert.equal(f.renderer._pending.length, 0);
});

test('level-zero cube and generated-mip copies retain the existing face-aware path', () => {
  const {f, destination, copy} = setup({dimension: 'cube'});
  destination.entry.shape.colors[0].generateMipmaps = true;
  copy(null, null, 0, 0);
  assert.equal(f.commands.length, 6); assert.ok(f.events.includes('target-copy'));
  assert.ok(!f.events.includes('target-write')); assert.equal(f.references, 0);
});
