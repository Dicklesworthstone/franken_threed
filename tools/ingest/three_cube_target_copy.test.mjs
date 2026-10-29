import test from 'node:test';
import assert from 'node:assert/strict';
import {cubeCopyFixture, box, point, pattern, checkRegion, deferred, failure} from './three_cube_target_copy_fixtures.mjs';

for (const level of [0, 1, 2]) test(`storage writes all six cube faces at authored mip ${level} through both production owners`, async () => {
  const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
  pattern(source.entry.native);
  assert.equal(f.copy(source, destination, null, null, level, level), undefined);
  assert.equal(f.submissions, 1); assert.equal(f.commands.length, 1);
  assert.equal(f.commands[0].to.mipLevel, level); assert.equal(f.commands[0].size[2], 6);
  const native = f.native(destination);
  checkRegion(source.entry.native, native, {sourceLevel: level, destinationLevel: level});
  assert.equal(destination.entry.references, 0); assert.equal(destination.texture.renderTarget, null);
  assert.equal(destination.entry.storage.diagnostics.mipGenerations, 0);
  assert.equal(f.textures.length, 2); assert.equal(f.targets.allocatedBytes, 504);
  await f.targets.whenIdle(); f.targets.dispose(); assert.equal(native.destroyed, 1);
});

for (const dimension of ['2d', '2d-array', '3d', 'cube']) test(`${dimension} target colors fill independently selected authored cube mips`, async () => {
  const f = cubeCopyFixture(), source = f.target({dimension, width: 8, depth: 6, mips: 4}), destination = f.target();
  const src = f.native(source); pattern(src);
  const count = dimension === '2d' ? 1 : 2;
  f.copy(source, destination, box(1, 1, 0, 2, 2, count), point(0, 0, 4), 1, 1);
  assert.equal(f.submissions, 1); assert.equal(f.commands.length, 1);
  checkRegion(src, f.native(destination), {sourceLevel: 1, destinationLevel: 1,
    sourceOrigin: [1, 1, 0], destinationOrigin: [0, 0, 4], size: [2, 2, count]});
  assert.equal(source.entry.references + destination.entry.references, 0);
  await f.targets.whenIdle(); f.targets.dispose();
});

for (const format of ['r8unorm', 'r16float', 'rg16float', 'rgba16float', 'r32float', 'rg32float', 'rgba32float', 'rg11b10ufloat'])
  test(`${format} storage-to-cube preserves raw bytes without filtering`, async () => {
    const f = cubeCopyFixture(); f.device.features.add('rg11b10ufloat-renderable');
    const destination = f.target({format}), source = f.storageTexture({format}); pattern(source.entry.native);
    f.copy(source, destination, box(0, 0, 1, 1, 2, 3), point(1, 0, 2), 1, 1);
    checkRegion(source.entry.native, f.native(destination), {sourceLevel: 1, destinationLevel: 1,
      sourceOrigin: [0, 0, 1], destinationOrigin: [1, 0, 2], size: [1, 2, 3]});
    await f.targets.whenIdle(); f.targets.dispose();
  });

test('NPOT one-texel cube tails retain all faces and source mip selection', async () => {
  const f = cubeCopyFixture(), destination = f.target({width: 7}), source = f.storageTexture({width: 3, mips: 2});
  pattern(source.entry.native); f.copy(source, destination, null, null, 1, 2);
  checkRegion(source.entry.native, f.native(destination), {sourceLevel: 1, destinationLevel: 2});
  assert.deepEqual(f.commands[0].size, [1, 1, 6]); await f.targets.whenIdle(); f.targets.dispose();
});

test('queued cube writes capture regions and generation policy but read submission-time pixels', async () => {
  const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
  const region = box(0, 0, 1, 2, 2, 2), position = point(0, 0, 4);
  f.renderer._preparing = true; f.copy(source, destination, region, position, 1, 1);
  assert.equal(f.renderer._pending.length, 1); assert.equal(destination.entry.references, 1);
  region.min.z = 99; position.z = 99; destination.texture.generateMipmaps = true;
  pattern(source.entry.native, 19); f.flush(); assert.deepEqual(f.errors, []);
  checkRegion(source.entry.native, f.native(destination), {sourceLevel: 1, destinationLevel: 1,
    sourceOrigin: [0, 0, 1], destinationOrigin: [0, 0, 4], size: [2, 2, 2]});
  assert.equal(destination.entry.references, 0); await f.targets.whenIdle(); f.targets.dispose();
});

test('cube policy is captured per call rather than inherited from an earlier allocation observation', async () => {
  const f = cubeCopyFixture(), destination = f.target({generateMipmaps: true}), source = f.storageTexture();
  f.targets.capture(destination.target, {initialize: true}); destination.texture.generateMipmaps = false;
  pattern(source.entry.native); f.copy(source, destination, null, null, 1, 1);
  checkRegion(source.entry.native, f.native(destination), {sourceLevel: 1, destinationLevel: 1});
  destination.texture.generateMipmaps = true;
  assert.throws(() => f.copy(source, destination, null, null, 1, 1), {code: 'THREE_STORAGE_COPY_LEVEL'});
  assert.equal(f.submissions, 1); assert.equal(destination.entry.references, 0);
  await f.targets.whenIdle(); f.targets.dispose();
});

test('generated higher cube mips reject before allocation and release both endpoints', () => {
  for (const storage of [true, false]) {
    const f = cubeCopyFixture(), destination = f.target({generateMipmaps: true});
    const source = storage ? f.storageTexture() : f.target();
    const allocations = f.textures.length;
    assert.throws(() => f.copy(source, destination, null, null, 1, 1), {code: 'THREE_STORAGE_COPY_LEVEL'});
    assert.equal(f.textures.length, allocations); assert.equal(f.commands.length, 0);
    assert.equal(destination.entry.references, 0); if (!storage) assert.equal(source.entry.references, 0);
    f.targets.dispose();
  }
});

test('bad final face, unknown mip and out-of-bounds XY reject the whole range before writes', () => {
  for (const args of [[box(0, 0, 0, 2, 2, 3), point(0, 0, 4), 1, 1],
    [null, null, 1, 3], [box(0, 0, 0, 2, 2, 2), point(1, 0, 0), 1, 1]]) {
    const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
    assert.throws(() => f.copy(source, destination, ...args));
    assert.equal(f.textures.length, 1); assert.equal(f.submissions, 0); assert.equal(destination.entry.references, 0);
    f.targets.dispose();
  }
});

test('empty cube ranges take no FIFO slot and allocate no destination', () => {
  const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
  f.renderer._preparing = true;
  f.copy(source, destination, box(0, 0, 6, 1, 1, 0), point(0, 0, 6));
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.textures.length, 1); assert.equal(destination.entry.references, 0);
  f.targets.dispose();
});

test('cube allocation budget and device layer limit reject before borrowing storage', () => {
  for (const change of [f => { f.device.limits.maxTextureArrayLayers = 5; },
    f => { f.device.limits.maxTextureDimension2D = 2; }, f => { f.storage.budgetError = failure('BUDGET'); }]) {
    const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture(); change(f);
    assert.throws(() => f.copy(source, destination)); assert.equal(f.textures.length, 1);
    assert.equal(f.events.length, 0); assert.equal(destination.entry.references, 0); f.targets.dispose();
  }
  const f = cubeCopyFixture({maxBytes: 503}), destination = f.target(), source = f.storageTexture();
  assert.throws(() => f.copy(source, destination), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.textures.length, 1); assert.equal(destination.entry.references, 0); f.targets.dispose();
});

test('full queues and earlier queue failure release captured cube generations', () => {
  for (const admission of [true, false]) {
    const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture(); f.renderer._preparing = true;
    if (admission) { f.renderer.queueError = failure('FULL'); assert.throws(() => f.copy(source, destination), {code: 'FULL'}); }
    else { f.copy(source, destination); f.renderer._pending[0].reject(failure('EARLIER')); f.flush(); assert.equal(f.errors.length, 1); }
    assert.equal(destination.entry.references, 0); assert.equal(f.commands.length, 0); f.targets.dispose();
  }
});

test('target disposal and renderer cancellation release stalled cube writes immediately', () => {
  for (const renderer of [true, false]) {
    const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture(); f.renderer._preparing = true;
    f.copy(source, destination);
    if (renderer) f.renderer._targetLifetime.abort(failure('CANCELLED')); else destination.dispose();
    assert.equal(destination.entry.references, 0); f.flush();
    assert.equal(f.commands.length, 0); assert.equal(f.errors.length, 1); assert.equal(f.textures.length, 1); f.targets.dispose();
  }
});

test('nested compute turns retain cube ownership until their actual submission position', async () => {
  const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
  let issue; const turn = deferred();
  f.renderer._computeTurn = {submit(run) { issue = run; return turn.promise; }};
  f.copy(source, destination); assert.equal(f.submissions, 0); assert.equal(destination.entry.references, 1);
  pattern(source.entry.native); issue(); turn.resolve(); await turn.promise;
  checkRegion(source.entry.native, f.native(destination));
  assert.equal(destination.entry.references, 0); await f.targets.whenIdle(); f.targets.dispose();
});

test('source validation rejects native cube completion without blocking submission', async () => {
  const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture(), pending = deferred();
  f.storage.completion = pending.promise; f.copy(source, destination);
  assert.equal(f.submissions, 1); const complete = f.targets.whenIdle();
  pending.reject(failure('SOURCE_GPU')); await assert.rejects(complete, {code: 'SOURCE_GPU'});
  assert.equal(destination.entry.references, 0); f.targets.dispose();
});

test('native validation errors and submission errors fail the same cube owner', async () => {
  for (const stage of ['scope', 'submit', 'finish']) {
    const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
    if (stage === 'scope') f.device.scopeError = {message: 'invalid cube copy'};
    else f.device[stage === 'submit' ? 'onSubmit' : 'onFinish'] = () => { throw failure('ENCODING'); };
    if (stage === 'scope') f.copy(source, destination);
    else assert.throws(() => f.copy(source, destination), {code: 'ENCODING'});
    await assert.rejects(f.targets.whenIdle());
    assert.equal(destination.entry.references, 0); assert.equal(destination.entry.busy, false); assert.equal(f.scopes.length, 0);
    f.targets.dispose();
  }
});

test('raw cube destinations remain nonreentrant and defer disposal until the borrow ends', () => {
  const f = cubeCopyFixture(), destination = f.target();
  const source = {entry: destination.entry, texture: destination.texture, device: f.device, depth: false, colorIndex: 0};
  let native;
  assert.throws(() => f.targets.withNativeColorDestination(source, texture => {
    native = texture;
    assert.throws(() => f.targets.withNativeColorDestination(source, () => {}), {code: 'THREE_TARGET_REENTRANT'});
    destination.dispose(); assert.equal(texture.destroyed, 0);
  }), {code: 'THREE_TARGET_STALE'});
  assert.equal(native.destroyed, 1); assert.equal(f.targets.allocatedBytes, 0); assert.equal(destination.entry.busy, false);
  f.targets.dispose();
});

test('cube color mip copies leave every private depth face unchanged', async () => {
  const f = cubeCopyFixture(), destination = f.target({depthBuffer: true}), source = f.storageTexture();
  pattern(source.entry.native); f.copy(source, destination, null, null, 1, 1);
  const depth = f.textures.find(texture => texture.descriptor.format.startsWith('depth'));
  assert.ok(depth); assert.equal(depth.descriptor.size.depthOrArrayLayers, 6);
  assert.ok(depth.levels[0].data.every(byte => byte === 213));
  assert.equal(f.textures.length, 3); checkRegion(source.entry.native, f.native(destination), {sourceLevel: 1, destinationLevel: 1});
  await f.targets.whenIdle(); f.targets.dispose();
});

test('MRT source selection writes the chosen color into an authored cube face', async () => {
  const f = cubeCopyFixture(), source = f.target({dimension: '2d', count: 2, mips: 1}), destination = f.target();
  f.targets.ensure(source.entry); source.texture = source.target.textures[1];
  let src; source.entry.storage.withTexture(texture => { src = texture; pattern(texture); }, {colorIndex: 1});
  f.copy(source, destination, box(1, 0, 0, 2, 2, 1), point(0, 0, 5), 0, 1);
  assert.equal(f.commands[0].from.texture, src);
  checkRegion(src, f.native(destination), {destinationLevel: 1, sourceOrigin: [1, 0, 0], destinationOrigin: [0, 0, 5], size: [2, 2, 1]});
  await f.targets.whenIdle(); f.targets.dispose();
});

test('copy-compatible sRGB target pairs retain cube bytes without color conversion', async () => {
  const f = cubeCopyFixture(), source = f.target({format: 'rgba8unorm-srgb'}), destination = f.target();
  const src = f.native(source); pattern(src); f.copy(source, destination, null, null, 1, 1);
  checkRegion(src, f.native(destination), {sourceLevel: 1, destinationLevel: 1});
  await f.targets.whenIdle(); f.targets.dispose();
});

test('cube version changes reject queued writes before allocating destination storage', () => {
  const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
  f.renderer._preparing = true; f.copy(source, destination); destination.texture.version++;
  f.flush(); assert.equal(f.errors[0].code, 'THREE_TARGET_STALE');
  assert.equal(f.textures.length, 1); assert.equal(f.submissions, 0); assert.equal(destination.entry.references, 0);
  f.targets.dispose();
});

test('disposal at encoder completion prevents submission and safely unwinds the native borrow', async () => {
  const f = cubeCopyFixture(), destination = f.target(), source = f.storageTexture();
  f.device.onFinish = () => destination.dispose();
  assert.throws(() => f.copy(source, destination), {code: 'THREE_TARGET_STALE'});
  assert.equal(f.submissions, 0); assert.equal(destination.entry.references, 0); assert.equal(f.targets.allocatedBytes, 0);
  assert.equal(f.textures[1].destroyed, 1); assert.equal(f.scopes.length, 0); f.targets.dispose();
});

test('raw cube writes cannot keep a native borrow across an await', () => {
  const f = cubeCopyFixture(), destination = f.target();
  const source = {entry: destination.entry, texture: destination.texture, device: f.device, depth: false, colorIndex: 0};
  assert.throws(() => f.targets.withNativeColorDestination(source, async () => {}), {code: 'GPU_TARGET_ASYNC_USE'});
  assert.equal(destination.entry.busy, false); assert.equal(f.scopes.length, 0); f.targets.dispose();
});

test('a target-to-cube copy and reverse authored copy finish without validation cycles', async () => {
  const f = cubeCopyFixture(), source = f.target(), destination = f.target();
  pattern(f.native(source)); f.copy(source, destination, null, null, 1, 1);
  f.copy(destination, source, null, null, 1, 1);
  assert.equal(f.submissions, 2);
  await Promise.race([f.targets.whenIdle(), new Promise((_, reject) => {
    const timer = setTimeout(() => reject(failure('VALIDATION_CYCLE')), 500); timer.unref();
  })]);
  assert.equal(source.entry.references + destination.entry.references, 0); f.targets.dispose();
});

for (const kind of ['storage', 'target']) test(`public copyTextureToTexture routes ${kind} into authored cube mips`, async () => {
  const {publicRenderer} = await import('./three_cube_target_copy_fixtures.mjs');
  const f = cubeCopyFixture(), renderer = publicRenderer(f), destination = f.target();
  const source = kind === 'storage' ? f.storageTexture() : f.target();
  const native = kind === 'storage' ? source.entry.native : f.native(source); pattern(native);
  const texture = kind === 'storage' ? source : source.texture;
  assert.equal(renderer.copyTextureToTexture(texture, destination.texture, null, null, 1, 1), undefined);
  checkRegion(native, f.native(destination), {sourceLevel: 1, destinationLevel: 1});
  assert.deepEqual(renderer.writeChecks, [destination.texture]); assert.equal(renderer.inspections.length, 1);
  await renderer.waitForGPU(); renderer.dispose();
});

test('public waitForGPU reports a cancelled cube request without allocating its destination', async () => {
  const {publicRenderer} = await import('./three_cube_target_copy_fixtures.mjs');
  const f = cubeCopyFixture(), renderer = publicRenderer(f), destination = f.target(), source = f.storageTexture();
  renderer._preparing = true; renderer.copyTextureToTexture(source, destination.texture);
  destination.dispose(); f.flush(); await assert.rejects(renderer.waitForGPU(), {code: 'THREE_TARGET_STALE'});
  assert.equal(f.submissions, 0); assert.equal(destination.entry.references, 0); assert.equal(f.textures.length, 1); renderer.dispose();
});
