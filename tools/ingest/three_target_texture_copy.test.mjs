import test from 'node:test';
import assert from 'node:assert/strict';
import {captureTargetToStorageCopy, captureStorageToTargetCopy} from './three_target_texture_copy.mjs';
import {submitTextureCopy} from './three_texture_copy_lifetime.mjs';
import {targetCopyFixture, box, point, failure} from './three_target_copy_fixtures.mjs';

function forward(f, source, destination, ...args) {
  const request = captureTargetToStorageCopy(f.renderer, source.texture, destination, ...args);
  return submitTextureCopy(f.renderer, request, f.rejected);
}
function reverse(f, source, destination, ...args) {
  const request = captureStorageToTargetCopy(f.renderer, source, destination.texture, ...args);
  return submitTextureCopy(f.renderer, request, f.rejected);
}

test('cube-to-storage copies all six native faces without a public owner back-pointer', async () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube', pointer: false}), destination = f.storageTexture();
  assert.equal(forward(f, source, destination), undefined);
  assert.equal(f.commands.length, 1);
  assert.equal(f.commands[0].from.texture, source.entry.native[0]);
  assert.equal(f.commands[0].to.texture, destination.entry.native);
  assert.deepEqual(f.commands[0].size, [4, 4, 6]);
  assert.equal(f.references, 0); assert.equal(f.releases, 1);
  await Promise.all(f.validations);
});

test('queued cube ranges, source mips and destination positions are immutable', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}), destination = f.storageTexture();
  f.renderer._preparing = true;
  const region = box(0, 0, 2, 2, 2, 3), position = point(0, 0, 1);
  forward(f, source, destination, region, position, 1, 1);
  region.min.z = 5; position.z = 5; source.texture.renderTarget = null;
  assert.equal(f.references, 1); assert.equal(f.commands.length, 0);
  f.flush();
  assert.deepEqual(f.commands[0].from.origin, [0, 0, 2]);
  assert.deepEqual(f.commands[0].to.origin, [0, 0, 1]);
  assert.deepEqual(f.commands[0].size, [2, 2, 3]);
  assert.equal(f.commands[0].from.mipLevel, 1); assert.equal(f.storage.lastWrite.mipLevel, 1);
  assert.equal(f.references, 0); assert.deepEqual(f.errors, []);
});

test('one cube face can seed a 2D texture or a mip-local volume slice range', () => {
  for (const dimension of ['2d', '3d']) {
    const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
    const destination = f.storageTexture({dimension, depth: dimension === '2d' ? 1 : 4});
    forward(f, source, destination, box(0, 0, 5, 2, 2, 1), point(0, 0, dimension === '2d' ? 0 : 1), 1, 1);
    assert.deepEqual(f.commands[0].size, [2, 2, 1]); assert.equal(f.commands[0].from.origin[2], 5);
    assert.equal(f.references, 0);
  }
});

test('Box2 selects the first face, not all cube layers', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}), destination = f.storageTexture({dimension: '2d', depth: 1});
  forward(f, source, destination, {isBox2: true, min: {x: 0, y: 0}, max: {x: 4, y: 4}});
  assert.deepEqual(f.commands[0].size, [4, 4, 1]);
});

test('invalid cube faces, mip levels and formats reject without retaining generations', () => {
  for (const args of [[box(0, 0, 6, 4, 4, 1)], [null, null, 3], [null, point(0, 0, 1)]]) {
    const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
    assert.throws(() => forward(f, source, f.storageTexture(), ...args));
    assert.equal(f.references, 0); assert.equal(f.commands.length, 0); assert.deepEqual(f.events, []);
  }
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
  assert.throws(() => forward(f, source, f.storageTexture({format: 'rgba16float'})), {code: 'THREE_STORAGE_COPY_FORMAT'});
  assert.equal(f.references, 0);
});

test('empty cube copies release admission without joining a blocked queue', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = true;
  forward(f, source, f.storageTexture(), box(0, 0, 0, 0, 4, 6));
  assert.equal(f.references, 0); assert.equal(f.renderer._pending.length, 0); assert.deepEqual(f.events, []);
});

test('full FIFO admission releases a captured target synchronously', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = true;
  f.renderer.queueError = failure('FULL');
  assert.throws(() => forward(f, source, f.storageTexture()), {code: 'FULL'});
  assert.equal(f.references, 0); assert.equal(f.releases, 1); assert.deepEqual(f.errors, []);
});

test('target disposal releases immediately while unrelated preparation remains blocked', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = true;
  forward(f, source, f.storageTexture()); source.dispose();
  assert.equal(f.references, 0); assert.equal(f.errors[0].code, 'TARGET_DISPOSED');
  f.flush(); assert.equal(f.commands.length, 0); assert.equal(f.errors.length, 1); assert.equal(f.releases, 1);
});

test('renderer cancellation releases copies that never enter their issue callback', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = true;
  forward(f, source, f.storageTexture());
  f.renderer._targetLifetime.abort(failure('CANCELLED'));
  assert.equal(f.references, 0); f.flush();
  assert.equal(f.commands.length, 0); assert.equal(f.errors[0].code, 'CANCELLED');
});

test('an earlier queue failure releases retained targets exactly once', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = true;
  forward(f, source, f.storageTexture());
  const request = f.renderer._pending.shift(); request.reject(failure('EARLIER')); request.reject(failure('AGAIN'));
  source.dispose(); request.issue();
  assert.equal(f.references, 0); assert.equal(f.releases, 1); assert.equal(f.errors.length, 1); assert.equal(f.commands.length, 0);
});

test('old target generations stay retained until their queued copy is submitted', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = true;
  forward(f, source, f.storageTexture());
  // Recognized internal resizing can preserve this entry; public resizing is
  // separately rejected by assertCurrent. The lease must survive the producer.
  assert.equal(f.references, 1); f.flush(); assert.equal(f.references, 0); assert.equal(f.commands.length, 1);
});

test('public resize and device changes reject before borrowing source GPU residency', () => {
  for (const change of [(f, source) => { source.entry.target.epoch++; }, f => { f.renderer._device = {}; }]) {
    const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = true;
    forward(f, source, f.storageTexture()); change(f, source); f.flush();
    assert.equal(f.commands.length, 0); assert.equal(f.references, 0); assert.equal(f.errors.length, 1);
    assert.ok(!f.events.includes('source-borrow'));
  }
});

test('source and destination budgets and cube layer limits precede all borrows', () => {
  for (const change of [f => { f.targets.budgetError = failure('BUDGET'); }, f => { f.storage.budgetError = failure('BUDGET'); },
    f => { f.device.limits.maxTextureArrayLayers = 5; }, f => { f.device.limits.maxTextureDimension2D = 2; }]) {
    const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); change(f);
    assert.throws(() => forward(f, source, f.storageTexture()));
    assert.equal(f.references, 0); assert.equal(f.commands.length, 0); assert.ok(!f.events.includes('source-borrow'));
  }
});

test('source mip validation failures reach storage completion without a queue wait', async () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
  source.entry.storage.validation = Promise.reject(failure('MIP_GPU')); source.entry.storage.validation.catch(() => {});
  forward(f, source, f.storageTexture());
  assert.equal(f.commands.length, 1); await assert.rejects(Promise.all(f.validations), {code: 'MIP_GPU'});
  assert.equal(f.references, 0);
});

test('encoding and queue submission errors release generations on warm and queued paths', () => {
  for (const queued of [false, true]) for (const stage of ['onFinish', 'onSubmit']) {
    const f = targetCopyFixture(), source = f.target({dimension: 'cube'}); f.renderer._preparing = queued;
    f.device[stage] = () => { throw failure('GPU'); };
    if (queued) { forward(f, source, f.storageTexture()); f.flush(); assert.equal(f.errors[0].code, 'GPU'); }
    else assert.throws(() => forward(f, source, f.storageTexture()), {code: 'GPU'});
    assert.equal(f.references, 0); assert.equal(f.commands.length, 0); assert.equal(f.releases, 1);
  }
});

test('encoder completion cannot redirect a stale target copy', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
  f.device.onFinish = () => { source.entry.target.epoch++; };
  assert.throws(() => forward(f, source, f.storageTexture()), {code: 'STALE_TARGET'});
  assert.equal(f.commands.length, 0); assert.equal(f.references, 0);
});

test('nested compute turns keep the copy in their submission position', async () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
  let resume, pending;
  f.renderer._computeTurn = {submit(issue) { return pending = new Promise(resolve => { resume = () => resolve(issue()); }); }};
  forward(f, source, f.storageTexture());
  assert.equal(f.references, 1); assert.equal(f.renderer._pending.length, 0); assert.equal(f.commands.length, 0);
  resume(); await pending; assert.equal(f.commands.length, 1); assert.equal(f.references, 0);
});

test('rejected compute turns release a copy that was never submitted', async () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
  f.renderer._computeTurn = {submit() { return Promise.reject(failure('TURN')); }};
  forward(f, source, f.storageTexture()); await Promise.resolve();
  assert.equal(f.references, 0); assert.equal(f.commands.length, 0); assert.equal(f.errors[0].code, 'TURN');
});

test('direct captured issue is single-use and also releases its generation lease', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
  const request = captureTargetToStorageCopy(f.renderer, source.texture, f.storageTexture());
  assert.equal(f.references, 1); request.issue(); assert.equal(f.references, 0);
  assert.throws(() => request.issue(), {code: 'THREE_STORAGE_COPY_STALE'}); request.release(); assert.equal(f.releases, 1);
});

test('storage-to-target copies also retain selected MRT destinations and validation', async () => {
  const f = targetCopyFixture(), destination = f.target({dimension: '2d-array', depth: 6, count: 2});
  destination.texture = destination.textures[1]; f.renderer._preparing = true;
  reverse(f, f.storageTexture(), destination);
  assert.equal(f.references, 1); f.flush(); assert.equal(f.references, 0);
  assert.equal(f.commands[0].to.texture, destination.entry.native[1]);
  assert.deepEqual(f.commands[0].size, [4, 4, 6]); await Promise.all(f.validations);
});

test('storage-to-target destination disposal cancels its queued write', () => {
  const f = targetCopyFixture(), destination = f.target({dimension: '2d-array', depth: 6}); f.renderer._preparing = true;
  reverse(f, f.storageTexture(), destination); destination.dispose();
  assert.equal(f.references, 0); f.flush(); assert.equal(f.commands.length, 0); assert.equal(f.errors.length, 1);
});

test('unsupported storage-to-target destinations release their admission', () => {
  for (const options of [{samples: 4}]) {
    const f = targetCopyFixture(), destination = f.target(options);
    assert.throws(() => reverse(f, f.storageTexture(), destination)); assert.equal(f.references, 0);
  }
});
