/** Copy orchestration against strict recording owner/device contracts. These
 * tests execute the production copy planner and scheduler, not WGSL or a GPU.
 * Sampled upload-history semantics have their own source-owner regression suite.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {copySampledTexture} from './three_sampled_texture_copy.mjs';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  promise.catch(() => {});
  return {promise, resolve, reject};
};
const shape = (options = {}) => ({width: 2, height: 2, depth: 1,
  mipLevelCount: 1, format: 'rgba8unorm', viewDimension: '2d', ...options});
const texture = (options = {}) => ({isTexture: true, alive: true, version: 1,
  bytes: Uint8Array.of(1, 2, 3, 4), shape: shape(options)});
const box = (x, y, z, width, height, depth) => ({isBox3: true,
  min: {x, y, z}, max: {x: x + width, y: y + height, z: z + depth}});
const point = (x, y, z) => ({isVector3: true, x, y, z});
const error = code => Object.assign(new Error(code), {code});

function fixture() {
  const events = [], commands = [], errors = [], active = new Set(), history = new Map(), pending = [];
  let transaction = false, rollbacks = 0, releases = 0;
  const source = texture(), destination = {...texture(), isStorageTexture: true};
  const device = {createCommandEncoder() {
    events.push('encoder');
    let copy;
    return {copyTextureToTexture(from, to, size) { copy = {from, to, size}; },
      finish() { device.onFinish?.(); return copy; }};
  }, queue: {submit(batch) {
    device.onSubmit?.(); events.push('submit');
    for (const copy of batch) commands.push({...copy, bytes: [...copy.from.texture.bytes]});
  }}};
  const storage = {
    capture(value) { return {texture: value, shape: {...value.shape}}; },
    assertCurrent(entry) {
      if (!entry.texture.alive || JSON.stringify(entry.shape) !== JSON.stringify(entry.texture.shape)) throw error('STALE_DESTINATION');
    },
    captureBinding(binding) { return {...binding}; },
    validateAllocations(entries) {
      events.push('preflight'); entries.forEach(entry => storage.assertCurrent(entry));
      if (storage.budgetError) throw storage.budgetError;
    },
    withTexture(entry, run, write, options) {
      storage.assertCurrent(entry); events.push('destination');
      const result = run(entry.native ??= {entry});
      events.push('written'); storage.lastWrite = write;
      const validation = Promise.resolve(options.validation); validation.catch(() => {}); pending.push(validation);
      return result;
    },
  };
  const inputs = {
    captureDispatch(run) {
      assert.equal(transaction, false); transaction = true;
      const before = new Map(history);
      try { return run(); }
      catch (e) { rollbacks++; history.clear(); for (const [key, value] of before) history.set(key, value); throw e; }
      finally { transaction = false; }
    },
    captureBinding(binding) {
      assert.equal(transaction, true); events.push('capture');
      const value = binding.texture;
      let record = history.get(value);
      if (!record || record.version !== value.version) {
        record = {version: value.version, bytes: value.bytes.slice(), shape: {...value.shape}};
        history.set(value, record);
      }
      const lease = {source: value, record}; active.add(lease); return lease;
    },
    copyShape(lease) { inputs.validateDispatch([lease]); return lease.record.shape; },
    preparation(leases) { return leases[0].source.ready ?? null; },
    validateDispatch(leases) {
      for (const lease of leases) if (!active.has(lease) || !lease.source.alive) throw error('STALE_SOURCE');
    },
    withTexture(lease, run) {
      inputs.validateDispatch([lease]); events.push('upload'); lease.source.onUpdate?.();
      inputs.validateDispatch([lease]); return run(lease.record);
    },
    whenIdle() {
      events.push('source-validation');
      // Match the real owner's synchronous snapshot of earlier storage scopes.
      return Promise.all([...pending, source.validation]);
    },
    release(lease) { assert.ok(active.delete(lease), 'each lease is released exactly once'); releases++; events.push('release'); },
  };
  const renderer = {
    _device: device, _storageTextures: storage, _getSampledTextureInputs: () => inputs,
    _targetLifetime: new AbortController(), _pending: [], _drain: null, _preparing: false,
    _enqueue(request) {
      if (renderer.queueError) throw renderer.queueError;
      events.push('enqueue'); renderer._pending.push(request);
    },
  };
  const copy = (region = null, position = null, sourceLevel = 0, destinationLevel = 0) =>
    copySampledTexture(renderer, source, destination, region, position, sourceLevel, destinationLevel, e => errors.push(e));
  async function flush() {
    while (renderer._pending.length) {
      const request = renderer._pending[0];
      try {
        const value = request.issue();
        request.resolve(request.awaitSubmission ? await value : value);
      } catch (e) { request.reject(e); }
      if (renderer._pending[0] === request) renderer._pending.shift();
    }
  }
  function dispose() {
    renderer._disposed = true; renderer._targetLifetime.abort(error('DISPOSED'));
    for (const request of renderer._pending.splice(0)) request.reject(error('DISPOSED'));
  }
  return {renderer, inputs, storage, source, destination, device, events, commands, errors, active, history, pending, copy, flush, dispose,
    get releases() { return releases; }, get rollbacks() { return rollbacks; }};
}

test('warm copies stay void and submit through the existing owners with one release', async () => {
  const f = fixture(); assert.equal(f.copy(), undefined);
  assert.equal(f.commands.length, 1); assert.deepEqual(f.commands[0].bytes, [1, 2, 3, 4]);
  assert.deepEqual(f.commands[0].size, [2, 2, 1]);
  assert.deepEqual(f.events, ['capture', 'preflight', 'upload', 'source-validation', 'destination', 'encoder', 'submit', 'written', 'release']);
  assert.equal(f.storage.lastWrite.store, true); assert.equal(f.storage.lastWrite.access, 'writeOnly');
  assert.equal(f.releases, 1); assert.equal(f.active.size, 0);
  await Promise.all(f.pending);
});

test('queued copies freeze versions, source shape, box and destination position', async () => {
  const f = fixture(); f.renderer._preparing = true;
  f.source.shape = shape({width: 8, height: 8}); f.destination.shape = shape({width: 8, height: 8});
  const region = box(2, 3, 0, 2, 2, 1), position = point(4, 5, 0);
  f.copy(region, position);
  f.source.bytes.fill(8); f.source.version++; f.copy(region, position);
  f.source.bytes.fill(99); f.source.shape = shape(); region.min.x = 7; position.y = 100;
  assert.equal(f.commands.length, 0); assert.equal(f.active.size, 2);
  await f.flush();
  assert.deepEqual(f.commands.map(c => c.bytes), [[1, 2, 3, 4], [8, 8, 8, 8]]);
  assert.ok(f.commands.every(c => c.from.origin.join() === '2,3,0' && c.to.origin.join() === '4,5,0'));
  assert.equal(f.releases, 2); assert.deepEqual(f.errors, []);
});

test('copies select authored source mips and destination mips without sampling or conversion', () => {
  const f = fixture(); f.source.shape = shape({width: 4, height: 4, mipLevelCount: 3});
  f.destination.shape = shape({width: 8, height: 8, mipLevelCount: 4});
  f.copy(null, null, 1, 2);
  assert.equal(f.commands[0].from.mipLevel, 1); assert.equal(f.commands[0].to.mipLevel, 2);
  assert.deepEqual(f.commands[0].size, [2, 2, 1]); assert.equal(f.storage.lastWrite.mipLevel, 2);
});

test('array/cube-face ranges and volume slices share raw integer copy planning', () => {
  for (const from of ['2d-array', '3d']) for (const to of ['2d-array', '3d']) {
    const f = fixture(); f.source.shape = shape({viewDimension: from, depth: 6, format: 'rgba32uint'});
    f.destination.shape = shape({viewDimension: to, depth: 4, format: 'rgba32uint'});
    f.copy(box(0, 0, 2, 2, 2, 2), point(0, 0, 1));
    assert.deepEqual(f.commands[0].from.origin, [0, 0, 2]);
    assert.deepEqual(f.commands[0].to.origin, [0, 0, 1]); assert.deepEqual(f.commands[0].size, [2, 2, 2]);
  }
});

test('bad formats, bounds and mip levels roll back admission without allocating or uploading', () => {
  for (const mutate of [f => { f.destination.shape.format = 'rgba16float'; },
    f => { f.destination.shape.width = 1; }]) {
    const f = fixture(); mutate(f); assert.throws(() => f.copy());
    assert.deepEqual(f.events, ['capture', 'release']); assert.equal(f.rollbacks, 1);
    assert.equal(f.history.size, 0); assert.equal(f.active.size, 0);
  }
  const f = fixture(); assert.throws(() => f.copy(null, null, 1), {code: 'THREE_STORAGE_COPY_LEVEL'});
  assert.equal(f.rollbacks, 1); assert.equal(f.commands.length, 0);
});

test('zero-area copies leave no queued work, allocation or accepted sampled history', () => {
  const f = fixture(); f.renderer._preparing = true;
  assert.equal(f.copy(box(0, 0, 0, 0, 2, 1)), undefined);
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.history.size, 0);
  assert.equal(f.rollbacks, 1); assert.equal(f.releases, 1); assert.equal(f.commands.length, 0);
});

test('a full FIFO rejects synchronously and restores the preceding capture history', () => {
  const f = fixture(); f.copy(); const accepted = f.history.get(f.source);
  f.renderer._preparing = true; f.renderer.queueError = error('FULL');
  f.source.bytes.fill(9); f.source.version++;
  assert.throws(() => f.copy(), {code: 'FULL'});
  assert.equal(f.history.get(f.source), accepted); assert.equal(f.active.size, 0);
  assert.equal(f.commands.length, 1); assert.deepEqual(f.errors, []);
});

test('destination budget refusal precedes irreversible sampled uploads', () => {
  const f = fixture(); f.storage.budgetError = error('BUDGET');
  assert.throws(() => f.copy(), {code: 'BUDGET'});
  assert.ok(!f.events.includes('upload')); assert.equal(f.commands.length, 0); assert.equal(f.releases, 1);
});

test('queued disposal, resizing and device changes fail before any source upload', async () => {
  for (const change of [f => { f.destination.alive = false; }, f => { f.destination.shape.width++; },
    f => { f.source.alive = false; }, f => { f.renderer._device = {}; }]) {
    const f = fixture(); f.renderer._preparing = true; f.copy(); change(f); await f.flush();
    assert.equal(f.commands.length, 0); assert.ok(!f.events.includes('upload'));
    assert.equal(f.errors.length, 1); assert.equal(f.releases, 1);
  }
});

test('source upload callbacks cannot publish into a disposed destination or a changed device', () => {
  for (const change of [f => { f.destination.alive = false; }, f => { f.renderer._device = {}; }]) {
    const f = fixture(); f.source.onUpdate = () => change(f);
    assert.throws(() => f.copy()); assert.equal(f.commands.length, 0); assert.equal(f.releases, 1);
  }
});

test('encoding and submission failures release captured sources on both scheduling paths', async () => {
  for (const deferredCall of [false, true]) for (const field of ['onFinish', 'onSubmit']) {
    const f = fixture(); f.renderer._preparing = deferredCall; f.device[field] = () => { throw error('GPU'); };
    if (deferredCall) { f.copy(); await f.flush(); assert.equal(f.errors[0].code, 'GPU'); }
    else assert.throws(() => f.copy(), {code: 'GPU'});
    assert.equal(f.active.size, 0); assert.equal(f.releases, 1); assert.equal(f.commands.length, 0);
  }
});

test('image preparation occupies its FIFO slot without waiting for GPU completion', async () => {
  const f = fixture(), ready = deferred(); f.source.ready = ready.promise;
  f.copy(); assert.equal(f.renderer._pending[0].awaitSubmission, true);
  f.renderer._pending.push({transfer: true, issue() { f.events.push('later'); }, resolve() {}, reject: assert.fail});
  const drain = f.flush(); await Promise.resolve();
  assert.equal(f.commands.length, 0); assert.ok(!f.events.includes('later'));
  ready.resolve(); await drain;
  assert.equal(f.commands.length, 1); assert.ok(f.events.indexOf('submit') < f.events.indexOf('later'));
  assert.equal(f.releases, 1);
});

test('image failure releases a queued capture and reports the original error', async () => {
  const f = fixture(), ready = deferred(); f.source.ready = ready.promise; f.copy();
  ready.reject(error('BITMAP')); await f.flush();
  assert.equal(f.errors[0].code, 'BITMAP'); assert.equal(f.commands.length, 0); assert.equal(f.releases, 1);
});

test('disposal cancels a blocked image copy and late readiness cannot resume it', async () => {
  const f = fixture(), ready = deferred(); f.source.ready = ready.promise; f.copy();
  const drain = f.flush(); await Promise.resolve(); f.dispose(); await drain;
  assert.equal(f.releases, 1); assert.equal(f.errors.length, 1);
  ready.resolve(); await Promise.resolve(); await Promise.resolve();
  assert.equal(f.commands.length, 0); assert.equal(f.releases, 1);
});

test('an earlier FIFO failure releases copies that never reach their issue callback', () => {
  const f = fixture(); f.renderer._preparing = true; f.copy();
  const request = f.renderer._pending.shift(); request.reject(error('EARLIER_FAILURE')); request.reject(error('AGAIN'));
  assert.equal(f.releases, 1); assert.equal(f.errors.length, 1); assert.equal(f.commands.length, 0);
});

test('source validation propagates to destination completion without a dependency cycle', async () => {
  const f = fixture(); f.source.validation = Promise.reject(error('SOURCE_GPU'));
  f.source.validation.catch(() => {}); f.copy();
  await assert.rejects(Promise.all(f.pending), {code: 'SOURCE_GPU'});
  const good = fixture(); good.copy(); good.copy();
  await Promise.all(good.pending); assert.equal(good.commands.length, 2);
});

test('nested compute turns receive the image submission barrier, not a GPU-completion barrier', async () => {
  const f = fixture(), ready = deferred(); f.source.ready = ready.promise;
  let turnResult, awaits;
  f.renderer._computeTurn = {submit(issue, awaitSubmission) { awaits = awaitSubmission; return turnResult = issue(); }};
  f.copy(); assert.equal(awaits, true); assert.equal(f.renderer._pending.length, 0);
  ready.resolve(); await turnResult;
  assert.equal(f.commands.length, 1); assert.equal(f.releases, 1);
});

function targetFixture(options = {}) {
  const f = fixture(); delete f.destination.isStorageTexture;
  f.destination.isRenderTargetTexture = true;
  const first = texture(options);
  const target = {alive: true, epoch: 1, textures: [first, f.destination]};
  f.destination.renderTarget = first.renderTarget = target;
  const targetShape = {width: 2, height: 2, layers: 1, sampleCount: 1, cube: false,
    viewDimension: '2d', ...options, colors: [shape(options), shape(options)], textures: target.textures};
  let refs = 0, released = 0;
  f.device.limits = {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048, maxTextureArrayLayers: 256};
  const targets = {
    getTextureTarget(value) { return value.renderTarget === target ? target : null; },
    capture(value) {
      assert.equal(value, target);
      return {target, epoch: target.epoch, shape: {...targetShape, textures: target.textures.slice()}};
    },
    assertCurrent(entry) { if (!target.alive || entry.epoch !== target.epoch) throw error('STALE_TARGET'); },
    retainGeneration(entry) {
      targets.assertCurrent(entry); refs++; let live = true;
      return () => { assert.ok(live); live = false; refs--; released++; };
    },
    validateNativeTextures(borrows) {
      f.events.push('target-preflight');
      for (const borrow of borrows) {
        targets.assertCurrent(borrow.entry);
        assert.equal(borrow.device, f.renderer._device); assert.equal(borrow.depth, false);
        assert.equal(borrow.texture, target.textures[borrow.colorIndex]);
      }
      if (targets.budgetError) throw targets.budgetError;
    },
    withNativeColorDestination(borrow, run, {validation}) {
      targets.validateNativeTextures([borrow]); f.events.push('target-destination');
      const result = run({target, colorIndex: borrow.colorIndex});
      const task = Promise.resolve(validation); task.catch(() => {}); f.pending.push(task);
      return result;
    },
  };
  f.renderer._targets = targets;
  return {...f, target, targets, targetShape,
    get targetReferences() { return refs; }, get targetReleases() { return released; }};
}

test('native target copies select the actual MRT attachment without a sampler or UV replica', async () => {
  const f = targetFixture(); f.source.shape.mipLevelCount = 2;
  f.copy(box(0, 0, 0, 1, 1, 1), point(1, 1, 0), 1);
  assert.equal(f.commands[0].to.texture.colorIndex, 1);
  assert.equal(f.commands[0].to.texture.target, f.target);
  assert.equal(f.commands[0].from.mipLevel, 1); assert.equal(f.commands[0].to.mipLevel, 0);
  assert.deepEqual(f.commands[0].to.origin, [1, 1, 0]);
  assert.equal(f.storage.lastWrite, undefined); assert.equal(f.targetReferences, 0); assert.equal(f.targetReleases, 1);
  await Promise.all(f.pending);
});

test('sampled versions remain independent across queued target copies', async () => {
  const f = targetFixture(); f.renderer._preparing = true;
  f.copy(); f.source.version++; f.source.bytes.fill(9); f.copy(); f.source.bytes.fill(77);
  assert.equal(f.targetReferences, 2); await f.flush();
  assert.deepEqual(f.commands.map(c => c.bytes), [[1, 2, 3, 4], [9, 9, 9, 9]]);
  assert.equal(f.targetReferences, 0); assert.equal(f.targetReleases, 2); assert.deepEqual(f.errors, []);
});

test('array and 3D target writes retain selected slices and raw float32 formats', () => {
  for (const viewDimension of ['2d-array', '3d']) {
    const f = targetFixture({viewDimension, layers: 4, format: 'rgba32float'});
    f.source.shape = shape({viewDimension: '3d', depth: 6, format: 'rgba32float'});
    f.copy(box(0, 0, 3, 2, 2, 2), point(0, 0, 1));
    assert.deepEqual(f.commands[0].size, [2, 2, 2]); assert.deepEqual(f.commands[0].to.origin, [0, 0, 1]);
    assert.equal(f.targetReferences, 0);
  }
});

test('unsupported target destinations fail before source capture or GPU work', () => {
  for (const [options, code] of [[{cube: true}, 'SOURCE'], [{sampleCount: 4}, 'SAMPLES']]) {
    const f = targetFixture(options);
    assert.throws(() => f.copy(), {code: 'THREE_STORAGE_COPY_' + code});
    assert.equal(f.active.size, 0); assert.equal(f.targetReferences, 0); assert.equal(f.commands.length, 0);
  }
  const f = targetFixture();
  assert.throws(() => f.copy(null, null, 0, 1), {code: 'THREE_STORAGE_COPY_LEVEL'});
  f.destination.isDepthTexture = true;
  assert.throws(() => f.copy(), {code: 'THREE_STORAGE_COPY_SOURCE'});
  delete f.destination.isDepthTexture; f.destination.renderTarget = null;
  assert.throws(() => f.copy(), {code: 'THREE_STORAGE_COPY_SOURCE'});
  assert.ok(!f.events.includes('capture')); assert.equal(f.targetReferences, 0);
});

test('target budgets and 2D/3D device limits are checked before uploading sampled pixels', () => {
  for (const viewDimension of ['2d', '3d']) {
    const f = targetFixture({viewDimension});
    f.device.limits[viewDimension === '3d' ? 'maxTextureDimension3D' : 'maxTextureDimension2D'] = 1;
    assert.throws(() => f.copy(), {code: 'THREE_STORAGE_COPY_LIMIT'});
    assert.ok(!f.events.includes('upload')); assert.equal(f.targetReferences, 0);
  }
  const f = targetFixture(); f.targets.budgetError = error('TARGET_BUDGET');
  assert.throws(() => f.copy(), {code: 'TARGET_BUDGET'});
  assert.ok(!f.events.includes('upload')); assert.equal(f.targetReferences, 0);
});

test('queued target disposal and generation changes cannot redirect an accepted copy', async () => {
  for (const change of [f => { f.target.alive = false; }, f => { f.target.epoch++; }]) {
    const f = targetFixture(); f.renderer._preparing = true; f.copy(); change(f); await f.flush();
    assert.equal(f.errors[0].code, 'STALE_TARGET'); assert.equal(f.targetReferences, 0);
    assert.equal(f.commands.length, 0); assert.ok(!f.events.includes('upload'));
  }
});

test('target format rejection, empty copies and full FIFOs release their generation lease', () => {
  for (const mode of ['format', 'empty', 'full']) {
    const f = targetFixture(); f.renderer._preparing = true;
    if (mode === 'format') f.source.shape.format = 'rgba16float';
    if (mode === 'full') f.renderer.queueError = error('FULL');
    if (mode === 'empty') f.copy(box(0, 0, 0, 0, 2, 1));
    else assert.throws(() => f.copy());
    assert.equal(f.targetReferences, 0); assert.equal(f.targetReleases, 1);
    assert.equal(f.active.size, 0); assert.equal(f.history.size, 0); assert.equal(f.commands.length, 0);
  }
});

test('pending target image copies cancel both owners and cannot resume after disposal', async () => {
  const f = targetFixture(), ready = deferred(); f.source.ready = ready.promise; f.copy();
  const drain = f.flush(); await Promise.resolve(); f.dispose(); await drain;
  assert.equal(f.targetReferences, 0); assert.equal(f.targetReleases, 1); assert.equal(f.active.size, 0);
  ready.resolve(); await Promise.resolve(); await Promise.resolve();
  assert.equal(f.commands.length, 0); assert.equal(f.targetReleases, 1);
});

test('target validation receives the source error and later writes cannot form a cycle', async () => {
  const f = targetFixture(); f.source.validation = Promise.reject(error('SOURCE_GPU'));
  f.source.validation.catch(() => {}); f.copy();
  await assert.rejects(Promise.all(f.pending), {code: 'SOURCE_GPU'});
  const good = targetFixture(); good.copy(); good.copy(); await Promise.all(good.pending);
  assert.equal(good.commands.length, 2); assert.equal(good.targetReferences, 0);
});

test('target changes in upload and encoder completion callbacks are rejected before submission', () => {
  for (const phase of ['upload', 'finish']) {
    const f = targetFixture(), change = () => { f.target.epoch++; };
    if (phase === 'upload') f.source.onUpdate = change; else f.device.onFinish = change;
    assert.throws(() => f.copy(), {code: 'STALE_TARGET'});
    assert.equal(f.commands.length, 0); assert.equal(f.targetReferences, 0); assert.equal(f.active.size, 0);
  }
});

test('registered target ownership does not require a public renderTarget back-pointer', () => {
  const f = targetFixture(); f.destination.renderTarget = null;
  f.targets.getTextureTarget = value => value === f.destination ? f.target : null;
  f.copy(); assert.equal(f.commands[0].to.texture.colorIndex, 1); assert.equal(f.targetReferences, 0);
});
