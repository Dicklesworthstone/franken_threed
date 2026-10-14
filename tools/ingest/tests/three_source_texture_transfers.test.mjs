import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeRendererTransfers} from '../three_renderer_transfers.mjs';
import {inspectGpuTargetCopy} from '../gpu_target_transfer.mjs';

class RendererError extends Error {
  constructor(code, message) { super(message); this.code = `F3D_RENDERER_${code}`; }
}
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};

// The actual controller/planners with a target-owner boundary double. No native
// shader, pixel parity, or GPU performance claims follow from these tests.
function fixture({initialized = true, fence = Promise.resolve(), maxRequests = 64} = {}) {
  const entries = new Map(), calls = [];
  const renderer = {
    _disposed: false, _deferredError: null, _initialized: initialized,
    _pending: [], _preparing: null, _drain: null, _targetLifetime: new AbortController(),
    _enqueue(request) { this._pending.push(request); },
    _targets: {
      capture(target) {
        const entry = entries.get(target);
        if (!entry) throw new RendererError('SOURCE', 'Unknown target');
        this.assertCurrent(entry); return entry;
      },
      assertCurrent(entry) {
        if (!entry.current) throw new RendererError('STALE', 'Target generation changed');
      },
      ensure(entry) {
        this.assertCurrent(entry);
        if (!entry.storage) {
          entry.storage = {whenIdle: () => fence};
          calls.push({kind: 'initialize', target: entry.target});
        }
      },
      copyColor(from, to, options) {
        this.ensure(from); this.ensure(to);
        const plan = inspectGpuTargetCopy(from.shape, to.shape, options);
        calls.push({kind: 'copy', from, to, plan});
        // Model content observed at submission, not a GPU readback.
        to.value = from.value;
        return !plan.empty;
      },
    },
  };
  const controller = createThreeRendererTransfers(renderer, RendererError, {maxRequests});
  function target(width = 8, height = 8, {format = 'rgba8unorm', sampleCount = 1, mipLevelCount = 1} = {}) {
    const target = {width, height};
    target.texture = {isTexture: true, isRenderTargetTexture: true, renderTarget: target};
    const entry = {target, current: true, value: 0, lifetime: new AbortController(), validation: Promise.resolve(),
      shape: {width, height, format, sampleCount, mipLevelCount, texture: target.texture}};
    entries.set(target, entry);
    return target;
  }
  function drain() {
    renderer._initialized = true;
    while (renderer._pending.length) {
      const request = renderer._pending.shift();
      try { const result = request.issue(); request.resolve?.(result); } catch (error) { request.reject?.(error); }
    }
  }
  return {renderer, controller, target, calls, entries, drain};
}

for (const bottomLeft of [false, true]) test(`source texture copy selects the live ${bottomLeft ? 'WebGL' : 'WebGPU'} origin`, async () => {
  const f = fixture({initialized: false}), a = f.target(9, 7, {mipLevelCount: 4}), b = f.target();
  // The WebGL constructor sets this AFTER creating the controller.
  f.renderer._bottomLeftOrigin = bottomLeft;
  const box = {min: {x: 1, y: 0}, max: {x: 3, y: 2}}, position = {x: 4, y: 1};
  const result = f.renderer.copyTextureToTextureAsync(a.texture, b.texture, box, position, 1);
  box.min.x = 999; box.max.y = 999; position.y = 999;
  assert.equal(f.calls.length, 0);
  f.drain(); await result;
  const copy = f.calls.find(call => call.kind === 'copy');
  assert.deepEqual(copy.plan.sourceOrigin, {x: 1, y: bottomLeft ? 1 : 0, z: 0});
  assert.deepEqual(copy.plan.destinationOrigin, {x: 4, y: bottomLeft ? 5 : 1, z: 0});
  assert.deepEqual(copy.plan.size, {width: 2, height: 2, depthOrArrayLayers: 1});
  assert.equal(copy.plan.sourceMipLevel, 1);
});

test('source default region covers the selected NPOT mip, including one-dimensional tails', async () => {
  const f = fixture(), a = f.target(9, 1, {mipLevelCount: 4}), b = f.target(2, 1);
  await f.renderer.copyTextureToTextureAsync(a.texture, b.texture, null, null, 2);
  assert.deepEqual(f.calls.at(-1).plan.size, {width: 2, height: 1, depthOrArrayLayers: 1});
});

test('source initRenderTarget is void, ordered, lazy before init and idempotent after allocation', async () => {
  const f = fixture({initialized: false}), target = f.target();
  assert.equal(f.renderer.initRenderTarget(target), undefined);
  assert.equal(f.calls.length, 0);
  const completion = f.renderer.initRenderTargetAsync(target);
  f.drain(); await completion; await f.controller.whenIdle();
  assert.equal(f.calls.length, 1);
  assert.equal(f.controller.pendingRequests, 0);
  assert.equal(f.controller.allocatedBytes, 0);
});

test('queued init/copy observes preceding writes without waiting for a GPU completion fence', async () => {
  const done = deferred(), f = fixture({initialized: false, fence: done.promise});
  const a = f.target(), b = f.target();
  f.renderer.initRenderTarget(a);
  f.renderer._pending.push({issue() { f.entries.get(a).value = 17; }});
  const result = f.renderer.copyTextureToTextureAsync(a.texture, b.texture);
  f.renderer._pending.push({issue() { f.entries.get(a).value = 29; }});
  f.drain();
  assert.equal(f.entries.get(b).value, 17);
  assert.equal(f.entries.get(a).value, 29);
  assert.equal(f.controller.pendingRequests, 2);
  done.resolve(); await result; await f.controller.whenIdle();
  assert.equal(f.controller.pendingRequests, 0);
});

test('ordinary void copy submits immediately and reports native validation at the renderer boundary', async () => {
  const invalid = deferred(), f = fixture(), a = f.target(), b = f.target();
  f.entries.get(b).validation = invalid.promise;
  assert.equal(f.renderer.copyTextureToTexture(a.texture, b.texture), undefined);
  assert.equal(f.calls.at(-1).kind, 'copy');
  const error = new Error('native validation');
  const idle = f.controller.whenIdle(); invalid.reject(error);
  await assert.rejects(idle, value => value === error);
  await Promise.resolve();
  assert.equal(f.renderer._deferredError, error);
});

test('source admission rejects unsupported mip/layer/texture and bad bounds synchronously without submission', () => {
  const f = fixture(), a = f.target(), b = f.target();
  const copy = (...args) => f.renderer.copyTextureToTexture(...args);
  assert.throws(() => copy(a.texture, b.texture, null, null, 0, 1), /destination mip zero/);
  assert.throws(() => copy(a.texture, b.texture, null, {x: 0, y: 0, z: 1}), /depth zero/);
  assert.throws(() => copy(a.texture, b.texture, {isBox3: true, min: {x: 0, y: 0, z: 1}, max: {x: 1, y: 1, z: 2}}), /2D source/);
  assert.throws(() => copy({}, b.texture), /target-owned/);
  assert.throws(() => copy(a.texture, a.texture), /In-place/);
  assert.throws(() => copy(a.texture, b.texture, null, null, 1), /not allocated/);
  assert.throws(() => copy(a.texture, b.texture, {min: {x: -1, y: 0}, max: {x: 2, y: 2}}), /non-negative/);
  assert.throws(() => copy(a.texture, b.texture, null, {x: 1, y: 0}), /exceeds/);
  assert.equal(f.calls.length, 0);
  assert.equal(f.controller.pendingRequests, 0);
});

test('async source admission rejects a promise and does not poison later valid operations', async () => {
  const f = fixture(), a = f.target(), b = f.target();
  await assert.rejects(f.renderer.copyTextureToTextureAsync(a.texture, b.texture, null, null, 0, 1), /mip zero/);
  assert.equal(f.renderer._deferredError, null);
  await f.renderer.copyTextureToTextureAsync(a.texture, b.texture);
});

test('target disposal cancels a deferred source copy before any allocation', async () => {
  const f = fixture({initialized: false}), a = f.target(), b = f.target();
  const result = f.renderer.copyTextureToTextureAsync(a.texture, b.texture);
  const error = new Error('target disposed'); f.entries.get(a).lifetime.abort(error);
  await assert.rejects(result, value => value === error);
  assert.equal(f.controller.pendingRequests, 0);
  f.drain(); assert.equal(f.calls.length, 0);
});

test('generation checks prevent queued texture handles from being redirected after resize', async () => {
  const f = fixture({initialized: false}), a = f.target(), b = f.target();
  const result = f.renderer.copyTextureToTextureAsync(a.texture, b.texture);
  f.entries.get(b).current = false; f.drain();
  await assert.rejects(result, /generation changed/);
  assert.equal(f.calls.length, 0);
});

test('init and source copies consume the same pending-request budget', async () => {
  const f = fixture({initialized: false, maxRequests: 1}), a = f.target(), b = f.target();
  const first = f.renderer.copyTextureToTextureAsync(a.texture, b.texture);
  await assert.rejects(f.renderer.initRenderTargetAsync(a), /budget/);
  assert.equal(f.renderer._pending.length, 1);
  f.drain(); await first;
  await f.renderer.initRenderTargetAsync(a);
});

test('source API installation preserves explicit subclass implementations', () => {
  const f = fixture(), copy = function () {};
  const subclass = Object.create({copyTextureToTexture: copy});
  createThreeRendererTransfers(subclass, RendererError);
  assert.equal(subclass.copyTextureToTexture, copy);
  assert.equal(Object.hasOwn(subclass, 'copyTextureToTexture'), false);
  assert.equal(typeof f.renderer.copyTextureToTexture, 'function');
});
