/** Private source effect targets retain accepted generations, not disposal. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {gpuPostprocessFixture} from './fixtures/animation/gpu_postprocess_fixture.mjs';

function fixture(t, options = {}) {
  const f = gpuPostprocessFixture();
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  const target = new T.RenderTarget(8, 4, {depthBuffer: false, type: T.HalfFloatType});
  t.after(() => { owner.dispose(); target.dispose(); });
  return {...f, owner, target};
}

test('effect resize scopes preserve every accepted generation across successive source updates', async t => {
  const f = fixture(t), first = f.owner.captureNativeTexture(f.target.texture);
  const releaseFirst = f.owner.retainGeneration(first.entry);
  f.owner.ensure(first.entry);
  f.owner.withNativeEffectTargets([f.target], () => f.target.setSize(16, 8));
  const second = f.owner.captureNativeTexture(f.target.texture), releaseSecond = f.owner.retainGeneration(second.entry);
  f.owner.ensure(second.entry);
  f.owner.withNativeEffectTargets([f.target], () => f.target.setSize(32, 16));
  const third = f.owner.captureNativeTexture(f.target.texture);
  assert.equal(f.owner.assertCurrent(first.entry), first.entry);
  assert.equal(f.owner.assertCurrent(second.entry), second.entry);
  const widths = [];
  for (const input of [first, second, third]) f.owner.withNativeTexture(input, texture => widths.push(texture.width));
  assert.deepEqual(widths, [8, 16, 32]);
  assert.equal(f.owner.allocatedBytes, (8 * 4 + 16 * 8 + 32 * 16) * 8);
  releaseFirst(); releaseSecond(); await f.owner.whenIdle();
  assert.equal(first.entry.released, true); assert.equal(second.entry.released, true);
  assert.equal(f.owner.allocatedBytes, 32 * 16 * 8);
});

test('zero source extents retain their logical viewport and charge the one-texel backing dimension', t => {
  for (const [width, height] of [[0, 4], [8, 0], [0, 0]]) {
    const f = fixture(t); f.target.setSize(width, height);
    const entry = f.owner.ensure(f.owner.capture(f.target));
    assert.deepEqual([entry.shape.width, entry.shape.height], [width || 1, height || 1]);
    assert.deepEqual(f.target.viewport.toArray(), [0, 0, width, height]);
    assert.equal(f.owner.allocatedBytes, (width || 1) * (height || 1) * 8);
    assert.deepEqual([entry.storage.diagnostics.width, entry.storage.diagnostics.height], [width || 1, height || 1]);
  }
});

test('effect resize scopes preserve queued generations across zero-size source mip levels', async t => {
  const f = fixture(t), first = f.owner.ensure(f.owner.capture(f.target));
  const releaseFirst = f.owner.retainGeneration(first);
  f.owner.withNativeEffectTargets([f.target], () => f.target.setSize(0, 0));
  const empty = f.owner.ensure(f.owner.capture(f.target)), releaseEmpty = f.owner.retainGeneration(empty);
  f.owner.withNativeEffectTargets([f.target], () => f.target.setSize(16, 8));
  const next = f.owner.ensure(f.owner.capture(f.target));
  for (const entry of [first, empty, next]) assert.equal(f.owner.assertCurrent(entry), entry);
  assert.deepEqual([first, empty, next].map(entry => entry.storage.diagnostics.width), [8, 1, 16]);
  assert.equal(f.owner.allocatedBytes, (8 * 4 + 1 + 16 * 8) * 8);
  releaseFirst(); releaseEmpty(); await f.owner.whenIdle();
  assert.equal(f.owner.allocatedBytes, 16 * 8 * 8);
});

test('one-texel backing does not admit negative, fractional or non-finite source extents', t => {
  for (const value of [-1, .5, NaN, Infinity]) {
    const f = fixture(t); f.target.setSize(value, 4);
    assert.throws(() => f.owner.capture(f.target), {code: 'THREE_TARGET_SIZE'});
  }
});

test('same-size explicit effect disposal invalidates its accepted generation inside a scope', t => {
  const f = fixture(t), entry = f.owner.capture(f.target), release = f.owner.retainGeneration(entry);
  f.owner.withNativeEffectTargets([f.target], () => f.target.dispose());
  assert.throws(() => f.owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'}); release();
});

test('a cold effect resize cannot hide later disposal of its first allocated generation', t => {
  const f = fixture(t); let entry, release;
  f.owner.withNativeEffectTargets([f.target], () => {
    f.target.setSize(16, 8);
    entry = f.owner.ensure(f.owner.capture(f.target)); release = f.owner.retainGeneration(entry);
    f.target.dispose();
  });
  assert.throws(() => f.owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'}); release();
});

test('a second effect disposal invalidates both the resized and previously accepted generation', t => {
  const f = fixture(t), first = f.owner.capture(f.target), releaseFirst = f.owner.retainGeneration(first);
  let second, releaseSecond;
  f.owner.withNativeEffectTargets([f.target], () => {
    f.target.setSize(16, 8);
    second = f.owner.capture(f.target); releaseSecond = f.owner.retainGeneration(second);
    f.target.dispose();
  });
  for (const entry of [first, second]) assert.throws(() => f.owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'});
  releaseFirst(); releaseSecond();
});

test('effect scopes do not preserve queued attachment-format mutations', t => {
  const f = fixture(t), entry = f.owner.capture(f.target), release = f.owner.retainGeneration(entry);
  f.owner.withNativeEffectTargets([f.target], () => {
    f.target.texture.type = T.FloatType; f.target.setSize(16, 8);
  });
  assert.throws(() => f.owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'}); release();
});

test('effect scope preflight and failures leave no registrations or disposal observers behind', t => {
  const f = fixture(t), entry = f.owner.capture(f.target), release = f.owner.retainGeneration(entry);
  let ran = false;
  assert.throws(() => f.owner.withNativeEffectTargets([f.target, {}], () => { ran = true; }), {code: 'THREE_TARGET_SOURCE'});
  assert.equal(ran, false);
  const listeners = f.target._listeners.dispose.length;
  assert.throws(() => f.owner.withNativeEffectTargets([f.target], () => {
    assert.throws(() => f.owner.withNativeEffectTargets([f.target], () => {}), {code: 'THREE_TARGET_REENTRANT'});
    throw new Error('effect update failed');
  }), /effect update failed/);
  assert.equal(f.target._listeners.dispose.length, listeners);
  assert.throws(() => f.owner.withNativeEffectTargets([f.target], () => Promise.resolve()), {code: 'THREE_TARGET_ASYNC_USE'});
  assert.equal(f.target._listeners.dispose.length, listeners);
  f.target.setSize(16, 8);
  assert.throws(() => f.owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'}); release();
});

test('detached effect generations retain the existing allocation budget until their leases end', async t => {
  const f = fixture(t, {maxBytes: 16 * 8 * 8}), entry = f.owner.ensure(f.owner.capture(f.target));
  const release = f.owner.retainGeneration(entry);
  f.owner.withNativeEffectTargets([f.target], () => f.target.setSize(16, 8));
  const next = f.owner.capture(f.target);
  assert.throws(() => f.owner.ensure(next), {code: 'THREE_TARGET_BUDGET'});
  release(); await f.owner.whenIdle(); f.owner.ensure(next);
  assert.equal(f.owner.allocatedBytes, 16 * 8 * 8);
});
