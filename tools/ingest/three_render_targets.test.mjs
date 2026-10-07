import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {targetDevice, targetThree as T} from './fixtures/target_residency_fixture.mjs';
const fixture = (options = {}) => {
  const f = targetDevice(), owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  return {...f, owner, target: new T.RenderTarget()};
};

test('selecting source targets is GPU-free; one persistent storage per target', async () => {
  const {owner, target, calls} = fixture();
  const e = owner.capture(target); assert.equal(calls.textures.length, 0);
  assert.equal(owner.capture(target), e);
  owner.ensure(e); owner.ensure(e); assert.equal(calls.textures.length, 2);
  assert.equal(owner.allocatedBytes, 4 * 2 * 8);
  await owner.whenIdle(); owner.dispose();
  assert.equal(calls.destroyed, 0); assert.ok(calls.textures.every(t => t.destroyed === 1));
});

for (const samples of [0, 4]) for (const type of [T.UnsignedByteType, T.HalfFloatType])
  test(`resolved texture sampling: samples=${samples}, type=${type}`, async () => {
    const {owner, target, calls} = fixture(); target.samples = samples; target.texture.type = type;
    const e = owner.capture(target), bindings = owner.bindingsFor();
    owner.render(e, frame => {
      assert.equal(frame.colorView.texture.sampleCount, samples || 1);
      assert.equal(!!frame.resolveTarget, samples === 4);
    });
    const a = bindings.get(target.texture), count = calls.submissions.length;
    const b = bindings.get(target.texture);
    assert.equal(a.view, b.view); assert.equal(a.sampler, b.sampler);
    assert.equal(calls.submissions.length, count, 'unchanged target does not need another UV pass');
    const pass = calls.submissions.at(-1).passes[0];
    assert.equal(pass.group.entries[0].resource.texture.sampleCount, 1, 'sample resolved color, never MSAA');
    assert.notEqual(pass.descriptor.colorAttachments[0].view.texture, pass.group.entries[0].resource.texture);
    assert.equal(pass.vertices, 3); assert.equal(pass.ended, true);
    assert.match(calls.modules[0].code, /i32\(size.y\) - 1 - i32\(p.y\)/);
    owner.render(e, () => {}); bindings.get(target.texture);
    assert.equal(calls.submissions.length, count + 1); assert.equal(e.copies, 2);
    assert.equal(a.view, bindings.get(target.texture).view, 'new content does not invalidate material bindings');
    await owner.whenIdle(); owner.dispose();
  });

test('independent ping-pong targets share pipelines, not attachments or content', () => {
  const {owner, target, calls} = fixture(), other = new T.RenderTarget();
  const readA = owner.bindingsFor(other), readB = owner.bindingsFor(target);
  const a = readA.get(target.texture), b = readB.get(other.texture);
  assert.notEqual(a.view, b.view); assert.equal(calls.pipelines.length, 1);
  assert.throws(() => readA.get(other.texture), {code: 'THREE_TARGET_FEEDBACK'});
  assert.throws(() => readB.get(target.texture), {code: 'THREE_TARGET_FEEDBACK'});
  owner.dispose();
});

test('sampler changes replace only the sampler; source versions stay acknowledged', () => {
  const {owner, target, calls} = fixture(), bindings = owner.bindingsFor();
  const a = bindings.get(target.texture); target.texture.wrapS = T.RepeatWrapping; target.texture.version++;
  target.texture.source.version++; const b = bindings.get(target.texture);
  assert.notEqual(a.sampler, b.sampler); assert.equal(a.view, b.view);
  assert.equal(b.sampler.addressModeU, 'repeat'); assert.equal(b.version, 1); assert.equal(b.sourceVersion, 1);
  assert.equal(calls.submissions.length, 1); owner.dispose();
});

test('resize/dispose invalidates queued tokens and all existing sampling views', () => {
  const {owner, target, calls} = fixture(), map = owner.bindingsFor(), e = owner.capture(target);
  const a = map.get(target.texture); const old = calls.textures.slice();
  target.setSize(8, 4);
  assert.throws(() => owner.ensure(e), {code: 'THREE_TARGET_STALE'});
  assert.ok(old.every(t => t.destroyed === 1));
  const b = map.get(target.texture); assert.notEqual(a.view, b.view);
  assert.equal(b.view.texture.width, 8); assert.equal(owner.size, 1);
  target.dispose(); assert.equal(owner.size, 0); assert.equal(owner.allocatedBytes, 0); owner.dispose();
});

test('direct configuration edits also invalidate already queued work', () => {
  const {owner, target} = fixture(), e = owner.capture(target); target.width++;
  assert.throws(() => owner.assertCurrent(e), {code: 'THREE_TARGET_STALE'});
  const next = owner.capture(target); assert.notEqual(next, e);
  owner.ensure(next); owner.dispose();
});

test('source disposal inside a submission does not destroy a borrowed attachment early', () => {
  const {owner, target, calls} = fixture(), e = owner.capture(target);
  assert.throws(() => owner.render(e, () => {
    target.dispose(); assert.ok(calls.textures.every(t => t.destroyed === 0));
  }), {code: 'THREE_TARGET_STALE'});
  assert.ok(calls.textures.every(t => t.destroyed === 1)); owner.dispose();
});

test('global allocation budget includes sampling replicas and remains retryable', () => {
  const {owner, target, calls} = fixture({maxBytes: 80});
  const e = owner.capture(target); owner.ensure(e); assert.equal(owner.allocatedBytes, 64);
  assert.throws(() => owner.bindingsFor().get(target.texture), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(calls.textures.length, 2); assert.equal(e.alive, true);
  target.setSize(2, 2); owner.bindingsFor().get(target.texture); assert.equal(owner.allocatedBytes, 48);
  owner.dispose();
});

test('allocation-free feedback/format refusals and bounded target count', () => {
  const {owner, target, calls} = fixture({maxTargets: 1});
  assert.throws(() => owner.bindingsFor(target).get(target.texture), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(calls.textures.length, 0);
  owner.capture(target); assert.throws(() => owner.capture(new T.RenderTarget()), {code: 'THREE_TARGET_BUDGET'});
  for (const mutate of [t => t.stencilBuffer = true, t => t.depthTexture = {}, t => t.samples = 2,
    t => t.texture.generateMipmaps = true, t => t.texture.type = 999, t => t.textures.push(t.texture), t => t.depth = 6]) {
    const t = new T.RenderTarget(); mutate(t); assert.throws(() => inspectThreeRenderTarget(t, T), /THREE_TARGET_/);
  }
  owner.dispose();
});

test('asynchronous GPU rejection fails the owner rather than publishing success', async () => {
  const {owner, target, controls, calls} = fixture(); owner.ensure(owner.capture(target));
  controls.error = {message: 'invalid UV pipeline'}; owner.bindingsFor().get(target.texture);
  await assert.rejects(owner.whenIdle(), /invalid UV pipeline/);
  await Promise.resolve(); await assert.rejects(owner.whenIdle(), /invalid UV pipeline/);
  owner.dispose(); assert.ok(calls.textures.every(t => t.destroyed === 1));
});

test('readback session uses native attachment, not the flipped sampling image', () => {
  const {owner, target} = fixture(), e = owner.ensure(owner.capture(target));
  const sample = owner.bindingsFor().get(target.texture);
  e.session.withTexture(texture => assert.notEqual(texture, sample.view.texture));
  target.dispose(); assert.equal(e.session.disposed, true);
  assert.throws(() => e.session.withTexture(() => {}), {code: 'THREE_TARGET_STALE'}); owner.dispose();
});

test('device loss cannot turn dead native textures into usable sampling bindings', async () => {
  const {owner, target, lose} = fixture(); owner.bindingsFor().get(target.texture);
  lose({message: 'lost device'}); await Promise.resolve();
  assert.throws(() => owner.bindingsFor().get(target.texture), {code: 'THREE_TARGET_GPU'});
  await assert.rejects(owner.whenIdle(), /lost device/); owner.dispose();
});
