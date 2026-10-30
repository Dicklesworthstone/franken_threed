import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {depthDevice} from './tests/depth_transfer_test_fixture.mjs';

// Real source ownership/events and a recording GPU boundary. Pixel evidence
// lives in tests/e2e/webgpu_renderer/native_shared_depth.mjs.
function fixture(samples = 0, options = {}) {
  const f = depthDevice(), depth = new T.DepthTexture(4, 3, T.FloatType);
  const a = new T.RenderTarget(4, 3, {depthTexture: depth, samples});
  const b = new T.RenderTarget(4, 3, {depthTexture: depth, samples});
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  return {...f, owner, depth, a, b};
}

for (const samples of [0, 4]) test(`shared depth uses one native image and one byte charge with ${samples || 1} samples`, async () => {
  const f = fixture(samples), {owner, a, b, depth} = f;
  const ea = owner.ensure(owner.capture(a)), eb = owner.ensure(owner.capture(b));
  let native;
  owner.render(ea, frame => { native = frame.depthView.texture; native.data.fill(.25); });
  owner.render(eb, frame => {
    assert.equal(frame.depthView.texture, native);
    assert.ok(native.data.every(value => value === .25)); native.data.fill(.75);
  });
  owner.render(ea, frame => assert.ok(frame.depthView.texture.data.every(value => value === .75)));
  assert.equal(depth.renderTarget, a); assert.equal(b.depthTexture, a.depthTexture);
  assert.equal(f.textures.filter(texture => texture.format === 'depth32float').length, 1);
  assert.equal(owner.allocatedBytes, 4 * 3 * (8 * (samples === 4 ? 5 : 1) + 4 * (samples || 1)));
  b.dispose(); assert.equal(native.destroyed, false); assert.equal(eb.alive, false);
  owner.render(ea, frame => assert.equal(frame.depthView.texture, native));
  assert.equal(owner.allocatedBytes, 4 * 3 * (4 * (samples === 4 ? 5 : 1) + 4 * (samples || 1)));
  await owner.whenIdle(); owner.dispose(); assert.ok(f.textures.every(texture => texture.destroyed));
});

test('borrower depth writes invalidate the shared WebGL depth replica, not owner color', () => {
  const f = fixture(), {owner, a, b, depth} = f;
  const ea = owner.ensure(owner.capture(a)), eb = owner.ensure(owner.capture(b));
  owner.render(ea, frame => frame.depthView.texture.data.fill(.25));
  const bindings = owner.bindingsFor(), first = bindings.get(depth), copies = ea.depthCopies;
  assert.ok(first.view.texture.data.every(value => value === .25));
  owner.render(eb, frame => frame.depthView.texture.data.fill(.75));
  assert.equal(bindings.get(depth).view, first.view); assert.equal(ea.depthCopies, copies + 1);
  assert.ok(first.view.texture.data.every(value => value === .75)); owner.dispose();
});

test('color-only borrower samples shared depth with independent color extent and samples', () => {
  const f = fixture(), {owner, a, b, depth} = f;
  const ea = owner.ensure(owner.capture(a));
  b.depthBuffer = false; b.samples = 4; b.setSize(2, 2);
  const eb = owner.capture(b), source = owner.captureNativeTexture(depth);
  owner.withNativeTexture(source, native => owner.render(eb, frame => {
    assert.equal(native.width, 4); assert.equal(native.height, 3);
    assert.equal(frame.depthView, undefined); assert.equal(frame.colorView.texture.width, 2);
    assert.equal(frame.colorView.texture.sampleCount, 4);
  }));
  assert.equal(f.textures.filter(texture => texture.format === 'depth32float').length, 1);
  assert.doesNotThrow(() => owner.bindingsFor(b).get(depth));
  assert.equal(ea.depthDirty, false); owner.dispose();
});

test('shared depth cannot be read while attached for writing in either borrow order', () => {
  const f = fixture(), {owner, a, b, depth} = f;
  owner.ensure(owner.capture(a)); const eb = owner.ensure(owner.capture(b));
  const source = owner.captureNativeTexture(depth);
  assert.throws(() => owner.bindingsFor(b).get(depth), {code: 'THREE_TARGET_FEEDBACK'});
  assert.throws(() => owner.withNativeTexture(source, () => owner.render(eb, () => {})), {code: 'THREE_TARGET_FEEDBACK'});
  assert.throws(() => owner.render(eb, () => owner.withNativeTexture(source, () => {})), {code: 'THREE_TARGET_FEEDBACK'});
  owner.dispose();
});

test('history color swaps do not hide shared depth read/write feedback', () => {
  const f = depthDevice(), owner = createThreeRenderTargets(T, {getDevice: () => f.device});
  const pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera());
  pass.getPreviousTextureNode(); pass.setSize(4, 3); owner.registerNativePass(pass);
  const a = pass.renderTarget, b = new T.RenderTarget(4, 3, {type: T.HalfFloatType, depthTexture: a.depthTexture});
  owner.ensure(owner.capture(a)); const eb = owner.ensure(owner.capture(b));
  const source = owner.captureNativeTexture(a.depthTexture);
  assert.throws(() => owner.withNativeTexture(source, () => owner.render(eb, () => {})), {code: 'THREE_TARGET_FEEDBACK'});
  owner.dispose();
});

test('attached depth dimension/sample mismatches fail without allocating private replacement depth', () => {
  for (const edit of [b => b.setSize(2, 2), b => { b.samples = 4; }]) {
    const f = fixture(), {owner, a, b} = f;
    owner.ensure(owner.capture(a)); edit(b); const count = f.textures.length;
    assert.throws(() => owner.ensure(owner.capture(b)), {code: 'THREE_TARGET_DEPTH'});
    assert.equal(f.textures.length, count); owner.dispose();
  }
});

test('borrower resize/dispose preserve owner depth; owner disposal invalidates every borrower', () => {
  const f = fixture(), {owner, a, b} = f;
  const ea = owner.ensure(owner.capture(a)), eb = owner.ensure(owner.capture(b));
  const original = f.textures.find(texture => texture.format === 'depth32float');
  b.setSize(8, 6); assert.equal(eb.alive, false); assert.equal(original.destroyed, false);
  assert.equal(owner.assertCurrent(ea), ea);
  b.setSize(4, 3); const next = owner.ensure(owner.capture(b));
  a.dispose(); assert.equal(next.alive, false); assert.equal(original.destroyed, true);
  assert.throws(() => owner.render(next, () => {}), {code: 'THREE_TARGET_STALE'}); owner.dispose();
});

test('owner color disposal and direct configuration changes retire dependent shared allocations', () => {
  for (const change of [f => f.a.texture.dispose(), f => { f.a.samples = 4; f.owner.capture(f.a); }]) {
    const f = fixture(), {owner, a, b} = f;
    owner.ensure(owner.capture(a)); const eb = owner.ensure(owner.capture(b));
    const resources = f.textures.slice(); change(f);
    assert.equal(eb.alive, false); assert.ok(resources.every(texture => texture.destroyed));
    assert.equal(owner.allocatedBytes, 0); owner.dispose();
  }
});

test('asynchronous borrower bindings keep shared depth charged until their last lease releases', () => {
  const f = fixture(0, {maxBytes: 192}), {owner, a, b} = f;
  owner.ensure(owner.capture(a)); owner.ensure(owner.capture(b));
  const binding = owner.bindingsFor().get(b.texture), finish = binding.retain();
  assert.equal(owner.allocatedBytes, 192);
  const resources = f.textures.slice(); a.dispose();
  assert.throws(() => binding.validate(), {code: 'THREE_TARGET_STALE'});
  assert.equal(owner.allocatedBytes, 192, 'borrowed depth and its owning allocation remain charged');
  assert.ok(resources.every(texture => !texture.destroyed));
  assert.throws(() => owner.ensure(owner.capture(new T.RenderTarget(1, 1))), {code: 'THREE_TARGET_BUDGET'});
  finish(); finish(); assert.equal(owner.allocatedBytes, 0);
  assert.ok(resources.every(texture => texture.destroyed)); owner.dispose();
});

test('accepted shared-depth generations survive automatic pass resizing together', async () => {
  const f = fixture(), {owner, a, b} = f;
  owner.ensure(owner.capture(a)); const eb = owner.ensure(owner.capture(b));
  const finish = owner.retainGeneration(eb), original = f.textures.find(texture => texture.format === 'depth32float');
  owner.withNativeEffectTargets([a, b], () => { a.setSize(2, 2); b.setSize(2, 2); });
  assert.equal(owner.assertCurrent(eb), eb);
  owner.ensure(owner.capture(a)); const next = owner.ensure(owner.capture(b));
  owner.render(eb, frame => { assert.equal(frame.depthView.texture, original); assert.equal(frame.colorView.texture.width, 4); });
  owner.render(next, frame => { assert.notEqual(frame.depthView.texture, original); assert.equal(frame.colorView.texture.width, 2); });
  finish(); await owner.whenIdle(); await owner.whenIdle();
  assert.equal(original.destroyed, true); assert.equal(next.alive, true); owner.dispose();
});

test('source clones distinguish independent owned depth from shared borrowed depth', () => {
  const f = fixture(), {owner, a, b, depth} = f;
  owner.ensure(owner.capture(a)); const ac = a.clone(), bc = b.clone();
  assert.notEqual(ac.depthTexture, depth); assert.equal(ac.depthTexture.renderTarget, ac);
  assert.equal(bc.depthTexture, depth); assert.equal(bc.depthTexture.renderTarget, a);
  const independent = owner.ensure(owner.capture(ac)), shared = owner.ensure(owner.capture(bc));
  let sourceDepth;
  owner.withNativeTexture(owner.captureNativeTexture(depth), native => { sourceDepth = native; });
  owner.render(shared, frame => assert.equal(frame.depthView.texture, sourceDepth));
  owner.render(independent, frame => assert.notEqual(frame.depthView.texture, sourceDepth)); owner.dispose();
});

test('borrower-first admission never eagerly initializes or resizes its source owner', () => {
  const f = fixture(), {owner, a, b, depth} = f;
  depth.image.width = depth.image.height = 1;
  const image = {...depth.image}, version = depth.version, eb = owner.capture(b);
  assert.equal(f.textures.length, 0);
  assert.throws(() => owner.ensure(eb), {code: 'THREE_TARGET_UNINITIALIZED'});
  assert.deepEqual(depth.image, image); assert.equal(depth.version, version); assert.equal(depth.renderTarget, a);
  owner.ensure(owner.capture(a)); assert.doesNotThrow(() => owner.ensure(eb)); owner.dispose();
});

test('shared depth charges its owning target against the same target-count budget', () => {
  const f = fixture(0, {maxTargets: 1});
  assert.throws(() => f.owner.capture(f.b), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.owner.size, 1); assert.equal(f.textures.length, 0); f.owner.dispose();
});
