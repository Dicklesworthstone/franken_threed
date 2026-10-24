/** Persistent native history identities; browser parity separately verifies shader pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';
import {fixture as transferFixture} from './fixtures/cube_renderer_transfer_fixture.mjs';

function fixture(options = {}) {
  const f = gpuPostprocessFixture(), create = f.device.createTexture;
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.features = new Set();
  f.device.createTexture = descriptor => {
    const texture = create(descriptor);
    texture.sampleCount = descriptor.sampleCount; texture.mipLevelCount = descriptor.mipLevelCount;
    return texture;
  };
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  return {...f, owner};
}
function source(samples = 0) {
  const pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(), {samples});
  pass.getTextureNode('normal'); pass.getPreviousTextureNode(); pass.setSize(16, 8);
  return pass;
}
const destroyed = textures => textures.forEach(texture => assert.equal(texture.destroyed, 1));
const resolved = frame => frame.resolveTarget?.texture ?? frame.colorView.texture;

for (const samples of [1, 4]) test(`raw history preserves each ${samples}-sample color residency and shares other outputs/depth`, async () => {
  const f = fixture(), raw = createGpuRenderTarget(f.device, {width: 16, height: 8,
    colorFormats: ['rgba16float', 'rgba8unorm'], sampleCount: samples, sampleDepth: true});
  let first, next;
  raw.withFrame(frame => { first = frame; }); raw.reserveColorHistory([0]);
  raw.withFrame(frame => { next = frame; }, {colorSlots: [1, 0]});
  assert.notEqual(next.colorView, first.colorView); assert.notEqual(resolved(next), resolved(first));
  assert.equal(next.colorViews[1], first.colorViews[1]); assert.equal(next.depthView, first.depthView);
  if (samples === 4) assert.notEqual(next.resolveTargets[0], first.resolveTargets[0]);
  raw.withTexture(texture => assert.equal(texture, resolved(next)), {colorSlots: [1, 0]});
  raw.withAttachments((colors, depth, history) => {
    assert.equal(colors[0], resolved(first)); assert.equal(history[0], resolved(next));
    assert.equal(depth, first.depthView.texture); assert.equal(history[1], null);
  });
  const bytes = 16 * 8 * (20 * (samples === 4 ? 5 : 1) + 4 * samples);
  assert.equal(raw.allocatedBytes, bytes);
  raw.reserveColorHistory([0]); assert.equal(raw.allocatedBytes, bytes);
  assert.throws(() => raw.withFrame(() => {}, {colorSlots: [0, 1]}), {code: 'GPU_TARGET_ATTACHMENT'});
  await raw.whenIdle(); raw.dispose(); destroyed(f.calls.textures); f.owner.dispose();
});

for (const samples of [0, 4]) test(`PassNode alternates persistent colors with ${samples || 1} samples and permits only detached feedback`, async () => {
  const f = fixture(), pass = source(samples), target = pass.renderTarget;
  f.owner.registerNativePass(pass);
  const original = target.texture, previous = pass._previousTextures.output;
  const images = new Map(); let sharedNormal, sharedDepth;
  for (let frameIndex = 0; frameIndex < 5; frameIndex++) {
    pass.toggleTexture('output'); f.owner.registerNativePass(pass);
    const entry = f.owner.capture(target), before = f.owner.captureNativeTexture(pass._previousTextures.output);
    const current = f.owner.captureNativeTexture(target.texture), depth = f.owner.captureNativeTexture(target.depthTexture);
    f.owner.render(entry, frame => {
      const image = resolved(frame), old = images.get(target.texture);
      if (old) assert.equal(image, old);
      images.set(target.texture, image);
      f.owner.withNativeTextures([before], bindings => {
        const history = bindings.get(before.texture);
        assert.notEqual(history, image);
        assert.equal(history.frameIndex, frameIndex ? frameIndex - 1 : undefined);
      });
      assert.throws(() => f.owner.withNativeTextures([current], () => {}), {code: 'THREE_TARGET_FEEDBACK'});
      assert.throws(() => f.owner.withNativeTextures([depth], () => {}), {code: 'THREE_TARGET_FEEDBACK'});
      if (sharedNormal) assert.equal(sharedNormal, frame.colorViews[1]);
      if (sharedDepth) assert.equal(sharedDepth, frame.depthView);
      sharedNormal = frame.colorViews[1]; sharedDepth = frame.depthView; image.frameIndex = frameIndex;
    });
    f.owner.withNativeTextures([before, current], bindings => {
      assert.notEqual(bindings.get(before.texture), bindings.get(current.texture));
      assert.equal(bindings.get(current.texture).frameIndex, frameIndex);
    });
  }
  assert.equal(images.size, 2); assert.equal(images.has(original), true); assert.equal(images.has(previous), true);
  const colors = 16 + 8, factor = samples === 4 ? 5 : 1;
  assert.equal(f.owner.allocatedBytes, 16 * 8 * (colors * factor + 4 * (samples || 1)));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.calls.textures);
});

test('history source captures group current and previous texture leases into one submitted native borrow', async () => {
  const f = fixture(), pass = source(); f.owner.registerNativePass(pass);
  const inputs = createThreeComputeTargets({targets: f.owner, device: f.device});
  const current = inputs.captureBinding({isSampledTexture: true, texture: pass.renderTarget.texture});
  const previous = inputs.captureBinding({isSampledTexture: true, texture: pass._previousTextures.output});
  inputs.withDispatch([current, previous], () => {
    const a = inputs.resolve(current), b = inputs.resolve(previous);
    assert.notEqual(a.texture, b.texture);
  });
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  inputs.withDispatch([current, previous], () => assert.notEqual(inputs.resolve(current).texture, inputs.resolve(previous).texture));
  inputs.release(current); inputs.release(previous); await inputs.whenIdle(); inputs.dispose();
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.calls.textures);
});

test('adding a history after rendering retains the current image, depth and unrelated MRT output', async () => {
  const f = fixture(), pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(), {samples: 0});
  pass.getTextureNode('normal'); pass.setSize(16, 8);
  let first;
  const original = f.owner.capture(pass.renderTarget);
  f.owner.render(original, frame => { first = frame; resolved(frame).value = 47; });
  pass.getPreviousTextureNode(); f.owner.registerNativePass(pass);
  f.owner.render(f.owner.capture(pass.renderTarget), frame => assert.equal(frame.colorView, first.colorView));
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  const history = f.owner.captureNativeTexture(pass._previousTextures.output);
  f.owner.render(f.owner.capture(pass.renderTarget), frame => {
    assert.equal(frame.colorViews[1], first.colorViews[1]); assert.equal(frame.depthView, first.depthView);
    f.owner.withNativeTexture(history, texture => assert.equal(texture.value, 47));
  });
  // A second late history needs another slot, not replacement of either existing image.
  pass.getPreviousTextureNode('normal'); f.owner.registerNativePass(pass); pass.toggleTexture('normal');
  f.owner.registerNativePass(pass);
  const normalHistory = f.owner.captureNativeTexture(pass._previousTextures.normal);
  f.owner.withNativeTexture(normalHistory, texture => assert.equal(texture, first.colorViews[1].texture));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.calls.textures);
});

test('automatic pass resize retains queued history generations until readers and GPU completion finish', async () => {
  const f = fixture(), pass = source(); f.owner.registerNativePass(pass);
  const a = f.owner.captureNativeTexture(pass.renderTarget.texture), b = f.owner.captureNativeTexture(pass._previousTextures.output);
  const releaseA = f.owner.retainGeneration(a.entry), releaseB = f.owner.retainGeneration(b.entry);
  f.owner.withNativeTextures([a, b], () => {});
  const initial = f.calls.textures.slice(), fence = deferred(); f.controls.completion = fence;
  f.owner.withNativePassResize(pass.renderTarget, 32, 16, () => pass.setSize(32, 16));
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  const next = f.owner.capture(pass.renderTarget); f.owner.render(next, frame => assert.equal(resolved(frame).width, 32));
  f.owner.withNativeTextures([a, b], textures => {
    for (const texture of textures.values()) assert.equal(texture.width, 16);
  });
  releaseA(); releaseB(); initial.forEach(texture => assert.equal(texture.destroyed, 0));
  fence.resolve(); await f.owner.whenIdle(); destroyed(initial);
  f.owner.dispose(); destroyed(f.calls.textures);
});

test('history budget rejection precedes allocation and explicit disposal invalidates both captured halves', async () => {
  const f = fixture({maxBytes: 16 * 8 * 28 - 1}), pass = source(); f.owner.registerNativePass(pass);
  const entry = f.owner.capture(pass.renderTarget);
  assert.throws(() => f.owner.ensure(entry), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
  for (const which of ['target', 'previous']) {
    const q = fixture(), p = source(); q.owner.registerNativePass(p);
    const a = q.owner.captureNativeTexture(p.renderTarget.texture), b = q.owner.captureNativeTexture(p._previousTextures.output);
    q.owner.withNativeTextures([a, b], () => {});
    (which === 'target' ? p.renderTarget : p._previousTextures.output).dispose();
    assert.throws(() => q.owner.assertCurrent(a.entry), {code: 'THREE_TARGET_STALE'});
    assert.throws(() => q.owner.assertCurrent(b.entry), {code: 'THREE_TARGET_STALE'});
    q.owner.dispose(); destroyed(q.calls.textures);
  }
});

test('selected history color handles preserve upload, readback and copy endpoints', async () => {
  const f = transferFixture(), owner = createThreeRenderTargets(T, {getDevice: () => f.device});
  const pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(), {samples: 0, type: T.UnsignedByteType});
  pass.getPreviousTextureNode(); pass.setSize(4, 4); owner.registerNativePass(pass);
  const current = owner.captureNativeTexture(pass.renderTarget.texture).entry;
  const previous = owner.captureNativeTexture(pass._previousTextures.output).entry;
  owner.ensure(current); owner.ensure(previous);
  const a = Uint8Array.from({length: 64}, (_, i) => i), b = Uint8Array.from(a, value => 255 - value);
  owner.writePixels(current, a); owner.writePixels(previous, b);
  assert.deepEqual(await current.storage.readPixels(), a); assert.deepEqual(await previous.storage.readPixels(), b);
  const other = owner.capture(new T.RenderTarget(4, 4, {depthTexture: new T.DepthTexture(4, 4)}));
  owner.copyColor(previous, other); assert.deepEqual(await other.storage.readPixels(), b);
  owner.writePixels(other, a); owner.copyColor(other, previous);
  assert.deepEqual(await previous.storage.readPixels(), a); assert.deepEqual(await current.storage.readPixels(), a);
  owner.copyDepth(previous, other);
  assert.throws(() => previous.storage.copyFrom(current.storage), {code: 'GPU_TARGET_FEEDBACK'});
  pass.toggleTexture('output'); owner.registerNativePass(pass);
  assert.equal(owner.capture(pass.renderTarget), previous);
  await owner.whenIdle(); owner.dispose(); f.dispose();
  f.textures.forEach(texture => assert.equal(texture.destroyed, true));
});

test('history allocation failure retires each partially allocated resolved/MSAA image once', () => {
  for (const offset of [1, 2]) {
    const f = fixture(), raw = createGpuRenderTarget(f.device, {width: 16, height: 8, sampleCount: 4});
    f.controls.textureFailure = f.calls.textures.length + offset;
    assert.throws(() => raw.reserveColorHistory([0]), /injected/);
    assert.equal(raw.failed, true); raw.dispose(); destroyed(f.calls.textures); f.owner.dispose();
  }
});
