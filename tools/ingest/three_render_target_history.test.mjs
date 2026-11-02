/** Persistent native history identities; browser parity separately verifies shader pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';
import {captureTargetColorCopy} from './three_target_texture_copy.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';
import {gpuTargetMipmapFixture} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';
import {fixture as transferFixture} from './fixtures/cube_renderer_transfer_fixture.mjs';
import {fixture as colorCopyFixture} from './fixtures/target_storage_copy_fixture.mjs';

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

test('source setup refreshes cold history metadata but shape edits invalidate retained or allocated generations', () => {
  for (const state of ['cold', 'retained', 'allocated']) {
    const f = fixture(), pass = source(); f.owner.registerNativePass(pass);
    const before = f.owner.capture(pass.renderTarget);
    const release = state === 'retained' ? f.owner.retainGeneration(before) : null;
    if (state === 'allocated') f.owner.ensure(before);
    pass.renderTarget.samples = 4;
    if (state === 'cold') {
      const current = f.owner.captureNativeTexture(pass.renderTarget.texture);
      const previous = f.owner.captureNativeTexture(pass._previousTextures.output);
      assert.equal(current.sampleData.samples, 4); assert.equal(previous.sampleData.samples, 4);
      assert.notEqual(current.entry, before); assert.equal(f.calls.textures.length, 0);
    } else assert.throws(() => f.owner.capture(pass.renderTarget), {code: 'THREE_TARGET_STALE'});
    assert.throws(() => f.owner.assertCurrent(before), {code: 'THREE_TARGET_STALE'});
    release?.(); f.owner.dispose(); destroyed(f.calls.textures);
  }
});

function mipFixture() {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  return f;
}
const mipPasses = f => f.submissions.flatMap(commands => commands.flatMap(command => command.passes));

for (const sampleCount of [1, 4]) test(`history mip chains retain independent physical images, caches and ${sampleCount}-sample budgets`, async () => {
  const f = mipFixture(), raw = createGpuRenderTarget(f.device, {width: 8, height: 4, sampleCount,
    colorFormats: ['rgba8unorm-srgb', 'rgba16float'], colorMipLevelCounts: [4, 3]});
  const msaa = sampleCount === 4 ? 4 : 0, colorCosts = [43 * 4 + 32 * 4 * msaa, 42 * 8 + 32 * 8 * msaa];
  const baseBytes = colorCosts[0] + colorCosts[1] + 32 * 4 * sampleCount;
  assert.equal(raw.allocatedBytes, baseBytes);
  let first;
  raw.withFrame(frame => { first = frame; });
  raw.reserveColorHistory([0]); assert.equal(raw.allocatedBytes, baseBytes + colorCosts[0]);
  raw.reserveColorHistory([1]); assert.equal(raw.allocatedBytes, baseBytes + colorCosts[0] + colorCosts[1]);
  const a = raw.selectColorSlots([0, 0]), b = raw.selectColorSlots([1, 1]);
  let other;
  b.withFrame(frame => { other = frame; });
  assert.equal(first.depthView, other.depthView);
  const aColors = (first.resolveTargets ?? first.colorViews).map(view => view.texture);
  const bColors = (other.resolveTargets ?? other.colorViews).map(view => view.texture);
  assert.deepEqual(bColors.map(texture => texture.mipLevelCount), [4, 3]);
  for (let index = 0; index < 2; index++) {
    assert.notEqual(aColors[index], bColors[index]);
    assert.equal(other.colorViews[index].texture.sampleCount, sampleCount);
    assert.equal((other.resolveTargets ?? other.colorViews)[index].descriptor.mipLevelCount, 1);
  }
  for (const [target, textures] of [[a, aColors], [b, bColors]]) {
    const start = mipPasses(f).length;
    assert.equal(target.generateMipmaps(), true);
    assert.deepEqual(mipPasses(f).slice(start).map(pass => pass.descriptor.colorAttachments[0].view.texture),
      [textures[0], textures[0], textures[0], textures[1], textures[1]]);
    assert.equal(target.generateMipmaps(), false);
  }
  const groups = f.groups.length;
  assert.equal(f.pipelines.length, 2, 'formats share pipelines across both history images');
  b.withFrame(() => {});
  assert.equal(a.generateMipmaps(), false, 'writing the alternate does not dirty the previous frame');
  assert.equal(b.generateMipmaps([1]), true); assert.equal(b.generateMipmaps([0]), true);
  assert.equal(f.groups.length, groups, 'each physical image reuses its mip views and bind groups');
  const mixed = raw.selectColorSlots([1, 0]); mixed.withFrame(() => {});
  assert.equal(a.generateMipmaps([0]), false); assert.equal(b.generateMipmaps([1]), false);
  const start = mipPasses(f).length; mixed.generateMipmaps();
  assert.deepEqual(mipPasses(f).slice(start).map(pass => pass.descriptor.colorAttachments[0].view.texture),
    [bColors[0], bColors[0], bColors[0], aColors[1], aColors[1]]);
  if (sampleCount === 4)
    assert.throws(() => b.withFrame(() => assert.fail('MSAA cannot render higher levels'), {mipLevel: 1}), {code: 'GPU_TARGET_MIP'});
  await raw.whenIdle(); raw.dispose(); destroyed(f.textures);
});

test('full history pyramid budgets reject before allocation and remain retryable', () => {
  for (const sampleCount of [1, 4]) {
    const f = mipFixture(), colorBytes = 43 * 4 + (sampleCount === 4 ? 32 * 4 * 4 : 0);
    const raw = createGpuRenderTarget(f.device, {width: 8, height: 4, format: 'rgba8unorm', depthFormat: null,
      mipLevelCount: 4, sampleCount, maxBytes: 2 * colorBytes - 1});
    const count = f.textures.length;
    assert.throws(() => raw.reserveColorHistory([0]), {code: 'GPU_TARGET_BUDGET'});
    assert.equal(f.textures.length, count); assert.equal(raw.allocatedBytes, colorBytes); assert.equal(raw.failed, false);
    raw.withFrame(() => {}); raw.dispose(); destroyed(f.textures);
  }
});

test('authored history mip frames retain selected colors and share each matching private depth level', async () => {
  const f = mipFixture(), raw = createGpuRenderTarget(f.device, {width: 8, height: 4,
    colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [4, 3]});
  raw.reserveColorHistory([0]);
  const a = raw.selectColorSlots([0, 0]), b = raw.selectColorSlots([1, 0]);
  let first, other;
  a.withFrame(frame => { first = frame; }, {mipLevel: 1});
  b.withFrame(frame => { other = frame; }, {mipLevel: 1});
  assert.notEqual(first.colorView.texture, other.colorView.texture);
  assert.equal(first.colorViews[1], other.colorViews[1]); assert.equal(first.depthView, other.depthView);
  assert.deepEqual(other.colorViews.map(view => view.descriptor.baseMipLevel), [1, 1]);
  const bytes = raw.allocatedBytes;
  b.withFrame(frame => assert.equal(frame, other), {mipLevel: 1});
  assert.equal(raw.allocatedBytes, bytes);
  b.withFrame(frame => {
    assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseMipLevel), [2, 2]);
    assert.deepEqual([frame.depthView.texture.width, frame.depthView.texture.height], [2, 1]);
    assert.notEqual(frame.depthView, other.depthView);
  }, {mipLevel: 2});
  a.withAttachments((colors, depth, history, withFrame) => {
    withFrame(frame => assert.equal(frame, other), {colorSlots: [1, 0], mipLevel: 1, generateColorMipmaps: []});
    assert.throws(() => withFrame(() => assert.fail('missing secondary mip'), {colorSlots: [1, 0], mipLevel: 3}), {code: 'GPU_TARGET_MIP'});
  });
  assert.equal(f.submissions.length, 0, 'authored levels survive sampling and independent writes');
  assert.equal(raw.allocatedBytes, bytes + 2 * 1 * 4);
  await raw.whenIdle(); raw.dispose(); destroyed(f.textures);
});

test('nested history writes filter captured resolved images after submission without reentering or escaping the borrow', async () => {
  const f = mipFixture(), raw = createGpuRenderTarget(f.device, {width: 8, height: 4, sampleCount: 4, depthFormat: null,
    colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [3, 2]});
  raw.reserveColorHistory([0, 1]);
  const a = raw.selectColorSlots([0, 0]), b = raw.selectColorSlots([1, 1]);
  a.generateMipmaps(); b.generateMipmaps();
  let leaked, previous, selected;
  a.withAttachments((colors, depth, history, withFrame) => {
    leaked = withFrame; previous = colors; selected = history;
    const slots = [1, 1], generate = [1], before = f.submissions.length;
    withFrame(frame => {
      assert.equal(frame.resolveTargets[1].texture, history[1]);
      assert.notEqual(frame.resolveTargets[1].texture, colors[1]);
      assert.throws(() => b.generateMipmaps(), {code: 'GPU_TARGET_REENTRANT'});
      assert.throws(() => raw.withFrame(() => {}), {code: 'GPU_TARGET_REENTRANT'});
      f.device.queue.submit([{passes: [], scene: true}]);
      slots[1] = 0; generate[0] = 0;
    }, {colorSlots: slots, generateColorMipmaps: generate});
    assert.equal(f.submissions.length, before + 2);
    assert.equal(f.submissions[before][0].scene, true);
    assert.equal(f.submissions[before + 1][0].passes[0].descriptor.colorAttachments[0].view.texture, history[1]);
    assert.throws(() => withFrame(() => assert.fail('invalid generation'), {generateColorMipmaps: [0, 0]}), {code: 'GPU_TARGET_ATTACHMENT'});
  });
  assert.equal(a.generateMipmaps(), false, 'held previous images retain their clean chains');
  assert.equal(b.generateMipmaps([1]), false); assert.equal(b.generateMipmaps([0]), true);
  assert.ok(previous.every((texture, index) => texture !== selected[index]));
  assert.throws(() => leaked(() => {}), {code: 'GPU_TARGET_BORROW'});
  a.withAttachments((colors, depth, history, withFrame) => {
    const before = f.submissions.length;
    assert.throws(() => withFrame(async () => {}, {colorSlots: [1, 1], generateColorMipmaps: [0]}), {code: 'GPU_TARGET_ASYNC_USE'});
    assert.equal(f.submissions.length, before, 'a rejected async callback cannot schedule filtering');
  });
  await raw.whenIdle(); raw.dispose(); destroyed(f.textures);
});

test('selected authored history mips preserve upload, copy and readback endpoints across resize', async () => {
  const f = transferFixture();
  const raw = createGpuRenderTarget(f.device, {width: 4, height: 4, format: 'rgba8unorm', depthFormat: null,
    mipLevelCount: 3, copyDestination: true});
  raw.reserveColorHistory([0]);
  const a = raw.selectColorSlots([0]), b = raw.selectColorSlots([1]);
  const first = Uint8Array.from({length: 16}, (_, index) => index + 1), previous = Uint8Array.from(first, value => 255 - value);
  a.writePixels(first, {destinationMipLevel: 1}); b.writePixels(previous, {destinationMipLevel: 1});
  const source = createGpuRenderTarget(f.device, {width: 4, height: 4, format: 'rgba8unorm', depthFormat: null,
    mipLevelCount: 3, copyDestination: true});
  const tail = new Uint8Array([19, 71, 131, 255]); source.writePixels(tail, {destinationMipLevel: 2});
  b.copyFrom(source, {sourceMipLevel: 2, destinationMipLevel: 2});
  assert.deepEqual(await b.readPixels({mipLevel: 2}), tail);
  assert.deepEqual(await a.readPixels({mipLevel: 2}), new Uint8Array(4));
  const pending = [a.readPixels({mipLevel: 1}), b.readPixels({mipLevel: 1})];
  raw.resize(8, 8);
  assert.deepEqual(await Promise.all(pending), [first, previous]);
  await raw.whenIdle(); raw.dispose(); source.dispose(); f.dispose();
  assert.ok(f.textures.every(texture => texture.destroyed));
});

function nativeMipFixture(options = {}) {
  const f = mipFixture();
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false, ...options});
  return {...f, owner};
}

for (const samples of [0, 4]) test(`native PassNode ${samples || 1}-sample pyramids preserve per-image flags, borrowed feedback and exact charges`, async () => {
  const f = nativeMipFixture(), pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(),
    {samples, type: T.UnsignedByteType, generateMipmaps: true, depthBuffer: false});
  pass.getTextureNode(); pass.getTextureNode('normal');
  pass.getTexture('normal').type = T.HalfFloatType; pass.getTexture('normal').mipmaps = [{}, {}, {}];
  pass.getPreviousTextureNode(); pass.getPreviousTextureNode('normal'); pass.setSize(8, 4);
  f.owner.registerNativePass(pass);
  const first = f.owner.capture(pass.renderTarget, {initialize: true});
  let firstImages;
  f.owner.render(first, frame => { firstImages = (frame.resolveTargets ?? frame.colorViews).map(view => view.texture); }, {generateColorMipmaps: [0, 1]});
  pass.toggleTexture('output'); pass.toggleTexture('normal'); f.owner.registerNativePass(pass);
  const next = f.owner.capture(pass.renderTarget), previous = ['output', 'normal'].map(name => f.owner.captureNativeTexture(pass._previousTextures[name]));
  let nextImages;
  f.owner.withNativeTextures(previous, bindings => f.owner.render(next, frame => {
    nextImages = (frame.resolveTargets ?? frame.colorViews).map(view => view.texture);
    for (let index = 0; index < 2; index++) {
      assert.equal(bindings.get(previous[index].texture), firstImages[index]);
      assert.notEqual(nextImages[index], firstImages[index]);
    }
    const current = f.owner.captureNativeTexture(pass.renderTarget.texture);
    assert.throws(() => f.owner.withNativeTexture(current, () => {}), {code: 'THREE_TARGET_FEEDBACK'});
  }, {generateColorMipmaps: [0, 1]}));
  const bytes = 2 * (43 * 4 + 42 * 8 + (samples === 4 ? 32 * 12 * 4 : 0));
  assert.equal(f.owner.allocatedBytes, bytes); assert.equal(next.storage.allocatedBytes, bytes);
  const physical = [...pass.renderTarget.textures, ...previous.map(source => source.texture)];
  for (const texture of physical) { texture.generateMipmaps = false; texture.mipmaps = [{}]; }
  f.owner.registerNativePass(pass);
  assert.equal(f.owner.capture(pass.renderTarget), next);
  assert.deepEqual(next.shape.colorMipLevelCounts, [4, 3]);
  const before = mipPasses(f).length;
  f.owner.render(next, () => {}, {generateColorMipmaps: []});
  f.owner.withNativeTextures(physical.map(texture => f.owner.captureNativeTexture(texture)), () => {});
  assert.equal(mipPasses(f).length, before, 'sampling and no-generation frames retain authored lower levels');
  f.owner.render(next, () => {}, {generateColorMipmaps: [1]});
  assert.deepEqual(mipPasses(f).slice(before).map(draw => draw.descriptor.colorAttachments[0].view.texture), [nextImages[1], nextImages[1]]);
  assert.equal(f.owner.allocatedBytes, bytes); assert.equal(f.pipelines.length, 2);
  physical[0].needsUpdate = true;
  assert.throws(() => f.owner.assertCurrent(next), {code: 'THREE_TARGET_STALE'});
  f.owner.dispose(); destroyed(f.textures);
});

test('a late single-color history adopts the existing generated image and keeps explicit LOD access', async () => {
  const f = nativeMipFixture(), pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(),
    {samples: 0, type: T.UnsignedByteType, generateMipmaps: true, depthBuffer: false});
  pass.getTextureNode(); pass.setSize(8, 4);
  const entry = f.owner.capture(pass.renderTarget, {initialize: true}); let original;
  f.owner.render(entry, (frame, texture) => { original = texture; }, {generateColorMipmaps: [0]});
  assert.equal(mipPasses(f).length, 3); assert.equal(f.owner.allocatedBytes, 43 * 4);
  pass.getPreviousTextureNode(); f.owner.registerNativePass(pass);
  const previous = f.owner.captureNativeTexture(pass._previousTextures.output);
  f.owner.withNativeTexture(previous, texture => assert.notEqual(texture, original));
  assert.equal(previous.sampler.lodMaxClamp, 3, 'ordinary LinearFilter retains explicit lower-LOD history access');
  assert.equal(f.owner.allocatedBytes, 2 * 43 * 4);
  pass.renderTarget.texture.generateMipmaps = false; previous.texture.generateMipmaps = false;
  f.owner.render(f.owner.capture(pass.renderTarget), (frame, texture) => assert.equal(texture, original), {generateColorMipmaps: []});
  f.owner.withNativeTexture(previous, () => {});
  assert.equal(mipPasses(f).length, 3, 'late adoption does not retain an implicit generation policy');
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  f.owner.withNativeTexture(f.owner.captureNativeTexture(pass._previousTextures.output), texture => assert.equal(texture, original));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('borrowed authored history mips respect the shared target budget and charge private depth once', async () => {
  const f = nativeMipFixture({maxBytes: 835}), pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(),
    {samples: 0, type: T.UnsignedByteType});
  pass.getTextureNode(); pass.getTextureNode('normal'); pass.getTexture('normal').type = T.HalfFloatType;
  for (const texture of pass.renderTarget.textures) texture.mipmaps = [{}, {}, {}];
  pass.renderTarget.depthTexture = null; pass.getPreviousTextureNode(); pass.setSize(8, 4);
  f.owner.registerNativePass(pass); const a = f.owner.capture(pass.renderTarget);
  f.owner.render(a, () => {});
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  const b = f.owner.capture(pass.renderTarget), previous = f.owner.captureNativeTexture(pass._previousTextures.output);
  const other = new T.RenderTarget(1, 1, {depthBuffer: false}); f.owner.ensure(f.owner.capture(other));
  assert.equal(f.owner.allocatedBytes, 804);
  const count = f.textures.length;
  f.owner.withNativeTexture(previous, () => {
    assert.throws(() => f.owner.render(b, () => assert.fail('shared budget must precede allocation'), {mipLevel: 1}), {code: 'THREE_TARGET_BUDGET'});
  });
  assert.equal(f.textures.length, count); assert.equal(f.owner.allocatedBytes, 804);
  other.dispose(); let depth;
  f.owner.withNativeTexture(previous, () => f.owner.render(b, frame => { depth = frame.depthView; }, {mipLevel: 1}));
  assert.equal(f.owner.allocatedBytes, 832); assert.equal(b.storage.allocatedBytes, 832);
  f.owner.ensure(a); f.owner.render(a, frame => assert.equal(frame.depthView, depth), {mipLevel: 1});
  assert.equal(f.owner.allocatedBytes, 832, 'switching history variants preserves the existing private-depth charge');
  await f.owner.whenIdle(); f.owner.dispose(); assert.equal(f.owner.allocatedBytes, 0); destroyed(f.textures);
});

test('automatic history resize retains accepted full chains when the next allocation disables generation', async () => {
  const f = nativeMipFixture(), pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(),
    {samples: 0, type: T.UnsignedByteType, generateMipmaps: true, depthBuffer: false});
  pass.getTextureNode(); pass.getPreviousTextureNode(); pass.setSize(8, 4); f.owner.registerNativePass(pass);
  const sources = [pass.getTexture('output'), pass.getPreviousTexture('output')].map(texture => f.owner.captureNativeTexture(texture));
  const releases = sources.map(source => f.owner.retainGeneration(source.entry));
  f.owner.withNativeTextures(sources, () => {});
  const initial = f.textures.slice(), fence = deferred(); f.controls.completion = fence;
  for (const source of sources) source.texture.generateMipmaps = false;
  f.owner.withNativePassResize(pass.renderTarget, 16, 8, () => pass.setSize(16, 8));
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  const current = f.owner.capture(pass.renderTarget); f.owner.render(current, () => {}, {generateColorMipmaps: []});
  assert.equal(current.shape.mipLevelCount, 1, 'the new allocation follows its own generation flag');
  f.owner.withNativeTextures(sources, textures => {
    for (const texture of textures.values()) {
      assert.equal(texture.width, 8); assert.equal(texture.mipLevelCount, 4);
    }
  });
  for (const release of releases) release(); initial.forEach(texture => assert.equal(texture.destroyed, 0));
  fence.resolve(); await f.owner.whenIdle(); destroyed(initial);
  f.owner.dispose(); destroyed(f.textures);
});

for (const type of [T.UnsignedByteType, T.FloatType]) test(`public raw target copies resolve previous history slots for type ${type} without a sampler`, async () => {
  const f = colorCopyFixture(), owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false});
  const pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(), {samples: 0, type, depthBuffer: false});
  pass.getTextureNode(); pass.renderTarget.texture.mipmaps = [{}, {}, {}]; pass.getPreviousTextureNode(); pass.setSize(4, 4);
  owner.registerNativePass(pass);
  const previous = owner.captureNativeTexture(pass._previousTextures.output, {sampled: false}), current = owner.capture(pass.renderTarget);
  if (type === T.FloatType)
    assert.throws(() => owner.captureNativeTexture(previous.texture), {code: 'THREE_TARGET_SAMPLER'});
  const ArrayType = type === T.FloatType ? Float32Array : Uint8Array;
  const pixels = new ArrayType([13, 101, 197, 255]); owner.writePixels(previous.entry, pixels, {destinationMipLevel: 2});
  owner.writePixels(current, new ArrayType([3, 4, 5, 255]), {destinationMipLevel: 2});
  const destination = new T.RenderTarget(1, 1, {type, depthBuffer: false});
  const renderer = {_targets: owner, _device: f.device, _shaderEncodedOutput: false};
  const request = captureTargetColorCopy(renderer, previous.texture, destination.texture, null, null, 2, 0);
  pass.toggleTexture('output'); owner.registerNativePass(pass);
  assert.equal(request.issue(), true);
  assert.deepEqual(await owner.ensure(owner.capture(destination)).storage.readPixels(), pixels);
  assert.equal(f.calls.samplers.length, 0); assert.equal(f.calls.passes.length, 0);
  request.release(); await owner.whenIdle(); owner.dispose(); f.dispose();
  destroyed(f.calls.textures);
});
