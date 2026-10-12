import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

const make = (f, options = {}) => createGpuRenderTarget(f.device, {width: 8, height: 4, sampleDepth: true, ...options});
const released = f => { for (const texture of f.textures) assert.equal(texture.destroyed, 1); };

for (const depthFormat of ['depth16unorm', 'depth24plus', 'depth32float']) for (const sampleCount of [1, 4])
  test(`sample the rendered ${depthFormat} attachment with ${sampleCount} sample(s), not a replacement`, async () => {
    const f = gpuTargetMipmapFixture(), target = make(f, {depthFormat, sampleCount});
    let written;
    target.withFrame(frame => { written = frame.depthView.texture; });
    assert.equal(target.withDepthTexture(depth => {
      assert.equal(depth, written);
      assert.equal(depth.format, depthFormat);
      assert.equal(depth.sampleCount, sampleCount);
      assert.equal(depth.mipLevelCount, 1);
      assert.equal(depth.usage, 4 | 16);
      assert.equal(depth.createView({aspect: 'depth-only'}).texture, written);
    }), true);
    target.withTexture(color => { assert.notEqual(color, written); assert.equal(color.sampleCount, 1); });
    assert.equal(f.submissions.length, 0, 'depth borrowing has no implicit resolve or copy');
    assert.equal(f.pipelines.length, 0);
    assert.equal(f.textures.length, sampleCount === 4 ? 3 : 2);
    assert.equal(target.diagnostics.sampleDepth, true);
    assert.deepEqual(target.rendererOptions, {format: 'rgba16float', depthFormat, sampleCount});
    assert.equal(target.allocatedBytes, 32 * (8 * (sampleCount === 4 ? 5 : 1) + (depthFormat === 'depth16unorm' ? 2 : 4) * sampleCount));
    await target.whenIdle(); target.dispose(); released(f);
  });

test('default render-only depth and depthless owners do not acquire sampled-depth usage', () => {
  for (const depthFormat of [null, 'depth24plus']) {
    const f = gpuTargetMipmapFixture(), target = make(f, {depthFormat, sampleDepth: false});
    target.withFrame(frame => {
      if (depthFormat) assert.equal(frame.depthView.texture.usage, 16);
      else assert.equal(frame.depthView, undefined);
    });
    assert.throws(() => target.withDepthTexture(() => assert.fail('no sampled depth')), {code: 'GPU_TARGET_DEPTH'});
    assert.equal(target.failed, false);
    assert.equal(f.textures.length, depthFormat ? 2 : 1);
    target.dispose(); released(f);
  }
});

test('depth borrows neither generate nor acknowledge pending color mip updates', () => {
  const f = gpuTargetMipmapFixture(), target = make(f, {generateMipmaps: true});
  target.withFrame(() => {});
  target.withDepthTexture(() => {});
  assert.equal(f.submissions.length, 0);
  assert.equal(target.diagnostics.mipGenerations, 0);
  target.withTexture(() => {});
  assert.equal(target.diagnostics.mipGenerations, 1);
  target.withFrame(() => {});
  target.withDepthTexture(() => {}); target.withDepthTexture(() => {});
  assert.equal(target.diagnostics.mipGenerations, 1);
  target.withTexture(() => {});
  assert.equal(target.diagnostics.mipGenerations, 2);
  target.dispose(); released(f);
});

test('depth-only consumers retain an old generation through its submission fence', async () => {
  const f = gpuTargetMipmapFixture(), target = make(f), first = f.textures.slice();
  let sampled;
  target.withDepthTexture(depth => { sampled = depth; });
  const fence = deferred(); f.controls.completion = fence;
  const bytes = target.allocatedBytes;
  target.resize(4, 2);
  assert.equal(target.diagnostics.retiredTargets, 1);
  assert.equal(target.allocatedBytes, bytes + 8 * 12);
  assert.ok(first.every(t => t.destroyed === 0));
  target.withDepthTexture(depth => { assert.notEqual(depth, sampled); assert.equal(depth.width, 4); });
  fence.resolve(); await target.whenIdle();
  assert.ok(first.every(t => t.destroyed === 1));
  assert.equal(target.allocatedBytes, 8 * 12);
  target.dispose(); released(f);
});

test('a throwing depth consumer still protects possibly submitted work', async () => {
  const f = gpuTargetMipmapFixture(), target = make(f), first = f.textures.slice();
  assert.throws(() => target.withDepthTexture(() => { throw Error('consumer'); }), /consumer/);
  assert.equal(target.failed, false);
  f.controls.completion = deferred(); target.resize(4, 2);
  assert.ok(first.every(t => t.destroyed === 0));
  f.controls.completion.resolve(); await target.whenIdle();
  target.dispose(); released(f);
});

test('depth generation peak budget includes retained storage and can be retried', async () => {
  const f = gpuTargetMipmapFixture(), target = make(f, {maxBytes: 480});
  target.withDepthTexture(() => {}); f.controls.completion = deferred();
  target.resize(4, 2);
  const allocated = f.textures.length;
  assert.throws(() => target.resize(8, 4), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.textures.length, allocated);
  assert.equal(target.failed, false);
  f.controls.completion.resolve(); await target.whenIdle(); target.resize(8, 4);
  target.dispose(); released(f);
});

test('zero dimensions suspend depth borrowers; resuming supplies the new attachment', async () => {
  const f = gpuTargetMipmapFixture(), target = make(f, {height: 0});
  assert.equal(target.withDepthTexture(() => assert.fail('suspended')), false);
  assert.equal(f.textures.length, 0);
  target.resize(8, 4); target.withDepthTexture(depth => assert.equal(depth.height, 4));
  target.resize(0, 4); await target.whenIdle();
  assert.equal(target.withDepthTexture(() => assert.fail('suspended')), false);
  assert.equal(target.allocatedBytes, 0);
  target.dispose(); released(f);
});

test('every borrowed handle rejects reentry through the depth API', async () => {
  const f = gpuTargetMipmapFixture(), target = make(f);
  for (const borrow of ['withFrame', 'withTexture', 'withDepthTexture']) {
    assert.throws(() => target[borrow](() => target.withDepthTexture(() => {})), {code: 'GPU_TARGET_REENTRANT'});
    assert.throws(() => target.withDepthTexture(() => target[borrow](() => {})), {code: 'GPU_TARGET_REENTRANT'});
  }
  for (const operation of [() => target.resize(2, 2), () => target.dispose()])
    assert.throws(() => target.withDepthTexture(operation), {code: 'GPU_TARGET_REENTRANT'});
  assert.throws(() => target.withDepthTexture(null), {code: 'GPU_TARGET_CALLBACK'});
  assert.throws(() => target.withDepthTexture(() => Promise.resolve()), {code: 'GPU_TARGET_ASYNC_USE'});
  assert.throws(() => target.withDepthTexture(() => Promise.reject(Error('late'))), {code: 'GPU_TARGET_ASYNC_USE'});
  await target.whenIdle(); target.dispose(); released(f);
});

test('sampled-depth option errors are allocation-free, including suspended targets', () => {
  const f = gpuTargetMipmapFixture();
  for (const options of [{sampleDepth: 1}, {sampleDepth: null}, {depthFormat: null}, {height: 0, depthFormat: null}, {depthFormat: 'depth24plus-stencil8'}])
    assert.throws(() => make(f, options), /GPU_TARGET_/);
  assert.equal(f.textures.length, 0);
});

test('disposal and device loss end waits and prohibit future depth borrows', async () => {
  for (const reason of ['dispose', 'loss']) {
    const f = gpuTargetMipmapFixture(), target = make(f);
    target.withDepthTexture(() => {}); f.controls.completion = deferred(); target.resize(4, 2);
    const idle = target.whenIdle();
    if (reason === 'dispose') target.dispose(); else f.lost.resolve({message: 'device lost'});
    const code = reason === 'dispose' ? 'GPU_TARGET_DISPOSED' : 'GPU_TARGET_DEVICE_LOST';
    await assert.rejects(idle, {code});
    assert.throws(() => target.withDepthTexture(() => {}), {code});
    released(f); f.controls.completion.resolve(); await Promise.resolve(); released(f);
    target.dispose();
  }
});

test('native validation errors cannot become apparently usable depth textures', async () => {
  const f = gpuTargetMipmapFixture(); f.controls.validation = {message: 'invalid sampled depth'};
  const target = make(f);
  await assert.rejects(target.whenIdle(), {code: 'GPU_TARGET_GPU'});
  assert.throws(() => target.withDepthTexture(() => {}), {code: 'GPU_TARGET_GPU'});
  assert.equal(target.allocatedBytes, 0); target.dispose(); released(f);
});

test('partial sampled-depth allocation failure releases already-created color and MSAA', () => {
  const f = gpuTargetMipmapFixture(), create = f.device.createTexture;
  f.device.createTexture = descriptor => {
    if (descriptor.format.startsWith('depth')) throw Error('depth allocation');
    return create(descriptor);
  };
  assert.throws(() => make(f, {sampleCount: 4}), /depth allocation/);
  assert.equal(f.scopes.length, 0); released(f);
});
