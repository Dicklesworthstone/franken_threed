import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

function fixture() {
  const f = gpuPostprocessFixture(), descriptors = [], create = f.device.createTexture;
  // The existing postprocess recorder assumes single-sample textures. Record
  // and retain the native descriptor here to exercise this owner's MSAA path.
  f.device.createTexture = d => {
    descriptors.push(d);
    const t = create(d); t.sampleCount = d.sampleCount; return t;
  };
  let fences = 0;
  const complete = f.device.queue.onSubmittedWorkDone;
  f.device.queue.onSubmittedWorkDone = () => { fences++; return complete(); };
  return {...f, descriptors, get fences() { return fences; }};
}
const make = (f, options = {}) => createGpuRenderTarget(f.device, {width: 16, height: 8, ...options});
const gone = textures => textures.forEach(t => assert.equal(t.destroyed, 1));

for (const sampleCount of [1, 4]) for (const format of ['rgba16float', 'rgba8unorm-srgb'])
  test(`${format}, ${sampleCount} samples: resolved sampling/readback and matching attachments`, async () => {
    const f = fixture(), t = make(f, {format, sampleCount});
    assert.deepEqual(t.rendererOptions, {format, sampleCount, depthFormat: 'depth24plus'});
    let color;
    assert.equal(t.withFrame((frame, texture) => {
      color = texture;
      assert.equal(texture.sampleCount, 1);
      assert.equal(texture.usage, 1 | 4 | 16);
      assert.equal(frame.colorView.texture.sampleCount, sampleCount);
      assert.equal(frame.depthView.texture.sampleCount, sampleCount);
      assert.equal(frame.colorView.texture.format, format);
      if (sampleCount === 4) assert.equal(frame.resolveTarget.texture, color);
      else assert.equal(frame.colorView.texture, color);
      assert.equal(Object.isFrozen(frame), true);
    }), true);
    t.withTexture(texture => { assert.equal(texture, color); });
    assert.equal(f.fences, 0);
    const bpp = format === 'rgba16float' ? 8 : 4;
    assert.equal(t.allocatedBytes, 16 * 8 * (bpp * (sampleCount === 4 ? 5 : 1) + 4 * sampleCount));
    await t.whenIdle();
    assert.equal(f.calls.scopes.length, 0);
    t.dispose(); gone(f.calls.textures);
    assert.equal(f.calls.deviceDestroyed, 0);
  });

test('depthless target allocates only its color; stable-size resizing changes nothing', () => {
  const f = fixture(), t = make(f, {depthFormat: null});
  assert.equal(f.calls.textures.length, 1);
  t.withFrame(frame => { assert.deepEqual(Object.keys(frame), ['colorView']); });
  assert.equal(t.resize(16, 8), t);
  assert.equal(f.calls.textures.length, 1);
  assert.equal(t.version, 0);
  t.dispose();
});

test('used targets survive replacement until the fence covering their last use', async () => {
  const f = fixture(), t = make(f, {sampleCount: 4}), first = f.calls.textures.slice();
  t.withTexture(() => {});
  const fence = deferred(); f.controls.completion = fence;
  t.resize(32, 16);
  assert.equal(t.version, 1);
  assert.equal(t.diagnostics.retiredTargets, 1);
  assert.equal(f.fences, 1);
  first.forEach(v => assert.equal(v.destroyed, 0));
  t.withFrame((frame, texture) => { assert.equal(texture.width, 32); });
  const before = t.allocatedBytes;
  fence.resolve(); await t.whenIdle();
  gone(first);
  assert.ok(t.allocatedBytes < before);
  assert.equal(t.diagnostics.retiredTargets, 0);
  t.dispose(); gone(f.calls.textures);
});

test('never-used storage can be retired immediately without a queue fence', () => {
  const f = fixture(), t = make(f), first = f.calls.textures.slice();
  t.resize(8, 8); gone(first);
  assert.equal(f.fences, 0);
  assert.equal(t.diagnostics.retiredTargets, 0);
  t.dispose();
});

test('peak budget rejection is allocation-free and retryable after retirement', async () => {
  const f = fixture(), t = make(f, {depthFormat: null, maxBytes: 2048});
  const fence = deferred(); f.controls.completion = fence;
  t.withFrame(() => {}); t.resize(8, 8);
  assert.equal(t.allocatedBytes, 1536);
  const count = f.calls.textures.length;
  assert.throws(() => t.resize(16, 8), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(t.failed, false); assert.equal(t.width, 8);
  assert.equal(f.calls.textures.length, count);
  fence.resolve(); await t.whenIdle(); t.resize(16, 8);
  assert.equal(t.width, 16); t.dispose();
});

test('zero extents suspend allocation and all callbacks; resume creates fresh storage', async () => {
  const f = fixture(), t = make(f, {width: 0});
  assert.equal(f.calls.textures.length, 0);
  assert.equal(t.suspended, true);
  const bad = () => assert.fail('suspended callback');
  assert.equal(t.withFrame(bad), false); assert.equal(t.withTexture(bad), false);
  t.resize(16, 8); t.withFrame(() => {});
  t.resize(16, 0); await t.whenIdle();
  assert.equal(t.allocatedBytes, 0);
  t.resize(16, 8); assert.equal(t.suspended, false);
  t.dispose(); gone(f.calls.textures);
});

for (const failure of [1, 2, 3]) test(`partial MSAA allocation failure ${failure} releases all ownership`, () => {
  const f = fixture(); f.controls.textureFailure = failure;
  assert.throws(() => make(f, {sampleCount: 4}), /injected texture/);
  gone(f.calls.textures); assert.equal(f.calls.scopes.length, 0);
});

test('replacement allocation failure retires current storage and is terminal', async () => {
  const f = fixture(), t = make(f);
  f.controls.textureFailure = 4;
  assert.throws(() => t.resize(8, 8), /injected texture/);
  assert.equal(t.failed, true); gone(f.calls.textures);
  await assert.rejects(t.whenIdle(), /injected texture/);
});

test('native validation errors are not reported as successful targets', async () => {
  const f = fixture(); f.controls.scopeError = {message: 'unsupported attachment'};
  const t = make(f);
  await assert.rejects(t.whenIdle(), {code: 'GPU_TARGET_GPU'});
  assert.equal(t.failed, true); assert.equal(t.allocatedBytes, 0);
  gone(f.calls.textures);
});

test('device loss ends stalled waits and destroys current plus retired textures', async () => {
  const f = fixture(), t = make(f);
  f.controls.completion = deferred(); t.withFrame(() => {}); t.resize(8, 8);
  const idle = t.whenIdle(); f.lost.resolve({message: 'lost'});
  await assert.rejects(idle, {code: 'GPU_TARGET_DEVICE_LOST'});
  gone(f.calls.textures);
  f.controls.completion.resolve(); await Promise.resolve();
  gone(f.calls.textures);
});

test('disposal interrupts idle and is idempotent without destroying a borrowed device', async () => {
  const f = fixture(), t = make(f); f.controls.completion = deferred();
  const idle = t.whenIdle(); t.dispose(); t.dispose();
  await assert.rejects(idle, {code: 'GPU_TARGET_DISPOSED'});
  assert.throws(() => t.resize(1, 1), {code: 'GPU_TARGET_DISPOSED'});
  assert.throws(() => t.withTexture(() => {}), {code: 'GPU_TARGET_DISPOSED'});
  gone(f.calls.textures); assert.equal(f.calls.deviceDestroyed, 0);
});

test('callback failures cannot discard already-submitted dependencies', async () => {
  const f = fixture(), t = make(f), textures = f.calls.textures.slice();
  assert.throws(() => t.withFrame(() => { throw Error('caller'); }), /caller/);
  assert.equal(t.failed, false);
  const fence = deferred(); f.controls.completion = fence;
  t.resize(8, 8); assert.equal(t.diagnostics.retiredTargets, 1);
  textures.forEach(v => assert.equal(v.destroyed, 0));
  fence.resolve(); await t.whenIdle(); gone(textures); t.dispose();
});

test('borrowed frame/texture operations reject reentry and asynchronous escape', async () => {
  const f = fixture(), t = make(f);
  for (const borrow of ['withFrame', 'withTexture']) {
    for (const op of [() => t.dispose(), () => t.resize(8, 8), () => t.withTexture(() => {})])
      assert.throws(() => t[borrow](op), {code: 'GPU_TARGET_REENTRANT'});
    assert.throws(() => t[borrow](() => Promise.resolve()), {code: 'GPU_TARGET_ASYNC_USE'});
    assert.throws(() => t[borrow](() => Promise.reject(Error('invalid async callback'))), {code: 'GPU_TARGET_ASYNC_USE'});
    assert.throws(() => t[borrow](null), {code: 'GPU_TARGET_CALLBACK'});
  }
  await t.whenIdle(); t.dispose();
});

test('dimensions, format, capacity and options are checked before any native allocation', () => {
  const f = fixture();
  for (const options of [{width: -1}, {height: 8193}, {width: 1.5}, {width: NaN}, {width: Infinity},
    {format: 'bad'}, {depthFormat: 'depth24plus-stencil8'}, {sampleCount: 2}, {maxBytes: 1},
    {maxBytes: Infinity}, {label: 4}, {unknown: true}]) assert.throws(() => make(f, options));
  assert.equal(f.calls.textures.length, 0);
});

test('a rejected retirement fence is terminal and does not leak either generation', async () => {
  const f = fixture(), t = make(f), fence = deferred();
  f.controls.completion = fence; t.withFrame(() => {}); t.resize(8, 8);
  const idle = t.whenIdle(); fence.reject(Error('queue failure'));
  await assert.rejects(idle, /queue failure/); gone(f.calls.textures);
});
