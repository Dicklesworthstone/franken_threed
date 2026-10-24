import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {transferDevice, deferred} from './fixtures/target_transfer_fixture.mjs';

const formats = ['rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb',
  'rgba16float', 'r32float', 'rg32float', 'rgba32float'];
const flush = () => new Promise(resolve => setImmediate(resolve));
function fixture(options = {}) {
  const f = transferDevice();
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 128});
  const target = createGpuRenderTarget(f.device, {width: 4, height: 2, format: 'rgba8unorm',
    depthFormat: null, copyDestination: true, ...options});
  return {...f, target};
}
function external(f, format = 'rgba8unorm') {
  const texture = f.device.createTexture({size: {width: 4, height: 2, depthOrArrayLayers: 1},
    dimension: '2d', format, sampleCount: 1, mipLevelCount: 1, usage: 1 | 8});
  for (let i = 0; i < texture.levels[0].bytes.length; i++) texture.levels[0].bytes[i] = (i * 73 + 11) & 255;
  return texture;
}
function copy(f, source, target = f.target, options = {}) {
  return target.withColorDestination(texture => {
    const encoder = f.device.createCommandEncoder();
    encoder.copyTextureToTexture({texture: source, mipLevel: 0, origin: {x: 0, y: 0, z: 0}},
      {texture, mipLevel: 0, origin: {x: 0, y: 0, z: 0}}, {width: 4, height: 2, depthOrArrayLayers: 1});
    f.device.queue.submit([encoder.finish()]);
  }, options);
}

// The recorder copies independent bytes, not shader output. In particular these
// cases certify no float conversion; they do not certify WGSL/browser parity.
for (const format of formats) for (const colorIndex of [0, 1, 2])
  test(`${format} slot ${colorIndex}: external GPU copy writes only the selected native color`, async () => {
    const colorFormats = ['rgba8unorm', 'rgba8unorm', 'rgba8unorm']; colorFormats[colorIndex] = format;
    const f = fixture({colorFormats, format: colorFormats[0], depthFormat: 'depth24plus-stencil8'});
    const source = external(f, format), bytes = f.target.allocatedBytes, textures = f.calls.textures.length;
    assert.equal(copy(f, source, f.target, {colorIndex}), true);
    f.target.withAttachments(colors => {
      for (let i = 0; i < colors.length; i++) assert.deepEqual(colors[i].levels[0].bytes,
        i === colorIndex ? source.levels[0].bytes : new Uint8Array(colors[i].levels[0].bytes.length));
    });
    assert.equal(f.calls.colorCopies.length, 1); assert.equal(f.calls.buffers.length, 0);
    assert.equal(f.calls.uploads.length, 0); assert.equal(f.calls.pipelines.length, 0);
    assert.equal(f.calls.textures.length, textures); assert.equal(f.target.allocatedBytes, bytes);
    await f.target.whenIdle(); f.target.dispose();
    assert.equal(source.destroyed, 0, 'borrowed source is not destination-owned');
    assert.ok(f.calls.textures.filter(t => t !== source).every(t => t.destroyed === 1));
    source.destroy(); assert.equal(f.calls.destroyed, 0);
  });

test('destination borrow neither samples nor generates stale mips before the write', async () => {
  const f = fixture({generateMipmaps: true}), source = external(f);
  copy(f, source); assert.equal(f.target.diagnostics.mipGenerations, 0);
  assert.equal(f.calls.submissions.length, 1);
  f.target.withTexture(() => {}); assert.equal(f.target.diagnostics.mipGenerations, 1);
  f.target.withTexture(() => {}); assert.equal(f.target.diagnostics.mipGenerations, 1);
  copy(f, source); assert.equal(f.target.diagnostics.mipGenerations, 1);
  f.target.withTexture(() => {}); assert.equal(f.target.diagnostics.mipGenerations, 2);
  await f.target.whenIdle(); f.target.dispose(); source.destroy();
});

test('copy capability, samples, attachment indices and callbacks reject before use', () => {
  const f = fixture(), disabled = createGpuRenderTarget(f.device),
    msaa = createGpuRenderTarget(f.device, {sampleCount: 4, copyDestination: true});
  const never = () => assert.fail('inadmissible destination callback');
  assert.throws(() => disabled.withColorDestination(never), {code: 'GPU_TARGET_USAGE'});
  assert.throws(() => msaa.withColorDestination(never), {code: 'GPU_TARGET_SAMPLES'});
  for (const colorIndex of [-1, 1, 0.5, NaN, Infinity, '0', null])
    assert.throws(() => f.target.withColorDestination(never, {colorIndex}), {code: 'GPU_TARGET_ATTACHMENT'});
  assert.throws(() => f.target.withColorDestination(null), {code: 'GPU_TARGET_CALLBACK'});
  assert.equal(f.calls.submissions.length, 0); assert.equal(f.target.failed, false);
  f.target.dispose(); disabled.dispose(); msaa.dispose();
});

test('suspended targets do not invoke writers, then resume with new storage', () => {
  const f = fixture({width: 0});
  assert.equal(f.target.withColorDestination(() => assert.fail('suspended')), false);
  assert.equal(f.target.allocatedBytes, 0);
  f.target.resize(4, 2); const source = external(f); copy(f, source);
  f.target.dispose(); source.destroy();
});

test('destination borrow protects native lifetime against reentry', async () => {
  const f = fixture();
  f.target.withColorDestination(() => {
    for (const run of [() => f.target.dispose(), () => f.target.resize(2, 2),
      () => f.target.withTexture(() => {}), () => f.target.withColorDestination(() => {})])
      assert.throws(run, {code: 'GPU_TARGET_REENTRANT'});
  });
  for (const value of [Promise.resolve(), Promise.reject(Error('async escape'))])
    assert.throws(() => f.target.withColorDestination(() => value), {code: 'GPU_TARGET_ASYNC_USE'});
  await f.target.whenIdle(); assert.equal(f.calls.scopes.length, 0); f.target.dispose();
});

test('external copies retain replaced destination generations until the queue fence', async () => {
  const f = fixture(), source = external(f), first = f.calls.textures[0], fence = deferred();
  copy(f, source); f.controls.completion = fence.promise; f.target.resize(2, 2);
  assert.equal(first.destroyed, 0); assert.equal(f.target.diagnostics.retiredTargets, 1);
  fence.resolve(); await f.target.whenIdle(); assert.equal(first.destroyed, 1);
  f.target.dispose(); source.destroy();
});

for (const kind of ['validation', 'throw']) test(`${kind}: failed destination work is not published as success`, async () => {
  const f = fixture(); await f.target.whenIdle();
  const failure = Error('external copy failed');
  if (kind === 'validation') f.target.withColorDestination(() => { f.controls.error = failure; });
  else assert.throws(() => f.target.withColorDestination(() => { throw failure; }), /external copy failed/);
  await assert.rejects(f.target.whenIdle(), /external copy failed/);
  assert.equal(f.target.failed, true); assert.equal(f.target.allocatedBytes, 0);
  assert.equal(f.calls.scopes.length, 0); f.target.dispose();
  assert.ok(f.calls.textures.every(t => t.destroyed === 1));
});

for (const fail of [false, true]) test(`source validation ${fail ? 'failure' : 'success'} gates readback commit, not GPU submission`, async () => {
  const f = fixture(), source = external(f), validation = deferred();
  copy(f, source, f.target, {validation: validation.promise});
  const output = new Uint8Array(32).fill(197), before = output.slice();
  let settled = false;
  const read = f.target.readPixels({output}); read.then(() => { settled = true; }, () => { settled = true; });
  const snapshot = f.target.validation; snapshot.catch(() => {});
  await flush();
  assert.equal(f.calls.submissions.length, 2, 'read copy submits before validation resolves');
  assert.equal(settled, false); assert.deepEqual(output, before);
  if (fail) {
    validation.reject(Error('producer validation failed'));
    await assert.rejects(read, /producer validation failed/);
    await assert.rejects(f.target.whenIdle(), /producer validation failed/);
    assert.deepEqual(output, before);
  } else {
    validation.resolve(); assert.equal(await read, output);
    assert.deepEqual(output, source.levels[0].bytes); await snapshot;
  }
  assert.ok(f.calls.buffers.every(b => b.destroyed === 1));
  f.target.dispose(); source.destroy();
});

test('producer validation propagates through another native target copy', async () => {
  const f = fixture(), source = external(f), validation = deferred();
  const next = createGpuRenderTarget(f.device, {width: 4, height: 2, format: 'rgba8unorm', depthFormat: null, copyDestination: true});
  copy(f, source, f.target, {validation: validation.promise}); next.copyFrom(f.target);
  const output = new Uint8Array(32).fill(199), read = next.readPixels({output}); read.catch(() => {});
  await flush(); assert.ok(output.every(n => n === 199));
  validation.reject(Error('bad producer'));
  await assert.rejects(read, /bad producer/); await assert.rejects(next.whenIdle(), /bad producer/);
  assert.ok(output.every(n => n === 199)); f.target.dispose(); next.dispose(); source.destroy();
});

for (const reason of ['dispose', 'device loss', 'abort']) test(`${reason} interrupts reads waiting for a producer`, async () => {
  const f = fixture(), validation = deferred(), controller = new AbortController();
  f.target.withColorDestination(() => {}, {validation: validation.promise});
  const output = new Uint8Array(32).fill(111), read = f.target.readPixels({output, signal: controller.signal});
  read.catch(() => {}); await flush();
  if (reason === 'dispose') f.target.dispose();
  else if (reason === 'device loss') f.lose({message: 'device loss'});
  else controller.abort(Error('abort'));
  await assert.rejects(read); assert.ok(output.every(n => n === 111));
  assert.ok(f.calls.buffers.every(b => b.destroyed === 1)); f.target.dispose(); validation.resolve();
});
