import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';
import {inspectGpuTargetWrite, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {mrtTransferDevice, layouts, bytes, deferred} from './fixtures/mrt_transfer_fixture.mjs';
const make = (f, colorFormats = ['rgba8unorm', 'rgba16float', 'r32float'], options = {}) =>
  createGpuRenderTarget(f.device, {width: 4, height: 3, colorFormats, copyDestination: true, ...options});
function payload(format, pixels = 12, seed = 7) {
  const [Type, components] = layouts.get(format), data = new Type(pixels * components);
  bytes(data).set(Uint8Array.from({length: data.byteLength}, (_, i) => (i * 71 + seed) % 256));
  return data;
}
const close = (f, ...targets) => {
  targets.forEach(target => target.dispose());
  assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
  assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
  assert.equal(f.calls.deviceDestroyed, 0);
};

for (const format of layouts.keys()) for (const colorIndex of [0, 1, 2])
  test(`${format}, attachment ${colorIndex}: typed upload/read preserves bits and other attachments`, async () => {
    const f = mrtTransferDevice(), formats = ['rgba8unorm', 'rgba16float', 'r32float']; formats[colorIndex] = format;
    const target = make(f, formats), input = payload(format), expected = bytes(input).slice();
    let colors, depth;
    target.withFrame(frame => { colors = frame.colorViews.map(view => view.texture); depth = frame.depthView.texture; });
    depth.planes[0][0].fill(173);
    assert.equal(target.writePixels(input, {colorIndex}), true);
    input.fill(0);
    const result = await target.readPixels({colorIndex});
    assert.equal(result.constructor, layouts.get(format)[0]); assert.deepEqual(bytes(result), expected);
    assert.equal(f.uploads[0].destination.texture, colors[colorIndex]);
    assert.equal(f.reads[0].source.texture, colors[colorIndex]);
    colors.forEach((texture, index) => { if (index !== colorIndex) assert.ok(texture.planes[0][0].every(value => value === 0)); });
    assert.ok(depth.planes[0][0].every(value => value === 173));
    await target.whenIdle(); close(f, target);
  });

for (const format of layouts.keys()) test(`${format}: cropped secondary upload/read honors stride, offset, flipY and output guards`, async () => {
  const f = mrtTransferDevice(), target = make(f, ['rgba8unorm', format]);
  const [Type, components] = layouts.get(format), input = payload(format, 4), rowBytes = input.byteLength / 2;
  const backing = new Uint8Array(rowBytes * 5).fill(213), view = new Type(backing.buffer, rowBytes);
  backing.set(bytes(input).subarray(0, rowBytes), rowBytes * 2);
  backing.set(bytes(input).subarray(rowBytes), rowBytes * 4);
  target.writePixels(view, {colorIndex: 1, x: 1, y: 0, width: 2, height: 2, offset: rowBytes, bytesPerRow: rowBytes * 2, flipY: true});
  const guarded = new Type(4 * components + 4).fill(11), output = guarded.subarray(2, 2 + 4 * components);
  assert.equal(await target.readPixels({colorIndex: 1, x: 1, y: 0, width: 2, height: 2, flipY: true, output}), output);
  assert.deepEqual(bytes(output), bytes(input)); assert.deepEqual([...guarded.subarray(0, 2)], [11, 11]);
  assert.deepEqual([...guarded.subarray(-2)], [11, 11]);
  assert.ok((await target.readPixels()).every(value => value === 0)); close(f, target);
});

test('cross-index GPU copies use selected formats, not incompatible primaries, without staging', async () => {
  const f = mrtTransferDevice(), source = make(f, ['r32float', 'rgba16float', 'rg32float']);
  const destination = make(f, ['rgba8unorm', 'rg32float', 'rgba16float']);
  for (const [sourceColorIndex, colorIndex, format] of [[1, 2, 'rgba16float'], [2, 1, 'rg32float']]) {
    const input = payload(format); source.writePixels(input, {colorIndex: sourceColorIndex});
    const buffers = f.calls.buffers.length;
    assert.equal(destination.copyFrom(source, {sourceColorIndex, colorIndex}), true);
    assert.equal(f.calls.buffers.length, buffers);
    assert.deepEqual(bytes(await destination.readPixels({colorIndex})), bytes(input));
  }
  assert.ok((await destination.readPixels()).every(value => value === 0));
  assert.throws(() => destination.copyFrom(source, {sourceColorIndex: 1, colorIndex: 1}), {code: 'GPU_TRANSFER_FORMAT'});
  await destination.whenIdle(); close(f, source, destination);
});

for (const format of ['rgba8unorm', 'bgra8unorm']) test(`${format}: secondary sRGB-compatible copies remain raw bits`, async () => {
  const f = mrtTransferDevice(), source = make(f, ['r32float', format]), destination = make(f, ['rgba16float', `${format}-srgb`]);
  const input = payload(format); source.writePixels(input, {colorIndex: 1});
  destination.copyFrom(source, {sourceColorIndex: 1, colorIndex: 1});
  assert.deepEqual(bytes(await destination.readPixels({colorIndex: 1})), bytes(input)); close(f, source, destination);
});

for (const direction of ['to-cube', 'from-cube']) test(`selected MRT attachment copies ${direction} with explicit face`, async () => {
  const f = mrtTransferDevice(), mrt = make(f, ['rgba16float', 'rgba8unorm'], {height: 4});
  const cube = make(f, ['rgba8unorm'], {height: 4, cube: true}), data = payload('rgba8unorm', 16);
  if (direction === 'to-cube') {
    mrt.writePixels(data, {colorIndex: 1}); cube.copyFrom(mrt, {sourceColorIndex: 1, face: 4});
    assert.deepEqual(await cube.readPixels({face: 4}), data);
    assert.ok((await cube.readPixels({face: 3})).every(value => value === 0));
  } else {
    cube.writePixels(data, {face: 3}); mrt.copyFrom(cube, {sourceFace: 3, colorIndex: 1});
    assert.deepEqual(await mrt.readPixels({colorIndex: 1}), data);
  }
  assert.ok((await mrt.readPixels()).every(value => value === 0)); close(f, mrt, cube);
});

test('selected source mip can populate secondary MRT storage', async () => {
  const f = mrtTransferDevice(), source = make(f, ['rgba16float'], {width: 8, height: 6, mipLevelCount: 2});
  const destination = make(f), data = payload('rgba16float');
  source.withFrame((frame, texture) => texture.planes[1][0].set(bytes(data)), {mipLevel: 1});
  destination.copyFrom(source, {sourceMipLevel: 1, colorIndex: 1});
  assert.equal(f.copies[0].source.mipLevel, 1);
  assert.deepEqual(bytes(await destination.readPixels({colorIndex: 1})), bytes(data)); close(f, source, destination);
});

test('MSAA reads and copies select the resolved color, never another attachment or its multisample image', async () => {
  const f = mrtTransferDevice(), source = make(f, ['rgba8unorm', 'rgba16float'], {sampleCount: 4});
  const destination = make(f), data = payload('rgba16float'); let resolved;
  source.withFrame(frame => {
    resolved = frame.resolveTargets[1].texture; resolved.planes[0][0].set(bytes(data));
    frame.colorViews[1].texture.planes[0][0].fill(227);
  });
  assert.deepEqual(bytes(await source.readPixels({colorIndex: 1})), bytes(data));
  destination.copyFrom(source, {sourceColorIndex: 1, colorIndex: 1});
  assert.equal(f.copies[0].source.texture, resolved); assert.equal(f.reads[0].source.texture, resolved);
  assert.deepEqual(bytes(await destination.readPixels({colorIndex: 1})), bytes(data));
  assert.throws(() => source.writePixels(data, {colorIndex: 1}), {code: 'GPU_TRANSFER_SAMPLES'});
  assert.throws(() => source.copyFrom(destination, {sourceColorIndex: 1, colorIndex: 1}), {code: 'GPU_TRANSFER_SAMPLES'});
  close(f, source, destination);
});

test('selection validation rejects before native calls, including empty and legacy single-color targets', async () => {
  const f = mrtTransferDevice(), target = make(f), other = make(f), single = make(f, ['rgba8unorm']);
  for (const colorIndex of [-1, 3, 1.5, NaN, Infinity, '1', null, true]) {
    assert.throws(() => target.writePixels(new Uint8Array(0), {colorIndex, width: 0}), {code: 'GPU_TRANSFER_ATTACHMENT'});
    assert.throws(() => target.copyFrom(other, {colorIndex, width: 0}), {code: 'GPU_TRANSFER_ATTACHMENT'});
    assert.throws(() => target.copyFrom(other, {sourceColorIndex: colorIndex, width: 0}), {code: 'GPU_TRANSFER_ATTACHMENT'});
    await assert.rejects(target.readPixels({colorIndex, width: 0}), {code: 'GPU_READBACK_ATTACHMENT'});
  }
  for (const key of ['colorIndex', 'sourceColorIndex']) {
    const value = Object.defineProperty({}, key, {get() { assert.fail('selector getter must not run'); }});
    assert.throws(() => target.copyFrom(other, value), {code: 'GPU_TRANSFER_OPTIONS'});
  }
  await assert.rejects(single.readPixels({colorIndex: 1}), {code: 'GPU_READBACK_ATTACHMENT'});
  assert.throws(() => single.writePixels(new Uint8Array(48), {colorIndex: 1}), {code: 'GPU_TRANSFER_ATTACHMENT'});
  assert.equal(f.uploads.length, 0); assert.equal(f.calls.submissions.length, 0); assert.equal(f.calls.buffers.length, 0);
  assert.equal(target.failed, false); close(f, target, other, single);
});

test('selected type, packed bytes and padded readback budgets are enforced before allocation', async () => {
  const f = mrtTransferDevice(), target = make(f, ['rgba8unorm', 'rgba32float'], {width: 17, height: 2});
  const data = payload('rgba32float', 34), shape = {...target.diagnostics, mipLevelCount: 1};
  assert.throws(() => target.writePixels(new Uint8Array(data.byteLength), {colorIndex: 1}), {code: 'GPU_TRANSFER_DATA'});
  assert.throws(() => target.writePixels(data, {colorIndex: 1, maxBytes: data.byteLength - 1}), {code: 'GPU_TRANSFER_BUDGET'});
  assert.throws(() => inspectGpuTargetCopy(shape, shape, {sourceColorIndex: 1, colorIndex: 1, maxBytes: data.byteLength - 1}), {code: 'GPU_TRANSFER_BUDGET'});
  await assert.rejects(target.readPixels({colorIndex: 1, maxBytes: 1023}), {code: 'GPU_READBACK_BUDGET'});
  await assert.rejects(target.readPixels({colorIndex: 1, output: new Uint8Array(1024)}), {code: 'GPU_READBACK_OUTPUT'});
  assert.equal(f.calls.buffers.length, 0); assert.equal(f.uploads.length, 0);
  const plan = inspectGpuTargetWrite(shape, data, {colorIndex: 1}); assert.equal(plan.packedBytes, data.byteLength);
  target.writePixels(data, {colorIndex: 1});
  assert.deepEqual(bytes(await target.readPixels({colorIndex: 1, maxBytes: 1024})), bytes(data)); close(f, target);
});

test('pending reads and copies preserve selected content and dimensions across edits and resize', async () => {
  const f = mrtTransferDevice(), source = make(f), destination = make(f), data = payload('rgba16float');
  f.controls.mapping = deferred(); f.controls.completion = deferred();
  source.writePixels(data, {colorIndex: 1});
  const options = {sourceColorIndex: 1, colorIndex: 1}; destination.copyFrom(source, options);
  const readOptions = {colorIndex: 1}, pending = destination.readPixels(readOptions);
  options.sourceColorIndex = options.colorIndex = readOptions.colorIndex = 2;
  source.writePixels(new Uint16Array(data.length), {colorIndex: 1});
  destination.writePixels(new Uint16Array(data.length), {colorIndex: 1});
  const old = f.calls.textures.slice(); source.resize(8, 6); destination.resize(8, 6);
  assert.ok(old.every(texture => texture.destroyed === 0));
  f.controls.mapping.resolve(); assert.deepEqual(bytes(await pending), bytes(data));
  f.controls.completion.resolve(); await Promise.all([source.whenIdle(), destination.whenIdle()]);
  assert.ok(old.every(texture => texture.destroyed === 1)); close(f, source, destination);
});

test('attachments share one in-flight readback budget and cancellation preserves output', async () => {
  const f = mrtTransferDevice(), target = make(f), signal = new AbortController(), output = new Uint16Array(48).fill(19);
  f.controls.mapping = deferred();
  const pending = target.readPixels({colorIndex: 1, output, signal: signal.signal, maxInFlightBytes: 768});
  const rejected = assert.rejects(pending, /cancel selected/);
  await assert.rejects(target.readPixels({colorIndex: 2, maxInFlightBytes: 100000}), {code: 'GPU_READBACK_BUDGET'});
  signal.abort(Error('cancel selected')); await rejected;
  assert.ok(output.every(value => value === 19)); assert.equal(f.calls.buffers[0].destroyed, 1);
  f.controls.mapping.resolve(); await target.readPixels({colorIndex: 2, maxInFlightBytes: 816}); close(f, target);
});

for (const stop of ['dispose', 'loss']) test(`${stop}: secondary reads reject and release staging once`, async () => {
  const f = mrtTransferDevice(), target = make(f), output = new Float32Array(12).fill(71);
  f.controls.mapping = deferred(); const pending = target.readPixels({colorIndex: 2, output});
  const rejected = assert.rejects(pending, {code: stop === 'dispose' ? 'GPU_TARGET_DISPOSED' : 'GPU_TARGET_DEVICE_LOST'});
  if (stop === 'dispose') target.dispose(); else f.lost.resolve({message: 'lost'});
  await rejected; assert.ok(output.every(value => value === 71)); f.controls.mapping.resolve();
  close(f, target);
});

test('validation failure in a selected upload propagates to its dependent copy', async () => {
  const f = mrtTransferDevice(), source = make(f), destination = make(f); await Promise.all([source.whenIdle(), destination.whenIdle()]);
  f.controls.scopeError = {message: 'secondary upload invalid'};
  source.writePixels(payload('rgba16float'), {colorIndex: 1}); destination.copyFrom(source, {sourceColorIndex: 1, colorIndex: 1});
  await assert.rejects(destination.whenIdle(), /secondary upload invalid/);
  assert.equal(source.failed, true); assert.equal(destination.failed, true); close(f, source, destination);
});

test('empty secondary reads retain their type and still validate selection without storage', async () => {
  const f = mrtTransferDevice(), target = make(f, undefined, {width: 0, height: 0});
  assert.ok(await target.readPixels({colorIndex: 1}) instanceof Uint16Array);
  assert.ok(await target.readPixels({colorIndex: 2}) instanceof Float32Array);
  assert.equal(target.writePixels(new Uint16Array(0), {colorIndex: 1}), false);
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.buffers.length, 0); close(f, target);
});

test('a session that ignores attachment selection fails instead of interpreting primary bytes as another format', async () => {
  const f = mrtTransferDevice(), target = make(f); let primary;
  target.withTexture(texture => { primary = texture; });
  await assert.rejects(readGpuTargetPixels({device: f.device, width: 4, height: 3, diagnostics: target.diagnostics,
    withTexture: run => run(primary)}, {colorIndex: 1}), {code: 'GPU_READBACK_SOURCE'});
  assert.equal(f.calls.buffers.length, 0); close(f, target);
});

test('attachment selection does not bypass device identity, reentry, or same-owner feedback guards', () => {
  const f = mrtTransferDevice(), target = make(f), source = make(f), foreignFixture = mrtTransferDevice(), foreign = make(foreignFixture);
  assert.throws(() => target.copyFrom(target, {sourceColorIndex: 1, colorIndex: 1}), {code: 'GPU_TARGET_FEEDBACK'});
  assert.throws(() => target.copyFrom(foreign, {sourceColorIndex: 1, colorIndex: 1}), {code: 'GPU_TARGET_DEVICE'});
  source.withTexture(() => assert.throws(() => target.copyFrom(source, {sourceColorIndex: 1, colorIndex: 1}), {code: 'GPU_TARGET_REENTRANT'}));
  assert.equal(f.copies.length, 0); close(f, target, source); close(foreignFixture, foreign);
});
