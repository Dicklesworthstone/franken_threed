import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget, inspectGpuRenderTargetColors} from './gpu_render_target.mjs';
import {transferDevice, deferred} from './fixtures/target_transfer_fixture.mjs';

// Byte movement and descriptor recording; this fixture does not execute WGSL.
const formats = [['r8unorm', Uint8Array], ['rg11b10ufloat', Uint32Array]];
function fixture() {
  const f = transferDevice();
  f.device.features.add('rg11b10ufloat-renderable');
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  return f;
}
const make = (f, options = {}) => createGpuRenderTarget(f.device,
  {width: 5, height: 4, depthFormat: null, copyDestination: true, ...options});

for (const [format, Type] of formats) for (const flipY of [false, true])
  test(`${format}: cropped transfers preserve raw texels and padded rows, flipY=${flipY}`, async () => {
    const f = fixture(), source = make(f, {format}), destination = make(f, {format});
    const input = new Type(17);
    // Packed values include special exponent and mantissa patterns. These are
    // storage words, not JS numbers to be unpacked, normalized or requantized.
    input.set(Type === Uint32Array ? [0, 0xffffffff, 0x7c0f87ff, 0x00000001, 0x80000000, 0,
      0xffc007c0, 0x12345678, 0xabcdef01] : [0, 1, 255, 127, 96, 0, 17, 33, 211]);
    const expected = new Type([input[1], input[2], input[3], input[6], input[7], input[8]]);
    source.writePixels(input, {x: 1, y: 1, width: 3, height: 2,
      offset: Type.BYTES_PER_ELEMENT, bytesPerRow: 5 * Type.BYTES_PER_ELEMENT, flipY});
    input.fill(0);
    destination.copyFrom(source, {sourceX: 1, sourceY: 1, x: 0, y: 1, width: 3, height: 2, flipY});
    assert.equal(f.calls.buffers.length, 0, 'GPU copies do not stage through CPU memory');
    const backing = new Type(10).fill(79), output = backing.subarray(1, 8);
    assert.equal(await destination.readPixels({x: 0, y: 1, width: 3, height: 2, flipY, output}), output);
    assert.deepEqual(output.subarray(0, 6), expected);
    assert.equal(backing[0], 79); assert.ok(backing.subarray(7).every(value => value === 79));
    assert.ok((await destination.readPixels({x: 4, width: 1})).every(value => value === 0));
    assert.equal(source.allocatedBytes, 20 * Type.BYTES_PER_ELEMENT);
    assert.equal(destination.diagnostics.colorCopies, 1);
    await destination.whenIdle(); source.dispose(); destination.dispose();
    assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
  });

test('packed target feature admission precedes native allocation and applies to secondary attachments', () => {
  const f = fixture(); f.device.features.clear();
  for (const colorFormats of [['rg11b10ufloat'], ['r8unorm', 'rg11b10ufloat']]) {
    assert.throws(() => make(f, {colorFormats}), {code: 'GPU_TARGET_FEATURE'});
    assert.throws(() => inspectGpuRenderTargetColors(f.device, colorFormats), {code: 'GPU_TARGET_FEATURE'});
  }
  assert.equal(f.calls.textures.length, 0);
  const r8 = make(f, {format: 'r8unorm'}); r8.dispose();
});

test('mixed AO/GI targets distinguish storage footprint from aligned render attachment cost', async () => {
  const f = fixture(), colorFormats = ['r8unorm', 'rg11b10ufloat'];
  f.device.limits.maxColorAttachmentBytesPerSample = 11;
  assert.throws(() => make(f, {colorFormats}), {code: 'GPU_TARGET_LIMIT'});
  assert.equal(f.calls.textures.length, 0);
  f.device.limits.maxColorAttachmentBytesPerSample = 12;
  assert.throws(() => make(f, {colorFormats, maxBytes: 99}), {code: 'GPU_TARGET_BUDGET'});
  const target = make(f, {colorFormats, maxBytes: 100});
  assert.equal(target.allocatedBytes, 100);
  target.withFrame(frame => assert.deepEqual(frame.colorViews.map(view => view.texture.format), colorFormats));
  const ao = Uint8Array.from({length: 20}, (_, index) => index * 11);
  const gi = Uint32Array.from({length: 20}, (_, index) => (0xf0000001 + index * 157) >>> 0);
  target.writePixels(ao); target.writePixels(gi, {colorIndex: 1});
  assert.deepEqual(await target.readPixels(), ao);
  assert.deepEqual(await target.readPixels({colorIndex: 1}), gi);
  await target.whenIdle(); target.dispose();
  // Attachment order changes the alignment: packed GI then scalar AO costs9.
  f.device.limits.maxColorAttachmentBytesPerSample = 9;
  assert.deepEqual(inspectGpuRenderTargetColors(f.device, colorFormats.slice().reverse()), colorFormats.slice().reverse());
});

for (const [format, Type] of formats) test(`${format}: four-sample attachments resolve without changing format`, async () => {
  const f = fixture(), target = make(f, {format, sampleCount: 4});
  assert.equal(target.allocatedBytes, 20 * Type.BYTES_PER_ELEMENT * 5);
  target.withFrame(frame => {
    assert.equal(frame.colorView.texture.sampleCount, 4);
    assert.equal(frame.resolveTarget.texture.sampleCount, 1);
    assert.equal(frame.colorView.texture.format, format);
    assert.equal(frame.resolveTarget.texture.format, format);
  });
  assert.throws(() => target.writePixels(new Type(20)), {code: 'GPU_TRANSFER_SAMPLES'});
  assert.ok(await target.readPixels() instanceof Type);
  await target.whenIdle(); target.dispose();
});

for (const [format, Type] of formats) test(`${format}: native mip generation retains compact storage and format`, async () => {
  const f = fixture(), target = make(f, {format, generateMipmaps: true});
  assert.equal(target.mipLevelCount, 3);
  assert.equal(target.allocatedBytes, (20 + 4 + 1) * Type.BYTES_PER_ELEMENT);
  target.writePixels(new Type(20)); target.withTexture(() => {});
  assert.equal(target.diagnostics.mipGenerations, 1);
  assert.equal(f.calls.pipelines[0].fragment.targets[0].format, format);
  assert.match(f.calls.modules[0].code, /textureSampleLevel/);
  assert.equal(f.calls.submissions.flatMap(submission => submission.passes).length, 2);
  await target.whenIdle(); target.dispose();
});

for (const [format, Type] of formats) test(`${format}: pending readback cancels without publishing into caller output`, async () => {
  const f = fixture(), gate = deferred(), target = make(f, {format});
  f.controls.mapping = gate.promise;
  const output = new Type(20).fill(39);
  const pending = target.readPixels({output});
  assert.equal(f.calls.buffers.length, 1);
  target.dispose(); await assert.rejects(pending);
  assert.ok(output.every(value => value === 39));
  assert.equal(f.calls.buffers[0].destroyed, 1);
  assert.equal(target.diagnostics.pendingReadbacks, 0);
  gate.resolve();
});
