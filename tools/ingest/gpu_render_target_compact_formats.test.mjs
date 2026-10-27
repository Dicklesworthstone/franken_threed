import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget, inspectGpuRenderTargetColors} from './gpu_render_target.mjs';
import {transferDevice, deferred} from './fixtures/target_transfer_fixture.mjs';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {inspectThreeRenderTarget} from './three_render_targets.mjs';

// Byte movement and descriptor recording; this fixture does not execute WGSL.
const formats = [['r8unorm', Uint8Array], ['rg11b10ufloat', Uint32Array]];
const halfFormats = [['r16float', Uint16Array, 1], ['rg16float', Uint16Array, 2]];
const allFormats = [...formats, ...halfFormats];
function fixture() {
  const f = transferDevice();
  f.device.features.add('rg11b10ufloat-renderable');
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  return f;
}
const make = (f, options = {}) => createGpuRenderTarget(f.device,
  {width: 5, height: 4, depthFormat: null, copyDestination: true, ...options});

for (const [format, Type, components = 1] of allFormats) test(`${format}: public queued uploads, copies and readbacks preserve component words`, async t => {
  const f = fixture(), gate = deferred(), Base = createWebGPURendererClass(T);
  class Renderer extends Base {
    async _init() { await gate.promise; this._device = f.device; this._initialized = true; return this; }
  }
  const renderer = new Renderer({canvas: {width: 5, height: 4, style: {}}});
  const options = {depthBuffer: false, format: Type === Uint32Array ? T.RGBFormat : components === 2 ? T.RGFormat : T.RedFormat,
    type: Type === Uint32Array ? T.UnsignedInt101111Type : Type === Uint16Array ? T.HalfFloatType : T.UnsignedByteType};
  const source = new T.RenderTarget(5, 4, options), destination = new T.RenderTarget(5, 4, options);
  t.after(async () => {
    gate.resolve(); renderer.dispose();
    while (renderer._drain) await renderer._drain.catch(() => {});
    source.dispose(); destination.dispose();
  });
  const input = Type.from({length: 20 * components}, (_, i) => Type === Uint32Array
    ? [0, 0xffffffff, 0x7c0f87ff, 0x80000000, 0x12345678][i % 5]
    : Type === Uint16Array ? [0x7c01, 0xfc00, 0x8000, 0x0001, 0x3c00][i % 5] : i * 13);
  const expected = input.slice();
  const write = renderer.writeRenderTargetPixelsAsync(source, input);
  const copy = renderer.copyTextureToTextureAsync(source.texture, destination.texture);
  const read = renderer.readRenderTargetPixelsAsync(destination, 0, 0, 5, 4);
  input.fill(0); gate.resolve();
  await Promise.all([write, copy]);
  const actual = await read; assert.ok(actual instanceof Type); assert.deepEqual(actual, expected);
  assert.equal(f.calls.colorCopies[0].source.texture.format, format);
  const backing = new Type(expected.length + 3).fill(39), output = backing.subarray(1);
  assert.equal(await renderer.readRenderTargetMipPixelsAsync(destination, 0, {output}), output);
  assert.deepEqual(output.subarray(0, expected.length), expected);
  assert.equal(backing[0], 39); assert.ok(output.subarray(expected.length).every(value => value === 39));
  if (Type === Uint16Array) {
    const binding = renderer._targets.bindingsFor().get(source.texture);
    assert.equal(binding.view.texture.format, format); assert.equal(binding.sampleType, 'float');
    assert.equal(binding.sampler.minFilter, 'linear'); assert.equal(binding.sampler.magFilter, 'linear');
  }
  assert.equal(renderer._targets.capture(source).references, 0);
  assert.equal(renderer._targets.capture(destination).references, 0);
  assert.equal(renderer.info.f3d.pendingTransfers, 0);
});

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

for (const [format, Type, components = 1] of allFormats) test(`${format}: four-sample attachments resolve without changing format`, async () => {
  const f = fixture(), target = make(f, {format, sampleCount: 4});
  assert.equal(target.allocatedBytes, 20 * components * Type.BYTES_PER_ELEMENT * 5);
  target.withFrame(frame => {
    assert.equal(frame.colorView.texture.sampleCount, 4);
    assert.equal(frame.resolveTarget.texture.sampleCount, 1);
    assert.equal(frame.colorView.texture.format, format);
    assert.equal(frame.resolveTarget.texture.format, format);
  });
  assert.throws(() => target.writePixels(new Type(20 * components)), {code: 'GPU_TRANSFER_SAMPLES'});
  const result = await target.readPixels();
  assert.ok(result instanceof Type); assert.equal(result.length, 20 * components);
  await target.whenIdle(); target.dispose();
});

for (const [format, Type, components = 1] of allFormats) test(`${format}: native mip generation retains compact storage and format`, async () => {
  const f = fixture(), target = make(f, {format, generateMipmaps: true});
  assert.equal(target.mipLevelCount, 3);
  assert.equal(target.allocatedBytes, (20 + 4 + 1) * components * Type.BYTES_PER_ELEMENT);
  target.writePixels(new Type(20 * components)); target.withTexture(() => {});
  assert.equal(target.diagnostics.mipGenerations, 1);
  assert.equal(f.calls.pipelines[0].fragment.targets[0].format, format);
  assert.match(f.calls.modules[0].code, /textureSampleLevel/);
  assert.equal(f.calls.submissions.flatMap(submission => submission.passes).length, 2);
  await target.whenIdle(); target.dispose();
});

for (const [format, Type, components = 1] of allFormats) test(`${format}: pending readback cancels without publishing into caller output`, async () => {
  const f = fixture(), gate = deferred(), target = make(f, {format});
  f.controls.mapping = gate.promise;
  const output = new Type(20 * components).fill(39);
  const pending = target.readPixels({output});
  assert.equal(f.calls.buffers.length, 1);
  target.dispose(); await assert.rejects(pending);
  assert.ok(output.every(value => value === 39));
  assert.equal(f.calls.buffers[0].destroyed, 1);
  assert.equal(target.diagnostics.pendingReadbacks, 0);
  gate.resolve();
});

for (const [format, Type, components] of halfFormats) {
  test(`${format}: source half-float targets retain native precision and sample budgets`, () => {
    const sourceFormat = components === 1 ? T.RedFormat : T.RGFormat;
    for (const samples of [0, 1, 4]) for (const colorSpace of [T.NoColorSpace, T.LinearSRGBColorSpace]) {
      const target = new T.RenderTarget(5, 4, {format: sourceFormat, type: T.HalfFloatType,
        depthBuffer: false, samples, colorSpace});
      const shape = inspectThreeRenderTarget(target, T);
      assert.equal(shape.format, format); assert.equal(shape.float32, false);
      assert.equal(shape.sampleCount, samples === 4 ? 4 : 1);
      assert.equal(shape.colorBytes, 20 * components * 2);
      assert.equal(shape.bytes, 20 * components * 2 * (samples === 4 ? 5 : 1));
      target.texture.colorSpace = T.SRGBColorSpace;
      assert.throws(() => inspectThreeRenderTarget(target, T), {code: 'THREE_TARGET_UNSUPPORTED'});
      target.dispose();
    }
  });

  test(`${format}: mixed attachments charge native storage and two-byte render alignment`, async () => {
    const f = fixture(), colorFormats = ['r8unorm', format], bytes = 20 * (1 + components * 2);
    f.device.features.clear();
    f.device.limits.maxColorAttachmentBytesPerSample = 1 + components * 2;
    assert.throws(() => make(f, {colorFormats}), {code: 'GPU_TARGET_LIMIT'});
    assert.equal(f.calls.textures.length, 0);
    f.device.limits.maxColorAttachmentBytesPerSample++;
    assert.throws(() => make(f, {colorFormats, maxBytes: bytes - 1}), {code: 'GPU_TARGET_BUDGET'});
    assert.equal(f.calls.textures.length, 0);
    const target = make(f, {colorFormats, maxBytes: bytes});
    assert.equal(target.allocatedBytes, bytes);
    target.withFrame(frame => assert.deepEqual(frame.colorViews.map(view => view.texture.format), colorFormats));
    const input = Type.from({length: 20 * components}, (_, i) => [0x7c01, 0xfc00, 0x8000, 0x0001][i % 4]);
    target.writePixels(input, {colorIndex: 1});
    assert.deepEqual(await target.readPixels({colorIndex: 1}), input);
    await target.whenIdle(); target.dispose();
    f.device.limits.maxColorAttachmentBytesPerSample--;
    assert.deepEqual(inspectGpuRenderTargetColors(f.device, colorFormats.slice().reverse()), colorFormats.slice().reverse());
  });
}
