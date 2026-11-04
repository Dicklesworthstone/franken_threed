/** Pinned source targets and the production native target/transfer owners.
 * The recording device independently copies bytes; it does not execute WGSL
 * or establish browser rasterization, GPU validation or performance parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {createGpuRenderTarget, inspectGpuRenderTargetColors} from './gpu_render_target.mjs';
import {mrtTransferDevice, deferred} from './fixtures/mrt_transfer_fixture.mjs';
import {transferDevice} from './fixtures/target_transfer_fixture.mjs';

const profiles = [
  ['r32uint', T.UnsignedIntType, T.RedIntegerFormat, T.RedFormat, Uint32Array, 1, 'uint'],
  ['rg32uint', T.UnsignedIntType, T.RGIntegerFormat, T.RGFormat, Uint32Array, 2, 'uint'],
  ['rgba32uint', T.UnsignedIntType, T.RGBAIntegerFormat, T.RGBAFormat, Uint32Array, 4, 'uint'],
  ['r32sint', T.IntType, T.RedIntegerFormat, T.RedFormat, Int32Array, 1, 'sint'],
  ['rg32sint', T.IntType, T.RGIntegerFormat, T.RGFormat, Int32Array, 2, 'sint'],
  ['rgba32sint', T.IntType, T.RGBAIntegerFormat, T.RGBAFormat, Int32Array, 4, 'sint'],
];
const sourceTarget = (profile, options = {}) => new T.RenderTarget(4, 3, {
  type: profile[1], format: profile[2], depthBuffer: false, ...options,
});
function payload(Type, count, seed = 0) {
  const words = Type === Uint32Array
    ? [0, 1, 16777217, 0xffffffff, 0x80000001, 0x7fffffff, 0x12345679]
    : [0, -1, 16777217, -16777217, -2147483648, 2147483647, 0x12345679];
  return Type.from({length: count}, (_, index) => words[(index + seed) % words.length]);
}
function fixture(t) {
  const f = mrtTransferDevice(), ready = deferred();
  f.device.features = new Set(); f.device.limits.maxTextureDimension3D = 2048;
  const Base = createWebGPURendererClass(T);
  // Only canvas/device initialization is replaced. Queuing, object capture,
  // allocation, uploads, texture copies and readback use production owners.
  class Renderer extends Base {
    async _init() { await ready.promise; this._device = f.device; this._initialized = true; return this; }
  }
  const renderer = new Renderer({canvas: {width: 4, height: 3, style: {}}});
  t.after(async () => {
    ready.resolve(); renderer.dispose();
    while (renderer._drain) await renderer._drain.catch(() => {});
    assert.equal(renderer._transfers.pendingRequests, 0);
    assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
    assert.equal(f.calls.deviceDestroyed, 0);
  });
  return {...f, renderer, ready: () => ready.resolve()};
}

for (const profile of profiles) for (const ordinary of [false, true])
  test(`${profile[0]}: pinned ${ordinary ? 'ordinary' : 'integer'} format alias retains typed native residency`, async t => {
    const [format, , integerFormat, ordinaryFormat, , components, sampleType] = profile;
    const f = mrtTransferDevice(), source = sourceTarget(profile, {format: ordinary ? ordinaryFormat : integerFormat});
    const owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false});
    t.after(() => { owner.dispose(); source.dispose(); });
    const shape = inspectThreeRenderTarget(source, T);
    assert.equal(shape.format, format); assert.equal(shape.integerType, sampleType);
    assert.equal(shape.bytes, 4 * 3 * components * 4); assert.equal(shape.sampleCount, 1);
    const capture = owner.captureNativeTexture(source.texture);
    assert.equal(capture.sampleType, sampleType); assert.equal(capture.viewDimension, '2d');
    assert.equal(capture.sampler.minFilter, 'nearest'); assert.equal(capture.sampler.magFilter, 'nearest');
    assert.equal(source.texture.minFilter, T.LinearFilter, 'texel loads do not rewrite authored filter state');
    assert.equal(source.texture.magFilter, T.LinearFilter);
    assert.throws(() => owner.bindingsFor().get(source.texture), {code: 'THREE_TARGET_UNSUPPORTED'},
      'integer targets cannot enter the fixed floating-point orientation replica');
    assert.equal(f.calls.textures.length, 0, 'native input capture does not allocate');
    owner.withNativeTexture(capture, texture => assert.equal(texture.format, format));
    assert.equal(f.calls.textures.length, 1); assert.equal(owner.allocatedBytes, shape.bytes);
    assert.equal(capture.entry.sample, null, 'native integer inputs share the target image');
    await owner.whenIdle();
  });

for (const profile of profiles)
  test(`${profile[0]}: public queued uploads, copies and typed readback preserve values beyond float32 precision`, async t => {
    const [format, , , , Type, components] = profile, f = fixture(t), r = f.renderer;
    const source = sourceTarget(profile), destination = sourceTarget(profile);
    t.after(() => { source.dispose(); destination.dispose(); });
    const data = payload(Type, 12 * components), expected = data.slice(), later = payload(Type, data.length, 2);
    const write = r.writeRenderTargetPixelsAsync(source, data);
    const copy = r.copyTextureToTextureAsync(source.texture, destination.texture);
    const read = r.readRenderTargetPixelsAsync(destination, 0, 0, 4, 3);
    const overwrite = r.writeRenderTargetPixelsAsync(destination, later);
    data.fill(37); assert.equal(f.calls.textures.length, 0); f.ready();
    await Promise.all([write, copy, overwrite]);
    const pixels = await read;
    assert.ok(pixels instanceof Type); assert.deepEqual(pixels, expected);
    assert.deepEqual(await r.readRenderTargetPixelsAsync(destination, 0, 0, 4, 3), later);
    assert.equal(f.copies[0].source.texture.format, format); assert.equal(f.copies[0].destination.texture.format, format);
    const backing = new Type(expected.length + 4).fill(39), output = backing.subarray(1);
    assert.equal(await r.readRenderTargetMipPixelsAsync(source, 0, {output}), output);
    assert.deepEqual(output.subarray(0, expected.length), expected);
    assert.equal(backing[0], 39); assert.ok(output.subarray(expected.length).every(value => value === 39));
    const wrong = Type === Int32Array ? new Uint32Array(expected.length) : new Int32Array(expected.length);
    const buffers = f.calls.buffers.length, uploads = f.uploads.length;
    await assert.rejects(r.readRenderTargetMipPixelsAsync(source, 0, {output: wrong}), {code: 'GPU_READBACK_OUTPUT'});
    await assert.rejects(r.writeRenderTargetPixelsAsync(source, wrong), {code: 'GPU_TRANSFER_DATA'});
    assert.equal(f.calls.buffers.length, buffers); assert.equal(f.uploads.length, uploads);
    assert.equal(r._targets.capture(source).references, 0); assert.equal(r._targets.capture(destination).references, 0);
  });

for (const profile of profiles) for (const flipY of [false, true])
  test(`${profile[0]}: authored mip transfer retains cropped rows and integer words, flipY=${flipY}`, async t => {
    const [format, , , , Type, components] = profile, f = mrtTransferDevice();
    const source = createGpuRenderTarget(f.device, {width: 8, height: 6, format,
      depthFormat: null, mipLevelCount: 3, copyDestination: true});
    const destination = createGpuRenderTarget(f.device, {width: 8, height: 6, format,
      depthFormat: null, mipLevelCount: 3, copyDestination: true});
    t.after(() => { source.dispose(); destination.dispose(); });
    const expected = payload(Type, 4 * components), rowWords = 3 * components;
    const input = new Type(rowWords * 3).fill(19);
    input.set(expected.subarray(0, 2 * components), components);
    input.set(expected.subarray(2 * components), rowWords + components);
    source.writePixels(input, {destinationMipLevel: 1, x: 1, y: 0, width: 2, height: 2,
      offset: components * 4, bytesPerRow: rowWords * 4, flipY});
    input.fill(0);
    destination.copyFrom(source, {sourceMipLevel: 1, destinationMipLevel: 1,
      sourceX: 1, sourceY: 0, x: 0, y: 1, width: 2, height: 2, flipY});
    assert.equal(f.calls.buffers.length, 0, 'integer copies stay on the texture path');
    assert.deepEqual(await destination.readPixels({mipLevel: 1, x: 0, y: 1, width: 2, height: 2, flipY}), expected);
    assert.equal(source.allocatedBytes, (48 + 12 + 2) * components * 4);
    assert.ok((await destination.readPixels()).every(value => value === 0));
    assert.equal(f.calls.pipelines.length, 0, 'authored integer mips require no filtering pipeline');
  });

for (const kind of ['array', 'volume', 'cube'])
  test(`${kind}: integer target copies address exact layers and authored mips`, async t => {
    const f = fixture(t), r = f.renderer, profile = profiles[5];
    const options = {type: profile[1], format: profile[2], depthBuffer: false};
    const source = kind === 'volume' ? new T.RenderTarget3D(4, 4, 4, options)
      : kind === 'cube' ? new T.CubeRenderTarget(4, options) : new T.RenderTarget(4, 4, {...options, depth: 4});
    source.texture.mipmaps = Array.from({length: kind === 'cube' ? 1 : 2}, () => ({}));
    t.after(() => source.dispose()); f.ready();
    // WebGPU treats a volume's entire mip as one copy subresource. Use distinct
    // levels for 3D; array layers and cube faces may be disjoint at one level.
    const sourceLevel = kind === 'volume' ? 0 : 1, width = 4 >> sourceLevel;
    const data = payload(Int32Array, width * width * 4), expected = new Int32Array(16);
    expected.set(data.subarray(0, 8)); expected.set(data.subarray(width * 4, width * 4 + 8), 8);
    await r.writeRenderTargetPixelsAsync(source, data, {destinationMipLevel: sourceLevel, face: 0});
    assert.equal(r.copyTextureToTexture(source.texture, source.texture,
      new T.Box3(new T.Vector3(0, 0, 0), new T.Vector3(2, 2, 1)), new T.Vector3(0, 0, 1), sourceLevel, 1), undefined);
    assert.deepEqual(await r.readRenderTargetMipPixelsAsync(source, 1, {face: 1}), expected);
    assert.equal(f.copies[0].source.texture, f.copies[0].destination.texture);
    assert.equal(f.copies[0].source.texture.format, 'rgba32sint');
    assert.equal(f.calls.pipelines.length, 0);
  });

test('mixed MRT selects integer attachment types, preserves independent images, and charges aligned component costs', async t => {
  const f = fixture(t), r = f.renderer, source = new T.RenderTarget(4, 3, {count: 3, depthBuffer: false});
  Object.assign(source.textures[1], {format: T.RedIntegerFormat, type: T.IntType});
  Object.assign(source.textures[2], {format: T.RGIntegerFormat, type: T.UnsignedIntType});
  t.after(() => source.dispose()); f.ready();
  const ids = payload(Int32Array, 12), counters = payload(Uint32Array, 24);
  await r.writeRenderTargetPixelsAsync(source, ids, {colorIndex: 1});
  await r.writeRenderTargetPixelsAsync(source, counters, {colorIndex: 2});
  assert.deepEqual(await r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1}), ids);
  assert.deepEqual(await r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 2}), counters);
  assert.ok((await r.readRenderTargetMipPixelsAsync(source)).every(value => value === 0));
  assert.equal(r._targets.capture(source).storage.allocatedBytes, 12 * (4 + 4 + 8));
  f.device.limits.maxColorAttachmentBytesPerSample = 7;
  assert.throws(() => inspectGpuRenderTargetColors(f.device, ['r8unorm', 'r32uint']), {code: 'GPU_TARGET_LIMIT'});
  f.device.limits.maxColorAttachmentBytesPerSample = 8;
  assert.deepEqual(inspectGpuRenderTargetColors(f.device, ['r8unorm', 'r32uint']), ['r8unorm', 'r32uint']);
  f.device.limits.maxColorAttachmentBytesPerSample = 5;
  assert.deepEqual(inspectGpuRenderTargetColors(f.device, ['r32uint', 'r8unorm']), ['r32uint', 'r8unorm']);
});

test('integer color restrictions reject before allocation, while selected float MRT mips still generate', async t => {
  const f = transferDevice();
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  for (const colorFormats of [['r32uint'], ['rgba8unorm', 'rgba32sint']]) {
    assert.throws(() => createGpuRenderTarget(f.device, {colorFormats, sampleCount: 4}), {code: 'GPU_TARGET_SAMPLES'});
    assert.throws(() => createGpuRenderTarget(f.device, {colorFormats, generateMipmaps: true}), {code: 'GPU_TARGET_MIP'});
  }
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.pipelines.length, 0);
  const target = createGpuRenderTarget(f.device, {width: 4, height: 4, colorFormats: ['r32uint', 'rgba16float'],
    depthFormat: null, colorMipLevelCounts: [3, 3], colorGenerateMipmaps: [false, true], copyDestination: true});
  t.after(() => target.dispose());
  target.writePixels(payload(Uint32Array, 16));
  target.writePixels(new Uint16Array(64).fill(0x3c00), {colorIndex: 1});
  const pipelines = f.calls.pipelines.length, submissions = f.calls.submissions.length;
  assert.throws(() => target.generateMipmaps([0]), {code: 'GPU_TARGET_MIP'});
  assert.equal(f.calls.pipelines.length, pipelines); assert.equal(f.calls.submissions.length, submissions);
  assert.equal(target.failed, false, 'unsupported integer filtering does not poison valid storage');
  assert.equal(target.generateMipmaps([1]), true);
  assert.equal(f.calls.pipelines.at(-1).fragment.targets[0].format, 'rgba16float');
  assert.equal(target.diagnostics.mipGenerations, 1);
  assert.deepEqual(await target.readPixels(), payload(Uint32Array, 16));
});

test('source integer restrictions and WebGL orientation refuse before any native side effect', t => {
  const f = mrtTransferDevice(), owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false});
  t.after(() => owner.dispose());
  for (const options of [{samples: 4}, {generateMipmaps: true}, {colorSpace: T.SRGBColorSpace},
    {colorSpace: T.LinearSRGBColorSpace}, {internalFormat: 'R32UI'}]) {
    const source = sourceTarget(profiles[0], options);
    assert.throws(() => owner.capture(source), {code: 'THREE_TARGET_UNSUPPORTED'}); source.dispose();
  }
  const source = sourceTarget(profiles[0]); source.texture.normalized = true;
  assert.throws(() => owner.capture(source), {code: 'THREE_TARGET_UNSUPPORTED'});
  source.texture.normalized = false;
  const webgl = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: true});
  assert.throws(() => webgl.capture(source), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.pipelines.length, 0); assert.equal(f.calls.submissions.length, 0);
  source.dispose(); webgl.dispose();
});

test('integer readback cancellation leaves caller output untouched and releases mapped storage', async t => {
  const f = fixture(t), r = f.renderer, source = sourceTarget(profiles[2]);
  t.after(() => source.dispose()); f.ready();
  await r.writeRenderTargetPixelsAsync(source, payload(Uint32Array, 48));
  f.controls.mapping = deferred();
  const output = new Uint32Array(48).fill(16777217);
  const pending = r.readRenderTargetMipPixelsAsync(source, 0, {output});
  source.dispose();
  await assert.rejects(pending);
  assert.ok(output.every(value => value === 16777217));
  f.controls.mapping.resolve();
});
