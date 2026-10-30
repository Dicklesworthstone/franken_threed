/** Pinned source descriptors and the real public renderer, FIFO, target and
 * transfer owners. Native texture copies use the existing independent byte
 * fixture; these tests do not claim browser rasterization or GPU parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass, createWebGLRendererClass} from './three_renderer.mjs';
import {mrtTransferDevice, deferred, bytes, layouts} from './fixtures/mrt_transfer_fixture.mjs';

function fixture(t, {gl = false, maxTransfers, maxTransferBytes} = {}) {
  const f = mrtTransferDevice(), init = deferred();
  const Base = (gl ? createWebGLRendererClass : createWebGPURendererClass)(T, {
    ...(maxTransfers === undefined ? {} : {maxTransfers}),
    ...(maxTransferBytes === undefined ? {} : {maxTransferBytes}),
  });
  // Target-only execution needs no canvas negotiation. Everything after this
  // browser initialization boundary remains the production implementation.
  class Renderer extends Base {
    async _init() {
      await init.promise;
      this._device = f.device; this._initialized = true;
      return this;
    }
  }
  const r = new Renderer({canvas: {width: 8, height: 8, style: {}}});
  const drain = async () => { while (r._drain) await r._drain; };
  t.after(async () => {
    init.resolve(); r.dispose(); await drain().catch(() => {});
    assert.equal(r._transfers.pendingRequests, 0);
    assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
  });
  return {...f, r, drain, async ready() { init.resolve(); await r.init(); }};
}

const textureOptions = {depthBuffer: false, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter};
const cube = () => new T.CubeRenderTarget(4, textureOptions);
function target(formats = ['rgba8unorm', 'rgba16float', 'r32float'], gl = false) {
  const C = gl ? T.WebGLRenderTarget : T.RenderTarget;
  const result = new C(4, 3, {...textureOptions, count: formats.length});
  formats.forEach((format, index) => Object.assign(result.textures[index], {
    type: format.includes('32float') ? T.FloatType : format === 'rgba16float' ? T.HalfFloatType : T.UnsignedByteType,
    format: format === 'r32float' ? T.RedFormat : format === 'rg32float' ? T.RGFormat : T.RGBAFormat,
    colorSpace: format.endsWith('-srgb') ? T.SRGBColorSpace : T.NoColorSpace,
  }));
  return result;
}
function payload(format, pixels = 12, seed = 7) {
  const [Type, components] = layouts.get(format), output = new Type(pixels * components);
  bytes(output).set(Uint8Array.from({length: output.byteLength}, (_, i) => (i * 73 + seed) % 256));
  return output;
}

function mipTarget(width, height, gl = true) {
  const result = new (gl ? T.WebGLRenderTarget : T.RenderTarget)(width, height, textureOptions);
  result.texture.mipmaps = Array.from({length: 1 + Math.floor(Math.log2(Math.max(width, height)))}, () => ({}));
  return result;
}

for (const gl of [false, true]) test(`queued destination mip copies preserve raw regions, selectors and readback (${gl})`, async t => {
  const f = fixture(t, {gl}), r = f.r, seed = mipTarget(4, 4, gl), source = mipTarget(8, 8, gl), destination = mipTarget(16, 16, gl);
  const input = payload('rgba8unorm', 16), expected = new Uint8Array([...input.subarray(20, 28), ...input.subarray(36, 44)]);
  const writing = r.writeRenderTargetPixelsAsync(seed, input);
  const seeding = r.copyRenderTargetPixelsAsync(seed, source, {destinationMipLevel: 1});
  const options = {sourceMipLevel: 1, destinationMipLevel: 2, sourceX: 1, sourceY: 1, x: 1, y: 0, width: 2, height: 2};
  const copying = r.copyRenderTargetPixelsAsync(source, destination, options);
  const reading = r.readRenderTargetMipPixelsAsync(destination, 2, {x: 1, y: 0, width: 2, height: 2});
  const base = r.readRenderTargetMipPixelsAsync(destination, 0), other = r.readRenderTargetMipPixelsAsync(destination, 1);
  options.sourceMipLevel = options.destinationMipLevel = options.x = options.y = 0; input.fill(0);
  assert.equal(f.calls.textures.length, 0); await f.ready();
  assert.deepEqual(await reading, expected); assert.ok((await base).every(value => value === 0));
  assert.ok((await other).every(value => value === 0)); await Promise.all([writing, seeding, copying]);
  assert.equal(f.copies[1].source.mipLevel, 1); assert.equal(f.copies[1].destination.mipLevel, 2);
  assert.equal(f.copies[1].destination.origin.y, gl ? 2 : 0);
  assert.equal(r._targets.capture(destination).storage.diagnostics.mipGenerations, 0);
  assert.equal(r._targets.capture(destination).sample, null); assert.equal(r.getRenderTarget(), null);
});

test('WebGL void and async source copies populate independent destination mips with captured physical rectangles', async t => {
  const f = fixture(t, {gl: true}), r = f.r, source = mipTarget(8, 8), destination = mipTarget(16, 16);
  const writing = r.writeRenderTargetPixelsAsync(source, payload('rgba8unorm', 64));
  const region = new T.Box2(new T.Vector2(2, 1), new T.Vector2(5, 3)), position = new T.Vector2(1, 4);
  assert.equal(r.copyTextureToTexture(source.texture, destination.texture, region, position, 0, 1), undefined);
  const copying = r.copyTextureToTextureAsync(source.texture, destination.texture,
    new T.Box2(new T.Vector2(0, 1), new T.Vector2(2, 3)), new T.Vector2(1, 0), 0, 2);
  region.min.set(99, 99); region.max.set(100, 100); position.set(99, 99);
  const first = r.readRenderTargetMipPixelsAsync(destination, 1, {x: 1, y: 4, width: 3, height: 2});
  const second = r.readRenderTargetMipPixelsAsync(destination, 2, {x: 1, y: 0, width: 2, height: 2});
  const expectedFirst = r.readRenderTargetMipPixelsAsync(source, 0, {x: 2, y: 1, width: 3, height: 2});
  const expectedSecond = r.readRenderTargetMipPixelsAsync(source, 0, {x: 0, y: 1, width: 2, height: 2});
  await f.ready(); await Promise.all([writing, copying]);
  assert.deepEqual(await first, await expectedFirst); assert.deepEqual(await second, await expectedSecond);
  assert.deepEqual(f.copies.map(copy => copy.destination.mipLevel), [1, 2]);
  assert.equal(f.copies[0].source.origin.y, 5); assert.equal(f.copies[0].destination.origin.y, 2);
  assert.equal(r._transfers.pendingRequests, 0); assert.equal(r._targets.capture(destination).storage.diagnostics.mipGenerations, 0);
});

for (const gl of [false, true]) test(`zero-origin source copies select both authored mip levels (${gl})`, async t => {
  const f = fixture(t, {gl}), r = f.r, seed = mipTarget(4, 4, gl), source = mipTarget(8, 8, gl), destination = mipTarget(16, 16, gl);
  await f.ready(); const input = payload('rgba8unorm', 16);
  await r.writeRenderTargetPixelsAsync(seed, input);
  await r.copyRenderTargetPixelsAsync(seed, source, {destinationMipLevel: 1});
  // Native r186 defaults to the base image extent; supply its mip-local box.
  // WebGL r186 derives the default extent from sourceLevel instead.
  const region = gl ? null : new T.Box2(new T.Vector2(), new T.Vector2(4, 4));
  await r.copyTextureToTextureAsync(source.texture, destination.texture, region, null, 1, 2);
  assert.deepEqual(await r.readRenderTargetMipPixelsAsync(destination, 2), input);
  const before = f.copies.length, allocations = f.calls.textures.length;
  if (gl) {
    assert.throws(() => r.copyTextureToTexture(source.texture, destination.texture,
      new T.Box2(new T.Vector2(1, 0), new T.Vector2(3, 2)), null, 1, 2), {code: 'F3D_RENDERER_UNSUPPORTED'});
    assert.throws(() => r.copyTextureToTexture(source.texture, destination.texture,
      new T.Box2(new T.Vector2(), new T.Vector2(2, 2)), new T.Vector2(1, 0), 1, 2), {code: 'F3D_RENDERER_UNSUPPORTED'});
  }
  const generated = mipTarget(16, 16, gl); generated.texture.generateMipmaps = true;
  await assert.rejects(r.copyRenderTargetPixelsAsync(source, generated, {destinationMipLevel: 1}), {code: 'GPU_TRANSFER_MIP'});
  assert.equal(f.copies.length, before); assert.equal(f.calls.textures.length, allocations);
  assert.equal(r._transfers.pendingRequests, 0);
});

test('WebGL destination mip copies preserve empty default source tails while explicit rectangles copy the allocated texel row', async t => {
  const f = fixture(t, {gl: true}), r = f.r, source = mipTarget(8, 1), destination = mipTarget(8, 1);
  await f.ready();
  await r.copyTextureToTextureAsync(source.texture, destination.texture, null, null, 1, 1);
  assert.equal(f.copies.length, 0); assert.equal(f.calls.textures.length, 0);
  await r.copyTextureToTextureAsync(source.texture, destination.texture,
    new T.Box2(new T.Vector2(), new T.Vector2(4, 1)), null, 1, 1);
  assert.equal(f.copies.length, 1); assert.equal(f.copies[0].source.mipLevel, 1); assert.equal(f.copies[0].destination.mipLevel, 1);
  assert.deepEqual(f.copies[0].size, {width: 4, height: 1, depthOrArrayLayers: 1});
});

test('native public readback reads all six source cube faces and defaults to +X', async t => {
  const f = fixture(t), r = f.r, source = cube(); await f.ready();
  for (let face = 0; face < 6; face++) {
    const data = payload('rgba8unorm', 16, face * 31);
    await r.writeRenderTargetPixelsAsync(source, data, {face});
    r.setRenderTarget(source, 5 - face);
    assert.deepEqual(await r.readRenderTargetPixelsAsync(source, 0, 0, 4, 4, 0, face), data);
    assert.equal(f.reads.at(-1).source.origin.z, face);
    assert.equal(r._activeCubeFace, 5 - face, 'readback never changes render selection');
  }
  assert.deepEqual(await r.readRenderTargetPixelsAsync(source, 0, 0, 4, 4), payload('rgba8unorm', 16, 0));
  assert.equal(source.texture.renderTarget, null, 'retain the pinned cube texture ownership');
});

for (const format of ['rgba8unorm', 'rgba8unorm-srgb', 'rgba16float', 'r32float', 'rg32float', 'rgba32float'])
  for (const gl of [false, true]) test(`public ${gl ? 'WebGL' : 'native'} MRT readback keeps ${format} precision and argument order`, async t => {
    const f = fixture(t, {gl}), r = f.r, source = target(['rgba8unorm', format], gl); await f.ready();
    const data = payload(format); await r.writeRenderTargetPixelsAsync(source, data, {colorIndex: 1});
    const [Type] = layouts.get(format), output = new Type(data.length + 3).fill(11);
    const result = gl ? await r.readRenderTargetPixelsAsync(source, 0, 0, 4, 3, output, undefined, 1)
      : await r.readRenderTargetPixelsAsync(source, 0, 0, 4, 3, 1);
    assert.ok(result instanceof Type);
    assert.deepEqual(bytes(result).subarray(0, data.byteLength), bytes(data));
    assert.equal(f.reads.at(-1).source.texture.format, format);
    if (gl) { assert.equal(result, output); assert.ok(output.subarray(data.length).every(value => value === 11)); }
    assert.equal(r.getRenderTarget(), null);
  });

test('WebGL public MRT rectangles retain bottom-left origin and caller view boundaries', async t => {
  const f = fixture(t, {gl: true}), r = f.r, source = target(undefined, true); await f.ready();
  const data = payload('rgba16float'); await r.writeRenderTargetPixelsAsync(source, data, {colorIndex: 1});
  const storage = new Uint16Array(24).fill(12345), output = storage.subarray(3, 21);
  assert.equal(await r.readRenderTargetPixelsAsync(source, 1, 0, 2, 2, output, undefined, 1), output);
  const expected = new Uint16Array([...data.subarray(4, 12), ...data.subarray(20, 28)]);
  assert.deepEqual(output.subarray(0, 16), expected);
  assert.ok(storage.subarray(0, 3).every(value => value === 12345));
  assert.ok(storage.subarray(19).every(value => value === 12345));
  assert.equal(f.reads.at(-1).source.origin.y, 1);
});

test('public cube reads keep FIFO byte history while delayed mapping does not block later writes', async t => {
  const f = fixture(t), r = f.r, source = cube(), mapping = deferred(); f.controls.mapping = mapping;
  const before = payload('rgba8unorm', 16, 1), after = payload('rgba8unorm', 16, 2);
  const writeBefore = r.writeRenderTargetPixelsAsync(source, before, {face: 2});
  const read = r.readRenderTargetPixelsAsync(source, 0, 0, 4, 4, 0, 2);
  const writeAfter = r.writeRenderTargetPixelsAsync(source, after, {face: 2});
  r.setRenderTarget(source, 4);
  assert.equal(f.calls.textures.length, 0, 'accept before initialization without GPU allocation');
  await f.ready(); await f.drain(); await Promise.all([writeBefore, writeAfter]);
  assert.equal(f.uploads.length, 2); assert.equal(f.reads.length, 1);
  assert.equal(r._pending.length, 0);
  let complete = false; read.then(() => { complete = true; });
  await new Promise(resolve => setImmediate(resolve)); assert.equal(complete, false);
  mapping.resolve(); assert.deepEqual(await read, before);
  assert.deepEqual(await r.readRenderTargetPixelsAsync(source, 0, 0, 4, 4, 0, 2), after);
  assert.equal(r._activeCubeFace, 4);
});

test('source reads share transfer request limits and release them after completion', async t => {
  const f = fixture(t, {maxTransfers: 1}), r = f.r, source = target();
  const read = r.readRenderTargetPixelsAsync(source, 0, 0, 1, 1, 1);
  assert.equal(r.info.f3d.pendingTransfers, 1);
  await assert.rejects(r.readRenderTargetPixelsAsync(source, 0, 0, 1, 1, 2), {code: 'F3D_RENDERER_LIMIT'});
  await assert.rejects(r.readRenderTargetMipPixelsAsync(source), {code: 'F3D_RENDERER_LIMIT'});
  await f.ready(); await read;
  assert.equal(r.info.f3d.pendingTransfers, 0);
  assert.ok(await r.readRenderTargetPixelsAsync(source, 0, 0, 1, 1, 2) instanceof Float32Array);
});

test('invalid public attachment and face selectors reject before allocation or queueing', async t => {
  const f = fixture(t), r = f.r, source = cube(), flat = target();
  for (const value of [-1, 6, 0.5, NaN, '1', null])
    await assert.rejects(r.readRenderTargetPixelsAsync(source, 0, 0, 0, 1, 0, value), {code: 'F3D_RENDERER_FACE'});
  for (const value of [-1, 3, 0.5, NaN, '1', null])
    await assert.rejects(r.readRenderTargetPixelsAsync(flat, 0, 0, 1, 1, value), {code: 'F3D_RENDERER_ATTACHMENT'});
  await assert.rejects(r.readRenderTargetPixelsAsync(flat, 0, 0, 1, 1, 0, 1), {code: 'F3D_RENDERER_FACE'});
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.buffers.length, 0);
  assert.equal(r._pending.length, 0); assert.equal(r.info.f3d.pendingTransfers, 0);
});

test('disposing a target cancels an in-flight public MRT read before caller bytes change', async t => {
  const f = fixture(t, {gl: true}), r = f.r, source = target(undefined, true); await f.ready();
  f.controls.mapping = deferred();
  const output = new Uint16Array(48).fill(12345);
  const read = r.readRenderTargetPixelsAsync(source, 0, 0, 4, 3, output, undefined, 1);
  assert.equal(f.reads.length, 1);
  source.dispose();
  await assert.rejects(read, {code: 'THREE_TARGET_STALE'});
  assert.ok(output.every(value => value === 12345));
  assert.equal(r.info.f3d.pendingTransfers, 0);
  f.controls.mapping.resolve();
});

test('resizing a queued target invalidates the selected generation before submission', async t => {
  const f = fixture(t), r = f.r, source = target();
  const read = r.readRenderTargetPixelsAsync(source, 0, 0, 1, 1, 2);
  source.setSize(8, 8);
  await assert.rejects(read, {code: 'THREE_TARGET_STALE'});
  await f.ready(); await f.drain();
  assert.equal(f.reads.length, 0); assert.equal(f.calls.buffers.length, 0);
});

test('caller output is published only after producer completion without blocking later submissions', async t => {
  const f = fixture(t, {gl: true}), r = f.r, source = target(['rgba8unorm'], true); await f.ready();
  const before = payload('rgba8unorm'), after = payload('rgba8unorm', 12, 35);
  await r.writeRenderTargetPixelsAsync(source, before);
  f.controls.completion = deferred();
  const output = new Uint8Array(before.length + 4).fill(211);
  const read = r.readRenderTargetPixelsAsync(source, 0, 0, 4, 3, output);
  const write = r.writeRenderTargetPixelsAsync(source, after);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.uploads.length, 2, 'mapping/completion cannot block a later write submission');
  assert.equal(f.calls.buffers[0].destroyed, 1, 'mapping has completed and staging was released');
  assert.ok(output.every(value => value === 211));
  f.controls.completion.resolve();
  assert.equal(await read, output); await write;
  assert.deepEqual(output.subarray(0, before.length), before);
  assert.ok(output.subarray(before.length).every(value => value === 211));
});

for (const failure of ['target disposal', 'renderer disposal', 'completion error'])
  test(`${failure} after mapping leaves the caller buffer unchanged`, async t => {
    const f = fixture(t, {gl: true}), r = f.r, source = target(['rgba8unorm'], true); await f.ready();
    await r.writeRenderTargetPixelsAsync(source, payload('rgba8unorm'));
    f.controls.completion = deferred();
    const output = new Uint8Array(48).fill(211);
    const read = r.readRenderTargetPixelsAsync(source, 0, 0, 4, 3, output);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(f.calls.buffers[0].destroyed, 1);
    assert.ok(output.every(value => value === 211));
    const expected = failure === 'target disposal' ? {code: 'THREE_TARGET_STALE'}
      : failure === 'renderer disposal' ? {code: 'F3D_RENDERER_DISPOSED'} : /producer completion failed/;
    const rejected = assert.rejects(read, expected);
    if (failure === 'target disposal') source.dispose();
    else if (failure === 'renderer disposal') r.dispose();
    else f.controls.completion.reject(new Error('producer completion failed'));
    await rejected;
    assert.ok(output.every(value => value === 211));
    assert.equal(r._transfers.allocatedBytes, 0);
    f.controls.completion.resolve();
  });

test('explicit caller cancellation after mapping also prevents output publication', async t => {
  const f = fixture(t), r = f.r, source = target(['rgba8unorm']); await f.ready();
  await r.writeRenderTargetPixelsAsync(source, payload('rgba8unorm'));
  f.controls.completion = deferred();
  const controller = new AbortController(), reason = new Error('cancelled after map'), output = new Uint8Array(48).fill(211);
  const read = r.readRenderTargetMipPixelsAsync(source, 0, {output, signal: controller.signal});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.calls.buffers[0].destroyed, 1);
  const rejected = assert.rejects(read, error => error === reason); controller.abort(reason); await rejected;
  assert.ok(output.every(value => value === 211)); assert.equal(r._transfers.allocatedBytes, 0);
  f.controls.completion.resolve();
});

test('private mapped output remains charged until publication or cancellation', async t => {
  const f = fixture(t, {maxTransferBytes: 16}), r = f.r, source = target(['rgba8unorm']); await f.ready();
  f.controls.completion = deferred();
  const read = r.readRenderTargetPixelsAsync(source, 0, 0, 2, 2);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.calls.buffers[0].destroyed, 1); assert.equal(r._transfers.allocatedBytes, 16);
  await assert.rejects(r.readRenderTargetPixelsAsync(source, 0, 0, 1, 1), {code: 'F3D_RENDERER_LIMIT'});
  f.controls.completion.resolve(); assert.equal((await read).length, 16);
  assert.equal(r._transfers.allocatedBytes, 0);
  assert.equal((await r.readRenderTargetPixelsAsync(source, 0, 0, 1, 1)).length, 4);
});

test('source mutation during completion and caller-buffer detachment cannot publish stale output', async t => {
  for (const detach of [false, true]) {
    const f = fixture(t, {gl: true}), r = f.r, source = target(['rgba8unorm'], true); await f.ready();
    await r.writeRenderTargetPixelsAsync(source, payload('rgba8unorm'));
    f.controls.completion = deferred();
    const output = new Uint8Array(48).fill(211);
    const read = r.readRenderTargetPixelsAsync(source, 0, 0, 4, 3, output);
    await new Promise(resolve => setImmediate(resolve));
    let transferred;
    if (detach) transferred = structuredClone(output.buffer, {transfer: [output.buffer]});
    else source.width++;
    const rejected = assert.rejects(read, {code: detach ? 'GPU_READBACK_OUTPUT' : 'THREE_TARGET_STALE'});
    f.controls.completion.resolve(); await rejected;
    assert.ok((detach ? new Uint8Array(transferred) : output).every(value => value === 211));
    assert.equal(r._transfers.allocatedBytes, 0);
  }
});

test('invalid caller output fails before target allocation or FIFO admission', async t => {
  const f = fixture(t, {gl: true}), source = target(['rgba16float'], true);
  for (const output of [new Uint8Array(96), new Uint16Array(47), new DataView(new ArrayBuffer(96))])
    await assert.rejects(f.r.readRenderTargetPixelsAsync(source, 0, 0, 4, 3, output), {code: 'GPU_READBACK_OUTPUT'});
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.buffers.length, 0);
  assert.equal(f.r._pending.length, 0); assert.equal(f.r._transfers.allocatedBytes, 0);
});
