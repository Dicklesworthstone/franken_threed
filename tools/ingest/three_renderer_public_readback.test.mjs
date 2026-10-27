/** Pinned source descriptors and the real public renderer, FIFO, target and
 * transfer owners. Native texture copies use the existing independent byte
 * fixture; these tests do not claim browser rasterization or GPU parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass, createWebGLRendererClass} from './three_renderer.mjs';
import {mrtTransferDevice, deferred, bytes, layouts} from './fixtures/mrt_transfer_fixture.mjs';

function fixture(t, {gl = false, maxTransfers} = {}) {
  const f = mrtTransferDevice(), init = deferred();
  const Base = (gl ? createWebGLRendererClass : createWebGPURendererClass)(T,
    maxTransfers === undefined ? {} : {maxTransfers});
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
