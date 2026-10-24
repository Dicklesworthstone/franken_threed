/** Production controller and source/native owners; source class descriptors,
 * queue driver and native GPU calls are controlled test boundaries. This does
 * not construct a renderer class, compile WGSL or establish browser parity. */
import assert from 'node:assert/strict';
import {targetThree} from './target_residency_fixture.mjs';
import {mrtTransferDevice, deferred, layouts, bytes} from './mrt_transfer_fixture.mjs';
import {createThreeRenderTargets} from '../three_render_targets.mjs';
import {createThreeRendererTransfers} from '../three_renderer_transfers.mjs';
export {deferred, layouts, bytes};
export const flush = () => new Promise(resolve => setImmediate(resolve));
export const T = {...targetThree, FloatType: 1015, RedFormat: 1028, RGFormat: 1030};
export const formats = ['rgba8unorm', 'rgba8unorm-srgb', 'rgba16float', 'r32float', 'rg32float', 'rgba32float'];
export function target(colorFormats = ['rgba8unorm', 'rgba16float', 'r32float'], {width = 4, height = 3, samples = 0} = {}) {
  const result = new T.RenderTarget(width, height); result.samples = samples;
  result.textures = colorFormats.map(format => {
    assert.ok(formats.includes(format));
    return Object.assign(new T.Texture(), {isRenderTargetTexture: true, renderTarget: result,
      type: format.includes('32float') ? T.FloatType : format === 'rgba16float' ? T.HalfFloatType : T.UnsignedByteType,
      format: format === 'r32float' ? T.RedFormat : format === 'rg32float' ? T.RGFormat : T.RGBAFormat,
      colorSpace: format.endsWith('-srgb') ? T.SRGBColorSpace : T.NoColorSpace,
      minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  });
  return result;
}
export function payload(format, pixels = 12, seed = 7) {
  const [Type, components] = layouts.get(format), result = new Type(pixels * components);
  bytes(result).set(Uint8Array.from({length: result.byteLength}, (_, i) => (i * 73 + seed) % 256));
  return result;
}
export class TransferError extends Error {
  constructor(code, message) { super(`F3D_RENDERER_${code}: ${message}`); this.code = `F3D_RENDERER_${code}`; }
}
export function fixture({gl = false, initialized = false, maxBytes, maxRequests, ownerBytes} = {}) {
  const f = mrtTransferDevice(), events = [];
  const renderer = {_initialized: initialized, _disposed: false, _deferredError: null,
    _pending: [], _drain: null, _preparing: null, _bottomLeftOrigin: gl, _targetLifetime: new AbortController(),
    _enqueue(request) { this._pending.push(request); },
    dispose() {
      if (this._disposed) return;
      this._disposed = true;
      const error = new TransferError('DISPOSED', 'Renderer disposed');
      this._targetLifetime.abort(error); this._targets.dispose();
      for (const request of this._pending.splice(0)) request.reject(error);
    }};
  const owner = renderer._targets = createThreeRenderTargets(T, {getDevice: () => renderer._initialized ? f.device : null,
    ...(ownerBytes === undefined ? {} : {maxBytes: ownerBytes}), flipY: gl});
  const controller = createThreeRendererTransfers(renderer, TransferError, {
    ...(maxBytes === undefined ? {} : {maxBytes}), ...(maxRequests === undefined ? {} : {maxRequests})});
  renderer.writeRenderTargetPixelsAsync = (...args) => controller.writePixels(...args);
  renderer.copyRenderTargetPixelsAsync = (...args) => controller.copyColor(...args);
  for (const [method, kind] of [['writePixels', 'upload'], ['copyColor', 'copy']]) {
    const original = owner[method];
    owner[method] = (...args) => { const result = original(...args); events.push(kind); return result; };
  }
  const encode = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const encoder = encode(), copy = encoder.copyTextureToBuffer;
    encoder.copyTextureToBuffer = (...args) => { events.push('read'); return copy(...args); };
    return encoder;
  };
  function drain() {
    assert.equal(renderer._initialized, true);
    renderer._drain = true;
    try {
      for (let request; (request = renderer._pending.shift());) {
        try { request.resolve(request.issue()); } catch (error) { request.reject(error); }
      }
    } finally { renderer._drain = null; }
  }
  function draw(source, run) {
    const entry = owner.capture(source);
    const pending = new Promise((resolve, reject) => renderer._enqueue({resolve, reject,
      issue() { events.push('draw'); owner.render(entry, run); return entry.storage.whenIdle(); }}));
    pending.catch(() => {}); return pending;
  }
  function close() {
    renderer.dispose();
    assert.equal(controller.pendingRequests, 0); assert.equal(controller.allocatedBytes, 0);
    assert.equal(owner.allocatedBytes, 0);
    assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
    assert.equal(f.calls.deviceDestroyed, 0);
  }
  return {...f, renderer, owner, controller, events, drain, draw, close,
    ready() { renderer._initialized = true; drain(); }};
}
