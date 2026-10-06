/** An owned offscreen rendering session on a borrowed GPUDevice.
 * Shares the canvas renderer factory contract, but never acquires a canvas
 * texture. Render, sample/copy use and resize are synchronous use boundaries;
 * preparation and completion waits can be cancelled without destroying the
 * caller's device. Native attachments come only from the owned target.
 */
import {createGpuRenderTarget} from './gpu_render_target.mjs';

export class GpuRenderTargetRendererError extends Error {
  constructor(code, message) {
    super(`GPU_OFFSCREEN_${code}: ${message}`);
    this.name = 'GpuRenderTargetRendererError';
    this.code = `GPU_OFFSCREEN_${code}`;
  }
}
const fail = (code, message) => { throw new GpuRenderTargetRendererError(code, message); };

export async function createGpuRenderTargetRenderer(device, createRenderer, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected offscreen options');
  for (const key of Object.keys(options))
    if (!['target', 'signal'].includes(key)) fail('OPTIONS', `Unknown offscreen option: ${key}`);
  if (typeof createRenderer !== 'function') fail('FACTORY', 'Expected a renderer factory');
  const {target: targetOptions = {}, signal} = options;
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' ||
      typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function'))
    fail('OPTIONS', 'Expected AbortSignal');
  const lifetime = new AbortController();
  let target = null, renderer = null, disposed = false, terminal = null, busy = false, preparing = false;
  let rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; });
  stopped.catch(() => {});
  const closed = () => disposed || terminal !== null;
  const wait = promise => Promise.race([promise, stopped]);
  const abort = () => stop(signal.reason ?? new GpuRenderTargetRendererError('ABORTED', 'Offscreen session aborted'));
  function release() {
    signal?.removeEventListener('abort', abort);
    const child = renderer, storage = target;
    renderer = target = null;
    try { child?.dispose(); } finally { storage?.dispose(); }
  }
  function stop(error) {
    if (closed()) return;
    terminal = error;
    rejectStopped(error);
    lifetime.abort(error);
    if (!busy) release();
  }
  function live() {
    if (disposed) fail('DISPOSED', 'Offscreen session is disposed');
    if (terminal !== null) throw terminal;
    if (target?.failed || target?.disposed || renderer?.failed || renderer?.disposed) {
      stop(new GpuRenderTargetRendererError('GPU', 'An owned offscreen stage failed'));
      throw terminal;
    }
  }
  function exclusive() {
    live();
    if (busy || preparing) fail('REENTRANT', 'Offscreen operations cannot overlap a render or preparation');
  }
  function childFailure(error) {
    if (target?.failed || renderer?.failed) stop(error);
    throw error;
  }
  const api = Object.freeze({
    device,
    get disposed() { return disposed; },
    get failed() { return terminal !== null || !!target?.failed || !!renderer?.failed; },
    get width() { return target?.width ?? 0; },
    get height() { return target?.height ?? 0; },
    get diagnostics() { return Object.freeze({target: target?.diagnostics ?? null, renderer: renderer?.diagnostics ?? null}); },
    resize(width, height) {
      exclusive();
      try { target.resize(width, height); return api; }
      catch (error) { return childFailure(error); }
    },
    render(input, frame = {}) {
      exclusive();
      if (!frame || typeof frame !== 'object' || Array.isArray(frame)) fail('FRAME', 'Expected frame options');
      for (const key of ['colorView', 'depthView', 'resolveTarget'])
        if (Object.hasOwn(frame, key)) fail('ATTACHMENT', 'Offscreen attachments are owned by the session');
      busy = true;
      try {
        return target.withFrame((attachments, texture) => {
          const result = renderer.render(input, {...frame, ...attachments}, texture);
          if (result && typeof result.then === 'function') {
            Promise.resolve(result).catch(() => {});
            fail('ASYNC_RENDER', 'Rendering must submit before its borrowed attachments are released');
          }
          live();
        });
      } catch (error) { return childFailure(error); }
      finally { busy = false; if (closed()) release(); }
    },
    withTexture(callback) {
      exclusive();
      busy = true;
      try { return target.withTexture(callback); }
      catch (error) { return childFailure(error); }
      finally { busy = false; if (closed()) release(); }
    },
    async prepare() {
      exclusive();
      preparing = true;
      try { await wait(renderer.prepare?.()); live(); return api; }
      catch (error) { return childFailure(error); }
      finally { preparing = false; }
    },
    async whenIdle() {
      exclusive();
      try { await wait(Promise.all([renderer.whenIdle(), target.whenIdle()])); live(); return api; }
      catch (error) { stop(error); throw error; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during an offscreen submission or texture use');
      if (disposed) return;
      disposed = true;
      const error = new GpuRenderTargetRendererError('DISPOSED', 'Offscreen session disposed');
      rejectStopped(error);
      lifetime.abort(error);
      release();
    },
  });
  signal?.addEventListener('abort', abort, {once: true});
  try {
    if (signal?.aborted) abort();
    live();
    target = createGpuRenderTarget(device, targetOptions);
    if (device.lost && typeof device.lost.then === 'function')
      device.lost.then(info => stop(new GpuRenderTargetRendererError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    // A late factory result still belongs to this owner and must be retired.
    const acquisition = Promise.resolve(createRenderer(device, target.rendererOptions, {signal: lifetime.signal})).then(value => {
      if (closed()) { value?.dispose?.(); throw terminal ?? new GpuRenderTargetRendererError('DISPOSED', 'Offscreen session disposed'); }
      if (!value || ['render', 'dispose', 'whenIdle'].some(key => typeof value[key] !== 'function')) {
        value?.dispose?.();
        fail('FACTORY', 'Renderer must provide render, dispose and whenIdle');
      }
      renderer = value;
      return value;
    });
    await wait(acquisition);
    live();
    return api;
  } catch (error) { stop(error); throw error; }
}
