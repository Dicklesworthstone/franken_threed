/** Device negotiation and owned canvas execution for explicit renderer factories.
 * A factory returns {render(input, frame), dispose(), whenIdle(), prepare?()}.
 * It borrows the device; the session owns the returned renderer. No frame loop.
 */
import {createGpuCanvasTarget, GpuCanvasError} from './gpu_canvas.mjs';
const fail = (code, message) => { throw new GpuCanvasError(code, message); };
const object = (value, label) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('OPTIONS', `Expected ${label}`);
  return value;
};
const lowerLimits = new Set(['minUniformBufferOffsetAlignment', 'minStorageBufferOffsetAlignment']);

export async function createGpuCanvasRenderer(canvas, createRenderer, options = {}) {
  object(options, 'renderer options');
  for (const key of Object.keys(options))
    if (!['device', 'gpu', 'powerPreference', 'requiredFeatures', 'requiredLimits', 'target', 'signal'].includes(key))
      fail('OPTIONS', `Unknown renderer option: ${key}`);
  if (typeof createRenderer !== 'function') fail('FACTORY', 'Expected an explicit renderer factory');
  const {
    device: suppliedDevice = null, gpu = globalThis.navigator?.gpu,
    powerPreference, signal,
  } = options;
  const targetOptions = {...object(options.target ?? {}, 'canvas target options')};
  const features = options.requiredFeatures ?? [];
  if (!Array.isArray(features) || features.length > 64 || features.some(f => typeof f !== 'string' || !f))
    fail('FEATURE', 'Required features must be a bounded array of names');
  const requiredFeatures = [...new Set(features)];
  const requiredLimits = {...object(options.requiredLimits ?? {}, 'required limits')};
  for (const value of Object.values(requiredLimits))
    if (!Number.isSafeInteger(value) || value < 0) fail('LIMIT', 'Required limits must be nonnegative safe integers');
  if (powerPreference !== undefined && !['low-power', 'high-performance'].includes(powerPreference))
    fail('OPTIONS', 'Invalid adapter power preference');
  if (signal && (typeof signal.aborted !== 'boolean' || typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected an AbortSignal');
  let device = suppliedDevice, ownsDevice = false, target = null, renderer = null;
  let disposed = false, terminal = null, busy = false, preparing = false, lastFrameRendered = false;
  let rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; });
  stopped.catch(() => {});
  const closed = () => disposed || terminal !== null;
  const aborted = () => stop(new GpuCanvasError('ABORTED', 'Canvas renderer initialization or lifetime was aborted'));
  function release() {
    signal?.removeEventListener('abort', aborted);
    const owned = renderer; renderer = null;
    try { if (typeof owned?.dispose === 'function') owned.dispose(); } finally {
      try { target?.dispose(); } finally {
        if (ownsDevice) { ownsDevice = false; device.destroy(); }
      }
    }
  }
  function stop(error) {
    if (closed()) return;
    terminal = error; rejectStopped(error);
    // AbortSignal delivery can be synchronous inside caller input getters.
    // The active native submission must leave its nonreentrant boundary first.
    if (!busy) release();
  }
  function live() {
    if (disposed) fail('DISPOSED', 'Canvas renderer is disposed');
    if (terminal) throw terminal;
    if (target?.failed || target?.disposed || renderer?.failed || renderer?.disposed) {
      stop(new GpuCanvasError('GPU', 'An owned canvas or renderer resource is unavailable'));
      throw terminal;
    }
  }
  function capabilities(host) {
    for (const feature of requiredFeatures)
      if (!host.features?.has(feature)) fail('FEATURE', `Required device feature is unavailable: ${feature}`);
    for (const [name, needed] of Object.entries(requiredLimits)) {
      const available = host.limits?.[name];
      if (!Number.isSafeInteger(available) || (lowerLimits.has(name) ? available > needed : available < needed))
        fail('LIMIT', `Required device limit is unavailable: ${name}`);
    }
  }
  const wait = promise => Promise.race([promise, stopped]);
  function exclusive(operation) {
    live();
    if (busy || preparing) fail('REENTRANT', 'Canvas rendering cannot overlap preparation or another operation');
    busy = true;
    try { return operation(); }
    catch (error) {
      if (target?.failed || renderer?.failed) stop(error);
      throw error;
    } finally { busy = false; if (closed()) release(); }
  }
  const api = Object.freeze({
    canvas,
    get device() { return device; }, get ownsDevice() { return ownsDevice; },
    get width() { return target?.width ?? 0; }, get height() { return target?.height ?? 0; },
    get rendererOptions() { return target?.rendererOptions; },
    get suspended() { return target?.suspended ?? true; },
    get lastFrameRendered() { return lastFrameRendered; },
    get disposed() { return disposed; }, get failed() { return terminal !== null || !!target?.failed || !!renderer?.failed; },
    get diagnostics() { return Object.freeze({canvas: target?.diagnostics ?? null, renderer: renderer?.diagnostics ?? null}); },
    render(input, frame = {}) {
      return exclusive(() => {
        object(frame, 'frame options');
        // Capture getters before borrowing any current presentation texture.
        const packet = {...frame};
        for (const key of ['colorView', 'depthView', 'resolveTarget'])
          if (Object.hasOwn(packet, key)) fail('FRAME', 'Canvas attachments cannot be overridden');
        live();
        lastFrameRendered = target.withFrame(attachments => renderer.render(input, {...packet, ...attachments}));
        live(); return api;
      });
    },
    resize(width, height) { return exclusive(() => { target.resize(width, height); return api; }); },
    setSize(width, height, pixelRatio = 1) {
      return exclusive(() => {
        if (![width, height].every(n => Number.isSafeInteger(n) && n >= 0) ||
            typeof pixelRatio !== 'number' || !Number.isFinite(pixelRatio) || pixelRatio <= 0)
          fail('SIZE', 'Expected nonnegative logical dimensions and a finite positive pixel ratio');
        target.resize(Math.floor(width * pixelRatio), Math.floor(height * pixelRatio));
        return api;
      });
    },
    async prepare() {
      live();
      if (busy || preparing) fail('REENTRANT', 'Preparation cannot overlap another operation');
      preparing = true;
      try { await wait(renderer.prepare?.()); live(); return api; }
      catch (error) { if (target?.failed || renderer?.failed) stop(error); throw error; }
      finally { preparing = false; }
    },
    async whenIdle() {
      live();
      if (busy || preparing) fail('REENTRANT', 'Await preparation before waiting for idle');
      try { await wait(Promise.all([target.whenIdle(), renderer.whenIdle()])); live(); return api; }
      catch (error) { stop(error); throw error; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose inside a synchronous render operation');
      if (disposed) return;
      disposed = true; rejectStopped(new GpuCanvasError('DISPOSED', 'Canvas renderer is disposed'));
      release();
    },
  });
  signal?.addEventListener('abort', aborted, {once: true});
  try {
    if (signal?.aborted) aborted();
    live();
    if (!device) {
      if (!gpu || typeof gpu.requestAdapter !== 'function') fail('UNAVAILABLE', 'WebGPU is unavailable in this host');
      const adapter = await wait(gpu.requestAdapter(powerPreference === undefined ? {} : {powerPreference}));
      live();
      if (!adapter) fail('ADAPTER', 'No WebGPU adapter is available');
      capabilities(adapter);
      // Publish ownership in the promise continuation so cancellation cannot
      // fall between device resolution and installation in the owner.
      const acquisition = Promise.resolve(adapter.requestDevice({requiredFeatures, requiredLimits})).then(value => {
        if (closed()) { value.destroy(); throw terminal; }
        device = value; ownsDevice = true; return value;
      });
      await wait(acquisition); live();
    }
    capabilities(device);
    if (device.lost && typeof device.lost.then === 'function')
      device.lost.then(info => stop(new GpuCanvasError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    targetOptions.format ??= gpu?.getPreferredCanvasFormat?.() ?? 'bgra8unorm';
    target = createGpuCanvasTarget(device, canvas, targetOptions);
    await wait(target.whenIdle()); live();
    const construction = Promise.resolve(createRenderer(device, target.rendererOptions)).then(value => {
      if (closed()) { value?.dispose?.(); throw terminal; }
      renderer = value;
      if (!value || ['render', 'dispose', 'whenIdle'].some(key => typeof value[key] !== 'function'))
        fail('FACTORY', 'Renderer factory must return render, dispose and whenIdle methods');
      return value;
    });
    await wait(construction); live();
    await wait(renderer.whenIdle()); live();
    return api;
  } catch (error) { stop(error); throw error; }
}
