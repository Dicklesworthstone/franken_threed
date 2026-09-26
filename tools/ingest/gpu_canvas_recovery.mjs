/** Explicit, bounded device-loss reconstruction for the existing canvas owner.
 * No automatic retry, frame replay, source-state snapshot or fallback backend.
 * Reinvoke the caller's renderer factory on a fresh device; keep application
 * scene/animation state outside that factory. Every generation has its own
 * abort signal, stopped operations, native handles and initialization boundary.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {GpuCanvasError} from './gpu_canvas.mjs';

const fail = (code, message) => { throw new GpuCanvasError(code, message); };
const leases = new WeakMap();
const object = (value, label) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('OPTIONS', `Expected ${label}`);
  return value;
};
const dimension = value => {
  if (!Number.isSafeInteger(value) || value < 0) fail('SIZE', 'Expected a nonnegative safe drawing-buffer dimension');
  return value;
};
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  // Waiters are optional. Disposal/loss must not create an unhandled rejection.
  promise.catch(() => {});
  return {promise, resolve, reject};
}

/** Same options/factory as createGpuCanvasRenderer, plus maxRecoveryAttempts
 * (default 3, range 0..64). recover() negotiates a fresh adapter/device after an
 * owned-device loss. After a borrowed-device loss, pass {device: replacement},
 * or explicitly {device: null} to opt into negotiation and device ownership.
 * Only successful initialization/recovery increments generation (starts at 1).
 */
export function createRecoverableGpuCanvasRenderer(canvas, createRenderer, options = {}) {
  return createRecoverableCanvas(createGpuCanvasRenderer, canvas, createRenderer, options);
}


/** Reconstruct the complete linear-HDR target/output stack, not only the scene.
 * Snapshot configuration and reserve lifetime ownership before the lazy import.
 * The direct canvas entry never loads HDR dependencies merely by importing it.
 */
export function createRecoverableGpuHdrCanvasRenderer(canvas, createRenderer, options = {}) {
  return createRecoverableCanvas(async (...args) => {
    const {createGpuHdrCanvasRenderer} = await import('./gpu_hdr_canvas.mjs');
    return createGpuHdrCanvasRenderer(...args);
  }, canvas, createRenderer, options);
}

async function createRecoverableCanvas(createSession, canvas, createRenderer, input) {
  object(input, 'recoverable canvas options');
  if (!canvas || typeof canvas.getContext !== 'function') fail('OWNERSHIP', 'Expected an exclusively owned canvas');
  if (typeof createRenderer !== 'function') fail('FACTORY', 'Expected a renderer reconstruction factory');
  const {maxRecoveryAttempts = 3, ...options} = {...input};
  if (!Number.isSafeInteger(maxRecoveryAttempts) || maxRecoveryAttempts < 0 || maxRecoveryAttempts > 64)
    fail('RECOVERY_LIMIT', 'maxRecoveryAttempts must be an integer in [0,64]');
  if (options.device != null && typeof options.device !== 'object') fail('RECOVERY_DEVICE', 'Expected a device object or null');
  const signal = options.signal;
  if (signal && (typeof signal.aborted !== 'boolean' || typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected an AbortSignal');
  // Configuration is immutable across retries. In particular, a caller cannot
  // silently lower capabilities or alter HDR/attachment semantics during awaits.
  options.gpu ??= globalThis.navigator?.gpu;
  for (const key of ['target', 'requiredLimits', 'renderTarget', 'output'])
    if (options[key] !== undefined) options[key] = Object.freeze({...object(options[key], key + ' options')});
  if (options.requiredFeatures !== undefined) {
    if (!Array.isArray(options.requiredFeatures)) fail('FEATURE', 'Expected required feature names');
    options.requiredFeatures = Object.freeze([...options.requiredFeatures]);
  }
  let width = dimension(options.target?.width ?? canvas.width),
    height = dimension(options.target?.height ?? canvas.height);
  if (leases.has(canvas)) fail('OWNERSHIP', 'A recoverable session already owns this canvas lifetime');
  const token = {}, devices = new WeakSet();
  leases.set(canvas, token);
  let active = null, lastReady = null, generation = 0, attempts = 0;
  let state = 'initializing', disposed = false, fatal = null, lastError = null, lastLoss = null;
  let busy = false, lastFrameRendered = false, rendererOptions;
  const unlink = () => {
    signal?.removeEventListener('abort', abort);
    if (leases.get(canvas) === token) leases.delete(canvas);
  };
  const abort = () => terminate(new GpuCanvasError('ABORTED', 'Recoverable canvas lifetime was aborted'));
  function live() {
    if (disposed) fail('DISPOSED', 'Recoverable canvas is disposed');
    if (fatal) throw fatal;
  }
  function retire(record) {
    if (!record || record.retired) return;
    record.retiring = true;
    // Base owners defer destruction out of synchronous native submission.
    record.abort.abort(record.reason);
    if (busy) return;
    record.retired = true;
    const session = record.session;
    record.session = null;
    try { session?.dispose(); }
    catch (error) {
      // A broken reconstruction factory's cleanup is not a recoverable loss.
      fatal = error; lastError = error; state = disposed ? 'disposed' : 'failed'; unlink();
    }
  }
  function stopRecord(record, error, info = null) {
    if (!record || record.stopped) return;
    record.stopped = true;
    record.reason = error;
    record.stop.reject(error);
    const loss = info === null ? null : Object.freeze({
      generation: record.published ? record.generation : null,
      recoveryAttempt: record.attempt,
      reason: typeof info.reason === 'string' ? info.reason : 'unknown',
      message: typeof info.message === 'string' ? info.message : 'Device lost',
    });
    if (loss) record.loss.resolve(loss); else record.loss.reject(error);
    if (active === record && record.published && !disposed && !fatal) {
      lastError = error;
      lastFrameRendered = false;
      if (loss) { lastLoss = loss; state = 'lost'; }
      else { fatal = error; state = 'failed'; unlink(); }
    }
    retire(record);
  }
  function terminate(error) {
    if (disposed || fatal) return;
    fatal = error; lastError = error; state = 'failed'; lastFrameRendered = false;
    unlink(); stopRecord(active, error);
  }
  function check(record) {
    live();
    if (record.stopped) throw record.reason;
    if (record !== active) fail('RECOVERY_STALE', 'The canvas operation belongs to a retired generation');
  }
  function ready() {
    live();
    if (busy || active?.preparing) fail('REENTRANT', 'Canvas operations cannot overlap submission or preparation');
    if (state === 'lost') throw lastError;
    if (state !== 'ready') fail('RECOVERING', 'Await device reconstruction before using the canvas');
    const record = active;
    if (record.session.failed || record.session.disposed) {
      // Session failures with no observed GPUDevice.lost are NOT permission to
      // switch a renderer or erase a shader validation error by retrying.
      const error = record.lifetime?.aborted ? record.lifetime.reason :
        new GpuCanvasError('GPU', 'The current canvas generation failed');
      throw operationError(record, error, true);
    }
    return record;
  }
  function operationError(record, error, terminal = false) {
    if (record.stopped) return record.reason;
    // Native loss observers run on the microtask queue. Let the actual lost
    // promise authorize recovery; a renderer merely throwing a similarly named
    // error is not a device-loss notification.
    if (record.lifetime?.aborted && record.lifetime.reason?.code === 'GPU_CANVAS_DEVICE_LOST')
      return record.lifetime.reason;
    if (terminal || record.session?.failed || record.session?.disposed) stopRecord(record, error);
    return error;
  }
  function invoke(method, ...args) {
    const record = ready();
    busy = true;
    try {
      record.session[method](...args);
      check(record);
      if (method === 'render') lastFrameRendered = record.session.lastFrameRendered;
      if (method === 'resize') { width = record.session.width; height = record.session.height; }
      return api;
    } catch (error) { throw operationError(record, error); }
    finally { busy = false; if (record.retiring) retire(record); }
  }
  async function waitFor(method) {
    const record = ready();
    if (method === 'prepare') record.preparing = true;
    try {
      await Promise.race([record.session[method](), record.stop.promise]);
      check(record);
      return api;
    } catch (error) { throw operationError(record, error, method === 'whenIdle'); }
    finally { if (method === 'prepare') record.preparing = false; }
  }
  function observeDevice(record, device) {
    check(record);
    if (!device || typeof device !== 'object' || !device.lost || typeof device.lost.then !== 'function')
      fail('RECOVERY_DEVICE', 'Recovery requires a device with a loss notification');
    if (devices.has(device)) fail('RECOVERY_DEVICE', 'A previous generation device cannot be reused');
    devices.add(device); record.device = device;
    Promise.resolve(device.lost).then(info => {
      // Owned devices are also destroyed after unrelated terminal errors. Do
      // not mislabel that deliberate teardown as a recoverable native loss.
      if (record.lifetime?.aborted && record.lifetime.reason?.code !== 'GPU_CANVAS_DEVICE_LOST') {
        stopRecord(record, record.lifetime.reason); return;
      }
      stopRecord(record, new GpuCanvasError('DEVICE_LOST', info?.message ?? 'Device lost'), info ?? {});
    }, error => stopRecord(record, error)).catch(error => {
      // A user-supplied device/renderer is not allowed to leave a detached
      // observer rejection or publish an otherwise invalid replacement.
      if (record === active && !disposed) terminate(error);
    });
  }
  async function open(suppliedDevice, recovering) {
    const record = {generation: generation + 1, attempt: attempts, suppliedDevice,
      abort: new AbortController(), stop: deferred(), loss: deferred(),
      stopped: false, retired: false, retiring: false, published: false, preparing: false,
      reason: null, session: null, device: null};
    active = record;
    state = recovering ? 'recovering' : 'initializing';
    lastFrameRendered = false;
    try {
      live();
      const construction = Promise.resolve(createSession(canvas, (device, attachments, context) => {
        record.lifetime = context.signal;
        observeDevice(record, device);
        return createRenderer(device, attachments, Object.freeze({...context,
          generation: record.generation, recoveryAttempt: record.attempt}));
      }, {...options, device: suppliedDevice, signal: record.abort.signal,
        target: {...options.target, width, height}})).then(session => {
        if (record.stopped || active !== record || disposed || fatal) {
          session.dispose();
          throw record.reason ?? fatal ?? new GpuCanvasError('DISPOSED', 'Reconstruction was cancelled');
        }
        record.session = session;
        return session;
      });
      await Promise.race([construction, record.stop.promise]);
      check(record);
      record.published = true;
      lastReady = record;
      generation = record.generation;
      width = record.session.width; height = record.session.height;
      rendererOptions = record.session.rendererOptions;
      state = 'ready'; lastError = null;
      return api;
    } catch (error) {
      const cause = record.stopped ? record.reason : error;
      stopRecord(record, cause);
      if (active === record && !disposed && !fatal) {
        lastError = cause;
        if (recovering) state = 'lost';
        else { fatal = cause; state = 'failed'; unlink(); }
      }
      throw cause;
    }
  }
  const api = Object.freeze({
    canvas,
    get device() { return active?.device ?? null; },
    get ownsDevice() { return active?.session?.ownsDevice ?? false; },
    get width() { return width; }, get height() { return height; },
    get rendererOptions() { return rendererOptions; },
    get suspended() { return state !== 'ready' || !width || !height; },
    get lastFrameRendered() { return lastFrameRendered; },
    get state() { return state; }, get generation() { return generation; },
    get recoveryAttempts() { return attempts; }, get maxRecoveryAttempts() { return maxRecoveryAttempts; },
    get recoverable() { return state === 'lost' && !fatal && attempts < maxRecoveryAttempts; },
    get lastLoss() { return lastLoss; }, get lastError() { return lastError; },
    get disposed() { return disposed; },
    get failed() { return state === 'lost' || state === 'failed' || !!active?.session?.failed; },
    get diagnostics() { return Object.freeze({state, generation, recoveryAttempts: attempts,
      maxRecoveryAttempts, lastLoss, session: active?.session?.diagnostics ?? null}); },
    render(input, frame) { return invoke('render', input, frame); },
    resize(w, h) {
      live();
      if (busy || active?.preparing && state === 'ready') fail('REENTRANT', 'Resize cannot overlap another operation');
      dimension(w); dimension(h);
      if (state === 'lost') { width = w; height = h; return api; }
      return invoke('resize', w, h);
    },
    setSize(w, h, pixelRatio = 1) {
      dimension(w); dimension(h);
      if (typeof pixelRatio !== 'number' || !Number.isFinite(pixelRatio) || pixelRatio <= 0)
        fail('SIZE', 'Expected a finite positive pixel ratio');
      return api.resize(Math.floor(w * pixelRatio), Math.floor(h * pixelRatio));
    },
    prepare() { return waitFor('prepare'); },
    whenIdle() { return waitFor('whenIdle'); },
    whenLost() {
      try { live(); return (lastReady ?? active).loss.promise; }
      catch (error) { return Promise.reject(error); }
    },
    async recover(settings = {}) {
      live();
      if (busy) fail('REENTRANT', 'Recovery cannot start inside a canvas operation');
      if (state !== 'lost') fail('RECOVERY_STATE', 'Recovery requires an observed device loss and no concurrent recovery');
      // Lock around input getters, before any capability requests or counters.
      busy = true;
      let device;
      try {
        const selected = {...object(settings, 'recovery options')};
        if (Object.keys(selected).some(key => key !== 'device')) fail('OPTIONS', 'Only replacement device is a recovery option');
        if (attempts >= maxRecoveryAttempts) fail('RECOVERY_LIMIT', 'Canvas recovery attempt budget exhausted');
        if (!Object.hasOwn(selected, 'device') && lastReady.suppliedDevice !== null)
          fail('RECOVERY_DEVICE_REQUIRED', 'Supply a fresh borrowed device, or device:null to request an owned device');
        device = Object.hasOwn(selected, 'device') ? selected.device : null;
        if (device !== null && (!device || typeof device !== 'object' || devices.has(device)))
          fail('RECOVERY_DEVICE', 'Expected a fresh replacement device or explicit null');
        live();
      } finally { busy = false; }
      attempts++;
      return open(device, true);
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose inside a synchronous canvas operation');
      if (disposed) return;
      disposed = true; state = 'disposed'; lastFrameRendered = false;
      unlink(); stopRecord(active, new GpuCanvasError('DISPOSED', 'Recoverable canvas is disposed'));
    },
  });
  signal?.addEventListener('abort', abort, {once: true});
  if (signal?.aborted) abort();
  return open(options.device ?? null, false);
}
