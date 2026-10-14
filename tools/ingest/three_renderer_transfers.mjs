/** Public renderer transfers share its render/readback submission FIFO.
 * Upload bytes/options and source target generations are captured at the call.
 * Copies capture GPU content at their ordered submission, without CPU readback.
 * Completion includes native validation and the submitted-work fence, but never
 * blocks the FIFO on GPU completion. Cancellation cannot undo submitted writes.
 * The controller also installs the source initRenderTarget/copyTextureToTexture
 * methods on this renderer instance (never on a global/prototype). Source void
 * calls report asynchronous failures at the next renderer boundary; the Async
 * extensions return the same completion promises. Both use this one owner/FIFO.
 * This is general WebGPU execution, not shader/pixel equivalence evidence.
 */
import {inspectGpuTargetWrite, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';

export function createThreeRendererTransfers(renderer, ErrorType, {
  maxBytes = 128 * 1024 * 1024, maxRequests = 64,
} = {}) {
  const fail = (code, message) => { throw new ErrorType(code, message); };
  if (![maxBytes, maxRequests].every(n => Number.isSafeInteger(n) && n > 0))
    fail('OPTIONS', 'Expected positive renderer transfer byte and request limits');
  const active = new Set(), watchers = new Map();
  let allocatedBytes = 0;
  const live = () => {
    if (renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
    if (renderer._deferredError) throw renderer._deferredError;
  };
  function options(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value) ||
        ![Object.prototype, null].includes(Object.getPrototypeOf(value)))
      fail('OPTIONS', 'Expected plain renderer transfer options');
    const copy = Object.create(null);
    let signal;
    for (const key of Reflect.ownKeys(value)) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (typeof key !== 'string' || !Object.hasOwn(descriptor, 'value'))
        fail('OPTIONS', 'Renderer transfer options cannot contain symbols or accessors');
      if (key === 'signal') signal = descriptor.value;
      else copy[key] = descriptor.value;
    }
    if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' ||
        typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function'))
      fail('OPTIONS', 'Expected AbortSignal');
    if (copy.flipY === undefined) copy.flipY = renderer._bottomLeftOrigin === true;
    return {value: copy, signal};
  }
  const stopped = signal => signal.reason ?? new ErrorType('ABORTED', 'Renderer transfer cancelled');
  function unwatch(state) {
    for (const signal of state.signals) {
      const watcher = watchers.get(signal);
      if (watcher?.states.delete(state) && watcher.states.size === 0) {
        signal.removeEventListener('abort', watcher.abort);
        watchers.delete(signal);
      }
    }
  }
  function watch(state) {
    for (const signal of state.signals) {
      let watcher = watchers.get(signal);
      if (!watcher) {
        watcher = {states: new Set(), abort: null};
        watcher.abort = () => {
          const error = stopped(signal);
          for (const pending of [...watcher.states]) pending.reject(error);
        };
        watchers.set(signal, watcher);
        signal.addEventListener('abort', watcher.abort, {once: true});
      }
      watcher.states.add(state);
      if (signal.aborted) { state.reject(stopped(signal)); break; }
    }
  }
  function schedule(entries, signal, bytes, makeAction) {
    live();
    const signals = [...new Set([renderer._targetLifetime.signal, ...entries.map(entry => entry.lifetime.signal), signal].filter(Boolean))];
    for (const current of signals) if (current.aborted) throw stopped(current);
    if (active.size >= maxRequests || bytes > maxBytes - allocatedBytes)
      fail('LIMIT', 'Pending renderer transfers exceed their shared request or byte budget');
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    const state = {signals, action: null, done: false, promise, reject: null};
    const finish = (error, result, failed = false) => {
      if (state.done) return;
      state.done = true; state.action = null;
      active.delete(state); allocatedBytes -= bytes; unwatch(state);
      if (failed) reject(error); else resolve(result);
    };
    state.reject = error => finish(error, undefined, true);
    active.add(state); allocatedBytes += bytes;
    // Observe internal failures even when callers use fire-and-forget source
    // wrappers. The returned promise still rejects for an awaiting caller.
    promise.catch(() => {});
    try {
      watch(state);
      if (state.done) return promise;
      state.action = makeAction();
      makeAction = null;
      const request = {
        transfer: true,
        issue() {
          if (state.done) return undefined;
          live();
          for (const entry of entries) renderer._targets.assertCurrent(entry);
          const action = state.action;
          state.action = null;
          // Must submit synchronously. Only native completion may cross await.
          const result = action();
          if (result && typeof result.then === 'function') {
            Promise.resolve(result).catch(() => {});
            fail('TRANSFER', 'Target transfer submission must be synchronous');
          }
          const completion = Promise.all(entries.flatMap(entry => [entry.validation, entry.storage?.whenIdle()]));
          return completion.then(() => result);
        },
        resolve(result) { Promise.resolve(result).then(value => finish(null, value), state.reject); },
        reject: state.reject,
      };
      // Same fast path as readback: no await is allowed before submission.
      if (renderer._initialized && !renderer._drain && !renderer._preparing && renderer._pending.length === 0)
        request.resolve(request.issue());
      else renderer._enqueue(request);
    } catch (error) { state.reject(error); }
    finally { makeAction = null; }
    return promise;
  }
  const api = Object.freeze({
    get pendingRequests() { return active.size; },
    get allocatedBytes() { return allocatedBytes; },
    async whenIdle() { await Promise.all([...active].map(state => state.promise)); },
    writePixels(target, data, supplied = {}) {
      try {
        live();
        const {value, signal} = options(supplied), entry = renderer._targets.capture(target);
        const plan = inspectGpuTargetWrite(entry.shape, data, value);
        // Pick built-in constructors, not a caller-controlled data.constructor.
        const Type = data instanceof Float32Array ? Float32Array : data instanceof Uint16Array ? Uint16Array : Uint8Array;
        const uploadOptions = Object.freeze({...value, width: plan.size.width, height: plan.size.height,
          offset: 0, bytesPerRow: plan.rowBytes});
        return schedule([entry], signal, plan.packedBytes, () => {
          // Copy bytes rather than float values (preserves NaN/half-float bits).
          // Strip input row padding; never retain the caller's unused tail.
          const packed = new Uint8Array(plan.packedBytes);
          if (!plan.empty) for (let row = 0; row < plan.size.height; row++)
            packed.set(plan.data.subarray(row * plan.bytesPerRow, row * plan.bytesPerRow + plan.rowBytes), row * plan.rowBytes);
          plan.data = null;
          const snapshot = new Type(packed.buffer);
          return () => renderer._targets.writePixels(entry, snapshot, uploadOptions);
        });
      } catch (error) { return Promise.reject(error); }
    },
    copyColor(source, destination, supplied = {}) {
      try {
        live();
        const {value, signal} = options(supplied);
        const from = renderer._targets.capture(source), to = renderer._targets.capture(destination);
        if (from === to) fail('TRANSFER', 'In-place render target copies are not admitted');
        const plan = inspectGpuTargetCopy(from.shape, to.shape, value);
        const copyOptions = Object.freeze({...value, width: plan.size.width, height: plan.size.height,
          sourceMipLevel: plan.sourceMipLevel});
        return schedule([from, to], signal, 0, () => () => renderer._targets.copyColor(from, to, copyOptions));
      } catch (error) { return Promise.reject(error); }
    },
  });
  // Keep source APIs next to the transfer owner, not a second submission path.
  // The renderer constructor creates this controller after its target owner and
  // FIFO. WebGL chooses its bottom-left origin later in the subclass constructor,
  // so sample that flag at the call, never when installing these methods.
  function textureEntry(texture) {
    if (texture?.isTexture !== true || texture.isRenderTargetTexture !== true ||
        !texture.renderTarget || texture.renderTarget.texture !== texture)
      fail('UNSUPPORTED', 'Texture copies require target-owned 2D color textures');
    return renderer._targets.capture(texture.renderTarget);
  }
  function initialize(target) {
    live();
    const entry = renderer._targets.capture(target);
    return schedule([entry], undefined, 0, () => () => { renderer._targets.ensure(entry); });
  }
  function copyTexture(source, destination, sourceRegion = null, destinationPosition = null,
    sourceLevel = 0, destinationLevel = 0) {
    live();
    if (destinationLevel !== 0)
      fail('UNSUPPORTED', 'Texture copies write destination mip zero and regenerate its derived mips');
    if (sourceRegion !== null && (!sourceRegion.min || !sourceRegion.max ||
        (sourceRegion.isBox3 && (sourceRegion.min.z !== 0 || sourceRegion.max.z !== 1))))
      fail('UNSUPPORTED', 'Expected a 2D source region or a single depth-zero slice');
    if (destinationPosition !== null && destinationPosition.z !== undefined && destinationPosition.z !== 0)
      fail('UNSUPPORTED', 'Texture copies require destination depth zero');
    const from = textureEntry(source), to = textureEntry(destination);
    if (from === to) fail('TRANSFER', 'In-place render target copies are not admitted');
    const value = {sourceMipLevel: sourceLevel, flipY: renderer._bottomLeftOrigin === true,
      sourceX: sourceRegion === null ? 0 : sourceRegion.min.x,
      sourceY: sourceRegion === null ? 0 : sourceRegion.min.y,
      x: destinationPosition === null ? 0 : destinationPosition.x,
      y: destinationPosition === null ? 0 : destinationPosition.y,
      ...(sourceRegion === null ? {} : {width: sourceRegion.max.x - sourceRegion.min.x,
        height: sourceRegion.max.y - sourceRegion.min.y})};
    // Validate before the void source call returns. Copy only scalar options;
    // later edits to a caller's Box2/Vector2 must not retarget a queued copy.
    const plan = inspectGpuTargetCopy(from.shape, to.shape, value);
    return api.copyColor(source.renderTarget, destination.renderTarget,
      {...value, width: plan.size.width, height: plan.size.height});
  }
  const deferred = error => { if (!renderer._disposed) renderer._deferredError ??= error; };
  const promised = operation => (...args) => {
    try { return operation(...args); } catch (error) { return Promise.reject(error); }
  };
  // Preserve explicit subclass implementations. r186 WebGLRenderer likewise
  // exposes its source transfer functions as instance methods.
  const methods = {
    initRenderTarget(target) { initialize(target).catch(deferred); },
    initRenderTargetAsync: promised(initialize),
    copyTextureToTexture(...args) { copyTexture(...args).catch(deferred); },
    copyTextureToTextureAsync: promised(copyTexture),
  };
  for (const [name, method] of Object.entries(methods))
    if (typeof renderer[name] !== 'function') renderer[name] = method;
  return api;
}
