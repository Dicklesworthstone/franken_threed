/** Public renderer transfers share its render/readback submission FIFO.
 * Upload bytes/options and source target generations are captured at the call.
 * Copies capture GPU content at their ordered submission, without CPU readback.
 * Completion includes native validation and the submitted-work fence, but never
 * blocks the FIFO on GPU completion. Cancellation cannot undo submitted writes.
 * The controller also installs the source initRenderTarget/copyTextureToTexture
 * methods on this renderer instance (never on a global/prototype). Source void
 * calls report asynchronous failures at the next renderer boundary; the Async
 * extensions return the same completion promises. Both use this one owner/FIFO.
 * Explicit mip/depth readback extensions also use this FIFO and request limit;
 * native readback owns the shared per-device staging/output byte budget. Pending
 * reads propagate renderer, target and caller cancellation to native mapping.
 * Target-owned DepthTextures use the same source copy APIs. The depth extension
 * preserves color/stencil and requires explicit consent for format conversion.
 * This is general WebGPU execution, not shader/pixel equivalence evidence.
 */
import {inspectGpuTargetWrite, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {inspectGpuDepthCopy} from './gpu_depth_transfer.mjs';
import {assertThreeNativeTextureWrite} from './three_node_shadows.mjs';

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
  function transferEntry(target) {
    const entry = renderer._targets.capture(target);
    if (entry.shape.cube) fail('UNSUPPORTED', 'Cube target transfers require explicit face-aware support');
    return entry;
  }
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
  function schedule(entries, signal, bytes, makeAction, readback = false) {
    live();
    const signals = [...new Set([renderer._targetLifetime.signal, ...entries.map(entry => entry.lifetime.signal), signal].filter(Boolean))];
    for (const current of signals) if (current.aborted) throw stopped(current);
    if (active.size >= maxRequests || bytes > maxBytes - allocatedBytes)
      fail('LIMIT', 'Pending renderer transfers exceed their shared request or byte budget');
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    const cancellation = readback ? new AbortController() : null;
    const state = {signals, action: null, done: false, promise, reject: null};
    const finish = (error, result, failed = false) => {
      if (state.done) return;
      state.done = true; state.action = null;
      active.delete(state); allocatedBytes -= bytes; unwatch(state);
      if (failed) reject(error); else resolve(result);
    };
    state.reject = error => {
      if (state.done) return;
      cancellation?.abort(error);
      finish(error, undefined, true);
    };
    active.add(state); allocatedBytes += bytes;
    // Observe internal failures even when callers use fire-and-forget source
    // wrappers. The returned promise still rejects for an awaiting caller.
    promise.catch(() => {});
    try {
      watch(state);
      if (state.done) return promise;
      state.action = makeAction(cancellation?.signal);
      makeAction = null;
      const request = {
        transfer: true,
        issue() {
          if (state.done) return undefined;
          live();
          for (const entry of entries) renderer._targets.assertCurrent(entry);
          const action = state.action;
          state.action = null;
          // Native readback submits before returning its mapping promise. No
          // await here: the next render/copy can submit while mapping is pending.
          const result = action();
          if (result && typeof result.then === 'function') {
            Promise.resolve(result).catch(() => {});
            if (!readback) fail('TRANSFER', 'Target transfer submission must be synchronous');
          }
          const completion = Promise.all([result, ...entries.flatMap(entry => [entry.validation, entry.storage?.whenIdle()])]);
          return completion.then(([value]) => value);
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
    // Renderer mixins can delegate source-shaped target copies to this same
    // owner even when their prototype method prevents instance installation.
    copyTexture(...args) { return copyTexture(...args); },
    writePixels(target, data, supplied = {}) {
      try {
        live();
        assertThreeNativeTextureWrite(renderer, target?.texture);
        const {value, signal} = options(supplied), entry = transferEntry(target);
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
    copyDepth(source, destination, supplied = {}) {
      try {
        live();
        assertThreeNativeTextureWrite(renderer, destination?.depthTexture);
        const {value, signal} = options(supplied);
        const from = transferEntry(source), to = transferEntry(destination);
        if (from === to) fail('TRANSFER', 'In-place render target copies are not admitted');
        const plan = inspectGpuDepthCopy({...from.shape, sampleDepth: from.shape.depthTexture !== null}, to.shape, value);
        const copyOptions = Object.freeze({...value, width: plan.width, height: plan.height});
        return schedule([from, to], signal, 0, () => () => renderer._targets.copyDepth(from, to, copyOptions));
      } catch (error) { return Promise.reject(error); }
    },
    copyColor(source, destination, supplied = {}) {
      try {
        live();
        assertThreeNativeTextureWrite(renderer, destination?.texture);
        const {value, signal} = options(supplied);
        const from = transferEntry(source), to = transferEntry(destination);
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
  function textureEntry(texture, depth = false) {
    const target = texture?.renderTarget;
    // Source DepthTexture has its own ownership pointer, not the color-only
    // isRenderTargetTexture flag. Never mutate a source texture to admit it.
    if (texture?.isTexture !== true || !target || (depth
      ? texture.isDepthTexture !== true || target.depthTexture !== texture
      : texture.isRenderTargetTexture !== true || target.texture !== texture))
      fail('UNSUPPORTED', 'Texture copies require matching target-owned 2D color or depth textures');
    return transferEntry(target);
  }
  function initialize(target) {
    live();
    const entry = renderer._targets.capture(target);
    return schedule([entry], undefined, 0, () => () => { renderer._targets.ensure(entry); });
  }
  function copyTexture(source, destination, sourceRegion = null, destinationPosition = null,
    sourceLevel = 0, destinationLevel = 0) {
    live();
    assertThreeNativeTextureWrite(renderer, destination);
    if (destinationLevel !== 0)
      fail('UNSUPPORTED', 'Texture copies write destination mip zero and regenerate its derived mips');
    if (sourceRegion !== null && (!sourceRegion.min || !sourceRegion.max ||
        (sourceRegion.isBox3 && (sourceRegion.min.z !== 0 || sourceRegion.max.z !== 1))))
      fail('UNSUPPORTED', 'Expected a 2D source region or a single depth-zero slice');
    if (destinationPosition !== null && destinationPosition.z !== undefined && destinationPosition.z !== 0)
      fail('UNSUPPORTED', 'Texture copies require destination depth zero');
    const depth = source?.isDepthTexture === true || destination?.isDepthTexture === true;
    if (depth && sourceLevel !== 0) fail('UNSUPPORTED', 'Depth textures have only mip level zero');
    const from = textureEntry(source, depth), to = textureEntry(destination, depth);
    if (from === to) fail('TRANSFER', 'In-place render target copies are not admitted');
    const value = {...(depth ? {} : {sourceMipLevel: sourceLevel}), flipY: renderer._bottomLeftOrigin === true,
      sourceX: sourceRegion === null ? 0 : sourceRegion.min.x,
      sourceY: sourceRegion === null ? 0 : sourceRegion.min.y,
      x: destinationPosition === null ? 0 : destinationPosition.x,
      y: destinationPosition === null ? 0 : destinationPosition.y,
      ...(sourceRegion === null ? {} : {width: sourceRegion.max.x - sourceRegion.min.x,
        height: sourceRegion.max.y - sourceRegion.min.y})};
    // Validate before the void source call returns. Copy only scalar options;
    // later edits to a caller's Box2/Vector2 must not retarget a queued copy.
    if (depth) {
      const plan = inspectGpuDepthCopy({...from.shape, sampleDepth: from.shape.depthTexture !== null}, to.shape, value);
      return api.copyDepth(source.renderTarget, destination.renderTarget,
        {...value, width: plan.width, height: plan.height});
    }
    const plan = inspectGpuTargetCopy(from.shape, to.shape, value);
    return api.copyColor(source.renderTarget, destination.renderTarget,
      {...value, width: plan.size.width, height: plan.size.height});
  }
  function readPixels(target, supplied = {}, mipLevel = 0, depth = false) {
    live();
    const {value, signal} = options(supplied), entry = transferEntry(target);
    const allowed = ['x', 'y', 'width', 'height', 'flipY', 'output', 'maxBytes', 'maxInFlightBytes',
      ...(depth ? ['sampleIndex'] : [])];
    for (const name of Object.keys(value))
      if (!allowed.includes(name)) fail('OPTIONS', `Unknown renderer readback option: ${name}`);
    if (depth && !entry.shape.depthTexture)
      fail('UNSUPPORTED', 'Attach a source DepthTexture before requesting depth readback');
    const levels = depth ? 1 : entry.shape.mipLevelCount;
    if (!Number.isSafeInteger(mipLevel) || mipLevel < 0 || mipLevel >= levels)
      fail('REGION', 'Readback mip level is not allocated');
    const w = Math.max(1, Math.floor(entry.shape.width / 2 ** mipLevel));
    const h = Math.max(1, Math.floor(entry.shape.height / 2 ** mipLevel));
    const {x = 0, y = 0, width = w - x, height = h - y} = value;
    if (![x, y, width, height].every(n => Number.isSafeInteger(n) && n >= 0) || x > w - width || y > h - height)
      fail('REGION', 'Readback bounds must be non-negative integers inside the selected mip');
    if (typeof value.flipY !== 'boolean' || ['maxBytes', 'maxInFlightBytes'].some(name =>
      value[name] !== undefined && (!Number.isSafeInteger(value[name]) || value[name] < 1)))
      fail('OPTIONS', 'Invalid renderer readback orientation or byte budget');
    if (depth) {
      const sampleIndex = value.sampleIndex === undefined && entry.shape.sampleCount === 1 ? 0 : value.sampleIndex;
      if (!Number.isSafeInteger(sampleIndex) || sampleIndex < 0 || sampleIndex >= entry.shape.sampleCount)
        fail('UNSUPPORTED', 'Depth readback requires an explicit in-range sampleIndex for multisampled targets');
    }
    const captured = Object.freeze({...value, x, y, width, height, mipLevel});
    return schedule([entry], signal, 0, combinedSignal => () => {
      renderer._targets.ensure(entry);
      return depth ? entry.storage.readDepthPixels({...captured, signal: combinedSignal})
        : entry.storage.readPixels({...captured, signal: combinedSignal});
    }, true);
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
    copyRenderTargetDepthPixelsAsync: api.copyDepth,
    readRenderTargetMipPixelsAsync: promised((target, mipLevel = 0, supplied = {}) => readPixels(target, supplied, mipLevel)),
    readRenderTargetDepthPixelsAsync: promised((target, supplied = {}) => readPixels(target, supplied, 0, true)),
  };
  for (const [name, method] of Object.entries(methods))
    if (typeof renderer[name] !== 'function') renderer[name] = method;
  return api;
}
