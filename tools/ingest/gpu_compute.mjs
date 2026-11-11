/** Owned WebGPU compute execution. Shader preparation is supplied by the caller;
 * this owner creates pipelines, keeps storage GPU-resident across dispatches,
 * submits real compute passes and performs asynchronous staging-buffer readback.
 * Repeated dispatches reuse bounded, pipeline-local bind groups by native
 * resource identity and buffer range, not by caller descriptor identity or
 * upload version. External/unknown texture resources are never cached.
 * General WebGPU execution, not a Rust specialization or a speedup claim. */
import {createComputeAttributeOwner} from './gpu_compute_attributes.mjs';
import {subscribeGpuDeviceLoss} from './gpu_device_loss.mjs';

export class GpuComputeError extends Error {
  constructor(code, message) {
    super(`GPU_COMPUTE_${code}: ${message}`);
    this.name = 'GpuComputeError'; this.code = `GPU_COMPUTE_${code}`;
  }
}
const fail = (code, message) => { throw new GpuComputeError(code, message); };
// WebGPU constants are also usable in command-recording Node tests.
const COPY_SRC = 4, COPY_DST = 8, INDEX = 16, VERTEX = 32, UNIFORM = 64, STORAGE = 128, INDIRECT = 256, MAP_READ = 1;
const positive = n => Number.isSafeInteger(n) && n > 0;
const readbackReservations = new WeakMap(), capturedReadbacks = new WeakSet();

/** Reserve a source readback handle before a deferred FIFO submission. A
 * release/dispose before submission cancels that request, and cannot let its
 * eventual cleanup reset a newer use of the same public handle. */
export function captureGpuReadback(target = null, offset = 0, count = -1) {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset % 4 ||
      count !== -1 && (!Number.isSafeInteger(count) || count < 0 || count % 4))
    fail('READBACK_RANGE', 'Readback offset and count must be nonnegative byte multiples of four, or count -1 for the remainder');
  const reusable = target?.isReadbackBuffer === true;
  if (target !== null && !reusable && !(target instanceof ArrayBuffer))
    fail('READBACK_TARGET', 'Readback target must be an ArrayBuffer or source ReadbackBuffer');
  if (reusable && (!Number.isSafeInteger(target.maxByteLength) || target.maxByteLength < 0 ||
      typeof target.addEventListener !== 'function' || typeof target.removeEventListener !== 'function'))
    fail('READBACK_TARGET', 'ReadbackBuffer requires a nonnegative byte capacity and source release/dispose events');
  if (reusable && target._mapped === true)
    fail('READBACK_MAPPED', 'ReadbackBuffer must be released before being used again');
  let invalid = null, finished = false, rejectCancellation;
  const cancellation = reusable ? new Promise((_, reject) => { rejectCancellation = reject; }) : null;
  cancellation?.catch(() => {});
  const request = {target, offset, count, reusable, capacity: reusable ? target.maxByteLength : null,
    check(bytes = 0) {
      if (invalid) throw invalid;
      if (finished) fail('READBACK_STALE', 'Readback request has already completed');
      if (target instanceof ArrayBuffer) {
        let view;
        try { view = new Uint8Array(target); } catch { fail('READBACK_TARGET', 'Readback target ArrayBuffer is detached'); }
        if (view.byteLength < bytes) fail('READBACK_TARGET', 'Readback target is smaller than the requested byte range');
      }
    },
    wait(promise) { return cancellation ? Promise.race([promise, cancellation]) : promise; },
    cancel(error) {
      if (invalid) return;
      invalid = error;
      if (reusable && readbackReservations.get(target) === request) {
        readbackReservations.delete(target); target.buffer = null; target._mapped = false;
      }
      request.onCancel?.(error);
      rejectCancellation?.(error);
    },
    finish(success) {
      if (finished) return;
      if (!success) request.cancel(new GpuComputeError('READBACK_CANCELLED', 'Readback did not complete'));
      finished = true;
      if (reusable) {
        target.removeEventListener('release', onRelease); target.removeEventListener('dispose', onDispose);
        if (readbackReservations.get(target) === request) readbackReservations.delete(target);
      }
    }};
  const onRelease = () => request.cancel(new GpuComputeError('READBACK_RELEASED', 'ReadbackBuffer was released before completion'));
  const onDispose = () => request.cancel(new GpuComputeError('READBACK_DISPOSED', 'ReadbackBuffer was disposed before completion'));
  request.check(count < 0 ? 0 : count);
  if (reusable) {
    target._mapped = true;
    readbackReservations.set(target, request);
    target.addEventListener('release', onRelease); target.addEventListener('dispose', onDispose);
  }
  capturedReadbacks.add(request);
  return request;
}

export function createGpuCompute(device, {maxBytes = 256 * 1024 * 1024, maxBuffers = 1024,
  maxPipelines = 256, maxCachedBindGroups = 256, signal = null, onReadbackChange = null} = {}) {
  if (!device?.queue || typeof device.createBuffer !== 'function' ||
      ![maxBytes, maxBuffers, maxPipelines].every(positive) ||
      !Number.isSafeInteger(maxCachedBindGroups) || maxCachedBindGroups < 0)
    fail('OPTIONS', 'Expected a GPUDevice, positive resource limits and a nonnegative bind-group cache limit');
  const buffers = new Map(), pipelines = new Map(), compiled = new Map(), ready = new Set(), staging = new Set(), pending = new Set();
  const readbacks = new Map();
  const bindGroups = new Map();
  let bindingIds = new WeakMap(), bindingUsers = new WeakMap(), bindingLayouts = new WeakMap(), nextBindingId = 0;
  const lifetime = new AbortController();
  const attributeOwner = createComputeAttributeOwner(device);
  let terminal = null, allocatedBytes = 0, readbackBytes = 0, releaseDeviceLoss = null;
  const live = () => { if (terminal) throw terminal; };
  function stop(error) {
    if (terminal) return;
    terminal = error instanceof Error ? error : new GpuComputeError('ABORTED', String(error ?? 'Compute execution aborted'));
    releaseDeviceLoss?.(); releaseDeviceLoss = null;
    lifetime.abort(terminal);
    attributeOwner.dispose();
    bindGroups.clear();
    bindingIds = new WeakMap(); bindingUsers = new WeakMap(); bindingLayouts = new WeakMap();
    for (const entry of [...readbacks.values()]) disposeReadback(entry, terminal);
    for (const entry of buffers.values()) entry.buffer.destroy();
    for (const buffer of staging) buffer.destroy();
    buffers.clear(); staging.clear(); ready.clear(); pipelines.clear(); compiled.clear(); allocatedBytes = 0;
    signal?.removeEventListener('abort', onAbort);
  }
  const onAbort = () => stop(signal.reason);
  if (signal?.aborted) onAbort(); else signal?.addEventListener('abort', onAbort, {once: true});
  // A borrowed device is never destroyed by this owner.
  if (!terminal && typeof device.lost?.then === 'function')
    releaseDeviceLoss = subscribeGpuDeviceLoss(device.lost,
      info => stop(new GpuComputeError('DEVICE_LOST', info?.message ?? 'WebGPU device lost')), stop);
  function interrupted(promise) {
    live();
    return new Promise((resolve, reject) => {
      const abort = () => reject(terminal);
      lifetime.signal.addEventListener('abort', abort, {once: true});
      Promise.resolve(promise).then(resolve, reject).finally(() => lifetime.signal.removeEventListener('abort', abort));
    });
  }
  function watch(promise) {
    const tracked = Promise.resolve(promise).catch(error => { stop(error); throw error; });
    pending.add(tracked);
    tracked.finally(() => pending.delete(tracked)).catch(() => {});
    return tracked;
  }
  // Never leave a device error scope installed across an await: the same device
  // is shared with rendering, uploads and other asynchronous pipeline builds.
  function checked(run, validations = null) {
    live();
    device.pushErrorScope?.('validation');
    try { return run(); }
    finally {
      if (device.popErrorScope) {
        const validation = watch(device.popErrorScope().then(error => {
          if (error) throw new GpuComputeError('VALIDATION', error.message);
        }));
        validations?.push(validation);
      }
    }
  }
  function budget(bytes) {
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > maxBytes - allocatedBytes) fail('BUDGET', 'Compute buffers exceed their shared byte budget');
  }
  function upload(key, array, version = 0, {uniform = false, ranges = [], admission = null} = {}) {
    live();
    if (!key || !['object', 'function'].includes(typeof key) || !ArrayBuffer.isView(array) || array instanceof DataView)
      fail('BUFFER', 'Expected an identity key and typed array');
    if (!uniform) attributeOwner.admit(key, array, version, admission);
    const bytes = array.byteLength, size = Math.max(4, Math.ceil(bytes / 4) * 4);
    const limit = uniform ? device.limits?.maxUniformBufferBindingSize : device.limits?.maxStorageBufferBindingSize;
    if (size > (device.limits?.maxBufferSize ?? Infinity) || size > (limit ?? Infinity))
      fail('LIMIT', 'Compute buffer exceeds device binding limits');
    const old = buffers.get(key), replacement = !old || old.size !== size || old.uniform !== uniform;
    if (old && !replacement && old.array === array && old.version === version) return old.buffer;
    // Validate the complete range list before allocating or issuing any writes.
    const partial = !replacement && ranges.length > 0;
    const writes = partial ? ranges.map(({start, count}) => {
      const offset = start * array.BYTES_PER_ELEMENT, length = count * array.BYTES_PER_ELEMENT;
      if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count < 0 ||
          !Number.isSafeInteger(offset + length) || offset + length > bytes || offset % 4 || length % 4)
        fail('RANGE', 'Storage update ranges must be in bounds and aligned to four bytes');
      return [offset, length];
    }) : [[0, size]];
    let entry = old;
    if (replacement) {
      if (!old && buffers.size >= maxBuffers) fail('BUDGET', 'Compute buffer count exceeds maxBuffers');
      // Charge both generations until the replacement is successfully created.
      budget(size);
      const buffer = checked(() => device.createBuffer({label: 'f3d-compute-buffer', size,
        usage: COPY_SRC | COPY_DST | (uniform ? UNIFORM : STORAGE | INDIRECT | INDEX | VERTEX)}));
      entry = {buffer, size, bytes, uniform, array: null, version: null};
      allocatedBytes += size; buffers.set(key, entry);
      if (old) { invalidateBindings(old.buffer); old.buffer.destroy(); allocatedBytes -= old.size; }
    }
    const source = new Uint8Array(array.buffer, array.byteOffset, bytes);
    checked(() => {
      for (const [offset, length] of writes) {
        if (!length) continue;
        if (length > bytes) {
          const padded = new Uint8Array(length); padded.set(source);
          device.queue.writeBuffer(entry.buffer, offset, padded);
        } else device.queue.writeBuffer(entry.buffer, offset, source, offset, length);
      }
    });
    entry.array = array; entry.version = version; entry.bytes = bytes;
    if (!uniform) attributeOwner.publish(key, entry.buffer, array, version, live, whenIdle, admission);
    else attributeOwner.release(key);
    return entry.buffer;
  }
  function release(key) {
    const entry = buffers.get(key);
    if (!entry) return;
    attributeOwner.release(key);
    invalidateBindings(entry.buffer);
    buffers.delete(key); entry.buffer.destroy(); allocatedBytes -= entry.size;
  }
  function bindingId(resource) {
    let id = bindingIds.get(resource);
    if (id === undefined) { id = ++nextBindingId; bindingIds.set(resource, id); }
    return id;
  }
  function evictBindings(key) {
    const record = bindGroups.get(key);
    if (!record) return;
    bindGroups.delete(key);
    for (const buffer of record.buffers) {
      const users = bindingUsers.get(buffer);
      users.delete(key);
      if (!users.size) bindingUsers.delete(buffer);
    }
  }
  function invalidateBindings(buffer) {
    for (const key of bindingUsers.get(buffer) ?? []) evictBindings(key);
  }
  function captureGroups(groups) {
    const groupLimit = device.limits?.maxBindGroups ?? 4;
    const bindingLimit = device.limits?.maxBindingsPerBindGroup ?? 1000;
    if (!Array.isArray(groups) || groups.length > groupLimit)
      fail('BINDINGS', 'Expected bounded compute bind groups');
    const indices = new Set();
    // Complete admission precedes any GPU allocation. Copy buffer range
    // descriptors so later host edits cannot change a cached group's meaning.
    return Array.from(groups, group => {
      const {index, entries} = group ?? {};
      if (!Number.isSafeInteger(index) || index < 0 || index >= groupLimit || indices.has(index) ||
          !Array.isArray(entries) || entries.length > bindingLimit)
        fail('BINDINGS', 'Expected distinct in-range group indices and bounded entries');
      indices.add(index);
      const slots = new Set(), resources = new Set(), keys = [];
      let cacheable = maxCachedBindGroups > 0;
      const captured = Array.from(entries, entry => {
        const {binding, resource} = entry ?? {};
        if (!Number.isSafeInteger(binding) || binding < 0 || binding >= bindingLimit || slots.has(binding) ||
            !resource || typeof resource !== 'object')
          fail('BINDINGS', 'Expected distinct in-range bindings and resource objects');
        slots.add(binding);
        // Native interface resources take precedence over dictionary members:
        // an application may attach a harmless `buffer` property to a view.
        const brand = Object.prototype.toString.call(resource);
        if (brand === '[object Object]' && 'buffer' in resource) {
          const {buffer, offset = 0, size} = resource;
          if (!buffer || typeof buffer !== 'object' || !Number.isSafeInteger(offset) || offset < 0 ||
              size !== undefined && (!Number.isSafeInteger(size) || size < 1))
            fail('BINDINGS', 'Expected a native buffer and nonnegative offset with optional positive size');
          resources.add(buffer);
          keys.push([binding, 'buffer', bindingId(buffer), offset, size ?? null]);
          return {binding, resource: {buffer, offset, ...(size === undefined ? {} : {size})}};
        }
        // GPUExternalTexture expires independently of object identity. Unknown
        // resources retain the uncached path rather than guessing their lifetime.
        if (brand !== '[object GPUTextureView]' && brand !== '[object GPUSampler]') cacheable = false;
        keys.push([binding, brand, bindingId(resource)]);
        return {binding, resource};
      });
      keys.sort((a, b) => a[0] - b[0]);
      return {index, entries: captured, resources, key: cacheable ? JSON.stringify(keys) : null};
    });
  }
  function resolveGroup(pipeline, snapshot) {
    const {index, entries, resources} = snapshot;
    // Auto layouts are exclusive to their pipeline, even for identical WGSL
    // binding declarations. Never share groups across pipeline identities.
    const key = snapshot.key === null ? null : `${bindingId(pipeline)}/${index}/${snapshot.key}`;
    const existing = key === null ? null : bindGroups.get(key);
    if (existing) {
      bindGroups.delete(key); bindGroups.set(key, existing);
      return {index, group: existing.group};
    }
    let layouts = bindingLayouts.get(pipeline);
    if (!layouts) bindingLayouts.set(pipeline, layouts = new Map());
    if (!layouts.has(index)) layouts.set(index, pipeline.getBindGroupLayout(index));
    live();
    const group = device.createBindGroup({layout: layouts.get(index), entries});
    live();
    if (key !== null) {
      while (bindGroups.size >= maxCachedBindGroups) evictBindings(bindGroups.keys().next().value);
      bindGroups.set(key, {group, buffers: resources});
      for (const buffer of resources) {
        let users = bindingUsers.get(buffer);
        if (!users) bindingUsers.set(buffer, users = new Set());
        users.add(key);
      }
    }
    return {index, group};
  }
  // Explicit layouts are necessary for unfilterable float32 textureLoad inputs:
  // automatic layouts infer sampleType:float even when no sampler is used.
  // Snapshot descriptors before yielding and include them in the pipeline key.
  function captureLayouts(layouts) {
    if (layouts === null) return null;
    if (!Array.isArray(layouts) || layouts.length > (device.limits?.maxBindGroups ?? 4) ||
        layouts.some(group => !group || !Array.isArray(group.entries)))
      fail('LAYOUT', 'Expected bounded bind group layout descriptors');
    return structuredClone(layouts);
  }
  function pipelineLayout(layouts) {
    return layouts === null ? 'auto' : device.createPipelineLayout({
      bindGroupLayouts: layouts.map(group => device.createBindGroupLayout(group)),
    });
  }
  function prepare({code, entryPoint = 'main', label = 'f3d-compute', bindGroupLayouts = null} = {}) {
    live();
    if (typeof code !== 'string' || !code.trim() || typeof entryPoint !== 'string' || !entryPoint)
      fail('SHADER', 'Expected WGSL source and an entry point');
    const layouts = captureLayouts(bindGroupLayouts);
    const key = JSON.stringify([code, entryPoint, layouts]);
    if (pipelines.has(key)) return pipelines.get(key);
    if (pipelines.size >= maxPipelines) fail('BUDGET', 'Compute pipeline count exceeds maxPipelines');
    const validations = [];
    const run = (async () => {
      const module = checked(() => device.createShaderModule({label, code}), validations);
      if (module.getCompilationInfo) {
        const info = await interrupted(module.getCompilationInfo());
        const errors = info.messages.filter(message => message.type === 'error');
        if (errors.length) fail('SHADER', errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
      }
      live();
      const descriptor = {label, layout: checked(() => pipelineLayout(layouts), validations), compute: {module, entryPoint}};
      const pipeline = await interrupted(checked(() => device.createComputePipelineAsync
        ? device.createComputePipelineAsync(descriptor) : device.createComputePipeline(descriptor), validations));
      // Native completion is not validation, especially with the synchronous
      // fallback. Join only this preparation's scopes, never unrelated work or
      // a GPU queue fence, before publishing an asynchronous success.
      await interrupted(Promise.all(validations));
      live();
      // A source-synchronous compute may have prepared the same key while this
      // job yielded. Adopt its validated result rather than overwrite its cache.
      if (pipelines.get(key) !== run) return pipelines.get(key);
      ready.add(pipeline); compiled.set(key, pipeline); return pipeline;
    })();
    pipelines.set(key, run);
    run.catch(() => { if (pipelines.get(key) === run) pipelines.delete(key); });
    return run;
  }
  /** Source renderer.compute() is synchronous after initialization. This path
   * compiles before submission; compileComputeAsync() can prewarm the same key.
   * Validation is still tracked by whenIdle(), never silently discarded. */
  function prepareSync({code, entryPoint = 'main', label = 'f3d-compute', bindGroupLayouts = null} = {}) {
    live();
    if (typeof code !== 'string' || !code.trim() || typeof entryPoint !== 'string' || !entryPoint)
      fail('SHADER', 'Expected WGSL source and an entry point');
    const layouts = captureLayouts(bindGroupLayouts);
    const key = JSON.stringify([code, entryPoint, layouts]);
    if (compiled.has(key)) return compiled.get(key);
    if (!pipelines.has(key) && pipelines.size >= maxPipelines) fail('BUDGET', 'Compute pipeline count exceeds maxPipelines');
    const validations = [];
    const pipeline = checked(() => {
      const module = device.createShaderModule({label, code});
      return device.createComputePipeline({label, layout: pipelineLayout(layouts), compute: {module, entryPoint}});
    }, validations);
    // Synchronous source dispatch remains synchronous. An asynchronous caller
    // sharing this provisional pipeline must still await its scope result.
    ready.add(pipeline); compiled.set(key, pipeline);
    const run = interrupted(Promise.all(validations)).then(() => { live(); return pipeline; });
    pipelines.set(key, run);
    run.catch(() => { if (pipelines.get(key) === run) pipelines.delete(key); });
    return pipeline;
  }
  function dispatch(pipeline, groups, workgroups, timestamp = null, operation = null) {
    live();
    if (!ready.has(pipeline)) fail('PIPELINE', 'Prepare the compute pipeline before dispatch');
    let direct = null;
    if (Array.isArray(workgroups)) {
      const limit = device.limits?.maxComputeWorkgroupsPerDimension ?? 65535;
      if (workgroups.length < 1 || workgroups.length > 3 ||
          !workgroups.every(n => Number.isSafeInteger(n) && n >= 0 && n <= limit))
        fail('DISPATCH', 'Workgroup counts must fit the device limits');
      direct = [workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1];
    } else {
      const {buffer, offset = 0} = workgroups ?? {};
      const entry = [...buffers.values()].find(e => e.buffer === buffer && !e.uniform);
      if (!entry || !Number.isSafeInteger(offset) || offset < 0 || offset % 4 || offset + 12 > entry.bytes)
        fail('DISPATCH', 'Indirect dispatch requires twelve in-bounds bytes of owned storage');
    }
    const snapshots = captureGroups(groups);
    const validations = timestamp ? [] : null;
    const issue = () => checked(() => {
      const bindings = snapshots.map(snapshot => resolveGroup(pipeline, snapshot));
      const encoder = device.createCommandEncoder({label: 'f3d-compute'});
      const descriptor = {label: 'f3d-compute', timestampWrites: timestamp?.writes()};
      if (operation) {
        const {backend, context} = operation.scope;
        const pass = backend.beginCompute(context, encoder, descriptor);
        backend.compute(context, operation.node, bindings, pipeline, operation.dispatchSize, pass, direct ?? workgroups);
        backend.finishCompute(context, pass);
      } else {
        const pass = encoder.beginComputePass(descriptor);
        pass.setPipeline(pipeline);
        for (const {index, group} of bindings) pass.setBindGroup(index, group);
        if (direct) pass.dispatchWorkgroups(...direct);
        else pass.dispatchWorkgroupsIndirect(workgroups.buffer, workgroups.offset ?? 0);
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
    }, validations);
    try { if (operation) operation.scope.run(issue); else issue(); }
    catch (error) { timestamp?.cancel(); throw error; }
    if (timestamp) timestamp.validate(Promise.all(validations));
  }
  function releaseReadback(entry, error) {
    const request = entry.request;
    if (!request) return;
    entry.request = null;
    request.cancel(error);
    entry.target.buffer = null; entry.target._mapped = false;
    entry.buffer.unmap();
  }
  function disposeReadback(entry, error = new GpuComputeError('READBACK_DISPOSED', 'ReadbackBuffer was disposed')) {
    if (readbacks.get(entry.target) !== entry) return;
    releaseReadback(entry, error);
    readbacks.delete(entry.target);
    entry.target.removeEventListener('release', entry.onRelease);
    entry.target.removeEventListener('dispose', entry.onDispose);
    entry.buffer.destroy(); allocatedBytes -= entry.size; readbackBytes -= entry.size;
    onReadbackChange?.(readbacks.size, readbackBytes);
  }
  function reusableReadback(request, copySize) {
    let entry = readbacks.get(request.target);
    // r186 retains the first native capacity until disposal, even if the
    // application later edits the public maxByteLength property.
    if (copySize > (entry?.size ?? request.capacity))
      fail('READBACK_TARGET', 'ReadbackBuffer capacity is smaller than the requested byte range');
    if (!entry) {
      if (request.capacity > (device.limits?.maxBufferSize ?? Infinity))
        fail('LIMIT', 'ReadbackBuffer capacity exceeds the device buffer size limit');
      if (staging.size + readbacks.size >= maxBuffers) fail('BUDGET', 'Readback buffers exceed the compute buffer count budget');
      budget(request.capacity);
      const buffer = checked(() => device.createBuffer({label: request.target.name || 'f3d-compute-readback',
        size: request.capacity, usage: MAP_READ | COPY_DST}));
      entry = {target: request.target, buffer, size: request.capacity, request: null};
      entry.onRelease = () => releaseReadback(entry, new GpuComputeError('READBACK_RELEASED', 'ReadbackBuffer was released before completion'));
      entry.onDispose = () => disposeReadback(entry);
      readbacks.set(entry.target, entry); allocatedBytes += entry.size; readbackBytes += entry.size;
      entry.target.addEventListener('release', entry.onRelease); entry.target.addEventListener('dispose', entry.onDispose);
      onReadbackChange?.(readbacks.size, readbackBytes);
    }
    entry.request = request;
    // Mapping can finish before another resource owner's completion proof.
    // A later facade failure must still release this native mapping, without
    // touching a newer request that reused the same public handle.
    request.onCancel = error => { if (entry.request === request) releaseReadback(entry, error); };
    return entry;
  }
  async function readCaptured(key, request) {
    live();
    if (!capturedReadbacks.has(request)) fail('READBACK_TARGET', 'Expected a captured readback request');
    request.check();
    const entry = buffers.get(key);
    if (!entry) fail('BUFFER', 'The storage attribute has not been uploaded');
    const {buffer: source} = entry, {offset, count, target} = request;
    const bytes = count === -1 ? entry.bytes - offset : count;
    if (offset > entry.bytes || bytes < 0 || bytes > entry.bytes - offset)
      fail('READBACK_RANGE', 'Readback byte range exceeds the resident storage allocation');
    request.check(bytes);
    // Low-level byte uploads may have a padded tail; the public source storage
    // API uses 32-bit components. Copy that padding only for an implicit tail.
    const size = Math.ceil(bytes / 4) * 4;
    let reusable = null, buffer;
    if (request.reusable) { reusable = reusableReadback(request, size); buffer = reusable.buffer; }
    else {
      if (staging.size + readbacks.size >= maxBuffers) fail('BUDGET', 'Readback buffers exceed the compute buffer count budget');
      budget(size);
      buffer = checked(() => device.createBuffer({label: 'f3d-compute-readback', size, usage: MAP_READ | COPY_DST}));
      staging.add(buffer); allocatedBytes += size;
    }
    let success = false;
    try {
      // Submit the snapshot before the first await so a following dispatch
      // cannot overtake this readback while mapAsync waits for completion.
      checked(() => {
        const encoder = device.createCommandEncoder({label: 'f3d-compute-readback'});
        encoder.copyBufferToBuffer(source, offset, buffer, 0, size);
        device.queue.submit([encoder.finish()]);
      });
      // The copy observes only work issued through this submission boundary.
      // Capture its outstanding validation before mapping yields: later uploads
      // and dispatches must not hold an already completed readback open.
      const validations = [...pending];
      await interrupted(request.wait(buffer.mapAsync(MAP_READ, 0, size)));
      await interrupted(request.wait(Promise.all(validations)));
      live(); request.check(bytes);
      const mapped = buffer.getMappedRange(0, size);
      let result;
      if (reusable) { target.buffer = mapped; result = target; }
      else if (target === null) result = mapped.slice(0, bytes);
      else { new Uint8Array(target).set(new Uint8Array(mapped, 0, bytes)); result = target; }
      success = true;
      return result;
    } finally {
      // stop() already destroyed and uncharged buffers on terminal paths.
      if (reusable) {
        if (!success && reusable.request === request)
          releaseReadback(reusable, new GpuComputeError('READBACK_CANCELLED', 'Readback did not complete'));
      } else if (staging.delete(buffer)) { buffer.unmap(); buffer.destroy(); allocatedBytes -= size; }
    }
  }
  async function read(key, target = null, offset = 0, count = -1) {
    const request = captureGpuReadback(target, offset, count);
    let success = false;
    try { const result = await readCaptured(key, request); success = true; return result; }
    finally { request.finish(success); }
  }
  async function whenIdle() {
    live();
    await interrupted(Promise.all([...pending, device.queue.onSubmittedWorkDone()]));
    live();
  }
  // Capture validation already issued by this owner. Resource preparation can
  // join this proof without waiting for GPU work or future unrelated scopes.
  async function validationSnapshot() {
    live();
    await interrupted(Promise.all([...pending]));
    live();
  }
  return {upload, release, prepare, prepareSync, dispatch, read, readCaptured, whenIdle, validationSnapshot,
    captureUpload: (key, array, version) => { live(); return attributeOwner.capture(key, array, version); },
    releaseUpload: admission => attributeOwner.releaseCapture(admission),
    getBuffer: key => { live(); return buffers.get(key)?.buffer ?? null; },
    get allocatedBytes() { return allocatedBytes; }, get bufferCount() { return buffers.size; },
    get readbackBufferCount() { return readbacks.size; }, get readbackBytes() { return readbackBytes; },
    get cachedBindGroupCount() { return bindGroups.size; },
    dispose: () => stop(new GpuComputeError('DISPOSED', 'Compute owner is disposed'))};
}
