/** Owned WebGPU compute execution. Shader preparation is supplied by the caller;
 * this owner creates pipelines, keeps storage GPU-resident across dispatches,
 * submits real compute passes and performs asynchronous staging-buffer readback.
 * General WebGPU execution, not a Rust specialization or a speedup claim. */
export class GpuComputeError extends Error {
  constructor(code, message) {
    super(`GPU_COMPUTE_${code}: ${message}`);
    this.name = 'GpuComputeError'; this.code = `GPU_COMPUTE_${code}`;
  }
}
const fail = (code, message) => { throw new GpuComputeError(code, message); };
// WebGPU constants are also usable in command-recording Node tests.
const COPY_SRC = 4, COPY_DST = 8, UNIFORM = 64, STORAGE = 128, INDIRECT = 256, MAP_READ = 1;
const positive = n => Number.isSafeInteger(n) && n > 0;

export function createGpuCompute(device, {maxBytes = 256 * 1024 * 1024, maxBuffers = 1024,
  maxPipelines = 256, signal = null} = {}) {
  if (!device?.queue || typeof device.createBuffer !== 'function' ||
      ![maxBytes, maxBuffers, maxPipelines].every(positive)) fail('OPTIONS', 'Expected a GPUDevice and positive resource limits');
  const buffers = new Map(), pipelines = new Map(), compiled = new Map(), ready = new Set(), staging = new Set(), pending = new Set();
  const lifetime = new AbortController();
  let terminal = null, allocatedBytes = 0;
  const live = () => { if (terminal) throw terminal; };
  function stop(error) {
    if (terminal) return;
    terminal = error instanceof Error ? error : new GpuComputeError('ABORTED', String(error ?? 'Compute execution aborted'));
    lifetime.abort(terminal);
    for (const entry of buffers.values()) entry.buffer.destroy();
    for (const buffer of staging) buffer.destroy();
    buffers.clear(); staging.clear(); ready.clear(); pipelines.clear(); compiled.clear(); allocatedBytes = 0;
    signal?.removeEventListener('abort', onAbort);
  }
  const onAbort = () => stop(signal.reason);
  if (signal?.aborted) onAbort(); else signal?.addEventListener('abort', onAbort, {once: true});
  // A borrowed device is never destroyed by this owner.
  device.lost?.then(info => stop(new GpuComputeError('DEVICE_LOST', info?.message ?? 'WebGPU device lost')),
    error => stop(error));
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
  function checked(run) {
    live();
    device.pushErrorScope?.('validation');
    try { return run(); }
    finally {
      if (device.popErrorScope) watch(device.popErrorScope().then(error => {
        if (error) throw new GpuComputeError('VALIDATION', error.message);
      }));
    }
  }
  function budget(bytes) {
    if (!positive(bytes) || bytes > maxBytes - allocatedBytes) fail('BUDGET', 'Compute buffers exceed their shared byte budget');
  }
  function upload(key, array, version = 0, {uniform = false, ranges = []} = {}) {
    live();
    if (!key || !['object', 'function'].includes(typeof key) || !ArrayBuffer.isView(array) || array instanceof DataView)
      fail('BUFFER', 'Expected an identity key and typed array');
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
        usage: COPY_SRC | COPY_DST | (uniform ? UNIFORM : STORAGE | INDIRECT)}));
      entry = {buffer, size, bytes, uniform, array: null, version: null};
      allocatedBytes += size; buffers.set(key, entry);
      if (old) { old.buffer.destroy(); allocatedBytes -= old.size; }
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
    return entry.buffer;
  }
  function release(key) {
    const entry = buffers.get(key);
    if (!entry) return;
    buffers.delete(key); entry.buffer.destroy(); allocatedBytes -= entry.size;
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
    const run = (async () => {
      const module = checked(() => device.createShaderModule({label, code}));
      if (module.getCompilationInfo) {
        const info = await interrupted(module.getCompilationInfo());
        const errors = info.messages.filter(message => message.type === 'error');
        if (errors.length) fail('SHADER', errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
      }
      live();
      const descriptor = {label, layout: checked(() => pipelineLayout(layouts)), compute: {module, entryPoint}};
      const pipeline = await interrupted(checked(() => device.createComputePipelineAsync
        ? device.createComputePipelineAsync(descriptor) : device.createComputePipeline(descriptor)));
      live(); ready.add(pipeline); compiled.set(key, pipeline); return pipeline;
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
    const pipeline = checked(() => {
      const module = device.createShaderModule({label, code});
      return device.createComputePipeline({label, layout: pipelineLayout(layouts), compute: {module, entryPoint}});
    });
    ready.add(pipeline); compiled.set(key, pipeline); pipelines.set(key, Promise.resolve(pipeline));
    return pipeline;
  }
  function dispatch(pipeline, groups, workgroups) {
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
    checked(() => {
      const bindings = groups.map(({index, entries}) => ({index, group: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(index), entries})}));
      const encoder = device.createCommandEncoder({label: 'f3d-compute'});
      const pass = encoder.beginComputePass({label: 'f3d-compute'});
      pass.setPipeline(pipeline);
      for (const {index, group} of bindings) pass.setBindGroup(index, group);
      if (direct) pass.dispatchWorkgroups(...direct);
      else pass.dispatchWorkgroupsIndirect(workgroups.buffer, workgroups.offset ?? 0);
      pass.end(); device.queue.submit([encoder.finish()]);
    });
  }
  async function read(key) {
    live();
    const entry = buffers.get(key);
    if (!entry) fail('BUFFER', 'The storage attribute has not been uploaded');
    const {size, bytes, buffer: source} = entry;
    budget(size);
    const buffer = checked(() => device.createBuffer({label: 'f3d-compute-readback', size, usage: MAP_READ | COPY_DST}));
    staging.add(buffer); allocatedBytes += size;
    try {
      // Submit the snapshot before the first await so a following dispatch
      // cannot overtake this readback while mapAsync waits for completion.
      checked(() => {
        const encoder = device.createCommandEncoder({label: 'f3d-compute-readback'});
        encoder.copyBufferToBuffer(source, 0, buffer, 0, size);
        device.queue.submit([encoder.finish()]);
      });
      await interrupted(buffer.mapAsync(MAP_READ));
      await interrupted(Promise.all([...pending]));
      live();
      return buffer.getMappedRange().slice(0, bytes);
    } finally {
      // stop() already destroyed and uncharged buffers on terminal paths.
      if (staging.delete(buffer)) { buffer.unmap(); buffer.destroy(); allocatedBytes -= size; }
    }
  }
  async function whenIdle() {
    live();
    await interrupted(Promise.all([...pending, device.queue.onSubmittedWorkDone()]));
    live();
  }
  return {upload, release, prepare, prepareSync, dispatch, read, whenIdle,
    getBuffer: key => { live(); return buffers.get(key)?.buffer ?? null; },
    get allocatedBytes() { return allocatedBytes; }, get bufferCount() { return buffers.size; },
    dispose: () => stop(new GpuComputeError('DISPOSED', 'Compute owner is disposed'))};
}
