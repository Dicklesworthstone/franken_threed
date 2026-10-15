/** r186 TSL compute on F3D-owned WebGPU pipelines and storage.
 * WGSLNodeBuilder and NodeFrame are retained source shader/update preparation;
 * no upstream renderer or backend is constructed or used to submit commands.
 * Currently admits buffer/uniform kernels (including atomics and indirect
 * dispatch), not texture bindings or compute-to-scene vertex-buffer sharing. */
import {createGpuCompute} from './gpu_compute.mjs';

export class ThreeComputeError extends Error {
  constructor(code, message) {
    super(`THREE_COMPUTE_${code}: ${message}`);
    this.name = 'ThreeComputeError'; this.code = `THREE_COMPUTE_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeComputeError(code, message); };
const listOf = nodes => {
  const list = Array.isArray(nodes) ? nodes.slice() : [nodes];
  if (!list.length || list.some(node => node?.isComputeNode !== true))
    fail('SOURCE', 'Expected a ComputeNode or a nonempty array of ComputeNodes');
  return list;
};

function createThreeCompute(renderer, three) {
  if (three.REVISION !== '186' || typeof three.WGSLNodeBuilder !== 'function' || !renderer._nodes?.nodeFrame?.updateNode)
    fail('SOURCE', 'Supply the pinned r186 three/webgpu namespace for TSL compute');
  const device = renderer._device, frame = renderer._nodes.nodeFrame;
  const gpu = createGpuCompute(device, {signal: renderer._targetLifetime.signal});
  const programs = new Map(), initialized = new Set(), listeners = new Map(), uploading = new WeakMap(), building = new Set();
  let disposed = false, uniformVersion = 0;
  function live() {
    if (disposed || renderer._disposed) fail('DISPOSED', 'Renderer compute execution is disposed');
  }
  function validateAttribute(attribute) {
    if (!(attribute?.isStorageBufferAttribute || attribute?.isStorageInstancedBufferAttribute || attribute?.isIndirectStorageBufferAttribute) ||
        attribute.isInterleavedBufferAttribute || attribute.normalized || !ArrayBuffer.isView(attribute.array) ||
        ![1, 2, 3, 4, 16].includes(attribute.itemSize) || !Number.isSafeInteger(attribute.count) || attribute.count < 1 ||
        attribute.array.length !== attribute.count * attribute.itemSize)
      fail('STORAGE', 'Expected an unnormalized scalar/vector/mat4 storage attribute with a complete typed array');
    if (![Float32Array, Int32Array, Uint32Array, Int16Array, Uint16Array, Int8Array, Uint8Array].includes(attribute.array.constructor))
      fail('STORAGE', 'Storage attributes require f32 or integer components');
  }
  function storage(attribute) {
    live(); validateAttribute(attribute);
    // Mirror source WebGPU storage admission: small integers become 32-bit,
    // and source vec3 storage arrays acquire WGSL's four-component stride.
    let array = attribute.array;
    if (array instanceof Int8Array || array instanceof Int16Array) array = new Int32Array(array);
    else if (array instanceof Uint8Array || array instanceof Uint16Array) array = new Uint32Array(array);
    if (attribute.itemSize === 3) {
      const padded = new array.constructor(attribute.count * 4);
      for (let i = 0; i < attribute.count; i++) padded.set(array.subarray(i * 3, i * 3 + 3), i * 4);
      array = padded; attribute.itemSize = 4;
      // Initial upload is necessarily complete; source ranges referred to the
      // original packed array and are no longer meaningful after padding.
      attribute.clearUpdateRanges?.();
    }
    attribute.array = array;
    const previous = uploading.get(attribute), version = attribute.version ?? 0;
    const changed = !previous || previous.array !== array || previous.version !== version;
    const buffer = gpu.upload(attribute, array, version, {ranges: attribute.updateRanges ?? []});
    if (changed) {
      uploading.set(attribute, {array, version});
      attribute.clearUpdateRanges?.(); attribute.onUploadCallback?.();
    }
    return buffer;
  }
  function forget(node) {
    const entry = programs.get(node);
    // Storage belongs to the renderer, not one kernel: other kernels, indirect
    // dispatches and readbacks can still refer to the same source attribute.
    if (entry) for (const group of entry.groups) for (const binding of group.bindings)
      if (binding.isUniformBuffer) gpu.release(binding);
    programs.delete(node);
  }
  function initialize(node) {
    if (initialized.has(node)) return;
    if (initialized.size >= 256) fail('BUDGET', 'Compute nodes exceed the renderer program budget');
    const onDispose = () => {
      forget(node); initialized.delete(node);
      node.removeEventListener?.('dispose', onDispose); listeners.delete(node);
    };
    initialized.add(node); listeners.set(node, onDispose); node.addEventListener?.('dispose', onDispose);
    try { node.onInitFunction?.call(node, {renderer}); }
    catch (error) { onDispose(); throw error; }
    live();
    if (!initialized.has(node)) fail('STALE', 'Compute node was disposed during initialization');
  }
  function workgroupSize(node) {
    const size = node.workgroupSize;
    const limits = device.limits ?? {};
    if (!Array.isArray(size) || size.length < 1 || size.length > 3 ||
        size.some((n, i) => !Number.isSafeInteger(n) || n < 1 ||
          n > (limits[['maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ'][i]] ?? [256, 256, 64][i])) ||
        size.reduce((a, b) => a * b, 1) > (limits.maxComputeInvocationsPerWorkgroup ?? 256))
      fail('DISPATCH', 'Compute workgroup size exceeds device limits');
    return size;
  }
  function program(node) {
    live(); workgroupSize(node);
    if (building.has(node)) fail('REENTRY', 'A compute node cannot initialize or compile itself recursively');
    building.add(node);
    try {
      initialize(node);
      const key = `${node.version}/${node.workgroupSize.join(',')}`;
      let entry = programs.get(node);
      if (entry?.key === key) return entry;
      const builder = new three.WGSLNodeBuilder(null, renderer);
      builder.compute = node; builder.build();
      const groups = builder.getBindings();
      // Reject unsupported binding families before uploading any source storage.
      for (const group of groups) for (const binding of group.bindings) {
        if (binding.isStorageBuffer) validateAttribute(binding.attribute);
        else if (!binding.isUniformBuffer) fail('BINDING', 'TSL compute currently admits storage and uniform buffers, not texture/sampler bindings');
      }
      forget(node);
      entry = {key, builder, groups, shader: {code: builder.computeShader, label: node.name || 'f3d-tsl-compute'}};
      programs.set(node, entry); return entry;
    } finally { building.delete(node); }
  }
  function withFrame(node, run) {
    const previous = {renderer: frame.renderer, compute: frame.compute, renderId: frame.renderId};
    frame.renderer = renderer; frame.compute = node; frame.renderId = renderer.info.calls;
    try { return run(); } finally { Object.assign(frame, previous); }
  }
  function update(entry, stage) {
    const names = {before: ['updateBeforeNodes', 'updateBeforeNode'], current: ['updateNodes', 'updateNode'], after: ['updateAfterNodes', 'updateAfterNode']};
    const [nodes, method] = names[stage];
    for (const node of entry.builder[nodes] ?? []) frame[method](node);
  }
  function bindings(entry) {
    return entry.groups.map((group, index) => ({index, entries: group.bindings.map((binding, slot) => {
      let buffer;
      if (binding.isStorageBuffer) buffer = storage(binding.attribute);
      else {
        binding.update?.();
        const array = binding.buffer instanceof ArrayBuffer ? new Uint8Array(binding.buffer) : binding.buffer;
        buffer = gpu.upload(binding, array, ++uniformVersion, {uniform: true});
        binding.clearUpdateRanges?.();
      }
      return {binding: slot, resource: {buffer}};
    })}));
  }
  function dispatchSize(node, override) {
    const size = override ?? (node.dispatchSize || node.count);
    if (size?.isIndirectStorageBufferAttribute) return {buffer: storage(size), offset: 0};
    if (Array.isArray(size)) return [size[0], size[1] || 1, size[2] || 1];
    if (!Number.isSafeInteger(size) || size < 0) fail('DISPATCH', 'Expected a nonnegative invocation count, workgroup dimensions or indirect storage attribute');
    const count = Math.ceil(size / workgroupSize(node).reduce((a, b) => a * b, 1));
    const limit = device.limits?.maxComputeWorkgroupsPerDimension ?? 65535;
    return count > limit ? [limit, Math.ceil(count / limit), 1] : [count, 1, 1];
  }
  function compute(list, override) {
    live();
    renderer.info.calls++; renderer.info.compute.calls++; renderer.info.compute.frameCalls++;
    renderer.inspector?.beginCompute?.(`compute:${renderer.info.compute.frameCalls}:f${renderer.info.frame}`, list);
    try {
      for (const node of list) {
        const entry = program(node);
        withFrame(node, () => {
          update(entry, 'before'); update(entry, 'current');
          const pipeline = gpu.prepareSync(entry.shader), groups = bindings(entry), size = dispatchSize(node, override);
          gpu.dispatch(pipeline, groups, size);
          update(entry, 'after');
        });
      }
    } finally { renderer.inspector?.finishCompute?.(); }
  }
  async function compile(list, onProgress) {
    for (let i = 0; i < list.length; i++) {
      live();
      const node = list[i], entry = program(node);
      // Initialize data and run the same source updates, but never dispatch.
      withFrame(node, () => { update(entry, 'before'); update(entry, 'current'); bindings(entry); });
      await gpu.prepare(entry.shader);
      live();
      if (programs.get(node) !== entry) fail('STALE', 'Compute node changed during asynchronous compilation');
      withFrame(node, () => update(entry, 'after'));
      if (onProgress) onProgress(typeof ProgressEvent === 'function'
        ? new ProgressEvent('progress', {lengthComputable: true, loaded: i + 1, total: list.length})
        : {type: 'progress', lengthComputable: true, loaded: i + 1, total: list.length});
    }
  }
  return {compute, compile, gpu,
    read(attribute) { live(); validateAttribute(attribute); return gpu.read(attribute); },
    dispose() {
      if (disposed) return; disposed = true;
      for (const [node, listener] of listeners) node.removeEventListener?.('dispose', listener);
      listeners.clear(); initialized.clear(); programs.clear(); gpu.dispose();
    }};
}

/** Install on the public class; reuse its existing render/transfer submission
 * FIFO. Each issue() submits synchronously, including first-use compilation.
 * Waiting for mapping or precompilation never holds the queue's submit slot. */
export function withThreeCompute(Base, THREE) {
  const owners = new WeakMap();
  const owner = renderer => {
    let value = owners.get(renderer);
    if (!value) { value = createThreeCompute(renderer, THREE); owners.set(renderer, value); }
    return value;
  };
  function submit(renderer, issue) {
    if (renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
    if (!renderer.isWebGPURenderer) fail('ROUTE', 'TSL compute requires the WebGPU renderer surface');
    if (renderer._deferredError) { const error = renderer._deferredError; renderer._deferredError = null; throw error; }
    if (renderer._initialized && !renderer._drain && !renderer._preparing) return issue();
    return new Promise((resolve, reject) => renderer._enqueue({transfer: true, issue, resolve, reject}));
  }
  return class WebGPURenderer extends Base {
    constructor(...args) {
      super(...args);
      if (this.isF3DRenderer && THREE.TSL?.context && !this.contextNode) this.contextNode = THREE.TSL.context();
    }
    compute(nodes, dispatchSize = null) {
      const list = listOf(nodes), size = Array.isArray(dispatchSize) ? dispatchSize.slice() : dispatchSize;
      const result = submit(this, () => owner(this).compute(list, size));
      // Source code may ignore compute()'s pre-initialization promise. Preserve
      // errors for the next public call rather than creating an unhandled rejection.
      result?.catch(error => { this._deferredError ??= error; });
      return result;
    }
    async computeAsync(nodes, dispatchSize = null) { await this.compute(nodes, dispatchSize); }
    async compileComputeAsync(nodes, onProgress = null) {
      const list = listOf(nodes);
      if (onProgress !== null && typeof onProgress !== 'function') fail('OPTIONS', 'onProgress must be a function');
      await submit(this, () => owner(this).compile(list, onProgress));
    }
    async getArrayBufferAsync(attribute) {
      return await submit(this, () => owner(this).read(attribute));
    }
    async waitForGPU() {
      await super.waitForGPU();
      const value = owners.get(this); if (value) await value.gpu.whenIdle();
    }
    dispose() { owners.get(this)?.dispose(); super.dispose(); }
  };
}
