/** r186 TSL compute on F3D-owned WebGPU pipelines and storage.
 * WGSLNodeBuilder and NodeFrame are retained source shader/update preparation;
 * no upstream renderer or backend is constructed or used to submit commands.
 * Initialized calls capture their uniforms, storage upload versions and dispatch
 * sizes before joining the renderer FIFO. Deferred GPU submission must not turn
 * distinct application updates into repeated execution of the last state.
 * StorageTexture kernels share native residency with scene materials. Ready
 * CPU data textures also provide read-only, call-time-versioned sampled inputs.
 * Native color and depth RenderTarget inputs borrow their attachments through
 * the FIFO submission, including both aspects of one target. No UV replica,
 * readback or CPU upload is introduced. Decoded image inputs remain unsupported. */
import {createGpuCompute} from './gpu_compute.mjs';
import {copyComputeBufferToTarget} from './gpu_compute_texture_copy.mjs';
import {createThreeStorageTextures, inspectThreeStorageTexture} from './three_storage_textures.mjs';
import {createGpuThreeTextures} from './three_textures.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';

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
  const textureOwner = createThreeComputeTextures({storage: renderer._storageTextures,
    poolFactory: () => createGpuThreeTextures(device, {three}),
    targetFactory: () => createThreeComputeTargets({targets: renderer._targets, device, signal: renderer._targetLifetime.signal})});
  const gpu = createGpuCompute(device, {signal: renderer._targetLifetime.signal});
  const programs = new Map(), initialized = new Set(), listeners = new Map(), building = new Set();
  const captured = new WeakMap(), uploaded = new WeakMap(), payloads = new Set(), compiling = new Set();
  const transfers = new Set();
  let transferFailure = null;
  const maxSnapshotBytes = 128 * 1024 * 1024;
  let disposed = false, uniformVersion = 0, snapshotBytes = 0, issuing = false;
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
  function normalize(attribute) {
    live(); validateAttribute(attribute);
    let array = attribute.array;
    if (array instanceof Int8Array || array instanceof Int16Array) array = new Int32Array(array);
    else if (array instanceof Uint8Array || array instanceof Uint16Array) array = new Uint32Array(array);
    if (attribute.itemSize === 3) {
      const padded = new array.constructor(attribute.count * 4);
      for (let i = 0; i < attribute.count; i++) padded.set(array.subarray(i * 3, i * 3 + 3), i * 4);
      array = padded; attribute.itemSize = 4;
      attribute.clearUpdateRanges?.();
    }
    attribute.array = array;
    return array;
  }
  function free(record) {
    if (!record.data) return;
    snapshotBytes -= record.charged; record.charged = 0; record.data = null; payloads.delete(record);
  }
  function snapshot(key, array, version, uniform, deferred, ranges = []) {
    if (!ArrayBuffer.isView(array) || array instanceof DataView) fail('BUFFER', 'Expected a typed uniform or storage buffer');
    const previous = !uniform && captured.get(key);
    if (previous?.source === array && previous.version === version && (previous.data || uploaded.get(key) === previous)) return previous;
    const charged = deferred ? array.byteLength : 0;
    if (charged > maxSnapshotBytes - snapshotBytes) fail('BUDGET', 'Queued compute uploads exceed the snapshot byte budget');
    const data = deferred ? new array.constructor(array) : array;
    const record = {key, source: array, version, uniform, data, charged, ranges, refs: 0};
    snapshotBytes += charged; payloads.add(record);
    if (!uniform) captured.set(key, record);
    return record;
  }
  function upload(record) {
    const {key, version, uniform, source, ranges} = record;
    if (uploaded.get(key) === record) {
      const buffer = gpu.getBuffer(key);
      if (buffer) return buffer;
    }
    if (!record.data) fail('STALE', 'A consumed upload snapshot cannot recreate released storage');
    const buffer = gpu.upload(key, record.data, version, {uniform, ranges});
    uploaded.set(key, record);
    if (!uniform) {
      // A later application update may already have new ranges. Acknowledging
      // this queued version must not erase the newer version's upload request.
      if (key.array === source && (key.version ?? 0) === version) key.clearUpdateRanges?.();
      key.onUploadCallback?.();
    }
    return buffer;
  }
  function releaseUniforms(entry) {
    for (const group of entry.groups) for (const binding of group.bindings)
      if (binding.isUniformBuffer) { gpu.release(binding); uploaded.delete(binding); }
  }
  function forget(node) {
    const entry = programs.get(node);
    if (entry) { entry.retired = true; if (!entry.uses) releaseUniforms(entry); }
    // Storage is renderer-owned, not owned by one kernel.
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
    const size = node.workgroupSize, limits = device.limits ?? {};
    if (!Array.isArray(size) || size.length < 1 || size.length > 3 ||
        size.some((n, i) => !Number.isSafeInteger(n) || n < 1 ||
          n > (limits[['maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ'][i]] ?? [256, 256, 64][i])) ||
        size.reduce((a, b) => a * b, 1) > (limits.maxComputeInvocationsPerWorkgroup ?? 256))
      fail('DISPATCH', 'Compute workgroup size exceeds device limits');
    return size;
  }
  function textureBinding(binding) {
    // NodeSampledTexture/NodeSampler update their cached .texture from .value.
    // Select the live value even before binding.update() is called for this turn.
    return {texture: binding.textureNode ? binding.textureNode.value : binding.texture,
      isSampler: binding.isSampler, isSampledTexture: binding.isSampledTexture,
      store: binding.store, access: binding.access, mipLevel: binding.mipLevel};
  }
  function captureTexture(binding) {
    return textureOwner.captureBinding(textureBinding(binding));
  }
  function textureSignature(groups) {
    return JSON.stringify(groups.map(group => group.bindings.map(binding =>
      binding.isSampledTexture || binding.isSampler
        ? textureOwner.layoutOf(textureBinding(binding)) : null)));
  }
  function program(node) {
    live(); workgroupSize(node);
    if (building.has(node)) fail('REENTRY', 'A compute node cannot initialize or compile itself recursively');
    building.add(node);
    try {
      initialize(node);
      const key = `${node.version}/${node.workgroupSize.join(',')}`;
      let entry = programs.get(node);
      if (entry?.key === key && entry.textureSignature === textureSignature(entry.groups)) return entry;
      const builder = new three.WGSLNodeBuilder(null, renderer);
      builder.compute = node; builder.build();
      const groups = builder.getBindings();
      for (const group of groups) for (const binding of group.bindings) {
        if (binding.isStorageBuffer) validateAttribute(binding.attribute);
        else if (binding.isSampledTexture || binding.isSampler) textureOwner.layoutOf(textureBinding(binding));
        else if (!binding.isUniformBuffer) fail('BINDING', 'Expected storage/uniform buffers or admitted texture bindings');
      }
      forget(node);
      entry = {key, builder, groups, uses: 0, retired: false, textureSignature: textureSignature(groups),
        shader: {code: builder.computeShader, label: node.name || 'f3d-tsl-compute'}};
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
  function capture(node, entry, override, deferred) {
    const held = new Set(), textureUses = [], epoch = listeners.get(node);
    let released = false;
    entry.uses++;
    const hold = record => { if (!held.has(record)) { held.add(record); record.refs++; } return record; };
    function release() {
      if (released) return; released = true;
      for (const record of held) if (--record.refs === 0) free(record);
      for (const texture of textureUses) textureOwner.release(texture);
      if (--entry.uses === 0 && entry.retired) releaseUniforms(entry);
    }
    function attributeRecord(attribute) {
      const array = normalize(attribute), version = attribute.version ?? 0;
      const ranges = (attribute.updateRanges ?? []).map(({start, count}) => ({start, count}));
      if (!Number.isSafeInteger(version) || version < 0 || ranges.some(({start, count}) =>
        !Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count < 0 || start + count > array.length))
        fail('STORAGE', 'Expected a nonnegative storage version and in-bounds update ranges');
      return hold(snapshot(attribute, array, version, false, deferred, ranges));
    }
    try {
      const groups = entry.groups.map((group, index) => ({index, entries: group.bindings.map((binding, slot) => {
        if (binding.isSampledTexture || binding.isSampler) {
          binding.update?.();
          const texture = captureTexture(binding);
          textureUses.push(texture);
          return {binding: slot, texture, layout: {binding: slot, visibility: 4, ...textureOwner.layout(texture)}};
        }
        let record;
        if (binding.isStorageBuffer) record = attributeRecord(binding.attribute);
        else {
          binding.update?.();
          const array = binding.buffer instanceof ArrayBuffer ? new Uint8Array(binding.buffer) : binding.buffer;
          record = hold(snapshot(binding, array, ++uniformVersion, true, deferred));
          binding.clearUpdateRanges?.();
        }
        return {binding: slot, record, layout: {binding: slot, visibility: 4,
          buffer: {type: binding.isStorageBuffer ? (binding.access === 'readOnly' ? 'read-only-storage' : 'storage') : 'uniform'}}};
      })}));
      if (textureSignature(entry.groups) !== entry.textureSignature)
        fail('PREPARE', 'Texture binding layout changed during node updates; prepare the new node layout before dispatch');
      textureOwner.validateDispatch(textureUses);
      const shader = textureUses.length ? {...entry.shader,
        bindGroupLayouts: groups.map(group => ({entries: group.entries.map(binding => binding.layout)}))} : entry.shader;
      const size = override ?? node.dispatchSize ?? node.count;
      let direct, indirect;
      if (size?.isIndirectStorageBufferAttribute) indirect = attributeRecord(size);
      else if (Array.isArray(size)) {
        if (size.length < 1 || size.length > 3) fail('DISPATCH', 'Expected one to three workgroup dimensions');
        direct = [size[0], size[1] ?? 1, size[2] ?? 1];
      } else {
        if (!Number.isSafeInteger(size) || size < 0) fail('DISPATCH', 'Expected a nonnegative invocation count, workgroup dimensions or indirect storage attribute');
        const count = Math.ceil(size / workgroupSize(node).reduce((a, b) => a * b, 1));
        const limit = device.limits?.maxComputeWorkgroupsPerDimension ?? 65535;
        direct = count > limit ? [limit, Math.ceil(count / limit), 1] : [count, 1, 1];
      }
      if (direct && direct.some(n => !Number.isSafeInteger(n) || n < 0 || n > (device.limits?.maxComputeWorkgroupsPerDimension ?? 65535)))
        fail('DISPATCH', 'Workgroup counts must fit the device limits');
      return {release, shader, validate: () => textureOwner.validateDispatch(textureUses), issue(dispatch = true) {
        live();
        if (listeners.get(node) !== epoch) fail('STALE', 'Compute node was disposed before its queued use');
        if (issuing) fail('REENTRY', 'Cannot submit compute from an upload callback');
        textureOwner.validateDispatch(textureUses);
        issuing = true;
        try { textureOwner.withDispatch(textureUses, () => {
          const bindings = groups.map(group => ({index: group.index, entries: group.entries.map(({binding, record, texture}) =>
            ({binding, resource: texture ? textureOwner.resolve(texture) : {buffer: upload(record)}}))}));
          const dimensions = indirect ? {buffer: upload(indirect), offset: 0} : direct;
          textureOwner.validateDispatch(textureUses);
          if (dispatch) {
            gpu.dispatch(gpu.prepareSync(shader), bindings, dimensions);
            textureOwner.written(textureUses);
          }
        }); } finally { issuing = false; }
      }};
    } catch (error) { release(); throw error; }
  }
  function compute(list, override, schedule, deferred) {
    live();
    if (issuing) fail('REENTRY', 'Cannot compute from an upload callback');
    renderer.info.calls++; renderer.info.compute.calls++; renderer.info.compute.frameCalls++;
    renderer.inspector?.beginCompute?.(`compute:${renderer.info.compute.frameCalls}:f${renderer.info.frame}`, list);
    const pending = [];
    try {
      for (const node of list) {
        const entry = program(node);
        withFrame(node, () => {
          update(entry, 'before'); update(entry, 'current');
          const request = capture(node, entry, override, deferred());
          let result;
          try { result = schedule(() => request.issue()); }
          catch (error) { request.release(); throw error; }
          if (result?.then) {
            const done = Promise.resolve(result).finally(request.release);
            // A later source callback may throw before we return this promise.
            done.catch(() => {}); pending.push(done);
          } else request.release();
          // Schedule first: a compute/readback started by an after-update node
          // must follow this kernel, even while render preparation is pending.
          update(entry, 'after');
        });
      }
    } finally { renderer.inspector?.finishCompute?.(); }
    return pending.length ? Promise.all(pending).then(() => {}) : undefined;
  }
  async function compile(list, onProgress) {
    for (let i = 0; i < list.length; i++) {
      live();
      const node = list[i], entry = program(node);
      const request = withFrame(node, () => {
        update(entry, 'before'); update(entry, 'current');
        return capture(node, entry, null, false);
      });
      try {
        request.issue(false);
        await gpu.prepare(request.shader);
        live(); request.validate();
        if (programs.get(node) !== entry) fail('STALE', 'Compute node changed during asynchronous compilation');
        withFrame(node, () => update(entry, 'after'));
        if (onProgress) onProgress(typeof ProgressEvent === 'function'
          ? new ProgressEvent('progress', {lengthComputable: true, loaded: i + 1, total: list.length})
          : {type: 'progress', lengthComputable: true, loaded: i + 1, total: list.length});
      } finally { request.release(); }
    }
  }
  return {compute, gpu,
    textureSampleData(texture) {
      return textureOwner.sampleData(texture);
    },
    compile(...args) {
      const task = compile(...args); compiling.add(task);
      task.finally(() => compiling.delete(task)).catch(() => {}); return task;
    },
    copyToTarget(attribute, destination, options) {
      live();
      if (transfers.size >= 64) fail('BUDGET', 'Too many compute-to-target copies are awaiting completion');
      const buffer = gpu.getBuffer(attribute);
      if (!buffer) fail('STORAGE', 'Compute must upload or produce this storage attribute before copying it');
      const task = copyComputeBufferToTarget(device, buffer, renderer._targets, destination, options,
        {signal: renderer._targetLifetime.signal, sourceCompletion: () => Promise.all([gpu.whenIdle(), textureOwner.whenIdle()])});
      transfers.add(task);
      task.catch(error => { transferFailure ??= error; }).finally(() => transfers.delete(task)).catch(() => {});
      return task;
    },
    async whenIdle() {
      await Promise.all([...compiling, ...transfers]);
      if (transferFailure) throw transferFailure;
      await Promise.all([gpu.whenIdle(), textureOwner.whenIdle()]);
    },
    read(attribute) {
      live(); validateAttribute(attribute);
      // Submit the buffer snapshot before waiting for texture-owner errors.
      return Promise.all([gpu.read(attribute), textureOwner.whenIdle()]).then(([bytes]) => bytes);
    },
    dispose() {
      if (disposed) return; disposed = true;
      for (const [node, listener] of listeners) node.removeEventListener?.('dispose', listener);
      for (const record of payloads) free(record);
      listeners.clear(); initialized.clear(); programs.clear(); textureOwner.dispose(); gpu.dispose();
    }};
}

/** All GPU work joins the renderer FIFO. Source updates and upload snapshots
 * are taken at the initialized call, not replayed when preparation finishes.
 * Before init(), source preparation follows upstream's asynchronous boundary. */
export function withThreeCompute(Base, THREE) {
  const owners = new WeakMap();
  const owner = renderer => {
    let value = owners.get(renderer);
    if (!value) { value = createThreeCompute(renderer, THREE); owners.set(renderer, value); }
    return value;
  };
  function check(renderer) {
    if (renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
    if (!renderer.isWebGPURenderer) fail('ROUTE', 'TSL compute requires the WebGPU renderer surface');
    if (renderer._deferredError) { const error = renderer._deferredError; renderer._deferredError = null; throw error; }
  }
  function submit(renderer, issue) {
    check(renderer);
    if (renderer._computeTurn || (renderer._initialized && !renderer._drain && !renderer._preparing)) return issue();
    return new Promise((resolve, reject) => renderer._enqueue({transfer: true, issue, resolve, reject}));
  }
  return class WebGPURenderer extends Base {
    constructor(...args) {
      super(...args);
      if (!this.isF3DRenderer) return; // Preserve an exact-backend constructor return.
      if (THREE.TSL?.context && !this.contextNode) this.contextNode = THREE.TSL.context();
      this._storageTextures = createThreeStorageTextures(THREE, {
        getDevice: () => this._device, signal: this._targetLifetime.signal,
      });
      // Compose the renderer's existing borrowed-texture seam once, before any
      // scene dispatcher is prepared. No source objects or GPU APIs are patched.
      const targets = this._targets;
      if (targets?.bindingsFor) {
        const bindingsFor = targets.bindingsFor.bind(targets);
        targets.bindingsFor = (destination, retained) => bindingsFor(destination, this._storageTextures.bindingsFor(retained));
      }
      // The pinned WGSL builder asks its backend for texture sample metadata.
      // Report the captured target's MSAA state and its resolved primary view.
      if (this._backend) {
        this._backend.utils ??= {};
        this._backend.utils.getTextureSampleData = texture => {
          if (!texture?.isStorageTexture) return owner(this).textureSampleData(texture);
          inspectThreeStorageTexture(texture, THREE);
          return {samples: 1, primarySamples: 1, isMSAA: false};
        };
        this._backend.hasFeature = name => !!this._device?.features?.has(name);
      }
    }
    compute(nodes, dispatchSize = null) {
      check(this);
      const list = listOf(nodes), size = Array.isArray(dispatchSize) ? dispatchSize.slice() : dispatchSize;
      const run = () => owner(this).compute(list, size, issue => submit(this, issue),
        () => !this._computeTurn && !!(this._drain || this._preparing));
      const result = this._initialized ? run() : submit(this, () => {
        // Nested onInit/before/after compute calls belong to this submit slot,
        // ahead of requests already queued after the pre-initialization call.
        this._computeTurn = (this._computeTurn ?? 0) + 1;
        try { return run(); } finally { this._computeTurn--; }
      });
      result?.catch(error => { this._deferredError ??= error; });
      return result;
    }
    async computeAsync(nodes, dispatchSize = null) { await this.compute(nodes, dispatchSize); }
    async compileComputeAsync(nodes, onProgress = null) {
      const list = listOf(nodes);
      if (onProgress !== null && typeof onProgress !== 'function') fail('OPTIONS', 'onProgress must be a function');
      await submit(this, () => owner(this).compile(list, onProgress));
    }
    /** F3D extension: copy GPU storage bytes into a normal 2D RenderTarget.
     * The texture can then be sampled by ordinary scene materials. No CPU
     * readback/re-upload or format conversion; row layout is expressed in bytes.
     * x/y are top-left coordinates; flipY reverses rows within the copied region. */
    async copyStorageBufferToRenderTargetAsync(attribute, target, options = {}) {
      check(this);
      if (!attribute || !(attribute.isStorageBufferAttribute || attribute.isStorageInstancedBufferAttribute || attribute.isIndirectStorageBufferAttribute))
        fail('STORAGE', 'Expected a source storage buffer attribute');
      if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected copy options');
      const destination = this._targets.capture(target), copiedOptions = {...options};
      return await submit(this, () => owner(this).copyToTarget(attribute, destination, copiedOptions));
    }
    async getArrayBufferAsync(attribute) { return await submit(this, () => owner(this).read(attribute)); }
    async waitForGPU() {
      await super.waitForGPU();
      const value = owners.get(this); if (value) await value.whenIdle();
      await this._storageTextures?.whenIdle();
    }
    dispose() { owners.get(this)?.dispose(); this._storageTextures?.dispose(); super.dispose(); }
  };
}
