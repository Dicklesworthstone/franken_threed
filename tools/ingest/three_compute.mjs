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
 * readback or CPU upload is introduced. Sampled image/canvas inputs retain
 * private call-time pixels through the same source texture owner. */
import {createGpuCompute} from './gpu_compute.mjs';
import {captureComputeAttributeView, refreshComputeAttributeView} from './gpu_compute_attributes.mjs';
import {copyComputeBufferToTarget} from './gpu_compute_texture_copy.mjs';
import {createThreeStorageTextures, inspectThreeStorageTexture} from './three_storage_textures.mjs';
import {createGpuThreeTextures} from './three_textures.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';
import {createThreeFramebufferTextures} from './three_framebuffer_textures.mjs';
import {createThreeDepthTextures} from './three_depth_textures.mjs';
import {withStorageTextureCopies} from './three_storage_texture_copy.mjs';
import {assertThreeNativeComputeOutsideCapture, assertThreeNativeTextureWrite} from './three_node_shadows.mjs';

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
  const textureOwner = renderer._getSampledTextureInputs();
  const gpu = createGpuCompute(device, {signal: renderer._targetLifetime.signal});
  const programs = new Map(), initialized = new Set(), listeners = new Map(), building = new Set();
  const captured = new WeakMap(), uploaded = new WeakMap(), payloads = new Set(), compiling = new Set();
  const readLeases = new WeakMap(), readResources = new WeakMap();
  const transfers = new Set();
  let transferFailure = null;
  const maxSnapshotBytes = 128 * 1024 * 1024;
  let disposed = false, uniformVersion = 0, snapshotBytes = 0, issuing = false, capturingReads = false;
  function live() {
    if (disposed || renderer._disposed) fail('DISPOSED', 'Renderer compute execution is disposed');
  }
  function validateAttribute(attribute) {
    if (!(attribute?.isStorageBufferAttribute || attribute?.isStorageInstancedBufferAttribute || attribute?.isIndirectStorageBufferAttribute) ||
        attribute.isInterleavedBufferAttribute || attribute.normalized || !ArrayBuffer.isView(attribute.array) ||
        !Number.isSafeInteger(attribute.itemSize) || attribute.itemSize < 1 || !Number.isSafeInteger(attribute.count) || attribute.count < 1 ||
        attribute.array.length !== attribute.count * attribute.itemSize)
      fail('STORAGE', 'Expected an unnormalized storage attribute with a positive item stride and a complete typed array');
    if (![Float32Array, Int32Array, Uint32Array, Int16Array, Uint16Array, Int8Array, Uint8Array].includes(attribute.array.constructor))
      fail('STORAGE', 'Storage attributes require f32 or integer components');
  }
  function normalize(attribute) {
    live(); validateAttribute(attribute);
    // r186 StructTypeNode supplies its padded stride in 32-bit words. Preserve
    // those bytes, including mixed integer fields and matrix-column padding;
    // a storage item need not be a vertex-sized scalar/vector (H8 uses 20 words).
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
    gpu.releaseUpload(record.admission); record.admission = null;
    snapshotBytes -= record.charged; record.charged = 0; record.data = null; payloads.delete(record);
  }
  function snapshot(key, array, version, uniform, deferred, ranges = []) {
    if (!ArrayBuffer.isView(array) || array instanceof DataView) fail('BUFFER', 'Expected a typed uniform or storage buffer');
    const previous = !uniform && captured.get(key);
    if (previous?.source === array && previous.version === version && (previous.data || uploaded.get(key) === previous)) return previous;
    const charged = deferred ? array.byteLength : 0;
    if (charged > maxSnapshotBytes - snapshotBytes) fail('BUDGET', 'Queued compute uploads exceed the snapshot byte budget');
    const data = deferred ? new array.constructor(array) : array;
    const admission = !uniform && deferred ? gpu.captureUpload(key, data, version) : null;
    const record = {key, source: array, byteLength: array.byteLength, version, uniform, data, charged, ranges, admission, refs: 0};
    snapshotBytes += charged; payloads.add(record);
    if (!uniform) captured.set(key, record);
    return record;
  }
  function deferUpload(record) {
    if (!record.data || record.charged) return;
    const charged = record.data.byteLength;
    if (charged > maxSnapshotBytes - snapshotBytes) fail('BUDGET', 'Queued compute uploads exceed the snapshot byte budget');
    const data = new record.data.constructor(record.data);
    const admission = record.uniform ? null : gpu.captureUpload(record.key, data, record.version);
    record.data = data; record.admission = admission; record.charged = charged;
    snapshotBytes += charged;
  }
  function upload(record) {
    const {key, version, uniform, source, ranges} = record;
    if (uploaded.get(key) === record) {
      const buffer = gpu.getBuffer(key);
      if (buffer) return buffer;
    }
    if (!record.data) fail('STALE', 'A consumed upload snapshot cannot recreate released storage');
    const buffer = gpu.upload(key, record.data, version, {uniform, ranges, admission: record.admission});
    uploaded.set(key, record);
    if (!uniform) {
      // A later application update may already have new ranges. Acknowledging
      // this queued version must not erase the newer version's upload request.
      if (key.array === source && (key.version ?? 0) === version) key.clearUpdateRanges?.();
      key.onUploadCallback?.();
    }
    return buffer;
  }
  function attributeRecord(attribute, deferred, indirect = false) {
    // Ordinary indirect BufferAttributes retain their raw words. Storage-backed
    // inputs retain the source backend's shared normalization (including vec3
    // padding), so compute and drawing observe the same allocation/layout.
    if (indirect && (!(attribute instanceof three.BufferAttribute) || attribute.isInterleavedBufferAttribute ||
        attribute.normalized || ![Uint32Array, Int32Array].includes(attribute.array?.constructor) ||
        !Number.isSafeInteger(attribute.itemSize) || attribute.itemSize < 1 ||
        !Number.isSafeInteger(attribute.count) || attribute.count < 1 ||
        attribute.array.length !== attribute.itemSize * attribute.count))
      fail('INDIRECT', 'Indirect drawing requires a complete unnormalized source 32-bit integer buffer attribute');
    const plainIndirect = indirect && !attribute.isStorageBufferAttribute && !attribute.isStorageInstancedBufferAttribute;
    const array = plainIndirect ? attribute.array : normalize(attribute), version = attribute.version ?? 0;
    const ranges = (attribute.updateRanges ?? []).map(({start, count}) => ({start, count}));
    if (!Number.isSafeInteger(version) || version < 0 || ranges.some(({start, count}) =>
      !Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count < 0 || start + count > array.length))
      fail('STORAGE', 'Expected a nonnegative storage version and in-bounds update ranges');
    return snapshot(attribute, array, version, false, deferred, ranges);
  }
  function readRecord(lease) {
    live();
    const value = readLeases.get(lease);
    if (!value || value.released) fail('STALE', 'Native graphics storage lease is foreign or released');
    return value.record;
  }
  function validateReads(leases) {
    live();
    const versions = new Map(), accesses = new Map();
    const access = (attribute, writable) => {
      const previous = accesses.get(attribute);
      if (previous !== undefined && previous !== writable)
        fail('STORAGE_CONFLICT', 'One native render pass cannot combine writable storage with a read-only or vertex/index use of the same attribute');
      accesses.set(attribute, writable);
    };
    for (const lease of leases) {
      const record = readRecord(lease), previous = versions.get(record.key);
      // One graphics pass cannot bind two queue-upload histories of the same
      // allocation. Reject before uploading either: otherwise both draws could
      // observe the later value. Independent render calls have FIFO boundaries.
      if (previous && previous !== record)
        fail('READ_CONFLICT', 'One native render cannot read conflicting upload versions of a storage attribute');
      if (!record.data && (uploaded.get(record.key) !== record || !gpu.getBuffer(record.key)))
        fail('STALE', 'Native graphics storage residency was released before its captured use');
      access(record.key, readLeases.get(lease).writable);
      versions.set(record.key, record);
    }
    return versions.values();
  }
  function admittedRead(previous) {
    // Retained bindings observe uploads admitted before this logical render.
    // Never scan current CPU data or select a later queued version at submit.
    const pending = captured.get(previous.key), resident = uploaded.get(previous.key);
    return pending && (pending.data || pending === resident) ? pending : resident ?? previous;
  }
  const storageReads = Object.freeze({
    captureDispatch(run) {
      live();
      if (capturingReads || issuing) fail('REENTRY', 'Cannot reenter native storage capture or upload');
      capturingReads = true;
      try { return run(); } finally { capturingReads = false; }
    },
    capture(attribute, {writable = false, indirect = false} = {}) {
      live();
      if (issuing) fail('REENTRY', 'Cannot capture native storage from an upload callback');
      if (typeof writable !== 'boolean' || typeof indirect !== 'boolean' || writable && indirect)
        fail('STORAGE', 'Native storage access must be a captured boolean and indirect use must be read-only');
      const record = attributeRecord(attribute, true, indirect), lease = Object.freeze({});
      record.refs++; readLeases.set(lease, {record, writable, indirect, released: false});
      return lease;
    },
    refreshIndirect(lease) {
      const previous = readRecord(lease);
      if (!readLeases.get(lease).indirect) fail('INDIRECT', 'Expected a captured indirect argument use');
      // Reused source bindings still read the current GPU allocation. Select
      // only an upload already admitted before this render: never scan current
      // CPU data, and never resolve a later queued compute version at submit.
      const record = admittedRead(previous);
      if (record.byteLength !== previous.byteLength)
        fail('INDIRECT', 'A reused indirect command cannot replace its captured buffer allocation size');
      const next = Object.freeze({});
      record.refs++; readLeases.set(next, {record, writable: false, indirect: true, released: false});
      return next;
    },
    refreshStorage(lease) {
      const previous = readRecord(lease), record = admittedRead(previous);
      if (record === previous) return null;
      if (record.byteLength !== previous.byteLength)
        fail('STORAGE', 'A retained storage binding cannot replace its captured buffer allocation size');
      const value = readLeases.get(lease), next = Object.freeze({});
      record.refs++; readLeases.set(next, {record, writable: value.writable, indirect: value.indirect, released: false});
      return next;
    },
    captureIndirectAlias(lease, indirectLease) {
      const previous = readRecord(lease), record = readRecord(indirectLease);
      if (!readLeases.get(indirectLease).indirect) fail('INDIRECT', 'Expected a captured indirect argument use');
      if (previous.key !== record.key) return null;
      if (record.byteLength !== previous.byteLength)
        fail('INDIRECT', 'A reused indirect storage alias cannot replace its captured buffer allocation size');
      // Retained shader/vertex bindings alias the same source GPU allocation.
      // Follow this draw's admitted indirect history, never a later CPU edit
      // or queued dispatch. Preserve access so writable aliases still fail.
      const value = readLeases.get(lease), next = Object.freeze({});
      record.refs++; readLeases.set(next, {record, writable: value.writable, indirect: value.indirect, released: false});
      return next;
    },
    validate: validateReads,
    vertexAttribute(lease, view) {
      const record = readRecord(lease);
      return captureComputeAttributeView(device, record.key, view);
    },
    refreshVertexAttribute(lease, view) {
      const record = readRecord(lease);
      return refreshComputeAttributeView(device, record.key, view, record.version);
    },
    withReads(leases, run) {
      const records = [...validateReads(leases)];
      if (issuing || capturingReads) fail('REENTRY', 'Cannot submit native storage during capture or upload');
      issuing = true;
      try {
        for (const record of records) {
          const buffer = upload(record);
          let resource = readResources.get(buffer);
          if (!resource) { resource = Object.freeze({buffer}); readResources.set(buffer, resource); }
          record.resource = resource;
        }
        return run();
      } finally { issuing = false; }
    },
    resolve(lease) {
      const record = readRecord(lease);
      if (!issuing || !record.resource) fail('STORAGE', 'Resolve native storage only during its submitted read interval');
      return record.resource;
    },
    release(lease) {
      const value = readLeases.get(lease);
      if (!value || value.released) return;
      value.released = true;
      if (--value.record.refs === 0) free(value.record);
    },
  });
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
    const texture = binding.textureNode ? binding.textureNode.value : binding.texture;
    return {texture,
      isSampler: binding.isSampler, isSampledTexture: binding.isSampledTexture,
      comparison: texture?.isDepthTexture === true && texture.compareFunction !== null &&
        binding.textureNode?.isSampleCompare?.() === true && renderer.hasCompatibility(three.Compatibility.TEXTURE_COMPARE),
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
  class NativeComputeNodeBuilder extends three.WGSLNodeBuilder {
    getUniformFromNode(node, ...args) {
      // TextureNode asks for its uniform before selecting array coordinates.
      // Prepare target-owned depth metadata here so even Fn-created textures
      // use the effective source dimension on their first compute-only use.
      if (node.value?.isDepthTexture === true && node.value.renderTarget != null)
        textureOwner.sampleData(node.value);
      return super.getUniformFromNode(node, ...args);
    }
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
      const builder = new NativeComputeNodeBuilder(null, renderer);
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
    const previous = Object.fromEntries(['renderer', 'compute', 'renderId', 'scene', 'object', 'camera', 'material']
      .map(key => [key, frame[key]]));
    Object.assign(frame, {renderer, compute: node, renderId: renderer.info.calls, scene: null, object: null, camera: null, material: null});
    try { return run(); } finally { Object.assign(frame, previous); }
  }
  function update(entry, stage) {
    const names = {before: ['updateBeforeNodes', 'updateBeforeNode'], current: ['updateNodes', 'updateNode'], after: ['updateAfterNodes', 'updateAfterNode']};
    const [nodes, method] = names[stage];
    // NodeManager resets context for each before/after node. Ordinary compute
    // updates use getNodeFrame(), whose compute and render objects are null.
    const context = {renderer, compute: stage === 'current' ? null : entry.builder.compute,
      scene: null, object: null, camera: null, material: null};
    if (stage === 'current') Object.assign(frame, context);
    for (const node of entry.builder[nodes] ?? []) {
      if (stage !== 'current') Object.assign(frame, context);
      frame[method](node);
    }
  }
  function capture(node, entry, override, deferred) {
    // Validate live binding identities before capturing any upload histories.
    // onInit kernels enter this same boundary independently of their parent.
    const resources = entry.groups.flatMap(group => group.bindings.map(binding => binding.isStorageBuffer
      ? binding.attribute : binding.isSampledTexture || binding.isSampler ? textureBinding(binding).texture : null)).filter(Boolean);
    const dispatch = override ?? node.dispatchSize ?? node.count;
    if (dispatch?.isIndirectStorageBufferAttribute) resources.push(dispatch);
    assertThreeNativeComputeOutsideCapture(renderer, resources);
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
    try {
      const groups = entry.groups.map((group, index) => ({index, entries: group.bindings.map((binding, slot) => {
        if (binding.isSampledTexture || binding.isSampler) {
          binding.update?.();
          const texture = captureTexture(binding);
          textureUses.push(texture);
          return {binding: slot, texture, layout: {binding: slot, visibility: 4, ...textureOwner.layout(texture)}};
        }
        let record;
        if (binding.isStorageBuffer) record = hold(attributeRecord(binding.attribute, deferred));
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
      if (size?.isIndirectStorageBufferAttribute) indirect = hold(attributeRecord(size, deferred));
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
      const preparation = textureOwner.preparation(textureUses);
      // Decide from the captured payloads: an unchanged alias can retain a
      // pending HTML snapshot after its live source becomes a canvas. Every
      // upload in this call must survive the image-readiness boundary too.
      if (preparation && !deferred) for (const record of held) deferUpload(record);
      const issue = (dispatch = true) => {
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
      };
      return {release, shader, asynchronous: preparation !== null,
        validate: () => textureOwner.validateDispatch(textureUses),
        issue(dispatch = true) { return preparation ? preparation.then(() => issue(dispatch)) : issue(dispatch); }};
    } catch (error) { release(); throw error; }
  }
  function compute(list, override, schedule, deferred) {
    live();
    assertThreeNativeComputeOutsideCapture(renderer);
    if (issuing || capturingReads) fail('REENTRY', 'Cannot compute from an upload callback or native storage capture');
    renderer.info.calls++; renderer.info.compute.calls++; renderer.info.compute.frameCalls++;
    renderer.inspector?.beginCompute?.(`compute:${renderer.info.compute.frameCalls}:f${renderer.info.frame}`, list);
    const pending = [];
    try {
      for (const node of list) {
        const entry = program(node);
        withFrame(node, () => {
          update(entry, 'before'); update(entry, 'current');
          textureOwner.captureDispatch(() => {
            const request = capture(node, entry, override, deferred());
            let result;
            try { result = schedule(() => request.issue(), request.asynchronous); }
            catch (error) { request.release(); throw error; }
            if (result?.then) {
              const done = Promise.resolve(result).finally(request.release);
              // A later source callback may throw before we return this promise.
              done.catch(() => {}); pending.push(done);
            } else request.release();
          });
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
        return textureOwner.captureDispatch(() => capture(node, entry, null, false));
      });
      try {
        await request.issue(false);
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
  return {compute, gpu, storageReads,
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
      listeners.clear(); initialized.clear(); programs.clear(); gpu.dispose();
    }};
}

/** All GPU work joins the renderer FIFO. Source updates and upload snapshots
 * are taken at the initialized call, not replayed when preparation finishes.
 * Before init(), source preparation follows upstream's asynchronous boundary. */
export function withThreeCompute(Base, THREE) {
  Base = withStorageTextureCopies(Base);
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
  function createTurn() {
    return {pending: null, submit(issue, awaitSubmission = false) {
      if (!this.pending) {
        const result = issue();
        if (awaitSubmission && result?.then) {
          this.pending = Promise.resolve(result); this.pending.catch(() => {});
        }
        return result;
      }
      let resolve, reject;
      const result = new Promise((a, b) => { resolve = a; reject = b; });
      // Completion can include a readback map or GPU fence. Only readiness
      // before submission belongs to the turn's ordering chain.
      this.pending = this.pending.then(() => {
        const value = issue(); resolve(value);
        return awaitSubmission ? value : undefined;
      });
      this.pending.catch(reject);
      return result;
    }};
  }
  function submit(renderer, issue, awaitSubmission = false, animationBarrier = false) {
    check(renderer);
    if (renderer._computeTurn) return renderer._computeTurn.submit(issue, awaitSubmission);
    if (!awaitSubmission && renderer._initialized && !renderer._drain && !renderer._preparing) return issue();
    let resolve, reject;
    const pending = new Promise((a, b) => { resolve = a; reject = b; });
    // Reserve outside the Promise executor: a full FIFO is a synchronous
    // capture failure, not an accepted texture version with a rejected promise.
    renderer._enqueue({transfer: true, awaitSubmission, animationBarrier, issue, resolve, reject});
    return pending;
  }
  return class WebGPURenderer extends Base {
    constructor(...args) {
      super(...args);
      if (!this.isF3DRenderer) return; // Preserve an exact-backend constructor return.
      if (THREE.TSL?.context && !this.contextNode) this.contextNode = THREE.TSL.context();
      // r186's builder reads this configuration for every declaration, even
      // with keyword diagnostics disabled. Preserve application-supplied flags.
      this.debug ??= {};
      this.debug.checkShaderErrors ??= true;
      this.debug.diagnostics ??= {};
      this.debug.diagnostics.keywords ??= false;
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
          if (!texture?.isStorageTexture) return this._getSampledTextureInputs().sampleData(texture);
          inspectThreeStorageTexture(texture, THREE);
          return {samples: 1, primarySamples: 1, isMSAA: false};
        };
        this._backend.hasFeature = name => !!this._device?.features?.has(name);
        // Match the pinned WebGPU backend's depth comparison compatibility
        // policy. TextureNode retains its source shader fallback on Android.
        const textureCompare = typeof navigator === 'undefined' || !/Android/.test(navigator.userAgent);
        this._backend.hasCompatibility = name => name === THREE.Compatibility.TEXTURE_COMPARE && textureCompare;
      }
    }
    hasCompatibility(name) {
      if (!this._initialized) throw new Error('THREE.Renderer: .hasCompatibility() called before the backend is initialized. Use "await renderer.init();" before using this method.');
      return this._backend.hasCompatibility(name);
    }
    /** Native graphics and compute observe the same captured texture versions.
     * This owner reuses the existing uploader and borrows target/storage images;
     * shader compilation does not initialize a compute pipeline or executor. */
    _getSampledTextureInputs() {
      // The owned texture adapter also serves WebGL-authored framebuffer maps;
      // public compute entry points still enforce the native renderer surface.
      if (this._shaderEncodedOutput) { if (this._disposed) fail('DISPOSED', 'Renderer is disposed'); }
      else check(this);
      if (!this._device) fail('DEVICE', 'Initialize the renderer before capturing sampled texture inputs');
      return this._sampledTextureInputs ??= createThreeComputeTextures({storage: this._storageTextures,
        poolFactory: ({imageSnapshots, mediaSnapshots}) => createGpuThreeTextures(this._device, {three: THREE, imageSnapshots, mediaSnapshots,
          nativeCube: this._shaderEncodedOutput !== true, nativeMedia: this._shaderEncodedOutput !== true}),
        targetFactory: () => createThreeComputeTargets({targets: this._targets, device: this._device,
          signal: this._targetLifetime.signal}),
        framebufferFactory: () => createThreeFramebufferTextures({three: THREE, device: this._device,
          getFormat: () => this._framebufferFormat(), signal: this._targetLifetime.signal, webgl: this._shaderEncodedOutput === true,
          submitInitialization: request => {
            if (this._programDrawContext) {
              this._programDrawContext.flush();
              try { request.issue(); } finally { this._releaseRequest(request); }
              return;
            }
            if (this._webglSubmissionDepth > 0 || !this._drain && !this._preparing && !this._pending.length) {
              try { request.issue(); } finally { this._releaseRequest(request); }
            } else this._enqueue(request);
          }}),
        depthFactory: () => createThreeDepthTextures({three: THREE, device: this._device,
          getSamples: () => this.currentSamples, signal: this._targetLifetime.signal})});
    }
    /** Graphics leases share compute's authoritative allocation and upload
     * history. An unchanged host version never overwrites GPU-produced bytes. */
    _getStorageInputs() {
      check(this);
      if (!this._device) fail('DEVICE', 'Initialize the renderer before capturing storage inputs');
      return owner(this).storageReads;
    }
    compute(nodes, dispatchSize = null) {
      check(this);
      assertThreeNativeComputeOutsideCapture(this);
      const list = listOf(nodes), size = Array.isArray(dispatchSize) ? dispatchSize.slice() : dispatchSize;
      const run = () => owner(this).compute(list, size, (issue, asynchronous) => submit(this, issue, asynchronous, true),
        () => this._computeTurn ? !!this._computeTurn.pending : !!(this._drain || this._preparing));
      const result = this._initialized ? run() : submit(this, () => {
        // Nested onInit/before/after compute calls belong to this submit slot,
        // ahead of requests already queued after the pre-initialization call.
        const previous = this._computeTurn, turn = this._computeTurn = createTurn();
        try {
          const value = run(); value?.catch(() => {});
          return turn.pending ? turn.pending.then(() => value) : value;
        } catch (error) {
          // Earlier accepted kernels still own their submission position when
          // a later source callback rejects this public call.
          if (turn.pending) return turn.pending.then(() => { throw error; }, () => { throw error; });
          throw error;
        } finally { this._computeTurn = previous; }
      }, true, true);
      result?.catch(error => { this._deferredError ??= error; });
      return result;
    }
    async computeAsync(nodes, dispatchSize = null) { await this.compute(nodes, dispatchSize); }
    async compileComputeAsync(nodes, onProgress = null) {
      assertThreeNativeComputeOutsideCapture(this);
      const list = listOf(nodes);
      if (onProgress !== null && typeof onProgress !== 'function') fail('OPTIONS', 'onProgress must be a function');
      await submit(this, () => owner(this).compile(list, onProgress), true);
    }
    /** F3D extension: copy GPU storage bytes into a normal 2D RenderTarget.
     * The texture can then be sampled by ordinary scene materials. No CPU
     * readback/re-upload or format conversion; row layout is expressed in bytes.
     * x/y are top-left coordinates; flipY reverses rows within the copied region. */
    async copyStorageBufferToRenderTargetAsync(attribute, target, options = {}) {
      check(this);
      assertThreeNativeTextureWrite(this, target?.texture);
      if (!attribute || !(attribute.isStorageBufferAttribute || attribute.isStorageInstancedBufferAttribute || attribute.isIndirectStorageBufferAttribute))
        fail('STORAGE', 'Expected a source storage buffer attribute');
      if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected copy options');
      const destination = this._targets.capture(target), copiedOptions = {...options};
      if (destination.shape.cube) fail('TARGET', 'Cube target buffer copies require explicit face-aware support');
      return await submit(this, () => owner(this).copyToTarget(attribute, destination, copiedOptions));
    }
    async getArrayBufferAsync(attribute) { return await submit(this, () => owner(this).read(attribute)); }
    async waitForGPU() {
      await super.waitForGPU();
      const value = owners.get(this); if (value) await value.whenIdle();
      await this._sampledTextureInputs?.whenIdle();
      await this._storageTextures?.whenIdle();
    }
    dispose() { owners.get(this)?.dispose(); this._sampledTextureInputs?.dispose(); this._storageTextures?.dispose(); super.dispose(); }
  };
}
