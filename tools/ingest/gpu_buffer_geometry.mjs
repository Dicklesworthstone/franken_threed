/** Mutable CPU-authored BufferGeometry residency for the explicit GPU renderer.
 *
 * Float32 position/normal/tangent/uv/RGB(A) streams and Uint16/Uint32 indices are
 * borrowed from ordinary Three.js BufferGeometry objects. Interleaved attributes
 * share one upload history, GPU allocation and vertex slot. No vertex repacking,
 * numerical conversion, source-array replacement or compute pass is performed.
 *
 * update() observes the pinned WebGLAttributes upload contract, NOT arbitrary
 * CPU edits: first use uploads all bytes without clearing ranges; later uploads
 * require a larger version, coalesce ranges exactly as r186, clear ranges before
 * onUpload, and acknowledge the version AFTER the callback. Four-byte padding
 * comes from the last GPU-visible shadow. A callback may throw after writes;
 * those writes stay visible and the version stays unacknowledged, as upstream.
 *
 * Explicit API, not a WebGLRenderer replacement or arbitrary-object optimizer.
 * Geometry shape/range validation errors are recoverable. Native errors/loss are
 * terminal. Submit every draw using a version BEFORE the next update(); recording
 * without submission does not capture GPU buffer contents. whenIdle() drains
 * uploads and errors; it does not imply that external draws were submitted.
 *
 * maxBytes bounds GPU allocations and, separately, byte-for-byte CPU shadows.
 * maxAttributes bounds retained source identities (including replaced ones).
 * Geometry.dispose() releases residency but permits update() to recreate it;
 * handle.dispose() permanently closes this adapter, never the source or device.
 */
export class BufferGeometryGpuError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'BufferGeometryGpuError'; this.code = code; }
}
const fail = (code, message) => { throw new BufferGeometryGpuError('GEOMETRY_GPU_' + code, message); };
const states = new WeakMap();
const FIELDS = Object.freeze({position: [0, 3], normal: [1, 3], tangent: [2, 4], uv: [3, 2], color: [4, 0]});
const align4 = n => Math.ceil(n / 4) * 4;
function integer(n, min, max, name) {
  if (!Number.isSafeInteger(n) || n < min || n > max) fail('SHAPE', `Invalid ${name}`);
  return n;
}
function storage(array) {
  if (!ArrayBuffer.isView(array) || array instanceof DataView ||
      !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable) fail('STORAGE', 'Expected a fixed, unshared typed array');
  try { new Uint8Array(array.buffer, 0, 0); } catch { fail('STORAGE', 'Detached attribute storage'); }
}
function ownerOf(attribute) { return attribute.isInterleavedBufferAttribute ? attribute.data : attribute; }
function describe(geometry) {
  if (geometry?.isInstancedBufferGeometry || Object.values(geometry?.morphAttributes ?? {}).some(a => a.length))
    fail('SHAPE', 'Instancing and morph deformation require their existing GPU paths');
  const attributes = geometry?.attributes;
  if (!attributes || !attributes.position) fail('SHAPE', 'Geometry requires a position attribute');
  const owners = new Map();
  const vertexCount = integer(attributes.position.count, 0, 0xffffffff, 'vertex count');
  // Source insertion order is retained for upload callbacks, not shader slots.
  for (const name of Object.keys(attributes)) {
    if (!Object.hasOwn(FIELDS, name)) fail('SHAPE', `Unsupported geometry attribute: ${name}`);
    const attribute = attributes[name], owner = ownerOf(attribute), array = owner?.array;
    storage(array);
    if (attribute.isInstancedBufferAttribute || owner.isInstancedInterleavedBuffer) fail('SHAPE', 'Instance streams require the instancing path');
    if (!(array instanceof Float32Array) || attribute.normalized || attribute.isFloat16BufferAttribute) {
      fail('FORMAT', 'Vertex streams require non-normalized Float32 attributes');
    }
    const [location, width] = FIELDS[name];
    const itemSize = integer(attribute.itemSize, 2, 4, 'item size');
    if (width ? itemSize !== width : itemSize !== 3 && itemSize !== 4) fail('SHAPE', `Invalid ${name} width`);
    const stride = attribute.isInterleavedBufferAttribute ? integer(owner.stride, itemSize, 512, 'stride') : itemSize;
    const offset = attribute.isInterleavedBufferAttribute ? integer(attribute.offset, 0, stride - itemSize, 'attribute offset') : 0;
    const count = integer(attribute.count, vertexCount, 0xffffffff, 'attribute count');
    if (count * stride > array.length) fail('SHAPE', 'Attribute count exceeds storage');
    let entry = owners.get(owner);
    if (!entry) owners.set(owner, entry = {owner, array, stride, attributes: [], requiredBytes: 0});
    if (entry.stride !== stride) fail('SHAPE', 'Shared attributes require one stride');
    entry.requiredBytes = Math.max(entry.requiredBytes, vertexCount ? ((vertexCount - 1) * stride + offset + itemSize) * 4 : 0);
    entry.attributes.push({shaderLocation: location, offset: offset * 4, format: 'float32x' + itemSize});
  }
  const index = geometry.index ?? null;
  let indexOwner = null, indexFormat = null, indexCount = 0;
  if (index) {
    if (index.isInterleavedBufferAttribute || index.itemSize !== 1 || index.normalized) fail('SHAPE', 'Expected scalar non-interleaved indices');
    indexOwner = index;
    storage(index.array);
    if (!(index.array instanceof Uint16Array) && !(index.array instanceof Uint32Array)) fail('FORMAT', 'Indices require Uint16Array or Uint32Array');
    indexFormat = index.array instanceof Uint16Array ? 'uint16' : 'uint32';
    indexCount = integer(index.count, 0, index.array.length, 'index count');
    if (owners.has(index)) fail('SHAPE', 'Index and vertex source identities must be distinct');
    owners.set(index, {owner: index, array: index.array, stride: 0, attributes: [], requiredBytes: indexCount * index.array.BYTES_PER_ELEMENT});
  }
  const streams = [...owners.values()].filter(e => e.stride !== 0);
  // Stable layouts do not depend on object ids, buffer generations or insertion
  // order of source properties. Canonicalize only binding slots, not callbacks.
  streams.sort((a, b) => Math.min(...a.attributes.map(x => x.shaderLocation)) - Math.min(...b.attributes.map(x => x.shaderLocation)));
  const layouts = streams.map(e => Object.freeze({arrayStride: e.stride * 4, stepMode: 'vertex',
    attributes: Object.freeze(e.attributes.sort((a,b) => a.shaderLocation-b.shaderLocation).map(Object.freeze))}));
  const channels = Object.freeze({normal: !!attributes.normal, tangent: !!attributes.tangent,
    uv: !!attributes.uv, colorSize: attributes.color?.itemSize ?? 0});
  return {owners, streams, layouts: Object.freeze(layouts), signature: JSON.stringify(layouts), channels,
    vertexCount, indexOwner, indexFormat, indexCount};
}
function range(geometry, extent) {
  const source = geometry.drawRange ?? {start: 0, count: Infinity};
  const start = integer(source.start, 0, Number.MAX_SAFE_INTEGER, 'draw-range start');
  const count = source.count === Infinity ? Infinity : integer(source.count, 0, Number.MAX_SAFE_INTEGER, 'draw-range count');
  const first = Math.min(start, extent);
  return Object.freeze({first, count: Math.min(count, extent - first)});
}
function checkRanges(owner, length) {
  if (!Array.isArray(owner.updateRanges) || typeof owner.clearUpdateRanges !== 'function' || typeof owner.onUploadCallback !== 'function') {
    fail('SHAPE', 'Expected BufferAttribute upload methods and ranges');
  }
  for (const r of owner.updateRanges) {
    integer(r?.start, 0, length, 'range start');
    integer(r?.count, 0, length - r.start, 'range count');
  }
}
// Deliberately the pinned source's +1, not an idealized half-open-range merge.
function mergeRanges(ranges) {
  ranges.sort((a,b) => a.start-b.start);
  let mergeIndex = 0;
  for (let i=1; i<ranges.length; i++) {
    const previous = ranges[mergeIndex], current = ranges[i];
    if (current.start <= previous.start + previous.count + 1) {
      previous.count = Math.max(previous.count, current.start + current.count - previous.start);
    } else ranges[++mergeIndex] = current;
  }
  ranges.length = mergeIndex + 1;
}

/** Internal renderer handshake; ordinary deformers return null. No public shape
 * duck-typing admits fake buffers or a foreign device into the mutable path. */
export function bufferGeometrySnapshot(handle, device) {
  const state = states.get(handle);
  if (!state) return null;
  if (state.device !== device) fail('DEVICE', 'Geometry belongs to a different device');
  state.live();
  const snapshot = state.current();
  if (!snapshot) fail('RELEASED', 'Call geometry.update() after releasing residency');
  return {...snapshot, drawRange: range(state.geometry, snapshot.indexBuffer ? snapshot.indexCount : snapshot.vertexCount)};
}

export function createGpuBufferGeometry(device, geometry, {
  maxBytes = 64 * 1024 * 1024, maxAttributes = 128, label = 'f3d-buffer-geometry',
} = {}) {
  integer(maxBytes, 1, Number.MAX_SAFE_INTEGER, 'byte budget');
  integer(maxAttributes, 1, 65536, 'attribute budget');
  if (typeof label !== 'string' || !device?.limits || typeof device.createBuffer !== 'function' ||
      typeof device.queue?.writeBuffer !== 'function' || typeof device.queue.onSubmittedWorkDone !== 'function' ||
      typeof device.pushErrorScope !== 'function' || typeof device.popErrorScope !== 'function' || typeof device.lost?.then !== 'function') {
    fail('DEVICE', 'Lend a live WebGPU device');
  }
  const maximum = integer(device.limits.maxBufferSize, 4, Number.MAX_SAFE_INTEGER, 'device buffer limit');
  let records = new Map(), allocatedBytes = 0, disposed = false, terminal = null, busy = false,
    current = null, version = 0, generation = 0, pending = Promise.resolve(), rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const stats = {allocations: 0, uploads: 0, uploadedBytes: 0, updates: 0};
  const worldMatrix = new Float64Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
  function release() {
    for (const record of records.values()) record.buffer.destroy();
    records = new Map(); allocatedBytes = 0; current = null; generation++;
  }
  function stop(error) {
    if (!terminal) { terminal = error; release(); rejectStopped(terminal); }
    return terminal;
  }
  function live() {
    if (disposed) fail('DISPOSED', 'Geometry adapter is disposed');
    if (terminal) throw terminal;
  }
  function native(fn) { try { return fn(); } catch (error) { throw stop(error); } }
  function write(record, array, start, end) {
    const from = start * array.BYTES_PER_ELEMENT, to = end * array.BYTES_PER_ELEMENT;
    if (to === from) return;
    record.shadow.set(new Uint8Array(array.buffer, array.byteOffset + from, to - from), from);
    const offset = Math.floor(from / 4) * 4, limit = align4(to);
    native(() => device.queue.writeBuffer(record.buffer, offset, record.shadow, offset, limit - offset));
    stats.uploads++; stats.uploadedBytes += limit-offset;
  }
  function update() {
    live(); if (busy) fail('REENTRANT', 'Cannot update geometry from an upload callback');
    busy = true;
    let scoped = false;
    try {
      const shape = describe(geometry);
      range(geometry, shape.indexOwner ? shape.indexCount : shape.vertexCount);
      let addedBytes = 0, addedCount = 0;
      // Admission completes before any allocation, queue write, range mutation
      // or callback. An invalid later attribute cannot partly upload the frame.
      for (const {owner, array, requiredBytes} of shape.owners.values()) {
        integer(owner.version, 0, Number.MAX_SAFE_INTEGER, 'upload version');
        const existing = records.get(owner), size = Math.max(4, align4(array.byteLength));
        if (typeof owner.onUploadCallback !== 'function') fail('SHAPE', 'Expected onUploadCallback');
        if (existing) {
          if (existing.version >= owner.version && requiredBytes > existing.logicalBytes) fail('SHAPE', 'Geometry counts exceed resident storage');
          if (existing.elementType !== array.constructor) fail('FORMAT', 'Changing a resident element type requires a new attribute');
          if (existing.version < owner.version) {
            if (existing.logicalBytes !== array.byteLength) fail('RESIZE', 'Resizing a resident attribute is not supported; replace the attribute');
            if (existing.version !== -1) checkRanges(owner, array.length);
          }
        } else {
          if (size > maximum) fail('LIMIT', 'Attribute exceeds maxBufferSize');
          addedCount++; addedBytes += size;
        }
      }
      if (allocatedBytes + addedBytes > maxBytes || records.size + addedCount > maxAttributes) fail('LIMIT', 'Geometry residency budget exceeded');
      native(() => { device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory'); }); scoped = true;
      for (const entry of shape.owners.values()) {
        const {owner} = entry, array = owner.array;
        // A previous onUpload may replace a later view without changing its
        // shape. Read that current view here, as the source uploader does.
        storage(array);
        if (array.constructor !== entry.array.constructor || array.length !== entry.array.length) {
          fail('SHAPE', 'An upload callback changed a later attribute storage shape');
        }
        integer(owner.version, 0, Number.MAX_SAFE_INTEGER, 'upload version');
        let record = records.get(owner);
        if (record && record.version !== -1 && record.version < owner.version) checkRanges(owner, array.length);
        if (!record) {
          const size = Math.max(4, align4(array.byteLength));
          const shadow = new Uint8Array(size);
          record = {buffer: native(() => device.createBuffer({label, size, usage: 4|8|16|32})),
            shadow, logicalBytes: array.byteLength, elementType: array.constructor, version: -1};
          // Own the allocation even when an upload callback throws; retry the
          // initial upload without retaining an orphan native buffer.
          records.set(owner, record); allocatedBytes += size; stats.allocations++;
          write(record, array, 0, array.length);
          owner.onUploadCallback();
          record.version = owner.version;
        } else if (record.version < owner.version) {
          const ranges = owner.updateRanges;
          if (record.version === -1 || ranges.length === 0) write(record, array, 0, array.length);
          else {
            mergeRanges(ranges);
            for (const r of ranges) {
              // WebGL2's explicit length=0 copies the remaining source view.
              write(record, array, r.start, r.count === 0 ? array.length : r.start + r.count);
            }
            owner.clearUpdateRanges();
          }
          owner.onUploadCallback();
          record.version = owner.version;
        }
      }
      const indexRecord = shape.indexOwner ? records.get(shape.indexOwner) : null;
      current = Object.freeze({layouts: shape.layouts, signature: shape.signature, channels: shape.channels,
        vertexBuffers: Object.freeze(shape.streams.map(e => records.get(e.owner).buffer)),
        vertexCount: shape.vertexCount, indexBuffer: indexRecord?.buffer ?? null,
        indexFormat: shape.indexFormat, indexCount: shape.indexCount, generation});
      version++; stats.updates++;
      return handle;
    } finally {
      if (scoped) {
        // Callback exceptions are not native errors. Pop both scopes without
        // yielding and preserve the callback's partial-write/version history.
        const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(values => {
          const error = values.find(Boolean);
          if (error) throw new BufferGeometryGpuError('GEOMETRY_GPU_DEVICE', error.message || 'GPU upload failed');
        }).catch(error => { throw stop(error); });
        pending = Promise.all([pending, errors]).then(() => {}); pending.catch(() => {});
      }
      busy = false;
    }
  }
  const onDispose = () => {
    if (busy) fail('REENTRANT', 'Cannot release geometry from an upload callback');
    if (!disposed && !terminal) release();
  };
  const handle = Object.freeze({
    update,
    get vertexBuffer() { return current?.vertexBuffers[0] ?? null; },
    get vertexCount() { return current?.vertexCount ?? 0; },
    get version() { return version; },
    get poseVersion() { return version; },
    get bufferBytes() { return allocatedBytes; },
    get shadowBytes() { return allocatedBytes; },
    get diagnostics() { return Object.freeze({...stats, generation, residentAttributes: records.size}); },
    get disposed() { return disposed; },
    get failed() { return terminal !== null; },
    worldMatrix,
    async whenIdle() {
      live(); const uploaded = pending;
      try { await Promise.race([Promise.all([uploaded, device.queue.onSubmittedWorkDone()]), stopped]); }
      catch (error) { if (!disposed && !terminal) stop(error); throw error; }
      live();
    },
    release() { live(); onDispose(); },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during upload');
      if (!disposed) {
        disposed = true; geometry.removeEventListener?.('dispose', onDispose); release();
        rejectStopped(new BufferGeometryGpuError('GEOMETRY_GPU_DISPOSED', 'Geometry adapter is disposed'));
      }
    },
  });
  states.set(handle, {device, geometry, live, current: () => current});
  device.lost.then(info => { if (!disposed) stop(new BufferGeometryGpuError('GEOMETRY_GPU_LOST', info?.message || 'Device lost')); },
    error => { if (!disposed) stop(error); });
  try { update(); geometry.addEventListener?.('dispose', onDispose); }
  catch (error) { handle.dispose(); throw error; }
  return handle;
}
