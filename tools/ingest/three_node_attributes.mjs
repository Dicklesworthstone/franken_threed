/** Captured source-generated 32-bit vertex streams for native node programs.
 * The pinned Instance/BufferAttribute nodes own their public attributes. Keep
 * their WebGPU version/range history at the logical render boundary, then give
 * the existing geometry uploader private owners containing that history. A
 * queued render can install its bytes without rewriting application attributes.
 * Public attribute inputs instead retain their original owner for the shared
 * ordinary-geometry capture, which owns their queued bytes and upload history.
 */
export function inspectThreeNodeStorageAttribute(three, attribute, reject) {
  const arrayType = attribute?.array?.constructor;
  const scalar = arrayType === Int8Array || arrayType === Int16Array ? Int32Array
    : arrayType === Uint8Array || arrayType === Uint16Array ? Uint32Array : arrayType;
  if (!(attribute instanceof three.BufferAttribute) ||
      !(attribute.isStorageBufferAttribute === true || attribute.isStorageInstancedBufferAttribute === true) ||
      attribute.isInterleavedBufferAttribute || attribute.normalized ||
      ![Float32Array, Int32Array, Uint32Array].includes(scalar) ||
      !(attribute.array.buffer instanceof ArrayBuffer) || attribute.array.buffer.resizable ||
      !Number.isSafeInteger(attribute.itemSize) || attribute.itemSize < 1 || attribute.itemSize > 4 ||
      !Number.isSafeInteger(attribute.count) || attribute.count < 1 ||
      attribute.array.length !== attribute.itemSize * attribute.count ||
      !Number.isSafeInteger(attribute.version) || attribute.version < 0 ||
      attribute.isInstancedBufferAttribute && attribute.meshPerAttribute !== 1)
    reject('ATTRIBUTE', 'Native storage vertex inputs require source f32/integer scalar/vector attributes and instance divisor one');
  // The source uploader promotes narrow integers to 32 bits and gives vec3
  // values a fourth alignment word. Keep one signature across those public
  // layout changes so generated vertices borrow the same shader/compute owner.
  const stride = attribute.itemSize === 3 ? 4 : attribute.itemSize;
  return {gpu: true, owner: attribute, array: attribute.array, interleaved: false,
    instanced: attribute.isInstancedBufferAttribute === true, stride, offset: 0,
    signature: JSON.stringify([scalar.name, attribute.count, stride,
      attribute.isInstancedBufferAttribute === true, attribute.meshPerAttribute])};
}

export function createThreeNodeAttributes(three, {maxBytes = 64 * 1024 * 1024, fail} = {}) {
  const reject = (code, message) => {
    if (fail) return fail(code, message);
    throw new Error(`THREE_NODE_ATTRIBUTES_${code}: ${message}`);
  };
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 4) reject('LIMIT', 'Invalid native attribute history budget');
  const owners = new Map(), geometries = new Map(), views = new WeakMap();
  let allocatedBytes = 0, disposed = false;
  const live = () => { if (disposed) reject('DISPOSED', 'Native attribute histories are disposed'); };
  function shape(attribute, shared = false) {
    if (attribute?.isStorageBufferAttribute || attribute?.isStorageInstancedBufferAttribute)
      return inspectThreeNodeStorageAttribute(three, attribute, reject);
    const interleaved = attribute?.isInterleavedBufferAttribute === true;
    const owner = interleaved ? attribute.data : attribute, array = owner?.array;
    if (![Float32Array, Int32Array, Uint32Array].includes(array?.constructor) || !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable ||
        !Number.isSafeInteger(array.length) || !Number.isSafeInteger(attribute?.count) || attribute.count < 0 ||
        !Number.isSafeInteger(attribute?.itemSize) || attribute.itemSize < 1 || attribute.itemSize > 4 ||
        attribute.normalized || !shared && attribute.gpuType === three.IntType ||
        !Number.isSafeInteger(owner.version) || owner.version < 0)
      reject('ATTRIBUTE', 'Native generated attributes require fixed, unshared Float32, Int32 or Uint32 scalar/vector streams');
    try { new Uint8Array(array.buffer, 0, 0); }
    catch { reject('ATTRIBUTE', 'Native generated attribute storage is detached'); }
    const stride = interleaved ? owner.stride : attribute.itemSize, offset = interleaved ? attribute.offset : 0;
    // Source WebGPU selects interleaved step mode from the owner only. A raw
    // BufferAttributeNode's instanced flag does not promote its plain owner.
    const instanced = interleaved ? owner.isInstancedInterleavedBuffer === true : attribute.isInstancedBufferAttribute === true;
    if (!Number.isSafeInteger(stride) || stride < attribute.itemSize || stride > 512 ||
        !Number.isSafeInteger(offset) || offset < 0 || offset + attribute.itemSize > stride ||
        array.length !== attribute.count * stride || (instanced && owner.meshPerAttribute !== 1))
      reject('ATTRIBUTE', 'Native generated streams require a complete source stride and instance divisor one');
    return {owner, array, interleaved, instanced, stride, offset,
      signature: JSON.stringify([array.constructor.name, array.length, attribute.count, stride, offset, attribute.itemSize, interleaved, instanced, owner.meshPerAttribute])};
  }
  function register(descriptor, geometry, attributes) {
    live();
    if (views.has(descriptor)) reject('ATTRIBUTE', 'Native generated geometry is already registered');
    const sources = attributes.map(({name, attribute, shared = false}) => ({name, attribute, shared, ...shape(attribute, shared)}));
    const ordinaryOwners = new Set(Object.values(geometry.attributes).map(attribute =>
      attribute.isInterleavedBufferAttribute ? attribute.data : attribute));
    if (sources.some(source => !source.gpu && !source.shared && ordinaryOwners.has(source.owner)))
      reject('ATTRIBUTE', 'Native generated streams cannot also be ordinary geometry sources');
    const newOwners = new Map();
    for (const source of sources) {
      if (source.gpu || source.shared) continue;
      const previous = owners.get(source.owner);
      if (previous && previous.bytes !== source.array.byteLength)
        reject('CHANGED', 'Replace a generated attribute when changing its resident storage size');
      if (!previous) newOwners.set(source.owner, source.array.byteLength);
    }
    const addedBytes = [...newOwners.values()].reduce((sum, bytes) => sum + bytes, 0);
    if (addedBytes > maxBytes - allocatedBytes) reject('LIMIT', 'Native generated attribute histories exceed their byte budget');
    let geometryState = geometries.get(geometry);
    if (!geometryState && sources.length) {
      const owned = new Set();
      const onDispose = () => {
        geometries.delete(geometry); geometry.removeEventListener('dispose', onDispose);
        for (const owner of owned) {
          const record = owners.get(owner); if (!record) continue;
          // Source geometry disposal destroys even shared attribute residency.
          // Pending packets retain their old arrays; the next use uploads anew.
          record.snapshot = null; record.version = -1; record.renderId = null;
          record.geometries.delete(geometry);
          if (!record.geometries.size) { owners.delete(owner); allocatedBytes -= record.bytes; }
        }
      };
      geometryState = {owned, onDispose}; geometries.set(geometry, geometryState);
      geometry.addEventListener('dispose', onDispose);
    }
    const privateOwners = new Map(), mapped = {}, gpuSources = [];
    for (const source of sources) {
      if (source.shared) {
        // This is an application-owned input, possibly also selected by an
        // ordinary geometry attribute. Preserve its identity so the shared
        // geometry capture consumes its upload history once at the source
        // boundary. It also owns queued bytes, budgets and disposal.
        mapped[source.name] = source.attribute;
        continue;
      }
      if (source.gpu) {
        // Metadata view only: the storage owner supplies the actual vertex
        // allocation. Do not clone or upload the source CPU simulation array.
        const view = Object.create(source.attribute);
        mapped[source.name] = view; gpuSources.push({...source, view});
        continue;
      }
      let record = owners.get(source.owner);
      if (!record) {
        record = {bytes: source.array.byteLength, geometries: new Set(), snapshot: null, version: -1, renderId: null};
        owners.set(source.owner, record); allocatedBytes += record.bytes;
      }
      record.geometries.add(geometry); geometryState.owned.add(source.owner);
      let target = privateOwners.get(source.owner);
      if (!target) {
        let owner;
        if (source.interleaved) owner = source.instanced
          ? new three.InstancedInterleavedBuffer(source.array, source.stride, 1)
          : new three.InterleavedBuffer(source.array, source.stride);
        else owner = source.instanced
          ? new three.InstancedBufferAttribute(source.array, source.attribute.itemSize, false, 1)
          : new three.BufferAttribute(source.array, source.attribute.itemSize);
        target = {owner, record, source: source.owner, installed: null};
        privateOwners.set(source.owner, target);
      }
      mapped[source.name] = source.interleaved
        ? new three.InterleavedBufferAttribute(target.owner, source.attribute.itemSize, source.offset, false)
        : target.owner;
    }
    const view = {geometry, get attributes() { return {...geometry.attributes, ...mapped}; }, get index() { return geometry.index; },
      get drawRange() { return geometry.drawRange; }, morphAttributes: geometry.morphAttributes,
      isInstancedBufferGeometry: geometry.isInstancedBufferGeometry === true};
    views.set(descriptor, {sources, privateOwners, gpuSources, view});
    return view;
  }
  function stateOf(descriptor) {
    live(); const state = views.get(descriptor);
    if (!state) reject('ATTRIBUTE', 'Native generated geometry is not registered');
    return state;
  }
  function validate(descriptor) {
    const state = stateOf(descriptor);
    // Validate the entire selection before consuming a later stream's ranges.
    for (const source of state.sources) {
      const current = shape(source.attribute, source.shared);
      if (current.owner !== source.owner || current.signature !== source.signature)
        reject('CHANGED', 'Native generated attribute identity or layout changed; compile the graph again');
    }
    for (const {owner: source, gpu} of state.sources) {
      if (gpu) continue;
      if (!Array.isArray(source.updateRanges) || source.updateRanges.some(range =>
        !Number.isSafeInteger(range?.start) || range.start < 0 || !Number.isSafeInteger(range?.count) ||
        range.count < 0 || range.start + range.count > source.array.length))
        reject('ATTRIBUTE', 'Native generated attributes require bounded source update ranges');
    }
    return state;
  }
  function capture(descriptor, renderId, captureVertexAttribute = null) {
    const state = validate(descriptor);
    if (state.gpuSources.length && typeof captureVertexAttribute !== 'function')
      reject('ATTRIBUTE', 'Native storage vertex streams require the renderer storage owner');
    const vertexPackets = state.gpuSources.map(source => captureVertexAttribute(source.attribute, source.view));
    if (!Number.isSafeInteger(renderId) || renderId < 0) reject('ATTRIBUTE', 'Native attribute capture requires a logical render id');
    let bytes = 0;
    const selected = [];
    for (const target of state.privateOwners.values()) {
      const {source, record} = target;
      if (record.renderId !== renderId || !record.snapshot) {
        if (!record.snapshot || record.version < source.version || source.usage === three.DynamicDrawUsage) {
          const first = !record.snapshot, ranges = source.updateRanges;
          const data = first || !ranges.length ? source.array.slice() : record.snapshot.slice();
          if (!first && ranges.length) {
            for (const {start, count} of ranges) data.set(source.array.subarray(start, start + count), start);
            source.clearUpdateRanges();
          }
          record.snapshot = data; record.version = source.version;
        }
        record.renderId = renderId;
      }
      selected.push({target, data: record.snapshot}); bytes += record.bytes;
    }
    return Object.freeze({bytes, install() {
      live();
      for (const packet of vertexPackets) packet.install();
      for (const {target, data} of selected) if (target.installed !== data) {
        target.owner.array = data; target.owner.needsUpdate = true; target.installed = data;
      }
    }});
  }
  function release() {
    // Program replacement alone does not retire the source attribute's GPU
    // history. Weakly held views also remain available to accepted compile-only
    // requests that have not reached their first upload yet. The descriptor's
    // lifetime releases the view; geometry/renderer disposal retires history.
  }
  return Object.freeze({register, capture, release, validate(descriptor) { validate(descriptor); },
    geometry: descriptor => stateOf(descriptor).view,
    storage: descriptor => stateOf(descriptor).gpuSources.map(source => source.attribute),
    byteLength: descriptor => [...stateOf(descriptor).privateOwners.values()].reduce((sum, target) => sum + target.record.bytes, 0),
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const [geometry, state] of geometries) geometry.removeEventListener('dispose', state.onDispose);
      geometries.clear(); owners.clear(); allocatedBytes = 0;
    },
    get diagnostics() { return Object.freeze({owners: owners.size, allocatedBytes, maxBytes}); }});
}
