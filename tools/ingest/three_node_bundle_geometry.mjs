/** CPU attribute histories retained by source BundleGroup draws. Every view is
 * immutable after capture; queued packets never install into a newer view.
 * Baselines and retained views share array generations, charged once per copy.
 */
import {snapshotComputeAttributeView} from './gpu_compute_attributes.mjs';

export function createThreeNodeBundleGeometry(three, {maxBytes = 64 * 1024 * 1024, fail} = {}) {
  const reject = (code, message) => {
    if (fail) return fail(code, message);
    const error = new Error(`THREE_NODE_BUNDLE_GEOMETRY_${code}: ${message}`);
    error.code = `THREE_NODE_BUNDLE_GEOMETRY_${code}`; throw error;
  };
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) reject('LIMIT', 'Invalid bundle geometry history budget');
  const owners = new Map(), geometries = new Map(), disposedSources = new WeakSet();
  let allocatedBytes = 0, disposed = false;
  const live = () => { if (disposed) reject('DISPOSED', 'Bundle geometry histories are disposed'); };
  const keep = generation => { generation.refs++; };
  const drop = generation => { if (--generation.refs === 0) allocatedBytes -= generation.array.byteLength; };
  const dropView = view => {
    if (--view.refs !== 0) return;
    if (view.storageView) dropView(view.storageView);
    for (const generation of view.generations) drop(generation);
  };
  function rememberView(view) {
    const variants = view.state.variants, previous = variants.get(view.key);
    if (previous && previous !== view) dropView(previous);
    variants.delete(view.key); variants.set(view.key, view);
    // One source geometry can serve several shader input selections, notably
    // the shared QuadMesh geometry used by scene and output passes. Keep their
    // immutable views warm without retaining unbounded layout/range variants.
    while (variants.size > 8) {
      const [key, oldest] = variants.entries().next().value;
      variants.delete(key); dropView(oldest);
    }
  }
  function lease(view) {
    view.refs++; let released = false;
    return Object.freeze({geometry: view.geometry, bytes: view.bytes,
      // A retired program's wrapper can still name a view retained by a bundle.
      retain() { if (view.refs === 0) reject('RELEASED', 'Bundle geometry view has retired'); return lease(view); },
      refreshUploaded(sameResidency = false) {
        if (view.refs === 0) reject('RELEASED', 'Bundle geometry view has retired');
        const selected = view.selected.map(source => {
          const generation = !source.gpu && owners.get(source.owner)?.generation;
          return !generation || generation === source.generation ||
            sameResidency && generation.residency !== source.generation.residency
            ? source : {...source, generation};
        });
        if (selected.every((source, index) => source === view.selected[index])) return null;
        const latest = view.state.variants.get(view.key);
        if (latest && selected.every((source, index) =>
          source.owner === latest.selected[index].owner && source.generation === latest.selected[index].generation)) {
          rememberView(latest); return lease(latest);
        }
        // Refresh only bytes already uploaded by another source draw. Retain
        // this accepted layout/range, without reading any live CPU attributes.
        const next = makeView(view.state, selected, view.geometry.drawRange,
          view.geometry.vertexCount, view.geometry.isInstancedBufferGeometry, view.key);
        if (latest === view || !latest && view.state.active) {
          rememberView(next);
          return lease(next);
        }
        const result = lease(next); dropView(next); return result;
      },
      refreshStorage(refresh) {
        if (view.refs === 0) reject('RELEASED', 'Bundle geometry view has retired');
        const attributes = {}; let changed = false;
        for (const [name, attribute] of Object.entries(view.geometry.attributes)) {
          const next = refresh(attribute) ?? attribute;
          attributes[name] = next; changed ||= next !== attribute;
        }
        if (!changed) return null;
        const previous = view.storageView;
        if (previous && Object.entries(attributes).every(([name, attribute]) => previous.geometry.attributes[name] === attribute))
          return lease(previous);
        // Only GPU metadata changes. Keep the exact retained CPU generations,
        // ranges and index instead of recapturing live source geometry bytes.
        for (const generation of view.generations) keep(generation);
        const selected = view.selected.map(source => source.gpu
          ? {...source, attribute: attributes[source.name], generation: attributes[source.name]} : source);
        const next = {refs: 1, state: view.state, key: view.key, selected, generations: view.generations, bytes: view.bytes,
          geometry: Object.freeze({...view.geometry, attributes: Object.freeze(attributes)})};
        if (previous) dropView(previous);
        view.storageView = next;
        return lease(next);
      },
      release() { if (!released) { released = true; dropView(view); } }});
  }
  function inspect(attribute) {
    const interleaved = attribute?.isInterleavedBufferAttribute === true;
    const owner = interleaved ? attribute.data : attribute, array = owner?.array;
    if (attribute?.isStorageBufferAttribute || attribute?.isStorageInstancedBufferAttribute ||
        owner?.isStorageBufferAttribute || owner?.isStorageInstancedBufferAttribute) {
      const view = !interleaved && snapshotComputeAttributeView(attribute);
      if (!view) reject('ATTRIBUTE', 'GPU storage attributes require their live storage owner');
      return {gpu: true, attribute: view, owner: attribute,
        shape: [view.count, view.itemSize, view.itemSize, 0, false, view.gpuType,
          false, view.isInstancedBufferAttribute === true, false, 1]};
    }
    if (!ArrayBuffer.isView(array) || array instanceof DataView || !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable ||
        !Number.isSafeInteger(owner.version) || owner.version < 0 ||
        !Number.isSafeInteger(attribute.count) || attribute.count < 0 ||
        !Number.isSafeInteger(attribute.itemSize) || attribute.itemSize < 1 || attribute.itemSize > 16)
      reject('ATTRIBUTE', 'Bundle geometry requires fixed CPU buffer attributes');
    try { new Uint8Array(array.buffer, 0, 0); } catch { reject('ATTRIBUTE', 'Bundle attribute storage is detached'); }
    const stride = interleaved ? owner.stride : attribute.itemSize, offset = interleaved ? attribute.offset : 0;
    if (!Number.isSafeInteger(stride) || stride < attribute.itemSize || !Number.isSafeInteger(offset) || offset < 0 ||
        offset + attribute.itemSize > stride || attribute.count && (attribute.count - 1) * stride + offset + attribute.itemSize > array.length)
      reject('ATTRIBUTE', 'Bundle attribute layout exceeds its source storage');
    // Native source uploads do not observe BufferAttribute.onUploadCallback;
    // the private owner must not inherit or invoke an application callback.
    if (!Array.isArray(owner.updateRanges) || owner.updateRanges.some(range => !Number.isSafeInteger(range?.start) || range.start < 0 ||
        !Number.isSafeInteger(range.count) || range.count < 0 || range.start + range.count > array.length))
      reject('ATTRIBUTE', 'Bundle attribute update ranges must fit their source array');
    return {attribute, owner, array, interleaved, stride, offset, usage: owner.usage,
      shape: [attribute.count, attribute.itemSize, stride, offset, attribute.normalized, attribute.gpuType,
        attribute.isFloat16BufferAttribute === true, attribute.isInstancedBufferAttribute === true,
        owner.isInstancedInterleavedBuffer === true, owner.meshPerAttribute]};
  }
  function capture(geometry, renderId, attributeNames = null) {
    live();
    if (!geometry?.attributes || !Number.isSafeInteger(renderId) || renderId < 0)
      reject('GEOMETRY', 'Bundle geometry capture needs attributes and a logical render id');
    // Geometries.updateAttributes uploads only the RenderObject's shader inputs.
    // An unused attribute must remain fresh until its first selected draw; one
    // selected interleaved view still uploads its complete shared owner.
    const attributes = attributeNames === null ? Object.entries(geometry.attributes)
      : [...new Set(attributeNames)].map(name => [name, geometry.attributes[name]]).filter(([, attribute]) => attribute !== undefined);
    const selection = attributes.map(([name, attribute]) => ({name, ...inspect(attribute)}));
    if (geometry.index) selection.push({name: null, ...inspect(geometry.index)});
    if (selection.some(source => source.gpu && source.name === null))
      reject('ATTRIBUTE', 'Captured node storage views are vertex inputs, not indices');
    const unique = new Map(selection.filter(source => !source.gpu).map(source => [source.owner, source]));
    let addedBytes = 0;
    for (const source of unique.values()) {
      const record = owners.get(source.owner);
      if (record?.generation && (record.type !== source.array.constructor || record.length !== source.array.length))
        reject('CHANGED', 'Replace a bundle attribute when changing its storage layout');
      source.previous = record;
      source.upload = !record?.generation || record.renderId !== renderId &&
        (record.version < source.owner.version || source.owner.usage === three.DynamicDrawUsage);
      if (source.upload) addedBytes += source.array.byteLength;
    }
    // Cached alternatives must not pin superseded uploads against the byte
    // budget. Accepted leases keep their own generations after cache eviction.
    for (const source of unique.values()) if (source.upload && source.previous?.generation) {
      for (const state of source.previous.geometries) for (const [key, view] of state.variants) {
        if (!view.generations.has(source.previous.generation)) continue;
        state.variants.delete(key); dropView(view);
      }
    }
    if (addedBytes > maxBytes - allocatedBytes) reject('LIMIT', 'Bundle geometry histories exceed their byte budget');
    // Validate and reserve the complete selection before consuming any ranges.
    for (const source of unique.values()) {
      let record = source.previous;
      if (!record) {
        record = {generation: null, version: -1, renderId: null, geometries: new Set()}; owners.set(source.owner, record);
      }
      if (source.upload) {
        const first = !record.generation, ranges = source.owner.updateRanges;
        const array = first || ranges.length === 0 ? source.array.slice() : record.generation.array.slice();
        if (!first && ranges.length) {
          for (const {start, count} of ranges) array.set(source.array.subarray(start, start + count), start);
          source.owner.clearUpdateRanges();
        }
        const residency = first ? {} : record.generation?.residency;
        if (record.generation) drop(record.generation);
        record.generation = {array, refs: 1, residency}; allocatedBytes += array.byteLength; record.version = source.owner.version;
        record.type = source.array.constructor; record.length = source.array.length;
      }
      record.renderId = renderId;
    }
    let state = geometries.get(geometry);
    if (!state) {
      let source = geometry;
      const seen = new Set();
      while (source.geometry && source.geometry !== source && !seen.has(source.geometry)) { seen.add(source); source = source.geometry; }
      state = {source, owned: new Set(), variants: new Map(), active: true};
      state.onDispose = () => {
        // r186 Geometries removes its first disposal listener but retains the
        // initialized record. Later uses upload missing residency without
        // rearming disposal, including uses through a newly compiled view.
        disposedSources.add(source);
        state.active = false;
        for (const view of state.variants.values()) dropView(view);
        state.variants.clear();
        // Source disposal invalidates every current attribute, even one this
        // geometry never uploaded but another geometry shares. Inspect only
        // identities here; unused arrays must not acquire an upload history.
        const current = Object.values(source.attributes ?? {}).map(attribute =>
          attribute.isInterleavedBufferAttribute ? attribute.data : attribute);
        if (source.index) current.push(source.index);
        for (const owner of new Set([...state.owned, ...current])) {
          const record = owners.get(owner); if (!record) continue;
          if (record.generation) { drop(record.generation); record.generation = null; }
          record.version = -1; record.renderId = null; record.geometries.delete(state);
          if (!record.geometries.size) owners.delete(owner);
        }
        geometries.delete(geometry); source.removeEventListener?.('dispose', state.onDispose);
      };
      geometries.set(geometry, state);
      if (!disposedSources.has(source)) source.addEventListener?.('dispose', state.onDispose);
    }
    for (const owner of unique.keys()) { state.owned.add(owner); owners.get(owner).geometries.add(state); }
    const selected = selection.map(source => ({...source, generation: source.gpu ? source.attribute : owners.get(source.owner).generation}));
    const range = geometry.drawRange ?? {start: 0, count: Infinity};
    // A vertex-index shader can omit position while source draw counts still
    // come from that attribute. Preserve the count without uploading its bytes.
    const vertexCount = geometry.attributes.position?.count ?? geometry.vertexCount;
    const key = JSON.stringify([selection.map(source => [source.name, ...source.shape]), range.start, range.count,
      geometry.isInstancedBufferGeometry === true, vertexCount]);
    const previous = state.variants.get(key);
    if (previous && selected.every((source, i) => source.owner === previous.selected[i].owner &&
        source.generation === previous.selected[i].generation)) {
      rememberView(previous); return lease(previous);
    }
    const view = makeView(state, selected, range, vertexCount, geometry.isInstancedBufferGeometry === true, key);
    rememberView(view);
    return lease(view);
  }
  function makeView(state, selected, range, vertexCount, instanced, key) {
    const privateOwners = new Map(), capturedAttributes = {}; let index = null;
    for (const source of selected) {
      if (source.gpu) { capturedAttributes[source.name] = source.attribute; continue; }
      const [count, itemSize, stride, offset, normalized, gpuType, half, instanceAttribute, instanceBuffer, divisor] = source.shape;
      let owner = privateOwners.get(source.owner);
      if (!owner) {
        const array = source.generation.array;
        owner = source.interleaved
          ? instanceBuffer ? new three.InstancedInterleavedBuffer(array, stride, divisor ?? 1) : new three.InterleavedBuffer(array, stride)
          : instanceAttribute ? new three.InstancedBufferAttribute(array, itemSize, normalized, divisor ?? 1)
            : new three.BufferAttribute(array, itemSize, normalized);
        owner.count = count; owner.usage = source.usage;
        if (!source.interleaved) { owner.gpuType = gpuType; if (half) owner.isFloat16BufferAttribute = true; }
        privateOwners.set(source.owner, owner);
      }
      const attribute = source.interleaved ? new three.InterleavedBufferAttribute(owner, itemSize, offset, normalized) : owner;
      if (source.interleaved && gpuType !== undefined) attribute.gpuType = gpuType;
      if (source.name === null) index = attribute; else capturedAttributes[source.name] = attribute;
    }
    const generations = new Set(selected.filter(source => !source.gpu).map(source => source.generation));
    for (const generation of generations) keep(generation);
    return {refs: 1, state, key, selected, generations,
      bytes: [...generations].reduce((sum, generation) => sum + generation.array.byteLength, 0),
      geometry: Object.freeze({attributes: Object.freeze(capturedAttributes), index, drawRange: Object.freeze({...range}), vertexCount,
        isInstancedBufferGeometry: instanced,
        addEventListener: (...args) => state.source.addEventListener?.(...args),
        removeEventListener: (...args) => state.source.removeEventListener?.(...args)})};
  }
  return Object.freeze({capture, hasHistory: owner => !!owners.get(owner)?.generation,
    dispose() {
      if (disposed) return; disposed = true;
      for (const state of [...geometries.values()]) state.onDispose();
      for (const record of owners.values()) if (record.generation) drop(record.generation);
      geometries.clear(); owners.clear();
    },
    get diagnostics() { return Object.freeze({allocatedBytes, owners: owners.size, geometries: geometries.size, maxBytes}); }});
}
