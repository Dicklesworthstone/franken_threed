/** Pinned wireframe edge-index histories at the native geometry-update boundary.
 * Source attributes stay public and unchanged. Each program view has one private
 * index owner; accepted packets install immutable arrays before GPU preparation.
 */
export function createThreeNodeWireframes(three, {maxBytes = 64 * 1024 * 1024, fail} = {}) {
  const reject = (code, message) => {
    if (fail) return fail(code, message);
    const error = new Error(`THREE_NODE_WIREFRAME_${code}: ${message}`);
    error.code = `THREE_NODE_WIREFRAME_${code}`; throw error;
  };
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) reject('LIMIT', 'Invalid native wireframe history budget');
  const geometries = new Map(), views = new WeakMap();
  let allocatedBytes = 0, disposed = false;
  const live = () => { if (disposed) reject('DISPOSED', 'Native wireframe histories are disposed'); };
  const sourceOf = geometry => {
    const position = geometry.attributes?.position, index = geometry.index ?? null;
    if (!position) reject('GEOMETRY', 'Native wireframes require source positions');
    const owner = index ?? position;
    return {position, index, owner, version: owner.version};
  };
  const matches = (record, source) => record.snapshot !== null && record.owner === source.owner && record.version === source.version;
  function layoutOf(source) {
    const array = source.owner.array;
    if (!ArrayBuffer.isView(array) || array instanceof DataView || !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable ||
        !Number.isSafeInteger(source.position.count) || source.position.count < 0)
      reject('GEOMETRY', 'Native wireframes require fixed CPU position/index arrays');
    try { new Uint8Array(array.buffer, 0, 0); }
    catch { reject('GEOMETRY', 'Native wireframe source storage is detached'); }
    // These are the literal r186 loop extents, including partial triangles.
    const triangles = source.index ? Math.ceil(array.length / 3) : Math.max(0, Math.ceil((array.length - 3) / 9));
    const indexCount = triangles * 6, ArrayType = source.position.count >= 65535 ? Uint32Array : Uint16Array;
    const bytes = indexCount * ArrayType.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(indexCount) || indexCount > 0xffffffff || bytes > maxBytes)
      reject('LIMIT', 'Native wireframe edge indices exceed their byte budget');
    return {indexCount, ArrayType, bytes};
  }
  function recordOf(geometry) {
    let record = geometries.get(geometry);
    if (!record) {
      record = {owner: null, version: null, snapshot: null, layout: null, bytes: 0, placeholder: null};
      record.onDispose = () => {
        allocatedBytes -= record.bytes; record.bytes = 0;
        record.snapshot = null; record.placeholder = null; record.layout = null;
        record.owner = null; record.version = null;
        geometries.delete(geometry); geometry.removeEventListener('dispose', record.onDispose);
      };
      geometries.set(geometry, record); geometry.addEventListener('dispose', record.onDispose);
    }
    return record;
  }
  function reserve(record, bytes) {
    if (bytes > maxBytes - allocatedBytes + record.bytes)
      reject('LIMIT', 'Native wireframe histories exceed their byte budget');
    allocatedBytes += bytes - record.bytes; record.bytes = bytes;
  }
  function register(descriptor, geometry, base = geometry) {
    live();
    if (views.has(descriptor)) reject('GEOMETRY', 'Native wireframe geometry is already registered');
    const source = sourceOf(geometry), previous = geometries.get(geometry);
    const layout = previous && matches(previous, source) ? previous.layout : layoutOf(source);
    if (layout.bytes > maxBytes - allocatedBytes + (previous?.bytes ?? 0))
      reject('LIMIT', 'Native wireframe histories exceed their byte budget');
    const record = recordOf(geometry);
    reserve(record, layout.bytes);
    let initial = matches(record, source) ? record.snapshot : record.placeholder;
    if (!(initial instanceof layout.ArrayType) || initial.length !== layout.indexCount)
      initial = record.placeholder = new layout.ArrayType(layout.indexCount);
    const index = new three.BufferAttribute(initial, 1);
    const makeView = index => Object.freeze({geometry, get attributes() { return base.attributes; }, get index() { return index; },
      get drawRange() { const range = geometry.drawRange; return {start: range.start * 2, count: range.count * 2}; },
      get morphAttributes() { return base.morphAttributes; },
      get isInstancedBufferGeometry() { return base.isInstancedBufferGeometry === true; },
      addEventListener: (...args) => geometry.addEventListener(...args),
      removeEventListener: (...args) => geometry.removeEventListener(...args)});
    const view = makeView(index);
    views.set(descriptor, {geometry, layout, index, view, makeView, snapshots: new WeakMap(), installed: null});
    return view;
  }
  function stateOf(descriptor) {
    live(); const state = views.get(descriptor);
    if (!state) reject('GEOMETRY', 'Native wireframe geometry is not registered');
    return state;
  }
  function capture(descriptor) {
    const state = stateOf(descriptor), source = sourceOf(state.geometry), record = recordOf(state.geometry);
    const layout = matches(record, source) ? record.layout : layoutOf(source);
    if (layout.indexCount !== state.layout.indexCount || layout.ArrayType !== state.layout.ArrayType)
      reject('CHANGED', 'Native wireframe index layout changed; compile the graph again');
    reserve(record, layout.bytes);
    if (!matches(record, source)) {
      const data = new layout.ArrayType(layout.indexCount), array = source.owner.array;
      for (let triangle = 0, output = 0; output < data.length; triangle += 3) {
        const a = source.index ? array[triangle] : triangle;
        const b = source.index ? array[triangle + 1] : triangle + 1;
        const c = source.index ? array[triangle + 2] : triangle + 2;
        data[output++] = a; data[output++] = b; data[output++] = b;
        data[output++] = c; data[output++] = c; data[output++] = a;
      }
      record.snapshot = data; record.layout = layout;
      record.owner = source.owner; record.version = source.version; record.placeholder = null;
    }
    const data = record.snapshot;
    // Different source versions can occur between two groups within one render.
    // Keep their GPU geometry owners separate even though compilation shares a
    // program. Weak snapshot keys retire old views with their accepted packets.
    let geometry = state.snapshots.get(data);
    if (!geometry) {
      geometry = state.makeView(new three.BufferAttribute(data, 1));
      state.snapshots.set(data, geometry);
    }
    return Object.freeze({bytes: layout.bytes, indexCount: layout.indexCount, geometry, install() {
      live();
      if (state.installed !== data) {
        state.index.array = data; state.index.count = layout.indexCount;
        state.index.needsUpdate = true; state.installed = data;
      }
    }});
  }
  return Object.freeze({register, capture, geometry: descriptor => stateOf(descriptor).view,
    byteLength: descriptor => stateOf(descriptor).layout.bytes,
    // Accepted packets and descriptors can outlive replacement in the program
    // cache. Their weakly held views retain private owners until no longer used.
    release() {},
    dispose() {
      if (disposed) return; disposed = true;
      for (const [geometry, record] of geometries) geometry.removeEventListener('dispose', record.onDispose);
      geometries.clear(); allocatedBytes = 0;
    },
    get diagnostics() { return Object.freeze({geometries: geometries.size, allocatedBytes, maxBytes}); }});
}
