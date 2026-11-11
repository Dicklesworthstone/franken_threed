/** Private geometry selectors for queued WebGL draws. The shared upload owner
 * supplies GPU-visible raw bytes; these views never read later CPU edits and
 * never run application upload callbacks. Public BufferGeometry identities and
 * arrays remain untouched. Disposal retires a generation after its queued uses. */
export function createThreePassGeometry(three, history, {maxGeometries = 4096, maxBytes = 128 * 1024 * 1024} = {}) {
  const sources = new Map(), states = new Set(), views = new WeakMap();
  let edgeBytes = 0, previewBytes = 0;
  const fail = message => { throw Object.assign(new Error(`GEOMETRY_GPU_HISTORY: ${message}`), {code: 'GEOMETRY_GPU_HISTORY'}); };
  const data = (object, key) => {
    for (let owner = object; owner; owner = Object.getPrototypeOf(owner)) {
      const descriptor = Object.getOwnPropertyDescriptor(owner, key);
      if (descriptor) return Object.hasOwn(descriptor, 'value') ? descriptor.value : undefined;
    }
  };
  const dataOnly = object => object && typeof object === 'object' &&
    Object.values(Object.getOwnPropertyDescriptors(object)).every(descriptor => Object.hasOwn(descriptor, 'value'));
  const integer = (value, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= 0 && value <= max;
  function attribute(source) {
    const interleaved = data(source, 'isInterleavedBufferAttribute') === true;
    const owner = interleaved ? data(source, 'data') : source;
    const prototype = interleaved ? three.InterleavedBuffer.prototype : three.BufferAttribute.prototype;
    if (!(source instanceof (interleaved ? three.InterleavedBufferAttribute : three.BufferAttribute)) ||
        !dataOnly(source) || !dataOnly(owner) || data(owner, 'isStorageBufferAttribute') ||
        data(owner, 'isStorageInstancedBufferAttribute') || data(owner, 'isGLBufferAttribute') ||
        data(owner, 'onUploadCallback') !== prototype.onUploadCallback ||
        data(owner, 'clearUpdateRanges') !== prototype.clearUpdateRanges)
      fail('Queued geometry requires ordinary CPU attributes without upload hooks');
    const array = data(owner, 'array');
    if (!ArrayBuffer.isView(array) || array instanceof DataView || Object.getPrototypeOf(array) !== array.constructor.prototype ||
        ['buffer', 'byteOffset', 'byteLength', 'length', 'constructor', 'slice', 'subarray'].some(key => Object.hasOwn(array, key)))
      fail('Queued geometry requires ordinary typed-array storage');
    const itemSize = data(source, 'itemSize'), count = data(source, 'count');
    const stride = interleaved ? data(owner, 'stride') : itemSize, offset = interleaved ? data(source, 'offset') : 0;
    // InterleavedBufferAttribute.count is a pinned getter on its prototype.
    const selectedCount = interleaved ? data(owner, 'count') : count;
    if (!integer(itemSize, 16) || itemSize < 1 || !integer(stride, 512) || stride < itemSize ||
        !integer(offset) || offset + itemSize > stride || !integer(selectedCount, 0xffffffff) || selectedCount * stride > array.length)
      fail('Invalid queued geometry attribute layout');
    const ranges = data(owner, 'updateRanges');
    if (!Array.isArray(ranges) || Object.getPrototypeOf(ranges) !== Array.prototype || !dataOnly(ranges) ||
        ranges.some(range => !dataOnly(range) || !integer(data(range, 'start')) || !integer(data(range, 'count')) ||
          data(range, 'start') + data(range, 'count') > array.length)) fail('Invalid queued geometry upload ranges');
    const fields = {itemSize, normalized: data(source, 'normalized') === true, gpuType: data(source, 'gpuType'),
      isFloat16BufferAttribute: data(source, 'isFloat16BufferAttribute') === true,
      isInstancedBufferAttribute: data(source, 'isInstancedBufferAttribute') === true,
      meshPerAttribute: data(owner, 'meshPerAttribute'), stride, offset, interleaved,
      isInstancedInterleavedBuffer: data(owner, 'isInstancedInterleavedBuffer') === true};
    return {source, owner, array, fields, count: selectedCount, key: JSON.stringify(fields)};
  }
  function inspect(source, maxValues) {
    if (!(source instanceof three.BufferGeometry) || !dataOnly(source) ||
        data(source, 'addEventListener') !== three.EventDispatcher.prototype.addEventListener ||
        data(source, 'removeEventListener') !== three.EventDispatcher.prototype.removeEventListener)
      fail('Queued geometry requires ordinary resource lifecycle methods');
    const attributes = data(source, 'attributes'), groups = data(source, 'groups'), range = data(source, 'drawRange');
    if (!dataOnly(attributes) || !Array.isArray(groups) || Object.getPrototypeOf(groups) !== Array.prototype ||
        !dataOnly(groups) || !dataOnly(range)) fail('Queued geometry requires ordinary selectors');
    const values = Object.keys(attributes).length * 16 + groups.length * 3 + 4;
    if (values > maxValues) fail('Pending geometry selector capacity exceeded');
    const selected = Object.entries(attributes).map(([name, value]) => [name, attribute(value)]);
    const index = data(source, 'index') == null ? null : attribute(data(source, 'index'));
    const start = data(range, 'start'), count = data(range, 'count');
    if (!integer(start) || count !== Infinity && !integer(count)) fail('Invalid queued geometry draw range');
    const selectedGroups = groups.map(group => {
      if (!dataOnly(group) || !integer(data(group, 'start')) ||
          data(group, 'count') !== Infinity && !integer(data(group, 'count')) || !integer(data(group, 'materialIndex')))
        fail('Invalid queued geometry group');
      return {start: data(group, 'start'), count: data(group, 'count'), materialIndex: data(group, 'materialIndex')};
    });
    const all = [...selected.map(([, entry]) => entry), ...(index ? [index] : [])];
    return {source, attributes: selected, index, all, values, groups: selectedGroups, drawRange: {start, count},
      instanced: data(source, 'isInstancedBufferGeometry') === true, instanceCount: data(source, 'instanceCount')};
  }
  function validate(shapes) {
    if (sources.size + [...shapes.keys()].filter(source => !sources.has(source)).length > maxGeometries)
      fail('Queued geometry identity capacity exceeded');
    let selectedBytes = edgeBytes;
    for (const shape of shapes.values()) if (shape.edgePlan && !shape.edgePlan.cached)
      selectedBytes += shape.edgePlan.bytes - (sources.get(shape.source)?.edge?.array.byteLength ?? 0);
    if (selectedBytes > maxBytes) fail('Wireframe edge history capacity exceeded');
  }
  const matches = (a, b) => a.instanced === b.instanced && a.attributes.length === b.attributes.length &&
    a.attributes.every(([name, entry], i) => name === b.attributes[i][0] && sameAttribute(entry, b.attributes[i][1])) &&
    (a.index === null ? b.index === null : b.index !== null && sameAttribute(a.index, b.index));
  const sameAttribute = (a, b) => a.source === b.source && a.owner === b.owner && a.key === b.key && a.array.constructor === b.array.constructor;
  function retire(state, force = false) {
    if (!state || state.closed) return;
    state.retired = true;
    if (force || !state.refs) {
      state.closed = true; retireWire(state); state.view.dispatchEvent({type: 'dispose'}); states.delete(state);
    }
  }
  function clearPreview(wire) {
    if (wire?.preview) { previewBytes -= wire.preview.edge.array.byteLength; wire.preview = null; }
  }
  function retireWire(owner) {
    clearPreview(owner.wire); owner.wire?.geometry.dispose(); owner.wire = null;
  }
  function sourceRecord(original) {
    let source = sources.get(original);
    if (source) return source;
    if (sources.size >= maxGeometries) fail('Queued geometry identity capacity exceeded');
    source = {state: null, edge: null, wire: null, dispose: null};
    source.dispose = () => {
      // Pinned WebGLGeometries removes current selectors and the separately
      // derived edge index; replacing a source attribute does not reset edges.
      for (const attribute of [...Object.values(original.attributes), original.index]) if (attribute)
        history.forget(attribute.isInterleavedBufferAttribute ? attribute.data : attribute);
      edgeBytes -= source.edge?.array.byteLength ?? 0;
      retireWire(source); original.removeEventListener('dispose', source.dispose);
      sources.delete(original); retire(source.state);
    };
    original.addEventListener('dispose', source.dispose); sources.set(original, source); return source;
  }
  function edgePlan(original) {
    const cached = sources.get(original)?.edge, index = original.index, position = original.attributes.position;
    if (cached && (index === null || !(cached.version < index.version)))
      return {cached, bytes: cached.array.byteLength};
    if (!position) fail('Wireframe geometry requires positions');
    const array = index ? index.array : position.array;
    if (!ArrayBuffer.isView(array) || array instanceof DataView) fail('Wireframe topology requires CPU index or position storage');
    const Type = position.count >= 65535 ? Uint32Array : Uint16Array;
    const triangles = index ? Math.ceil(array.length / 3) : Math.max(0, Math.ceil((array.length / 3 - 1) / 3));
    const length = triangles * 6, bytes = length * Type.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(bytes) || bytes > maxBytes) fail('Wireframe edge history capacity exceeded');
    return {array, indexed: index !== null, version: index ? index.version : position.version, Type, length, bytes};
  }
  function buildEdges(plan) {
    const array = new plan.Type(plan.length);
    for (let offset = 0, triangle = 0; offset < array.length; offset += 6, triangle += 3) {
      const a = plan.indexed ? plan.array[triangle] : triangle;
      const b = plan.indexed ? plan.array[triangle + 1] : triangle + 1;
      const c = plan.indexed ? plan.array[triangle + 2] : triangle + 2;
      array.set([a, b, b, c, c, a], offset);
    }
    const attribute = new three.BufferAttribute(array, 1); attribute.version = plan.version;
    history.markPrivate(attribute); return attribute;
  }
  function acceptEdges(original, plan) {
    if (plan.cached) return plan.cached;
    const source = sourceRecord(original), bytes = edgeBytes - (source.edge?.array.byteLength ?? 0) + plan.bytes;
    if (bytes > maxBytes) fail('Wireframe edge history capacity exceeded');
    source.edge = buildEdges(plan); edgeBytes = bytes; return source.edge;
  }
  function wireframeGeometry(input, {accept = true} = {}) {
    const state = views.get(input), original = state?.original ?? input;
    const plan = state?.acceptedEdge ? null : edgePlan(original);
    // A queued generation can outlive public disposal. Its accepted edges are
    // sufficient; deriving them must not resurrect a source cache/listener.
    const owner = state ?? sourceRecord(original);
    const wire = owner.wire ??= {geometry: new three.BufferGeometry(), edge: null, preview: null};
    history.markPrivateGeometry(wire.geometry);
    // Compilation may preview layouts, but never establishes the source edge
    // version. First render still reads CPU edits made after compileAsync().
    let edge = state?.acceptedEdge ?? plan.cached;
    if (edge || accept) {
      edge ??= acceptEdges(original, plan); clearPreview(wire);
    } else {
      const previous = wire.preview;
      if (!previous || previous.plan.array !== plan.array || previous.plan.version !== plan.version ||
          previous.plan.length !== plan.length || previous.plan.Type !== plan.Type || previous.plan.indexed !== plan.indexed) {
        const bytes = previewBytes - (previous?.edge.array.byteLength ?? 0) + plan.bytes;
        if (bytes > maxBytes) fail('Wireframe preparation preview capacity exceeded');
        edge = buildEdges(plan); previewBytes = bytes; wire.preview = {plan, edge};
      } else edge = previous.edge;
    }
    const derived = wire.geometry;
    derived.attributes = input.attributes; derived.morphAttributes = input.morphAttributes;
    if (wire.edge !== edge) {
      let index = derived.index;
      // Stable-size updates reuse one private upload identity instead of
      // retaining another GPU index allocation on every procedural rebuild.
      if (!index || index.array.constructor !== edge.array.constructor || index.array.length !== edge.array.length) {
        if (index) history.retirePrivate(index);
        index = new three.BufferAttribute(edge.array, 1); history.markPrivate(index); derived.setIndex(index);
      } else { index.array = edge.array; index.count = edge.count; index.needsUpdate = true; }
      wire.edge = edge;
    }
    const range = input.drawRange;
    derived.setDrawRange(range.start * 2, range.count === Infinity ? Infinity : range.count * 2);
    return derived;
  }
  function makeState(shape) {
    const owners = new Map(), attributes = {};
    function view(entry) {
      const {owner, fields, array} = entry;
      let storage = owners.get(owner);
      if (!storage) {
        const empty = new array.constructor(0);
        storage = fields.interleaved ? new three.InterleavedBuffer(empty, fields.stride)
          : new three.BufferAttribute(empty, fields.itemSize, fields.normalized);
        if (fields.isInstancedInterleavedBuffer) storage.isInstancedInterleavedBuffer = true;
        if (fields.meshPerAttribute !== undefined) storage.meshPerAttribute = fields.meshPerAttribute;
        history.markPrivate(storage); owners.set(owner, storage);
      }
      const attribute = fields.interleaved
        ? new three.InterleavedBufferAttribute(storage, fields.itemSize, fields.offset, fields.normalized) : storage;
      attribute.isFloat16BufferAttribute = fields.isFloat16BufferAttribute;
      attribute.isInstancedBufferAttribute = fields.isInstancedBufferAttribute;
      if (fields.gpuType !== undefined) attribute.gpuType = fields.gpuType;
      return attribute;
    }
    for (const [name, entry] of shape.attributes) attributes[name] = view(entry);
    const index = shape.index ? view(shape.index) : null;
    const geometry = Object.create(shape.source, Object.fromEntries(Object.entries({attributes, index, _listeners: {},
      morphAttributes: {}, groups: [], drawRange: {start: 0, count: Infinity}, isInstancedBufferGeometry: shape.instanced,
      instanceCount: shape.instanceCount,
      addEventListener: three.EventDispatcher.prototype.addEventListener,
      removeEventListener: three.EventDispatcher.prototype.removeEventListener,
      dispatchEvent: three.EventDispatcher.prototype.dispatchEvent,
    }).map(([key, value]) => [key, {value, writable: true, configurable: true, enumerable: true}])));
    // Layout reuse needs identities and scalar descriptors, not a retained
    // alias to every array the source has since replaced or released.
    const layout = entry => ({source: entry.source, owner: entry.owner, key: entry.key, array: {constructor: entry.array.constructor}});
    const signature = {instanced: shape.instanced, attributes: shape.attributes.map(([name, entry]) => [name, layout(entry)]),
      index: shape.index ? layout(shape.index) : null};
    const state = {view: geometry, attributes, index, owners, shape: signature, refs: 0, retired: false, closed: false,
      original: shape.source, acceptedEdge: null, wire: null};
    history.markPrivateGeometry(geometry);
    views.set(geometry, state); states.add(state); return state;
  }
  function capture(shape, accepted) {
    const source = sourceRecord(shape.source), edge = shape.edgePlan ? acceptEdges(shape.source, shape.edgePlan) : null;
    if (!source.state || !matches(source.state.shape, shape)) {
      retire(source.state); source.state = makeState(shape);
    }
    const state = source.state, arrays = new Map(); state.refs++;
    for (const entry of shape.all) arrays.set(entry.owner, {array: accepted.get(entry.owner), count: entry.count});
    let released = false;
    return {view: state.view,
      install() {
        if (released || state.closed) fail('Queued geometry was released before submission');
        state.view.attributes = state.attributes; state.view.index = state.index; state.view.morphAttributes = {};
        state.view.groups = shape.groups; state.view.drawRange = shape.drawRange; state.view.instanceCount = shape.instanceCount;
        state.view.isInstancedBufferGeometry = shape.instanced;
        state.acceptedEdge = edge;
        for (const [owner, {array, count}] of arrays) {
          const view = state.owners.get(owner); view.count = count;
          if (array && view.array !== array) { view.array = array; view.needsUpdate = true; }
        }
      },
      safe() {
        return shape.all.every(({owner, fields}) => data(owner, 'onUploadCallback') ===
          (fields.interleaved ? three.InterleavedBuffer.prototype : three.BufferAttribute.prototype).onUploadCallback);
      },
      release() { if (!released) { released = true; state.refs--; if (state.retired) retire(state); } },
    };
  }
  return {inspect, validate, capture, edgePlan, wireframeGeometry,
    live(source) {
      const state = sources.get(source)?.state;
      if (!state || state.closed || !dataOnly(source)) return source;
      // Keep the prepared geometry identity when a callback-bearing draw moves
      // back to original live attributes. Upload history supplies the same raw
      // versions, and callbacks still receive the original public geometry.
      for (const key of ['attributes', 'index', 'morphAttributes', 'groups', 'drawRange', 'instanceCount']) state.view[key] = source[key];
      state.view.isInstancedBufferGeometry = source.isInstancedBufferGeometry === true;
      state.acceptedEdge = null;
      return state.view;
    },
    dispose() {
      for (const [source, state] of sources) { source.removeEventListener('dispose', state.dispose); retireWire(state); }
      sources.clear(); edgeBytes = 0; for (const state of states) retire(state, true);
    },
  };
}
