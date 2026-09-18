/** KHR_draco_mesh_compression -> ordinary glTF accessor storage.
 * Lend a configured Three r186 DRACOLoader (decodeGeometry). The adapter owns
 * each returned temporary geometry, not the decoder, workers or native runtime.
 * Importing this file starts no I/O, worker, Wasm, GPU or application clock.
 */
export class GltfDracoError extends Error {
  constructor(code, message, options) { super(`${code}: ${message}`, options); this.name = 'GltfDracoError'; this.code = code; }
}
const EXT = 'KHR_draco_mesh_compression';
const fail = (code, message) => { throw new GltfDracoError('GLTF_DRACO_' + code, message); };
const integer = (value, label, min = 0) => {
  if (!Number.isSafeInteger(value) || value < min) fail('LAYOUT', `Invalid ${label}`);
  return value;
};
const object = (value, label) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('LAYOUT', `Expected ${label}`);
  return value;
};
const list = (value, label) => {
  if (!Array.isArray(value)) fail('LAYOUT', `Expected ${label} array`);
  return value;
};
const index = (values, i, label) => {
  integer(i, label); if (i >= values.length) fail('LAYOUT', `Invalid ${label}`);
  return object(values[i], label);
};
const abort = signal => { if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError'); };
const formats = new Map([
  [5120, [Int8Array, 1, 'setInt8']], [5121, [Uint8Array, 1, 'setUint8']],
  [5122, [Int16Array, 2, 'setInt16']], [5123, [Uint16Array, 2, 'setUint16']],
  [5125, [Uint32Array, 4, 'setUint32']], [5126, [Float32Array, 4, 'setFloat32']],
]);
const widths = new Map([['SCALAR', 1], ['VEC2', 2], ['VEC3', 3], ['VEC4', 4]]);
const aligned = value => Math.ceil(value / 4) * 4;
function byteView(value) {
  const buffer = ArrayBuffer.isView(value) ? value.buffer : value;
  if (!(buffer instanceof ArrayBuffer) || buffer.resizable) fail('BUFFER', 'Expected fixed unshared bytes');
  try { return ArrayBuffer.isView(value) ? new Uint8Array(buffer, value.byteOffset, value.byteLength) : new Uint8Array(buffer); }
  catch { fail('BUFFER', 'Detached bytes'); }
}
function attributeLayout(attribute, count, width, Type = null) {
  object(attribute, 'decoded attribute');
  if (attribute.itemSize !== width || attribute.count !== count) fail('DECODE', 'Decoded attribute count/width disagrees with accessor');
  const interleaved = attribute.isInterleavedBufferAttribute === true;
  const array = interleaved ? attribute.data?.array : attribute.array;
  if (!ArrayBuffer.isView(array) || array instanceof DataView || (Type && !(array instanceof Type))) fail('DECODE', 'Decoder returned an incorrect component type');
  byteView(array);
  const stride = interleaved ? integer(attribute.data.stride, 'decoded stride', width) : width;
  const offset = interleaved ? integer(attribute.offset, 'decoded attribute offset') : 0;
  if (offset + width > stride || !Number.isSafeInteger((count - 1) * stride + offset + width) ||
      (count - 1) * stride + offset + width > array.length) fail('DECODE', 'Decoded attribute exceeds its storage');
  return {array, stride, offset};
}
// Every result transfers temporary-geometry ownership. Dispose late results even
// if the foreign worker ignores cancellation; observe late rejection as well.
async function receive(pending, signal) {
  pending = Promise.resolve(pending); pending.catch(() => {});
  let adopted = false, onAbort;
  const dispose = geometry => { try { geometry?.dispose?.(); } catch {} };
  try {
    abort(signal);
    const cancelled = signal && new Promise((_, reject) => {
      onAbort = () => reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
      signal.addEventListener('abort', onAbort, {once: true}); if (signal.aborted) onAbort();
    });
    const geometry = await (cancelled ? Promise.race([pending, cancelled]) : pending);
    abort(signal); adopted = true; return geometry;
  } finally {
    if (onAbort) signal.removeEventListener('abort', onAbort);
    if (!adopted) pending.then(dispose, () => {});
  }
}

/** Preflight before resource I/O. Limits account for copied compressed ranges
 * and owned decoded buffers, not third-party decoder scratch/worker memory.
 * A source accessor shared by other primitives is NEVER rewritten in place.
 * Decode appends private accessors/views/buffers and remaps only this primitive.
 * TRIANGLES are supported by this DRACOLoader triangle-list adapter; compressed
 * TRIANGLE_STRIP needs a topology-aware decoder and is refused, never guessed.
 */
export function prepareDracoMeshes(model, {
  decoder = null, maxEncodedBytes = 128 * 1024 * 1024, maxDecodedBytes = 128 * 1024 * 1024,
  maxDecodedBufferBytes = 64 * 1024 * 1024, maxPrimitives = 4096,
} = {}) {
  integer(maxEncodedBytes, 'encoded budget'); integer(maxDecodedBytes, 'decoded budget');
  integer(maxDecodedBufferBytes, 'decoded buffer budget', 1); integer(maxPrimitives, 'primitive budget', 1);
  if (model?.asset?.version !== '2.0') fail('LAYOUT', 'Expected glTF 2.0');
  const meshes = list(model.meshes ?? [], 'meshes');
  const hasDraco = meshes.some(mesh => Array.isArray(mesh?.primitives) && mesh.primitives.some(p => Object.hasOwn(p?.extensions ?? {}, EXT)));
  const required = list(model.extensionsRequired ?? [], 'required extensions').includes(EXT);
  if (!hasDraco && !required) return Object.freeze({decodedBytes: 0, async decode(buffers, {signal} = {}) {
    abort(signal); return {json: model, sourceJson: model, buffers, decodedBytes: 0, decodedPrimitives: 0};
  }});
  const available = decoder != null && decoder.supported !== false;
  if (required && !available) fail('DECODER', 'A configured DRACOLoader is required');
  if (hasDraco && available && typeof decoder.decodeGeometry !== 'function') fail('DECODER', 'Expected DRACOLoader.decodeGeometry');
  list(model.extensionsUsed ?? [], 'used extensions');
  const snapshot = structuredClone(model), accessors = list(snapshot.accessors ?? [], 'accessors');
  const views = list(snapshot.bufferViews ?? [], 'bufferViews'), buffers = list(snapshot.buffers ?? [], 'buffers');
  const records = []; let encodedBytes = 0, reservedBytes = 0;
  const reserve = size => {
    if (!Number.isSafeInteger(size) || size > maxDecodedBufferBytes || size > maxDecodedBytes - reservedBytes) fail('LIMIT', 'Decoded Draco storage exceeds byte budget');
    reservedBytes += size;
  };
  function accessor(at, label, isIndex = false) {
    const a = index(accessors, at, label), format = formats.get(a.componentType), width = widths.get(a.type);
    if (!format || !width || (isIndex && (width !== 1 || ![5121, 5123, 5125].includes(a.componentType) || a.normalized === true))) fail('LAYOUT', 'Unsupported Draco accessor format');
    integer(a.count, 'accessor count', 1);
    if (a.normalized !== undefined && (typeof a.normalized !== 'boolean' || (a.normalized && ![5120, 5121, 5122, 5123].includes(a.componentType)))) fail('LAYOUT', 'Invalid normalization');
    if (a.sparse !== undefined && available) fail('LAYOUT', 'Compressed accessors cannot also supply sparse overrides');
    const stride = isIndex ? format[1] : aligned(width * format[1]);
    const size = aligned(a.count * stride);
    if (available) reserve(size);
    return {at, count: a.count, componentType: a.componentType, width, stride, size};
  }
  for (let mesh = 0; mesh < (snapshot.meshes ?? []).length; mesh++) {
    const primitives = list(snapshot.meshes[mesh]?.primitives, 'mesh primitives');
    for (let primitive = 0; primitive < primitives.length; primitive++) {
      const p = object(primitives[primitive], 'primitive');
      if (!Object.hasOwn(p.extensions ?? {}, EXT)) continue;
      if (records.length >= maxPrimitives) fail('LIMIT', 'Too many Draco primitives');
      const ext = object(p.extensions[EXT], 'Draco extension'), attrs = object(p.attributes, 'primitive attributes');
      const mappings = object(ext.attributes, 'Draco attribute map'), names = Object.keys(mappings);
      if (!names.length || names.length > 64 || Object.keys(attrs).length > 64) fail('LIMIT', 'Invalid Draco attribute count');
      if (available && (p.mode ?? 4) !== 4) fail('TOPOLOGY', 'DRACOLoader triangle-list decoding requires TRIANGLES');
      if (![4, 5].includes(p.mode ?? 4)) fail('TOPOLOGY', 'Draco requires triangle topology');
      const source = index(views, ext.bufferView, 'Draco bufferView'), sourceBuffer = index(buffers, source.buffer, 'Draco buffer');
      integer(sourceBuffer.byteLength, 'source buffer length', 1); integer(source.byteOffset ?? 0, 'source offset'); integer(source.byteLength, 'source length', 1);
      if (source.byteLength > sourceBuffer.byteLength - (source.byteOffset ?? 0) || source.byteStride !== undefined) fail('LAYOUT', 'Invalid compressed view range/stride');
      if (available && source.byteLength > maxEncodedBytes - encodedBytes) fail('LIMIT', 'Compressed Draco snapshots exceed budget');
      if (available) encodedBytes += source.byteLength;
      const positions = index(accessors, attrs.POSITION, 'POSITION accessor'); integer(positions.count, 'vertex count', 1);
      for (const at of Object.values(attrs)) if (index(accessors, at, 'attribute accessor').count !== positions.count) fail('LAYOUT', 'Primitive attribute counts disagree');
      const attributes = names.map((name, slot) => {
        if (!Object.hasOwn(attrs, name)) fail('LAYOUT', 'Draco attribute is not a primitive attribute');
        const uniqueID = integer(mappings[name], 'Draco unique attribute ID');
        if (uniqueID > 0xffffffff) fail('LAYOUT', 'Draco unique attribute ID exceeds uint32');
        return {name, key: 'a' + slot, uniqueID, ...accessor(attrs[name], name)};
      });
      const indices = p.indices === undefined ? null : accessor(p.indices, 'index accessor', true);
      if (indices && (p.mode ?? 4) === 4 && indices.count % 3) fail('LAYOUT', 'Triangle index count must be divisible by three');
      // Optional compressed inputs use only explicit real fallback storage. Do
      // not turn missing compressed-only accessors into zero-filled geometry.
      if (!available) for (const field of [...attributes, ...(indices ? [indices] : [])]) {
        const a = accessors[field.at];
        if (a.bufferView === undefined && a.sparse === undefined) fail('FALLBACK', 'Optional Draco needs backed or sparse fallback accessors');
      }
      records.push({mesh, primitive, source: ext.bufferView, attributes, indices, count: positions.count});
    }
  }
  let consumed = false;
  return Object.freeze({decodedBytes: reservedBytes, async decode(supplied, {signal} = {}) {
    if (consumed) fail('STATE', 'Draco plan is single-use'); consumed = true; abort(signal);
    if (!Array.isArray(supplied) || supplied.length !== buffers.length) fail('BUFFER', 'Supply buffers in original index order');
    const sources = available ? records.map(record => {
      const view = views[record.source];
      if (Object.keys(view.extensions ?? {}).length) fail('LAYOUT', 'Resolve compressed bufferView extensions before Draco decoding');
      const data = byteView(supplied[view.buffer]), start = view.byteOffset ?? 0;
      if (data.length < buffers[view.buffer].byteLength || view.byteLength > data.length - start) fail('BUFFER', 'Truncated Draco buffer');
      return data.slice(start, start + view.byteLength).buffer;
    }) : [];
    const json = structuredClone(snapshot), output = supplied.slice();
    // JSON loaded from bytes has no aliases, but direct callers can share a
    // primitive object or attribute map. Keep each projected instance private.
    if (json.meshes) json.meshes = json.meshes.map(mesh => ({...mesh, primitives: mesh.primitives.map(p => ({
      ...p, ...(p.attributes ? {attributes: {...p.attributes}} : {}), ...(p.extensions ? {extensions: {...p.extensions}} : {}),
    }))}));
    json.buffers ??= []; json.bufferViews ??= []; json.accessors ??= [];
    let decodedBytes = 0, decodedPrimitives = 0;
    function allocate(size) {
      if (!Number.isSafeInteger(size) || size > maxDecodedBufferBytes || size > maxDecodedBytes - decodedBytes) fail('LIMIT', 'Decoded Draco storage exceeds byte budget');
      decodedBytes += size; return new Uint8Array(size);
    }
    function publish(data, definition, stride, target) {
      const buffer = output.length, bufferView = json.bufferViews.length;
      output.push(data); json.buffers.push({byteLength: data.length});
      json.bufferViews.push({buffer, byteLength: data.length, target, ...(target === 34962 ? {byteStride: stride} : {})});
      const a = {...definition, bufferView, byteOffset: 0}; delete a.sparse;
      return json.accessors.push(a) - 1;
    }
    for (let i = 0; i < records.length; i++) {
      abort(signal); const record = records[i], p = json.meshes[record.mesh].primitives[record.primitive];
      if (available) {
        const taskConfig = {attributeIDs: {}, attributeTypes: {}, useUniqueIDs: true, vertexColorSpace: 'srgb-linear'};
        for (const a of record.attributes) {
          taskConfig.attributeIDs[a.key] = a.uniqueID; taskConfig.attributeTypes[a.key] = formats.get(a.componentType)[0].name;
        }
        let geometry;
        try {
          // Unique private buffers are transferable by DRACOLoader. Generated
          // attribute names avoid color conversion and hostile source-name keys.
          geometry = await receive(decoder.decodeGeometry(sources[i], taskConfig), signal); abort(signal);
          object(geometry, 'decoded geometry'); object(geometry.attributes, 'decoded attributes');
          for (const a of record.attributes) {
            const [Type, bytes, setter] = formats.get(a.componentType);
            const source = attributeLayout(geometry.attributes[a.key], a.count, a.width, Type);
            const data = allocate(a.size), view = new DataView(data.buffer);
            for (let v = 0; v < a.count; v++) for (let c = 0; c < a.width; c++) {
              const value = source.array[source.offset + v * source.stride + c];
              if (!Number.isFinite(value)) fail('DECODE', 'Non-finite decoded attribute');
              view[setter](v * a.stride + c * bytes, value, true);
            }
            p.attributes[a.name] = publish(data, accessors[a.at], a.stride, 34962);
          }
          const declared = record.indices, indexAttribute = geometry.index;
          if (declared && !indexAttribute) fail('DECODE', 'Decoder omitted required mesh indices');
          if (indexAttribute) {
            const count = integer(indexAttribute.count, 'decoded index count', 1);
            if (count % 3 || (declared && count !== declared.count)) fail('DECODE', 'Decoded index count disagrees with triangle accessor');
            const source = attributeLayout(indexAttribute, count, 1);
            if (![Uint8Array, Uint16Array, Uint32Array].some(Type => source.array instanceof Type) || indexAttribute.normalized === true) fail('DECODE', 'Decoded indices must be unsigned integers');
            const componentType = declared?.componentType ?? 5125, [, size, setter] = formats.get(componentType);
            const data = allocate(aligned(count * size)), view = new DataView(data.buffer), restart = 2 ** (size * 8) - 1;
            for (let k = 0; k < count; k++) {
              const value = source.array[source.offset + k * source.stride];
              if (value >= record.count || value >= restart) fail('DECODE', 'Decoded index is outside the accessor domain');
              view[setter](k * size, value, true);
            }
            p.indices = publish(data, declared ? accessors[declared.at] : {type: 'SCALAR', componentType, count}, size, 34963);
          } else if (record.count % 3) fail('DECODE', 'Nonindexed triangles need complete vertex triples');
          decodedPrimitives++;
        } catch (error) {
          abort(signal); if (error instanceof GltfDracoError) throw error;
          throw new GltfDracoError('GLTF_DRACO_DECODE', `Draco failed for mesh ${record.mesh} primitive ${record.primitive}`, {cause: error});
        } finally {
          // Disposal exceptions must not hide decoding/cancellation failures.
          try { geometry?.dispose?.(); } catch {}
        }
      }
      delete p.extensions[EXT]; if (!Object.keys(p.extensions).length) delete p.extensions;
    }
    for (const key of ['extensionsRequired', 'extensionsUsed']) if (json[key]) json[key] = json[key].filter(name => name !== EXT);
    abort(signal); return {json, sourceJson: snapshot, buffers: output, decodedBytes, decodedPrimitives};
  }});
}

/** Preloaded JSON/buffers. The decoder owns fetching its own code and worker
 * lifecycle; the F3D adapter never invokes preload, dispose, or a decoder clock.
 */
export async function decodeDracoMeshes(model, buffers, options = {}) {
  return prepareDracoMeshes(model, options).decode(buffers, {signal: options.signal});
}
