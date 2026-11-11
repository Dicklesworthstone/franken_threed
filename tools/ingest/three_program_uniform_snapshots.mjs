/** Accepted data for ordinary WebGL shader uniforms. Source materials and
 * dictionaries remain the program-cache identities; each queued use owns only
 * its copied values. Pending dictionaries are candidates, never publications:
 * a failed earlier compilation cannot select the next draw's uniform owner. */
import {captureThreeUniformGroup} from './three_uniform_group_state.mjs';

export function createThreeProgramUniformSnapshots(T, {
  maxBytes = 32 * 1024 * 1024, maxCaptureBytes = 2 * 1024 * 1024,
  maxMaterials = 256, maxValues = 1048576,
} = {}) {
  const fail = (code, message) => {
    const error = new Error(`THREE_PROGRAM_UNIFORM_SNAPSHOT_${code}: ${message}`);
    error.code = `THREE_PROGRAM_UNIFORM_SNAPSHOT_${code}`; throw error;
  };
  if (![maxBytes, maxCaptureBytes, maxMaterials, maxValues].every(value => Number.isSafeInteger(value) && value > 0))
    fail('LIMIT', 'Expected positive uniform snapshot capacities');
  const unsupported = Symbol('live uniform inputs');
  const pending = new Map(), frames = new Set();
  let allocatedBytes = 0, disposed = false;
  const math = new Set(['Color', 'Vector2', 'Vector3', 'Vector4', 'Matrix3', 'Matrix4', 'Quaternion', 'Euler', 'Plane', 'Sphere', 'Box3']
    .map(name => T[name]?.prototype).filter(Boolean));
  const typed = new Map([Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array,
    Float32Array, Float64Array, ...(typeof Float16Array === 'function' ? [Float16Array] : [])].map(Type => [Type.prototype, Type]));
  const data = (object, key) => {
    for (let current = object; current; current = Object.getPrototypeOf(current)) {
      const descriptor = Object.getOwnPropertyDescriptor(current, key);
      if (descriptor) {
        if (!Object.hasOwn(descriptor, 'value')) throw unsupported;
        return descriptor.value;
      }
    }
  };
  const descriptors = object => {
    const result = Object.getOwnPropertyDescriptors(object);
    if (Reflect.ownKeys(result).some(key => typeof key !== 'string') ||
        Object.values(result).some(descriptor => !Object.hasOwn(descriptor, 'value'))) throw unsupported;
    return result;
  };
  const plain = object => object !== null && typeof object === 'object' &&
    [Object.prototype, null].includes(Object.getPrototypeOf(object));
  const uniform = value => {
    if (!plain(value) && !(T.Uniform && Object.getPrototypeOf(value ?? {}) === T.Uniform.prototype)) throw unsupported;
    const fields = descriptors(value);
    if (!Object.hasOwn(fields, 'value') || data(value, 'needsUpdate') === false) throw unsupported;
    return fields;
  };
  function release(record) {
    if (record.released) return;
    record.released = true;
    for (const [material, entry] of record.entries) {
      const candidates = pending.get(material);
      for (const source of entry.uniforms.keys()) {
        const count = candidates.get(source) - 1;
        if (count) candidates.set(source, count); else candidates.delete(source);
      }
      if (!candidates.size) pending.delete(material);
    }
    allocatedBytes -= record.bytes; record.entries.clear(); frames.delete(record);
  }
  function capture(materials, activeSource = () => undefined) {
    if (disposed) fail('DISPOSED', 'Uniform snapshot owner is disposed');
    const accepted = new Map(), registrations = [];
    let bytes = 0, values = 0;
    for (const material of new Set(materials)) {
      let localBytes = 0, localValues = 0;
      const memo = new Map(), groups = [], uniformObjects = new WeakSet();
      const inspectUniform = value => { uniform(value); uniformObjects.add(value); };
      const charge = (size, count = 1) => {
        localBytes += size; localValues += count;
        if (!Number.isSafeInteger(localBytes) || bytes + localBytes > maxCaptureBytes ||
            bytes + localBytes > maxBytes - allocatedBytes || values + localValues > maxValues)
          fail('LIMIT', 'Pending shader uniform snapshots exceed their data budget');
      };
      function copy(value, depth = 0) {
        if (depth > 32) throw unsupported;
        if (value === null || value === undefined || typeof value === 'number' || typeof value === 'boolean') {
          charge(8); return value;
        }
        if (typeof value === 'string') { charge(8 + value.length * 2); return value; }
        if (typeof value !== 'object') throw unsupported;
        if (memo.has(value)) return memo.get(value);
        if (T.Texture && value instanceof T.Texture) {
          if (data(value, 'onUpdate') != null) throw unsupported;
          charge(64); memo.set(value, value); return value;
        }
        const prototype = Object.getPrototypeOf(value), Type = typed.get(prototype);
        if (ArrayBuffer.isView(value)) {
          if (!Type || ['buffer', 'byteOffset', 'byteLength', 'length', 'constructor', 'slice', 'subarray'].some(key => Object.hasOwn(value, key))) throw unsupported;
          const buffer = value.buffer;
          if (!(buffer instanceof ArrayBuffer) || Object.getPrototypeOf(buffer) !== ArrayBuffer.prototype ||
              ['byteLength', 'resizable', 'slice'].some(key => Object.hasOwn(buffer, key)) || buffer.resizable) throw unsupported;
          try { new Uint8Array(buffer, 0, 0); } catch { throw unsupported; }
          charge(64 + value.byteLength, value.length + 1);
          if (Reflect.ownKeys(value).some(key => typeof key !== 'string' || !/^(0|[1-9][0-9]*)$/.test(key))) throw unsupported;
          const result = new Type(value.length);
          new Uint8Array(result.buffer).set(new Uint8Array(buffer, value.byteOffset, value.byteLength));
          memo.set(value, result); return result;
        }
        const isGroup = T.UniformsGroup && prototype === T.UniformsGroup.prototype;
        const isUniform = T.Uniform && prototype === T.Uniform.prototype || uniformObjects.has(value);
        const array = Array.isArray(value);
        if (array ? prototype !== Array.prototype : !plain(value) && !math.has(prototype) && !isUniform && !isGroup) throw unsupported;
        if (array) charge(64 + value.length * 8, value.length + 1); else charge(64);
        const fields = descriptors(value);
        if (isGroup && (data(value, 'addEventListener') !== T.EventDispatcher.prototype.addEventListener ||
            data(value, 'removeEventListener') !== T.EventDispatcher.prototype.removeEventListener ||
            data(value, 'dispose') !== T.UniformsGroup.prototype.dispose)) throw unsupported;
        const sourceUniforms = isGroup ? data(value, 'uniforms') : null;
        if (isGroup) {
          if (!Array.isArray(sourceUniforms) || Object.getPrototypeOf(sourceUniforms) !== Array.prototype) throw unsupported;
          descriptors(sourceUniforms);
          for (const item of sourceUniforms) {
            if (Array.isArray(item)) {
              if (Object.getPrototypeOf(item) !== Array.prototype) throw unsupported;
              descriptors(item);
            }
            for (const itemUniform of Array.isArray(item) ? item : [item]) inspectUniform(itemUniform);
          }
        }
        const result = array ? [] : Object.create(prototype); memo.set(value, result);
        for (const [key, descriptor] of Object.entries(fields)) {
          if (array && key === 'length' || math.has(prototype) && key === '_onChangeCallback' ||
              (isUniform || isGroup) && ['__data', '__offset', '__cache', '__size'].includes(key) ||
              isGroup && key === '_listeners') continue;
          charge(key.length * 2 + 16);
          Object.defineProperty(result, key, {...descriptor, value: copy(descriptor.value, depth + 1)});
        }
        if (array) Object.defineProperty(result, 'length', fields.length);
        if (isGroup) {
          const identities = new WeakMap();
          for (const item of sourceUniforms) for (const itemUniform of Array.isArray(item) ? item : [item]) {
            identities.set(memo.get(itemUniform), itemUniform);
          }
          groups.push({source: value, captured: result, identities});
        }
        return result;
      }
      try {
        if (!material || ![T.ShaderMaterial?.prototype, T.RawShaderMaterial?.prototype].includes(Object.getPrototypeOf(material))) continue;
        descriptors(material);
        if (data(material, 'onBeforeRender') !== T.Material.prototype.onBeforeRender ||
            data(material, 'onBeforeCompile') !== T.Material.prototype.onBeforeCompile ||
            data(material, 'customProgramCacheKey') !== T.Material.prototype.customProgramCacheKey ||
            data(material, 'lights') === true || data(material, 'fog') === true) continue;
        const source = data(material, 'uniforms'), active = activeSource(material);
        const candidates = new Set([source, ...(active === undefined ? [] : [active]), ...(pending.get(material)?.keys() ?? [])]);
        // A skipped GL upload needs the program's existing upload state. Keep
        // that established live path, and never inspect the skipped value.
        for (const dictionary of candidates) {
          if (!plain(dictionary)) throw unsupported;
          for (const field of Object.values(descriptors(dictionary))) inspectUniform(field.value);
        }
        const uniforms = new Map([...candidates].map(dictionary => [dictionary, copy(dictionary)]));
        const sourceGroups = data(material, 'uniformsGroups');
        const capturedGroups = sourceGroups === undefined ? undefined : copy(sourceGroups);
        if (sourceGroups !== undefined && (!Array.isArray(sourceGroups) ||
            sourceGroups.some(group => Object.getPrototypeOf(group ?? {}) !== T.UniformsGroup?.prototype))) throw unsupported;
        if (accepted.size >= maxMaterials) fail('LIMIT', 'Shader uniform snapshot material capacity exceeded');
        accepted.set(material, {source, uniforms, groups: capturedGroups});
        registrations.push(...groups); bytes += localBytes; values += localValues;
      } catch (error) { if (error !== unsupported) throw error; }
    }
    if (!accepted.size) return null;
    if (pending.size + [...accepted.keys()].filter(material => !pending.has(material)).length > maxMaterials)
      fail('LIMIT', 'Pending shader uniform material capacity exceeded');
    // No candidate or group generation is published until the complete frame
    // has passed admission. Releasing a failed/aborted frame removes its inputs.
    for (const {source, captured, identities} of registrations) captureThreeUniformGroup(source, captured, identities);
    for (const [material, entry] of accepted) {
      let candidates = pending.get(material);
      if (!candidates) pending.set(material, candidates = new Map());
      for (const source of entry.uniforms.keys()) candidates.set(source, (candidates.get(source) ?? 0) + 1);
    }
    const record = {entries: accepted, bytes, released: false}; frames.add(record); allocatedBytes += bytes;
    return Object.freeze({bytes,
      sourceFor: material => record.entries.get(material)?.source,
      uniformsFor: (material, source) => record.entries.get(material)?.uniforms.get(source) ?? source,
      groupsFor: material => record.entries.get(material)?.groups,
      has: material => record.entries.has(material),
      release: () => release(record),
    });
  }
  return Object.freeze({capture,
    dispose() { if (!disposed) { disposed = true; for (const record of [...frames]) release(record); pending.clear(); } },
    get allocatedBytes() { return allocatedBytes; },
  });
}
