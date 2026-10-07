/** Per-call state for small, callback-free ShaderMaterial Mesh passes.
 *
 * FullScreenQuad deliberately reuses one mesh and mutable uniforms. Preparation
 * may await pipeline creation, so borrowing that mesh would draw later values.
 * Captures instead install into a private, stable execution view. The application
 * objects are never temporarily rewritten across an await. Textures keep their
 * resource identity: this captures shader inputs, not copies of GPU images.
 */
export class ThreePassSnapshotError extends Error {
  constructor(code, message) {
    super(`THREE_PASS_${code}: ${message}`);
    this.name = 'ThreePassSnapshotError'; this.code = `THREE_PASS_${code}`;
  }
}
const fail = (code, message) => { throw new ThreePassSnapshotError(code, message); };
const own = (object, key) => {
  const d = Object.getOwnPropertyDescriptor(object, key);
  if (d && !Object.hasOwn(d, 'value')) fail('ACCESSOR', `Cannot capture accessor-backed ${key}`);
  return d?.value;
};
const view = source => Object.create(Object.getPrototypeOf(source));
const put = (object, key, value, enumerable = true) => Object.defineProperty(object, key,
  {value, writable: true, configurable: true, enumerable});
const positive = n => Number.isSafeInteger(n) && n > 0;
const TYPED_ARRAYS = new Map([Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array].map(Type => [Type.prototype, Type]));
const MATH = ['Vector2', 'Vector3', 'Vector4', 'Matrix3', 'Matrix4', 'Color', 'Quaternion', 'Euler', 'Plane', 'Sphere', 'Box3', 'Layers', 'Uniform'];
const TEXTURE_STATE = ['version', 'format', 'type', 'colorSpace', 'mapping', 'channel', 'wrapS', 'wrapT', 'magFilter', 'minFilter',
  'anisotropy', 'flipY', 'premultiplyAlpha', 'generateMipmaps', 'rotation', 'matrixAutoUpdate'];

export function createThreePassSnapshots(T, {maxBytes = 32 * 1024 * 1024, maxCaptureBytes = 2 * 1024 * 1024,
  maxRoots = 64, maxVertices = 6, maxValues = 65536} = {}) {
  if (![maxBytes, maxCaptureBytes, maxRoots, maxVertices, maxValues].every(positive)) fail('OPTIONS', 'Expected positive snapshot limits');
  const states = new WeakMap(), retained = new Set(), records = new Set();
  const math = new Set(MATH.map(name => T[name]?.prototype).filter(Boolean));
  let disposed = false, allocatedBytes = 0;
  function live() { if (disposed) fail('DISPOSED', 'Pass snapshot owner is disposed'); }
  const plain = object => Object.getPrototypeOf(object) === Object.prototype || Object.getPrototypeOf(object) === null;
  function eligible(root, camera) {
    // General scenes, instancing, user hooks and large geometry keep their existing
    // explicit live-source path. No user clone()/copy()/serialization hook runs.
    if (!T.Mesh || Object.getPrototypeOf(root ?? {}) !== T.Mesh.prototype || !T.Camera || !(camera instanceof T.Camera)) return false;
    const m = own(root, 'material'), g = own(root, 'geometry');
    if (!m || ![T.ShaderMaterial?.prototype, T.RawShaderMaterial?.prototype].includes(Object.getPrototypeOf(m)) ||
        !T.BufferGeometry || !(g instanceof T.BufferGeometry) || root.children?.length || root.isInstancedMesh || root.isSkinnedMesh || g.isInstancedBufferGeometry) return false;
    if (camera.children?.length || camera.isArrayCamera || (!camera.isPerspectiveCamera && !camera.isOrthographicCamera)) return false;
    if (root.onBeforeRender !== T.Object3D.prototype.onBeforeRender || root.onAfterRender !== T.Object3D.prototype.onAfterRender ||
        root.updateMatrixWorld !== T.Object3D.prototype.updateMatrixWorld ||
        root.updateMatrix !== T.Object3D.prototype.updateMatrix ||
        camera.updateMatrixWorld !== T.Camera.prototype.updateMatrixWorld ||
        camera.updateMatrix !== T.Object3D.prototype.updateMatrix ||
        m.onBeforeRender !== T.Material.prototype.onBeforeRender || m.onBeforeCompile !== T.Material.prototype.onBeforeCompile ||
        m.customProgramCacheKey !== T.Material.prototype.customProgramCacheKey) return false;
    const attributes = own(g, 'attributes'), position = attributes && own(attributes, 'position');
    if (!position || !positive(position.count) || position.count > maxVertices || Object.values(g.morphAttributes ?? {}).some(a => a.length)) return false;
    for (const a of [...Object.values(attributes), ...(g.index ? [g.index] : [])]) {
      const owner = a.isInterleavedBufferAttribute ? a.data : a;
      const expected = owner.isInterleavedBuffer ? T.InterleavedBuffer?.prototype : T.BufferAttribute?.prototype;
      if (!expected || owner.onUploadCallback !== expected.onUploadCallback) return false;
    }
    return true;
  }
  function collect(record) {
    if (record.pending || record.installed || !records.delete(record)) return;
    allocatedBytes -= record.bytes;
    for (const [texture, listener] of record.listeners) texture.removeEventListener('dispose', listener);
    record.listeners.length = 0;
    record.data = record.camera = null;
  }
  function capture(root, camera) {
    live();
    if (!eligible(root, camera)) return null;
    let state = states.get(root);
    if (!state && retained.size >= maxRoots) fail('BUDGET', 'Too many retained fullscreen pass roots');
    let bytes = 0, values = 0;
    const charge = n => {
      bytes += n;
      if (!Number.isSafeInteger(bytes) || bytes > maxCaptureBytes || bytes > maxBytes - allocatedBytes) fail('BUDGET', 'Pass snapshots exceed their byte budget');
      if (++values > maxValues) fail('BUDGET', 'Pass snapshot contains too many values');
    };
    const memo = new Map(), textures = new Map();
    function textureState(t) {
      const result = TEXTURE_STATE.map(k => own(t, k));
      result.push(t.source, t.source?.version, own(t, 'renderTarget'));
      for (const k of ['offset', 'repeat', 'center']) { const v = own(t, k); result.push(v?.x, v?.y); }
      // Auto-updated matrices are derived from the transform fields above; a
      // previous pass may legitimately update that derived matrix during prepare.
      if (t.matrixAutoUpdate === false) result.push(...(t.matrix?.elements ?? []));
      return result;
    }
    function copy(value, depth = 0) {
      if (depth > 32) fail('BUDGET', 'Pass snapshot nesting exceeds 32 levels');
      if (value === null || value === undefined || ['number', 'boolean'].includes(typeof value)) { charge(8); return value; }
      if (typeof value === 'string') { charge(8 + value.length * 2); return value; }
      if (typeof value !== 'object') fail('VALUE', 'Pass inputs must be data, not callable values');
      if (T.Texture && value instanceof T.Texture) {
        if (!textures.has(value)) { charge(256); textures.set(value, textureState(value)); }
        return value;
      }
      if (memo.has(value)) return memo.get(value);
      if (ArrayBuffer.isView(value)) {
        if (typeof SharedArrayBuffer !== 'undefined' && value.buffer instanceof SharedArrayBuffer) fail('VALUE', 'Shared-memory pass inputs need an explicit synchronization boundary');
        charge(value.byteLength + 64);
        const buffer = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
        const Type = TYPED_ARRAYS.get(Object.getPrototypeOf(value));
        if (!Type && Object.getPrototypeOf(value) !== DataView.prototype) fail('VALUE', 'Unsupported typed-array pass input');
        const clone = Type ? new Type(buffer) : new DataView(buffer);
        memo.set(value, clone); return clone;
      }
      if (!Array.isArray(value) && !plain(value) && !math.has(Object.getPrototypeOf(value)))
        fail('VALUE', 'Unsupported mutable uniform value; use plain structs, arrays or source math values');
      const clone = Array.isArray(value) ? [] : view(value); memo.set(value, clone); charge(64 + (Array.isArray(value) ? value.length * 8 : 0));
      for (const key of Reflect.ownKeys(value)) {
        if (key === 'length' && Array.isArray(value)) continue;
        // These are internal math change notifications, not shader inputs.
        if (key === '_onChangeCallback' && math.has(Object.getPrototypeOf(value))) continue;
        if (typeof key !== 'string') fail('VALUE', 'Symbol-backed pass data is not admitted');
        charge(key.length * 2 + 16); put(clone, key, copy(own(value, key), depth + 1));
      }
      if (Array.isArray(value)) clone.length = value.length;
      return clone;
    }
    function fields(source, special = {}) {
      const out = view(source); charge(128);
      for (const key of Object.getOwnPropertyNames(source)) {
        if (Object.hasOwn(special, key)) continue;
        if (['_listeners', 'userData'].includes(key)) continue;
        const value = own(source, key);
        // Only the default methods admitted above may be own properties.
        if (typeof value === 'function') {
          const prototype = Object.getPrototypeOf(source);
          if (value !== prototype[key]) fail('VALUE', `Custom callable pass field: ${key}`);
          put(out, key, value); continue;
        }
        put(out, key, copy(value));
      }
      for (const [key, value] of Object.entries(special)) put(out, key, value);
      return out;
    }
    const sourceMaterial = root.material, sourceGeometry = root.geometry;
    const attributes = {}, attributeSources = {}, owners = new Map(), interleavedSources = new Map();
    function attribute(a) {
      if (owners.has(a)) return owners.get(a);
      const special = {};
      if (a.isInterleavedBufferAttribute) { const data = a.data; interleavedSources.set(a, data); special.data = attribute(data); }
      const cloned = fields(a, special); owners.set(a, cloned); return cloned;
    }
    for (const [name, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(sourceGeometry.attributes))) {
      if (!Object.hasOwn(descriptor, 'value')) fail('ACCESSOR', `Accessor-backed geometry attribute: ${name}`);
      attributes[name] = attribute(descriptor.value); attributeSources[name] = descriptor.value;
    }
    const indexSource = sourceGeometry.index, index = indexSource ? attribute(indexSource) : null;
    const geometry = fields(sourceGeometry, {attributes, index, morphAttributes: {}, indirect: null});
    const material = fields(sourceMaterial);
    // Updating default transforms is part of the source render boundary. Do this
    // once on the original, then freeze only the private execution matrices.
    if (root.matrixWorldAutoUpdate === true) root.updateMatrixWorld();
    if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
    const rootData = fields(root, {geometry, material, parent: null, children: [], animations: [],
      matrixAutoUpdate: false, matrixWorldAutoUpdate: false, matrixWorldNeedsUpdate: false});
    const cameraData = fields(camera, {parent: null, children: [], animations: [],
      matrixAutoUpdate: false, matrixWorldAutoUpdate: false, matrixWorldNeedsUpdate: false});
    if (!state) {
      state = {root: view(root), record: null, material: null, geometry: null, attributeViews: new Map()};
      // Type flags suffice until the first serialized install; it is not an
      // independently traversed render root until that request is prepared.
      put(state.root, 'isMesh', true); put(state.root, 'isObject3D', true);
      states.set(root, state); retained.add(state);
    }
    const record = {state, data: rootData, camera: cameraData, bytes, pending: true, installed: false, listeners: [], stale: false};
    allocatedBytes += bytes; records.add(record);
    for (const t of textures.keys()) {
      const listener = () => { record.stale = true; };
      record.listeners.push([t, listener]); t.addEventListener('dispose', listener);
    }
    function check() {
      live();
      if (!record.pending) fail('RELEASED', 'Pass capture was already released');
      if (record.stale) fail('STALE', 'A captured pass texture was disposed before submission');
      for (const [t, expected] of textures) {
        const current = textureState(t);
        if (current.length !== expected.length || current.some((v, i) => !Object.is(v, expected[i])))
          fail('STALE', 'A captured pass texture changed before submission');
      }
    }
    function overwrite(target, source) {
      for (const key of Object.getOwnPropertyNames(target)) if (key !== '_listeners' && !Object.hasOwn(source, key)) delete target[key];
      for (const key of Object.getOwnPropertyNames(source)) put(target, key, source[key]);
    }
    function install() {
      check();
      if (state.record === record) return;
      const previous = state.record;
      if (previous) { previous.installed = false; collect(previous); }
      if (state.materialSource !== sourceMaterial) { state.material = view(material); state.materialSource = sourceMaterial; }
      overwrite(state.material, material);
      if (state.geometrySource !== sourceGeometry) {
        state.geometry = view(geometry); state.geometrySource = sourceGeometry; state.attributeViews.clear();
      }
      const installedOwners = new Map();
      function installAttribute(source, captured) {
        if (installedOwners.has(source)) return installedOwners.get(source);
        let target = state.attributeViews.get(source);
        if (!target) { target = view(captured); state.attributeViews.set(source, target); }
        overwrite(target, captured); installedOwners.set(source, target);
        if (captured.isInterleavedBufferAttribute) target.data = installAttribute(interleavedSources.get(source), captured.data);
        return target;
      }
      const current = {};
      for (const [name, captured] of Object.entries(attributes)) current[name] = installAttribute(attributeSources[name], captured);
      const installedIndex = index ? installAttribute(indexSource, index) : null;
      overwrite(state.geometry, geometry); state.geometry.attributes = current; state.geometry.index = installedIndex;
      for (const source of state.attributeViews.keys()) if (!installedOwners.has(source)) state.attributeViews.delete(source);
      overwrite(state.root, rootData); state.root.material = state.material; state.root.geometry = state.geometry;
      record.installed = true; state.record = record;
    }
    return Object.freeze({root: state.root, camera: cameraData, bytes, check, install,
      release() { if (!record.pending) return; record.pending = false; collect(record); }});
  }
  return Object.freeze({capture,
    get allocatedBytes() { return allocatedBytes; }, get rootCount() { return retained.size; },
    dispose() {
      if (disposed) return; disposed = true;
      for (const record of [...records]) { record.pending = record.installed = false; collect(record); }
      retained.clear();
    },
  });
}
