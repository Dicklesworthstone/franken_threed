/** Per-call state for small, callback-free shader and basic-material Mesh passes.
 *
 * FullScreenQuad deliberately reuses one mesh and mutable uniforms. Preparation
 * may await pipeline creation, so borrowing that mesh would draw later values.
 * Captures instead install into a private, stable execution view. The application
 * objects are never temporarily rewritten across an await. Textures keep their
 * resource identity: this captures shader inputs, not copies of GPU images.
 */
import {captureThreeUniformGroup} from './three_uniform_group_state.mjs';
import {claimWebGLGeometrySnapshot} from './gpu_buffer_geometry.mjs';

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
  'anisotropy', 'compareFunction', 'flipY', 'premultiplyAlpha', 'generateMipmaps', 'rotation', 'matrixAutoUpdate'];

/** The projection portion of pinned Renderer._updateCamera. Reversal is a
 * source-visible, one-way camera transition, including ArrayCamera children.
 * Matrix-world updates stay at the caller's existing source-order boundary. */
export function updateThreeCameraProjection(T, renderer, camera) {
  let changed = false;
  if (renderer?.reversedDepthBuffer === true && camera.reversedDepth !== true) {
    camera._reversedDepth = true;
    if (camera.isArrayCamera) for (const child of camera.cameras) child._reversedDepth = true;
    changed = true;
  }
  const coordinateSystem = renderer?.coordinateSystem ?? T.WebGPUCoordinateSystem;
  if (camera.coordinateSystem !== coordinateSystem) {
    camera.coordinateSystem = coordinateSystem;
    if (camera.isArrayCamera) for (const child of camera.cameras) child.coordinateSystem = coordinateSystem;
    changed = true;
  }
  if (changed) {
    if (typeof camera.updateProjectionMatrix !== 'function') fail('CAMERA', 'The source camera cannot update its projection');
    camera.updateProjectionMatrix();
    if (camera.isArrayCamera) for (const child of camera.cameras) child.updateProjectionMatrix();
  }
}

/** Native callbacks observe the source camera before this private execution
 * view is captured. Queued fixed draws then keep that call's projection, pose,
 * layers and viewport without invoking source clone/copy hooks. */
export function captureThreeCamera(T, source, array = false) {
  if (!(source instanceof T.Camera) || (!source.isPerspectiveCamera && !source.isOrthographicCamera) ||
      (!array && source.isArrayCamera) || source.isMultiViewCamera)
    fail('CAMERA', 'Native camera views require perspective or orthographic cameras');
  const result = view(source);
  for (const key of ['id', 'uuid', 'name', 'type', 'isCamera', 'isPerspectiveCamera', 'isOrthographicCamera',
    'isArrayCamera', 'isMultiViewCamera', 'coordinateSystem', 'near', 'far', 'zoom', 'fov', 'aspect', '_reversedDepth',
    'filmGauge', 'filmOffset', 'left', 'right', 'top', 'bottom'])
    if (key in source) put(result, key, source[key]);
  // WebGL cube captures can reach their first reversed-depth draw after being
  // queued. Their projection update needs the accepted lens/frustum inputs,
  // including a private view offset, rather than later source-camera edits.
  put(result, 'view', source.view === null ? null : {...source.view});
  for (const key of ['matrixWorld', 'matrixWorldInverse', 'projectionMatrix', 'projectionMatrixInverse'])
    put(result, key, new T.Matrix4().copy(source[key]));
  const layers = new T.Layers(); layers.mask = source.layers.mask; put(result, 'layers', layers);
  if (source.viewport) put(result, 'viewport', new T.Vector4().copy(source.viewport));
  put(result, 'parent', null); put(result, 'children', []); put(result, 'matrixWorldAutoUpdate', false);
  return result;
}

/** Background selectors and numeric controls belong to the accepted render,
 * including an explicit null while retained PMREM captures a Color scene.
 * Textures/nodes retain resource identity; mutable Color/Euler values do not. */
export function captureThreeBackground(T, scene) {
  if (scene.isScene !== true) return undefined;
  const background = scene.background, rotation = scene.backgroundRotation;
  return Object.freeze({
    background: background?.isColor ? new T.Color().copy(background) : background,
    backgroundNode: scene.backgroundNode,
    backgroundBlurriness: scene.backgroundBlurriness, backgroundIntensity: scene.backgroundIntensity,
    backgroundRotation: rotation?.isEuler ? new T.Euler(rotation.x, rotation.y, rotation.z, rotation.order) : rotation,
  });
}

/** An ArrayCamera render may wait for scene preparation. Keep every view's
 * call-time matrices, layers and viewport in a private data view; later changes
 * to the cameras array or its children must not retarget an earlier render.
 * The renderer performs the source matrix updates before calling this helper.
 * No source clone/copy hook or temporary write to an application camera runs.
 */
export function captureThreeArrayCamera(T, camera, maxViews = 256) {
  if (!(camera instanceof T.Camera) || camera.isArrayCamera !== true ||
      !Array.isArray(camera.cameras) || camera.cameras.length > maxViews || camera.isMultiViewCamera)
    fail('CAMERA', 'Expected an ordinary ArrayCamera within the view capacity');
  const copied = new Map();
  function capture(source, array = false) {
    if (copied.has(source)) return copied.get(source);
    const result = captureThreeCamera(T, source, array); copied.set(source, result);
    return result;
  }
  const result = capture(camera, true);
  put(result, 'cameras', camera.cameras.map(child => capture(child)));
  return result;
}

export function createThreePassSnapshots(T, {targets = null, maxBytes = 32 * 1024 * 1024, maxCaptureBytes = 2 * 1024 * 1024,
  maxRoots = 64, maxVertices = 6, maxValues = 65536} = {}) {
  if (![maxBytes, maxCaptureBytes, maxRoots, maxVertices, maxValues].every(positive)) fail('OPTIONS', 'Expected positive snapshot limits');
  const states = new WeakMap(), retained = new Set(), records = new Set();
  const math = new Set(MATH.map(name => T[name]?.prototype).filter(Boolean));
  let disposed = false, allocatedBytes = 0, pmremScenes = null;
  function live() { if (disposed) fail('DISPOSED', 'Pass snapshot owner is disposed'); }
  const plain = object => Object.getPrototypeOf(object) === Object.prototype || Object.getPrototypeOf(object) === null;
  function eligible(root, camera, vertexLimit = maxVertices) {
    // General scenes, instancing, user hooks and large geometry keep their existing
    // explicit live-source path. No user clone()/copy()/serialization hook runs.
    if (!T.Mesh || Object.getPrototypeOf(root ?? {}) !== T.Mesh.prototype || !T.Camera || !(camera instanceof T.Camera)) return false;
    const m = own(root, 'material'), g = own(root, 'geometry');
    if (!m || ![T.ShaderMaterial?.prototype, T.RawShaderMaterial?.prototype, T.MeshBasicMaterial?.prototype].includes(Object.getPrototypeOf(m)) ||
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
    if (!position || !positive(position.count) || position.count > vertexLimit || Object.values(g.morphAttributes ?? {}).some(a => a.length)) return false;
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
    for (const release of record.targetReleases) release();
    record.targetReleases.length = 0;
    record.data = record.camera = null;
  }
  function capture(root, camera, overrideMaterial = null, vertexLimit = maxVertices) {
    live();
    if (overrideMaterial) {
      if (![T.ShaderMaterial?.prototype, T.RawShaderMaterial?.prototype].includes(Object.getPrototypeOf(overrideMaterial)) ||
          Object.values(Object.getOwnPropertyDescriptors(overrideMaterial)).some(d => !Object.hasOwn(d, 'value')) ||
          overrideMaterial.onBeforeRender !== T.Material.prototype.onBeforeRender ||
          overrideMaterial.onBeforeCompile !== T.Material.prototype.onBeforeCompile ||
          overrideMaterial.customProgramCacheKey !== T.Material.prototype.customProgramCacheKey) return null;
    } else if (!eligible(root, camera, vertexLimit)) return null;
    const identity = overrideMaterial ?? root;
    let state = states.get(identity);
    if (!state && retained.size >= maxRoots) fail('BUDGET', 'Too many retained pass inputs');
    let bytes = 0, values = 0;
    const charge = n => {
      bytes += n;
      if (!Number.isSafeInteger(bytes) || bytes > maxCaptureBytes || bytes > maxBytes - allocatedBytes - (pmremScenes?.bytes ?? 0)) fail('BUDGET', 'Pass snapshots exceed their byte budget');
      if (++values > maxValues) fail('BUDGET', 'Pass snapshot contains too many values');
    };
    const memo = new Map(), textures = new Map(), targetEntries = new Map();
    function textureState(t) {
      const result = TEXTURE_STATE.map(k => own(t, k));
      const target = own(t, 'renderTarget');
      result.push(t.source, t.source?.version, target);
      // The retained owner validates an exact generation across a verified
      // composer resize. Other texture/sampler edits still invalidate capture.
      if (target && !targetEntries.has(t)) result.push(target.width, target.height, target.depth, target.samples, target.texture,
        target.depthBuffer, target.stencilBuffer, target.depthTexture, target.resolveColorBuffer);
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
      if (T.UniformsGroup && Object.getPrototypeOf(value) === T.UniformsGroup.prototype) {
        const clone = view(value); memo.set(value, clone); charge(128);
        for (const key of ['id', 'name', 'usage', 'isUniformsGroup']) put(clone, key, copy(own(value, key), depth + 1));
        const uniforms = own(value, 'uniforms');
        put(clone, 'uniforms', copy(uniforms, depth + 1));
        const sources = new WeakMap();
        for (const item of uniforms) for (const uniform of Array.isArray(item) ? item : [item])
          sources.set(memo.get(uniform), uniform);
        captureThreeUniformGroup(value, clone, sources);
        return clone;
      }
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
        // The group owner retains its layout/upload history. Copying these
        // derived fields would duplicate it and retain obsolete buffer payloads.
        if (Object.getPrototypeOf(value) === T.Uniform?.prototype && ['__data', '__offset'].includes(key)) continue;
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
    const sourceMaterial = overrideMaterial ?? root.material, sourceGeometry = root?.geometry;
    const sourceUniforms = own(sourceMaterial, 'uniforms');
    const attributes = {}, attributeSources = {}, owners = new Map(), interleavedSources = new Map();
    function attribute(a) {
      if (owners.has(a)) return owners.get(a);
      const special = {};
      if (a.isInterleavedBufferAttribute) { const data = a.data; interleavedSources.set(a, data); special.data = attribute(data); }
      const cloned = fields(a, special); owners.set(a, cloned); return cloned;
    }
    for (const [name, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(sourceGeometry?.attributes ?? {}))) {
      if (!Object.hasOwn(descriptor, 'value')) fail('ACCESSOR', `Accessor-backed geometry attribute: ${name}`);
      attributes[name] = attribute(descriptor.value); attributeSources[name] = descriptor.value;
    }
    const indexSource = sourceGeometry?.index, index = indexSource ? attribute(indexSource) : null;
    const geometry = sourceGeometry ? fields(sourceGeometry, {attributes, index, morphAttributes: {}, indirect: null}) : null;
    const material = fields(sourceMaterial);
    // Updating default transforms is part of the source render boundary. Do this
    // once on the original, then freeze only the private execution matrices.
    if (root?.matrixWorldAutoUpdate === true) root.updateMatrixWorld();
    if (camera?.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
    const rootData = root ? fields(root, {geometry, material, parent: null, children: [], animations: [],
      matrixAutoUpdate: false, matrixWorldAutoUpdate: false, matrixWorldNeedsUpdate: false}) : null;
    const cameraData = camera ? fields(camera, {parent: null, children: [], animations: [],
      matrixAutoUpdate: false, matrixWorldAutoUpdate: false, matrixWorldNeedsUpdate: false}) : null;
    const targetReleases = [], heldTargets = new Set();
    try {
      for (const texture of textures.keys()) {
        const target = targets?.getTextureTarget(texture);
        if (!target) continue;
        const entry = targets.capture(target);
        if (!heldTargets.has(entry)) {
          targetReleases.push(targets.retainGeneration(entry)); heldTargets.add(entry);
        }
        targetEntries.set(texture, entry);
        textures.set(texture, textureState(texture));
      }
    } catch (error) {
      for (const release of targetReleases) release();
      throw error;
    }
    if (!state) {
      state = {root: root ? view(root) : null, record: null, material: overrideMaterial ? view(material) : null,
        materialSource: overrideMaterial, geometry: null, attributeViews: new Map(), uniformViews: new WeakMap()};
      // Type flags suffice until the first serialized install; it is not an
      // independently traversed render root until that request is prepared.
      if (state.root) { put(state.root, 'isMesh', true); put(state.root, 'isObject3D', true); }
      states.set(identity, state); retained.add(state);
    }
    const record = {state, data: rootData, camera: cameraData, bytes, pending: true, installed: false, listeners: [], stale: false,
      targetReleases};
    allocatedBytes += bytes; records.add(record);
    const resources = new Set(textures.keys());
    for (const t of textures.keys()) if (t.renderTarget) resources.add(t.renderTarget);
    const retainedTargets = new Set([...heldTargets].map(entry => entry.target));
    for (const target of retainedTargets) resources.add(target);
    for (const resource of resources) {
      // Target disposal is classified by its existing owner: a trusted resize
      // preserves this generation, while explicit disposal still retires it.
      const listener = () => { if (!retainedTargets.has(resource)) record.stale = true; };
      record.listeners.push([resource, listener]); resource.addEventListener('dispose', listener);
    }
    function check() {
      live();
      if (!record.pending) fail('RELEASED', 'Pass capture was already released');
      if (record.stale) fail('STALE', 'A captured pass texture was disposed before submission');
      for (const entry of heldTargets) {
        try { targets.assertCurrent(entry); }
        catch { fail('STALE', 'A captured pass target changed or was disposed before submission'); }
      }
      for (const [t, expected] of textures) {
        const current = textureState(t);
        if (current.length !== expected.length || current.some((v, i) => !Object.is(v, expected[i])))
          fail('STALE', 'A captured pass texture changed before submission');
      }
    }
    function overwrite(target, source, listeners = true) {
      for (const key of Object.getOwnPropertyNames(target)) if ((!listeners || key !== '_listeners') && !Object.hasOwn(source, key)) delete target[key];
      for (const key of Object.getOwnPropertyNames(source)) put(target, key, source[key]);
    }
    function install() {
      check();
      if (state.record === record) return;
      const previous = state.record;
      if (previous) { previous.installed = false; collect(previous); }
      if (state.materialSource !== sourceMaterial) { state.material = view(material); state.materialSource = sourceMaterial; }
      // WebGL programs retain their first uniforms dictionary. Keep its private
      // identity while installing each accepted value set, just as mutations
      // of the original dictionary remain visible to an already linked program.
      // Application preset lists must not pin old copied payloads: identity is
      // needed only while an execution view or linked program retains the copy.
      if (sourceUniforms && typeof sourceUniforms === 'object') {
        let uniforms = state.uniformViews.get(sourceUniforms)?.deref();
        if (!uniforms) { uniforms = view(material.uniforms); state.uniformViews.set(sourceUniforms, new WeakRef(uniforms)); }
        overwrite(uniforms, material.uniforms, false); material.uniforms = uniforms;
      }
      overwrite(state.material, material);
      if (overrideMaterial) { record.installed = true; state.record = record; return; }
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
    return Object.freeze({root: state.root, camera: cameraData, material: overrideMaterial ? state.material : null,
      bytes, targetEntries, check, install,
      release() { if (!record.pending) return; record.pending = false; collect(record); }});
  }
  return Object.freeze({capture,
    bindDevice(device) { live(); pmremScenes?.bindDevice(device); },
    capturePMREMScene(root, camera, device, options) {
      live();
      pmremScenes ??= createPMREMSceneSnapshots(T, {maxBytes, maxCaptureBytes, maxRoots, maxValues,
        available: () => maxBytes - allocatedBytes});
      return pmremScenes.capture(root, camera, device, options);
    },
    // The retained r186 PMREM helper draws six faces in one 36-vertex mesh.
    // Its caller verifies the generator's private mesh/material identities;
    // keep this larger capture profile restricted to the source plane layout.
    capturePMREM(root, camera) {
      const geometry = own(root, 'geometry'), attributes = geometry && own(geometry, 'attributes');
      const position = attributes && own(attributes, 'position'), direction = attributes && own(attributes, 'outputDirection');
      if (!attributes || Object.keys(attributes).length !== 2 || geometry.index !== null ||
          ![position, direction].every(attribute => attribute?.itemSize === 3 && attribute.count === 36 &&
            Object.getPrototypeOf(attribute.array) === Float32Array.prototype)) return null;
      return capture(root, camera, null, 36);
    },
    // A verified pass's scene-wide ShaderMaterial override reuses the same
    // bounded value copies and sampled-target leases as a fullscreen pass.
    // Its stable private material is installed only during serialized work.
    captureMaterial(material) { return material ? capture(null, null, material) : null; },
    // A retained generator has a finite private mesh generation. Retire those
    // roots only after its accepted draws drain, without evicting application
    // fullscreen roots or resetting their shared uniform/upload identities.
    retire(root) {
      if (pmremScenes?.has(root)) { pmremScenes.retire(root); return; }
      const state = states.get(root);
      if (!state) return;
      const owned = [...records].filter(record => record.state === state);
      if (owned.some(record => record.pending)) fail('PENDING', 'Cannot retire a pass root with queued draws');
      for (const record of owned) { record.installed = false; collect(record); }
      for (const value of [state.root, state.material, state.geometry])
        if (value) for (const key of Object.getOwnPropertyNames(value)) delete value[key];
      state.record = state.material = state.geometry = state.materialSource = state.geometrySource = null;
      state.attributeViews.clear(); state.uniformViews = new WeakMap();
      states.delete(root); retained.delete(state);
    },
    get allocatedBytes() { return allocatedBytes + (pmremScenes?.bytes ?? 0); }, get rootCount() { return retained.size + (pmremScenes?.roots ?? 0); },
    dispose() {
      if (disposed) return; disposed = true; pmremScenes?.dispose();
      for (const record of [...records]) { record.pending = record.installed = false; collect(record); }
      // WeakMap entries and renderer lookup keys may outlive disposal. Leave no
      // copied uniform/geometry payload reachable through those private views.
      for (const state of retained) {
        const data = state.root ?? state.material;
        if (data) for (const key of Object.getOwnPropertyNames(data)) delete data[key];
        state.record = state.material = state.geometry = state.materialSource = state.geometrySource = null;
        state.attributeViews.clear(); state.uniformViews = new WeakMap();
      }
      retained.clear();
    },
  });
}

/** Detached, callback-free WebGL PMREM scene views. These retain source upload
 * versions across captures; their attribute owner is intentionally separate
 * from ordinary WebGL scene bridges until an explicit disposal boundary. */
function createPMREMSceneSnapshots(T, {maxBytes, maxCaptureBytes, maxRoots, maxValues, available}) {
  const states = new WeakMap(), retained = new Set(), records = new Set(), histories = new Map(), resources = new Map();
  const duplicate = value => new (TYPED_ARRAYS.get(Object.getPrototypeOf(value)))(value);
  const token = {}, math = new Set([...MATH, 'Fog', 'FogExp2'].map(name => T[name]?.prototype).filter(Boolean));
  let bytes = 0, historyBytes = 0, disposed = false, boundDevice = null;
  const field = (object, key) => {
    for (let value = object; value; value = Object.getPrototypeOf(value))
      if (Object.hasOwn(value, key)) return own(value, key);
  };
  const live = () => { if (disposed) fail('DISPOSED', 'PMREM scene snapshot owner is disposed'); };
  const releaseHistory = owner => {
    const history = histories.get(owner);
    if (history) { histories.delete(owner); historyBytes -= history.array.byteLength; history.release?.(); }
  };
  function bindDevice(device) {
    live();
    if (!device || boundDevice && boundDevice !== device) fail('SCENE', 'PMREM geometry histories require one renderer device');
    if (boundDevice === device) return;
    const claimed = [];
    try {
      for (const [owner, history] of histories) if (!history.release)
        claimed.push([history, claimWebGLGeometrySnapshot(device, owner, token)]);
    } catch (error) { for (const [, release] of claimed) release(); throw error; }
    for (const [history, release] of claimed) history.release = release;
    boundDevice = device;
  }
  function observe(resource, owners) {
    let record = resources.get(resource);
    if (!record) {
      const listener = () => {
        resource.removeEventListener('dispose', listener);
        resources.delete(resource); for (const owner of record.owners) releaseHistory(owner);
      };
      resources.set(resource, record = {owners: new Set(), listener}); resource.addEventListener('dispose', listener);
    }
    for (const owner of owners) record.owners.add(owner);
  }
  function collect(record) {
    if (record.pending || record.installed || !records.delete(record)) return;
    bytes -= record.bytes; record.data = record.sources = null;
  }
  function capture(root, camera, device, {cube = false, position = null, sortObjects = true} = {}) {
    live();
    if (device) bindDevice(device);
    if (!(root instanceof T.Object3D) || !(camera instanceof T.Camera) || camera.isArrayCamera ||
        camera.updateMatrixWorld !== T.Camera.prototype.updateMatrixWorld)
      fail('SCENE', 'PMREM scene capture requires source objects and an ordinary camera');
    let state = states.get(root);
    if (!state && retained.size >= maxRoots) fail('BUDGET', 'Too many PMREM scene roots');
    const nodes = [], stack = [root], seen = new Set(), sourceResources = new Map();
    const sourceOwners = geometry => [...Object.values(geometry.attributes), ...(geometry.index ? [geometry.index] : [])]
      .map(attribute => attribute.isInterleavedBufferAttribute ? attribute.data : attribute);
    while (stack.length) {
      const object = stack.pop();
      if (!(object instanceof T.Object3D) || seen.has(object) || seen.size >= maxValues || !Array.isArray(object.children))
        fail('SCENE', 'PMREM capture requires a bounded source hierarchy');
      seen.add(object); nodes.push(object);
      if (field(object, 'onBeforeRender') !== T.Object3D.prototype.onBeforeRender || field(object, 'onAfterRender') !== T.Object3D.prototype.onAfterRender ||
          field(object, 'updateMatrixWorld') !== T.Object3D.prototype.updateMatrixWorld || field(object, 'updateMatrix') !== T.Object3D.prototype.updateMatrix)
        fail('SCENE', 'PMREM scene callbacks require their source synchronous draw owner');
      if (object.isScene && (object.background && !object.background.isColor || object.environment || object.overrideMaterial))
        fail('SCENE', 'PMREM scene capture currently requires a solid background without an environment or override');
      if (object.isLOD || object.isSkinnedMesh || object.isBatchedMesh || object.isLine || object.isPoints || object.isSprite ||
          object.isLight && object.castShadow || object.isInstancedMesh && object.morphTexture)
        fail('SCENE', 'PMREM capture requires rigid meshes, CPU instances and unshadowed source lights');
      if (object.isMesh) {
        const geometry = object.geometry, materials = Array.isArray(object.material) ? object.material : [object.material];
        if (!(geometry instanceof T.BufferGeometry) || Object.values(geometry.morphAttributes).some(list => list.length))
          fail('SCENE', 'PMREM scene geometry requires source BufferGeometry without morph targets');
        for (const material of materials) if (material && (
          !(material instanceof T.Material) || field(material, 'onBeforeRender') !== T.Material.prototype.onBeforeRender ||
          field(material, 'onBeforeCompile') !== T.Material.prototype.onBeforeCompile ||
          field(material, 'customProgramCacheKey') !== T.Material.prototype.customProgramCacheKey || material.transmission > 0 || material.wireframe))
          fail('SCENE', 'PMREM scene materials require ordinary source programs without custom hooks or transmission');
        sourceResources.set(geometry, sourceOwners(geometry));
        if (field(object, 'intersectsFrustum') !== T.Mesh.prototype.intersectsFrustum ||
            field(geometry, 'computeBoundingSphere') !== T.BufferGeometry.prototype.computeBoundingSphere ||
            object.isInstancedMesh && field(object, 'computeBoundingSphere') !== T.InstancedMesh.prototype.computeBoundingSphere)
          fail('SCENE', 'PMREM bounds require ordinary source computation');
        if (object.isInstancedMesh) sourceResources.set(object, [object.instanceMatrix, ...(object.instanceColor ? [object.instanceColor] : [])]);
      }
      stack.push(...object.children);
    }
    if (root.matrixWorldAutoUpdate === true) root.updateMatrixWorld();
    if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();

    const frusta = [], project = new T.Matrix4(), forward = [1, 1, 1, -1, -1, -1], up = [1, -1, 1, 1, 1, 1];
    for (let face = 0; face < (cube ? 6 : 1); face++) {
      if (cube) {
        const axis = face % 3;
        camera.position.copy(position); camera.up.set(0, axis === 1 ? 0 : up[face], axis === 1 ? up[face] : 0);
        const look = new T.Vector3().copy(position); look.setComponent(axis, look.getComponent(axis) + forward[face]); camera.lookAt(look);
        camera.updateMatrixWorld();
      }
      frusta.push(new T.Frustum().setFromProjectionMatrix(project.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
        camera.coordinateSystem, camera.reversedDepth));
    }
    const selectedOwners = new Set(), visible = [root];
    while (visible.length) {
      const object = visible.pop(); if (object.visible === false) continue;
      if (object.isMesh && object.layers.test(camera.layers) &&
          (!object.frustumCulled || frusta.some(frustum => frustum.intersectsObject(object)))) {
        const geometry = object.geometry, materials = Array.isArray(object.material)
          ? geometry.groups.map(group => [object.material[group.materialIndex], group]) : [[object.material, null]];
        // WebGLObjects uploads vertex/instance data before material selection.
        for (const attribute of Object.values(geometry.attributes))
          selectedOwners.add(attribute.isInterleavedBufferAttribute ? attribute.data : attribute);
        if (object.isInstancedMesh) for (const owner of sourceResources.get(object)) selectedOwners.add(owner);
        if (sortObjects) {
          const bounds = object.boundingSphere !== undefined ? object : geometry;
          if (bounds.boundingSphere === null) bounds.computeBoundingSphere();
        }
        if (geometry.index && materials.some(([material, group]) => {
          const range = geometry.drawRange, start = Math.max(0, range.start, group?.start ?? 0),
            end = Math.min(geometry.index.count, range.start + range.count, group ? group.start + group.count : Infinity);
          return material?.visible && end - start >= 0 && end - start !== Infinity;
        })) selectedOwners.add(geometry.index);
      }
      visible.push(...object.children);
    }
    let charged = 0, values = 0, addedHistory = 0;
    const pending = new Map(), sources = new Map(), memo = new Map();
    const charge = size => {
      charged += size;
      if (++values > maxValues || charged > maxCaptureBytes || charged + bytes + historyBytes + addedHistory > Math.min(maxBytes, available()))
        fail('BUDGET', 'PMREM scene snapshots exceed their byte/value budget');
    };
    const attributeOwners = new Set([...sourceResources.values()].flat());
    function uploaded(owner) {
      if (pending.has(owner)) return pending.get(owner);
      const previous = histories.get(owner), array = own(owner, 'array'), version = own(owner, 'version');
      const prototype = owner.isInterleavedBuffer ? T.InterleavedBuffer.prototype : T.BufferAttribute.prototype;
      if (!TYPED_ARRAYS.has(Object.getPrototypeOf(array ?? {})) || !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable ||
          field(owner, 'onUploadCallback') !== prototype.onUploadCallback || field(owner, 'clearUpdateRanges') !== prototype.clearUpdateRanges ||
          owner.isStorageBufferAttribute || owner.isStorageInstancedBufferAttribute || !Number.isSafeInteger(version) || version < 0)
        fail('SCENE', 'PMREM attributes require fixed CPU arrays and default upload callbacks');
      if (previous && (previous.array.constructor !== array.constructor || previous.array.length !== array.length))
        fail('SCENE', 'Replace or dispose resized PMREM attributes before another capture');
      const copies = !previous || version > previous.version ? 2 : 1;
      if (array.byteLength > maxCaptureBytes - charged ||
          charged + copies * array.byteLength + bytes + historyBytes + addedHistory > Math.min(maxBytes, available()))
        fail('BUDGET', 'PMREM attribute history exceeds the remaining snapshot budget');
      const next = {array: previous?.array ?? duplicate(array), version: previous?.version ?? version, clear: false, release: previous?.release};
      if (!previous) {
        addedHistory += array.byteLength;
        if (boundDevice) next.release = claimWebGLGeometrySnapshot(boundDevice, owner, token);
      } else if (version > previous.version) {
        next.array = duplicate(previous.array); next.version = version;
        const ranges = owner.updateRanges.map(range => ({start: range.start, count: range.count, source: range}));
        for (const range of ranges) if (!Number.isSafeInteger(range.start) || range.start < 0 ||
            !Number.isSafeInteger(range.count) || range.count < 0 || range.start + range.count > array.length)
          fail('SCENE', 'Invalid PMREM attribute update range');
        ranges.sort((a, b) => a.start - b.start);
        const merged = [];
        for (const range of ranges) {
          const last = merged.at(-1);
          if (last && range.start <= last.start + last.count + 1) last.count = Math.max(last.count, range.start + range.count - last.start);
          else merged.push(range);
        }
        if (!merged.length) next.array.set(array);
        else for (const {start, count} of merged) next.array.set(array.subarray(start, count ? start + count : array.length), start);
        next.clear = merged.length > 0; next.ranges = merged;
      }
      pending.set(owner, next); return next;
    }
    function copy(value, depth = 0) {
      if (depth > 48) fail('BUDGET', 'PMREM scene data nesting is too deep');
      if (value == null || ['number', 'boolean', 'string'].includes(typeof value)) { charge(typeof value === 'string' ? value.length * 2 : 8); return value; }
      if (typeof value !== 'object') fail('SCENE', 'PMREM capture requires ordinary data fields');
      if (value instanceof T.Texture) fail('SCENE', 'PMREM scene texture inputs require their explicit uploaded-history owner');
      if (memo.has(value)) return memo.get(value);
      if (ArrayBuffer.isView(value)) {
        if (!TYPED_ARRAYS.has(Object.getPrototypeOf(value)) || !(value.buffer instanceof ArrayBuffer) || value.buffer.resizable)
          fail('SCENE', 'PMREM capture requires ordinary fixed typed arrays');
        charge(value.byteLength); const result = duplicate(value); memo.set(value, result); return result;
      }
      const object = value instanceof T.Object3D, geometry = value instanceof T.BufferGeometry, material = value instanceof T.Material;
      const attribute = attributeOwners.has(value) || value.isInterleavedBufferAttribute === true;
      const prototype = Object.getPrototypeOf(value), array = Array.isArray(value);
      if (!object && !geometry && !material && !attribute && !array && !math.has(prototype) &&
          prototype !== Object.prototype && prototype !== null)
        fail('SCENE', 'Unsupported mutable PMREM scene field');
      charge(128); const result = array ? [] : view(value); memo.set(value, result); sources.set(result, value);
      const upload = attributeOwners.has(value) ? selectedOwners.has(value) ? uploaded(value) : histories.get(value) : null;
      for (const key of Object.getOwnPropertyNames(value)) {
        if (['_listeners', 'userData', '_onChangeCallback'].includes(key) || array && key === 'length' ||
            object && ['parent', 'children', 'animations', 'shadow'].includes(key)) continue;
        let input = own(value, key);
        if (upload && key === 'array') input = upload.array;
        if (upload && key === 'updateRanges') input = [];
        if (typeof input === 'function') {
          if (input !== prototype[key]) fail('SCENE', 'Custom callable PMREM fields require their source owner');
          put(result, key, input); continue;
        }
        put(result, key, copy(input, depth + 1));
      }
      if (array) result.length = value.length;
      if (object) {
        put(result, 'parent', null); put(result, 'children', copy(value.children, depth + 1));
        put(result, 'animations', []); put(result, 'matrixAutoUpdate', false); put(result, 'matrixWorldAutoUpdate', false);
        put(result, 'matrixWorldNeedsUpdate', false);
        if (value.shadow) put(result, 'shadow', value.shadow); // castShadow was excluded above.
      }
      return result;
    }
    let data;
    try {
      data = copy(root);
      if (root.isScene && root.background?.isColor) data.background = null;
      for (const [owner, history] of pending) {
        if (!histories.has(owner)) historyBytes += history.array.byteLength;
        histories.set(owner, history);
        if (history.clear) { for (const range of history.ranges) range.source.count = range.count; owner.clearUpdateRanges(); }
        delete history.ranges;
      }
      for (const [resource, owners] of sourceResources) {
        const selected = owners.filter(owner => pending.has(owner)); if (selected.length) observe(resource, selected);
      }
    } catch (error) {
      for (const [owner, history] of pending) if (!histories.has(owner)) history.release?.();
      throw error;
    }
    if (!state) { state = {root: view(root), views: new WeakMap(), record: null}; states.set(root, state); retained.add(state); }
    state.views.set(root, new WeakRef(state.root));
    const record = {state, data, sources, bytes: charged, pending: 0, installed: false}; records.add(record); bytes += charged;
    function install() {
      live(); if (!record.pending) fail('RELEASED', 'PMREM scene capture was released');
      if (state.record === record) return;
      if (state.record) { state.record.installed = false; collect(state.record); }
      const installed = new Map();
      function apply(value) {
        const source = record.sources.get(value);
        if (!source) return value;
        if (installed.has(source)) return installed.get(source);
        let target = state.views.get(source)?.deref();
        if (!target) { target = Array.isArray(value) ? [] : view(value); state.views.set(source, new WeakRef(target)); }
        installed.set(source, target);
        for (const key of Object.getOwnPropertyNames(target)) if (key !== '_listeners' && !(Array.isArray(target) && key === 'length') && !Object.hasOwn(value, key)) delete target[key];
        for (const key of Object.getOwnPropertyNames(value)) if (!(Array.isArray(value) && key === 'length')) put(target, key, apply(value[key]));
        if (Array.isArray(value)) target.length = value.length;
        return target;
      }
      apply(record.data); record.installed = true; state.record = record;
    }
    function retain(camera) {
      live(); record.pending++;
      if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
      let released = false;
      return Object.freeze({root: state.root, camera: captureThreeCamera(T, camera), bytes: charged, install, retain,
        release() { if (!released) { released = true; record.pending--; collect(record); } }});
    }
    return retain(camera);
  }
  return {capture, bindDevice, has: root => states.has(root), get bytes() { return bytes + historyBytes; }, get roots() { return retained.size; },
    retire(root) {
      const state = states.get(root); if (!state) return;
      for (const record of records) if (record.state === state && record.pending) fail('PENDING', 'PMREM scene still has queued captures');
      for (const record of [...records]) if (record.state === state) { record.installed = false; collect(record); }
      for (const key of Object.getOwnPropertyNames(state.root)) delete state.root[key];
      state.views = new WeakMap(); state.record = null; retained.delete(state); states.delete(root);
    },
    dispose() {
      disposed = true;
      for (const record of [...records]) { record.pending = record.installed = false; collect(record); }
      for (const [resource, record] of resources) resource.removeEventListener('dispose', record.listener);
      resources.clear(); for (const owner of histories.keys()) releaseHistory(owner);
      for (const state of retained) { for (const key of Object.getOwnPropertyNames(state.root)) delete state.root[key]; state.record = null; state.views = new WeakMap(); }
      retained.clear();
    }};
}
