/** Per-request values for ordinary fixed-path materials.
 * Source identities remain cache/disposal keys; only callback-free value data is
 * copied. Geometry, texture objects and their upload histories remain live.
 */
export class ThreeMaterialSnapshotError extends Error {
  constructor(code, message) {
    super(`THREE_MATERIAL_SNAPSHOT_${code}: ${message}`);
    this.name = 'ThreeMaterialSnapshotError'; this.code = `THREE_MATERIAL_SNAPSHOT_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeMaterialSnapshotError(code, message); };

// Inspect eligibility without advancing an application's accessor or callback.
const accessor = Symbol('accessor');
const data = (object, key) => {
  for (let current = object; current; current = Object.getPrototypeOf(current)) {
    const descriptor = Object.getOwnPropertyDescriptor(current, key);
    if (descriptor) return Object.hasOwn(descriptor, 'value') ? descriptor.value : accessor;
  }
};
const dataOnly = object => Object.values(Object.getOwnPropertyDescriptors(object))
  .every(descriptor => Object.hasOwn(descriptor, 'value'));
const mapPrototype = Map.prototype, mapForEach = mapPrototype.forEach;
const mapSize = Object.getOwnPropertyDescriptor(mapPrototype, 'size').get;
const selectorArray = (array, limit) => {
  if (!Array.isArray(array) || Object.getPrototypeOf(array) !== Array.prototype) return null;
  const descriptors = Object.getOwnPropertyDescriptors(array);
  if (descriptors.length.value > limit) fail('LIMIT', 'Material selection array capacity exceeded');
  if (Reflect.ownKeys(descriptors).some(key => typeof key !== 'string' || key !== 'length' &&
      (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= 4294967295)) ||
      Object.values(descriptors).some(descriptor => !Object.hasOwn(descriptor, 'value'))) return null;
  return Object.defineProperties([], descriptors);
};

/** WebGL projects a material selector before drawing it. Retain that selector
 * across asynchronous preparation, while the selected material and its shader
 * uniforms stay live. Hooks that could change selection during projection keep
 * their existing live-source path instead of being moved ahead of source work.
 */
function captureSelections(three, root, {camera, overrideMaterial, exclusions, allowInstances, maxNodes, maxMaterials, maxValues}) {
  const materials = new Map(), objects = new Map(), seen = new Set(), instances = new Set(), stack = [root];
  let values = 0;
  const claim = count => {
    if (count > maxValues - values) fail('LIMIT', 'Pending material snapshot value capacity exceeded');
    values += count;
  };
  // Only the pinned pass's temporary hidden list/cache is captured. Ordinary scene
  // visibility and LOD updates retain their existing submission behavior.
  let excluded = null;
  if (exclusions !== null && Object.getPrototypeOf(exclusions) === mapPrototype) {
    // OutlinePass owns an ordinary Map. Do not call application-overridable
    // keys()/iterators or Array.from() while retaining its temporary selection.
    if (Reflect.ownKeys(exclusions).length) return null;
    let size;
    try { size = mapSize.call(exclusions); } catch { return null; }
    if (size > maxNodes) fail('LIMIT', 'Pass exclusion cache capacity exceeded');
    excluded = []; mapForEach.call(exclusions, (_visible, object) => excluded.push(object));
  } else if (exclusions !== null) excluded = selectorArray(exclusions, maxNodes);
  if (exclusions !== null && (!excluded || excluded.some(object => !(object instanceof three.Object3D)))) return null;
  const excludedObjects = excluded ? new Set(excluded) : null;
  if (excludedObjects) claim(excludedObjects.size);
  const ordinaryInstances = object => {
    if (!allowInstances || !(object instanceof three.InstancedMesh) || data(object, 'morphTexture') !== null ||
        data(object, 'computeBoundingSphere') !== three.InstancedMesh.prototype.computeBoundingSphere ||
        data(object, 'getMatrixAt') !== three.InstancedMesh.prototype.getMatrixAt) return false;
    const sphere = data(object, 'boundingSphere');
    if (sphere !== null && (!(sphere instanceof three.Sphere) || !dataOnly(sphere) ||
        !(data(sphere, 'center') instanceof three.Vector3) || !dataOnly(data(sphere, 'center')))) return false;
    const count = data(object, 'count');
    if (!Number.isSafeInteger(count) || count < 0) return false;
    for (const [key, width] of [['instanceMatrix', 16], ['instanceColor', 3]]) {
      const attribute = data(object, key);
      if (key === 'instanceColor' && attribute === null) continue;
      if (!(attribute instanceof three.InstancedBufferAttribute) || !dataOnly(attribute) ||
          data(attribute, 'itemSize') !== width || data(attribute, 'normalized') !== false || data(attribute, 'meshPerAttribute') !== 1 ||
          data(attribute, 'isStorageInstancedBufferAttribute') ||
          data(attribute, 'onUploadCallback') !== three.BufferAttribute.prototype.onUploadCallback ||
          data(attribute, 'clearUpdateRanges') !== three.BufferAttribute.prototype.clearUpdateRanges) return false;
      const array = data(attribute, 'array'), capacity = data(attribute, 'count'), version = data(attribute, 'version');
      if (!(array instanceof Float32Array) || Object.getPrototypeOf(array) !== Float32Array.prototype ||
          ['buffer', 'byteOffset', 'byteLength', 'length', 'constructor', 'slice', 'subarray'].some(key => Object.hasOwn(array, key)) ||
          !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable ||
          !Number.isSafeInteger(capacity) || capacity < count || capacity * width !== array.length ||
          !Number.isSafeInteger(version) || version < 0) return false;
      const ranges = selectorArray(data(attribute, 'updateRanges'), maxValues);
      if (!ranges || ranges.some(range => !range || Object.getPrototypeOf(range) !== Object.prototype || !dataOnly(range) ||
          !Number.isSafeInteger(data(range, 'start')) || !Number.isSafeInteger(data(range, 'count')) ||
          data(range, 'start') < 0 || data(range, 'count') < 0 || data(range, 'start') + data(range, 'count') > array.length)) return false;
    }
    const morphs = data(data(object, 'geometry'), 'morphAttributes');
    return morphs && dataOnly(morphs) && Object.values(morphs).every(value => Array.isArray(value) && value.length === 0);
  };
  const ordinaryObject = object => {
    if (!(object instanceof three.Object3D) || !dataOnly(object)) return false;
    const prototype = object instanceof three.Camera ? three.Camera.prototype : three.Object3D.prototype;
    if (data(object, 'updateMatrixWorld') !== prototype.updateMatrixWorld ||
        data(object, 'updateMatrix') !== three.Object3D.prototype.updateMatrix ||
        data(object, 'onBeforeRender') !== three.Object3D.prototype.onBeforeRender ||
        data(object, 'onAfterRender') !== three.Object3D.prototype.onAfterRender ||
        data(object, 'isSkinnedMesh') || data(object, 'isBatchedMesh') ||
        data(object, 'isInstancedMesh') && !ordinaryInstances(object)) return false;
    if (data(object, 'isLOD') && data(object, 'update') !== three.LOD?.prototype.update) return false;
    const type = [three.Mesh, three.Line, three.Points, three.Sprite].find(Type => Type && object instanceof Type);
    if (type && data(object, 'intersectsFrustum') !== type.prototype.intersectsFrustum) return false;
    const layers = data(object, 'layers');
    if (!(layers instanceof three.Layers) || !dataOnly(layers) || data(layers, 'test') !== three.Layers.prototype.test ||
        data(layers, 'mask') === accessor ||
        !data(object, 'isInstancedMesh') && (data(object, 'boundingSphere') !== undefined || data(object, 'computeBoundingSphere') !== undefined)) return false;
    return true;
  };
  const ordinaryGeometry = geometry => {
    if (geometry === undefined) return true;
    if (!(geometry instanceof three.BufferGeometry) || !dataOnly(geometry) ||
        data(geometry, 'computeBoundingSphere') !== three.BufferGeometry.prototype.computeBoundingSphere) return false;
    const attributes = data(geometry, 'attributes'), morphs = data(geometry, 'morphAttributes');
    if (!attributes || !morphs || !dataOnly(attributes) || !dataOnly(morphs)) return false;
    const groups = selectorArray(data(geometry, 'groups'), maxValues);
    if (!groups || groups.some(group => !group || typeof group !== 'object' ||
        ![Object.prototype, null].includes(Object.getPrototypeOf(group)) || !dataOnly(group))) return false;
    const sources = [...Object.values(attributes), data(geometry, 'index')];
    for (const group of Object.values(morphs)) {
      const array = selectorArray(group, maxValues);
      if (!array) return false;
      for (const attribute of array) sources.push(attribute);
    }
    for (const attribute of sources) {
      if (attribute == null) continue;
      const interleaved = data(attribute, 'isInterleavedBufferAttribute');
      if (interleaved === accessor) return false;
      const owner = interleaved ? data(attribute, 'data') : attribute;
      const prototype = interleaved ? three.InterleavedBuffer?.prototype : three.BufferAttribute?.prototype;
      if (!owner || owner === accessor || !prototype || data(owner, 'onUploadCallback') !== prototype.onUploadCallback) return false;
    }
    return true;
  };
  const ordinaryMaterial = material => material instanceof three.Material && dataOnly(material) &&
    data(material, 'onBeforeRender') === three.Material.prototype.onBeforeRender &&
    data(material, 'onBeforeCompile') === three.Material.prototype.onBeforeCompile &&
    data(material, 'customProgramCacheKey') === three.Material.prototype.customProgramCacheKey;
  const captureMaterial = material => {
    if (material == null || materials.has(material)) return true;
    if (!ordinaryMaterial(material)) return false;
    if (materials.size >= maxMaterials) fail('LIMIT', 'Material snapshot identity capacity exceeded');
    claim(1); materials.set(material, material);
    return true;
  };
  // ArrayCamera callbacks and custom camera children retain their established
  // per-view projection order. Callback-free ordinary cameras may be captured
  // independently of these live material objects.
  if (camera && (!ordinaryObject(camera) || data(camera, 'isArrayCamera') ||
      data(camera, 'children')?.length !== 0)) return null;
  if (!captureMaterial(overrideMaterial)) return null;
  while (stack.length) {
    const object = stack.pop();
    if (!ordinaryObject(object)) return null;
    if (seen.has(object)) fail('GRAPH', 'Expected an acyclic source hierarchy');
    seen.add(object);
    if (data(object, 'isInstancedMesh')) instances.add(object);
    const children = selectorArray(data(object, 'children'), maxNodes);
    if (!children) return null;
    if (seen.size + stack.length + children.length > maxNodes) fail('LIMIT', 'Material snapshot node capacity exceeded');
    if (excludedObjects?.has(object)) for (const child of children) if (!excludedObjects.has(child)) {
      claim(1); excludedObjects.add(child);
    }
    for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
    if (!ordinaryGeometry(data(object, 'geometry'))) return null;
    const material = data(object, 'material');
    if (material === accessor) return null;
    if (material === undefined) continue;
    const selected = Array.isArray(material) ? selectorArray(material, maxMaterials) : material;
    if (Array.isArray(material) && !selected) return null;
    claim(1 + (Array.isArray(selected) ? selected.length : 1));
    objects.set(object, selected);
    for (const source of Array.isArray(selected) ? selected : [selected]) if (!captureMaterial(source)) return null;
  }
  let selection;
  const cameraSafe = () => {
    const perspective = data(camera, 'isPerspectiveCamera') === true;
    if (!camera || !ordinaryObject(camera) || data(camera, 'isArrayCamera') || data(camera, 'isMultiViewCamera') ||
        !perspective && data(camera, 'isOrthographicCamera') !== true || data(camera, 'children')?.length !== 0) return false;
    const prototype = perspective ? three.PerspectiveCamera.prototype : three.OrthographicCamera.prototype;
    if (data(camera, 'updateProjectionMatrix') !== prototype.updateProjectionMatrix ||
        perspective && data(camera, 'getFilmWidth') !== prototype.getFilmWidth) return false;
    for (const object of seen) {
      // A queued rigid pass owns these selectors already. New children and
      // geometry replacements belong to the next call, while callbacks on the
      // original source objects and captured resources remain guarded.
      const captured = selection?.instanceFrame;
      const geometry = captured?.geometries.has(object) ? captured.geometries.get(object) : data(object, 'geometry');
      if (!ordinaryObject(object) || !ordinaryGeometry(geometry)) return false;
      const children = captured?.hierarchy.get(object) ?? selectorArray(data(object, 'children'), maxNodes);
      if (!children || children.some(child => !seen.has(child))) return false;
    }
    for (const material of materials.keys()) if (!ordinaryMaterial(material)) return false;
    return true;
  };
  // Automatic LOD projection still follows the original live scene path. The
  // required pass scope rejects it before destination retention; ordinary
  // rendering must not newly fail because optional instance capture is enabled.
  if (instances.size && [...seen].some(object => data(object, 'isLOD') && data(object, 'autoUpdate'))) return null;
  selection = objects.size ? {materials, objects, values, cameraSafe, excludedObjects, instances} : null;
  return selection;
}

export function captureThreeMaterialState(three, root, {overrideMaterial = null, nodeFrame = null,
  selectionOnly = false, camera = null, exclusions = null, allowInstances = false, maxNodes = 16384, maxMaterials = 16384, maxValues = 1048576} = {}) {
  if (![maxNodes, maxMaterials, maxValues].every(n => Number.isSafeInteger(n) && n >= 0) || !maxNodes || !maxMaterials)
    fail('LIMIT', 'Invalid material snapshot capacity');
  if (selectionOnly) return captureSelections(three, root, {camera, overrideMaterial, exclusions, allowInstances, maxNodes, maxMaterials, maxValues});
  const materials = new Map(), objects = new Map(), seen = new Set(), copies = new WeakMap(), stack = [root];
  const valuePrototypes = new Set(['Color', 'Vector2', 'Vector3', 'Vector4', 'Quaternion', 'Euler', 'Matrix3', 'Matrix4', 'Plane']
    .map(name => three[name]?.prototype).filter(Boolean));
  let values = 0;
  const claim = count => {
    if (count > maxValues - values) fail('LIMIT', 'Pending material snapshot value capacity exceeded');
    values += count;
  };
  const copyValue = value => {
    if (value === null || typeof value !== 'object') return value;
    const prototype = Object.getPrototypeOf(value), array = Array.isArray(value);
    // Textures, nodes, resources, userData and custom value classes keep identity.
    if (array ? prototype !== Array.prototype : !valuePrototypes.has(prototype)) return value;
    if (copies.has(value)) return copies.get(value);
    const descriptors = Object.getOwnPropertyDescriptors(value);
    // Do not invoke user accessors or change their receiver. Such custom values
    // retain the existing live-value contract; ordinary numeric values copy.
    if (Object.values(descriptors).some(d => !Object.hasOwn(d, 'value'))) return value;
    claim(Object.keys(descriptors).length);
    const result = array ? [] : Object.create(prototype);
    copies.set(value, result);
    for (const descriptor of Object.values(descriptors)) descriptor.value = copyValue(descriptor.value);
    Object.defineProperties(result, descriptors);
    return result;
  };
  const captureMaterial = material => {
    if (material === null || typeof material !== 'object' || materials.has(material)) return;
    if (materials.size >= maxMaterials) fail('LIMIT', 'Material snapshot identity capacity exceeded');
    const descriptors = Object.getOwnPropertyDescriptors(material);
    claim(Object.keys(descriptors).length);
    const state = Object.create(Object.getPrototypeOf(material));
    materials.set(material, state);
    for (const [key, descriptor] of Object.entries(descriptors))
      if (key !== 'userData' && Object.hasOwn(descriptor, 'value')) descriptor.value = copyValue(descriptor.value);
    // Accessor descriptors remain descriptors. The fixed mapper rejects custom
    // material accessors before reading them, exactly as it does for the source.
    Object.defineProperties(state, descriptors);
  };
  while (stack.length) {
    const object = stack.pop();
    if (!(object instanceof three.Object3D) || seen.has(object) || !Array.isArray(object.children))
      fail('GRAPH', 'Expected an acyclic source hierarchy');
    seen.add(object);
    if (seen.size + stack.length + object.children.length > maxNodes) fail('LIMIT', 'Material snapshot node capacity exceeded');
    for (let i = object.children.length - 1; i >= 0; i--) stack.push(object.children[i]);
    // Native packets already own material selection and values. Never revisit
    // their excluded objects or manufacture another native update invocation.
    if (nodeFrame?.objects?.has(object) || nodeFrame?.excludedObjects?.has(object)) continue;
    const descriptor = Object.getOwnPropertyDescriptor(object, 'material');
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) continue;
    const material = descriptor.value;
    let selected = material;
    if (Array.isArray(material)) {
      if (material.length > maxMaterials) fail('LIMIT', 'Material snapshot array capacity exceeded');
      // Ordinary material arrays contain source references, not cloned values.
      const descriptors = Object.getOwnPropertyDescriptors(material);
      if (Object.getPrototypeOf(material) !== Array.prototype ||
          Reflect.ownKeys(descriptors).some(key => typeof key !== 'string' || key !== 'length' &&
            (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= 4294967295)) ||
          Object.values(descriptors).some(d => !Object.hasOwn(d, 'value'))) continue;
      selected = []; Object.defineProperties(selected, descriptors);
    }
    claim(1 + (Array.isArray(selected) ? selected.length : 1));
    objects.set(object, selected);
    for (const source of Array.isArray(selected) ? selected : [selected]) captureMaterial(source);
  }
  if (!objects.size) return null;
  captureMaterial(overrideMaterial);
  return {materials, objects, values};
}
