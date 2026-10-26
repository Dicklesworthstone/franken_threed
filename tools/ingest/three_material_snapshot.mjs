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

export function captureThreeMaterialState(three, root, {overrideMaterial = null, nodeFrame = null,
  maxNodes = 16384, maxMaterials = 16384, maxValues = 1048576} = {}) {
  if (![maxNodes, maxMaterials, maxValues].every(n => Number.isSafeInteger(n) && n >= 0) || !maxNodes || !maxMaterials)
    fail('LIMIT', 'Invalid material snapshot capacity');
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
