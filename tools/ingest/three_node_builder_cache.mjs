/** Pinned r186 RenderObject material/geometry cache identity. No GPU ownership.
 * NodeBuilderState sharing depends on these source keys, rather than texture
 * identity or equal WGSL. Keep getter order and the source numeric reductions.
 */
const prototypeKeys = new WeakMap();

export function threeNodeBuilderMaterialKey(three, renderer, object, material, contextId, clippingKey = '') {
  // RenderObject invokes the hook before snapshotting enumerable properties or
  // consulting the constructor's getter cache. A hook may change either.
  let key = material.customProgramCacheKey();
  const properties = Object.keys(material);
  let getters = prototypeKeys.get(material.constructor);
  if (!getters) {
    getters = [];
    for (let prototype = Object.getPrototypeOf(material); prototype; prototype = Object.getPrototypeOf(prototype))
      for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(prototype)))
        if (typeof descriptor.get === 'function') getters.push(key);
    prototypeKeys.set(material.constructor, getters);
  }
  for (const property of [...properties, ...getters]) {
    if (/^(is[A-Z]|_)|^(visible|version|uuid|name|opacity|userData)$/.test(property)) continue;
    const value = material[property];
    let part;
    if (value !== null && typeof value === 'number') part = property === 'side' ? String(value) : value !== 0 ? '1' : '0';
    else if (value !== null && typeof value === 'object') {
      part = '{';
      if (value.isTexture) {
        part += value.mapping;
        if (renderer.backend.isWebGPUBackend === true) {
          part += value.magFilter; part += value.minFilter;
          part += value.wrapS; part += value.wrapT; part += value.wrapR;
        }
      }
      part += '}';
    } else part = String(value);
    key += part + ',';
  }
  key += clippingKey + ',';
  const geometry = object.geometry;
  if (geometry) {
    for (const name of Object.keys(geometry.attributes).sort()) {
      const attribute = geometry.attributes[name];
      key += name + ',';
      if (attribute.data) key += attribute.data.stride + ',';
      if (attribute.offset) key += attribute.offset + ',';
      if (attribute.itemSize) key += attribute.itemSize + ',';
      if (attribute.normalized) key += 'n,';
    }
    for (const name of Object.keys(geometry.morphAttributes).sort()) {
      const targets = geometry.morphAttributes[name];
      key += 'morph-' + name + ',';
      for (let index = 0, length = targets.length; index < length; index++) key += targets[index].id + ',';
    }
    if (geometry.index) key += 'index,';
  }
  if (object.skeleton) key += object.skeleton.bones.length + ',';
  if (object.isBatchedMesh) {
    key += object._matricesTexture.uuid + ',';
    if (object._colorsTexture !== null) key += object._colorsTexture.uuid + ',';
  }
  if (object.isInstancedMesh || object.count > 1) key += object.uuid + ',';
  key += contextId + ',' + object.receiveShadow + ',';
  return three.NodeUtils.hashString(key);
}

/** Source NodeBuilderState.createBindings: wrappers are per render object;
 * underlying nodes (including captured PMREM inputs) remain shared. */
export function cloneThreeNodeBuilderBindings(builder) {
  return builder.getBindings().map(group => group.bindings[0].groupNode.shared === true ? group :
    new group.constructor(group.name, group.bindings.map(binding => binding.clone())));
}
