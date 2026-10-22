/** Native r186 TSL graphics preparation. The source WGSLNodeBuilder and NodeFrame
 * are retained helpers; this module constructs no renderer/backend and submits
 * no GPU commands. Descriptors preserve the builder's native bind groups.
 *
 * The first profile is rigid, unlit meshes with float attributes and ordinary
 * scalar/vector/matrix node uniforms. Texture, storage, lighting, clipping and
 * MRT graphs fail explicitly. capture() is a logical-draw operation, never a
 * preparation retry: it runs source before/current updates and owns the bytes.
 * after() belongs after that draw has been accepted. Callers capture before any
 * asynchronous pipeline preparation and retain the resulting byte arrays.
 */
export class ThreeNodeProgramError extends Error {
  constructor(code, message, options) {
    super(`THREE_NODE_PROGRAM_${code}: ${message}`, options);
    this.name = 'ThreeNodeProgramError'; this.code = `THREE_NODE_PROGRAM_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeNodeProgramError(code, message); };
const SLOTS = new Set(['colorNode', 'positionNode', 'vertexNode', 'fragmentNode', 'opacityNode', 'alphaTestNode', 'outputNode']);
const TYPES = new Set(['float', 'int', 'uint', 'bool', 'color', 'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4',
  'uvec2', 'uvec3', 'uvec4', 'bvec2', 'bvec3', 'bvec4', 'mat2', 'mat3', 'mat4']);
const FRAME_FIELDS = ['renderer', 'scene', 'camera', 'object', 'material', 'compute', 'frameId', 'renderId', 'time', 'deltaTime'];
const nodeStates = new WeakMap();

/** True only for materials with an authored graphics slot. Default built-ins
 * continue through the existing fixed shader path. Admission is separate. */
export function hasThreeNodeProgram(material) {
  return material?.isNodeMaterial === true && Object.keys(material).some(key => key.endsWith('Node') && material[key] != null);
}

export function createThreeNodePrograms({three, renderer, maxPrograms = 4096, maxNodes = 8192,
  maxShaderBytes = 16 * 1024 * 1024, maxUniformBytes = 256 * 1024, maxBindGroups = 4} = {}) {
  if (three?.REVISION !== '186' || typeof three.WGSLNodeBuilder !== 'function' ||
      typeof renderer?._nodes?.nodeFrame?.updateNode !== 'function')
    fail('SOURCE', 'Supply the pinned r186 namespace and the renderer NodeFrame');
  for (const [name, value] of Object.entries({maxPrograms, maxNodes, maxShaderBytes, maxUniformBytes, maxBindGroups}))
    if (!Number.isSafeInteger(value) || value < 1) fail('LIMIT', `Invalid ${name}`);
  const frame = renderer._nodes.nodeFrame, cache = new WeakMap(), entries = new Set();
  let disposed = false, shaderBytes = 0, serial = 0;
  const live = () => { if (disposed) fail('DISPOSED', 'Native node program preparation is disposed'); };
  function inspect(material, object, {scene, camera} = {}) {
    live();
    if (!(material instanceof three.NodeMaterial) ||
        ![three.MeshBasicNodeMaterial.prototype, three.NodeMaterial.prototype].includes(Object.getPrototypeOf(material)))
      fail('MATERIAL', 'Native node graphs currently require MeshBasicNodeMaterial or unlit NodeMaterial');
    if (!material.isMeshBasicNodeMaterial && material.lights !== false)
      fail('LIGHTS', 'Base NodeMaterial graphs must be unlit');
    if (!(object instanceof three.Mesh) || object.isSkinnedMesh || object.isInstancedMesh || object.isBatchedMesh ||
        !(object.geometry instanceof three.BufferGeometry) || object.geometry.isInstancedBufferGeometry ||
        Object.values(object.geometry.morphAttributes ?? {}).some(list => list.length))
      fail('OBJECT', 'Native node graphs currently require rigid ordinary meshes');
    if (camera && (!(camera instanceof three.Camera) || camera.isArrayCamera || camera.isMultiViewCamera || camera.reversedDepth))
      fail('CAMERA', 'Native node graphs require an ordinary non-reversed camera');
    if (object.castShadow || object.receiveShadow) fail('SHADOW', 'Native node graph shadows are not admitted');
    if ((scene?.fog && material.fog !== false) || material.clippingPlanes?.length || renderer.clippingPlanes?.length)
      fail('EFFECT', 'Native node graph fog and clipping require their own shader context');
    if (material.wireframe || material.alphaHash || material.alphaToCoverage)
      fail('RASTER', 'Native node graph wireframe, hashed alpha and alpha-to-coverage are not admitted');
    if (material.transparent && material.side === three.DoubleSide && !material.forceSinglePass)
      fail('RASTER', 'Transparent DoubleSide node graphs require forceSinglePass until side variants are supported');
    if (Object.keys(renderer.contextNode?.getFlowContextData?.() ?? {}).length)
      fail('GRAPH', 'Custom renderer node contexts require an explicit graphics context adapter');
    if (material.onBeforeCompile !== three.Material.prototype.onBeforeCompile ||
        material.onBeforeRender !== three.Material.prototype.onBeforeRender ||
        material.customProgramCacheKey !== three.NodeMaterial.prototype.customProgramCacheKey)
      fail('HOOK', 'Custom material shader/render hooks are not admitted');
    for (const key of Object.keys(material)) {
      const value = material[key];
      if (value?.isTexture) fail('TEXTURE', `Native node graph material texture ${key} is not admitted`);
      if (key.endsWith('Node') && value != null && !SLOTS.has(key))
        fail('GRAPH', `Native node graph slot ${key} is not admitted`);
    }
    if (renderer.getMRT?.() != null) fail('MRT', 'Native node graphs currently write one color attachment');
    const seen = new Set(), stack = [...SLOTS].map(key => material[key]).filter(Boolean);
    while (stack.length) {
      const node = stack.pop();
      if (seen.has(node)) continue;
      seen.add(node);
      if (seen.size > maxNodes) fail('LIMIT', 'Native node graph exceeds its node budget');
      checkNode(node);
      if (typeof node.getChildren === 'function') for (const child of node.getChildren()) stack.push(child);
    }
    return material;
  }
  function checkNode(node) {
    if (!node?.isNode) fail('GRAPH', 'Expected source TSL nodes');
    if (node.isTextureNode || node.isStorageTextureNode || node.value?.isTexture)
      fail('TEXTURE', 'Texture graphs require native texture bindings and are not admitted');
    if (node.isStorageBufferNode || node.isBufferNode || node.isBufferAttributeNode || node.isComputeNode || node.isAtomicFunctionNode)
      fail('STORAGE', 'Storage, explicit buffer and compute graphs are not admitted');
    if (node.isLightingNode || node.isLightsNode || node.isLightingContextNode || node.light?.isLight)
      fail('LIGHTS', 'Lighting graphs require a native lighting context');
    if (node.isOutputStructNode || node.isMRTNode) fail('MRT', 'Native node graphs currently write one color attachment');
  }
  function keyOf(material, object, {camera} = {}) {
    // Match the source's graph cache key. Uniform values never participate.
    // Object/material references remain separate cache dimensions, preserving
    // builder nodes which close over a particular source object.
    return JSON.stringify([material.version, material.customProgramCacheKey(), material.side,
      material.vertexColors, material.flatShading, material.transparent, material.blending, material.premultipliedAlpha,
      material.alphaTest > 0, material.depthWrite, material.depthTest, material.lights,
      object.geometry.id, camera?.isPerspectiveCamera === true,
      Object.entries(object.geometry.attributes).map(([name, a]) => [name, a.itemSize, a.normalized, a.gpuType, a.array?.constructor.name,
        a.isInterleavedBufferAttribute, a.data?.array?.constructor.name, a.data?.stride, a.offset]),
      renderer.logarithmicDepthBuffer, renderer.reversedDepthBuffer]);
  }
  function retire(entry) {
    if (!entries.delete(entry)) return;
    shaderBytes -= entry.bytes;
    entry.material.removeEventListener('dispose', entry.onDispose);
    entry.geometry.removeEventListener('dispose', entry.onDispose);
    if (cache.get(entry.object)?.get(entry.material) === entry) cache.get(entry.object).delete(entry.material);
  }
  function compile(material, object, context = {}) {
    inspect(material, object, context);
    const key = keyOf(material, object, context), byMaterial = cache.get(object);
    const previous = byMaterial?.get(material);
    if (previous?.key === key) return previous.descriptor;
    if (entries.size - (previous ? 1 : 0) >= maxPrograms) fail('LIMIT', 'Native node program capacity exceeded');
    const builder = new three.WGSLNodeBuilder(object, renderer);
    builder.material = material; builder.scene = context.scene ?? null; builder.camera = context.camera ?? null;
    try { builder.build(); }
    catch (cause) {
      if (cause instanceof ThreeNodeProgramError) throw cause;
      throw new ThreeNodeProgramError('BUILD', `Pinned node construction failed: ${cause?.message ?? cause}`, {cause});
    }
    if (builder.nodes.size > maxNodes) fail('LIMIT', 'Generated native node graph exceeds its node budget');
    for (const node of builder.nodes) checkNode(node);
    const bindings = builder.getBindings();
    if (bindings.length > Math.min(maxBindGroups, renderer._device?.limits?.maxBindGroups ?? 4))
      fail('LIMIT', 'Native node graph exceeds its bind group budget');
    let uniformBytes = 0;
    const groups = bindings.map((group, index) => {
      if (group.bindings.length !== 1 || group.bindings[0].isNodeUniformsGroup !== true)
        fail('BINDING', 'Native graphics groups currently require one scalar/vector/matrix uniform buffer each');
      const binding = group.bindings[0], byteLength = binding.byteLength;
      if (binding.uniforms.some(u => !TYPES.has(u.nodeUniform?.type)))
        fail('UNIFORM', 'Native node uniforms must be ordinary scalars, vectors or matrices');
      if (!Number.isSafeInteger(byteLength) || byteLength < 4 || byteLength % 4 ||
          byteLength > (renderer._device?.limits?.maxUniformBufferBindingSize ?? 65536))
        fail('LIMIT', 'Native node uniform buffer exceeds device binding limits');
      uniformBytes += byteLength;
      return Object.freeze({index, name: group.name, byteLength});
    });
    if (uniformBytes > maxUniformBytes) fail('LIMIT', 'Native node draw exceeds its uniform byte budget');
    const attributes = builder.attributes.map((attribute, location) => {
      const components = {float: 1, vec2: 2, vec3: 3, vec4: 4}[attribute.type];
      const source = object.geometry.attributes[attribute.name];
      if (!components || !source || source.itemSize !== components || source.gpuType === three.IntType ||
          source.isInstancedBufferAttribute || source.data?.isInstancedInterleavedBuffer)
        fail('ATTRIBUTE', `Native node attribute ${attribute.name} requires a matching scalar/vector float geometry attribute`);
      return Object.freeze({name: attribute.name, location, locations: 1, components, scalar: 'float'});
    });
    const vertex = builder.vertexShader, fragment = builder.fragmentShader, bytes = 2 * (vertex.length + fragment.length);
    if (bytes > maxShaderBytes - shaderBytes + (previous?.bytes ?? 0)) fail('LIMIT', 'Native node shader byte budget exceeded');
    const descriptor = Object.freeze({nodeProgram: true, key: `native-node-${++serial}`, vertex, fragment,
      attributes: Object.freeze(attributes), attributesKey: JSON.stringify(attributes), groups: Object.freeze(groups)});
    nodeStates.set(descriptor, {owner: api, builder, bindings, material, object});
    const entry = {key, descriptor, material, object, geometry: object.geometry, bytes};
    entry.onDispose = () => retire(entry);
    if (previous) retire(previous);
    const next = byMaterial ?? new Map(); next.set(material, entry); cache.set(object, next);
    entries.add(entry); shaderBytes += bytes;
    material.addEventListener('dispose', entry.onDispose); object.geometry.addEventListener('dispose', entry.onDispose);
    return descriptor;
  }
  function stateOf(descriptor) {
    live(); const state = nodeStates.get(descriptor);
    if (state?.owner !== api) fail('SOURCE', 'Native node descriptor belongs to another preparation owner');
    return state;
  }
  function withFrame(state, context, run) {
    const previous = Object.fromEntries(FRAME_FIELDS.map(key => [key, frame[key]]));
    Object.assign(frame, {renderer, object: context.object ?? state.object, material: context.material ?? state.material,
      camera: context.camera ?? null, scene: context.scene ?? null, compute: null,
      renderId: context.renderId ?? renderer.info.calls});
    for (const key of ['frameId', 'time', 'deltaTime']) if (context[key] !== undefined) frame[key] = context[key];
    try { return run(); } finally { Object.assign(frame, previous); }
  }
  function capture(descriptor, context = {}) {
    const state = stateOf(descriptor);
    return withFrame(state, context, () => {
      for (const node of state.builder.updateBeforeNodes) frame.updateBeforeNode(node);
      for (const node of state.builder.updateNodes) frame.updateNode(node);
      return state.bindings.map((group, index) => {
        const binding = group.bindings[0]; binding.update();
        const buffer = binding.buffer;
        if (buffer.byteLength !== descriptor.groups[index].byteLength) fail('CHANGED', 'Native node uniform layout changed during its update');
        const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength).slice();
        binding.clearUpdateRanges(); return bytes;
      });
    });
  }
  function after(descriptor, context = {}) {
    const state = stateOf(descriptor);
    return withFrame(state, context, () => { for (const node of state.builder.updateAfterNodes) frame.updateAfterNode(node); });
  }
  const api = Object.freeze({inspect, compile, capture, after,
    release(descriptor) { for (const entry of entries) if (entry.descriptor === descriptor) { retire(entry); break; } },
    dispose() { if (disposed) return; disposed = true; for (const entry of [...entries]) retire(entry); },
    get diagnostics() { return Object.freeze({programs: entries.size, shaderBytes, maxPrograms, maxShaderBytes}); }});
  return api;
}
