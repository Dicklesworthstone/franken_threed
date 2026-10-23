/** Native r186 TSL graphics preparation. The source WGSLNodeBuilder and NodeFrame
 * are retained helpers; this module constructs no renderer/backend and submits
 * no GPU commands. Descriptors preserve the builder's native bind groups.
 *
 * The profile is rigid meshes, unlit or Toon/Lambert/Phong/Standard/Physical
 * with ordinary source lights, float attributes, scalar/vector/matrix uniforms,
 * sampled 2D float textures and read-only scalar/vector or flat-struct storage buffers.
 * Physical direct lighting retains the source DFG texture, clearcoat, sheen,
 * iridescence, anisotropy, retroreflection and source PMREM environment lighting. Transmission,
 * writable/atomic storage, shadows, clipping and MRT graphs fail explicitly.
 * capture() is a logical-draw operation, never a
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
const LIT_SLOTS = new Set(['normalNode', 'emissiveNode']);
const PHONG_SLOTS = new Set(['shininessNode', 'specularNode']);
const STANDARD_SLOTS = new Set(['metalnessNode', 'roughnessNode']);
const PHYSICAL_SLOTS = new Set(['clearcoatNode', 'clearcoatRoughnessNode', 'clearcoatNormalNode',
  'sheenNode', 'sheenRoughnessNode', 'iridescenceNode', 'iridescenceIORNode', 'iridescenceThicknessNode',
  'anisotropyNode', 'retroreflectivityNode', 'iorNode', 'specularColorNode', 'specularIntensityNode']);
const GRAPH_SLOTS = [...SLOTS, ...LIT_SLOTS, ...PHONG_SLOTS, ...STANDARD_SLOTS, ...PHYSICAL_SLOTS];
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
  const standardPrototypes = [three.MeshStandardNodeMaterial.prototype, three.MeshPhysicalNodeMaterial.prototype];
  const litPrototypes = [three.MeshToonNodeMaterial.prototype, three.MeshLambertNodeMaterial.prototype,
    three.MeshPhongNodeMaterial.prototype, ...standardPrototypes];
  const lightClasses = new Map([[three.AmbientLight, three.AmbientLightNode], [three.DirectionalLight, three.DirectionalLightNode],
    [three.HemisphereLight, three.HemisphereLightNode], [three.PointLight, three.PointLightNode], [three.SpotLight, three.SpotLightNode]]);
  const environmentNodes = new WeakMap(), environmentVersions = new WeakMap();
  let disposed = false, shaderBytes = 0, serial = 0;
  const live = () => { if (disposed) fail('DISPOSED', 'Native node program preparation is disposed'); };
  const litMaterial = material => litPrototypes.includes(Object.getPrototypeOf(material));
  function inspectStorage(node) {
    const attribute = node.value, struct = node.structTypeNode;
    const type = /^(?:(i|u)?vec([234])|(float|int|uint))$/.exec(node.nodeType ?? '');
    const components = type?.[2] ? Number(type[2]) : 1;
    const scalar = type?.[1] === 'i' || type?.[3] === 'int' ? Int32Array
      : type?.[1] === 'u' || type?.[3] === 'uint' ? Uint32Array : Float32Array;
    let stride = components === 3 ? 4 : components;
    if (struct) {
      if (node.nodeType !== 'struct' || struct.isStructTypeNode !== true ||
          !Array.isArray(struct.membersLayout) || !struct.membersLayout.length || struct.membersLayout.length > maxNodes ||
          struct.membersLayout.some(member => !member || typeof member.name !== 'string' || member.atomic ||
            !/^(?:float|int|uint|[iu]?vec[234]|mat[234])$/.test(member.type)))
        fail('STORAGE', 'Native storage structs require non-atomic scalar/vector/matrix 32-bit members');
      // Use the pinned source layout, including vec3 member alignment and mat3
      // column padding. Mixed integer/float fields share the original bytes.
      stride = three.StructTypeNode.prototype.getLength.call(struct);
      if (!Number.isSafeInteger(stride) || stride < 1 || stride === 3)
        fail('STORAGE', 'Native storage structs require a complete source stride without ambiguous three-word padding');
    }
    if ((!type && !struct) || node.isAtomic || node.isPBO ||
        !['readOnly', 'readWrite'].includes(node.access) || attribute?.isIndirectStorageBufferAttribute ||
        !(attribute?.isStorageBufferAttribute || attribute?.isStorageInstancedBufferAttribute) ||
        attribute.isInterleavedBufferAttribute || attribute.normalized ||
        !(struct ? [Float32Array, Int32Array, Uint32Array].some(Type => attribute.array instanceof Type) : attribute.array instanceof scalar) ||
        !Number.isSafeInteger(attribute.count) || attribute.count < 1 ||
        !(struct ? attribute.itemSize === stride : [components, stride].includes(attribute.itemSize)) ||
        attribute.array.length !== attribute.count * attribute.itemSize ||
        !Number.isSafeInteger(node.bufferCount) || node.bufferCount < 0 || node.bufferCount > attribute.count)
      fail('STORAGE', 'Native storage reads require matching non-atomic scalar/vector or flat-struct 32-bit storage attributes');
    if (attribute.onUploadCallback !== three.BufferAttribute.prototype.onUploadCallback)
      fail('HOOK', 'Native storage reads require the source upload callback');
    const bytes = attribute.count * stride * 4;
    if (bytes > (renderer._device?.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024) ||
        bytes > (renderer._device?.limits?.maxBufferSize ?? 256 * 1024 * 1024))
      fail('LIMIT', 'Native storage input exceeds device buffer binding limits');
    return bytes;
  }
  function storageSignature(nodes) {
    const aliases = new Map();
    return JSON.stringify(nodes.map(node => {
      const bytes = inspectStorage(node), attribute = node.value;
      const unsized = node.bufferCount === 0;
      if (unsized && !aliases.has(attribute)) aliases.set(attribute, aliases.size);
      const struct = node.structTypeNode;
      // NodeStorageBuffer.attribute follows node.value on each draw. Compatible
      // replacements retain bindings; only unsized nodes share by attribute, so
      // their alias partition must remain stable. Fixed arrays bind separately.
      return [unsized ? aliases.get(attribute) : null, node.nodeType, node.bufferCount, attribute.count, bytes, node.groupNode?.name,
        struct ? [struct.name, struct.membersLayout.map(({name, type}) => [name, type])] : null];
    }));
  }
  function sourceLights(material, {lights = [], lightingEnabled = renderer.lighting?.enabled !== false} = {}) {
    if (!litMaterial(material) || material.lights !== true || material.fragmentNode != null) return [];
    if (typeof lightingEnabled !== 'boolean' || lightingEnabled !== (renderer.lighting?.enabled !== false))
      fail('LIGHTS', 'Captured lighting enablement must match the source renderer during construction');
    if (!lightingEnabled) return [];
    if (!Array.isArray(lights)) fail('LIGHTS', 'Supply the selected source scene lights');
    if (lights.length > 8) fail('LIMIT', 'Native lit node graphs currently admit at most eight selected scene lights');
    for (const light of lights) {
      const nodeClass = lightClasses.get(light?.constructor);
      if (!nodeClass || Object.getPrototypeOf(light) !== light.constructor.prototype || light.isLight !== true ||
          renderer.library.getLightNodeClass(light.constructor) !== nodeClass)
        fail('LIGHTS', 'Native lit graphs require ordinary Ambient, Directional, Hemisphere, Point or Spot lights');
      if (light.castShadow || light.map != null || light.colorNode != null)
        fail('LIGHTS', 'Native lit graph shadows, projected maps and custom light nodes are not admitted');
    }
    // The pinned LightsNode retains analytic nodes in its module WeakMap by the
    // real light identity. Never clone lights or instantiate per-program light
    // nodes: that would repeat FRAME updates across objects and recompiles.
    return lights.slice().sort((a, b) => a.id - b.id);
  }
  function sourceEnvironment(material, {scene} = {}) {
    if (!standardPrototypes.includes(Object.getPrototypeOf(material)) || material.lights !== true ||
        material.fragmentNode != null || renderer.lighting?.enabled === false) return null;
    if (scene?.environmentNode != null)
      fail('ENVIRONMENT', 'Custom scene environment nodes require an explicit lighting context');
    const texture = scene?.environment ?? null;
    if (texture === null) return null;
    if (!(texture instanceof three.Texture) || texture.isF3DSceneEnvironment || texture.isF3DPMREMSource ||
        ![three.EquirectangularReflectionMapping, three.EquirectangularRefractionMapping, three.CubeUVReflectionMapping].includes(texture.mapping))
      fail('ENVIRONMENT', 'Native PBR environments require a ready equirectangular image or source CubeUV texture');
    assertEnvironmentVersion(texture);
    inspectTexture(texture);
    return texture;
  }
  function assertEnvironmentVersion(texture) {
    if (texture && environmentVersions.has(texture) && environmentVersions.get(texture) !== texture.pmremVersion)
      fail('ENVIRONMENT_REFRESH', 'In-place source PMREM regeneration requires a nested preparation boundary; use a different environment texture identity for fresh source filtering');
  }
  function environmentNodeFor(texture) {
    if (!texture) return null;
    let node = environmentNodes.get(texture);
    if (!node) { node = three.TSL.texture(texture); environmentNodes.set(texture, node); }
    return node;
  }
  function inspectTexture(texture) {
    if (!(texture instanceof three.Texture) || texture.isStorageTexture || texture.isCubeTexture ||
        texture.is3DTexture || texture.isArrayTexture || texture.isData3DTexture || texture.isDataArrayTexture ||
        texture.isCompressedTexture || texture.isDepthTexture || texture.isVideoTexture || texture.isVideoFrameTexture ||
        texture.isFramebufferTexture || texture.isExternalTexture)
      fail('TEXTURE', 'Native graphics nodes currently sample ordinary 2D textures and color render targets');
    if (![three.UnsignedByteType, three.HalfFloatType, three.FloatType].includes(texture.type) ||
        [three.RedIntegerFormat, three.RGIntegerFormat, three.RGBAIntegerFormat, three.RGBIntegerFormat].includes(texture.format) ||
        texture.type === three.FloatType && !renderer.hasFeature('float32-filterable'))
      fail('TEXTURE', 'Native graphics texture formats must have filterable float samples');
    const layout = renderer._getSampledTextureInputs?.().layoutOf({texture, isSampledTexture: true, isSampler: true, store: false});
    if (layout && (layout.texture?.sampleType !== 'float' || layout.texture.viewDimension !== '2d' || layout.texture.multisampled))
      fail('TEXTURE', 'Native graphics texture bindings require a single-sampled float 2D view');
    if (typeof renderer.backend?.utils?.getTextureSampleData !== 'function')
      fail('SOURCE', 'The native renderer must provide source texture sample metadata');
    if (renderer.backend.utils.getTextureSampleData(texture).primarySamples !== 1)
      fail('TEXTURE', 'Native graphics texture bindings require resolved color views');
    return texture;
  }
  function textureSignature(nodes) {
    const aliases = new Map(), checked = new Set();
    return JSON.stringify(nodes.map(node => {
      const texture = node.value;
      if (!checked.has(texture)) { inspectTexture(texture); checked.add(texture); }
      // The pinned builder aliases uniforms by texture UUID. Keep that alias
      // partition, not the identities themselves, in the structural signature:
      // replacing one compatible sampled image must not rebuild the shader.
      if (!aliases.has(texture.uuid)) aliases.set(texture.uuid, aliases.size);
      return [aliases.get(texture.uuid), texture.type, texture.format, texture.colorSpace, texture.channel,
        texture.isRenderTargetTexture === true, texture.normalized === true,
        texture.minFilter === three.NearestFilter && texture.magFilter === three.NearestFilter,
        node.sampler, node.updateMatrix, node.groupNode?.name];
    }));
  }
  function inspect(material, object, context = {}) {
    live();
    const {scene, camera} = context;
    if (!(material instanceof three.NodeMaterial) ||
        ![three.MeshBasicNodeMaterial.prototype, three.NodeMaterial.prototype, ...litPrototypes].includes(Object.getPrototypeOf(material)))
      fail('MATERIAL', 'Native node graphs require Basic, Toon, Lambert, Phong, Standard, Physical or unlit base NodeMaterial');
    const prototype = Object.getPrototypeOf(material), lit = litMaterial(material),
      phong = prototype === three.MeshPhongNodeMaterial.prototype, standard = standardPrototypes.includes(prototype),
      physical = prototype === three.MeshPhysicalNodeMaterial.prototype;
    if (!lit && !material.isMeshBasicNodeMaterial && material.lights !== false)
      fail('LIGHTS', 'Base NodeMaterial graphs must be unlit');
    sourceLights(material, context);
    if (physical && (material.useTransmission || material.useDispersion))
      fail('TRANSMISSION', 'Native physical transmission and dispersion require the captured scene-color pass');
    sourceEnvironment(material, context);
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
      if (key.endsWith('Node') && value != null && !SLOTS.has(key) && !(lit && LIT_SLOTS.has(key)) &&
          !(phong && PHONG_SLOTS.has(key)) && !(standard && STANDARD_SLOTS.has(key)) && !(physical && PHYSICAL_SLOTS.has(key)))
        fail('GRAPH', `Native node graph slot ${key} is not admitted`);
    }
    if (renderer.getMRT?.() != null) fail('MRT', 'Native node graphs currently write one color attachment');
    const seen = new Set(), stack = GRAPH_SLOTS.map(key => material[key]).filter(Boolean);
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
  function checkNode(node, lightsNode = null, lights = [], environmentNode = null, boundTextures = null) {
    if (!node?.isNode) fail('GRAPH', 'Expected source TSL nodes');
    if (node.isStorageTextureNode || node.isCubeTextureNode || node.isTexture3DNode)
      fail('TEXTURE', 'Storage, cube and volume texture graphs need additional native bindings');
    if (node.isTextureNode) {
      // Source shader functions introduce unbound TextureNode parameters with
      // an empty placeholder image. Only real native bindings need residency;
      // authored graph inputs were already inspected before construction.
      if (boundTextures === null || boundTextures.has(node.value)) inspectTexture(node.value);
      if (node.compareNode !== null || node.depthNode !== null)
        fail('TEXTURE', 'Texture comparison and array-layer sampling are not admitted');
    } else if (node.value?.isTexture && !(environmentNode && node.constructor === three.PMREMNode && node.value === environmentNode.value))
      fail('TEXTURE', 'Native graphics textures require source TextureNode bindings');
    if (node.isStorageBufferNode) inspectStorage(node);
    else if (node.isBufferNode || node.isBufferAttributeNode || node.isComputeNode || node.isAtomicFunctionNode)
      fail('STORAGE', 'Explicit uniform buffers, attribute conversions, atomic and compute graphs are not admitted');
    if (node.isAssignNode) {
      const seen = new Set(), targets = [node.targetNode];
      while (targets.length) {
        const target = targets.pop(); if (!target || seen.has(target)) continue;
        seen.add(target);
        if (target.isStorageBufferNode) fail('STORAGE', 'Native graphics storage bindings are read-only');
        // An explicit local variable owns a value copied from its initializer;
        // assigning it does not write back to the source storage address.
        if (target.isVarNode && target.intent === false) continue;
        if (target.structNode) targets.push(target.structNode);
        // Assignment addresses follow the node/target chain; an index expression
        // may read storage without making the assigned local variable writable.
        if (target.node) targets.push(target.node);
        if (target.targetNode) targets.push(target.targetNode);
      }
    }
    if (node.isLightingNode || node.isLightsNode || node.isLightingContextNode || node.light?.isLight) {
      const admitted = lightsNode && (node === lightsNode ||
        environmentNode && node.constructor === three.EnvironmentNode && node.envNode === environmentNode ||
        node.isAnalyticLightNode && lights.includes(node.light) &&
        Object.getPrototypeOf(node) === lightClasses.get(node.light.constructor)?.prototype);
      if (!admitted) fail('LIGHTS', 'Only the captured source scene light and environment context is admitted');
    }
    if (node.isOutputStructNode || node.isMRTNode) fail('MRT', 'Native node graphs currently write one color attachment');
  }
  function keyOf(material, object, {camera, lightingEnabled = renderer.lighting?.enabled !== false} = {}, lights = [], environment = null) {
    // Match the source's graph cache key. Uniform values never participate.
    // Object/material references remain separate cache dimensions, preserving
    // builder nodes which close over a particular source object.
    return JSON.stringify([material.version, material.customProgramCacheKey(), material.side,
      material.vertexColors, material.flatShading, material.transparent, material.blending, material.premultipliedAlpha,
      material.alphaTest > 0, material.depthWrite, material.depthTest, material.lights,
      object.geometry.id, camera?.isPerspectiveCamera === true,
      environment ? [environment.id, environment.mapping, environment.isRenderTargetTexture === true, environment.isPMREMTexture === true] : null,
      Object.entries(object.geometry.attributes).map(([name, a]) => [name, a.itemSize, a.normalized, a.gpuType, a.array?.constructor.name,
        a.isInterleavedBufferAttribute, a.data?.array?.constructor.name, a.data?.stride, a.offset]),
      renderer.logarithmicDepthBuffer, renderer.reversedDepthBuffer,
      litMaterial(material) && material.fragmentNode == null ? [lightingEnabled, lights.map(light => light.id)] : null,
      Object.getPrototypeOf(material) === three.MeshPhysicalNodeMaterial.prototype ?
        [material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useRetroreflection] : null]);
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
    const lights = sourceLights(material, context), environment = sourceEnvironment(material, context);
    const key = keyOf(material, object, context, lights, environment), byMaterial = cache.get(object);
    const previous = byMaterial?.get(material);
    if (previous?.key === key && textureSignature(previous.textureNodes) === previous.textureSignature &&
        storageSignature(previous.storageNodes) === previous.storageSignature) return previous.descriptor;
    if (entries.size - (previous ? 1 : 0) >= maxPrograms) fail('LIMIT', 'Native node program capacity exceeded');
    const builder = new three.WGSLNodeBuilder(object, renderer);
    builder.material = material; builder.context.material = material;
    builder.scene = context.scene ?? null; builder.camera = context.camera ?? null;
    const lightsNode = litMaterial(material) ? three.TSL.lights(lights) : null;
    if (lightsNode) builder.lightsNode = lightsNode;
    const environmentNode = environmentNodeFor(environment);
    if (environmentNode) builder.environmentNode = environmentNode;
    try { builder.build(); }
    catch (cause) {
      if (cause instanceof ThreeNodeProgramError) throw cause;
      throw new ThreeNodeProgramError('BUILD', `Pinned node construction failed: ${cause?.message ?? cause}`, {cause});
    }
    // PMREM setup has now generated the source atlas. A later in-place refresh
    // would recursively capture renderer calls inside a whole-render texture
    // transaction. Refuse that boundary explicitly, before source callbacks;
    // changing environment identity still builds a fresh atlas normally.
    if (environment && environment.mapping !== three.CubeUVReflectionMapping)
      environmentVersions.set(environment, environment.pmremVersion);
    if (builder.nodes.size > maxNodes) fail('LIMIT', 'Generated native node graph exceeds its node budget');
    const bindings = builder.getBindings();
    const boundTextures = new Set(bindings.flatMap(group => group.bindings)
      .filter(binding => binding.textureNode?.isTextureNode).map(binding => binding.textureNode.value));
    for (const node of builder.nodes) checkNode(node, lightsNode, lights, environmentNode, boundTextures);
    if (bindings.length > Math.min(maxBindGroups, renderer._device?.limits?.maxBindGroups ?? 4))
      fail('LIMIT', 'Native node graph exceeds its bind group budget');
    let uniformBytes = 0;
    const sampledCounts = [0, 0], samplerCounts = [0, 0], storageCounts = [0, 0];
    const groups = bindings.map((group, index) => {
      let byteLength = 0;
      const entries = group.bindings.map((binding, slot) => {
        const base = {binding: slot, visibility: binding.visibility};
        if (binding.isNodeUniformsGroup === true) {
          if (byteLength) fail('BINDING', 'Native graphics groups admit one uniform block per source group');
          byteLength = binding.byteLength;
          if (binding.uniforms.some(u => !TYPES.has(u.nodeUniform?.type)))
            fail('UNIFORM', 'Native node uniforms must be ordinary scalars, vectors or matrices');
          if (!Number.isSafeInteger(byteLength) || byteLength < 4 || byteLength % 4 ||
              byteLength > (renderer._device?.limits?.maxUniformBufferBindingSize ?? 65536))
            fail('LIMIT', 'Native node uniform buffer exceeds device binding limits');
          return Object.freeze({...base, buffer: Object.freeze({type: 'uniform', hasDynamicOffset: true, minBindingSize: byteLength})});
        }
        if (binding.isStorageBuffer && binding.nodeUniform?.isStorageBufferNode) {
          const minBindingSize = inspectStorage(binding.nodeUniform);
          for (let stage = 0; stage < 2; stage++) if (binding.visibility & (1 << stage))
            if (++storageCounts[stage] > (renderer._device?.limits?.maxStorageBuffersPerShaderStage ?? 8))
              fail('LIMIT', 'Native node storage bindings exceed per-stage device limits');
          // r186 WGSLNodeBuilder.getNodeAccess makes non-atomic graphics reads
          // read-only even when the same source node is writable in compute.
          return Object.freeze({...base, buffer: Object.freeze({type: 'read-only-storage', hasDynamicOffset: false, minBindingSize})});
        }
        if (!binding.textureNode?.isTextureNode || binding.store || (!binding.isSampledTexture && !binding.isSampler))
          fail('BINDING', 'Native graphics groups require uniform blocks or ordinary sampled 2D texture bindings');
        inspectTexture(binding.textureNode.value);
        const sampled = binding.isSampledTexture === true, counts = sampled ? sampledCounts : samplerCounts;
        const limit = renderer._device?.limits?.[sampled ? 'maxSampledTexturesPerShaderStage' : 'maxSamplersPerShaderStage'] ?? 16;
        for (let stage = 0; stage < 2; stage++) if (binding.visibility & (1 << stage))
          if (++counts[stage] > limit) fail('LIMIT', 'Native node texture bindings exceed per-stage device limits');
        return Object.freeze(sampled ? {...base, texture: Object.freeze({sampleType: 'float', viewDimension: '2d', multisampled: false})} :
          {...base, sampler: Object.freeze({type: 'filtering'})});
      });
      uniformBytes += byteLength;
      return Object.freeze({index, name: group.name, byteLength, entries: Object.freeze(entries)});
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
    const textureNodes = [...builder.nodes].filter(node => node.isTextureNode && boundTextures.has(node.value));
    const signature = textureSignature(textureNodes);
    // Node.build() deduplicates unsized storage by attribute before addNode().
    // Walk the original child edges too, including graphs created inside Fn,
    // so changing one alias cannot silently keep the old shared binding.
    const graphNodes = new Set(builder.nodes);
    for (const node of graphNodes) {
      if (graphNodes.size > maxNodes) fail('LIMIT', 'Generated native node graph exceeds its node budget');
      if (typeof node.getChildren === 'function') for (const child of node.getChildren()) graphNodes.add(child);
    }
    const storageNodes = [...graphNodes].filter(node => node.isStorageBufferNode);
    const storageKey = storageSignature(storageNodes);
    nodeStates.set(descriptor, {owner: api, builder, bindings, material, object, environment, textureNodes, textureSignature: signature,
      storageNodes, storageSignature: storageKey});
    const entry = {key, descriptor, material, object, geometry: object.geometry, bytes, textureNodes, textureSignature: signature,
      storageNodes, storageSignature: storageKey};
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
  function readTextures(state) {
    if (textureSignature(state.textureNodes) !== state.textureSignature)
      fail('CHANGED', 'Native node texture layout, UV channel or binding aliases changed; compile the graph again');
    const textures = [];
    for (const [group, source] of state.bindings.entries()) for (const [binding, value] of source.bindings.entries())
      if (value.isSampledTexture || value.isSampler) textures.push(Object.freeze({group, binding,
        texture: value.textureNode.value, isSampler: value.isSampler === true, isSampledTexture: value.isSampledTexture === true,
        store: false, access: null, mipLevel: 0}));
    return Object.freeze(textures);
  }
  function readStorage(state) {
    if (storageSignature(state.storageNodes) !== state.storageSignature)
      fail('CHANGED', 'Native storage layout, struct schema or binding aliases changed; compile the graph again');
    const storage = [];
    for (const [group, source] of state.bindings.entries()) for (const [binding, value] of source.bindings.entries())
      if (value.isStorageBuffer) storage.push(Object.freeze({group, binding, attribute: value.attribute}));
    return Object.freeze(storage);
  }
  function captureDraw(descriptor, context = {}) {
    const state = stateOf(descriptor);
    assertEnvironmentVersion(state.environment);
    // Reject a stale or newly atomic schema before observable NodeFrame updates;
    // readStorage below also catches mutations performed by those updates.
    if (state.storageNodes.length) readStorage(state);
    return withFrame(state, context, () => {
      for (const node of state.builder.updateBeforeNodes) frame.updateBeforeNode(node);
      for (const node of state.builder.updateNodes) frame.updateNode(node);
      const textures = readTextures(state);
      const storage = readStorage(state);
      const uniforms = state.bindings.map((group, index) => {
        const binding = group.bindings.find(value => value.isNodeUniformsGroup);
        if (!binding) return new Uint8Array(0);
        binding.update();
        const buffer = binding.buffer;
        if (buffer.byteLength !== descriptor.groups[index].byteLength) fail('CHANGED', 'Native node uniform layout changed during its update');
        const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength).slice();
        binding.clearUpdateRanges(); return bytes;
      });
      return {uniforms, textures, storage};
    });
  }
  function capture(descriptor, context = {}) { return captureDraw(descriptor, context).uniforms; }
  function after(descriptor, context = {}) {
    const state = stateOf(descriptor);
    return withFrame(state, context, () => { for (const node of state.builder.updateAfterNodes) frame.updateAfterNode(node); });
  }
  const api = Object.freeze({inspect, compile, capture, captureDraw, after,
    textures(descriptor) { return readTextures(stateOf(descriptor)); },
    storage(descriptor) { return readStorage(stateOf(descriptor)); },
    release(descriptor) { for (const entry of entries) if (entry.descriptor === descriptor) { retire(entry); break; } },
    dispose() { if (disposed) return; disposed = true; for (const entry of [...entries]) retire(entry); },
    get diagnostics() { return Object.freeze({programs: entries.size, shaderBytes, maxPrograms, maxShaderBytes}); }});
  return api;
}
