/** Native r186 TSL graphics preparation. The source WGSLNodeBuilder and NodeFrame
 * are retained helpers; this module constructs no renderer/backend and submits
 * no GPU commands. Descriptors preserve the builder's native bind groups.
 *
 * The profile is rigid/skinned meshes and CPU/storage-backed InstancedMesh objects, unlit
 * or Toon/Lambert/Phong/Standard/Physical node materials and corresponding
 * built-in materials with authored slots, ordinary or registered direct analytic
 * lights, source float/integer attributes, scalar/vector/matrix uniforms,
 * source bone palettes, authored typed uniform arrays and multiple uniform blocks per source group,
 * sampled 2D/CPU-data cube textures, target-owned depth comparisons and read-only
 * scalar/vector/mat4 or flat-struct storage buffers.
 * Physical direct lighting retains the source DFG texture, clearcoat, sheen,
 * iridescence, anisotropy, retroreflection and source PMREM environment lighting.
 * Directional/spot/point Basic/PCF shadows and the registered SunLight cascade lifecycle
 * retain source shadow nodes and ShadowMaterial receivers. Transmission,
 * writable/atomic storage, VSM shadows, clipping and MRT fail explicitly.
 * capture() is a logical-draw operation, never a
 * preparation retry: it runs source before/current updates and owns the bytes.
 * after() belongs after that draw has been accepted. Callers capture before any
 * asynchronous pipeline preparation and retain the resulting byte arrays.
 */
import {createThreeNodeAttributes} from './three_node_attributes.mjs';
import {registerThreeNativeShadow, updateThreeNativeShadow} from './three_node_shadows.mjs';

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
const SHADOW_SLOTS = new Set(['castShadowPositionNode', 'castShadowNode', 'maskShadowNode',
  'receivedShadowPositionNode', 'receivedShadowNode']);
const GRAPH_SLOTS = [...SLOTS, ...LIT_SLOTS, ...PHONG_SLOTS, ...STANDARD_SLOTS, ...PHYSICAL_SLOTS, ...SHADOW_SLOTS, 'depthNode'];
// These source material accessors lower to ordinary 2D texture bindings. Light
// maps/AO introduce additional lighting nodes; environment maps have a separate
// PMREM context. Keep those dependencies explicit instead of silently ignoring them.
const MATERIAL_MAPS = new Set(['map', 'alphaMap', 'emissiveMap', 'normalMap', 'bumpMap', 'displacementMap',
  'specularMap', 'specularIntensityMap', 'specularColorMap', 'roughnessMap', 'metalnessMap', 'gradientMap',
  'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap', 'sheenColorMap', 'sheenRoughnessMap',
  'iridescenceMap', 'iridescenceThicknessMap', 'anisotropyMap', 'retroreflectivityMap']);
const TYPES = new Set(['float', 'int', 'uint', 'bool', 'color', 'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4',
  'uvec2', 'uvec3', 'uvec4', 'bvec2', 'bvec3', 'bvec4', 'mat2', 'mat3', 'mat4']);
const FRAME_FIELDS = ['renderer', 'scene', 'camera', 'object', 'material', 'compute', 'frameId', 'renderId', 'time', 'deltaTime'];
const nodeStates = new WeakMap();

/** True only for materials with an authored graphics slot. Default built-ins
 * continue through the existing fixed shader path. Admission is separate. */
export function hasThreeNodeProgram(material) {
  return material?.isMaterial === true && Object.keys(material).some(key => key.endsWith('Node') && material[key] != null);
}

export function createThreeNodePrograms({three, renderer, maxPrograms = 4096, maxNodes = 8192,
  maxShaderBytes = 16 * 1024 * 1024, maxUniformBytes = 256 * 1024, maxAttributeBytes = 64 * 1024 * 1024, maxBindGroups = 4} = {}) {
  if (three?.REVISION !== '186' || typeof three.WGSLNodeBuilder !== 'function' ||
      typeof renderer?._nodes?.nodeFrame?.updateNode !== 'function')
    fail('SOURCE', 'Supply the pinned r186 namespace and the renderer NodeFrame');
  for (const [name, value] of Object.entries({maxPrograms, maxNodes, maxShaderBytes, maxUniformBytes, maxAttributeBytes, maxBindGroups}))
    if (!Number.isSafeInteger(value) || value < 1) fail('LIMIT', `Invalid ${name}`);
  const frame = renderer._nodes.nodeFrame, cache = new WeakMap(), entries = new Set();
  const standardPrototypes = [three.MeshStandardNodeMaterial.prototype, three.MeshPhysicalNodeMaterial.prototype];
  const litPrototypes = [three.MeshToonNodeMaterial.prototype, three.MeshLambertNodeMaterial.prototype,
    three.MeshPhongNodeMaterial.prototype, three.ShadowNodeMaterial.prototype, ...standardPrototypes];
  const lightClasses = new Map([[three.AmbientLight, three.AmbientLightNode], [three.DirectionalLight, three.DirectionalLightNode],
    [three.HemisphereLight, three.HemisphereLightNode], [three.PointLight, three.PointLightNode], [three.SpotLight, three.SpotLightNode]]);
  const lightClassIds = new WeakMap();
  let lightClassSerial = 0;
  const environmentNodes = new WeakMap(), environmentVersions = new WeakMap(), uniformBuffers = new WeakMap();
  const generatedAttributes = createThreeNodeAttributes(three, {maxBytes: maxAttributeBytes, fail});
  const instanceArrays = new WeakMap(), sourceIds = new WeakMap(); let sourceSerial = 0;
  const materialViews = new WeakMap();
  const sourceMaterialClasses = new Map([
    [three.MeshBasicMaterial, three.MeshBasicNodeMaterial], [three.MeshToonMaterial, three.MeshToonNodeMaterial],
    [three.MeshLambertMaterial, three.MeshLambertNodeMaterial], [three.MeshPhongMaterial, three.MeshPhongNodeMaterial],
    [three.MeshStandardMaterial, three.MeshStandardNodeMaterial], [three.MeshPhysicalMaterial, three.MeshPhysicalNodeMaterial],
    [three.ShadowMaterial, three.ShadowNodeMaterial]
  ].map(([Source, Native]) => [Source.prototype, {Source, Native}]));
  let disposed = false, shaderBytes = 0, serial = 0;
  const live = () => { if (disposed) fail('DISPOSED', 'Native node program preparation is disposed'); };
  const litMaterial = material => litPrototypes.includes(Object.getPrototypeOf(material));
  function prepareMaterial(material) {
    if (material instanceof three.NodeMaterial) return material;
    const classes = material instanceof three.Material && sourceMaterialClasses.get(Object.getPrototypeOf(material));
    if (!classes || material.type !== classes.Source.name ||
        renderer.library.getMaterialNodeClass(material.type) !== classes.Native ||
        renderer.library.fromMaterial !== three.StandardNodeLibrary.prototype.fromMaterial)
      fail('MATERIAL', 'Native source materials require an ordinary built-in class and its pinned node-library conversion');
    let record = materialViews.get(material);
    if (!record) {
      const view = new classes.Native();
      record = {view, defaults: new Map(Object.entries(view)), keys: new Set()};
      materialViews.set(material, record);
    }
    // Match NodeLibrary.fromMaterial's enumerable-property copy without making
    // a new material on every admission/cache lookup. Reset removed properties
    // to source defaults; otherwise deleting an authored slot would retain it.
    const keys = new Set();
    for (const key in material) { record.view[key] = material[key]; keys.add(key); }
    for (const key of record.keys) if (!keys.has(key)) {
      if (record.defaults.has(key)) record.view[key] = record.defaults.get(key);
      else delete record.view[key];
    }
    record.keys = keys;
    return record.view;
  }
  const rawUniformBuffer = node => node instanceof three.BufferNode && !node.isStorageBufferNode && !node.isArrayBufferNode;
  function sourceId(value) {
    if (value === null || (typeof value !== 'object' && typeof value !== 'function')) return value;
    let id = sourceIds.get(value); if (id === undefined) { id = ++sourceSerial; sourceIds.set(value, id); }
    return id;
  }
  function inspectAttributeInput(node, object) {
    if (node.value instanceof Float32Array) return;
    const inputs = object?.isInstancedMesh ? [object.instanceMatrix, object.instanceColor].filter(Boolean) : [];
    const value = node.value;
    // Raw arrays receive a private owner from BufferAttributeNode. Instance.js
    // also constructs separate color/interleaved owners from admitted CPU
    // instance arrays. Public geometry owners need residency shared across all
    // geometry routes; copying them here would consume their partial ranges
    // without preserving an ordinary attribute alias's GPU-visible history.
    // r186's vec4 factory leaves node.instanced false for the matrix columns;
    // their InstancedInterleavedBuffer is the authoritative instancing marker.
    const instanced = node.instanced === true ||
      value instanceof three.InstancedInterleavedBuffer && value.isInstancedInterleavedBuffer === true;
    const derivedInstance = instanced && !inputs.includes(value) &&
      (value instanceof three.InstancedBufferAttribute || value instanceof three.InstancedInterleavedBuffer) &&
      inputs.some(input => input instanceof three.InstancedBufferAttribute && !input.isStorageInstancedBufferAttribute &&
        value.array === input.array);
    if (!derivedInstance)
      fail('ATTRIBUTE', 'Native generated streams require raw Float32Array inputs or the source-derived CPU instance owners; public attribute inputs need shared residency');
  }
  function attributeSignature(nodes, object) {
    return JSON.stringify(nodes.map(node => {
      inspectAttributeInput(node, object);
      const attribute = node.attribute, owner = attribute?.isInterleavedBufferAttribute ? attribute.data : attribute;
      const array = owner?.array, inputArray = ArrayBuffer.isView(node.value) ? node.value : node.value?.array;
      if (!attribute || !array || inputArray !== array)
        fail('CHANGED', 'Native buffer attribute nodes must retain their source backing array; replace the node or instance attribute');
      if (owner.isStorageBufferAttribute || owner.isStorageInstancedBufferAttribute || owner.isIndirectStorageBufferAttribute)
        fail('ATTRIBUTE', 'Compute-backed attribute conversion requires shared GPU residency, not a CPU attribute snapshot');
      return [node.id, node.nodeType, node.bufferType, node.bufferStride, node.bufferOffset, node.instanced, node.usage,
        sourceId(node.value), sourceId(attribute), sourceId(owner), sourceId(array), array.length, array.byteLength,
        attribute.count, attribute.itemSize, attribute.normalized, attribute.gpuType, attribute.isInterleavedBufferAttribute,
        attribute.offset, attribute.isInstancedBufferAttribute, owner.stride, owner.meshPerAttribute,
        owner.isInstancedInterleavedBuffer, owner.usage, sourceId(owner.onUploadCallback)];
    }));
  }
  function inspectUniformBuffer(node) {
    // BufferNode emits the caller's array layout verbatim. Unlike UniformArrayNode,
    // it inserts no per-element padding: scalar/vec2 arrays therefore do not have
    // a legal uniform-address-space stride in the pinned WGSL output.
    const type = /^(?:(i|u)?vec([34])|mat([234])|color)$/.exec(node.nodeType ?? '');
    const matrix = Number(type?.[3] ?? 0), stride = matrix === 3 ? 12 : matrix ? matrix * matrix : 4;
    const tail = matrix === 3 ? 11 : type?.[2] === '3' || node.nodeType === 'color' ? 3 : stride;
    const Scalar = type?.[1] === 'i' ? Int32Array : type?.[1] === 'u' ? Uint32Array : Float32Array;
    const array = node.value, required = (node.bufferCount - 1) * stride + tail;
    if (!type || !rawUniformBuffer(node) || !(array instanceof Scalar) ||
        !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable ||
        !Number.isSafeInteger(node.bufferCount) || node.bufferCount < 1 ||
        !Number.isSafeInteger(required) || array.length < required ||
        !Array.isArray(node.updateRanges) || node.updateRanges.length > maxNodes ||
        node.updateRanges.some(range => !range || !Number.isSafeInteger(range.start) || range.start < 0 ||
          !Number.isSafeInteger(range.count) || range.count < 0 || range.start + range.count > array.length))
      fail('UNIFORM', 'Native uniform arrays require fixed unshared 32-bit arrays, valid component ranges and a matching padded vec3/vec4 or matrix layout');
    // NodeUniformBuffer pads the allocation, not the uploaded array. A final
    // vec3/mat3 padding word may be absent; extra source capacity is also legal.
    const bytes = Math.ceil(array.byteLength / 16) * 16;
    if (bytes > (renderer._device?.limits?.maxUniformBufferBindingSize ?? 65536) || bytes > maxUniformBytes)
      fail('LIMIT', 'Native uniform array exceeds its binding or draw byte budget');
    return bytes;
  }
  function uniformSignature(nodes, builder) {
    return JSON.stringify(nodes.map(node => {
      const bytes = inspectUniformBuffer(node), group = node.groupNode;
      // Source uniforms share by their node hash, never by typed-array identity.
      // Values and same-layout replacements remain live NodeUniformBuffer inputs.
      return [node.getUniformHash(builder), node.nodeType, node.bufferType, node.bufferCount, bytes,
        node.name, group?.id, group?.name, group?.order, group?.shared, group?.updateType];
    }));
  }
  function inspectStorage(node) {
    const attribute = node.value, struct = node.structTypeNode;
    const type = /^(?:(i|u)?vec([234])|(float|int|uint|mat4))$/.exec(node.nodeType ?? '');
    const components = type?.[3] === 'mat4' ? 16 : type?.[2] ? Number(type[2]) : 1;
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
      fail('STORAGE', 'Native storage reads require matching non-atomic scalar/vector/mat4 or flat-struct 32-bit storage attributes');
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
  function inspectInstances(object) {
    if (!object.isInstancedMesh) return null;
    if (!(object instanceof three.InstancedMesh) || object.isSkinnedMesh || object.isBatchedMesh || object.morphTexture != null ||
        !(object.geometry instanceof three.BufferGeometry) || object.geometry.isInstancedBufferGeometry ||
        Object.values(object.geometry.morphAttributes ?? {}).some(list => list.length))
      fail('OBJECT', 'Native instancing requires source InstancedMesh objects without skinning or morph data');
    const cpu = [];
    const shape = (attribute, type) => {
      if (attribute instanceof three.StorageInstancedBufferAttribute && attribute.isStorageInstancedBufferAttribute === true) {
        const bytes = inspectStorage({value: attribute, nodeType: type, bufferCount: attribute.count, access: 'readWrite'});
        if (attribute.isInstancedBufferAttribute !== true) fail('OBJECT', 'Native instance attributes must retain their source instancing flag');
        return [attribute.id, attribute.count, bytes, 'storage'];
      }
      const array = attribute?.array, stride = type === 'mat4' ? 16 : 3;
      if (!(attribute instanceof three.InstancedBufferAttribute) || attribute.isInstancedBufferAttribute !== true ||
          attribute.isStorageInstancedBufferAttribute || attribute.isInterleavedBufferAttribute || attribute.normalized ||
          attribute.gpuType === three.IntType || !(array instanceof Float32Array) ||
          !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable || attribute.itemSize !== stride ||
          !Number.isSafeInteger(attribute.count) || attribute.count < 1 || attribute.count > 0xffffffff ||
          array.length !== attribute.count * stride || attribute.meshPerAttribute !== 1 ||
          !Number.isSafeInteger(attribute.version) || attribute.version < 0)
        fail('OBJECT', 'Native CPU instances require fixed unshared Float32 matrix/RGB attributes with divisor one');
      if (attribute.onUploadCallback !== three.BufferAttribute.prototype.onUploadCallback)
        fail('HOOK', 'Native CPU instance attributes require the source upload callback');
      if (instanceArrays.has(attribute) && instanceArrays.get(attribute) !== array)
        fail('CHANGED', 'Source instance nodes retain their original CPU array; replace the instance attribute to change its backing array');
      if (array.byteLength > maxAttributeBytes || array.byteLength > (renderer._device?.limits?.maxBufferSize ?? 256 * 1024 * 1024))
        fail('LIMIT', 'Native CPU instance input exceeds its byte budget');
      cpu.push(attribute);
      return [attribute.id, attribute.count, array.byteLength, 'cpu', sourceId(array), attribute.usage];
    };
    const matrix = shape(object.instanceMatrix, 'mat4'), color = object.instanceColor == null ? null : shape(object.instanceColor, 'vec3');
    const uniformLimit = matrix[3] === 'cpu' ? renderer.backend?.capabilities?.getUniformBufferLimit?.() : null;
    if (matrix[3] === 'cpu' && (!Number.isSafeInteger(uniformLimit) || uniformLimit < 16))
      fail('SOURCE', 'Native CPU instances require the source uniform buffer size capability');
    // The source instancedMesh Fn captures these attribute objects at build.
    // Packed vec3 colors and their compute-padded form share one shader layout.
    return {signature: JSON.stringify([matrix, color, uniformLimit]), capacity: Math.min(matrix[1], color?.[1] ?? Infinity), cpu};
  }
  function assertInstances(object, signature) {
    if ((inspectInstances(object)?.signature ?? null) !== signature)
      fail('CHANGED', 'Native instance matrix/color identity or layout changed; compile the graph again');
  }
  function inspectSkinning(object) {
    if (!object.isSkinnedMesh) return null;
    const skeleton = object.skeleton, matrices = skeleton?.boneMatrices;
    const count = skeleton?.bones?.length;
    if (!(object instanceof three.SkinnedMesh) || object.isInstancedMesh || object.isBatchedMesh ||
        !(skeleton instanceof three.Skeleton) || !Array.isArray(skeleton.bones) ||
        !Number.isSafeInteger(count) || count < 1 ||
        !(matrices instanceof Float32Array) || !(matrices.buffer instanceof ArrayBuffer) || matrices.buffer.resizable ||
        matrices.length < count * 16 || !Array.isArray(skeleton.boneInverses) || skeleton.boneInverses.length !== count ||
        !(object.bindMatrix instanceof three.Matrix4) || !(object.bindMatrixInverse instanceof three.Matrix4) ||
        object.geometry?.attributes.skinIndex?.itemSize !== 4 || object.geometry?.attributes.skinWeight?.itemSize !== 4)
      fail('OBJECT', 'Native skinning requires a source SkinnedMesh, complete skeleton palette, bind matrices and four-component skin attributes');
    const uniformLimit = renderer.backend?.capabilities?.getUniformBufferLimit?.();
    if (!Number.isSafeInteger(uniformLimit) || uniformLimit < 16)
      fail('SOURCE', 'Native skinning requires the source uniform buffer size capability');
    // Skinning.js closes over the bone count and chooses its uniform/texture
    // route at construction. Its references remain live: replacing a compatible
    // skeleton or palette must not freeze source identity or trigger CPU skinning.
    return JSON.stringify([count, uniformLimit, count * 64 <= uniformLimit ? matrices.byteLength : null]);
  }
  function assertSkinning(object, signature) {
    if (inspectSkinning(object) !== signature)
      fail('CHANGED', 'Native skeleton palette layout changed; compile the graph again');
  }
  function sourceLightClass(light) {
    const Source = light?.constructor;
    if (!(light instanceof three.Light) || typeof Source !== 'function' ||
        Object.getPrototypeOf(light) !== Source.prototype || light.isLight !== true)
      fail('LIGHTS', 'Native lights require exact source Light instances');
    const Native = renderer.library.getLightNodeClass(Source), builtin = lightClasses.get(Source);
    if (builtin) {
      if (Native !== builtin) fail('LIGHTS', 'Native built-in lights require their pinned node-library mappings');
      return Native;
    }
    // Addons such as r186 SunLight register a direct AnalyticLightNode subclass.
    // Admit that public extension point without cloning the light or replacing
    // the source analytic node's FRAME lifecycle. Other light models stay gated.
    const prototype = Native?.prototype;
    const direct = prototype && Object.getOwnPropertyDescriptor(prototype, 'setupDirect');
    if (renderer.library.getLightNodeClass !== three.StandardNodeLibrary.prototype.getLightNodeClass ||
        Object.getPrototypeOf(Source.prototype) !== three.Light.prototype ||
        !prototype || Object.getPrototypeOf(prototype) !== three.AnalyticLightNode.prototype ||
        prototype.constructor !== Native || typeof direct?.value !== 'function' ||
        Object.getOwnPropertyNames(prototype).some(key => !['constructor', 'setupDirect', 'setupShadowNode'].includes(key)))
      fail('LIGHTS', 'Registered native lights require a direct analytic node with the pinned source lifecycle');
    return Native;
  }
  function lightSignature(lights) {
    const id = value => {
      if (!lightClassIds.has(value)) lightClassIds.set(value, ++lightClassSerial);
      return lightClassIds.get(value);
    };
    return JSON.stringify(lights.map(light => {
      const Native = sourceLightClass(light);
      if (light.map != null || light.colorNode != null)
        fail('LIGHTS', 'Native lit graph projected maps and custom light color nodes are not admitted');
      const shadow = light.castShadow && renderer.shadowMap?.enabled === true ? light.shadow : null;
      if (shadow && (!(shadow instanceof three.LightShadow) ||
          ![three.BasicShadowMap, three.PCFShadowMap].includes(renderer.shadowMap.type) || renderer.shadowMap.transmitted ||
          shadow.filterNode != null || shadow.shadowNode != null ||
          ![three.DirectionalLight, three.SpotLight, three.PointLight].includes(light.constructor) &&
            !(light.isSunLight === true && shadow.isSunLightShadow === true)))
        fail('SHADOW', 'Native source shadows require ordinary directional, spot, point or registered SunLight Basic/PCF maps');
      return [light.id, id(light.constructor), id(Native), id(Native.prototype.setupDirect),
        id(Native.prototype.setupShadowNode), light.castShadow === true, shadow ? id(shadow) : null,
        renderer.shadowMap?.enabled === true, renderer.shadowMap?.type, renderer.shadowMap?.transmitted === true];
    }));
  }
  function checkAnalyticLight(node, lights) {
    if (!node.isAnalyticLightNode || !lights.includes(node.light)) return false;
    const Native = sourceLightClass(node.light);
    if (Object.getPrototypeOf(node) !== Native.prototype) return false;
    if (!lightClasses.has(node.light.constructor)) {
      const source = three.AnalyticLightNode.prototype;
      if (node.updateType !== 'frame' || node.updateBeforeType !== 'none' || node.updateAfterType !== 'none' ||
          node.setupDirect !== Native.prototype.setupDirect ||
          ['build', 'setup', 'analyze', 'generate', 'getHash', 'updateReference', 'getUpdateType',
            'getUpdateBeforeType', 'getUpdateAfterType', 'update', 'updateBefore', 'updateAfter']
            .some(key => node[key] !== source[key]))
        fail('LIGHTS', 'Registered analytic nodes must retain the pinned source setup and update lifecycle');
    }
    return true;
  }
  function sourceLights(material, {lights = [], lightingEnabled = renderer.lighting?.enabled !== false} = {}) {
    if (!litMaterial(material) || material.lights !== true || material.fragmentNode != null) return [];
    if (typeof lightingEnabled !== 'boolean' || lightingEnabled !== (renderer.lighting?.enabled !== false))
      fail('LIGHTS', 'Captured lighting enablement must match the source renderer during construction');
    if (!lightingEnabled) return [];
    if (!Array.isArray(lights)) fail('LIGHTS', 'Supply the selected source scene lights');
    if (lights.length > 8) fail('LIMIT', 'Native lit node graphs currently admit at most eight selected scene lights');
    lightSignature(lights);
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
    return texture === null ? null : sourcePMREMTexture(texture);
  }
  function sourcePMREMTexture(texture) {
    if (!(texture instanceof three.Texture) || texture.isF3DSceneEnvironment || texture.isF3DPMREMSource ||
        ![three.EquirectangularReflectionMapping, three.EquirectangularRefractionMapping, three.CubeUVReflectionMapping].includes(texture.mapping))
      fail('ENVIRONMENT', 'Native PBR environments require a ready equirectangular image or source CubeUV texture');
    inspectTexture(texture);
    return texture;
  }
  function assertEnvironmentVersion(texture) {
    if (texture && environmentVersions.has(texture) && environmentVersions.get(texture) !== texture.pmremVersion)
      fail('ENVIRONMENT_REFRESH', 'In-place PMREM regeneration must precede authored draw updates at the first native draw preparation boundary');
  }
  function inspectPMREMNode(node) {
    const texture = sourcePMREMTexture(node.value), target = node._generator?._pingPongRenderTarget;
    if (target && texture.mapping !== three.CubeUVReflectionMapping && texture.isPMREMTexture !== true) {
      // Match the pinned generator's source size and CubeUV target allocation.
      // Resizing would dispose resources still held by accepted queued draws.
      const size = 2 ** Math.floor(Math.log2(texture.image.width / 4));
      if (target.width !== 3 * Math.max(size, 16 * 7) || target.height !== 4 * size)
        fail('ENVIRONMENT_REFRESH', 'Changing a prepared PMREM source atlas size requires a new PMREMNode to preserve queued render work');
    }
    return texture;
  }
  function pmremSignature(nodes) {
    return JSON.stringify(nodes.map(node => {
      const texture = inspectPMREMNode(node);
      // Source PMREMNode.setup chooses its atlas and Y orientation from this
      // input. A same-format replacement still needs source setup to run again.
      return [texture.id, texture.mapping, texture.isRenderTargetTexture === true, texture.isPMREMTexture === true];
    }));
  }
  function environmentNodeFor(texture) {
    if (!texture) return null;
    let node = environmentNodes.get(texture);
    if (!node) { node = three.TSL.texture(texture); environmentNodes.set(texture, node); }
    return node;
  }
  function inspectTexture(texture) {
    const array = texture?.isDataArrayTexture === true;
    if (!(texture instanceof three.Texture) || texture.isStorageTexture ||
        texture.is3DTexture || texture.isArrayTexture && !array || texture.isData3DTexture ||
        array && !(texture instanceof three.DataArrayTexture) ||
        texture.isCompressedTexture || texture.isVideoTexture || texture.isVideoFrameTexture ||
        texture.isFramebufferTexture || texture.isExternalTexture)
      fail('TEXTURE', 'Native graphics nodes sample ordinary 2D/data-array textures, CPU-data cubes and target-owned color/depth textures');
    const depth = texture.isDepthTexture === true;
    if (depth && (!(texture instanceof three.DepthTexture) || !(texture.renderTarget instanceof three.RenderTarget) ||
        texture.renderTarget.depthTexture !== texture ||
        texture.isCubeTexture && (!(texture instanceof three.CubeDepthTexture) || texture.isCubeDepthTexture !== true)))
      fail('TEXTURE', 'Native depth sampling requires an attached source DepthTexture or CubeDepthTexture');
    const viewDimension = texture.isCubeTexture === true ? 'cube' : array ? '2d-array' : '2d';
    if (!depth && viewDimension === 'cube' && (!(texture instanceof three.CubeTexture) || texture.isRenderTargetTexture ||
        ![three.CubeReflectionMapping, three.CubeRefractionMapping].includes(texture.mapping) ||
        !Array.isArray(texture.image) || texture.image.length !== 6 || texture.image.some(face =>
          !(face instanceof three.DataTexture) || face.isDataTexture !== true ||
          !ArrayBuffer.isView(face.image?.data) || face.image.data instanceof DataView)))
      fail('TEXTURE', 'Native cube sampling requires six CPU DataTexture faces with source reflection or refraction mapping');
    if (!depth && (![three.UnsignedByteType, three.HalfFloatType, three.FloatType].includes(texture.type) ||
        [three.RedIntegerFormat, three.RGIntegerFormat, three.RGBAIntegerFormat, three.RGBIntegerFormat].includes(texture.format)))
      fail('TEXTURE', 'Native graphics texture formats must have floating-point samples');
    if (depth && texture.compareFunction !== null && typeof renderer.hasCompatibility !== 'function')
      fail('SOURCE', 'Native depth comparison requires source renderer compatibility metadata');
    const sampleType = depth ? 'depth' : texture.type === three.FloatType && !renderer.hasFeature('float32-filterable')
      ? 'unfilterable-float' : 'float';
    if (sampleType === 'unfilterable-float' &&
        (texture.magFilter !== three.NearestFilter ||
         ![three.NearestFilter, three.NearestMipmapNearestFilter].includes(texture.minFilter)))
      fail('TEXTURE', 'Native float32 texture filtering requires the float32-filterable feature; nearest texel reads remain available');
    const layout = renderer._getSampledTextureInputs?.().layoutOf({texture, isSampledTexture: true, isSampler: true, store: false});
    if (layout && (layout.texture?.sampleType !== sampleType || layout.texture.viewDimension !== viewDimension || layout.texture.multisampled))
      fail('TEXTURE', 'Native graphics texture bindings require a matching single-sampled color or depth view');
    if (typeof renderer.backend?.utils?.getTextureSampleData !== 'function')
      fail('SOURCE', 'The native renderer must provide source texture sample metadata');
    if (renderer.backend.utils.getTextureSampleData(texture).primarySamples !== 1)
      fail('TEXTURE', 'Native graphics texture bindings require resolved color views');
    return layout?.texture ?? {sampleType, viewDimension, multisampled: false};
  }
  function comparison(node, texture = node.value) {
    return texture.isDepthTexture === true && texture.compareFunction !== null &&
      node.isSampleCompare?.() === true && renderer.hasCompatibility(three.Compatibility.TEXTURE_COMPARE);
  }
  function textureValue(node, references = null) {
    const reference = references?.get(node.getBase());
    if (!reference) return node.value;
    let value = reference.material;
    for (const property of reference.properties) value = value?.[property];
    return value;
  }
  function textureSignature(nodes, references = null) {
    const aliases = new Map(), checked = new Set();
    return JSON.stringify(nodes.map(node => {
      const texture = references ? textureValue(node, references) : node.value;
      if (!checked.has(texture)) { inspectTexture(texture); checked.add(texture); }
      // The pinned builder aliases uniforms by texture UUID. Keep that alias
      // partition, not the identities themselves, in the structural signature:
      // replacing one compatible sampled image must not rebuild the shader.
      if (!aliases.has(texture.uuid)) aliases.set(texture.uuid, aliases.size);
      return [aliases.get(texture.uuid), texture.type, texture.format, texture.colorSpace, texture.channel,
        texture.isRenderTargetTexture === true, texture.normalized === true, texture.isCubeTexture === true,
        texture.isDataArrayTexture === true,
        texture.isCubeTexture === true ? texture.mapping : null,
        texture.isDepthTexture === true, texture.isDepthTexture === true ? texture.compareFunction : null,
        node.isSampleCompare?.() === true, texture.isDepthTexture === true && texture.compareFunction !== null
          ? renderer.hasCompatibility(three.Compatibility.TEXTURE_COMPARE) : null,
        texture.minFilter === three.NearestFilter && texture.magFilter === three.NearestFilter,
        node.sampler, node.updateMatrix, node.groupNode?.name];
    }));
  }
  function materialMapSignature(material) {
    const maps = [...MATERIAL_MAPS].filter(key => material[key]?.isTexture);
    return JSON.stringify([maps, material.normalMapType,
      textureSignature(maps.map(key => ({value: material[key]})))]);
  }
  function inspect(material, object, context = {}) {
    live();
    const sourceMaterial = material;
    material = prepareMaterial(material);
    const {scene, camera} = context;
    if (!(material instanceof three.NodeMaterial) ||
        ![three.MeshBasicNodeMaterial.prototype, three.NodeMaterial.prototype, ...litPrototypes].includes(Object.getPrototypeOf(material)))
      fail('MATERIAL', 'Native node graphs require Basic, Toon, Lambert, Phong, Standard, Physical, Shadow or unlit base NodeMaterial');
    const prototype = Object.getPrototypeOf(material), lit = litMaterial(material),
      phong = prototype === three.MeshPhongNodeMaterial.prototype, standard = standardPrototypes.includes(prototype),
      physical = prototype === three.MeshPhysicalNodeMaterial.prototype;
    if (!lit && !material.isMeshBasicNodeMaterial && material.lights !== false)
      fail('LIGHTS', 'Base NodeMaterial graphs must be unlit');
    sourceLights(material, context);
    if (physical && (material.useTransmission || material.useDispersion))
      fail('TRANSMISSION', 'Native physical transmission and dispersion require the captured scene-color pass');
    sourceEnvironment(material, context);
    if (!(object instanceof three.Mesh) || object.isBatchedMesh ||
        !(object.geometry instanceof three.BufferGeometry) || object.geometry.isInstancedBufferGeometry ||
        Object.values(object.geometry.morphAttributes ?? {}).some(list => list.length))
      fail('OBJECT', 'Native node graphs require rigid/skinned meshes or CPU/storage-backed InstancedMesh objects');
    inspectInstances(object);
    inspectSkinning(object);
    if (camera && (!(camera instanceof three.Camera) || camera.isArrayCamera || camera.isMultiViewCamera || camera.reversedDepth))
      fail('CAMERA', 'Native node graphs require an ordinary non-reversed camera');
    if ((scene?.fog && material.fog !== false) || material.clippingPlanes?.length || renderer.clippingPlanes?.length)
      fail('EFFECT', 'Native node graph fog and clipping require their own shader context');
    if (material.wireframe || material.alphaHash || material.alphaToCoverage)
      fail('RASTER', 'Native node graph wireframe, hashed alpha and alpha-to-coverage are not admitted');
    if (material.transparent && material.side === three.DoubleSide && !material.forceSinglePass)
      fail('RASTER', 'Transparent DoubleSide node graphs require forceSinglePass until side variants are supported');
    if (Object.keys(renderer.contextNode?.getFlowContextData?.() ?? {}).length)
      fail('GRAPH', 'Custom renderer node contexts require an explicit graphics context adapter');
    if (sourceMaterial.onBeforeCompile !== three.Material.prototype.onBeforeCompile ||
        sourceMaterial.onBeforeRender !== three.Material.prototype.onBeforeRender ||
        sourceMaterial.customProgramCacheKey !== (sourceMaterial.isNodeMaterial === true
          ? three.NodeMaterial.prototype.customProgramCacheKey : three.Material.prototype.customProgramCacheKey))
      fail('HOOK', 'Custom material shader/render hooks are not admitted');
    for (const key of Object.keys(material)) {
      const value = material[key];
      if (value?.isTexture) {
        if (!MATERIAL_MAPS.has(key)) fail('TEXTURE', `Native node graph material texture ${key} is not admitted`);
        if (value.isCubeTexture) fail('TEXTURE', 'Native material map slots require ordinary 2D textures');
        inspectTexture(value);
      }
      if (key.endsWith('Node') && value != null && !SLOTS.has(key) && !(lit && LIT_SLOTS.has(key)) &&
          !(phong && PHONG_SLOTS.has(key)) && !(standard && STANDARD_SLOTS.has(key)) && !(physical && PHYSICAL_SLOTS.has(key)) &&
          !SHADOW_SLOTS.has(key) && !(key === 'depthNode' && material.isShadowPassMaterial === true))
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
      if (node instanceof three.BufferAttributeNode) inspectAttributeInput(node, object);
      if (typeof node.getChildren === 'function') for (const child of node.getChildren()) stack.push(child);
    }
    return material;
  }
  function checkNode(node, lightsNode = null, lights = [], environmentNode = null, boundTextures = null) {
    if (!node?.isNode) fail('GRAPH', 'Expected source TSL nodes');
    if (node.isStorageTextureNode || node.isTexture3DNode)
      fail('TEXTURE', 'Storage and volume texture graphs need additional native bindings');
    if (node.isTextureNode) {
      if (node.isCubeTextureNode && (node.constructor !== three.CubeTextureNode ||
          Object.getPrototypeOf(node) !== three.CubeTextureNode.prototype || node.sampler !== true))
        fail('TEXTURE', 'Native cube graphs require the source CubeTextureNode with sampled direction coordinates');
      if ((node.isCubeTextureNode === true) !== (node.value?.isCubeTexture === true))
        fail('TEXTURE', 'Native texture nodes must match their 2D or cube source view');
      // Source shader functions introduce unbound TextureNode parameters with
      // an empty placeholder image. Only real native bindings need residency;
      // authored graph inputs were already inspected before construction.
      if (boundTextures === null || boundTextures.has(node.value)) {
        inspectTexture(node.value);
        if (node.isSampleCompare?.() === true && (!node.value.isDepthTexture || node.value.compareFunction === null || !node.sampler))
          fail('TEXTURE', 'Native comparison sampling requires an attached depth texture with a comparison function');
      }
      if (node.depthNode !== null && node.value?.isDataArrayTexture !== true)
        fail('TEXTURE', 'Native layer coordinates require a source DataArrayTexture');
    } else if (node.constructor === three.PMREMNode) inspectPMREMNode(node);
    else if (node.value?.isTexture)
      fail('TEXTURE', 'Native graphics textures require source TextureNode or PMREMNode bindings');
    if (node.isStorageBufferNode) inspectStorage(node);
    else if (rawUniformBuffer(node)) inspectUniformBuffer(node);
    else if (!(node instanceof three.BufferAttributeNode) &&
        (node.isBufferNode || node.isBufferAttributeNode || node.isComputeNode || node.isAtomicFunctionNode))
      fail('STORAGE', 'Managed uniform arrays, atomic and compute graphs are not admitted');
    if (node.isAssignNode) {
      const seen = new Set(), targets = [node.targetNode];
      while (targets.length) {
        const target = targets.pop(); if (!target || seen.has(target)) continue;
        seen.add(target);
        if (target.isStorageBufferNode) fail('STORAGE', 'Native graphics storage bindings are read-only');
        if (rawUniformBuffer(target)) fail('UNIFORM', 'Native graphics uniform buffers are read-only');
        if (target instanceof three.BufferAttributeNode) fail('ATTRIBUTE', 'Native graphics buffer attributes are read-only');
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
        checkAnalyticLight(node, lights) || lights.includes(node.light) && registerThreeNativeShadow(three, renderer, node));
      if (!admitted) fail('LIGHTS', 'Only the captured source scene light and environment context is admitted');
    }
    if (node.isOutputStructNode || node.isMRTNode) fail('MRT', 'Native node graphs currently write one color attachment');
  }
  function keyOf(material, object, {camera, lightingEnabled = renderer.lighting?.enabled !== false} = {}, lights = [], environment = null, instanceSignature = null, skinSignature = null) {
    // Match the source's graph cache key. Uniform values never participate.
    // Object/material references remain separate cache dimensions, preserving
    // builder nodes which close over a particular source object.
    return JSON.stringify([material.version, three.NodeMaterial.prototype.customProgramCacheKey.call(material), material.side,
      material.vertexColors, material.flatShading, material.transparent, material.blending, material.premultipliedAlpha,
      material.alphaTest > 0, material.depthWrite, material.depthTest, material.lights,
      object.geometry.id, camera?.isPerspectiveCamera === true, object.receiveShadow === true,
      instanceSignature, skinSignature, materialMapSignature(material),
      environment ? [environment.id, environment.mapping, environment.isRenderTargetTexture === true, environment.isPMREMTexture === true] : null,
      Object.entries(object.geometry.attributes).map(([name, a]) => [name, a.itemSize, a.normalized, a.gpuType, a.array?.constructor.name,
        a.isInterleavedBufferAttribute, a.data?.array?.constructor.name, a.data?.stride, a.offset]),
      renderer.logarithmicDepthBuffer, renderer.reversedDepthBuffer,
      litMaterial(material) && material.fragmentNode == null ? [lightingEnabled, lightSignature(lights)] : null,
      Object.getPrototypeOf(material) === three.MeshPhysicalNodeMaterial.prototype ?
        [material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useRetroreflection] : null]);
  }
  function retire(entry) {
    if (!entries.delete(entry)) return;
    shaderBytes -= entry.bytes;
    entry.material.removeEventListener('dispose', entry.onDispose);
    entry.geometry.removeEventListener('dispose', entry.onDispose);
    if (entry.attributeNodes.length) generatedAttributes.release(entry.descriptor);
    if (cache.get(entry.object)?.get(entry.material) === entry) cache.get(entry.object).delete(entry.material);
  }
  function compile(material, object, context = {}) {
    const preparedMaterial = inspect(material, object, context);
    const lights = sourceLights(preparedMaterial, context), environment = sourceEnvironment(preparedMaterial, context);
    const lightKey = lightSignature(lights);
    const instances = inspectInstances(object), instanceSignature = instances?.signature ?? null;
    const skinSignature = inspectSkinning(object);
    const key = keyOf(preparedMaterial, object, context, lights, environment, instanceSignature, skinSignature), byMaterial = cache.get(object);
    const previous = byMaterial?.get(material);
    if (previous?.key === key && pmremSignature(previous.pmremNodes) === previous.pmremSignature &&
        textureSignature(previous.textureNodes, previous.textureReferences) === previous.textureSignature &&
        storageSignature(previous.storageNodes) === previous.storageSignature &&
        uniformSignature(previous.uniformNodes, previous.builder) === previous.uniformSignature &&
        attributeSignature(previous.attributeNodes, object) === previous.attributeSignature) return previous.descriptor;
    if (entries.size - (previous ? 1 : 0) >= maxPrograms) fail('LIMIT', 'Native node program capacity exceeded');
    // setup() itself may filter a PMREM source. An existing stale atlas must
    // reach the first draw's ordered preparation boundary through its cached
    // graph, rather than being rewritten while another graph is constructed.
    assertEnvironmentVersion(environment);
    for (const entry of entries) for (const node of entry.pmremNodes) assertEnvironmentVersion(node.value);
    const builder = new three.WGSLNodeBuilder(object, renderer);
    builder.material = material; builder.context.material = material;
    builder.scene = context.scene ?? null; builder.camera = context.camera ?? null;
    const lightsNode = litMaterial(preparedMaterial) ? three.TSL.lights(lights) : null;
    if (lightsNode) builder.lightsNode = lightsNode;
    const environmentNode = environmentNodeFor(environment);
    if (environmentNode) builder.environmentNode = environmentNode;
    try { builder.build(); }
    catch (cause) {
      if (cause instanceof ThreeNodeProgramError) throw cause;
      throw new ThreeNodeProgramError('BUILD', `Pinned node construction failed: ${cause?.message ?? cause}`, {cause});
    }
    assertInstances(object, instanceSignature);
    assertSkinning(object, skinSignature);
    if (lightSignature(lights) !== lightKey) fail('CHANGED', 'Native light registration changed during source graph construction');
    // Initial source setup has generated its atlas. Later versions are
    // prepared at a source-ordered boundary outside draw resource capture.
    const pmremNodes = [...builder.nodes].filter(node => node.constructor === three.PMREMNode);
    for (const node of pmremNodes) if (node.value.mapping !== three.CubeUVReflectionMapping && !node.value.isPMREMTexture)
      environmentVersions.set(node.value, node.value.pmremVersion);
    const pmremKey = pmremSignature(pmremNodes);
    if (builder.nodes.size > maxNodes) fail('LIMIT', 'Generated native node graph exceeds its node budget');
    const bindings = builder.getBindings();
    const boundTextures = new Set(bindings.flatMap(group => group.bindings)
      .filter(binding => binding.textureNode?.isTextureNode).map(binding => binding.textureNode.value));
    for (const node of builder.nodes) {
      checkNode(node, lightsNode, lights, environmentNode, boundTextures);
      if (node instanceof three.BufferAttributeNode) inspectAttributeInput(node, object);
    }
    if (bindings.length > Math.min(maxBindGroups, renderer._device?.limits?.maxBindGroups ?? 4))
      fail('LIMIT', 'Native node graph exceeds its bind group budget');
    let uniformBytes = 0, uniformCount = 0;
    const alignment = renderer._device?.limits?.minUniformBufferOffsetAlignment ?? 256;
    if (!Number.isSafeInteger(alignment) || alignment < 1 || !Number.isInteger(Math.log2(alignment)))
      fail('LIMIT', 'Native uniform offsets require a valid device alignment');
    const sampledCounts = [0, 0], samplerCounts = [0, 0], storageCounts = [0, 0], uniformCounts = [0, 0];
    const groups = bindings.map((group, index) => {
      let byteLength = 0;
      const uniforms = [];
      const entries = group.bindings.map((binding, slot) => {
        const base = {binding: slot, visibility: binding.visibility};
        if (binding.isNodeUniformsGroup === true || binding.isNodeUniformBuffer === true) {
          const size = binding.isNodeUniformBuffer ? inspectUniformBuffer(binding.nodeUniform) : binding.byteLength;
          if (binding.isNodeUniformsGroup && binding.uniforms.some(u => !TYPES.has(u.nodeUniform?.type)))
            fail('UNIFORM', 'Native node uniforms must be ordinary scalars, vectors or matrices');
          if (!Number.isSafeInteger(size) || size < 4 || size % 4 ||
              size > (renderer._device?.limits?.maxUniformBufferBindingSize ?? 65536))
            fail('LIMIT', 'Native node uniform buffer exceeds device binding limits');
          if (++uniformCount > (renderer._device?.limits?.maxDynamicUniformBuffersPerPipelineLayout ?? 8))
            fail('LIMIT', 'Native node uniform bindings exceed the dynamic buffer pipeline limit');
          for (let stage = 0; stage < 2; stage++) if (binding.visibility & (1 << stage))
            if (++uniformCounts[stage] > (renderer._device?.limits?.maxUniformBuffersPerShaderStage ?? 12))
              fail('LIMIT', 'Native node uniform bindings exceed per-stage device limits');
          const offset = Math.ceil(byteLength / alignment) * alignment;
          byteLength = offset + size;
          if (!Number.isSafeInteger(byteLength) || byteLength > maxUniformBytes - uniformBytes)
            fail('LIMIT', 'Native node draw exceeds its uniform byte budget');
          uniforms.push(Object.freeze({binding: slot, offset, byteLength: size}));
          return Object.freeze({...base, buffer: Object.freeze({type: 'uniform', hasDynamicOffset: true, minBindingSize: size})});
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
          fail('BINDING', 'Native graphics groups require uniform blocks or ordinary sampled 2D or cube texture bindings');
        const textureLayout = inspectTexture(binding.textureNode.value);
        const sampled = binding.isSampledTexture === true, counts = sampled ? sampledCounts : samplerCounts;
        const limit = renderer._device?.limits?.[sampled ? 'maxSampledTexturesPerShaderStage' : 'maxSamplersPerShaderStage'] ?? 16;
        for (let stage = 0; stage < 2; stage++) if (binding.visibility & (1 << stage))
          if (++counts[stage] > limit) fail('LIMIT', 'Native node texture bindings exceed per-stage device limits');
        return Object.freeze(sampled ? {...base, texture: Object.freeze({...textureLayout})} :
          {...base, sampler: Object.freeze({type: comparison(binding.textureNode) ? 'comparison' :
            textureLayout.sampleType === 'float' ? 'filtering' : 'non-filtering'})});
      });
      uniformBytes += byteLength;
      return Object.freeze({index, name: group.name, byteLength, uniforms: Object.freeze(uniforms), entries: Object.freeze(entries)});
    });
    if (uniformBytes > maxUniformBytes) fail('LIMIT', 'Native node draw exceeds its uniform byte budget');
    const nodeAttributes = builder.getAttributesArray(), generated = [];
    const attributes = nodeAttributes.map((attribute, location) => {
      const vector = /^(?:(u|i)?vec([234])|(float|int|uint))$/.exec(attribute.type);
      const components = vector ? Number(vector[2] ?? 1) : 0;
      const scalar = vector?.[1] === 'u' || vector?.[3] === 'uint' ? 'uint'
        : vector?.[1] === 'i' || vector?.[3] === 'int' ? 'int' : 'float';
      const node = attribute.node, source = node?.attribute ?? object.geometry.attributes[attribute.name];
      if (!components || !source || source.itemSize !== components ||
          (node ? !(node instanceof three.BufferAttributeNode) : source.isInstancedBufferAttribute || source.data?.isInstancedInterleavedBuffer))
        fail('ATTRIBUTE', `Native node attribute ${attribute.name} requires a matching scalar/vector geometry attribute`);
      if (node) generated.push({name: attribute.name, attribute: source});
      return Object.freeze({name: attribute.name, location, locations: 1, components, scalar});
    });
    const vertex = builder.vertexShader, fragment = builder.fragmentShader, bytes = 2 * (vertex.length + fragment.length);
    if (bytes > maxShaderBytes - shaderBytes + (previous?.bytes ?? 0)) fail('LIMIT', 'Native node shader byte budget exceeded');
    const descriptor = Object.freeze({nodeProgram: true, key: `native-node-${++serial}`, vertex, fragment,
      attributes: Object.freeze(attributes), attributesKey: JSON.stringify(attributes), groups: Object.freeze(groups),
      ...(instances ? {instanceCapacity: instances.capacity} : {})});
    // Node.build() deduplicates unsized storage by attribute before addNode().
    // Walk the original child edges too, including graphs created inside Fn,
    // so changing one alias cannot silently keep the old shared binding.
    const graphNodes = new Set(builder.nodes);
    for (const node of graphNodes) {
      if (graphNodes.size > maxNodes) fail('LIMIT', 'Generated native node graph exceeds its node budget');
      if (typeof node.getChildren === 'function') for (const child of node.getChildren()) graphNodes.add(child);
    }
    // Include TextureNodes omitted by source UUID binding deduplication. Source
    // MaterialNode caches its references across materials, so read their live
    // source properties for cache/preflight checks without running node updates.
    // This also detects a map that stops sharing an authored texture binding.
    const textureReferences = new Map();
    for (const node of graphNodes) if (node.constructor === three.MaterialReferenceNode && node.node?.isTextureNode &&
        node.updateReference === three.MaterialReferenceNode.prototype.updateReference)
      textureReferences.set(node.node.getBase(), {material: node.material ?? material, properties: node.properties.slice()});
    const textureNodes = [...graphNodes].filter(node => node.isTextureNode && boundTextures.has(node.value));
    const signature = textureSignature(textureNodes, textureReferences);
    const storageNodes = [...graphNodes].filter(node => node.isStorageBufferNode);
    const storageKey = storageSignature(storageNodes);
    const uniformNodes = [...graphNodes].filter(rawUniformBuffer), uniformKey = uniformSignature(uniformNodes, builder);
    const analyticLightNodes = [...builder.nodes].filter(node => node.isAnalyticLightNode);
    const attributeNodes = [...graphNodes].filter(node => node instanceof three.BufferAttributeNode && node.attribute);
    const attributeKey = attributeSignature(attributeNodes, object);
    // Instance.js caches derived color/interleaved attributes by the original
    // attribute identity. Refuse stale cached arrays even if another renderer
    // constructed the source nodes before this preparation owner saw them.
    for (const attribute of instances?.cpu ?? []) {
      if (!uniformNodes.some(node => node.value === attribute.array) &&
          !generated.some(({attribute: value}) => (value.isInterleavedBufferAttribute ? value.data.array : value.array) === attribute.array))
        fail('CHANGED', 'Source instance construction retained a different CPU array; replace the instance attribute');
    }
    if (generated.length) generatedAttributes.register(descriptor, object.geometry, generated);
    for (const attribute of instances?.cpu ?? []) instanceArrays.set(attribute, attribute.array);
    nodeStates.set(descriptor, {owner: api, descriptor, builder, bindings, material, object, lights, lightSignature: lightKey, analyticLightNodes,
      environment, instanceSignature, skinSignature, pmremNodes, pmremSignature: pmremKey,
      textureNodes, textureReferences, textureSignature: signature, materialMapSignature: materialMapSignature(material),
      storageNodes, storageSignature: storageKey, uniformNodes, uniformSignature: uniformKey,
      attributeNodes, attributeSignature: attributeKey, geometry: object.geometry, generated: generated.length > 0});
    const entry = {key, descriptor, builder, material, object, geometry: object.geometry, bytes, pmremNodes, pmremSignature: pmremKey,
      textureNodes, textureReferences, textureSignature: signature,
      storageNodes, storageSignature: storageKey, uniformNodes, uniformSignature: uniformKey,
      attributeNodes, attributeSignature: attributeKey};
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
  function checkPMREM(state, versions = true) {
    if (pmremSignature(state.pmremNodes) !== state.pmremSignature)
      fail('CHANGED', 'Native PMREM input identity or sampling orientation changed; compile the graph again');
    if (versions) {
      assertEnvironmentVersion(state.environment);
      for (const node of state.pmremNodes) assertEnvironmentVersion(node.value);
    }
  }
  function preparePMREM(descriptors, context = {}, atSceneStart = false) {
    const states = descriptors.map(stateOf), stale = new Set();
    for (const state of states) {
      checkPMREM(state, false);
      for (const node of state.pmremNodes)
        if (environmentVersions.has(node.value) && environmentVersions.get(node.value) !== node.value.pmremVersion)
          stale.add(node.value);
    }
    if (!stale.size) return;
    if (!atSceneStart)
      fail('ENVIRONMENT_REFRESH', 'In-place PMREM regeneration cannot precede fixed scene draws without a native background prefix');
    const first = states[0], prefix = [];
    for (const node of first.builder.updateBeforeNodes) {
      // Moving an application callback ahead of an earlier draw would change
      // its visible source pixels and effects. Only the exact source prefix
      // can run here; all remaining before/current/after updates stay in place.
      if (node.constructor !== three.PMREMNode || Object.getPrototypeOf(node) !== three.PMREMNode.prototype ||
          node.updateBefore !== three.PMREMNode.prototype.updateBefore ||
          node.updateFromTexture !== three.PMREMNode.prototype.updateFromTexture ||
          node.getUpdateBeforeType !== three.Node.prototype.getUpdateBeforeType ||
          node.updateReference !== three.Node.prototype.updateReference || node.updateBeforeType !== 'render') break;
      prefix.push(node);
    }
    const covered = new Set(prefix.map(node => node.value));
    if ([...stale].some(texture => !covered.has(texture)))
      fail('ENVIRONMENT_REFRESH', 'In-place PMREM regeneration requires the first native draw to begin with its source PMREM updates');
    // These child filter renders enter the existing FIFO after earlier accepted
    // frames and before this frame. Their leases are never part of the outer
    // draw transaction, so a later authored callback refusal cannot undo them.
    withFrame(first, context, () => {
      for (const node of prefix) {
        frame.updateBeforeNode(node);
        if (environmentVersions.has(node.value)) {
          if (node._pmrem?.renderTarget?.pmremVersion !== node.value.pmremVersion)
            fail('ENVIRONMENT_REFRESH', 'The source PMREM update did not prepare the requested environment version');
          environmentVersions.set(node.value, node.value.pmremVersion);
        }
      }
    });
    for (const state of states) checkPMREM(state);
  }
  function checkLights(state) {
    if (lightSignature(state.lights) !== state.lightSignature)
      fail('CHANGED', 'Native light constructors, registrations or direct-light graphs changed; compile the graph again');
    for (const node of state.analyticLightNodes) if (!checkAnalyticLight(node, state.lights))
      fail('LIGHTS', 'Native analytic light identity changed after source construction');
  }
  function checkMaterialMaps(state) {
    if (materialMapSignature(state.material) !== state.materialMapSignature)
      fail('CHANGED', 'Native material map slots, layout, normal mode or aliases changed; compile the graph again');
  }
  function readTextures(state, liveReferences = false) {
    checkMaterialMaps(state);
    // Preparation reads source properties without executing updates. Draw
    // capture uses the values produced by the actual NodeFrame update order;
    // a later application callback may already have edited the public map.
    const references = liveReferences ? state.textureReferences : null;
    if (textureSignature(state.textureNodes, references) !== state.textureSignature)
      fail('CHANGED', 'Native node texture layout, UV channel or binding aliases changed; compile the graph again');
    const textures = [];
    for (const [group, source] of state.bindings.entries()) for (const [binding, value] of source.bindings.entries())
      if (value.isSampledTexture || value.isSampler) textures.push(Object.freeze({group, binding,
        texture: liveReferences ? textureValue(value.textureNode, references) : value.textureNode.value,
        isSampler: value.isSampler === true, isSampledTexture: value.isSampledTexture === true,
        comparison: comparison(value.textureNode, liveReferences ? textureValue(value.textureNode, references) : value.textureNode.value),
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
  function checkUniformBuffers(state) {
    if (uniformSignature(state.uniformNodes, state.builder) !== state.uniformSignature)
      fail('CHANGED', 'Native uniform array layout, group or binding aliases changed; compile the graph again');
  }
  function checkAttributes(state) {
    if (attributeSignature(state.attributeNodes, state.object) !== state.attributeSignature)
      fail('CHANGED', 'Native source attribute identity, backing array or layout changed; compile the graph again');
    if (state.generated) generatedAttributes.validate(state.descriptor);
  }
  function readUniformBuffer(binding, byteLength) {
    let history = uniformBuffers.get(binding);
    if (!history) {
      // Source createUniformBuffer allocates zeroed storage without an initial
      // upload. Even the first update may write only selected component ranges.
      history = {bytes: new Uint8Array(byteLength), version: undefined};
      uniformBuffers.set(binding, history);
    }
    if (history.bytes.byteLength !== byteLength) fail('CHANGED', 'Native uniform buffer allocation changed');
    const group = binding.groupNode;
    if (group.updateType === 'object' || history.version !== group.version) {
      binding.update();
      const array = binding.buffer, bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
      if (binding.updateRanges.length === 0) history.bytes.set(bytes);
      else for (const {start, count} of binding.updateRanges)
        history.bytes.set(bytes.subarray(start * 4, (start + count) * 4), start * 4);
      history.version = group.version;
      // Separate source bindings can refer to the same node. Consume its ranges
      // in binding order, just as Bindings._update does for the pinned backend.
      binding.clearUpdateRanges();
    }
    return history.bytes;
  }
  function checkCapture(state, context) {
    const object = context.object ?? state.object;
    checkMaterialMaps(state);
    checkLights(state);
    assertInstances(object, state.instanceSignature);
    assertSkinning(object, state.skinSignature);
    checkPMREM(state);
    // Reject a stale or newly atomic schema before observable NodeFrame updates;
    // readStorage below also catches mutations performed by those updates.
    if (state.storageNodes.length) readStorage(state);
    checkUniformBuffers(state);
    checkAttributes(state);
    return object;
  }
  function runBefore(state) {
    for (const node of state.builder.updateBeforeNodes) {
      if (node instanceof three.ShadowNode) {
        if (!state.lights.includes(node.light)) fail('SHADOW', 'A source shadow node changed its captured light identity');
        registerThreeNativeShadow(three, renderer, node);
        updateThreeNativeShadow(three, renderer, node, frame);
      } else frame.updateBeforeNode(node);
      // PMREM refresh keeps its separate ordered-prefix contract. Shadow
      // recursion runs outside draw-resource capture at its source boundary.
      checkPMREM(state);
    }
  }
  function before(descriptor, context = {}) {
    const state = stateOf(descriptor); checkCapture(state, context);
    return withFrame(state, context, () => runBefore(state));
  }
  function captureDraw(descriptor, context = {}) {
    const state = stateOf(descriptor), object = checkCapture(state, context);
    return withFrame(state, context, () => {
      if (!context.beforeDone) runBefore(state);
      assertInstances(object, state.instanceSignature);
      assertSkinning(object, state.skinSignature);
      checkAttributes(state);
      // Match Renderer._renderObjectDirect: source geometry uploads occur after
      // updateBefore and before ordinary node updates and uniform uploads.
      const attributes = state.generated ? generatedAttributes.capture(descriptor, frame.renderId) : null;
      for (const node of state.builder.updateNodes) {
        frame.updateNode(node);
        checkPMREM(state);
      }
      assertInstances(object, state.instanceSignature);
      assertSkinning(object, state.skinSignature);
      checkLights(state);
      checkPMREM(state);
      checkUniformBuffers(state);
      checkAttributes(state);
      const textures = readTextures(state);
      const storage = readStorage(state);
      const uniforms = state.bindings.map((group, index) => {
        const layout = descriptor.groups[index], packet = new Uint8Array(layout.byteLength);
        for (const member of layout.uniforms) {
          const binding = group.bindings[member.binding];
          if (binding.isNodeUniformBuffer) packet.set(readUniformBuffer(binding, member.byteLength), member.offset);
          else {
            binding.update();
            const buffer = binding.buffer;
            if (buffer.byteLength !== member.byteLength) fail('CHANGED', 'Native node uniform layout changed during its update');
            packet.set(new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength), member.offset);
            binding.clearUpdateRanges();
          }
        }
        return packet;
      });
      return {uniforms, textures, storage, attributes};
    });
  }
  function capture(descriptor, context = {}) { return captureDraw(descriptor, context).uniforms; }
  function after(descriptor, context = {}) {
    const state = stateOf(descriptor);
    return withFrame(state, context, () => { for (const node of state.builder.updateAfterNodes) frame.updateAfterNode(node); });
  }
  const api = Object.freeze({inspect, compile, preparePMREM, before, capture, captureDraw, after,
    geometry(descriptor) { const state = stateOf(descriptor); return state.generated ? generatedAttributes.geometry(descriptor) : state.geometry; },
    attributeBytes(descriptor) { return stateOf(descriptor).generated ? generatedAttributes.byteLength(descriptor) : 0; },
    captureAttributes(descriptor, context = {}) {
      const state = stateOf(descriptor); assertInstances(context.object ?? state.object, state.instanceSignature); checkAttributes(state);
      return state.generated ? generatedAttributes.capture(descriptor, context.renderId ?? renderer.info.calls) : null;
    },
    textures(descriptor) { return readTextures(stateOf(descriptor), true); },
    storage(descriptor) { return readStorage(stateOf(descriptor)); },
    release(descriptor) { for (const entry of entries) if (entry.descriptor === descriptor) { retire(entry); break; } },
    dispose() { if (disposed) return; disposed = true; for (const entry of [...entries]) retire(entry); generatedAttributes.dispose(); },
    get diagnostics() { return Object.freeze({programs: entries.size, shaderBytes, maxPrograms, maxShaderBytes}); }});
  return api;
}
