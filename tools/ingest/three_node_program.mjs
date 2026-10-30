/** Native r186 TSL graphics preparation. The source WGSLNodeBuilder and NodeFrame
 * are retained helpers; this module constructs no renderer/backend and submits
 * no GPU commands. Descriptors preserve the builder's native bind groups.
 *
 * The profile is rigid/skinned/morphed meshes, source sprites, points and lines,
 * including source Line2NodeMaterial wide-line mesh expansion and VolumeNodeMaterial
 * raymarched scattering/emission,
 * InstancedBufferGeometry and CPU/storage-backed InstancedMesh objects, unlit
 * or Toon/Lambert/Phong/Standard/Physical node materials and corresponding
 * built-in materials with authored slots, ordinary or registered direct analytic
 * lights, source float/integer attributes, scalar/vector/matrix uniforms,
 * source bone palettes, authored typed uniform arrays and multiple uniform blocks per source group,
 * sampled 2D/data-array/3D/cube textures, native cube render-target color,
 * GPU-authored 2D/array/3D storage inputs, including lossless signed/unsigned texel loads,
 * target-owned depth comparisons and read-only
 * scalar/vector/matrix or flat-struct storage buffers.
 * Physical direct lighting retains the source DFG texture, clearcoat, sheen,
 * iridescence, anisotropy, retroreflection and source PMREM environment lighting.
 * Source Fog/FogExp2 and directional/spot/point Basic/PCF/VSM shadows retain
 * native material composition, including source Point/Sun VSM fallbacks.
 * Registered SunLight cascades and LightProbeGrid irradiance retain source nodes
 * and their update lifecycle; ShadowMaterial retains source receivers. Single-output source
 * RenderPipeline graphs retain their PassNode lifecycle and output context.
 * Named MRT outputs share native attachments and retain source graph construction.
 * ClippingGroup planes retain source union/intersection and alpha-to-coverage
 * composition. Physical transmission and dispersion retain source framebuffer
 * nodes and require the renderer's ordered scene-color copy owner. Fragment
 * integer atomics share compute storage. VSM retains the source moment and blur passes.
 * capture() is a logical-draw operation, never a
 * preparation retry: it runs source before/current updates and owns the bytes.
 * after() belongs after that draw has been accepted. Callers capture before any
 * asynchronous pipeline preparation and retain the resulting byte arrays.
 */
import {createThreeNodeAttributes, inspectThreeNodeStorageAttribute} from './three_node_attributes.mjs';
import {inspectThreeNativeShadowFilter, inspectThreeNativeShadowGraph, captureThreeNativeShadowGraphs, installThreeNativeShadowTargets,
  registerThreeNativeShadow, updateThreeNativeShadow, withThreeNativeComputePrerequisites} from './three_node_shadows.mjs';
import {inspectThreePipelineContext, inspectThreeLightingContext, inspectThreeNativePass,
  updateThreeNativePass, inspectThreeMRT, inspectThreeNativeCamera} from './three_node_passes.mjs';
import {inspectThreeNodeClipping, updateThreeNodeClipping} from './three_node_clipping.mjs';
import {createThreeNodeEffects} from './three_node_effects.mjs';
import {inspectThreeNativeBatch} from './three_node_batch.mjs';
import {createThreeNodeWireframes} from './three_node_wireframe.mjs';

export class ThreeNodeProgramError extends Error {
  constructor(code, message, options) {
    super(`THREE_NODE_PROGRAM_${code}: ${message}`, options);
    this.name = 'ThreeNodeProgramError'; this.code = `THREE_NODE_PROGRAM_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeNodeProgramError(code, message); };
const SLOTS = new Set(['colorNode', 'positionNode', 'vertexNode', 'fragmentNode', 'opacityNode', 'alphaTestNode', 'outputNode', 'mrtNode', 'aoNode', 'depthNode', 'lightsNode', 'geometryNode']);
const SPRITE_SLOTS = new Set(['rotationNode', 'scaleNode']);
const POINT_SLOTS = new Set(['sizeNode']);
const DASH_SLOTS = new Set(['offsetNode', 'dashScaleNode', 'dashSizeNode', 'gapSizeNode']);
const VOLUME_FUNCTION_SLOTS = new Set(['scatteringNode', 'scatteringEmissiveNode']);
const LIT_SLOTS = new Set(['normalNode', 'emissiveNode']);
const PHONG_SLOTS = new Set(['shininessNode', 'specularNode']);
const STANDARD_SLOTS = new Set(['metalnessNode', 'roughnessNode', 'envNode']);
const PHYSICAL_SLOTS = new Set(['clearcoatNode', 'clearcoatRoughnessNode', 'clearcoatNormalNode',
  'sheenNode', 'sheenRoughnessNode', 'iridescenceNode', 'iridescenceIORNode', 'iridescenceThicknessNode',
  'anisotropyNode', 'retroreflectivityNode', 'iorNode', 'specularColorNode', 'specularIntensityNode',
  'transmissionNode', 'thicknessNode', 'attenuationDistanceNode', 'attenuationColorNode', 'dispersionNode']);
const SSS_SLOTS = new Set(['thicknessColorNode', 'thicknessDistortionNode', 'thicknessAmbientNode',
  'thicknessAttenuationNode', 'thicknessPowerNode', 'thicknessScaleNode']);
const SHADOW_SLOTS = new Set(['castShadowPositionNode', 'castShadowNode', 'maskShadowNode',
  'receivedShadowPositionNode', 'receivedShadowNode']);
const GRAPH_SLOTS = [...SLOTS, ...SPRITE_SLOTS, ...POINT_SLOTS, ...DASH_SLOTS, ...LIT_SLOTS, ...PHONG_SLOTS, ...STANDARD_SLOTS, ...PHYSICAL_SLOTS, ...SSS_SLOTS, ...SHADOW_SLOTS];
// Match WebGPUUtils: primitive type belongs to the object, not its material.
const topologyOf = (object, material) => object.isPoints === true ? 'points'
  : object.isLineSegments === true || object.isMesh === true && material.wireframe === true ? 'lines'
  : object.isLine === true ? 'line-strip' : 'triangles';
const primitiveLayout = (object, material) => {
  const topology = topologyOf(object, material), index = object.geometry.index;
  return {topology, stripIndexFormat: topology === 'line-strip' && index
    ? index.array instanceof Uint16Array ? 'uint16' : 'uint32' : null,
    wireframe: object.isMesh === true && material.wireframe === true};
};
// These source material accessors lower to ordinary 2D texture bindings. AO and
// light maps additionally retain their exact source material lighting nodes.
// Environment maps use their separate source PMREM identity and context below.
const MATERIAL_MAPS = new Set(['map', 'alphaMap', 'matcap', 'emissiveMap', 'normalMap', 'bumpMap', 'displacementMap',
  'specularMap', 'specularIntensityMap', 'specularColorMap', 'roughnessMap', 'metalnessMap', 'gradientMap',
  'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap', 'sheenColorMap', 'sheenRoughnessMap',
  'iridescenceMap', 'iridescenceThicknessMap', 'anisotropyMap', 'retroreflectivityMap', 'aoMap', 'lightMap',
  'transmissionMap', 'thicknessMap']);
const TYPES = new Set(['float', 'int', 'uint', 'bool', 'color', 'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4',
  'uvec2', 'uvec3', 'uvec4', 'bvec2', 'bvec3', 'bvec4', 'mat2', 'mat3', 'mat4']);
const FRAME_FIELDS = ['renderer', 'scene', 'camera', 'object', 'material', 'compute', 'frameId', 'renderId', 'time', 'deltaTime'];
const nodeStates = new WeakMap();

/** True only for materials with an authored graphics slot. Default built-ins
 * continue through the existing fixed shader path. Admission is separate. */
export function hasThreeNodeProgram(material) {
  return material?.isMaterial === true && Object.keys(material).some(key => key.endsWith('Node') && material[key] != null);
}

/** Built-in media maps need the same immutable queued frames as authored nodes. */
export function hasThreeNativeMedia(material) {
  if (material?.isMaterial !== true) return false;
  for (const slot of MATERIAL_MAPS)
    if (material[slot]?.isVideoTexture === true || material[slot]?.isVideoFrameTexture === true) return true;
  return false;
}

export function createThreeNodePrograms({three, renderer, maxPrograms = 4096, maxNodes = 8192,
  maxShaderBytes = 16 * 1024 * 1024, maxUniformBytes = 256 * 1024, maxAttributeBytes = 64 * 1024 * 1024, maxBindGroups = 4} = {}) {
  if (three?.REVISION !== '186' || typeof three.WGSLNodeBuilder !== 'function' ||
      typeof renderer?._nodes?.nodeFrame?.updateNode !== 'function')
    fail('SOURCE', 'Supply the pinned r186 namespace and the renderer NodeFrame');
  for (const [name, value] of Object.entries({maxPrograms, maxNodes, maxShaderBytes, maxUniformBytes, maxAttributeBytes, maxBindGroups}))
    if (!Number.isSafeInteger(value) || value < 1) fail('LIMIT', `Invalid ${name}`);
  const frame = renderer._nodes.nodeFrame, cache = new WeakMap(), entries = new Set();
  const effects = createThreeNodeEffects(three, renderer, {fail});
  const physicalPrototypes = [three.MeshPhysicalNodeMaterial.prototype,
    ...(three.MeshSSSNodeMaterial ? [three.MeshSSSNodeMaterial.prototype] : [])];
  const standardPrototypes = [three.MeshStandardNodeMaterial.prototype, ...physicalPrototypes];
  const basicEnvironmentPrototypes = [three.MeshBasicNodeMaterial.prototype, three.MeshLambertNodeMaterial.prototype,
    three.MeshPhongNodeMaterial.prototype];
  const normalPrototypes = [three.MeshNormalNodeMaterial.prototype, three.MeshMatcapNodeMaterial.prototype];
  const primitivePrototypes = [three.PointsNodeMaterial, three.LineBasicNodeMaterial, three.LineDashedNodeMaterial,
    three.Line2NodeMaterial]
    .filter(Boolean).map(Type => Type.prototype);
  const litPrototypes = [three.MeshToonNodeMaterial.prototype, three.MeshLambertNodeMaterial.prototype,
    three.MeshPhongNodeMaterial.prototype, three.ShadowNodeMaterial.prototype, ...standardPrototypes,
    ...(three.VolumeNodeMaterial ? [three.VolumeNodeMaterial.prototype] : [])];
  const lightClasses = new Map([[three.AmbientLight, three.AmbientLightNode], [three.DirectionalLight, three.DirectionalLightNode],
    [three.HemisphereLight, three.HemisphereLightNode], [three.PointLight, three.PointLightNode], [three.SpotLight, three.SpotLightNode],
    [three.LightProbe, three.LightProbeNode], [three.RectAreaLight, three.RectAreaLightNode]]);
  const lightClassIds = new WeakMap();
  let lightClassSerial = 0;
  const environmentNodes = new WeakMap(), environmentVersions = new WeakMap(), uniformBuffers = new WeakMap(), fogNodes = new WeakMap();
  const generatedAttributes = createThreeNodeAttributes(three, {maxBytes: maxAttributeBytes, fail});
  const wireframes = createThreeNodeWireframes(three, {maxBytes: maxAttributeBytes, fail});
  const bundleGeometryRecords = new Set();
  const fogResourceRecords = new Set();
  const instanceArrays = new WeakMap(), sourceIds = new WeakMap(); let sourceSerial = 0;
  const materialViews = new WeakMap();
  const sourceMaterialClasses = new Map([
    [three.MeshBasicMaterial, three.MeshBasicNodeMaterial], [three.MeshToonMaterial, three.MeshToonNodeMaterial],
    [three.MeshLambertMaterial, three.MeshLambertNodeMaterial], [three.MeshPhongMaterial, three.MeshPhongNodeMaterial],
    [three.MeshStandardMaterial, three.MeshStandardNodeMaterial], [three.MeshPhysicalMaterial, three.MeshPhysicalNodeMaterial],
    [three.MeshNormalMaterial, three.MeshNormalNodeMaterial], [three.MeshMatcapMaterial, three.MeshMatcapNodeMaterial],
    [three.ShadowMaterial, three.ShadowNodeMaterial], [three.SpriteMaterial, three.SpriteNodeMaterial],
    [three.PointsMaterial, three.PointsNodeMaterial], [three.LineBasicMaterial, three.LineBasicNodeMaterial],
    [three.LineDashedMaterial, three.LineDashedNodeMaterial]
  ].filter(([Source, Native]) => Source && Native).map(([Source, Native]) => [Source.prototype, {Source, Native}]));
  // Share immutable source text only. Each render object keeps its own builder,
  // observer and bindings, since source nodes may close over its live values.
  const shaderPairs = new Map();
  let disposed = false, shaderBytes = 0, serial = 0;
  function retainShaders(pair) {
    if (pair.references++ !== 0) return;
    let fragments = shaderPairs.get(pair.vertex);
    if (!fragments) shaderPairs.set(pair.vertex, fragments = new Map());
    fragments.set(pair.fragment, pair); shaderBytes += pair.bytes;
  }
  function releaseShaders(pair) {
    if (--pair.references !== 0) return;
    const fragments = shaderPairs.get(pair.vertex);
    fragments.delete(pair.fragment);
    if (fragments.size === 0) shaderPairs.delete(pair.vertex);
    shaderBytes -= pair.bytes;
  }
  const live = () => { if (disposed) fail('DISPOSED', 'Native node program preparation is disposed'); };
  const litMaterial = material => litPrototypes.includes(Object.getPrototypeOf(material));
  // Basic uses an empty source lighting context for AO and its PI-scaled light
  // map, while remaining independent of the scene's analytic lights.
  const lightingMaterial = material => litMaterial(material) ||
    Object.getPrototypeOf(material) === three.MeshBasicNodeMaterial.prototype;
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
    // Native setters such as SpriteNodeMaterial.sizeAttenuation can advance
    // this private view while copying. The original material owns invalidation.
    record.view.version = material.version;
    record.keys = keys;
    return record.view;
  }
  const rawUniformBuffer = node => node instanceof three.BufferNode && !node.isStorageBufferNode && !node.isArrayBufferNode;
  const managedUniformBuffer = node => node instanceof three.UniformArrayNode && node.isArrayBufferNode === true;
  const nativeUniformBuffer = node => rawUniformBuffer(node) || managedUniformBuffer(node);
  function sourceId(value) {
    if (value === null || (typeof value !== 'object' && typeof value !== 'function')) return value;
    let id = sourceIds.get(value); if (id === undefined) { id = ++sourceSerial; sourceIds.set(value, id); }
    return id;
  }
  function inspectAttributeInput(node, object) {
    if (node.value?.isStorageBufferAttribute || node.value?.isStorageInstancedBufferAttribute) {
      inspectThreeNodeStorageAttribute(three, node.value, fail);
      if (node.attribute !== node.value || node.bufferOffset !== 0 || node.bufferStride !== 0)
        fail('ATTRIBUTE', 'Native storage vertex inputs require the source scalar/vector attribute without interleaved conversion');
      return;
    }
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
      if (owner?.isStorageBufferAttribute || owner?.isStorageInstancedBufferAttribute) {
        const shape = inspectThreeNodeStorageAttribute(three, owner, fail);
        return [node.id, node.nodeType, node.bufferType, node.bufferStride, node.bufferOffset, node.instanced, node.usage,
          sourceId(node.value), sourceId(attribute), shape.signature];
      }
      if (!attribute || !array || inputArray !== array)
        fail('CHANGED', 'Native buffer attribute nodes must retain their source backing array; replace the node or instance attribute');
      return [node.id, node.nodeType, node.bufferType, node.bufferStride, node.bufferOffset, node.instanced, node.usage,
        sourceId(node.value), sourceId(attribute), sourceId(owner), sourceId(array), array.length, array.byteLength,
        attribute.count, attribute.itemSize, attribute.normalized, attribute.gpuType, attribute.isInterleavedBufferAttribute,
        attribute.offset, attribute.isInstancedBufferAttribute, owner.stride, owner.meshPerAttribute,
        owner.isInstancedInterleavedBuffer, owner.usage, sourceId(owner.onUploadCallback)];
    }));
  }
  function inspectUniformArray(node) {
    const array = node.array, type = node.elementType;
    if (!managedUniformBuffer(node) ||
        !(Array.isArray(array) || ArrayBuffer.isView(array) && !(array instanceof DataView)) ||
        !Number.isSafeInteger(array.length) || array.length < 1 || array.length > maxNodes ||
        !/^(?:float|int|uint|color|[iu]?vec[234]|mat[234])$/.test(type ?? '') ||
        node.paddedType !== three.UniformArrayNode.prototype.getPaddedType.call(node))
      fail('UNIFORM', 'Native managed uniform arrays require bounded source scalar/vector/color/matrix inputs and their pinned padded type');
    const bytes = array.length * (node.paddedType === 'mat4' ? 64 : 16);
    if (bytes > (renderer._device?.limits?.maxUniformBufferBindingSize ?? 65536) || bytes > maxUniformBytes)
      fail('LIMIT', 'Native managed uniform array exceeds its binding or draw byte budget');
  }
  function inspectUniformBuffer(node) {
    // BufferNode emits the caller's array layout verbatim. Unlike UniformArrayNode,
    // it inserts no per-element padding: scalar/vec2 arrays therefore do not have
    // a legal uniform-address-space stride in the pinned WGSL output.
    const managed = managedUniformBuffer(node);
    if (managed) inspectUniformArray(node);
    const type = /^(?:(i|u)?vec([34])|mat([234])|color)$/.exec((managed ? node.bufferType : node.nodeType) ?? '');
    const matrix = Number(type?.[3] ?? 0), stride = matrix === 3 ? 12 : matrix ? matrix * matrix : 4;
    const tail = matrix === 3 ? 11 : type?.[2] === '3' || node.nodeType === 'color' ? 3 : stride;
    const Scalar = type?.[1] === 'i' ? Int32Array : type?.[1] === 'u' ? Uint32Array : Float32Array;
    const array = node.value, required = (node.bufferCount - 1) * stride + tail;
    if (!type || !nativeUniformBuffer(node) || !(array instanceof Scalar) ||
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
        node.name, group?.id, group?.name, group?.order, group?.shared, group?.updateType,
        managedUniformBuffer(node) ? [node.elementType, node.paddedType] : null];
    }));
  }
  function inspectStorage(node) {
    const attribute = node.value, struct = node.structTypeNode;
    const atomic = node.isAtomic === true;
    if (atomic && (struct || !['int', 'uint'].includes(node.nodeType)))
      fail('STORAGE', 'Native fragment atomics require scalar signed or unsigned 32-bit storage');
    const type = /^(?:(i|u)?vec([234])|(float|int|uint|mat[234]))$/.exec(node.nodeType ?? '');
    const matrix = /^mat([234])$/.exec(node.nodeType ?? '')?.[1];
    const components = matrix ? Number(matrix) ** 2 : type?.[2] ? Number(type[2]) : 1;
    const scalar = type?.[1] === 'i' || type?.[3] === 'int' ? Int32Array
      : type?.[1] === 'u' || type?.[3] === 'uint' ? Uint32Array : Float32Array;
    // Storage matrices use aligned column vectors: mat2 is four words, mat3
    // is twelve (three padded vec3 columns), and mat4 is sixteen. The source
    // attribute uploader only pads itemSize=3, not a packed nine-word mat3;
    // require the authored matrix stride rather than rewriting its CPU bytes.
    let stride = matrix === '3' ? 12 : components === 3 ? 4 : components;
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
    // PBO is a WebGL-fallback hint. The pinned StorageArrayElementNode only
    // observes it when storageBuffer is unavailable; native WGSL keeps the
    // ordinary shared storage allocation, including for authored PBO nodes.
    if ((!type && !struct) || node.isAtomic && !atomic ||
        !['readOnly', 'readWrite'].includes(node.access) || attribute?.isIndirectStorageBufferAttribute ||
        !(attribute?.isStorageBufferAttribute || attribute?.isStorageInstancedBufferAttribute) ||
        attribute.isInterleavedBufferAttribute || attribute.normalized ||
        !(struct ? [Float32Array, Int32Array, Uint32Array].some(Type => attribute.array instanceof Type) : attribute.array instanceof scalar) ||
        !Number.isSafeInteger(attribute.count) || attribute.count < 1 ||
        !(struct || matrix ? attribute.itemSize === stride : [components, stride].includes(attribute.itemSize)) ||
        attribute.array.length !== attribute.count * attribute.itemSize ||
        !Number.isSafeInteger(node.bufferCount) || node.bufferCount < 0 || node.bufferCount > attribute.count)
      fail('STORAGE', 'Native storage requires matching scalar/vector/matrix or flat-struct 32-bit attributes; mat3 columns require padding and atomics require scalar integers');
    if (attribute.onUploadCallback !== three.BufferAttribute.prototype.onUploadCallback)
      fail('HOOK', 'Native shader storage requires the source upload callback');
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
        node.isAtomic === true, node.access,
        struct ? [struct.name, struct.membersLayout.map(({name, type}) => [name, type])] : null];
    }));
  }
  function inspectInstances(object) {
    if (!object.isInstancedMesh) return null;
    if (!(object instanceof three.InstancedMesh) || object.isSkinnedMesh || object.isBatchedMesh ||
        !(object.geometry instanceof three.BufferGeometry) || object.geometry.isInstancedBufferGeometry ||
        object.morphTexture != null && !object.geometry.morphAttributes.position?.length)
      fail('OBJECT', 'Native instancing requires source InstancedMesh objects, with position targets for per-instance morph textures');
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
  function inspectMorphs(object) {
    const geometry = object.geometry, attributes = geometry?.morphAttributes;
    const position = attributes?.position, normal = attributes?.normal, color = attributes?.color;
    if (![position, normal, color].some(list => list?.length)) return null;
    const count = position?.length, vertices = geometry.attributes.position?.count;
    const influences = object.morphTargetInfluences;
    // Match Morph.js's construction branch. Its instance texture contains each
    // row's weights, while the source base multiplier remains an object uniform.
    // Absolute morphs therefore still require the source influence array; do
    // not invent a different base calculation from the texture's first texel.
    const texture = object.count > 1 && object.morphTexture != null ? object.morphTexture : null;
    const influenceArray = Array.isArray(influences) || ArrayBuffer.isView(influences) && !(influences instanceof DataView);
    if (object.isBatchedMesh || !Array.isArray(position) ||
        !Number.isSafeInteger(count) || count < 1 || !Number.isSafeInteger(vertices) || vertices < 1 ||
        ((!texture || !geometry.morphTargetsRelative || influences != null) && (!influenceArray || influences.length !== count)))
      fail('OBJECT', 'Native morphs require source position targets and matching influence arrays for the source uniform/base path');
    if (texture) {
      if (!(texture instanceof three.DataTexture) || texture.isDataTexture !== true)
        fail('OBJECT', 'Native per-instance morph weights require a source DataTexture');
      inspectTexture(texture);
    }
    if (count > (renderer._device?.limits?.maxTextureArrayLayers ?? 256))
      fail('LIMIT', 'Native morph targets exceed the device texture layer limit');
    for (const [name, list] of [['position', position], ['normal', normal], ['color', color]]) {
      if (list === undefined) continue;
      if (!Array.isArray(list) || list.length !== count || list.some(attribute =>
        !(attribute instanceof three.BufferAttribute || attribute instanceof three.InterleavedBufferAttribute) ||
        attribute.count !== vertices || ![3, ...(name === 'color' ? [4] : [])].includes(attribute.itemSize)))
        fail('OBJECT', 'Native morph targets require matching source attribute counts and vector layouts');
    }
    // The pinned Morph.js owns packing and caches by geometry/count. Influence
    // values and relative mode stay live; attribute edits do not silently
    // regenerate a source texture that remains cached until geometry disposal.
    return JSON.stringify([count, vertices, normal !== undefined && geometry.hasAttribute('normal'), color !== undefined,
      texture ? sourceId(texture) : null]);
  }
  function assertMorphs(object, signature) {
    if (inspectMorphs(object) !== signature)
      fail('CHANGED', 'Native morph target layout changed; compile the graph again');
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
    // SunLight contributes through setupDirect; LightProbeGrid contributes
    // irradiance from its atlas through setup and updates its own uniforms.
    // Both use the source cached AnalyticLightNode and its FRAME scheduling.
    const prototype = Native?.prototype;
    const direct = prototype && Object.getOwnPropertyDescriptor(prototype, 'setupDirect');
    const grid = light.isLightProbeGrid === true;
    const methods = grid ? ['constructor', 'setup', 'update'] : ['constructor', 'setupDirect', 'setupShadowNode'];
    if (renderer.library.getLightNodeClass !== three.StandardNodeLibrary.prototype.getLightNodeClass ||
        Object.getPrototypeOf(Source.prototype) !== three.Light.prototype ||
        !prototype || Object.getPrototypeOf(prototype) !== three.AnalyticLightNode.prototype ||
        prototype.constructor !== Native || (grid
          ? ['setup', 'update'].some(key => typeof Object.getOwnPropertyDescriptor(prototype, key)?.value !== 'function')
          : typeof direct?.value !== 'function') ||
        Object.getOwnPropertyNames(prototype).some(key => !methods.includes(key)))
      fail('LIGHTS', 'Registered native lights require a direct analytic node with the pinned source lifecycle');
    if (grid && (!(light.boundingBox instanceof three.Box3) || !(light.resolution instanceof three.Vector3) ||
        light.texture !== null && (inspectTexture(light.texture).viewDimension !== '3d' || integerTextureType(light.texture))))
      fail('LIGHTS', 'Native probe grids require source bounds, resolution and a floating-point 3D SH atlas');
    return Native;
  }
  function lightSignature(lights) {
    const id = value => {
      if (!lightClassIds.has(value)) lightClassIds.set(value, ++lightClassSerial);
      return lightClassIds.get(value);
    };
    return JSON.stringify(lights.map(light => {
      const Native = sourceLightClass(light);
      // SpotLightNode calls its authored color with projected coordinates;
      // other analytic nodes consume an authored color graph directly. Keep
      // those source entrances distinct and let the pinned builder call Fn.
      const spot = Native === three.SpotLightNode, color = light.colorNode || null;
      if (light.map != null && !spot)
        fail('LIGHTS', 'Native projected maps require the source SpotLightNode');
      if (color && (spot ? typeof color !== 'function' || typeof color.getCacheKey !== 'function' : !(color instanceof three.Node)))
        fail('LIGHTS', 'Native spot color nodes require a callable projection with the source cache key; other light colors require source node graphs');
      const map = spot && !color ? light.map || null : null;
      if (map && (inspectTexture(map).viewDimension !== '2d' || integerTextureType(map)))
        fail('LIGHTS', 'Projected spot maps require floating-point source 2D textures');
      // A projected map is a render-update TextureNode inside the source
      // lighting graph. Rebuild on replacement so layout and binding aliases
      // are checked before that callback selects the new resource.
      const projected = map ? [sourceId(map), textureSignature([{value: map}])] : null;
      const authored = color ? [sourceId(color), color.getCacheKey()] : null;
      const shadow = light.castShadow && renderer.shadowMap?.enabled === true ? light.shadow : null;
      const filter = shadow ? inspectThreeNativeShadowFilter(shadow) : null;
      const shadowGraph = shadow ? inspectThreeNativeShadowGraph(three, shadow.shadowNode, light) : null;
      if (shadow && (!(shadow instanceof three.LightShadow) ||
          ![three.BasicShadowMap, three.PCFShadowMap, three.VSMShadowMap].includes(renderer.shadowMap.type) ||
          ![three.DirectionalLight, three.SpotLight, three.PointLight].includes(light.constructor) &&
            !(light.isSunLight === true && shadow.isSunLightShadow === true)))
        fail('SHADOW', 'Native source shadows require ordinary directional, spot, point or registered SunLight Basic/PCF/VSM maps');
      // The grid's setup closes over the atlas identity and its falloff branch.
      // Bounds, resolution, positive falloff and intensity remain live uniforms.
      const grid = light.isLightProbeGrid === true ? [light.texture ?
        [sourceId(light.texture), textureSignature([{value: light.texture}])] : null, light.falloff > 0] : null;
      return [light.id, id(light.constructor), id(Native), id(Native.prototype.setup), id(Native.prototype.update), id(Native.prototype.setupDirect),
        id(Native.prototype.setupDirectRectArea), id(Native.prototype.setupShadowNode), light.castShadow === true, shadow ? id(shadow) : null,
        renderer.shadowMap?.enabled === true, renderer.shadowMap?.type, renderer.shadowMap?.transmitted === true,
        filter ? [sourceId(filter), filter.isFn === true ? filter.getCacheKey() : null] : null,
        shadowGraph ? [sourceId(shadowGraph), shadowGraph.getCacheKey()] : null, projected, authored, grid];
    }));
  }
  function checkAnalyticLight(node, lights) {
    if (!node.isAnalyticLightNode || !lights.includes(node.light)) return false;
    const Native = sourceLightClass(node.light);
    if (Object.getPrototypeOf(node) !== Native.prototype) return false;
    if (!lightClasses.has(node.light.constructor)) {
      const source = three.AnalyticLightNode.prototype;
      const grid = node.light.isLightProbeGrid === true;
      if (node.updateType !== 'frame' || node.updateBeforeType !== 'none' || node.updateAfterType !== 'none' ||
          node.setupDirect !== Native.prototype.setupDirect ||
          ['build', 'setup', 'analyze', 'generate', 'getHash', 'updateReference', 'getUpdateType',
            'getUpdateBeforeType', 'getUpdateAfterType', 'update', 'updateBefore', 'updateAfter']
            .some(key => node[key] !== (grid && (key === 'setup' || key === 'update') ? Native.prototype[key] : source[key])))
        fail('LIGHTS', 'Registered analytic nodes must retain the pinned source setup and update lifecycle');
    }
    return true;
  }
  function authoredLightsNode(material) {
    if (material.lightsNode == null || material.fragmentNode != null || !lightingMaterial(material)) return null;
    const node = material.lightsNode;
    if (!(node instanceof three.LightsNode) || Object.getPrototypeOf(node) !== three.LightsNode.prototype ||
        ['getScope', 'getLights', 'getBuiltinLights', 'setup', 'setupLightsNode', 'setupDirectLight', 'setupDirectRectAreaLight']
          .some(key => node[key] !== three.LightsNode.prototype[key]))
      fail('GRAPH', 'Authored material lighting requires the pinned LightsNode composition');
    const inputs = node.getBuiltinLights();
    if (!Array.isArray(inputs) || inputs.some(input => !(input instanceof three.Light || input instanceof three.Node)))
      fail('GRAPH', 'Authored lighting lists require source lights or source node contributors');
    if (inputs.length > maxNodes) fail('LIMIT', 'Authored light inputs exceed the native node budget');
    return node;
  }
  function authoredLightings(node) {
    return node ? node.getBuiltinLights().filter(input => input instanceof three.Node) : [];
  }
  function lightingSignature(material, lights) {
    const node = authoredLightsNode(material);
    // LightsNode keeps its list in a private field, outside Node.getChildren().
    // Capture the selected identities explicitly; contributor uniforms remain
    // live data while their source graph keys describe structural changes.
    return JSON.stringify([lightSignature(lights), node ? [sourceId(node), node.getCacheKey(),
      node.getBuiltinLights().map(input => input.isNode === true
        ? [sourceId(input), input.getCacheKey()] : input.id)] : null]);
  }
  function sourceLights(material, {lights = [], lightsNode = null, lightingEnabled = renderer.lighting?.enabled !== false} = {}) {
    const authored = authoredLightsNode(material);
    if (!authored && ((!litMaterial(material) && !(lightsNode && lightingMaterial(material))) ||
        material.lights !== true || material.fragmentNode != null)) return [];
    if (typeof lightingEnabled !== 'boolean' || !lightsNode && lightingEnabled !== (renderer.lighting?.enabled !== false))
      fail('LIGHTS', 'Captured lighting enablement must match the source renderer during construction');
    // NodeMaterial gives an explicit lightsNode precedence even when the scene
    // lighting manager or the material's ordinary light collection is disabled.
    if (authored) lights = authored.getBuiltinLights().flatMap(input => input instanceof three.Light
      ? [input] : input.isAnalyticLightNode === true ? [input.light] : []);
    else if (!lightingEnabled && !lightsNode) return [];
    if (!Array.isArray(lights)) fail('LIGHTS', 'Supply the selected source scene lights');
    // The source forward graph includes every selected light. Its generated
    // node, shader and device-binding budgets below bound the actual work;
    // the fixed shader's eight-light storage layout is not a native limit.
    if (lights.length > maxNodes) fail('LIMIT', 'Selected scene lights exceed the native node budget');
    lightSignature(lights);
    // The pinned LightsNode retains analytic nodes in its module WeakMap by the
    // real light identity. Never clone lights or instantiate per-program light
    // nodes: that would repeat FRAME updates across objects and recompiles.
    return lights.slice().sort((a, b) => a.id - b.id);
  }
  const basicEnvironmentMaterial = material => basicEnvironmentPrototypes.includes(Object.getPrototypeOf(material));
  const environmentEnabled = material => (standardPrototypes.includes(Object.getPrototypeOf(material)) || basicEnvironmentMaterial(material)) &&
    material.lights === true && material.fragmentNode == null && renderer.lighting?.enabled !== false;
  const environmentGraph = (material, {scene} = {}) => environmentEnabled(material)
    ? material.envNode || (!basicEnvironmentMaterial(material) && !material.envMap && scene?.environmentNode?.isNode ? scene.environmentNode : null) : null;
  function basicEnvironmentSignature(material, environment) {
    // BasicLightingModel chooses combine during construction; CubeTextureNode
    // chooses reflection/refraction from its mapping. Both remain structural
    // even when source callbacks mutate the same material or texture identity.
    return basicEnvironmentMaterial(material) && environmentEnabled(material) && (material.envNode || environment)
      ? JSON.stringify([material.combine, environment?.mapping ?? null]) : null;
  }
  function sourceBasicEnvironment(texture) {
    const cube = texture?.isCubeTexture === true;
    if (!(texture instanceof three.Texture) || texture.isF3DSceneEnvironment || texture.isF3DPMREMSource ||
        texture.isStorageTexture || texture.isDepthTexture || texture.isDataArrayTexture || texture.isData3DTexture ||
        !(cube ? [three.CubeReflectionMapping, three.CubeRefractionMapping]
          : [three.EquirectangularReflectionMapping, three.EquirectangularRefractionMapping]).includes(texture.mapping) ||
        !cube && !(texture.image?.height > 0))
      fail('ENVIRONMENT', 'Native Basic/Lambert/Phong environments require ready source cube or equirectangular reflection/refraction textures');
    inspectTexture(texture);
    return texture;
  }
  function environmentGraphKey(material, context) {
    const node = environmentGraph(material, context);
    return node ? JSON.stringify([sourceId(node), node.getCacheKey()]) : null;
  }
  const materialEnvironment = (material, texture) => texture !== null && !material.envNode && material.envMap === texture;
  function sourceEnvironment(material, {scene} = {}) {
    if (!environmentEnabled(material)) return null;
    const node = environmentGraph(material, {scene}), basic = basicEnvironmentMaterial(material);
    const inspectEnvironment = basic ? sourceBasicEnvironment : sourcePMREMTexture;
    // The source environment nodes resolve material/texture inputs while
    // constructing PMREM or cube conversion nodes. A changed source identity
    // needs fresh setup even if its authored node object remains unchanged.
    if (node?.isTextureNode) return inspectEnvironment(node.value);
    if (node?.isMaterialReferenceNode) return inspectEnvironment(material[node.property]);
    if (node) {
      if (basic) fail('ENVIRONMENT', 'Pinned BasicEnvironmentNode requires a texture or material-reference envNode');
      return null;
    }
    // A material environment overrides the scene, including its custom node.
    // Keep the source material reference in the builder so the pinned accessors
    // select envMapIntensity/envMapRotation at each logical draw.
    // Basic/Lambert/Phong have material-local environments only. Unlike PBR,
    // their source setup never substitutes the scene's environment or node.
    const texture = material.envMap || (!basic && scene?.environment) || null;
    return texture === null ? null : inspectEnvironment(texture);
  }
  function sourcePMREMTexture(texture) {
    if (!(texture instanceof three.Texture) || texture.isF3DSceneEnvironment || texture.isF3DPMREMSource || texture.isStorageTexture ||
        ![three.EquirectangularReflectionMapping, three.EquirectangularRefractionMapping, three.CubeUVReflectionMapping,
          three.CubeReflectionMapping, three.CubeRefractionMapping].includes(texture.mapping))
      fail('ENVIRONMENT', 'Native PBR environments require a ready equirectangular image, data or decoded cube faces, or source CubeUV texture');
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
      const inputSize = texture.isCubeTexture
        ? texture.image[0].width || texture.image[0].image.width : texture.image.width / 4;
      const size = 2 ** Math.floor(Math.log2(inputSize));
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
  const integerTextureType = texture => texture?.isDepthTexture === true ? null : three.IntType !== undefined && texture?.type === three.IntType ? 'sint'
    : three.UnsignedIntType !== undefined && texture?.type === three.UnsignedIntType ? 'uint' : null;
  function inspectIntegerRead(node, texture = node.value) {
    if (!integerTextureType(texture)) return;
    // Integer IDs and bitfields must stay integer shader values. The source
    // textureLoad/texture3DLoad factories select this sampler-free path; no
    // normalized coordinates, interpolation or implicit float conversion.
    if (node.sampler !== false || node.compareNode != null || node.isSampleCompare?.() === true ||
        node.biasNode != null || node.gradNode != null || node.gatherNode != null)
      fail('TEXTURE', 'Native integer graphics inputs require sampler-free texel loads without comparison, bias, gradients or gather');
  }
  function storageTextureLayout(node, texture = node.value, mipLevel = 0) {
    if (typeof three.StorageTextureNode !== 'function' || !(node instanceof three.StorageTextureNode) ||
        node.storeNode !== null || texture?.isStorageTexture !== true ||
        node.compareNode != null || node.isSampleCompare?.() === true || node.biasNode != null ||
        node.gradNode != null || node.gatherNode != null)
      fail('TEXTURE', 'Native graphics storage textures require source read-only texel loads without stores, comparison, bias, gradients or gather');
    // r186 getNodeAccess always selects readOnly for non-atomic graphics
    // storage. Keep its actual storage view, including the compiled mip, rather
    // than replacing it with a sampled texture or inventing a sampler.
    const layout = renderer._getSampledTextureInputs?.().layoutOf({texture, isSampledTexture: true,
      store: true, access: 'readOnly', mipLevel});
    if (layout?.storageTexture?.access !== 'read-only')
      fail('TEXTURE', 'Native storage texel loads require the shared renderer storage texture owner');
    return layout.storageTexture;
  }
  function inspectTexture(texture) {
    const storage = texture?.isStorageTexture === true;
    const framebuffer = texture?.isFramebufferTexture === true;
    const media = texture?.isVideoTexture === true;
    const target = texture?.isRenderTargetTexture === true || texture?.isDepthTexture === true && texture.renderTarget != null;
    const targetLayout = target && texture instanceof three.Texture
      ? renderer._getSampledTextureInputs?.().layoutOf({texture, isSampledTexture: true, isSampler: true, store: false}) : null;
    // The shared target owner initializes attached depth dimensions before
    // WGSLNodeBuilder selects its texture type and implicit array coordinates.
    const compressed = texture?.isCompressedTexture === true ||
      texture?.isCubeTexture === true && texture.image?.[0]?.isCompressedTexture === true;
    const compressedArray = texture?.isCompressedArrayTexture === true;
    const array = texture?.isDataArrayTexture === true || compressedArray || (storage || target) && texture.isArrayTexture === true;
    const dataVolume = texture?.isData3DTexture === true && !storage && !target;
    const targetVolume = target && targetLayout?.texture?.viewDimension === '3d';
    const volume = dataVolume || targetVolume || storage && texture.is3DTexture === true;
    if (!(texture instanceof three.Texture) ||
        texture.is3DTexture && !volume || volume && (array || texture.isCubeTexture) ||
        texture.isArrayTexture && !array || texture.isData3DTexture && !dataVolume && !targetVolume ||
        (dataVolume || targetVolume) && (typeof three.Data3DTexture !== 'function' || !(texture instanceof three.Data3DTexture)) ||
        array && !storage && !target && !(compressedArray
          ? typeof three.CompressedArrayTexture === 'function' && texture instanceof three.CompressedArrayTexture
          : texture instanceof three.DataArrayTexture) ||
        texture.isVideoFrameTexture && !media ||
        texture.isExternalTexture)
      fail('TEXTURE', 'Native graphics nodes sample ordinary or compressed 2D/array/cube textures, data volumes, 2D/array/3D storage and target-owned color/depth textures');
    // Compute owns these images. Admission must use the same residency owner,
    // never the CPU texture pool or an independently inferred storage layout.
    // TextureNode/Texture3DNode sampling remains read-only. Volume coordinates
    // come from the pinned Texture3DNode, never an implicit array-layer adapter.
    if (storage && (texture.isDepthTexture || typeof renderer._getSampledTextureInputs !== 'function'))
      fail('TEXTURE', 'Native storage sampling requires color and the shared renderer texture owner');
    if (framebuffer && (!(texture instanceof three.FramebufferTexture) ||
        Object.getPrototypeOf(texture) !== three.FramebufferTexture.prototype || texture.isDepthTexture ||
        texture.isCubeTexture || array || volume || typeof renderer._getSampledTextureInputs !== 'function'))
      fail('TEXTURE', 'Native framebuffer sampling requires a source color FramebufferTexture and the shared renderer texture owner');
    // Source Textures.updateTexture selects this type from the public target;
    // the GPU owner separately selects the actual framebuffer format (including
    // the canvas output intermediate). Normalize before freezing the layout so
    // the first ordered copy does not appear to mutate a compiled texture type.
    if (framebuffer) {
      const inputs = renderer._getSampledTextureInputs();
      if (typeof inputs?.prepareFramebuffer !== 'function')
        fail('TEXTURE', 'Framebuffer inputs require the shared versioned renderer texture owner');
      inputs.prepareFramebuffer(texture, renderer.getRenderTarget()?.texture.type ?? three.UnsignedByteType);
    }
    const depth = texture.isDepthTexture === true;
    if (depth && (!(texture instanceof three.DepthTexture) || texture.renderTarget != null &&
        (!(texture.renderTarget instanceof three.RenderTarget) || texture.renderTarget.depthTexture !== texture) ||
        texture.isCubeTexture && (!(texture instanceof three.CubeDepthTexture) || texture.isCubeDepthTexture !== true ||
          texture.renderTarget == null)))
      fail('TEXTURE', 'Native depth sampling requires a source DepthTexture or attached CubeDepthTexture');
    const viewDimension = volume ? '3d' : texture.isCubeTexture === true ? 'cube' : array ? '2d-array' : '2d';
    if (!depth && viewDimension === 'cube' && (!(texture instanceof three.CubeTexture) &&
        !(compressed && typeof three.CompressedCubeTexture === 'function' && texture instanceof three.CompressedCubeTexture) ||
        ![three.CubeReflectionMapping, three.CubeRefractionMapping].includes(texture.mapping) ||
        !Array.isArray(texture.image) || texture.image.length !== 6 || (target
          ? typeof renderer._getSampledTextureInputs !== 'function' : !compressed && texture.image.some(face =>
          !(face instanceof three.DataTexture && face.isDataTexture === true &&
            ArrayBuffer.isView(face.image?.data) && !(face.image.data instanceof DataView)) &&
          !(typeof renderer._getSampledTextureInputs === 'function' &&
            ['ImageData', 'ImageBitmap', 'HTMLImageElement', 'HTMLCanvasElement', 'OffscreenCanvas'].some(name =>
              typeof globalThis[name] === 'function' && face instanceof globalThis[name]))))))
      fail('TEXTURE', 'Native cube sampling requires six source data/image/compressed faces or a registered native cube target with reflection or refraction mapping');
    const integerType = integerTextureType(texture);
    const integerData = integerType && (dataVolume || typeof three.DataTexture === 'function' &&
      texture instanceof three.DataTexture && texture.isDataTexture === true && !volume) &&
      !array && !texture.isCubeTexture && !target;
    if (integerType && ((!storage && !integerData) || texture.normalized === true || texture.colorSpace !== three.NoColorSpace))
      fail('TEXTURE', 'Native integer graphics inputs require unnormalized storage, DataTexture or Data3DTexture inputs with NoColorSpace');
    const packedTarget = target && three.UnsignedInt101111Type !== undefined &&
      texture.type === three.UnsignedInt101111Type && texture.format === three.RGBFormat;
    if (!depth && !integerType && (!packedTarget && ![three.UnsignedByteType, three.HalfFloatType, three.FloatType].includes(texture.type) ||
        [three.RedIntegerFormat, three.RGIntegerFormat, three.RGBAIntegerFormat, three.RGBIntegerFormat].includes(texture.format)))
      fail('TEXTURE', 'Native graphics texture formats must have floating-point samples');
    if (depth && texture.compareFunction !== null && typeof renderer.hasCompatibility !== 'function')
      fail('SOURCE', 'Native depth comparison requires source renderer compatibility metadata');
    const compressedLayout = compressed
      ? renderer._getSampledTextureInputs?.().layoutOf({texture, isSampledTexture: true, isSampler: true, store: false}) : null;
    if (compressed && !compressedLayout?.texture)
      fail('TEXTURE', 'Missing shared GPU texture sampling layout');
    // Compressed block formats (including BC6H HDR) have filterable float
    // samples independently of the source's upload type. The shared owner
    // validates source classes, mip blocks, color space and device features.
    const sampleType = compressed ? compressedLayout.texture.sampleType : integerType ?? (depth ? 'depth' : texture.type === three.FloatType && !renderer.hasFeature('float32-filterable')
      ? 'unfilterable-float' : 'float');
    if (sampleType === 'unfilterable-float' &&
        (texture.magFilter !== three.NearestFilter ||
         ![three.NearestFilter, three.NearestMipmapNearestFilter].includes(texture.minFilter)))
      fail('TEXTURE', 'Native float32 texture filtering requires the float32-filterable feature; nearest texel reads remain available');
    if (typeof renderer.backend?.utils?.getTextureSampleData !== 'function')
      fail('SOURCE', 'The native renderer must provide source texture sample metadata');
    const {primarySamples} = renderer.backend.utils.getTextureSampleData(texture), multisampled = primarySamples > 1;
    if (primarySamples !== 1 && !(primarySamples === 4 && depth && viewDimension === '2d'))
      fail('TEXTURE', 'Native graphics require resolved color or one/four-sample 2D depth views');
    const layout = targetLayout ?? compressedLayout ?? renderer._getSampledTextureInputs?.().layoutOf({texture, isSampledTexture: true, isSampler: true, store: false});
    if ((media || storage || dataVolume || integerData || framebuffer || target && ['cube', '2d-array', '3d'].includes(viewDimension)) && !layout?.texture)
      fail('TEXTURE', 'Missing shared GPU texture sampling layout');
    if (layout && (layout.texture?.sampleType !== sampleType || layout.texture.viewDimension !== viewDimension ||
        (layout.texture.multisampled === true) !== multisampled))
      fail('TEXTURE', 'Native graphics texture bindings must match the source dimension and sample count');
    return layout?.texture ?? {sampleType, viewDimension, multisampled};
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
    const aliases = new Map(), checked = new Map();
    return JSON.stringify(nodes.map(node => {
      const texture = references ? textureValue(node, references) : node.value;
      if (!aliases.has(texture.uuid)) aliases.set(texture.uuid, aliases.size);
      if (node.isStorageTextureNode) {
        const layout = storageTextureLayout(node, texture);
        // The source binding freezes mipLevel during shader construction.
        // A later setMipLevel alone does not rebind it; material invalidation
        // or a new graph selects a new view. Exact formats remain structural
        // because storage texture declarations include their format in WGSL.
        return [aliases.get(texture.uuid), 'storageTexture', layout.format, layout.viewDimension,
          texture.channel, node.sampler, node.updateMatrix, node.groupNode?.name];
      }
      inspectIntegerRead(node, texture);
      if (!checked.has(texture)) checked.set(texture, inspectTexture(texture));
      // The pinned builder aliases uniforms by texture UUID. Keep that alias
      // partition, not the identities themselves, in the structural signature:
      // replacing one compatible sampled image must not rebuild the shader.
      // Validated filterable colors share the source f32 sample/filter shader
      // and native binding layout, including FloatType when the device enables
      // float32-filterable. Effects choose precision in updateBefore, after
      // consumer construction. Keep unfilterable, integer and depth types
      // separate; the image owner still validates exact formats/generations.
      // FloatType compressed sources still select the pinned builder's manual
      // filtering path without float32-filterable, despite a float GPU layout.
      const type = checked.get(texture).sampleType === 'float' &&
        (texture.type !== three.FloatType || renderer.hasFeature('float32-filterable')) ? 'float-color' : texture.type;
      return [aliases.get(texture.uuid), type, texture.format, texture.colorSpace, texture.channel,
        texture.isRenderTargetTexture === true, texture.isFramebufferTexture === true, texture.normalized === true, texture.isCubeTexture === true,
        texture.isDataArrayTexture === true, texture.isStorageTexture === true, texture.isArrayTexture === true,
        texture.isCompressedTexture === true, texture.isCompressedArrayTexture === true, texture.isCompressedCubeTexture === true,
        texture.isVideoTexture === true, texture.isVideoFrameTexture === true,
        texture.is3DTexture === true, texture.isData3DTexture === true, node.isTexture3DNode === true,
        texture.is3DTexture === true || texture.isData3DTexture === true ? [texture.wrapS, texture.wrapT, texture.wrapR] : null,
        texture.isCubeTexture === true ? texture.mapping : null,
        renderer.backend.utils.getTextureSampleData(texture).primarySamples,
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
  // Sprite shader construction branches on attenuation and the presence of
  // an anchor reference. Values of center/rotation/scale stay live uniforms.
  function spriteSignature(material, object) {
    if (Object.getPrototypeOf(material) === three.PointsNodeMaterial?.prototype)
      return JSON.stringify([material.sizeAttenuation, object.isPoints === true]);
    return Object.getPrototypeOf(material) === three.SpriteNodeMaterial.prototype
      ? JSON.stringify([material.sizeAttenuation, object.center?.isVector2 === true]) : null;
  }
  function wideLineSignature(material) {
    if (Object.getPrototypeOf(material) !== three.Line2NodeMaterial?.prototype) return null;
    // Width, dash sizes and offset values are live uniforms. These controls
    // instead change expansion, attributes, fragment coverage or framebuffer
    // dependencies, so an already captured program cannot accept a new shape.
    return JSON.stringify([material.worldUnits, material.dashed, material.transparent, material.vertexColors,
      ...[...DASH_SLOTS].map(key => sourceId(material[key]))]);
  }
  function volumeSignature(material) {
    if (Object.getPrototypeOf(material) !== three.VolumeNodeMaterial?.prototype) return null;
    // The source calls these graph constructors with positionRay while building
    // its loop. Keep their identities/graphs structural; steps and captured
    // uniforms remain live values at the source RENDER/OBJECT frequencies.
    return JSON.stringify(['offsetNode', 'depthNode', ...VOLUME_FUNCTION_SLOTS].map(key => {
      const node = material[key];
      return [sourceId(node), node?.isNode === true ? node.getCacheKey() : null];
    }));
  }
  function geometrySignature(material) {
    const node = material.geometryNode;
    return node == null ? null : JSON.stringify([sourceId(node), node.getCacheKey()]);
  }
  function inspectCompute(node) {
    const source = three.ComputeNode?.prototype;
    if (!source || Object.getPrototypeOf(node) !== source ||
        ['setup', 'generate', 'updateBefore', 'getUpdateBeforeType'].some(key => node[key] !== source[key]) ||
        ['getUpdateType', 'getUpdateAfterType', 'updateReference'].some(key => node[key] !== three.Node.prototype[key]) ||
        !['none', 'frame', 'render', 'object'].includes(node.updateBeforeType) ||
        node.updateType !== 'none' || node.updateAfterType !== 'none')
      fail('STORAGE', 'Render-triggered compute requires the pinned ComputeNode construction and update lifecycle');
  }
  function sssSignature(material) {
    if (Object.getPrototypeOf(material) !== three.MeshSSSNodeMaterial?.prototype) return null;
    // The retained SSS model consumes these graphs during construction. Slot
    // replacement (including disabling thicknessColorNode) needs a new graph;
    // node uniform values remain live at their source update frequencies.
    return JSON.stringify([...SSS_SLOTS].map(key => {
      const node = material[key];
      return [sourceId(node), node?.isNode === true ? node.getCacheKey() : null];
    }));
  }
  const wideLineCoverage = material => Object.getPrototypeOf(material) === three.Line2NodeMaterial?.prototype &&
    material.alphaToCoverage && renderer.currentSamples > 0;
  function transmissionSignature(material) {
    if (!physicalPrototypes.includes(Object.getPrototypeOf(material))) return null;
    return JSON.stringify([material.useTransmission, material.useDispersion, material.useTransmission ? material.side : null,
      ...['transmissionNode', 'thicknessNode', 'attenuationDistanceNode', 'attenuationColorNode', 'dispersionNode']
        .map(key => sourceId(material[key]))]);
  }
  function fogGraph(material, {scene} = {}) {
    // NodeManager.getFogNode gives a truthy authored graph precedence over the
    // ordinary fog, while NodeMaterial.setupOutput consumes it only when enabled.
    const node = material.fog === true ? scene?.fogNode || null : null;
    if (node && (node.isNode !== true || typeof node.getCacheKey !== 'function'))
      fail('FOG', 'Native scene fog graphs require a source TSL node');
    return node;
  }
  function sourceFog(material, {scene} = {}) {
    if (material.fog !== true || fogGraph(material, {scene})) return null;
    const fog = scene?.fog;
    if (fog == null) return null;
    const linear = Object.getPrototypeOf(fog) === three.Fog.prototype,
      exponential = Object.getPrototypeOf(fog) === three.FogExp2.prototype;
    if (!(linear && fog.isFog === true && !fog.isFogExp2 ||
        exponential && fog.isFogExp2 === true && !fog.isFog) ||
        !(fog.color instanceof three.Color) || fog.color.isColor !== true)
      fail('FOG', 'Native scene fog requires an ordinary source Fog/FogExp2 and Color');
    return fog;
  }
  function fogSignature(material, context) {
    const node = fogGraph(material, context), fog = sourceFog(material, context);
    return JSON.stringify([material.fog === true, sourceId(fog), fog ? fog.isFogExp2 ? 'exp2' : 'linear' : null,
      node ? [sourceId(node), node.getCacheKey()] : null]);
  }
  function fogNodeFor(fog) {
    if (!fog) return null;
    const exponential = fog.isFogExp2 === true, previous = fogNodes.get(fog);
    if (previous?.exponential === exponential) return previous.node;
    // Retain NodeManager.updateFog's exact source graph and render group. Its
    // references read live source fields through NodeFrame; only graph identity
    // and kind belong in the program key, never color/range/density values.
    const {reference, renderGroup} = three.TSL;
    const color = reference('color', 'color', fog).setGroup(renderGroup);
    const node = exponential
      ? three.TSL.fog(color, three.TSL.densityFogFactor(reference('density', 'float', fog).setGroup(renderGroup)))
      : three.TSL.fog(color, three.TSL.rangeFogFactor(reference('near', 'float', fog).setGroup(renderGroup),
        reference('far', 'float', fog).setGroup(renderGroup)));
    fogNodes.set(fog, {exponential, node});
    return node;
  }
  function inspect(material, object, context = {}) {
    live();
    const sourceMaterial = material;
    material = prepareMaterial(material);
    const {scene, camera} = context;
    if (!(material instanceof three.NodeMaterial) ||
        ![three.MeshBasicNodeMaterial.prototype, three.SpriteNodeMaterial.prototype, three.NodeMaterial.prototype, ...primitivePrototypes, ...normalPrototypes, ...litPrototypes].includes(Object.getPrototypeOf(material)))
      fail('MATERIAL', 'Native node graphs require Basic, Normal, Matcap, Toon, Lambert, Phong, Standard, Physical, SSS, Shadow, Sprite, Points, LineBasic, LineDashed, Line2, Volume or unlit base NodeMaterial');
    const prototype = Object.getPrototypeOf(material), lit = litMaterial(material),
      phong = prototype === three.MeshPhongNodeMaterial.prototype, standard = standardPrototypes.includes(prototype),
      physical = physicalPrototypes.includes(prototype), sss = prototype === three.MeshSSSNodeMaterial?.prototype,
      points = prototype === three.PointsNodeMaterial?.prototype,
      sprite = prototype === three.SpriteNodeMaterial.prototype || points,
      dashed = prototype === three.LineDashedNodeMaterial?.prototype || prototype === three.Line2NodeMaterial?.prototype,
      volume = prototype === three.VolumeNodeMaterial?.prototype;
    if (!lit && !material.isMeshBasicNodeMaterial && material.lights !== false)
      fail('LIGHTS', 'Base NodeMaterial graphs must be unlit');
    const lights = sourceLights(material, context), lightsNode = authoredLightsNode(material);
    const authoredNodes = new Set(authoredLightings(lightsNode));
    if (physical && material.useTransmission && (typeof renderer.copyFramebufferToTexture !== 'function' ||
        typeof renderer.getCanvasTarget !== 'function' || typeof renderer._getSampledTextureInputs !== 'function'))
      fail('TRANSMISSION', 'Native physical transmission and dispersion require the captured scene-color pass');
    sourceEnvironment(material, context);
    if (!(object instanceof three.Mesh || object instanceof three.Sprite ||
        three.Points && object instanceof three.Points || three.Line && object instanceof three.Line && !object.isLineLoop) ||
        !(object.geometry instanceof three.BufferGeometry) ||
        object.geometry.isInstancedBufferGeometry && !(object.geometry instanceof three.InstancedBufferGeometry))
      fail('OBJECT', 'Native node graphs require source meshes, sprites, points or non-loop lines with BufferGeometry');
    inspectInstances(object);
    inspectThreeNativeBatch(three, object, fail);
    inspectSkinning(object);
    inspectMorphs(object);
    if (camera) inspectThreeNativeCamera(three, camera);
    sourceFog(material, context);
    inspectThreeNodeClipping(three, context, maxNodes, fail);
    if (material.clippingPlanes?.length || renderer.clippingPlanes?.length)
      fail('EFFECT', 'Native WebGPU clipping uses source ClippingGroup objects, not legacy material or renderer planes');
    if (typeof material.alphaToCoverage !== 'boolean')
      fail('RASTER', 'Native alpha-to-coverage requires the source boolean flag');
    if (material.transparent && material.side === three.DoubleSide && !material.forceSinglePass)
      fail('RASTER', 'Transparent DoubleSide node graphs require captured FrontSide and BackSide draw variants');
    inspectThreeLightingContext(three, renderer.contextNode);
    const pipelineContext = inspectThreePipelineContext(three, renderer, material, object, effects);
    // The pinned WebGPU renderer invokes object hooks only. Material's legacy
    // onBeforeRender callback is inert on this source route, including when set.
    if (sourceMaterial.onBeforeCompile !== three.Material.prototype.onBeforeCompile ||
        sourceMaterial.customProgramCacheKey !== (sourceMaterial.isNodeMaterial === true
          ? three.NodeMaterial.prototype.customProgramCacheKey : three.Material.prototype.customProgramCacheKey))
      fail('HOOK', 'Custom material shader/render hooks are not admitted');
    for (const key of Object.keys(material)) {
      const value = material[key];
      if (value?.isTexture) {
        // sourceEnvironment validates only the selected, active input. An
        // envNode or disabled lighting leaves an unused map unobserved.
        if (key === 'envMap' && (standard || basicEnvironmentMaterial(material))) continue;
        if (!MATERIAL_MAPS.has(key)) fail('TEXTURE', `Native node graph material texture ${key} is not admitted`);
        if (integerTextureType(value))
          fail('TEXTURE', 'Native integer inputs require authored texel loads, not floating-point material map slots');
        if (value.isCubeTexture || value.is3DTexture || value.isData3DTexture || value.isStorageTexture && value.isArrayTexture)
          fail('TEXTURE', 'Native material map slots require ordinary 2D textures');
        inspectTexture(value);
      }
      if (key.endsWith('Node') && value != null && !SLOTS.has(key) && !(sprite && SPRITE_SLOTS.has(key)) && !(lit && LIT_SLOTS.has(key)) &&
          !(points && POINT_SLOTS.has(key)) && !(dashed && DASH_SLOTS.has(key)) &&
          !(volume && (key === 'offsetNode' || VOLUME_FUNCTION_SLOTS.has(key))) &&
          !(key === 'normalNode' && normalPrototypes.includes(prototype)) &&
          !(phong && PHONG_SLOTS.has(key)) && !(standard && STANDARD_SLOTS.has(key)) &&
          !(key === 'envNode' && basicEnvironmentMaterial(material)) && !(physical && PHYSICAL_SLOTS.has(key)) &&
          !(sss && SSS_SLOTS.has(key)) &&
          !SHADOW_SLOTS.has(key) &&
          !(key === 'contextNode' && pipelineContext !== null))
        fail('GRAPH', `Native node graph slot ${key} is not admitted`);
    }
    const mrt = inspectThreeMRT(three, renderer, material);
    const seen = new Set(), stack = [...GRAPH_SLOTS.map(key => key === 'lightsNode' ? lightsNode : material[key]).filter(Boolean),
      ...(volume ? [...VOLUME_FUNCTION_SLOTS].map(key => material[key]).filter(node => typeof node === 'function' && node.isNode === true) : []),
      ...authoredNodes,
      ...[environmentGraph(material, context), fogGraph(material, context)].filter(Boolean), ...(mrt?.nodes ?? [])];
    while (stack.length) {
      const node = stack.pop();
      if (seen.has(node)) continue;
      seen.add(node);
      if (seen.size > maxNodes) fail('LIMIT', 'Native node graph exceeds its node budget');
      checkNode(node, lightsNode, lights, null, null, material, authoredNodes);
      if (node instanceof three.BufferAttributeNode) inspectAttributeInput(node, object);
      // A compute body has its own writable-resource validation at dispatch.
      // Its generated graphics output is checked after source construction.
      if (!node.isComputeNode && typeof node.getChildren === 'function') for (const child of node.getChildren()) stack.push(child);
    }
    return material;
  }
  function sourceMaterialLighting(node, material) {
    if (!material || material.lights !== true || material.fragmentNode != null) return false;
    const prototype = Object.getPrototypeOf(node);
    if (prototype === three.EnvironmentNode.prototype && standardPrototypes.includes(Object.getPrototypeOf(material)) ||
        prototype === three.BasicEnvironmentNode?.prototype && basicEnvironmentMaterial(material)) {
      const reference = node.envNode;
      if (material.envNode) return reference === material.envNode;
      return material.envMap?.isTexture === true && reference instanceof three.MaterialReferenceNode &&
        Object.getPrototypeOf(reference) === three.MaterialReferenceNode.prototype &&
        reference.property === 'envMap' && reference.material === null &&
        reference.uniformType === (material.envMap.isCubeTexture ? 'cubeTexture' : 'texture') &&
        reference.updateReference === three.MaterialReferenceNode.prototype.updateReference;
    }
    if (prototype === three.AONode.prototype)
      return (material.aoNode != null || material.aoMap?.isTexture === true) && node.aoNode === three.TSL.ambientOcclusion;
    if (material.lightMap?.isTexture !== true) return false;
    return material.isMeshBasicNodeMaterial === true
      ? prototype === three.BasicLightMapNode.prototype && node.lightMapNode === three.TSL.materialLightMap
      : prototype === three.IrradianceNode.prototype && node.node === three.TSL.materialLightMap;
  }
  function checkNode(node, lightsNode = null, lights = [], environmentNode = null, boundTextures = null, material = null, contextLightings = null, shadowGraphs = null) {
    if (!node?.isNode) fail('GRAPH', 'Expected source TSL nodes');
    if (node.isComputeNode) { inspectCompute(node); return; }
    if (node.isPassNode === true) inspectThreeNativePass(three, renderer, node);
    if (node.isPassTextureNode === true) {
      if (node.passNode?.isPassNode === true) inspectThreeNativePass(three, renderer, node.passNode);
      else effects.inspectTexture(node);
    }
    if (node.isTexture3DNode && (node.constructor !== three.Texture3DNode ||
        Object.getPrototypeOf(node) !== three.Texture3DNode.prototype || node.gatherNode != null))
      fail('TEXTURE', 'Native volume graphs require the pinned Texture3DNode without unsupported 3D gathers');
    if (node.isStorageTextureNode) storageTextureLayout(node);
    else if (node.isTextureNode) {
      if (node.isCubeTextureNode && (node.constructor !== three.CubeTextureNode ||
          Object.getPrototypeOf(node) !== three.CubeTextureNode.prototype || node.sampler !== true))
        fail('TEXTURE', 'Native cube graphs require the source CubeTextureNode with sampled direction coordinates');
      if ((node.isCubeTextureNode === true) !== (node.value?.isCubeTexture === true))
        fail('TEXTURE', 'Native texture nodes must match their 2D or cube source view');
      // Source shader functions introduce unbound TextureNode parameters with
      // an empty placeholder image. Only real native bindings need residency;
      // authored graph inputs were already inspected before construction.
      if (boundTextures === null || boundTextures.has(node.value)) {
        if ((node.isTexture3DNode === true) !== (node.value?.is3DTexture === true || node.value?.isData3DTexture === true))
          fail('TEXTURE', 'Native volume samples require matching Texture3DNode and source 3D texture dimensions');
        inspectIntegerRead(node);
        const layout = inspectTexture(node.value);
        if (layout.multisampled && (node.isSampleCompare?.() === true || node.biasNode != null ||
            node.gradNode != null || node.gatherNode != null || node.sampler !== false && node.levelNode != null &&
            (node.value.minFilter === three.LinearFilter || node.value.magFilter === three.LinearFilter)))
          fail('TEXTURE', 'Multisampled depth supports source texel loads without comparison, gather, gradients, bias or filtered levels');
        if (node.isSampleCompare?.() === true && (!node.value.isDepthTexture || node.value.compareFunction === null || !node.sampler))
          fail('TEXTURE', 'Native comparison sampling requires an attached depth texture with a comparison function');
      }
      if (node.depthNode !== null && node.value?.isDataArrayTexture !== true && node.value?.isCompressedArrayTexture !== true &&
          !(node.value?.isArrayTexture === true && (node.value.isStorageTexture === true ||
            node.value.isRenderTargetTexture === true || node.value.isDepthTexture === true && node.value.renderTarget != null)))
        fail('TEXTURE', 'Native layer coordinates require source data, compressed, storage or target-owned array textures');
    } else if (node.constructor === three.PMREMNode) inspectPMREMNode(node);
    else if (node.value?.isTexture)
      fail('TEXTURE', 'Native graphics textures require source TextureNode or PMREMNode bindings');
    if (node.isStorageBufferNode) inspectStorage(node);
    else if (managedUniformBuffer(node)) inspectUniformArray(node);
    else if (rawUniformBuffer(node)) inspectUniformBuffer(node);
    else if (!(node instanceof three.BufferAttributeNode) &&
        (node.isBufferNode || node.isBufferAttributeNode || node.isComputeNode))
      fail('STORAGE', 'Unsupported managed buffers and compute graphs are not admitted');
    if (node.isAssignNode) {
      const seen = new Set(), targets = [node.targetNode];
      while (targets.length) {
        const target = targets.pop(); if (!target || seen.has(target)) continue;
        seen.add(target);
        if (target.isStorageBufferNode) fail('STORAGE', 'Native graphics storage bindings are read-only');
        if (nativeUniformBuffer(target)) fail('UNIFORM', 'Native graphics uniform buffers are read-only');
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
    if (node.isLightingNode || node.isLightsNode || node.isLightingContextNode || node.light?.isLight || node instanceof three.ShadowNode) {
      const admitted = lightsNode && (node === lightsNode || sourceMaterialLighting(node, material) ||
        contextLightings?.has(node) && !node.isAnalyticLightNode && !node.light?.isLight ||
        environmentNode && node.constructor === three.EnvironmentNode && node.envNode === environmentNode ||
        checkAnalyticLight(node, lights) || shadowGraphs?.roots.has(node) && !(node instanceof three.ShadowNode) ||
        (lights.includes(node.light) || shadowGraphs?.children.has(node)) &&
          registerThreeNativeShadow(three, renderer, node, shadowGraphs?.children.get(node)));
      if (!admitted) fail('LIGHTS', 'Only the captured source scene light and environment context is admitted');
    }
    if (node.isOutputStructNode && Object.getPrototypeOf(node) !== three.OutputStructNode.prototype &&
        Object.getPrototypeOf(node) !== three.MRTNode.prototype)
      fail('MRT', 'Native fragment output structs require the pinned output node construction');
  }
  function keyOf(material, object, {camera, scene, lightsNode = null, lightingEnabled = renderer.lighting?.enabled !== false} = {}, lights = [], environment = null, instanceSignature = null, skinSignature = null, morphSignature = null, fogKey = null, clippingKey = '0/0') {
    // Match the source's graph cache key. Uniform values never participate.
    // Object/material references remain separate cache dimensions, preserving
    // builder nodes which close over a particular source object.
    return JSON.stringify([material.version, three.NodeMaterial.prototype.customProgramCacheKey.call(material), material.side,
      material.vertexColors, material.flatShading, material.transparent, material.blending, material.premultipliedAlpha,
      object.isMesh === true && material.wireframe === true ? [object.geometry.index?.array.length ?? null,
        object.geometry.attributes.position?.array.length ?? null, object.geometry.attributes.position?.count >= 65535] : null,
      material.alphaTest > 0, material.alphaHash === true, material.alphaToCoverage, material.alphaToCoverage && renderer.currentSamples > 1,
      wideLineCoverage(material),
      material.depthWrite, material.depthTest, material.lights, clippingKey,
      spriteSignature(material, object), wideLineSignature(material), volumeSignature(material), sssSignature(material), geometrySignature(material),
      primitiveLayout(object, material),
      object.geometry.id, object.geometry.isInstancedBufferGeometry === true,
      camera?.isPerspectiveCamera === true, object.receiveShadow === true,
      camera?.isArrayCamera === true ? [camera.id, camera.cameras.length] : null,
      instanceSignature, inspectThreeNativeBatch(three, object, fail), skinSignature, morphSignature, fogKey, materialMapSignature(material),
      inspectThreePipelineContext(three, renderer, material, object, effects),
      inspectThreeLightingContext(three, renderer.contextNode),
      inspectThreeMRT(three, renderer, material)?.key ?? null, environmentGraphKey(material, {scene}),
      basicEnvironmentSignature(material, environment),
      environment ? [environment.id, environment.mapping, environment.isRenderTargetTexture === true,
        environment.isPMREMTexture === true, materialEnvironment(material, environment)] : null,
      Object.entries(object.geometry.attributes).map(([name, a]) => [name,
        (a.isStorageBufferAttribute || a.isStorageInstancedBufferAttribute) && a.itemSize === 3 ? 4 : a.itemSize,
        a.normalized, a.gpuType, a.array?.constructor.name,
        a.isInterleavedBufferAttribute, a.data?.array?.constructor.name, a.data?.stride, a.offset,
        a.isInstancedBufferAttribute === true, a.data?.isInstancedInterleavedBuffer === true]),
      renderer.logarithmicDepthBuffer, renderer.reversedDepthBuffer,
      lightingMaterial(material) && material.fragmentNode == null ? [lightingEnabled, lightsNode && sourceId(lightsNode), lightingSignature(material, lights)] : null,
      physicalPrototypes.includes(Object.getPrototypeOf(material)) ?
        [material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useRetroreflection,
          material.useTransmission, material.useDispersion] : null]);
  }
  function retire(entry) {
    if (!entries.delete(entry)) return;
    const state = nodeStates.get(entry.descriptor);
    for (const record of state?.bundleLeases ?? []) releaseBundleGeometry(record);
    for (const record of state?.fogRecords?.values() ?? []) releaseFogResources(record);
    releaseShaders(entry.shaders);
    entry.material.removeEventListener('dispose', entry.onDispose);
    entry.geometry.removeEventListener('dispose', entry.onDispose);
    if (entry.attributeNodes.length) generatedAttributes.release(entry.descriptor);
    if (entry.descriptor.wireframe) wireframes.release(entry.descriptor);
    const materials = cache.get(entry.object), passes = materials?.get(entry.material);
    if (passes?.get(entry.passId) === entry) {
      passes.delete(entry.passId);
      if (!passes.size) materials.delete(entry.material);
    }
  }
  function compile(material, object, context = {}) {
    const preparedMaterial = inspect(material, object, context);
    const lights = sourceLights(preparedMaterial, context), environment = sourceEnvironment(preparedMaterial, context);
    const lightKey = lightingSignature(preparedMaterial, lights);
    const instances = inspectInstances(object), instanceSignature = instances?.signature ?? null;
    const batchSignature = inspectThreeNativeBatch(three, object, fail);
    const skinSignature = inspectSkinning(object);
    const morphSignature = inspectMorphs(object), pipelineSignature = inspectThreePipelineContext(three, renderer, material, object, effects);
    const rendererContextSignature = inspectThreeLightingContext(three, renderer.contextNode);
    const fogKey = fogSignature(preparedMaterial, context), alphaHash = preparedMaterial.alphaHash === true;
    const volumeKey = volumeSignature(preparedMaterial), sssKey = sssSignature(preparedMaterial), geometryKey = geometrySignature(preparedMaterial);
    const clippingKey = inspectThreeNodeClipping(three, context, maxNodes, fail);
    const mrtKey = inspectThreeMRT(three, renderer, preparedMaterial)?.key ?? null;
    let key = keyOf(preparedMaterial, object, context, lights, environment, instanceSignature, skinSignature, morphSignature, fogKey, clippingKey);
    const byMaterial = cache.get(object);
    // Source render objects distinguish the transparent back-side pass from
    // the ordinary front-side pass. Both may own generated attribute streams;
    // compiling one must not retire the other's accepted descriptor.
    const passId = context.passId ?? null, passes = byMaterial?.get(material);
    const previous = passes?.get(passId);
    const previousState = previous && nodeStates.get(previous.descriptor);
    if (previousState?.basicEnvironmentSignature !== null && previousState?.basicEnvironmentSignature !== undefined &&
        basicEnvironmentSignature(preparedMaterial, environment) !== null) {
      const [combine, mapping] = JSON.parse(previousState.basicEnvironmentSignature);
      const sameGraph = environmentGraphKey(preparedMaterial, context) === previousState.environmentGraph;
      // The pinned RenderObject reduces numeric material cache values to zero/
      // nonzero, so needsUpdate cannot distinguish Mix from Add. TextureNode's
      // graph key likewise excludes the input texture's mapping. Do not compile
      // a new shader where that source renderer would keep its previous shader.
      if (sameGraph && combine !== preparedMaterial.combine && combine !== 0 && preparedMaterial.combine !== 0)
        fail('ENVIRONMENT_CACHE', 'Pinned Mix/Add changes require disposing the prior material or replacing its environment graph');
      if (sameGraph && preparedMaterial.envNode?.isTextureNode && preparedMaterial.envMap !== environment && mapping !== environment?.mapping)
        fail('ENVIRONMENT_CACHE', 'Pinned texture envNode mapping changes require replacing the environment node or disposing the material');
    }
    if (previous?.key === key && pmremSignature(previous.pmremNodes) === previous.pmremSignature &&
        textureSignature(previous.textureNodes, previous.textureReferences) === previous.textureSignature &&
        storageSignature(previous.storageNodes) === previous.storageSignature &&
        uniformSignature(previous.uniformNodes, previous.builder) === previous.uniformSignature &&
        attributeSignature(previous.attributeNodes, object) === previous.attributeSignature) {
      updateThreeNodeClipping(previous.builder.clippingContext, context);
      return previous.descriptor;
    }
    if (entries.size - (previous ? 1 : 0) >= maxPrograms) fail('LIMIT', 'Native node program capacity exceeded');
    // setup() itself may filter a PMREM source. An existing stale atlas must
    // reach the first draw's ordered preparation boundary through its cached
    // graph, rather than being rewritten while another graph is constructed.
    assertEnvironmentVersion(environment);
    for (const entry of entries) for (const node of entry.pmremNodes) assertEnvironmentVersion(node.value);
    const builder = new three.WGSLNodeBuilder(object, renderer);
    const captureShadowGraphs = captureThreeNativeShadowGraphs(three, builder);
    effects.installBuilder(builder);
    installThreeNativeShadowTargets(renderer, builder);
    builder.material = material; builder.context.material = material;
    builder.scene = context.scene ?? null; builder.camera = context.camera ?? null;
    builder.clippingContext = Object.fromEntries(['unionPlanes', 'intersectionPlanes'].map(key =>
      [key, (context.clippingContext?.[key] ?? []).map(plane => plane.clone())]));
    const authored = authoredLightsNode(preparedMaterial);
    // Keep the scene entrance distinct from material.lightsNode, just as the
    // pinned NodeManager does. NodeMaterial selects the authored graph itself.
    if (lightingMaterial(preparedMaterial)) builder.lightsNode = context.lightsNode ?? three.TSL.lights(authored
      ? context.lightingEnabled === false ? [] : context.lights ?? [] : lights);
    const lightsNode = authored ?? builder.lightsNode;
    const environmentNode = environmentGraph(preparedMaterial, context) || environmentNodeFor(environment);
    if (environmentNode) builder.environmentNode = environmentNode;
    builder.fogNode = fogGraph(preparedMaterial, context) || fogNodeFor(sourceFog(preparedMaterial, context));
    // ComputeNode.setup constructs its body in this graphics builder, but its
    // generate() emits only outputComputeNode. Keep source update lists intact
    // while separating setup-only compute writes from actual graphics nodes.
    const computeChildren = new Set(), graphicsNodes = new Set(), generatedNodes = new Set(), addChain = builder.addChain;
    builder.addChain = function (node) {
      (this.chaining.some(parent => parent.isComputeNode) ? computeChildren : graphicsNodes).add(node);
      if (this.buildStage === 'generate') generatedNodes.add(node);
      return addChain.call(this, node);
    };
    try { builder.build(); }
    catch (cause) {
      if (cause instanceof ThreeNodeProgramError) throw cause;
      throw new ThreeNodeProgramError('BUILD', `Pinned node construction failed: ${cause?.message ?? cause}`, {cause});
    } finally { builder.addChain = addChain; }
    const graphicsNode = node => node.isComputeNode || !computeChildren.has(node) || graphicsNodes.has(node) || generatedNodes.has(node);
    assertInstances(object, instanceSignature);
    assertBatch(object, batchSignature);
    assertSkinning(object, skinSignature);
    assertMorphs(object, morphSignature);
    if (inspectThreeLightingContext(three, renderer.contextNode, 'CHANGED') !== rendererContextSignature)
      fail('CHANGED', 'Native lighting context changed during source graph construction');
    if ((inspectThreeMRT(three, renderer, material)?.key ?? null) !== mrtKey)
      fail('CHANGED', 'Native MRT context changed during source graph construction');
    if (fogSignature(prepareMaterial(material), context) !== fogKey) fail('CHANGED', 'Native scene fog changed during source graph construction');
    if ((material.alphaHash === true) !== alphaHash) fail('CHANGED', 'Native alpha hashing changed during source graph construction');
    if (volumeSignature(prepareMaterial(material)) !== volumeKey)
      fail('CHANGED', 'Native volume graphs changed during source construction');
    if (sssSignature(prepareMaterial(material)) !== sssKey)
      fail('CHANGED', 'Native subsurface scattering graphs changed during source construction');
    if (geometrySignature(prepareMaterial(material)) !== geometryKey)
      fail('CHANGED', 'Native geometry graph changed during source construction');
    if (inspectThreeNodeClipping(three, context, maxNodes, fail) !== clippingKey)
      fail('CHANGED', 'Native clipping layout changed during source graph construction');
    if (lightingSignature(prepareMaterial(material), sourceLights(prepareMaterial(material), context)) !== lightKey)
      fail('CHANGED', 'Native light selection or registration changed during source graph construction');
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
    const analyticLightNodes = [...builder.nodes].filter(node => checkAnalyticLight(node, lights));
    const shadowGraphs = captureShadowGraphs(analyticLightNodes);
    // Explicit contributors and AO/GI hooks share the retained source context.
    // Their generated graphs still pass every ordinary node/resource check.
    const contextLightings = new Set(authoredLightings(authored));
    for (const node of builder.nodes) if (Object.getPrototypeOf(node) === three.LightingContextNode.prototype && node.node === lightsNode) {
      for (const lighting of node.materialLightings) {
        if (typeof builder.context.getAO === 'function' && Object.getPrototypeOf(lighting) === three.AONode.prototype &&
            lighting.aoNode === three.TSL.ambientOcclusion ||
            typeof builder.context.getGI === 'function' && Object.getPrototypeOf(lighting) === three.IrradianceNode.prototype)
          contextLightings.add(lighting);
      }
    }
    for (const node of builder.nodes) {
      if (!graphicsNode(node)) continue;
      checkNode(node, lightsNode, lights, environmentNode, boundTextures, preparedMaterial, contextLightings, shadowGraphs);
      if (node instanceof three.BufferAttributeNode) inspectAttributeInput(node, object);
    }
    // Producer inspection above registers source-owned pipeline state created
    // during setup (including TRAA's view-offset owner).
    if (inspectThreePipelineContext(three, renderer, material, object, effects) !== pipelineSignature)
      fail('CHANGED', 'Native material context changed during source graph construction');
    if (bindings.length > Math.min(maxBindGroups, renderer._device?.limits?.maxBindGroups ?? 4))
      fail('LIMIT', 'Native node graph exceeds its bind group budget');
    let uniformBytes = 0, uniformCount = 0;
    const alignment = renderer._device?.limits?.minUniformBufferOffsetAlignment ?? 256;
    if (!Number.isSafeInteger(alignment) || alignment < 1 || !Number.isInteger(Math.log2(alignment)))
      fail('LIMIT', 'Native uniform offsets require a valid device alignment');
    const sampledCounts = [0, 0], samplerCounts = [0, 0], storageCounts = [0, 0], storageTextureCounts = [0, 0], uniformCounts = [0, 0];
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
          const atomic = binding.nodeUniform.isAtomic === true;
          if (atomic && binding.visibility !== 2)
            fail('STORAGE', 'Source atomic storage is supported only in the fragment stage of native graphics');
          for (let stage = 0; stage < 2; stage++) if (binding.visibility & (1 << stage))
            if (++storageCounts[stage] > (renderer._device?.limits?.maxStorageBuffersPerShaderStage ?? 8))
              fail('LIMIT', 'Native node storage bindings exceed per-stage device limits');
          // Match r186 getNodeAccess and WebGPUBindingUtils: fragment atomics
          // are writable; ordinary graphics storage remains read-only.
          return Object.freeze({...base, buffer: Object.freeze({type: atomic ? 'storage' : 'read-only-storage', hasDynamicOffset: false, minBindingSize})});
        }
        if (binding.textureNode?.isStorageTextureNode && binding.isSampledTexture && binding.store) {
          const layout = storageTextureLayout(binding.textureNode, binding.textureNode.value, binding.mipLevel);
          if (binding.access !== 'readOnly' || (binding.visibility & ~3) !== 0)
            fail('TEXTURE', 'Native storage texture bindings require read-only vertex or fragment visibility');
          for (let stage = 0; stage < 2; stage++) if (binding.visibility & (1 << stage))
            if (++storageTextureCounts[stage] > (renderer._device?.limits?.maxStorageTexturesPerShaderStage ?? 4))
              fail('LIMIT', 'Native storage texture bindings exceed per-stage device limits');
          return Object.freeze({...base, storageTexture: Object.freeze({...layout})});
        }
        if (!binding.textureNode?.isTextureNode || binding.store || (!binding.isSampledTexture && !binding.isSampler))
          fail('BINDING', 'Native graphics groups require uniform blocks or ordinary sampled 2D, array, cube or 3D texture bindings');
        const textureLayout = inspectTexture(binding.textureNode.value);
        const sampled = binding.isSampledTexture === true, counts = sampled ? sampledCounts : samplerCounts;
        if (!sampled && integerTextureType(binding.textureNode.value))
          fail('BINDING', 'Native integer texel loads must not acquire a sampler binding');
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
      const storage = source?.isStorageBufferAttribute || source?.isStorageInstancedBufferAttribute;
      const padded = storage && source.itemSize === 4 && components === 3;
      if (!components || !source || source.itemSize !== components && !padded ||
          node && !(node instanceof three.BufferAttributeNode))
        fail('ATTRIBUTE', `Native node attribute ${attribute.name} requires a matching scalar/vector geometry attribute`);
      if (node || storage) generated.push({name: attribute.name, attribute: source});
      return Object.freeze({name: attribute.name, location, locations: 1, components, scalar});
    });
    const vertex = builder.vertexShader, fragment = builder.fragmentShader;
    const shared = shaderPairs.get(vertex)?.get(fragment);
    const shaders = shared ?? {vertex, fragment, bytes: 2 * (vertex.length + fragment.length), references: 0};
    // A replacement can reclaim a pair only when no other live program uses
    // it. A failed build must leave both the prior descriptor and budget live.
    const reclaim = previous?.shaders.references === 1 && previous.shaders !== shaders ? previous.shaders.bytes : 0;
    if (!shared && shaders.bytes > maxShaderBytes - shaderBytes + reclaim)
      fail('LIMIT', 'Native node shader byte budget exceeded');
    builder.vertexShader = shaders.vertex; builder.fragmentShader = shaders.fragment;
    const outputStruct = builder.structs.fragment.find(struct => struct.output === true);
    const colorCount = outputStruct?.members.length ?? 1;
    if (colorCount > (renderer.getRenderTarget?.()?.textures.length ?? 1))
      fail('MRT', 'Native fragment outputs exceed the current color attachments');
    let cameraIndex = null;
    if (context.camera?.isArrayCamera && context.camera.cameras.length) {
      const index = groups.findIndex(group => group.name === 'cameraIndex');
      if (index !== -1) {
        const group = groups[index], binding = bindings[index].bindings[0], uniform = binding?.uniforms?.[0];
        // The source backend replaces this dedicated group with one uint index
        // per child. Preserve its layout without changing the public TSL node.
        if (group.entries.length !== 1 || group.uniforms.length !== 1 || binding.uniforms.length !== 1 ||
            uniform.nodeUniform.type !== 'uint' || uniform.nodeUniform.node.name !== 'u_cameraIndex')
          fail('CAMERA', 'Native ArrayCamera requires the source camera-index uniform group');
        cameraIndex = Object.freeze({group: index, offset: group.uniforms[0].offset + uniform.offset * 4});
      }
    }
    const descriptor = Object.freeze({nodeProgram: true, key: `native-node-${++serial}`,
      vertex: shaders.vertex, fragment: shaders.fragment, colorCount,
      cameraIndex,
      ...primitiveLayout(object, preparedMaterial),
      attributes: Object.freeze(attributes), attributesKey: JSON.stringify(attributes), groups: Object.freeze(groups),
      ...(instances ? {instanceCapacity: instances.capacity} : {})});
    // Node.build() deduplicates unsized storage by attribute before addNode().
    // Walk the original child edges too, including graphs created inside Fn,
    // so changing one alias cannot silently keep the old shared binding.
    const graphNodes = new Set([...builder.nodes].filter(graphicsNode));
    for (const node of graphNodes) {
      if (graphNodes.size > maxNodes) fail('LIMIT', 'Generated native node graph exceeds its node budget');
      if (!node.isComputeNode && typeof node.getChildren === 'function')
        for (const child of node.getChildren()) if (graphicsNode(child)) graphNodes.add(child);
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
    const uniformNodes = [...graphNodes].filter(nativeUniformBuffer), uniformKey = uniformSignature(uniformNodes, builder);
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
    if (descriptor.wireframe) wireframes.register(descriptor, object.geometry,
      generated.length ? generatedAttributes.geometry(descriptor) : object.geometry);
    for (const attribute of instances?.cpu ?? []) instanceArrays.set(attribute, attribute.array);
    nodeStates.set(descriptor, {owner: api, descriptor, builder, bindings, material, object, lights, lightSignature: lightKey, analyticLightNodes, shadowGraphs,
      environment, materialEnvironment: materialEnvironment(preparedMaterial, environment), environmentGraph: environmentGraphKey(preparedMaterial, context),
      basicEnvironmentSignature: basicEnvironmentSignature(preparedMaterial, environment),
      transmissionSignature: transmissionSignature(preparedMaterial),
      spriteSignature: spriteSignature(preparedMaterial, object),
      wideLineSignature: wideLineSignature(preparedMaterial),
      volumeSignature: volumeKey,
      sssSignature: sssKey, geometrySignature: geometryKey,
      instanceSignature, batchSignature, skinSignature, morphSignature, pipelineSignature, rendererContextSignature,
      fogSignature: fogKey, mrtSignature: mrtKey,
      clippingSignature: clippingKey, alphaHash, alphaToCoverage: material.alphaToCoverage, wideLineCoverage: wideLineCoverage(material),
      clippingSamples: material.alphaToCoverage && renderer.currentSamples > 1, pmremNodes, pmremSignature: pmremKey,
      integerTextures: textureNodes.some(node => integerTextureType(node.value)),
      storageTextures: textureNodes.some(node => node.isStorageTextureNode),
      textureNodes, textureReferences, textureSignature: signature, materialMapSignature: materialMapSignature(material),
      storageNodes, storageSignature: storageKey, uniformNodes, uniformSignature: uniformKey,
      attributeNodes, attributeSignature: attributeKey, geometry: object.geometry, generated: generated.length > 0});
    // Source geometry constructors may add named storage attributes. Adopt
    // their accepted layout now, including the canonical vec3 padding above,
    // instead of rebuilding that constructor and repeating onInit next frame.
    if (geometryKey !== null) key = keyOf(preparedMaterial, object, context, lights, environment,
      instanceSignature, skinSignature, morphSignature, fogKey, clippingKey);
    const entry = {key, passId, descriptor, builder, material, object, geometry: object.geometry, shaders, pmremNodes, pmremSignature: pmremKey,
      textureNodes, textureReferences, textureSignature: signature,
      storageNodes, storageSignature: storageKey, uniformNodes, uniformSignature: uniformKey,
      attributeNodes, attributeSignature: attributeKey};
    entry.onDispose = () => retire(entry);
    // Source shared UniformArray bindings survive graph reconstruction, notably
    // an ArrayCamera view-count change. A newly validated layout owns a fresh
    // allocation; accepted older draws already retain their immutable packets.
    for (const group of groups) for (const member of group.uniforms) {
      const binding = bindings[group.index].bindings[member.binding], history = uniformBuffers.get(binding);
      if (history && history.bytes.byteLength !== member.byteLength) uniformBuffers.delete(binding);
    }
    // Acquire before retiring to preserve an unchanged pair's cache identity.
    retainShaders(shaders);
    if (previous) retire(previous);
    const next = byMaterial ?? new Map(), nextPasses = passes ?? new Map();
    nextPasses.set(passId, entry); next.set(material, nextPasses); cache.set(object, next);
    entries.add(entry);
    material.addEventListener('dispose', entry.onDispose); object.geometry.addEventListener('dispose', entry.onDispose);
    return descriptor;
  }
  function stateOf(descriptor) {
    live(); const state = nodeStates.get(descriptor);
    if (state?.owner !== api) fail('SOURCE', 'Native node descriptor belongs to another preparation owner');
    return state;
  }
  // Render bundles retain source render objects independently of the next
  // projection. In particular, static is a refresh policy, not a replacement
  // for versioned bundle structure. Use the pinned builder's actual observer.
  function bundleRecord(state, context) {
    if (context.bundleToken === undefined) {
      // The source observer does not count scene.fogNode as an authored
      // material slot. Ordinary materials therefore keep their object-group
      // fog uniforms and resource bindings on SHARED refresh, even though the
      // fog callbacks still run. The same observer controls ordinary indirect
      // geometry: changing only its argument attribute does not force uploads.
      if (context.fogRenderContext === undefined || state.builder.observer.hasNode) return null;
      const records = state.fogRecords ??= new Map(), key = context.fogRenderContext;
      let record = records.get(key);
      if (!record) {
        const renderObject = {object: context.object ?? state.object, geometry: state.geometry, material: state.material,
          scene: context.scene ?? state.builder.scene, camera: context.camera ?? state.builder.camera,
          context: context.renderContext ?? {}, bundle: null,
          lightsNode: state.builder.lightsNode ?? {getBuiltinLights: () => []}};
        record = {renderObject, packet: null, refreshType: three.RenderObjectRefreshType.FULL, fog: true, resources: null};
        records.set(key, record);
      }
      record.renderObject.camera = context.camera ?? state.builder.camera;
      record.renderObject.context = context.renderContext ?? record.renderObject.context;
      return record;
    }
    const token = context.bundleToken;
    if ((typeof token !== 'object' && typeof token !== 'function') || token === null ||
        context.bundle?.isBundleGroup !== true)
      fail('SOURCE', 'Native bundle refresh requires a stable render-object token and source BundleGroup');
    const records = state.bundleRecords ??= new WeakMap();
    let record = records.get(token);
    if (!record) {
      const renderObject = {object: context.object ?? state.object, geometry: state.geometry, material: state.material,
        scene: context.scene ?? state.builder.scene, camera: context.camera ?? state.builder.camera,
        context: context.renderContext ?? {}, bundle: context.bundle,
        lightsNode: state.builder.lightsNode ?? {getBuiltinLights: () => []}};
      record = {renderObject, packet: null, refreshType: three.RenderObjectRefreshType.FULL};
      records.set(token, record);
    }
    if (record.renderObject.bundle !== context.bundle)
      fail('SOURCE', 'A native bundle render-object token cannot change its group');
    return record;
  }
  function refresh(descriptor, context = {}) {
    const state = stateOf(descriptor), record = bundleRecord(state, context);
    if (!record) return three.RenderObjectRefreshType.FULL;
    return withFrame(state, context, () => record.refreshType =
      state.builder.observer.needsRefresh(record.renderObject, frame));
  }
  function bundleRefresh(record, context) {
    const kind = context.refreshType ?? record?.refreshType ?? three.RenderObjectRefreshType.FULL;
    if (kind !== three.RenderObjectRefreshType.NONE && kind !== three.RenderObjectRefreshType.SHARED &&
        kind !== three.RenderObjectRefreshType.FULL ||
        !record && kind !== three.RenderObjectRefreshType.FULL ||
        record && !record.packet && kind !== three.RenderObjectRefreshType.FULL)
      fail('SOURCE', 'Native bundle refresh requires a previous full capture');
    return kind;
  }
  function releaseBundleGeometry(record) {
    record.ownedBundleGeometry?.release?.();
    if (record.ownedIndirect) record.ownedIndirect.owner.release(record.ownedIndirect.lease);
    record.ownedIndirect = null;
    record.ownedBundleGeometry = null;
    record.leaseSet?.delete(record); bundleGeometryRecords.delete(record);
  }
  function releaseFogResources(record) {
    if (record.resources) {
      for (const binding of record.resources.textures) record.resources.textureOwner.release(binding.lease);
      for (const binding of record.resources.storage) record.resources.storageOwner.release(binding.lease);
      for (const lease of record.resources.vertexStorage ?? []) record.resources.storageOwner.release(lease);
      if (record.resources.indirect) record.resources.storageOwner.release(record.resources.indirect.lease);
    }
    record.resources = null; fogResourceRecords.delete(record);
  }
  function retainFogResources(descriptor, context, resources) {
    const record = bundleRecord(stateOf(descriptor), context);
    if (!record?.fog) return;
    const retained = [];
    try {
      for (const binding of resources.textures) {
        resources.textureOwner.retain(binding.lease);
        retained.push(() => resources.textureOwner.release(binding.lease));
      }
      for (const binding of resources.storage) {
        resources.storageOwner.retain(binding.lease);
        retained.push(() => resources.storageOwner.release(binding.lease));
      }
      for (const lease of resources.vertexStorage ?? []) {
        resources.storageOwner.retain(lease);
        retained.push(() => resources.storageOwner.release(lease));
      }
      if (resources.indirect) {
        resources.storageOwner.retain(resources.indirect.lease);
        retained.push(() => resources.storageOwner.release(resources.indirect.lease));
      }
    } catch (error) {
      for (const release of retained.reverse()) release();
      throw error;
    }
    releaseFogResources(record);
    record.resources = resources; fogResourceRecords.add(record);
  }
  function storeBundlePacket(state, record, packet) {
    const previous = record.ownedBundleGeometry, previousIndirect = record.ownedIndirect;
    // Source observer packets can survive a bundle re-recording. Keep their
    // geometry-time argument lease independently of each submitted draw lease.
    if (packet.indirect) packet.indirect.owner.retain(packet.indirect.lease);
    record.packet = packet; record.ownedBundleGeometry = packet.bundleGeometry ?? null;
    record.ownedIndirect = packet.indirect;
    if (typeof record.ownedBundleGeometry?.release === 'function' || record.ownedIndirect) {
      record.leaseSet = state.bundleLeases ??= new Set();
      record.leaseSet.add(record); bundleGeometryRecords.add(record);
    } else {
      record.leaseSet?.delete(record); bundleGeometryRecords.delete(record);
    }
    if (previous !== record.ownedBundleGeometry) previous?.release?.();
    if (previousIndirect) previousIndirect.owner.release(previousIndirect.lease);
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
    if (lightingSignature(prepareMaterial(state.material), state.lights) !== state.lightSignature)
      fail('CHANGED', 'Native light selection, registrations or authored lighting graphs changed; compile the graph again');
    for (const node of state.analyticLightNodes) if (!checkAnalyticLight(node, state.lights))
      fail('LIGHTS', 'Native analytic light identity changed after source construction');
    for (const [node, light] of state.shadowGraphs.roots) inspectThreeNativeShadowGraph(three, node, light);
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
      if (value.isSampledTexture || value.isSampler) {
        const texture = liveReferences ? textureValue(value.textureNode, references) : value.textureNode.value;
        if (value.store) storageTextureLayout(value.textureNode, texture, value.mipLevel);
        textures.push(Object.freeze({group, binding, texture,
        isSampler: value.isSampler === true, isSampledTexture: value.isSampledTexture === true,
        comparison: comparison(value.textureNode, texture),
        store: value.store === true, access: value.store === true ? value.access : null,
        mipLevel: value.store === true ? value.mipLevel : 0}));
      }
    return Object.freeze(textures);
  }
  function readStorage(state) {
    if (storageSignature(state.storageNodes) !== state.storageSignature)
      fail('CHANGED', 'Native storage layout, struct schema or binding aliases changed; compile the graph again');
    const storage = [], attributes = new Map();
    for (const [group, source] of state.bindings.entries()) for (const [binding, value] of source.bindings.entries())
      if (value.isStorageBuffer) {
        const attribute = value.attribute;
        const writable = state.descriptor.groups[group].entries.find(entry => entry.binding === binding).buffer.type === 'storage';
        if (attributes.has(attribute) && (writable || attributes.get(attribute)))
          fail('STORAGE', 'A native draw cannot bind the same attribute to multiple bindings when one is writable');
        attributes.set(attribute, writable);
        storage.push(Object.freeze({group, binding, attribute, writable}));
      }
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
      if (binding.isNodeUniformsGroup && bytes.byteLength !== byteLength)
        fail('CHANGED', 'Native node uniform layout changed during its update');
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
  function checkPipelineContext(state, object = state.object) {
    if (inspectThreeLightingContext(three, renderer.contextNode, 'CHANGED') !== state.rendererContextSignature)
      fail('CHANGED', 'Native renderer lighting context changed; compile the material graph again');
    if (inspectThreePipelineContext(three, renderer, state.material, object, effects) !== state.pipelineSignature)
      fail('CHANGED', 'Native render-pipeline context changed; compile its output graph again');
    if ((inspectThreeMRT(three, renderer, state.material)?.key ?? null) !== state.mrtSignature)
      fail('CHANGED', 'Native MRT graph or attachment layout changed; compile its output graph again');
  }
  function checkFog(state, context) {
    if (fogSignature(prepareMaterial(state.material), {scene: context.scene ?? state.builder.scene}) !== state.fogSignature)
      fail('CHANGED', 'Native scene fog identity, graph, kind or material enable flag changed; compile the graph again');
  }
  function checkEnvironment(state, context) {
    const material = prepareMaterial(state.material), environmentContext = {scene: context.scene ?? state.builder.scene};
    const environment = sourceEnvironment(material, environmentContext);
    if (environment !== state.environment || materialEnvironment(material, environment) !== state.materialEnvironment ||
        environmentGraphKey(material, environmentContext) !== state.environmentGraph ||
        basicEnvironmentSignature(material, environment) !== state.basicEnvironmentSignature)
      fail('CHANGED', 'Native material or scene environment changed; compile the graph again');
  }
  function checkTransmission(state) {
    if (transmissionSignature(prepareMaterial(state.material)) !== state.transmissionSignature)
      fail('CHANGED', 'Native transmission or dispersion graph changed; compile the physical draw variant again');
  }
  function checkVolume(state) {
    if (volumeSignature(prepareMaterial(state.material)) !== state.volumeSignature)
      fail('CHANGED', 'Native volume scattering, emission, offset or depth graphs changed; compile the volume graph again');
  }
  function checkGeometry(state) {
    if (geometrySignature(prepareMaterial(state.material)) !== state.geometrySignature)
      fail('CHANGED', 'Native geometry graph changed; compile the source geometry constructor again');
  }
  function checkSSS(state) {
    if (sssSignature(prepareMaterial(state.material)) !== state.sssSignature)
      fail('CHANGED', 'Native subsurface thickness graphs changed; compile the SSS graph again');
  }
  function checkClipping(state, context) {
    if ((state.material.alphaHash === true) !== state.alphaHash)
      fail('CHANGED', 'Native alpha hashing changed; compile the material graph again');
    if (inspectThreeNodeClipping(three, context, maxNodes, fail) !== state.clippingSignature ||
        state.material.alphaToCoverage !== state.alphaToCoverage ||
        wideLineCoverage(state.material) !== state.wideLineCoverage ||
        (state.material.alphaToCoverage && renderer.currentSamples > 1) !== state.clippingSamples)
      fail('CHANGED', 'Native clipping layout or coverage mode changed; compile the graph again');
  }
  function assertBatch(object, signature) {
    if (inspectThreeNativeBatch(three, object, fail) !== signature)
      fail('CHANGED', 'Native batch texture identities changed; compile the source batching graph again');
  }
  function checkWireframe(state, object = state.object) {
    const primitive = primitiveLayout(object, state.material);
    if (Object.entries(primitive).some(([key, value]) => state.descriptor[key] !== value))
      fail('CHANGED', 'Native primitive topology or strip index format changed; compile the graph again');
  }
  function captureGeometry(state, renderId, captureVertexAttribute) {
    const generated = state.generated ? generatedAttributes.capture(state.descriptor, renderId, captureVertexAttribute) : null;
    const wireframe = state.descriptor.wireframe ? wireframes.capture(state.descriptor, renderId) : null;
    if (!wireframe) return generated;
    return Object.freeze({bytes: wireframe.bytes + (generated?.bytes ?? 0), indexCount: wireframe.indexCount,
      geometry: wireframe.geometry,
      install() { generated?.install(); wireframe.install(); }});
  }
  function checkCapture(state, context) {
    const object = context.object ?? state.object;
    checkWireframe(state, object);
    if (spriteSignature(prepareMaterial(state.material), object) !== state.spriteSignature)
      fail('CHANGED', 'Native sprite attenuation or center reference changed; compile the billboard graph again');
    if (wideLineSignature(prepareMaterial(state.material)) !== state.wideLineSignature)
      fail('CHANGED', 'Native wide-line expansion, dashes, colors or transparency changed; compile the line graph again');
    checkVolume(state);
    checkSSS(state);
    checkGeometry(state);
    checkPipelineContext(state, object);
    checkFog(state, context);
    checkEnvironment(state, context);
    checkTransmission(state);
    checkClipping(state, context);
    updateThreeNodeClipping(state.builder.clippingContext, context);
    checkMaterialMaps(state);
    // Catch stale signedness, aliases and newly sampled integer reads before
    // observable callbacks. Live material references are read without updates;
    // readTextures also revalidates mutations made by the actual node updates.
    if (state.integerTextures || state.storageTextures || state.textureNodes.some(node =>
      node.isStorageTextureNode || integerTextureType(textureValue(node, state.textureReferences))))
      readTextures(state, true);
    checkLights(state);
    assertInstances(object, state.instanceSignature);
    assertBatch(object, state.batchSignature);
    assertSkinning(object, state.skinSignature);
    assertMorphs(object, state.morphSignature);
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
        const provenance = state.shadowGraphs.children.get(node);
        if (!state.lights.includes(node.light) && !provenance) fail('SHADOW', 'A source shadow node changed its captured light identity');
        registerThreeNativeShadow(three, renderer, node, provenance);
        updateThreeNativeShadow(three, renderer, node, frame);
      } else if (node.isPassNode === true) updateThreeNativePass(three, renderer, node, frame);
      else if (node.isComputeNode) {
        inspectCompute(node);
        withThreeNativeComputePrerequisites(renderer, () => frame.updateBeforeNode(node), true);
      } else if (effects.owns(node)) effects.update(node, frame);
      else frame.updateBeforeNode(node);
      // PMREM refresh keeps its separate ordered-prefix contract. Shadow
      // recursion runs outside draw-resource capture at its source boundary.
      checkPMREM(state);
    }
  }
  function before(descriptor, context = {}) {
    const state = stateOf(descriptor), record = bundleRecord(state, context), kind = bundleRefresh(record, context);
    if (kind === three.RenderObjectRefreshType.NONE) return;
    if (kind === three.RenderObjectRefreshType.FULL) checkCapture(state, context);
    return withFrame(state, context, () => runBefore(state));
  }
  function captureDraw(descriptor, context = {}) {
    const state = stateOf(descriptor), record = bundleRecord(state, context), kind = bundleRefresh(record, context);
    if (kind !== three.RenderObjectRefreshType.FULL) return withFrame(state, context, () => {
      // SHARED runs the complete node lifecycle, but only shared uniform
      // bindings reach the GPU. NONE consumes any shared bytes another source
      // render object already refreshed without evaluating nodes or bindings.
      if (kind === three.RenderObjectRefreshType.SHARED) {
        if (!context.beforeDone) runBefore(state);
        for (const node of state.builder.updateNodes) { frame.updateNode(node); checkPMREM(state); }
      }
      const previous = record.packet;
      const uniforms = state.bindings.map((group, index) => {
        const packet = previous.uniforms[index].slice();
        for (const member of descriptor.groups[index].uniforms) {
          const binding = group.bindings[member.binding];
          if (binding.groupNode.shared !== true) continue;
          const bytes = kind === three.RenderObjectRefreshType.SHARED
            ? readUniformBuffer(binding, member.byteLength) : uniformBuffers.get(binding)?.bytes;
          if (bytes) packet.set(bytes, member.offset);
        }
        return packet;
      });
      return record.packet = {...previous, uniforms, refreshType: kind,
        ...(record.fog ? {fogResources: record.resources} : {})};
    });
    const object = checkCapture(state, context);
    return withFrame(state, context, () => {
      let bundleGeometry = null;
      try {
      if (!context.beforeDone) runBefore(state);
      checkPipelineContext(state, object);
      assertInstances(object, state.instanceSignature);
      assertBatch(object, state.batchSignature);
      assertSkinning(object, state.skinSignature);
      assertMorphs(object, state.morphSignature);
      checkFog(state, context);
      checkEnvironment(state, context);
      checkTransmission(state);
      checkVolume(state);
      checkSSS(state);
      checkGeometry(state);
      checkAttributes(state);
      checkWireframe(state);
      // Match Renderer._renderObjectDirect: source geometry uploads occur after
      // updateBefore and before ordinary node updates and uniform uploads.
      const attributes = captureGeometry(state, frame.renderId, context.captureVertexAttribute);
      // Bundle commands retain this upload generation even when updateNodes
      // subsequently mutates the source geometry. The scene owner owns the
      // geometry/lease snapshot; compilation and shared refresh never invoke it.
      bundleGeometry = context.captureGeometry?.(attributes?.geometry ??
        (state.generated ? generatedAttributes.geometry(descriptor) : state.geometry), attributes, descriptor.attributes);
      const indirect = context.captureIndirect?.() ?? null;
      for (const node of state.builder.updateNodes) {
        frame.updateNode(node);
        checkPMREM(state);
      }
      assertInstances(object, state.instanceSignature);
      assertBatch(object, state.batchSignature);
      assertSkinning(object, state.skinSignature);
      assertMorphs(object, state.morphSignature);
      checkLights(state);
      checkPMREM(state);
      checkUniformBuffers(state);
      checkAttributes(state);
      checkPipelineContext(state, object);
      checkFog(state, context);
      checkEnvironment(state, context);
      checkTransmission(state);
      checkVolume(state);
      checkSSS(state);
      checkGeometry(state);
      checkClipping(state, context);
      checkWireframe(state);
      const textures = readTextures(state);
      const storage = readStorage(state);
      const uniforms = state.bindings.map((group, index) => {
        const layout = descriptor.groups[index], packet = new Uint8Array(layout.byteLength);
        for (const member of layout.uniforms) {
          const binding = group.bindings[member.binding];
          // Both source arrays and scalar/vector groups obey their group
          // version. Fog shares the camera's render group, which must keep its
          // accepted bytes until the source render update advances that group.
          packet.set(readUniformBuffer(binding, member.byteLength), member.offset);
        }
        return packet;
      });
      const packet = {uniforms, textures, storage, attributes, indirect,
        ...(record ? {refreshType: kind} : {}),
        ...(context.captureGeometry ? {bundleGeometry} : {}),
        ...(descriptor.wireframe ? {wireframeIndexCount: attributes.indexCount, programGeometry: attributes.geometry} : {})};
      if (record) storeBundlePacket(state, record, packet);
      return packet;
      } catch (error) {
        bundleGeometry?.release?.();
        throw error;
      }
    });
  }
  function capture(descriptor, context = {}) { return captureDraw(descriptor, context).uniforms; }
  function after(descriptor, context = {}) {
    const state = stateOf(descriptor);
    if (bundleRefresh(bundleRecord(state, context), context) === three.RenderObjectRefreshType.NONE) return;
    return withFrame(state, context, () => { for (const node of state.builder.updateAfterNodes) frame.updateAfterNode(node); });
  }
  const api = Object.freeze({inspect, compile, preparePMREM, refresh, before, capture, captureDraw, after, retainFogResources,
    hasCompilationUpdates(descriptor) {
      const state = stateOf(descriptor);
      return state.builder.updateBeforeNodes.some(node => node.isComputeNode ||
        state.shadowGraphs.roots.has(node) && !(node instanceof three.ShadowNode));
    },
    geometry(descriptor) { const state = stateOf(descriptor); return descriptor.wireframe ? wireframes.geometry(descriptor)
      : state.generated ? generatedAttributes.geometry(descriptor) : state.geometry; },
    attributeBytes(descriptor) { const state = stateOf(descriptor); return (state.generated ? generatedAttributes.byteLength(descriptor) : 0)
      + (descriptor.wireframe ? wireframes.byteLength(descriptor) : 0); },
    captureAttributes(descriptor, context = {}) {
      const state = stateOf(descriptor); assertInstances(context.object ?? state.object, state.instanceSignature); checkAttributes(state);
      checkWireframe(state);
      return captureGeometry(state, context.renderId ?? renderer.info.calls, context.captureVertexAttribute);
    },
    textures(descriptor) { return readTextures(stateOf(descriptor), true); },
    storage(descriptor) { return readStorage(stateOf(descriptor)); },
    vertexStorage(descriptor) { return stateOf(descriptor).generated ? generatedAttributes.storage(descriptor) : []; },
    release(descriptor) { for (const entry of entries) if (entry.descriptor === descriptor) { retire(entry); break; } },
    dispose() { if (disposed) return; disposed = true; for (const entry of [...entries]) retire(entry);
      for (const record of bundleGeometryRecords) releaseBundleGeometry(record);
      for (const record of fogResourceRecords) releaseFogResources(record);
      generatedAttributes.dispose(); wireframes.dispose(); effects.dispose(); },
    get diagnostics() { return Object.freeze({programs: entries.size, shaderBytes, maxPrograms, maxShaderBytes}); }});
  return api;
}
