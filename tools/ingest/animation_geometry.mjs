/**
 * Core glTF geometry -> the existing CPU/GPU animation deformer contract.
 * No fetching, renderer construction, image decoding, device or frame loop.
 * The caller supplies parsed JSON and buffers (or a lazy synchronous provider).
 *
 * Decode only mesh nodes reachable from the selected scene, retaining original
 * node/mesh/primitive/material IDs and all decoded source attributes. Shared
 * mesh instances have independent writable output arrays and original node IDs.
 * TRIANGLES/STRIP/FAN become triangle lists with source winding and degenerates
 * preserved. Missing normals produce expanded flat-shaded triangles; source
 * tangents are then ignored as glTF requires. Skinned/position-morphed flat
 * meshes request face normals rebuilt AFTER deformation on both CPU and GPU.
 * No MikkTSpace tangents are guessed.
 * Skin sets are combined vertex-major, up to the deformer's 32 influences.
 *
 * KHR_mesh_quantization adds integer positions/UVs and signed normalized
 * normals/tangents, including position/normal/tangent morph deltas. The shared
 * accessor reader expands to Float64 without adding a dequantization transform;
 * source node/inverse-bind and texture transforms already carry that operation.
 * Other extensions/codecs and nontriangle primitives need the source loader/backend:
 * an explicit error is a refusal of this opt-in route, not a compatibility pass.
 * Materials/textures/cameras/lights are NOT interpreted by this geometry layer.
 * Budgets independently bound cached accessor components and emitted components,
 * including expanded streams, repeated instances, indices and generated normals.
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes
 */
import {createGltfAccessorReader} from './animation_gltf.mjs';
import {AnimationPoseError} from './animation_runtime.mjs';
const fail = (code, message) => { throw new AnimationPoseError('GLTF_GEOMETRY_' + code, message); };
const object = (value, label) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('SHAPE', `Expected ${label} object`);
  return value;
};
const list = (value, label) => {
  if (!Array.isArray(value)) fail('SHAPE', `Expected ${label} array`);
  return value;
};
const index = (values, at, label) => {
  if (!Array.isArray(values) || !Number.isSafeInteger(at) || at < 0 || at >= values.length) fail('INDEX', `Invalid ${label} index`);
  return values[at];
};
const noExtensions = (value, label) => {
  if (value.extensions !== undefined && Object.keys(object(value.extensions, 'extensions')).length) {
    fail('EXTENSION', `Extended ${label} requires its source loader`);
  }
};
const widths = {SCALAR:1, VEC2:2, VEC3:3, VEC4:4};
const baseNames = {POSITION:'positions', NORMAL:'normals', TANGENT:'tangents'};
const unsigned = [5121, 5123], integerTypes = [5120, 5121, 5122, 5123];

/**
 * {scene=model.scene??0, maxComponents=16777216, maxPrimitives=4096}
 * -> {format, scene, primitives, diagnostics, accessorComponents, outputComponents}
 * Each primitive has {node,mesh,primitive,material,geometry,indices,attributes}.
 * attributes[name] = {width, values}; values share only that primitive's matching
 * geometry array. Material adapters select TEXCOORD_n/COLOR_0 from this table.
 */
export function decodeGltfGeometry(model, suppliedBuffers, {
  scene = model?.scene ?? 0, maxComponents = 16777216, maxPrimitives = 4096,
} = {}) {
  if (model?.asset?.version !== '2.0') fail('VERSION', 'Expected glTF 2.0');
  if (!Number.isSafeInteger(maxComponents) || maxComponents < 1 ||
      !Number.isSafeInteger(maxPrimitives) || maxPrimitives < 1 || maxPrimitives > 65536) fail('LIMIT', 'Invalid decode budgets');
  const quantized = list(model.extensionsRequired ?? [], 'required extensions').includes('KHR_mesh_quantization');
  const nodes = list(model.nodes ?? [], 'nodes');
  if (nodes.length > 65536) fail('LIMIT', 'Too many nodes');
  const selected = object(index(model.scenes, scene, 'scene'), 'scene'); noExtensions(selected, 'scene');
  const parents = new Int32Array(nodes.length).fill(-1), children = [];
  // Validate the whole forest, even disconnected cycles; do not recursively
  // traverse untrusted depth or let a repeated child multiply output work.
  for (let n = 0; n < nodes.length; n++) {
    const node = object(nodes[n], 'node');
    children[n] = list(node.children ?? [], 'children');
    for (const child of children[n]) {
      index(nodes, child, 'child');
      if (child === n || parents[child] !== -1) fail('HIERARCHY', 'Self-parent, duplicate child or multiple parents');
      parents[child] = n;
    }
  }
  const forest = [];
  for (let n = 0; n < nodes.length; n++) if (parents[n] === -1) forest.push(n);
  for (let n = 0; n < forest.length; n++) for (const child of children[forest[n]]) forest.push(child);
  if (forest.length !== nodes.length) fail('HIERARCHY', 'Cyclic node hierarchy');
  const roots = list(selected.nodes ?? [], 'scene roots'), seen = new Set(), order = [];
  for (const root of roots) {
    index(nodes, root, 'scene root');
    if (parents[root] !== -1 || seen.has(root)) fail('HIERARCHY', 'Scene roots must be distinct forest roots');
    seen.add(root); order.push(root);
  }
  for (let n = 0; n < order.length; n++) for (const child of children[order[n]]) order.push(child);
  const reader = createGltfAccessorReader(model, suppliedBuffers, {maxComponents});
  let outputComponents = 0;
  const charge = length => {
    if (!Number.isSafeInteger(length) || length < 0 || length > maxComponents - outputComponents) fail('LIMIT', 'Emitted geometry component budget exceeded');
    outputComponents += length;
  };
  function attribute(accessorIndex, name, count, morph = false) {
    const a = reader.read(accessorIndex), type = a.type, component = a.componentType;
    let valid = false;
    if (Object.hasOwn(baseNames, name)) {
      const shape = type === (name === 'TANGENT' && !morph ? 'VEC4' : 'VEC3');
      const signed = component === 5120 || component === 5122;
      // Quantized morph displacements are signed. Normal/tangent directions
      // additionally require normalization; position integers keep their literal
      // values when normalized=false. Never infer scales from accessor min/max.
      const integer = name === 'POSITION' ? (morph ? signed : integerTypes.includes(component)) : signed && a.normalized === true;
      valid = shape && (component === 5126 ? !a.normalized : quantized && integer);
    }
    else if (!morph && /^TEXCOORD_(0|[1-9]\d*)$/.test(name)) valid = type === 'VEC2' && (component === 5126 ? !a.normalized :
      (unsigned.includes(component) && a.normalized === true) || (quantized && integerTypes.includes(component)));
    else if (!morph && /^COLOR_(0|[1-9]\d*)$/.test(name)) valid = ['VEC3','VEC4'].includes(type) && (component === 5126 ? !a.normalized : unsigned.includes(component) && a.normalized === true);
    else if (!morph && /^JOINTS_(0|[1-9]\d*)$/.test(name)) valid = type === 'VEC4' && unsigned.includes(component) && !a.normalized;
    else if (!morph && /^WEIGHTS_(0|[1-9]\d*)$/.test(name)) valid = type === 'VEC4' && (component === 5126 ? !a.normalized : unsigned.includes(component) && a.normalized === true);
    else if (!morph && name.startsWith('_')) valid = Object.hasOwn(widths, type) && (component === 5126 || integerTypes.includes(component));
    if (!valid || (count !== undefined && a.count !== count)) fail('ATTRIBUTE', `Invalid ${name} accessor type, normalization or vertex count`);
    if (a.bufferView !== undefined) {
      const view = index(model.bufferViews, a.bufferView, 'attribute bufferView');
      if ((a.byteOffset ?? 0) % 4 || (view.target !== undefined && view.target !== 34962) ||
          (a.count > 1 && view.byteStride === undefined && a.values.length / a.count * ({5120:1,5121:1,5122:2,5123:2,5126:4}[component]) % 4)) {
        fail('ALIGNMENT', 'Vertex attributes require four-byte offsets/strides and ARRAY_BUFFER target');
      }
    }
    if (name.startsWith('COLOR_') || name.startsWith('WEIGHTS_')) {
      for (const v of a.values) if (v < 0 || v > 1) fail('ATTRIBUTE', `${name} values must be in [0,1]`);
    }
    if (name === 'TANGENT' && !morph) for (let i = 3; i < a.values.length; i += 4) {
      if (Math.abs(a.values[i]) !== 1) fail('ATTRIBUTE', 'Tangent W must be -1 or 1');
    }
    return {width: widths[type], values: a.values};
  }
  const primitives = [], diagnostics = [];
  for (const nodeIndex of order) {
    const node = nodes[nodeIndex];
    if (node.extensions?.EXT_mesh_gpu_instancing) fail('EXTENSION', 'Instanced transforms require the source loader');
    if (node.mesh === undefined) continue;
    const mesh = object(index(model.meshes, node.mesh, 'mesh'), 'mesh'); noExtensions(mesh, 'mesh');
    const sourcePrimitives = list(mesh.primitives, 'mesh primitives');
    if (!sourcePrimitives.length || sourcePrimitives.length > maxPrimitives - primitives.length) fail('LIMIT', 'Mesh primitive count exceeds budget');
    const targetCount = list(sourcePrimitives[0]?.targets ?? [], 'morph targets').length;
    if (targetCount > 4096 || sourcePrimitives.some(p => list(p?.targets ?? [], 'morph targets').length !== targetCount)) fail('MORPH', 'Inconsistent or excessive morph target count');
    const skin = node.skin === undefined ? null : object(index(model.skins, node.skin, 'skin'), 'skin');
    if (skin && (!Array.isArray(skin.joints) || !skin.joints.length || skin.joints.length > 65536)) fail('SKIN', 'Invalid skin joints');
    for (let primitiveIndex = 0; primitiveIndex < sourcePrimitives.length; primitiveIndex++) {
      const source = object(sourcePrimitives[primitiveIndex], 'primitive'); noExtensions(source, 'primitive');
      const mode = source.mode ?? 4;
      if (![4,5,6].includes(mode)) fail('TOPOLOGY', 'Nontriangle primitives require the source renderer');
      const sourceAttrs = object(source.attributes, 'attributes');
      if (!Object.hasOwn(sourceAttrs, 'POSITION')) fail('ATTRIBUTE', 'Primitive needs source positions');
      const attributes = Object.create(null);
      attributes.POSITION = attribute(sourceAttrs.POSITION, 'POSITION');
      const count = attributes.POSITION.values.length / 3;
      for (const [name, accessorIndex] of Object.entries(sourceAttrs)) if (name !== 'POSITION') attributes[name] = attribute(accessorIndex, name, count);
      const sourceTargets = list(source.targets ?? [], 'morph targets');
      const targets = sourceTargets.map(target => {
        const result = Object.create(null);
        for (const [name, accessorIndex] of Object.entries(object(target, 'morph target'))) {
          if (!Object.hasOwn(baseNames, name) || !Object.hasOwn(attributes, name)) fail('MORPH', 'Morph delta requires a supported base attribute');
          result[name] = attribute(accessorIndex, name, count, true);
        }
        return result;
      });
      let sourceIndices = null;
      if (source.indices !== undefined) {
        const a = reader.read(source.indices);
        if (a.type !== 'SCALAR' || ![5121,5123,5125].includes(a.componentType) || a.normalized) fail('INDEX', 'Indices must be unsigned integer SCALAR');
        if (a.bufferView !== undefined) {
          const view = index(model.bufferViews, a.bufferView, 'index bufferView');
          if (view.byteStride !== undefined || (view.target !== undefined && view.target !== 34963)) fail('INDEX', 'Indices cannot be strided or use an attribute target');
        }
        const restart = {5121:255,5123:65535,5125:4294967295}[a.componentType];
        for (const v of a.values) if (v >= count || v === restart) fail('INDEX', 'Index outside vertices or forbidden restart value');
        sourceIndices = a.values;
      }
      const extent = sourceIndices?.length ?? count;
      if (extent < 3 || (mode === 4 && extent % 3)) fail('TOPOLOGY', 'Invalid triangle index count');
      const indexCount = mode === 4 ? extent : (extent - 2) * 3;
      charge(indexCount);
      let indices = new Uint32Array(indexCount);
      const at = i => sourceIndices === null ? i : sourceIndices[i];
      if (mode === 4) for (let i = 0; i < extent; i++) indices[i] = at(i);
      else for (let i = 0; i < extent - 2; i++) {
        // Keep the source's first/provoking vertex and cyclic fan order too.
        indices[i*3] = at(mode === 5 ? i : i+1);
        indices[i*3+1] = at(mode === 5 ? i+1+(i%2) : i+2);
        indices[i*3+2] = at(mode === 5 ? i+2-(i%2) : 0);
      }
      const flat = !Object.hasOwn(attributes, 'NORMAL');
      const dynamicFlat = flat && (skin !== null || targets.some(target => target.POSITION));
      const expand = entry => {
        const length = flat ? indexCount * entry.width : entry.values.length; charge(length);
        if (!flat) return {width: entry.width, values: entry.values.slice()};
        const values = new Float64Array(length);
        for (let v = 0; v < indexCount; v++) for (let c = 0; c < entry.width; c++) values[v*entry.width+c] = entry.values[indices[v]*entry.width+c];
        return {width: entry.width, values};
      };
      for (const name of Object.keys(attributes)) {
        if (flat && name === 'TANGENT') { delete attributes[name]; diagnostics.push({node:nodeIndex,primitive:primitiveIndex,reason:'IGNORED_TANGENTS_WITHOUT_NORMALS'}); }
        else attributes[name] = expand(attributes[name]);
      }
      for (const target of targets) for (const name of Object.keys(target)) {
        if (flat && name === 'TANGENT') delete target[name]; else target[name] = expand(target[name]);
      }
      if (flat) {
        const positions = attributes.POSITION.values; charge(positions.length);
        const normals = new Float64Array(positions.length);
        for (let i = 0; i < positions.length; i += 9) {
          const ax=positions[i+3]-positions[i], ay=positions[i+4]-positions[i+1], az=positions[i+5]-positions[i+2];
          const bx=positions[i+6]-positions[i], by=positions[i+7]-positions[i+1], bz=positions[i+8]-positions[i+2];
          const x=ay*bz-az*by, y=az*bx-ax*bz, z=ax*by-ay*bx, length=Math.hypot(x,y,z);
          for (let v = 0; v < 3; v++) normals.set(length ? [x/length,y/length,z/length] : [0,0,0], i+v*3);
        }
        attributes.NORMAL = {width:3, values:normals}; indices = null;
        diagnostics.push({node:nodeIndex,primitive:primitiveIndex,reason:'GENERATED_FLAT_NORMALS'});
      }
      const geometry = {node:nodeIndex};
      if (dynamicFlat) {
        geometry.flatNormals = true;
        diagnostics.push({node:nodeIndex,primitive:primitiveIndex,reason:'DYNAMIC_FLAT_NORMALS'});
      }
      for (const [name, field] of Object.entries(baseNames)) if (attributes[name]) geometry[field] = attributes[name].values;
      geometry.morphTargets = targets.map(target => Object.fromEntries(Object.entries(target).map(([name, entry]) => [baseNames[name], entry.values])));
      const sets = Object.keys(attributes).filter(name => name.startsWith('JOINTS_')).map(name => Number(name.slice(7))).sort((a,b)=>a-b);
      const weightSets = Object.keys(attributes).filter(name => name.startsWith('WEIGHTS_'));
      if (sets.length !== weightSets.length || sets.some((set,i) => set !== i || !attributes['WEIGHTS_'+i])) fail('SKIN', 'Joint and weight sets must be paired and consecutive');
      if (skin) {
        if (!sets.length || sets.length > 8) fail('SKIN', 'Skinned geometry requires 1..8 complete joint/weight sets');
        const influences = sets.length * 4, vertices = geometry.positions.length / 3; charge(vertices*influences*2);
        const joints = new Uint32Array(vertices*influences), weights = new Float64Array(vertices*influences);
        for (let v = 0; v < vertices; v++) {
          let total = 0; const used = new Set();
          for (const set of sets) for (let k = 0; k < 4; k++) {
            const j = attributes['JOINTS_'+set].values[v*4+k], w = attributes['WEIGHTS_'+set].values[v*4+k], offset=v*influences+set*4+k;
            if (j >= skin.joints.length || (w > 0 && used.has(j))) fail('SKIN', 'Out-of-range joint or repeated nonzero influence');
            if (w > 0) used.add(j); joints[offset]=j; weights[offset]=w; total+=w;
          }
          if (Math.abs(total-1) > 1e-4) fail('SKIN', 'Skin weights must sum to one; no hidden renormalization');
        }
        Object.assign(geometry, {joints, weights, influences});
      }
      const material = source.material ?? null;
      if (material !== null) object(index(model.materials, material, 'material'), 'material');
      primitives.push({node:nodeIndex, mesh:node.mesh, primitive:primitiveIndex, material, geometry, indices, attributes});
    }
  }
  return {format:'f3d-gltf-geometry-v1', scene, primitives, diagnostics, accessorComponents:reader.components, outputComponents};
}
