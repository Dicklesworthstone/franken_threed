/** Lower EXT_mesh_gpu_instancing TRS into the existing animated node/mesh path.
 * This is a bounded compatibility path, NOT hardware-instanced submission.
 * Original node indices and hierarchy stay stable; synthetic mesh children are
 * appended. A node's children, camera and light are not duplicated. Only the
 * mesh is instanced, with world = nodeWorld * instance(T * R * S).
 *
 * No I/O, GPU, clock, new accessor decoder or source mutation. The returned JSON
 * structurally shares unchanged data; consume it synchronously or snapshot it
 * before yielding. instanceOrigins maps synthetic pose-node IDs to source IDs.
 * Skinned nodes and custom per-instance shader attributes need the source route.
 * https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_mesh_gpu_instancing
 */
import {createGltfAccessorReader} from './animation_gltf.mjs';
import {AnimationPoseError} from './animation_runtime.mjs';
const EXT = 'EXT_mesh_gpu_instancing';
const fail = (code, message) => { throw new AnimationPoseError('GLTF_INSTANCING_' + code, message); };
const object = (value, label) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('SHAPE', `Expected ${label} object`);
  return value;
};
const list = (value, label) => {
  if (!Array.isArray(value)) fail('SHAPE', `Expected ${label} array`);
  return value;
};
const integer = (value, label, min = 0, max = Number.MAX_SAFE_INTEGER) => {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail('LIMIT', `Invalid ${label}`);
  return value;
};
function indexed(values, index, label) {
  list(values, label); integer(index, label, 0, values.length - 1);
  return object(values[index], label);
}
function fields(value, allowed, label) {
  object(value, label);
  for (const key of Object.keys(value)) if (!allowed.includes(key)) fail('UNSUPPORTED', `Unsupported ${label} field: ${key}`);
}

/** maxInstances bounds aggregate synthetic nodes, not each batch separately.
 * maxComponents independently bounds decoded instance accessors and expanded
 * TRS/rest-weight/keyframe components. Ordinary geometry has its own decoder
 * budget, including every expanded drawable; no large batch is silently cut.
 */
export function expandGltfInstances(model, suppliedBuffers, {
  maxInstances = 4096, maxComponents = 16777216,
} = {}) {
  integer(maxInstances, 'instance budget', 1, 65536);
  integer(maxComponents, 'component budget', 1);
  if (model?.asset?.version !== '2.0') fail('VERSION', 'Expected glTF 2.0');
  const original = list(model.nodes ?? [], 'nodes');
  if (original.length > 65536) fail('LIMIT', 'Too many pose nodes');
  const plans = [], byNode = new Map();
  let instanceCount = 0, expandedComponents = 0;
  function charge(count) {
    integer(count, 'expanded component count');
    if (count > maxComponents - expandedComponents) fail('LIMIT', 'Expanded instance data exceeds component budget');
    expandedComponents += count;
  }
  // Validate descriptor counts and all expansion sizes before decoding instance
  // arrays or building synthetic nodes. Huge declared counts cannot allocate.
  for (let node = 0; node < original.length; node++) {
    const source = object(original[node], 'node');
    const extension = source.extensions?.[EXT];
    if (extension === undefined) continue;
    fields(extension, ['attributes', 'extras', 'extensions'], 'instancing extension');
    if (Object.keys(object(extension.extensions ?? {}, 'extensions')).length) fail('UNSUPPORTED', 'Extended instance behavior needs the source route');
    const mesh = indexed(model.meshes, source.mesh, 'instanced mesh');
    if (source.skin !== undefined) fail('UNSUPPORTED', 'Skinned instancing needs the source skinning route');
    const attributes = object(extension.attributes, 'instance attributes'), entries = Object.entries(attributes);
    if (!entries.length) fail('SHAPE', 'Instancing needs at least one attribute');
    let count;
    for (const [semantic, index] of entries) {
      if (!['TRANSLATION', 'ROTATION', 'SCALE'].includes(semantic)) fail('UNSUPPORTED', `Instance attribute needs the source route: ${semantic}`);
      const a = indexed(model.accessors, index, 'instance accessor');
      integer(a.count, 'instance count', 1, maxInstances);
      if (count !== undefined && count !== a.count) fail('COUNT', 'All instance attributes must have the same count');
      count = a.count;
      const rotation = semantic === 'ROTATION';
      if (a.type !== (rotation ? 'VEC4' : 'VEC3') ||
          !(a.componentType === 5126 ? a.normalized === undefined || a.normalized === false :
            rotation && [5120, 5122].includes(a.componentType) && a.normalized === true)) {
        fail('ATTRIBUTE', `Invalid ${semantic} accessor type or normalization`);
      }
    }
    if (count > maxInstances - instanceCount || count > 65536 - original.length - instanceCount) fail('LIMIT', 'Aggregate instance/pose-node budget exceeded');
    const primitives = list(mesh.primitives, 'mesh primitives');
    if (!primitives.length) fail('SHAPE', 'Instanced mesh needs primitives');
    const width = list(primitives[0]?.targets ?? [], 'morph targets').length;
    if (width > 4096 || primitives.some(p => list(p?.targets ?? [], 'morph targets').length !== width)) fail('SHAPE', 'Inconsistent or excessive instance morph targets');
    if (source.weights !== undefined) {
      const weights = list(source.weights, 'node morph weights');
      if (weights.length !== width || weights.some(v => typeof v !== 'number' || !Number.isFinite(v))) fail('SHAPE', 'Invalid instance morph weights');
    }
    charge(count * (10 + width));
    const first = original.length + instanceCount;
    const plan = {node, source, count, first, entries};
    plans.push(plan); byNode.set(node, plan); instanceCount += count;
  }
  const declarations = {};
  for (const key of ['extensionsUsed', 'extensionsRequired']) if (model[key] !== undefined) {
    const names = list(model[key], key);
    if (names.some(name => typeof name !== 'string')) fail('SHAPE', 'Extension names must be strings');
    if (names.includes(EXT)) declarations[key] = names.filter(name => name !== EXT);
  }
  if (!plans.length) return Object.freeze({json: Object.keys(declarations).length ? {...model, ...declarations} : model,
    instanceOrigins: Object.freeze({}), instanceCount: 0, accessorComponents: 0, expandedComponents: 0});

  // Appending nodes must never make an originally invalid node reference valid.
  // Keep every source reference inside the original node domain; generated IDs
  // are reserved exclusively for this lowering step.
  const sourceNode = value => integer(value, 'source node reference', 0, original.length - 1);
  for (const node of original) for (const child of list(node.children ?? [], 'children')) sourceNode(child);
  for (const scene of list(model.scenes ?? [], 'scenes')) for (const root of list(scene.nodes ?? [], 'scene roots')) sourceNode(root);
  for (const skin of list(model.skins ?? [], 'skins')) {
    for (const joint of list(skin.joints, 'skin joints')) sourceNode(joint);
    if (skin.skeleton !== undefined) sourceNode(skin.skeleton);
  }

  // Morph animation targets the mesh, not its transform-only parent. Duplicate
  // just these channels, preserving clip/sampler identity and channel ordering.
  // Count their eventual decoded arrays now, before the downstream pose decoder
  // could multiply one large accessor into thousands of copies.
  const animations = list(model.animations ?? [], 'animations');
  if (animations.length > 4096) fail('LIMIT', 'Too many animation clips');
  const animationPlans = animations.map(animation => {
    object(animation, 'animation');
    const channels = list(animation.channels, 'animation channels');
    let total = 0, changed = false;
    for (const channel of channels) {
      object(channel, 'animation channel');
      const target = channel.target;
      if (target?.node !== undefined) sourceNode(target.node);
      const plan = target?.path === 'weights' ? byNode.get(target.node) : undefined;
      total += plan?.count ?? 1;
      if (total > 262144) fail('LIMIT', 'Expanded animation channel budget exceeded');
      if (!plan) continue;
      changed = true;
      const sampler = indexed(animation.samplers, channel.sampler, 'animation sampler');
      const times = indexed(model.accessors, sampler.input, 'animation input');
      const values = indexed(model.accessors, sampler.output, 'animation output');
      integer(times.count, 'animation input count', 1); integer(values.count, 'animation output count', 1);
      if (times.type !== 'SCALAR' || values.type !== 'SCALAR') fail('ATTRIBUTE', 'Morph channels require scalar accessors');
      charge(plan.count * (times.count + values.count));
    }
    return {animation, channels, changed};
  });
  const reader = createGltfAccessorReader(model, suppliedBuffers, {maxComponents});
  for (const plan of plans) {
    plan.attributes = Object.fromEntries(plan.entries.map(([semantic, index]) => [semantic, reader.read(index)]));
    const rotation = plan.attributes.ROTATION;
    if (rotation) for (let i = 0; i < plan.count; i++) {
      const q = rotation.values.subarray(i * 4, i * 4 + 4), norm = Math.hypot(...q);
      // The pose runtime tolerates 1e-3 for FLOAT rotations and 0.01 for
      // quantized rotations. Normalize only quantized values when materializing
      // their FLOAT node representation, not arbitrary malformed quaternions.
      if (!(norm > 0) || Math.abs(norm - 1) > (rotation.componentType === 5126 ? 1e-3 : 0.01)) fail('ROTATION', 'Instance quaternion must be normalized');
    }
  }
  const nodes = original.slice(), origins = {};
  for (const plan of plans) {
    const {node, source, count, first, attributes} = plan;
    const children = source.children === undefined ? [] : list(source.children, 'children').slice();
    const extensions = {...source.extensions}; delete extensions[EXT];
    const parent = {...source, extensions, children}; delete parent.mesh; delete parent.weights;
    nodes[node] = parent;
    for (let i = 0; i < count; i++) {
      const child = {mesh: source.mesh};
      for (const [semantic, field, width] of [['TRANSLATION', 'translation', 3], ['ROTATION', 'rotation', 4], ['SCALE', 'scale', 3]]) {
        const accessor = attributes[semantic]; if (!accessor) continue;
        child[field] = Array.from(accessor.values.subarray(i * width, (i + 1) * width));
        if (semantic === 'ROTATION' && accessor.componentType !== 5126) {
          const norm = Math.hypot(...child.rotation); child.rotation = child.rotation.map(v => v / norm);
        }
      }
      if (source.weights !== undefined) child.weights = source.weights.slice();
      if (source.name !== undefined) child.name = source.name;
      const generated = first + i; nodes.push(child); children.push(generated);
      origins[generated] = Object.freeze({node, instance: i});
    }
  }
  const json = {...model, ...declarations, nodes};
  if (animationPlans.some(p => p.changed)) json.animations = animationPlans.map(({animation, channels, changed}) => {
    if (!changed) return animation;
    const output = [];
    for (const channel of channels) {
      const target = channel.target, plan = target?.path === 'weights' ? byNode.get(target.node) : undefined;
      if (!plan) output.push(channel);
      else for (let i = 0; i < plan.count; i++) output.push({...channel, target: {...target, node: plan.first + i}});
    }
    return {...animation, channels: output};
  });
  return Object.freeze({json, instanceOrigins: Object.freeze(origins), instanceCount,
    accessorComponents: reader.components, expandedComponents});
}
