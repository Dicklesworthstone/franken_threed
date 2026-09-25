/** Native r186 Mesh/SkinnedMesh -> the existing animation pose/geometry contract.
 * Captures source state; never advances AnimationMixer, evaluates vertices per
 * frame, changes source geometry, or owns a Skeleton/Scene/device. Static source
 * edits follow BufferAttribute.needsUpdate and require a new binding.
 */
export class ThreeDeformationError extends Error {
  constructor(code, message) {
    super(`THREE_DEFORMATION_${code}: ${message}`);
    this.name = 'ThreeDeformationError';
    this.code = `THREE_DEFORMATION_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeDeformationError(code, message); };
const integer = (n, low, high, label) => {
  if (!Number.isSafeInteger(n) || n < low || n > high) fail('LIMIT', `Invalid ${label}`);
  return n;
};
const finite = (n, label) => {
  if (typeof n !== 'number' || !Number.isFinite(n) || !Number.isFinite(Math.fround(n)))
    fail('VALUE', `${label} must fit finite f32`);
  return n;
};
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const fields = [['position', 'positions', 3], ['normal', 'normals', 3], ['tangent', 'tangents', 4]];
const getters = ['getX', 'getY', 'getZ', 'getW'];
const defaults = {maxVertices: 1048576, maxJoints: 1024, maxMorphTargets: 256, maxComponents: 16777216};

export function hasThreeDeformation(source) {
  return source?.isSkinnedMesh === true || Object.values(source?.geometry?.morphAttributes ?? {}).some(a => a?.length);
}

function configuration(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected binding options');
  for (const key of Object.keys(options)) if (key !== 'three' && !Object.hasOwn(defaults, key)) fail('OPTIONS', `Unknown binding option: ${key}`);
  const limits = {...defaults, ...options};
  for (const key of Object.keys(defaults)) integer(limits[key], 1, 16777216, key);
  if (limits.maxMorphTargets > 4096) fail('LIMIT', 'At most 4096 morph targets are supported');
  if (limits.three?.REVISION !== '186') fail('SOURCE', 'Supply the pinned Three.js r186 module');
  return limits;
}
function inspect(source, limits) {
  const {three} = limits, g = source?.geometry;
  if (typeof three.Mesh !== 'function' || !(source instanceof three.Mesh) || source.isInstancedMesh || source.isBatchedMesh)
    fail('SOURCE', 'Expected one native, non-instanced Mesh or SkinnedMesh');
  if (!(g instanceof three.BufferGeometry) || g.isInstancedBufferGeometry || !g.attributes || !g.morphAttributes)
    fail('GEOMETRY', 'Expected native BufferGeometry');
  const signature = [g, g.morphTargetsRelative], attributes = new Map();
  let components = 0;
  const charge = n => {
    components += n;
    if (!Number.isSafeInteger(components) || components > limits.maxComponents) fail('LIMIT', 'Source deformation component budget exceeded');
  };
  function attribute(a, width, count, label) {
    if (!a) fail('GEOMETRY', `Missing ${label}`);
    const interleaved = a.isInterleavedBufferAttribute === true, owner = interleaved ? a.data : a;
    const C = interleaved ? three.InterleavedBufferAttribute : a.isFloat16BufferAttribute ? three.Float16BufferAttribute : three.BufferAttribute;
    if (typeof C !== 'function' || !(a instanceof C) || a.itemSize !== width || typeof a.normalized !== 'boolean' ||
        !ArrayBuffer.isView(owner?.array) || owner.array instanceof DataView)
      fail('GEOMETRY', `Invalid ${label} layout`);
    integer(a.count, 1, label === 'index' ? limits.maxComponents : limits.maxVertices, `${label} count`);
    if (count !== null && a.count !== count) fail('GEOMETRY', `${label} count differs from positions`);
    integer(owner.version, 0, Number.MAX_SAFE_INTEGER, `${label} version`);
    const proto = C.prototype;
    for (let i = 0; i < width; i++) if (typeof a[getters[i]] !== 'function' || a[getters[i]] !== proto[getters[i]])
      fail('HOOK', `Custom ${label} component readers are not admitted`);
    const upload = interleaved ? three.InterleavedBuffer.prototype.onUploadCallback : three.BufferAttribute.prototype.onUploadCallback;
    if (owner.onUploadCallback !== upload) fail('HOOK', 'Custom attribute upload callbacks are not admitted');
    const stride = interleaved ? integer(owner.stride, width, 16777216, 'attribute stride') : width;
    const offset = interleaved ? integer(a.offset, 0, stride - width, 'attribute offset') : 0;
    if ((a.count - 1) * stride + offset + width > owner.array.length) fail('GEOMETRY', `${label} storage is truncated`);
    charge(a.count * width);
    signature.push(a, owner, owner.array, owner.version, width, a.count, a.normalized, stride, offset);
    attributes.set(label, a);
    return a;
  }
  const position = attribute(g.attributes.position, 3, null, 'position'), count = position.count;
  for (const [name, , width] of fields.slice(1)) if (g.attributes[name]) attribute(g.attributes[name], width, count, name);
  if (g.attributes.uv) attribute(g.attributes.uv, 2, count, 'uv');
  if (g.attributes.color) {
    if (![3, 4].includes(g.attributes.color.itemSize)) fail('GEOMETRY', 'Vertex colors require RGB or RGBA');
    attribute(g.attributes.color, g.attributes.color.itemSize, count, 'color');
  }
  if (g.index) {
    attribute(g.index, 1, null, 'index');
    if (g.index.normalized) fail('GEOMETRY', 'Index attributes cannot be normalized');
  }
  const morph = g.morphAttributes;
  for (const [name, list] of Object.entries(morph)) {
    if (!Array.isArray(list)) fail('GEOMETRY', 'Morph attributes must be arrays');
    if (list.length && !['position', 'normal'].includes(name)) fail('GEOMETRY', `Unsupported morph channel: ${name}`);
  }
  const targets = Math.max(morph.position?.length ?? 0, morph.normal?.length ?? 0);
  integer(targets, 0, limits.maxMorphTargets, 'morph target count');
  if (typeof g.morphTargetsRelative !== 'boolean') fail('GEOMETRY', 'Expected boolean morphTargetsRelative');
  signature.push(targets);
  for (const name of ['position', 'normal']) {
    const list = morph[name] ?? [];
    if (list.length && (list.length !== targets || !g.attributes[name])) fail('GEOMETRY', 'Morph channels require matching targets and base attributes');
    signature.push(name, list.length);
    for (let i = 0; i < list.length; i++) attribute(list[i], 3, count, `${name}/${i}`);
  }
  const skinned = source.isSkinnedMesh === true, skeleton = skinned ? source.skeleton : null;
  signature.push(skinned, skeleton);
  if (skinned) {
    if (typeof three.SkinnedMesh !== 'function' || !(source instanceof three.SkinnedMesh) ||
        typeof three.Skeleton !== 'function' || !(skeleton instanceof three.Skeleton) ||
        !Array.isArray(skeleton.bones) || !Array.isArray(skeleton.boneInverses)) fail('SKIN', 'Expected a native bound Skeleton');
    const joints = integer(skeleton.bones.length, 1, limits.maxJoints, 'joint count');
    if (skeleton.boneInverses.length !== joints || !['attached', 'detached'].includes(source.bindMode)) fail('SKIN', 'Invalid inverse binds or bind mode');
    signature.push(joints);
    for (let i = 0; i < joints; i++) {
      if (!(skeleton.bones[i] instanceof three.Bone)) fail('SKIN', 'Missing native bone');
      signature.push(skeleton.bones[i], skeleton.boneInverses[i]);
    }
    attribute(g.attributes.skinIndex, 4, count, 'skinIndex');
    attribute(g.attributes.skinWeight, 4, count, 'skinWeight');
    if (g.attributes.skinIndex.normalized) fail('SKIN', 'Skin indices cannot be normalized');
    charge(joints * 16);
  }
  return {signature, attributes, count, targets, skeleton, components};
}

/** Metadata-only validation, useful before any GPU allocation or source scan. */
export function inspectThreeDeformation(source, options) {
  const s = inspect(source, configuration(options));
  return Object.freeze({signature: Object.freeze(s.signature), vertexCount: s.count, targetCount: s.targets,
    jointCount: s.skeleton?.bones.length ?? 0, components: s.components});
}
function matrix(value, out, label) {
  const e = value?.elements;
  if (!e || e.length !== 16) fail('SKIN', `Missing ${label}`);
  for (let i = 0; i < 16; i++) out[i] = finite(e[i], label);
  if (out[3] !== 0 || out[7] !== 0 || out[11] !== 0 || out[15] !== 1) fail('SKIN', `${label} must be affine`);
}
function multiply(a, b, out) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let v = 0;
    for (let k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = finite(v, 'Skin matrix product');
  }
}

export function createThreeDeformationBinding(source, options) {
  const limits = configuration(options), shape = inspect(source, limits), g = source.geometry;
  let disposed = false, invalidated = false, busy = false;
  const pose = {nodeCount: 1, version: 0, disposed: false, sample() { fail('CLOCK', 'Advance the source mixer, then capture the binding'); },
    worldMatrices: new Float64Array(16), jointMatrices: new Float64Array((shape.skeleton?.bones.length ?? 0) * 16),
    morphWeights: new Float64Array(shape.targets), morphOffsets: new Uint32Array([0, shape.targets]),
    instances: shape.skeleton ? [{node: 0, offset: 0, jointCount: shape.skeleton.bones.length}] : []};
  const read = (name, width) => {
    const a = shape.attributes.get(name);
    if (!a) return undefined;
    const out = new Float32Array(a.count * width);
    for (let i = 0; i < a.count; i++) for (let c = 0; c < width; c++) out[i * width + c] = finite(a[getters[c]](i), name);
    return out;
  };
  const geometry = {node: 0, positions: read('position', 3), morphTargets: []};
  for (const [name, field, width] of fields.slice(1)) if (shape.attributes.has(name)) geometry[field] = read(name, width);
  for (let i = 0; i < shape.targets; i++) {
    const target = {};
    for (const [name, field] of fields.slice(0, 2)) {
      const values = read(`${name}/${i}`, 3);
      if (values) {
        if (!g.morphTargetsRelative) for (let j = 0; j < values.length; j++) values[j] = finite(values[j] - geometry[field][j], 'Absolute morph delta');
        target[field] = values;
      }
    }
    geometry.morphTargets.push(target);
  }
  if (shape.skeleton) {
    geometry.joints = read('skinIndex', 4); geometry.weights = read('skinWeight', 4);
    for (let i = 0; i < shape.count; i++) {
      let total = 0;
      for (let k = 0; k < 4; k++) {
        const j = i * 4 + k;
        integer(geometry.joints[j], 0, shape.skeleton.bones.length - 1, 'skin index');
        if (geometry.weights[j] < 0) fail('SKIN', 'Skin weights must be nonnegative');
        total += geometry.weights[j];
      }
      if (Math.abs(total - 1) > 1e-4) fail('SKIN', 'Skin weights must sum to one; normalize them explicitly on the source');
    }
  }
  const indices = read('index', 1);
  if (indices) for (const index of indices) integer(index, 0, shape.count - 1, 'vertex index');
  const colors = read('color', g.attributes.color?.itemSize ?? 3);
  let vertexColors = colors;
  if (colors && g.attributes.color.itemSize === 3) {
    vertexColors = new Float32Array(shape.count * 4);
    for (let i = 0; i < shape.count; i++) { vertexColors.set(colors.subarray(i * 3, i * 3 + 3), i * 4); vertexColors[i * 4 + 3] = 1; }
  }
  const surface = Object.freeze({indices: indices ? Uint32Array.from(indices) : null, texCoords: read('uv', 2) ?? null, vertexColors: vertexColors ?? null});
  const world = new Float64Array(16), palette = new Float64Array(pose.jointMatrices.length), morphWeights = new Float64Array(shape.targets);
  const bind = new Float64Array(16), inverse = new Float64Array(16), bone = new Float64Array(16), rest = new Float64Array(16);
  const ab = new Float64Array(16), abc = new Float64Array(16), result = new Float64Array(16);
  const onDispose = () => { invalidated = true; };
  const check = () => {
    if (disposed) fail('DISPOSED', 'Source deformation binding is disposed');
    if (invalidated || !same(shape.signature, inspect(source, limits).signature)) fail('PREPARE', 'Source deformation layout/content changed; create a new binding through prepare()');
  };
  const binding = Object.freeze({source, pose, geometry, surface,
    signature: Object.freeze(shape.signature), vertexCount: shape.count, indexCount: indices?.length ?? 0,
    get disposed() { return disposed; },
    matches() { return !disposed && !invalidated && same(shape.signature, inspect(source, limits).signature); },
    check,
    capture() {
      if (busy) fail('REENTRANT', 'Source pose capture cannot be reentered');
      busy = true;
      try {
        check();
        if (pose.version === Number.MAX_SAFE_INTEGER) fail('LIMIT', 'Source pose version exhausted');
        matrix(source.matrixWorld, world, 'mesh world matrix');
        const weights = source.morphTargetInfluences;
        if (shape.targets && (!Array.isArray(weights) || weights.length !== shape.targets)) fail('MORPH', 'Morph influences must match the prepared targets');
        for (let i = 0; i < shape.targets; i++) morphWeights[i] = finite(weights[i], 'Morph influence');
        if (shape.skeleton) {
          matrix(source.bindMatrix, bind, 'bind matrix'); matrix(source.bindMatrixInverse, inverse, 'inverse bind matrix');
          for (let i = 0; i < shape.skeleton.bones.length; i++) {
            matrix(shape.skeleton.bones[i].matrixWorld, bone, 'bone world matrix');
            matrix(shape.skeleton.boneInverses[i], rest, 'bone inverse');
            multiply(bone, rest, ab); multiply(inverse, ab, abc); multiply(abc, bind, result);
            palette.set(result, i * 16);
          }
        }
        check();
        pose.worldMatrices.set(world); pose.jointMatrices.set(palette); pose.morphWeights.set(morphWeights); pose.version++;
        return binding;
      } finally { busy = false; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during source capture');
      if (!disposed) { disposed = true; pose.disposed = true; g.removeEventListener('dispose', onDispose); }
    },
  });
  binding.capture();
  g.addEventListener('dispose', onDispose);
  return binding;
}
