/**
 * CPU mesh deformation driven by a packed animation player. The application's
 * loader supplies decoded, tightly packed geometry; this is not a model loader
 * or renderer. No DOM, GPU, Wasm, timers or network services are initialized.
 *
 * Morph POSITION/NORMAL/TANGENT deltas are applied before linear blend skinning.
 * Skin indices address the node instance's mesh-local palette, NOT glTF node
 * indices. Directions use the weighted skin matrix with w=0, as in the pinned
 * Three.js skinnormal_vertex chunk; they are not normalized here. The renderer
 * still applies its normal/model transforms and normalizes. Tangent w survives.
 *
 * Outputs are mesh-local Float32 arrays with stable identities. worldMatrix is
 * the matching node-to-world transform; bounds encloses the published positions.
 * Use these buffers for an ordinary mesh: applying morphing/skinning to them
 * again would double-deform the geometry. Material/UV/index data stays with the
 * loader. This is CPU execution, not a GPU acceleration or full renderer claim.
 */
import {AnimationPoseError} from './animation_runtime.mjs';
const fail = (code, message) => { throw new AnimationPoseError(code, message); };
const finite = (value, label) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail('ANIMATION_DEFORM_VALUE', `${label} must be finite`);
  return value;
};
function storage(value, length, label) {
  if (!ArrayBuffer.isView(value) || value instanceof DataView ||
      !(value.buffer instanceof ArrayBuffer) || value.buffer.resizable || value.length !== length) {
    fail('ANIMATION_DEFORM_STORAGE', `${label} must have fixed, unshared, attached storage`);
  }
  try { new Uint8Array(value.buffer, 0, 0); }
  catch { fail('ANIMATION_DEFORM_STORAGE', `${label} is detached`); }
}

/**
 * createAnimationDeformer(pose, {node, positions, normals?, tangents?,
 *   morphTargets?: [{positions?, normals?, tangents?}],
 *   joints?, weights?, influences?: 4}, {maxComponents?: 16777216})
 *
 * positions/normals and morph deltas are XYZ; base tangents are XYZW. joints
 * and weights are vertex-major, with 1..32 influences per vertex. Weights must
 * be nonnegative and sum to one (1e-4 tolerance); they are never renormalized.
 * Decode normalized glTF integer weights before passing them here.
 *
 * Geometry is copied at construction. update() reads the borrowed pose, but
 * never advances or disposes it. A failed update publishes nothing. Evaluation
 * reuses construction-time scratch rather than allocating per-vertex objects.
 */
export function createAnimationDeformer(pose, geometry, {maxComponents = 16777216} = {}) {
  if (!pose || !Number.isSafeInteger(pose.nodeCount) || pose.nodeCount < 0 ||
      !Array.isArray(pose.instances) || typeof pose.sample !== 'function') {
    fail('ANIMATION_DEFORM_PLAYER', 'Expected an animation pose player');
  }
  if (!geometry || typeof geometry !== 'object') fail('ANIMATION_DEFORM_GEOMETRY', 'Expected decoded geometry');
  if (!Number.isSafeInteger(maxComponents) || maxComponents < 1) fail('ANIMATION_DEFORM_LIMIT', 'Invalid component budget');
  const node = geometry.node;
  if (!Number.isInteger(node) || node < 0 || node >= pose.nodeCount) fail('ANIMATION_DEFORM_NODE', 'Invalid mesh node');
  const length = geometry.positions?.length;
  if (!Number.isSafeInteger(length) || length < 3 || length % 3 !== 0 || length > maxComponents) {
    fail('ANIMATION_DEFORM_GEOMETRY', 'Positions must contain a bounded, nonempty XYZ array');
  }
  const vertexCount = length / 3;
  const palette = pose.jointMatrices, morphWeights = pose.morphWeights, worldMatrices = pose.worldMatrices;
  const paletteLength = palette?.length, morphLength = morphWeights?.length;
  storage(palette, paletteLength, 'Joint palette');
  storage(morphWeights, morphLength, 'Morph weights');
  storage(worldMatrices, pose.nodeCount * 16, 'World matrices');
  storage(pose.morphOffsets, pose.nodeCount + 1, 'Morph offsets');
  const morphStart = pose.morphOffsets[node], morphEnd = pose.morphOffsets[node + 1];
  if (!Number.isSafeInteger(morphStart) || !Number.isSafeInteger(morphEnd) ||
      morphStart < 0 || morphEnd < morphStart || morphEnd > morphLength) {
    fail('ANIMATION_DEFORM_MORPH', 'Invalid node morph range');
  }
  const targets = geometry.morphTargets ?? [];
  if (!Array.isArray(targets) || targets.length !== morphEnd - morphStart) {
    fail('ANIMATION_DEFORM_MORPH', 'Geometry must provide every morph target for this node');
  }
  let consumed = 0;
  function copy(value, count, label) {
    if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== count) {
      fail('ANIMATION_DEFORM_GEOMETRY', `${label} requires ${count} components`);
    }
    if (ArrayBuffer.isView(value)) storage(value, count, label);
    consumed += count;
    if (consumed > maxComponents) fail('ANIMATION_DEFORM_LIMIT', 'Geometry component budget exceeded');
    return Float64Array.from(value, item => finite(item, label));
  }
  const fields = [{name: 'positions', width: 3, base: copy(geometry.positions, length, 'Positions')}];
  if (geometry.normals !== undefined) fields.push({name: 'normals', width: 3, base: copy(geometry.normals, length, 'Normals')});
  if (geometry.tangents !== undefined) {
    const base = copy(geometry.tangents, vertexCount * 4, 'Tangents');
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      if (Math.abs(base[vertex * 4 + 3]) !== 1) fail('ANIMATION_DEFORM_GEOMETRY', 'Tangent handedness must be -1 or 1');
    }
    fields.push({name: 'tangents', width: 4, base});
  }
  const fieldNames = new Set(fields.map(field => field.name));
  const deltas = targets.map(target => {
    if (!target || typeof target !== 'object' || Array.isArray(target)) fail('ANIMATION_DEFORM_MORPH', 'Invalid morph target');
    const result = {};
    for (const name of Object.keys(target)) {
      if (!fieldNames.has(name)) fail('ANIMATION_DEFORM_MORPH', `Morph target ${name} lacks a supported base attribute`);
      result[name] = copy(target[name], length, `Morph ${name}`);
    }
    return result;
  });
  const found = pose.instances.filter(instance => instance.node === node);
  if (found.length > 1) fail('ANIMATION_DEFORM_SKIN', 'Duplicate palette instances for mesh node');
  const skin = found.length ? {...found[0]} : null;
  let joints = null, weights = null, influences = 0;
  if (skin) {
    if (!Number.isSafeInteger(skin.offset) || skin.offset < 0 || skin.offset % 16 ||
        !Number.isSafeInteger(skin.jointCount) || skin.jointCount < 1 ||
        skin.offset + skin.jointCount * 16 > paletteLength) fail('ANIMATION_DEFORM_SKIN', 'Invalid instance palette range');
    influences = geometry.influences ?? 4;
    if (!Number.isInteger(influences) || influences < 1 || influences > 32) fail('ANIMATION_DEFORM_SKIN', 'Expected 1..32 influences per vertex');
    joints = copy(geometry.joints, vertexCount * influences, 'Joint indices');
    weights = copy(geometry.weights, vertexCount * influences, 'Skin weights');
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      let total = 0;
      for (let k = 0; k < influences; k++) {
        const i = vertex * influences + k;
        if (!Number.isInteger(joints[i]) || joints[i] < 0 || joints[i] >= skin.jointCount) fail('ANIMATION_DEFORM_SKIN', 'Joint index exceeds instance palette');
        if (weights[i] < 0) fail('ANIMATION_DEFORM_SKIN', 'Skin weights must be nonnegative');
        total += weights[i];
      }
      if (Math.abs(total - 1) > 1e-4) fail('ANIMATION_DEFORM_SKIN', 'Skin weights must sum to one');
    }
  } else if (geometry.joints !== undefined || geometry.weights !== undefined || geometry.influences !== undefined) {
    fail('ANIMATION_DEFORM_SKIN', 'Skin attributes require a skinned node instance');
  }
  // Scratch and publication have disjoint storage. Output size is bounded by
  // the already charged base attributes, rather than caller-selected lengths.
  const output = {positions: null, normals: null, tangents: null};
  for (const field of fields) {
    field.scratch = new Float32Array(field.base.length);
    field.output = output[field.name] = new Float32Array(field.base.length);
  }
  const worldMatrix = new Float64Array(16), nextWorld = new Float64Array(16);
  const minimum = new Float32Array(3), maximum = new Float32Array(3);
  const nextMin = new Float32Array(3), nextMax = new Float32Array(3);
  const bounds = Object.freeze({min: minimum, max: maximum});
  let version = -1, poseVersion = -1, disposed = false, busy = false;
  function checkOutputs() {
    for (const field of fields) storage(field.output, field.base.length, `Output ${field.name}`);
    storage(worldMatrix, 16, 'Output world matrix');
    storage(minimum, 3, 'Output minimum'); storage(maximum, 3, 'Output maximum');
  }
  function update() {
    if (disposed) fail('ANIMATION_DEFORM_DISPOSED', 'Mesh deformer has been disposed');
    if (busy) fail('ANIMATION_REENTRANT', 'Mesh deformation cannot be reentered');
    busy = true;
    try {
      if (pose.disposed) fail('ANIMATION_DISPOSED', 'Animation player has been disposed');
      const nextPoseVersion = pose.version;
      if (!Number.isSafeInteger(nextPoseVersion) || nextPoseVersion < 0) fail('ANIMATION_DEFORM_PLAYER', 'Invalid pose version');
      storage(palette, paletteLength, 'Joint palette'); storage(morphWeights, morphLength, 'Morph weights');
      storage(worldMatrices, pose.nodeCount * 16, 'World matrices'); checkOutputs();
      for (let i = 0; i < 16; i++) nextWorld[i] = finite(worldMatrices[node * 16 + i], 'World matrix');
      if (nextWorld[3] !== 0 || nextWorld[7] !== 0 || nextWorld[11] !== 0 || nextWorld[15] !== 1) fail('ANIMATION_DEFORM_SKIN', 'World matrix must be affine');
      for (let i = morphStart; i < morphEnd; i++) finite(morphWeights[i], 'Morph weight');
      if (skin) for (let joint = 0; joint < skin.jointCount; joint++) {
        const o = skin.offset + joint * 16;
        for (let c = 0; c < 16; c++) finite(palette[o + c], 'Joint palette');
        if (palette[o + 3] !== 0 || palette[o + 7] !== 0 || palette[o + 11] !== 0 || palette[o + 15] !== 1) fail('ANIMATION_DEFORM_SKIN', 'Joint matrices must be affine');
      }
      nextMin.fill(Infinity); nextMax.fill(-Infinity);
      for (const field of fields) for (let vertex = 0; vertex < vertexCount; vertex++) {
        const a = vertex * field.width, d = vertex * 3;
        let x = field.base[a], y = field.base[a + 1], z = field.base[a + 2];
        for (let target = 0; target < deltas.length; target++) {
          const values = deltas[target][field.name], weight = morphWeights[morphStart + target];
          if (!values || weight === 0) continue;
          x += weight * values[d]; y += weight * values[d + 1]; z += weight * values[d + 2];
        }
        if (skin) {
          let sx = 0, sy = 0, sz = 0;
          const w = field.name === 'positions' ? 1 : 0;
          for (let k = 0; k < influences; k++) {
            const i = vertex * influences + k, weight = weights[i];
            if (weight === 0) continue;
            const o = skin.offset + joints[i] * 16;
            sx += weight * (palette[o] * x + palette[o + 4] * y + palette[o + 8] * z + w * palette[o + 12]);
            sy += weight * (palette[o + 1] * x + palette[o + 5] * y + palette[o + 9] * z + w * palette[o + 13]);
            sz += weight * (palette[o + 2] * x + palette[o + 6] * y + palette[o + 10] * z + w * palette[o + 14]);
          }
          x = sx; y = sy; z = sz;
        }
        field.scratch[a] = x; field.scratch[a + 1] = y; field.scratch[a + 2] = z;
        if (field.width === 4) field.scratch[a + 3] = field.base[a + 3];
        for (let axis = 0; axis < 3; axis++) {
          const value = finite(field.scratch[a + axis], 'Deformed Float32 attribute');
          if (field.name === 'positions') {
            nextMin[axis] = Math.min(nextMin[axis], value);
            nextMax[axis] = Math.max(nextMax[axis], value);
          }
        }
      }
      // No caller code runs during publication; failure above leaves all
      // published attributes, bounds, transform and version stamps unchanged.
      for (const field of fields) field.output.set(field.scratch);
      worldMatrix.set(nextWorld); minimum.set(nextMin); maximum.set(nextMax);
      poseVersion = nextPoseVersion; version++; return deformer;
    } finally { busy = false; }
  }
  const deformer = Object.freeze({...output, node, vertexCount, worldMatrix, bounds,
    update,
    get version() { return version; }, get poseVersion() { return poseVersion; },
    get disposed() { return disposed; },
    dispose() { if (busy) fail('ANIMATION_REENTRANT', 'Cannot dispose during deformation'); disposed = true; },
  });
  update(); return deformer;
}
