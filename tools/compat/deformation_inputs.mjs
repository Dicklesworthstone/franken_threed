/** Flat retained-geometry snapshots for the compiled Rust deformation batch. */
function reject(code, detail) {
  const error = new Error(`${code}: ${detail}`); error.code = code; error.reason = code; throw error;
}
function requireThat(condition, code, detail) { if (!condition) reject(code, detail); }
function ordinaryBuffer(array) {
  if (!ArrayBuffer.isView(array) || array instanceof DataView ||
      !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable) return false;
  try { new Uint8Array(array.buffer, 0, 0); return true; } catch { return false; }
}
function checkedSize(value, name) {
  requireThat(Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff,
    'DEFORMATION_SIZE', `${name} must be a nonnegative u32`);
  return value;
}
function budget(value, limit, name) {
  requireThat(Number.isSafeInteger(value) && value <= limit, 'DEFORMATION_BUDGET', `${name} exceeds ${limit}`);
}
function attribute(attribute, itemSize, count, name) {
  const target = attribute?.isInterleavedBufferAttribute ? attribute.data : attribute;
  requireThat(attribute && attribute.itemSize === itemSize && attribute.count === count &&
    !attribute.isInstancedBufferAttribute && !target?.isInstancedInterleavedBuffer && ordinaryBuffer(target?.array),
    'DEFORMATION_ATTRIBUTE', `${name} needs ${count} ordinary ${itemSize}-component vertices`);
  requireThat(!Object.hasOwn(target, 'onUploadCallback') &&
    (!target.onUploadCallback || target.onUploadCallback === Object.getPrototypeOf(target)?.onUploadCallback),
    'UNSUPPORTED_UPLOAD_CALLBACK', `${name} has a custom upload callback`);
  for (const getter of ['getX', 'getY', 'getZ', 'getW'].slice(0, itemSize)) {
    requireThat(typeof attribute[getter] === 'function', 'DEFORMATION_ATTRIBUTE', `${name}.${getter} is required`);
  }
  return attribute;
}
function values(attribute, size, count, destination, offset, name, integers = false) {
  const getters = ['getX', 'getY', 'getZ', 'getW'];
  for (let vertex = 0; vertex < count; vertex++) for (let axis = 0; axis < size; axis++) {
    const value = attribute[getters[axis]](vertex);
    requireThat(Number.isFinite(value) && (!integers || Number.isInteger(value) && value >= 0 && value <= 0xffffffff),
      'DEFORMATION_VALUE', `${name}[${vertex},${axis}] is not ${integers ? 'a u32' : 'finite'}`);
    destination[offset + vertex * size + axis] = value;
  }
}
function matrix(matrix, destination, offset, name) {
  const source = matrix?.elements;
  requireThat((Array.isArray(source) || ordinaryBuffer(source)) && source.length === 16,
    'DEFORMATION_BIND', `${name} needs sixteen numeric matrix elements`);
  for (let i = 0; i < 16; i++) {
    requireThat(Number.isFinite(source[i]), 'DEFORMATION_BIND', `${name}[${i}] is not finite`);
    destination[offset + i] = source[i];
  }
}

/**
 * Capture immutable base geometry, current influences and evaluated palettes.
 * This is an explicit fresh-data snapshot, not a persistent GPU attribute cache.
 * Call after the application's chosen matrix/bind/palette update boundary. In
 * particular, skeleton.boneMatrices is consumed as-is; no Skeleton.update(),
 * hierarchy traversal, AnimationMixer tick, bind-mode repair, or normalization
 * happens here. Float32 palette padding is excluded from the native bank.
 *
 * Ordinary Three.js BufferAttribute getters decode normalized/interleaved data.
 * Position and normal morph families are recognized; normals are not consumed
 * by the unlit, untextured MeshBasicMaterial draw path. Color/unknown morphs,
 * instancing, shared/resizable storage, and upload callbacks reject explicitly.
 */
export function captureDeformationBatch(meshes, {
  maxVertices = 1_000_000, maxMorphComponents = 24_000_000, maxJoints = 65_536,
} = {}) {
  requireThat(Array.isArray(meshes) && meshes.length > 0, 'EMPTY_MESH_BATCH', 'an ordered nonempty mesh array is required');
  for (const [name, value] of Object.entries({ maxVertices, maxMorphComponents, maxJoints })) checkedSize(value, name);
  const rows = [];
  let vertexCount = 0, morphComponents = 0, morphCount = 0, jointCount = 0, influences = 0, skinnedCount = 0, admittedMorphs = 0;
  // Bound the mesh table too: zero-vertex meshes must not bypass every budget.
  budget(meshes.length, Math.max(1, maxVertices), 'mesh count');
  for (let index = 0; index < meshes.length; index++) {
    const mesh = meshes[index], geometry = mesh?.geometry;
    requireThat(mesh?.isMesh && !mesh.isInstancedMesh && !mesh.isBatchedMesh && geometry?.isBufferGeometry,
      'DEFORMATION_MESH', `mesh ${index} needs scalar BufferGeometry`);
    const count = checkedSize(geometry.attributes?.position?.count, `mesh ${index} vertex count`);
    budget(vertexCount + count, maxVertices, 'vertex count');
    const position = attribute(geometry.attributes.position, 3, count, `mesh ${index} position`);
    const families = geometry.morphAttributes ?? {};
    for (const [name, targets] of Object.entries(families)) {
      requireThat(Array.isArray(targets) && (name === 'position' || name === 'normal' || targets.length === 0),
        'DEFORMATION_MORPH', `mesh ${index}: unsupported ${name} morph family`);
    }
    const targets = families.position ?? [], normals = families.normal ?? [];
    const familyCount = Math.max(targets.length, normals.length);
    const weights = mesh.morphTargetInfluences ?? [];
    requireThat((Array.isArray(weights) || ordinaryBuffer(weights)) &&
      (familyCount === 0 || weights.length === familyCount) &&
      (targets.length === 0 || targets.length === familyCount) &&
      (normals.length === 0 || normals.length === familyCount),
      'DEFORMATION_MORPH', `mesh ${index}: morph families and influences disagree`);
    budget(morphComponents + count * 3 * targets.length, maxMorphComponents, 'morph components');
    budget(admittedMorphs + familyCount, maxMorphComponents, 'morph influence count');
    admittedMorphs += familyCount;
    for (const [family, list] of [['position', targets], ['normal', normals]]) {
      for (const target of list) attribute(target, 3, count, `${family} morph`);
    }
    for (let i = 0; i < familyCount; i++) requireThat(Number.isFinite(weights[i]), 'DEFORMATION_MORPH', 'morph weight is not finite');
    const skinned = mesh.isSkinnedMesh === true;
    let paletteCount = 0, skinIndex = null, skinWeight = null, palette = null;
    if (skinned) {
      requireThat(Array.isArray(mesh.skeleton?.bones), 'DEFORMATION_SKELETON', `mesh ${index}: missing skeleton bones`);
      paletteCount = checkedSize(mesh.skeleton.bones.length, 'joint count');
      budget(jointCount + paletteCount, maxJoints, 'joint count');
      palette = mesh.skeleton.boneMatrices;
      requireThat(palette instanceof Float32Array && ordinaryBuffer(palette) && palette.length >= paletteCount * 16,
        'DEFORMATION_SKELETON', `mesh ${index}: an evaluated Float32 boneMatrices bank is required`);
      skinIndex = attribute(geometry.attributes.skinIndex, 4, count, 'skinIndex');
      requireThat(!skinIndex.normalized, 'DEFORMATION_ATTRIBUTE', 'skinIndex must not be normalized');
      skinWeight = attribute(geometry.attributes.skinWeight, 4, count, 'skinWeight');
    }
    rows.push({ mesh, geometry, position, count, targets, weights, skinned, paletteCount, palette, skinIndex, skinWeight,
      positionOffset: vertexCount * 3, morphOffset: morphComponents, weightOffset: morphCount,
      jointOffset: jointCount * 16, influenceOffset: influences, bindOffset: skinnedCount * 32 });
    vertexCount += count; morphComponents += count * 3 * targets.length; morphCount += targets.length;
    jointCount += paletteCount;
    if (skinned) { influences += count * 4; skinnedCount++; }
  }
  const layout = new Uint32Array(rows.length * 4), positions = new Float64Array(vertexCount * 3);
  const morphPositions = new Float64Array(morphComponents), morphWeights = new Float64Array(morphCount);
  const jointIndices = new Uint32Array(influences), jointWeights = new Float64Array(influences);
  const jointPalettes = new Float64Array(jointCount * 16), bindMatrices = new Float64Array(skinnedCount * 32);
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    layout.set([row.count, row.targets.length, row.paletteCount,
      (row.geometry.morphTargetsRelative === true ? 1 : 0) | (row.skinned ? 2 : 0)], i * 4);
    values(row.position, 3, row.count, positions, row.positionOffset, 'position');
    for (let target = 0; target < row.targets.length; target++) {
      values(row.targets[target], 3, row.count, morphPositions, row.morphOffset + target * row.count * 3, 'morph position');
      morphWeights[row.weightOffset + target] = row.weights[target];
    }
    if (row.skinned) {
      values(row.skinIndex, 4, row.count, jointIndices, row.influenceOffset, 'skinIndex', true);
      values(row.skinWeight, 4, row.count, jointWeights, row.influenceOffset, 'skinWeight');
      for (let j = 0; j < row.paletteCount * 16; j++) {
        requireThat(Number.isFinite(row.palette[j]), 'DEFORMATION_SKELETON', `joint palette component ${j} is not finite`);
        jointPalettes[row.jointOffset + j] = row.palette[j];
      }
      matrix(row.mesh.bindMatrix, bindMatrices, row.bindOffset, 'bindMatrix');
      matrix(row.mesh.bindMatrixInverse, bindMatrices, row.bindOffset + 16, 'bindMatrixInverse');
    }
  }
  return { rows, layout, positions, morphPositions, morphWeights, jointIndices, jointWeights, jointPalettes, bindMatrices,
    vertexCount, skinnedCount, deformedMeshCount: rows.filter(row => row.skinned || row.targets.length > 0).length };
}

/** Perform one synchronous compiled evaluation; no host math implementation. */
export function evaluateDeformationBatch(input, wasm) {
  const native = wasm?.f3d_deform_position_batch;
  requireThat(typeof native === 'function', 'DEFORMATION_MISSING_WASM', 'compiled f3d_deform_position_batch is required');
  const output = native.call(wasm, input.layout, input.positions, input.morphPositions, input.morphWeights,
    input.jointIndices, input.jointWeights, input.jointPalettes, input.bindMatrices);
  requireThat(output instanceof Float32Array && ordinaryBuffer(output) && output.length === input.positions.length,
    'DEFORMATION_WASM_OUTPUT', 'compiled result must have one finite xyz tuple per input vertex');
  for (const value of output) requireThat(Number.isFinite(value), 'DEFORMATION_WASM_OUTPUT', 'compiled output is not finite');
  // Own the result before other Wasm exports may grow memory or reuse scratch.
  return new Float32Array(output);
}
