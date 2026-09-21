/**
 * Bulk retained-Skeleton update through the Rust f64 matrix kernel.
 * Call after the application's explicit world-matrix update boundary, including
 * bindThreeTransformHierarchy(...).update() when that route owns the transforms.
 * Does not traverse bones, sample animation, calculate inverse binds, update
 * SkinnedMesh bind modes, or monkey-patch Skeleton.update().
 */
const owners = new WeakMap();
const identity = Object.freeze([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const setF32 = Function.call.bind(Float32Array.prototype.set);
function copyF32(values, count = values.length) {
  const copy = new Float32Array(count);
  for (let i = 0; i < count; i++) copy[i] = values[i];
  return copy;
}

function fail(code, detail) {
  const error = new Error(`${code}: ${detail}`);
  error.code = code;
  return error;
}
function requireThat(condition, code, detail) {
  if (!condition) throw fail(code, detail);
}
function localBuffer(view) {
  if (!(view.buffer instanceof ArrayBuffer) || view.buffer.resizable) return false;
  try { new Uint8Array(view.buffer, 0, 0); return true; } catch { return false; }
}
function matrixElements(matrix) {
  const values = matrix?.elements;
  requireThat((Array.isArray(values) || values instanceof Float64Array || values instanceof Float32Array) &&
    values.length === 16, 'SKELETON_MATRIX', 'world and inverse-bind matrices need 16 numeric elements');
  if (!Array.isArray(values)) {
    requireThat(localBuffer(values), 'SKELETON_SHARED_BUFFER', 'shared/resizable matrix storage is not admitted');
  }
  for (let i = 0; i < 16; i++) {
    const descriptor = Object.getOwnPropertyDescriptor(values, String(i));
    requireThat(descriptor && 'value' in descriptor && typeof descriptor.value === 'number',
      'SKELETON_MATRIX', 'accessors, holes, and nonnumeric matrix elements are not admitted');
  }
  return values;
}
function range(view, elements = view.length) {
  return { buffer: view.buffer, start: view.byteOffset, end: view.byteOffset + elements * view.BYTES_PER_ELEMENT };
}
// Group and sweep byte intervals, rather than comparing every rig against all
// joints. Read/read overlap is legal; any overlap involving a write is not.
function validateRanges(sources, destinations) {
  const banks = new Map();
  for (const [items, write] of [[sources, false], [destinations, true]]) {
    for (const item of items) {
      if (item.start === item.end) continue;
      let bank = banks.get(item.buffer);
      if (!bank) { bank = []; banks.set(item.buffer, bank); }
      bank.push({ start: item.start, end: item.end, write });
    }
  }
  for (const bank of banks.values()) {
    bank.sort((a, b) => a.start - b.start);
    let readEnd = 0, writeEnd = 0;
    for (const item of bank) {
      requireThat(item.start >= writeEnd && (!item.write || item.start >= readEnd),
        'SKELETON_ALIAS', 'palette writes overlap another palette or matrix input');
      if (item.write) writeEnd = Math.max(writeEnd, item.end);
      else readEnd = Math.max(readEnd, item.end);
    }
  }
}
function same(values, packed, offset, count) {
  if (values.length < count) return false;
  for (let i = 0; i < count; i++) if (!Object.is(values[i], packed[offset + i])) return false;
  return true;
}

/**
 * Bind a fixed set of retained skeleton identities. Bone membership, inverse
 * binds and boneTexture/boneMatrices residency may change BETWEEN updates.
 *
 * One bulk Wasm call handles the entire batch. Publication preserves existing
 * Float32Array identity and padding used by Skeleton.computeBoneTexture().
 * Matrices remain f64 through multiplication and narrow only at the same public
 * boneMatrices boundary as pinned r186. Missing (null/undefined) bones use an
 * identity world matrix. No JavaScript multiplication implementation exists here.
 *
 * Input/transport failure publishes no palette. After all matrices commit,
 * texture.needsUpdate is set in skeleton order, even for unchanged/empty rigs.
 * Texture setter effects are not transactional: a throwing custom setter leaves
 * the already-published matrices in place and propagates its original exception.
 *
 * Intended for ordinary retained data objects/arrays, not Proxies or custom
 * accessor-driven matrix storage. Shared/resizable buffers and overlapping
 * source/destination or destination/destination ranges reject explicitly.
 * No rendering acceleration or GPU conformance is established by this binding.
 */
export function bindThreeSkeletonPalettes(wasm, skeletons, { maxJoints = 65536 } = {}) {
  const native = wasm?.f3d_batch_skeleton_palette;
  requireThat(typeof native === 'function',
    'SKELETON_MISSING_WASM', 'compiled f3d_batch_skeleton_palette is required');
  requireThat(Array.isArray(skeletons), 'SKELETON_INPUT', 'skeletons must be an array');
  requireThat(Number.isSafeInteger(maxJoints) && maxJoints >= 0 && maxJoints <= 0x7fffffff / 16,
    'SKELETON_BUDGET', 'maxJoints must be a bounded nonnegative integer');
  const members = skeletons.slice();
  const unique = new Set();
  for (const skeleton of members) {
    requireThat(skeleton !== null && typeof skeleton === 'object', 'SKELETON_INPUT', 'invalid skeleton object');
    requireThat(!unique.has(skeleton), 'SKELETON_DUPLICATE', 'include shared skeleton identities only once');
    requireThat(!owners.has(skeleton), 'SKELETON_OWNERSHIP', 'skeleton already has a palette binding');
    unique.add(skeleton);
  }
  const token = {};
  for (const skeleton of members) owners.set(skeleton, token);
  const compute = native.bind(wasm);
  let disposed = false, busy = false;
  function live() { requireThat(!disposed, 'SKELETON_DISPOSED', 'palette binding is disposed'); }
  function release() {
    for (const skeleton of members) if (owners.get(skeleton) === token) owners.delete(skeleton);
  }

  function snapshot() {
    const rows = [], sources = [], destinations = [];
    let jointCount = 0;
    for (const skeleton of members) {
      const bones = skeleton.bones, inverses = skeleton.boneInverses;
      requireThat(Array.isArray(bones) && Array.isArray(inverses) && bones.length === inverses.length,
        'SKELETON_SHAPE', 'bones and inverse-bind arrays must have matching counts');
      const offset = jointCount * 16, count = bones.length;
      jointCount += count;
      requireThat(jointCount <= maxJoints, 'SKELETON_BUDGET', 'joint snapshot exceeds maxJoints');
      const destination = skeleton.boneMatrices;
      requireThat(destination instanceof Float32Array && destination.length >= count * 16,
        'SKELETON_DESTINATION', 'boneMatrices must be a sufficiently large Float32Array');
      requireThat(localBuffer(destination), 'SKELETON_SHARED_BUFFER', 'shared/resizable boneMatrices are not admitted');
      const boneRefs = bones.slice(), inverseRefs = inverses.slice(), worldRefs = [], worldArrays = [], inverseArrays = [];
      for (let j = 0; j < count; j++) {
        const world = boneRefs[j] == null ? null : boneRefs[j].matrixWorld;
        const worldArray = boneRefs[j] == null ? identity : matrixElements(world);
        const inverseArray = matrixElements(inverseRefs[j]);
        worldRefs.push(world); worldArrays.push(worldArray); inverseArrays.push(inverseArray);
        for (const values of [worldArray, inverseArray]) if (ArrayBuffer.isView(values)) sources.push(range(values));
      }
      destinations.push(range(destination, count * 16));
      rows.push({ skeleton, bones, inverses, boneRefs, inverseRefs, worldRefs, worldArrays, inverseArrays,
        destination, destinationLength: destination.length, previous: copyF32(destination, count * 16),
        texture: skeleton.boneTexture, offset, count });
    }
    validateRanges(sources, destinations);
    const worlds = new Float64Array(jointCount * 16), inverses = new Float64Array(jointCount * 16);
    for (const row of rows) for (let j = 0; j < row.count; j++) {
      worlds.set(row.worldArrays[j], row.offset + j * 16);
      inverses.set(row.inverseArrays[j], row.offset + j * 16);
    }
    return { rows, worlds, inverses, jointCount };
  }

  function check(input) {
    live();
    for (const row of input.rows) {
      const { skeleton, count } = row;
      requireThat(skeleton.bones === row.bones && skeleton.boneInverses === row.inverses &&
        row.bones.length === count && row.inverses.length === count &&
        skeleton.boneMatrices === row.destination && row.destination.length === row.destinationLength &&
        localBuffer(row.destination) && skeleton.boneTexture === row.texture &&
        same(row.destination, row.previous, 0, count * 16),
        'SKELETON_STALE_INPUT', 'skeleton storage or palette changed before publication');
      for (let j = 0; j < count; j++) {
        const bone = row.bones[j];
        requireThat(bone === row.boneRefs[j] && row.inverses[j] === row.inverseRefs[j] &&
          (bone == null || bone.matrixWorld === row.worldRefs[j] &&
            matrixElements(bone.matrixWorld) === row.worldArrays[j]) &&
          matrixElements(row.inverses[j]) === row.inverseArrays[j] &&
          same(row.worldArrays[j], input.worlds, row.offset + j * 16, 16) &&
          same(row.inverseArrays[j], input.inverses, row.offset + j * 16, 16),
          'SKELETON_STALE_INPUT', 'joint world matrix or inverse bind changed before publication');
      }
    }
    live();
  }

  return Object.freeze({
    get disposed() { return disposed; },
    get skeletonCount() { return members.length; },
    update() {
      live();
      requireThat(!busy, 'SKELETON_REENTRANT', 'palette update cannot reenter');
      busy = true;
      try {
        const input = snapshot();
        check(input);
        const result = input.jointCount ? compute(input.worlds, input.inverses) : new Float32Array();
        requireThat(result instanceof Float32Array && result.length === input.jointCount * 16 && localBuffer(result),
          'SKELETON_WASM_OUTPUT', 'Wasm must return an owned Float32Array with 16 components per joint');
        // Copy before retained-object validation; no borrowed native view escapes.
        const output = copyF32(result);
        check(input);
        // No user callbacks between the final validation and all numeric writes.
        for (const row of input.rows) {
          setF32(row.destination, output.subarray(row.offset, row.offset + row.count * 16), 0);
        }
        for (const row of input.rows) if (row.texture != null) row.texture.needsUpdate = true;
        return Object.freeze({ skeletonCount: members.length, jointCount: input.jointCount,
          paletteBytes: output.byteLength });
      } finally { busy = false; }
    },
    /** Release only this CPU binding, not skeletons, textures, or a GPU device. */
    dispose() { if (!disposed) { disposed = true; release(); } },
  });
}
