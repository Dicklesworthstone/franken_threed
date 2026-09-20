/**
 * Explicit retained-object binding to the Rust transform hierarchy.
 *
 * This is a synchronous matrix-snapshot API, not a replacement for arbitrary
 * Object3D.updateMatrixWorld overrides. It does not drive animation, run hooks,
 * update camera inverses, or change matrixWorldNeedsUpdate. The caller selects
 * complete, fixed node membership and owns the update boundary. Parent changes
 * within that membership are supported; outside parents require a new binding.
 *
 * All TRS composition and parent/world matrix products run in compiled Wasm.
 * This shell only reads identities/properties, packs typed arrays, and publishes
 * a current result to the same retained matrix arrays. There is no JS fallback.
 */
const owners = new WeakMap();
const methods = ['reparent', 'setLocalMatrices', 'setWorldMatrices', 'setWorldAuto',
  'solve', 'worldMatrices', 'changedIndices', 'solveStats', 'free'];

function fail(code, message) {
  const error = new Error(`${code}: ${message}`);
  error.code = code;
  return error;
}

function matrixElements(matrix, label, writable) {
  const values = matrix?.elements;
  if (!(Array.isArray(values) || values instanceof Float64Array) || values.length !== 16) {
    throw fail('HIERARCHY_MATRIX', `${label} must contain 16 f64-compatible elements`);
  }
  for (let i = 0; i < 16; i++) {
    const property = Object.getOwnPropertyDescriptor(values, String(i));
    if (!property || !('value' in property) || typeof property.value !== 'number' ||
        (writable && !property.writable)) {
      throw fail('HIERARCHY_MATRIX', `${label}[${i}] must be an ordinary ${writable ? 'writable ' : ''}numeric element`);
    }
  }
  if (typeof SharedArrayBuffer !== 'undefined' && values.buffer instanceof SharedArrayBuffer) {
    throw fail('HIERARCHY_MATRIX', `${label} cannot use concurrently mutable shared memory`);
  }
  return values;
}

function readComponents(value, names, label) {
  if (!value) throw fail('HIERARCHY_TRS', `${label} is missing`);
  return names.map(name => {
    const component = value[name];
    if (typeof component !== 'number') throw fail('HIERARCHY_TRS', `${label}.${name} must be numeric`);
    return component;
  });
}

function sameComponents(owner, names, expected) {
  return names.every((name, i) => Object.is(owner[name], expected[i]));
}

function checkedOutput(value, count, label) {
  if (!(value instanceof Float64Array) || value.length !== count ||
      (typeof SharedArrayBuffer !== 'undefined' && value.buffer instanceof SharedArrayBuffer)) {
    throw fail('HIERARCHY_WASM_OUTPUT', `${label} must be an owned Float64Array of length ${count}`);
  }
  return value;
}

/**
 * Bind ordinary Three.js Object3D/Group/Scene/Mesh matrices to compiled Wasm.
 * Cameras and SkinnedMesh require extra source update semantics and reject.
 * Public object and matrix-array identities are retained. Dispose the binding
 * before binding the same objects elsewhere or changing its node membership.
 *
 * @param {object} wasm Initialized f3d_runtime wasm-bindgen module.
 * @param {object[]} objects Explicit node slots, in any order, with all parents included.
 */
export function bindThreeTransformHierarchy(wasm, objects) {
  if (typeof wasm?.F3dTransformHierarchy !== 'function') {
    throw fail('HIERARCHY_MISSING_WASM', 'compiled F3dTransformHierarchy is required');
  }
  if (!Array.isArray(objects) || objects.length > 0x7fffffff) {
    throw fail('HIERARCHY_NODES', 'objects must be a fixed array of signed-32-bit-indexable nodes');
  }
  const nodes = objects.slice();
  const slots = new Map(nodes.map((node, index) => [node, index]));
  if (slots.size !== nodes.length) throw fail('HIERARCHY_NODES', 'duplicate object identity');
  for (const node of nodes) {
    if (!node || node.isObject3D !== true || node.isCamera || node.isSkinnedMesh ||
        Object.hasOwn(node, 'updateMatrixWorld') || Object.hasOwn(node, 'updateWorldMatrix')) {
      throw fail('HIERARCHY_NODES', 'ordinary Object3D matrices without camera, skin, or own update hooks are required');
    }
    if (owners.has(node)) throw fail('HIERARCHY_OWNER', 'an object already has a hierarchy writer');
  }
  const token = Symbol('hierarchy owner');
  const indices = Uint32Array.from(nodes, (_, index) => index);
  let native = null, disposed = false, busy = false;
  for (const node of nodes) owners.set(node, token);

  function releaseOwnership() {
    for (const node of nodes) if (owners.get(node) === token) owners.delete(node);
  }
  function freeNative() {
    const old = native;
    native = null;
    if (typeof old?.free === 'function') old.free();
  }
  function checkLive() {
    if (disposed) throw fail('HIERARCHY_DISPOSED', 'the binding is disposed');
  }

  function snapshot() {
    const parents = new Int32Array(nodes.length);
    const locals = new Float64Array(nodes.length * 16);
    const worlds = new Float64Array(nodes.length * 16);
    const flags = new Uint8Array(nodes.length);
    const rows = [], autoNodes = [], positions = [], rotations = [], scales = [];
    const worldNodes = [], externalWorlds = [];
    const destinations = new Set();
    const memoryRanges = new Map();
    for (let index = 0; index < nodes.length; index++) {
      const node = nodes[index];
      if (typeof node.matrixAutoUpdate !== 'boolean' || typeof node.matrixWorldAutoUpdate !== 'boolean') {
        throw fail('HIERARCHY_FLAGS', 'matrixAutoUpdate and matrixWorldAutoUpdate must be explicit booleans');
      }
      const parent = node.parent;
      if (parent !== null && !slots.has(parent)) throw fail('HIERARCHY_PARENT', 'parent is outside the bound node membership');
      parents[index] = parent === null ? -1 : slots.get(parent);
      const localMatrix = node.matrix, worldMatrix = node.matrixWorld;
      const local = matrixElements(localMatrix, `local matrix ${index}`, node.matrixAutoUpdate);
      const world = matrixElements(worldMatrix, `world matrix ${index}`, node.matrixWorldAutoUpdate);
      // Even read-only source banks must not alias a publication destination:
      // writing a world result must never overwrite another node's local input.
      if (destinations.has(local) || destinations.has(world) || local === world) {
        throw fail('HIERARCHY_ALIAS', 'local and world matrix arrays must have independent identities');
      }
      destinations.add(local); destinations.add(world);
      for (const values of [local, world]) {
        if (values instanceof Float64Array) {
          const ranges = memoryRanges.get(values.buffer) ?? [];
          ranges.push([values.byteOffset, values.byteOffset + values.byteLength]);
          memoryRanges.set(values.buffer, ranges);
        }
      }
      locals.set(local, index * 16); worlds.set(world, index * 16);
      flags[index] = Number(node.matrixWorldAutoUpdate);
      const row = { node, parent, localMatrix, worldMatrix, local, world,
        autoLocal: node.matrixAutoUpdate, autoWorld: node.matrixWorldAutoUpdate,
        originalLocal: Array.from(local), originalWorld: Array.from(world) };
      if (row.autoLocal) {
        row.position = node.position; row.quaternion = node.quaternion; row.scale = node.scale;
        row.p = readComponents(row.position, ['x', 'y', 'z'], 'position');
        row.q = readComponents(row.quaternion, ['x', 'y', 'z', 'w'], 'quaternion');
        row.s = readComponents(row.scale, ['x', 'y', 'z'], 'scale');
        autoNodes.push(index); positions.push(...row.p); rotations.push(...row.q); scales.push(...row.s);
      }
      if (!row.autoWorld) { worldNodes.push(index); externalWorlds.push(...world); }
      rows.push(row);
    }
    for (const ranges of memoryRanges.values()) {
      ranges.sort((a, b) => a[0] - b[0]);
      for (let i = 1; i < ranges.length; i++) {
        if (ranges[i][0] < ranges[i - 1][1]) {
          throw fail('HIERARCHY_ALIAS', 'typed matrix views cannot overlap in memory');
        }
      }
    }
    if (autoNodes.length) {
      if (typeof wasm.f3d_batch_compose_scalar !== 'function') {
        throw fail('HIERARCHY_MISSING_WASM', 'automatic local matrices require compiled f3d_batch_compose_scalar');
      }
      const composed = checkedOutput(wasm.f3d_batch_compose_scalar(
        new Float64Array(positions), new Float64Array(rotations), new Float64Array(scales)),
      autoNodes.length * 16, 'composed local matrices');
      autoNodes.forEach((node, i) => locals.set(composed.subarray(i * 16, i * 16 + 16), node * 16));
    }
    return { parents, locals, worlds, flags, rows,
      worldNodes: new Uint32Array(worldNodes), externalWorlds: new Float64Array(externalWorlds) };
  }

  function checkSnapshot(input) {
    for (const row of input.rows) {
      const n = row.node;
      if (n.parent !== row.parent || n.matrix !== row.localMatrix || n.matrixWorld !== row.worldMatrix ||
          n.matrixAutoUpdate !== row.autoLocal || n.matrixWorldAutoUpdate !== row.autoWorld ||
          n.matrix.elements !== row.local || n.matrixWorld.elements !== row.world ||
          !row.originalLocal.every((v, i) => Object.is(v, row.local[i])) ||
          !row.originalWorld.every((v, i) => Object.is(v, row.world[i])) ||
          (row.autoLocal && (n.position !== row.position || n.quaternion !== row.quaternion || n.scale !== row.scale ||
            !sameComponents(row.position, ['x', 'y', 'z'], row.p) ||
            !sameComponents(row.quaternion, ['x', 'y', 'z', 'w'], row.q) ||
            !sameComponents(row.scale, ['x', 'y', 'z'], row.s)))) {
        throw fail('HIERARCHY_STALE_INPUT', 'retained input changed before result publication');
      }
      // Recheck writeability after calling compiled exports; no half-publication
      // if a host callback froze or replaced a later destination.
      matrixElements(row.localMatrix, 'local publication', row.autoLocal);
      matrixElements(row.worldMatrix, 'world publication', row.autoWorld);
    }
    checkLive();
  }

  try {
    const initial = snapshot();
    checkSnapshot(initial);
    native = new wasm.F3dTransformHierarchy(initial.parents, initial.locals, initial.worlds, initial.flags);
    for (const method of methods) {
      if (typeof native?.[method] !== 'function') throw fail('HIERARCHY_MISSING_WASM', `compiled hierarchy lacks ${method}`);
    }
    checkSnapshot(initial);
  } catch (error) {
    disposed = true;
    releaseOwnership();
    try { freeNative(); } catch { /* Preserve the constructor's original error. */ }
    throw error;
  }

  return Object.freeze({
    get nodeCount() { return nodes.length; },
    get disposed() { return disposed; },
    /** Pack the current source snapshot, solve in Rust, then publish synchronously. */
    update() {
      checkLive();
      if (busy) throw fail('HIERARCHY_REENTRANT', 'hierarchy update cannot reenter');
      busy = true;
      try {
        const input = snapshot();
        checkSnapshot(input);
        native.reparent(input.parents);
        native.setWorldAuto(indices, input.flags);
        native.setLocalMatrices(indices, input.locals);
        native.setWorldMatrices(input.worldNodes, input.externalWorlds);
        const revision = native.solve();
        if (typeof revision !== 'bigint') throw fail('HIERARCHY_WASM_OUTPUT', 'revision must be a lossless BigInt');
        const worlds = checkedOutput(native.worldMatrices(revision), nodes.length * 16, 'world matrices');
        const changed = native.changedIndices(revision);
        const stats = native.solveStats();
        if (!(changed instanceof Uint32Array) || !(stats instanceof Uint32Array) || stats.length !== 5 ||
            stats.some(value => value > nodes.length) || stats[0] !== stats[1] + stats[2] + stats[3] ||
            stats[4] !== changed.length || changed.some((node, i) => node >= nodes.length || i > 0 && node <= changed[i - 1])) {
          throw fail('HIERARCHY_WASM_OUTPUT', 'invalid solve statistics or changed-node identities');
        }
        if (native.revision !== revision) throw fail('HIERARCHY_STALE_RESULT', 'native revision changed before publication');
        checkedOutput(worlds, nodes.length * 16, 'world matrices');
        checkSnapshot(input);
        // Ordinary writable numeric elements were validated for EVERY target
        // before the first write. Do not call overridable Matrix4.fromArray hooks.
        for (let node = 0; node < nodes.length; node++) {
          const row = input.rows[node];
          for (let i = 0; i < 16; i++) {
            if (row.autoLocal) row.local[i] = input.locals[node * 16 + i];
            if (row.autoWorld) row.world[i] = worlds[node * 16 + i];
          }
        }
        return Object.freeze({ revision, visited: stats[0], multiplied: stats[1],
          copiedRoots: stats[2], preservedWorlds: stats[3], changedNodes: changed.slice() });
      } finally {
        busy = false;
        if (disposed) freeNative();
      }
    },
    /** Release only the owned Rust hierarchy, never the retained Three.js objects. */
    dispose() {
      if (disposed) return;
      disposed = true;
      releaseOwnership();
      if (!busy) freeNative();
    },
  });
}
