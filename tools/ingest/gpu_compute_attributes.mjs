/** Internal compute -> graphics-buffer ownership seam. A source storage attribute
 * identifies one published allocation per device. Geometry only borrows it:
 * no readback, CPU upload, duplicate allocation or ownership transfer occurs.
 * Publication is after compute upload (including compileComputeAsync). Native
 * buffer usages must include VERTEX as well as STORAGE from creation onward;
 * index borrowers additionally require INDEX usage. */
const devices = new WeakMap(), vertexViews = new WeakMap(), immutableVertexViews = new WeakMap(), refreshedVertexViews = new WeakMap();
export class ComputeAttributeError extends Error {
  constructor(code, message) {
    super(`COMPUTE_ATTRIBUTE_${code}: ${message}`);
    this.name = 'ComputeAttributeError'; this.code = `COMPUTE_ATTRIBUTE_${code}`;
  }
}
const fail = (code, message) => { throw new ComputeAttributeError(code, message); };
export const isComputeAttribute = value => value?.isStorageBufferAttribute === true ||
  value?.isStorageInstancedBufferAttribute === true || value?.isIndirectStorageBufferAttribute === true;

export function createComputeAttributeOwner(device) {
  let table = devices.get(device);
  if (!table) devices.set(device, table = new WeakMap());
  const owned = new Set(), identity = {}, captures = new WeakMap();
  let disposed = false;
  function release(attribute) {
    if (table.get(attribute)?.identity === identity) table.delete(attribute);
    owned.delete(attribute);
  }
  function admit(attribute, array, version = attribute?.version, admission = null) {
      if (!isComputeAttribute(attribute)) return;
      if (disposed) fail('DISPOSED', 'Compute attribute owner is disposed');
      const existing = table.get(attribute);
      if (existing && existing.identity !== identity)
        fail('OWNER', 'The same storage attribute cannot have two compute owners on one device');
      if (admission !== null) {
        const captured = captures.get(admission);
        if (!captured || captured.attribute !== attribute || captured.array !== array || captured.version !== version)
          fail('PUBLICATION', 'Queued upload must use its admitted attribute, array and source version');
        return captured;
      }
      const stride = attribute.itemSize, count = attribute.count;
      if (!Number.isSafeInteger(stride) || stride < 1 || !Number.isSafeInteger(count) || count < 1 ||
          stride * count !== array.length || array.BYTES_PER_ELEMENT !== 4)
        fail('SHAPE', 'Publish the normalized 32-bit storage layout before vertex use');
      if (!Number.isSafeInteger(version) || version < 0 || version !== attribute.version)
        fail('PUBLICATION', 'Compute upload must acknowledge the current source version');
      return {stride, count};
  }
  return {
    admit,
    // The renderer captures this proof while the source version is current.
    // Only that exact copied array may later upload after the source advances.
    // Direct uploads retain their current-version check; a boolean bypass would
    // let an unadmitted stale CPU array overwrite GPU simulation state.
    capture(attribute, array, version) {
      const shape = admit(attribute, array, version);
      if (!shape) return null;
      const admission = Object.freeze({});
      captures.set(admission, {...shape, attribute, array, version});
      return admission;
    },
    releaseCapture(admission) { if (admission) captures.delete(admission); },
    publish(attribute, buffer, array, version, live, whenIdle, admission = null) {
      if (!isComputeAttribute(attribute)) return;
      const {stride, count} = admit(attribute, array, version, admission);
      // A borrowed GPU allocation must actually support vertex fetch. Reject
      // malformed or foreign publications before they become visible to draws.
      const required = array.byteLength;
      if (!buffer || !Number.isSafeInteger(buffer.size) || buffer.size < required ||
          !Number.isSafeInteger(buffer.usage) || (buffer.usage & (128 | 32)) !== (128 | 32) ||
          typeof live !== 'function' || typeof whenIdle !== 'function')
        fail('PUBLICATION', 'Shared vertex storage needs a live, sufficiently sized STORAGE|VERTEX buffer at an admitted source version');
      const previous = table.get(attribute);
      const resource = previous?.buffer === buffer && previous.byteLength === array.byteLength
        ? previous.resource : Object.freeze({buffer, offset: 0, size: array.byteLength});
      table.set(attribute, {identity, buffer, stride, count, version, scalar: array.constructor,
        byteLength: array.byteLength, live, whenIdle, resource});
      owned.add(attribute);
    },
    release,
    dispose() { if (disposed) return; disposed = true; for (const attribute of owned) release(attribute); },
  };
}

/** Capture a read-only shader use at the logical draw, then resolve it in the
 * renderer FIFO after any preceding compute upload/dispatch. The lease keeps
 * the requested source version and shape, never CPU bytes: ordinary CPU edits
 * cannot overwrite GPU-authored simulation state. A preceding queued upload
 * may publish after capture and after the application has advanced its live
 * attribute again. Resolution therefore compares the captured request with
 * compute's publication, not with the attribute's later mutable fields.
 *
 * This is a borrow of the existing compute owner, not an uploader. It neither
 * allocates nor extends GPU lifetime, and released/replaced owners or resolved
 * allocations are rejected. No release operation is needed for this lease. */
export function captureComputeStorage(device, attribute) {
  const array = attribute?.array, stride = attribute?.itemSize, count = attribute?.count,
    version = attribute?.version;
  if (!isComputeAttribute(attribute) || attribute.isIndirectStorageBufferAttribute ||
      attribute.isInterleavedBufferAttribute || attribute.normalized ||
      !ArrayBuffer.isView(array) || ![Float32Array, Int32Array, Uint32Array].includes(array.constructor) ||
      !Number.isSafeInteger(stride) || stride < 1 || !Number.isSafeInteger(count) || count < 1 ||
      stride * count !== array.length || !Number.isSafeInteger(version) || version < 0)
    fail('SHAPE', 'Graphics storage requires a complete normalized 32-bit storage attribute and nonnegative version');
  const scalar = array.constructor, byteLength = array.byteLength,
    initial = devices.get(device)?.get(attribute);
  function atVersion(version, previous = null) {
    let resolved = previous;
    function current() {
      // Retain the actual terminal reason even when disposal/loss has removed
      // the owner from the shared table already.
      initial?.live(); resolved?.live();
      const entry = devices.get(device)?.get(attribute);
      if (!entry) fail('UNPREPARED', 'Compute must publish the captured storage version on the drawing device before submission');
      entry.live();
      if ((initial && entry.identity !== initial.identity) ||
          (resolved && (entry.identity !== resolved.identity || entry.buffer !== resolved.buffer)))
        fail('STALE', 'Captured shader storage was released or replaced before its queued use');
      if (entry.stride !== stride || entry.count !== count || entry.scalar !== scalar || entry.byteLength !== byteLength)
        fail('SHAPE', 'Published shader storage does not match the captured attribute layout');
      if (entry.version !== version)
        fail('UNPREPARED', 'Compute must publish the captured storage attribute version before drawing');
      resolved ??= entry;
      return entry;
    }
    return Object.freeze({version, byteLength,
      get ownerIdentity() { return resolved?.identity ?? initial?.identity; },
      resolve: () => current().resource,
      assertCurrent() { current(); },
      atVersion(next) {
        if (!Number.isSafeInteger(next) || next < 0) fail('PUBLICATION', 'A storage read requires an admitted nonnegative version');
        // Derive a new proof; the old queued read remains strict. Preserve any
        // resolved allocation identity and the original shape/owner checks.
        return atVersion(next, resolved);
      },
      whenIdle() {
        // Retiring a geometry waits for the allocation it already used. A later
        // FIFO upload may legitimately advance that buffer's content version.
        const allocation = resolved ?? current(); allocation.live();
        return allocation.whenIdle();
      }});
  }
  return atVersion(version);
}

/** Private geometry view of a captured storage generation. Only this module
 * installs its proof; ordinary objects cannot supply a foreign GPUBuffer. The
 * view identity stays stable across frames so geometry caches remain bounded.
 * Its captured proof is immutable even when a later frame installs a new one. */
export function captureComputeAttributeView(device, source, view) {
  const lease = captureComputeStorage(device, source);
  const shape = {array: source.array, itemSize: source.itemSize, count: source.count, version: source.version,
    isStorageBufferAttribute: source.isStorageBufferAttribute === true,
    isStorageInstancedBufferAttribute: source.isStorageInstancedBufferAttribute === true,
    isInstancedBufferAttribute: source.isInstancedBufferAttribute === true,
    normalized: false, meshPerAttribute: 1, gpuType: source.gpuType, usage: source.usage};
  const scalar = shape.array.constructor, byteLength = shape.array.byteLength;
  const proof = Object.freeze({device, source, lease, shape: Object.freeze(shape),
    scalar, byteLength, stride: shape.itemSize, count: shape.count});
  return Object.freeze({install() {
    Object.assign(view, shape);
    vertexViews.set(view, proof);
  }});
}

/** Preserve the proof currently installed in a generated vertex view. A later
 * queued draw can install another source version in that same mutable view;
 * immutable geometry must keep this version while borrowing its GPU buffer.
 * This copies metadata only and never resolves, uploads or reads GPU contents.
 */
export function snapshotComputeAttributeView(view) {
  const proof = vertexViews.get(view);
  if (!proof) return null;
  const previous = immutableVertexViews.get(view), old = previous?.proof;
  if (old && old.device === proof.device && old.source === proof.source && old.lease.ownerIdentity === proof.lease.ownerIdentity &&
      Object.keys(proof.shape).every(key => old.shape[key] === proof.shape[key])) return previous.view;
  const snapshot = Object.freeze(Object.assign(Object.create(Object.getPrototypeOf(view)), proof.shape));
  vertexViews.set(snapshot, proof); immutableVertexViews.set(view, {view: snapshot, proof});
  return snapshot;
}

/** Retained draw commands keep their vertex layout and allocation, but an
 * earlier admitted compute/upload can advance its content version. Derive a
 * fresh immutable proof for that draw's exact storage lease without inspecting
 * the source's later mutable array, shape, or version. */
export function refreshComputeAttributeView(device, source, view, version) {
  const proof = vertexViews.get(view);
  if (!proof || proof.source !== source) return null;
  if (proof.device !== device) fail('OWNER', 'Captured vertex storage belongs to another device');
  if (proof.shape.version === version) return view;
  const previous = refreshedVertexViews.get(view);
  if (previous?.proof === proof && previous.version === version) return previous.view;
  const lease = proof.lease.atVersion(version), shape = Object.freeze({...proof.shape, version});
  const snapshot = Object.freeze(Object.assign(Object.create(Object.getPrototypeOf(view)), shape));
  vertexViews.set(snapshot, Object.freeze({...proof, shape, lease}));
  refreshedVertexViews.set(view, {proof, version, view: snapshot});
  return snapshot;
}

/** Only buffers published by the compute owner on this device are admitted.
 * needsUpdate remains a compute upload request: execute/precompile that kernel
 * before drawing a newer host version. Never upload stale CPU simulation state
 * from the geometry owner, and never infer current GPU bounds from that array. */
export function computeAttributeSnapshot(device, attribute, {index = false, integer = false} = {}) {
  const view = vertexViews.get(attribute);
  if (view) {
    if (view.device !== device) fail('OWNER', 'Captured vertex storage belongs to another device');
    if (index) fail('FORMAT', 'Captured node vertex streams cannot be used as indices');
    if (integer ? ![Int32Array, Uint32Array].includes(view.scalar) : view.scalar !== Float32Array)
      fail('FORMAT', 'Captured storage signedness must match its vertex shader input');
    const resource = view.lease.resolve();
    return Object.freeze({buffer: resource.buffer, arrayStride: view.stride * 4, count: view.count,
      byteLength: view.byteLength, scalar: view.scalar, indexFormat: null,
      assertCurrent: view.lease.assertCurrent, whenIdle: view.lease.whenIdle});
  }
  const table = devices.get(device), entry = table?.get(attribute);
  if (!isComputeAttribute(attribute) || !entry)
    fail('UNPREPARED', 'Compute or compileComputeAsync must upload this storage attribute on the drawing device first');
  function assertCurrent() {
    entry.live();
    const current = table.get(attribute);
    if (current?.identity !== entry.identity || current.buffer !== entry.buffer ||
        current.stride !== entry.stride || current.count !== entry.count)
      fail('STALE', 'Shared vertex storage was released, replaced or resized; update geometry before drawing');
    if (attribute.isInterleavedBufferAttribute || attribute.normalized ||
        attribute.itemSize !== entry.stride || attribute.count !== entry.count ||
        !(attribute.array instanceof entry.scalar) || attribute.array.byteLength !== entry.byteLength)
      fail('SHAPE', 'The source attribute no longer describes its published GPU storage');
    if (attribute.version !== current.version)
      fail('UNPREPARED', 'Submit the pending storage attribute version through compute before drawing');
  }
  assertCurrent();
  if (index && integer) fail('FORMAT', 'Index and integer-vertex modes are distinct');
  if (index) {
    if (entry.scalar !== Uint32Array || entry.stride !== 1)
      fail('FORMAT', 'Shared index streams require scalar Uint32 storage');
    if ((entry.buffer.usage & 16) !== 16) fail('FORMAT', 'Shared index storage requires INDEX buffer usage');
  } else if (integer) {
    if (entry.scalar !== Int32Array && entry.scalar !== Uint32Array)
      fail('FORMAT', 'Shared integer vertex streams require Int32 or Uint32 storage');
  } else if (entry.scalar !== Float32Array) fail('FORMAT', 'Shared ordinary vertex and instance streams require Float32 storage');
  return Object.freeze({buffer: entry.buffer, arrayStride: entry.stride * 4, count: entry.count,
    byteLength: entry.byteLength, scalar: entry.scalar, indexFormat: index ? 'uint32' : null, assertCurrent, whenIdle: entry.whenIdle});
}
