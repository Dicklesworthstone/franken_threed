/** Internal compute -> vertex-buffer ownership seam. A source storage attribute
 * identifies one published allocation per device. Geometry only borrows it:
 * no readback, CPU upload, duplicate allocation or ownership transfer occurs.
 * Publication is after compute upload (including compileComputeAsync). Native
 * buffer usages must include VERTEX as well as STORAGE from creation onward. */
const devices = new WeakMap();
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
  const owned = new Set(), identity = {};
  let disposed = false;
  function release(attribute) {
    if (table.get(attribute)?.identity === identity) table.delete(attribute);
    owned.delete(attribute);
  }
  function admit(attribute, array) {
      if (!isComputeAttribute(attribute)) return;
      if (disposed) fail('DISPOSED', 'Compute attribute owner is disposed');
      const existing = table.get(attribute);
      if (existing && existing.identity !== identity)
        fail('OWNER', 'The same storage attribute cannot have two compute owners on one device');
      const stride = attribute.itemSize, count = attribute.count;
      if (!Number.isSafeInteger(stride) || stride < 1 || !Number.isSafeInteger(count) || count < 1 ||
          stride * count !== array.length || array.BYTES_PER_ELEMENT !== 4)
        fail('SHAPE', 'Publish the normalized 32-bit storage layout before vertex use');
  }
  return {
    admit,
    publish(attribute, buffer, array, version, live, whenIdle) {
      if (!isComputeAttribute(attribute)) return;
      admit(attribute, array);
      // A borrowed GPU allocation must actually support vertex fetch. Reject
      // malformed or foreign publications before they become visible to draws.
      const stride = attribute.itemSize, count = attribute.count;
      const required = array.byteLength;
      if (!buffer || !Number.isSafeInteger(buffer.size) || buffer.size < required ||
          !Number.isSafeInteger(buffer.usage) || (buffer.usage & (128 | 16 | 32)) !== (128 | 16 | 32) ||
          typeof live !== 'function' || typeof whenIdle !== 'function' ||
          !Number.isSafeInteger(version) || version < 0 || version !== attribute.version)
        fail('PUBLICATION', 'Shared draw storage needs a live, sufficiently sized STORAGE|INDEX|VERTEX buffer at the current source version');
      table.set(attribute, {identity, buffer, stride, count, version, scalar: array.constructor,
        byteLength: array.byteLength, live, whenIdle});
      owned.add(attribute);
    },
    release,
    dispose() { if (disposed) return; disposed = true; for (const attribute of owned) release(attribute); },
  };
}

/** Only buffers published by the compute owner on this device are admitted.
 * needsUpdate remains a compute upload request: execute/precompile that kernel
 * before drawing a newer host version. Never upload stale CPU simulation state
 * from the geometry owner, and never infer current GPU bounds from that array. */
export function computeAttributeSnapshot(device, attribute, {index = false} = {}) {
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
  if (index) {
    if (entry.scalar !== Uint32Array || entry.stride !== 1)
      fail('FORMAT', 'Shared index streams require scalar Uint32 storage');
    if ((entry.buffer.usage & 16) !== 16) fail('FORMAT', 'Shared index storage requires INDEX buffer usage');
  } else if (entry.scalar !== Float32Array) fail('FORMAT', 'Shared ordinary vertex and instance streams require Float32 storage');
  return Object.freeze({buffer: entry.buffer, arrayStride: entry.stride * 4, count: entry.count,
    byteLength: entry.byteLength, indexFormat: index ? 'uint32' : null, assertCurrent, whenIdle: entry.whenIdle});
}
