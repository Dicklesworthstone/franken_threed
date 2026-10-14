/** Queue-ordered color readback. The texture copy is submitted synchronously;
 * only buffer mapping waits. Returned rows exclude WebGPU's 256-byte padding.
 * mipLevel selects an allocated color mip; bounds and omitted extents are local
 * to that mip, including NPOT and 1D tails. No implicit downsampling is performed.
 * BGRA targets return RGBA bytes; float32 targets preserve full precision and
 * half-float values remain their original Uint16 bit patterns. Scalar/vector
 * targets return one/two components per pixel, without inventing RGBA channels.
 * maxBytes bounds each staging buffer. maxInFlightBytes bounds shared per-device
 * staging plus internally allocated output for all unfinished reads. At most 64
 * reads may be in flight per device; budget failures are retryable after drain.
 */
export class GpuTargetReadbackError extends Error {
  constructor(code, message) {
    super(`GPU_READBACK_${code}: ${message}`);
    this.name = 'GpuTargetReadbackError'; this.code = `GPU_READBACK_${code}`;
  }
}
const fail = (code, message) => { throw new GpuTargetReadbackError(code, message); };
const FORMATS = new Map([
  ['rgba8unorm', [Uint8Array, 4, false]], ['rgba8unorm-srgb', [Uint8Array, 4, false]],
  ['bgra8unorm', [Uint8Array, 4, true]], ['bgra8unorm-srgb', [Uint8Array, 4, true]],
  ['rgba16float', [Uint16Array, 4, false]], ['rgba32float', [Float32Array, 4, false]],
  ['r32float', [Float32Array, 1, false]], ['rg32float', [Float32Array, 2, false]],
]);

// One device-loss handler, not one permanently retained promise closure per
// frame. Completed reads remove their waiter and release all staging/output data.
const devices = new WeakMap();
function deviceState(device) {
  let state = devices.get(device);
  if (!state) {
    state = {waiters: new Set(), error: null, bytes: 0, reservations: new Map()};
    devices.set(device, state);
    const stop = cause => {
      state.error = cause instanceof Error ? cause : new GpuTargetReadbackError('DEVICE_LOST', cause?.message ?? 'Device lost');
      for (const waiter of state.waiters) waiter(state.error);
      state.waiters.clear();
    };
    device.lost?.then(info => stop(new GpuTargetReadbackError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
  }
  return state;
}
function reserve(state, bytes, limit) {
  if (state.error) throw state.error;
  // A later caller cannot relax the limit promised to an unfinished read.
  let ceiling = limit;
  for (const activeLimit of state.reservations.values()) ceiling = Math.min(ceiling, activeLimit);
  if (state.reservations.size >= 64 || bytes > ceiling - state.bytes)
    fail('BUDGET', 'Concurrent readbacks exceed the shared byte or request budget');
  const token = {};
  state.reservations.set(token, limit); state.bytes += bytes;
  return () => {
    if (state.reservations.delete(token)) state.bytes -= bytes;
  };
}

export async function readGpuTargetPixels(session, {x = 0, y = 0, width, height, mipLevel = 0, flipY = false, output,
  signal, maxBytes = 64 * 1024 * 1024, maxInFlightBytes = 128 * 1024 * 1024} = {}) {
  if (!session || typeof session.withTexture !== 'function') fail('SOURCE', 'Expected an offscreen session');
  if (session.disposed || session.failed) fail('SOURCE', 'Offscreen session is no longer usable');
  for (const value of [session.width, session.height])
    if (!Number.isSafeInteger(value) || value < 0) fail('SOURCE', 'Expected finite integer target dimensions');
  const diagnostics = session.diagnostics;
  let mipLevelCount = diagnostics?.target?.mipLevelCount ?? diagnostics?.mipLevelCount ?? session.mipLevelCount ?? 1;
  // Suspended owners have no allocation, but still admit an empty base-level read.
  if (mipLevelCount === 0 && (session.width === 0 || session.height === 0)) mipLevelCount = 1;
  const maxMipLevels = session.width && session.height ? 1 + Math.floor(Math.log2(Math.max(session.width, session.height))) : 1;
  if (!Number.isSafeInteger(mipLevelCount) || mipLevelCount < 1 || mipLevelCount > maxMipLevels)
    fail('SOURCE', 'Invalid allocated mip count');
  if (!Number.isSafeInteger(mipLevel) || mipLevel < 0 || mipLevel >= mipLevelCount)
    fail('REGION', 'Readback mip level is not allocated');
  const divisor = 2 ** mipLevel;
  const mipWidth = session.width === 0 ? 0 : Math.max(1, Math.floor(session.width / divisor));
  const mipHeight = session.height === 0 ? 0 : Math.max(1, Math.floor(session.height / divisor));
  if (width === undefined) width = mipWidth - x;
  if (height === undefined) height = mipHeight - y;
  for (const value of [x, y, width, height])
    if (!Number.isSafeInteger(value) || value < 0) fail('REGION', 'Readback bounds must be non-negative integers');
  if (x > mipWidth - width || y > mipHeight - height) fail('REGION', 'Readback region exceeds the target');
  if (typeof flipY !== 'boolean' || ![maxBytes, maxInFlightBytes].every(n => Number.isSafeInteger(n) && n > 0))
    fail('OPTIONS', 'Invalid readback options');
  const layout = FORMATS.get(diagnostics?.target?.format ?? diagnostics?.format);
  if (!layout) fail('FORMAT', 'Readback requires RGBA/BGRA8, RGBA16F, or R/RG/RGBA32F color storage');
  const [Type, components, bgra] = layout;
  const rowBytes = width * components * Type.BYTES_PER_ELEMENT, bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const size = bytesPerRow * height, elements = width * height * components;
  const packedBytes = elements * Type.BYTES_PER_ELEMENT, device = session.device;
  if (!device || typeof device.createBuffer !== 'function') fail('SOURCE', 'Expected a session device');
  if (![size, packedBytes, bytesPerRow].every(Number.isSafeInteger) || size > maxBytes ||
      (device.limits?.maxBufferSize !== undefined && size > device.limits.maxBufferSize))
    fail('BUDGET', 'Padded readback exceeds the byte budget or device buffer limit');
  if (output !== undefined && (!(output instanceof Type) || output.length < elements))
    fail('OUTPUT', 'Output array has the wrong component type or insufficient capacity');
  if (signal !== undefined && (!signal || typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function'))
    fail('OPTIONS', 'Expected AbortSignal');
  const checkAbort = () => {
    if (signal?.aborted) throw signal.reason ?? new GpuTargetReadbackError('ABORTED', 'Readback cancelled');
  };
  checkAbort();
  if (width === 0 || height === 0) return output ?? new Type(0);
  const state = deviceState(device);
  const charged = size + (output === undefined ? packedBytes : 0);
  if (!Number.isSafeInteger(charged)) fail('BUDGET', 'Readback allocation exceeds safe integer precision');
  const release = reserve(state, charged, maxInFlightBytes);
  let buffer = null, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const abort = () => rejectStopped(signal.reason ?? new GpuTargetReadbackError('ABORTED', 'Readback cancelled'));
  state.waiters.add(rejectStopped);
  try {
    signal?.addEventListener('abort', abort, {once: true});
    checkAbort();
    const result = output ?? new Type(elements);
    let validation, error;
    session.withTexture(texture => {
      device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
      try {
        buffer = device.createBuffer({label: 'Render target pixel readback', size, usage: 1 | 8});
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({texture, mipLevel, origin: {x, y: flipY ? mipHeight - y - height : y, z: 0}},
          {buffer, bytesPerRow, rowsPerImage: height}, {width, height, depthOrArrayLayers: 1});
        device.queue.submit([encoder.finish()]);
      } catch (cause) { error = cause; }
      const invalid = device.popErrorScope(), oom = device.popErrorScope();
      validation = Promise.all([invalid, oom]).then(errors => {
        const cause = errors.find(Boolean);
        if (cause) fail('GPU', cause.message ?? 'GPU rejected the readback copy');
      });
      validation.catch(() => {});
    });
    if (error) throw error;
    if (!buffer) fail('SOURCE', 'Target storage was suspended before the copy');
    await Promise.race([Promise.all([validation, buffer.mapAsync(1)]), stopped]);
    // Mapping may win Promise.race in the same turn as cancellation/device loss.
    // The caller's array is committed only after all three conditions hold.
    checkAbort();
    if (state.error) throw state.error;
    const mapped = new Uint8Array(buffer.getMappedRange());
    if (mapped.byteLength < size) fail('GPU', 'Mapped readback buffer is smaller than the submitted copy');
    const destination = new Uint8Array(result.buffer, result.byteOffset, packedBytes);
    for (let row = 0; row < height; row++) {
      const sourceRow = flipY ? height - row - 1 : row;
      const source = mapped.subarray(sourceRow * bytesPerRow, sourceRow * bytesPerRow + rowBytes);
      const offset = row * rowBytes;
      if (bgra) {
        for (let pixel = 0; pixel < rowBytes; pixel += 4) {
          destination[offset + pixel] = source[pixel + 2];
          destination[offset + pixel + 1] = source[pixel + 1];
          destination[offset + pixel + 2] = source[pixel];
          destination[offset + pixel + 3] = source[pixel + 3];
        }
      } else destination.set(source, offset);
    }
    return result;
  } finally {
    state.waiters.delete(rejectStopped);
    try {
      signal?.removeEventListener('abort', abort);
    } finally {
      try {
        if (buffer) { try { buffer.unmap(); } finally { buffer.destroy(); } }
      } finally { release(); }
    }
  }
}
