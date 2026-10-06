/** Queue-ordered RGBA readback. The texture copy is submitted synchronously;
 * only buffer mapping waits. Returned rows exclude WebGPU's 256-byte padding.
 * Half-float values remain their original Uint16 bit patterns.
 */
export class GpuTargetReadbackError extends Error {
  constructor(code, message) {
    super(`GPU_READBACK_${code}: ${message}`);
    this.name = 'GpuTargetReadbackError'; this.code = `GPU_READBACK_${code}`;
  }
}
const fail = (code, message) => { throw new GpuTargetReadbackError(code, message); };

// One device-loss handler, not one permanently retained promise closure per
// frame. Completed reads remove their waiter and release all staging/output data.
const deviceLoss = new WeakMap();
function watchLoss(device, notify) {
  let state = deviceLoss.get(device);
  if (!state) {
    state = {waiters: new Set(), error: null}; deviceLoss.set(device, state);
    const stop = error => {
      state.error = error;
      for (const waiter of state.waiters) waiter(error);
      state.waiters.clear();
    };
    device.lost?.then(info => stop(new GpuTargetReadbackError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
  }
  if (state.error) notify(state.error); else state.waiters.add(notify);
  return () => state.waiters.delete(notify);
}

export async function readGpuTargetPixels(session, {x, y, width, height, flipY = false, output,
  signal, maxBytes = 64 * 1024 * 1024} = {}) {
  if (!session || typeof session.withTexture !== 'function') fail('SOURCE', 'Expected an offscreen session');
  if (session.disposed || session.failed) fail('SOURCE', 'Offscreen session is no longer usable');
  for (const value of [x, y, width, height])
    if (!Number.isSafeInteger(value) || value < 0) fail('REGION', 'Readback bounds must be non-negative integers');
  if (x > session.width - width || y > session.height - height) fail('REGION', 'Readback region exceeds the target');
  if (typeof flipY !== 'boolean' || !Number.isSafeInteger(maxBytes) || maxBytes < 1) fail('OPTIONS', 'Invalid readback options');
  const format = session.diagnostics.target?.format;
  const Type = format === 'rgba16float' ? Uint16Array : ['rgba8unorm', 'rgba8unorm-srgb'].includes(format) ? Uint8Array : null;
  if (!Type) fail('FORMAT', 'Readback supports RGBA8 and RGBA16F targets');
  const rowBytes = width * 4 * Type.BYTES_PER_ELEMENT, bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const size = bytesPerRow * height, elements = width * height * 4;
  const device = session.device;
  if (!Number.isSafeInteger(size) || size > maxBytes || (device.limits?.maxBufferSize !== undefined && size > device.limits.maxBufferSize))
    fail('BUDGET', 'Padded readback exceeds the byte budget or device buffer limit');
  if (output !== undefined && (!(output instanceof Type) || output.length < elements))
    fail('OUTPUT', 'Output array has the wrong component type or insufficient capacity');
  if (signal !== undefined && (!signal || typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function'))
    fail('OPTIONS', 'Expected AbortSignal');
  if (signal?.aborted) throw signal.reason ?? new GpuTargetReadbackError('ABORTED', 'Readback cancelled');
  const result = output ?? new Type(elements);
  if (width === 0 || height === 0) return result;
  let buffer = null, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const abort = () => rejectStopped(signal.reason ?? new GpuTargetReadbackError('ABORTED', 'Readback cancelled'));
  signal?.addEventListener('abort', abort, {once: true});
  const unwatch = watchLoss(device, rejectStopped);
  try {
    let validation, error;
    session.withTexture(texture => {
      device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
      try {
        buffer = device.createBuffer({label: 'Render target pixel readback', size, usage: 1 | 8});
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({texture, origin: {x, y: flipY ? session.height - y - height : y, z: 0}},
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
    const mapped = new Uint8Array(buffer.getMappedRange());
    // Do not write a caller buffer until mapping and validation both succeed.
    const destination = new Uint8Array(result.buffer, result.byteOffset, elements * Type.BYTES_PER_ELEMENT);
    for (let row = 0; row < height; row++) {
      const sourceRow = flipY ? height - row - 1 : row;
      destination.set(mapped.subarray(sourceRow * bytesPerRow, sourceRow * bytesPerRow + rowBytes), row * rowBytes);
    }
    return result;
  } finally {
    unwatch();
    signal?.removeEventListener('abort', abort);
    if (buffer) { try { buffer.unmap(); } finally { buffer.destroy(); } }
  }
}
