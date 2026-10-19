/** Explicit pixel transfers for GPU-authored simulation textures.
 * The renderer's storage owner is the sole texture allocator. Uploads capture
 * logical pixel bytes at the call, then occupy the same FIFO as compute/copies.
 * These are F3D extensions, not a replacement for source Texture.needsUpdate.
 */
export class ThreeStorageTransferError extends Error {
  constructor(code, message) {
    super(`THREE_STORAGE_TRANSFER_${code}: ${message}`);
    this.name = 'ThreeStorageTransferError'; this.code = `THREE_STORAGE_TRANSFER_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeStorageTransferError(code, message); };
const natural = n => Number.isSafeInteger(n) && n >= 0;
const positive = n => natural(n) && n > 0;

/** Pixel coordinates address the selected mip. Array depth does not shrink. */
export function inspectStoragePixelRegion(shape, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected pixel transfer options');
  const allowed = new Set(['x', 'y', 'z', 'width', 'height', 'depth', 'mipLevel', 'offset', 'bytesPerRow', 'rowsPerImage', 'flipY']);
  for (const key of Object.keys(options)) if (!allowed.has(key)) fail('OPTIONS', `Unknown pixel transfer option: ${key}`);
  const {mipLevel = 0, x = 0, y = 0, z = 0, flipY = false} = options;
  if (!shape || !['2d', '2d-array', '3d'].includes(shape.viewDimension) ||
      ![shape.width, shape.height, shape.depth, shape.mipLevelCount, shape.texelBytes].every(positive))
    fail('SHAPE', 'Expected an admitted storage texture shape');
  if (!natural(mipLevel) || mipLevel >= shape.mipLevelCount) fail('LEVEL', 'Pixel mip level is not allocated');
  const extent = [shape.width, shape.height, shape.depth].map((n, axis) =>
    axis === 2 && shape.viewDimension !== '3d' ? n : Math.max(1, Math.floor(n / 2 ** mipLevel)));
  const origin = [x, y, z], size = [options.width ?? extent[0] - x, options.height ?? extent[1] - y, options.depth ?? extent[2] - z];
  if (typeof flipY !== 'boolean' || !origin.every(natural) || !size.every(positive) ||
      size.some((n, i) => origin[i] > extent[i] || n > extent[i] - origin[i]))
    fail('BOUNDS', 'Pixel region must contain positive integer texels inside the selected mip');
  const rowBytes = size[0] * shape.texelBytes, bytes = rowBytes * size[1] * size[2];
  const {offset = 0, bytesPerRow = rowBytes, rowsPerImage = size[1]} = options;
  const end = offset + ((size[2] - 1) * rowsPerImage + size[1] - 1) * bytesPerRow + rowBytes;
  if (!natural(offset) || !positive(bytesPerRow) || bytesPerRow < rowBytes || bytesPerRow % shape.texelBytes ||
      !positive(rowsPerImage) || rowsPerImage < size[1] || !positive(bytes) || !positive(end))
    fail('LAYOUT', 'Expected a safe byte offset, whole-texel row pitch and complete image stride');
  return Object.freeze({origin: Object.freeze(origin), size: Object.freeze(size), mipLevel, flipY,
    rowBytes, bytes, offset, bytesPerRow, rowsPerImage, end});
}

function sourceBytes(data) {
  const buffer = ArrayBuffer.isView(data) ? data.buffer : data;
  if (!(buffer instanceof ArrayBuffer) || buffer.resizable) fail('DATA', 'Expected fixed, unshared ArrayBuffer or view bytes');
  try { return ArrayBuffer.isView(data) ? new Uint8Array(buffer, data.byteOffset, data.byteLength) : new Uint8Array(buffer); }
  catch { fail('DATA', 'Pixel source buffer is detached'); }
}

export function createStorageTextureTransfers(renderer, {maxSnapshotBytes = 128 * 1024 * 1024, maxPendingTransfers = 64} = {}) {
  if (!renderer || !positive(maxSnapshotBytes) || !positive(maxPendingTransfers)) fail('OPTIONS', 'Expected renderer and positive transfer budgets');
  const pending = new Set(), captures = new Set();
  let snapshotBytes = 0, terminal = null, disposed = false, watched = null;
  let rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const signal = renderer._targetLifetime?.signal;
  function release(request) {
    if (!captures.delete(request)) return;
    snapshotBytes -= request.plan.bytes; request.data = null;
  }
  function stop(error) {
    if (terminal) return;
    terminal = error instanceof Error ? error : new ThreeStorageTransferError('ABORTED', String(error ?? 'Transfers stopped'));
    for (const request of captures) release(request);
    rejectStopped(terminal); signal?.removeEventListener('abort', abort);
  }
  const abort = () => stop(signal.reason);
  if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, {once: true});
  function live() {
    if (terminal) throw terminal;
    if (disposed || renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
    if (!renderer.isWebGPURenderer) fail('ROUTE', 'Pixel transfers require the WebGPU renderer surface');
    if (!renderer._initialized) fail('INITIALIZATION', 'Initialize the renderer before transferring storage pixels');
    if (renderer._deferredError) { const error = renderer._deferredError; renderer._deferredError = null; throw error; }
    const device = renderer._device;
    if (!device || !renderer._storageTextures) fail('DEVICE', 'Renderer has no storage texture device');
    if (watched && watched !== device) fail('DEVICE', 'Storage transfers cannot migrate between devices');
    if (!watched) {
      watched = device;
      device.lost?.then(info => stop(new ThreeStorageTransferError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    }
    return device;
  }
  function schedule(issue) {
    if (renderer._computeTurn || (!renderer._drain && !renderer._preparing)) return issue();
    return new Promise((resolve, reject) => renderer._enqueue({transfer: true, issue, resolve, reject}));
  }
  function track(result) {
    const task = Promise.race([Promise.resolve(result), stopped]);
    pending.add(task);
    task.catch(error => { stop(error); renderer._deferredError ??= error; }).finally(() => pending.delete(task)).catch(() => {});
    return task;
  }
  function write(texture, data, options = {}) {
    let request;
    try {
      const device = live(), owner = renderer._storageTextures;
      if (pending.size >= maxPendingTransfers) fail('BUDGET', 'Too many storage transfers are awaiting completion');
      const entry = owner.capture(texture), plan = inspectStoragePixelRegion(entry.shape, options);
      const binding = owner.captureBinding({texture, isSampledTexture: true, store: true, access: 'writeOnly', mipLevel: plan.mipLevel});
      const source = sourceBytes(data);
      if (source.byteLength < plan.end) fail('DATA', 'Pixel bytes do not fill the requested row/image layout');
      if (plan.bytes > maxSnapshotBytes - snapshotBytes) fail('BUDGET', 'Queued storage pixels exceed maxSnapshotBytes');
      const packed = new Uint8Array(plan.bytes), [, height, depth] = plan.size;
      for (let z = 0; z < depth; z++) for (let y = 0; y < height; y++) {
        const from = plan.offset + (z * plan.rowsPerImage + (plan.flipY ? height - 1 - y : y)) * plan.bytesPerRow;
        packed.set(source.subarray(from, from + plan.rowBytes), (z * height + y) * plan.rowBytes);
      }
      request = {entry, plan, binding, data: packed}; captures.add(request); snapshotBytes += plan.bytes;
      const result = schedule(() => {
        try {
          live(); owner.assertCurrent(entry);
          if (!request.data) fail('STALE', 'Storage pixel capture was released');
          // Borrow through the shared owner: allocation limits, reentry,
          // validation scopes, mip writes and completion stay in one place.
          owner.withTexture(entry, native => {
            device.queue.writeTexture({texture: native, mipLevel: plan.mipLevel, origin: plan.origin},
              request.data, {bytesPerRow: plan.rowBytes, rowsPerImage: height}, plan.size);
          }, binding);
          return owner.whenIdle();
        } finally { release(request); }
      });
      return track(result).finally(() => release(request));
    } catch (error) { if (request) release(request); return Promise.reject(error); }
  }
  return {write,
    get diagnostics() { return {snapshotBytes, pendingTransfers: pending.size}; },
    async whenIdle() { live(); await Promise.race([Promise.all([...pending]), stopped]); live(); },
    dispose() { if (!disposed) { disposed = true; stop(new ThreeStorageTransferError('DISPOSED', 'Storage pixel transfers disposed')); } },
  };
}
