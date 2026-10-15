/** Bit-preserving, GPU-only storage-buffer -> source RenderTarget color copy.
 * No mapping, CPU upload, temporary texture or precision conversion is used.
 * Aligned rows use one copy; tight unaligned rows (or flipY) use row copies in
 * one command buffer. The normal target owner maintains mipmaps/UV replicas.
 * Native coordinates are top-left. This is an explicit F3D extension, not an
 * implementation claim for upstream texture-backed TSL compute bindings. */
export class GpuComputeTextureCopyError extends Error {
  constructor(code, message) {
    super(`GPU_COMPUTE_TEXTURE_${code}: ${message}`);
    this.name = 'GpuComputeTextureCopyError'; this.code = `GPU_COMPUTE_TEXTURE_${code}`;
  }
}
const fail = (code, message) => { throw new GpuComputeTextureCopyError(code, message); };
const TEXEL_BYTES = new Map([['rgba8unorm', 4], ['rgba8unorm-srgb', 4], ['rgba16float', 8],
  ['r32float', 4], ['rg32float', 8], ['rgba32float', 16]]);
const KEYS = new Set(['x', 'y', 'width', 'height', 'offset', 'bytesPerRow', 'flipY']);
const integer = n => Number.isSafeInteger(n) && n >= 0;

/** Validate the complete copy before target allocation or command encoding.
 * offset and bytesPerRow are bytes in GPU storage, including vec3 padding. */
export function inspectComputeTextureCopy(byteLength, target, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options) || Object.keys(options).some(k => !KEYS.has(k)))
    fail('OPTIONS', 'Expected buffer-to-target copy options');
  const texelBytes = TEXEL_BYTES.get(target?.format);
  if (!texelBytes || target.sampleCount !== 1) fail('TARGET', 'Expected a single-sample RGBA8/RGBA16F/R/RG/RGBA32F target');
  const {x = 0, y = 0, width = target.width - x, height = target.height - y,
    offset = 0, bytesPerRow = width * texelBytes, flipY = false} = options;
  if (![byteLength, target.width, target.height, x, y, width, height, offset, bytesPerRow].every(integer) ||
      x > target.width || width > target.width - x || y > target.height || height > target.height - y ||
      typeof flipY !== 'boolean' || offset % texelBytes || bytesPerRow % texelBytes || bytesPerRow < width * texelBytes)
    fail('LAYOUT', 'Expected in-bounds integer extents and texel-aligned byte offsets/row strides');
  const empty = width === 0 || height === 0;
  const end = empty ? offset : offset + (height - 1) * bytesPerRow + width * texelBytes;
  if (!Number.isSafeInteger(end) || end > byteLength) fail('BOUNDS', 'Copy reads beyond the GPU storage buffer');
  return Object.freeze({x, y, width, height, offset, bytesPerRow, flipY, empty,
    singleCopy: height <= 1 || (!flipY && bytesPerRow % 256 === 0)});
}

function cancellable(promise, signals) {
  return new Promise((resolve, reject) => {
    const handlers = signals.filter(Boolean).map(signal => [signal, () => reject(signal.reason ?? new GpuComputeTextureCopyError('ABORTED', 'Copy cancelled'))]);
    for (const [signal, abort] of handlers) {
      signal.addEventListener('abort', abort, {once: true});
      if (signal.aborted) abort();
    }
    Promise.resolve(promise).then(resolve, reject).finally(() => {
      for (const [signal, abort] of handlers) signal.removeEventListener('abort', abort);
    });
  });
}

/** The caller lends its OWN compute buffer and target owner from the same
 * renderer/device. Both must remain owned; no raw GPU handle is exposed by the
 * public renderer API. Submit synchronously before returning the completion. */
export function copyComputeBufferToTarget(device, buffer, targets, destination, options = {}, {signal, sourceCompletion} = {}) {
  targets.assertCurrent(destination);
  if (signal?.aborted) throw signal.reason ?? new GpuComputeTextureCopyError('ABORTED', 'Copy cancelled');
  if (!buffer || !(buffer.usage & 4)) fail('SOURCE', 'Expected GPU storage with COPY_SRC usage');
  const plan = inspectComputeTextureCopy(buffer.size, destination.shape, options);
  if (plan.empty) return Promise.resolve(false);
  if (destination.busy) fail('REENTRANT', 'Cannot copy during a target submission');
  targets.ensure(destination);
  const depthDirty = destination.depthDirty;
  let error;
  device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
  try {
    targets.render(destination, (_frame, texture) => {
      const encoder = device.createCommandEncoder({label: 'F3D compute storage to target'});
      if (plan.singleCopy) {
        const source = {buffer, offset: plan.offset};
        if (plan.height > 1) source.bytesPerRow = plan.bytesPerRow;
        encoder.copyBufferToTexture(source, {texture, mipLevel: 0, origin: [plan.x, plan.y, 0]}, [plan.width, plan.height, 1]);
      } else {
        // WebGPU's 256-byte row alignment applies to multi-row copies, not a
        // single row. Tight compute arrays need neither CPU nor staging repacks.
        for (let row = 0; row < plan.height; row++) {
          const sourceRow = plan.flipY ? plan.height - 1 - row : row;
          encoder.copyBufferToTexture({buffer, offset: plan.offset + sourceRow * plan.bytesPerRow},
            {texture, mipLevel: 0, origin: [plan.x, plan.y + row, 0]}, [plan.width, 1, 1]);
        }
      }
      targets.assertCurrent(destination);
      device.queue.submit([encoder.finish()]);
    });
  } catch (cause) { error = cause; }
  finally {
    // withFrame invalidates color/mips and the target sampling replica. It also
    // conservatively dirties depth for draws; this operation never writes depth.
    destination.depthDirty = depthDirty;
  }
  // Error scopes may not span awaits on a renderer's shared GPUDevice.
  const validation = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(values => {
    if (error) throw error;
    const failure = values.find(Boolean);
    if (failure) throw new GpuComputeTextureCopyError('GPU', failure.message ?? 'GPU texture copy failed');
  });
  const produced = error ? Promise.resolve() : sourceCompletion?.();
  destination.validation = Promise.all([destination.validation, validation, produced]).then(() => {});
  destination.validation.catch(cause => {
    destination.error ??= cause; destination.lifetime.abort(cause);
  });
  if (error) throw error;
  const completed = Promise.all([destination.validation, destination.storage.whenIdle(),
    device.queue.onSubmittedWorkDone()]);
  return cancellable(completed, [signal, destination.lifetime.signal]).then(() => {
    targets.assertCurrent(destination); return true;
  });
}
