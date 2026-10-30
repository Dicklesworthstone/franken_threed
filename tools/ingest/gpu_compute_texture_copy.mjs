/** Bit-preserving, GPU-only storage-buffer -> source RenderTarget color copy.
 * No mapping, CPU upload, temporary texture or precision conversion is used.
 * Aligned rows use one copy; tight unaligned rows (or flipY) use row copies in
 * one command buffer. Selected authored mips and cube/array/volume slices share
 * the normal target owner's allocation, mip tracking and sampling invalidation.
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
const KEYS = new Set(['x', 'y', 'width', 'height', 'offset', 'bytesPerRow', 'flipY', 'face', 'destinationMipLevel']);
const integer = n => Number.isSafeInteger(n) && n >= 0;

/** Validate the complete copy before target allocation or command encoding.
 * offset and bytesPerRow are bytes in GPU storage, including vec3 padding. */
export function inspectComputeTextureCopy(byteLength, target, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options) || Object.keys(options).some(k => !KEYS.has(k)))
    fail('OPTIONS', 'Expected buffer-to-target copy options');
  const texelBytes = TEXEL_BYTES.get(target?.format);
  if (!texelBytes || target.sampleCount !== 1) fail('TARGET', 'Expected a single-sample RGBA8/RGBA16F/R/RG/RGBA32F target');
  const {destinationMipLevel = 0} = options, mipLevelCount = target.mipLevelCount ?? 1;
  if (!integer(destinationMipLevel) || !integer(mipLevelCount) || mipLevelCount < 1 || destinationMipLevel >= mipLevelCount)
    fail('LEVEL', 'Destination mip level is not allocated');
  if ((destinationMipLevel > 0 || target.cube) && target.generateMipmaps !== false)
    fail('MIP', 'Higher destination mips and cube faces require authored, non-generated storage');
  const divisor = 2 ** destinationMipLevel;
  const mipWidth = target.width === 0 ? 0 : Math.max(1, Math.floor(target.width / divisor));
  const mipHeight = target.height === 0 ? 0 : Math.max(1, Math.floor(target.height / divisor));
  const layers = target.cube ? 6 : target.layers ?? 1;
  const slices = target.volume ? Math.max(1, Math.floor(layers / divisor)) : layers;
  const face = options.face === undefined && !target.cube && !target.array && !target.volume && layers === 1 ? 0 : options.face;
  if (!integer(layers) || layers < 1 || !integer(face) || face >= slices)
    fail('FACE', 'Explicitly select an allocated cube face, array layer or mip-local volume slice');
  const {x = 0, y = 0, width = mipWidth - x, height = mipHeight - y,
    offset = 0, bytesPerRow = width * texelBytes, flipY = false} = options;
  if (![byteLength, target.width, target.height, x, y, width, height, offset, bytesPerRow].every(integer) ||
      x > mipWidth || width > mipWidth - x || y > mipHeight || height > mipHeight - y ||
      typeof flipY !== 'boolean' || offset % texelBytes || bytesPerRow % texelBytes || bytesPerRow < width * texelBytes)
    fail('LAYOUT', 'Expected in-bounds integer extents and texel-aligned byte offsets/row strides');
  const empty = width === 0 || height === 0;
  const end = empty ? offset : offset + (height - 1) * bytesPerRow + width * texelBytes;
  if (!Number.isSafeInteger(end) || end > byteLength) fail('BOUNDS', 'Copy reads beyond the GPU storage buffer');
  return Object.freeze({x, y, width, height, offset, bytesPerRow, flipY, empty, face, destinationMipLevel,
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
  const write = {entry: destination, texture: destination.shape.textures[0], device, depth: false, colorIndex: 0};
  // Budget admission and allocation failures remain retryable before write scopes.
  targets.validateNativeTextures([write]);
  targets.ensure(destination);
  let error;
  device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
  try {
    targets.withNativeColorDestination(write, texture => {
      const encoder = device.createCommandEncoder({label: 'F3D compute storage to target'});
      if (plan.singleCopy) {
        const source = {buffer, offset: plan.offset};
        if (plan.height > 1) source.bytesPerRow = plan.bytesPerRow;
        encoder.copyBufferToTexture(source, {texture, mipLevel: plan.destinationMipLevel, origin: [plan.x, plan.y, plan.face]}, [plan.width, plan.height, 1]);
      } else {
        // WebGPU's 256-byte row alignment applies to multi-row copies, not a
        // single row. Tight compute arrays need neither CPU nor staging repacks.
        for (let row = 0; row < plan.height; row++) {
          const sourceRow = plan.flipY ? plan.height - 1 - row : row;
          encoder.copyBufferToTexture({buffer, offset: plan.offset + sourceRow * plan.bytesPerRow},
            {texture, mipLevel: plan.destinationMipLevel, origin: [plan.x, plan.y + row, plan.face]}, [plan.width, 1, 1]);
        }
      }
      targets.assertCurrent(destination);
      device.queue.submit([encoder.finish()]);
    });
  } catch (cause) { error = cause; }
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
