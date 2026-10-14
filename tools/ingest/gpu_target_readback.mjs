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
      state.depthPipelines?.clear();
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

// Sampling depth into a storage buffer supports rectangular reads of opaque
// depth24plus formats without COPY_SRC usage, filtering or a color conversion.
// WebGPU depth/stencil texture-to-buffer copies cannot provide that contract.
const DEPTH_FORMATS = new Set(['depth16unorm', 'depth24plus', 'depth32float',
  'depth24plus-stencil8', 'depth32float-stencil8']);
function depthReadbackPipeline(device, state, multisampled) {
  const pipelines = state.depthPipelines ??= new Map();
  if (pipelines.has(multisampled)) return pipelines.get(multisampled);
  const module = device.createShaderModule({label: 'Depth pixel readback', code: `
struct Region {
  origin: vec2u,
  extent: vec2u,
  sampleIndex: u32,
  flipY: u32,
  padding: vec2u,
}
@group(0) @binding(0) var source: ${multisampled ? 'texture_depth_multisampled_2d' : 'texture_depth_2d'};
@group(0) @binding(1) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(2) var<uniform> region: Region;
@compute @workgroup_size(8, 8) fn cs(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= region.extent)) { return; }
  let row = select(id.y, region.extent.y - 1u - id.y, region.flipY != 0u);
  let coordinate = vec2i(region.origin + vec2u(id.x, row));
  pixels[id.y * region.extent.x + id.x] = textureLoad(source, coordinate, ${multisampled ? 'i32(region.sampleIndex)' : '0'});
}`});
  const pipeline = device.createComputePipeline({label: 'Depth pixel readback', layout: 'auto', compute: {module, entryPoint: 'cs'}});
  pipelines.set(multisampled, pipeline);
  return pipeline;
}

/** Read sampled depth as tightly packed Float32Array values in native depth
 * space, not linearized distance. Requires sampleDepth on the target. MSAA reads
 * require an explicit sampleIndex: samples are never averaged or resolved.
 * Submission captures depth before returning its promise. The same per-device
 * budgets and cancellation gate as color readback own all temporary buffers;
 * maxBytes limits each data buffer, maxInFlightBytes also charges the compute
 * output, 32-byte uniform and any internally allocated host output.
 */
export async function readGpuTargetDepth(session, {x = 0, y = 0, width, height, sampleIndex,
  mipLevel = 0, flipY = false, output, signal, maxBytes = 64 * 1024 * 1024,
  maxInFlightBytes = 128 * 1024 * 1024} = {}) {
  if (!session || typeof session.withDepthTexture !== 'function' || session.disposed || session.failed)
    fail('SOURCE', 'Expected a live sampled-depth session');
  for (const value of [session.width, session.height])
    if (!Number.isSafeInteger(value) || value < 0 || value > 0x7fffffff)
      fail('SOURCE', 'Expected finite integer depth dimensions');
  const shape = session.diagnostics?.target ?? session.diagnostics;
  if (!DEPTH_FORMATS.has(shape?.depthFormat)) fail('FORMAT', 'Expected a depth attachment');
  if (shape.sampleDepth === false) fail('SOURCE', 'Enable sampleDepth before reading depth');
  const samples = shape.sampleCount;
  if (![1, 4].includes(samples)) fail('SOURCE', 'Expected one or four depth samples');
  if (sampleIndex === undefined && samples === 1) sampleIndex = 0;
  if (!Number.isSafeInteger(sampleIndex) || sampleIndex < 0 || sampleIndex >= samples)
    fail('SAMPLES', 'Supply a valid explicit sampleIndex for multisampled depth');
  if (mipLevel !== 0) fail('REGION', 'Depth attachments have only mip level zero');
  if (width === undefined) width = session.width - x;
  if (height === undefined) height = session.height - y;
  for (const value of [x, y, width, height])
    if (!Number.isSafeInteger(value) || value < 0) fail('REGION', 'Depth readback bounds must be non-negative integers');
  if (x > session.width - width || y > session.height - height) fail('REGION', 'Depth readback region exceeds the target');
  if (typeof flipY !== 'boolean' || ![maxBytes, maxInFlightBytes].every(n => Number.isSafeInteger(n) && n > 0))
    fail('OPTIONS', 'Invalid depth readback options');
  const device = session.device, elements = width * height, bytes = elements * 4;
  if (!device || typeof device.createBuffer !== 'function') fail('SOURCE', 'Expected a session device');
  const groupsX = Math.ceil(width / 8), groupsY = Math.ceil(height / 8);
  if (!Number.isSafeInteger(bytes) || bytes > maxBytes ||
      (device.limits?.maxBufferSize !== undefined && Math.max(bytes, 32) > device.limits.maxBufferSize) ||
      (device.limits?.maxStorageBufferBindingSize !== undefined && bytes > device.limits.maxStorageBufferBindingSize) ||
      (device.limits?.maxComputeWorkgroupsPerDimension !== undefined &&
        Math.max(groupsX, groupsY) > device.limits.maxComputeWorkgroupsPerDimension))
    fail('BUDGET', 'Depth readback exceeds its byte budget or device compute limits');
  if (output !== undefined && (!(output instanceof Float32Array) || output.length < elements))
    fail('OUTPUT', 'Depth output must be a Float32Array with sufficient capacity');
  if (signal !== undefined && (!signal || typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function'))
    fail('OPTIONS', 'Expected AbortSignal');
  const checkAbort = () => {
    if (signal?.aborted) throw signal.reason ?? new GpuTargetReadbackError('ABORTED', 'Depth readback cancelled');
  };
  checkAbort();
  if (width === 0 || height === 0) return output ?? new Float32Array(0);
  if (['createShaderModule', 'createComputePipeline', 'createBindGroup', 'createCommandEncoder', 'pushErrorScope', 'popErrorScope']
    .some(key => typeof device[key] !== 'function') || typeof device.queue?.writeBuffer !== 'function' || typeof device.queue?.submit !== 'function')
    fail('SOURCE', 'Depth readback requires WebGPU compute and queue submission');
  const state = deviceState(device), charged = bytes * (output === undefined ? 3 : 2) + 32;
  if (!Number.isSafeInteger(charged)) fail('BUDGET', 'Depth readback allocation exceeds safe integer precision');
  const release = reserve(state, charged, maxInFlightBytes);
  let staging = null, storage = null, uniform = null, rejectStopped, pipeline;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const abort = () => rejectStopped(signal.reason ?? new GpuTargetReadbackError('ABORTED', 'Depth readback cancelled'));
  state.waiters.add(rejectStopped);
  try {
    signal?.addEventListener('abort', abort, {once: true});
    checkAbort();
    const result = output ?? new Float32Array(elements);
    let validation, error;
    session.withDepthTexture(texture => {
      device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
      try {
        pipeline = depthReadbackPipeline(device, state, samples > 1);
        storage = device.createBuffer({label: 'Depth readback compute output', size: bytes, usage: 128 | 4});
        staging = device.createBuffer({label: 'Depth readback mapping', size: bytes, usage: 1 | 8});
        uniform = device.createBuffer({label: 'Depth readback region', size: 32, usage: 64 | 8});
        device.queue.writeBuffer(uniform, 0, new Uint32Array([x, flipY ? session.height - y - height : y,
          width, height, sampleIndex, flipY ? 1 : 0, 0, 0]));
        const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
          {binding: 0, resource: texture.createView({aspect: 'depth-only', baseMipLevel: 0, mipLevelCount: 1})},
          {binding: 1, resource: {buffer: storage}}, {binding: 2, resource: {buffer: uniform}},
        ]});
        const encoder = device.createCommandEncoder({label: 'Depth pixel readback'}), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
        encoder.copyBufferToBuffer(storage, 0, staging, 0, bytes);
        device.queue.submit([encoder.finish()]);
      } catch (cause) { error = cause; }
      const invalid = device.popErrorScope(), oom = device.popErrorScope();
      validation = Promise.all([invalid, oom]).then(errors => {
        const cause = errors.find(Boolean);
        if (cause) fail('GPU', cause.message ?? 'GPU rejected depth readback');
      });
      validation.catch(() => {});
    });
    if (error) throw error;
    if (!staging) fail('SOURCE', 'Depth storage was suspended before submission');
    await Promise.race([Promise.all([validation, staging.mapAsync(1)]), stopped]);
    checkAbort();
    if (state.error) throw state.error;
    const mapped = new Uint8Array(staging.getMappedRange());
    if (mapped.byteLength < bytes) fail('GPU', 'Mapped depth buffer is smaller than the submitted region');
    new Uint8Array(result.buffer, result.byteOffset, bytes).set(mapped.subarray(0, bytes));
    return result;
  } catch (error) {
    // Do not retain an asynchronously invalid pipeline for later reads.
    if (pipeline && state.depthPipelines?.get(samples > 1) === pipeline) state.depthPipelines.delete(samples > 1);
    throw error;
  } finally {
    state.waiters.delete(rejectStopped);
    try {
      signal?.removeEventListener('abort', abort);
    } finally {
      try {
        if (staging) { try { staging.unmap(); } finally { staging.destroy(); } }
      } finally {
        try { storage?.destroy(); }
        finally { try { uniform?.destroy(); } finally { release(); } }
      }
    }
  }
}
