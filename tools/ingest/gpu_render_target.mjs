/** Owned offscreen color/depth/MSAA attachments for immediate WebGPU rendering.
 * The resolved color texture supports sampling and COPY_SRC readback. All native
 * handles are borrowed only inside withFrame/withTexture, with consumers submitted
 * before returning. generateMipmaps:true allocates a full resolved color pyramid;
 * withTexture refreshes it on the GPU after the last withFrame write, before the
 * consumer runs. Attachments always select level zero; MSAA/depth stay single-level.
 * No framebuffer emulation, CPU pixel copies, or frame loop.
 */
export class GpuRenderTargetError extends Error {
  constructor(code, message) {
    super(`GPU_TARGET_${code}: ${message}`);
    this.name = 'GpuRenderTargetError';
    this.code = `GPU_TARGET_${code}`;
  }
}
const fail = (code, message) => { throw new GpuRenderTargetError(code, message); };
const COLOR_BYTES = new Map([
  ['rgba8unorm', 4], ['rgba8unorm-srgb', 4],
  ['bgra8unorm', 4], ['bgra8unorm-srgb', 4], ['rgba16float', 8],
]);

// One source level per pass keeps sampled/attachment subresources disjoint.
// sRGB textures decode before bilinear filtering and encode in the attachment.
// Destination pixel centers map to the next floor-halved extent, including NPOT
// and one-dimensional tails. This is bilinear reduction, not an area-box filter.
const MIP_WGSL = `
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let size = max(textureDimensions(source) / 2u, vec2u(1u));
  return textureSampleLevel(source, linearSampler, p.xy / vec2f(size), 0.0);
}`;

export function createGpuRenderTarget(device, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    fail('OPTIONS', 'Expected render target options');
  for (const key of Object.keys(options))
    if (!['width', 'height', 'format', 'depthFormat', 'sampleCount', 'maxBytes', 'label', 'generateMipmaps'].includes(key))
      fail('OPTIONS', `Unknown target option: ${key}`);
  const {
    width: initialWidth = 1, height: initialHeight = 1,
    format = 'rgba16float', depthFormat = 'depth24plus', sampleCount = 1,
    maxBytes = 256 * 1024 * 1024, label = 'Offscreen render target', generateMipmaps = false,
  } = options;
  if (!COLOR_BYTES.has(format) || ![null, 'depth24plus', 'depth32float'].includes(depthFormat))
    fail('FORMAT', 'Unsupported color or depth format');
  if (![1, 4].includes(sampleCount)) fail('SAMPLES', 'Expected one or four samples');
  if (typeof generateMipmaps !== 'boolean') fail('OPTIONS', 'generateMipmaps must be boolean');
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || typeof label !== 'string')
    fail('OPTIONS', 'Expected a positive byte budget and string label');
  if (!device || ['createTexture', 'pushErrorScope', 'popErrorScope'].some(k => typeof device[k] !== 'function') ||
      typeof device.queue?.onSubmittedWorkDone !== 'function') fail('DEVICE', 'Expected a WebGPU device');
  const limit = device.limits?.maxTextureDimension2D;
  if (!Number.isSafeInteger(limit) || limit < 1) fail('DEVICE', 'Missing texture dimension limit');
  const dimension = value => {
    if (!Number.isSafeInteger(value) || value < 0 || value > limit)
      fail('SIZE', 'Target dimensions must fit the device limit');
    return value;
  };
  let width = dimension(initialWidth), height = dimension(initialHeight);
  const rendererOptions = Object.freeze({format, depthFormat, sampleCount});
  let current = null, disposed = false, terminal = null, busy = false, version = 0;
  let validation = Promise.resolve(), retirement = Promise.resolve();
  let mipPipeline = null, mipSampler = null, mipGenerations = 0;
  const levels = (w, h) => generateMipmaps && w && h ? 1 + Math.floor(Math.log2(Math.max(w, h))) : 1;
  const retired = new Set();
  let rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; });
  stopped.catch(() => {});
  const cost = (w, h) => {
    // Depth24plus is charged conservatively at four bytes/texel. Driver-private
    // allocation overhead is opaque and is not presented as measured memory.
    const base = w * h;
    let texels = base, mw = w, mh = h;
    for (let level = 1; level < levels(w, h); level++) {
      mw = Math.max(1, Math.floor(mw / 2)); mh = Math.max(1, Math.floor(mh / 2));
      texels += mw * mh;
    }
    const bytes = COLOR_BYTES.get(format) * (texels + (sampleCount === 4 ? base * 4 : 0)) +
      (depthFormat === null ? 0 : base * 4 * sampleCount);
    if (!Number.isSafeInteger(bytes)) fail('BUDGET', 'Target byte count exceeds safe integer precision');
    return bytes;
  };
  const allocated = () => (current?.bytes ?? 0) + [...retired].reduce((n, pair) => n + pair.bytes, 0);
  const destroy = pair => {
    pair?.color?.destroy(); pair?.multisample?.destroy(); pair?.depth?.destroy();
  };
  function release() {
    destroy(current); current = null;
    for (const pair of retired) destroy(pair);
    retired.clear();
    mipPipeline = mipSampler = null;
  }
  function stop(error) {
    if (disposed || terminal !== null) return;
    terminal = error; rejectStopped(error);
    if (!busy) release();
  }
  function live() {
    if (disposed) fail('DISPOSED', 'Render target is disposed');
    if (terminal !== null) throw terminal;
  }
  function exclusive() {
    live();
    if (busy) fail('REENTRANT', 'Target operations cannot be reentered');
  }
  function checked(operation) {
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let value, error;
    try { value = operation(); } catch (cause) { error = cause; }
    // Never keep a scope open over a host await on the borrowed device.
    const invalid = device.popErrorScope(), oom = device.popErrorScope();
    validation = Promise.all([validation, invalid, oom]).then(([, a, b]) => {
      if (error) throw error;
      if (a || b) throw new GpuRenderTargetError('GPU', (a || b).message ?? 'Target allocation failed');
    });
    validation.catch(stop);
    if (error) { stop(error); throw error; }
    return value;
  }
  function allocate(w, h, bytes) {
    if (w === 0 || h === 0) return null;
    const pair = {bytes, used: false, color: null, multisample: null, depth: null,
      mipLevelCount: levels(w, h), mipsDirty: true, mipSteps: null};
    try {
      checked(() => {
        const size = {width: w, height: h, depthOrArrayLayers: 1};
        pair.color = device.createTexture({label: label + ' color', size, format,
          dimension: '2d', sampleCount: 1, mipLevelCount: pair.mipLevelCount, usage: 1 | 4 | 16});
        pair.view = pair.color.createView({baseMipLevel: 0, mipLevelCount: 1});
        if (sampleCount === 4) {
          pair.multisample = device.createTexture({label: label + ' MSAA', size, format,
            dimension: '2d', sampleCount, mipLevelCount: 1, usage: 16});
          pair.multisampleView = pair.multisample.createView();
        }
        if (depthFormat !== null) {
          pair.depth = device.createTexture({label: label + ' depth', size, format: depthFormat,
            dimension: '2d', sampleCount, mipLevelCount: 1, usage: 16});
          pair.depthView = pair.depth.createView();
        }
      });
      pair.frame = Object.freeze({colorView: pair.multisampleView ?? pair.view,
        ...(pair.depthView ? {depthView: pair.depthView} : {}),
        ...(pair.multisampleView ? {resolveTarget: pair.view} : {})});
      return pair;
    } catch (error) { destroy(pair); throw error; }
  }
  function refreshMips(pair) {
    if (!pair.mipsDirty || pair.mipLevelCount === 1) return;
    checked(() => {
      if (['createShaderModule', 'createRenderPipeline', 'createSampler', 'createBindGroup', 'createCommandEncoder']
        .some(key => typeof device[key] !== 'function') || typeof device.queue.submit !== 'function')
        fail('DEVICE', 'Mip generation requires WebGPU render pipelines and queue submission');
      if (!mipPipeline) {
        const module = device.createShaderModule({label: label + ' mipmap', code: MIP_WGSL});
        mipPipeline = device.createRenderPipeline({label: label + ' mipmap', layout: 'auto',
          vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format}]},
          primitive: {topology: 'triangle-list'}});
        mipSampler = device.createSampler({minFilter: 'linear', magFilter: 'linear',
          addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
      }
      if (!pair.mipSteps) {
        const steps = [];
        for (let level = 1; level < pair.mipLevelCount; level++) {
          const source = pair.color.createView({baseMipLevel: level - 1, mipLevelCount: 1});
          const view = pair.color.createView({baseMipLevel: level, mipLevelCount: 1});
          const group = device.createBindGroup({layout: mipPipeline.getBindGroupLayout(0), entries: [
            {binding: 0, resource: source}, {binding: 1, resource: mipSampler},
          ]});
          steps.push({view, group});
        }
        pair.mipSteps = steps;
      }
      const encoder = device.createCommandEncoder({label: label + ' mip chain'});
      for (const {view, group} of pair.mipSteps) {
        const pass = encoder.beginRenderPass({colorAttachments: [{view,
          loadOp: 'clear', storeOp: 'store', clearValue: {r: 0, g: 0, b: 0, a: 0}}]});
        pass.setPipeline(mipPipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
      }
      device.queue.submit([encoder.finish()]);
    });
    pair.mipsDirty = false; mipGenerations++;
  }
  function retire(pair) {
    if (!pair) return;
    if (!pair.used) { destroy(pair); return; }
    retired.add(pair);
    let pending;
    try { pending = device.queue.onSubmittedWorkDone(); }
    catch (error) { stop(error); throw error; }
    const fence = Promise.resolve(pending).then(() => {
      if (retired.delete(pair)) destroy(pair);
    });
    retirement = Promise.all([retirement, fence]);
    retirement.catch(stop);
  }
  function borrow(callback, render) {
    exclusive();
    if (typeof callback !== 'function') fail('CALLBACK', 'Expected a synchronous submitted-use callback');
    if (!current) return false;
    busy = true;
    try {
      current.used = true; // Even a throwing callback may have submitted work.
      if (render) current.mipsDirty = true;
      else refreshMips(current);
      const result = render ? callback(current.frame, current.color) : callback(current.color);
      if (result && typeof result.then === 'function') {
        Promise.resolve(result).catch(() => {});
        fail('ASYNC_USE', 'Submit all work before returning; borrowed textures cannot cross an await');
      }
      live();
      return true;
    } finally { busy = false; if (disposed || terminal !== null) release(); }
  }
  const api = Object.freeze({
    rendererOptions,
    get width() { return width; }, get height() { return height; },
    get mipLevelCount() { return current?.mipLevelCount ?? 0; },
    get suspended() { return width === 0 || height === 0; },
    get version() { return version; }, get disposed() { return disposed; },
    get failed() { return terminal !== null; }, get allocatedBytes() { return allocated(); },
    get diagnostics() { return Object.freeze({width, height, version, format, depthFormat, sampleCount,
      allocatedBytes: allocated(), retiredTargets: retired.size, mipLevelCount: current?.mipLevelCount ?? 0, mipGenerations}); },
    resize(w, h) {
      exclusive(); dimension(w); dimension(h);
      if (w === width && h === height) return api;
      if (version === Number.MAX_SAFE_INTEGER) fail('VERSION', 'Target version exhausted');
      const bytes = cost(w, h);
      // Budget failures are retryable after existing retirement fences drain.
      if (bytes > maxBytes - allocated()) fail('BUDGET', 'Peak target allocation exceeds maxBytes');
      let replacement = null;
      busy = true;
      try {
        replacement = allocate(w, h, bytes);
        const previous = current;
        current = replacement; replacement = null;
        width = w; height = h; version++;
        retire(previous); live();
        return api;
      } catch (error) { destroy(replacement); stop(error); throw error; }
      finally { busy = false; if (terminal !== null) release(); }
    },
    withFrame(callback) { return borrow(callback, true); },
    withTexture(callback) { return borrow(callback, false); },
    async whenIdle() {
      exclusive();
      try {
        await Promise.race([Promise.all([validation, retirement, device.queue.onSubmittedWorkDone()]), stopped]);
        live(); return api;
      } catch (error) { stop(error); throw error; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during a borrowed target use');
      if (disposed) return;
      disposed = true; rejectStopped(new GpuRenderTargetError('DISPOSED', 'Render target disposed'));
      release();
    },
  });
  try {
    const bytes = cost(width, height);
    if (bytes > maxBytes) fail('BUDGET', 'Initial target exceeds maxBytes');
    current = allocate(width, height, bytes);
    if (device.lost && typeof device.lost.then === 'function')
      device.lost.then(info => stop(new GpuRenderTargetError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    return api;
  } catch (error) { disposed = true; release(); throw error; }
}
