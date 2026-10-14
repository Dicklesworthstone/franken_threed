/** Depth-only GPU transfers. No color attachment, CPU readback or implicit resolve.
 * Owners must keep both attachments borrowed until submitGpuDepthCopy returns.
 * Submission is synchronous; its promise covers validation and GPU completion.
 * The only transient allocation is a bounded 32-byte uniform per pending copy.
 */
export class GpuDepthTransferError extends Error {
  constructor(code, message) {
    super(`GPU_DEPTH_TRANSFER_${code}: ${message}`);
    this.name = 'GpuDepthTransferError'; this.code = `GPU_DEPTH_TRANSFER_${code}`;
  }
}
const fail = (code, message) => { throw new GpuDepthTransferError(code, message); };
const FORMATS = new Map([
  ['depth16unorm', 2], ['depth24plus', 4], ['depth32float', 4],
  ['depth24plus-stencil8', 8], ['depth32float-stencil8', 8],
]);
const integer = (value, label, min = 0, max = Number.MAX_SAFE_INTEGER) => {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail('OPTIONS', `Invalid ${label}`);
  return value;
};
const states = new WeakMap(), admittedPlans = new WeakSet(), UNIFORM_BYTES = 32;

/** Allocation-free admission. Coordinates address the selected depth attachment,
 * not color mip levels. flipY changes both origins, not the row order of a copy.
 * Different depth formats require explicit allowFormatConversion:true because
 * destination quantization is otherwise an observable precision change.
 */
export function inspectGpuDepthCopy(source, destination, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(options))) fail('OPTIONS', 'Expected plain depth copy options');
  const allowed = ['sourceX', 'sourceY', 'x', 'y', 'width', 'height', 'flipY', 'sampleIndex',
    'allowFormatConversion', 'maxBytes', 'maxInFlightBytes'];
  for (const key of Reflect.ownKeys(options))
    if (!allowed.includes(key) || !Object.hasOwn(Object.getOwnPropertyDescriptor(options, key), 'value'))
      fail('OPTIONS', `Unknown or accessor-backed depth copy option: ${String(key)}`);
  for (const shape of [source, destination]) {
    if (!FORMATS.has(shape?.depthFormat)) fail('FORMAT', 'Expected an admitted depth attachment');
    integer(shape.width, 'depth width', 0, 0x7fffffff); integer(shape.height, 'depth height', 0, 0x7fffffff);
    if (![1, 4].includes(shape.sampleCount)) fail('SAMPLES', 'Depth copies require one or four samples');
  }
  if (source.sampleDepth !== true) fail('USAGE', 'Enable sampleDepth on the source target');
  const {sourceX = 0, sourceY = 0, x = 0, y = 0, width = source.width - sourceX,
    height = source.height - sourceY, flipY = false, sampleIndex,
    allowFormatConversion = false, maxBytes = 64 * 1024 * 1024, maxInFlightBytes = 64 * 1024} = options;
  if (typeof flipY !== 'boolean' || typeof allowFormatConversion !== 'boolean') fail('OPTIONS', 'Expected boolean depth copy flags');
  if (source.depthFormat !== destination.depthFormat && !allowFormatConversion)
    fail('FORMAT', 'Different depth formats require explicit allowFormatConversion');
  for (const [label, value] of Object.entries({sourceX, sourceY, x, y, width, height})) integer(value, label);
  if (sourceX > source.width - width || sourceY > source.height - height ||
      x > destination.width - width || y > destination.height - height) fail('REGION', 'Depth copy region exceeds an attachment');
  integer(maxBytes, 'depth copy byte budget', 1); integer(maxInFlightBytes, 'in-flight depth copy budget', 1);
  let mode;
  if (source.sampleCount === 1 && destination.sampleCount === 1) {
    if (sampleIndex !== undefined && sampleIndex !== 0) fail('SAMPLES', 'Single-sample depth requires sample zero');
    mode = 'single';
  } else if (source.sampleCount === 4 && destination.sampleCount === 4) {
    if (sampleIndex !== undefined) fail('SAMPLES', 'Equal-MSAA copies preserve all samples; do not select one');
    mode = 'per-sample';
  } else if (source.sampleCount === 4 && destination.sampleCount === 1) {
    if (!Number.isSafeInteger(sampleIndex) || sampleIndex < 0 || sampleIndex >= 4)
      fail('SAMPLES', 'MSAA to single-sample depth requires an explicit sampleIndex from zero to three');
    mode = 'selected';
  } else fail('SAMPLES', 'Single-sample depth is not implicitly replicated into MSAA storage');
  const bytes = width * height * Math.max(FORMATS.get(source.depthFormat), FORMATS.get(destination.depthFormat)) * source.sampleCount;
  if (!Number.isSafeInteger(bytes) || bytes > maxBytes) fail('BUDGET', 'Depth copy exceeds its byte budget');
  const plan = Object.freeze({
    empty: width === 0 || height === 0, width, height,
    sourceX, sourceY: flipY ? source.height - sourceY - height : sourceY,
    x, y: flipY ? destination.height - y - height : y,
    sourceWidth: source.width, sourceHeight: source.height,
    destinationWidth: destination.width, destinationHeight: destination.height,
    sourceFormat: source.depthFormat, format: destination.depthFormat,
    sourceSampleCount: source.sampleCount, sampleCount: destination.sampleCount,
    mode, sampleIndex: sampleIndex ?? 0, maxInFlightBytes,
  });
  admittedPlans.add(plan);
  return plan;
}

function shader(mode) {
  const multi = mode !== 'single';
  return `
struct Parameters { origins: vec4u, control: vec4u }
@group(0) @binding(0) var source: ${multi ? 'texture_depth_multisampled_2d' : 'texture_depth_2d'};
@group(0) @binding(1) var<uniform> parameters: Parameters;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f${mode === 'per-sample' ? ', @builtin(sample_index) sample: u32' : ''}) -> @builtin(frag_depth) f32 {
  let pixel = vec2i(p.xy) - vec2i(parameters.origins.zw) + vec2i(parameters.origins.xy);
  return textureLoad(source, pixel, ${mode === 'per-sample' ? 'i32(sample)' : multi ? 'i32(parameters.control.x)' : '0'});
}`;
}
function stateFor(device) {
  let state = states.get(device);
  if (state) return state;
  state = {pipelines: new Map(), bytes: 0, jobs: new Set(), lost: null};
  states.set(device, state);
  const lost = cause => {
    state.lost = cause instanceof Error ? cause : new GpuDepthTransferError('DEVICE_LOST', cause?.message ?? 'Device lost');
    state.pipelines.clear();
    for (const job of [...state.jobs]) job.finish(state.lost, true);
  };
  if (device.lost && typeof device.lost.then === 'function') device.lost.then(lost, lost);
  return state;
}

/** Called only inside the owners' synchronous borrows. Cancellation after this
 * function returns cannot undo the submitted write, nor release its uniform
 * before the submitted-work fence. Pipelines are shared per device and profile;
 * an invalid pipeline is evicted rather than poisoning subsequent owners.
 */
export function submitGpuDepthCopy(device, texture, depthView, plan) {
  if (!admittedPlans.has(plan)) fail('PLAN', 'Expected an immutable inspected depth copy plan');
  if (!device || ['createShaderModule', 'createRenderPipeline', 'createBindGroup', 'createBuffer',
    'createCommandEncoder', 'pushErrorScope', 'popErrorScope'].some(key => typeof device[key] !== 'function') ||
      ['writeBuffer', 'submit', 'onSubmittedWorkDone'].some(key => typeof device.queue?.[key] !== 'function'))
    fail('DEVICE', 'Depth copies require a WebGPU device with rendering and completion fences');
  if (plan.empty) return Promise.resolve(false);
  if (!texture || typeof texture.createView !== 'function' || !depthView) fail('SOURCE', 'Expected borrowed source depth and destination view');
  if (texture.format !== plan.sourceFormat || texture.sampleCount !== plan.sourceSampleCount ||
      texture.width !== plan.sourceWidth || texture.height !== plan.sourceHeight ||
      texture.dimension !== '2d' || texture.depthOrArrayLayers !== 1 || (texture.usage & 4) === 0)
    fail('SOURCE', 'Source depth texture does not match its captured profile or sampling usage');
  if (['maxBufferSize', 'maxUniformBufferBindingSize'].some(name =>
      device.limits?.[name] !== undefined && device.limits[name] < UNIFORM_BYTES))
    fail('BUDGET', 'Depth copy uniform exceeds a device buffer limit');
  const state = stateFor(device);
  if (state.lost) throw state.lost;
  let ceiling = plan.maxInFlightBytes;
  for (const pending of state.jobs) ceiling = Math.min(ceiling, pending.limit);
  if (state.jobs.size >= 64 || UNIFORM_BYTES > ceiling - state.bytes) fail('BUDGET', 'Pending depth copies exceed their uniform budget');
  const key = `${plan.format}/${plan.sampleCount}/${plan.mode}`;
  let buffer = null, cached = null, scopeCount = 0, failure, failed = false, submitted = false, fence;
  const remember = error => { if (!failed) { failed = true; failure = error; } };
  state.bytes += UNIFORM_BYTES;
  let resolve, reject;
  const completion = new Promise((yes, no) => { resolve = yes; reject = no; });
  completion.catch(() => {});
  const job = {limit: plan.maxInFlightBytes, finish(error, rejected = false) {
    if (!state.jobs.delete(job)) return;
    state.bytes -= UNIFORM_BYTES;
    try { buffer?.destroy(); }
    catch (cause) { if (!rejected) { error = cause; rejected = true; } }
    finally { buffer = null; }
    if (rejected) reject(error); else resolve(true);
  }};
  state.jobs.add(job);
  const scopes = [];
  try {
    for (const filter of ['out-of-memory', 'validation']) { device.pushErrorScope(filter); scopeCount++; }
    cached = state.pipelines.get(key);
    if (!cached) {
      const module = device.createShaderModule({label: 'Depth-only target copy', code: shader(plan.mode)});
      const pipeline = device.createRenderPipeline({label: 'Depth-only target copy', layout: 'auto',
        vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: []},
        primitive: {topology: 'triangle-list'}, multisample: {count: plan.sampleCount},
        depthStencil: {format: plan.format, depthWriteEnabled: true, depthCompare: 'always',
          ...(plan.format.endsWith('-stencil8') ? {stencilReadMask: 0, stencilWriteMask: 0} : {})}});
      cached = {pipeline, validation: Promise.resolve()};
      state.pipelines.set(key, cached);
    }
    buffer = device.createBuffer({label: 'Depth copy region', size: UNIFORM_BYTES, usage: 64 | 8});
    device.queue.writeBuffer(buffer, 0, new Uint32Array([
      plan.sourceX, plan.sourceY, plan.x, plan.y, plan.sampleIndex, 0, 0, 0,
    ]));
    const group = device.createBindGroup({layout: cached.pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: texture.createView({aspect: 'depth-only', baseMipLevel: 0, mipLevelCount: 1})},
      {binding: 1, resource: {buffer}},
    ]});
    const encoder = device.createCommandEncoder({label: 'Depth-only target copy'});
    const pass = encoder.beginRenderPass({colorAttachments: [], depthStencilAttachment: {
      view: depthView, depthLoadOp: 'load', depthStoreOp: 'store', depthReadOnly: false,
      ...(plan.format.endsWith('-stencil8') ? {stencilReadOnly: true} : {}),
    }});
    pass.setPipeline(cached.pipeline); pass.setBindGroup(0, group);
    pass.setScissorRect(plan.x, plan.y, plan.width, plan.height);
    pass.draw(3); pass.end();
    // Treat even a throwing submit as possibly submitted work.
    submitted = true;
    device.queue.submit([encoder.finish()]);
  } catch (error) { remember(error); }
  finally {
    while (scopeCount > 0) {
      scopeCount--;
      try { scopes.push(Promise.resolve(device.popErrorScope())); }
      catch (error) { remember(error); }
    }
    if (submitted) {
      try { fence = Promise.resolve(device.queue.onSubmittedWorkDone()); }
      catch (error) { remember(error); }
    }
  }
  const checked = Promise.all(scopes).then(errors => {
    if (failed) throw failure;
    const error = errors.find(Boolean);
    if (error) throw new GpuDepthTransferError('GPU', error.message ?? 'Depth copy validation failed');
  });
  const validation = Promise.all([cached?.validation, checked]).then(() => {});
  if (cached) {
    cached.validation = validation;
    validation.catch(() => { if (state.pipelines.get(key) === cached) state.pipelines.delete(key); });
  }
  // allSettled is deliberate: validation may fail while native work still uses
  // the buffer. Releasing on Promise.all's first rejection would race that work.
  Promise.allSettled([validation, fence]).then(results => {
    const error = results.find(result => result.status === 'rejected');
    job.finish(error ? error.reason : state.lost, !!error || state.lost !== null);
  });
  return completion;
}
