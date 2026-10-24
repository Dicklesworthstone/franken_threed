/** Owned offscreen color/depth/MSAA attachments for immediate WebGPU rendering.
 * The resolved color texture supports sampling and COPY_SRC readback. All native
 * handles are borrowed only inside withFrame/withTexture, with consumers submitted
 * before returning. generateMipmaps:true allocates a full resolved color pyramid;
 * withTexture refreshes it on the GPU after the last withFrame write, before the
 * consumer runs. mipLevelCount can instead allocate an authored pyramid without
 * automatic generation. withFrame(callback, {mipLevel}) selects one color level;
 * nonzero levels use private matching depth storage allocated on first use.
 * Nonzero mip rendering requires one sample and private (not sampled) depth.
 * sampleDepth:true exposes the actual depth attachment through withDepthTexture.
 * withSampledTextures borrows resolved color and sampled depth together under
 * one lifetime guard. Depth-only reads do not refresh or dirty color mipmaps.
 * A four-sample depth attachment stays multisampled: no depth resolve, averaging
 * or precision conversion is inferred. Consumers choose their shader sample index.
 * colorFormats adds ordered 2D mip-zero MRT attachments under the same owner.
 * Frames expose colorViews/resolveTargets, retaining their primary aliases.
 * withAttachments borrows all resolved colors and optional sampled depth at once;
 * withTexture's colorIndex selects one color. Every attachment shares retirement.
 * readPixels/writePixels also accept colorIndex. copyFrom selects independent
 * sourceColorIndex/colorIndex attachments; all default to zero and preserve
 * the selected format without conversion, including resolved MSAA sources.
 * Combined depth/stencil formats expose both aspects in frame.depthView. Sample
 * depth with a depth-only view; depth32float-stencil8 requires its device feature.
 * R/RG/RGBA32F storage stays full precision. Its mipmaps use textureLoad-based
 * bilinear reduction without requiring optional float32 filtering. Float32
 * targets currently admit one sample only; no precision fallback is inferred.
 * readPixels() submits a bounded copy immediately and returns tightly packed
 * typed color data. mipLevel selects an allocated color level, with bounds and
 * default extents in that level. Pending reads are cancelled on disposal or failure.
 * readDepthPixels() reads sampled depth as native-space Float32 values; MSAA
 * requires an explicit sampleIndex. Depth never travels through RGBA/half floats.
 * copyDestination:true additionally admits writePixels(data, options) and
 * copyFrom(sourceTarget, options). These queue-ordered, level-zero writes dirty
 * the mip chain; copies can read a selected source mip or resolved MSAA color.
 * Upload/copy destinations must be single-sample. No implicit color conversion.
 * copyDepthFrom(sourceTarget, options) writes only depth, preserving color and
 * stencil. Equal MSAA counts preserve every sample; MSAA-to-single requires an
 * explicit sampleIndex. Different depth formats require allowFormatConversion.
 * Enable sampleDepth on the source and copyDestination on the destination.
 * Submission is synchronous; whenIdle includes validation and the copy fence.
 * No framebuffer emulation or frame loop.
 * cube:true admits six square single-sample mip-zero attachments. withFrame's
 * face selects one 2D attachment; depth borrowers retain the entire native cube.
 * readPixels({face}) and readDepthPixels({face}) select one explicit cube face,
 * retaining bounded staging, call-time submission and cancellation on disposal.
 * writePixels(data, {face}) and copyFrom(source, {sourceFace, face}) transfer
 * one explicit cube color face, including cube/2D copies. copyDepthFrom uses
 * the same face selection for depth-only copies. Cube mip generation remains
 * unsupported; no implicit multisample depth resolve is introduced.
 */
import {readGpuTargetPixels, readGpuTargetDepth} from './gpu_target_readback.mjs';
import {inspectGpuTargetWrite, gpuTargetWriteData, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {inspectGpuDepthCopy, submitGpuDepthCopy} from './gpu_depth_transfer.mjs';
// Authenticate target handles and device identity without exposing GPU textures.
const transferTargets = new WeakMap();

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
  ['r32float', 4], ['rg32float', 8], ['rgba32float', 16],
]);

/** Render attachment limits use pixel byte cost, not texture storage bytes. */
export function inspectGpuRenderTargetColors(device, formats) {
  if (!Array.isArray(formats) || !formats.length || formats.some(format => !COLOR_BYTES.has(format)))
    fail('FORMAT', 'Expected supported color attachment formats');
  if (formats.length > 1) {
    const count = device?.limits?.maxColorAttachments, limit = device?.limits?.maxColorAttachmentBytesPerSample;
    if (!Number.isSafeInteger(count) || !Number.isSafeInteger(limit))
      fail('DEVICE', 'MRT requires the device color attachment limits');
    // WebGPU assigns normalized RGBA8/BGRA8 an eight-byte render cost despite
    // their four-byte storage footprint. Component alignment is 1, 2 or 4.
    const bytes = formats.reduce((total, format) => {
      const alignment = format.includes('32float') ? 4 : format === 'rgba16float' ? 2 : 1;
      const cost = format.includes('8unorm') ? 8 : COLOR_BYTES.get(format);
      return Math.ceil(total / alignment) * alignment + cost;
    }, 0);
    if (formats.length > count || bytes > limit)
      fail('LIMIT', 'Color attachments exceed device count or bytes-per-sample limits');
  }
  return Object.freeze(formats.slice());
}

// Combined formats may use eight bytes per sample on the selected backend.
// Charge the conservative format footprint, not measured driver overhead.
const DEPTH_BYTES = new Map([[null, 0], ['depth16unorm', 2], ['depth24plus', 4], ['depth32float', 4],
  ['depth24plus-stencil8', 8], ['depth32float-stencil8', 8]]);

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

// Explicit non-filtering layout plus four textureLoad operations avoids silently
// depending on float32-filterable. Match the existing bilinear pixel-center
// convention, including odd extents and 1D tails, at the original f32 precision.
const FLOAT32_MIP_WGSL = `
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let size = textureDimensions(source);
  let nextSize = max(size / 2u, vec2u(1u));
  let q = p.xy * vec2f(size) / vec2f(nextSize) - vec2f(0.5);
  let lo = vec2i(floor(q));
  let hi = vec2i(size) - vec2i(1);
  let a = textureLoad(source, clamp(lo, vec2i(0), hi), 0);
  let b = textureLoad(source, clamp(lo + vec2i(1, 0), vec2i(0), hi), 0);
  let c = textureLoad(source, clamp(lo + vec2i(0, 1), vec2i(0), hi), 0);
  let d = textureLoad(source, clamp(lo + vec2i(1, 1), vec2i(0), hi), 0);
  let weight = fract(q);
  return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}`;

export function createGpuRenderTarget(device, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    fail('OPTIONS', 'Expected render target options');
  for (const key of Object.keys(options))
    if (!['width', 'height', 'format', 'colorFormats', 'depthFormat', 'sampleCount', 'maxBytes', 'label', 'generateMipmaps', 'mipLevelCount', 'sampleDepth', 'copyDestination', 'cube'].includes(key))
      fail('OPTIONS', `Unknown target option: ${key}`);
  const {
    width: initialWidth = 1, height: initialHeight = 1,
    format = options.colorFormats?.[0] ?? 'rgba16float', depthFormat = 'depth24plus', sampleCount = 1,
    maxBytes = 256 * 1024 * 1024, label = 'Offscreen render target', generateMipmaps = false,
    mipLevelCount: requestedMipLevels, sampleDepth = false, copyDestination = false, cube = false,
  } = options;
  const colorFormats = inspectGpuRenderTargetColors(device, options.colorFormats ?? [format]), multiple = colorFormats.length > 1;
  if (colorFormats[0] !== format) fail('FORMAT', 'The primary format must match colorFormats[0]');
  if (!COLOR_BYTES.has(format) || !DEPTH_BYTES.has(depthFormat))
    fail('FORMAT', 'Unsupported color or depth format');
  if (![1, 4].includes(sampleCount)) fail('SAMPLES', 'Expected one or four samples');
  const float32 = ['r32float', 'rg32float', 'rgba32float'].includes(format);
  if (colorFormats.some(value => value.includes('32float')) && sampleCount !== 1)
    fail('SAMPLES', 'Float32 color targets currently require one sample');
  if (typeof copyDestination !== 'boolean') fail('OPTIONS', 'copyDestination must be boolean');
  if (typeof sampleDepth !== 'boolean') fail('OPTIONS', 'sampleDepth must be boolean');
  if (sampleDepth && depthFormat === null) fail('DEPTH', 'Sampled depth requires a depth attachment');
  if (typeof generateMipmaps !== 'boolean') fail('OPTIONS', 'generateMipmaps must be boolean');
  if (typeof cube !== 'boolean') fail('OPTIONS', 'cube must be boolean');
  if (multiple && (cube || generateMipmaps || requestedMipLevels !== undefined && requestedMipLevels !== 1))
    fail('UNSUPPORTED', 'Multiple color attachments currently require 2D mip-zero targets');
  if (cube && (sampleCount !== 1 || generateMipmaps || requestedMipLevels !== undefined && requestedMipLevels !== 1))
    fail('UNSUPPORTED', 'Cube targets require one sample and mip zero');
  if (requestedMipLevels !== undefined && (!Number.isSafeInteger(requestedMipLevels) || requestedMipLevels < 1))
    fail('OPTIONS', 'mipLevelCount must be a positive integer');
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || typeof label !== 'string')
    fail('OPTIONS', 'Expected a positive byte budget and string label');
  if (!device || ['createTexture', 'pushErrorScope', 'popErrorScope'].some(k => typeof device[k] !== 'function') ||
      typeof device.queue?.onSubmittedWorkDone !== 'function') fail('DEVICE', 'Expected a WebGPU device');
  if (depthFormat === 'depth32float-stencil8' && device.features?.has?.('depth32float-stencil8') !== true)
    fail('FEATURE', 'depth32float-stencil8 must be enabled on the supplied device');
  const limit = device.limits?.maxTextureDimension2D;
  if (!Number.isSafeInteger(limit) || limit < 1) fail('DEVICE', 'Missing texture dimension limit');
  if (cube && (!Number.isSafeInteger(device.limits?.maxTextureArrayLayers) || device.limits.maxTextureArrayLayers < 6))
    fail('DEVICE', 'Cube targets require capacity for six texture array layers');
  const dimension = value => {
    if (!Number.isSafeInteger(value) || value < 0 || value > limit)
      fail('SIZE', 'Target dimensions must fit the device limit');
    return value;
  };
  let width = dimension(initialWidth), height = dimension(initialHeight);
  if (cube && width !== height) fail('SIZE', 'Cube faces must be square');
  const layers = cube ? 6 : 1;
  const rendererOptions = Object.freeze({format, depthFormat, sampleCount, ...(multiple ? {colorFormats} : {})});
  let current = null, disposed = false, terminal = null, busy = false, version = 0;
  let validation = Promise.resolve(), retirement = Promise.resolve();
  let mipPipeline = null, mipSampler = null, mipGenerations = 0, colorUploads = 0, colorCopies = 0, depthCopies = 0;
  const levels = (w, h) => {
    if (!w || !h) return 1;
    const maximum = 1 + Math.floor(Math.log2(Math.max(w, h)));
    if (requestedMipLevels > maximum) fail('SIZE', 'mipLevelCount exceeds the target extent');
    return requestedMipLevels ?? (generateMipmaps ? maximum : 1);
  };
  const transferShape = () => ({width, height, format, colorFormats, sampleCount, mipLevelCount: levels(width, height), cube});
  const depthShape = () => ({width, height, depthFormat, sampleCount, sampleDepth, cube});
  const retired = new Set(), readbacks = new Set();
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
    const bytes = layers * (colorFormats.reduce((sum, value) => sum + COLOR_BYTES.get(value), 0) * (texels + (sampleCount === 4 ? base * 4 : 0)) +
      base * DEPTH_BYTES.get(depthFormat) * sampleCount);
    if (!Number.isSafeInteger(bytes)) fail('BUDGET', 'Target byte count exceeds safe integer precision');
    return bytes;
  };
  const allocated = () => (current?.bytes ?? 0) + [...retired].reduce((n, pair) => n + pair.bytes, 0);
  const destroy = pair => {
    for (const texture of pair?.colors ?? []) texture.destroy();
    for (const texture of pair?.multisamples ?? []) texture.destroy();
    pair?.depth?.destroy();
    for (const depth of pair?.mipDepths?.values() ?? []) depth.destroy();
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
    for (const controller of readbacks) controller.abort(error);
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
    const pair = {bytes, used: false, colors: [], multisamples: [], color: null, multisample: null, depth: null, mipDepths: new Map(), frames: new Map(),
      mipLevelCount: levels(w, h), mipsDirty: true, mipSteps: null};
    try {
      checked(() => {
        const size = {width: w, height: h, depthOrArrayLayers: layers};
        const faceZero = cube ? {dimension: '2d', baseArrayLayer: 0, arrayLayerCount: 1} : {};
        pair.views = []; pair.multisampleViews = [];
        for (const [index, attachmentFormat] of colorFormats.entries()) {
          const color = device.createTexture({label: `${label} color ${index}`, size, format: attachmentFormat,
            dimension: '2d', sampleCount: 1, mipLevelCount: pair.mipLevelCount, usage: 1 | 4 | 16 | (copyDestination ? 2 : 0)});
          pair.colors.push(color);
          pair.views.push(color.createView({baseMipLevel: 0, mipLevelCount: 1, ...faceZero}));
          if (sampleCount === 4) {
            const multisample = device.createTexture({label: `${label} MSAA ${index}`, size, format: attachmentFormat,
              dimension: '2d', sampleCount, mipLevelCount: 1, usage: 16});
            pair.multisamples.push(multisample); pair.multisampleViews.push(multisample.createView());
          }
        }
        pair.color = pair.colors[0]; pair.view = pair.views[0];
        pair.multisample = pair.multisamples[0]; pair.multisampleView = pair.multisampleViews[0];
        if (depthFormat !== null) {
          pair.depth = device.createTexture({label: label + ' depth', size, format: depthFormat,
            dimension: '2d', sampleCount, mipLevelCount: 1, usage: 16 | (sampleDepth ? 4 : 0)});
          pair.depthView = pair.depth.createView(faceZero);
        }
      });
      pair.frame = Object.freeze({colorView: pair.multisampleView ?? pair.view,
        ...(multiple ? {colorViews: Object.freeze((sampleCount === 4 ? pair.multisampleViews : pair.views).slice()),
          ...(sampleCount === 4 ? {resolveTargets: Object.freeze(pair.views.slice())} : {})} : {}),
        ...(pair.depthView ? {depthView: pair.depthView} : {}),
        ...(pair.multisampleView ? {resolveTarget: pair.view} : {})});
      pair.frames.set(0, pair.frame);
      if (cube) pair.cubeFrames = checked(() => Array.from({length: 6}, (_, face) => face === 0 ? pair.frame : Object.freeze({
        colorView: pair.color.createView({dimension: '2d', baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: face, arrayLayerCount: 1}),
        ...(pair.depth ? {depthView: pair.depth.createView({dimension: '2d', baseMipLevel: 0, mipLevelCount: 1,
          baseArrayLayer: face, arrayLayerCount: 1})} : {}),
      })));
      return pair;
    } catch (error) { destroy(pair); throw error; }
  }
  function refreshMips(pair) {
    if (!generateMipmaps || !pair.mipsDirty || pair.mipLevelCount === 1) return;
    checked(() => {
      const methods = ['createShaderModule', 'createRenderPipeline', 'createBindGroup', 'createCommandEncoder',
        ...(float32 ? ['createBindGroupLayout', 'createPipelineLayout'] : ['createSampler'])];
      if (methods.some(key => typeof device[key] !== 'function') || typeof device.queue.submit !== 'function')
        fail('DEVICE', 'Mip generation requires WebGPU render pipelines and queue submission');
      if (!mipPipeline) {
        const module = device.createShaderModule({label: label + ' mipmap', code: float32 ? FLOAT32_MIP_WGSL : MIP_WGSL});
        const layout = float32 ? device.createPipelineLayout({bindGroupLayouts: [device.createBindGroupLayout({entries: [
          {binding: 0, visibility: 2, texture: {sampleType: 'unfilterable-float', viewDimension: '2d', multisampled: false}},
        ]})]}) : 'auto';
        mipPipeline = device.createRenderPipeline({label: label + ' mipmap', layout,
          vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format}]},
          primitive: {topology: 'triangle-list'}});
        if (!float32) mipSampler = device.createSampler({minFilter: 'linear', magFilter: 'linear',
          addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
      }
      if (!pair.mipSteps) {
        const steps = [];
        for (let level = 1; level < pair.mipLevelCount; level++) {
          const source = pair.color.createView({baseMipLevel: level - 1, mipLevelCount: 1});
          const view = pair.color.createView({baseMipLevel: level, mipLevelCount: 1});
          const group = device.createBindGroup({layout: mipPipeline.getBindGroupLayout(0), entries: [
            {binding: 0, resource: source}, ...(float32 ? [] : [{binding: 1, resource: mipSampler}]),
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
  function frameAllocationBytes(mipLevel = 0, face = 0) {
    exclusive();
    if (!Number.isSafeInteger(face) || face < 0 || face >= layers)
      fail('FACE', cube ? 'Cube face must be an integer from zero to five' : '2D targets require face zero');
    if (!Number.isSafeInteger(mipLevel) || mipLevel < 0 || mipLevel >= (current?.mipLevelCount ?? 1))
      fail('MIP', 'Render mip level is not allocated');
    if (mipLevel > 0 && (sampleCount !== 1 || sampleDepth))
      fail('MIP', 'Nonzero mip rendering requires single-sample color and private depth');
    if (!current || current.frames.has(mipLevel) || depthFormat === null) return 0;
    return Math.max(1, Math.floor(width / 2 ** mipLevel)) * Math.max(1, Math.floor(height / 2 ** mipLevel)) * DEPTH_BYTES.get(depthFormat);
  }
  function frameAt(mipLevel, bytes, face) {
    if (cube) return current.cubeFrames[face];
    if (current.frames.has(mipLevel)) return current.frames.get(mipLevel);
    if (bytes > maxBytes - allocated()) fail('BUDGET', 'Mip depth allocation exceeds maxBytes');
    let depth = null;
    try {
      const frame = checked(() => {
        const colorView = current.color.createView({baseMipLevel: mipLevel, mipLevelCount: 1});
        if (depthFormat !== null) {
          depth = device.createTexture({label: `${label} mip ${mipLevel} depth`,
            size: {width: Math.max(1, Math.floor(width / 2 ** mipLevel)), height: Math.max(1, Math.floor(height / 2 ** mipLevel)), depthOrArrayLayers: 1},
            format: depthFormat, dimension: '2d', sampleCount: 1, mipLevelCount: 1, usage: 16});
        }
        return Object.freeze({colorView, ...(depth ? {depthView: depth.createView()} : {})});
      });
      if (depth) current.mipDepths.set(mipLevel, depth);
      current.frames.set(mipLevel, frame); current.bytes += bytes;
      return frame;
    } catch (error) { depth?.destroy(); throw error; }
  }
  function borrow(callback, kind, mipLevel = 0, face = 0, colorIndex = 0) {
    exclusive();
    if (!Number.isSafeInteger(colorIndex) || colorIndex < 0 || colorIndex >= colorFormats.length)
      fail('ATTACHMENT', 'Color attachment index is out of range');
    if ((kind === 'depth' || kind === 'sampled') && !sampleDepth) fail('DEPTH', 'Enable sampleDepth before borrowing the depth texture');
    if (typeof callback !== 'function') fail('CALLBACK', 'Expected a synchronous submitted-use callback');
    const extraBytes = kind === 'frame' ? frameAllocationBytes(mipLevel, face) : 0;
    if (!current) return false;
    // Budget rejection is retryable and happens before marking this owner busy.
    if (extraBytes > maxBytes - allocated()) fail('BUDGET', 'Mip depth allocation exceeds maxBytes');
    busy = true;
    try {
      const frame = kind === 'frame' ? frameAt(mipLevel, extraBytes, face) : null;
      current.used = true; // Even a throwing callback may have submitted work.
      if (kind === 'frame') current.mipsDirty = true;
      else if (kind === 'color' || kind === 'sampled' || kind === 'attachments') refreshMips(current);
      const result = kind === 'frame' ? callback(frame, current.color)
        : kind === 'sampled' ? callback(current.color, current.depth)
        : kind === 'attachments' ? callback(Object.freeze(current.colors.slice()), sampleDepth ? current.depth : null)
        : callback(kind === 'depth' ? current.depth : current.colors[colorIndex]);
      if (result && typeof result.then === 'function') {
        Promise.resolve(result).catch(() => {});
        fail('ASYNC_USE', 'Submit all work before returning; borrowed textures cannot cross an await');
      }
      live();
      return true;
    } finally { busy = false; if (disposed || terminal !== null) release(); }
  }
  async function read(operation, options = {}) {
    exclusive();
    if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected readback options');
    const caller = options.signal;
    if (caller !== undefined && (!caller || typeof caller.addEventListener !== 'function' || typeof caller.removeEventListener !== 'function'))
      fail('OPTIONS', 'Expected AbortSignal');
    const controller = new AbortController(), abort = () => controller.abort(caller.reason);
    readbacks.add(controller);
    try {
      caller?.addEventListener('abort', abort, {once: true});
      if (caller?.aborted) abort();
      // Capture the extent and submit before any await: resize and subsequent
      // draws cannot change which version of the texture this call observes.
      return await operation({device, width, height, diagnostics: {target: {...transferShape(), depthFormat, sampleDepth}},
        get disposed() { return disposed; }, get failed() { return terminal !== null; },
        withTexture: api.withTexture, withDepthTexture: api.withDepthTexture}, {...options, signal: controller.signal});
    } finally {
      readbacks.delete(controller);
      caller?.removeEventListener('abort', abort);
    }
  }
  const api = Object.freeze({
    rendererOptions,
    get width() { return width; }, get height() { return height; },
    get mipLevelCount() { return current?.mipLevelCount ?? 0; },
    get suspended() { return width === 0 || height === 0; },
    get version() { return version; }, get disposed() { return disposed; },
    get failed() { return terminal !== null; }, get allocatedBytes() { return allocated(); },
    get diagnostics() { return Object.freeze({width, height, version, format, ...(multiple ? {colorFormats} : {}), depthFormat, sampleCount, sampleDepth, copyDestination, cube, layers, colorUploads, colorCopies, depthCopies,
      allocatedBytes: allocated(), retiredTargets: retired.size, mipLevelCount: current?.mipLevelCount ?? 0, mipGenerations, pendingReadbacks: readbacks.size}); },
    resize(w, h) {
      exclusive(); dimension(w); dimension(h);
      if (cube && w !== h) fail('SIZE', 'Cube faces must be square');
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
    frameAllocationBytes,
    withFrame(callback, {mipLevel = 0, face = 0} = {}) { return borrow(callback, 'frame', mipLevel, face); },
    withTexture(callback, {colorIndex = 0} = {}) { return borrow(callback, 'color', 0, 0, colorIndex); },
    withAttachments(callback) { return borrow(callback, 'attachments'); },
    withDepthTexture(callback) { return borrow(callback, 'depth'); },
    withSampledTextures(callback) { return borrow(callback, 'sampled'); },
    writePixels(data, options = {}) {
      exclusive();
      if (!copyDestination) fail('USAGE', 'Enable copyDestination before uploading target color');
      const plan = inspectGpuTargetWrite(transferShape(), data, options);
      if (plan.empty) return false;
      if (typeof device.queue.writeTexture !== 'function') fail('DEVICE', 'Color uploads require GPUQueue.writeTexture');
      const upload = gpuTargetWriteData(plan);
      busy = true;
      try {
        current.used = true; current.mipsDirty = true;
        checked(() => device.queue.writeTexture({texture: current.colors[plan.colorIndex], mipLevel: 0, origin: plan.origin},
          upload.data, upload.layout, plan.size));
        live(); colorUploads++; return true;
      } finally { busy = false; if (disposed || terminal !== null) release(); }
    },
    copyFrom(source, options = {}) {
      exclusive();
      if (!copyDestination) fail('USAGE', 'Enable copyDestination before copying target color');
      const other = transferTargets.get(source);
      if (!other) fail('SOURCE', 'Expected a native render target handle');
      if (other.device !== device) fail('DEVICE', 'Target copies require the same GPUDevice');
      if (source === api) fail('FEEDBACK', 'In-place target copies are not admitted');
      other.exclusive();
      const plan = inspectGpuTargetCopy(other.shape(), transferShape(), options);
      if (plan.empty) return false;
      busy = true;
      try {
        current.used = true; current.mipsDirty = true;
        other.borrow(texture => checked(() => {
          const encoder = device.createCommandEncoder({label: label + ' color copy'});
          encoder.copyTextureToTexture({texture, mipLevel: plan.sourceMipLevel, origin: plan.sourceOrigin},
            {texture: current.colors[plan.colorIndex], mipLevel: 0, origin: plan.destinationOrigin}, plan.size);
          device.queue.submit([encoder.finish()]);
        }), plan.sourceColorIndex);
        // A source mip-generation/upload error must also fail the dependent
        // copy, even if its own copy error scope happened to validate first.
        validation = Promise.all([validation, other.validation()]).then(() => {});
        validation.catch(stop);
        live(); colorCopies++; return true;
      } finally { busy = false; if (disposed || terminal !== null) release(); }
    },
    copyDepthFrom(source, options = {}) {
      exclusive();
      if (!copyDestination) fail('USAGE', 'Enable copyDestination before copying target depth');
      const other = transferTargets.get(source);
      if (!other) fail('SOURCE', 'Expected a native render target handle');
      if (other.device !== device) fail('DEVICE', 'Depth copies require the same GPUDevice');
      if (source === api) fail('FEEDBACK', 'In-place depth copies are not admitted');
      other.exclusive();
      const plan = inspectGpuDepthCopy(other.depthShape(), depthShape(), options);
      if (plan.empty) return false;
      busy = true;
      try {
        current.used = true;
        let completion;
        other.borrowDepth(texture => {
          const view = cube ? current.cubeFrames[plan.face].depthView : current.depthView;
          completion = submitGpuDepthCopy(device, texture, view, plan);
        });
        // Depth-only writes must not regenerate color mips. Source validation
        // remains a dependency even if this pass itself validates successfully.
        validation = Promise.all([validation, other.validation(), completion]).then(() => {});
        validation.catch(stop);
        live(); depthCopies++; return true;
      } finally { busy = false; if (disposed || terminal !== null) release(); }
    },
    readPixels(options = {}) { return read(readGpuTargetPixels, options); },
    readDepthPixels(options = {}) { return read(readGpuTargetDepth, options); },
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
      disposed = true;
      const error = new GpuRenderTargetError('DISPOSED', 'Render target disposed');
      rejectStopped(error);
      for (const controller of readbacks) controller.abort(error);
      release();
    },
  });
  try {
    const bytes = cost(width, height);
    if (bytes > maxBytes) fail('BUDGET', 'Initial target exceeds maxBytes');
    current = allocate(width, height, bytes);
    if (device.lost && typeof device.lost.then === 'function')
      device.lost.then(info => stop(new GpuRenderTargetError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    transferTargets.set(api, {device, exclusive, shape: transferShape, depthShape,
      borrowDepth: callback => borrow(callback, 'depth'),
      borrow: (callback, colorIndex = 0) => borrow(callback, 'color', 0, 0, colorIndex), validation: () => validation});
    return api;
  } catch (error) { disposed = true; release(); throw error; }
}
