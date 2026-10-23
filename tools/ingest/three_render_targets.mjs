/** Source RenderTarget residency shared by all scenes of one renderer.
 * Native attachments stay top-left for rendering/readback. A lazily allocated
 * GPU-only sampling image preserves the selected source surface's texture UVs:
 * native WebGPU keeps rows unchanged; WebGL flips vertically. It is refreshed
 * once per written version, never downloaded or re-uploaded.
 * generateMipmaps:true keeps a full native color pyramid and copies each level
 * with the selected orientation into the sampling pyramid. Both chains are
 * charged, including NPOT tails.
 * The WebGL surface samples ordinary DepthTexture attachments through a separate
 * UV-oriented r32float image: no color conversion or half-float quantization.
 * Native node/compute inputs borrow the depth attachment directly, with distinct
 * ordinary and comparison samplers captured for each accepted use.
 * Private stencil buffers and UnsignedInt248 DepthStencilFormat attachments
 * retain both aspects for rendering; depth sampling selects only the depth aspect.
 * Source FloatType R/RG/RGBA attachments retain full precision through storage,
 * UV replicas and readback. Nearest sampling needs no optional device feature;
 * linear filtering requires float32-filterable. This owner does not extend the
 * separate draw renderer's admitted attachment/pipeline formats.
 * writePixels(entry, data, options) and copyColor(source, destination, options)
 * update native level-zero color and lazily invalidate only the color replicas.
 * copyDepth(source, destination, options) instead invalidates only the depth
 * replica, preserving color mipmaps and stencil. All copies share admission,
 * allocation budgets, generation checks and nonreentrant native submission.
 * These explicit owner operations do not enqueue the public renderer's FIFO.
 * This is general WebGPU execution, not Rust specialization or a speedup claim.
 */
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {inspectGpuTargetWrite, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {inspectGpuDepthCopy} from './gpu_depth_transfer.mjs';

export class ThreeRenderTargetError extends Error {
  constructor(code, message) {
    super(`THREE_TARGET_${code}: ${message}`);
    this.name = 'ThreeRenderTargetError'; this.code = `THREE_TARGET_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeRenderTargetError(code, message); };
const positive = n => Number.isSafeInteger(n) && n > 0;

function depthComparison(texture, three) {
  if (texture.compareFunction == null) return null;
  const compares = new Map([[three.NeverCompare, 'never'], [three.LessCompare, 'less'],
    [three.EqualCompare, 'equal'], [three.LessEqualCompare, 'less-equal'],
    [three.GreaterCompare, 'greater'], [three.GreaterEqualCompare, 'greater-equal'],
    [three.AlwaysCompare, 'always'], [three.NotEqualCompare, 'not-equal']]);
  const compare = compares.get(texture.compareFunction);
  if (!compare) fail('SAMPLER', 'Expected a source depth comparison function');
  return compare;
}

export function inspectThreeRenderTarget(target, three) {
  if (!three?.RenderTarget || !(target instanceof three.RenderTarget)) fail('SOURCE', 'Expected a source RenderTarget');
  if (!positive(target.width) || !positive(target.height)) fail('SIZE', 'Target dimensions must be positive integers');
  if (target.isWebGLCubeRenderTarget || target.depth !== 1 || target.multiview || target.textures?.length !== 1)
    fail('UNSUPPORTED', 'Only single-color 2D render targets are admitted');
  const texture = target.texture;
  if (!(texture instanceof three.Texture) || texture.isCubeTexture || texture.isDataArrayTexture || texture.isData3DTexture ||
      texture.renderTarget !== target || texture.isRenderTargetTexture !== true)
    fail('SOURCE', 'Expected the target-owned 2D color texture');
  if (target.stencilBuffer && !target.depthBuffer)
    fail('UNSUPPORTED', 'Source stencil targets require depthBuffer');
  if (![0, 1, 4].includes(target.samples)) fail('UNSUPPORTED', 'Target samples must be zero, one or four');
  if (target.samples === 4 && target.resolveColorBuffer === false) fail('UNSUPPORTED', 'MSAA targets must resolve their color');
  const float32 = three.FloatType !== undefined && texture.type === three.FloatType;
  const floatFormat = texture.format === three.RGBAFormat ? 'rgba32float'
    : three.RedFormat !== undefined && texture.format === three.RedFormat ? 'r32float'
    : three.RGFormat !== undefined && texture.format === three.RGFormat ? 'rg32float' : null;
  if (texture.internalFormat != null || (texture.format !== three.RGBAFormat && !(float32 && floatFormat)))
    fail('UNSUPPORTED', 'Targets require RGBA, or FloatType R/RG, with no internal format override');
  if (typeof texture.generateMipmaps !== 'boolean') fail('UNSUPPORTED', 'generateMipmaps must be boolean');
  const linear = texture.colorSpace === three.NoColorSpace || texture.colorSpace === three.LinearSRGBColorSpace;
  let format;
  if (texture.type === three.UnsignedByteType && (linear || texture.colorSpace === three.SRGBColorSpace))
    format = linear ? 'rgba8unorm' : 'rgba8unorm-srgb';
  else if (texture.type === three.HalfFloatType && linear) format = 'rgba16float';
  else if (float32 && linear) format = floatFormat;
  else fail('UNSUPPORTED', 'Target color must be RGBA8, linear RGBA16F or linear R/RG/RGBA32F');
  if (float32 && target.samples === 4) fail('UNSUPPORTED', 'Float32 targets currently require zero or one source samples');
  const sampleCount = target.samples === 4 ? 4 : 1, depthTexture = target.depthTexture ?? null;
  let depthFormat = target.depthBuffer ? (target.stencilBuffer ? 'depth24plus-stencil8' : 'depth24plus') : null;
  if (depthTexture !== null) {
    if (typeof three.DepthTexture !== 'function' || !(depthTexture instanceof three.DepthTexture) ||
        depthTexture.isDepthTexture !== true || depthTexture.renderTarget !== target)
      fail('SOURCE', 'Expected the target-owned source DepthTexture; shared depth attachments are not admitted');
    if (!target.depthBuffer || sampleCount !== 1 || target.useArrayDepthTexture ||
        depthTexture.isArrayTexture || depthTexture.isCubeTexture || (depthTexture.image?.depth ?? 1) !== 1)
      fail('UNSUPPORTED', 'Source depth sampling requires a single-sample 2D depth attachment');
    const combined = three.DepthStencilFormat !== undefined && depthTexture.format === three.DepthStencilFormat;
    if ((!combined && depthTexture.format !== three.DepthFormat) || depthTexture.internalFormat != null ||
        depthTexture.colorSpace !== three.NoColorSpace || depthTexture.generateMipmaps !== false ||
        depthTexture.mipmaps?.length || depthTexture.flipY !== false)
      fail('UNSUPPORTED', 'Depth textures require no mipmaps, UV flipping or color conversion');
    depthComparison(depthTexture, three);
    // WebGLTextures.setupDepthTexture selects the attachment from the texture's
    // format, not stencilBuffer: a depth-only texture has no stencil aspect,
    // while DepthStencilFormat attaches both even with stencilBuffer:false.
    if (combined) {
      if (three.UnsignedInt248Type === undefined || depthTexture.type !== three.UnsignedInt248Type)
        fail('UNSUPPORTED', 'Source depth/stencil textures require UnsignedInt248Type');
      depthFormat = 'depth24plus-stencil8';
    } else if (depthTexture.type === three.UnsignedIntType) depthFormat = 'depth24plus';
    else if (depthTexture.type === three.FloatType) depthFormat = 'depth32float';
    else fail('UNSUPPORTED', 'Source depth type must be UnsignedIntType or FloatType');
    // Match the ordinary sampler2D depth-read profile. Do not pretend a missing
    // float32-filterable device feature can be replaced by lower precision.
    if (![three.NearestFilter, three.LinearFilter].includes(depthTexture.minFilter) ||
        ![three.NearestFilter, three.LinearFilter].includes(depthTexture.magFilter) || depthTexture.anisotropy !== 1)
      fail('SAMPLER', 'Depth textures require nearest/linear filtering and anisotropy one');
    samplerDescriptor(depthTexture, three, 1);
  }
  const generateMipmaps = texture.generateMipmaps;
  const maximumMipLevels = 1 + Math.floor(Math.log2(Math.max(target.width, target.height)));
  if (texture.mipmaps !== undefined && !Array.isArray(texture.mipmaps)) fail('UNSUPPORTED', 'Expected source mipmap descriptors');
  // Source Textures.getMipLevels uses authored descriptors before automatic
  // generation. Render-target descriptors allocate levels, not uploaded pixels.
  const mipLevelCount = texture.mipmaps?.length || (generateMipmaps ? maximumMipLevels : 1);
  if (mipLevelCount > maximumMipLevels) fail('SIZE', 'Too many target mip levels for its extent');
  const texelBytes = format === 'rgba32float' ? 16 : ['rgba16float', 'rg32float'].includes(format) ? 8 : 4;
  const baseColorBytes = target.width * target.height * texelBytes;
  let colorBytes = baseColorBytes, w = target.width, h = target.height;
  for (let level = 1; level < mipLevelCount; level++) {
    w = Math.max(1, Math.floor(w / 2)); h = Math.max(1, Math.floor(h / 2));
    colorBytes += w * h * texelBytes;
  }
  const bytes = colorBytes + (sampleCount === 4 ? baseColorBytes * 4 : 0) +
    (depthFormat ? target.width * target.height * (depthFormat.endsWith('-stencil8') ? 8 : 4) * sampleCount : 0);
  if (!Number.isSafeInteger(bytes)) fail('BUDGET', 'Target byte size exceeds safe integer precision');
  return {width: target.width, height: target.height, format, depthFormat, sampleCount, bytes, colorBytes, texture, generateMipmaps, mipLevelCount, float32,
    depthTexture, depthSampleBytes: depthTexture ? target.width * target.height * 4 : 0,
    key: `${target.width}/${target.height}/${format}/${depthFormat}/${sampleCount}/${generateMipmaps}/${mipLevelCount}`};
}

/** Validate a destination before any GPU allocation or queued submission. */
export function inspectThreeRenderTargetMip(shape, mipLevel = 0) {
  if (!Number.isSafeInteger(mipLevel) || mipLevel < 0 || mipLevel >= shape.mipLevelCount)
    fail('MIP', 'Render mip level is not allocated');
  if (mipLevel > 0 && (shape.sampleCount !== 1 || shape.depthTexture !== null))
    fail('MIP', 'Nonzero target mips require single-sample color and private depth');
  return {width: Math.max(1, Math.floor(shape.width / 2 ** mipLevel)),
    height: Math.max(1, Math.floor(shape.height / 2 ** mipLevel))};
}

function samplerDescriptor(texture, three, mipLevelCount) {
  const filters = new Map([[three.NearestFilter, 'nearest'], [three.LinearFilter, 'linear']]);
  const minification = new Map([
    [three.NearestFilter, ['nearest', 'linear', false]],
    [three.LinearFilter, ['linear', 'linear', false]],
    [three.NearestMipmapNearestFilter, ['nearest', 'nearest', true]],
    [three.NearestMipmapLinearFilter, ['nearest', 'linear', true]],
    [three.LinearMipmapNearestFilter, ['linear', 'nearest', true]],
    [three.LinearMipmapLinearFilter, ['linear', 'linear', true]],
  ]);
  const modes = new Map([[three.ClampToEdgeWrapping, 'clamp-to-edge'], [three.RepeatWrapping, 'repeat'], [three.MirroredRepeatWrapping, 'mirror-repeat']]);
  const magFilter = filters.get(texture.magFilter), min = minification.get(texture.minFilter);
  const addressModeU = modes.get(texture.wrapS), addressModeV = modes.get(texture.wrapT);
  if (!magFilter || !min || !addressModeU || !addressModeV || !Number.isFinite(texture.anisotropy) || texture.anisotropy < 1)
    fail('SAMPLER', 'Expected source target filters, wrap modes and positive anisotropy');
  const [minFilter, mipmapFilter, mips] = min;
  return {magFilter, minFilter, mipmapFilter, addressModeU, addressModeV,
    lodMaxClamp: mips ? mipLevelCount - 1 : 0,
    maxAnisotropy: magFilter === 'linear' && minFilter === 'linear' && mipmapFilter === 'linear'
      ? Math.min(16, Math.floor(texture.anisotropy)) : 1};
}

const FLIP_WGSL = flipY => `
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let size = textureDimensions(source);
  return textureLoad(source, vec2i(i32(p.x), ${flipY ? 'i32(size.y) - 1 - i32(p.y)' : 'i32(p.y)'}), 0);
}`;

// A scalar R32F output preserves textureLoad's f32 depth value. Sampling the
// replica uses the program renderer's existing sampler2D float binding route.
// Its red channel is depth; G/B are zero and A is one. Never use RGBA8 or half
// floats here: small depth errors become large reconstructed-distance errors.
const DEPTH_FLIP_WGSL = flipY => `
@group(0) @binding(0) var source: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) f32 {
  let size = textureDimensions(source);
  return textureLoad(source, vec2i(i32(p.x), ${flipY ? 'i32(size.y) - 1 - i32(p.y)' : 'i32(p.y)'}), 0);
}`;

/** getDevice is lazy so selecting a target before renderer.init() performs no GPU work. */
export function createThreeRenderTargets(three, {getDevice, maxTargets = 256, maxBytes = 512 * 1024 * 1024, flipY = true} = {}) {
  if (typeof getDevice !== 'function' || !positive(maxTargets) || !positive(maxBytes) || typeof flipY !== 'boolean')
    fail('OPTIONS', 'Invalid target owner options');
  const entries = new Map(), pipelines = new Map();
  let disposed = false, allocatedBytes = 0, terminal = null;
  const live = () => { if (disposed) fail('DISPOSED', 'Render target owner is disposed'); if (terminal) throw terminal; };
  function release(entry) {
    if (entry.busy || entry.released) return;
    entry.released = true;
    try { entry._dispatcher?.dispose(); }
    finally {
      entry.storage?.dispose(); entry.sample?.destroy(); entry.depthSample?.destroy();
      allocatedBytes -= entry.charged; entry.charged = 0;
      entry.storage = entry.sample = entry.view = entry.group = entry.flipSteps = null;
      entry.depthSample = entry.depthView = entry.depthGroup = entry.depthSampler = null;
      entry._wanted.clear();
    }
  }
  function retire(entry) {
    if (!entry.alive) return;
    entry.alive = false;
    entry.lifetime.abort(new ThreeRenderTargetError('STALE', 'Source render target disposed or replaced'));
    entry.target.removeEventListener('dispose', entry.onDispose);
    entry.shape.texture.removeEventListener('dispose', entry.onDispose);
    entry.shape.depthTexture?.removeEventListener('dispose', entry.onDispose);
    if (entries.get(entry.target) === entry) entries.delete(entry.target);
    release(entry);
  }
  function capture(target) {
    live();
    const shape = inspectThreeRenderTarget(target, three);
    let entry = entries.get(target);
    if (entry && (entry.shape.key !== shape.key || entry.shape.texture !== shape.texture || entry.shape.depthTexture !== shape.depthTexture)) { retire(entry); entry = null; }
    if (entry) return entry;
    if (entries.size >= maxTargets) fail('BUDGET', 'Render target count exceeds maxTargets');
    entry = {target, shape, alive: true, released: false, busy: false, charged: 0, storage: null, sample: null,
      dirty: true, copies: 0, depthDirty: true, depthCopies: 0, depthSample: null, error: null, lifetime: new AbortController(), validation: Promise.resolve(), _wanted: new Map(), _dispatcher: null};
    entry.onDispose = () => retire(entry);
    target.addEventListener('dispose', entry.onDispose);
    shape.texture.addEventListener('dispose', entry.onDispose);
    shape.depthTexture?.addEventListener('dispose', entry.onDispose);
    entries.set(target, entry);
    return entry;
  }
  function assertCurrent(entry) {
    live();
    if (!entry?.alive || entries.get(entry.target) !== entry) fail('STALE', 'Target was disposed or replaced before its queued use');
    const shape = inspectThreeRenderTarget(entry.target, three);
    if (shape.key !== entry.shape.key || shape.texture !== entry.shape.texture || shape.depthTexture !== entry.shape.depthTexture) fail('STALE', 'Target configuration changed before its queued use');
    if (entry.error) throw entry.error;
    if (entry.storage?.failed) fail('GPU', 'Native target storage failed');
    return entry;
  }
  function budget(bytes) {
    if (bytes > maxBytes - allocatedBytes) fail('BUDGET', 'Render targets exceed their shared byte budget');
  }
  function ensure(entry) {
    assertCurrent(entry);
    if (entry.storage) return entry;
    budget(entry.shape.bytes);
    const device = getDevice();
    if (!device) fail('DEVICE', 'Initialize the renderer before acquiring target storage');
    const {width, height, format, depthFormat, sampleCount, generateMipmaps, mipLevelCount} = entry.shape;
    entry.storage = createGpuRenderTarget(device, {width, height, format, depthFormat, sampleCount, maxBytes, generateMipmaps, mipLevelCount,
      sampleDepth: entry.shape.depthTexture !== null, copyDestination: true});
    entry.charged += entry.shape.bytes; allocatedBytes += entry.shape.bytes;
    // The readback helper consumes this session-shaped object. It copies the
    // original attachment, not the UV-oriented sampling replica.
    entry.session = {device, get width() { return width; }, get height() { return height; },
      get disposed() { return !entry.alive; }, get failed() { return !!entry.error || !!entry.storage?.failed; },
      get diagnostics() { return {target: {format, mipLevelCount}}; },
      withTexture(callback) { assertCurrent(entry); return entry.storage.withTexture(callback); }};
    return entry;
  }
  function checked(entry, operation) {
    const device = getDevice();
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let result, error;
    try { result = operation(device); } catch (cause) { error = cause; }
    const invalid = device.popErrorScope(), oom = device.popErrorScope();
    entry.validation = Promise.all([entry.validation, invalid, oom]).then(([, a, b]) => {
      if (error) throw error;
      if (a || b) throw new ThreeRenderTargetError('GPU', (a || b).message ?? 'Target sampling failed');
    });
    entry.validation.catch(cause => { terminal ??= cause; entry.error = cause; retire(entry); });
    if (error) { terminal ??= error; entry.error = error; retire(entry); throw error; }
    return result;
  }
  function colorSampler(entry, texture) {
    const device = getDevice();
    const filterable = !entry.shape.float32 || device.features?.has?.('float32-filterable') === true;
    const descriptor = samplerDescriptor(texture, three, entry.shape.mipLevelCount);
    if (!filterable) {
      if (texture.magFilter !== three.NearestFilter ||
          (texture.minFilter !== three.NearestFilter && texture.minFilter !== three.NearestMipmapNearestFilter))
        fail('SAMPLER', 'Linear float32 color sampling requires the device float32-filterable feature');
      // A non-filtering sampler must use nearest for all three filters, even
      // when the mip filter is inactive because lodMaxClamp is zero.
      descriptor.mipmapFilter = 'nearest'; descriptor.maxAnisotropy = 1;
    }
    return {descriptor, sampleType: filterable ? 'float' : 'unfilterable-float'};
  }
  function sampling(entry, texture) {
    assertCurrent(entry);
    if (entry.busy) fail('REENTRANT', 'Cannot sample color during a target submission');
    const device = getDevice();
    if (!device) fail('DEVICE', 'Initialize the renderer before sampling target color');
    const {descriptor, sampleType} = colorSampler(entry, texture);
    const samplerKey = JSON.stringify(descriptor);
    ensure(entry);
    entry.busy = true;
    try {
      if (!entry.sample) {
        budget(entry.shape.colorBytes);
        checked(entry, device => {
          const {width, height, format, mipLevelCount} = entry.shape;
          entry.sample = device.createTexture({label: 'Source target UV sampling image',
            size: {width, height, depthOrArrayLayers: 1}, format, dimension: '2d', sampleCount: 1, mipLevelCount, usage: 4 | 16});
          entry.charged += entry.shape.colorBytes; allocatedBytes += entry.shape.colorBytes;
          entry.view = entry.sample.createView();
          let pipeline = pipelines.get(format);
          if (!pipeline) {
            const module = device.createShaderModule({label: 'Source target UV orientation', code: FLIP_WGSL(flipY)});
            // textureLoad does not need filtering, but an automatic layout would
            // still infer sampleType:float and reject a feature-free float32 view.
            const layout = entry.shape.float32 ? device.createPipelineLayout({bindGroupLayouts: [device.createBindGroupLayout({entries: [
              {binding: 0, visibility: 2, texture: {sampleType: 'unfilterable-float', viewDimension: '2d', multisampled: false}},
            ]})]}) : 'auto';
            pipeline = device.createRenderPipeline({label: 'Source target UV orientation', layout,
              vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format}]},
              primitive: {topology: 'triangle-list'}});
            pipelines.set(format, pipeline);
          }
          entry.pipeline = pipeline;
          entry.storage.withTexture(source => {
            entry.flipSteps = [];
            for (let level = 0; level < mipLevelCount; level++) {
              const settings = {baseMipLevel: level, mipLevelCount: 1};
              const view = entry.sample.createView(settings);
              const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
                entries: [{binding: 0, resource: source.createView(settings)}]});
              entry.flipSteps.push({view, group});
            }
            entry.group = entry.flipSteps[0].group;
          });
        });
      }
      if (entry.samplerKey !== samplerKey) {
        entry.sampler = checked(entry, device => device.createSampler(descriptor));
        entry.samplerKey = samplerKey;
      }
      if (entry.dirty) {
        checked(entry, device => entry.storage.withTexture(() => {
          assertCurrent(entry);
          const encoder = device.createCommandEncoder({label: 'Source target UV orientation'});
          for (const {view, group} of entry.flipSteps) {
            const pass = encoder.beginRenderPass({colorAttachments: [{view, loadOp: 'clear', storeOp: 'store', clearValue: {r: 0, g: 0, b: 0, a: 0}}]});
            pass.setPipeline(entry.pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
          }
          assertCurrent(entry);
          device.queue.submit([encoder.finish()]);
        }));
        entry.dirty = false; entry.copies++;
      }
      assertCurrent(entry);
      return {view: entry.view, sampler: entry.sampler, sampleType, viewDimension: '2d',
        version: texture.version, sourceVersion: texture.source.version};
    } finally { entry.busy = false; if (!entry.alive) release(entry); }
  }
  function depthSampling(entry, texture) {
    assertCurrent(entry);
    if (texture.compareFunction != null)
      fail('UNSUPPORTED', 'Comparison depth sampling requires the native depth binding, not the WebGL float replica');
    if (entry.busy) fail('REENTRANT', 'Cannot sample depth during a target submission');
    const device = getDevice();
    if (!device) fail('DEVICE', 'Initialize the renderer before sampling target depth');
    const filterable = device.features?.has?.('float32-filterable') === true;
    if (!filterable && (texture.minFilter !== three.NearestFilter || texture.magFilter !== three.NearestFilter))
      fail('SAMPLER', 'Linear depth sampling requires the device float32-filterable feature');
    const descriptor = {...samplerDescriptor(texture, three, 1), mipmapFilter: 'nearest', maxAnisotropy: 1};
    const samplerKey = JSON.stringify(descriptor);
    ensure(entry);
    // Retirement may be requested by a throwing/reentrant host boundary. Do not
    // release storage until the native withDepthTexture borrow has unwound.
    entry.busy = true;
    try {
      if (!entry.depthSample) {
        budget(entry.shape.depthSampleBytes);
        checked(entry, device => {
          const {width, height} = entry.shape;
          entry.depthSample = device.createTexture({label: 'Source target depth UV sampling image',
            size: {width, height, depthOrArrayLayers: 1}, format: 'r32float', dimension: '2d',
            sampleCount: 1, mipLevelCount: 1, usage: 4 | 16});
          entry.charged += entry.shape.depthSampleBytes; allocatedBytes += entry.shape.depthSampleBytes;
          entry.depthView = entry.depthSample.createView();
          let pipeline = pipelines.get('depth-to-r32float');
          if (!pipeline) {
            const module = device.createShaderModule({label: 'Source target depth UV orientation', code: DEPTH_FLIP_WGSL(flipY)});
            pipeline = device.createRenderPipeline({label: 'Source target depth UV orientation', layout: 'auto',
              vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format: 'r32float'}]},
              primitive: {topology: 'triangle-list'}});
            pipelines.set('depth-to-r32float', pipeline);
          }
          entry.depthPipeline = pipeline;
          entry.storage.withDepthTexture(source => {
            entry.depthGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
              entries: [{binding: 0, resource: source.createView({aspect: 'depth-only'})}]});
          });
        });
      }
      if (entry.depthSamplerKey !== samplerKey) {
        entry.depthSampler = checked(entry, device => device.createSampler(descriptor));
        entry.depthSamplerKey = samplerKey;
      }
      if (entry.depthDirty) {
        checked(entry, device => entry.storage.withDepthTexture(() => {
          assertCurrent(entry);
          const encoder = device.createCommandEncoder({label: 'Source target depth UV orientation'});
          const pass = encoder.beginRenderPass({colorAttachments: [{view: entry.depthView,
            loadOp: 'clear', storeOp: 'store', clearValue: {r: 0, g: 0, b: 0, a: 0}}]});
          pass.setPipeline(entry.depthPipeline); pass.setBindGroup(0, entry.depthGroup); pass.draw(3); pass.end();
          assertCurrent(entry);
          device.queue.submit([encoder.finish()]);
        }));
        entry.depthDirty = false; entry.depthCopies++;
      }
      assertCurrent(entry);
      return {view: entry.depthView, sampler: entry.depthSampler,
        sampleType: filterable ? 'float' : 'unfilterable-float', viewDimension: '2d',
        version: texture.version, sourceVersion: texture.source.version};
    } finally { entry.busy = false; if (!entry.alive) release(entry); }
  }
  // r186 assigns renderTarget on DepthTexture without setting the color-only
  // isRenderTargetTexture flag. Never mutate source texture flags to route it.
  const targetTexture = texture => (texture?.isRenderTargetTexture === true || texture?.isDepthTexture === true) && texture.renderTarget != null;
  // The existing scene bridge accepts a Map of externally owned bindings. Its
  // has/get calls always resolve the current target generation, so stale views
  // trigger the bridge's normal re-preparation boundary before submission.
  function bindingsFor(destination = null, retained = new Map()) {
    if (!(retained instanceof Map)) fail('OPTIONS', 'Expected a Map of borrowed texture bindings');
    return new class extends Map {
      has(texture) { return targetTexture(texture) || retained.has(texture); }
      get(texture) {
        if (!targetTexture(texture)) return retained.get(texture);
        if (texture.renderTarget === destination) fail('FEEDBACK', 'A render pass cannot sample its own color or depth target');
        const entry = capture(texture.renderTarget);
        if (entry.shape.depthTexture === texture) return depthSampling(entry, texture);
        if (entry.shape.texture !== texture) fail('STALE', 'Texture no longer belongs to its source target');
        return sampling(entry, texture);
      }
    }();
  }
  function copyTarget(source, destination, options, depth = false) {
    assertCurrent(source); assertCurrent(destination);
    if (source === destination) fail('FEEDBACK', 'In-place target copies are not admitted');
    if (source.busy || destination.busy) fail('REENTRANT', 'Cannot copy during a target submission');
    const plan = depth
      ? inspectGpuDepthCopy({...source.shape, sampleDepth: source.shape.depthTexture !== null}, destination.shape, options)
      : inspectGpuTargetCopy(source.shape, destination.shape, options);
    if (plan.empty) return false;
    budget((source.storage ? 0 : source.shape.bytes) + (destination.storage ? 0 : destination.shape.bytes));
    ensure(source); ensure(destination);
    source.busy = destination.busy = true;
    if (depth) destination.depthDirty = true; else destination.dirty = true;
    try {
      const result = destination.storage[depth ? 'copyDepthFrom' : 'copyFrom'](source.storage, options);
      assertCurrent(source); assertCurrent(destination); return result;
    } finally {
      source.busy = destination.busy = false;
      try { if (!source.alive) release(source); }
      finally { if (!destination.alive) release(destination); }
    }
  }
  return {
    capture, ensure, assertCurrent, bindingsFor,
    // Native WGSL uses the attachment's top-left coordinates, not the WebGL
    // bridge's UV-flipped sampling replica. Capture only generation and sampler
    // metadata; GPU contents are observed at the consumer's FIFO submission.
    captureNativeTexture(texture, {comparison = false} = {}) {
      live();
      if (!targetTexture(texture))
        fail('SOURCE', 'Expected a target-owned 2D color or depth texture for native sampling');
      const entry = capture(texture.renderTarget), device = getDevice();
      const depth = texture.isDepthTexture === true;
      if ((depth ? entry.shape.depthTexture : entry.shape.texture) !== texture)
        fail('STALE', 'Texture no longer belongs to its source target');
      if (!device) fail('DEVICE', 'Initialize the renderer before capturing native target inputs');
      if (comparison && (!depth || texture.compareFunction == null))
        fail('SAMPLER', 'Comparison sampling requires a source depth texture with compareFunction');
      // Native depth formats are depth bindings, not the WebGL bridge's R32F
      // replica. r186 forces ordinary depth samplers to nearest even when the
      // texture requests linear filtering. Comparison uses the source filters
      // and compare function; one depth view may have both kinds of consumers.
      const {descriptor, sampleType} = depth
        ? {descriptor: {...samplerDescriptor(texture, three, 1), mipmapFilter: 'nearest', maxAnisotropy: 1,
          ...(comparison ? {compare: depthComparison(texture, three)} : {magFilter: 'nearest', minFilter: 'nearest'})}, sampleType: 'depth'}
        : colorSampler(entry, texture);
      const samples = entry.shape.sampleCount;
      return Object.freeze({entry, texture, device, depth, sampler: Object.freeze(descriptor), sampleType,
        sampleData: Object.freeze({samples, primarySamples: 1, isMSAA: samples > 1})});
    },
    validateNativeTextures(sources) {
      live();
      let bytes = 0; const seen = new Set();
      for (const source of sources) {
        const {entry, texture, device} = source;
        assertCurrent(entry);
        if ((source.depth ? entry.shape.depthTexture : entry.shape.texture) !== texture)
          fail('STALE', 'Native texture no longer belongs to its captured target');
        if (device !== getDevice() || entry.session && entry.session.device !== device)
          fail('DEVICE', 'Captured native target belongs to another GPUDevice');
        if (entry.busy) fail('REENTRANT', 'Cannot sample a target during its submission');
        if (!seen.has(entry) && !entry.storage) bytes += entry.shape.bytes;
        seen.add(entry);
      }
      budget(bytes);
    },
    withNativeTexture(source, callback) {
      if (typeof callback !== 'function') fail('CALLBACK', 'Expected a synchronous submitted-use callback');
      return this.withNativeTextures([source], textures => callback(textures.get(source.texture)));
    },
    withNativeTextures(sources, callback) {
      if (!Array.isArray(sources) || !sources.length || typeof callback !== 'function')
        fail('CALLBACK', 'Expected captured native inputs and a synchronous submitted-use callback');
      const {entry, device} = sources[0];
      this.validateNativeTextures(sources);
      if (sources.some(source => source.entry !== entry)) fail('SOURCE', 'A native borrow group must belong to one target');
      const color = sources.some(source => !source.depth), depth = sources.some(source => source.depth);
      ensure(entry);
      if (entry.session.device !== device) fail('DEVICE', 'Native target storage belongs to another GPUDevice');
      const invoke = (nativeColor, nativeDepth) => {
        const textures = new Map(sources.map(source => [source.texture, source.depth ? nativeDepth : nativeColor]));
        const result = callback(textures); assertCurrent(entry); return result;
      };
      entry.busy = true;
      try {
        if (color && depth) return entry.storage.withSampledTextures(invoke);
        if (depth) return entry.storage.withDepthTexture(native => invoke(null, native));
        return entry.storage.withTexture(native => invoke(native, null));
      } finally { entry.busy = false; if (!entry.alive) release(entry); }
    },
    writePixels(entry, data, options = {}) {
      assertCurrent(entry);
      if (entry.busy) fail('REENTRANT', 'Cannot upload during a target submission');
      if (inspectGpuTargetWrite(entry.shape, data, options).empty) return false;
      ensure(entry);
      entry.busy = true; entry.dirty = true;
      try { const result = entry.storage.writePixels(data, options); assertCurrent(entry); return result; }
      finally { entry.busy = false; if (!entry.alive) release(entry); }
    },
    copyColor(source, destination, options = {}) { return copyTarget(source, destination, options); },
    copyDepth(source, destination, options = {}) { return copyTarget(source, destination, options, true); },
    get allocatedBytes() { return allocatedBytes; }, get size() { return entries.size; },
    render(entry, callback, {mipLevel = 0} = {}) {
      assertCurrent(entry); inspectThreeRenderTargetMip(entry.shape, mipLevel);
      ensure(entry);
      if (entry.busy) fail('REENTRANT', 'Cannot reenter a target submission');
      budget(entry.storage.frameAllocationBytes(mipLevel));
      const allocatedBefore = entry.storage.allocatedBytes;
      entry.busy = true; entry.dirty = true;
      // Higher private depth levels never modify the source's level-zero depth.
      if (mipLevel === 0) entry.depthDirty = true;
      try { return entry.storage.withFrame((frame, texture) => { const result = callback(frame, texture); assertCurrent(entry); return result; }, {mipLevel}); }
      finally {
        const delta = entry.storage.allocatedBytes - allocatedBefore;
        entry.charged += delta; allocatedBytes += delta;
        entry.busy = false; if (!entry.alive) release(entry);
      }
    },
    async whenIdle() {
      live();
      try { await Promise.all([...entries.values()].flatMap(entry => [entry.validation, entry.storage?.whenIdle(), entry._dispatcher?.whenIdle()])); }
      catch (error) { throw terminal ?? error; }
      live();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of [...entries.values()]) retire(entry);
      pipelines.clear();
    },
  };
}
