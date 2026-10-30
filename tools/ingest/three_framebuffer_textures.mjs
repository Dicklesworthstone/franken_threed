/** Native color FramebufferTexture residency for viewport nodes and transmission.
 * Captures destination generations and sampler state without reading GPU pixels.
 * The existing target owner supplies storage, validation and mip generation;
 * copies and their mips are encoded together at the captured draw boundary.
 */
import {createGpuRenderTarget} from './gpu_render_target.mjs';

export class ThreeFramebufferTextureError extends Error {
  constructor(code, message) {
    super(`THREE_FRAMEBUFFER_TEXTURE_${code}: ${message}`);
    this.name = 'ThreeFramebufferTextureError'; this.code = `THREE_FRAMEBUFFER_TEXTURE_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeFramebufferTextureError(code, message); };
const positive = value => Number.isSafeInteger(value) && value > 0;
const texelBytes = new Map([['rgba8unorm', 4], ['bgra8unorm', 4], ['rgba8unorm-srgb', 4],
  ['bgra8unorm-srgb', 4], ['rgba16float', 8], ['r32float', 4], ['rg32float', 8], ['rgba32float', 16]]);

export function createThreeFramebufferTextures({three, device, getFormat, signal = null,
  maxTextures = 256, maxBytes = 256 * 1024 * 1024, maxBindings = 4096, maxSamplers = 256, webgl = false, submitInitialization = null} = {}) {
  if (typeof three?.FramebufferTexture !== 'function' || !device?.queue || typeof getFormat !== 'function' ||
      ![maxTextures, maxBytes, maxBindings, maxSamplers].every(positive) || typeof webgl !== 'boolean' ||
      submitInitialization !== null && typeof submitInitialization !== 'function')
    fail('OPTIONS', 'Expected pinned texture constructors, a GPUDevice, framebuffer format and positive budgets');
  const current = new Map(), entries = new Set(), leases = new Map(), tokens = new WeakSet(), watches = new Map(), samplers = new Map(), pending = new Set();
  const copyPipelines = new Map();
  let bytes = 0, active = null, terminal = null, disposed = false;
  const live = () => { if (disposed) fail('DISPOSED', 'Framebuffer textures are disposed'); if (terminal) throw terminal; };
  function track(task) {
    pending.add(task); task.finally(() => pending.delete(task)).catch(() => {});
    task.catch(error => { terminal ??= error; });
  }
  function retire(entry) {
    if (entry.retiring || entry.borrowed) return;
    entry.retiring = true;
    const release = () => { entry.storage?.dispose(); entries.delete(entry); bytes -= entry.shape.bytes; };
    if (entry.storage) track(entry.storage.whenIdle().then(release, error => { release(); throw error; }));
    else release();
  }
  function discard(entry) {
    entry.alive = false;
    if (current.get(entry.texture) === entry) current.delete(entry.texture);
    retire(entry);
  }
  function watch(texture) {
    if (watches.has(texture)) return;
    const listener = () => {
      for (const entry of entries) if (entry.texture === texture) discard(entry);
      texture.removeEventListener('dispose', listener); watches.delete(texture);
    };
    texture.addEventListener('dispose', listener); watches.set(texture, listener);
  }
  function stop(error) {
    terminal ??= error;
    for (const entry of entries) discard(entry);
    for (const [texture, listener] of watches) texture.removeEventListener('dispose', listener);
    watches.clear(); samplers.clear(); copyPipelines.clear(); signal?.removeEventListener('abort', abort);
  }
  const abort = () => stop(signal.reason ?? new ThreeFramebufferTextureError('ABORTED', 'Renderer lifetime ended'));
  if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, {once: true});
  device.lost?.then(info => stop(new ThreeFramebufferTextureError('LOST', info?.message ?? 'WebGPU device lost')), stop);
  function inspect(texture, format = getFormat(), layoutOnly = false) {
    live();
    if (!(texture instanceof three.FramebufferTexture) || texture.isFramebufferTexture !== true ||
        texture.isDepthTexture || texture.isCubeTexture || texture.isArrayTexture || texture.is3DTexture)
      fail('SOURCE', 'Expected a pinned color FramebufferTexture');
    if (webgl) {
      const linear = texture.colorSpace === three.NoColorSpace || texture.colorSpace === three.LinearSRGBColorSpace;
      if (texture.format !== three.RGBAFormat || !linear && texture.colorSpace !== three.SRGBColorSpace)
        fail('FORMAT', 'WebGL framebuffer snapshots require RGBA with linear or sRGB color space');
      format = texture.type === three.UnsignedByteType ? `rgba8unorm${linear ? '' : '-srgb'}`
        : linear && texture.type === three.HalfFloatType ? 'rgba16float'
          : linear && texture.type === three.FloatType ? 'rgba32float' : null;
    }
    if (!texelBytes.has(format) || texture.mipmaps?.length || texture.internalFormat != null || typeof texture.generateMipmaps !== 'boolean')
      fail('FORMAT', 'Expected an admitted framebuffer format with no authored mip or internal format override');
    const sampleType = format.includes('32float') && !device.features?.has?.('float32-filterable') ? 'unfilterable-float' : 'float';
    // Source viewport textures have no extent until their first updateBefore.
    // Shader layout inspection needs the format, never an allocation or size.
    if (layoutOnly) return {format, sampleType, mipLevelCount: 1};
    const width = texture.image?.width, height = texture.image?.height;
    if (!positive(width) || !positive(height) || width > device.limits.maxTextureDimension2D || height > device.limits.maxTextureDimension2D ||
        (texture.image.depth ?? 1) !== 1 || texture.image.data != null)
      fail('SHAPE', 'Framebuffer texture dimensions must fit a 2D GPU attachment');
    const mipLevelCount = texture.generateMipmaps || webgl && texture.minFilter !== three.NearestFilter && texture.minFilter !== three.LinearFilter
      ? 1 + Math.floor(Math.log2(Math.max(width, height))) : 1;
    let size = 0;
    for (let level = 0; level < mipLevelCount; level++)
      size += Math.max(1, Math.floor(width / 2 ** level)) * Math.max(1, Math.floor(height / 2 ** level)) * texelBytes.get(format);
    if (!Number.isSafeInteger(size)) fail('BUDGET', 'Framebuffer size exceeds integer precision');
    return Object.freeze({width, height, format, mipLevelCount, sampleType, bytes: size});
  }
  function shapeOf(texture, layoutOnly = false) {
    live();
    // ViewportTextureNode restores generateMipmaps after copying. Source texture
    // allocation remains unchanged until its public version advances.
    const entry = current.get(texture);
    return entry?.version === texture.version ? entry.shape : inspect(texture, undefined, layoutOnly);
  }
  function samplerOf(texture, shape) {
    const filters = new Map([[three.NearestFilter, 'nearest'], [three.LinearFilter, 'linear']]);
    const minFilters = new Map([[three.NearestFilter, ['nearest', 'nearest', false]], [three.LinearFilter, ['linear', 'nearest', false]],
      [three.NearestMipmapNearestFilter, ['nearest', 'nearest', true]], [three.NearestMipmapLinearFilter, ['nearest', 'linear', true]],
      [three.LinearMipmapNearestFilter, ['linear', 'nearest', true]], [three.LinearMipmapLinearFilter, ['linear', 'linear', true]]]);
    const wraps = new Map([[three.ClampToEdgeWrapping, 'clamp-to-edge'], [three.RepeatWrapping, 'repeat'], [three.MirroredRepeatWrapping, 'mirror-repeat']]);
    const magFilter = filters.get(texture.magFilter), min = minFilters.get(texture.minFilter);
    const addressModeU = wraps.get(texture.wrapS), addressModeV = wraps.get(texture.wrapT);
    if (!magFilter || !min || !addressModeU || !addressModeV || !Number.isFinite(texture.anisotropy) || texture.anisotropy < 1 || texture.compareFunction != null)
      fail('SAMPLER', 'Expected ordinary source texture filtering and wrapping');
    const [minFilter, mipmapFilter, useMips] = min;
    if (shape.sampleType === 'unfilterable-float' && [magFilter, minFilter, mipmapFilter].includes('linear'))
      fail('SAMPLER', 'Linear float32 framebuffer sampling requires float32-filterable');
    return Object.freeze({magFilter, minFilter, mipmapFilter, addressModeU, addressModeV,
      lodMaxClamp: useMips ? shape.mipLevelCount - 1 : 0,
      maxAnisotropy: [magFilter, minFilter, mipmapFilter].every(value => value === 'linear') ? Math.min(16, Math.floor(texture.anisotropy)) : 1});
  }
  function bindingLayout(binding, shape) {
    if (binding?.store || !binding?.isSampler && !binding?.isSampledTexture)
      fail('BINDING', 'Framebuffer textures are read-only sampled inputs');
    return binding.isSampler && !binding.isSampledTexture
      ? {sampler: {type: shape.sampleType === 'float' ? 'filtering' : 'non-filtering'}}
      : {texture: {sampleType: shape.sampleType, viewDimension: '2d', multisampled: false}};
  }
  function entryOf(texture, format) {
    live();
    let entry = current.get(texture);
    if (entry?.version === texture.version) return entry;
    // WebGLTextures changes its allocation only when the source/cache key
    // changes (or disposal), not for another upload request to the same image.
    const key = webgl ? [texture.wrapS, texture.wrapT, texture.wrapR || 0, texture.magFilter, texture.minFilter,
      texture.anisotropy, texture.internalFormat, texture.format, texture.type, texture.generateMipmaps,
      texture.premultiplyAlpha, texture.flipY, texture.unpackAlignment, texture.colorSpace].join() : null;
    if (webgl && entry?.source === texture.source && entry.key === key) return entry;
    const shape = inspect(texture, format);
    if (entries.size >= maxTextures || shape.bytes > maxBytes - bytes)
      fail('BUDGET', 'Captured framebuffer generations exceed texture or byte capacity');
    const next = {texture, version: texture.version, source: texture.source, key, shape, refs: 0, storage: null, view: null,
      sampler: webgl ? samplerOf(texture, shape) : null, uploadedSourceVersion: -1, alive: true, borrowed: false, retiring: false};
    watch(texture); entries.add(next); current.set(texture, next); bytes += shape.bytes;
    if (entry?.refs === 0) discard(entry);
    return next;
  }
  function lease(entry, state) {
    if (leases.size >= maxBindings) fail('BUDGET', 'Queued framebuffer bindings exceed maxBindings');
    const token = Object.freeze({}); leases.set(token, {entry, ...state}); tokens.add(token); entry.refs++; return token;
  }
  function validate(token) {
    live(); const state = leases.get(token);
    if (!state?.entry.alive) fail('STALE', 'Framebuffer texture was disposed or its captured use was released');
    return state;
  }
  function ensure(entry) {
    if (!entry.storage) entry.storage = createGpuRenderTarget(device, {width: entry.shape.width, height: entry.shape.height,
      format: entry.shape.format, depthFormat: null, sampleCount: 1, copyDestination: true,
      mipLevelCount: entry.shape.mipLevelCount, maxBytes: entry.shape.bytes, label: 'f3d-framebuffer-texture'});
  }
  function initializeWebGL(texture, entry = entryOf(texture)) {
    ensure(entry);
    if (entry.uploadedSourceVersion >= 0 && entry.version === texture.version) return entry;
    if (entry.uploadedSourceVersion !== texture.source.version) {
      if (texture.generateMipmaps && entry.shape.mipLevelCount > 1) {
        const token = lease(entry, {initialize: true}), release = () => api.release(token);
        const issue = () => {
          validate(token);
          entry.storage.withColorDestination((native, encodeMipmaps) => {
            const encoder = device.createCommandEncoder({label: 'f3d-webgl-framebuffer-initialize'});
            encodeMipmaps(encoder); device.queue.submit([encoder.finish()]);
          });
        };
        try {
          if (submitInitialization) submitInitialization({framebufferCopy: true, issue, releaseFramebuffer: release});
          else { try { issue(); } finally { release(); } }
        } catch (error) { release(); throw error; }
      }
      // r186 acknowledges the source before onUpdate and the texture afterward.
      // The callback runs before choosing the copy's current source framebuffer.
      entry.uploadedSourceVersion = texture.source.version;
      texture.onUpdate?.(texture);
      if (!entry.alive) fail('STALE', 'Framebuffer texture was disposed during initialization');
    }
    entry.version = texture.version; return entry;
  }
  function sampler(descriptor) {
    const key = JSON.stringify(descriptor); let result = samplers.get(key);
    if (!result) {
      result = device.createSampler(descriptor);
      if (samplers.size >= maxSamplers) samplers.delete(samplers.keys().next().value);
    } else samplers.delete(key);
    samplers.set(key, result); return result;
  }
  function encodeWebGLCopy(state, encoder, source, destination) {
    const {rectangle: r, entry, sourceLevel, sourceFace, level} = state;
    const width = Math.max(1, source.width >> sourceLevel), height = Math.max(1, source.height >> sourceLevel);
    if (source.sampleCount !== 1 || r.x + r.z > width || r.y + r.w > height ||
        sourceLevel >= source.mipLevelCount || sourceFace >= (source.dimension === '3d'
          ? Math.max(1, source.depthOrArrayLayers >> sourceLevel) : source.depthOrArrayLayers))
      fail('COPY', 'Framebuffer source must contain the selected resolved attachment and copied extent');
    if (source.format.endsWith('-srgb') !== entry.shape.format.endsWith('-srgb'))
      fail('FORMAT', 'WebGL framebuffer copies require matching storage color encoding');
    const format = entry.shape.format.replace(/-srgb$/, ''), sourceFormat = source.format.replace(/-srgb$/, '');
    if (!texelBytes.has(source.format)) fail('FORMAT', 'Expected a normalized or floating-point color framebuffer');
    if (sourceFormat.includes('float') !== format.includes('float'))
      fail('FORMAT', 'WebGL framebuffer copies require matching normalized or floating-point format classes');
    if (sourceFormat.replace(/^bgra/, 'rgba') !== format)
      fail('FORMAT', 'WebGL framebuffer copies require matching RGBA channel precision');
    const volume = source.dimension === '3d', key = `${format}/${volume}`;
    let pipeline = copyPipelines.get(key);
    if (!pipeline) {
      const module = device.createShaderModule({label: 'WebGL framebuffer texel copy', code: `
@group(0) @binding(0) var source: texture_${volume ? '3d' : '2d'}<f32>;
@group(0) @binding(1) var<uniform> region: vec4u;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let xy = vec2i(i32(region.x) + i32(p.x), i32(region.z) - 1 - i32(region.y) - i32(p.y));
  return textureLoad(source, ${volume ? 'vec3i(xy, i32(region.w))' : 'xy'}, 0);
}`});
      const layout = device.createBindGroupLayout({entries: [
        {binding: 0, visibility: 2, texture: {sampleType: 'unfilterable-float', viewDimension: volume ? '3d' : '2d'}},
        {binding: 1, visibility: 2, buffer: {type: 'uniform'}},
      ]});
      pipeline = device.createRenderPipeline({label: 'WebGL framebuffer texel copy',
        layout: device.createPipelineLayout({bindGroupLayouts: [layout]}),
        vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format}]},
        primitive: {topology: 'triangle-list'}});
      copyPipelines.set(key, pipeline);
    }
    state.parameters ??= device.createBuffer({label: 'WebGL framebuffer copy region', size: 16, usage: 64 | 8});
    device.queue.writeBuffer(state.parameters, 0, new Uint32Array([r.x, r.y, height, sourceFace]));
    const view = source.createView({format: sourceFormat, dimension: volume ? '3d' : '2d', baseMipLevel: sourceLevel, mipLevelCount: 1,
      ...(volume ? {} : {baseArrayLayer: sourceFace, arrayLayerCount: 1})});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: view}, {binding: 1, resource: {buffer: state.parameters}},
    ]});
    const pass = encoder.beginRenderPass({colorAttachments: [{view: destination.createView({format, baseMipLevel: level, mipLevelCount: 1}),
      loadOp: 'load', storeOp: 'store'}]});
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.setViewport(0, 0, r.z, r.w, 0, 1); pass.draw(3); pass.end();
  }
  const api = {
    owns(token) { return tokens.has(token); },
    layoutOf(binding) { const shape = shapeOf(binding.texture, true); samplerOf(binding.texture, shape); return bindingLayout(binding, shape); },
    sampleData(texture) { const shape = shapeOf(texture, true); samplerOf(texture, shape); return {samples: 1, primarySamples: 1, isMSAA: false}; },
    captureBinding(binding) {
      if (leases.size >= maxBindings) fail('BUDGET', 'Queued framebuffer bindings exceed maxBindings');
      const shape = shapeOf(binding.texture), layout = bindingLayout(binding, shape), sampler = samplerOf(binding.texture, shape);
      return lease(entryOf(binding.texture, shape.format), {layout, sampler: binding.isSampler && !binding.isSampledTexture ? sampler : null});
    },
    prepareFramebuffer(texture, type) {
      live();
      if (webgl) { initializeWebGL(texture); return; }
      // Source Textures.updateTexture returns early for an allocated version.
      // The public type can differ from the backend framebuffer's actual format.
      if (current.get(texture)?.version !== texture.version) texture.type = type;
    },
    captureProgramBindings() {
      live(); if (!webgl) fail('BINDING', 'Program bindings require the WebGL framebuffer profile');
      const bindings = new Map();
      try {
        for (const [texture, entry] of current) bindings.set(texture, lease(entry, {program: true}));
        return bindings;
      } catch (error) { for (const token of bindings.values()) this.release(token); throw error; }
    },
    binding(texture, {initialize = false, snapshot = null} = {}) {
      live(); if (!webgl) fail('BINDING', 'Direct program bindings require the WebGL framebuffer profile');
      const entry = snapshot ? validate(snapshot).entry : entryOf(texture);
      if (initialize && current.get(texture) === entry) initializeWebGL(texture, entry);
      ensure(entry);
      entry.storage.withColorDestination(native => { entry.view ??= native.createView(); });
      return {view: entry.view, sampler: sampler(entry.sampler), sampleType: entry.shape.sampleType, viewDimension: '2d',
        version: texture.version, sourceVersion: texture.source.version};
    },
    captureFramebuffer(texture, {format = getFormat(), rectangle = null, level = 0, sourceLevel = 0, sourceFace = 0} = {}) {
      live();
      if (leases.size >= maxBindings) fail('BUDGET', 'Queued framebuffer bindings exceed maxBindings');
      const shape = webgl ? entryOf(texture).shape : inspect(texture, format);
      const region = rectangle ?? {x: 0, y: 0, z: shape.width, w: shape.height};
      const rect = Object.freeze({x: region.x, y: region.y, z: region.z, w: region.w});
      if (![rect.x, rect.y, rect.z, rect.w].every(value => Number.isSafeInteger(value) && value >= 0) ||
          rect.z > shape.width || rect.w > shape.height) fail('RECTANGLE', 'Framebuffer copy rectangle must fit the destination');
      const entry = entryOf(texture, format);
      if (![level, sourceLevel, sourceFace].every(value => Number.isSafeInteger(value) && value >= 0) ||
          level >= entry.shape.mipLevelCount || !webgl && (level || sourceLevel || sourceFace))
        fail('MIP', 'Framebuffer copy levels must address allocated source and destination images');
      if (rect.z > Math.max(1, entry.shape.width >> level) || rect.w > Math.max(1, entry.shape.height >> level))
        fail('RECTANGLE', 'Framebuffer copy exceeds its allocated texture generation');
      return lease(entry, {copy: true, rectangle: rect, level, sourceLevel, sourceFace, generateMipmaps: !webgl && texture.generateMipmaps});
    },
    layout(token) { return validate(token).layout; },
    validateDispatch(tokens) { for (const token of tokens) validate(token); },
    withDispatch(tokens, run) {
      live();
      if (active) fail('REENTRY', 'Framebuffer texture borrowing cannot be nested');
      if (typeof run !== 'function') fail('CALLBACK', 'Expected synchronous submitted use');
      const unique = [...new Set(tokens.map(token => validate(token).entry))];
      active = new Map();
      const visit = index => {
        if (index === unique.length) {
          const result = run();
          if (result?.then) { Promise.resolve(result).catch(() => {}); fail('ASYNC_USE', 'Submit framebuffer work before releasing its textures'); }
          return result;
        }
        const entry = unique[index]; ensure(entry); entry.borrowed = true;
        try {
          let result;
          entry.storage.withColorDestination((texture, encodeMipmaps) => {
            active.set(entry, {texture, encodeMipmaps}); result = visit(index + 1);
          });
          return result;
        } finally { entry.borrowed = false; if (!entry.alive) retire(entry); }
      };
      try { return visit(0); }
      finally { active = null; }
    },
    resolve(token) {
      const state = validate(token), borrowed = active?.get(state.entry);
      if (!borrowed) fail('BORROW', 'Resolve framebuffer inputs inside their submitted use');
      if (state.copy) fail('BINDING', 'A framebuffer copy marker is not a sampled binding');
      if (state.sampler) return sampler(state.sampler);
      return state.entry.view ??= borrowed.texture.createView({dimension: '2d', baseMipLevel: 0, mipLevelCount: state.entry.shape.mipLevelCount});
    },
    encodeFramebufferCopy(token, encoder, source) {
      const state = validate(token), borrowed = active?.get(state.entry);
      if (!state.copy || !borrowed) fail('BORROW', 'Encode a captured framebuffer copy inside its submitted use');
      if (webgl) return encodeWebGLCopy(state, encoder, source, borrowed.texture);
      const {rectangle: r, entry} = state;
      if (!encoder || typeof encoder.copyTextureToTexture !== 'function' || source?.format !== entry.shape.format || source.sampleCount !== 1 ||
          r.x + r.z > source.width || r.y + r.w > source.height)
        fail('COPY', 'Framebuffer source must have the captured format, one resolved sample and the copied extent');
      encoder.copyTextureToTexture({texture: source, origin: {x: r.x, y: r.y, z: 0}, mipLevel: 0},
        {texture: borrowed.texture, origin: {x: 0, y: 0, z: 0}, mipLevel: 0}, {width: r.z, height: r.w, depthOrArrayLayers: 1});
      if (state.generateMipmaps) borrowed.encodeMipmaps(encoder);
    },
    release(token) {
      const state = leases.get(token); if (!state) return;
      state.parameters?.destroy();
      leases.delete(token); const entry = state.entry;
      if (--entry.refs === 0 && current.get(entry.texture) !== entry) discard(entry);
    },
    async whenIdle() {
      live(); if (active) fail('REENTRY', 'Cannot await framebuffer completion inside a submitted use');
      await Promise.all([...pending, ...[...entries].filter(entry => entry.alive && entry.storage).map(entry => entry.storage.whenIdle())]); live();
    },
    get diagnostics() { return {textures: current.size, generations: entries.size, bindings: leases.size, bytes}; },
    dispose() { if (disposed) return; disposed = true; stop(new ThreeFramebufferTextureError('DISPOSED', 'Framebuffer owner disposed')); },
  };
  return api;
}
