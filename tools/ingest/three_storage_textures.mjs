/** Renderer-owned 2D StorageTexture residency shared by TSL compute and scenes.
 * No source pixels, readbacks, upload shadows or retained renderer submissions.
 * Views keep native WebGPU orientation. Writes may refresh the mip pyramid on
 * the GPU; ordinary material bindings sample the same allocation. */
export class ThreeStorageTextureError extends Error {
  constructor(code, message) {
    super(`THREE_STORAGE_TEXTURE_${code}: ${message}`);
    this.name = 'ThreeStorageTextureError'; this.code = `THREE_STORAGE_TEXTURE_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeStorageTextureError(code, message); };
const positive = n => Number.isSafeInteger(n) && n > 0;
const accessNames = new Map([['readOnly', 'read-only'], ['writeOnly', 'write-only'], ['readWrite', 'read-write']]);

export function inspectThreeStorageTexture(texture, three) {
  if (typeof three?.Texture !== 'function' || !(texture instanceof three.Texture) || !texture.isStorageTexture)
    fail('SOURCE', 'Expected a pinned Three StorageTexture');
  if (texture.is3DTexture || texture.isArrayTexture || texture.isCubeTexture || texture.isRenderTargetTexture ||
      !positive(texture.image?.width) || !positive(texture.image?.height) || (texture.image.depth ?? 1) !== 1)
    fail('SHAPE', 'Storage residency currently requires a positive, two-dimensional texture');
  if (texture.image.data != null || texture.mipmaps?.length || texture.internalFormat != null ||
      ![three.NoColorSpace, three.LinearSRGBColorSpace].includes(texture.colorSpace))
    fail('FORMAT', 'Storage textures require linear GPU-authored data and no internal-format override');
  let format, texelBytes;
  if (texture.type === three.UnsignedByteType && texture.format === three.RGBAFormat) [format, texelBytes] = ['rgba8unorm', 4];
  else if (texture.type === three.HalfFloatType && texture.format === three.RGBAFormat) [format, texelBytes] = ['rgba16float', 8];
  else if (texture.type === three.FloatType) {
    if (texture.format === three.RedFormat) [format, texelBytes] = ['r32float', 4];
    else if (texture.format === three.RGFormat) [format, texelBytes] = ['rg32float', 8];
    else if (texture.format === three.RGBAFormat) [format, texelBytes] = ['rgba32float', 16];
  }
  if (!format) fail('FORMAT', 'Storage color must be RGBA8, RGBA16F or R/RG/RGBA32F');
  if (typeof texture.generateMipmaps !== 'boolean' || typeof texture.mipmapsAutoUpdate !== 'boolean')
    fail('MIPS', 'Expected boolean storage mipmap controls');
  const {width, height} = texture.image;
  const mipLevelCount = texture.generateMipmaps ? 1 + Math.floor(Math.log2(Math.max(width, height))) : 1;
  let bytes = 0;
  for (let level = 0; level < mipLevelCount; level++)
    bytes += Math.max(1, Math.floor(width / 2 ** level)) * Math.max(1, Math.floor(height / 2 ** level)) * texelBytes;
  if (!Number.isSafeInteger(bytes)) fail('BUDGET', 'Storage texture size exceeds safe integer precision');
  return {width, height, format, texelBytes, mipLevelCount, bytes,
    float32: texture.type === three.FloatType, key: `${width}/${height}/${format}/${mipLevelCount}`};
}

function samplerOf(texture, three, shape, device) {
  const filters = new Map([[three.NearestFilter, 'nearest'], [three.LinearFilter, 'linear']]);
  const minFilters = new Map([[three.NearestFilter, ['nearest', 'nearest', false]], [three.LinearFilter, ['linear', 'nearest', false]],
    [three.NearestMipmapNearestFilter, ['nearest', 'nearest', true]], [three.NearestMipmapLinearFilter, ['nearest', 'linear', true]],
    [three.LinearMipmapNearestFilter, ['linear', 'nearest', true]], [three.LinearMipmapLinearFilter, ['linear', 'linear', true]]]);
  const wraps = new Map([[three.ClampToEdgeWrapping, 'clamp-to-edge'], [three.RepeatWrapping, 'repeat'], [three.MirroredRepeatWrapping, 'mirror-repeat']]);
  const magFilter = filters.get(texture.magFilter), min = minFilters.get(texture.minFilter);
  const addressModeU = wraps.get(texture.wrapS), addressModeV = wraps.get(texture.wrapT);
  if (!magFilter || !min || !addressModeU || !addressModeV || !Number.isFinite(texture.anisotropy) || texture.anisotropy < 1 || texture.compareFunction != null)
    fail('SAMPLER', 'Expected ordinary source texture filters and wrapping');
  const [minFilter, mipmapFilter, useMips] = min;
  const filtering = magFilter === 'linear' || minFilter === 'linear' || mipmapFilter === 'linear';
  if (shape.float32 && filtering && !device?.features?.has?.('float32-filterable'))
    fail('SAMPLER', 'Linear float32 sampling requires float32-filterable; use nearest filters');
  return {magFilter, minFilter, mipmapFilter, addressModeU, addressModeV,
    lodMaxClamp: useMips ? shape.mipLevelCount - 1 : 0,
    maxAnisotropy: magFilter === 'linear' && minFilter === 'linear' && mipmapFilter === 'linear' ? Math.min(16, Math.floor(texture.anisotropy)) : 1};
}

// Center-sampled bilinear reduction, including NPOT extents. textureLoad keeps
// float32 mips full precision even on devices without float32-filterable.
const MIPS = `
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(source));
  let target = max(size / 2, vec2i(1));
  let uv = p.xy / vec2f(target) * vec2f(size) - 0.5;
  let base = vec2i(floor(uv)); let f = fract(uv);
  let hi = size - 1;
  let a = textureLoad(source, clamp(base, vec2i(0), hi), 0);
  let b = textureLoad(source, clamp(base + vec2i(1,0), vec2i(0), hi), 0);
  let c = textureLoad(source, clamp(base + vec2i(0,1), vec2i(0), hi), 0);
  let d = textureLoad(source, clamp(base + vec2i(1,1), vec2i(0), hi), 0);
  return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}`;

export function createThreeStorageTextures(three, {getDevice, signal = null, maxTextures = 256, maxBytes = 256 * 1024 * 1024} = {}) {
  if (typeof getDevice !== 'function' || !positive(maxTextures) || !positive(maxBytes)) fail('OPTIONS', 'Expected storage texture owner options');
  const entries = new Map(), mipPipelines = new Map(), pending = new Set();
  let allocatedBytes = 0, terminal = null, watchedDevice = null;
  const stopped = new AbortController();
  const live = () => { if (terminal) throw terminal; };
  function retire(entry) {
    if (!entry.alive) return;
    entry.alive = false; entry.texture.removeEventListener('dispose', entry.onDispose);
    entry.native?.destroy(); allocatedBytes -= entry.charged; entry.charged = 0;
    if (entries.get(entry.texture) === entry) entries.delete(entry.texture);
  }
  function stop(error) {
    if (terminal) return;
    terminal = error instanceof Error ? error : new ThreeStorageTextureError('ABORTED', String(error ?? 'Texture execution stopped'));
    stopped.abort(terminal);
    for (const entry of [...entries.values()]) retire(entry);
    mipPipelines.clear(); signal?.removeEventListener('abort', abort);
  }
  const abort = () => stop(signal.reason);
  if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, {once: true});
  function device() {
    live(); const value = getDevice();
    if (!value) fail('DEVICE', 'Initialize the renderer before acquiring texture residency');
    if (watchedDevice && watchedDevice !== value) fail('DEVICE', 'Storage textures cannot migrate between devices');
    if (!watchedDevice) {
      watchedDevice = value;
      value.lost?.then(info => stop(new ThreeStorageTextureError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    }
    return value;
  }
  function checked(run) {
    const d = device(); d.pushErrorScope('out-of-memory'); d.pushErrorScope('validation');
    try { return run(d); }
    catch (error) { stop(error); throw error; }
    finally {
      const task = Promise.all([d.popErrorScope(), d.popErrorScope()]).then(errors => {
        const error = errors.find(Boolean); if (error) throw new ThreeStorageTextureError('GPU', error.message);
      }).catch(error => { stop(error); throw error; });
      pending.add(task); task.finally(() => pending.delete(task)).catch(() => {});
    }
  }
  function capture(texture) {
    live(); const shape = inspectThreeStorageTexture(texture, three);
    let entry = entries.get(texture);
    if (entry && entry.shape.key !== shape.key) { retire(entry); entry = null; }
    if (!entry) {
      if (entries.size >= maxTextures) fail('BUDGET', 'Storage texture count exceeds maxTextures');
      entry = {texture, shape, alive: true, native: null, view: null, charged: 0, mipsDirty: false, views: new Map(), samplers: new Map()};
      entry.onDispose = () => retire(entry);
      texture.addEventListener('dispose', entry.onDispose); entries.set(texture, entry);
    }
    return entry;
  }
  function assertCurrent(entry) {
    live();
    if (!entry?.alive || entries.get(entry.texture) !== entry || inspectThreeStorageTexture(entry.texture, three).key !== entry.shape.key)
      fail('STALE', 'Storage texture was resized, disposed or reconfigured before its queued use');
    return entry;
  }
  function ensure(entry) {
    assertCurrent(entry);
    if (entry.native) return entry;
    const d = device(), {width, height, mipLevelCount, format, bytes} = entry.shape;
    if (width > d.limits.maxTextureDimension2D || height > d.limits.maxTextureDimension2D) fail('LIMIT', 'Storage texture exceeds device dimensions');
    if (bytes > maxBytes - allocatedBytes) fail('BUDGET', 'Storage texture allocations exceed maxBytes');
    // All uses share this allocation. RENDER_ATTACHMENT serves GPU mip updates.
    checked(d => {
      entry.native = d.createTexture({label: entry.texture.name || 'f3d-storage-texture',
        size: [width, height, 1], dimension: '2d', mipLevelCount, sampleCount: 1, format, usage: 1 | 2 | 4 | 8 | 16});
      allocatedBytes += bytes; entry.charged = bytes;
      entry.view = entry.native.createView();
    });
    return entry;
  }
  function captureBinding(binding) {
    const entry = capture(binding.texture), d = device();
    if (binding.isSampler) return {entry, kind: 'sampler', sampler: samplerOf(binding.texture, three, entry.shape, d)};
    if (!binding.isSampledTexture) fail('BINDING', 'Expected a sampled or storage texture binding');
    if (!binding.store) return {entry, kind: 'sampled'};
    const access = accessNames.get(binding.access), mipLevel = binding.mipLevel ?? 0;
    if (!access || !Number.isSafeInteger(mipLevel) || mipLevel < 0 || mipLevel >= entry.shape.mipLevelCount)
      fail('BINDING', 'Expected source storage access and an allocated mip level');
    return {entry, kind: 'storage', access, mipLevel, autoMips: binding.texture.mipmapsAutoUpdate};
  }
  function layout(snapshot) {
    const {entry, kind} = snapshot;
    if (kind === 'sampler') return {sampler: {type: 'filtering'}};
    if (kind === 'storage') return {storageTexture: {access: snapshot.access, format: entry.shape.format, viewDimension: '2d'}};
    return {texture: {sampleType: entry.shape.float32 && !device().features?.has?.('float32-filterable') ? 'unfilterable-float' : 'float', viewDimension: '2d', multisampled: false}};
  }
  function refreshMips(entry) {
    if (!entry.mipsDirty || entry.shape.mipLevelCount === 1) return;
    checked(d => {
      let pipeline = mipPipelines.get(entry.shape.format);
      if (!pipeline) {
        const module = d.createShaderModule({label: 'f3d-storage-mips', code: MIPS});
        const bgl = d.createBindGroupLayout({entries: [{binding: 0, visibility: 2, texture: {sampleType: 'unfilterable-float'}}]});
        pipeline = d.createRenderPipeline({layout: d.createPipelineLayout({bindGroupLayouts: [bgl]}),
          vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format: entry.shape.format}]}, primitive: {topology: 'triangle-list'}});
        mipPipelines.set(entry.shape.format, pipeline);
      }
      const encoder = d.createCommandEncoder({label: 'f3d-storage-mips'});
      for (let level = 1; level < entry.shape.mipLevelCount; level++) {
        const group = d.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [{binding: 0,
          resource: entry.native.createView({baseMipLevel: level - 1, mipLevelCount: 1})}]});
        const pass = encoder.beginRenderPass({colorAttachments: [{view: entry.native.createView({baseMipLevel: level, mipLevelCount: 1}), loadOp: 'clear', storeOp: 'store', clearValue: [0,0,0,0]}]});
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
      }
      d.queue.submit([encoder.finish()]);
    });
    entry.mipsDirty = false;
  }
  function resolve(snapshot) {
    const entry = ensure(snapshot.entry);
    if (snapshot.kind === 'sampler') {
      const key = JSON.stringify(snapshot.sampler);
      if (!entry.samplers.has(key)) {
        if (entry.samplers.size >= 64) entry.samplers.clear();
        entry.samplers.set(key, checked(d => d.createSampler(snapshot.sampler)));
      }
      return entry.samplers.get(key);
    }
    if (snapshot.kind === 'sampled') { refreshMips(entry); return entry.view; }
    if (!entry.views.has(snapshot.mipLevel)) entry.views.set(snapshot.mipLevel,
      checked(() => entry.native.createView({baseMipLevel: snapshot.mipLevel, mipLevelCount: 1})));
    return entry.views.get(snapshot.mipLevel);
  }
  function validateDispatch(snapshots) {
    const uses = new Map();
    for (const s of snapshots) {
      assertCurrent(s.entry);
      if (s.kind === 'sampler') continue;
      const list = uses.get(s.entry) ?? []; list.push(s); uses.set(s.entry, list);
    }
    for (const list of uses.values()) for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      const a = list[i], b = list[j];
      const writes = s => s.kind === 'storage' && s.access !== 'read-only';
      const overlap = a.kind === 'sampled' || b.kind === 'sampled' || a.mipLevel === b.mipLevel;
      // Overlapping storage bindings with the same access are legal; mixing a
      // sampled read or read-only storage with a writer is not a usage scope.
      if (overlap && ((writes(a) && !writes(b)) || (!writes(a) && writes(b))))
        fail('FEEDBACK', 'A dispatch cannot sample/read a subresource also bound for storage writing; use separate textures or a read-write storage binding');
    }
  }
  function written(snapshots) {
    const dirty = new Set();
    for (const s of snapshots) if (s.kind === 'storage' && s.access !== 'read-only' && s.mipLevel === 0 && s.autoMips) {
      s.entry.mipsDirty = true; dirty.add(s.entry);
    }
    // Source auto-update occurs after each dispatch, before a later kernel can
    // load a generated level or manually overwrite one of its mips.
    for (const entry of dirty) refreshMips(entry);
  }
  function bindingsFor(retained = new Map()) {
    return new class extends Map {
      has(texture) { return texture?.isStorageTexture === true || retained.has(texture); }
      get(texture) {
        if (!texture?.isStorageTexture) return retained.get(texture);
        const sample = captureBinding({texture, isSampledTexture: true}), sampler = captureBinding({texture, isSampler: true});
        return {view: resolve(sample), sampler: resolve(sampler), sampleType: layout(sample).texture.sampleType,
          viewDimension: '2d', version: texture.version, sourceVersion: texture.source.version};
      }
    }();
  }
  async function whenIdle() {
    live(); if (!watchedDevice) return;
    let abort;
    const cancelled = new Promise((_, reject) => { abort = () => reject(terminal); stopped.signal.addEventListener('abort', abort, {once: true}); });
    try { await Promise.race([Promise.all([...pending, watchedDevice.queue.onSubmittedWorkDone()]), cancelled]); live(); }
    finally { stopped.signal.removeEventListener('abort', abort); }
  }
  return {capture, assertCurrent, ensure, captureBinding, layout, resolve, validateDispatch, written, bindingsFor, whenIdle,
    get allocatedBytes() { return allocatedBytes; }, get size() { return entries.size; },
    dispose: () => stop(new ThreeStorageTextureError('DISPOSED', 'Storage texture owner disposed'))};
}
