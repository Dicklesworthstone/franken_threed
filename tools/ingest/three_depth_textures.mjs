/** Detached source DepthTexture inputs, including the initial TRAA history.
 * Source Textures.updateTexture allocates depth storage even at version zero.
 * No CPU upload, source-object mutation or invented render target is required.
 * Each captured version remains zero initialized and resident until its queued
 * readers and submitted-work fence release it; dispose invalidates queued uses.
 */
export class ThreeDepthTextureError extends Error {
  constructor(code, message) {
    super(`THREE_DEPTH_TEXTURE_${code}: ${message}`);
    this.name = 'ThreeDepthTextureError'; this.code = `THREE_DEPTH_TEXTURE_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeDepthTextureError(code, message); };
const positive = value => Number.isSafeInteger(value) && value > 0;

export function createThreeDepthTextures({three, device, signal = null, maxTextures = 256,
  maxBytes = 64 * 1024 * 1024, maxBindings = 4096, maxSamplers = 256, maxPending = 1024} = {}) {
  if (typeof three?.DepthTexture !== 'function' || !device?.queue ||
      ![maxTextures, maxBytes, maxBindings, maxSamplers, maxPending].every(positive))
    fail('OPTIONS', 'Expected pinned depth textures, a GPUDevice and positive resource limits');
  const current = new Map(), entries = new Set(), watches = new Map(), leases = new Map(), tokens = new WeakSet();
  const samplers = new Map(), pending = new Set();
  let bytes = 0, active = null, terminal = null, disposed = false, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const live = () => { if (disposed) fail('DISPOSED', 'Depth texture owner is disposed'); if (terminal) throw terminal; };
  function collect(entry) {
    if (entry.alive || entry.borrowed || entry.inFlight || !entries.delete(entry)) return;
    entry.native?.destroy(); entry.native = null; entry.view = null; bytes -= entry.shape.bytes;
  }
  function discard(entry) {
    entry.alive = false;
    if (current.get(entry.texture) === entry) current.delete(entry.texture);
    collect(entry);
  }
  function stop(error) {
    if (terminal) return;
    terminal = error && (typeof error === 'object' || typeof error === 'function')
      ? error : new ThreeDepthTextureError('ABORTED', String(error ?? 'Renderer lifetime ended'));
    rejectStopped(terminal);
    for (const entry of entries) discard(entry);
    for (const [texture, listener] of watches) texture.removeEventListener('dispose', listener);
    watches.clear(); samplers.clear(); signal?.removeEventListener('abort', abort);
  }
  const abort = () => stop(signal.reason ?? new ThreeDepthTextureError('ABORTED', 'Renderer lifetime ended'));
  if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, {once: true});
  device.lost?.then(info => stop(new ThreeDepthTextureError('LOST', info?.message ?? 'WebGPU device lost')), stop);
  function track(task) {
    pending.add(task); task.catch(stop); task.finally(() => pending.delete(task)).catch(() => {}); return task;
  }
  function checked(run) {
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let result, error;
    try { result = run(); } catch (cause) { error = cause; }
    const scopes = [device.popErrorScope(), device.popErrorScope()];
    track(Promise.all(scopes).then(errors => {
      if (error) throw error;
      const invalid = errors.find(Boolean);
      if (invalid) throw new ThreeDepthTextureError('GPU', invalid.message ?? 'Depth texture allocation or binding failed');
    }));
    if (error) { stop(error); throw error; }
    return result;
  }
  function watch(texture) {
    if (watches.has(texture)) return;
    const listener = () => {
      for (const entry of entries) if (entry.texture === texture) discard(entry);
      texture.removeEventListener('dispose', listener); watches.delete(texture);
    };
    texture.addEventListener('dispose', listener); watches.set(texture, listener);
  }
  function inspect(texture) {
    live();
    if (!(texture instanceof three.DepthTexture) || Object.getPrototypeOf(texture) !== three.DepthTexture.prototype ||
        texture.isDepthTexture !== true || texture.renderTarget != null || texture.isCubeTexture ||
        texture.isArrayTexture || texture.is3DTexture || texture.isRenderTargetTexture || texture.isStorageTexture)
      fail('SOURCE', 'Expected an ordinary detached source 2D DepthTexture');
    if (texture.format !== three.DepthFormat || texture.internalFormat != null || texture.colorSpace !== three.NoColorSpace ||
        texture.generateMipmaps !== false || texture.mipmaps?.length || texture.flipY !== false || texture.image?.data != null)
      fail('FORMAT', 'Detached depth requires no pixels, mipmaps, color transform or internal format override');
    const format = texture.type === three.UnsignedIntType ? 'depth24plus' : texture.type === three.FloatType ? 'depth32float' : null;
    if (!format) fail('FORMAT', 'Detached depth requires UnsignedIntType or FloatType');
    const width = texture.image?.width || 1, height = texture.image?.height || 1;
    if (!positive(width) || !positive(height) || (texture.image?.depth ?? 1) !== 1 ||
        width > device.limits.maxTextureDimension2D || height > device.limits.maxTextureDimension2D)
      fail('SHAPE', 'Detached depth dimensions must fit one 2D GPU texture');
    const size = width * height * 4;
    if (!Number.isSafeInteger(size)) fail('BUDGET', 'Depth texture byte size exceeds integer precision');
    return Object.freeze({width, height, format, bytes: size});
  }
  function shapeOf(texture) {
    live(); const entry = current.get(texture);
    return entry?.version === texture.version ? entry.shape : inspect(texture);
  }
  function describe(binding) {
    live();
    if (!binding || binding.store || !binding.isSampledTexture && !binding.isSampler)
      fail('BINDING', 'Detached depth textures are read-only sampled inputs');
    const texture = binding.texture, shape = shapeOf(texture);
    const filters = new Map([[three.NearestFilter, 'nearest'], [three.LinearFilter, 'linear']]);
    const wraps = new Map([[three.ClampToEdgeWrapping, 'clamp-to-edge'], [three.RepeatWrapping, 'repeat'], [three.MirroredRepeatWrapping, 'mirror-repeat']]);
    const minFilter = filters.get(texture.minFilter), magFilter = filters.get(texture.magFilter);
    const addressModeU = wraps.get(texture.wrapS), addressModeV = wraps.get(texture.wrapT);
    if (!minFilter || !magFilter || !addressModeU || !addressModeV || texture.anisotropy !== 1)
      fail('SAMPLER', 'Detached depth requires nearest/linear filtering and anisotropy one');
    const comparison = binding.comparison === true;
    const comparisons = new Map([[three.NeverCompare, 'never'], [three.LessCompare, 'less'], [three.EqualCompare, 'equal'],
      [three.LessEqualCompare, 'less-equal'], [three.GreaterCompare, 'greater'], [three.NotEqualCompare, 'not-equal'],
      [three.GreaterEqualCompare, 'greater-equal'], [three.AlwaysCompare, 'always']]);
    if (texture.compareFunction != null && !comparisons.has(texture.compareFunction) || comparison && texture.compareFunction == null)
      fail('SAMPLER', 'Depth comparison requires a source compare function');
    const sampler = Object.freeze({addressModeU, addressModeV, mipmapFilter: 'nearest', maxAnisotropy: 1,
      minFilter: comparison ? minFilter : 'nearest', magFilter: comparison ? magFilter : 'nearest',
      ...(comparison ? {compare: comparisons.get(texture.compareFunction)} : {})});
    const layout = binding.isSampler && !binding.isSampledTexture
      ? {sampler: {type: comparison ? 'comparison' : 'non-filtering'}}
      : {texture: {sampleType: 'depth', viewDimension: '2d', multisampled: false}};
    return {texture, shape, layout, sampler: binding.isSampler && !binding.isSampledTexture ? sampler : null};
  }
  function entryOf(texture, shape) {
    let entry = current.get(texture);
    if (entry?.version === texture.version) return entry;
    if (entries.size >= maxTextures || shape.bytes > maxBytes - bytes)
      fail('BUDGET', 'Detached depth generations exceed texture or byte capacity');
    const next = {texture, version: texture.version, shape, refs: 0, native: null, view: null, alive: true, borrowed: false, inFlight: 0};
    watch(texture); entries.add(next); current.set(texture, next); bytes += shape.bytes;
    if (entry?.refs === 0) discard(entry);
    return next;
  }
  function validate(token) {
    live(); const record = leases.get(token);
    if (!record?.entry.alive) fail('STALE', 'Detached depth was disposed or its captured use was released');
    return record;
  }
  function ensure(entry) {
    if (!entry.native) checked(() => {
      entry.native = device.createTexture({label: 'f3d-detached-depth', size: {width: entry.shape.width, height: entry.shape.height, depthOrArrayLayers: 1},
        dimension: '2d', format: entry.shape.format, sampleCount: 1, mipLevelCount: 1, usage: 1 | 2 | 4 | 16});
    });
  }
  return {
    owns(token) { return tokens.has(token); },
    layoutOf(binding) { return describe(binding).layout; },
    sampleData(texture) { describe({texture, isSampledTexture: true}); return {samples: 1, primarySamples: 1, isMSAA: false}; },
    captureBinding(binding) {
      if (active) fail('REENTRY', 'Cannot capture detached depth during a submitted use');
      const record = describe(binding);
      if (leases.size >= maxBindings) fail('BUDGET', 'Queued detached-depth bindings exceed maxBindings');
      const entry = entryOf(record.texture, record.shape), token = Object.freeze({});
      entry.refs++; leases.set(token, {entry, layout: record.layout, sampler: record.sampler}); tokens.add(token); return token;
    },
    layout(token) { return validate(token).layout; },
    validateDispatch(tokens) { for (const token of tokens) validate(token); },
    withDispatch(tokens, run) {
      live();
      if (active) fail('REENTRY', 'Cannot nest detached-depth submission');
      if (typeof run !== 'function') fail('CALLBACK', 'Expected synchronous submitted use');
      if (!tokens.length) return run();
      const unique = new Set(tokens.map(token => validate(token).entry));
      if (pending.size + unique.size + tokens.length + 1 > maxPending)
        fail('BUDGET', 'Too many detached-depth operations await completion');
      for (const entry of unique) ensure(entry);
      active = unique;
      for (const entry of unique) entry.borrowed = true;
      try {
        const result = run();
        if (result?.then) { Promise.resolve(result).catch(() => {}); fail('ASYNC_USE', 'Submit detached-depth work before returning its borrow'); }
        for (const token of tokens) validate(token);
        return result;
      } finally {
        for (const entry of unique) entry.inFlight++;
        let completion;
        try { completion = Promise.resolve(device.queue.onSubmittedWorkDone()); }
        catch (error) { completion = Promise.reject(error); }
        track(completion.finally(() => {
          for (const entry of unique) { entry.inFlight--; collect(entry); }
        }));
        for (const entry of unique) { entry.borrowed = false; collect(entry); }
        active = null;
      }
    },
    resolve(token) {
      const record = validate(token), entry = record.entry;
      if (!active?.has(entry)) fail('BORROW', 'Resolve detached depth inside its submitted use');
      if (record.sampler) {
        const key = JSON.stringify(record.sampler); let sampler = samplers.get(key);
        if (!sampler) {
          sampler = checked(() => device.createSampler(record.sampler));
          if (samplers.size >= maxSamplers) samplers.delete(samplers.keys().next().value);
        } else samplers.delete(key);
        samplers.set(key, sampler); return sampler;
      }
      return entry.view ??= checked(() => entry.native.createView({dimension: '2d', aspect: 'depth-only', baseMipLevel: 0, mipLevelCount: 1}));
    },
    release(token) {
      const record = leases.get(token); if (!record) return;
      leases.delete(token); const entry = record.entry;
      if (--entry.refs === 0 && current.get(entry.texture) !== entry) discard(entry);
    },
    async whenIdle() {
      live(); if (active) fail('REENTRY', 'Cannot await depth completion inside a submitted use');
      await Promise.race([Promise.all([...pending]), stopped]); live();
    },
    get diagnostics() { return {textures: current.size, generations: entries.size, bindings: leases.size, bytes}; },
    dispose() { if (disposed) return; disposed = true; stop(new ThreeDepthTextureError('DISPOSED', 'Depth texture owner disposed')); },
  };
}
