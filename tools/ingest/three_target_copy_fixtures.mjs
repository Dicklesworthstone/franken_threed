/** Recording contracts, not a WebGPU implementation or a pixel oracle. */
export const box = (x, y, z, width, height, depth) => ({isBox3: true,
  min: {x, y, z}, max: {x: x + width, y: y + height, z: z + depth}});
export const point = (x, y, z) => ({isVector3: true, x, y, z});
export const failure = code => Object.assign(new Error(code), {code});
export function targetCopyFixture() {
  const events = [], commands = [], errors = [], entries = new Map(), validations = [];
  let refs = 0, releases = 0;
  const device = {
    limits: {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048, maxTextureArrayLayers: 256},
    createCommandEncoder() {
      let copy;
      return {copyTextureToTexture(from, to, size) { copy = {from, to, size}; },
        finish() { device.onFinish?.(); return copy; }};
    },
    queue: {submit(batch) { device.onSubmit?.(); events.push('submit'); commands.push(...batch); }},
  };
  const targets = {
    getTextureTarget(texture) { return entries.get(texture)?.target ?? null; },
    capture(target) { return [...entries.values()].find(entry => entry.target === target); },
    retainGeneration(entry) {
      targets.assertCurrent(entry); refs++; let alive = true;
      return () => { if (!alive) throw failure('DOUBLE_RELEASE'); alive = false; refs--; releases++; };
    },
    assertCurrent(entry) {
      if (!entry.alive || entry.target.epoch !== entry.epoch) throw failure('STALE_TARGET');
    },
    validateNativeTextures(sources) {
      events.push('target-preflight');
      for (const source of sources) {
        targets.assertCurrent(source.entry);
        if (source.entry.shape.textures[source.colorIndex] !== source.texture) throw failure('STALE_ATTACHMENT');
        if (source.device !== device || source.depth) throw failure('WRONG_BORROW');
        if (source.entry.shape.layers > device.limits.maxTextureArrayLayers && !source.entry.shape.volume) throw failure('LAYERS');
      }
      if (targets.budgetError) throw targets.budgetError;
    },
    withNativeTexture(source, run) {
      events.push('source-borrow'); targets.assertCurrent(source.entry);
      targets.onBorrow?.(source);
      return run(source.entry.native[source.colorIndex]);
    },
    withNativeColorDestination(destination, run, {validation}) {
      events.push('target-write'); targets.assertCurrent(destination.entry);
      const value = run(destination.entry.native[destination.colorIndex]);
      validation.catch(() => {}); validations.push(validation); return value;
    },
  };
  const storage = {
    capture(texture) {
      if (!texture?.isStorageTexture) throw failure('STORAGE');
      return texture.entry;
    },
    assertCurrent(entry) { if (!entry.alive || entry.texture.shape !== entry.shape) throw failure('STALE_STORAGE'); },
    captureBinding(binding) {
      const {mipLevel} = binding;
      if (!Number.isSafeInteger(mipLevel) || mipLevel < 0 || mipLevel >= binding.texture.shape.mipLevelCount) throw failure('STORAGE_MIP');
      return {...binding};
    },
    validateAllocations(values) {
      events.push('storage-preflight'); values.forEach(storage.assertCurrent);
      if (storage.budgetError) throw storage.budgetError;
    },
    withTexture(entry, run, write, options = {}) {
      events.push(write ? 'storage-write' : 'storage-read'); storage.assertCurrent(entry);
      const result = run(entry.native);
      storage.lastWrite = write;
      if (options.validation) { options.validation.catch(() => {}); validations.push(options.validation); }
      return result;
    },
    get validation() { return storage.completion ?? Promise.resolve(); },
  };
  const renderer = {_device: device, _targets: targets, _storageTextures: storage,
    _targetLifetime: new AbortController(), _pending: [], _preparing: false,
    _enqueue(request) {
      if (renderer.queueError) throw renderer.queueError;
      events.push('enqueue'); renderer._pending.push(request);
    }};
  function target({dimension = '2d', width = 4, height = 4, depth = 1, mips = 3,
    format = 'rgba8unorm', samples = 1, count = 1, pointer = true} = {}) {
    const cube = dimension === 'cube', volume = dimension === '3d', array = dimension === '2d-array';
    const target = {epoch: 1}, textures = Array.from({length: count}, () => ({isTexture: true,
      isRenderTargetTexture: true, renderTarget: pointer ? target : null}));
    const colors = textures.map(() => ({width, height, format, mipLevelCount: mips}));
    const entry = {target, epoch: 1, alive: true, lifetime: new AbortController(),
      shape: {...colors[0], cube, volume, array, layers: cube ? 6 : depth,
        viewDimension: dimension, sampleCount: samples, colors, textures, colorFormats: colors.map(c => c.format)},
      native: textures.map((_, colorIndex) => ({target, colorIndex})),
      validation: Promise.resolve(), storage: {validation: Promise.resolve()}};
    for (const texture of textures) entries.set(texture, entry);
    return {entry, texture: textures[0], textures, dispose() { entry.alive = false; entry.lifetime.abort(failure('TARGET_DISPOSED')); }};
  }
  function storageTexture({dimension = '2d-array', width = 4, height = 4, depth = 6, mips = 3, format = 'rgba8unorm'} = {}) {
    const texture = {isTexture: true, isStorageTexture: true,
      shape: {width, height, depth, mipLevelCount: mips, viewDimension: dimension, format}};
    texture.entry = {texture, shape: texture.shape, alive: true, native: {storage: texture}};
    return texture;
  }
  function flush() {
    for (const request of renderer._pending.splice(0)) {
      try { request.resolve(request.issue()); } catch (error) { request.reject(error); }
    }
  }
  return {renderer, device, targets, storage, events, commands, errors, validations, target, storageTexture, flush,
    rejected: error => errors.push(error), get references() { return refs; }, get releases() { return releases; }};
}
