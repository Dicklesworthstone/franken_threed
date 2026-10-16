/** Read-only CPU data textures at the TSL compute boundary.
 *
 * The existing source texture pool performs format admission and native uploads.
 * Private descriptors (not Three constructors/clones) hold call-time pixel and
 * sampler versions while a renderer FIFO is blocked. No source IDs are allocated,
 * no source arrays are replaced, and no retained renderer submits GPU work.
 * StorageTexture bindings continue to use the renderer's shared storage owner.
 */
export class ThreeComputeTextureError extends Error {
  constructor(code, message) {
    super(`THREE_COMPUTE_TEXTURE_${code}: ${message}`);
    this.name = 'ThreeComputeTextureError'; this.code = `THREE_COMPUTE_TEXTURE_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeComputeTextureError(code, message); };
const positive = n => Number.isSafeInteger(n) && n > 0;

/** poolFactory returns the existing createGpuThreeTextures owner, lazily, after
 * renderer initialization. Its inspect/prepare/binding/retain contract is shared
 * with scene textures; this adapter does not implement another texture uploader.
 * The storage owner is borrowed and is not disposed by this adapter.
 */
export function createThreeComputeTextures({storage, poolFactory, targetFactory = null,
  maxTextures = 256, maxSnapshotBytes = 128 * 1024 * 1024} = {}) {
  if (!storage || typeof poolFactory !== 'function' || !positive(maxTextures) || !positive(maxSnapshotBytes))
    fail('OPTIONS', 'Expected storage, source texture pool and positive texture budgets');
  const current = new Map(), records = new Set(), watches = new Map(), sources = new Map();
  const leases = new WeakSet();
  let targetInputs = null;
  const targets = () => {
    if (typeof targetFactory !== 'function') fail('SOURCE', 'Native render-target inputs require the renderer target owner');
    return targetInputs ??= targetFactory();
  };
  const isTarget = binding => binding?.texture?.isRenderTargetTexture === true ||
    binding?.texture?.isDepthTexture === true && binding.texture.renderTarget != null;
  let pool = null, disposed = false, snapshotBytes = 0, resolving = false, poolDirty = false;
  const live = () => { if (disposed) fail('DISPOSED', 'Compute texture inputs are disposed'); };
  const sourcePool = () => { live(); return pool ??= poolFactory(); };
  const isStorage = binding => binding?.texture?.isStorageTexture === true;
  function describe(binding) {
    live();
    const t = binding?.texture;
    if (!t || !(t.isDataTexture || t.isDataArrayTexture || t.isData3DTexture || t.isCompressedTexture || t.isCubeTexture) ||
        t.isRenderTargetTexture || t.isVideoTexture ||
        t.isCubeTexture && (!Array.isArray(t.image) || t.image.length !== 6))
      fail('SOURCE', 'Compute sampled inputs require ready 2D/array/3D data textures or CPU-backed cube/compressed textures');
    if (binding.store) fail('ACCESS', 'CPU data textures are read-only compute inputs; use StorageTexture for writes');
    if (!binding.isSampler && !binding.isSampledTexture) fail('BINDING', 'Expected a source texture or sampler binding');
    const shape = sourcePool().inspect(t);
    if (!shape.data) fail('SOURCE', 'Expected CPU pixel or compressed-block data');
    if (t.updateRanges?.length || t.layerUpdates?.size)
      fail('RANGES', 'Partial compute texture updates require an existing GPU-visible snapshot');
    return shape;
  }
  function inputLayout(binding, shape) {
    // SampledTexture also has isSampler=true in r186.
    if (binding.isSampler && !binding.isSampledTexture)
      return {sampler: {type: shape.sampleType === 'float' ? 'filtering' : 'non-filtering'}};
    return {texture: {sampleType: shape.sampleType, viewDimension: shape.viewDimension, multisampled: false}};
  }
  function layoutOf(binding) {
    if (isStorage(binding)) return storage.layout(storage.captureBinding(binding));
    if (isTarget(binding)) return targets().layoutOf(binding);
    return inputLayout(binding, describe(binding));
  }
  function syncPool() {
    if (resolving) { poolDirty = true; return; }
    poolDirty = false;
    if (pool) { if (disposed) pool.dispose(); else pool.retain([...records].map(record => record.texture)); }
  }
  function retire(record) {
    if (!record.alive) return;
    record.alive = false; records.delete(record);
    if (--record.pixels.refs === 0) {
      snapshotBytes -= record.pixels.bytes;
      const variants = sources.get(record.source);
      variants.delete(record.pixels.key);
      if (!variants.size) sources.delete(record.source);
    }
    // A stale queued lease may survive until a blocked FIFO drains. It must
    // not keep retired CPU copies alive outside the snapshot byte budget.
    record.texture = null; record.binding = null; record.pixels = null;
    if (current.get(record.original) === record) current.delete(record.original);
    if (![...records].some(other => other.original === record.original)) {
      const listener = watches.get(record.original);
      if (listener) record.original.removeEventListener('dispose', listener);
      watches.delete(record.original);
    }
  }
  function watch(original) {
    if (watches.has(original)) return;
    const listener = () => {
      // A texture's dispose event invalidates queued uses, not just future ones.
      for (const record of records) if (record.original === original) retire(record);
      original.removeEventListener('dispose', listener); watches.delete(original);
      syncPool();
    };
    original.addEventListener('dispose', listener); watches.set(original, listener);
  }
  function privateDescriptor(original) {
    // No Texture.clone()/Source constructors: those advance public UUID/ID
    // state. Private listener and upload state never aliases the application.
    const descriptors = Object.getOwnPropertyDescriptors(original);
    for (const key of ['source', 'image', 'mipmaps', 'updateRanges', 'layerUpdates', '_listeners', 'onUpdate']) delete descriptors[key];
    const texture = Object.create(Object.getPrototypeOf(original), descriptors);
    texture.updateRanges = []; texture.onUpdate = null;
    if (original.layerUpdates instanceof Set) texture.layerUpdates = new Set();
    return texture;
  }
  function snapshot(original, shape) {
    const source = original.source, version = original.version, sourceVersion = source.version;
    const key = JSON.stringify([sourceVersion, shape.key]);
    let pixels = sources.get(source)?.get(key);
    if (!pixels) {
      // Source aliases share one immutable upload version and one native
      // resource, just as the source pool does. Copy only image/mip records;
      // never traverse arbitrary application objects attached to textures.
      const copies = new Map(), images = new Map(); let bytes = 0;
      const copyData = data => {
        if (data == null) return data;
        if (!ArrayBuffer.isView(data) || data instanceof DataView) fail('SOURCE', 'Expected typed texture data');
        if (copies.has(data)) return copies.get(data);
        if (data.byteLength > maxSnapshotBytes - snapshotBytes - bytes)
          fail('BUDGET', 'Captured compute texture pixels exceed maxSnapshotBytes');
        const copy = new data.constructor(data); copies.set(data, copy); bytes += copy.byteLength;
        return copy;
      };
      function copyImage(image) {
        if (!image) return image;
        if (images.has(image)) return images.get(image);
        if (Array.isArray(image)) {
          const result = []; images.set(image, result);
          for (const face of image) result.push(copyImage(face));
          return result;
        }
        // Uncompressed CubeTexture faces are DataTextures; authored cube mips
        // can themselves be CubeTextures. Compressed faces may be plain mip
        // records or CompressedTextures. Preserve prototypes without public IDs.
        if (image.isTexture || image.isDataTexture || image.isCompressedTexture || image.isCubeTexture) {
          const texture = privateDescriptor(image); images.set(image, texture);
          texture.source = {...image.source, data: copyImage(image.image)};
          texture.mipmaps = image.mipmaps.map(copyImage);
          return texture;
        }
        const result = {...image}; images.set(image, result);
        if ('data' in image) result.data = copyData(image.data);
        if ('image' in image) result.image = copyImage(image.image);
        if (Array.isArray(image.mipmaps)) result.mipmaps = image.mipmaps.map(copyImage);
        return result;
      }
      const image = copyImage(original.image), mipmaps = original.mipmaps.map(copyImage);
      pixels = {key, source: {...source, data: image}, mipmaps, bytes, refs: 0};
    }
    const texture = privateDescriptor(original);
    texture.source = pixels.source; texture.mipmaps = pixels.mipmaps;
    texture.onUpdate = () => {
      // A queued old upload cannot acknowledge a newer source's mip request.
      if (shape.manual && !shape.cube && original.source === source && original.version === version && source.version === sourceVersion)
        original.generateMipmaps = false;
      original.onUpdate?.call(original, original);
    };
    return {original, texture, key: shape.key, source, version, sourceVersion,
      pixels, refs: 0, alive: true, binding: null};
  }
  function captureBinding(binding) {
    live();
    if (isStorage(binding)) return storage.captureBinding(binding);
    if (resolving) fail('REENTRY', 'Cannot capture compute texture inputs from an upload callback');
    if (isTarget(binding)) return targets().captureBinding(binding);
    const shape = describe(binding), original = binding.texture;
    let record = current.get(original);
    if (!record || record.key !== shape.key || record.source !== original.source ||
        record.version !== original.version || record.sourceVersion !== original.source.version) {
      if (record && record.refs === 0) { retire(record); syncPool(); }
      if (records.size >= maxTextures) fail('BUDGET', 'Captured compute texture versions exceed maxTextures');
      const replacement = snapshot(original, shape);
      watch(original);
      const pixels = replacement.pixels;
      if (pixels.refs++ === 0) {
        let variants = sources.get(replacement.source);
        if (!variants) sources.set(replacement.source, variants = new Map());
        variants.set(pixels.key, pixels); snapshotBytes += pixels.bytes;
      }
      records.add(replacement);
      current.set(original, replacement); record = replacement;
    }
    record.refs++;
    const lease = {record, layout: inputLayout(binding, shape), sampler: binding.isSampler && !binding.isSampledTexture, released: false};
    leases.add(lease); return lease;
  }
  function validate(snapshot) {
    if (snapshot.released || !snapshot.record.alive)
      fail('STALE', 'Sampled texture was disposed or its captured use was released');
  }
  return {
    layoutOf, captureBinding,
    sampleData(texture) {
      if (isTarget({texture})) return targets().sampleData(texture);
      layoutOf({texture, isSampledTexture: true});
      return {samples: 1, primarySamples: 1, isMSAA: false};
    },
    withDispatch(snapshots, run) {
      live();
      return targetInputs ? targetInputs.withDispatch(snapshots.filter(s => targetInputs.owns(s)), run) : run();
    },
    layout(snapshot) { live(); if (targetInputs?.owns(snapshot)) return targetInputs.layout(snapshot); return leases.has(snapshot) ? snapshot.layout : storage.layout(snapshot); },
    validateDispatch(snapshots) {
      live();
      const owned = [];
      for (const snapshot of snapshots) {
        if (leases.has(snapshot)) validate(snapshot);
        else if (targetInputs?.owns(snapshot)) targetInputs.validateDispatch([snapshot]);
        else owned.push(snapshot);
      }
      storage.validateDispatch(owned);
    },
    resolve(snapshot) {
      live();
      if (targetInputs?.owns(snapshot)) return targetInputs.resolve(snapshot);
      if (!leases.has(snapshot)) return storage.resolve(snapshot);
      validate(snapshot);
      if (resolving) fail('REENTRY', 'Cannot resolve compute textures from an upload callback');
      const record = snapshot.record;
      if (!record.binding) {
        resolving = true;
        try {
          sourcePool().prepare([record.texture]);
          validate(snapshot);
          record.binding = sourcePool().binding(record.texture);
        } finally { resolving = false; if (poolDirty) syncPool(); }
      }
      return snapshot.sampler ? record.binding.sampler : record.binding.view;
    },
    written(snapshots) { live(); storage.written(snapshots.filter(snapshot => !leases.has(snapshot) && !targetInputs?.owns(snapshot))); },
    release(snapshot) {
      if (targetInputs?.owns(snapshot)) { targetInputs.release(snapshot); return; }
      if (!leases.has(snapshot) || snapshot.released) return;
      snapshot.released = true;
      const record = snapshot.record;
      if (--record.refs === 0 && current.get(record.original) !== record) { retire(record); syncPool(); }
    },
    async whenIdle() { live(); await Promise.all([storage.whenIdle(), pool?.whenIdle(), targetInputs?.whenIdle()]); live(); },
    get diagnostics() { return {textures: current.size, versions: records.size, snapshotBytes}; },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const [original, listener] of watches) original.removeEventListener('dispose', listener);
      watches.clear();
      for (const record of records) retire(record);
      current.clear(); syncPool(); targetInputs?.dispose();
    },
  };
}
