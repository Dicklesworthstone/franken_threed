/** Read-only data, decoded images and canvas textures at the TSL compute boundary.
 *
 * The existing source texture pool performs format admission and native uploads.
 * Private descriptors (not Three constructors/clones) hold call-time pixel and
 * sampler versions while a renderer FIFO is blocked. No source IDs are allocated,
 * no source arrays are replaced, and no retained renderer submits GPU work.
 * Partial rows/layers compose a complete immutable image from the preceding
 * captured source version (or zero-initialized storage), not unmarked CPU edits.
 * ImageData/ImageBitmap inputs use synchronous structured-clone snapshots.
 * Mutable images/canvases start immutable ImageBitmap capture synchronously
 * and await only bitmap readiness in the FIFO, retaining native color metadata.
 * Only private images are retired; application images are never transferred.
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
const instance = (value, name) => typeof globalThis[name] === 'function' && value instanceof globalThis[name];
const decodedImage = image => instance(image, 'ImageData') || instance(image, 'ImageBitmap');
const mutableImage = image => instance(image, 'HTMLImageElement') || instance(image, 'HTMLCanvasElement') || instance(image, 'OffscreenCanvas');

/** poolFactory returns the existing createGpuThreeTextures owner, lazily, after
 * renderer initialization. Its inspect/prepare/binding/retain contract is shared
 * with scene textures; this adapter does not implement another texture uploader.
 * The storage owner is borrowed and is not disposed by this adapter.
 */
export function createThreeComputeTextures({storage, poolFactory, targetFactory = null, framebufferFactory = null,
  maxTextures = 256, maxSnapshotBytes = 128 * 1024 * 1024} = {}) {
  if (!storage || typeof poolFactory !== 'function' || !positive(maxTextures) || !positive(maxSnapshotBytes))
    fail('OPTIONS', 'Expected storage, source texture pool and positive texture budgets');
  const current = new Map(), records = new Set(), watches = new Map(), sources = new Map(), histories = new Map();
  const leases = new WeakSet(), imageSnapshots = new WeakMap();
  let captureState = null;
  let targetInputs = null;
  let framebufferInputs = null;
  const framebuffers = () => {
    if (typeof framebufferFactory !== 'function') fail('SOURCE', 'Framebuffer inputs require the renderer framebuffer owner');
    return framebufferInputs ??= framebufferFactory();
  };
  const isFramebuffer = binding => binding?.texture?.isFramebufferTexture === true;
  const targets = () => {
    if (typeof targetFactory !== 'function') fail('SOURCE', 'Native render-target inputs require the renderer target owner');
    return targetInputs ??= targetFactory();
  };
  const isTarget = binding => binding?.texture?.isRenderTargetTexture === true ||
    binding?.texture?.isDepthTexture === true && binding.texture.renderTarget != null;
  let pool = null, disposed = false, snapshotBytes = 0, resolving = false, poolDirty = false;
  const live = () => { if (disposed) fail('DISPOSED', 'Compute texture inputs are disposed'); };
  const sourcePool = () => { live(); return pool ??= poolFactory({imageSnapshots}); };
  const isStorage = binding => binding?.texture?.isStorageTexture === true;
  function describe(binding, {initialize = false} = {}) {
    live();
    const t = binding?.texture;
    if (!t || !(t.isTexture || t.isDataTexture || t.isDataArrayTexture || t.isData3DTexture || t.isCompressedTexture || t.isCubeTexture) ||
        t.isRenderTargetTexture || t.isVideoTexture ||
        t.isCubeTexture && (!Array.isArray(t.image) || t.image.length !== 6))
      fail('SOURCE', 'Compute sampled inputs require ready source data, decoded images or canvases');
    if (binding.store) fail('ACCESS', 'Source textures are read-only compute inputs; use StorageTexture for writes');
    if (!binding.isSampler && !binding.isSampledTexture) fail('BINDING', 'Expected a source texture or sampler binding');
    const shape = sourcePool().inspect(t, {initialize});
    if (!shape.data && (!Array.isArray(shape.images) || !shape.images.length || !shape.images.every(image => decodedImage(image) || mutableImage(image))))
      fail('SOURCE', 'Compute image inputs require ImageData, ImageBitmap, HTML images or canvases');
    updatePlan(t, shape);
    return shape;
  }
  // Mirror the source pool's admitted physical upload spans. The pool still
  // validates formats, row extents and device limits and uploads the completed
  // private image. Keeping the deltas on the CPU is necessary for queued older
  // versions; applying them to one mutable native texture would lose FIFO state.
  function updatePlan(texture, shape) {
    const ranges = (texture.updateRanges ?? []).map(range => ({...range}));
    const layers = [...texture.layerUpdates ?? []];
    if (!ranges.length && !layers.length) return null;
    if (shape.blockUploads) {
      if (ranges.length || !shape.array || !positive(shape.layers) || !positive(shape.levels) ||
          !Array.isArray(shape.uploads) || shape.uploads.length !== shape.levels)
        fail('RANGES', 'Compressed partial updates require an admitted array mip chain');
      for (const layer of layers) if (!Number.isSafeInteger(layer) || layer < 0 || layer >= shape.layers)
        fail('RANGES', 'Updated compressed layer must belong to the source array');
      const plans = new Map();
      for (const [level, upload] of shape.uploads.entries()) {
        const image = texture.mipmaps[level], data = image?.data;
        if (upload.image !== image || upload.level !== level || upload.layer !== 0 || upload.depth !== shape.layers ||
            !positive(upload.sliceBytes) || !(data instanceof Uint8Array) ||
            upload.sliceBytes * (shape.layers - 1) >= data.byteLength)
          fail('RANGES', 'Compressed mip metadata does not describe every source layer');
        // sliceBytes comes from the native inspector's block geometry, not
        // texel dimensions or UNPACK_ALIGNMENT. RGBA fallback mips may omit
        // unused trailing padding on the last layer's last row.
        plans.set(image, {level, spans: layers.map(layer => {
          const offset = layer * upload.sliceBytes;
          return [offset, Math.min(upload.sliceBytes, data.byteLength - offset)];
        })});
      }
      return plans;
    }
    const {width, height, channels, bpc, alignment} = shape;
    if (![width, height, channels, bpc, alignment].every(positive) ||
        shape.cube || shape.volume || shape.manual || shape.blockUploads ||
        !ArrayBuffer.isView(texture.image?.data) || texture.image.data instanceof DataView)
      fail('RANGES', 'Partial snapshots require admitted base-level row or array-layer data');
    const rowBytes = width * channels * bpc, pitch = Math.ceil(rowBytes / alignment) * alignment;
    const spans = [];
    if (ranges.length) {
      if (layers.length || shape.layered || shape.integral || channels !== 4 || bpc !== 1 || shape.flipY)
        fail('RANGES', 'Partial rows require unflipped RGBA byte base pixels');
      for (const {start, count} of ranges) {
        if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count < 0 ||
            start + count > width * height * 4 || Math.floor(start / 4) % width + Math.ceil(count / 4) > width)
          fail('RANGES', 'Partial updates must fit one source pixel row');
      }
      ranges.sort((a, b) => a.start - b.start);
      const row = i => Math.floor(Math.floor(i / 4) / width), merged = [];
      for (const next of ranges) {
        const last = merged.at(-1);
        // Preserve r186's one-component adjacency and pixel rounding. In
        // particular, start/count need not be multiples of four components.
        if (last && next.start <= last.start + last.count + 1 && row(next.start) === row(last.start) &&
            row(next.start + next.count - 1) === row(next.start))
          last.count = Math.max(last.count, next.start + next.count - last.start);
        else merged.push({...next});
      }
      for (const {start, count} of merged) {
        const pixel = Math.floor(start / 4), offset = Math.floor(pixel / width) * pitch + pixel % width * 4;
        if (count) spans.push([offset, Math.ceil(count / 4) * 4]);
      }
    } else {
      if (!shape.array || !positive(shape.layers) || pitch !== rowBytes)
        fail('RANGES', 'Partial array layers require tightly packed source rows');
      const slice = pitch * height;
      for (const layer of layers) {
        if (!Number.isSafeInteger(layer) || layer < 0 || layer >= shape.layers)
          fail('RANGES', 'Updated layer must belong to the source array');
        spans.push([layer * slice, slice]);
      }
    }
    if (spans.some(([offset, size]) => !Number.isSafeInteger(offset + size) || offset + size > texture.image.data.byteLength))
      fail('RANGES', 'Partial upload span exceeds the source data');
    return new Map([[texture.image, {level: null, spans}]]);
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
    if (isFramebuffer(binding)) return framebuffers().layoutOf(binding);
    // Shader preparation needs storage metadata before the source requests its
    // first upload (BatchedMesh prepares its indirect IDs at onBeforeRender).
    // Capture still validates readiness and owns only the requested versions.
    return inputLayout(binding, describe(binding, {initialize: true}));
  }
  function syncPool() {
    if (resolving) { poolDirty = true; return; }
    poolDirty = false;
    if (pool) { if (disposed) pool.dispose(); else pool.retain([...records].map(record => record.texture)); }
  }
  function retainPixels(source, pixels) {
    if (pixels.refs++ !== 0) return;
    let variants = sources.get(source);
    if (!variants) sources.set(source, variants = new Map());
    variants.set(pixels.key, pixels); snapshotBytes += pixels.bytes;
  }
  function releasePixels(source, pixels) {
    if (--pixels.refs !== 0) return;
    pixels.disposeImages?.();
    snapshotBytes -= pixels.bytes;
    const variants = sources.get(source);
    variants.delete(pixels.key);
    if (!variants.size) sources.delete(source);
  }
  function setHistory(source, key, pixels) {
    let group = histories.get(source);
    if (!group) histories.set(source, group = new Map());
    const previous = group.get(key);
    if (previous === pixels) return;
    retainPixels(source, pixels); group.set(key, pixels);
    if (previous) releasePixels(source, previous);
  }
  function pruneHistories() {
    for (const [source, group] of histories) {
      for (const [key, pixels] of group) if (![...records].some(record => record.source === source && record.key === key)) {
        group.delete(key); releasePixels(source, pixels);
      }
      if (!group.size) histories.delete(source);
    }
  }
  function retire(record) {
    if (!record.alive) return;
    record.alive = false; records.delete(record);
    releasePixels(record.source, record.pixels);
    pruneHistories();
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
  function snapshot(original, shape, existingPixels = null) {
    const texture = privateDescriptor(original);
    const updates = updatePlan(original, shape), imageAtCapture = original.image;
    const rangesAtCapture = original.updateRanges, rangeValues = (rangesAtCapture ?? []).map(r => ({ref: r, start: r.start, count: r.count}));
    const layersAtCapture = original.layerUpdates, layerValues = [...layersAtCapture ?? []];
    const dataAtCapture = imageAtCapture?.data, mipsAtCapture = original.mipmaps;
    const mipValues = shape.blockUploads ? mipsAtCapture.map(image => ({image, data: image.data})) : null;
    const source = original.source, version = original.version, sourceVersion = existingPixels?.source.version ?? source.version;
    const key = JSON.stringify([sourceVersion, shape.key]);
    let pixels = existingPixels ?? sources.get(source)?.get(key);
    if (!pixels) {
      // Residency is keyed by the same complete descriptor as the native pool.
      // A new source/shape/sampler starts from zero, just like a fresh GPUTexture.
      const previousPixels = updates ? histories.get(source)?.get(shape.key) : null;
      if (updates && shape.blockUploads && !previousPixels)
        fail('RANGES', 'Upload a complete compressed mip chain before partial layer updates');
      // Source aliases share one immutable upload version and one native
      // resource, just as the source pool does. Copy only image/mip records;
      // never traverse arbitrary application objects attached to textures.
      const copies = new Map(), images = new Map(), bitmaps = new Set(), pendingImages = [];
      let bytes = 0, retired = false;
      const disposeImages = () => {
        retired = true;
        for (const bitmap of bitmaps) bitmap.close(); bitmaps.clear();
      };
      const copyData = (data, image) => {
        if (data == null) return data;
        if (!ArrayBuffer.isView(data) || data instanceof DataView) fail('SOURCE', 'Expected typed texture data');
        const partial = updates?.get(image), copyKey = partial ?? data;
        if (copies.has(copyKey)) return copies.get(copyKey);
        if (data.byteLength > maxSnapshotBytes - snapshotBytes - bytes)
          fail('BUDGET', 'Captured compute texture pixels exceed maxSnapshotBytes');
        let copy;
        if (partial) {
          const previous = partial.level === null ? previousPixels?.source.data?.data : previousPixels?.mipmaps[partial.level]?.data;
          copy = new data.constructor(data.length);
          const destination = new Uint8Array(copy.buffer, copy.byteOffset, copy.byteLength);
          if (previous) {
            if (previous.constructor !== data.constructor) fail('RANGES', 'Partial snapshot component type changed');
            destination.set(new Uint8Array(previous.buffer, previous.byteOffset, Math.min(previous.byteLength, copy.byteLength)));
          }
          const incoming = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          for (const [offset, size] of partial.spans) destination.set(incoming.subarray(offset, offset + size), offset);
        } else copy = new data.constructor(data);
        copies.set(copyKey, copy); bytes += copy.byteLength;
        return copy;
      };
      function copyImage(image) {
        if (!image) return image;
        if (images.has(image)) return images.get(image);
        if (mutableImage(image)) {
          const width = image.naturalWidth ?? image.width, height = image.naturalHeight ?? image.height;
          const htmlImage = instance(image, 'HTMLImageElement');
          const size = width * height * 4;
          if (!positive(width) || !positive(height) || !positive(size) || size > maxSnapshotBytes - snapshotBytes - bytes)
            fail('BUDGET', 'Captured compute image pixels exceed maxSnapshotBytes');
          if (typeof globalThis.createImageBitmap !== 'function')
            fail('SOURCE', 'Mutable compute images require createImageBitmap capture');
          // The HTML algorithm copies pixels at invocation, then queues only
          // promise resolution. Decoded images need unpremultiplied capture to
          // preserve low/zero-alpha RGB. Canvases retain their native premultiplied
          // representation and color metadata: forcing an unpremultiplied bitmap
          // or rasterizing into a different gamut adds an observable conversion.
          const captured = {width, height};
          const pending = Promise.resolve(globalThis.createImageBitmap(image,
            {premultiplyAlpha: htmlImage ? 'none' : 'default', colorSpaceConversion: 'default'})).then(bitmap => {
            if (!instance(bitmap, 'ImageBitmap') || bitmap.width !== width || bitmap.height !== height) {
              if (instance(bitmap, 'ImageBitmap')) bitmap.close();
              fail('SOURCE', 'The host did not capture the intrinsic image pixels');
            }
            if (retired) { bitmap.close(); return; }
            bitmaps.add(bitmap); imageSnapshots.set(captured, bitmap);
          });
          // No caller context is queried and no source canvas is resized or
          // transferred. Browser origin checks apply to capture and upload.
          // Disposal or rejected FIFO admission can precede bitmap delivery.
          pending.catch(() => {}); pendingImages.push(pending);
          images.set(image, captured); bytes += size;
          return captured;
        }
        if (decodedImage(image)) {
          // ImageBitmap and ImageData have platform internal slots; spreading
          // them produces an invalid external-copy source. Structured cloning
          // copies the pixels synchronously, without transfer, canvas rasterizing,
          // a decoder task, source constructors or image ownership changes.
          // Opaque bitmaps are charged their logical RGBA8 pixel footprint, as
          // in the source texture pool, not an estimate of browser process RAM.
          const size = instance(image, 'ImageData') ? image.data.byteLength : image.width * image.height * 4;
          if (!positive(size) || size > maxSnapshotBytes - snapshotBytes - bytes)
            fail('BUDGET', 'Captured compute image pixels exceed maxSnapshotBytes');
          if (typeof globalThis.structuredClone !== 'function')
            fail('SOURCE', 'Decoded compute inputs require synchronous structuredClone support');
          const copy = globalThis.structuredClone(image);
          if (copy !== image && instance(copy, 'ImageBitmap')) bitmaps.add(copy);
          if (copy === image || !decodedImage(copy) || copy.width !== image.width || copy.height !== image.height)
            fail('SOURCE', 'The host did not produce an independent decoded image snapshot');
          images.set(image, copy); bytes += size;
          return copy;
        }
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
        if ('data' in image) result.data = copyData(image.data, image);
        if ('image' in image) result.image = copyImage(image.image);
        if (Array.isArray(image.mipmaps)) result.mipmaps = image.mipmaps.map(copyImage);
        return result;
      }
      try {
        const image = copyImage(original.image), mipmaps = original.mipmaps.map(copyImage);
        const ready = pendingImages.length ? Promise.all(pendingImages).then(() => {}) : null;
        ready?.catch(() => {});
        pixels = {key, descriptorKey: shape.key, source: {...source, data: image}, mipmaps, bytes, refs: 0,
          disposeImages, ready, readyState: ready ? 'pending' : 'ready', readyError: null};
        if (ready) ready.then(() => { pixels.readyState = 'ready'; }, error => {
          pixels.readyState = 'failed'; pixels.readyError = error;
        });
      } catch (error) { disposeImages(); throw error; }
    }
    let record;
    texture.source = pixels.source; texture.mipmaps = pixels.mipmaps;
    texture.onUpdate = () => {
      // A source history can outlive the alias that first uploaded it. Recreate
      // native residency as needed without replaying that version's callback.
      if (pixels.uploaded) return;
      pixels.uploaded = true;
      captureState?.uploaded.add(record);
      // Acknowledging an older queued upload must not erase a newer request,
      // including a request mutated without a version bump inside user code.
      if (original.source === source && original.version === version && source.version === sourceVersion &&
          original.image === imageAtCapture && original.image?.data === dataAtCapture &&
          (!mipValues || original.mipmaps === mipsAtCapture && original.mipmaps.length === mipValues.length &&
            mipValues.every((value, i) => original.mipmaps[i] === value.image && value.image.data === value.data))) {
        if (shape.manual && !shape.cube) original.generateMipmaps = false;
        if (updates && original.updateRanges === rangesAtCapture && rangesAtCapture?.length === rangeValues.length &&
            rangeValues.every((r, i) => rangesAtCapture[i] === r.ref && r.ref.start === r.start && r.ref.count === r.count))
          original.clearUpdateRanges?.();
        if (updates && original.layerUpdates === layersAtCapture && layersAtCapture?.size === layerValues.length &&
            layerValues.every(layer => layersAtCapture.has(layer))) original.clearLayerUpdates?.();
      }
      original.onUpdate?.call(original, original);
    };
    return record = {original, texture, key: shape.key, source, version, sourceVersion,
      pixels, refs: 0, alive: true, binding: null};
  }
  function captureBinding(binding) {
    live();
    if (isStorage(binding)) return storage.captureBinding(binding);
    if (resolving) fail('REENTRY', 'Cannot capture compute texture inputs from an upload callback');
    if (isTarget(binding)) return targets().captureBinding(binding);
    if (isFramebuffer(binding)) return framebuffers().captureBinding(binding);
    const shape = describe(binding), original = binding.texture;
    let record = current.get(original);
    const unchanged = record && record.key === shape.key && record.source === original.source &&
      record.version === original.version;
    // An unused alias can increment Source.version without requesting an upload
    // through this texture. Only an already captured alias payload changes the
    // shared residency; a later uncaptured version must not hide that payload.
    const sharedPixels = unchanged ? histories.get(original.source)?.get(shape.key) : null;
    if (!unchanged || sharedPixels && sharedPixels !== record.pixels) {
      if (records.size >= maxTextures) fail('BUDGET', 'Captured compute texture versions exceed maxTextures');
      const replacement = snapshot(original, shape, sharedPixels);
      const pixels = replacement.pixels;
      try { watch(original); }
      catch (error) { if (pixels.refs === 0) pixels.disposeImages?.(); throw error; }
      retainPixels(replacement.source, pixels);
      records.add(replacement);
      setHistory(replacement.source, replacement.key, pixels);
      current.set(original, replacement);
      if (record && record.refs === 0 && !captureState?.records.has(record)) { retire(record); syncPool(); }
      record = replacement;
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
    prepareFramebuffer(texture, type) { live(); framebuffers().prepareFramebuffer(texture, type); },
    captureFramebuffer(texture, options) { live(); return framebuffers().captureFramebuffer(texture, options); },
    encodeFramebufferCopy(snapshot, encoder, source) { live(); return framebuffers().encodeFramebufferCopy(snapshot, encoder, source); },
    preparation(snapshots) {
      const pending = new Set();
      for (const snapshot of snapshots) if (leases.has(snapshot)) {
        validate(snapshot);
        const pixels = snapshot.record.pixels;
        if (pixels.readyState === 'failed') throw pixels.readyError;
        if (pixels.readyState === 'pending') pending.add(pixels.ready);
      }
      if (!pending.size) return null;
      const ready = Promise.all(pending).then(() => {});
      // FIFO admission can fail, or an earlier operation can keep this request
      // waiting after a bitmap rejects. Observe it now without swallowing the
      // failure returned to the eventual submission.
      ready.catch(() => {});
      return ready;
    },
    // Public compute validates the whole request and reserves its FIFO slot
    // inside this synchronous transaction. Rejected requests must not become
    // the baseline for a later partial upload. Existing versions remain charged
    // until commit/rollback; no temporary copy is exempt from the byte budget.
    captureDispatch(run) {
      live();
      if (captureState || resolving) fail('REENTRY', 'Cannot nest a compute texture capture transaction');
      if (typeof run !== 'function') fail('OPTIONS', 'Expected a synchronous capture callback');
      const before = {current: new Map(current), records: new Set(records), uploaded: new Set(),
        histories: new Map([...histories].map(([source, group]) => [source, new Map(group)]))};
      // Pin history-only payloads too: disposing an uploading alias must not
      // make rollback resurrect an uncharged or already-retired baseline.
      for (const [source, group] of before.histories) for (const pixels of group.values()) retainPixels(source, pixels);
      captureState = before;
      try {
        const result = run();
        if (result?.then) {
          Promise.resolve(result).catch(() => {});
          fail('ASYNC_USE', 'Compute texture capture cannot cross an await');
        }
        return result;
      } catch (error) {
        // Admission is reversible; a completed native upload is not. In
        // particular onUpdate may throw AFTER the pool acknowledges its source.
        // Preserve those versions without replaying callbacks on a later retry.
        const uploaded = [...before.uploaded].filter(record => record.alive);
        current.clear();
        for (const [original, record] of before.current) if (record.alive) current.set(original, record);
        for (const record of uploaded) current.set(record.original, record);
        for (const record of records) if (!before.records.has(record) && !before.uploaded.has(record)) retire(record);
        for (const [source, group] of histories) for (const pixels of group.values()) releasePixels(source, pixels);
        histories.clear();
        for (const [source, group] of before.histories) for (const [key, pixels] of group)
          if ([...records].some(record => record.source === source && record.key === key)) setHistory(source, key, pixels);
        for (const record of uploaded) setHistory(record.source, record.key, record.pixels);
        throw error;
      } finally {
        captureState = null;
        for (const record of records) if (record.refs === 0 && current.get(record.original) !== record) retire(record);
        for (const [source, group] of before.histories) for (const pixels of group.values()) releasePixels(source, pixels);
        pruneHistories(); syncPool();
      }
    },
    sampleData(texture) {
      if (isTarget({texture})) return targets().sampleData(texture);
      if (isFramebuffer({texture})) return framebuffers().sampleData(texture);
      layoutOf({texture, isSampledTexture: true});
      return {samples: 1, primarySamples: 1, isMSAA: false};
    },
    withDispatch(snapshots, run) {
      live();
      const submit = () => framebufferInputs ? framebufferInputs.withDispatch(snapshots.filter(s => framebufferInputs.owns(s)), run) : run();
      return targetInputs ? targetInputs.withDispatch(snapshots.filter(s => targetInputs.owns(s)), submit) : submit();
    },
    layout(snapshot) { live(); if (targetInputs?.owns(snapshot)) return targetInputs.layout(snapshot);
      if (framebufferInputs?.owns(snapshot)) return framebufferInputs.layout(snapshot);
      return leases.has(snapshot) ? snapshot.layout : storage.layout(snapshot); },
    validateDispatch(snapshots) {
      live();
      const owned = [];
      for (const snapshot of snapshots) {
        if (leases.has(snapshot)) validate(snapshot);
        else if (targetInputs?.owns(snapshot)) targetInputs.validateDispatch([snapshot]);
        else if (framebufferInputs?.owns(snapshot)) framebufferInputs.validateDispatch([snapshot]);
        else owned.push(snapshot);
      }
      storage.validateDispatch(owned);
    },
    resolve(snapshot) {
      live();
      if (targetInputs?.owns(snapshot)) return targetInputs.resolve(snapshot);
      if (framebufferInputs?.owns(snapshot)) return framebufferInputs.resolve(snapshot);
      if (!leases.has(snapshot)) return storage.resolve(snapshot);
      validate(snapshot);
      if (snapshot.record.pixels.readyState === 'failed') throw snapshot.record.pixels.readyError;
      if (snapshot.record.pixels.readyState === 'pending') fail('NOT_READY', 'Captured image is still preparing');
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
    written(snapshots) { live(); storage.written(snapshots.filter(snapshot => !leases.has(snapshot) && !targetInputs?.owns(snapshot) && !framebufferInputs?.owns(snapshot))); },
    release(snapshot) {
      if (targetInputs?.owns(snapshot)) { targetInputs.release(snapshot); return; }
      if (framebufferInputs?.owns(snapshot)) { framebufferInputs.release(snapshot); return; }
      if (!leases.has(snapshot) || snapshot.released) return;
      snapshot.released = true;
      const record = snapshot.record;
      if (--record.refs === 0 && current.get(record.original) !== record && !captureState?.records.has(record)) { retire(record); syncPool(); }
    },
    async whenIdle() { live(); await Promise.all([storage.whenIdle(), pool?.whenIdle(), targetInputs?.whenIdle(), framebufferInputs?.whenIdle()]); live(); },
    get diagnostics() { return {textures: current.size, versions: records.size, snapshotBytes,
      ...(framebufferInputs ? {framebuffers: framebufferInputs.diagnostics} : {})}; },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const [original, listener] of watches) original.removeEventListener('dispose', listener);
      watches.clear();
      for (const record of records) retire(record);
      current.clear(); syncPool(); targetInputs?.dispose(); framebufferInputs?.dispose();
    },
  };
}
