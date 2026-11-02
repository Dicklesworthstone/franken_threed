/** Sampled pixels -> sampled/storage/target, and GPU producers -> sampled.
 * Capture with the existing sampled owner; borrow its native residency only at the renderer's FIFO slot.
 * This layer neither uploads pixels nor changes the source texture/CPU array.
 */
import {ThreeStorageCopyError, inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';
import {captureTargetColor} from './three_target_texture_copy.mjs';
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };
const EMPTY = Symbol('empty sampled copy');

/** Raw destination capture deliberately does not use sampled target bindings:
 * copying float32 color must not require a filtering feature or a UV replica. */
function captureDestination(renderer, texture, level, inputs) {
  const device = renderer._device;
  if (texture?.isStorageTexture === true) {
    const storage = renderer._storageTextures;
    if (!storage) fail('DEVICE', 'Initialize the storage texture owner before copying');
    const entry = storage.capture(texture);
    const write = storage.captureBinding({texture, isSampledTexture: true,
      store: true, access: 'writeOnly', mipLevel: level});
    return {shape: entry.shape,
      check: () => storage.assertCurrent(entry),
      preflight: () => storage.validateAllocations([entry]),
      withTexture: (run, validation) => storage.withTexture(entry, run, write, {validation})};
  }
  if (texture?.isTexture === true && !texture.isRenderTargetTexture && !texture.isDepthTexture && !texture.isFramebufferTexture) {
    if (renderer.isWebGPURenderer !== true || renderer._shaderEncodedOutput === true)
      fail('ROUTE', 'Ordinary sampled copy destinations require the native WebGPU renderer surface');
    const lease = inputs.captureCopyDestination(texture, level);
    try {
      return {lease, shape: inputs.copyShape(lease),
        check: () => inputs.validateDispatch([lease]),
        preflight: source => inputs.validateCopyAllocations(source ? [source, lease] : [lease]),
        release: () => inputs.release(lease)};
    } catch (error) { inputs.release(lease); throw error; }
  }
  const targets = renderer._targets, target = targets?.getTextureTarget(texture);
  if (!texture?.isRenderTargetTexture || texture.isDepthTexture || !target)
    fail('SOURCE', 'Expected a storage texture or target-owned color destination');
  let entry = targets.capture(target), colorIndex = entry.shape.textures.indexOf(texture);
  if (colorIndex < 0 && entry.historyGroup)
    ({entry, colorIndex} = targets.captureNativeTexture(texture, {sampled: false}));
  if (entry.shape.sampleCount !== 1) fail('SAMPLES', 'Sampled copies require a single-sample target destination');
  if (colorIndex < 0) fail('STALE', 'Color texture no longer belongs to its render target');
  const color = entry.shape.colors[colorIndex];
  const explicitMipmaps = !renderer._shaderEncodedOutput && (entry.historyGroup || entry.shape.textures.length > 1 ||
    !entry.shape.cube && !entry.shape.array && !entry.shape.volume);
  // CubeCamera can toggle generation without changing the native allocation.
  // Capture this call's policy; authored faces share its six array layers.
  const generateMipmaps = entry.shape.cube || explicitMipmaps
    ? texture.generateMipmaps : color.generateMipmaps;
  if (entry.shape.cube && generateMipmaps !== false)
    fail('LEVEL', 'Cube destinations require an authored, non-generated mip chain');
  // Authored chains already belong to the target's native allocation. The
  // shared planner validates the selected mip and its independent XYZ extent.
  // Do not write a derived level: the owner's next mip refresh would replace it.
  if (level !== 0 && generateMipmaps !== false)
    fail('LEVEL', 'Nonzero target destination mips require an authored, non-generated chain');
  const source = Object.freeze({entry, texture, device, depth: false, colorIndex});
  return {shape: {...color, depth: entry.shape.layers ?? 1,
      viewDimension: entry.shape.cube ? '2d-array' : entry.shape.viewDimension},
    check() {
      targets.assertCurrent(entry);
      if (entry.shape.textures[colorIndex] !== texture) fail('STALE', 'Captured color texture was replaced');
    },
    preflight() {
      targets.validateNativeTextures([source]);
      const volume = entry.shape.viewDimension === '3d';
      const limit = device.limits?.[volume ? 'maxTextureDimension3D' : 'maxTextureDimension2D'];
      if (!Number.isSafeInteger(limit) || entry.shape.width > limit || entry.shape.height > limit ||
          volume && entry.shape.layers > limit)
        fail('LIMIT', 'Destination target exceeds the device texture dimension limit');
      // The allocation, not just the copied region, must fit. A volume's color
      // uses 3D limits but its independently allocated depth stays 2D/2D-array.
      const hasDepth = entry.shape.depthFormat != null;
      if (volume && hasDepth) {
        const depthLimit = device.limits?.maxTextureDimension2D;
        if (!Number.isSafeInteger(depthLimit) || entry.shape.width > depthLimit || entry.shape.height > depthLimit)
          fail('LIMIT', 'Destination depth exceeds the device 2D texture dimension limit');
      }
      const array = entry.shape.cube || entry.shape.viewDimension === '2d-array';
      if (array || hasDepth && entry.shape.depthArray) {
        const layerLimit = device.limits?.maxTextureArrayLayers;
        const layers = Math.max(array ? entry.shape.layers : 1, hasDepth ? entry.shape.depthLayers ?? 1 : 1);
        if (!Number.isSafeInteger(layerLimit) || layers > layerLimit)
          fail('LIMIT', 'Destination target exceeds the device texture array layer limit');
      }
    },
    withTexture: (run, validation) => targets.withNativeColorDestination(source, run, {validation,
      generateMipmaps: explicitMipmaps && level === 0 && generateMipmaps === true}),
    release: targets.retainGeneration(entry)};
}

/** Reuse raw producer ownership: copying a float target or a selected array
 * slice must not allocate a sampled view, UV replica or another upload pool. */
function captureProducedSource(renderer, texture) {
  if (texture?.isStorageTexture) {
    const storage = renderer._storageTextures;
    if (!storage) fail('DEVICE', 'Initialize the storage texture owner before copying');
    const entry = storage.capture(texture);
    return {shape: entry.shape, check: () => storage.assertCurrent(entry),
      preflight: () => storage.validateAllocations([entry]),
      withTexture: run => storage.withTexture(entry, run),
      validation: () => storage.validation,
      watch(cancel) {
        const disposed = () => { try { storage.assertCurrent(entry); } catch (error) { cancel(error); } };
        texture.addEventListener('dispose', disposed);
        return () => texture.removeEventListener('dispose', disposed);
      }};
  }
  if (!texture?.isRenderTargetTexture) return null;
  if (!renderer._targets) fail('DEVICE', 'Initialize the target owner before copying');
  let release;
  try {
    const source = captureTargetColor(renderer, texture, entry => { release = renderer._targets.retainGeneration(entry); });
    const signal = source.entry.lifetime?.signal;
    return {...source, release,
      withTexture: run => renderer._targets.withNativeTexture(source.source, run),
      validation: () => Promise.all([source.entry.validation, source.entry.storage.validation]),
      watch(cancel) {
        const disposed = () => cancel(signal.reason ?? new ThreeStorageCopyError('STALE', 'Source target was disposed'));
        if (signal?.aborted) disposed(); else signal?.addEventListener('abort', disposed, {once: true});
        return () => signal?.removeEventListener('abort', disposed);
      }};
  } catch (error) { release?.(); throw error; }
}

export function copySampledTexture(renderer, sourceTexture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0, onError = () => {}) {
  const device = renderer._device;
  if (!device || typeof renderer._getSampledTextureInputs !== 'function')
    fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  const inputs = renderer._getSampledTextureInputs(), signal = renderer._targetLifetime?.signal;
  let lease, destination, producer, unwatchProducer, released = false, cancelWait = null;
  const cancelled = () => signal?.reason instanceof Error ? signal.reason
    : new ThreeStorageCopyError('DISPOSED', 'Renderer is disposed');
  const release = () => {
    if (released) return;
    released = true;
    signal?.removeEventListener('abort', abort);
    unwatchProducer?.(); unwatchProducer = null;
    try { if (lease) inputs.release(lease); }
    finally {
      try { producer?.release?.(); }
      finally {
        try { destination?.release?.(); }
        finally { cancelWait?.(cancelled()); cancelWait = null; }
      }
    }
  };
  const abort = () => release();
  const check = () => {
    if (renderer._disposed || signal?.aborted) throw cancelled();
    if (released) fail('STALE', 'Sampled copy was released before submission');
    if (renderer._device !== device) fail('DEVICE', 'Texture copies cannot migrate between devices');
  };
  let reported = false;
  const rejected = error => {
    release();
    if (!reported) { reported = true; onError(error); }
  };
  try {
    // Include FIFO admission in the transaction. A full queue or a rejected
    // descriptor must not become the baseline for later partial source uploads.
    inputs.captureDispatch(() => {
      check();
      destination = captureDestination(renderer, destinationTexture, destinationLevel, inputs);
      producer = captureProducedSource(renderer, sourceTexture);
      if (producer && !destination.lease) fail('SOURCE', 'Raw producers on this path require an ordinary sampled destination');
      if (!producer) lease = inputs.captureBinding({texture: sourceTexture, isSampledTexture: true});
      const sourceShape = producer?.shape ?? inputs.copyShape(lease);
      if (destination.lease && !producer && (sourceShape.viewDimension !== '2d' || sourceLevel !== 0 && region === null))
        fail('SOURCE', 'Ordinary sampled destinations require a 2D source and an explicit region for nonzero source mips');
      // The pinned native API defaults to base image width/height and ONE
      // layer, including for a layered or nonzero-mip source. Explicit boxes
      // address mip-local texels. Never silently substitute a smaller extent.
      const sourceRegion = producer && region === null ? {isBox2: true,
        min: {x: 0, y: 0}, max: {x: sourceShape.width, y: sourceShape.height}} : region;
      const compatible = shape => producer ? {...shape, format: shape.format.replace(/-srgb$/, '')} : shape;
      const plan = inspectStorageTextureCopy(compatible(sourceShape), compatible(destination.shape),
        sourceRegion, position, sourceLevel, destinationLevel,
        destination.lease && lease ? inputs.sameTexture(lease, destination.lease) : false);
      // Roll back a no-op too: its never-uploaded pixels are not a new history.
      if (plan.empty) throw EMPTY;
      const sampled = [lease, destination.lease].filter(Boolean);
      const ready = inputs.preparation(sampled);
      unwatchProducer = producer?.watch(rejected);
      signal?.addEventListener('abort', abort, {once: true});
      check();
      const submit = () => {
        try {
          check(); destination.check(); producer?.check(); if (lease) inputs.validateDispatch([lease]);
          // Check destination budgets before a source upload can issue effects.
          destination.preflight(lease);
          producer?.preflight();
          const encode = (nativeSource, nativeDestination) => {
            const encoder = device.createCommandEncoder({label: 'f3d-sampled-texture-copy'});
            encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
              {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
            const commands = encoder.finish();
            check(); destination.check(); producer?.check(); if (lease) inputs.validateDispatch([lease]);
            device.queue.submit([commands]);
          };
          if (producer) {
            // User upload callbacks run before either raw GPU borrow. A source
            // owner closes its validation scopes after the synchronous borrow;
            // link those already-issued results without waiting for the queue.
            inputs.resolve(destination.lease);
            let resolve, reject;
            const validation = new Promise((yes, no) => { resolve = yes; reject = no; });
            validation.catch(() => {});
            try {
              producer.withTexture(nativeSource => inputs.writeCopyDestination(destination.lease,
                nativeDestination => encode(nativeSource, nativeDestination), {validation}));
              resolve(producer.validation());
            } catch (error) { reject(error); throw error; }
            return;
          }
          if (destination.lease) {
            inputs.copyTextures(lease, destination.lease, encode);
            return;
          }
          inputs.withTexture(lease, nativeSource => {
            check(); destination.check();
            // Snapshot earlier validation BEFORE entering the destination scope.
            // A later inputs.whenIdle() would include this write through storage
            // and make the destination's validation depend on itself.
            const validation = inputs.whenIdle(); validation.catch(() => {});
            destination.withTexture(nativeDestination => {
              encode(nativeSource, nativeDestination);
            }, validation);
          });
        } finally { release(); }
      };
      const issue = ready ? () => new Promise((resolve, reject) => {
        if (released) { reject(cancelled()); return; }
        cancelWait = reject;
        // Observe both outcomes even if cancellation wins. A late bitmap must
        // never restart a disposed copy or publish any destination bytes.
        ready.then(() => {
          cancelWait = null;
          try { resolve(submit()); } catch (error) { reject(error); }
        }, error => { cancelWait = null; release(); reject(error); });
      }) : submit;
      if (renderer._computeTurn) renderer._computeTurn.submit(issue, !!ready)?.catch?.(rejected);
      else if (!ready && !renderer._drain && !renderer._preparing && !renderer._pending?.length) issue();
      else renderer._enqueue({transfer: true, awaitSubmission: !!ready, issue, resolve() {}, reject: rejected});
    });
  } catch (error) {
    release();
    if (error !== EMPTY) throw error;
  }
}
