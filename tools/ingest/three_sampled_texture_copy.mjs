/** Source sampled pixels -> shared storage or native target color.
 * Capture with the existing sampled owner; borrow its native residency only at the renderer's FIFO slot.
 * This layer neither uploads pixels nor changes the source texture/CPU array.
 */
import {ThreeStorageCopyError, inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };
const EMPTY = Symbol('empty sampled copy');

/** Raw destination capture deliberately does not use sampled target bindings:
 * copying float32 color must not require a filtering feature or a UV replica. */
function captureDestination(renderer, texture, level) {
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
  const targets = renderer._targets, target = targets?.getTextureTarget(texture);
  if (!texture?.isRenderTargetTexture || texture.isDepthTexture || !target)
    fail('SOURCE', 'Expected a storage texture or target-owned color destination');
  const entry = targets.capture(target);
  if (entry.shape.sampleCount !== 1) fail('SAMPLES', 'Sampled copies require a single-sample target destination');
  const colorIndex = entry.shape.textures.indexOf(texture);
  if (colorIndex < 0) fail('STALE', 'Color texture no longer belongs to its render target');
  const color = entry.shape.colors[colorIndex];
  // CubeCamera can toggle generation without changing the native allocation.
  // Capture this call's policy; authored faces share its six array layers.
  const generateMipmaps = entry.shape.cube ? texture.generateMipmaps : color.generateMipmaps;
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
    withTexture: (run, validation) => targets.withNativeColorDestination(source, run, {validation}),
    release: targets.retainGeneration(entry)};
}

export function copySampledTexture(renderer, sourceTexture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0, onError = () => {}) {
  const device = renderer._device;
  if (!device || typeof renderer._getSampledTextureInputs !== 'function')
    fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  const inputs = renderer._getSampledTextureInputs(), signal = renderer._targetLifetime?.signal;
  let lease, destination, released = false, cancelWait = null;
  const cancelled = () => signal?.reason instanceof Error ? signal.reason
    : new ThreeStorageCopyError('DISPOSED', 'Renderer is disposed');
  const release = () => {
    if (released) return;
    released = true;
    signal?.removeEventListener('abort', abort);
    try { if (lease) inputs.release(lease); }
    finally {
      try { destination?.release?.(); }
      finally { cancelWait?.(cancelled()); cancelWait = null; }
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
      destination = captureDestination(renderer, destinationTexture, destinationLevel);
      lease = inputs.captureBinding({texture: sourceTexture, isSampledTexture: true});
      const plan = inspectStorageTextureCopy(inputs.copyShape(lease), destination.shape,
        region, position, sourceLevel, destinationLevel);
      // Roll back a no-op too: its never-uploaded pixels are not a new history.
      if (plan.empty) throw EMPTY;
      const ready = inputs.preparation([lease]);
      signal?.addEventListener('abort', abort, {once: true});
      check();
      const submit = () => {
        try {
          check(); destination.check(); inputs.validateDispatch([lease]);
          // Check destination budgets before a source upload can issue effects.
          destination.preflight();
          inputs.withTexture(lease, nativeSource => {
            check(); destination.check();
            // Snapshot earlier validation BEFORE entering the destination scope.
            // A later inputs.whenIdle() would include this write through storage
            // and make the destination's validation depend on itself.
            const validation = inputs.whenIdle(); validation.catch(() => {});
            destination.withTexture(nativeDestination => {
              const encoder = device.createCommandEncoder({label: 'f3d-sampled-texture-copy'});
              encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
                {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
              const commands = encoder.finish();
              check(); destination.check(); inputs.validateDispatch([lease]);
              device.queue.submit([commands]);
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
