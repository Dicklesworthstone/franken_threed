/** Source sampled pixels -> shared GPU storage. Capture with the existing
 * sampled owner; borrow its native residency only at the renderer's FIFO slot.
 * This layer neither uploads pixels nor changes the source texture/CPU array.
 */
import {ThreeStorageCopyError, inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };
const EMPTY = Symbol('empty sampled copy');

export function copySampledTextureToStorage(renderer, sourceTexture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0, onError = () => {}) {
  const device = renderer._device, storage = renderer._storageTextures;
  if (!device || !storage || typeof renderer._getSampledTextureInputs !== 'function')
    fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  const inputs = renderer._getSampledTextureInputs(), signal = renderer._targetLifetime?.signal;
  let lease, released = false, cancelWait = null;
  const cancelled = () => signal?.reason instanceof Error ? signal.reason
    : new ThreeStorageCopyError('DISPOSED', 'Renderer is disposed');
  const release = () => {
    if (released) return;
    released = true;
    signal?.removeEventListener('abort', abort);
    if (lease) inputs.release(lease);
    cancelWait?.(cancelled()); cancelWait = null;
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
      const destination = storage.capture(destinationTexture);
      lease = inputs.captureBinding({texture: sourceTexture, isSampledTexture: true});
      const plan = inspectStorageTextureCopy(inputs.copyShape(lease), destination.shape,
        region, position, sourceLevel, destinationLevel);
      // Roll back a no-op too: its never-uploaded pixels are not a new history.
      if (plan.empty) throw EMPTY;
      const write = storage.captureBinding({texture: destinationTexture, isSampledTexture: true,
        store: true, access: 'writeOnly', mipLevel: destinationLevel});
      const ready = inputs.preparation([lease]);
      signal?.addEventListener('abort', abort, {once: true});
      check();
      const submit = () => {
        try {
          check(); storage.assertCurrent(destination); inputs.validateDispatch([lease]);
          // Check destination budgets before a source upload can issue effects.
          storage.validateAllocations([destination]);
          inputs.withTexture(lease, nativeSource => {
            check(); storage.assertCurrent(destination);
            // Snapshot earlier validation BEFORE entering the destination scope.
            // A later inputs.whenIdle() would include this write through storage
            // and make the destination's validation depend on itself.
            const validation = inputs.whenIdle(); validation.catch(() => {});
            storage.withTexture(destination, nativeDestination => {
              const encoder = device.createCommandEncoder({label: 'f3d-sampled-to-storage-copy'});
              encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
                {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
              const commands = encoder.finish();
              check(); storage.assertCurrent(destination); inputs.validateDispatch([lease]);
              device.queue.submit([commands]);
            }, write, {validation});
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
