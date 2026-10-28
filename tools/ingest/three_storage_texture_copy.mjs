/** Source-shaped, queue-ordered GPU texture copies. */
import {createStorageTextureTransfers} from './three_storage_texture_transfer.mjs';
import {ThreeStorageCopyError} from './three_texture_copy_plan.mjs';
import {captureTargetToStorageCopy, captureStorageToTargetCopy} from './three_target_texture_copy.mjs';
import {copySampledTexture} from './three_sampled_texture_copy.mjs';
import {assertThreeNativeTextureWrite} from './three_node_shadows.mjs';
export {ThreeStorageCopyError, inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };
const ordinaryTargetTexture = texture => texture?.renderTarget != null && texture.isStorageTexture !== true &&
  (texture.isRenderTargetTexture === true || texture.isDepthTexture === true);

/** Install below withThreeCompute's facade. Its waitForGPU already waits on the
 * shared storage owner; this layer additionally reports rejected FIFO entries.
 * An initialized source call stays void, including when preparation defers it.
 */
export function withStorageTextureCopies(Base) {
  const failures = new WeakMap(), transfers = new WeakMap();
  const pixels = renderer => {
    let owner = transfers.get(renderer);
    if (!owner) { owner = createStorageTextureTransfers(renderer); transfers.set(renderer, owner); }
    return owner;
  };
  return class StorageCopyRenderer extends Base {
    copyTextureToTexture(source, destination, region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      assertThreeNativeTextureWrite(this, destination);
      // The target transfer facade installs source methods per instance. This
      // inherited method suppresses that installation, so super cannot reach it.
      // Dispatch only two ordinary target endpoints; storage admission stays here.
      if (ordinaryTargetTexture(source) && ordinaryTargetTexture(destination) &&
          typeof this._transfers?.copyTexture === 'function') {
        this._transfers.copyTexture(source, destination, region, position, sourceLevel, destinationLevel)
          .catch(error => { this._deferredError ??= error; });
        return;
      }
      if (!this.isWebGPURenderer) fail('ROUTE', 'Storage texture copies require the WebGPU renderer surface');
      if (!this._initialized) fail('INITIALIZATION', 'Initialize the renderer before copying textures');
      const rejected = error => {
        if (!failures.has(this)) failures.set(this, error);
        this._deferredError ??= error;
      };
      let issue;
      if (source?.isRenderTargetTexture && destination?.isStorageTexture) {
        const request = captureTargetToStorageCopy(this, source, destination, region, position, sourceLevel, destinationLevel);
        issue = request.issue;
      } else if (source?.isStorageTexture && ordinaryTargetTexture(destination)) {
        const request = captureStorageToTargetCopy(this, source, destination, region, position, sourceLevel, destinationLevel);
        issue = request.issue;
      } else if (source?.isStorageTexture && destination?.isStorageTexture) {
        const owner = this._storageTextures;
        const request = owner.captureCopy(source, destination, region, position, sourceLevel, destinationLevel);
        issue = () => { owner.copy(request); };
      } else if (source?.isTexture && !source.isStorageTexture && !source.isRenderTargetTexture &&
          (destination?.isStorageTexture || destination?.isRenderTargetTexture)) {
        copySampledTexture(this, source, destination, region, position, sourceLevel, destinationLevel, rejected);
        this.inspector?.copyTextureToTexture?.(source, destination);
        return;
      } else {
        if (typeof super.copyTextureToTexture === 'function')
          return super.copyTextureToTexture(source, destination, region, position, sourceLevel, destinationLevel);
        fail('SOURCE', 'Expected storage textures or copies between storage and target-owned color');
      }
      if (this._computeTurn) this._computeTurn.submit(issue)?.catch?.(rejected);
      else if (!this._drain && !this._preparing && !this._pending?.length) issue();
      else this._enqueue({transfer: true, issue, resolve() {}, reject: rejected});
      this.inspector?.copyTextureToTexture?.(source, destination);
    }
    /** F3D extension: upload exact raw pixel bytes without a seeding kernel.
     * Captures bytes/options now; completion includes queued GPU validation.
     * Byte offsets are relative to the supplied view, not its backing buffer. */
    writeStorageTexturePixelsAsync(texture, data, options = {}) {
      try {
        assertThreeNativeTextureWrite(this, texture);
        return pixels(this).write(texture, data, options);
      } catch (error) { return Promise.reject(error); }
    }
    /** F3D extension: snapshot a GPU texture region into tightly packed bytes.
     * Captures the FIFO copy point, not a later texture state at map completion. */
    readStorageTexturePixelsAsync(texture, options = {}) {
      return pixels(this).read(texture, options);
    }
    dispose() { transfers.get(this)?.dispose(); super.dispose(); }
    async waitForGPU() {
      await super.waitForGPU();
      if (failures.has(this)) throw failures.get(this);
      await transfers.get(this)?.whenIdle();
    }
  };
}
