/** Connect source RenderTarget color to GPU-authored storage without a CPU
 * image, UV replica, sampler, conversion shader or additional texture owner.
 * The renderer facade schedules the captured request in its ordinary FIFO.
 */
import {ThreeStorageCopyError, inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };

export function captureTargetToStorageCopy(renderer, texture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
  const targets = renderer._targets, storage = renderer._storageTextures, device = renderer._device;
  if (!targets || !storage || !device) fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  if (!texture?.isRenderTargetTexture || texture.isDepthTexture || !texture.renderTarget || !destinationTexture?.isStorageTexture)
    fail('SOURCE', 'Expected target-owned color and a storage texture destination');
  const entry = targets.capture(texture.renderTarget);
  if (entry.shape.texture !== texture) fail('STALE', 'Color texture no longer belongs to its render target');
  // A transfer does not sample. Build the owner-native borrow record directly
  // from capture(), rather than imposing a float-filtering feature or creating
  // a sampler through captureNativeTexture(). The resolved color is single-
  // sample even when the source render attachment is multisampled.
  const source = Object.freeze({entry, texture, device, depth: false});
  const shape = {...entry.shape, depth: 1, viewDimension: '2d'};
  const destination = storage.capture(destinationTexture);
  const plan = inspectStorageTextureCopy(shape, destination.shape, region, position, sourceLevel, destinationLevel);
  const write = plan.empty ? null : storage.captureBinding({texture: destinationTexture,
    isSampledTexture: true, store: true, access: 'writeOnly', mipLevel: destinationLevel});
  return Object.freeze({issue() {
    if (renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
    if (renderer._device !== device) fail('DEVICE', 'Texture copies cannot migrate between devices');
    targets.assertCurrent(entry); storage.assertCurrent(destination);
    if (entry.shape.texture !== texture) fail('STALE', 'Captured color texture was replaced');
    if (plan.empty) return false;
    // Check both budgets and all device extents before allocating either side.
    targets.validateNativeTextures([source]); storage.validateAllocations([destination]);
    const limit = device.limits?.maxTextureDimension2D;
    if (!Number.isSafeInteger(limit) || shape.width > limit || shape.height > limit)
      fail('LIMIT', 'Source target exceeds the device texture dimension limit');
    targets.withNativeTexture(source, nativeSource => storage.withTexture(destination, nativeDestination => {
      const encoder = device.createCommandEncoder({label: 'f3d-target-to-storage-copy'});
      encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
        {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
      targets.assertCurrent(entry); storage.assertCurrent(destination);
      if (renderer._disposed || renderer._device !== device) fail('STALE', 'Renderer changed during a texture copy');
      device.queue.submit([encoder.finish()]);
    }, write));
    return true;
  }});
}
