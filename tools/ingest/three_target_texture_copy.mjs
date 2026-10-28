/** Connect source RenderTarget color and GPU-authored storage without a CPU
 * image, UV replica, sampler, conversion shader or additional texture owner.
 * The renderer facade schedules the captured request in its ordinary FIFO.
 */
import {ThreeStorageCopyError, inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };

export function captureTargetToStorageCopy(renderer, texture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
  const targets = renderer._targets, storage = renderer._storageTextures, device = renderer._device;
  if (!targets || !storage || !device) fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  const target = targets.getTextureTarget(texture);
  if (!texture?.isRenderTargetTexture || texture.isDepthTexture || !target || !destinationTexture?.isStorageTexture)
    fail('SOURCE', 'Expected target-owned color and a storage texture destination');
  const entry = targets.capture(target);
  if (entry.shape.cube) fail('SOURCE', 'Cube target copies require explicit face-aware support');
  const colorIndex = entry.shape.textures.indexOf(texture);
  if (colorIndex < 0) fail('STALE', 'Color texture no longer belongs to its render target');
  // A transfer does not sample. Build the owner-native borrow record directly
  // from capture(), rather than imposing a float-filtering feature or creating
  // a sampler through captureNativeTexture(). The resolved color is single-
  // sample even when the source render attachment is multisampled.
  const source = Object.freeze({entry, texture, device, depth: false, colorIndex});
  const shape = {...entry.shape.colors[colorIndex], depth: entry.shape.layers ?? 1, viewDimension: entry.shape.viewDimension};
  const destination = storage.capture(destinationTexture);
  const plan = inspectStorageTextureCopy(shape, destination.shape, region, position, sourceLevel, destinationLevel);
  const write = plan.empty ? null : storage.captureBinding({texture: destinationTexture,
    isSampledTexture: true, store: true, access: 'writeOnly', mipLevel: destinationLevel});
  return Object.freeze({issue() {
    if (renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
    if (renderer._device !== device) fail('DEVICE', 'Texture copies cannot migrate between devices');
    targets.assertCurrent(entry); storage.assertCurrent(destination);
    if (entry.shape.textures[colorIndex] !== texture) fail('STALE', 'Captured color texture was replaced');
    if (plan.empty) return false;
    // Check both budgets and all device extents before allocating either side.
    targets.validateNativeTextures([source]); storage.validateAllocations([destination]);
    const limit = device.limits?.[shape.viewDimension === '3d' ? 'maxTextureDimension3D' : 'maxTextureDimension2D'];
    if (!Number.isSafeInteger(limit) || shape.width > limit || shape.height > limit || shape.viewDimension === '3d' && shape.depth > limit)
      fail('LIMIT', 'Source target exceeds the device texture dimension limit');
    targets.withNativeTexture(source, nativeSource => storage.withTexture(destination, nativeDestination => {
      const encoder = device.createCommandEncoder({label: 'f3d-target-to-storage-copy'});
      encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
        {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
      targets.assertCurrent(entry); storage.assertCurrent(destination);
      if (renderer._disposed || renderer._device !== device) fail('STALE', 'Renderer changed during a texture copy');
      device.queue.submit([encoder.finish()]);
    }, write, {validation: Promise.all([entry.validation, entry.storage.validation])}));
    return true;
  }});
}

/** Copy a storage mip or array layer into a selected single-sample target color.
 * Keep GPU contents at the submission point, while the source-shaped box,
 * destination position, formats and attachment identities are captured now.
 */
export function captureStorageToTargetCopy(renderer, texture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
  const targets = renderer._targets, storage = renderer._storageTextures, device = renderer._device;
  if (!targets || !storage || !device) fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  const target = targets.getTextureTarget(destinationTexture);
  if (!texture?.isStorageTexture || !destinationTexture?.isRenderTargetTexture ||
      destinationTexture.isDepthTexture || !target)
    fail('SOURCE', 'Expected a storage texture and target-owned color destination');
  const entry = targets.capture(target);
  if (entry.shape.cube) fail('SOURCE', 'Cube target copies require explicit face-aware support');
  if (entry.shape.sampleCount !== 1) fail('SAMPLES', 'Storage copies require a single-sample target destination');
  if (destinationLevel !== 0) fail('LEVEL', 'Storage copies write target mip zero; derived mips regenerate on demand');
  const colorIndex = entry.shape.textures.indexOf(destinationTexture);
  if (colorIndex < 0) fail('STALE', 'Color texture no longer belongs to its render target');
  const destination = Object.freeze({entry, texture: destinationTexture, device, depth: false, colorIndex});
  const shape = {...entry.shape.colors[colorIndex], depth: entry.shape.layers ?? 1, viewDimension: entry.shape.viewDimension};
  const source = storage.capture(texture);
  const plan = inspectStorageTextureCopy(source.shape, shape, region, position, sourceLevel, destinationLevel);
  return Object.freeze({issue() {
    if (renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
    if (renderer._device !== device) fail('DEVICE', 'Texture copies cannot migrate between devices');
    targets.assertCurrent(entry); storage.assertCurrent(source);
    if (entry.shape.textures[colorIndex] !== destinationTexture) fail('STALE', 'Captured color texture was replaced');
    if (plan.empty) return false;
    targets.validateNativeTextures([destination]); storage.validateAllocations([source]);
    const limit = device.limits?.[shape.viewDimension === '3d' ? 'maxTextureDimension3D' : 'maxTextureDimension2D'];
    if (!Number.isSafeInteger(limit) || shape.width > limit || shape.height > limit || shape.viewDimension === '3d' && shape.depth > limit)
      fail('LIMIT', 'Destination target exceeds the device texture dimension limit');
    // The source borrow closes its validation scope only after the destination
    // submits. Resolve with the snapshot synchronously after that scope closes:
    // a deferred getter could include later reverse copies and create a cycle.
    let resolve, reject;
    const validation = new Promise((yes, no) => { resolve = yes; reject = no; });
    validation.catch(() => {});
    try {
      storage.withTexture(source, nativeSource => targets.withNativeColorDestination(destination, nativeDestination => {
        const encoder = device.createCommandEncoder({label: 'f3d-storage-to-target-copy'});
        encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
          {texture: nativeDestination, mipLevel: 0, origin: plan.destinationOrigin}, plan.size);
        targets.assertCurrent(entry); storage.assertCurrent(source);
        if (renderer._disposed || renderer._device !== device) fail('STALE', 'Renderer changed during a texture copy');
        device.queue.submit([encoder.finish()]);
      }, {validation}));
      resolve(storage.validation);
      return true;
    } catch (error) { reject(error); throw error; }
  }});
}
