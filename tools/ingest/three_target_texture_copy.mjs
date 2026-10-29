/** Connect source RenderTarget color and GPU-authored storage without a CPU
 * image, UV replica, sampler, conversion shader or additional texture owner.
 * The renderer facade schedules the captured request in its ordinary FIFO.
 */
import {ThreeStorageCopyError, inspectStorageTextureCopy} from './three_texture_copy_plan.mjs';
import {inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {captureTextureCopy} from './three_texture_copy_lifetime.mjs';
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };

function targetColor(renderer, texture, retain) {
  const targets = renderer._targets, device = renderer._device;
  const target = targets.getTextureTarget(texture);
  if (!texture?.isRenderTargetTexture || texture.isDepthTexture || !target)
    fail('SOURCE', 'Expected target-owned color');
  const entry = targets.capture(target), colorIndex = entry.shape.textures.indexOf(texture);
  if (colorIndex < 0) fail('STALE', 'Color texture no longer belongs to its render target');
  retain(entry);
  // A raw copy addresses cube faces as native array layers. Never introduce
  // a cube sampler, source-face permutation, CPU upload or UV-flipped replica.
  const shape = {...entry.shape.colors[colorIndex], depth: entry.shape.layers ?? 1,
    viewDimension: entry.shape.cube ? '2d-array' : entry.shape.viewDimension};
  const source = Object.freeze({entry, texture, device, depth: false, colorIndex});
  return {entry, shape, source,
    check() {
      targets.assertCurrent(entry);
      if (entry.shape.textures[colorIndex] !== texture) fail('STALE', 'Captured color texture was replaced');
    },
    preflight() {
      targets.validateNativeTextures([source]);
      const volume = shape.viewDimension === '3d';
      const limit = device.limits?.[volume ? 'maxTextureDimension3D' : 'maxTextureDimension2D'];
      if (!Number.isSafeInteger(limit) || shape.width > limit || shape.height > limit || volume && shape.depth > limit)
        fail('LIMIT', 'Target exceeds the device texture dimension limit');
    }};
}
function checkRenderer(renderer, device) {
  if (renderer._disposed) fail('DISPOSED', 'Renderer is disposed');
  if (renderer._device !== device) fail('DEVICE', 'Texture copies cannot migrate between devices');
}

export function captureTargetToStorageCopy(renderer, texture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
  const targets = renderer._targets, storage = renderer._storageTextures, device = renderer._device;
  if (!targets || !storage || !device) fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  return captureTextureCopy(renderer, retain => {
    if (!destinationTexture?.isStorageTexture) fail('SOURCE', 'Expected a storage texture destination');
    const source = targetColor(renderer, texture, retain), destination = storage.capture(destinationTexture);
    const plan = inspectStorageTextureCopy(source.shape, destination.shape, region, position, sourceLevel, destinationLevel);
    const write = plan.empty ? null : storage.captureBinding({texture: destinationTexture,
      isSampledTexture: true, store: true, access: 'writeOnly', mipLevel: destinationLevel});
    return {empty: plan.empty, issue() {
      checkRenderer(renderer, device); source.check(); storage.assertCurrent(destination);
      if (plan.empty) return false;
      // Check both budgets and device extents before allocating either side.
      source.preflight(); storage.validateAllocations([destination]);
      targets.withNativeTexture(source.source, nativeSource => storage.withTexture(destination, nativeDestination => {
        const encoder = device.createCommandEncoder({label: 'f3d-target-to-storage-copy'});
        encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
          {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
        const commands = encoder.finish();
        source.check(); storage.assertCurrent(destination); checkRenderer(renderer, device);
        device.queue.submit([commands]);
      }, write, {validation: Promise.all([source.entry.validation, source.entry.storage.validation])}));
      return true;
    }};
  });
}

/** Copy a storage mip or array layer into a selected single-sample target color.
 * Keep GPU contents at the submission point, while the source-shaped box,
 * destination position, formats and attachment identities are captured now.
 */
export function captureStorageToTargetCopy(renderer, texture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
  const targets = renderer._targets, storage = renderer._storageTextures, device = renderer._device;
  if (!targets || !storage || !device) fail('DEVICE', 'Initialize the renderer and its texture owners before copying');
  return captureTextureCopy(renderer, retain => {
    if (!texture?.isStorageTexture) fail('SOURCE', 'Expected a storage texture source');
    const destination = targetColor(renderer, destinationTexture, retain);
    if (destination.entry.shape.cube) fail('SOURCE', 'Cube target destinations require explicit face-aware write support');
    if (destination.entry.shape.sampleCount !== 1) fail('SAMPLES', 'Storage copies require a single-sample target destination');
    if (destinationLevel !== 0) fail('LEVEL', 'Storage copies write target mip zero; derived mips regenerate on demand');
    const source = storage.capture(texture);
    const plan = inspectStorageTextureCopy(source.shape, destination.shape, region, position, sourceLevel, destinationLevel);
    return {empty: plan.empty, issue() {
      checkRenderer(renderer, device); destination.check(); storage.assertCurrent(source);
      if (plan.empty) return false;
      destination.preflight(); storage.validateAllocations([source]);
      // Snapshot source validation after its borrow closes, before a later
      // reverse copy can make the producer depend on this destination again.
      let resolve, reject;
      const validation = new Promise((yes, no) => { resolve = yes; reject = no; });
      validation.catch(() => {});
      try {
        storage.withTexture(source, nativeSource => targets.withNativeColorDestination(destination.source, nativeDestination => {
          const encoder = device.createCommandEncoder({label: 'f3d-storage-to-target-copy'});
          encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
            {texture: nativeDestination, mipLevel: 0, origin: plan.destinationOrigin}, plan.size);
          const commands = encoder.finish();
          destination.check(); storage.assertCurrent(source); checkRenderer(renderer, device);
          device.queue.submit([commands]);
        }, {validation}));
        resolve(storage.validation);
        return true;
      } catch (error) { reject(error); throw error; }
    }};
  });
}

/** Source-shaped cube/array/volume color copies through the target owner's
 * existing face-aware transfer path. One accepted range has one FIFO slot;
 * each slice keeps the same owner validation, mip and history-color policy.
 */
export function captureTargetColorCopy(renderer, texture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
  const targets = renderer._targets, device = renderer._device;
  if (!targets || !device) fail('DEVICE', 'Initialize the renderer and its target owner before copying');
  return captureTextureCopy(renderer, retain => {
    const source = targetColor(renderer, texture, retain), destination = targetColor(renderer, destinationTexture, retain);
    if (source.entry === destination.entry || source.entry.historyGroup && source.entry.historyGroup === destination.entry.historyGroup)
      fail('FEEDBACK', 'In-place target color copies are not admitted');
    if (destinationLevel !== 0) fail('LEVEL', 'Target color copies write mip zero; derived mips remain owner-managed');
    if (destination.entry.shape.sampleCount !== 1) fail('SAMPLES', 'Target color copies require a single-sample destination');
    // Native copies preserve bits across sRGB view-compatible formats. The
    // target transfer planner below still validates the original formats.
    const compatible = shape => ({...shape, format: shape.format.replace(/-srgb$/, '')});
    const plan = inspectStorageTextureCopy(compatible(source.shape), compatible(destination.shape),
      region, position, sourceLevel, destinationLevel);
    const [sourceX, sourceY, sourceFace] = plan.sourceOrigin, [x, y, face] = plan.destinationOrigin;
    const [width, height, count] = plan.size;
    const base = Object.freeze({sourceX, sourceY, x, y, width, height, sourceMipLevel: sourceLevel,
      sourceColorIndex: source.source.colorIndex, colorIndex: destination.source.colorIndex, flipY: false});
    const step = index => ({...base, sourceFace: sourceFace + index, face: face + index});
    // Extents are contiguous; the checked XYZ plan plus both endpoint slices
    // validates every middle layer without building an unbounded request list.
    if (!plan.empty) {
      inspectGpuTargetCopy(source.entry.shape, destination.entry.shape, step(0));
      inspectGpuTargetCopy(source.entry.shape, destination.entry.shape, step(count - 1));
    }
    return {empty: plan.empty, issue() {
      checkRenderer(renderer, device); source.check(); destination.check();
      if (plan.empty) return false;
      source.preflight(); destination.preflight();
      // Admit the combined allocation, not two individually affordable owners.
      // No layer may be written before all destination extents/budgets pass.
      targets.validateNativeTextures([source.source, destination.source]);
      for (let index = 0; index < count; index++) {
        checkRenderer(renderer, device); source.check(); destination.check();
        targets.copyColor(source.entry, destination.entry, step(index));
      }
      return true;
    }};
  });
}
