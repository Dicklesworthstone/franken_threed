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
    viewDimension: entry.shape.cube ? '2d-array' : entry.shape.viewDimension,
    // CubeCamera can toggle this flag without replacing the allocation. Capture
    // this use's policy, not an earlier entry's temporary generation setting.
    ...(entry.shape.cube ? {generateMipmaps: texture.generateMipmaps} : {})};
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
      // Check the whole allocation, not only the selected mip's copied region.
      // A volume's color is 3D, but its independently owned depth is 2D/array.
      const hasDepth = entry.shape.depthFormat != null;
      if (volume && hasDepth) {
        const depthLimit = device.limits?.maxTextureDimension2D;
        if (!Number.isSafeInteger(depthLimit) || shape.width > depthLimit || shape.height > depthLimit)
          fail('LIMIT', 'Target depth exceeds the device 2D texture dimension limit');
      }
      const array = shape.viewDimension === '2d-array';
      if (array || hasDepth && entry.shape.depthArray) {
        const layerLimit = device.limits?.maxTextureArrayLayers;
        const layers = Math.max(array ? shape.depth : 1, hasDepth ? entry.shape.depthLayers ?? 1 : 1);
        if (!Number.isSafeInteger(layerLimit) || layers > layerLimit)
          fail('LIMIT', 'Target exceeds the device texture array layer limit');
      }
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

/** Copy storage mips, array layers or volume slices into an authored target mip.
 * The selected color must be single-sample; its native allocation stays shared.
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
    if (destination.entry.shape.sampleCount !== 1) fail('SAMPLES', 'Storage copies require a single-sample target destination');
    // Explicitly authored levels already share this native allocation. Never
    // write a derived level: the next automatic mip refresh would overwrite it.
    if (destinationLevel !== 0 && destination.shape.generateMipmaps !== false)
      fail('LEVEL', 'Nonzero target destination mips require an authored, non-generated chain');
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
            {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
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

/** Source-shaped cube/array/volume color copies through existing target owners.
 * Non-cube destinations and authored cube mips copy one complete native range.
 * Cube level-zero writes retain the face-aware path. One range has one FIFO slot.
 */
export function captureTargetColorCopy(renderer, texture, destinationTexture,
  region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
  const targets = renderer._targets, device = renderer._device;
  if (!targets || !device) fail('DEVICE', 'Initialize the renderer and its target owner before copying');
  return captureTextureCopy(renderer, retain => {
    const source = targetColor(renderer, texture, retain), destination = targetColor(renderer, destinationTexture, retain);
    if (source.entry === destination.entry || source.entry.historyGroup && source.entry.historyGroup === destination.entry.historyGroup)
      fail('FEEDBACK', 'In-place target color copies are not admitted');
    if (destinationLevel !== 0 && destination.shape.generateMipmaps !== false)
      fail('LEVEL', 'Nonzero target destination mips require an authored, non-generated chain');
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
    // Reuse the raw transfer format/byte-budget checks in destination-mip
    // coordinates, without changing the actual allocation or owner metadata.
    const divisor = 2 ** destinationLevel, destinationShape = destination.entry.shape;
    const mipShape = destinationLevel === 0 ? destinationShape : {...destinationShape,
      width: Math.max(1, Math.floor(destinationShape.width / divisor)),
      height: Math.max(1, Math.floor(destinationShape.height / divisor)),
      layers: destinationShape.volume ? Math.max(1, Math.floor(destinationShape.layers / divisor)) : destinationShape.layers};
    if (!plan.empty) {
      inspectGpuTargetCopy(source.entry.shape, mipShape, step(0));
      inspectGpuTargetCopy(source.entry.shape, mipShape, step(count - 1));
    }
    return {empty: plan.empty, issue() {
      checkRenderer(renderer, device); source.check(); destination.check();
      if (plan.empty) return false;
      source.preflight(); destination.preflight();
      // Admit the combined allocation, not two individually affordable owners.
      // No layer may be written before all destination extents/budgets pass.
      targets.validateNativeTextures([source.source, destination.source]);
      if (!destination.entry.shape.cube || destinationLevel !== 0) {
        // One native copy covers every contiguous layer/slice. The destination
        // borrow marks the same level-zero mip chain and sampling replica dirty
        // as copyColor, without one submission/source borrow per array layer.
        // Authored higher mips share that ownership but never auto-regenerate.
        let resolve, reject;
        const validation = new Promise((yes, no) => { resolve = yes; reject = no; });
        validation.catch(() => {});
        try {
          targets.withNativeTexture(source.source, nativeSource =>
            targets.withNativeColorDestination(destination.source, nativeDestination => {
              const encoder = device.createCommandEncoder({label: 'f3d-target-color-range-copy'});
              encoder.copyTextureToTexture({texture: nativeSource, mipLevel: plan.sourceLevel, origin: plan.sourceOrigin},
                {texture: nativeDestination, mipLevel: plan.destinationLevel, origin: plan.destinationOrigin}, plan.size);
              const commands = encoder.finish();
              checkRenderer(renderer, device); source.check(); destination.check();
              device.queue.submit([commands]);
            }, {validation}));
          // Snapshot after the source borrow (including generated source mips),
          // before a subsequent reverse copy can depend on this destination.
          resolve(Promise.all([source.entry.validation, source.entry.storage.validation]));
          return true;
        } catch (error) { reject(error); throw error; }
      }
      for (let index = 0; index < count; index++) {
        checkRenderer(renderer, device); source.check(); destination.check();
        targets.copyColor(source.entry, destination.entry, step(index));
      }
      return true;
    }};
  });
}
