/** Device-loss reconstruction of the admitted source-owned Three.js canvas.
 * Reuse the live source Scene/module and reconstruct only device-local owners.
 * Native texture bindings cannot survive a device change. This convenience path
 * therefore uses owned source-texture residency; custom borrowed GPU bindings
 * need an application factory via createRecoverableGpuCanvasRenderer instead.
 */
import {createRecoverableGpuCanvasRenderer, createRecoverableGpuHdrCanvasRenderer} from './gpu_canvas_recovery.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {GpuCanvasError} from './gpu_canvas.mjs';

const fail = message => { throw new GpuCanvasError('OPTIONS', message); };
const copy = (value, label) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`Expected ${label} options`);
  return {...value};
};

export function createRecoverableGpuThreeCanvas(canvas, scene, options = {}) {
  return sourceCanvas(createRecoverableGpuCanvasRenderer, canvas, scene, options);
}

/** Retains the existing opaque whole-image tone-mapping profile. Recovery does
 * not add per-material toneMapped exclusions or preserve lost HDR history.
 */
export function createRecoverableGpuThreeHdrCanvas(canvas, scene, options = {}) {
  return sourceCanvas(createRecoverableGpuHdrCanvasRenderer, canvas, scene, options);
}

function sourceCanvas(createCanvas, canvas, scene, options) {
  const {three, scene: sourceOptions = {}, ...canvasOptions} = copy(options, 'source canvas');
  const source = copy(sourceOptions, 'source scene');
  if (Object.hasOwn(source, 'signal') || Object.hasOwn(source, 'three'))
    fail('Supply the lifetime signal and pinned three module at the top level');
  if (Object.hasOwn(source, 'textures') || source.autoTextures === false)
    throw new GpuCanvasError('RECOVERY_BINDINGS',
      'Recoverable source canvases require automatic textures; recreate borrowed bindings in an explicit renderer factory');
  if (source.autoTextures !== undefined && typeof source.autoTextures !== 'boolean')
    fail('Expected boolean automatic texture ownership');
  source.autoTextures = true;
  const renderer = source.renderer === undefined ? {} : copy(source.renderer, 'source renderer');
  for (const key of ['texture', 'geometry', 'deformation', 'shadow', 'environment', 'background']) {
    if (source[key] === undefined) continue;
    if (source[key] === null && ['shadow', 'environment', 'background'].includes(key)) continue;
    source[key] = Object.freeze(copy(source[key], 'source ' + key));
  }
  return createCanvas(canvas, (device, attachments, {signal}) => {
    for (const [key, value] of Object.entries(attachments))
      if (renderer[key] !== undefined && renderer[key] !== value)
        throw new GpuCanvasError('FORMAT', `Source renderer ${key} differs from the render target`);
    // The source graph is intentionally live, not a snapshot of the first
    // generation: recovery observes its current geometry, pixels and poses.
    return createGpuThreeScene(device, scene, {...source, three, signal,
      renderer: {...renderer, ...attachments}});
  }, canvasOptions);
}
