/** Ready-to-render canvas ownership for the admitted live Three.js scene path.
 * The source Scene and camera remain caller-owned. These are explicit async
 * factories, not WebGLRenderer/WebGPURenderer constructor replacements.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {GpuCanvasError} from './gpu_canvas.mjs';

export function createGpuThreeCanvas(canvas, scene, options = {}) {
  return createSourceCanvas(createGpuCanvasRenderer, canvas, scene, options);
}

/** Whole-image tone mapping over the existing linear scene profile. This is an
 * explicit output effect: every pixel (including backgrounds and materials with
 * toneMapped:false) is mapped. It is not source per-material tone-map parity.
 */
export function createGpuThreeHdrCanvas(canvas, scene, options = {}) {
  return createSourceCanvas(createGpuHdrCanvasRenderer, canvas, scene, options);
}

function createSourceCanvas(createCanvas, canvas, scene, options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw new GpuCanvasError('OPTIONS', 'Expected source canvas options');
  const {three, scene: sourceOptions = {}, ...canvasOptions} = options;
  if (!sourceOptions || typeof sourceOptions !== 'object' || Array.isArray(sourceOptions))
    throw new GpuCanvasError('OPTIONS', 'Expected source scene options');
  const source = {...sourceOptions};
  if (Object.hasOwn(source, 'three')) throw new GpuCanvasError('OPTIONS', 'Supply one top-level pinned three module');
  if (source.renderer !== undefined && (!source.renderer || typeof source.renderer !== 'object' || Array.isArray(source.renderer)))
    throw new GpuCanvasError('OPTIONS', 'Expected source renderer options');
  const renderer = {...source.renderer};
  for (const key of ['texture', 'geometry']) {
    if (source[key] === undefined) continue;
    if (!source[key] || typeof source[key] !== 'object' || Array.isArray(source[key]))
      throw new GpuCanvasError('OPTIONS', `Expected source ${key} options`);
    source[key] = {...source[key]};
  }
  // Attachment compatibility is fixed by the render target, not caller pipeline
  // state. Reject conflicts instead of silently changing their meaning.
  return createCanvas(canvas, (device, attachments) => {
    for (const [key, value] of Object.entries(attachments))
      if (renderer[key] !== undefined && renderer[key] !== value)
        throw new GpuCanvasError('FORMAT', `Source renderer ${key} differs from the render target`);
    return createGpuThreeScene(device, scene, {...source, three, renderer: {...renderer, ...attachments}});
  }, canvasOptions);
}
