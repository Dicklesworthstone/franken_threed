/** Ready-to-render canvas ownership for the admitted live Three.js scene path.
 * The source Scene and camera remain caller-owned. This is an explicit async
 * factory, not a WebGLRenderer/WebGPURenderer constructor replacement.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {GpuCanvasError} from './gpu_canvas.mjs';

export function createGpuThreeCanvas(canvas, scene, options = {}) {
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
  // Attachment compatibility is fixed by the canvas, not caller pipeline state.
  // Reject conflicting values instead of silently changing their meaning.
  return createGpuCanvasRenderer(canvas, (device, attachments) => {
    for (const [key, value] of Object.entries(attachments))
      if (renderer[key] !== undefined && renderer[key] !== value)
        throw new GpuCanvasError('FORMAT', `Source renderer ${key} differs from the canvas target`);
    return createGpuThreeScene(device, scene, {...source, three, renderer: {...renderer, ...attachments}});
  }, canvasOptions);
}
