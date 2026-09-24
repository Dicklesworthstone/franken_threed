/** Owned WebGPU canvas attachments for the explicit rendering paths.
 * No frame loop, camera mutation, device ownership, or retained renderer.
 * withFrame() lends views and its second-argument presentation texture only
 * for synchronous, immediately submitted work.
 * Canvas textures are reacquired on every use, never retained across host turns.
 */
export class GpuCanvasError extends Error {
  constructor(code, message) {
    super(`GPU_CANVAS_${code}: ${message}`);
    this.name = 'GpuCanvasError';
    this.code = `GPU_CANVAS_${code}`;
  }
}
const fail = (code, message) => { throw new GpuCanvasError(code, message); };
const owners = new WeakMap();
const RENDER_ATTACHMENT = 16;

export function createGpuCanvasTarget(device, canvas, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    fail('OPTIONS', 'Expected canvas options');
  for (const key of Object.keys(options))
    if (!['format', 'depthFormat', 'sampleCount', 'alphaMode', 'maxBytes', 'width', 'height'].includes(key))
      fail('OPTIONS', `Unknown canvas option: ${key}`);
  const {
    format: canvasFormat = 'bgra8unorm', depthFormat = 'depth24plus', sampleCount = 1,
    alphaMode = 'opaque', maxBytes = 128 * 1024 * 1024,
    width: initialWidth = canvas?.width, height: initialHeight = canvas?.height,
  } = options;
  if (!['rgba8unorm', 'bgra8unorm'].includes(canvasFormat))
    fail('FORMAT', 'Canvas format must be rgba8unorm or bgra8unorm');
  if (![null, 'depth24plus', 'depth32float'].includes(depthFormat) || ![1, 4].includes(sampleCount))
    fail('FORMAT', 'Expected optional depth24plus/depth32float and one or four samples');
  // The existing explicit material renderer produces straight-alpha blending.
  // Transparent canvas compositing needs a separate output-alpha conversion.
  if (alphaMode !== 'opaque') fail('ALPHA', 'This direct canvas path requires opaque compositing');
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) fail('BUDGET', 'Invalid attachment budget');
  if (!canvas || typeof canvas.getContext !== 'function' || owners.has(canvas))
    fail('OWNERSHIP', 'Supply an exclusively owned canvas without another live canvas target');
  if (!device || typeof device.createTexture !== 'function' ||
      typeof device.pushErrorScope !== 'function' || typeof device.popErrorScope !== 'function' ||
      typeof device.queue?.onSubmittedWorkDone !== 'function') fail('DEVICE', 'Expected a WebGPU device');
  const limit = device.limits?.maxTextureDimension2D;
  if (!Number.isSafeInteger(limit) || limit < 1) fail('DEVICE', 'Missing device texture-dimension limit');
  const dimension = value => {
    if (!Number.isSafeInteger(value) || value < 0 || value > limit)
      fail('SIZE', 'Drawing-buffer dimensions must be nonnegative integers within device limits');
    return value;
  };
  dimension(initialWidth); dimension(initialHeight);
  const format = `${canvasFormat}-srgb`;
  const rendererOptions = Object.freeze({format, depthFormat, sampleCount});
  let context, configured = false, current = null, width = 0, height = 0;
  let disposed = false, terminal = null, busy = false, version = 0, frames = 0;
  let validation = Promise.resolve(), rejectStopped;
  const retired = new Set(), token = {};
  const stopped = new Promise((_, reject) => { rejectStopped = reject; });
  stopped.catch(() => {});
  const bytes = () => (current?.bytes ?? 0) + [...retired].reduce((n, t) => n + t.bytes, 0);
  const destroy = t => { t?.color?.destroy(); t?.depth?.destroy(); };
  function release() {
    destroy(current); current = null;
    for (const t of retired) destroy(t);
    retired.clear();
    // A late loss/validation completion must not unconfigure a replacement owner.
    if (owners.get(canvas) === token) {
      if (configured) context.unconfigure();
      configured = false;
      owners.delete(canvas);
    }
  }
  function stop(error) {
    if (disposed || terminal) return;
    terminal = error;
    release();
    rejectStopped(error);
  }
  function live() {
    if (disposed) fail('DISPOSED', 'Canvas target is disposed');
    if (terminal) throw terminal;
  }
  function exclusive(operation) {
    live();
    if (busy) fail('REENTRANT', 'Canvas ownership cannot change during a frame or resize');
    busy = true;
    try { return operation(); } finally { busy = false; }
  }
  function checked(operation) {
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    let result, thrown;
    try { result = operation(); } catch (error) { thrown = error; }
    const invalid = device.popErrorScope(), oom = device.popErrorScope();
    validation = Promise.all([validation, invalid, oom]).then(([, a, b]) => {
      if (a || b) throw new GpuCanvasError('GPU', (a || b).message ?? 'Native canvas validation failed');
    });
    validation.catch(stop);
    if (thrown) { stop(thrown); throw thrown; }
    return result;
  }
  function resize(w, h) {
    dimension(w); dimension(h);
    if (version && width === w && height === h && canvas.width === w && canvas.height === h) return target;
    // depth24plus uses a conservative four-byte texel charge. Swapchain and
    // driver-private allocations are not part of this owned-attachment budget.
    const cost = w * h * sampleCount * ((depthFormat ? 4 : 0) + (sampleCount > 1 ? 4 : 0));
    const overlap = bytes();
    if (!Number.isSafeInteger(cost) || cost > maxBytes - overlap)
      fail('BUDGET', 'Resize exceeds the attachment budget including in-flight old targets');
    const next = {color: null, depth: null, depthView: null, colorView: null, bytes: cost, used: false};
    checked(() => {
      try {
        const descriptor = {size: [w, h], sampleCount, usage: RENDER_ATTACHMENT};
        if (w && h) {
          if (sampleCount > 1) {
            next.color = device.createTexture({...descriptor, format, label: 'f3d-canvas-msaa'});
            next.colorView = next.color.createView();
          }
          if (depthFormat) {
            next.depth = device.createTexture({...descriptor, format: depthFormat, label: 'f3d-canvas-depth'});
            next.depthView = next.depth.createView();
          }
        }
        canvas.width = w; canvas.height = h;
      } catch (error) { destroy(next); throw error; }
    });
    const previous = current;
    current = next; width = w; height = h; version++;
    if (previous?.used) {
      retired.add(previous);
      // Fence only on structural resize, not in the ordinary frame hot path.
      let fence;
      try { fence = device.queue.onSubmittedWorkDone(); } catch (error) { stop(error); throw error; }
      previous.done = Promise.resolve(fence).then(() => {
        if (retired.delete(previous)) destroy(previous);
      });
      previous.done.catch(stop);
    } else destroy(previous);
    return target;
  }
  function withFrame(draw) {
    if (typeof draw !== 'function') fail('FRAME', 'Expected a synchronous draw function');
    if (canvas.width !== width || canvas.height !== height)
      fail('SIZE_CHANGED', 'Canvas dimensions changed outside the target; call resize() first');
    if (!width || !height) return false;
    let presentation;
    const attachments = checked(() => {
      const texture = context.getCurrentTexture();
      presentation = texture;
      const view = texture.createView({format});
      return Object.freeze({colorView: current.colorView ?? view,
        ...(current.depthView ? {depthView: current.depthView} : {}),
        ...(current.colorView ? {resolveTarget: view} : {})});
    });
    // A caller can submit before throwing: never retire this target as unused.
    current.used = true;
    const result = draw(attachments, presentation);
    if (result && typeof result.then === 'function') {
      Promise.resolve(result).catch(() => {});
      fail('ASYNC_FRAME', 'Frame views must be consumed and submitted synchronously; returned work is not cancelled');
    }
    frames++;
    return true;
  }
  const target = Object.freeze({
    device, canvas, rendererOptions, format, canvasFormat, depthFormat, sampleCount,
    resize(w, h) { return exclusive(() => resize(w, h)); },
    withFrame(draw) { return exclusive(() => withFrame(draw)); },
    get width() { return width; }, get height() { return height; },
    get suspended() { return !width || !height; },
    get disposed() { return disposed; }, get failed() { return terminal !== null; },
    get diagnostics() { return Object.freeze({version, frames, attachmentBytes: bytes(), retiredTargets: retired.size}); },
    async whenIdle() {
      live();
      if (busy) fail('REENTRANT', 'Cannot wait during a frame or resize');
      try {
        await Promise.race([Promise.all([validation, ...[...retired].map(t => t.done),
          current?.used ? device.queue.onSubmittedWorkDone() : undefined]), stopped]);
        live(); return target;
      } catch (error) { stop(error); throw error; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during a frame or resize');
      if (disposed) return;
      disposed = true;
      release();
      rejectStopped(new GpuCanvasError('DISPOSED', 'Canvas target is disposed'));
    },
  });
  owners.set(canvas, token);
  try {
    context = canvas.getContext('webgpu');
    if (!context || typeof context.configure !== 'function' || typeof context.unconfigure !== 'function' ||
        typeof context.getCurrentTexture !== 'function') fail('CONTEXT', 'The canvas cannot provide a WebGPU context');
    // Guard native operations just like subsequent resizes and submissions.
    exclusive(() => {
      checked(() => {
        configured = true;
        context.configure({device, format: canvasFormat, usage: RENDER_ATTACHMENT,
          alphaMode, colorSpace: 'srgb', viewFormats: [format]});
      });
      resize(initialWidth, initialHeight);
    });
    if (device.lost && typeof device.lost.then === 'function')
      device.lost.then(info => stop(new GpuCanvasError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    return target;
  } catch (error) { release(); throw error; }
}
