/** Owned WebGPU canvas attachments for the explicit rendering paths.
 * No frame loop, camera mutation, device ownership, or retained renderer.
 * withFrame() lends views, its second-argument presentation texture and third-
 * argument depth texture only for synchronous, immediately submitted work.
 * Lazy texture arguments are accessors; borrowing depth does not present color.
 * Canvas textures are reacquired on every use, never retained across host turns.
 * Optional frameHistory retains owned attachments for one non-null frameToken;
 * a new token starts a fresh buffer, while null keeps direct canvas borrowing.
 * Straight-alpha presentation keeps encoded drawing-buffer texels separate
 * from the premultiplied page image, including framebuffer-copy sources.
 */
import {subscribeGpuDeviceLoss} from './gpu_device_loss.mjs';

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
const COPY_SRC = 1;
const COPY_DST = 2;
const TEXTURE_BINDING = 4;
const PREMULTIPLY_PRESENTATION = `
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(positions[index], 0, 1);
}
@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let color = textureLoad(source, vec2i(position.xy), 0);
  return vec4f(color.rgb * color.a, color.a);
}`;

export function createGpuCanvasTarget(device, canvas, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    fail('OPTIONS', 'Expected canvas options');
  for (const key of Object.keys(options))
    if (!['format', 'depthFormat', 'sampleCount', 'alphaMode', 'maxBytes', 'width', 'height', 'srgbView', 'toneMappingMode', 'colorSpace', 'preserveDrawingBuffer', 'frameHistory', 'premultiplyPresentation'].includes(key))
      fail('OPTIONS', `Unknown canvas option: ${key}`);
  const {
    format: canvasFormat = 'bgra8unorm', depthFormat = 'depth24plus', sampleCount = 1,
    alphaMode = 'opaque', maxBytes = 128 * 1024 * 1024,
    width: initialWidth = canvas?.width, height: initialHeight = canvas?.height,
    srgbView = canvasFormat !== 'rgba16float', toneMappingMode = 'standard', preserveDrawingBuffer = false, frameHistory = false,
    premultiplyPresentation = false, colorSpace = 'srgb',
  } = options;
  if (typeof srgbView !== 'boolean') fail('OPTIONS', 'srgbView must be boolean');
  if (typeof preserveDrawingBuffer !== 'boolean') fail('OPTIONS', 'preserveDrawingBuffer must be boolean');
  if (typeof frameHistory !== 'boolean') fail('OPTIONS', 'frameHistory must be boolean');
  if (typeof premultiplyPresentation !== 'boolean') fail('OPTIONS', 'premultiplyPresentation must be boolean');
  if (!['rgba8unorm', 'bgra8unorm', 'rgba16float'].includes(canvasFormat) || canvasFormat === 'rgba16float' && srgbView)
    fail('FORMAT', 'Canvas format must be rgba8unorm, bgra8unorm or rgba16float without an sRGB view');
  if (!['standard', 'extended'].includes(toneMappingMode))
    fail('FORMAT', 'Canvas tone mapping must be standard or extended');
  if (!['srgb', 'display-p3'].includes(colorSpace))
    fail('FORMAT', 'Canvas display color space must be srgb or display-p3');
  if (![null, 'depth24plus', 'depth32float', 'depth24plus-stencil8'].includes(depthFormat) || ![1, 4].includes(sampleCount))
    fail('FORMAT', 'Expected optional depth24plus/depth32float/depth24plus-stencil8 and one or four samples');
  // 'premultiplied' composites the canvas over the page. The explicit renderer
  // blends color with (src-alpha, one-minus-src-alpha) and alpha with (one,
  // one-minus-src-alpha), so content over a zero-alpha clear is premultiplied in
  // linear space before the sRGB view encodes it -- the same order as the source
  // WebGPU output pass. Callers must clear with premultiplied colors.
  if (!['opaque', 'premultiplied'].includes(alphaMode)) fail('ALPHA', 'Canvas alpha mode must be opaque or premultiplied');
  if (premultiplyPresentation && (srgbView || canvasFormat === 'rgba16float' || alphaMode !== 'premultiplied'))
    fail('ALPHA', 'Straight-alpha presentation requires a plain 8-bit view and premultiplied canvas');
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) fail('BUDGET', 'Invalid attachment budget');
  if (!canvas || typeof canvas.getContext !== 'function' || owners.has(canvas))
    fail('OWNERSHIP', 'Supply an exclusively owned canvas without another live canvas target');
  if (!device || typeof device.createTexture !== 'function' ||
      typeof device.pushErrorScope !== 'function' || typeof device.popErrorScope !== 'function' ||
      typeof device.queue?.onSubmittedWorkDone !== 'function') fail('DEVICE', 'Expected a WebGPU device');
  if ((preserveDrawingBuffer || frameHistory || premultiplyPresentation) && (typeof device.createCommandEncoder !== 'function' || typeof device.queue.submit !== 'function'))
    fail('DEVICE', 'Drawing-buffer history requires native encoded copies');
  if (premultiplyPresentation && ['createShaderModule', 'createRenderPipeline', 'createBindGroup'].some(name => typeof device[name] !== 'function'))
    fail('DEVICE', 'Straight-alpha presentation requires native render pipelines');
  const limit = device.limits?.maxTextureDimension2D;
  if (!Number.isSafeInteger(limit) || limit < 1) fail('DEVICE', 'Missing device texture-dimension limit');
  const dimension = value => {
    if (!Number.isSafeInteger(value) || value < 0 || value > limit)
      fail('SIZE', 'Drawing-buffer dimensions must be nonnegative integers within device limits');
    return value;
  };
  dimension(initialWidth); dimension(initialHeight);
  // srgbView:false lends the plain 8-bit view; the renderer must then encode
  // sRGB itself (shader-side transfer, WebGL-style encoded blending).
  const format = srgbView ? `${canvasFormat}-srgb` : canvasFormat;
  const colorBytes = canvasFormat === 'rgba16float' ? 8 : 4;
  const rendererOptions = Object.freeze({format, depthFormat, sampleCount});
  let context, configured = false, current = null, width = 0, height = 0;
  let disposed = false, terminal = null, busy = false, version = 0, frames = 0;
  let validation = Promise.resolve(), rejectStopped, releaseDeviceLoss = null;
  let presentationPipeline;
  // getCurrentTexture returns the same object until browser presentation or
  // replacement. Weak keys identify direct drawing-buffer lifetimes without
  // retaining a swapchain texture or view across host turns.
  const presentationTokens = new WeakMap();
  const retired = new Set(), token = {};
  const stopped = new Promise((_, reject) => { rejectStopped = reject; });
  stopped.catch(() => {});
  const bytes = () => (current?.bytes ?? 0) + [...retired].reduce((n, t) => n + t.bytes, 0);
  const destroy = t => { t?.color?.destroy(); t?.depth?.destroy(); t?.preserved?.destroy(); };
  function release() {
    releaseDeviceLoss?.(); releaseDeviceLoss = null;
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
  function allocatePreserved(allocation, w, h) {
    allocation.preserved = device.createTexture({size: [w, h], sampleCount: 1,
      format: canvasFormat, viewFormats: srgbView ? [format] : [],
      usage: RENDER_ATTACHMENT | COPY_SRC | TEXTURE_BINDING, label: 'f3d-canvas-preserved'});
    allocation.preservedView = allocation.preserved.createView({format});
  }
  function resize(w, h) {
    dimension(w); dimension(h);
    if (version && width === w && height === h && canvas.width === w && canvas.height === h) return target;
    // Charge combined depth/stencil conservatively at eight bytes, as the
    // offscreen owner does; depth24plus alone uses four. Swapchain/driver
    // allocations are outside this owned-attachment budget.
    const depthBytes = depthFormat === 'depth24plus-stencil8' ? 8 : depthFormat ? 4 : 0;
    const cost = w * h * (sampleCount * (depthBytes + (sampleCount > 1 ? colorBytes : 0)) +
      (preserveDrawingBuffer ? colorBytes : 0));
    const overlap = bytes();
    if (!Number.isSafeInteger(cost) || cost > maxBytes - overlap)
      fail('BUDGET', 'Resize exceeds the attachment budget including in-flight old targets');
    const next = {color: null, depth: null, depthView: null, colorView: null,
      preserved: null, preservedView: null, initialized: false, frameToken: null, bytes: cost, used: false};
    checked(() => {
      try {
        const descriptor = {size: [w, h], sampleCount, usage: RENDER_ATTACHMENT};
        if (w && h) {
          if (sampleCount > 1) {
            next.color = device.createTexture({...descriptor, format, label: 'f3d-canvas-msaa'});
            next.colorView = next.color.createView();
          }
          if (depthFormat) {
            next.depth = device.createTexture({...descriptor, usage: RENDER_ATTACHMENT | COPY_SRC,
              format: depthFormat, label: 'f3d-canvas-depth'});
            next.depthView = next.depth.createView();
          }
          if (preserveDrawingBuffer) allocatePreserved(next, w, h);
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
  function withFrame(draw, {lazy = false, frameToken = null} = {}) {
    if (typeof draw !== 'function') fail('FRAME', 'Expected a synchronous draw function');
    if (typeof lazy !== 'boolean') fail('FRAME', 'Expected a boolean lazy acquisition flag');
    if (canvas.width !== width || canvas.height !== height)
      fail('SIZE_CHANGED', 'Canvas dimensions changed outside the target; call resize() first');
    if (!width || !height) return false;
    const history = preserveDrawingBuffer || frameHistory && frameToken !== null || premultiplyPresentation;
    let presentation, canvasTexture, views = null, historyToken = frameToken;
    const allocationBytes = () => {
      const extra = current.preserved ? 0 : width * height * colorBytes;
      if (!Number.isSafeInteger(extra) || extra > maxBytes - bytes())
        fail('BUDGET', 'Frame history exceeds the attachment budget including in-flight old targets');
      return extra;
    };
    const initialize = () => {
      if (!history) {
        // Direct uses may modify shared depth/MSAA attachments. A later history
        // borrow must not combine that state with an older owned color image.
        current.initialized = false; current.frameToken = null;
        return;
      }
      if (current.initialized && (preserveDrawingBuffer || current.frameToken === historyToken)) return;
      const extra = allocationBytes();
      // A fresh WebGL drawing buffer begins with transparent black, depth one
      // and stencil zero, independently of clear masks. A depth-only borrower
      // also needs these initialized contents, without acquiring presentation.
      // A callback token can span several browser presentation turns; only the
      // first actual borrow for its token starts a new drawing buffer.
      checked(() => {
        if (!current.preserved) { allocatePreserved(current, width, height); current.bytes += extra; }
        const encoder = device.createCommandEncoder({label: 'f3d-canvas-initialize'});
        const depth = current.depthView ? {view: current.depthView, depthLoadOp: 'clear',
          depthStoreOp: 'store', depthClearValue: 1,
          ...(depthFormat === 'depth24plus-stencil8'
            ? {stencilLoadOp: 'clear', stencilStoreOp: 'store', stencilClearValue: 0} : {})} : undefined;
        const pass = encoder.beginRenderPass({colorAttachments: [{view: current.colorView ?? current.preservedView,
          ...(current.colorView ? {resolveTarget: current.preservedView} : {}),
          clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store'}],
          ...(depth ? {depthStencilAttachment: depth} : {})});
        pass.end(); device.queue.submit([encoder.finish()]); current.initialized = true; current.frameToken = historyToken;
      });
    };
    const acquire = () => {
      if (premultiplyPresentation && !preserveDrawingBuffer && frameToken === null && !views) {
        allocationBytes(); // A refused allocation must not acquire a page image.
        canvasTexture = checked(() => context.getCurrentTexture());
        historyToken = presentationTokens.get(canvasTexture) ?? {};
      }
      initialize();
      views ??= checked(() => {
        if (history) {
          presentation = current.preserved;
          return {colorView: current.colorView ?? current.preservedView, depthView: current.depthView,
            resolveTarget: current.colorView ? current.preservedView : undefined};
        }
        const texture = context.getCurrentTexture();
        presentation = texture;
        const view = texture.createView({format});
        return {colorView: current.colorView ?? view, depthView: current.depthView, resolveTarget: current.colorView ? view : undefined};
      });
      // A caller can submit before throwing: never retire this target as unused.
      current.used = true;
      return views;
    };
    // Lazy attachments acquire the presentation texture only when a renderer
    // first reads a view. A draw that throws before that point (for example an
    // explicit preparation boundary) presents nothing, so the canvas keeps its
    // last presented image instead of compositing an empty swapchain texture.
    let attachments;
    if (lazy) {
      attachments = {get colorView() { return acquire().colorView; }};
      if (current.depthView) Object.defineProperty(attachments, 'depthView', {enumerable: true, get: () => acquire().depthView});
      if (current.colorView) Object.defineProperty(attachments, 'resolveTarget', {enumerable: true, get: () => acquire().resolveTarget});
      Object.freeze(attachments);
    } else {
      const v = acquire();
      attachments = Object.freeze({colorView: v.colorView,
        ...(v.depthView ? {depthView: v.depthView} : {}),
        ...(v.resolveTarget ? {resolveTarget: v.resolveTarget} : {})});
    }
    const depth = () => {
      // A direct depth-only borrow has no presentation lifetime to observe.
      // As on the direct canvas path, it never acquires a color image. Public
      // framebuffer copies use a full borrow and therefore resolve its epoch.
      if (!premultiplyPresentation || preserveDrawingBuffer || frameToken !== null) initialize();
      current.used = true; return current.depth;
    };
    try {
      const result = draw(attachments, lazy ? () => acquire() && presentation : presentation, lazy ? depth : current.depth);
      if (result && typeof result.then === 'function') {
        Promise.resolve(result).catch(() => {});
        fail('ASYNC_FRAME', 'Frame views must be consumed and submitted synchronously; returned work is not cancelled');
      }
    } finally {
      // Keep color/depth/MSAA storage owned across presentation boundaries.
      // The swapchain remains a same-turn borrow. Straight-alpha output applies
      // only the compositor conversion here, after source blending/quantization;
      // its owned drawing buffer and framebuffer copies remain unmodified.
      // Issued partial draws remain visible if a later source callback throws.
      if (views && history && current?.preserved && !terminal) checked(() => {
        const target = canvasTexture ?? context.getCurrentTexture(), encoder = device.createCommandEncoder({label: 'f3d-canvas-present'});
        if (premultiplyPresentation) {
          if (!presentationPipeline) {
            const module = device.createShaderModule({code: PREMULTIPLY_PRESENTATION, label: 'f3d-canvas-premultiply'});
            presentationPipeline = device.createRenderPipeline({label: 'f3d-canvas-premultiply', layout: 'auto',
              vertex: {module, entryPoint: 'vertex'}, fragment: {module, entryPoint: 'fragment', targets: [{format: canvasFormat}]},
              primitive: {topology: 'triangle-list'}});
          }
          current.presentationGroup ??= device.createBindGroup({layout: presentationPipeline.getBindGroupLayout(0),
            entries: [{binding: 0, resource: current.preservedView}]});
          const pass = encoder.beginRenderPass({label: 'f3d-canvas-premultiply', colorAttachments: [{
            view: target.createView({format: canvasFormat}), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0]}]});
          pass.setPipeline(presentationPipeline); pass.setBindGroup(0, current.presentationGroup); pass.draw(3); pass.end();
          presentationTokens.set(target, historyToken);
        } else encoder.copyTextureToTexture({texture: current.preserved}, {texture: target}, [width, height, 1]);
        device.queue.submit([encoder.finish()]);
      });
    }
    if (!views) return false;
    frames++;
    return true;
  }
  const target = Object.freeze({
    device, canvas, rendererOptions, format, canvasFormat, depthFormat, sampleCount,
    resize(w, h) { return exclusive(() => resize(w, h)); },
    withFrame(draw, options) { return exclusive(() => withFrame(draw, options)); },
    get alphaMode() { return alphaMode; },
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
        context.configure({device, format: canvasFormat, usage: RENDER_ATTACHMENT | COPY_SRC | TEXTURE_BINDING | (preserveDrawingBuffer || frameHistory ? COPY_DST : 0),
          alphaMode, colorSpace, viewFormats: srgbView ? [format] : [], toneMapping: {mode: toneMappingMode}});
      });
      resize(initialWidth, initialHeight);
    });
    if (device.lost && typeof device.lost.then === 'function')
      releaseDeviceLoss = subscribeGpuDeviceLoss(device.lost,
        info => stop(new GpuCanvasError('DEVICE_LOST', info?.message ?? 'Device lost')), stop);
    return target;
  } catch (error) { release(); throw error; }
}
