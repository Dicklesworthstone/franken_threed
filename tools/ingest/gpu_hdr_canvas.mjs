/** Opaque whole-image HDR canvas composition, not per-material tone mapping.
 * The factory supplies a synchronous linear-sRGB renderer; this owner renders
 * into rgba16float (optionally MSAA), then runs the existing output pass onto
 * the same-turn canvas texture. No readback, texture copy or frame loop.
 */
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {animationOutputSettings, createGpuAnimationOutput} from './animation_output.mjs';

export class GpuHdrCanvasError extends Error {
  constructor(code, message) {
    super(`GPU_HDR_${code}: ${message}`);
    this.name = 'GpuHdrCanvasError'; this.code = `GPU_HDR_${code}`;
  }
}
const fail = (code, message) => { throw new GpuHdrCanvasError(code, message); };
const capture = (value, allowed, label) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('OPTIONS', `Expected ${label}`);
  const copy = {...value};
  for (const key of Object.keys(copy)) if (!allowed.includes(key)) fail('OPTIONS', `Unknown ${label} option: ${key}`);
  return copy;
};
const OUTPUT_KEYS = ['toneMapping', 'exposure'];
const DEFAULTS = Object.freeze({toneMapping: 'aces-filmic', exposure: 1, inputAlpha: 'straight', outputAlpha: 'opaque'});

export async function createGpuHdrCanvasRenderer(canvas, createRenderer, options = {}) {
  const all = capture(options, ['device', 'gpu', 'powerPreference', 'requiredFeatures', 'requiredLimits',
    'target', 'signal', 'renderTarget', 'output'], 'HDR canvas');
  if (typeof createRenderer !== 'function') fail('FACTORY', 'Expected a linear-scene renderer factory');
  const {renderTarget = {}, output = {}, ...session} = all;
  const hdr = {depthFormat: 'depth24plus', sampleCount: 1, maxBytes: 256 * 1024 * 1024,
    ...capture(renderTarget, ['depthFormat', 'sampleCount', 'maxBytes', 'label'], 'HDR target')};
  if (![null, 'depth24plus', 'depth32float'].includes(hdr.depthFormat) || ![1, 4].includes(hdr.sampleCount))
    fail('OPTIONS', 'Expected optional depth24plus/depth32float and one or four scene samples');
  if (!Number.isSafeInteger(hdr.maxBytes) || hdr.maxBytes < 1 || (hdr.label !== undefined && typeof hdr.label !== 'string'))
    fail('OPTIONS', 'Invalid HDR target budget or label');
  const defaults = Object.freeze(animationOutputSettings(capture(output, OUTPUT_KEYS, 'output'), DEFAULTS));
  const target = capture(session.target ?? {}, ['format', 'depthFormat', 'sampleCount', 'alphaMode',
    'maxBytes', 'width', 'height'], 'presentation target');
  if ((target.depthFormat !== undefined && target.depthFormat !== null) ||
      (target.sampleCount !== undefined && target.sampleCount !== 1))
    fail('OPTIONS', 'Use renderTarget for scene depth/MSAA; the output pass is single-sample and depthless');
  session.target = {...target, depthFormat: null, sampleCount: 1};
  return createGpuCanvasRenderer(canvas, async (device, presentation, {signal}) => {
    let offscreen = null, finalOutput = null, renderer = null;
    let disposed = false, terminal = null, busy = false, preparing = false;
    let rejectStopped;
    const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
    const closed = () => disposed || terminal !== null;
    const wait = promise => Promise.race([promise, stopped]);
    const aborted = () => stop(signal.reason ?? new GpuHdrCanvasError('ABORTED', 'HDR owner stopped'));
    function release() {
      signal.removeEventListener('abort', aborted);
      const a = renderer, b = finalOutput, c = offscreen;
      renderer = finalOutput = offscreen = null;
      try { a?.dispose?.(); } finally { try { b?.dispose(); } finally { c?.dispose(); } }
    }
    function stop(error) {
      if (closed()) return;
      terminal = error; rejectStopped(error);
      if (!busy) release();
    }
    function live() {
      if (disposed) fail('DISPOSED', 'HDR composition is disposed');
      if (terminal !== null) throw terminal;
      if ([renderer, finalOutput, offscreen].some(value => value?.failed || value?.disposed)) {
        stop(new GpuHdrCanvasError('GPU', 'An owned HDR stage failed'));
        throw terminal;
      }
    }
    function childFailure(error) {
      if (renderer?.failed || finalOutput?.failed || offscreen?.failed) stop(error);
      throw error;
    }
    const api = Object.freeze({
      get disposed() { return disposed; },
      get failed() { return terminal !== null || !!renderer?.failed || !!finalOutput?.failed || !!offscreen?.failed; },
      get diagnostics() { return Object.freeze({scene: renderer?.diagnostics ?? null,
        hdr: offscreen?.diagnostics ?? null, outputSubmissions: finalOutput?.version ?? 0,
        outputBytes: finalOutput?.allocatedBytes ?? 0, defaults}); },
      render(input, frame, texture) {
        live(); if (busy || preparing) fail('REENTRANT', 'HDR rendering cannot overlap another operation');
        busy = true;
        try {
          const {output: selectedOutput = {}, colorView, depthView, resolveTarget, ...sceneFrame} = frame;
          const selected = animationOutputSettings(capture(selectedOutput, OUTPUT_KEYS, 'frame output'), defaults);
          if (depthView || resolveTarget) fail('ATTACHMENT', 'Final HDR presentation must be depthless and single-sample');
          // This profile renders a complete opaque scene before whole-image tone
          // mapping. It does not pretend to preserve selectively unmapped layers.
          if ((sceneFrame.loadOp ?? 'clear') !== 'clear' || (sceneFrame.depthLoadOp ?? 'clear') !== 'clear')
            fail('FRAME', 'HDR canvas frames require clear color/depth loads; use explicit targets for multipass loads');
          const clear = sceneFrame.clearColor ?? [0, 0, 0, 1];
          if ((!Array.isArray(clear) && !ArrayBuffer.isView(clear)) || clear.length !== 4)
            fail('FRAME', 'Expected an opaque RGBA clear color');
          sceneFrame.clearColor = Array.from(clear);
          if (sceneFrame.clearColor.some(v => typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) ||
              sceneFrame.clearColor[3] !== 1) fail('FRAME', 'HDR canvas clear alpha must be one');
          live();
          offscreen.resize(texture.width, texture.height);
          offscreen.withFrame((attachments, source) => {
            const result = renderer.render(input, {...sceneFrame, ...attachments});
            if (result && typeof result.then === 'function') {
              Promise.resolve(result).catch(() => {});
              fail('ASYNC_RENDER', 'Scene rendering must submit synchronously');
            }
            live();
            finalOutput.render({source, target: texture, ...selected});
          });
          live(); return api;
        } catch (error) { return childFailure(error); }
        finally { busy = false; if (closed()) release(); }
      },
      async prepare() {
        live(); if (busy || preparing) fail('REENTRANT', 'HDR preparation cannot overlap another operation');
        preparing = true;
        try { await wait(renderer.prepare?.()); live(); return api; }
        catch (error) { return childFailure(error); }
        finally { preparing = false; }
      },
      async whenIdle() {
        live(); if (busy || preparing) fail('REENTRANT', 'Cannot await HDR idle during another operation');
        try { await wait(Promise.all([renderer.whenIdle(), finalOutput.whenIdle(), offscreen.whenIdle()])); live(); return api; }
        catch (error) { stop(error); throw error; }
      },
      dispose() {
        if (busy) fail('REENTRANT', 'Cannot dispose inside HDR submission');
        if (disposed) return;
        disposed = true; rejectStopped(new GpuHdrCanvasError('DISPOSED', 'HDR owner disposed')); release();
      },
    });
    signal.addEventListener('abort', aborted, {once: true});
    try {
      if (signal.aborted) aborted(); live();
      offscreen = createGpuRenderTarget(device, {...hdr, width: canvas.width, height: canvas.height, format: 'rgba16float'});
      const outputReady = createGpuAnimationOutput(device, {...defaults, format: presentation.format, outputColorSpace: 'srgb', signal}).then(value => {
        if (closed()) { value.dispose(); throw terminal; }
        finalOutput = value; return value;
      });
      await wait(outputReady);
      live();
      const acquisition = Promise.resolve(createRenderer(device, offscreen.rendererOptions, {signal})).then(value => {
        if (closed()) { value?.dispose?.(); throw terminal; }
        renderer = value;
        if (!value || ['render', 'dispose', 'whenIdle'].some(key => typeof value[key] !== 'function'))
          fail('FACTORY', 'Scene factory must return render, dispose and whenIdle');
        return value;
      });
      await wait(acquisition); live();
      return api;
    } catch (error) { stop(error); throw error; }
  }, session);
}
