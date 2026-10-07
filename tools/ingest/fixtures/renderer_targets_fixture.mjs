/** Scene/canvas factory doubles: this tests the actual renderer's public API,
 * target owner, GPU descriptors and queue sequencing, not rasterized pixels. */
import {targetDevice, targetThree} from './target_residency_fixture.mjs';
class Object3D {
  name = '';
  children = [];
  traverse(fn) { fn(this); for (const c of this.children) c.traverse(fn); }
}
class Scene extends Object3D {
  isScene = true; background = null; fog = null; environment = null; overrideMaterial = null;
}
class Camera extends Object3D {}
class Color {
  isColor = true;
  constructor(r = 0, g = 0, b = 0) { Object.assign(this, {r, g, b}); }
  set(c) { return this.copy(c?.isColor ? c : new Color((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)); }
  copy(c) { Object.assign(this, {r: c.r, g: c.g, b: c.b}); return this; }
}
class Vector2 { set(x, y) { Object.assign(this, {x, y}); return this; } }
export const T = {...targetThree, Scene, Object3D, Camera, PerspectiveCamera: Camera, Color, Vector2,
  WebGPUCoordinateSystem: 2001, NoToneMapping: 0, ACESFilmicToneMapping: 4, AgXToneMapping: 6, PCFShadowMap: 1};
export const createThreeProgramSupport = options => options;
export const createThreeProgramPMREM = () => {};
export const createThreeProgramShadows = () => {};
export async function createGpuThreeScene(device, scene, options) {
  let disposed = false, binding;
  const prepare = async () => {
    if (scene.gate) await Promise.race([scene.gate, new Promise((_, reject) => {
      if (options.signal.aborted) reject(options.signal.reason);
      else options.signal.addEventListener('abort', () => reject(options.signal.reason), {once: true});
    })]);
    if (scene.failPrepare) throw scene.failPrepare;
    if (scene.requiresHdr && options.program) throw Object.assign(new Error('needs HDR fallback'), {code: 'THREE_SCENE_TONE_MAPPING'});
    if (scene.sample) binding = options.textures.get(scene.sample);
    scene.prepared = (scene.prepared ?? 0) + 1;
  };
  await prepare();
  return {
    get disposed() { return disposed; }, failed: false, prepare,
    diagnostics: {sourceDraws: 1, drawCalls: 1, geometryCount: 1, geometryBytes: 12, instanceBytes: 0, deformationBytes: 0, rendererBytes: 16},
    render(camera, frame) {
      if (disposed) throw Error('disposed bridge');
      const sampled = scene.sample ? options.textures.get(scene.sample) : null;
      if (sampled && (sampled.view !== binding.view || sampled.sampler !== binding.sampler))
        throw Object.assign(new Error('texture changed'), {code: 'THREE_SCENE_PREPARE'});
      scene.onDraw?.();
      const output = frame.resolveTarget ?? frame.colorView;
      output.texture.value = scene.value ?? 0;
      device.queue.submit([{scene: scene.name, frame, sampled, program: options.program?.state(), renderOptions: options.renderer}]);
    },
    whenIdle: () => device.queue.onSubmittedWorkDone(), dispose() { disposed = true; },
  };
}
export async function createGpuCanvasRenderer(canvas, factory, {device, target}) {
  const lifetime = new AbortController();
  const child = factory(device, {format: 'rgba8unorm-srgb', depthFormat: 'depth24plus', sampleCount: 1}, {signal: lifetime.signal});
  return {device,
    render(input, frame) { canvas.acquired++; child.render(input, {...frame, colorView: {texture: canvas.texture}}); },
    prepare: () => child.prepare(), whenIdle: () => child.whenIdle(),
    resize(w, h) { canvas.width = w; canvas.height = h; },
    dispose() { lifetime.abort(); child.dispose(); canvas.closed++; }};
}
export const createGpuHdrCanvasRenderer = createGpuCanvasRenderer;
export function fixture() {
  const f = targetDevice();
  const canvas = {width: 300, height: 150, style: {}, acquired: 0, closed: 0, texture: {value: 0}, getContext() { return {}; }};
  const buffers = [], copies = [], draws = [];
  f.device.createBuffer = desc => {
    const data = new ArrayBuffer(desc.size);
    const b = {data, destroyed: 0, unmap() {}, destroy() { b.destroyed++; },
      mapAsync() { return f.mapGate ?? Promise.resolve(); }, getMappedRange() { return data; }};
    buffers.push(b); return b;
  };
  const submit = f.device.queue.submit;
  f.device.queue.submit = commands => {
    for (const command of commands) {
      if (command.scene !== undefined) draws.push(command);
      for (const copy of command.copies ?? []) {
        copies.push(copy);
        new Uint8Array(copy.target.buffer.data).fill(copy.source.texture.value ?? 0);
      }
    }
    submit(commands);
  };
  return Object.assign(f, {canvas, buffers, copies, draws});
}
export function scene(name, value = 0) { return Object.assign(new T.Scene(), {name, value}); }
export function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; }
