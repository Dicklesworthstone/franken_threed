/** Real facade/target/readback code; this factory records pass inputs instead of executing shaders. */
import {passThree, passFixture} from './pass_snapshot_fixture.mjs';
import {targetThree, targetDevice} from './target_residency_fixture.mjs';
class Color { constructor(r = 0, g = 0, b = 0) { Object.assign(this, {r, g, b, isColor: true}); } set(c) { return this.copy(c?.isColor ? c : new Color()); } copy(c) { Object.assign(this, {r: c.r, g: c.g, b: c.b}); return this; } }
class Scene extends passThree.Object3D { isScene = true; background = null; environment = null; fog = null; overrideMaterial = null; }
export const T = {...passThree, ...targetThree, Color, Scene, PerspectiveCamera: passThree.Camera, NoToneMapping: 0, ACESFilmicToneMapping: 4, AgXToneMapping: 6, PCFShadowMap: 1};
export {passFixture};
export const createThreeProgramSupport = options => options;
export const createThreeProgramPMREM = () => {};
export const createThreeProgramShadows = () => {};
const boundary = () => Object.assign(new Error('prepare required'), {code: 'THREE_SCENE_PREPARE'});
async function gate(device, signal) {
  if (!device.prepareGate) return;
  let abort;
  try {
    await Promise.race([device.prepareGate, new Promise((_, reject) => {
      abort = () => reject(signal.reason); if (signal.aborted) abort(); else signal.addEventListener('abort', abort, {once: true});
    })]);
  } finally { signal.removeEventListener('abort', abort); }
}
export async function createGpuThreeScene(device, root, options) {
  let material, signature, texture, binding, disposed = false;
  const key = () => root.material?.vertexShader + '/' + root.material?.fragmentShader;
  const prepare = async () => {
    device.preparations.push({root, factor: root.material?.uniforms.factor?.value});
    await gate(device, options.signal);
    if (root.material?.fragmentShader === 'reject') throw new Error('rejected program');
    material = root.material; signature = key(); texture = material?.uniforms.input?.value;
    binding = texture ? options.textures.get(texture) : null;
  };
  await prepare();
  return {get disposed() { return disposed; }, failed: false, prepare,
    diagnostics: {sourceDraws: 1, drawCalls: 1, geometryCount: 1, geometryBytes: 60, instanceBytes: 0, deformationBytes: 0, rendererBytes: 128},
    render(camera, frame) {
      if (disposed) throw Error('disposed scene');
      if (device.retry) { device.retry--; throw boundary(); }
      const input = root.material?.uniforms.input?.value;
      if (root.material !== material || key() !== signature || input !== texture) throw boundary();
      const sampled = input ? options.textures.get(input) : null;
      if (sampled && sampled.view !== binding.view) throw boundary();
      const factor = root.material?.uniforms.factor?.value ?? 0, direction = root.material?.uniforms.direction?.value;
      const output = frame.resolveTarget ?? frame.colorView; output.texture.value = factor;
      const draw = {factor, direction: direction ? [direction.x, direction.y] : null, shader: root.material?.fragmentShader,
        root, material: root.material, geometry: root.geometry, position: root.geometry?.attributes.position.array[0],
        worldX: root.matrixWorld?.elements[12], cameraX: camera.matrixWorld?.elements[12], input, sampled, frame,
        program: options.program?.state()};
      device.draws.push(draw); device.queue.submit([{draw}]);
    },
    whenIdle: () => device.queue.onSubmittedWorkDone(), dispose() { disposed = true; },
  };
}
export async function createGpuCanvasRenderer(canvas, factory, {device}) {
  const lifetime = new AbortController(), child = factory(device, {format: 'rgba8unorm-srgb', depthFormat: 'depth24plus', sampleCount: 1}, {signal: lifetime.signal});
  return {device, prepare: () => child.prepare(), whenIdle: () => child.whenIdle(),
    render(input, frame) { canvas.acquired++; child.render(input, {...frame, colorView: {texture: canvas.texture}}); },
    resize(w, h) { canvas.width = w; canvas.height = h; }, dispose() { lifetime.abort(); child.dispose(); }};
}
export const createGpuHdrCanvasRenderer = createGpuCanvasRenderer;
export function fixture() {
  const f = targetDevice(); Object.assign(f.device, {draws: [], preparations: []});
  f.canvas = {width: 64, height: 32, style: {}, texture: {value: 0}, acquired: 0};
  f.copies = []; f.buffers = [];
  f.device.createBuffer = d => {
    const data = new ArrayBuffer(d.size), b = {destroyed: 0, data, unmap() {}, destroy() { this.destroyed++; },
      mapAsync: () => f.mapGate ?? Promise.resolve(), getMappedRange: () => data}; f.buffers.push(b); return b;
  };
  const submit = f.device.queue.submit;
  f.device.queue.submit = commands => {
    for (const c of commands) for (const copy of c.copies ?? []) {
      f.copies.push(copy); new Uint8Array(copy.target.buffer.data).fill(copy.source.texture.value ?? 0);
    }
    submit(commands);
  };
  return f;
}
export function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; }
