/** WebGL-authored output: pinned shader/clear oracles and recorded native
 * canvas lifetimes. These checks do not execute WGSL or establish pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {WebGLProgram} from '../../upstream/three.js/src/renderers/webgl/WebGLProgram.js';
import {WebGLBackground} from '../../upstream/three.js/src/renderers/webgl/WebGLBackground.js';
import {ColorManagement as sourceColorManagement} from '../../upstream/three.js/src/math/ColorManagement.js';
import {DisplayP3ColorSpace, LinearDisplayP3ColorSpace, DisplayP3ColorSpaceImpl,
  LinearDisplayP3ColorSpaceImpl} from '../../upstream/three.js/examples/jsm/math/ColorSpaces.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {threeProgramSources} from './three_program.mjs';
import {compileEsslProgram} from './essl_wgsl.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const registered = new WeakSet();
function spaces(t) {
  if (registered.has(t)) return;
  registered.add(t);
  const definitions = {[DisplayP3ColorSpace]: DisplayP3ColorSpaceImpl, [LinearDisplayP3ColorSpace]: LinearDisplayP3ColorSpaceImpl};
  for (const management of [T.ColorManagement, sourceColorManagement]) {
    const old = Object.fromEntries(Object.keys(definitions).map(key => [key, management.spaces[key]]));
    management.define(definitions);
    t.after(() => { for (const [key, value] of Object.entries(old)) {
      if (value === undefined) delete management.spaces[key]; else management.spaces[key] = value;
    } });
  }
}
function fixture(t, options = {}) {
  spaces(t);
  const device = geometryDevice(), textures = [], configurations = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      createView(view = {}) { return {texture: this, ...view}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => descriptor; device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const encoder = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const result = encoder(), copies = [];
    return {...result, copyTextureToTexture(source, destination, size) { copies.push({source, destination, size, draws: []}); },
      finish() { return [...result.finish(), ...copies]; }};
  };
  let configuration, unconfigured = 0;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; configurations.push(value); },
    unconfigure() { unconfigured++; },
    getCurrentTexture() { return Object.assign(device.createTexture({size: [32, 32], format: configuration.format, usage: configuration.usage}), {swapchain: true}); },
  })};
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    const first = device.snapshots.length; submit(commands);
    for (const draws of device.snapshots.slice(first)) for (const draw of draws) draw.canvasColorSpace = configuration?.colorSpace;
  };
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device, ...options});
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 1, .1, 100);
  const material = new T.MeshBasicMaterial({color: new T.Color(.12, .5, .8)}), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  mesh.position.z = -3; scene.add(mesh);
  t.after(() => { renderer.dispose(); material.dispose(); mesh.geometry.dispose(); });
  return {device, textures, configurations, renderer, scene, camera, material, mesh, get unconfigured() { return unconfigured; }};
}
function sourceProgram(parameters) {
  const source = {}, gl = {VERTEX_SHADER: 'vertex', FRAGMENT_SHADER: 'fragment', createProgram: () => ({}),
    createShader: type => ({type}), shaderSource(shader, code) { source[shader.type] = code; },
    compileShader() {}, attachShader() {}, bindAttribLocation() {}, linkProgram() {}};
  new WebGLProgram({getContext: () => gl}, 'output-space-source', parameters, {});
  return compileEsslProgram(source.vertex, source.fragment);
}
function sourceClear(color, alpha, space, premultiply) {
  let result;
  const background = WebGLBackground({getRenderTarget: () => null, outputColorSpace: space}, null,
    {buffers: {color: {setClear(r, g, b, a, multiply) {
      result = (multiply ? [r * a, g * a, b * a, a] : [r, g, b, a]).map(v => Math.max(0, Math.min(1, v)));
    }}}}, null, true, premultiply);
  background.setClearColor(color, alpha); background.dispose(); return result;
}
const draws = f => f.device.snapshots.flat();
const color = value => Array.isArray(value) ? value : [value.r, value.g, value.b, value.a];

for (const space of [T.LinearSRGBColorSpace, DisplayP3ColorSpace, LinearDisplayP3ColorSpace])
test(`${space} uses the unchanged source output program and its drawing-buffer gamut`, async t => {
  const f = fixture(t); f.renderer.outputColorSpace = space;
  await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f).at(-1);
  const parameters = threeProgramSources(T, f.material, f.mesh, {toneMapping: T.NoToneMapping, outputColorSpace: space}).parameters;
  const reference = sourceProgram(parameters);
  assert.equal(draw.pipeline.vertex.module.code, reference.vertex);
  assert.equal(draw.pipeline.fragment.module.code, reference.fragment);
  assert.equal(draw.canvasColorSpace, space === T.LinearSRGBColorSpace ? 'srgb' : 'display-p3');
  assert.equal(f.renderer._session.rendererOptions.format, 'bgra8unorm');
  assert.equal(f.renderer._hdr, false, 'conversion occurs in source material shaders before blending');
});

test('clear colors and solid backgrounds use the source output conversion and premultiplication', async t => {
  for (const premultipliedAlpha of [true, false]) {
    const f = fixture(t, {alpha: true, premultipliedAlpha});
    for (const space of [T.SRGBColorSpace, T.LinearSRGBColorSpace, DisplayP3ColorSpace, LinearDisplayP3ColorSpace]) {
      f.renderer.outputColorSpace = space;
      const clear = new T.Color(.1, .4, .8); f.renderer.setClearColor(clear, .375);
      assert.deepEqual(f.renderer._frame(f.scene).clearColor, sourceClear(clear, .375, space, premultipliedAlpha));
      f.scene.background = clear; await f.renderer.renderAsync(f.scene, f.camera);
      const pass = f.device.passes.filter(p => p.desc.label !== 'f3d-canvas-premultiply' &&
        p.desc.colorAttachments[0]?.loadOp === 'clear' && p.draws.length).at(-1);
      assert.deepEqual(color(pass.desc.colorAttachments[0].clearValue), sourceClear(clear, 1, space, premultipliedAlpha));
    }
  }
});

test('queued output changes retain each accepted shader space and canvas gamut', async t => {
  const f = fixture(t); await f.renderer.init();
  let release, arrive;
  const gate = new Promise(resolve => { release = resolve; }), entered = new Promise(resolve => { arrive = resolve; });
  const create = f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync = async descriptor => { arrive(); await gate; return create(descriptor); };
  const selected = [T.LinearSRGBColorSpace, DisplayP3ColorSpace, LinearDisplayP3ColorSpace, T.SRGBColorSpace];
  f.material.onBeforeCompile = () => {};
  for (const [index, space] of selected.entries()) {
    f.renderer.outputColorSpace = space; f.renderer.render(f.scene, f.camera);
    if (index === 0) await entered;
  }
  release(); await f.renderer.waitForGPU();
  assert.equal(draws(f).length, selected.length);
  for (const [i, space] of selected.entries()) {
    const parameters = threeProgramSources(T, f.material, f.mesh, {toneMapping: T.NoToneMapping, outputColorSpace: space}).parameters;
    assert.equal(draws(f)[i].pipeline.fragment.module.code, sourceProgram(parameters).fragment);
    assert.equal(draws(f)[i].canvasColorSpace, space.startsWith('display-p3') ? 'display-p3' : 'srgb');
  }
  assert.equal(f.renderer.outputColorSpace, T.SRGBColorSpace);
});

test('same-gamut transfer edits preserve history, while gamut round trips reset every attachment', async t => {
  const f = fixture(t, {preserveDrawingBuffer: true, stencil: true});
  f.renderer.outputColorSpace = DisplayP3ColorSpace;
  await f.renderer.renderAsync(f.scene, f.camera);
  const session = f.renderer._session, history = f.textures.findLast(texture => texture.label === 'f3d-canvas-preserved' && !texture.destroyed);
  const depth = f.textures.findLast(texture => texture.label === 'f3d-canvas-depth' && !texture.destroyed);
  f.renderer.autoClear = false; f.renderer.outputColorSpace = LinearDisplayP3ColorSpace;
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(f.renderer._session, session); assert.equal(history.destroyed, undefined); assert.equal(depth.destroyed, undefined);
  const before = f.unconfigured;
  f.renderer.outputColorSpace = T.LinearSRGBColorSpace;
  f.renderer.outputColorSpace = DisplayP3ColorSpace;
  await f.renderer.waitForGPU();
  assert.ok(history.destroyed); assert.ok(depth.destroyed); assert.ok(f.unconfigured > before);
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.notEqual(f.renderer._session, session);
  const initialize = f.device.passes.filter(pass => pass.desc.depthStencilAttachment?.depthClearValue === 1 &&
    pass.desc.colorAttachments[0]?.clearValue?.every?.(value => value === 0)).at(-1);
  assert.ok(initialize, 'a new drawing buffer starts with transparent black and depth one');
  assert.equal(initialize.desc.depthStencilAttachment.stencilClearValue, 0);
});

test('offscreen rendering keeps its working-linear output independently of canvas selection', async t => {
  const f = fixture(t), target = new T.WebGLRenderTarget(32, 32); t.after(() => target.dispose());
  f.renderer.outputColorSpace = DisplayP3ColorSpace; f.material.onBeforeCompile = () => {};
  f.renderer.setRenderTarget(target); await f.renderer.renderAsync(f.scene, f.camera);
  const support = f.renderer._targets.capture(target)._programSupport;
  assert.equal(support.state().outputColorSpace, T.LinearSRGBColorSpace); assert.equal(support.state().canvasOutput, false);
  const parameters = threeProgramSources(T, f.material, f.mesh, {toneMapping: T.NoToneMapping, outputColorSpace: T.LinearSRGBColorSpace}).parameters;
  assert.equal(draws(f).at(-1).pipeline.fragment.module.code, sourceProgram(parameters).fragment);
});

for (const hook of ['onBeforeRender', 'onAfterRender', 'onBeforeCompile', 'customProgramCacheKey'])
test(`output mutation inside ${hook} fails before changing the accepted selector or clearing history`, async t => {
  const f = fixture(t, {preserveDrawingBuffer: true});
  f.renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const receiver = hook.endsWith('Render') ? f.mesh : f.material;
  receiver[hook] = () => { f.renderer.outputColorSpace = DisplayP3ColorSpace; return 'changed'; };
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), /inside rendering callbacks requires exact renderer execution/);
  assert.equal(f.renderer.outputColorSpace, T.LinearSRGBColorSpace);
  assert.equal(f.renderer._drawingBufferColorSpace, 'srgb');
  assert.equal(f.renderer._programCompileCallbackDepth ?? 0, 0);
  assert.equal(f.renderer._webglSubmissionDepth ?? 0, 0);
});
