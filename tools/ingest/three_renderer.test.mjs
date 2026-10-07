/** Drop-in WebGPURenderer facade over real pinned r186 source objects and the
 * actual canvas/scene/renderer modules. GPU work goes to the byte-accurate
 * recording device: these tests prove command/queue behavior, not shader
 * execution, rasterized pixels or performance.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createWebGPURendererClass, createWebGLRendererClass, F3DRendererError} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
const WebGPURenderer = createWebGPURendererClass(T);

function canvasFixture() {
  const calls = [];
  let config;
  const context = {
    configure(d) { config = d; calls.push(['configure', d]); }, unconfigure() { calls.push(['unconfigure']); },
    getCurrentTexture() { const t = {width: canvas.width, height: canvas.height, dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, usage: 16, format: config?.format, swapchain: true, createView(d) { return {texture: t, ...d}; }}; calls.push(['acquire', t]); return t; },
  };
  const canvas = {width: 300, height: 150, style: {}, getContext() { return context; }};
  return {canvas, calls, acquired: () => calls.filter(c => c[0] === 'acquire').length};
}
function sceneFixture() {
  const scene = new T.Scene();
  const material = new T.MeshPhongMaterial({color: 0x3366cc});
  const mesh = new T.Mesh(new T.BoxGeometry(), material);
  scene.add(mesh, new T.AmbientLight(0xffffff, 1), new T.DirectionalLight(0xffffff, 2));
  const camera = new T.PerspectiveCamera(60, 2, 0.1, 100);
  camera.position.z = 4;
  return {scene, material, mesh, camera};
}
function recordingDevice() {
  const d = geometryDevice();
  Object.assign(d.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  d.textures = [];
  d.createTexture = desc => { const t = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc, width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1], destroyed: false, createView(o = {}) { return {attachment: t, ...o}; }, destroy() { t.destroyed = true; }}; d.textures.push(t); return t; };
  d.createSampler = desc => ({...desc});
  d.queue.writeTexture = () => {};
  d.queue.copyExternalImageToTexture = () => {};
  d.createRenderPipeline = desc => ({...desc, getBindGroupLayout: () => ({})});
  // Full-screen output passes draw without vertex streams; record them apart.
  d.outputPasses = [];
  const encode = d.createCommandEncoder;
  d.createCommandEncoder = () => {
    const e = encode();
    return {...e, beginRenderPass(desc) {
      if (!desc.colorAttachments[0]?.view?.texture?.swapchain || desc.depthStencilAttachment) {
        const pass = e.beginRenderPass(desc), record = d.passes.at(-1);
        return {...pass, setViewport(...v) { record.viewport = v; }, setScissorRect(...r) { record.scissor = r; }};
      }
      const pass = {desc, draws: []}; d.outputPasses.push(pass);
      return {setPipeline() {}, setBindGroup() {}, draw(n) { pass.draws.push(n); }, end() {}};
    }};
  };
  return d;
}
async function create(parameters = {}) {
  const device = recordingDevice(), c = canvasFixture();
  const renderer = new WebGPURenderer({canvas: c.canvas, device, ...parameters});
  return {renderer, device, ...c};
}
const flush = async renderer => { while (renderer._drain) await renderer._drain; };
const passes = d => d.passes;

test('source construction surface: render before init throws the source error', async () => {
  const {renderer, canvas} = await create();
  assert.equal(renderer.domElement, canvas); assert.equal(renderer.isWebGPURenderer, true);
  assert.equal(renderer.getClearAlpha(), 0); assert.equal(renderer.autoClear, true);
  assert.equal(renderer.outputColorSpace, T.SRGBColorSpace); assert.equal(renderer.toneMapping, T.NoToneMapping);
  const f = sceneFixture();
  assert.throws(() => renderer.render(f.scene, f.camera), /called before the backend is initialized/);
  renderer.setPixelRatio(2); renderer.setSize(200, 100);
  assert.equal(canvas.width, 400); assert.equal(canvas.height, 200); assert.equal(canvas.style.width, '200px');
  assert.deepEqual(renderer.getSize(new T.Vector2()).toArray(), [200, 100]);
  assert.deepEqual(renderer.getDrawingBufferSize(new T.Vector2()).toArray(), [400, 200]);
  assert.throws(() => new WebGPURenderer({forceWebGL: true}), {code: 'F3D_RENDERER_ROUTE'});
  // With the pinned upstream class supplied, a runtime WebGL request constructs it unchanged.
  const Routed = createWebGPURendererClass(T, {exactBackend: T.WebGPURenderer});
  globalThis.document ??= {createElementNS: () => ({width: 300, height: 150, style: {}, getContext() { return null; }})};
  const exact = new Routed({canvas: {width: 1, height: 1, style: {}, getContext() { return null; }}, forceWebGL: true});
  assert.ok(exact instanceof T.WebGPURenderer); assert.equal(exact.isF3DRenderer, undefined);
  assert.equal(new Routed({canvas: canvasFixture().canvas}).isF3DRenderer, true);
  assert.throws(() => new WebGPURenderer({canvas, bogus: 1}), {code: 'F3D_RENDERER_OPTIONS'});
});

test('ordinary first frame: render defers to preparation without presenting an empty texture', async () => {
  const {renderer, device, acquired, calls} = await create({antialias: true});
  await renderer.init();
  assert.equal(calls.find(c => c[0] === 'configure')[1].alphaMode, 'premultiplied');
  const f = sceneFixture();
  renderer.render(f.scene, f.camera);
  assert.equal(acquired(), 0, 'no swapchain texture is acquired before the scene is prepared');
  assert.equal(renderer.info.f3d.deferredRenders, 1);
  await flush(renderer);
  assert.equal(acquired(), 1); assert.equal(renderer.info.f3d.presentedRenders, 1);
  assert.equal(device.submissions.length > 0, true);
  // Steady state: immediate synchronous submission with live source values.
  for (let i = 0; i < 3; i++) {
    f.mesh.rotation.y = 0.3 * i; f.material.color.setRGB(0.1 * i, 0.2, 0.3);
    renderer.info.reset(); // Without setAnimationLoop the source leaves resets to the application.
    renderer.render(f.scene, f.camera);
  }
  assert.equal(acquired(), 4); assert.equal(renderer.info.f3d.deferredRenders, 1);
  assert.equal(renderer.info.render.drawCalls, 1);
  const pass = passes(device).at(-1).desc;
  // Default alpha:true clears with the premultiplied transparent clear color.
  assert.deepEqual(pass.colorAttachments[0].clearValue, {r: 0, g: 0, b: 0, a: 0});
  assert.equal(pass.colorAttachments[0].loadOp, 'clear');
  assert.ok(pass.colorAttachments[0].resolveTarget, 'antialias resolves into the presentation view');
  await renderer.waitForGPU(); renderer.dispose();
  assert.ok(calls.some(c => c[0] === 'unconfigure'));
});

test('clear color state, scene background and autoClear map to source clear semantics', async () => {
  const {renderer, device} = await create({alpha: false});
  const f = sceneFixture();
  await renderer.compileAsync(f.scene, f.camera);
  renderer.setClearColor(0xff0000, 0.5);
  renderer.render(f.scene, f.camera);
  const red = new T.Color(0xff0000);
  assert.deepEqual(passes(device).at(-1).desc.colorAttachments[0].clearValue, {r: red.r, g: red.g, b: red.b, a: 1});
  f.scene.background = new T.Color(0x00ff00);
  renderer.render(f.scene, f.camera);
  const green = new T.Color(0x00ff00);
  assert.deepEqual(passes(device).at(-1).desc.colorAttachments[0].clearValue, {r: green.r, g: green.g, b: green.b, a: 1});
  f.scene.background = null; renderer.autoClear = false;
  renderer.render(f.scene, f.camera);
  const last = passes(device).at(-1).desc;
  assert.equal(last.colorAttachments[0].loadOp, 'load'); assert.equal(last.depthStencilAttachment.depthLoadOp, 'load');
  assert.equal(renderer.info.f3d.deferredRenders, 0, 'compileAsync prepared ahead of the first frame');
  renderer.dispose();
});

test('structural edits defer only the affected frames and preserve application callbacks', async () => {
  const {renderer, acquired} = await create();
  const f = sceneFixture();
  await renderer.compileAsync(f.scene, f.camera);
  renderer.render(f.scene, f.camera);
  assert.equal(acquired(), 1);
  const added = new T.Mesh(new T.SphereGeometry(0.5), new T.MeshLambertMaterial({color: 0xffaa00}));
  f.scene.add(added);
  renderer.render(f.scene, f.camera);
  assert.equal(acquired(), 1, 'a frame needing preparation does not present');
  renderer.render(f.scene, f.camera); // Both calls remain observable, including callbacks and target dependencies.
  assert.equal(renderer.info.f3d.deferredRenders, 2);
  renderer.info.reset();
  await flush(renderer);
  assert.equal(acquired(), 3); assert.equal(renderer.info.render.drawCalls, 4);
  renderer.render(f.scene, f.camera);
  assert.equal(acquired(), 4); assert.equal(renderer.info.f3d.deferredRenders, 2);
  renderer.dispose();
});

test('feature profile changes rebuild the scene owner at the preparation boundary', async () => {
  const {renderer, acquired} = await create();
  const f = sceneFixture();
  await renderer.compileAsync(f.scene, f.camera);
  const before = renderer._dispatcher.entry(f.scene).bridge;
  f.scene.fog = new T.Fog(0x000000, 1, 10);
  renderer.render(f.scene, f.camera);
  assert.equal(acquired(), 0);
  await flush(renderer);
  const after = renderer._dispatcher.entry(f.scene).bridge;
  assert.notEqual(after, before); assert.equal(before.disposed, true); assert.equal(acquired(), 1);
  renderer.dispose();
});

test('unsupported source content surfaces as an explicit error instead of a silent omission', async () => {
  const {renderer} = await create();
  await renderer.init();
  const f = sceneFixture();
  f.mesh.material = new T.MeshMatcapMaterial();
  renderer.render(f.scene, f.camera);
  await flush(renderer).catch(() => {});
  assert.throws(() => renderer.render(f.scene, f.camera), /Unsupported source material/);
  renderer.setRenderTarget(null);
  assert.throws(() => renderer.setRenderTarget(new T.RenderTarget(4, 4, {stencilBuffer: true})), {code: 'THREE_TARGET_UNSUPPORTED'});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  assert.throws(() => renderer.render(f.scene, f.camera), F3DRendererError);
  renderer.dispose();
});

test('tone mapping selects the HDR output pass at initialization', async () => {
  const {renderer, device} = await create();
  renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.5;
  const f = sceneFixture();
  f.scene.background = new T.Color(0x202020);
  await renderer.compileAsync(f.scene, f.camera);
  renderer.render(f.scene, f.camera);
  assert.equal(device.outputPasses.length, 1, 'whole-image output pass');
  assert.ok(!device.passes.at(-1).desc.colorAttachments[0].view.texture?.swapchain, 'scene renders offscreen');
  // Default alpha:true: the transparent clear reaches a premultiplied output pass.
  f.scene.background = null;
  renderer.render(f.scene, f.camera);
  assert.deepEqual(device.passes.at(-1).desc.colorAttachments[0].clearValue, {r: 0, g: 0, b: 0, a: 0});
  renderer.dispose();
});

test('tone mapping can change after init: the output path is rebuilt on the same device', async () => {
  const {renderer, device} = await create();
  const f = sceneFixture();
  await renderer.compileAsync(f.scene, f.camera);
  renderer.render(f.scene, f.camera);
  assert.equal(device.outputPasses.length, 0);
  renderer.toneMapping = T.AgXToneMapping;
  renderer.render(f.scene, f.camera);
  await flush(renderer);
  assert.equal(device.outputPasses.length, 1, 'deferred frame presented through the output pass');
  renderer.toneMapping = T.NoToneMapping;
  renderer.render(f.scene, f.camera);
  await flush(renderer);
  renderer.render(f.scene, f.camera);
  assert.equal(device.outputPasses.length, 1, 'direct path again');
  renderer.dispose();
});

test('viewport, scissor and clear() follow source CanvasTarget semantics', async () => {
  const {renderer, device} = await create();
  renderer.setPixelRatio(2); renderer.setSize(100, 50);
  const f = sceneFixture();
  await renderer.compileAsync(f.scene, f.camera);
  renderer.setViewport(10, 5, 40, 20);
  renderer.setScissor(-5, 0, 200, 20); renderer.setScissorTest(true);
  renderer.render(f.scene, f.camera);
  assert.deepEqual(renderer.getViewport(new T.Vector4()).toArray(), [10, 5, 40, 20]);
  // Pixel-ratio scaling, top-left origin, scissor clamped into the drawing buffer.
  assert.deepEqual(device.passes.at(-1).viewport, [20, 10, 80, 40, 0, 1]);
  assert.deepEqual(device.passes.at(-1).scissor, [0, 0, 200, 40]);
  renderer.setSize(100, 50); // setSize resets the viewport to the full canvas
  assert.deepEqual(renderer.getViewport(new T.Vector4()).toArray(), [0, 0, 100, 50]);
  renderer.autoClear = false;
  renderer.setScissorTest(false);
  renderer.setClearColor(0x00ff00, 1);
  renderer.clear();
  await flush(renderer);
  const clearPass = device.passes.at(-1).desc;
  assert.equal(clearPass.colorAttachments[0].loadOp, 'clear');
  assert.equal(renderer.getMaxAnisotropy(), 16);
  renderer.dispose();
});

test('setAnimationLoop initializes, drives callbacks each host frame and stops on null', async () => {
  const {renderer, canvas} = await create();
  const frames = [];
  let next = null;
  canvas.ownerDocument = {defaultView: {requestAnimationFrame(cb) { next = cb; return 7; }, cancelAnimationFrame() { next = null; }}};
  const f = sceneFixture();
  await renderer.setAnimationLoop(time => { frames.push(time); renderer.render(f.scene, f.camera); });
  assert.equal(renderer.initialized, true);
  next(16); next(32);
  assert.deepEqual(frames, [16, 32]); assert.equal(renderer.info.frame, 2);
  await flush(renderer);
  next(48);
  assert.equal(renderer.info.f3d.presentedRenders, 3);
  await renderer.setAnimationLoop(null);
  assert.equal(next, null); assert.equal(renderer.getAnimationLoop(), null);
  renderer.dispose();
});

test('WebGLRenderer surface: synchronous construction, frames before init are deferred, GL-only state fails', async () => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const WebGLRenderer = createWebGLRendererClass(TC);
  const device = recordingDevice(), c = canvasFixture();
  const renderer = new WebGLRenderer({canvas: c.canvas, device, antialias: false, powerPreference: 'default'});
  assert.equal(renderer.isWebGLRenderer, true); assert.equal(renderer.isWebGPURenderer, false);
  assert.equal(renderer.getClearAlpha(), 1, 'source WebGL default alpha:false');
  const scene = new TC.Scene(), mesh = new TC.Mesh(new TC.BoxGeometry(), new TC.MeshBasicMaterial({color: 0xff0000}));
  scene.add(mesh);
  const camera = new TC.PerspectiveCamera(50, 2, 0.1, 10); camera.position.z = 3;
  renderer.render(scene, camera); // before the device exists: no throw, deferred
  await flush(renderer);
  assert.equal(c.acquired(), 1);
  assert.equal(c.calls.find(x => x[0] === 'configure')[1].alphaMode, 'opaque');
  // Bottom-left viewport origin, rounded like gl.viewport.
  renderer.setPixelRatio(1); renderer.setSize(100, 50);
  renderer.setViewport(10, 5, 40, 20);
  renderer.render(scene, camera);
  await flush(renderer);
  renderer.render(scene, camera);
  assert.deepEqual(device.passes.at(-1).viewport, [10, 25, 40, 20, 0, 1]);
  // Renderer-level clipping planes enable the bridge clipping profile.
  renderer.clippingPlanes = [new TC.Plane(new TC.Vector3(1, 0, 0), 0)];
  renderer.render(scene, camera);
  await flush(renderer);
  assert.equal(renderer._dispatcher.entry(scene).clipping, true);
  for (const name of ['extensions', 'state', 'properties'])
    assert.throws(() => renderer[name], {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(renderer.capabilities.getMaxAnisotropy(), 16);
  assert.throws(() => renderer.capabilities.maxTextureSize, {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.throws(() => renderer.getContext(), {code: 'F3D_RENDERER_UNSUPPORTED'});
  renderer.dispose();
});


test('source RenderTarget rendering and subsequent material sampling share owned GPU residency', async () => {
  const {renderer, acquired} = await create();
  const f = sceneFixture(), target = new T.RenderTarget(16, 8, {samples: 4});
  renderer.setRenderTarget(target);
  await renderer.compileAsync(f.scene, f.camera);
  renderer.render(f.scene, f.camera);
  assert.equal(acquired(), 0);
  const native = renderer._targets.capture(target);
  assert.equal(native.storage.rendererOptions.sampleCount, 4);
  const screen = new T.Scene();
  screen.add(new T.Mesh(new T.PlaneGeometry(2, 2), new T.MeshBasicMaterial({map: target.texture})));
  renderer.setRenderTarget(null);
  await renderer.compileAsync(screen, f.camera);
  renderer.render(screen, f.camera);
  assert.equal(acquired(), 1); assert.equal(native.copies, 1);
  const sampledView = native.view;
  renderer.setRenderTarget(target); renderer.render(f.scene, f.camera);
  renderer.setRenderTarget(null); renderer.render(screen, f.camera);
  assert.equal(native.view, sampledView); assert.equal(native.copies, 2);
  await renderer.waitForGPU(); target.dispose(); renderer.dispose();
});
