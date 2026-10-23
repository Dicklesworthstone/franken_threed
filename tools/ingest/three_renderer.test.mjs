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
    renderer.info.reset(); // These synchronous calls occur in the same host frame.
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
  f.mesh.material = new T.MeshDepthMaterial();
  renderer.render(f.scene, f.camera);
  await flush(renderer).catch(() => {});
  assert.throws(() => renderer.render(f.scene, f.camera), /Unsupported source material/);
  renderer.setRenderTarget(null);
  assert.throws(() => renderer.setRenderTarget(new T.RenderTarget(4, 4, {samples: 2})), {code: 'THREE_TARGET_UNSUPPORTED'});
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

test('native initialization starts the frame clock and null stops callbacks without stopping node updates', async () => {
  const {renderer, canvas} = await create();
  const frames = [];
  let next = null;
  canvas.ownerDocument = {defaultView: {requestAnimationFrame(cb) { next = cb; return 7; }, cancelAnimationFrame() { next = null; }}};
  const f = sceneFixture();
  await renderer.setAnimationLoop(time => { frames.push(time); renderer.render(f.scene, f.camera); });
  assert.equal(renderer.initialized, true);assert.equal(renderer.info.frame,1);
  next(16); await flush(renderer); next(32);
  assert.deepEqual(frames, [16, 32]); assert.equal(renderer.info.frame, 3);
  await flush(renderer);
  next(48);
  assert.equal(renderer.info.f3d.presentedRenders, 3);
  await renderer.setAnimationLoop(null);
  assert.equal(typeof next,'function');assert.equal(renderer.getAnimationLoop(), null);
  next(64);assert.deepEqual(frames,[16,32,48]);assert.equal(renderer.info.frame,5);
  const staleTick=next;renderer.dispose();assert.equal(next,null);
  staleTick(80);assert.equal(renderer.info.frame,5,'an already scheduled callback cannot advance a disposed owner');
});

function animationHost(canvas) {
  const scheduled = new Map(); let serial = 0;
  canvas.ownerDocument = {defaultView: {
    requestAnimationFrame(callback) { const id = ++serial; scheduled.set(id, callback); return id; },
    cancelAnimationFrame(id) { scheduled.delete(id); }
  }};
  return {scheduled, step(time) {
    assert.equal(scheduled.size, 1, 'one source animation tick is scheduled');
    const [id, callback] = scheduled.entries().next().value;
    scheduled.delete(id); callback(time);
  }};
}

test('native animation waits for cold submission and resumes before GPU completion without losing accepted work', async t => {
  const {renderer, device, canvas, acquired} = await create(), host = animationHost(canvas);
  const frames = [], f = sceneFixture();
  let releasePipeline, pipelineStarted, finishGPU;
  const gate = new Promise(resolve => { releasePipeline = resolve; });
  const started = new Promise(resolve => { pipelineStarted = resolve; });
  device.completion = new Promise(resolve => { finishGPU = resolve; });
  t.after(() => { releasePipeline(); finishGPU(); renderer.dispose(); });
  await renderer.init();
  const createPipeline = device.createRenderPipelineAsync.bind(device);
  device.createRenderPipelineAsync = async descriptor => {
    pipelineStarted(); await gate; return createPipeline(descriptor);
  };
  await renderer.setAnimationLoop(time => { frames.push(time); renderer.render(f.scene, f.camera); });
  host.step(16); await started;
  assert.equal(renderer._pending.length, 1); assert.equal(acquired(), 0);
  host.step(32);
  assert.equal(host.scheduled.size, 0, 'preparation suspends admission of further animation frames');
  assert.deepEqual(frames, [16]); assert.equal(renderer.info.frame, 2);
  assert.equal(renderer._pending.length, 1, 'the accepted draw remains in its original FIFO slot');
  await renderer.setAnimationLoop(time => { frames.push(time * 10); renderer.render(f.scene, f.camera); });
  releasePipeline(); await flush(renderer);
  assert.equal(acquired(), 1, 'the accepted cold frame is submitted before resuming');
  assert.equal(host.scheduled.size, 1, 'GPU completion is still pending but submission can resume animation');
  host.step(48);
  assert.deepEqual(frames, [16, 480], 'resume uses the replacement callback and fresh host time');
  assert.equal(renderer.info.frame, 3); assert.equal(acquired(), 2);
  await renderer.setAnimationLoop(null); host.step(64);
  assert.equal(renderer.info.frame, 4, 'null callbacks preserve the source internal frame clock');
  assert.deepEqual(frames, [16, 480]);
});

test('native animation resumes after a preparation failure and disposal makes a waiting resume inert', async t => {
  for (const dispose of [false, true]) {
    const {renderer, device, canvas} = await create(), host = animationHost(canvas), f = sceneFixture();
    let releasePipeline, pipelineStarted;
    const gate = new Promise(resolve => { releasePipeline = resolve; });
    const started = new Promise(resolve => { pipelineStarted = resolve; });
    t.after(() => { releasePipeline(); renderer.dispose(); });
    await renderer.init();
    const failure = new Error('cold pipeline failed');
    device.createRenderPipelineAsync = async () => { pipelineStarted(); await gate; throw failure; };
    await renderer.setAnimationLoop(() => renderer.render(f.scene, f.camera));
    host.step(16); await started; host.step(32);
    const pending = renderer._drain;
    if (dispose) renderer.dispose();
    releasePipeline(); await pending.catch(() => {});
    if (dispose) {
      assert.equal(host.scheduled.size, 0); assert.equal(renderer.info.frame, 2);
    } else {
      assert.equal(host.scheduled.size, 1, 'a failed drain does not silently stop the source loop');
      assert.throws(() => host.step(48), /cold pipeline failed/);
      assert.equal(host.scheduled.size, 1, 'the next source tick was scheduled before the callback threw');
      renderer.dispose(); assert.equal(host.scheduled.size, 0);
    }
  }
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
  // compile(): the scene's materials, synchronously; preparation starts now.
  const compiled = renderer.compile(scene, camera);
  assert.ok(compiled instanceof Set && compiled.size === 1 && compiled.has(mesh.material));
  renderer.render(scene, camera); // before the device exists: no throw, deferred
  await flush(renderer);
  assert.equal(c.acquired(), 1);
  // r186 WebGLRenderer always creates an alpha:true context (alpha only sets the clear alpha).
  assert.equal(c.calls.find(x => x[0] === 'configure')[1].alphaMode, 'premultiplied');
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
  for (const name of ['extensions', 'properties'])
    assert.throws(() => renderer[name], {code: 'F3D_RENDERER_UNSUPPORTED'});
  // WebGLState: only the true depth facts (CubeCamera/PMREM/Reflector reads).
  assert.equal(renderer.state.buffers.depth.getReversed(), false);
  renderer.state.buffers.depth.setMask(true);
  assert.throws(() => renderer.state.buffers.depth.setMask(false), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.throws(() => renderer.state.buffers.stencil, {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.throws(() => renderer.state.viewport, {code: 'F3D_RENDERER_UNSUPPORTED'});
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

for (const antialias of [false, true]) test(`canvas stencil supports native materials and independent automatic clears (MSAA ${antialias})`, async () => {
  const {renderer, device} = await create({stencil: true, antialias}), f = sceneFixture();
  Object.assign(f.material, {stencilWrite: true, stencilRef: 5, stencilZPass: T.ReplaceStencilOp});
  try {
    await renderer.compileAsync(f.scene, f.camera); renderer.setClearStencil(5);
    renderer.render(f.scene, f.camera);
    const first = device.passes.at(-1);
    assert.equal(first.desc.depthStencilAttachment.view.attachment.format, 'depth24plus-stencil8');
    assert.equal(first.desc.depthStencilAttachment.view.attachment.sampleCount, antialias ? 4 : 1);
    assert.equal(first.desc.depthStencilAttachment.stencilLoadOp, 'clear');
    assert.equal(first.desc.depthStencilAttachment.stencilClearValue, 5);
    assert.equal(device.snapshots.at(-1)[0].stencilReference, 5);
    renderer.autoClearStencil = false; renderer.render(f.scene, f.camera);
    const preserved = device.passes.at(-1).desc;
    assert.equal(preserved.colorAttachments[0].loadOp, 'clear');
    assert.equal(preserved.depthStencilAttachment.depthLoadOp, 'clear');
    assert.equal(preserved.depthStencilAttachment.stencilLoadOp, 'load');
    renderer.autoClear = false; renderer.autoClearStencil = true; renderer.render(f.scene, f.camera);
    const loaded = device.passes.at(-1).desc;
    assert.equal(loaded.colorAttachments[0].loadOp, 'load');
    assert.equal(loaded.depthStencilAttachment.depthLoadOp, 'load');
    assert.equal(loaded.depthStencilAttachment.stencilLoadOp, 'load');
  } finally { renderer.dispose(); }
});

test('manual clear methods independently preserve color, depth and stencil', async () => {
  const {renderer, device} = await create({stencil: true}); await renderer.init();
  try {
    renderer.autoClear = false; renderer.setClearStencil(31);
    for (const [method, expected] of [['clearStencil', ['load', 'load', 'clear']],
      ['clearColor', ['clear', 'load', 'load']], ['clearDepth', ['load', 'clear', 'load']],
      ['clear', ['clear', 'clear', 'clear']], ['clearStencilAsync', ['load', 'load', 'clear']],
      ['clearColorAsync', ['clear', 'load', 'load']], ['clearDepthAsync', ['load', 'clear', 'load']],
      ['clearAsync', ['clear', 'clear', 'clear']]]) {
      await renderer[method](); await flush(renderer);
      const pass = device.passes.at(-1).desc;
      assert.deepEqual([pass.colorAttachments[0].loadOp, pass.depthStencilAttachment.depthLoadOp,
        pass.depthStencilAttachment.stencilLoadOp], expected);
      if (expected[2] === 'clear') assert.equal(pass.depthStencilAttachment.stencilClearValue, 31);
    }
    renderer.clear(true, true, false); await flush(renderer);
    assert.equal(device.passes.at(-1).desc.depthStencilAttachment.stencilLoadOp, 'load');
    const before = device.submissions.length; renderer.clear(false, false, false);
    assert.equal(device.submissions.length, before);
  } finally { renderer.dispose(); }
});

for (const method of ['clearStencil', 'clearStencilAsync']) test(`queued ${method} captures destination, value and masks before later calls`, async () => {
  const {renderer, device, acquired} = await create({stencil: true}); await renderer.init();
  const a = new T.RenderTarget(16, 8, {stencilBuffer: true}), b = new T.RenderTarget(16, 8, {stencilBuffer: true});
  try {
    renderer.setRenderTarget(a); renderer.setClearStencil(11); const first = renderer[method]();
    renderer.setRenderTarget(b); renderer.setClearStencil(22); const second = renderer[method]();
    renderer.setRenderTarget(null); renderer.setClearStencil(33); const third = renderer[method]();
    renderer.setClearStencil(99); await Promise.all([first, second, third]); await flush(renderer);
    const calls = device.passes.map(p => p.desc);
    assert.deepEqual(calls.map(p => p.depthStencilAttachment.stencilClearValue), [11, 22, 33]);
    assert.ok(calls.every(p => p.colorAttachments[0].loadOp === 'load' && p.depthStencilAttachment.depthLoadOp === 'load'));
    assert.notEqual(calls[0].depthStencilAttachment.view.attachment, calls[1].depthStencilAttachment.view.attachment);
    assert.equal(acquired(), 1); assert.equal(renderer.getClearStencil(), 99);
  } finally { a.dispose(); b.dispose(); renderer.dispose(); }
});

test('scissored stencil clears use a bounded native clear draw and retain other aspects', async () => {
  const {renderer, device} = await create({stencil: true}); await renderer.init();
  try {
    renderer.setScissor(4, 5, 20, 10); renderer.setScissorTest(true); renderer.setClearStencil(17);
    renderer.clearStencil(); await flush(renderer);
    const clear = device.passes.at(-2), main = device.passes.at(-1);
    assert.deepEqual(clear.scissor, [4, 5, 20, 10]);
    assert.equal(clear.desc.colorAttachments[0].loadOp, 'load');
    assert.equal(clear.desc.depthStencilAttachment.depthLoadOp, 'load');
    assert.equal(clear.desc.depthStencilAttachment.stencilLoadOp, 'load');
    assert.equal(clear.draws[0].pipeline.fragment.targets[0].writeMask, 0);
    assert.equal(clear.draws[0].pipeline.depthStencil.depthWriteEnabled, false);
    assert.equal(clear.draws[0].pipeline.depthStencil.stencilFront.passOp, 'replace');
    assert.equal(clear.draws[0].stencilReference, 17);
    assert.equal(main.desc.depthStencilAttachment.stencilLoadOp, 'load');
  } finally { renderer.dispose(); }
});

test('stencil clear values preserve public state and ignore dormant attachment controls', async () => {
  const {renderer, device} = await create({stencil: true}); await renderer.init();
  try {
    renderer.setClearStencil(300); await renderer.clearStencilAsync();
    assert.equal(device.passes.at(-1).desc.depthStencilAttachment.stencilClearValue, 44);
    assert.equal(renderer.getClearStencil(), 300);
    renderer.setClearStencil(-1); const effects = device.submissions.length;
    assert.throws(() => renderer.clearStencil(), {code: 'F3D_RENDERER_MATERIAL'});
    assert.equal(device.submissions.length, effects);
    const target = new T.RenderTarget(8, 8);
    renderer.setRenderTarget(target); renderer.clearStencil();
    assert.equal(device.submissions.length, effects, 'a target without stencil has no stencil clear');
    renderer.clearDepth(); await flush(renderer);
    assert.equal(device.passes.at(-1).desc.depthStencilAttachment.stencilLoadOp, undefined);
    target.dispose();
  } finally { renderer.dispose(); }
});

test('clear initialization and asynchronous stencil completion follow the source API', async () => {
  const {renderer, device} = await create({stencil: true});
  try {
    assert.throws(() => renderer.clearStencil(), /clear\(\) called before the backend is initialized/);
    await assert.rejects(renderer.clearStencilAsync(), /clear\(\) called before the backend is initialized/);
    await renderer.clearAsync(false, false, true);
    assert.equal(device.passes.at(-1).desc.depthStencilAttachment.stencilLoadOp, 'clear');
    assert.equal(renderer._drain, null, 'the async clear resolves after its deferred submission');
    renderer.setClearStencil(12); await renderer.clearStencilAsync();
    assert.equal(device.passes.at(-1).desc.depthStencilAttachment.stencilClearValue, 12);
  } finally { renderer.dispose(); }
});

test('an asynchronous clear reports a preceding deferred failure before claiming success', async () => {
  const {renderer, device} = await create({stencil: true}); await renderer.init();
  try {
    const f = sceneFixture(); f.mesh.material = new T.MeshDepthMaterial();
    renderer.render(f.scene, f.camera); await flush(renderer).catch(() => {});
    const submissions = device.submissions.length;
    await assert.rejects(renderer.clearStencilAsync(), /Unsupported source material/);
    assert.equal(device.submissions.length, submissions);
    renderer.setClearStencil(19); await renderer.clearStencilAsync();
    assert.equal(device.passes.at(-1).desc.depthStencilAttachment.stencilClearValue, 19);
  } finally { renderer.dispose(); }
});
