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
    getCurrentTexture() { const t = {width: canvas.width, height: canvas.height, dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, usage: config?.usage ?? 16, format: config?.format, swapchain: true, createView(d) { return {texture: t, ...d}; }}; calls.push(['acquire', t]); return t; },
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
  d.canvasCopies = [];
  const encode = d.createCommandEncoder;
  d.createCommandEncoder = () => {
    const e = encode();
    return {...e, copyTextureToTexture(source, destination, extent) {
      assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
      assert.equal(source.texture.sampleCount, 1); assert.equal(destination.texture.sampleCount, 1);
      d.canvasCopies.push({source, destination, extent});
    }, beginRenderPass(desc) {
      if (!desc.colorAttachments[0]?.view?.texture?.swapchain || desc.depthStencilAttachment) {
        const pass = e.beginRenderPass(desc), record = d.passes.at(-1);
        return {...pass, setViewport(...v) { record.viewport = v; }, setScissorRect(...r) { record.scissor = r; }};
      }
      const pass = {desc, draws: []}; d.outputPasses.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; }, setBindGroup(index, group) { pass.group = group; },
        draw(n) { pass.draws.push(n); }, end() {}};
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
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const hostTurn = () => new Promise(resolve => setImmediate(resolve));

function deviceEventsFixture() {
  const device = recordingDevice(), loss = deferred(); let destroyed = 0;
  device.features = new Set(['core-features-and-limits']); device.lost = loss.promise;
  device.destroy = () => { destroyed++; loss.resolve({reason: 'destroyed', message: ''}); };
  return {device, loss, destroyed: () => destroyed};
}
async function sourceDeviceEvents(t, fixture) {
  const source = new T.WebGPURenderer({canvas: canvasFixture().canvas, device: fixture.device});
  // The actual pinned backend wires its events without needing source render
  // managers or rasterization. GPU objects are the same recording boundary.
  await source.backend.init(source);
  t.after(async () => {
    fixture.device.onuncapturederror = null; fixture.loss.resolve({reason: 'destroyed'});
    await source.backend.dispose();
  });
  return source;
}

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
  const Routed = createWebGPURendererClass(T, {exactBackend: T.WebGPURenderer,
    gpu: {requestAdapter() { assert.fail('Construction must not request an adapter'); }}});
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

test('disposing an unused renderer never requests a device or starts its frame clock', () => {
  const c = canvasFixture(); let requests = 0;
  const Renderer = createWebGPURendererClass(T, {gpu: {requestAdapter() { requests++; return Promise.resolve(null); }}});
  const renderer = new Renderer({canvas: c.canvas});
  renderer.dispose();
  assert.equal(requests, 0);
  assert.equal(renderer.initialized, false);
  assert.equal(renderer.getAnimationLoop(), null);
  assert.equal(renderer.info.frame, 0);
  assert.deepEqual(c.calls, []);
});

test('initialization disposal rejects a pending adapter without requesting a late device or canvas', async () => {
  const adapter = deferred(), c = canvasFixture(), device = recordingDevice();
  let requests = 0, destroyed = 0, rejected;
  device.destroy = () => { destroyed++; };
  const Renderer = createWebGPURendererClass(T, {gpu: {requestAdapter: () => adapter.promise}});
  const renderer = new Renderer({canvas: c.canvas});
  const init = renderer.init();
  const animation = renderer.setAnimationLoop(() => assert.fail('A disposed initialization cannot install a callback'));
  animation.catch(() => {});
  init.catch(error => { rejected = error; });
  assert.equal(renderer.init(), init, 'concurrent callers share initialization');
  renderer.dispose();
  await hostTurn();
  const stopped = rejected;
  adapter.resolve({requestDevice() { requests++; return Promise.resolve(device); }});
  await assert.rejects(init, {code: 'F3D_RENDERER_DISPOSED'});
  await assert.rejects(animation, {code: 'F3D_RENDERER_DISPOSED'});
  await hostTurn();
  assert.equal(stopped?.code, 'F3D_RENDERER_DISPOSED', 'disposal settles init before the host adapter');
  assert.equal(requests, 0);
  assert.equal(destroyed, 0, 'no device was requested or owned');
  assert.deepEqual(c.calls, []);
  assert.equal(renderer.initialized, false);
  assert.equal(renderer.getAnimationLoop(), null);
  assert.equal(renderer.info.frame, 0);
  assert.equal(renderer.init(), init, 'a disposed initialization cannot restart');
});

test('initialization disposal retires a late owned device once without publishing it', async () => {
  const acquisition = deferred(), requested = deferred(), c = canvasFixture(), device = recordingDevice();
  let destroyed = 0, rejected;
  device.destroy = () => { destroyed++; };
  const Renderer = createWebGPURendererClass(T, {gpu: {requestAdapter: async () => ({
    requestDevice() { requested.resolve(); return acquisition.promise; },
  })}});
  const renderer = new Renderer({canvas: c.canvas}), init = renderer.init();
  const animation = renderer.setAnimationLoop(() => assert.fail('A late device cannot revive a disposed loop'));
  animation.catch(() => {});
  init.catch(error => { rejected = error; });
  await requested.promise;
  renderer.dispose();
  await hostTurn();
  const stopped = rejected;
  acquisition.resolve(device);
  await assert.rejects(init, {code: 'F3D_RENDERER_DISPOSED'});
  await assert.rejects(animation, {code: 'F3D_RENDERER_DISPOSED'});
  await hostTurn();
  assert.equal(stopped?.code, 'F3D_RENDERER_DISPOSED', 'disposal does not await requestDevice');
  assert.equal(destroyed, 1);
  assert.equal(renderer._device, null, 'late ownership is retired before publication');
  assert.equal(renderer._ownsDevice, false);
  assert.deepEqual(c.calls, []);
  assert.equal(renderer.initialized, false);
  assert.equal(renderer.getAnimationLoop(), null);
  renderer.dispose();
  assert.equal(destroyed, 1);
});

test('initialization failure releases its canvas and owned device while preserving a borrowed device', async t => {
  for (const borrowed of [false, true]) {
    const c = canvasFixture(), device = recordingDevice(), failure = new Error('output pipeline rejected');
    let destroyed = 0, requests = 0;
    device.destroy = () => { destroyed++; };
    device.createRenderPipelineAsync = async () => { throw failure; };
    const Renderer = createWebGPURendererClass(T, {gpu: {requestAdapter: async () => ({
      requestDevice: async () => { requests++; return device; },
    })}});
    const renderer = new Renderer({canvas: c.canvas, ...(borrowed ? {device} : {})});
    t.after(() => renderer.dispose());
    const init = renderer.init();
    await assert.rejects(init, error => error === failure);
    assert.equal(renderer.initialized, false);
    assert.equal(renderer.info.frame, 0);
    assert.equal(requests, borrowed ? 0 : 1);
    assert.equal(destroyed, borrowed ? 0 : 1, 'failed initialization releases owned GPU state without manual dispose');
    assert.equal(c.calls.filter(call => call[0] === 'configure').length, 1);
    assert.equal(c.calls.filter(call => call[0] === 'unconfigure').length, 1);
    assert.ok(device.textures.every(texture => texture.destroyed));
    assert.ok(device.buffers.every(buffer => buffer.destroyed));
    assert.equal(renderer.init(), init, 'failure retains its original result and does not renegotiate');
    renderer.dispose();
    assert.equal(destroyed, borrowed ? 0 : 1);
  }
});

test('initialization disposal aborts pending output preparation and cannot unconfigure a replacement', async t => {
  for (const borrowed of [false, true]) {
    const c = canvasFixture(), device = recordingDevice(), pipeline = deferred(), started = deferred();
    let destroyed = 0, rejected;
    device.destroy = () => { destroyed++; };
    device.createRenderPipelineAsync = descriptor => { started.resolve(descriptor); return pipeline.promise; };
    const Renderer = createWebGPURendererClass(T, {gpu: {requestAdapter: async () => ({requestDevice: async () => device})}});
    const renderer = new Renderer({canvas: c.canvas, ...(borrowed ? {device} : {})});
    t.after(() => renderer.dispose());
    const init = renderer.init();
    init.catch(error => { rejected = error; });
    const descriptor = await started.promise;
    renderer.dispose();
    await hostTurn();
    const stopped = rejected, unconfigured = c.calls.filter(call => call[0] === 'unconfigure').length;
    const replacement = new WebGPURenderer({canvas: c.canvas, device: recordingDevice()});
    t.after(() => replacement.dispose());
    let replacementError;
    try { await replacement.init(); }
    catch (error) { replacementError = error; }
    finally { pipeline.resolve(descriptor); }
    await assert.rejects(init, {code: 'F3D_RENDERER_DISPOSED'});
    await hostTurn();
    assert.equal(stopped?.code, 'F3D_RENDERER_DISPOSED', 'disposal settles init before native compilation');
    assert.equal(unconfigured, 1, 'the pending session relinquishes its canvas promptly');
    assert.equal(destroyed, borrowed ? 0 : 1);
    assert.equal(renderer._session, null);
    assert.equal(renderer.initialized, false);
    assert.equal(renderer.info.frame, 0);
    assert.ok(device.textures.every(texture => texture.destroyed));
    assert.ok(device.buffers.every(buffer => buffer.destroyed));
    assert.equal(replacementError, undefined, 'the new owner initializes before the old pipeline resolves');
    assert.equal(c.calls.filter(call => call[0] === 'unconfigure').length, unconfigured);
    assert.equal(replacement.initialized, true);
    replacement.dispose();
  }
});

for (const info of [{reason: 'unknown', message: 'adapter removed'}, {reason: '', message: ''}, {reason: 'destroyed'}])
test(`native device callbacks preserve pinned payloads, receiver and live handlers (${info.reason || 'empty'})`, async t => {
  const original = deviceEventsFixture(), native = deviceEventsFixture();
  const source = await sourceDeviceEvents(t, original);
  const renderer = new WebGPURenderer({canvas: canvasFixture().canvas, device: native.device});
  t.after(() => renderer.dispose()); await renderer.init(); await renderer.init();
  const sourceErrors = [], errors = [], sourceLosses = [], losses = [];
  source.onError = function (value) { assert.equal(this, source); sourceErrors.push(value); };
  renderer.onError = function (value) { assert.equal(this, renderer); errors.push(value); };
  source.onDeviceLost = function (value) { assert.equal(this, source); sourceLosses.push(value); };
  renderer.onDeviceLost = function (value) { assert.equal(this, renderer); losses.push(value); };
  class GPUValidationError extends Error {}
  for (const event of [{error: new GPUValidationError('invalid binding')}, {error: null}]) {
    original.device.onuncapturederror(event); native.device.onuncapturederror(event);
    assert.deepEqual(errors.at(-1), sourceErrors.at(-1));
    assert.equal(errors.at(-1).originalEvent, event);
  }
  const failure = new Error('application error handler');
  source.onError = renderer.onError = () => { throw failure; };
  assert.throws(() => original.device.onuncapturederror({}), error => error === failure);
  assert.throws(() => native.device.onuncapturederror({}), error => error === failure);
  original.loss.resolve(info); native.loss.resolve(info); await hostTurn();
  assert.deepEqual(losses, sourceLosses);
  assert.equal(losses.length, info.reason === 'destroyed' ? 0 : 1);
  if (losses.length) assert.equal(losses[0].originalEvent, info);
  assert.equal(renderer._isDeviceLost, source._isDeviceLost);
  assert.equal(renderer._isDeviceLost, false, 'custom handlers retain the source choice of loss-state policy');
});

test('default native device diagnostics and post-loss work follow the pinned source', async t => {
  const original = deviceEventsFixture(), native = deviceEventsFixture();
  const source = await sourceDeviceEvents(t, original);
  const renderer = new WebGPURenderer({canvas: canvasFixture().canvas, device: native.device});
  t.after(() => renderer.dispose()); await renderer.init();
  const previous = T.getConsoleFunction(), logs = [];
  T.setConsoleFunction((...args) => logs.push(args)); t.after(() => T.setConsoleFunction(previous));
  const event = {error: new Error('uncaptured validation')};
  original.device.onuncapturederror(event); native.device.onuncapturederror(event);
  assert.deepEqual(logs[1], logs[0]); assert.equal(renderer._isDeviceLost, false);
  const info = {reason: 'unknown', message: 'GPU reset'};
  original.loss.resolve(info); native.loss.resolve(info); await hostTurn();
  assert.equal(logs.length, 4); assert.deepEqual(logs[3], logs[2]);
  assert.equal(source._isDeviceLost, true); assert.equal(renderer._isDeviceLost, true);
  // Only the source backend was initialized above. Post-loss public calls must
  // return before they need any of its other managers or validate scene inputs.
  source._initialized = true;
  const submissions = native.device.submissions.length;
  for (const operation of [r => r.render(null, null), r => r.compileAsync(null, null),
    r => r.compute(null), r => r.compileComputeAsync(null)]) {
    assert.equal(await operation(source), undefined); assert.equal(await operation(renderer), undefined);
  }
  assert.equal(native.device.submissions.length, submissions);
  assert.equal(renderer._pending.length, 0);
});

test('device loss during initialization reaches the application before failed-session cleanup', async t => {
  const native = deviceEventsFixture(), gate = deferred(), started = deferred(), c = canvasFixture();
  const renderer = new WebGPURenderer({canvas: c.canvas, device: native.device}), losses = [];
  renderer.onDeviceLost = info => losses.push(info);
  const pipeline = native.device.createRenderPipelineAsync.bind(native.device);
  native.device.createRenderPipelineAsync = async descriptor => { started.resolve(); await gate.promise; return pipeline(descriptor); };
  t.after(() => { gate.resolve(); renderer.dispose(); });
  const init = renderer.init(); init.catch(() => {}); await started.promise;
  const info = {reason: 'unknown', message: 'GPU reset during initialization'};
  native.loss.resolve(info);
  await assert.rejects(init, {code: 'GPU_CANVAS_DEVICE_LOST'});
  assert.equal(losses.length, 1); assert.equal(losses[0].originalEvent, info);
  assert.equal(native.device.onuncapturederror, null);
  assert.equal(native.destroyed(), 0, 'a supplied device remains borrowed after initialization fails');
});

for (const borrowed of [false, true]) for (const replaceHandler of [false, true])
test(`device callbacks retire on disposal without clobbering another handler (borrowed=${borrowed}, replaced=${replaceHandler})`, async t => {
  const native = deviceEventsFixture(), Renderer = createWebGPURendererClass(T, {gpu: {
    requestAdapter: async () => ({requestDevice: async () => native.device})}});
  const renderer = new Renderer({canvas: canvasFixture().canvas, ...(borrowed ? {device: native.device} : {})});
  t.after(() => renderer.dispose()); await renderer.init();
  let calls = 0;
  renderer.onDeviceLost = renderer.onError = () => { calls++; };
  const stale = native.device.onuncapturederror, replacement = () => {};
  if (replaceHandler) native.device.onuncapturederror = replacement;
  renderer.dispose(); renderer.dispose();
  assert.equal(native.device.onuncapturederror, replaceHandler ? replacement : null);
  assert.equal(native.destroyed(), borrowed ? 0 : 1);
  native.loss.resolve({reason: 'unknown', message: 'late device loss'});
  stale({error: new Error('late uncaptured error')}); await hostTurn();
  assert.equal(calls, 0, 'neither queued loss nor a retained event handler can revive the disposed owner');
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
  renderer.outputColorSpace = T.DisplayP3ColorSpace;
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
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
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

test('native animation stopped during initialization stays stopped like the pinned source', async t => {
  const gate = deferred(), c = canvasFixture(), device = recordingDevice(), host = animationHost(c.canvas);
  device.destroy = () => {};
  const Renderer = createWebGPURendererClass(T, {gpu: {requestAdapter: () => gate.promise}});
  const renderer = new Renderer({canvas: c.canvas});
  t.after(() => { gate.resolve({requestDevice: async () => device}); renderer.dispose(); });
  let callbackCalls = 0, sourceLoop = null, sourceStopped = false, stopped = false;
  const callback = () => { callbackCalls++; };
  // Run the pinned public method with only its initialization/animation owners
  // controlled, so the comparison keeps the source's actual await ordering.
  const source = {_initialized: false, init: () => gate.promise,
    _animation: {setAnimationLoop(value) { sourceLoop = value; }}};
  const sourceSetLoop = T.WebGPURenderer.prototype.setAnimationLoop;
  const sourceStart = sourceSetLoop.call(source, callback);
  const sourceStop = sourceSetLoop.call(source, null).then(() => { sourceStopped = true; });
  const start = renderer.setAnimationLoop(callback);
  const stop = renderer.setAnimationLoop(null).then(() => { stopped = true; });
  await hostTurn();
  const stoppedWhileInitializing = [sourceStopped, stopped];
  gate.resolve({requestDevice: async () => device});
  await Promise.all([sourceStart, sourceStop, start, stop]);
  assert.equal(sourceLoop, null);
  assert.equal(renderer.getAnimationLoop(), sourceLoop, 'a cold start cannot overwrite its later stop');
  assert.deepEqual(stoppedWhileInitializing, [false, false], 'both source setters wait for initialization');
  host.step(16);
  assert.equal(callbackCalls, 0, 'the cancelled callback never executes on a later frame');
  assert.equal(renderer.info.frame, 2, 'the source internal frame clock continues after stopping callbacks');
});

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

test('straight-alpha WebGL clears match the pinned Background and State and remain encoded until presentation', async () => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const {WebGLBackground} = await import(pathToFileURL(path.join(root, 'src/renderers/webgl/WebGLBackground.js')));
  const {WebGLState} = await import(pathToFileURL(path.join(root, 'src/renderers/webgl/WebGLState.js')));
  const clears = [], constants = new Map();
  const gl = new Proxy({
    clearColor(...value) { clears.push(value); }, createTexture: () => ({}),
    getParameter(name) {
      if (name === gl.VERSION) return 'WebGL 2.0';
      if (name === gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) return 16;
      return [0, 0, 300, 150];
    },
  }, {get(object, key) {
    if (key in object) return object[key];
    if (/^[A-Z_0-9]+$/.test(key)) { if (!constants.has(key)) constants.set(key, constants.size + 1); return constants.get(key); }
    return () => {};
  }});
  const state = WebGLState(gl, {get: () => null});
  for (const alpha of [false, true]) for (const premultipliedAlpha of [false, true]) {
    const device = recordingDevice(), c = canvasFixture(), Renderer = createWebGLRendererClass(TC);
    const renderer = new Renderer({canvas: c.canvas, device, alpha, premultipliedAlpha});
    await renderer.init();
    const reference = WebGLBackground({getRenderTarget: () => null, outputColorSpace: TC.SRGBColorSpace}, null, state, null, alpha, premultipliedAlpha);
    for (const clearAlpha of [0, .25, .75, 1]) {
      const color = new TC.Color(.1, .4, .8);
      reference.setClearColor(color, clearAlpha); renderer.setClearColor(color, clearAlpha);
      assert.deepEqual(renderer._frame(new TC.Scene()).clearColor, clears.at(-1));
    }
    assert.equal(renderer._hdr, false); assert.equal(renderer._session.rendererOptions.format, 'bgra8unorm');
    // Like real context attributes, the construction value remains fixed.
    renderer.premultipliedAlpha = !premultipliedAlpha;
    assert.deepEqual(renderer._frame(new TC.Scene()).clearColor, clears.at(-1));
    renderer.dispose(); reference.dispose();
  }
});

for (const preserveDrawingBuffer of [false, true]) test(`straight-alpha facade renders and copies the original framebuffer, preserved=${preserveDrawingBuffer}`, async () => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const device = recordingDevice(), c = canvasFixture(), Renderer = createWebGLRendererClass(TC,
    {gpu: {getPreferredCanvasFormat: () => 'rgba8unorm'}});
  const renderer = new Renderer({canvas: c.canvas, device, alpha: true, premultipliedAlpha: false, preserveDrawingBuffer});
  const scene = new TC.Scene(), material = new TC.MeshBasicMaterial({color: 0xff0000, transparent: true, opacity: .5});
  const mesh = new TC.Mesh(new TC.BoxGeometry(), material), camera = new TC.PerspectiveCamera(50, 2, .1, 10);
  camera.position.z = 3; scene.add(mesh); renderer.setClearColor(new TC.Color(.1, .4, .8), .25);
  await renderer.renderAsync(scene, camera);
  const source = device.textures.find(t => t.label === 'f3d-canvas-preserved');
  assert.ok(source); assert.equal(source.format, 'rgba8unorm'); assert.equal(source.sampleCount, 1);
  const output = device.outputPasses.at(-1);
  assert.equal(output.desc.label, 'f3d-canvas-premultiply'); assert.deepEqual(output.draws, [3]);
  assert.equal(output.group.entries[0].resource.attachment, source);
  assert.deepEqual(output.pipeline.fragment.targets, [{format: 'rgba8unorm'}]);
  const drawing = device.passes.find(p => p.desc.colorAttachments[0].view.attachment === source && p.draws.length);
  assert.ok(drawing); assert.equal(drawing.desc.colorAttachments[0].clearValue.a, .25);
  assert.equal(device.canvasCopies.length, 0, 'presentation does not overwrite or copy premultiplied pixels back');
  const texture = new TC.FramebufferTexture(4, 4);
  renderer.copyFramebufferToTexture(texture); await flush(renderer);
  const copied = device.passes.find(pass => pass.draws.some(draw => draw.pipeline.label === 'WebGL framebuffer texel copy'));
  assert.ok(copied);
  const copy = copied.draws.find(draw => draw.pipeline.label === 'WebGL framebuffer texel copy');
  assert.equal(copy.groups.get(0).group.entries[0].resource.attachment, source);
  assert.notEqual(copied.desc.colorAttachments[0].view.attachment, source);
  assert.equal(source.swapchain, undefined); assert.equal(device.outputPasses.at(-1).group.entries[0].resource.attachment, source);
  renderer.dispose(); texture.dispose(); mesh.geometry.dispose(); material.dispose();
});

test('straight-alpha shader preparation errors retain the original canvas owner without whole-image HDR fallback', async () => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const device = recordingDevice(), c = canvasFixture(), Renderer = createWebGLRendererClass(TC);
  const renderer = new Renderer({canvas: c.canvas, device, premultipliedAlpha: false});
  const scene = new TC.Scene(), material = new TC.MeshBasicMaterial(), geometry = new TC.BoxGeometry(), camera = new TC.PerspectiveCamera();
  scene.add(new TC.Mesh(geometry, material)); await renderer.renderAsync(scene, camera);
  const session = renderer._session, error = Object.assign(new Error('source program requires unavailable output'), {code: 'THREE_SCENE_TONE_MAPPING'});
  material.onBeforeCompile = () => { throw error; }; material.needsUpdate = true;
  renderer.toneMapping = TC.ACESFilmicToneMapping;
  await assert.rejects(renderer.compileAsync(scene, camera), e => e === error);
  assert.equal(renderer._session, session); assert.equal(renderer._hdr, false);
  assert.notEqual(renderer._hdrFallback, true); assert.equal(c.calls.some(call => call[0] === 'unconfigure'), false);
  renderer.dispose(); geometry.dispose(); material.dispose();
});

test('WebGL material and procedural geometry changes do not acquire unused canvas images', async t => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const device = recordingDevice(), c = canvasFixture(), Renderer = createWebGLRendererClass(TC);
  const renderer = new Renderer({canvas: c.canvas, device});
  const materials = [new TC.MeshBasicMaterial(), new TC.MeshPhongMaterial(), new TC.ShaderMaterial({
    vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'void main() { gl_FragColor = vec4(0.2, 0.6, 0.9, 1.0); }',
  })];
  const geometries = [new TC.BoxGeometry(), new TC.PlaneGeometry(1, 1, 3, 3)];
  const scene = new TC.Scene(), mesh = new TC.Mesh(geometries[0], materials[0]);
  const camera = new TC.PerspectiveCamera(50, 2, .1, 10); camera.position.z = 3;
  scene.add(mesh, new TC.AmbientLight());
  t.after(() => { renderer.dispose(); for (const value of [...materials, ...geometries]) value.dispose(); });
  await renderer.init(); renderer.render(scene, camera); await flush(renderer);
  assert.equal(c.acquired(), 1); assert.equal(device.submissions.length, 1);
  for (const change of [() => { mesh.material = materials[1]; }, () => { mesh.material = materials[2]; },
    () => { mesh.geometry = geometries[1]; }]) {
    const acquired = c.acquired(), submissions = device.submissions.length;
    change(); renderer.render(scene, camera);
    assert.equal(renderer._pending.length, 1, 'the changed source draw needs preparation');
    assert.equal(device.submissions.length, submissions);
    assert.equal(c.acquired(), acquired, 'deferred draws must leave the presented image untouched');
    await flush(renderer);
    assert.equal(device.submissions.length, submissions + 1);
    assert.equal(c.acquired(), acquired + 1, 'acquire once, when the prepared draw is submitted');
    renderer.render(scene, camera);
    assert.equal(device.submissions.length, submissions + 2, 'the warmed draw remains synchronous');
    assert.equal(c.acquired(), acquired + 2);
  }
});

const submittedSurfaceDraws = device => device.snapshots.flat().filter(draw => draw.streams.has(0));
function submittedMaterialColor(draw) {
  const {group, offsets} = draw.groups.get(0), resource = group.entries[0].resource;
  const layout = group.layout.entries[0].buffer, bytes = draw.contents.get(resource.buffer);
  const offset = (resource.offset ?? 0) + (offsets[0] ??
    (layout.type === 'read-only-storage' ? draw.args.at(-1) * layout.minBindingSize : 0));
  return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset + offset).slice(16, 19));
}

test('WebGL accepted material selection survives a GUI replacement during pipeline preparation while values stay live', async t => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const device = recordingDevice(), c = canvasFixture(), Renderer = createWebGLRendererClass(TC);
  const renderer = new Renderer({canvas: c.canvas, device}), gate = deferred(), started = deferred();
  const first = new TC.MeshBasicMaterial({color: 0xff0000, side: TC.FrontSide});
  const next = new TC.MeshBasicMaterial({color: 0x0000ff, side: TC.BackSide});
  const scene = new TC.Scene(), mesh = new TC.Mesh(new TC.PlaneGeometry(2, 2), first);
  const camera = new TC.PerspectiveCamera(50, 2, .1, 10); camera.position.z = 3;
  t.after(() => { gate.resolve(); renderer.dispose(); mesh.geometry.dispose(); first.dispose(); next.dispose(); });
  await renderer.compileAsync(scene, camera); scene.add(mesh);
  const createPipeline = device.createRenderPipelineAsync.bind(device);
  device.createRenderPipelineAsync = async descriptor => { started.resolve(); await gate.promise; return createPipeline(descriptor); };
  renderer.render(scene, camera); await started.promise;
  assert.equal(c.acquired(), 0); assert.equal(device.submissions.length, 0);
  mesh.material = next;
  first.color.setRGB(.25, .5, .75);
  gate.resolve(); await flush(renderer);
  const accepted = submittedSurfaceDraws(device);
  assert.equal(accepted.length, 1, 'the accepted frame completes once after its pipeline is ready');
  assert.equal(accepted[0].pipeline.primitive.cullMode, 'back', 'the accepted source material still selects the raster state');
  assert.deepEqual(submittedMaterialColor(accepted[0]), [.25, .5, .75], 'selection capture does not clone material values');
  assert.equal(mesh.material, next, 'rendering never restores a stale selector onto the application object');
  assert.equal(c.acquired(), 1); assert.equal(renderer._deferredError, null);
  renderer.render(scene, camera); await flush(renderer);
  const draws = submittedSurfaceDraws(device);
  assert.equal(draws.length, 2); assert.equal(c.acquired(), 2);
  assert.equal(draws[1].pipeline.primitive.cullMode, 'front');
  assert.deepEqual(submittedMaterialColor(draws[1]), [0, 0, 1], 'the next accepted request sees the GUI selection');
  assert.equal(renderer._deferredError, null); assert.equal(renderer._materialCaptureValues, 0);
});

test('WebGL accepted material-array slots survive in-place GUI edits during preparation', async t => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const device = recordingDevice(), c = canvasFixture(), Renderer = createWebGLRendererClass(TC);
  const renderer = new Renderer({canvas: c.canvas, device}), gate = deferred(), started = deferred();
  const materials = [0xff0000, 0x0000ff, 0x00ff00].map(color => new TC.MeshBasicMaterial({color}));
  const selected = materials.slice(0, 2), geometry = new TC.PlaneGeometry(2, 2);
  geometry.clearGroups(); geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 1);
  const scene = new TC.Scene(), mesh = new TC.Mesh(geometry, selected);
  const camera = new TC.PerspectiveCamera(50, 2, .1, 10); camera.position.z = 3;
  t.after(() => { gate.resolve(); renderer.dispose(); geometry.dispose(); for (const material of materials) material.dispose(); });
  await renderer.compileAsync(scene, camera); scene.add(mesh);
  const createPipeline = device.createRenderPipelineAsync.bind(device);
  device.createRenderPipelineAsync = async descriptor => { started.resolve(); await gate.promise; return createPipeline(descriptor); };
  renderer.render(scene, camera); await started.promise;
  selected[0] = materials[2]; selected[1] = materials[0];
  gate.resolve(); await flush(renderer);
  const byGroup = draws => [...draws].sort((a, b) => a.args[2] - b.args[2]).map(submittedMaterialColor);
  assert.deepEqual(byGroup(submittedSurfaceDraws(device)), [[1, 0, 0], [0, 0, 1]]);
  assert.equal(mesh.material, selected); assert.deepEqual(selected, [materials[2], materials[0]]);
  assert.equal(c.acquired(), 1); assert.equal(renderer._deferredError, null);
  renderer.render(scene, camera); await flush(renderer);
  const draws = submittedSurfaceDraws(device);
  assert.equal(draws.length, 4); assert.equal(c.acquired(), 2);
  assert.deepEqual(byGroup(draws.slice(2)), [[0, 1, 0], [1, 0, 0]]);
  assert.equal(renderer._deferredError, null); assert.equal(renderer._materialCaptureValues, 0);
});

test('WebGL hooked draws retain original source identities and live shader uniforms across preparation', async t => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const device = recordingDevice(), c = canvasFixture(), Renderer = createWebGLRendererClass(TC);
  const renderer = new Renderer({canvas: c.canvas, device}), gate = deferred(), started = deferred();
  const tint = new TC.Vector3(.125, .25, .5), material = new TC.ShaderMaterial({uniforms: {tint: {value: tint}},
    vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 tint; void main() { gl_FragColor = vec4(tint, 1.0); }'});
  const scene = new TC.Scene(), mesh = new TC.Mesh(new TC.PlaneGeometry(2, 2), material);
  const camera = new TC.PerspectiveCamera(50, 2, .1, 10); camera.position.z = 3;
  const callbacks = [];
  mesh.onBeforeRender = function (actualRenderer, actualScene, actualCamera, geometry, selected) {
    assert.equal(this, mesh); assert.equal(actualRenderer, renderer); assert.equal(actualScene, scene);
    assert.equal(actualCamera, camera); assert.equal(geometry, mesh.geometry); assert.equal(selected, material);
    callbacks.push('before'); tint.x = .875;
  };
  mesh.onAfterRender = function () { callbacks.push('after'); };
  t.after(() => { gate.resolve(); renderer.dispose(); mesh.geometry.dispose(); material.dispose(); });
  await renderer.compileAsync(scene, camera); scene.add(mesh);
  const createPipeline = device.createRenderPipelineAsync.bind(device);
  device.createRenderPipelineAsync = async descriptor => { started.resolve(); await gate.promise; return createPipeline(descriptor); };
  renderer.render(scene, camera); await started.promise;
  assert.deepEqual(callbacks, [], 'preparation never replays draw callbacks');
  tint.set(.25, .625, .375);
  gate.resolve(); await flush(renderer);
  assert.deepEqual(callbacks, ['before', 'after']); assert.equal(c.acquired(), 1);
  const [draw] = submittedSurfaceDraws(device);
  const words = [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
    .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
    .flatMap((entry, index) => {
      const resource = group.entries.find(binding => binding.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4));
    }));
  assert.ok(words.some((value, index) => value === .875 && words[index + 1] === .625 && words[index + 2] === .375),
    'the actual shader upload observes the same live uniform object after the draw callback');
  assert.equal(material.uniforms.tint.value, tint); assert.equal(renderer._materialCaptureValues, 0);
  assert.equal(renderer._deferredError, null);
});

test('WebGL animation preserves procedural updates across cold preparation and resumes before GPU completion', async t => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  const device = recordingDevice(), c = canvasFixture(), host = animationHost(c.canvas);
  const Renderer = createWebGLRendererClass(TC), renderer = new Renderer({canvas: c.canvas, device});
  const scene = new TC.Scene(), mesh = new TC.Mesh(new TC.BoxGeometry(), new TC.MeshPhongMaterial());
  const camera = new TC.PerspectiveCamera(50, 2, .1, 10); camera.position.z = 3;
  scene.add(mesh, new TC.AmbientLight(0xffffff));
  let releasePipeline, pipelineStarted, finishGPU;
  const gate = new Promise(resolve => { releasePipeline = resolve; });
  const started = new Promise(resolve => { pipelineStarted = resolve; });
  device.completion = new Promise(resolve => { finishGPU = resolve; });
  t.after(() => { releasePipeline(); finishGPU(); renderer.dispose(); });
  const createPipeline = device.createRenderPipelineAsync.bind(device);
  device.createRenderPipelineAsync = async descriptor => { pipelineStarted(); await gate; return createPipeline(descriptor); };
  const frames = [];
  const update = time => {
    frames.push(time); mesh.position.x = frames.length * .125;
    mesh.geometry.attributes.position.array[0] = frames.length * .25;
    mesh.geometry.attributes.position.needsUpdate = true;
    renderer.render(scene, camera);
  };
  assert.equal(renderer.setAnimationLoop(update), undefined, 'WebGL loop installation stays synchronous');
  host.step(16); await started;
  const version = mesh.geometry.attributes.position.version;
  assert.equal(renderer._pending.length, 1); assert.equal(c.acquired(), 0);
  host.step(32);
  assert.equal(host.scheduled.size, 0); assert.deepEqual(frames, [16]);
  assert.equal(mesh.position.x, .125); assert.equal(mesh.geometry.attributes.position.version, version);
  assert.equal(renderer._pending.length, 1, 'no extra mutable H2-style geometry frame was accepted');
  renderer.setAnimationLoop(update); host.step(48);
  assert.equal(host.scheduled.size, 0, 'replacement with the same function also suspends safely');
  releasePipeline(); await flush(renderer);
  assert.equal(c.acquired(), 1); assert.equal(host.scheduled.size, 1, 'one fresh RAF, with GPU execution still pending');
  host.step(64);
  assert.deepEqual(frames, [16, 64]); assert.equal(c.acquired(), 2); assert.equal(mesh.position.x, .25);
  assert.equal(device.canvasCopies.length, 2, 'each accepted animation frame presents owned color');
  assert.equal(device.canvasCopies[0].source.texture, device.canvasCopies[1].source.texture, 'reuse bounded history storage');
  assert.equal(device.passes.filter(pass => pass.desc.colorAttachments[0]?.clearValue?.[3] === 0 &&
    pass.desc.depthStencilAttachment?.depthClearValue === 1 && pass.draws.length === 0).length, 2,
  'default non-preserved animation starts each callback with fresh color and depth');
  renderer.setAnimationLoop(null); assert.equal(host.scheduled.size, 0);
});

test('WebGL animation replacement, stop, disposal and failed preparation do not revive stale callbacks', async t => {
  const TC = await import(pathToFileURL(path.join(root, 'build/three.module.js')));
  for (const action of ['replace', 'stop', 'dispose', 'fail']) {
    const device = recordingDevice(), c = canvasFixture(), host = animationHost(c.canvas);
    const Renderer = createWebGLRendererClass(TC), renderer = new Renderer({canvas: c.canvas, device});
    const scene = new TC.Scene(); scene.add(new TC.Mesh(new TC.BoxGeometry(), new TC.MeshBasicMaterial()));
    const camera = new TC.PerspectiveCamera(50, 2, .1, 10); camera.position.z = 3;
    let releasePipeline, pipelineStarted;
    const gate = new Promise(resolve => { releasePipeline = resolve; });
    const started = new Promise(resolve => { pipelineStarted = resolve; });
    const failure = new Error('WebGL cold pipeline failed');
    const createPipeline = device.createRenderPipelineAsync.bind(device);
    device.createRenderPipelineAsync = async descriptor => {
      pipelineStarted(); await gate; if (action === 'fail') throw failure; return createPipeline(descriptor);
    };
    t.after(() => { releasePipeline(); renderer.dispose(); });
    const frames = [];
    renderer.setAnimationLoop(time => { frames.push(time); renderer.render(scene, camera); });
    host.step(16); await started; host.step(32);
    const pending = renderer._drain;
    if (action === 'replace') renderer.setAnimationLoop(time => { frames.push(time * 10); renderer.render(scene, camera); });
    if (action === 'stop') renderer.setAnimationLoop(null);
    if (action === 'dispose') renderer.dispose();
    releasePipeline(); await pending.catch(() => {});
    if (action === 'stop' || action === 'dispose') {
      assert.equal(host.scheduled.size, 0, action); assert.deepEqual(frames, [16]);
    } else if (action === 'replace') {
      assert.equal(host.scheduled.size, 1); host.step(48); assert.deepEqual(frames, [16, 480]);
    } else {
      assert.equal(host.scheduled.size, 1);
      assert.throws(() => host.step(48), /WebGL cold pipeline failed/);
      assert.equal(host.scheduled.size, 1, 'failure stays observable without silently killing the loop');
    }
    renderer.dispose(); assert.equal(host.scheduled.size, 0);
  }
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
    assert.equal(first.draws[0].stencilReference, 5);
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
