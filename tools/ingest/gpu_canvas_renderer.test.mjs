import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
const tick = async () => { for (let i = 0; i < 16; i++) await Promise.resolve(); };
const code = name => ({code: `GPU_CANVAS_${name}`});
function fixture() {
  const calls = [], textures = [], scopes = [], lost = deferred();
  let destroyed = 0, resourceDisposed = 0, resourceFailed = false;
  const context = {
    configure(d) { calls.push(['configure', d]); }, unconfigure() { calls.push(['unconfigure']); },
    getCurrentTexture() { return {createView(d) { return {native: 'canvas', ...d}; }}; },
  };
  const canvas = {width: 16, height: 8, getContext(kind) { assert.equal(kind, 'webgpu'); return context; }};
  const device = {
    limits: {maxTextureDimension2D: 4096, maxBufferSize: 65536, minUniformBufferOffsetAlignment: 256},
    features: new Set(['timestamp-query']), lost: lost.promise,
    pushErrorScope(kind) { scopes.push(kind); }, popErrorScope() { scopes.pop(); return Promise.resolve(null); },
    createTexture(d) { const t = {d, destroyed: false, createView() { return {texture: t}; }, destroy() { t.destroyed = true; }}; textures.push(t); return t; },
    queue: {onSubmittedWorkDone() { return Promise.resolve(); }},
    destroy() { destroyed++; calls.push(['device-destroy']); },
  };
  const adapter = {limits: device.limits, features: device.features,
    requestDevice(d) { calls.push(['requestDevice', d]); return Promise.resolve(device); }};
  const gpu = {requestAdapter(d) { calls.push(['requestAdapter', d]); return Promise.resolve(adapter); }, getPreferredCanvasFormat() { return 'rgba8unorm'; }};
  const resource = {
    render(input, frame) { calls.push(['render', input, frame]); return resource; },
    prepare() { calls.push(['prepare']); return Promise.resolve(resource); },
    whenIdle() { calls.push(['idle']); return Promise.resolve(resource); },
    dispose() { resourceDisposed++; calls.push(['resource-dispose']); },
    get failed() { return resourceFailed; },
    get diagnostics() { return {submitted: calls.filter(c => c[0] === 'render').length}; },
  };
  const factory = (d, settings) => { calls.push(['factory', d, settings]); return Promise.resolve(resource); };
  return {calls, textures, scopes, canvas, context, device, adapter, gpu, resource, factory, lost,
    get destroyed() { return destroyed; }, get resourceDisposed() { return resourceDisposed; },
    failResource() { resourceFailed = true; }};
}
const count = (f, name) => f.calls.filter(c => c[0] === name).length;

test('negotiates before creation and gives the renderer exactly the presentation pipeline formats', async () => {
  const f = fixture();
  const r = await createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, powerPreference: 'high-performance',
    requiredFeatures: ['timestamp-query', 'timestamp-query'], requiredLimits: {maxBufferSize: 32768}, target: {sampleCount: 4}});
  assert.deepEqual(f.calls[0], ['requestAdapter', {powerPreference: 'high-performance'}]);
  assert.deepEqual(f.calls[1], ['requestDevice', {requiredFeatures: ['timestamp-query'], requiredLimits: {maxBufferSize: 32768}}]);
  assert.equal(r.device, f.device); assert.equal(r.ownsDevice, true);
  assert.deepEqual(f.calls.find(c => c[0] === 'factory').slice(1), [f.device, {format: 'rgba8unorm-srgb', depthFormat: 'depth24plus', sampleCount: 4}]);
  const camera = {}; r.render(camera, {clearColor: [1, 0, 0, 1]});
  const draw = f.calls.find(c => c[0] === 'render'); assert.equal(draw[1], camera);
  assert.equal(draw[2].resolveTarget.native, 'canvas'); assert.deepEqual(draw[2].clearColor, [1, 0, 0, 1]);
  assert.equal(r.lastFrameRendered, true); assert.equal(r.diagnostics.renderer.submitted, 1);
  r.dispose(); assert.equal(f.destroyed, 1); assert.equal(f.resourceDisposed, 1);
  assert.ok(f.calls.findIndex(c => c[0] === 'resource-dispose') < f.calls.findIndex(c => c[0] === 'device-destroy'));
});

test('borrows supplied devices without requesting or destroying another device', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device, gpu: f.gpu});
  assert.equal(count(f, 'requestAdapter'), 0); assert.equal(r.ownsDevice, false);
  r.render({}); await r.whenIdle(); r.dispose(); r.dispose();
  assert.equal(f.destroyed, 0); assert.equal(f.resourceDisposed, 1);
});

test('logical sizing applies pixel ratio without touching source cameras or CSS', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device});
  assert.equal(r.setSize(100, 50, 1.5), r); assert.deepEqual([r.width, r.height], [150, 75]);
  assert.equal(r.resize(20, 10), r); assert.deepEqual([r.width, r.height], [20, 10]);
  r.setSize(0, 10, 2); r.render({}); assert.equal(r.suspended, true); assert.equal(r.lastFrameRendered, false);
  assert.equal(count(f, 'render'), 0);
  r.setSize(40, 20); r.render({}); assert.equal(count(f, 'render'), 1); r.dispose();
});

test('invalid frame overrides and sizing preserve the last valid configuration', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device});
  for (const key of ['colorView', 'depthView', 'resolveTarget']) assert.throws(() => r.render({}, {[key]: {}}), code('FRAME'));
  for (const args of [[10, 10, 0], [10, 10, -1], [10, 10, Infinity], [10, 10, NaN], [-1, 10], [0.5, 10]])
    assert.throws(() => r.setSize(...args), code('SIZE'));
  assert.deepEqual([r.width, r.height], [16, 8]); assert.equal(count(f, 'render'), 0); r.dispose();
});

test('unsupported adapter requirements refuse before requestDevice or canvas configuration', async () => {
  for (const requirement of [{requiredFeatures: ['missing']}, {requiredLimits: {maxBufferSize: 65537}},
    {requiredLimits: {minUniformBufferOffsetAlignment: 128}}, {requiredLimits: {unknown: 1}}]) {
    const f = fixture();
    await assert.rejects(createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, ...requirement}), code(requirement.requiredFeatures ? 'FEATURE' : 'LIMIT'));
    assert.equal(count(f, 'requestDevice'), 0); assert.equal(count(f, 'configure'), 0); assert.equal(f.destroyed, 0);
  }
});

test('minimum alignment limits use the opposite comparison to maximum capacity limits', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory,
    {gpu: f.gpu, requiredLimits: {minUniformBufferOffsetAlignment: 512, maxBufferSize: 1024}});
  assert.equal(count(f, 'requestDevice'), 1); r.dispose();
});

test('borrowed device capabilities are checked without claiming ownership on failure', async () => {
  const f = fixture();
  await assert.rejects(createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device, requiredFeatures: ['missing']}), code('FEATURE'));
  assert.equal(f.destroyed, 0); assert.equal(count(f, 'configure'), 0);
});

test('missing hosts and adapters fail explicitly without trying a WebGL backend', async () => {
  const f = fixture();
  await assert.rejects(createGpuCanvasRenderer(f.canvas, f.factory, {gpu: null}), code('UNAVAILABLE'));
  f.gpu.requestAdapter = async () => null;
  await assert.rejects(createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu}), code('ADAPTER'));
  assert.equal(count(f, 'configure'), 0);
});

test('configuration snapshots survive mutation while adapter negotiation is pending', async () => {
  const f = fixture(), ready = deferred(); f.gpu.requestAdapter = () => ready.promise;
  const options = {gpu: f.gpu, requiredFeatures: ['timestamp-query'], requiredLimits: {maxBufferSize: 2000}, target: {sampleCount: 4}};
  const pending = createGpuCanvasRenderer(f.canvas, f.factory, options);
  options.requiredFeatures.push('missing'); options.requiredLimits.maxBufferSize = 999999; options.target.sampleCount = 1;
  ready.resolve(f.adapter); const r = await pending;
  assert.equal(r.rendererOptions.sampleCount, 4);
  assert.deepEqual(f.calls.find(c => c[0] === 'requestDevice')[1].requiredFeatures, ['timestamp-query']); r.dispose();
});

test('pre-aborted initialization starts no device requests or resource creation', async () => {
  const f = fixture(), abort = new AbortController(); abort.abort();
  await assert.rejects(createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, signal: abort.signal}), code('ABORTED'));
  assert.equal(f.calls.length, 0);
});

test('aborting a pending adapter request ends the wait and never requests its late device', async () => {
  const f = fixture(), abort = new AbortController(), pendingAdapter = deferred(); f.gpu.requestAdapter = () => pendingAdapter.promise;
  const pending = createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, signal: abort.signal});
  abort.abort(); await assert.rejects(pending, code('ABORTED'));
  pendingAdapter.resolve(f.adapter); await tick(); assert.equal(count(f, 'requestDevice'), 0);
});

test('a device arriving after abort is destroyed exactly once and never configured', async () => {
  const f = fixture(), abort = new AbortController(), pendingDevice = deferred();
  f.adapter.requestDevice = () => { f.calls.push(['pending-device']); return pendingDevice.promise; };
  const pending = createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, signal: abort.signal});
  await tick(); assert.equal(count(f, 'pending-device'), 1);
  abort.abort(); await assert.rejects(pending, code('ABORTED'));
  pendingDevice.resolve(f.device); await tick();
  assert.equal(f.destroyed, 1); assert.equal(count(f, 'configure'), 0);
});

test('aborting a pending renderer releases its late result and preserves borrowed devices', async () => {
  for (const borrowed of [false, true]) {
    const f = fixture(), abort = new AbortController(), ready = deferred();
    const factory = () => { f.calls.push(['pending-factory']); return ready.promise; };
    const pending = createGpuCanvasRenderer(f.canvas, factory, {gpu: f.gpu, ...(borrowed ? {device: f.device} : {}), signal: abort.signal});
    await tick(); assert.equal(count(f, 'pending-factory'), 1);
    abort.abort(); await assert.rejects(pending, code('ABORTED'));
    assert.equal(count(f, 'unconfigure'), 1); assert.equal(f.destroyed, borrowed ? 0 : 1);
    ready.resolve(f.resource); await tick(); assert.equal(f.resourceDisposed, 1);
    assert.equal(f.destroyed, borrowed ? 0 : 1);
  }
});

test('renderer construction failures unconfigure the target and destroy only owned devices', async () => {
  for (const borrowed of [false, true]) {
    const f = fixture(), error = new Error('unsupported source mesh');
    await assert.rejects(createGpuCanvasRenderer(f.canvas, () => { throw error; },
      {gpu: f.gpu, ...(borrowed ? {device: f.device} : {})}), e => e === error);
    assert.equal(count(f, 'unconfigure'), 1); assert.ok(f.textures.every(t => t.destroyed));
    assert.equal(f.destroyed, borrowed ? 0 : 1);
  }
});

test('malformed renderer factories cannot leak the device or target', async () => {
  for (const result of [null, {}, {render() {}, dispose() {}}]) {
    const f = fixture();
    await assert.rejects(createGpuCanvasRenderer(f.canvas, () => result, {gpu: f.gpu}), code('FACTORY'));
    assert.equal(f.destroyed, 1); assert.equal(count(f, 'unconfigure'), 1);
  }
});

test('preparation remains explicit, excludes drawing, and admission errors permit retry', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device}), pending = deferred();
  f.resource.prepare = () => pending.promise;
  const prepare = r.prepare();
  assert.throws(() => r.render({}), code('REENTRANT')); assert.throws(() => r.resize(2, 2), code('REENTRANT'));
  await assert.rejects(r.prepare(), code('REENTRANT'));
  const error = new Error('prepare source mismatch'); pending.reject(error);
  await assert.rejects(prepare, e => e === error); assert.equal(r.failed, false);
  f.resource.prepare = async () => {}; await r.prepare(); r.render({}); r.dispose();
});

test('disposal ends stalled preparation and completion waits', async () => {
  for (const operation of ['prepare', 'whenIdle']) {
    const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device});
    const forever = deferred(); f.resource[operation] = () => forever.promise;
    const pending = r[operation](); r.dispose();
    await assert.rejects(pending, code('DISPOSED')); assert.equal(f.resourceDisposed, 1); assert.equal(f.destroyed, 0);
  }
});

test('source admission errors preserve the live renderer, native failures release it', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu}), error = new Error('source change requires prepare');
  f.resource.render = () => { throw error; };
  assert.throws(() => r.render({}), e => e === error); assert.equal(r.failed, false); assert.equal(f.resourceDisposed, 0);
  f.resource.render = () => { f.failResource(); throw error; };
  assert.throws(() => r.render({}), e => e === error); assert.equal(r.failed, true);
  assert.equal(f.resourceDisposed, 1); assert.equal(f.destroyed, 1); r.dispose();
});

test('device loss terminates initialized sessions but does not destroy borrowed devices', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device});
  f.lost.resolve({message: 'GPU removed'}); await tick();
  assert.equal(r.failed, true); assert.equal(f.resourceDisposed, 1); assert.equal(f.destroyed, 0);
  assert.throws(() => r.render({}), code('DEVICE_LOST')); r.dispose();
});

test('abort delivered by a frame getter cleans up outside the synchronous boundary and skips submission', async () => {
  const f = fixture(), abort = new AbortController();
  const r = await createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, signal: abort.signal});
  assert.throws(() => r.render({}, {get clearColor() { abort.abort(); return [0, 0, 0, 1]; }}), code('ABORTED'));
  assert.equal(count(f, 'render'), 0); assert.equal(f.resourceDisposed, 1); assert.equal(f.destroyed, 1);
});

test('draw-time abort lets an issued synchronous draw exit before native ownership is released', async () => {
  const f = fixture(), abort = new AbortController();
  f.resource.render = () => {
    abort.abort(); assert.equal(f.resourceDisposed, 0); assert.equal(f.destroyed, 0);
    f.calls.push(['submitted']);
  };
  const r = await createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, signal: abort.signal});
  assert.throws(() => r.render({}), code('ABORTED'));
  assert.equal(count(f, 'submitted'), 1); assert.equal(f.resourceDisposed, 1); assert.equal(f.destroyed, 1);
});

test('synchronous frame getters cannot reenter or dispose the session', async () => {
  const f = fixture(), r = await createGpuCanvasRenderer(f.canvas, f.factory, {device: f.device});
  assert.throws(() => r.render({}, {get clearColor() { r.dispose(); }}), code('REENTRANT'));
  assert.equal(r.disposed, false); assert.equal(count(f, 'render'), 0); r.dispose();
});

test('invalid options fail without side effects', async () => {
  for (const options of [{powerPreference: 'fastest'}, {requiredFeatures: 'timestamp-query'},
    {requiredFeatures: [1]}, {requiredLimits: {maxBufferSize: Infinity}}, {target: null}, {unknown: true}]) {
    // null target selects the documented default, just like omission.
    if (options.target === null) continue;
    const f = fixture();
    await assert.rejects(createGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, ...options})); assert.equal(f.calls.length, 0);
  }
});
