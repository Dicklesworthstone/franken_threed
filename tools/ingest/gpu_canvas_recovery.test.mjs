import test from 'node:test';
import assert from 'node:assert/strict';
import {createRecoverableGpuCanvasRenderer} from './gpu_canvas_recovery.mjs';
import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
import {GpuCanvasError} from './gpu_canvas.mjs';
import {recoveryFixture, deferred, turns} from './gpu_canvas_recovery_test_fixture.mjs';

const code = expected => error => error?.code === 'GPU_CANVAS_' + expected;
async function setup(options = {}, factory) {
  const f = recoveryFixture(), first = f.enqueue(f.device('first'));
  const app = await createRecoverableGpuCanvasRenderer(f.canvas, factory ? factory(f) : f.factory, {gpu: f.gpu, ...options});
  return {...f, first, app};
}
async function lose(f) { f.app.device.lose(); return f.app.whenLost(); }
const ownedTextures = d => d.textures.filter(t => !t.borrowed);

test('initial generation uses production adapter, device, canvas and renderer owners', async () => {
  const f = await setup({target: {sampleCount: 4, depthFormat: 'depth32float'}});
  assert.equal(f.app.state, 'ready'); assert.equal(f.app.generation, 1); assert.equal(f.app.recoveryAttempts, 0);
  assert.equal(f.app.ownsDevice, true); assert.equal(f.app.device, f.first); assert.equal(f.app.failed, false);
  assert.deepEqual(f.app.rendererOptions, {format: 'bgra8unorm-srgb', depthFormat: 'depth32float', sampleCount: 4});
  assert.equal(f.context.acquisitions, 0); assert.equal(f.renderers[0].renders.length, 0);
  assert.equal(f.app.render('camera'), f.app); await f.app.whenIdle();
  assert.equal(f.app.lastFrameRendered, true); assert.equal(f.context.acquisitions, 1);
  assert.ok(f.renderers[0].renders[0].frame.resolveTarget);
  f.app.dispose(); assert.equal(f.first.destroyCount, 1); assert.ok(ownedTextures(f.first).every(t => t.destroyed === 1));
});

test('loss retires GPU resources and recovery rebuilds without replaying source effects', async () => {
  const source = {scene: {name: 'caller scene'}, animationTime: 42};
  const f = await setup(); f.app.render(source.scene); const old = f.renderers[0];
  const pendingLoss = f.app.whenLost(), info = await lose(f);
  assert.equal(await pendingLoss, info); assert.ok(Object.isFrozen(info));
  assert.deepEqual(info, {generation: 1, recoveryAttempt: 0, reason: 'unknown', message: 'Lost first'});
  assert.equal(f.app.state, 'lost'); assert.equal(f.app.failed, true); assert.equal(f.app.recoverable, true);
  assert.equal(f.app.suspended, true); assert.equal(f.app.lastFrameRendered, false);
  assert.equal(old.disposed, true); assert.equal(old.lifetime.signal.aborted, true); assert.equal(f.context.device, null);
  assert.equal(f.gpu.requests.length, 1, 'loss must not automatically request an adapter');
  assert.throws(() => f.app.render(source.scene), code('DEVICE_LOST'));
  const next = f.enqueue(f.device('next'));
  assert.equal(await f.app.recover(), f.app); assert.equal(f.app.device, next);
  assert.equal(f.app.generation, 2); assert.equal(f.app.recoveryAttempts, 1); assert.equal(f.app.state, 'ready');
  assert.equal(f.app.lastError, null); assert.equal(f.app.lastLoss, info); assert.equal(f.app.lastFrameRendered, false);
  assert.equal(f.context.acquisitions, 1, 'reconstruction must not acquire a presentation texture');
  assert.equal(source.animationTime, 42); assert.equal(f.renderers[1].renders.length, 0);
  assert.equal(f.renderers[1].lifetime.generation, 2); assert.equal(f.renderers[1].lifetime.recoveryAttempt, 1);
  assert.notEqual(f.renderers[1].lifetime.signal, old.lifetime.signal);
  f.app.render(source.scene); assert.equal(f.renderers[1].renders[0].input, source.scene);
  assert.equal(old.renders.length, 1); assert.equal(f.first.destroyCount, 1);
  assert.ok(ownedTextures(f.first).every(t => t.destroyed === 1)); f.app.dispose(); assert.equal(next.destroyCount, 1);
});

test('committed drawing-buffer dimensions survive recovery; loss-time resize is deferred', async () => {
  const f = await setup({target: {width: 20, height: 10}});
  f.app.setSize(15, 9, 2); assert.deepEqual([f.app.width, f.app.height], [30, 18]);
  await lose(f); f.enqueue(f.device('next'));
  await f.app.recover(); assert.deepEqual([f.canvas.width, f.canvas.height], [30, 18]);
  await lose(f); const oldTextures = f.devices.flatMap(d => d.textures).length;
  assert.equal(f.app.resize(40, 24), f.app); assert.deepEqual([f.app.width, f.app.height], [40, 24]);
  assert.deepEqual([f.canvas.width, f.canvas.height], [30, 18]);
  assert.equal(f.devices.flatMap(d => d.textures).length, oldTextures);
  f.enqueue(f.device('third')); await f.app.recover(); assert.deepEqual([f.canvas.width, f.canvas.height], [40, 24]);
  f.app.dispose();
});

test('zero-sized suspended canvases reconstruct without allocating attachments or a frame', async () => {
  const f = await setup({target: {width: 0, height: 0, sampleCount: 4}});
  assert.equal(f.app.suspended, true); f.app.render({}); assert.equal(f.app.lastFrameRendered, false);
  await lose(f); const next = f.enqueue(f.device('next')); await f.app.recover(); f.app.render({});
  assert.equal(f.context.acquisitions, 0); assert.equal(next.textures.length, 0);
  f.app.resize(8, 8).render({}); assert.equal(f.context.acquisitions, 1); f.app.dispose();
});

test('borrowed devices require an explicit fresh replacement and are never destroyed', async () => {
  const f = recoveryFixture(), first = f.device('borrowed');
  f.app = await createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {device: first, gpu: f.gpu});
  await lose(f); assert.equal(first.destroyCount, 0);
  await assert.rejects(f.app.recover(), code('RECOVERY_DEVICE_REQUIRED'));
  await assert.rejects(f.app.recover({device: first}), code('RECOVERY_DEVICE'));
  await assert.rejects(f.app.recover({device: undefined}), code('RECOVERY_DEVICE'));
  assert.equal(f.app.recoveryAttempts, 0); assert.equal(f.gpu.requests.length, 0);
  const next = f.device('replacement'); await f.app.recover({device: next});
  assert.equal(f.app.ownsDevice, false); assert.equal(f.app.device, next); assert.equal(f.gpu.requests.length, 0);
  f.app.dispose(); assert.equal(next.destroyCount, 0); assert.equal(f.renderers[1].disposeCount, 1);
});

test('explicit null transitions borrowed recovery to acquired device ownership', async () => {
  const f = recoveryFixture(), first = f.device('borrowed');
  f.app = await createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {device: first, gpu: f.gpu});
  await lose(f); const owned = f.enqueue(f.device('owned')); await f.app.recover({device: null});
  assert.equal(f.app.ownsDevice, true); await lose(f); const next = f.enqueue(f.device('next'));
  await f.app.recover(); assert.equal(f.app.device, next); assert.equal(owned.destroyCount, 1);
  f.app.dispose(); assert.equal(first.destroyCount, 0); assert.equal(next.destroyCount, 1);
});

test('recovery can lend a replacement after an owned generation without destroying that device', async () => {
  const f = await setup(); await lose(f);
  const next = f.device('borrowed'); await f.app.recover({device: next});
  await lose(f); await assert.rejects(f.app.recover(), code('RECOVERY_DEVICE_REQUIRED'));
  assert.equal(f.gpu.requests.length, 1); assert.equal(next.destroyCount, 0); f.app.dispose();
});

test('capability requests are snapshotted and revalidated for each fresh adapter and device', async () => {
  const f = recoveryFixture();
  const first = f.enqueue(f.device('first', {features: ['float32-filterable']}));
  const options = {gpu: f.gpu, powerPreference: 'high-performance', requiredFeatures: ['float32-filterable'],
    requiredLimits: {maxBufferSize: 1048576, minUniformBufferOffsetAlignment: 256}, target: {sampleCount: 4}};
  f.app = await createRecoverableGpuCanvasRenderer(f.canvas, f.factory, options);
  options.requiredFeatures.length = 0; options.requiredLimits.maxBufferSize = 1; options.target.sampleCount = 1;
  await lose(f); f.enqueue(f.device('incapable'));
  await assert.rejects(f.app.recover(), code('FEATURE')); assert.equal(f.app.state, 'lost');
  assert.equal(f.app.generation, 1); assert.equal(f.gpu.deviceRequests.length, 1);
  const next = f.enqueue(f.device('capable', {features: ['float32-filterable']})); await f.app.recover();
  assert.equal(f.app.device, next); assert.equal(f.app.rendererOptions.sampleCount, 4);
  assert.deepEqual(f.gpu.deviceRequests[1], {requiredFeatures: ['float32-filterable'], requiredLimits: {maxBufferSize: 1048576, minUniformBufferOffsetAlignment: 256}});
  assert.ok(f.gpu.requests.every(r => r.powerPreference === 'high-performance'));
  f.app.dispose(); assert.equal(first.destroyCount, 1);
});

test('new-device dimensions and attachment budgets are rechecked rather than downscaled', async () => {
  const f = await setup(); f.app.resize(128, 64); await lose(f);
  const small = f.enqueue(f.device('small', {limits: {maxTextureDimension2D: 32}}));
  await assert.rejects(f.app.recover(), code('SIZE')); assert.equal(small.destroyCount, 1);
  assert.equal(f.app.state, 'lost'); assert.deepEqual([f.app.width, f.app.height], [128, 64]);
  f.app.resize(16, 16); f.enqueue(f.device('next')); await f.app.recover(); assert.equal(f.app.generation, 2); f.app.dispose();
});

test('failed reconstruction consumes one bounded attempt but publishes no generation', async () => {
  const f = await setup({maxRecoveryAttempts: 2}); await lose(f);
  await assert.rejects(f.app.recover(), code('ADAPTER')); assert.equal(f.app.recoveryAttempts, 1);
  await assert.rejects(f.app.recover(), code('ADAPTER')); assert.equal(f.app.recoveryAttempts, 2);
  assert.equal(f.app.state, 'lost'); assert.equal(f.app.generation, 1); assert.equal(f.app.recoverable, false);
  const count = f.gpu.requests.length; await assert.rejects(f.app.recover(), code('RECOVERY_LIMIT'));
  assert.equal(f.gpu.requests.length, count); assert.equal(f.app.recoveryAttempts, 2); f.app.dispose();
});

test('a ready canvas cannot be arbitrarily restarted or have recovery options silently changed', async () => {
  const f = await setup(); await assert.rejects(f.app.recover(), code('RECOVERY_STATE'));
  await lose(f);
  for (const settings of [null, [], {requiredFeatures: []}, {target: {width: 1}}, {device: false}])
    await assert.rejects(f.app.recover(settings), GpuCanvasError);
  assert.equal(f.app.recoveryAttempts, 0); assert.equal(f.gpu.requests.length, 1); f.app.dispose();
});

test('concurrent recovery and all GPU operations during reconstruction are rejected', async () => {
  const f = await setup(); await lose(f); const held = deferred(); f.adapters.push(held.promise);
  const recovery = f.app.recover(); assert.equal(f.app.state, 'recovering');
  await assert.rejects(f.app.recover(), code('RECOVERY_STATE'));
  assert.throws(() => f.app.render({}), code('RECOVERING')); assert.throws(() => f.app.resize(4, 4), code('RECOVERING'));
  await assert.rejects(f.app.prepare(), code('RECOVERING')); await assert.rejects(f.app.whenIdle(), code('RECOVERING'));
  const next = f.device('next'); held.resolve(f.adapter(next)); await recovery;
  assert.equal(f.app.recoveryAttempts, 1); assert.equal(f.app.generation, 2); f.app.dispose();
});

test('recovery option getters cannot recursively launch recovery or mutate its requested size', async () => {
  const f = await setup(); await lose(f); const next = f.device('next'); let recursive;
  await f.app.recover({get device() {
    recursive = f.app.recover(); recursive.catch(() => {});
    assert.throws(() => f.app.resize(4, 4), code('REENTRANT'));
    assert.throws(() => f.app.dispose(), code('REENTRANT')); return next;
  }});
  await assert.rejects(recursive, code('REENTRANT')); assert.equal(f.app.generation, 2);
  assert.deepEqual([f.app.width, f.app.height], [16, 12]); f.app.dispose();
});

test('loss cancels a pending preparation; its late completion cannot unlock or corrupt a new generation', async () => {
  const f = await setup(), oldReady = deferred(); f.renderers[0].prepareWait = oldReady.promise;
  const old = f.app.prepare(); old.catch(() => {}); await lose(f);
  await assert.rejects(old, code('DEVICE_LOST')); f.enqueue(f.device('next')); await f.app.recover();
  const nextReady = deferred(); f.renderers[1].prepareWait = nextReady.promise;
  const preparing = f.app.prepare(); oldReady.resolve(); await turns();
  assert.throws(() => f.app.render({}), code('REENTRANT'));
  nextReady.resolve(); await preparing; f.app.render({}); assert.equal(f.app.generation, 2); f.app.dispose();
});

test('old queue/renderer idle failures reject old callers but never stop the replacement', async () => {
  const f = await setup(), oldIdle = deferred(); f.app.render({}); f.renderers[0].idleWait = oldIdle.promise;
  const idle = f.app.whenIdle(); idle.catch(() => {}); await lose(f); await assert.rejects(idle, code('DEVICE_LOST'));
  f.enqueue(f.device('next')); await f.app.recover();
  oldIdle.reject(new Error('late old queue failure')); await turns();
  assert.equal(f.app.state, 'ready'); f.app.render({}); await f.app.whenIdle(); f.app.dispose();
});

test('an idle waiter does not clear a separately pending preparation lock', async () => {
  const f = await setup(), idle = deferred(), ready = deferred(); f.renderers[0].idleWait = idle.promise;
  const waiting = f.app.whenIdle(); f.renderers[0].prepareWait = ready.promise; const preparing = f.app.prepare();
  idle.resolve(); await waiting; assert.throws(() => f.app.render({}), code('REENTRANT'));
  ready.resolve(); await preparing; f.app.dispose();
});

test('disposal interrupts a hung requestDevice and retires the late acquired device exactly once', async () => {
  const f = await setup(); await lose(f); const acquisition = deferred(), late = f.device('late');
  f.adapters.push(f.adapter(late, {devicePromise: acquisition.promise}));
  const recovery = f.app.recover(); recovery.catch(() => {}); await turns();
  assert.equal(f.gpu.deviceRequests.length, 2); f.app.dispose(); await assert.rejects(recovery, code('DISPOSED'));
  acquisition.resolve(late); await turns();
  assert.equal(late.destroyCount, 1); assert.equal(f.context.configurations.length, 1); assert.equal(f.renderers.length, 1);
  assert.equal(f.app.state, 'disposed'); assert.equal(f.app.generation, 1); f.app.dispose();
});

test('external abort during recovery cancels work and prevents all future recovery', async () => {
  const signal = new AbortController(); const f = await setup({signal: signal.signal}); await lose(f);
  const held = deferred(); f.adapters.push(held.promise); const recovery = f.app.recover(); recovery.catch(() => {});
  signal.abort(); await assert.rejects(recovery, code('ABORTED')); assert.equal(f.app.state, 'failed');
  await assert.rejects(f.app.recover(), code('ABORTED')); await assert.rejects(f.app.whenLost(), code('ABORTED'));
  const next = f.device('not requested'); held.resolve(f.adapter(next)); await turns();
  assert.equal(f.gpu.deviceRequests.length, 1); assert.equal(next.destroyCount, 0); f.app.dispose();
});

test('abort inside a frame getter exits the synchronous boundary before resource retirement', async () => {
  const signal = new AbortController(); const f = await setup({signal: signal.signal});
  assert.throws(() => f.app.render({}, {get tag() { signal.abort(); return 'unused'; }}), code('ABORTED'));
  assert.equal(f.context.acquisitions, 0); assert.equal(f.first.destroyCount, 1);
  assert.equal(f.renderers[0].disposeCount, 1); assert.equal(f.app.state, 'failed'); f.app.dispose();
});

test('late asynchronous factory results are disposed without unconfiguring a different canvas owner', async () => {
  const f = recoveryFixture(), first = f.enqueue(f.device('first')), gate = deferred(); let pendingRenderer;
  const factory = (d, a, l) => { const r = f.factory(d, a, l); if (l.generation > 1) { pendingRenderer = r; return gate.promise; } return r; };
  f.app = await createRecoverableGpuCanvasRenderer(f.canvas, factory, {gpu: f.gpu});
  await lose(f); const oldDevice = f.enqueue(f.device('pending'));
  const recovery = f.app.recover(); recovery.catch(() => {}); await turns(50); assert.ok(pendingRenderer);
  f.app.dispose(); await assert.rejects(recovery, code('DISPOSED'));
  assert.equal(pendingRenderer.lifetime.signal.aborted, true); assert.equal(oldDevice.destroyCount, 1);
  const replacement = f.enqueue(f.device('replacement'));
  const other = await createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu});
  const unconfigurations = f.context.unconfigurations; gate.resolve(pendingRenderer); await turns(50);
  assert.equal(pendingRenderer.disposeCount, 1); assert.equal(f.context.unconfigurations, unconfigurations);
  assert.equal(f.context.device, replacement); other.render({}); await other.whenIdle(); other.dispose(); assert.equal(first.destroyCount, 1);
});

test('loss of the replacement during construction is retryable but never a published generation', async () => {
  const f = recoveryFixture(), first = f.enqueue(f.device('first')); const gate = deferred(); let pending;
  f.app = await createRecoverableGpuCanvasRenderer(f.canvas, (d, a, l) => {
    const r = f.factory(d, a, l); if (d.name === 'bad') { pending = r; return gate.promise; } return r;
  }, {gpu: f.gpu});
  await lose(f); const bad = f.enqueue(f.device('bad')); const recovery = f.app.recover(); recovery.catch(() => {});
  await turns(50); assert.ok(pending); bad.lose(); await assert.rejects(recovery, code('DEVICE_LOST'));
  assert.equal(f.app.state, 'lost'); assert.equal(f.app.generation, 1); assert.equal(f.app.recoveryAttempts, 1);
  f.enqueue(f.device('good')); await f.app.recover(); assert.equal(f.app.generation, 2);
  gate.resolve(pending); await turns(); assert.equal(pending.disposeCount, 1); assert.equal(f.app.state, 'ready'); f.app.dispose();
});

test('ordinary caller frame errors neither trigger recovery nor retire working state', async () => {
  const f = await setup(); const mistake = new Error('bad camera'); f.renderers[0].error = mistake;
  assert.throws(() => f.app.render({}), e => e === mistake); assert.equal(f.app.state, 'ready'); assert.equal(f.first.destroyCount, 0);
  await assert.rejects(f.app.recover(), code('RECOVERY_STATE'));
  f.renderers[0].error = null; f.app.render({}); await f.app.whenIdle(); f.app.dispose();
});

test('terminal renderer validation is not relabeled as loss when teardown destroys the owned device', async () => {
  const f = await setup(), invalid = new Error('invalid shader');
  const waiter = f.app.whenLost(); waiter.catch(() => {}); f.renderers[0].error = invalid; f.renderers[0].failed = true;
  await assert.rejects(f.app.whenIdle()); await turns(); assert.equal(f.app.state, 'failed'); assert.equal(f.app.lastLoss, null);
  await assert.rejects(waiter); const requests = f.gpu.requests.length;
  await assert.rejects(f.app.recover()); assert.equal(f.gpu.requests.length, requests); f.app.dispose();
});

test('native queue validation failure remains terminal even if its device-loss callback races rejection delivery', async () => {
  const f = await setup(), queue = deferred(); f.first.queue.fence = queue.promise; f.app.render({});
  const waiter = f.app.whenLost(); waiter.catch(() => {}); const idle = f.app.whenIdle(); idle.catch(() => {});
  queue.reject(new Error('native queue failed')); await assert.rejects(idle); await turns();
  assert.equal(f.app.state, 'failed'); assert.equal(f.app.lastLoss, null); assert.equal(f.app.recoverable, false);
  await assert.rejects(waiter); f.app.dispose();
});

test('loss waiters reset only after a replacement is fully ready', async () => {
  const f = await setup(); const loss1 = await lose(f); const held = deferred(); f.adapters.push(held.promise);
  const recovery = f.app.recover(); assert.equal(await f.app.whenLost(), loss1);
  const next = f.device('next'); held.resolve(f.adapter(next)); await recovery;
  let resolved = false; const loss2 = f.app.whenLost().then(info => { resolved = true; return info; });
  await turns(); assert.equal(resolved, false); next.lose(); assert.equal((await loss2).generation, 2); f.app.dispose();
});

test('disposed generations reject outstanding loss waiters and release logical canvas ownership', async () => {
  const f = await setup(); const pending = f.app.whenLost(); pending.catch(() => {}); f.app.dispose();
  await assert.rejects(pending, code('DISPOSED')); await assert.rejects(f.app.whenLost(), code('DISPOSED'));
  await assert.rejects(f.app.recover(), code('DISPOSED'));
  f.enqueue(f.device('another')); const next = await createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu}); next.dispose();
});

test('a lost recoverable owner retains its logical lease until explicitly disposed', async () => {
  const f = await setup(); await lose(f);
  await assert.rejects(createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu}), code('OWNERSHIP'));
  assert.equal(f.gpu.requests.length, 1); f.app.dispose();
});

test('a separately configured plain owner is never stolen or unconfigured by a failed recovery', async () => {
  const f = await setup(); await lose(f); const borrowed = f.device('separate');
  const other = await createGpuCanvasRenderer(f.canvas, f.factory, {device: borrowed});
  const rejected = f.enqueue(f.device('rejected')); await assert.rejects(f.app.recover(), code('OWNERSHIP'));
  assert.equal(f.context.device, borrowed); assert.equal(rejected.destroyCount, 1); other.render({}); other.dispose();
  f.enqueue(f.device('next')); await f.app.recover(); f.app.render({}); f.app.dispose();
});

test('invalid limits, dimensions, factories, signals and failed first initialization leave the canvas reusable', async () => {
  const f = recoveryFixture();
  for (const maxRecoveryAttempts of [-1, 65, 0.5, Infinity, NaN, null])
    await assert.rejects(createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, maxRecoveryAttempts}), code('RECOVERY_LIMIT'));
  await assert.rejects(createRecoverableGpuCanvasRenderer(f.canvas, null, {gpu: f.gpu}), code('FACTORY'));
  await assert.rejects(createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, signal: {}}), code('OPTIONS'));
  await assert.rejects(createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, target: {width: -1}}), code('SIZE'));
  await assert.rejects(createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu}), code('ADAPTER'));
  f.enqueue(f.device('first')); const app = await createRecoverableGpuCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, maxRecoveryAttempts: 0});
  app.device.lose(); await app.whenLost(); assert.equal(app.recoverable, false); await assert.rejects(app.recover(), code('RECOVERY_LIMIT')); app.dispose();
});
