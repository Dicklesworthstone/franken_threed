import assert from 'node:assert/strict';
import test from 'node:test';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {subscribeGpuDeviceLoss} from './gpu_device_loss.mjs';
import {createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';

function lossFixture() {
  let resolve, reject, subscriptions = 0;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  const then = promise.then.bind(promise);
  promise.then = (...args) => { subscriptions++; return then(...args); };
  return {promise, resolve, reject, get subscriptions() { return subscriptions; }};
}

test('disposed owners remove subscriptions without accumulating native promise reactions', async () => {
  const lost = lossFixture(), seen = [];
  for (let index = 0; index < 1000; index++) {
    const release = subscribeGpuDeviceLoss(lost.promise, () => assert.fail('disposed owner'));
    release(); release();
  }
  subscribeGpuDeviceLoss(lost.promise, info => seen.push(info));
  assert.equal(lost.subscriptions, 1);
  const info = {reason: 'unknown', message: 'adapter removed'};
  lost.resolve(info); await Promise.resolve();
  assert.deepEqual(seen, [info]);
});

test('loss cleanup can retire another owner before its notification', async () => {
  const lost = lossFixture(), seen = [];
  let releaseSecond;
  subscribeGpuDeviceLoss(lost.promise, () => { seen.push('first'); releaseSecond(); });
  releaseSecond = subscribeGpuDeviceLoss(lost.promise, () => assert.fail('retired child'));
  subscribeGpuDeviceLoss(lost.promise, () => seen.push('third'));
  lost.resolve({reason: 'unknown'}); await Promise.resolve();
  assert.deepEqual(seen, ['first', 'third']);
});

for (const rejected of [false, true]) test(`late loss subscriptions remain asynchronous and cancellable (${rejected})`, async () => {
  const lost = lossFixture(), reason = new Error('native loss rejection'), seen = [];
  const receive = value => seen.push(value);
  subscribeGpuDeviceLoss(lost.promise, receive, receive);
  if (rejected) lost.reject(reason); else lost.resolve(reason);
  await Promise.resolve(); assert.deepEqual(seen, [reason]);
  subscribeGpuDeviceLoss(lost.promise, receive, receive);
  const release = subscribeGpuDeviceLoss(lost.promise, () => assert.fail('cancelled'), () => assert.fail('cancelled'));
  release(); assert.deepEqual(seen, [reason]);
  await Promise.resolve(); assert.deepEqual(seen, [reason, reason]);
  assert.equal(lost.subscriptions, 1);
});

test('geometry residency release preserves the loss subscription for legal reuse', async () => {
  const lost = lossFixture(), device = {
    lost: lost.promise, limits: {maxBufferSize: 65536},
    createBuffer: () => ({destroy() {}}), pushErrorScope() {}, popErrorScope: async () => null,
    queue: {writeBuffer() {}, onSubmittedWorkDone: async () => {}},
  };
  const geometry = {attributes: {position: {array: new Float32Array(9), count: 3, itemSize: 3, version: 0,
    updateRanges: [], clearUpdateRanges() { this.updateRanges = []; }, onUploadCallback() {}}}};
  const owner = createGpuBufferGeometry(device, geometry);
  await owner.whenIdle(); owner.release(); owner.update();
  assert.equal(owner.vertexCount, 3);
  lost.resolve({message: 'lost after residency recreation'}); await Promise.resolve();
  await assert.rejects(owner.whenIdle(), {code: 'GEOMETRY_GPU_LOST'});
  assert.equal(owner.bufferBytes, 0); owner.dispose();
});

test('disposed canvas, target, session and geometry graphs are collectible while their borrowed device stays live', () => {
  // Run with an explicit collector in a child so the normal test command needs
  // no V8 flags. These are real production owners over a recording GPU boundary,
  // not an assertion about actual driver allocation or rendered pixels.
  const source = `
    import assert from 'node:assert/strict';
    import {createGpuCanvasTarget} from './gpu_canvas.mjs';
    import {createGpuCanvasRenderer} from './gpu_canvas_renderer.mjs';
    import {createGpuRenderTarget} from './gpu_render_target.mjs';
    import {createGpuRenderTargetRenderer} from './gpu_render_target_renderer.mjs';
    import {createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';
    import {subscribeGpuDeviceLoss} from './gpu_device_loss.mjs';
    let lose, subscriptions = 0;
    const lost = new Promise(resolve => { lose = resolve; }), then = lost.then.bind(lost);
    lost.then = (...args) => { subscriptions++; return then(...args); };
    const device = {lost, limits: {maxTextureDimension2D: 4096, maxBufferSize: 65536},
      pushErrorScope() {}, popErrorScope: async () => null,
      queue: {onSubmittedWorkDone: async () => {}, writeBuffer() {}},
      createBuffer: () => ({destroy() {}}),
      createTexture: () => ({createView: () => ({}), destroy() {}}),
      destroy() { assert.fail('borrowed device'); }};
    const canvas = () => ({width: 16, height: 8, getContext: () => ({
      configure() {}, unconfigure() {}, getCurrentTexture() {}})});
    const factory = () => ({render() {}, dispose() {}, whenIdle: async () => {}});
    const refs = [];
    async function cycle() {
      for (let index = 0; index < 50; index++) {
        const surface = canvas(), target = createGpuCanvasTarget(device, surface);
        await target.whenIdle(); target.dispose(); refs.push(new WeakRef(surface), new WeakRef(target));
        const session = await createGpuCanvasRenderer(canvas(), factory, {device});
        session.dispose(); refs.push(new WeakRef(session), new WeakRef(session.canvas));
        const output = createGpuRenderTarget(device);
        await output.whenIdle(); output.dispose(); refs.push(new WeakRef(output));
        const offscreen = await createGpuRenderTargetRenderer(device, factory);
        offscreen.dispose(); refs.push(new WeakRef(offscreen));
        const geometry = {attributes: {position: {array: new Float32Array(9), count: 3, itemSize: 3, version: 0,
          updateRanges: [], clearUpdateRanges() { this.updateRanges = []; }, onUploadCallback() {}}}};
        const gpu = createGpuBufferGeometry(device, geometry);
        await gpu.whenIdle(); gpu.dispose(); refs.push(new WeakRef(geometry), new WeakRef(gpu));
      }
    }
    await cycle();
    for (let turn = 0; turn < 20; turn++) { await new Promise(setImmediate); global.gc(); }
    assert.equal(refs.filter(ref => ref.deref()).length, 0, 'disposed source graphs must not await device loss');
    assert.equal(subscriptions, 1, 'churn must not accumulate dormant promise reactions');
    // A faulty subscriber retains the rejection behavior of its independent
    // promise reaction and cannot prevent other live owners from cleaning up.
    const failure = new Error('cleanup failure'), errors = [];
    process.on('unhandledRejection', reason => errors.push(reason));
    let survivor = false;
    subscribeGpuDeviceLoss(device.lost, () => { throw failure; });
    subscribeGpuDeviceLoss(device.lost, () => { survivor = true; });
    lose({reason: 'unknown'}); await new Promise(setImmediate);
    assert.equal(survivor, true); assert.deepEqual(errors, [failure]);
  `;
  execFileSync(process.execPath, ['--expose-gc', '--input-type=module', '-e', source], {
    cwd: fileURLToPath(new URL('.', import.meta.url)), encoding: 'utf8', timeout: 20000,
  });
});
