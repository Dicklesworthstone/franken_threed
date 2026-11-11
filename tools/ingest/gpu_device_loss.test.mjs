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

test('disposed texture, environment and recoverable canvas owners release their live device subscriptions', () => {
  const source = `
    import assert from 'node:assert/strict';
    import {createGpuThreeTextures} from './three_textures.mjs';
    import {createGpuThreeEnvironment} from './three_environment.mjs';
    import {createGpuThreeBackground} from './three_background.mjs';
    import {createThreeStorageTextures} from './three_storage_textures.mjs';
    import {createRecoverableGpuCanvasRenderer} from './gpu_canvas_recovery.mjs';
    import {T} from './fixtures/target_storage_copy_fixture.mjs';
    class DataTexture extends T.Texture {
      constructor() {
        super();
        Object.assign(this, {image: {data: new Uint16Array(32).fill(0x3c00), width: 4, height: 2},
          isDataTexture: true, type: T.HalfFloatType, internalFormat: null, mapping: 303,
          colorSpace: T.LinearSRGBColorSpace, premultiplyAlpha: false, unpackAlignment: 1,
          onUpdate: null, updateRanges: [], version: 1});
        this.source = {data: this.image, dataReady: true, version: 1};
      }
    }
    const three = {...T, REVISION: '186', DataTexture, EquirectangularReflectionMapping: 303};
    let lose, subscriptions = 0;
    const lost = new Promise(resolve => { lose = resolve; }), then = lost.then.bind(lost);
    lost.then = (...args) => { subscriptions++; return then(...args); };
    // Do not keep a test-side allocation log: only production owners may retain
    // these source graphs. Shader compilation/submission is a recorded boundary.
    const device = {lost, limits: {maxTextureDimension2D: 4096, maxBufferSize: 65536,
        minUniformBufferOffsetAlignment: 256},
      pushErrorScope() {}, popErrorScope: async () => null,
      queue: {onSubmittedWorkDone: async () => { if (device.failFence) throw new Error('injected completion failure'); },
        writeBuffer() {}, writeTexture() {}, submit() {}},
      createBuffer: () => ({destroy() {}}),
      createTexture: descriptor => {
        if (device.failTexture) throw new Error('injected allocation failure');
        return {...descriptor, width: descriptor.size[0], height: descriptor.size[1],
        depthOrArrayLayers: descriptor.size[2] ?? 1, dimension: descriptor.dimension ?? '2d',
        sampleCount: descriptor.sampleCount ?? 1, mipLevelCount: descriptor.mipLevelCount ?? 1,
        createView() { return {texture: this}; }, destroy() {}};
      },
      createSampler: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
      createShaderModule: () => ({}), createRenderPipelineAsync: async () => {
        if (device.failPipeline) throw new Error('injected compilation failure');
        return {};
      }, createBindGroup: () => ({}),
      createCommandEncoder: () => ({beginRenderPass: () => ({setPipeline() {}, setBindGroup() {}, draw() {}, end() {}}),
        finish: () => ({})}),
      destroy() { assert.fail('borrowed device'); }};
    const canvas = () => ({width: 4, height: 2, getContext: () => ({
      configure() {}, unconfigure() {}, getCurrentTexture() {}})});
    const factory = () => ({render() {}, dispose() {}, whenIdle: async () => {}});
    const environmentOptions = {three, size: 2, diffuseSize: 1, lutSize: 2, samples: 1};
    const refs = [];
    async function cycle() {
      for (let index = 0; index < 20; index++) {
        const panorama = new DataTexture(), backdrop = new DataTexture();
        const environment = await createGpuThreeEnvironment(device, panorama, environmentOptions);
        const background = await createGpuThreeBackground(device, backdrop, {three});
        const textures = createGpuThreeTextures(device, {three});
        const storage = createThreeStorageTextures(three, {getDevice: () => device});
        const texture = new T.StorageTexture(); storage.ensure(storage.capture(texture));
        await storage.whenIdle();
        const session = await createRecoverableGpuCanvasRenderer(canvas(), factory, {device});
        environment.dispose(); background.dispose(); textures.dispose(); storage.dispose(); session.dispose();
        for (const value of [panorama, backdrop, environment, background, textures, storage, texture, session, session.canvas])
          refs.push(new WeakRef(value));
      }
    }
    await cycle();
    async function failedConstruction() {
      const panorama = new DataTexture(), backdrop = new DataTexture(), surface = canvas();
      device.failPipeline = true;
      await assert.rejects(createGpuThreeEnvironment(device, panorama, environmentOptions), /injected compilation failure/);
      await assert.rejects(createGpuThreeBackground(device, backdrop, {three}), /injected compilation failure/);
      device.failPipeline = false;
      const storage = createThreeStorageTextures(three, {getDevice: () => device}), texture = new T.StorageTexture();
      device.failTexture = true;
      assert.throws(() => storage.ensure(storage.capture(texture)), /injected allocation failure/);
      device.failTexture = false;
      const textures = createGpuThreeTextures(device, {three});
      device.failFence = true;
      await assert.rejects(textures.whenIdle(), /injected completion failure/);
      device.failFence = false;
      await assert.rejects(createRecoverableGpuCanvasRenderer(surface, () => { throw new Error('injected factory failure'); },
        {device}), /injected factory failure/);
      for (const value of [panorama, backdrop, surface, storage, texture, textures]) refs.push(new WeakRef(value));
    }
    await failedConstruction();
    for (let turn = 0; turn < 20; turn++) { await new Promise(setImmediate); global.gc(); }
    assert.equal(refs.filter(ref => ref.deref()).length, 0, 'disposed texture and canvas graphs must not await device loss');
    assert.equal(subscriptions, 1, 'texture and recovery churn must share one native loss reaction');
    const environment = await createGpuThreeEnvironment(device, new DataTexture(), environmentOptions);
    const background = await createGpuThreeBackground(device, new DataTexture(), {three});
    const textures = createGpuThreeTextures(device, {three});
    const sampled = new DataTexture();
    textures.prepare([sampled]); sampled.dispose(); textures.prepare([sampled]);
    textures.retain([]); textures.prepare([sampled]);
    const storage = createThreeStorageTextures(three, {getDevice: () => device});
    const stored = new T.StorageTexture();
    storage.ensure(storage.capture(stored)); stored.dispose(); storage.ensure(storage.capture(stored));
    const session = await createRecoverableGpuCanvasRenderer(canvas(), factory, {device});
    lose({reason: 'unknown', message: 'device removed'}); await new Promise(setImmediate);
    assert.equal(environment.failed, true); assert.equal(environment.allocatedBytes, 0);
    assert.equal(background.failed, true); assert.equal(background.allocatedBytes, 0);
    assert.equal(textures.failed, true);
    await assert.rejects(storage.whenIdle(), {code: 'THREE_STORAGE_TEXTURE_DEVICE_LOST'});
    assert.equal(storage.allocatedBytes, 0);
    assert.equal(session.state, 'lost'); assert.equal(session.recoverable, true);
    environment.dispose(); background.dispose(); textures.dispose(); storage.dispose(); session.dispose();
  `;
  execFileSync(process.execPath, ['--expose-gc', '--input-type=module', '-e', source], {
    cwd: fileURLToPath(new URL('.', import.meta.url)), encoding: 'utf8', timeout: 20000,
  });
});
