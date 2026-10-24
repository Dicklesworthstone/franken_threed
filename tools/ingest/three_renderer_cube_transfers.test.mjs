/** Public cube-transfer admission with actual pinned source classes, renderer,
 * target/storage owners and the existing native byte/descriptor recorder.
 * No shader execution or pixel-parity claim follows from this recording suite. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {transferDevice, deferred} from './fixtures/target_transfer_fixture.mjs';

function fixture(t, {holdInitialization = false} = {}) {
  const f = transferDevice(), initialization = deferred();
  f.device.limits.maxTextureArrayLayers = 256;
  const gpu = {async requestAdapter() {
    await initialization.promise;
    return {features: new Set(), async requestDevice() { return f.device; }};
  }};
  const Renderer = createWebGPURendererClass(T, holdInitialization ? {gpu} : {});
  const canvas = {width: 4, height: 4, style: {}, getContext: () => ({
    configure() {}, unconfigure() {}, getCurrentTexture() { assert.fail('Target transfers must not acquire the canvas'); },
  })};
  const renderer = new Renderer({canvas, depth: false, ...(holdInitialization ? {} : {device: f.device})});
  const cube = new T.CubeRenderTarget(4); cube.depthTexture = new T.CubeDepthTexture(4);
  const ordinary = new T.RenderTarget(4, 4); ordinary.depthTexture = new T.DepthTexture(4, 4);
  const attribute = new T.StorageBufferAttribute(new Uint32Array(16), 4);
  t.after(() => { initialization.resolve(); renderer.dispose(); });
  return {...f, renderer, cube, ordinary, attribute, initialization};
}

function activity(f) {
  const r = f.renderer;
  return {pending: r._pending.length, transfers: r.info.f3d.pendingTransfers,
    transferBytes: r.info.f3d.pendingTransferBytes, targetBytes: r._targets.allocatedBytes,
    storageBytes: r._storageTextures.allocatedBytes, storageEntries: r._storageTextures.size,
    textures: f.calls.textures.length, buffers: f.calls.buffers.length, uploads: f.calls.uploads.length,
    copies: f.calls.colorCopies.length, submissions: f.calls.submissions.length};
}

test('public 2D target uploads, copies and readbacks reject actual cubes before initializing or queuing work', async t => {
  const f = fixture(t), r = f.renderer, {cube, ordinary, attribute} = f;
  const before = activity(f);
  const operations = [
    ['color upload', () => r.writeRenderTargetPixelsAsync(cube, new Uint8Array(64))],
    ['color copy source', () => r.copyRenderTargetPixelsAsync(cube, ordinary)],
    ['color copy destination', () => r.copyRenderTargetPixelsAsync(ordinary, cube)],
    ['depth copy source', () => r.copyRenderTargetDepthPixelsAsync(cube, ordinary)],
    ['depth copy destination', () => r.copyRenderTargetDepthPixelsAsync(ordinary, cube)],
    ['default face color readback', () => r.readRenderTargetPixelsAsync(cube, 0, 0, 4, 4)],
    ['mip-zero color readback', () => r.readRenderTargetMipPixelsAsync(cube, 0)],
    ['depth readback', () => r.readRenderTargetDepthPixelsAsync(cube)],
    ['storage buffer upload', () => r.copyStorageBufferToRenderTargetAsync(attribute, cube), 'THREE_COMPUTE_TARGET'],
  ];
  for (const [name, run, code = 'F3D_RENDERER_UNSUPPORTED'] of operations) {
    await assert.rejects(run(), {code, message: /Cube target/}, name);
    assert.deepEqual(activity(f), before, `${name} must reject before reserving or submitting a transfer`);
  }
  assert.equal(r._initPromise, null); assert.equal(r._deferredError, null);
  assert.equal(cube.texture.renderTarget, null); assert.equal(cube.depthTexture.isRenderTargetTexture, false);
  // Rejected admission must not poison the next valid public operation.
  await r.writeRenderTargetPixelsAsync(ordinary, new Uint8Array(64).fill(19));
  assert.equal(f.calls.uploads.length, 1); assert.ok(f.calls.uploads[0].data.every(value => value === 19));
  assert.equal(r._targets.capture(cube).storage, null);
});

test('cube texture copies cannot enter a blocked public FIFO through depth ownership or a supplied color owner pointer', async t => {
  const f = fixture(t, {holdInitialization: true}), r = f.renderer, {cube, ordinary, attribute} = f;
  const accepted = r.writeRenderTargetPixelsAsync(ordinary, new Uint8Array(64).fill(29));
  assert.ok(r._drain); assert.equal(r._pending.length, 1); assert.equal(f.calls.textures.length, 0);
  const before = activity(f), refusal = {code: 'F3D_RENDERER_UNSUPPORTED', message: /Cube target/};
  // Pinned CubeRenderTarget leaves this pointer null. Supplying the correct
  // pointer must not turn an otherwise unsupported cube into a 2D face-zero copy.
  assert.equal(cube.texture.renderTarget, null); cube.texture.renderTarget = cube;
  for (const [name, source, destination] of [
    ['depth source', cube.depthTexture, ordinary.depthTexture],
    ['depth destination', ordinary.depthTexture, cube.depthTexture],
    ['color source', cube.texture, ordinary.texture],
    ['color destination', ordinary.texture, cube.texture],
  ]) {
    assert.throws(() => r.copyTextureToTexture(source, destination), refusal, name);
    await assert.rejects(r.copyTextureToTextureAsync(source, destination), refusal, name);
    assert.deepEqual(activity(f), before, `${name} must leave the accepted upload as the only queued request`);
  }
  await assert.rejects(r.copyStorageBufferToRenderTargetAsync(attribute, cube),
    {code: 'THREE_COMPUTE_TARGET', message: /Cube target/});
  assert.deepEqual(activity(f), before); assert.equal(r._deferredError, null);
  f.initialization.resolve(); await accepted; await r.waitForGPU();
  assert.equal(f.calls.uploads.length, 1); assert.ok(f.calls.uploads[0].data.every(value => value === 29));
  assert.equal(r._targets.capture(cube).storage, null);
  assert.equal(r.info.f3d.pendingTransfers, 0); assert.equal(r.info.f3d.pendingTransferBytes, 0);
});

test('public cube-to-storage copies reject before capturing or allocating destination storage', async t => {
  const f = fixture(t), r = f.renderer, {cube} = f;
  await r.init();
  const destination = new T.StorageTexture(4, 4), before = activity(f);
  assert.throws(() => r.copyTextureToTexture(cube.texture, destination), {code: 'THREE_STORAGE_COPY_SOURCE'});
  assert.deepEqual(activity(f), before); assert.equal(cube.texture.renderTarget, null);
  cube.texture.renderTarget = cube;
  assert.throws(() => r.copyTextureToTexture(cube.texture, destination),
    {code: 'THREE_STORAGE_COPY_SOURCE', message: /Cube target/});
  assert.throws(() => r.copyTextureToTexture(cube.depthTexture, destination), {code: 'THREE_STORAGE_COPY_SOURCE'});
  assert.throws(() => r.copyTextureToTexture(destination, cube.texture), {code: 'THREE_STORAGE_COPY_SOURCE'});
  assert.deepEqual(activity(f), before); assert.equal(r._deferredError, null);
  assert.equal(destination.version, 0); assert.equal(r._storageTextures.size, 0);
});

test('public initRenderTarget still initializes real cube color and depth once without a face transfer', async t => {
  const f = fixture(t, {holdInitialization: true}), r = f.renderer, {cube} = f;
  assert.equal(r.initRenderTarget(cube), undefined);
  const complete = r.initRenderTargetAsync(cube);
  assert.equal(f.calls.textures.length, 0); assert.equal(r._pending.length, 2);
  f.initialization.resolve(); await complete; await r.waitForGPU();
  assert.equal(f.calls.textures.length, 2);
  assert.deepEqual(f.calls.textures.map(texture => [texture.format, texture.size.depthOrArrayLayers, texture.sampleCount]),
    [['rgba8unorm', 6, 1], ['depth24plus', 6, 1]]);
  assert.equal(r._targets.allocatedBytes, 4 * 4 * 6 * 8);
  assert.equal(cube.texture.renderTarget, null); assert.equal(cube.depthTexture.renderTarget, cube);
  assert.equal(r.info.f3d.pendingTransfers, 0); assert.equal(r.info.f3d.pendingTransferBytes, 0);
  const before = activity(f);
  await r.initRenderTargetAsync(cube); await r.waitForGPU();
  assert.deepEqual(activity(f), before);
  assert.equal(f.calls.uploads.length, 0); assert.equal(f.calls.colorCopies.length, 0); assert.equal(f.calls.submissions.length, 0);
});
