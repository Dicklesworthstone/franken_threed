import test from 'node:test';
import assert from 'node:assert/strict';
import {withStorageTextureCopies} from './three_storage_texture_copy.mjs';
import {withThreeNativeCapture} from './three_node_shadows.mjs';
import {fixture, T, target, setFormat, payload, raw, formats, box, position, deferred, flush} from './fixtures/target_storage_copy_fixture.mjs';

// Actual copy facade, storage pixel transfers, both source owners, native owner
// and copy/readback helpers. Only source class identities, GPU calls and the
// renderer's queue/compute-turn driver are structural doubles.
class Base {
  constructor() {
    const f = this.fixture = fixture(); Object.assign(this, f.renderer);
    this.isWebGPURenderer = true; this._initialized = true; this._pending = [];
    this._targetLifetime = new AbortController(); this._completions = []; this.limit = 64;
    this.events = []; this.inspector = {copyTextureToTexture: (...args) => this.events.push(args)};
  }
  _enqueue(request) { if (this._pending.length >= this.limit) throw new Error('queue limit'); this._pending.push(request); }
  drain() {
    while (this._pending.length) {
      const request = this._pending.shift();
      try { const result = request.issue(); request.resolve(result); }
      catch (error) { request.reject(error); }
    }
  }
  async waitForGPU() { this.drain(); await Promise.all([this._targets.whenIdle(), this._storageTextures.whenIdle()]); }
  dispose() { this._disposed = true; this._targetLifetime.abort(new Error('renderer disposed')); this.fixture.dispose(); }
}
const Renderer = withStorageTextureCopies(Base);
const texture = format => setFormat(new T.StorageTexture(), format);
const read = (r, value, colorIndex = 0) => r._targets.ensure(r._targets.capture(value)).storage.readPixels({colorIndex});

for (const [format] of formats) test(`${format}: public upload -> storage-to-target -> target-to-storage -> readback retains bytes`, async () => {
  const renderer = new Renderer(), source = texture(format), destination = target({colors: ['rgba8unorm', format]}), back = texture(format);
  const data = payload(format); await renderer.writeStorageTexturePixelsAsync(source, data);
  const before = renderer.fixture.calls.fences;
  assert.equal(renderer.copyTextureToTexture(source, destination.textures[1]), undefined);
  assert.equal(renderer.fixture.calls.fences, before);
  assert.equal(renderer.copyTextureToTexture(destination.textures[1], back), undefined);
  assert.deepEqual(await renderer.readStorageTexturePixelsAsync(back), raw(data));
  assert.deepEqual(raw(await read(renderer, destination, 1)), raw(data));
  assert.equal(renderer.events.length, 2); await renderer.waitForGPU(); renderer.dispose();
});

test('queued uploads and both copy directions preserve ordering and independent destination histories', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), first = target(), second = target(), back = texture('rgba8unorm');
  renderer._preparing = true;
  const a = renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(31));
  renderer.copyTextureToTexture(source, first.texture);
  const b = renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(79));
  renderer.copyTextureToTexture(source, second.texture);
  renderer.copyTextureToTexture(first.texture, back);
  const c = renderer.readStorageTexturePixelsAsync(back);
  assert.equal(renderer.fixture.calls.textures.length, 0); assert.equal(renderer._pending.length, 6);
  renderer._preparing = false; renderer.drain(); await Promise.all([a, b]);
  assert.ok((await c).every(value => value === 31));
  assert.ok((await read(renderer, first)).every(value => value === 31));
  assert.ok((await read(renderer, second)).every(value => value === 79));
  await renderer.waitForGPU(); renderer.dispose();
});

test('GPU completion and mapping never block later copy submission in the shared queue', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target(), fence = deferred(), mapping = deferred();
  renderer._preparing = true; renderer.fixture.controls.fence = fence; renderer.fixture.controls.mapping = mapping;
  const upload = renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(44));
  const storageRead = renderer.readStorageTexturePixelsAsync(source);
  renderer.copyTextureToTexture(source, destination.texture); renderer._preparing = false; renderer.drain();
  assert.equal(renderer.fixture.calls.copies.length, 1, 'copy submitted while upload/read promises still await completion');
  fence.resolve(); mapping.resolve(); await Promise.all([upload, storageRead]);
  assert.ok((await read(renderer, destination)).every(value => value === 44)); renderer.dispose();
});

test('source boxes and positions are captured when the public copy is accepted', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target();
  await renderer.writeStorageTexturePixelsAsync(source, payload('rgba8unorm')); renderer._preparing = true;
  const region = box(1, 1, 2, 2), at = position(2, 2);
  renderer.copyTextureToTexture(source, destination.texture, region, at);
  region.min.x = 0; region.max.x = 4; at.x = 0;
  renderer._preparing = false; renderer.drain();
  assert.deepEqual(renderer.fixture.calls.copies[0].source.origin, [1, 1, 0]);
  assert.deepEqual(renderer.fixture.calls.copies[0].destination.origin, [2, 2, 0]);
  await renderer.waitForGPU(); renderer.dispose();
});

test('nested-write protection uses the selected attachment rather than blocking independent siblings', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target({colors: ['rgba8unorm', 'rgba8unorm']});
  await renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(71));
  withThreeNativeCapture(renderer, sampled => {
    sampled(destination.textures[1]);
    assert.throws(() => renderer.copyTextureToTexture(source, destination.textures[1]), /pending parent pass/);
    renderer.copyTextureToTexture(source, destination.texture);
  });
  assert.equal(renderer.fixture.calls.copies.length, 1);
  assert.ok((await read(renderer, destination)).every(value => value === 71));
  assert.ok((await read(renderer, destination, 1)).every(value => value === 0)); renderer.dispose();
});

test('storage copies invoked during a pending compute turn remain in that turn ordering', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target(), gate = deferred();
  const tasks = []; renderer._computeTurn = {submit(issue) { const task = gate.promise.then(issue); tasks.push(task); return task; }};
  const upload = renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(92));
  assert.equal(renderer.copyTextureToTexture(source, destination.texture), undefined);
  renderer._computeTurn = null; assert.equal(renderer.fixture.calls.textures.length, 0);
  gate.resolve(); await Promise.all([upload, ...tasks]);
  assert.ok((await read(renderer, destination)).every(value => value === 92)); renderer.dispose();
});

test('a stale queued destination reports an error at waitForGPU and does not allocate endpoints', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target({colors: ['rgba8unorm', 'rgba8unorm']});
  renderer._preparing = true; renderer.copyTextureToTexture(source, destination.textures[1]);
  destination.textures.reverse(); renderer._preparing = false;
  await assert.rejects(renderer.waitForGPU(), /STALE/); assert.equal(renderer.fixture.calls.textures.length, 0);
  assert.throws(() => renderer.copyTextureToTexture(source, destination.texture), /STALE/); renderer.dispose();
});

test('renderer disposal before queued copies prevents native work', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target(); renderer._preparing = true;
  renderer.copyTextureToTexture(source, destination.texture); renderer.dispose(); renderer.drain();
  assert.match(renderer._deferredError.message, /DISPOSED/); assert.equal(renderer.fixture.calls.textures.length, 0);
});

test('source API routes remain explicit and raw copies do not require float filtering', async () => {
  const renderer = new Renderer(), source = texture('rgba32float'), destination = target({colors: ['rgba32float']});
  renderer._initialized = false; assert.throws(() => renderer.copyTextureToTexture(source, destination.texture), /INITIALIZATION/);
  renderer._initialized = true; renderer.isWebGPURenderer = false;
  assert.throws(() => renderer.copyTextureToTexture(source, destination.texture), /ROUTE/);
  renderer.isWebGPURenderer = true; const data = payload('rgba32float'); await renderer.writeStorageTexturePixelsAsync(source, data);
  renderer.copyTextureToTexture(source, destination.texture);
  assert.deepEqual(raw(await read(renderer, destination)), raw(data)); assert.equal(renderer.fixture.calls.samplers.length, 0);
  renderer.dispose();
});

test('the new branch preserves ordinary-target delegation and storage-to-storage routing', async () => {
  const renderer = new Renderer(), a = target(), b = target(), seen = [];
  renderer._transfers = {copyTexture(...args) { seen.push(args); return Promise.resolve(); }};
  renderer.copyTextureToTexture(a.texture, b.texture); assert.deepEqual(seen[0], [a.texture, b.texture, null, null, 0, 0]);
  const source = texture('rgba8unorm'), destination = texture('rgba8unorm');
  await renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(58)); renderer.copyTextureToTexture(source, destination);
  assert.ok((await renderer.readStorageTexturePixelsAsync(destination)).every(value => value === 58)); renderer.dispose();
});

test('copy validation failures are observed by the target and facade completion boundary', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target();
  await renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(22));
  renderer._targets.ensure(renderer._targets.capture(destination)); await renderer._targets.whenIdle();
  renderer.fixture.controls.onCopy = () => { renderer.fixture.controls.validation = {message: 'native copy invalid'}; };
  renderer.copyTextureToTexture(source, destination.texture);
  await assert.rejects(read(renderer, destination), /native copy invalid/);
  await assert.rejects(renderer.waitForGPU(), /native copy invalid/); renderer.dispose();
});

test('queue-capacity rejection cannot partially submit a storage-to-target copy', () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target();
  renderer.limit = 0; renderer._preparing = true;
  assert.throws(() => renderer.copyTextureToTexture(source, destination.texture), /queue limit/);
  assert.equal(renderer.fixture.calls.textures.length, 0); assert.equal(renderer.fixture.calls.submissions.length, 0); renderer.dispose();
});


test('a copy cannot overtake already queued work after preparation ends but before draining starts', async () => {
  const renderer = new Renderer(), source = texture('rgba8unorm'), destination = target();
  renderer._preparing = true;
  const upload = renderer.writeStorageTexturePixelsAsync(source, new Uint8Array(64).fill(107));
  renderer._preparing = false;
  renderer.copyTextureToTexture(source, destination.texture);
  assert.equal(renderer._pending.length, 2); assert.equal(renderer.fixture.calls.textures.length, 0);
  renderer.drain(); await upload; assert.ok((await read(renderer, destination)).every(value => value === 107)); renderer.dispose();
});
