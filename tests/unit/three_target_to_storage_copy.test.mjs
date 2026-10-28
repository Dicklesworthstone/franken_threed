import test from 'node:test';
import assert from 'node:assert/strict';
import {captureTargetToStorageCopy} from '../../tools/ingest/three_target_texture_copy.mjs';
import {THREE, setup, renderTarget, storageTexture, box, at, deferred, sourceBytes, storageBytes} from './native_texture_copy_fixture.mjs';
const copies = context => context.device.calls.filter(call => call[0] === 'copy');

for (const format of ['rgba8unorm', 'rgba16float', 'r32float', 'rg32float', 'rgba32float'])
  test(`rendered ${format} color copies bit-for-bit without sampler, upload, readback or UV replica`, async () => {
    const c = setup(), target = renderTarget(4, 4, format), destination = storageTexture(4, 4, format);
    const bytes = sourceBytes(c, target);
    for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 137 + 17) % 256;
    assert.equal(c.renderer.copyTextureToTexture(target.texture, destination), undefined);
    assert.deepEqual(storageBytes(c, destination), bytes);
    assert.equal(copies(c).length, 1); assert.equal(c.device.textures.length, 2); assert.equal(c.device.buffers.length, 0);
    assert.equal(c.device.calls.some(call => ['upload', 'mip', 'compute'].includes(call[0])), false);
    assert.equal(c.targets.capture(target).dirty, false); assert.equal(c.targets.capture(target).depthDirty, false);
    await c.renderer.waitForGPU(); c.renderer.dispose();
  });

test('target region copied into one array layer leaves all other bytes untouched', () => {
  const c = setup(), target = renderTarget(), destination = storageTexture(4, 4, 'rgba8unorm', 3);
  const bytes = sourceBytes(c, target); bytes.set(Array.from({length: bytes.length}, (_, i) => i));
  storageBytes(c, destination).fill(255);
  c.renderer.copyTextureToTexture(target.texture, destination, box(1, 1, 2, 2), at(0, 2, 1));
  const output = storageBytes(c, destination);
  for (let z = 0; z < 3; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) for (let k = 0; k < 4; k++) {
    const copied = z === 1 && x < 2 && y >= 2;
    assert.equal(output[((z * 4 + y) * 4 + x) * 4 + k], copied ? bytes[((y - 1) * 4 + x + 1) * 4 + k] : 255);
  }
  c.renderer.dispose();
});

test('source MSAA resolves are copied instead of the multisampled render attachment', () => {
  const c = setup(), target = renderTarget(4, 4, 'rgba8unorm', 4), destination = storageTexture();
  sourceBytes(c, target).fill(91); c.renderer.copyTextureToTexture(target.texture, destination);
  assert.equal(copies(c)[0][1].texture.descriptor.sampleCount, 1);
  assert.ok(storageBytes(c, destination).every(n => n === 91)); c.renderer.dispose();
});

test('selected source mip is refreshed before copying and target UV/depth state stays untouched', () => {
  const c = setup(), target = renderTarget(8, 8), destination = storageTexture(4, 4);
  target.texture.generateMipmaps = true; sourceBytes(c, target, 1).fill(37);
  c.renderer.copyTextureToTexture(target.texture, destination, null, null, 1);
  assert.ok(storageBytes(c, destination).every(n => n === 37));
  assert.deepEqual(c.device.calls.filter(call => ['target-mips', 'copy'].includes(call[0])).map(call => call[0]), ['target-mips', 'copy']);
  assert.equal(copies(c)[0][1].mipLevel, 1); assert.equal(c.targets.capture(target).depthDirty, false); c.renderer.dispose();
});

test('manual destination mip copy preserves the base and emits no automatic reduction', () => {
  const c = setup(), target = renderTarget(2, 2), destination = storageTexture(4, 4);
  destination.generateMipmaps = true; destination.mipmapsAutoUpdate = true;
  storageBytes(c, destination).fill(123); sourceBytes(c, target).fill(47);
  c.renderer.copyTextureToTexture(target.texture, destination, null, null, 0, 1);
  assert.ok(storageBytes(c, destination).every(n => n === 123));
  assert.ok(storageBytes(c, destination, 1).every(n => n === 47));
  assert.equal(c.device.calls.some(call => call[0] === 'mip'), false); c.renderer.dispose();
});

test('deferred render/copy/compute order observes source GPU contents at submission, not capture', async () => {
  const c = setup(), target = renderTarget(), destination = storageTexture(), gate = deferred(), order = [];
  const producer = c.enqueue({prepare: gate.promise, issue() { sourceBytes(c, target).fill(91); order.push('produce'); }});
  c.renderer.copyTextureToTexture(target.texture, destination);
  assert.equal(c.device.textures.length, 0);
  const consumer = c.enqueue({issue() { assert.ok(storageBytes(c, destination).every(n => n === 91)); order.push('consume'); }});
  const later = c.enqueue({issue() { sourceBytes(c, target).fill(17); }});
  gate.resolve(); await Promise.all([producer, consumer, later]); await c.renderer.waitForGPU();
  assert.deepEqual(order, ['produce', 'consume']); assert.ok(storageBytes(c, destination).every(n => n === 91));
  c.renderer.dispose();
});

test('queued copy captures mutable boxes, positions and destination mip-update policy', async () => {
  const c = setup(), target = renderTarget(), destination = storageTexture(), gate = deferred();
  destination.generateMipmaps = true; destination.mipmapsAutoUpdate = true;
  const producer = c.enqueue({prepare: gate.promise, issue() { sourceBytes(c, target).fill(88); }});
  const region = box(0, 0, 1, 1), position = at(2, 2);
  c.renderer.copyTextureToTexture(target.texture, destination, region, position);
  region.max.x = 99; position.x = 99; destination.mipmapsAutoUpdate = false;
  gate.resolve(); await producer; await c.renderer.waitForGPU();
  assert.deepEqual(copies(c)[0].slice(1).map((value, index) => index === 2 ? value : value.origin), [[0, 0, 0], [2, 2, 0], [1, 1, 1]]);
  assert.equal(c.device.calls.filter(call => call[0] === 'mip').length, 2); c.renderer.dispose();
});

for (const endpoint of ['target', 'storage']) test(`${endpoint} disposal cancels the queued generation and surfaces through waitForGPU`, async () => {
  const c = setup(), target = renderTarget(), destination = storageTexture(), gate = deferred();
  const blocking = c.enqueue({prepare: gate.promise, issue() {}});
  c.renderer.copyTextureToTexture(target.texture, destination);
  (endpoint === 'target' ? target : destination).dispose(); gate.resolve(); await blocking;
  await assert.rejects(c.renderer.waitForGPU(), /STALE/);
  assert.throws(() => c.renderer.copyTextureToTexture(target.texture, destination), /STALE/);
  assert.equal(copies(c).length, 0); c.renderer.dispose();
});

for (const budget of ['maxStorageBytes', 'maxTargetBytes']) test(`${budget} preflight allocates neither endpoint on refusal`, () => {
  const c = setup({[budget]: 1}), target = renderTarget(), destination = storageTexture();
  assert.throws(() => c.renderer.copyTextureToTexture(target.texture, destination), /BUDGET/);
  assert.equal(c.device.textures.length, 0); c.renderer.dispose();
});

test('both endpoint dimensions and destination array layers are checked before source allocation', () => {
  for (const invalid of ['source', 'destination', 'layers']) {
    const c = setup(), target = renderTarget(invalid === 'source' ? 8 : 4, 4), destination = storageTexture(invalid === 'destination' ? 8 : 4, 4, 'rgba8unorm', invalid === 'layers' ? 2 : 1);
    c.device.limits.maxTextureDimension2D = 4; c.device.limits.maxTextureArrayLayers = 1;
    assert.throws(() => c.renderer.copyTextureToTexture(target.texture, destination, box(0, 0, 1, 1)), /LIMIT/);
    assert.equal(c.device.textures.length, 0); c.renderer.dispose();
  }
});

test('format conversion, depth and detached color fail without a copy', () => {
  const c = setup(), target = renderTarget(), destination = storageTexture(4, 4, 'r32float');
  assert.throws(() => c.renderer.copyTextureToTexture(target.texture, destination), /FORMAT/);
  target.texture.isDepthTexture = true;
  assert.throws(() => c.renderer.copyTextureToTexture(target.texture, storageTexture()), /SOURCE/);
  target.texture.isDepthTexture = false;
  const detached = storageTexture(); detached.isStorageTexture = false; detached.isRenderTargetTexture = true; detached.renderTarget = target;
  assert.throws(() => c.renderer.copyTextureToTexture(detached, storageTexture()), /STALE/);
  assert.equal(c.device.textures.length, 0); c.renderer.dispose();
});

test('empty regions are allocation-free and invalid mip levels are rejected synchronously', () => {
  const c = setup(), target = renderTarget(), destination = storageTexture();
  c.renderer.copyTextureToTexture(target.texture, destination, box(0, 0, 0, 4));
  assert.throws(() => c.renderer.copyTextureToTexture(target.texture, destination, null, null, 1), /LEVEL/);
  assert.equal(c.device.textures.length, 0); assert.equal(copies(c).length, 0); c.renderer.dispose();
});

test('in-place target reconfiguration and foreign devices invalidate captured copies', () => {
  for (const change of ['size', 'texture', 'device', 'busy']) {
    const c = setup(), target = renderTarget(), destination = storageTexture();
    const request = captureTargetToStorageCopy(c.renderer, target.texture, destination);
    if (change === 'size') target.width++;
    if (change === 'texture') target.texture = storageTexture();
    if (change === 'device') c.setDevice({});
    if (change === 'busy') c.targets.capture(target).busy = true;
    assert.throws(() => request.issue(), /STALE|DEVICE|REENTRANT/);
    assert.equal(c.device.textures.length, 0); c.renderer.dispose();
  }
});

test('copy validation failure rejects completion and retires destination storage', async () => {
  const c = setup(), target = renderTarget(), destination = storageTexture();
  sourceBytes(c, target); storageBytes(c, destination);
  c.device.validationError('invalid native copy');
  c.renderer.copyTextureToTexture(target.texture, destination);
  await assert.rejects(c.renderer.waitForGPU(), /invalid native copy/);
  assert.equal(c.storage.allocatedBytes, 0); assert.equal(c.targets.allocatedBytes, 64); c.renderer.dispose();
});

test('source disposal during encoding prevents queue submission', () => {
  const c = setup(), target = renderTarget(), destination = storageTexture();
  const create = c.device.createCommandEncoder;
  c.device.createCommandEncoder = descriptor => { const encoder = create(descriptor); target.dispose(); return encoder; };
  assert.throws(() => c.renderer.copyTextureToTexture(target.texture, destination), /STALE/);
  assert.equal(copies(c).length, 0); c.renderer.dispose();
});

test('waitForGPU includes completion, while the initialized copy call never waits for the fence', async () => {
  const completion = deferred(), c = setup({completion}), target = renderTarget(), destination = storageTexture();
  c.renderer.copyTextureToTexture(target.texture, destination);
  let finished = false; const done = c.renderer.waitForGPU().then(() => { finished = true; });
  await Promise.resolve(); await Promise.resolve(); assert.equal(finished, false);
  completion.resolve(); await done; c.renderer.dispose();
});

test('nested compute submit turns do not enqueue copies after later commands', () => {
  const c = setup(), target = renderTarget(), destination = storageTexture();
  sourceBytes(c, target).fill(9); c.renderer._computeTurn = {submit: issue => issue()}; c.renderer._preparing = true;
  c.renderer.copyTextureToTexture(target.texture, destination);
  assert.ok(storageBytes(c, destination).every(n => n === 9)); assert.equal(c.renderer._pending.length, 0); c.renderer.dispose();
});

test('a full FIFO refuses admission without allocation or an inspector success event', () => {
  const c = setup(), target = renderTarget(), destination = storageTexture();
  c.renderer._preparing = true; c.renderer._enqueue = () => { throw new Error('queue full'); };
  assert.throws(() => c.renderer.copyTextureToTexture(target.texture, destination), /queue full/);
  assert.equal(c.device.textures.length, 0); assert.equal(c.device.calls.some(call => call[0] === 'inspect-copy'), false); c.renderer.dispose();
});

test('storage native borrows preflight duplicate bytes, reject reentry and preserve captured writes', async () => {
  const c = setup({maxStorageBytes: 64}), texture = storageTexture(), entry = c.storage.capture(texture);
  c.storage.validateAllocations([entry, entry]); assert.equal(c.device.textures.length, 0);
  c.storage.withTexture(entry, native => {
    assert.equal(native, entry.native);
    assert.throws(() => c.storage.withTexture(entry, () => {}), /REENTRANT/);
  });
  assert.equal(c.storage.allocatedBytes, 64);
  assert.throws(() => c.storage.withTexture(entry, () => {}, {entry: {}, kind: 'storage', access: 'write-only'}), /ACCESS/);
  await c.storage.whenIdle(); c.renderer.dispose();
});
