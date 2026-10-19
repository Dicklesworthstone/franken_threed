import test from 'node:test';
import assert from 'node:assert/strict';
import {createStorageTextureTransfers} from '../../tools/ingest/three_storage_texture_transfer.mjs';
import {deferred} from './storage_texture_fixture.mjs';
import {setup, texture, formats} from './storage_pixel_fixture.mjs';

for (const dimension of ['2d', '2d-array', '3d']) for (const format of formats) {
  test(`readback ${format[2]} ${dimension} returns exact packed bytes without retaining mapped storage`, async () => {
    const s = setup(), t = texture(dimension, 3, 2, 2, format);
    const length = 3 * 2 * (dimension === '2d' ? 1 : 2) * format[3];
    const input = Uint8Array.from({length}, (_, i) => (i * 181 + 255) & 255);
    await s.renderer.writeStorageTexturePixelsAsync(t, input);
    const output = await s.renderer.readStorageTexturePixelsAsync(t);
    assert.ok(output instanceof Uint8Array); assert.deepEqual(output, input);
    assert.equal(s.device.textures.length, 1); assert.equal(s.device.buffers.length, 1);
    assert.equal(s.device.buffers[0].destroyed, 1); assert.equal(s.device.buffers[0].data.byteLength, 0);
    assert.equal(output.byteLength, input.byteLength); assert.equal(t.image.data, undefined); s.renderer.dispose();
  });
}

test('read/write/read preserves each FIFO state while GPU mapping is delayed', async () => {
  const mapping = deferred(), s = setup({mapping}), t = texture('2d', 1, 1);
  await s.renderer.writeStorageTexturePixelsAsync(t, new Uint32Array([0xffffffff]));
  s.renderer._drain = true;
  const options = {width: 1}, first = s.renderer.readStorageTexturePixelsAsync(t, options);
  const write = s.renderer.writeStorageTexturePixelsAsync(t, new Uint32Array([16777217]));
  const second = s.renderer.readStorageTexturePixelsAsync(t, options);
  options.width = 300;
  s.renderer.flush(); await write;
  assert.deepEqual(s.events, ['write', 'read', 'map', 'write', 'read', 'map']);
  assert.equal(new Uint32Array(s.bytes(t).buffer)[0], 16777217);
  mapping.resolve();
  assert.equal(new Uint32Array((await first).buffer)[0], 0xffffffff);
  assert.equal(new Uint32Array((await second).buffer)[0], 16777217);
  assert.ok(s.device.buffers.every(b => b.destroyed === 1)); s.renderer.dispose();
});

test('regional volume readback strips per-row/layer padding and flips each slice independently', async () => {
  const s = setup(), t = texture('3d', 4, 4, 4);
  const input = Uint32Array.from({length: 64}, (_, i) => 0xff000000 + i);
  await s.renderer.writeStorageTexturePixelsAsync(t, input);
  const bytes = await s.renderer.readStorageTexturePixelsAsync(t, {x: 1, y: 1, z: 1, width: 2, height: 2, depth: 2, flipY: true});
  const expected = [];
  for (let z = 1; z <= 2; z++) for (let y = 2; y >= 1; y--) for (let x = 1; x <= 2; x++) expected.push(input[(z * 4 + y) * 4 + x]);
  assert.deepEqual([...new Uint32Array(bytes.buffer)], expected);
  assert.equal(bytes.length, 32); assert.equal(s.device.buffers[0].size, 3 * 256 + 8); s.renderer.dispose();
});

for (const dim of ['2d-array', '3d']) test(`read selected ${dim} mip without mixing depth extents`, async () => {
  const s = setup(), t = texture(dim, 8, 4, 4); t.generateMipmaps = true;
  const data = new Uint8Array(4 * 2 * (dim === '3d' ? 2 : 4) * 4).fill(93);
  await s.renderer.writeStorageTexturePixelsAsync(t, data, {mipLevel: 1});
  assert.deepEqual(await s.renderer.readStorageTexturePixelsAsync(t, {mipLevel: 1}), data);
  assert.ok((await s.renderer.readStorageTexturePixelsAsync(t)).every(n => n === 0));
  assert.equal(s.device.textures.length, 1); s.renderer.dispose();
});

test('fresh storage readback initializes zero native texels without an upload', async () => {
  const s = setup(), output = await s.renderer.readStorageTexturePixelsAsync(texture());
  assert.deepEqual(output, new Uint8Array(24)); assert.equal(s.writes.length, 0); s.renderer.dispose();
});

test('readback budget includes native row padding and packed output before allocation', async () => {
  const s = setup(), t = texture('2d', 1, 2);
  const transfers = createStorageTextureTransfers(s.renderer, {maxReadbackBytes: 267});
  await assert.rejects(transfers.read(t), /BUDGET/); // native 260 + output 8 = 268
  assert.equal(s.device.textures.length, 0); assert.equal(s.device.buffers.length, 0);
  assert.equal(transfers.diagnostics.readbackBytes, 0); transfers.dispose(); s.renderer.dispose();
});

test('aggregate readback reservation is held through mapping and released on completion', async () => {
  const mapping = deferred(), s = setup({mapping}), t = texture('2d', 1, 2);
  const transfers = createStorageTextureTransfers(s.renderer, {maxReadbackBytes: 268});
  const result = transfers.read(t);
  assert.equal(transfers.diagnostics.readbackBytes, 268);
  await assert.rejects(transfers.read(t), /BUDGET/);
  mapping.resolve(); await result;
  assert.equal(transfers.diagnostics.readbackBytes, 0); await transfers.read(t); transfers.dispose(); s.renderer.dispose();
});

test('readback device buffer limits account for padded rows, not logical byte count', async () => {
  const s = setup(); s.device.limits.maxBufferSize = 259;
  await assert.rejects(s.renderer.readStorageTexturePixelsAsync(texture('2d', 1, 2)), /LIMIT/);
  assert.equal(s.device.buffers.length, 0); assert.equal(s.device.textures.length, 0); s.renderer.dispose();
});

test('reads and uploads share the incomplete-transfer bound', async () => {
  const mapping = deferred(), s = setup({mapping}), t = texture();
  const transfers = createStorageTextureTransfers(s.renderer, {maxPendingTransfers: 1});
  const read = transfers.read(t);
  await assert.rejects(transfers.write(t, new Uint8Array(24)), /BUDGET/);
  mapping.resolve(); await read; await transfers.write(t, new Uint8Array(24)); transfers.dispose(); s.renderer.dispose();
});

test('rejected FIFO admission frees staging reservations without allocating a buffer', async () => {
  const s = setup({queueLimit: 0}), transfers = createStorageTextureTransfers(s.renderer);
  s.renderer._drain = true;
  await assert.rejects(transfers.read(texture()), /FIFO full/);
  assert.equal(transfers.diagnostics.readbackBytes, 0); assert.equal(s.device.buffers.length, 0); transfers.dispose(); s.renderer.dispose();
});

for (const mutation of ['resize', 'dispose']) test(`source ${mutation} before copy rejects stale queued readback`, async () => {
  const s = setup(), t = texture(); s.renderer._drain = true;
  const result = s.renderer.readStorageTexturePixelsAsync(t);
  if (mutation === 'resize') t.image.width++; else t.dispose();
  const rejected = assert.rejects(result, /STALE/); s.renderer.flush(); await rejected;
  assert.equal(s.device.buffers.length, 0); assert.equal(s.device.textures.length, 0); s.renderer.dispose();
});

test('disposing source after copy submission does not revoke an independent readback snapshot', async () => {
  const mapping = deferred(), s = setup({mapping}), t = texture();
  const data = new Uint8Array(24).fill(11);
  await s.renderer.writeStorageTexturePixelsAsync(t, data);
  const result = s.renderer.readStorageTexturePixelsAsync(t); t.dispose();
  mapping.resolve(); assert.deepEqual(await result, data); assert.equal(s.device.buffers[0].destroyed, 1); s.renderer.dispose();
});

for (const when of ['queued', 'mapping']) for (const reason of ['abort', 'dispose', 'loss']) {
  test(`${reason} during ${when} cancels readback and releases native/CPU reservations`, async () => {
    const mapping = deferred(), s = setup({mapping}), transfers = createStorageTextureTransfers(s.renderer);
    s.renderer._drain = when === 'queued';
    const result = transfers.read(texture()), rejected = assert.rejects(result, /cancel|DISPOSED|DEVICE_LOST|disposed/);
    if (reason === 'abort') s.renderer._targetLifetime.abort(new Error('cancel readback'));
    else if (reason === 'dispose') transfers.dispose();
    else s.device.lose({message: 'device removed'});
    await rejected;
    assert.equal(transfers.diagnostics.readbackBytes, 0);
    assert.ok(s.device.buffers.every(b => b.destroyed === 1));
    s.renderer.flush(); mapping.resolve(); await Promise.resolve(); transfers.dispose(); s.renderer.dispose();
  });
}

for (const options of [{mapError: 'mapping failed'}, {rangeError: 'mapped range failed'}, {onRead: d => d.validationError('invalid texture copy')}]) {
  test(`readback propagates ${Object.keys(options)[0]} and destroys staging exactly once`, async () => {
    const s = setup(options), transfers = createStorageTextureTransfers(s.renderer);
    await assert.rejects(transfers.read(texture()), /failed|invalid texture copy/);
    assert.equal(transfers.diagnostics.readbackBytes, 0); assert.equal(s.device.buffers[0].destroyed, 1);
    await assert.rejects(transfers.whenIdle(), /failed|invalid texture copy/); transfers.dispose(); s.renderer.dispose();
  });
}

test('readback refuses upload layouts and invalid regions without allocating', async () => {
  const s = setup(), t = texture();
  for (const options of [{offset: 0}, {bytesPerRow: 12}, {rowsPerImage: 2}, {z: 1}, {height: 0}, {mipLevel: 30}, {flipY: 'true'}, null])
    await assert.rejects(s.renderer.readStorageTexturePixelsAsync(t, options));
  assert.equal(s.device.buffers.length, 0); assert.equal(s.device.textures.length, 0); s.renderer.dispose();
});

test('waitForGPU waits for pending readback mapping and packed result extraction', async () => {
  const mapping = deferred(), s = setup({mapping});
  const read = s.renderer.readStorageTexturePixelsAsync(texture());
  let done = false; const idle = s.renderer.waitForGPU().then(() => { done = true; });
  await Promise.resolve(); assert.equal(done, false);
  mapping.resolve(); await Promise.all([read, idle]); assert.equal(done, true); s.renderer.dispose();
});


test('readback uses the current owner borrow guard and releases rejected reservations', async () => {
  const s = setup(), t = texture(), transfers = createStorageTextureTransfers(s.renderer);
  const entry = s.owner.capture(t); let nested;
  s.owner.withTexture(entry, () => { nested = transfers.read(t); nested.catch(() => {}); });
  await assert.rejects(nested, /REENTRANT/);
  assert.equal(transfers.diagnostics.readbackBytes, 0); assert.equal(s.device.buffers.length, 0);
  await transfers.read(t); transfers.dispose(); s.renderer.dispose();
});


test('post-submit source invalidation handles a later mapping rejection and destroys its buffer', async () => {
  const mapping = deferred(); let t;
  const s = setup({mapping, onRead() { t.dispose(); }}); t = texture();
  const transfers = createStorageTextureTransfers(s.renderer);
  await assert.rejects(transfers.read(t), /STALE/);
  assert.equal(transfers.diagnostics.readbackBytes, 0); assert.equal(s.device.buffers[0].destroyed, 1);
  mapping.resolve(); await new Promise(resolve => setImmediate(resolve));
  transfers.dispose(); s.renderer.dispose();
});
