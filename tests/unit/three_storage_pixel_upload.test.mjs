import test from 'node:test';
import assert from 'node:assert/strict';
import {createStorageTextureTransfers, inspectStoragePixelRegion} from '../../tools/ingest/three_storage_texture_transfer.mjs';
import {deferred} from './storage_texture_fixture.mjs';
import {setup, texture, formats, THREE} from './storage_pixel_fixture.mjs';

for (const dimension of ['2d', '2d-array', '3d']) for (const format of formats) {
  test(`upload ${format[2]} ${dimension} preserves every byte in shared native storage`, async () => {
    const s = setup(), t = texture(dimension, 3, 2, 2, format);
    const length = 3 * 2 * (dimension === '2d' ? 1 : 2) * format[3];
    const input = Uint8Array.from({length}, (_, i) => (i * 151 + 255) & 255), image = t.image, source = t.source;
    await s.renderer.writeStorageTexturePixelsAsync(t, input);
    assert.deepEqual(s.bytes(t), input); assert.equal(s.device.textures.length, 1);
    assert.equal(s.owner.bindingsFor().get(t).view.texture, s.device.textures[0]);
    assert.equal(t.image, image); assert.equal(t.source, source); assert.equal(t.version, 0); assert.equal(t.image.data, undefined);
    assert.equal(s.device.buffers.length, 0); assert.equal(s.writes.length, 1); s.renderer.dispose();
  });
}

test('view-relative offsets, padded rows/images and flipY address only selected volume texels', async () => {
  const s = setup(), t = texture('3d', 4, 4, 4);
  const backing = new Uint8Array(200).fill(199), view = new DataView(backing.buffer, 8, 144);
  const input = new Uint8Array(backing.buffer, 8, 144);
  for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) input.fill(10 + z * 2 + y, 4 + z * 48 + y * 16, 12 + z * 48 + y * 16);
  const before = backing.slice();
  await s.renderer.writeStorageTexturePixelsAsync(t, view, {x: 1, y: 1, z: 1, width: 2, height: 2, depth: 2,
    offset: 4, bytesPerRow: 16, rowsPerImage: 3, flipY: true});
  const expected = new Uint8Array(4 * 4 * 4 * 4);
  for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) {
    const at = (((z + 1) * 4 + y + 1) * 4 + 1) * 4;
    expected.fill(10 + z * 2 + 1 - y, at, at + 8);
  }
  assert.deepEqual(s.bytes(t), expected); assert.deepEqual(backing, before);
  assert.equal(s.writes[0].layout.bytesPerRow, 8); assert.equal(s.writes[0].bytes.length, 32); s.renderer.dispose();
});

test('queued uploads capture bytes/options before caller mutation and interleave with native copies', async () => {
  const s = setup(), a = texture('2d', 1, 1), b = texture('2d', 1, 1);
  s.renderer._drain = true;
  const data = new Uint32Array([0xffffffff]), options = {width: 1};
  const first = s.renderer.writeStorageTexturePixelsAsync(a, data, options);
  s.renderer.copyTextureToTexture(a, b);
  data[0] = 16777217;
  const second = s.renderer.writeStorageTexturePixelsAsync(a, data, options);
  options.width = 999; structuredClone(data.buffer, {transfer: [data.buffer]});
  assert.equal(s.device.textures.length, 0); s.renderer.flush(); await Promise.all([first, second]);
  assert.deepEqual(s.events, ['write', 'copy', 'write']);
  assert.equal(new Uint32Array(s.bytes(a).buffer)[0], 16777217);
  assert.equal(new Uint32Array(s.bytes(b).buffer)[0], 0xffffffff); s.renderer.dispose();
});

for (const dim of ['2d-array', '3d']) test(`manual ${dim} mips use correct depth and preserve base level`, async () => {
  const s = setup(), t = texture(dim, 8, 4, 4); t.generateMipmaps = true;
  const depth = dim === '3d' ? 2 : 4, data = new Uint8Array(4 * 2 * depth * 4).fill(17);
  await s.renderer.writeStorageTexturePixelsAsync(t, data, {mipLevel: 1});
  assert.deepEqual(s.bytes(t, 1), data); assert.ok(s.bytes(t).every(n => n === 0));
  assert.equal(s.device.calls.some(c => ['mip', 'dispatch'].includes(c[0])), false); s.renderer.dispose();
});

test('automatic mip policy is captured before a queued base upload', async () => {
  const s = setup(), t = texture('2d', 4, 4); t.generateMipmaps = true; t.mipmapsAutoUpdate = true;
  s.renderer._preparing = true;
  const write = s.renderer.writeStorageTexturePixelsAsync(t, new Uint8Array(64)); t.mipmapsAutoUpdate = false;
  s.renderer.flush(); await write;
  assert.equal(s.device.calls.filter(c => c[0] === 'mip').length, 2); s.renderer.dispose();
});

test('integer base upload rejects implicit mip interpolation before allocation', async () => {
  const s = setup(), t = texture('2d', 4, 4, 1, formats.at(-1)); t.generateMipmaps = true; t.mipmapsAutoUpdate = true;
  await assert.rejects(s.renderer.writeStorageTexturePixelsAsync(t, new Uint8Array(256)), /MIPS/);
  assert.equal(s.device.textures.length, 0); s.renderer.dispose();
});

test('invalid regions, layouts, sources and arithmetic overflow produce no GPU work', async () => {
  const s = setup(), t = texture();
  for (const options of [{x: -1}, {width: 0}, {depth: 2}, {mipLevel: 1}, {width: 2.5}, {flipY: 1},
    {bytesPerRow: 8}, {bytesPerRow: 13}, {rowsPerImage: 1}, {offset: -1}, {rowsPerImage: Number.MAX_SAFE_INTEGER + 1},
    {offset: Number.MAX_SAFE_INTEGER}, {mip: 1}, null, []])
    await assert.rejects(s.renderer.writeStorageTexturePixelsAsync(t, new Uint8Array(128), options));
  for (const data of [new Uint8Array(23), [], new SharedArrayBuffer(24), new ArrayBuffer(24, {maxByteLength: 48})])
    await assert.rejects(s.renderer.writeStorageTexturePixelsAsync(t, data));
  const detached = new ArrayBuffer(24); structuredClone(detached, {transfer: [detached]});
  await assert.rejects(s.renderer.writeStorageTexturePixelsAsync(t, detached), /DATA/);
  assert.equal(s.device.textures.length, 0); assert.equal(s.writes.length, 0); s.renderer.dispose();
});

test('aggregate snapshot budget rejects excess requests without disturbing accepted captures', async () => {
  const s = setup(), t = texture('2d', 1, 1), transfers = createStorageTextureTransfers(s.renderer, {maxSnapshotBytes: 4});
  s.renderer._drain = true;
  const first = transfers.write(t, new Uint8Array(4));
  await assert.rejects(transfers.write(t, new Uint8Array(4)), /BUDGET/);
  assert.equal(transfers.diagnostics.snapshotBytes, 4);
  s.renderer.flush(); await first;
  assert.equal(transfers.diagnostics.snapshotBytes, 0);
  await transfers.write(t, new Uint8Array(4)); transfers.dispose(); s.renderer.dispose();
});

test('GPU completion is included and pending count bounds issued transfers', async () => {
  const completion = deferred(), s = setup({completion}), t = texture('2d', 1, 1);
  const transfers = createStorageTextureTransfers(s.renderer, {maxPendingTransfers: 1});
  const first = transfers.write(t, new Uint8Array(4)); let settled = false; first.then(() => { settled = true; });
  await Promise.resolve(); assert.equal(settled, false); assert.equal(s.writes.length, 1);
  assert.equal(transfers.diagnostics.snapshotBytes, 0);
  await assert.rejects(transfers.write(t, new Uint8Array(4)), /BUDGET/);
  completion.resolve(); await first; await transfers.whenIdle(); transfers.dispose(); s.renderer.dispose();
});

test('a full FIFO releases rejected source snapshots', async () => {
  const s = setup({queueLimit: 0}), transfers = createStorageTextureTransfers(s.renderer);
  s.renderer._drain = true;
  await assert.rejects(transfers.write(texture(), new Uint8Array(24)), /FIFO full/);
  assert.equal(transfers.diagnostics.snapshotBytes, 0); assert.equal(s.device.textures.length, 0); transfers.dispose(); s.renderer.dispose();
});

for (const mutation of ['resize', 'dispose']) test(`${mutation} before submission rejects stale uploads`, async () => {
  const s = setup(), t = texture(); s.renderer._drain = true;
  const result = s.renderer.writeStorageTexturePixelsAsync(t, new Uint8Array(24));
  if (mutation === 'resize') t.image.width++; else t.dispose();
  const rejected = assert.rejects(result, /STALE/); s.renderer.flush(); await rejected;
  assert.equal(s.writes.length, 0); assert.equal(s.device.textures.length, 0); s.renderer.dispose();
});

test('abort and device loss release captured bytes without waiting for the FIFO', async () => {
  for (const loss of [false, true]) {
    const s = setup(), transfers = createStorageTextureTransfers(s.renderer); s.renderer._drain = true;
    const result = transfers.write(texture(), new Uint8Array(24));
    const rejected = assert.rejects(result, /cancelled|DEVICE_LOST/);
    if (loss) s.device.lose({message: 'lost'}); else s.renderer._targetLifetime.abort(new Error('cancelled'));
    await rejected; assert.equal(transfers.diagnostics.snapshotBytes, 0);
    s.renderer.flush(); assert.equal(s.writes.length, 0); transfers.dispose(); s.renderer.dispose();
  }
});

test('write validation failures reject both public completion and waitForGPU', async () => {
  const s = setup({onWrite: d => d.validationError('invalid write')}), t = texture();
  await assert.rejects(s.renderer.writeStorageTexturePixelsAsync(t, new Uint8Array(24)), /invalid write/);
  await assert.rejects(s.renderer.waitForGPU(), /invalid write/); s.renderer.dispose();
});

test('allocation budget and device limits reject before an upload', async () => {
  for (const options of [{storage: {maxBytes: 1}}, {}]) {
    const s = setup(options), t = texture();
    if (!options.storage) s.device.limits.maxTextureDimension2D = 1;
    await assert.rejects(s.renderer.writeStorageTexturePixelsAsync(t, new Uint8Array(24)), /BUDGET|LIMIT/);
    assert.equal(s.writes.length, 0); assert.equal(s.device.textures.length, 0); s.renderer.dispose();
  }
});

test('uninitialized, exact-backend and disposed renderers never accept uploads', async () => {
  for (const patch of [{_initialized: false}, {isWebGPURenderer: false}, {_disposed: true}]) {
    const s = setup(); Object.assign(s.renderer, patch);
    await assert.rejects(s.renderer.writeStorageTexturePixelsAsync(texture(), new Uint8Array(24)), /INITIALIZATION|ROUTE|DISPOSED/);
    assert.equal(s.device.textures.length, 0); s.renderer.dispose();
  }
});

test('region metadata is immutable and volume versus array mip depth is explicit', () => {
  const shape = {width: 8, height: 4, depth: 8, mipLevelCount: 4, texelBytes: 4, viewDimension: '3d'};
  assert.deepEqual(inspectStoragePixelRegion(shape, {mipLevel: 2}).size, [2, 1, 2]);
  const array = inspectStoragePixelRegion({...shape, viewDimension: '2d-array'}, {mipLevel: 2});
  assert.deepEqual(array.size, [2, 1, 8]); assert.throws(() => { array.origin[0] = 1; }, TypeError);
});


test('upload borrows native storage through the owner and rejects same-resource reentry', async () => {
  let nested, transfers, t;
  const s = setup({onWrite() { nested = transfers.write(t, new Uint8Array(24)); nested.catch(() => {}); }});
  transfers = createStorageTextureTransfers(s.renderer); t = texture();
  await transfers.write(t, new Uint8Array(24));
  await assert.rejects(nested, /REENTRANT/);
  assert.equal(s.writes.length, 1); assert.equal(transfers.diagnostics.snapshotBytes, 0);
  transfers.dispose(); s.renderer.dispose();
});
