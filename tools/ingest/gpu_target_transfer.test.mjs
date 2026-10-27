import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {inspectGpuTargetWrite, gpuTargetWriteData, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {transferDevice, deferred} from './fixtures/target_transfer_fixture.mjs';
const cases = [
  ['rgba8unorm', Uint8Array, 4], ['rgba8unorm-srgb', Uint8Array, 4],
  ['bgra8unorm', Uint8Array, 4], ['bgra8unorm-srgb', Uint8Array, 4],
  ['r16float', Uint16Array, 1], ['rg16float', Uint16Array, 2],
  ['rgba16float', Uint16Array, 4], ['r32float', Float32Array, 1],
  ['rg32float', Float32Array, 2], ['rgba32float', Float32Array, 4],
];
const make = (f, options = {}) => createGpuRenderTarget(f.device, {
  width: 5, height: 4, depthFormat: null, format: 'rgba8unorm', copyDestination: true, ...options});
const raw = data => new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
function payload(Type, elements) {
  const out = new Type(elements), bytes = raw(out);
  // Include bit patterns that cannot be preserved by a float/number round-trip.
  if (Type === Float32Array) {
    const bits = new Uint32Array(out.buffer), patterns = [0x80000000, 0x7fc00042, 0x3f800001, 0x7f800000, 0x00000001];
    for (let i = 0; i < bits.length; i++) bits[i] = patterns[i % patterns.length];
  } else for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 73 + 11) % 256;
  return out;
}

for (const [format, Type, components] of cases) for (const flipY of [false, true])
  test(`${format}/flip=${flipY}: subregion uploads round-trip bytes, with offsets, strides and untouched neighbors`, async () => {
    const f = transferDevice(), t = make(f, {format});
    const rowElements = 2 * components, strideElements = 3 * components;
    const input = payload(Type, 3 + strideElements + rowElements + 7), view = input.subarray(2, input.length - 2);
    const original = raw(input).slice();
    assert.equal(t.writePixels(view, {x: 1, y: 1, width: 2, height: 2, offset: Type.BYTES_PER_ELEMENT,
      bytesPerRow: strideElements * Type.BYTES_PER_ELEMENT, flipY}), true);
    assert.equal(f.calls.uploads.length, 1); assert.equal(t.diagnostics.colorUploads, 1);
    assert.equal(f.calls.uploads[0].destination.origin.y, flipY ? 1 : 1);
    assert.deepEqual(raw(input), original, 'caller bytes are never changed');
    const expected = new Type(2 * rowElements);
    raw(expected).set(raw(input).subarray(3 * Type.BYTES_PER_ELEMENT, (3 + rowElements) * Type.BYTES_PER_ELEMENT));
    raw(expected).set(raw(input).subarray((3 + strideElements) * Type.BYTES_PER_ELEMENT,
      (3 + strideElements + rowElements) * Type.BYTES_PER_ELEMENT), rowElements * Type.BYTES_PER_ELEMENT);
    input.fill(0); // writeTexture snapshots the upload before returning.
    const read = await t.readPixels({x: 1, y: 1, width: 2, height: 2, flipY});
    assert.ok(read instanceof Type); assert.deepEqual(raw(read), raw(expected));
    assert.ok(raw(await t.readPixels({x: 0, y: 0, width: 1, height: 4})).every(n => n === 0));
    assert.equal(f.calls.uploads[0].layout.offset, 0);
    assert.ok(f.calls.uploads[0].data.byteLength <= (strideElements + rowElements) * Type.BYTES_PER_ELEMENT);
    await t.whenIdle(); t.dispose(); assert.ok(f.calls.textures.every(v => v.destroyed === 1));
  });

for (const [format, Type, components] of cases) test(`${format}: GPU copies preserve bytes and do not allocate CPU staging`, async () => {
  const f = transferDevice(), source = make(f, {format}), destination = make(f, {format, height: 7});
  const data = payload(Type, 5 * 4 * components); source.writePixels(data);
  const textures = f.calls.textures.length;
  assert.equal(destination.copyFrom(source, {sourceX: 1, sourceY: 0, x: 2, y: 1, width: 2, height: 3, flipY: true}), true);
  assert.equal(f.calls.buffers.length, 0); assert.equal(f.calls.textures.length, textures);
  const copy = f.calls.colorCopies[0]; assert.equal(copy.source.origin.y, 1); assert.equal(copy.destination.origin.y, 3);
  assert.equal(destination.diagnostics.colorCopies, 1);
  const a = await source.readPixels({x: 1, y: 0, width: 2, height: 3, flipY: true});
  const b = await destination.readPixels({x: 2, y: 1, width: 2, height: 3, flipY: true});
  assert.deepEqual(raw(a), raw(b));
  await destination.whenIdle(); source.dispose(); destination.dispose();
});

test('copy-compatible sRGB pairs copy raw bytes rather than color-converting', async () => {
  const f = transferDevice(), source = make(f), destination = make(f, {format: 'rgba8unorm-srgb'});
  const data = payload(Uint8Array, 80); source.writePixels(data); destination.copyFrom(source);
  assert.deepEqual(await destination.readPixels(), data); source.dispose(); destination.dispose();
});

test('uploads and copies dirty destination mipmaps exactly once per written version', async () => {
  const f = transferDevice(), a = make(f, {generateMipmaps: true}), b = make(f, {generateMipmaps: true});
  a.withTexture(() => {}); b.withTexture(() => {});
  a.writePixels(new Uint8Array(80)); assert.equal(a.diagnostics.mipGenerations, 1);
  b.copyFrom(a); assert.equal(a.diagnostics.mipGenerations, 2); assert.equal(b.diagnostics.mipGenerations, 1);
  b.withTexture(() => {}); b.withTexture(() => {}); assert.equal(b.diagnostics.mipGenerations, 2);
  b.copyFrom(a); assert.equal(a.diagnostics.mipGenerations, 2);
  b.withTexture(() => {}); assert.equal(b.diagnostics.mipGenerations, 3);
  await b.whenIdle(); a.dispose(); b.dispose();
});

test('a selected NPOT source mip is copied into destination level zero', async () => {
  const f = transferDevice(), a = make(f, {width: 7, height: 5, generateMipmaps: true}), b = make(f, {width: 3, height: 2});
  // Passes are not rasterized by this recorder: seed a distinct payload in each
  // mip to assert which subresource the copy command actually selects.
  a.withTexture(texture => texture.levels.forEach((level, i) => level.bytes.fill(41 + i)));
  b.copyFrom(a, {sourceMipLevel: 1});
  assert.equal(f.calls.colorCopies[0].source.mipLevel, 1);
  assert.deepEqual(f.calls.colorCopies[0].size, {width: 3, height: 2, depthOrArrayLayers: 1});
  assert.ok((await b.readPixels()).every(n => n === 42)); a.dispose(); b.dispose();
});

test('MSAA sources copy their resolved single-sample image, never the multisample attachment', async () => {
  const f = transferDevice(), a = make(f, {sampleCount: 4}), b = make(f);
  a.withFrame((frame, color) => { assert.notEqual(frame.colorView.texture, color); color.levels[0].bytes.fill(79); });
  b.copyFrom(a); assert.equal(f.calls.colorCopies[0].source.texture.sampleCount, 1);
  assert.ok((await b.readPixels()).every(n => n === 79)); a.dispose(); b.dispose();
});

test('invalid regions, budgets, data, options and MSAA destinations reject before queue work', () => {
  const f = transferDevice(), t = make(f), data = new Uint8Array(128);
  for (const options of [{x: -1}, {width: 6}, {height: 5}, {offset: -1}, {offset: 100}, {bytesPerRow: 3},
    {bytesPerRow: 2 ** 32}, {maxBytes: 1}, {maxBytes: 0}, {flipY: 1}, {mipLevel: 1}, {x: NaN}, {height: Infinity}])
    assert.throws(() => t.writePixels(data, options), /GPU_TRANSFER_/);
  for (const value of [null, [], new Uint16Array(80), new Uint8Array(1), new DataView(new ArrayBuffer(80)),
    new Uint8Array(new SharedArrayBuffer(80)), new Uint8Array(new ArrayBuffer(80, {maxByteLength: 160}))])
    assert.throws(() => t.writePixels(value), /GPU_TRANSFER_/);
  let called = false;
  assert.throws(() => t.writePixels(data, {get width() { called = true; return 1; }}), /GPU_TRANSFER_OPTIONS/);
  assert.equal(called, false);
  const msaa = make(f, {sampleCount: 4});
  assert.throws(() => msaa.writePixels(data), /GPU_TRANSFER_SAMPLES/);
  assert.throws(() => msaa.copyFrom(t), /GPU_TRANSFER_SAMPLES/);
  assert.equal(f.calls.uploads.length, 0); assert.equal(f.calls.colorCopies.length, 0); assert.equal(t.failed, false);
  t.writePixels(new Uint8Array(80)); t.dispose(); msaa.dispose();
});

test('copy destination capability is opt-in; no-op transfers submit nothing and preserve mip state', () => {
  const f = transferDevice(), disabled = make(f, {copyDestination: false}), t = make(f, {generateMipmaps: true});
  assert.throws(() => disabled.writePixels(new Uint8Array(80)), /GPU_TARGET_USAGE/);
  assert.throws(() => disabled.copyFrom(t), /GPU_TARGET_USAGE/);
  assert.throws(() => make(f, {copyDestination: 1}), /GPU_TARGET_OPTIONS/);
  assert.equal(t.writePixels(new Uint8Array(0), {x: 5, width: 0}), false);
  assert.equal(t.copyFrom(disabled, {width: 0, height: 0}), false);
  assert.equal(f.calls.uploads.length, 0); assert.equal(f.calls.submissions.length, 0);
  assert.equal(t.diagnostics.colorCopies, 0); assert.equal(t.diagnostics.colorUploads, 0);
  t.dispose(); disabled.dispose();
});

test('cross-device, forged, in-place, wrong-format, dead and out-of-bounds copies are explicit refusals', () => {
  const f = transferDevice(), t = make(f), source = make(f), otherDevice = make(transferDevice()), otherFormat = make(f, {format: 'rgba16float'});
  assert.throws(() => t.copyFrom({}), /GPU_TARGET_SOURCE/);
  assert.throws(() => t.copyFrom(t), /GPU_TARGET_FEEDBACK/);
  assert.throws(() => t.copyFrom(otherDevice), /GPU_TARGET_DEVICE/);
  assert.throws(() => t.copyFrom(otherFormat), /GPU_TRANSFER_FORMAT/);
  for (const options of [{width: 6}, {sourceX: -1}, {sourceY: 5}, {sourceMipLevel: 1}, {x: 1},
    {flipY: 0}, {maxBytes: 1}, {destinationMipLevel: 1}, {width: Number.MAX_SAFE_INTEGER}])
    assert.throws(() => t.copyFrom(source, options), /GPU_TRANSFER_/);
  source.dispose(); assert.throws(() => t.copyFrom(source), /GPU_TARGET_DISPOSED/);
  assert.equal(f.calls.submissions.length, 0); t.dispose(); otherDevice.dispose(); otherFormat.dispose();
});

test('plain uploads borrow only the requested bytes; transformed uploads use bounded packed staging', () => {
  const shape = {width: 2, height: 2, format: 'rgba8unorm', sampleCount: 1};
  const data = new Uint8Array(1024), plan = inspectGpuTargetWrite(shape, data, {offset: 4, bytesPerRow: 12});
  const packed = gpuTargetWriteData(plan); assert.equal(packed.data.buffer, data.buffer);
  assert.equal(packed.data.byteOffset, 4); assert.equal(packed.data.byteLength, 20);
  const transformed = gpuTargetWriteData(inspectGpuTargetWrite(shape, data, {offset: 4, bytesPerRow: 12, flipY: true}));
  assert.notEqual(transformed.data.buffer, data.buffer); assert.equal(transformed.data.byteLength, 16);
  assert.equal(transformed.layout.bytesPerRow, 8);
  assert.throws(() => inspectGpuTargetCopy({...shape, mipLevelCount: 1}, shape, {sourceMipLevel: -1}), /GPU_TRANSFER_REGION/);
});

test('uploaded and copied generations retire only after their submission fence', async () => {
  const f = transferDevice(), a = make(f), b = make(f); a.writePixels(new Uint8Array(80)); b.copyFrom(a);
  const old = f.calls.textures.slice(), fence = deferred(); f.controls.completion = fence.promise;
  a.resize(2, 2); b.resize(2, 2); assert.ok(old.every(t => t.destroyed === 0));
  fence.resolve(); await Promise.all([a.whenIdle(), b.whenIdle()]); assert.ok(old.every(t => t.destroyed === 1));
  a.dispose(); b.dispose(); assert.ok(f.calls.textures.every(t => t.destroyed === 1));
});

test('native write and copy critical sections reject reentry without losing valid operations', async () => {
  const f = transferDevice(), a = make(f), b = make(f);
  f.controls.onUpload = () => {
    assert.throws(() => a.resize(1, 1), /GPU_TARGET_REENTRANT/);
    assert.throws(() => a.dispose(), /GPU_TARGET_REENTRANT/);
  };
  a.writePixels(new Uint8Array(80));
  f.controls.onEncodeCopy = () => {
    for (const t of [a, b]) assert.throws(() => t.dispose(), /GPU_TARGET_REENTRANT/);
    assert.throws(() => b.copyFrom(a), /GPU_TARGET_REENTRANT/);
  };
  b.copyFrom(a); await b.whenIdle(); a.dispose(); b.dispose();
});

for (const kind of ['upload', 'copy']) test(`${kind}: native validation failure is terminal and releases destination ownership`, async () => {
  const f = transferDevice(), a = make(f), b = make(f); await Promise.all([a.whenIdle(), b.whenIdle()]);
  f.controls.error = {message: 'transfer rejected'};
  if (kind === 'upload') b.writePixels(new Uint8Array(80)); else b.copyFrom(a);
  await assert.rejects(b.whenIdle(), /transfer rejected/); assert.equal(b.failed, true); assert.equal(b.allocatedBytes, 0);
  a.dispose(); b.dispose(); assert.ok(f.calls.textures.every(t => t.destroyed === 1));
});

test('source upload validation errors propagate through already-submitted dependent copies', async () => {
  const f = transferDevice(), a = make(f), b = make(f); await Promise.all([a.whenIdle(), b.whenIdle()]);
  f.controls.error = {message: 'bad source upload'}; a.writePixels(new Uint8Array(80)); b.copyFrom(a);
  await assert.rejects(b.whenIdle(), /bad source upload/); assert.equal(b.failed, true);
  a.dispose(); b.dispose(); assert.ok(f.calls.textures.every(t => t.destroyed === 1));
});

test('device loss rejects subsequent transfers and releases native storage without owning the device', async () => {
  const f = transferDevice(), a = make(f), b = make(f); a.writePixels(new Uint8Array(80)); b.copyFrom(a);
  f.lose({message: 'lost during transfer'}); await Promise.resolve();
  assert.throws(() => a.writePixels(new Uint8Array(80)), /GPU_TARGET_DEVICE_LOST/);
  await assert.rejects(b.whenIdle(), /GPU_TARGET_DEVICE_LOST/);
  a.dispose(); b.dispose(); assert.ok(f.calls.textures.every(t => t.destroyed === 1)); assert.equal(f.calls.destroyed, 0);
});
