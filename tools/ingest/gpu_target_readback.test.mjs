import assert from 'node:assert/strict';
import test from 'node:test';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

function fixture(format = 'rgba8unorm') {
  const f = gpuPostprocessFixture(), buffers = [], copies = [];
  let mapping = null, mapError = null, unmaps = 0;
  const componentBytes = format === 'rgba16float' ? 2 : 1;
  f.device.createBuffer = descriptor => {
    const bytes = new Uint8Array(descriptor.size);
    const buffer = {bytes, descriptor, destroyed: 0, mapAsync() { return mapError ? Promise.reject(mapError) : mapping?.promise ?? Promise.resolve(); },
      getMappedRange() { return bytes.buffer; }, unmap() { unmaps++; }, destroy() { this.destroyed++; }};
    buffers.push(buffer); return buffer;
  };
  f.device.createCommandEncoder = () => ({
    copyTextureToBuffer(source, destination, size) {
      copies.push({source, destination, size});
      const rowBytes = size.width * 4 * componentBytes;
      for (let row = 0; row < size.height; row++) for (let i = 0; i < rowBytes; i++)
        destination.buffer.bytes[row * destination.bytesPerRow + i] = (source.origin.y + row) * 20 + source.origin.x * 4 * componentBytes + i;
    },
    finish() { return {}; },
  });
  const session = {device: f.device, width: 8, height: 6, diagnostics: {target: {format}},
    withTexture(callback) { callback({format}); return true; }};
  return {...f, session, buffers, copies, get unmaps() { return unmaps; },
    set mapping(value) { mapping = value; }, set mapError(value) { mapError = value; }};
}
const region = {x: 1, y: 1, width: 3, height: 2};

for (const format of ['rgba8unorm', 'rgba8unorm-srgb', 'rgba16float'])
  test(`${format}: padded GPU rows become tightly packed typed output`, async () => {
    const f = fixture(format), result = await readGpuTargetPixels(f.session, region);
    const bpc = format === 'rgba16float' ? 2 : 1;
    assert.ok(result instanceof (bpc === 2 ? Uint16Array : Uint8Array));
    assert.equal(result.length, 24);
    const bytes = new Uint8Array(result.buffer), rowBytes = 12 * bpc;
    for (let row = 0; row < 2; row++) for (let i = 0; i < rowBytes; i++)
      assert.equal(bytes[row * rowBytes + i], 20 * (row + 1) + 4 * bpc + i);
    assert.equal(f.copies[0].destination.bytesPerRow, 256);
    assert.equal(f.buffers[0].descriptor.usage, 9);
    assert.equal(f.buffers[0].destroyed, 1); assert.equal(f.unmaps, 1);
    assert.equal(f.calls.scopes.length, 0);
  });

test('copy submission precedes the asynchronous map completion', async () => {
  const f = fixture(), gate = deferred(); f.mapping = gate;
  const pending = readGpuTargetPixels(f.session, region);
  assert.equal(f.calls.submissions.length, 1);
  assert.equal(f.buffers[0].destroyed, 0);
  gate.resolve(); await pending;
  assert.equal(f.buffers[0].destroyed, 1);
});

test('WebGL origin and output row order are both converted, retaining an output subarray tail', async () => {
  const f = fixture(), backing = new Uint8Array(40).fill(211), output = backing.subarray(4, 36);
  const result = await readGpuTargetPixels(f.session, {...region, output, flipY: true});
  assert.equal(result, output);
  assert.deepEqual(f.copies[0].source.origin, {x: 1, y: 3, z: 0});
  assert.equal(output[0], 84); assert.equal(output[12], 64);
  assert.deepEqual([...backing.subarray(0, 4)], [211, 211, 211, 211]);
  assert.ok(backing.subarray(28).every(value => value === 211));
});

test('zero extent, invalid bounds, wrong output and excessive budgets never allocate', async () => {
  const f = fixture();
  assert.equal((await readGpuTargetPixels(f.session, {...region, width: 0})).length, 0);
  for (const edit of [{x: -1}, {width: 20}, {y: 0.5}, {maxBytes: 255}, {output: new Uint16Array(24)}, {output: new Uint8Array(2)}])
    await assert.rejects(readGpuTargetPixels(f.session, {...region, ...edit}), /GPU_READBACK_/);
  assert.equal(f.buffers.length, 0); assert.equal(f.calls.submissions.length, 0);
});

for (const kind of ['validation', 'mapping', 'loss', 'abort']) test(`${kind} failure leaves caller output unchanged and destroys staging`, async () => {
  const f = fixture(), output = new Uint8Array(24).fill(211), controller = new AbortController();
  if (kind === 'validation') f.controls.scopeError = {message: 'copy rejected'};
  if (kind === 'mapping') f.mapError = Error('mapping rejected');
  if (kind === 'loss' || kind === 'abort') f.mapping = deferred();
  const pending = readGpuTargetPixels(f.session, {...region, output, signal: controller.signal});
  if (kind === 'loss') f.lost.resolve({message: 'lost'});
  if (kind === 'abort') controller.abort(Error('cancelled'));
  await assert.rejects(pending);
  assert.ok(output.every(value => value === 211));
  assert.equal(f.buffers[0].destroyed, 1); assert.equal(f.unmaps, 1);
});

test('pre-aborted reads do not allocate staging', async () => {
  const f = fixture(), controller = new AbortController(); controller.abort(Error('cancelled'));
  await assert.rejects(readGpuTargetPixels(f.session, {...region, signal: controller.signal}), /cancelled/);
  assert.equal(f.buffers.length, 0);
});

test('repeated reads install only one device-loss promise handler', async () => {
  const f = fixture(); let listeners = 0;
  f.device.lost = {then() { listeners++; }};
  for (let i = 0; i < 5; i++) await readGpuTargetPixels(f.session, region);
  assert.equal(listeners, 1);
});
