import assert from 'node:assert/strict';
import test from 'node:test';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';

// Tests only the native-call boundary and byte movement, not WGSL execution.
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const layouts = new Map([
  ['rgba8unorm', [Uint8Array, 4]], ['rgba8unorm-srgb', [Uint8Array, 4]],
  ['bgra8unorm', [Uint8Array, 4]], ['bgra8unorm-srgb', [Uint8Array, 4]],
  ['r16float', [Uint16Array, 1]], ['rg16float', [Uint16Array, 2]],
  ['rgba16float', [Uint16Array, 4]], ['rgba32float', [Float32Array, 4]],
  ['r32float', [Float32Array, 1]], ['rg32float', [Float32Array, 2]],
]);
function fixture(format = 'rgba8unorm') {
  const lost = deferred(), buffers = [], copies = [], submissions = [], scopes = [];
  const controls = {mapping: null, mapError: null, scopeError: null, allocationError: null,
    unmapError: null, shortMap: false, fill: null};
  const [Type, components] = layouts.get(format) ?? [Uint8Array, 4];
  const device = {
    limits: {maxBufferSize: 1024 * 1024}, lost: lost.promise,
    pushErrorScope(scope) { scopes.push(scope); },
    popErrorScope() {
      assert.ok(scopes.pop());
      const error = controls.scopeError; controls.scopeError = null;
      return Promise.resolve(error);
    },
    createBuffer(descriptor) {
      if (controls.allocationError) throw controls.allocationError;
      const buffer = {descriptor, bytes: new Uint8Array(descriptor.size), destroyed: 0, unmapped: 0,
        mapAsync() { return controls.mapError ? Promise.reject(controls.mapError) : controls.mapping?.promise ?? Promise.resolve(); },
        getMappedRange() { return controls.shortMap ? new ArrayBuffer(0) : this.bytes.buffer; },
        unmap() { this.unmapped++; if (controls.unmapError) throw controls.unmapError; },
        destroy() { this.destroyed++; }};
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() { return {
      copyTextureToBuffer(source, destination, size) {
        copies.push({source, destination, size});
        for (let row = 0; row < size.height; row++) {
          const packed = new Type(size.width * components);
          for (let col = 0; col < size.width; col++) for (let c = 0; c < components; c++)
            packed[col * components + c] = controls.fill
              ? controls.fill(source.origin.x + col, source.origin.y + row, c)
              : (source.origin.y + row) * 20 + (source.origin.x + col) * components + c;
          destination.buffer.bytes.set(new Uint8Array(packed.buffer), row * destination.bytesPerRow);
        }
      },
      finish() { return {}; },
    }; },
    queue: {submit(commands) { submissions.push(commands); }},
  };
  const session = {device, width: 8, height: 6, diagnostics: {target: {format}},
    withTexture(callback) { callback({format}); return true; }};
  return {session, device, controls, lost, buffers, copies, submissions, scopes};
}
const region = {x: 1, y: 1, width: 3, height: 2};
const pixel = {x: 0, y: 0, width: 1, height: 1};

for (const [format, [Type, components]] of layouts) {
  test(`${format}: typed, tightly packed rows preserve channel order and copy origin`, async () => {
    const f = fixture(format), result = await readGpuTargetPixels(f.session, region);
    assert.ok(result instanceof Type);
    assert.equal(result.length, region.width * region.height * components);
    for (let row = 0; row < 2; row++) for (let col = 0; col < 3; col++) for (let c = 0; c < components; c++) {
      const sourceChannel = format.startsWith('bgra') ? [2, 1, 0, 3][c] : c;
      assert.equal(result[(row * 3 + col) * components + c], (row + 1) * 20 + (col + 1) * components + sourceChannel);
    }
    assert.deepEqual(f.copies[0].source.origin, {x: 1, y: 1, z: 0});
    assert.equal(f.copies[0].destination.bytesPerRow, 256);
    assert.equal(f.buffers[0].descriptor.size, 512);
    assert.equal(f.buffers[0].descriptor.usage, 9);
    assert.equal(f.buffers[0].unmapped, 1); assert.equal(f.buffers[0].destroyed, 1);
    assert.equal(f.scopes.length, 0);
  });
}

test('float32 values are never quantized to half floats, including signed zero', async () => {
  const f = fixture('rgba32float'), values = [70000.125, -70000.5, Math.fround(1e-20), -0];
  f.controls.fill = (_x, _y, c) => values[c];
  const result = await readGpuTargetPixels(f.session, pixel);
  values.forEach((value, index) => assert.ok(Object.is(result[index], value)));
});

for (const [format, components] of [['r16float', 1], ['rg16float', 2], ['rgba16float', 4]])
  test(`${format}: half-float bit patterns, including NaN encodings, are unchanged`, async () => {
    const f = fixture(format), bits = [0x7c01, 0xfc00, 0x8000, 0x0001];
    f.controls.fill = (x, _y, c) => bits[x * components + c];
    assert.deepEqual([...await readGpuTargetPixels(f.session, {...pixel, width: 4 / components})], bits);
  });

test('BGRA swizzle composes with bottom-left origin, reversed rows and an output subarray', async () => {
  const f = fixture('bgra8unorm'), backing = new Uint8Array(40).fill(211), output = backing.subarray(4, 36);
  assert.equal(await readGpuTargetPixels(f.session, {...region, output, flipY: true}), output);
  assert.deepEqual(f.copies[0].source.origin, {x: 1, y: 3, z: 0});
  assert.deepEqual([...output.subarray(0, 4)], [86, 85, 84, 87]);
  assert.deepEqual([...output.subarray(12, 16)], [66, 65, 64, 67]);
  assert.ok(backing.subarray(0, 4).every(value => value === 211));
  assert.ok(backing.subarray(28).every(value => value === 211));
});

test('zero extent and invalid input do not allocate or submit', async () => {
  const f = fixture();
  assert.equal((await readGpuTargetPixels(f.session, {...region, width: 0})).length, 0);
  for (const edit of [{x: -1}, {width: 20}, {y: 0.5}, {maxBytes: 255}, {maxInFlightBytes: 0},
    {output: new Uint16Array(24)}, {output: new Uint8Array(2)}])
    await assert.rejects(readGpuTargetPixels(f.session, {...region, ...edit}), /GPU_READBACK_/);
  f.session.width = NaN;
  await assert.rejects(readGpuTargetPixels(f.session, region), /GPU_READBACK_SOURCE/);
  assert.equal(f.buffers.length, 0); assert.equal(f.submissions.length, 0);
});

test('device buffer limit is checked against padded bytes, not packed pixels', async () => {
  const f = fixture(); f.device.limits.maxBufferSize = 511;
  await assert.rejects(readGpuTargetPixels(f.session, region), /GPU_READBACK_BUDGET/);
  assert.equal(f.buffers.length, 0);
});

test('concurrent reads share a budget across sessions and cannot relax an active limit', async () => {
  const f = fixture(), gate = deferred(); f.controls.mapping = gate;
  const first = readGpuTargetPixels(f.session, {...pixel, maxInFlightBytes: 500});
  assert.equal(f.submissions.length, 1, 'copy submits before mapping finishes');
  // Each read owns 256 staging bytes plus four output bytes.
  await assert.rejects(readGpuTargetPixels({...f.session}, {...pixel, maxInFlightBytes: 1000}), /GPU_READBACK_BUDGET/);
  assert.equal(f.buffers.length, 1, 'budget rejection must precede native allocation');
  gate.resolve(); await first;
  await readGpuTargetPixels(f.session, {...pixel, maxInFlightBytes: 260});
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
});

test('a supplied output is not charged as a second owned allocation', async () => {
  const f = fixture(), gate = deferred(); f.controls.mapping = gate;
  const reads = [0, 1].map(() => readGpuTargetPixels(f.session,
    {...pixel, output: new Uint8Array(4), maxInFlightBytes: 512}));
  assert.equal(f.buffers.length, 2);
  await assert.rejects(readGpuTargetPixels(f.session, {...pixel, maxInFlightBytes: 512}), /GPU_READBACK_BUDGET/);
  gate.resolve(); await Promise.all(reads);
});

test('in-flight budgets are per device, not global', async () => {
  const a = fixture(), b = fixture(), gate = deferred(); a.controls.mapping = gate;
  const pending = readGpuTargetPixels(a.session, {...pixel, maxInFlightBytes: 260});
  await readGpuTargetPixels(b.session, {...pixel, maxInFlightBytes: 260});
  gate.resolve(); await pending;
});

test('the request count is bounded even for tiny copies with a large byte budget', async () => {
  const f = fixture(), gate = deferred(); f.controls.mapping = gate;
  const pending = Array.from({length: 64}, () => readGpuTargetPixels(f.session, pixel));
  await assert.rejects(readGpuTargetPixels(f.session, pixel), /GPU_READBACK_BUDGET/);
  assert.equal(f.buffers.length, 64);
  gate.resolve(); await Promise.all(pending);
  await readGpuTargetPixels(f.session, pixel);
});

for (const kind of ['validation', 'mapping', 'loss', 'abort', 'shortMap']) {
  test(`${kind}: caller output stays unchanged and staging is released`, async () => {
    const f = fixture(), output = new Uint8Array(24).fill(211), controller = new AbortController();
    if (kind === 'validation') f.controls.scopeError = {message: 'copy rejected'};
    if (kind === 'mapping') f.controls.mapError = Error('map rejected');
    if (kind === 'shortMap') f.controls.shortMap = true;
    if (kind === 'loss' || kind === 'abort') f.controls.mapping = deferred();
    const pending = readGpuTargetPixels(f.session, {...region, output, signal: controller.signal, maxInFlightBytes: 512});
    if (kind === 'loss') f.lost.resolve({message: 'lost'});
    if (kind === 'abort') controller.abort(Error('cancelled'));
    await assert.rejects(pending);
    assert.ok(output.every(value => value === 211));
    assert.equal(f.buffers[0].destroyed, 1); assert.equal(f.buffers[0].unmapped, 1);
    if (kind !== 'loss') {
      f.controls.mapping = f.controls.mapError = null; f.controls.shortMap = false;
      await readGpuTargetPixels(f.session, {...region, output, maxInFlightBytes: 512});
    }
  });
}

test('allocation and cleanup exceptions release reservations', async () => {
  const f = fixture(); f.controls.allocationError = Error('allocation failed');
  await assert.rejects(readGpuTargetPixels(f.session, {...pixel, maxInFlightBytes: 260}), /allocation failed/);
  f.controls.allocationError = null; f.controls.unmapError = Error('unmap failed');
  await assert.rejects(readGpuTargetPixels(f.session, {...pixel, maxInFlightBytes: 260}), /unmap failed/);
  assert.equal(f.buffers[0].destroyed, 1);
  f.controls.unmapError = null;
  await readGpuTargetPixels(f.session, {...pixel, maxInFlightBytes: 260});
});

test('pre-aborted and already-lost requests do not allocate', async () => {
  const f = fixture(), controller = new AbortController(); controller.abort(Error('cancelled'));
  await assert.rejects(readGpuTargetPixels(f.session, {...pixel, signal: controller.signal}), /cancelled/);
  assert.equal(f.buffers.length, 0);
  await readGpuTargetPixels(f.session, pixel);
  f.lost.resolve({message: 'device gone'}); await Promise.resolve();
  await assert.rejects(readGpuTargetPixels(f.session, pixel), /GPU_READBACK_DEVICE_LOST/);
  assert.equal(f.buffers.length, 1);
});

test('same-turn cancellation cannot commit a successful map into caller output', async () => {
  const f = fixture(), output = new Uint8Array(4).fill(211), controller = new AbortController();
  const pending = readGpuTargetPixels(f.session, {...pixel, output, signal: controller.signal});
  queueMicrotask(() => controller.abort(Error('cancelled')));
  await assert.rejects(pending, /cancelled/);
  assert.ok(output.every(value => value === 211));
});

test('repeated reads subscribe to device loss once', async () => {
  const f = fixture(); let count = 0;
  f.device.lost = {then() { count++; }};
  for (let i = 0; i < 5; i++) await readGpuTargetPixels(f.session, pixel);
  assert.equal(count, 1);
});

test('unknown formats reject rather than guessing a component layout', async () => {
  const f = fixture('depth24plus');
  await assert.rejects(readGpuTargetPixels(f.session, pixel), /GPU_READBACK_FORMAT/);
  assert.equal(f.buffers.length, 0);
});
