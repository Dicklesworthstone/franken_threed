import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';

function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return {promise, resolve, reject};
}
function gpu(options = {}) {
  const calls = [], buffers = [], lost = deferred();
  let scopeError = null;
  const device = {
    calls, buffers, lost: lost.promise, lose: lost.resolve,
    limits: {maxComputeWorkgroupsPerDimension: 65535},
    pushErrorScope: () => calls.push(['push']),
    popErrorScope: () => { const error = scopeError; scopeError = null; calls.push(['pop']); return Promise.resolve(error); },
    validationError: message => { scopeError = {message}; },
    createBuffer({size, usage}) {
      const buffer = {size, usage, data: new Uint8Array(size), destroyed: 0, unmapped: 0,
        destroy() { this.destroyed++; }, unmap() { this.unmapped++; },
        mapAsync: () => options.mapping?.promise ?? Promise.resolve(),
        getMappedRange() { return this.data.buffer; }};
      buffers.push(buffer); return buffer;
    },
    createShaderModule: ({code}) => ({code, getCompilationInfo: async () => ({messages: options.messages ?? []})}),
    createComputePipeline: descriptor => { calls.push(['pipeline', descriptor]); return {getBindGroupLayout: index => index}; },
    createComputePipelineAsync: async descriptor => { calls.push(['pipeline', descriptor]); return {getBindGroupLayout: index => index}; },
    createBindGroup: descriptor => descriptor,
    createCommandEncoder() {
      const operations = [];
      return {
        beginComputePass() { calls.push(['begin']); return {
          setPipeline: pipeline => calls.push(['setPipeline', pipeline]),
          setBindGroup: (index, group) => calls.push(['group', index, group]),
          dispatchWorkgroups: (...counts) => operations.push(() => calls.push(['dispatch', ...counts])),
          dispatchWorkgroupsIndirect: (buffer, offset) => operations.push(() => calls.push(['indirect', buffer, offset])),
          end: () => calls.push(['end'])}; },
        copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
          operations.push(() => { target.data.set(source.data.subarray(sourceOffset, sourceOffset + size), targetOffset); calls.push(['copy']); });
        },
        finish: () => operations,
      };
    },
    queue: {
      writeBuffer(buffer, offset, array, start = 0, size = array.byteLength - start) {
        buffer.data.set(new Uint8Array(array.buffer, array.byteOffset + start, size), offset);
        calls.push(['write', offset, size]);
      },
      submit(commands) { calls.push(['submit']); for (const command of commands) for (const operation of command) operation(); },
      onSubmittedWorkDone: () => options.completion?.promise ?? Promise.resolve(),
    },
  };
  return device;
}
const shader = {code: '@compute @workgroup_size(1) fn main() {}'};

test('dispatch owns a real compute encoder, pipeline, bind groups and submit', async () => {
  const device = gpu(), owner = createGpuCompute(device), key = {};
  const buffer = owner.upload(key, new Uint32Array([1, 2, 3]));
  const pipeline = await owner.prepare(shader);
  owner.dispatch(pipeline, [{index: 0, entries: [{binding: 0, resource: {buffer}}]}], [8, 2]);
  assert.deepEqual(device.calls.filter(c => c[0] === 'dispatch'), [['dispatch', 8, 2, 1]]);
  assert.equal(device.calls.filter(c => c[0] === 'submit').length, 1);
  assert.equal(device.calls.find(c => c[0] === 'group')[2].entries[0].resource.buffer, buffer);
  await owner.whenIdle(); owner.dispose();
});

test('unchanged CPU versions never overwrite GPU-produced storage', () => {
  const device = gpu(), owner = createGpuCompute(device), key = {}, array = new Float32Array([1, 2]);
  const buffer = owner.upload(key, array, 0);
  new Float32Array(buffer.data.buffer)[0] = 100;
  assert.equal(owner.upload(key, array, 0), buffer);
  assert.equal(new Float32Array(buffer.data.buffer)[0], 100);
  array[1] = 9;
  owner.upload(key, array, 1, {ranges: [{start: 1, count: 1}]});
  assert.deepEqual([...new Float32Array(buffer.data.buffer)], [100, 9]);
  assert.equal(device.calls.filter(c => c[0] === 'write').length, 2);
  owner.dispose();
});

test('typed-array subviews and padded trailing bytes are uploaded exactly', async () => {
  const device = gpu(), owner = createGpuCompute(device), key = {};
  const array = new Uint8Array([99, 1, 2, 3, 99]).subarray(1, 4);
  const buffer = owner.upload(key, array);
  assert.deepEqual([...buffer.data], [1, 2, 3, 0]);
  assert.deepEqual([...new Uint8Array(await owner.read(key))], [1, 2, 3]);
  assert.equal(owner.allocatedBytes, 4); owner.dispose();
});

test('readback snapshots are submitted before later work and mappings may finish out of order', async () => {
  const mapping = deferred(), device = gpu({mapping}), owner = createGpuCompute(device), key = {};
  const array = new Uint32Array([7]), buffer = owner.upload(key, array);
  const first = owner.read(key);
  new Uint32Array(buffer.data.buffer)[0] = 11;
  const second = owner.read(key);
  assert.equal(device.calls.filter(c => c[0] === 'copy').length, 2);
  mapping.resolve();
  assert.deepEqual([...new Uint32Array(await first)], [7]);
  assert.deepEqual([...new Uint32Array(await second)], [11]);
  assert.equal(owner.allocatedBytes, 4);
  assert.equal(device.buffers[1].unmapped, 1);
  assert.equal(device.buffers[2].destroyed, 1); owner.dispose();
});

test('indirect dispatch uses GPU-resident counts, not the stale CPU array', async () => {
  const device = gpu(), owner = createGpuCompute(device), key = {};
  const buffer = owner.upload(key, new Uint32Array([9, 1, 1, 1]));
  const pipeline = await owner.prepare(shader);
  owner.dispatch(pipeline, [], {buffer, offset: 4});
  assert.deepEqual(device.calls.filter(c => c[0] === 'indirect'), [['indirect', buffer, 4]]);
  assert.throws(() => owner.dispatch(pipeline, [], {buffer, offset: 8}), /DISPATCH/);
  assert.throws(() => owner.dispatch(pipeline, [], {buffer: {}, offset: 0}), /DISPATCH/);
  owner.dispose();
});

test('invalid direct dispatches fail before command encoding', async () => {
  const device = gpu(), owner = createGpuCompute(device), pipeline = await owner.prepare(shader);
  for (const counts of [[], [-1], [1.5], [65536], [1, 1, 1, 1], [NaN]])
    assert.throws(() => owner.dispatch(pipeline, [], counts), /DISPATCH/);
  assert.equal(device.calls.some(c => c[0] === 'begin'), false);
  owner.dispatch(pipeline, [], [0]); owner.dispose();
});

test('range admission is atomic and preserves GPU-owned values', () => {
  const device = gpu(), owner = createGpuCompute(device), key = {}, array = new Uint32Array([1, 2]);
  const buffer = owner.upload(key, array);
  const before = device.calls.length;
  assert.throws(() => owner.upload(key, array, 1, {ranges: [{start: 0, count: 1}, {start: 2, count: 1}]}), /RANGE/);
  assert.equal(device.calls.length, before);
  assert.deepEqual([...new Uint32Array(buffer.data.buffer)], [1, 2]); owner.dispose();
});

test('resource and device limits reject allocations and readback over budget', async () => {
  const device = gpu(), owner = createGpuCompute(device, {maxBytes: 16, maxBuffers: 1}), key = {};
  owner.upload(key, new Uint32Array(3));
  assert.throws(() => owner.upload({}, new Uint32Array(1)), /BUDGET/);
  await assert.rejects(owner.read(key), /BUDGET/);
  assert.equal(owner.allocatedBytes, 12);
  device.limits.maxStorageBufferBindingSize = 4;
  assert.throws(() => owner.upload(key, new Uint32Array(2), 1), /LIMIT/);
  owner.release(key); assert.equal(owner.allocatedBytes, 0); owner.dispose();
});

test('storage replacement retires the old generation without leaking its charge', () => {
  const device = gpu(), owner = createGpuCompute(device), key = {};
  const old = owner.upload(key, new Uint32Array([1]));
  const replacement = owner.upload(key, new Uint32Array([2, 3]), 1);
  assert.notEqual(old, replacement); assert.equal(old.destroyed, 1);
  assert.equal(owner.bufferCount, 1); assert.equal(owner.allocatedBytes, 8);
  owner.dispose(); owner.dispose(); assert.equal(replacement.destroyed, 1); assert.equal(owner.allocatedBytes, 0);
});

test('pipeline preparation is cached and shader failures never dispatch', async () => {
  const device = gpu(), owner = createGpuCompute(device, {maxPipelines: 1});
  const a = owner.prepare(shader), b = owner.prepare(shader);
  assert.equal(a, b); assert.equal(await a, await b);
  assert.equal(device.calls.filter(c => c[0] === 'pipeline').length, 1);
  assert.throws(() => owner.prepare({code: 'different'}), /BUDGET/); owner.dispose();
  const bad = createGpuCompute(gpu({messages: [{type: 'error', lineNum: 2, linePos: 3, message: 'bad WGSL'}]}));
  await assert.rejects(bad.prepare(shader), /2:3 bad WGSL/); bad.dispose();
});

test('device error scopes close synchronously before asynchronous compilation yields', async () => {
  const device = gpu(), owner = createGpuCompute(device);
  const run = owner.prepare(shader);
  assert.deepEqual(device.calls.slice(0, 2), [['push'], ['pop']]);
  await run; await owner.whenIdle(); owner.dispose();
});

test('disposal rejects a stuck mapping and frees staging exactly once', async () => {
  const mapping = deferred(), device = gpu({mapping}), owner = createGpuCompute(device), key = {};
  owner.upload(key, new Uint32Array([1]));
  const read = owner.read(key);
  owner.dispose(); await assert.rejects(read, /DISPOSED/);
  assert.equal(owner.allocatedBytes, 0);
  for (const buffer of device.buffers) assert.equal(buffer.destroyed, 1);
  mapping.resolve();
});

test('aborting a borrowed device rejects pending completion without destroying the device', async () => {
  const completion = deferred(), controller = new AbortController(), device = gpu({completion});
  let destroyed = false; device.destroy = () => { destroyed = true; };
  const owner = createGpuCompute(device, {signal: controller.signal});
  const idle = owner.whenIdle(); controller.abort(new Error('cancelled'));
  await assert.rejects(idle, /cancelled/); assert.equal(destroyed, false);
});

test('device loss and validation errors become terminal rather than reported success', async () => {
  const device = gpu(), owner = createGpuCompute(device);
  owner.upload({}, new Uint32Array([1]));
  device.lose({message: 'adapter removed'}); await Promise.resolve();
  assert.throws(() => owner.upload({}, new Uint32Array([1])), /DEVICE_LOST/);
  assert.equal(owner.allocatedBytes, 0);
  const invalid = gpu(), other = createGpuCompute(invalid);
  invalid.validationError('invalid binding'); other.upload({}, new Uint32Array([1]));
  await assert.rejects(other.whenIdle(), /invalid binding/);
  assert.equal(other.allocatedBytes, 0);
});


test('synchronous source compute preparation submits without yielding and reuses warm pipelines', async () => {
  const device = gpu(), owner = createGpuCompute(device);
  const pipeline = owner.prepareSync(shader);
  owner.dispatch(pipeline, [], [1]);
  assert.equal(device.calls.filter(c => c[0] === 'dispatch').length, 1);
  assert.equal(await owner.prepare(shader), pipeline);
  assert.equal(owner.prepareSync(shader), pipeline);
  assert.equal(device.calls.filter(c => c[0] === 'pipeline').length, 1);
  owner.dispose();
  const otherDevice = gpu(), other = createGpuCompute(otherDevice);
  const prewarmed = await other.prepare(shader);
  assert.equal(other.prepareSync(shader), prewarmed);
  assert.equal(otherDevice.calls.filter(c => c[0] === 'pipeline').length, 1);
  other.dispose();
});
