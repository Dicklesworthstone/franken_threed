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
      const mapped = [];
      const detach = () => { for (const array of mapped.splice(0)) structuredClone(array, {transfer: [array]}); };
      const buffer = {size, usage, data: new Uint8Array(size), destroyed: 0, unmapped: 0,
        destroy() { this.destroyed++; detach(); }, unmap() { this.unmapped++; detach(); },
        mapAsync: () => options.mapping?.promise ?? Promise.resolve(),
        getMappedRange(offset = 0, length = size - offset) {
          const array = this.data.buffer.slice(offset, offset + length); mapped.push(array); return array;
        }};
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
          operations.push(() => { target.data.set(source.data.subarray(sourceOffset, sourceOffset + size), targetOffset);
            calls.push(['copy', sourceOffset, size, target]); });
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
function readbackTarget(maxByteLength) {
  const listeners = new Map();
  return {isReadbackBuffer: true, maxByteLength, buffer: null, _mapped: false,
    addEventListener(type, listener) { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type).add(listener); },
    removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    release() { for (const listener of [...listeners.get('release') ?? []]) listener(); },
    dispose() { for (const listener of [...listeners.get('dispose') ?? []]) listener(); },
    get listenerCount() { return [...listeners.values()].reduce((sum, group) => sum + group.size, 0); }};
}

function delayedScopes() {
  const device = gpu(), scopes = [];
  device.popErrorScope = () => {
    device.calls.push(['pop']);
    const scope = deferred(); scopes.push(scope); return scope.promise;
  };
  return {device, scopes};
}
const nextTurn = () => new Promise(resolve => setImmediate(resolve));

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

test('ranged readback copies only selected bytes and preserves caller-owned ArrayBuffer tails', async t => {
  const device = gpu(), owner = createGpuCompute(device), key = {};
  t.after(() => owner.dispose());
  owner.upload(key, new Uint32Array([10, 20, 30, 40, 50, 60]));
  const target = new Uint32Array([99, 99, 99, 99]).buffer;
  assert.equal(await owner.read(key, target, 4, 8), target);
  assert.deepEqual([...new Uint32Array(target)], [20, 30, 99, 99]);
  assert.deepEqual([...new Uint32Array(await owner.read(key, null, 16))], [50, 60]);
  assert.equal((await owner.read(key, null, 24, 0)).byteLength, 0);
  assert.deepEqual(device.calls.filter(c => c[0] === 'copy').map(c => c.slice(1, 3)), [[4, 8], [16, 8], [24, 0]]);
  assert.deepEqual(device.buffers.slice(1).map(buffer => buffer.size), [8, 8, 0]);
  assert.equal(owner.allocatedBytes, 24);
});

test('invalid readback ranges and destinations fail without allocating or copying', async t => {
  const device = gpu(), owner = createGpuCompute(device), key = {};
  t.after(() => owner.dispose()); owner.upload(key, new Uint32Array([1, 2, 3, 4]));
  const detached = new ArrayBuffer(16); structuredClone(detached, {transfer: [detached]});
  for (const [target, offset, count] of [[null, -4, 4], [null, 2, 4], [null, 0, 2], [null, 0, -2],
    [null, 0, NaN], [null, 0, Infinity], [null, 0, 1.5], [null, 20, -1], [null, 12, 8],
    [new ArrayBuffer(4), 0, 8], [new ArrayBuffer(4), 4, -1], [new Uint32Array(4), 0, 4], [detached, 0, 4],
    [readbackTarget(4), 0, 8]]) await assert.rejects(owner.read(key, target, offset, count), /READBACK_(RANGE|TARGET)/);
  assert.equal(device.buffers.length, 1); assert.equal(device.calls.some(c => c[0] === 'copy'), false);
});

test('source readback handles retain one staging buffer until release or disposal', async t => {
  const changes = [], device = gpu(), owner = createGpuCompute(device, {onReadbackChange: (...change) => changes.push(change)}), key = {};
  t.after(() => owner.dispose());
  const storage = owner.upload(key, new Uint32Array([10, 20, 30, 40, 50, 60])), target = readbackTarget(32);
  const first = owner.read(key, target, 4, 8);
  assert.equal(target._mapped, true); assert.equal(target.buffer, null);
  await assert.rejects(owner.read(key, target, 0, 4), /READBACK_MAPPED/);
  assert.equal(await first, target); assert.deepEqual([...new Uint32Array(target.buffer)], [20, 30]);
  const mapped = target.buffer, native = device.buffers[1];
  target.release(); assert.equal(mapped.byteLength, 0); assert.equal(target.buffer, null); assert.equal(target._mapped, false);
  target.maxByteLength = 4; // Source retains its first native capacity until disposal.
  new Uint32Array(storage.data.buffer)[4] = 123;
  assert.equal(await owner.read(key, target, 16, 8), target);
  assert.deepEqual([...new Uint32Array(target.buffer)], [123, 60]);
  assert.equal(device.buffers.length, 2); assert.equal(native.destroyed, 0);
  assert.equal(owner.readbackBufferCount, 1); assert.equal(owner.readbackBytes, 32); assert.equal(owner.allocatedBytes, 56);
  target.dispose(); assert.equal(native.destroyed, 1); assert.equal(target._mapped, false); assert.equal(target.buffer, null);
  assert.equal(target.listenerCount, 0); assert.equal(owner.allocatedBytes, 24);
  assert.deepEqual(changes, [[1, 32], [0, 0]]);
  await owner.read(key, target, 0, 4); assert.equal(device.buffers.length, 3); assert.equal(device.buffers[2].size, 4);
});

for (const action of ['release', 'dispose'])
test(`${action} cancels a pending handle map without invalidating its next use`, async t => {
  const mapping = deferred(), device = gpu({mapping}), owner = createGpuCompute(device), key = {}, target = readbackTarget(16);
  t.after(() => { mapping.resolve(); owner.dispose(); });
  const storage = owner.upload(key, new Uint32Array([1, 2, 3, 4]));
  const first = owner.read(key, target, 0, 8), rejected = assert.rejects(first, /READBACK_(RELEASED|DISPOSED)/);
  target[action]();
  new Uint32Array(storage.data.buffer)[2] = 77;
  const second = owner.read(key, target, 8, 4);
  await rejected; assert.equal(target._mapped, true);
  mapping.resolve(); await second;
  assert.deepEqual([...new Uint32Array(target.buffer)], [77]);
  assert.equal(target._mapped, true); assert.equal(device.buffers.length, action === 'release' ? 2 : 3);
});

test('mapped handle disposal and device loss release retained allocation and source state once', async () => {
  const device = gpu(), owner = createGpuCompute(device), key = {}, target = readbackTarget(16);
  owner.upload(key, new Uint32Array([1, 2])); await owner.read(key, target);
  const mapped = target.buffer;
  device.lose({message: 'readback device lost'}); await Promise.resolve();
  assert.equal(mapped.byteLength, 0); assert.equal(target.buffer, null); assert.equal(target._mapped, false);
  assert.equal(target.listenerCount, 0); assert.equal(owner.readbackBufferCount, 0); assert.equal(owner.allocatedBytes, 0);
  target.release(); target.dispose(); owner.dispose();
  for (const buffer of device.buffers) assert.equal(buffer.destroyed, 1);
});

test('reusable readbacks enforce device and shared budgets while allowing small ranges of large buffers', async t => {
  const device = gpu(), owner = createGpuCompute(device, {maxBytes: 40}), key = {};
  t.after(() => owner.dispose()); owner.upload(key, new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8]));
  await assert.rejects(owner.read(key), /BUDGET/);
  const target = readbackTarget(8);
  await owner.read(key, target, 24, 8); assert.deepEqual([...new Uint32Array(target.buffer)], [7, 8]);
  target.release();
  await assert.rejects(owner.read(key, readbackTarget(4), 0, 4), /BUDGET/);
  target.dispose(); device.limits.maxBufferSize = 32;
  const oversized = readbackTarget(36);
  await assert.rejects(owner.read(key, oversized, 0, 4), /LIMIT/); assert.equal(oversized._mapped, false);
  assert.equal(device.buffers.length, 2); assert.equal(owner.allocatedBytes, 32);
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

for (const [index, operation] of ['shader', 'layout', 'pipeline'].entries()) {
  test(`asynchronous preparation waits for ${operation} validation before publishing success`, async t => {
    const {device, scopes} = delayedScopes(), owner = createGpuCompute(device);
    t.after(() => { for (const scope of scopes) scope.resolve(null); owner.dispose(); });
    let settled = false;
    const preparation = owner.prepare(shader);
    preparation.then(() => { settled = true; }, () => { settled = true; });
    await nextTurn();
    assert.equal(scopes.length, 3);
    for (const [i, scope] of scopes.entries()) if (i !== index) scope.resolve(null);
    await nextTurn();
    assert.equal(settled, false, 'native pipeline completion alone is not validation');
    scopes[index].resolve({message: `${operation} is invalid`});
    await assert.rejects(preparation, new RegExp(`GPU_COMPUTE_VALIDATION: ${operation} is invalid`));
    assert.throws(() => owner.prepareSync(shader), /GPU_COMPUTE_VALIDATION/);
    assert.equal(device.calls.some(c => c[0] === 'submit'), false);
  });
}

test('synchronous fallback compilation also awaits validation in asynchronous preparation', async t => {
  const {device, scopes} = delayedScopes(), owner = createGpuCompute(device);
  device.createComputePipelineAsync = undefined;
  t.after(() => { for (const scope of scopes) scope.resolve(null); owner.dispose(); });
  let settled = false;
  const preparation = owner.prepare(shader).then(value => { settled = true; return value; });
  await nextTurn();
  assert.equal(settled, false);
  for (const scope of scopes) scope.resolve(null);
  const pipeline = await preparation;
  assert.equal(owner.prepareSync(shader), pipeline);
  owner.dispatch(pipeline, [], [1]);
  assert.equal(device.calls.filter(c => c[0] === 'dispatch').length, 1);
});

test('synchronous preparation stays immediate but its async cache waits for validation', async t => {
  const {device, scopes} = delayedScopes(), owner = createGpuCompute(device);
  t.after(() => { for (const scope of scopes) scope.resolve(null); owner.dispose(); });
  const pipeline = owner.prepareSync(shader);
  owner.dispatch(pipeline, [], [1]);
  assert.equal(device.calls.filter(c => c[0] === 'dispatch').length, 1);
  let settled = false;
  const preparation = owner.prepare(shader);
  preparation.then(() => { settled = true; }, () => { settled = true; });
  await nextTurn();
  assert.equal(settled, false);
  scopes[0].resolve({message: 'synchronous pipeline is invalid'});
  await assert.rejects(preparation, /GPU_COMPUTE_VALIDATION: synchronous pipeline is invalid/);
});

test('independent preparation never waits for another shader or a GPU queue fence', async t => {
  const {device, scopes} = delayedScopes(), owner = createGpuCompute(device);
  device.queue.onSubmittedWorkDone = () => assert.fail('compilation must not fence GPU submissions');
  t.after(() => { for (const scope of scopes) scope.resolve(null); owner.dispose(); });
  const first = owner.prepare(shader);
  await nextTurn();
  let secondSettled = false;
  const second = owner.prepare({code: '@compute @workgroup_size(2) fn main() {}'});
  second.then(() => { secondSettled = true; }, () => { secondSettled = true; });
  await nextTurn();
  assert.equal(scopes.length, 6);
  scopes.slice(0, 3).forEach(scope => scope.resolve(null));
  assert.ok(await first);
  assert.equal(secondSettled, false);
  scopes.slice(3).forEach(scope => scope.resolve(null));
  assert.notEqual(await second, await first);
});

test('a synchronous request overtaking async preparation leaves one consistent warm cache', async t => {
  const {device, scopes} = delayedScopes(), owner = createGpuCompute(device);
  t.after(() => { for (const scope of scopes) scope.resolve(null); owner.dispose(); });
  const first = owner.prepare(shader);
  await nextTurn();
  const synchronous = owner.prepareSync(shader), second = owner.prepare(shader);
  assert.equal(scopes.length, 4);
  scopes.forEach(scope => scope.resolve(null));
  assert.equal(await first, synchronous);
  assert.equal(await second, synchronous);
  assert.equal(owner.prepareSync(shader), synchronous);
  assert.equal(await owner.prepare(shader), synchronous);
});

test('disposal interrupts pipeline validation without waiting for the device callback', async () => {
  const {device, scopes} = delayedScopes(), owner = createGpuCompute(device);
  const preparation = owner.prepare(shader);
  await nextTurn();
  owner.dispose();
  await assert.rejects(preparation, /DISPOSED/);
  scopes.forEach(scope => scope.resolve(null));
  await nextTurn();
  assert.throws(() => owner.prepare(shader), /DISPOSED/);
});

test('resource validation snapshots capture issued scopes without a queue fence', async t => {
  const {device, scopes} = delayedScopes(), owner = createGpuCompute(device);
  device.queue.onSubmittedWorkDone = () => assert.fail('validation must not fence GPU submissions');
  t.after(() => { for (const scope of scopes) scope.resolve(null); owner.dispose(); });
  owner.upload({}, new Uint32Array([1]));
  let settled = false;
  const validation = owner.validationSnapshot().then(() => { settled = true; });
  owner.upload({}, new Uint32Array([2]));
  await nextTurn();
  assert.equal(settled, false);
  scopes.slice(0, 2).forEach(scope => scope.resolve(null));
  await validation;
  assert.equal(settled, true, 'later issued scopes are not part of this snapshot');
});
