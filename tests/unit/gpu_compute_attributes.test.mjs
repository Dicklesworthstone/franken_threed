import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {computeAttributeSnapshot, captureComputeStorage} from '../../tools/ingest/gpu_compute_attributes.mjs';
import {gpuDevice, attribute, deferred} from './compute_geometry_fixture.mjs';

test('compute publication lends the identical storage buffer with vertex usage', async () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(12), 4);
  const buffer = compute.upload(source, source.array, source.version);
  const view = computeAttributeSnapshot(device, source);
  assert.equal(view.buffer, buffer); assert.equal(view.arrayStride, 16); assert.equal(view.count, 3);
  assert.ok(buffer.usage & 32); assert.ok(buffer.usage & 128);
  assert.equal(device.buffers.length, 1); await view.whenIdle(); compute.dispose();
});

test('borrowing never uploads stale CPU data over GPU results', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array([0, 0, 0, 0]), 4);
  const buffer = compute.upload(source, source.array, 0);
  new Float32Array(buffer.bytes.buffer)[0] = 27; source.array[0] = -99;
  for (let i = 0; i < 5; i++) assert.equal(computeAttributeSnapshot(device, source).buffer, buffer);
  assert.equal(new Float32Array(buffer.bytes.buffer)[0], 27);
  assert.equal(device.events.filter(e => e[0] === 'upload').length, 1); compute.dispose();
});

test('pending needsUpdate is an explicit compute boundary, not a second geometry uploader', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(8), 4);
  compute.upload(source, source.array, 0); source.version++;
  assert.throws(() => computeAttributeSnapshot(device, source), /UNPREPARED/);
  compute.upload(source, source.array, source.version);
  assert.equal(computeAttributeSnapshot(device, source).count, 2); compute.dispose();
});

test('resizing replaces storage and rejects previously borrowed generations', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(4), 4);
  const old = compute.upload(source, source.array, 0), view = computeAttributeSnapshot(device, source);
  source.array = new Float32Array(8); source.count = 2; source.version++;
  const next = compute.upload(source, source.array, source.version);
  assert.notEqual(next, old); assert.equal(old.destroyed, 1); assert.throws(view.assertCurrent, /STALE/);
  assert.equal(computeAttributeSnapshot(device, source).buffer, next); compute.dispose();
});

test('later writes to the same allocation keep existing vertex bindings valid', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(8), 4);
  compute.upload(source, source.array, 0); const view = computeAttributeSnapshot(device, source);
  source.version++; source.array[0] = 7; compute.upload(source, source.array, source.version);
  assert.doesNotThrow(view.assertCurrent); assert.equal(new Float32Array(view.buffer.bytes.buffer)[0], 7); compute.dispose();
});

test('foreign devices and unpublished attributes cannot supply native vertex handles', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(4), 4);
  assert.throws(() => computeAttributeSnapshot(device, source), /UNPREPARED/);
  compute.upload(source, source.array, 0);
  assert.throws(() => computeAttributeSnapshot(gpuDevice(), source), /UNPREPARED/);
  assert.throws(() => computeAttributeSnapshot(device, {...source}), /UNPREPARED/); compute.dispose();
});

test('two owners of the same source/device fail before allocating or writing', () => {
  const device = gpuDevice(), first = createGpuCompute(device), second = createGpuCompute(device), source = attribute(new Float32Array(4), 4);
  const original = first.upload(source, source.array, 0), before = device.events.length;
  assert.throws(() => second.upload(source, source.array, 0), /OWNER/);
  assert.equal(device.events.length, before); second.dispose();
  assert.equal(computeAttributeSnapshot(device, source).buffer, original); first.dispose();
});

test('release and owner disposal revoke borrows without destroying another owner', () => {
  const device = gpuDevice(), first = createGpuCompute(device), second = createGpuCompute(device);
  const a = attribute(new Float32Array(4), 4), b = attribute(new Float32Array(4), 4);
  const one = first.upload(a, a.array), two = second.upload(b, b.array), view = computeAttributeSnapshot(device, a);
  first.release(a); assert.equal(one.destroyed, 1); assert.throws(view.assertCurrent, /STALE/);
  assert.throws(() => computeAttributeSnapshot(device, a), /UNPREPARED/);
  first.dispose(); assert.equal(computeAttributeSnapshot(device, b).buffer, two); second.dispose();
  assert.equal(two.destroyed, 1);
});

test('integer storage remains computable but cannot impersonate float vertices', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Uint32Array(4), 4);
  compute.upload(source, source.array); assert.throws(() => computeAttributeSnapshot(device, source), /FORMAT/); compute.dispose();
});

test('shape and normalization edits are detected before native vertex use', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(8), 4);
  compute.upload(source, source.array); const view = computeAttributeSnapshot(device, source);
  source.normalized = true; assert.throws(view.assertCurrent, /SHAPE/); source.normalized = false;
  source.itemSize = 2; assert.throws(view.assertCurrent, /SHAPE/); compute.dispose();
});

test('device loss revokes storage publication and borrowed completion waits', async () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(4), 4), fence = deferred();
  device.queue.onSubmittedWorkDone = () => fence.promise;
  compute.upload(source, source.array); const view = computeAttributeSnapshot(device, source), idle = view.whenIdle();
  device.lose({message: 'lost adapter'}); await assert.rejects(idle, /DEVICE_LOST/);
  assert.throws(view.assertCurrent, /DEVICE_LOST/); assert.throws(() => computeAttributeSnapshot(device, source), /UNPREPARED/);
});

test('shader storage can capture before its preceding compute publication without allocating or uploading', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(8), 4);
  const before = device.events.length, lease = captureComputeStorage(device, source);
  assert.equal(device.events.length, before); assert.throws(lease.resolve, /UNPREPARED/);
  const buffer = compute.upload(source, source.array, source.version), published = device.events.length;
  const resource = lease.resolve();
  assert.deepEqual(resource, {buffer, offset: 0, size: 32}); assert.equal(Object.isFrozen(resource), true);
  assert.equal(lease.resolve(), resource); assert.equal(captureComputeStorage(device, source).resolve(), resource);
  assert.equal(device.events.length, published); compute.dispose();
});

test('queued shader reads resolve their captured version after live source state has advanced', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array([1, 2, 3, 4]), 4);
  compute.upload(source, source.array, 0);
  source.version = 1; source.array[0] = 11;
  const upload = new Float32Array(source.array), admission = compute.captureUpload(source, upload, 1);
  const lease = captureComputeStorage(device, source);
  assert.throws(lease.resolve, /UNPREPARED/);
  // The preceding compute's captured upload still describes version 1 even
  // after an application callback requests version 2 and resizes the source.
  source.version = 2; source.array = new Float32Array(8).fill(22); source.count = 2;
  const buffer = compute.upload(source, upload, 1, {admission}); compute.releaseUpload(admission);
  assert.equal(lease.resolve().buffer, buffer); assert.equal(lease.byteLength, 16); assert.equal(lease.version, 1);
  assert.equal(new Float32Array(buffer.bytes.buffer)[0], 11); assert.doesNotThrow(lease.assertCurrent);
  const newer = captureComputeStorage(device, source);
  assert.throws(newer.resolve, /SHAPE/);
  compute.upload(source, source.array, 2); assert.equal(newer.resolve().size, 32);
  assert.throws(lease.resolve, /STALE/); compute.dispose();
});

test('shader borrowing observes GPU-written bytes and ignores unrequested CPU edits', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array([1, 2, 3, 4]), 4);
  const buffer = compute.upload(source, source.array, 0), lease = captureComputeStorage(device, source);
  new Float32Array(buffer.bytes.buffer)[0] = 73; source.array.fill(-999);
  for (let i = 0; i < 4; i++) assert.equal(lease.resolve().buffer, buffer);
  assert.equal(new Float32Array(buffer.bytes.buffer)[0], 73);
  assert.equal(device.events.filter(event => event[0] === 'upload').length, 1); compute.dispose();
});

test('stable allocations retain storage resource identity across versions but old version leases fail', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(4), 4);
  compute.upload(source, source.array, 0);
  const lease = captureComputeStorage(device, source), original = lease.resolve();
  source.version++; source.array[0] = 5; compute.upload(source, source.array, source.version);
  assert.throws(lease.assertCurrent, /UNPREPARED/);
  assert.equal(captureComputeStorage(device, source).resolve(), original); compute.dispose();
});

test('shader storage supports normalized integer and instanced struct buffers without vertex format restrictions', () => {
  for (const ArrayType of [Float32Array, Int32Array, Uint32Array]) {
    const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new ArrayType(40), 20);
    delete source.isStorageBufferAttribute; source.isStorageInstancedBufferAttribute = true;
    const buffer = compute.upload(source, source.array, 0), lease = captureComputeStorage(device, source);
    assert.deepEqual(lease.resolve(), {buffer, offset: 0, size: 160}); compute.dispose();
  }
});

test('indirect storage borrowing preserves the published GPU allocation and unrequested CPU history', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Uint32Array([3, 1, 0, 0]), 1);
  source.isIndirectStorageBufferAttribute = true;
  const lease = captureComputeStorage(device, source), buffer = compute.upload(source, source.array, 0);
  new Uint32Array(buffer.bytes.buffer)[0] = 6; source.array[0] = 99;
  assert.equal(lease.resolve().buffer, buffer); assert.equal(lease.resolve().size, 16);
  assert.equal(buffer.usage & (128 | 256), 128 | 256);
  assert.equal(new Uint32Array(buffer.bytes.buffer)[0], 6);
  assert.equal(device.events.filter(event => event[0] === 'upload').length, 1); compute.dispose();
  assert.throws(lease.resolve, /DISPOSED|STALE/);
});

test('shader storage rejects malformed source layouts and mismatched publications without GPU side effects', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = attribute(new Float32Array(8), 4);
  for (const invalid of [{...source, normalized: true}, {...source, isInterleavedBufferAttribute: true},
    {...source, version: -1}, {...source, count: 0},
    {...source, itemSize: 3}, attribute(new Uint8Array(8), 4), attribute(new Float64Array(8), 4),
    attribute(new Float32Array(8), 4, false)])
    assert.throws(() => captureComputeStorage(device, invalid), /SHAPE/);
  assert.equal(device.events.length, 0);
  const lease = captureComputeStorage(device, source);
  source.itemSize = 2; source.count = 4; compute.upload(source, source.array, 0);
  assert.throws(lease.resolve, /SHAPE/);
  assert.throws(captureComputeStorage(gpuDevice(), source).resolve, /UNPREPARED/); compute.dispose();
});

test('shader leases cannot borrow a replacement owner and device loss retains its terminal reason', async () => {
  const device = gpuDevice(), first = createGpuCompute(device), source = attribute(new Float32Array(4), 4);
  first.upload(source, source.array, 0); const stale = captureComputeStorage(device, source); stale.resolve();
  first.release(source);
  const second = createGpuCompute(device); second.upload(source, source.array, 0);
  assert.throws(stale.resolve, /STALE/);
  const lease = captureComputeStorage(device, source), fence = deferred(); lease.resolve();
  device.queue.onSubmittedWorkDone = () => fence.promise;
  const idle = lease.whenIdle(); device.lose({message: 'shader storage loss'});
  await assert.rejects(idle, /DEVICE_LOST/); assert.throws(lease.resolve, /DEVICE_LOST/);
  first.dispose(); second.dispose();
});
