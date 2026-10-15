import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {computeAttributeSnapshot} from '../../tools/ingest/gpu_compute_attributes.mjs';
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
