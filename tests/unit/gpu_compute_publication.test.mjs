import test from 'node:test';
import assert from 'node:assert/strict';
import {createComputeAttributeOwner, computeAttributeSnapshot} from '../../tools/ingest/gpu_compute_attributes.mjs';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {gpuDevice} from './compute_geometry_fixture.mjs';

const attribute = () => ({isStorageBufferAttribute: true, itemSize: 4, count: 2, version: 0, normalized: false, array: new Float32Array(8)});
const buffer = (size = 32, usage = 128 | 32) => ({size, usage});
const publish = (owner, a, b = buffer(), version = a.version) =>
  owner.publish(a, b, a.array, version, () => {}, async () => {});

test('only correctly sized storage+vertex GPU buffers may be borrowed', () => {
  const device = {}, owner = createComputeAttributeOwner(device), a = attribute();
  for (const bad of [null, {}, buffer(16), buffer(32, 128), buffer(32, 32), buffer(32, 0)]) {
    assert.throws(() => publish(owner, a, bad), /PUBLICATION/);
    assert.throws(() => computeAttributeSnapshot(device, a), /UNPREPARED/);
  }
  const good = buffer();
  publish(owner, a, good);
  assert.equal(computeAttributeSnapshot(device, a).buffer, good);
  owner.dispose();
});

test('publication cannot acknowledge a future or stale host version', () => {
  const device = {}, owner = createComputeAttributeOwner(device), a = attribute();
  publish(owner, a);
  a.version = 2;
  for (const version of [0, 1, 3, -1, NaN])
    assert.throws(() => publish(owner, a, buffer(), version), /PUBLICATION/);
  assert.throws(() => computeAttributeSnapshot(device, a), /UNPREPARED/);
  publish(owner, a, buffer(), 2);
  assert.equal(computeAttributeSnapshot(device, a).count, 2);
  owner.dispose();
});

test('invalid competing owner cannot replace published allocation', () => {
  const device = {}, first = createComputeAttributeOwner(device), second = createComputeAttributeOwner(device), a = attribute();
  const original = buffer();
  publish(first, a, original);
  assert.throws(() => publish(second, a, buffer()), /OWNER/);
  assert.equal(computeAttributeSnapshot(device, a).buffer, original);
  first.dispose();
  publish(second, a);
  assert.notEqual(computeAttributeSnapshot(device, a).buffer, original);
  second.dispose();
});

test('INDEX usage is required only when borrowing a scalar Uint32 stream as indices', () => {
  const device = {}, owner = createComputeAttributeOwner(device);
  const a = {...attribute(), array: new Uint32Array([0, 1, 2]), itemSize: 1, count: 3};
  const vertexOnly = buffer(12);
  publish(owner, a, vertexOnly);
  assert.equal(computeAttributeSnapshot(device, a, {integer: true}).buffer, vertexOnly);
  assert.throws(() => computeAttributeSnapshot(device, a, {index: true}), /INDEX buffer usage/);
  const indexed = buffer(12, 128 | 32 | 16);
  publish(owner, a, indexed);
  assert.equal(computeAttributeSnapshot(device, a, {index: true}).buffer, indexed);
  assert.throws(() => computeAttributeSnapshot(device, a, {index: true, integer: true}), /distinct/);
  owner.dispose();
});

test('rejected compute upload versions cannot mutate an already published buffer', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), a = attribute();
  const native = compute.upload(a, a.array, a.version);
  const original = Array.from(native.bytes), before = device.events.length;
  a.array.fill(42);
  for (const version of [1, -1, NaN, 0.5]) {
    assert.throws(() => compute.upload(a, a.array, version), /PUBLICATION/);
    assert.deepEqual(Array.from(native.bytes), original);
    assert.equal(device.events.length, before);
    assert.equal(computeAttributeSnapshot(device, a).buffer, native);
  }
  a.version = 2;
  assert.throws(() => compute.upload(a, a.array, 0), /PUBLICATION/);
  assert.deepEqual(Array.from(native.bytes), original);
  assert.equal(device.events.length, before);
  compute.upload(a, a.array, a.version);
  assert.equal(new Float32Array(native.bytes.buffer)[0], 42);
  assert.equal(computeAttributeSnapshot(device, a).buffer, native);
  compute.dispose();
});
