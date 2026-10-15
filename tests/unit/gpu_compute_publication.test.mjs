import test from 'node:test';
import assert from 'node:assert/strict';
import {createComputeAttributeOwner, computeAttributeSnapshot} from '../../tools/ingest/gpu_compute_attributes.mjs';

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
