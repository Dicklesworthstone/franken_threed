import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {createGpuBufferGeometry, bufferGeometrySnapshot} from '../../tools/ingest/gpu_buffer_geometry.mjs';
import {gpuDevice, attribute, geometry} from './compute_geometry_fixture.mjs';
import {createComputeAttributeOwner, captureComputeAttributeView, computeAttributeSnapshot,
  snapshotComputeAttributeView, refreshComputeAttributeView} from '../../tools/ingest/gpu_compute_attributes.mjs';

const position = () => attribute(new Float32Array(12), 4);
const indices = values => attribute(new Uint32Array(values), 1);

test('compute-authored indices bind the exact storage allocation without CPU topology shadows', () => {
  const device = gpuDevice(), compute = createGpuCompute(device);
  const p = position(), i = indices([0, 1, 2, 2, 1, 3]);
  compute.upload(p, p.array, p.version);
  const ib = compute.upload(i, i.array, i.version);
  const before = device.events.length;
  const handle = createGpuBufferGeometry(device, geometry({position:p}, i), {maxBytes:1,maxInitialBytes:0});
  const state = bufferGeometrySnapshot(handle, device);
  assert.equal(state.indexBuffer, ib);
  assert.equal(state.indexFormat, 'uint32');
  assert.equal(state.indexCount, 6);
  assert.equal(state.readIndex, null);
  assert.deepEqual(state.drawRange, {first:0,count:6});
  assert.equal(handle.bufferBytes, 0);
  assert.equal(device.events.length, before);
  handle.dispose(); assert.equal(ib.destroyed, 0); compute.dispose();
});

test('GPU index writes remain resident across geometry updates and draw ranges', () => {
  const device = gpuDevice(), compute = createGpuCompute(device);
  const p = position(), i = indices([0,1,2,2,1,3]);
  compute.upload(p,p.array,p.version);
  const ib = compute.upload(i,i.array,i.version);
  new Uint32Array(ib.bytes.buffer).set([3,2,1,1,2,0]);
  const g = geometry({position:p},i), handle = createGpuBufferGeometry(device,g);
  i.array.fill(99); // unversioned CPU edits cannot overwrite GPU topology.
  handle.update();
  assert.deepEqual([...new Uint32Array(ib.bytes.buffer)], [3,2,1,1,2,0]);
  g.drawRange={start:2,count:3};
  assert.deepEqual(bufferGeometrySnapshot(handle,device).drawRange,{first:2,count:3});
  handle.dispose(); compute.dispose();
});

test('pending index versions must pass through compute before drawing', () => {
  const device = gpuDevice(), compute = createGpuCompute(device);
  const p = position(), i = indices([0,1,2]);
  compute.upload(p,p.array,p.version); compute.upload(i,i.array,i.version);
  const handle=createGpuBufferGeometry(device,geometry({position:p},i));
  i.array[0]=2; i.version++;
  assert.throws(()=>bufferGeometrySnapshot(handle,device),/UNPREPARED/);
  compute.upload(i,i.array,i.version);
  handle.update();
  assert.equal(bufferGeometrySnapshot(handle,device).indexBuffer,compute.getBuffer(i));
  handle.dispose(); compute.dispose();
});

test('computed indices reject non-32-bit storage and non-scalar geometry', () => {
  const device=gpuDevice(), compute=createGpuCompute(device), p=position();
  compute.upload(p,p.array,p.version);
  const bad16=attribute(new Uint16Array([0,1,2]),1);
  assert.throws(()=>compute.upload(bad16,bad16.array,bad16.version),/SHAPE/);
  const badVec=attribute(new Uint32Array([0,0,1,1,2,2]),2);
  compute.upload(badVec,badVec.array,badVec.version);
  assert.throws(()=>createGpuBufferGeometry(device,geometry({position:p},badVec)),/scalar non-interleaved indices/);
  compute.dispose();
});

test('captured index geometry borrows its accepted allocation after the source version advances', () => {
  const device = gpuDevice(), compute = createGpuCompute(device);
  const p = position(), source = indices([0, 1, 2, 2, 1, 3]);
  compute.upload(p, p.array, p.version);
  const buffer = compute.upload(source, source.array, source.version), view = Object.create(source);
  captureComputeAttributeView(device, source, view).install();
  const captured = snapshotComputeAttributeView(view);
  source.array.fill(99); source.version++;
  const before = device.events.length;
  const handle = createGpuBufferGeometry(device, geometry({position: p}, captured), {maxBytes: 1, maxInitialBytes: 0});
  const state = bufferGeometrySnapshot(handle, device);
  assert.equal(state.indexBuffer, buffer); assert.equal(state.indexFormat, 'uint32');
  assert.equal(state.indexCount, 6); assert.equal(state.readIndex, null);
  assert.deepEqual([...new Uint32Array(buffer.bytes.buffer)], [0, 1, 2, 2, 1, 3]);
  assert.equal(device.events.length, before, 'capturing and borrowing never upload unrequested CPU topology');
  assert.equal(handle.bufferBytes, 0);
  handle.dispose(); assert.equal(buffer.destroyed, 0); compute.dispose();
});

test('refreshed index proofs follow only admitted upload versions and retain earlier queued proofs', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), source = indices([0, 1, 2]);
  const buffer = compute.upload(source, source.array, source.version), view = Object.create(source);
  captureComputeAttributeView(device, source, view).install();
  const first = snapshotComputeAttributeView(view), old = computeAttributeSnapshot(device, first, {index: true});
  source.array.set([2, 0, 1]); source.version++;
  const next = refreshComputeAttributeView(device, source, first, source.version);
  assert.equal(next, refreshComputeAttributeView(device, source, first, source.version));
  assert.throws(() => computeAttributeSnapshot(device, next, {index: true}), /UNPREPARED/);
  old.assertCurrent();
  compute.upload(source, source.array, source.version);
  source.array.fill(99); source.version++;
  const current = computeAttributeSnapshot(device, next, {index: true});
  assert.equal(current.buffer, buffer); assert.equal(current.arrayStride, 4);
  assert.equal(current.count, 3); assert.equal(current.indexFormat, 'uint32');
  assert.deepEqual([...new Uint32Array(buffer.bytes.buffer)], [2, 0, 1]);
  assert.throws(() => old.assertCurrent(), /UNPREPARED/);
  assert.throws(() => computeAttributeSnapshot(gpuDevice(), next, {index: true}), /OWNER/);
  compute.dispose();
  assert.throws(() => current.assertCurrent(), /DISPOSED/);
});

test('captured index views enforce scalar Uint32 layout and INDEX usage without weakening vertex validation', () => {
  const device = gpuDevice(), owner = createComputeAttributeOwner(device);
  for (const [array, itemSize] of [[new Float32Array(3), 1], [new Int32Array(3), 1], [new Uint32Array(6), 2]]) {
    const source = attribute(array, itemSize), view = Object.create(source);
    captureComputeAttributeView(device, source, view).install();
    assert.throws(() => computeAttributeSnapshot(device, view, {index: true}), /scalar Uint32/);
  }
  const source = indices([0, 1, 2]), view = Object.create(source);
  const buffer = device.createBuffer({size: source.array.byteLength, usage: 128 | 32});
  owner.publish(source, buffer, source.array, source.version, () => {}, async () => {});
  captureComputeAttributeView(device, source, view).install();
  assert.throws(() => computeAttributeSnapshot(device, view, {index: true}), /INDEX buffer usage/);
  assert.throws(() => computeAttributeSnapshot(device, view, {index: true, integer: true}), /distinct/);
  assert.throws(() => computeAttributeSnapshot(device, view), /signedness/);
  assert.equal(computeAttributeSnapshot(device, view, {integer: true}).indexFormat, null);
  owner.dispose();
});
