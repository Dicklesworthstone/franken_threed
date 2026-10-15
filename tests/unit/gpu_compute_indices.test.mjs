import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {createGpuBufferGeometry, bufferGeometrySnapshot} from '../../tools/ingest/gpu_buffer_geometry.mjs';
import {gpuDevice, attribute, geometry} from './compute_geometry_fixture.mjs';

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
