import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {createGpuProgramGeometry, programGeometrySnapshot} from '../../tools/ingest/gpu_buffer_geometry.mjs';
import {gpuDevice, attribute, geometry} from './compute_geometry_fixture.mjs';

const reflected = (name, scalar, components, location = 0) => ({name, scalar, components, location, locations:1});

function shared(array, itemSize, gpuType = 1013) {
  const a = attribute(array, itemSize);
  a.gpuType = gpuType;
  return a;
}

test('reflected uint attributes borrow compute-owned Uint32 storage without conversion', () => {
  const device=gpuDevice(), compute=createGpuCompute(device), ids=shared(new Uint32Array([4,8,15,16]),1);
  const native=compute.upload(ids,ids.array,ids.version);
  new Uint32Array(native.bytes.buffer).set([23,42,7,9]);
  const handle=createGpuProgramGeometry(device,geometry({id:ids}),[reflected('id','uint',1,3)]);
  const state=programGeometrySnapshot(handle,device);
  assert.equal(state.vertexBuffers[0],native);
  assert.equal(state.layouts[0].attributes[0].format,'uint32');
  assert.equal(state.layouts[0].attributes[0].shaderLocation,3);
  assert.equal(handle.bufferBytes,0);
  ids.array.fill(99); handle.update();
  assert.deepEqual([...new Uint32Array(native.bytes.buffer)],[23,42,7,9]);
  handle.dispose(); compute.dispose();
});

test('reflected signed vectors borrow Int32 storage with native sint formats', () => {
  const device=gpuDevice(), compute=createGpuCompute(device), value=shared(new Int32Array([-1,2,-3,4, 5,-6,7,-8]),4);
  const native=compute.upload(value,value.array,value.version);
  const handle=createGpuProgramGeometry(device,geometry({value}),[reflected('value','int',4,5)]);
  const state=programGeometrySnapshot(handle,device);
  assert.equal(state.vertexBuffers[0],native);
  assert.equal(state.layouts[0].arrayStride,16);
  assert.equal(state.layouts[0].attributes[0].format,'sint32x4');
  handle.dispose(); compute.dispose();
});

test('integer shared attributes enforce shader signedness and gpuType identity', () => {
  const device=gpuDevice(), compute=createGpuCompute(device);
  const u=shared(new Uint32Array([1,2,3,4]),1), s=shared(new Int32Array([1,2,3,4]),1);
  compute.upload(u,u.array,u.version); compute.upload(s,s.array,s.version);
  assert.throws(()=>createGpuProgramGeometry(device,geometry({v:u}),[reflected('v','int',1)]),/signedness/);
  assert.throws(()=>createGpuProgramGeometry(device,geometry({v:s}),[reflected('v','uint',1)]),/signedness/);
  u.gpuType=0;
  assert.throws(()=>createGpuProgramGeometry(device,geometry({v:u}),[reflected('v','uint',1)]),/gpuType IntType/);
  compute.dispose();
});

test('program geometry can combine compute indices and integer vertex IDs', () => {
  const device=gpuDevice(), compute=createGpuCompute(device);
  const ids=shared(new Uint32Array([10,20,30,40]),1);
  const index=attribute(new Uint32Array([0,2,3]),1);
  compute.upload(ids,ids.array,ids.version); const ib=compute.upload(index,index.array,index.version);
  const handle=createGpuProgramGeometry(device,geometry({id:ids},index),[reflected('id','uint',1)]);
  const state=programGeometrySnapshot(handle,device);
  assert.equal(state.indexBuffer,ib); assert.equal(state.indexFormat,'uint32'); assert.equal(state.readIndex,null);
  assert.deepEqual(state.drawRange,{first:0,count:3});
  handle.dispose(); compute.dispose();
});
