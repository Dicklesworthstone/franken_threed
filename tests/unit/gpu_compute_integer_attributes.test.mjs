import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {createGpuProgramGeometry, programGeometrySnapshot} from '../../tools/ingest/gpu_buffer_geometry.mjs';
import {gpuDevice, attribute, geometry} from './compute_geometry_fixture.mjs';
import {createProgramMeshes} from '../../tools/ingest/animation_program_mesh.mjs';

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

test('program staging borrows GPU indices for smooth strips and refuses CPU-only topology lowering', async t => {
  // Recording device: verifies the real staging consumer, not WGSL execution.
  const device = gpuDevice(), compute = createGpuCompute(device);
  Object.assign(device, {createBindGroupLayout: d => d, createPipelineLayout: d => d,
    createRenderPipelineAsync: async d => d});
  const ids = shared(new Uint32Array([10,20,30]), 1);
  const index = attribute(new Uint32Array([0,1,0xffffffff,1,2]), 1);
  const vertices = compute.upload(ids, ids.array, ids.version);
  const indices = compute.upload(index, index.array, index.version);
  const gpu = createGpuProgramGeometry(device, geometry({id: ids}, index), [reflected('id','uint',1)]);
  const fail = (code, message) => { throw Object.assign(new Error(message), {code}); };
  const meshes = createProgramMeshes({device, format: 'rgba8unorm', depthFormat: null, sampleCount: 1,
    maxDraws: 4, label: 'shared-index', fail, lost: device.lost,
    scoped(_device, fn) { return {value: fn(), errors: Promise.resolve()}; }});
  t.after(() => { meshes.dispose(); gpu.dispose(); compute.dispose(); });
  const add = (topology, flat = false) => meshes.add(gpu, {topology, stripIndexFormat: 'uint32',
    raster: {blend: null, frontFace: 'ccw', cullMode: 'none', writeMask: 15},
    program: {vertex: '// recording fixture', fragment: '// recording fixture',
      reflection: {uniformBufferSize: 16, textures: [], outputs: [{location: 0}], varyings: [{flat}]}}});
  const strip = await add('line-strip');
  meshes.begin();
  const command = {}, before = device.events.length;
  meshes.stage(strip, {programUniforms: new Uint8Array(16)}, command);
  assert.equal(command.indexBuffer, indices);
  assert.equal(command.vertexBuffers[0], vertices);
  assert.equal(command.indexFormat, 'uint32');
  assert.equal(command.pipeline.primitive.stripIndexFormat, 'uint32');
  assert.equal(command.count, 5);
  assert.equal(device.events.length, before);
  for (const [topology, flat] of [['triangles', false], ['lines', false], ['line-strip', true]]) {
    const record = await add(topology, flat);
    assert.throws(() => meshes.stage(record, {programUniforms: new Uint8Array(16)}, {}), /resident index history/);
  }
});
