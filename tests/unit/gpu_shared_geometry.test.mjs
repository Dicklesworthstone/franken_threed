import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuCompute} from '../../tools/ingest/gpu_compute.mjs';
import {createGpuBufferGeometry, bufferGeometrySnapshot, createGpuInstanceAttributes,
  instanceAttributesSnapshot, inspectInstanceAttributes, createGpuProgramGeometry,
  programGeometrySnapshot} from '../../tools/ingest/gpu_buffer_geometry.mjs';
import {gpuDevice, attribute, geometry, deferred} from './compute_geometry_fixture.mjs';
const source = (n = 3, stride = 4) => attribute(new Float32Array(n * stride), stride);
function upload(compute, attribute) { return compute.upload(attribute, attribute.array, attribute.version); }
const floats = buffer => [...new Float32Array(buffer.bytes.buffer)];
const positionLayout = {arrayStride: 16, stepMode: 'vertex', attributes: [{shaderLocation: 0, offset: 0, format: 'float32x3'}]};

test('padded storage positions bind the exact compute allocation without uploads or shadows', async () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(), buffer = upload(compute, position);
  new Float32Array(buffer.bytes.buffer).set([1,2,3,99, 4,5,6,99, 7,8,9,99]);
  const before = device.events.length, g = geometry({position}), handle = createGpuBufferGeometry(device, g, {maxBytes: 1, maxInitialBytes: 0});
  const snapshot = bufferGeometrySnapshot(handle, device);
  assert.equal(snapshot.vertexBuffers[0], buffer); assert.deepEqual(snapshot.layouts, [positionLayout]);
  assert.equal(snapshot.source, g); assert.equal(snapshot.vertexCount, 3);
  assert.equal(handle.bufferBytes, 0); assert.equal(handle.shadowBytes, 0); assert.equal(position.uploads, 0);
  assert.equal(device.events.length, before); assert.equal(handle.diagnostics.sharedAttributes, 1);
  await handle.whenIdle(); handle.dispose(); assert.equal(buffer.destroyed, 0); compute.dispose();
});

test('mixed CPU/GPU streams retain exact CPU version and callback semantics', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(), gpu = upload(compute, position);
  const normal = attribute(new Float32Array([0,0,1, 0,0,1, 0,0,1]), 3, false);
  const handle = createGpuBufferGeometry(device, geometry({normal, position}));
  let state = bufferGeometrySnapshot(handle, device); const cpu = state.vertexBuffers[1];
  assert.equal(state.vertexBuffers[0], gpu); assert.equal(normal.uploads, 1); assert.equal(handle.bufferBytes, 36);
  normal.array[0] = 999; new Float32Array(gpu.bytes.buffer)[0] = 42;
  handle.update(); assert.equal(floats(cpu)[0], 0); assert.equal(floats(gpu)[0], 42); assert.equal(normal.uploads, 1);
  normal.array[1] = 7; normal.version++; normal.updateRanges.push({start: 1, count: 1});
  handle.update(); assert.deepEqual(floats(cpu).slice(0, 3), [0,7,1]); assert.equal(normal.uploads, 2);
  assert.equal(position.uploads, 0); assert.equal(normal.updateRanges.length, 0);
  handle.dispose(); assert.equal(cpu.destroyed, 1); assert.equal(gpu.destroyed, 0); compute.dispose();
});

test('one storage source used at two vertex locations occupies one binding slot', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), shared = source(); upload(compute, shared);
  const handle = createGpuBufferGeometry(device, geometry({position: shared, normal: shared}), {maxAttributes: 1});
  const state = bufferGeometrySnapshot(handle, device);
  assert.equal(state.vertexBuffers.length, 1);
  assert.deepEqual(state.layouts[0].attributes.map(a => a.shaderLocation), [0,1]);
  assert.equal(handle.diagnostics.sharedAttributes, 1); handle.dispose(); compute.dispose();
});

test('ordinary indices retain uploaded topology and draw-range semantics with shared vertices', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(); upload(compute, position);
  const index = attribute(new Uint16Array([2,0,1]), 1, false), g = geometry({position}, index);
  const handle = createGpuBufferGeometry(device, g, {maxBytes: 8});
  let state = bufferGeometrySnapshot(handle, device); assert.equal(state.indexFormat, 'uint16');
  assert.deepEqual([0,1,2].map(state.readIndex), [2,0,1]); index.array[0] = 0;
  handle.update(); state = bufferGeometrySnapshot(handle, device); assert.equal(state.readIndex(0), 2);
  g.drawRange = {start: 1, count: 1}; assert.deepEqual(bufferGeometrySnapshot(handle, device).drawRange, {first: 1, count: 1});
  assert.equal(handle.bufferBytes, 8); handle.dispose(); compute.dispose();
});

test('GPU replacement invalidates the old snapshot and advances geometry generation on update', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source();
  const oldBuffer = upload(compute, position), handle = createGpuBufferGeometry(device, geometry({position}));
  const old = bufferGeometrySnapshot(handle, device);
  position.array = new Float32Array(16); position.count = 4; position.version++;
  const next = upload(compute, position);
  assert.equal(oldBuffer.destroyed, 1); assert.throws(() => bufferGeometrySnapshot(handle, device), /STALE/);
  handle.update(); const state = bufferGeometrySnapshot(handle, device);
  assert.equal(state.vertexBuffers[0], next); assert.equal(state.vertexCount, 4); assert.ok(state.generation > old.generation);
  assert.equal(state.signature, old.signature); handle.dispose(); assert.equal(next.destroyed, 0); compute.dispose();
});

test('ordinary scene disposal and recreation do not destroy shared compute buffers', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(), buffer = upload(compute, position);
  const g = geometry({position}), a = createGpuBufferGeometry(device, g), b = createGpuBufferGeometry(device, g);
  g.dispose(); assert.equal(buffer.destroyed, 0); assert.throws(() => bufferGeometrySnapshot(a, device), /RELEASED/);
  a.update(); b.update(); assert.equal(bufferGeometrySnapshot(a, device).vertexBuffers[0], buffer);
  a.dispose(); b.dispose(); assert.equal(buffer.destroyed, 0); compute.dispose(); assert.equal(buffer.destroyed, 1);
});

test('adding and removing computed streams does not re-upload unchanged CPU streams', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = attribute(new Float32Array(9), 3, false);
  const g = geometry({position}), handle = createGpuBufferGeometry(device, g), normal = source(); upload(compute, normal);
  position.array.fill(999); g.attributes.normal = normal; handle.update();
  assert.deepEqual(floats(bufferGeometrySnapshot(handle, device).vertexBuffers[0]), new Array(9).fill(0));
  delete g.attributes.normal; handle.update(); assert.equal(position.uploads, 1);
  handle.dispose(); assert.equal(compute.getBuffer(normal).destroyed, 0); compute.dispose();
});

test('unprepared later shared attributes fail before any earlier CPU upload or callback', () => {
  const device = gpuDevice(), position = attribute(new Float32Array(9), 3, false), normal = source();
  assert.throws(() => createGpuBufferGeometry(device, geometry({position, normal})), /UNPREPARED/);
  assert.equal(device.buffers.length, 0); assert.equal(position.uploads, 0);
});

test('CPU callbacks retiring an earlier shared stream cannot publish a dangling snapshot', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(); upload(compute, position);
  const normal = attribute(new Float32Array(9), 3, false);
  normal.onUploadCallback = () => compute.release(position);
  assert.throws(() => createGpuBufferGeometry(device, geometry({position, normal})), /STALE/);
  assert.equal(device.buffers[0].destroyed, 1); assert.equal(device.buffers[1].destroyed, 1); compute.dispose();
});

test('shared entries count toward identity budgets, not duplicate byte budgets', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(), normal = source();
  upload(compute, position); upload(compute, normal); const before = device.events.length;
  assert.throws(() => createGpuBufferGeometry(device, geometry({position, normal}), {maxAttributes: 1}), /budget/);
  assert.equal(device.events.length, before);
  const handle = createGpuBufferGeometry(device, geometry({position, normal}), {maxBytes: 1, maxInitialBytes: 0});
  assert.equal(handle.bufferBytes, 0); handle.dispose(); compute.dispose();
});

test('program reflection binds shared vec3 fields and matrices at their actual input locations', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(), matrix = source(2,16);
  matrix.isInstancedBufferAttribute = true; matrix.meshPerAttribute = 1;
  const p = upload(compute, position), m = upload(compute, matrix);
  const attributes = [{name:'position', location:2, locations:1, components:3, scalar:'float'},
    {name:'matrix', location:5, locations:4, components:4, scalar:'float'}];
  const handle = createGpuProgramGeometry(device, geometry({position, matrix}), attributes);
  const state = programGeometrySnapshot(handle, device);
  assert.deepEqual(state.vertexBuffers, [p,m]); assert.equal(state.instanceCapacity, 2);
  assert.equal(state.layouts[0].attributes[0].format, 'float32x3');
  assert.equal(state.layouts[1].arrayStride, 64); assert.equal(state.layouts[1].stepMode, 'instance');
  assert.deepEqual(state.layouts[1].attributes.map(a => [a.shaderLocation,a.offset]), [[5,0],[6,16],[7,32],[8,48]]);
  handle.dispose(); compute.dispose();
});

test('shared program instances never simulate divisors by copying stale CPU data', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source();
  position.isInstancedBufferAttribute = true; position.meshPerAttribute = 2; upload(compute, position);
  assert.throws(() => createGpuProgramGeometry(device, geometry({position}), [
    {name:'position',location:0,locations:1,components:3,scalar:'float'}]), /divisor one/); compute.dispose();
});

function instanceAttribute(count, width) {
  return {...source(count, width), isStorageInstancedBufferAttribute: true, isInstancedBufferAttribute: true, meshPerAttribute: 1};
}
test('InstancedMesh matrices and padded RGB bind compute storage with correct strides', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), matrix = instanceAttribute(3,16), color = instanceAttribute(3,4);
  const m = upload(compute, matrix), c = upload(compute, color);
  const mesh = {isInstancedMesh:true, count:2, morphTexture:null, instanceMatrix:matrix, instanceColor:color};
  const inspected = inspectInstanceAttributes(mesh), handle = createGpuInstanceAttributes(device, mesh, {maxBytes:1});
  const state = instanceAttributesSnapshot(handle, device);
  assert.equal(state.signature, inspected.signature); assert.equal(state.instanceCount, 2);
  assert.deepEqual(state.vertexBuffers, [m,c]); assert.deepEqual(state.layouts.map(l => l.arrayStride), [64,16]);
  assert.equal(state.layouts[1].attributes[0].format, 'float32x3'); assert.equal(handle.bufferBytes, 0);
  mesh.count = 1; assert.equal(instanceAttributesSnapshot(handle, device).instanceCount, 1);
  mesh.count = 4; assert.throws(() => instanceAttributesSnapshot(handle, device), /instance count/);
  handle.dispose(); assert.equal(m.destroyed + c.destroyed, 0); compute.dispose();
});

test('shared instance inspection is metadata-only and never needs a GPU upload', () => {
  const matrix = instanceAttribute(1,16), color = instanceAttribute(1,4);
  const inspected = inspectInstanceAttributes({isInstancedMesh:true, count:1, morphTexture:null, instanceMatrix:matrix, instanceColor:color});
  assert.equal(inspected.capacity, 1); assert.equal(inspected.channels.instanceColor, true);
  assert.equal(matrix.uploads + color.uploads, 0);
});

test('CPU instance color and computed matrices keep separate allocation ownership', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), matrix = instanceAttribute(1,16);
  const color = {...attribute(new Float32Array([1,0,0]), 3, false), isInstancedBufferAttribute:true, meshPerAttribute:1};
  const buffer = upload(compute, matrix), handle = createGpuInstanceAttributes(device,
    {isInstancedMesh:true, count:1, morphTexture:null, instanceMatrix:matrix, instanceColor:color});
  assert.equal(handle.bufferBytes, 12); assert.equal(color.uploads, 1);
  handle.dispose(); assert.equal(buffer.destroyed, 0); compute.dispose();
});

test('geometry completion includes compute validation and cancellation', async () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(), fence = deferred(); upload(compute, position);
  const handle = createGpuBufferGeometry(device, geometry({position}));
  device.queue.onSubmittedWorkDone = () => fence.promise;
  const idle = handle.whenIdle(); compute.dispose();
  await assert.rejects(idle, /DISPOSED/); assert.equal(handle.failed, true); handle.dispose();
});

test('compute indices borrow storage while ordinary non-float vertices still fail', () => {
  const device = gpuDevice(), compute = createGpuCompute(device), position = source(), index = attribute(new Uint32Array([0,1,2]),1);
  upload(compute, position); upload(compute, index);
  const before = device.events.length;
  const handle = createGpuBufferGeometry(device, geometry({position}, index));
  const snapshot = bufferGeometrySnapshot(handle, device);
  assert.equal(snapshot.indexBuffer, compute.getBuffer(index));
  assert.equal(snapshot.readIndex, null);
  assert.equal(handle.bufferBytes, 0);
  assert.equal(handle.shadowBytes, 0);
  assert.equal(device.events.length, before);
  handle.dispose();
  const integerPosition = attribute(new Uint32Array(12),4); upload(compute, integerPosition);
  assert.throws(() => createGpuBufferGeometry(device, geometry({position:integerPosition})), /FORMAT/); compute.dispose();
});

test('ordinary normalized, interleaved and released-array uploads still follow the original uploader', () => {
  const device = gpuDevice(), position = attribute(new Float32Array(9),3,false);
  const data = {...attribute(new Uint8Array([255,0,0, 0,255,0, 0,0,255]),3,false), stride:3};
  const color = {isInterleavedBufferAttribute:true,data,offset:0,itemSize:3,count:3,normalized:true};
  const handle = createGpuBufferGeometry(device, geometry({position,color}));
  assert.deepEqual(floats(bufferGeometrySnapshot(handle,device).vertexBuffers[1]), [1,0,0,0,1,0,0,0,1]);
  data.array[0] = 0; handle.update(); assert.equal(floats(bufferGeometrySnapshot(handle,device).vertexBuffers[1])[0],1);
  position.array = null; handle.update(); assert.equal(handle.vertexCount,3);
  handle.dispose(); assert.ok(device.buffers.every(b => b.destroyed === 1));
});
