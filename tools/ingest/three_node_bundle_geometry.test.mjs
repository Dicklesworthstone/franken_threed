import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodeBundleGeometry} from './three_node_bundle_geometry.mjs';
import {createGpuProgramGeometry, programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {createComputeAttributeOwner, captureComputeAttributeView, computeAttributeSnapshot,
  snapshotComputeAttributeView} from './gpu_compute_attributes.mjs';

const positionProgram = [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}];
function geometry() {
  return new T.BufferGeometry().setAttribute('position', new T.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
}
function histories(t, options) {
  const value = createThreeNodeBundleGeometry(T, options); t.after(() => value.dispose()); return value;
}

test('bundle geometry shares unchanged generations across captures and freezes source ranges', t => {
  const owner = histories(t), source = geometry(), a = owner.capture(source, 1);
  source.attributes.position.array[0] = 7;
  const b = owner.capture(source, 2);
  assert.equal(a.geometry, b.geometry); assert.notEqual(a, b);
  assert.equal(a.geometry.attributes.position.array[0], 0);
  assert.equal(owner.diagnostics.allocatedBytes, 36);
  source.setDrawRange(1, 2);
  const c = owner.capture(source, 3);
  assert.notEqual(c.geometry, a.geometry);
  assert.equal(c.geometry.attributes.position.array, a.geometry.attributes.position.array);
  assert.deepEqual(a.geometry.drawRange, {start: 0, count: Infinity});
  assert.deepEqual(c.geometry.drawRange, {start: 1, count: 2});
  assert.ok(Object.isFrozen(c.geometry)); assert.ok(Object.isFrozen(c.geometry.attributes));
  a.release(); b.release(); c.release();
});

test('source version increases, partial ranges and DynamicDrawUsage obey one upload per render', t => {
  const owner = histories(t), source = geometry(), attribute = source.attributes.position;
  attribute.addUpdateRange(0, 1);
  const first = owner.capture(source, 1);
  assert.deepEqual(attribute.updateRanges, [{start: 0, count: 1}], 'initial allocation does not consume ranges');
  attribute.array.fill(9); attribute.needsUpdate = true;
  const sameRender = owner.capture(source, 1);
  assert.equal(sameRender.geometry, first.geometry);
  const second = owner.capture(source, 2);
  assert.deepEqual([...second.geometry.attributes.position.array], [9, 0, 0, 1, 0, 0, 0, 1, 0]);
  assert.deepEqual(attribute.updateRanges, []);
  attribute.version = 0; attribute.array.fill(4);
  const decreased = owner.capture(source, 3);
  assert.equal(decreased.geometry, second.geometry, 'version decreases do not request a source upload');
  attribute.setUsage(T.DynamicDrawUsage);
  const dynamic = owner.capture(source, 4);
  assert.deepEqual([...dynamic.geometry.attributes.position.array], Array(9).fill(4));
  attribute.array.fill(5);
  const repeated = owner.capture(source, 4);
  assert.equal(repeated.geometry, dynamic.geometry);
  const next = owner.capture(source, 5);
  assert.deepEqual([...next.geometry.attributes.position.array], Array(9).fill(5));
  for (const packet of [first, sameRender, second, decreased, dynamic, repeated, next]) packet.release();
});

test('interleaved, integer, index and instance layouts reach the existing GPU geometry owner', t => {
  const owner = histories(t), source = new T.InstancedBufferGeometry();
  const data = new T.InterleavedBuffer(new Float32Array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]), 6);
  source.setAttribute('position', new T.InterleavedBufferAttribute(data, 3, 0));
  source.setAttribute('normal', new T.InterleavedBufferAttribute(data, 3, 3));
  source.setAttribute('ids', new T.InstancedBufferAttribute(new Uint16Array([4, 8, 12, 16]), 2, false, 3));
  source.setIndex(new T.BufferAttribute(new Uint32Array([0, 1]), 1));
  const captured = owner.capture(source, 1), view = captured.geometry;
  assert.equal(view.attributes.position.data, view.attributes.normal.data);
  assert.notEqual(view.attributes.position.data, data);
  assert.equal(view.attributes.normal.offset, 3); assert.equal(view.attributes.position.data.stride, 6);
  assert.equal(view.attributes.ids.meshPerAttribute, 3);
  assert.ok(view.attributes.ids.array instanceof Uint16Array); assert.ok(view.index.array instanceof Uint32Array);
  assert.equal(captured.bytes, 64); assert.equal(owner.diagnostics.allocatedBytes, 64);
  const device = geometryDevice(), gpu = createGpuProgramGeometry(device, view, [...positionProgram,
    {name: 'normal', location: 1, locations: 1, components: 3, scalar: 'float'},
    {name: 'ids', location: 2, locations: 1, components: 2, scalar: 'uint'}], {attributeTypes: 'webgpu'});
  t.after(() => gpu.dispose());
  const snapshot = programGeometrySnapshot(gpu, device);
  assert.equal(snapshot.layouts[0].arrayStride, 24); assert.equal(snapshot.layouts[1].stepMode, 'instance');
  assert.equal(snapshot.instanceCapacity, 2, 'native WebGPU ignores source instance divisors');
  assert.equal(snapshot.indexFormat, 'uint32'); assert.equal(snapshot.vertexCount, 2);
  assert.deepEqual([...new Uint32Array(snapshot.vertexBuffers[1].data)], [4, 8, 12, 16]);
  data.array.fill(99); data.needsUpdate = true; gpu.update();
  assert.equal(new Float32Array(snapshot.vertexBuffers[0].data)[0], 0, 'later source edits cannot change the accepted view');
  captured.release();
});

test('private generated streams and half-float flags join ordinary attributes without rewriting source arrays', t => {
  const owner = histories(t), source = geometry(), original = source.attributes.position.array;
  const generated = new T.InstancedBufferAttribute(new Float32Array([1, 2, 3, 4]), 4);
  const half = new T.Float16BufferAttribute(new Uint16Array([0x3c00, 0x4000, 0x4200]), 1);
  const program = {geometry: source, get attributes() { return {...source.attributes, generated, half}; },
    index: new T.BufferAttribute(new Uint16Array([0, 1, 2]), 1), drawRange: source.drawRange};
  const a = owner.capture(program, 1);
  generated.array = new Float32Array([5, 6, 7, 8]); generated.needsUpdate = true;
  source.attributes.position.array[0] = 9; source.attributes.position.needsUpdate = true;
  const b = owner.capture(program, 2);
  assert.deepEqual([...a.geometry.attributes.generated.array], [1, 2, 3, 4]);
  assert.deepEqual([...b.geometry.attributes.generated.array], [5, 6, 7, 8]);
  assert.equal(a.geometry.attributes.position.array[0], 0); assert.equal(b.geometry.attributes.position.array[0], 9);
  assert.equal(source.attributes.position.array, original);
  assert.equal(b.geometry.attributes.half.isFloat16BufferAttribute, true);
  assert.deepEqual([...b.geometry.attributes.half.array], [...half.array]);
  assert.ok(b.geometry.index.array instanceof Uint16Array);
  a.release(); b.release();
});

test('queued generations and independent retain leases share the bounded history budget', t => {
  const owner = histories(t, {maxBytes: 72}), source = geometry(), a = owner.capture(source, 1), held = a.retain();
  a.release(); a.release();
  const another = a.retain(); another.release();
  source.attributes.position.array[0] = 4; source.attributes.position.needsUpdate = true;
  const b = owner.capture(source, 2);
  assert.equal(owner.diagnostics.allocatedBytes, 72);
  source.attributes.position.array[0] = 8; source.attributes.position.addUpdateRange(0, 1); source.attributes.position.needsUpdate = true;
  assert.throws(() => owner.capture(source, 3), {code: 'THREE_NODE_BUNDLE_GEOMETRY_LIMIT'});
  assert.deepEqual(source.attributes.position.updateRanges, [{start: 0, count: 1}]);
  assert.equal(held.geometry.attributes.position.array[0], 0);
  held.release(); assert.equal(owner.diagnostics.allocatedBytes, 36);
  assert.throws(() => a.retain(), {code: 'THREE_NODE_BUNDLE_GEOMETRY_RELEASED'});
  const c = owner.capture(source, 3);
  assert.equal(c.geometry.attributes.position.array[0], 8); assert.equal(b.geometry.attributes.position.array[0], 4);
  b.release(); c.release(); owner.dispose(); assert.equal(owner.diagnostics.allocatedBytes, 0);
});

test('source disposal forwards residency events while accepted data survives and later captures start fresh', t => {
  const owner = histories(t), source = geometry(), first = owner.capture(source, 1), device = geometryDevice();
  const gpu = createGpuProgramGeometry(device, first.geometry, positionProgram, {attributeTypes: 'webgpu'});
  t.after(() => gpu.dispose());
  source.attributes.position.array[0] = 7;
  source.dispose();
  assert.throws(() => programGeometrySnapshot(gpu, device), {code: 'GEOMETRY_GPU_RELEASED'});
  assert.equal(first.geometry.attributes.position.array[0], 0);
  gpu.update(); assert.equal(new Float32Array(programGeometrySnapshot(gpu, device).vertexBuffers[0].data)[0], 0);
  const next = owner.capture(source, 2);
  assert.equal(next.geometry.attributes.position.array[0], 7);
  owner.dispose(); assert.equal(owner.diagnostics.allocatedBytes, 72, 'accepted leases survive owner disposal');
  first.release(); next.release(); assert.equal(owner.diagnostics.allocatedBytes, 0);
  assert.throws(() => owner.capture(source, 3), {code: 'THREE_NODE_BUNDLE_GEOMETRY_DISPOSED'});
});

test('invalid ranges, mutable storage layouts and unproven GPU storage inputs fail before publishing new histories', t => {
  const owner = histories(t), source = geometry(), first = owner.capture(source, 1), attribute = source.attributes.position;
  attribute.addUpdateRange(8, 2); attribute.needsUpdate = true;
  assert.throws(() => owner.capture(source, 2), {code: 'THREE_NODE_BUNDLE_GEOMETRY_ATTRIBUTE'});
  assert.equal(owner.diagnostics.allocatedBytes, 36);
  attribute.clearUpdateRanges(); attribute.array = new Float32Array(12); attribute.count = 4;
  assert.throws(() => owner.capture(source, 2), {code: 'THREE_NODE_BUNDLE_GEOMETRY_CHANGED'});
  const storage = geometry(); storage.attributes.position.isStorageBufferAttribute = true;
  assert.throws(() => owner.capture(storage, 2), {code: 'THREE_NODE_BUNDLE_GEOMETRY_ATTRIBUTE'});
  assert.equal(first.geometry.attributes.position.count, 3); first.release();
});

test('immutable geometry retains captured GPU versions without copying simulation arrays or changing CPU histories', async t => {
  const owner = histories(t), source = geometry(), device = geometryDevice();
  const storage = new T.StorageInstancedBufferAttribute(new Float32Array([1,2,3,0,4,5,6,0]), 4);
  const publication = createComputeAttributeOwner(device); t.after(() => publication.dispose());
  const buffer = device.createBuffer({size: storage.array.byteLength, usage: 128 | 32});
  let waits = 0;
  const publish = () => publication.publish(storage, buffer, storage.array, storage.version,
    () => {}, () => { waits++; return Promise.resolve(); });
  publish();
  const mutableView = Object.create(storage);
  captureComputeAttributeView(device, storage, mutableView).install();
  const program = {geometry: source, attributes: {...source.attributes, offsets: mutableView}, drawRange: source.drawRange};
  const first = owner.capture(program, 1), firstView = first.geometry.attributes.offsets;
  assert.notEqual(firstView, mutableView); assert.ok(Object.isFrozen(firstView));
  assert.equal(firstView.array, storage.array, 'GPU metadata retains the source identity without a CPU clone');
  assert.equal(first.bytes, 36); assert.equal(owner.diagnostics.allocatedBytes, 36);
  const firstGpu = computeAttributeSnapshot(device, firstView);
  assert.equal(firstGpu.buffer, buffer); assert.equal(firstGpu.arrayStride, 16);

  storage.array.fill(7); storage.needsUpdate = true;
  captureComputeAttributeView(device, storage, mutableView).install();
  const second = owner.capture(program, 2), secondView = second.geometry.attributes.offsets;
  assert.notEqual(second.geometry, first.geometry); assert.notEqual(secondView, firstView);
  assert.equal(firstView.version, 0); assert.equal(secondView.version, 1);
  assert.equal(first.geometry.attributes.position.array, second.geometry.attributes.position.array);
  assert.equal(owner.diagnostics.allocatedBytes, 36, 'storage bytes do not consume the CPU snapshot budget');
  assert.throws(() => computeAttributeSnapshot(device, secondView), {code: 'COMPUTE_ATTRIBUTE_UNPREPARED'});
  assert.equal(computeAttributeSnapshot(device, firstView).buffer, buffer);
  publish();
  assert.equal(computeAttributeSnapshot(device, secondView).buffer, buffer);
  assert.throws(() => firstGpu.assertCurrent(), {code: 'COMPUTE_ATTRIBUTE_UNPREPARED'});
  await firstGpu.whenIdle(); assert.equal(waits, 1, 'retirement waits the old allocation without reading an obsolete content version');
  captureComputeAttributeView(device, storage, mutableView).install();
  const warm = owner.capture(program, 3);
  assert.equal(warm.geometry, second.geometry, 'unchanged storage proofs preserve native geometry cache identity');
  assert.equal(device.writes.length, 0, 'metadata capture never uploads CPU simulation arrays');
  for (const frame of [first, second, warm]) frame.release();
});

test('GPU view proofs retain accepted shape and cannot be forged or used as indices', t => {
  const owner = histories(t), source = geometry(), device = geometryDevice();
  const storage = new T.StorageBufferAttribute(new Uint32Array([1,2,3,4,5,6,7,8]), 4);
  const publication = createComputeAttributeOwner(device); t.after(() => publication.dispose());
  const buffer = device.createBuffer({size: storage.array.byteLength, usage: 128 | 32});
  publication.publish(storage, buffer, storage.array, 0, () => {}, () => Promise.resolve());
  const view = Object.create(storage), forged = Object.create(storage);
  captureComputeAttributeView(device, storage, view).install();
  Object.assign(forged, view);
  assert.equal(snapshotComputeAttributeView(forged), null);
  source.setAttribute('indicesAsVertex', forged);
  assert.throws(() => owner.capture(source, 1), {code: 'THREE_NODE_BUNDLE_GEOMETRY_ATTRIBUTE'});
  assert.equal(owner.diagnostics.allocatedBytes, 0);
  source.setAttribute('indicesAsVertex', view);
  view.count = 99; view.itemSize = 1; view.array = new Uint32Array(99);
  const accepted = owner.capture(source, 1), captured = accepted.geometry.attributes.indicesAsVertex;
  assert.equal(captured.count, 2); assert.equal(captured.itemSize, 4); assert.equal(captured.array, storage.array);
  assert.equal(computeAttributeSnapshot(device, captured, {integer: true}).buffer, buffer);
  assert.throws(() => computeAttributeSnapshot(geometryDevice(), captured, {integer: true}), {code: 'COMPUTE_ATTRIBUTE_OWNER'});
  source.setIndex(view);
  assert.throws(() => owner.capture(source, 2), {code: 'THREE_NODE_BUNDLE_GEOMETRY_ATTRIBUTE'});
  assert.equal(owner.diagnostics.allocatedBytes, 36); accepted.release();
});
