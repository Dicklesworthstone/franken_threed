import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceWebGPUAttributeUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUAttributeUtils.js';
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

test('selected native inputs leave unused attributes and their pending ranges fresh until first use', t => {
  const owner = histories(t), source = geometry();
  const color = new T.BufferAttribute(new Uint8Array([10,20,30,40,50,60,70,80,90]), 3, true);
  source.setAttribute('color', color); color.addUpdateRange(0, 3); color.needsUpdate = true;
  const first = owner.capture(source, 1, ['position']);
  assert.deepEqual(Object.keys(first.geometry.attributes), ['position']);
  assert.equal(owner.diagnostics.owners, 1); assert.equal(first.bytes, 36);
  color.array.fill(90); source.attributes.position.array[0] = 7;
  const second = owner.capture(source, 2, ['position', 'color']);
  assert.ok(second.geometry.attributes.color.array instanceof Uint8Array);
  assert.equal(second.geometry.attributes.color.normalized, true);
  assert.deepEqual([...second.geometry.attributes.color.array], Array(9).fill(90));
  assert.equal(second.geometry.attributes.position.array[0], 0);
  assert.deepEqual(color.updateRanges, [{start: 0, count: 3}], 'first upload keeps source ranges');
  color.array.fill(120); color.needsUpdate = true;
  const inactive = owner.capture(source, 3, ['position']);
  assert.deepEqual(color.updateRanges, [{start: 0, count: 3}], 'inactive inputs consume neither versions nor ranges');
  const active = owner.capture(source, 4, ['position', 'color']);
  assert.deepEqual([...active.geometry.attributes.color.array], [120,120,120,90,90,90,90,90,90]);
  assert.deepEqual(color.updateRanges, []);
  for (const packet of [first, second, inactive, active]) packet.release();
});

test('a selected interleaved input uploads the whole owner and preserves stale bytes for later views', t => {
  const owner = histories(t), source = new T.BufferGeometry();
  const data = new T.InterleavedBuffer(new Float32Array([0,0,0,.25,.5,1,1,0,0,.5,.75,1]), 6);
  source.setAttribute('position', new T.InterleavedBufferAttribute(data, 3, 0));
  source.setAttribute('normal', new T.InterleavedBufferAttribute(data, 3, 3));
  const first = owner.capture(source, 1, ['position']);
  data.array[3] = 99;
  const second = owner.capture(source, 2, ['normal']);
  assert.equal(second.geometry.attributes.position, undefined);
  assert.equal(second.geometry.attributes.normal.data.array, first.geometry.attributes.position.data.array);
  assert.equal(second.geometry.attributes.normal.getX(0), .25);
  assert.equal(second.geometry.vertexCount, 2); assert.equal(second.bytes, data.array.byteLength);
  assert.equal(owner.diagnostics.owners, 1);
  first.release(); second.release();
});

test('vertex-index programs retain source nonindexed draw counts without capturing position bytes', t => {
  const owner = histories(t), source = geometry(), first = owner.capture(source, 1, []), device = geometryDevice();
  assert.deepEqual(Object.keys(first.geometry.attributes), []); assert.equal(first.bytes, 0);
  assert.equal(owner.diagnostics.owners, 0); assert.equal(first.geometry.vertexCount, 3);
  const gpu = createGpuProgramGeometry(device, first.geometry, [], {attributeTypes: 'webgpu'});
  t.after(() => gpu.dispose()); assert.equal(programGeometrySnapshot(gpu, device).vertexCount, 3);
  source.attributes.position.array.fill(7);
  const selected = owner.capture(source, 2, ['position']);
  assert.deepEqual([...selected.geometry.attributes.position.array], Array(9).fill(7));
  first.release(); selected.release();
});

test('alternating shader input selections reuse immutable views and share uploaded generations', t => {
  const owner = histories(t, {maxBytes: 108}), source = geometry(), packets = [];
  source.setAttribute('normal', new T.Float32BufferAttribute(Array(9).fill(.5), 3));
  const capture = (renderId, inputs) => { const packet = owner.capture(source, renderId, inputs); packets.push(packet); return packet; };
  const empty = capture(1, []), positions = capture(2, ['position']), both = capture(3, ['position', 'normal']);
  assert.equal(capture(4, []).geometry, empty.geometry);
  assert.equal(capture(5, ['position']).geometry, positions.geometry);
  assert.equal(capture(6, ['position', 'normal']).geometry, both.geometry);
  assert.equal(both.geometry.attributes.position.array, positions.geometry.attributes.position.array);
  assert.equal(owner.diagnostics.allocatedBytes, 72, 'input variants charge their shared uploads once');
  source.attributes.position.array[0] = 7; source.attributes.position.needsUpdate = true;
  const next = capture(7, ['position']);
  assert.notEqual(next.geometry, positions.geometry);
  assert.equal(next.geometry.attributes.position.array[0], 7);
  assert.equal(positions.geometry.attributes.position.array[0], 0);
  assert.equal(capture(8, []).geometry, empty.geometry, 'unused uploads do not invalidate the vertex-index view');
  const nextBoth = capture(9, ['position', 'normal']);
  assert.equal(nextBoth.geometry.attributes.position.array, next.geometry.attributes.position.array);
  assert.equal(both.geometry.attributes.position.array[0], 0);
  assert.equal(capture(10, ['position']).geometry, next.geometry);
  for (const packet of packets) packet.release();
  assert.equal(owner.diagnostics.allocatedBytes, 72, 'retired upload generations leave the byte budget');
  source.dispose(); assert.equal(owner.diagnostics.allocatedBytes, 0);
});

test('layout variant caching stays bounded while accepted views survive eviction and disposal', t => {
  const owner = histories(t), source = geometry(), first = owner.capture(source, 1);
  const firstGeometry = first.geometry; first.release();
  let recent;
  for (let i = 1; i <= 32; i++) {
    source.setDrawRange(i, 1);
    const packet = owner.capture(source, i + 1); recent = packet.geometry; packet.release();
  }
  assert.throws(() => first.retain(), {code: 'THREE_NODE_BUNDLE_GEOMETRY_RELEASED'});
  const kept = owner.capture(source, 34);
  assert.equal(kept.geometry, recent); assert.equal(owner.diagnostics.allocatedBytes, 36);
  source.setDrawRange(0, Infinity);
  const recreated = owner.capture(source, 35); assert.notEqual(recreated.geometry, firstGeometry);
  owner.dispose(); assert.equal(owner.diagnostics.allocatedBytes, 36);
  assert.deepEqual(kept.geometry.drawRange, {start: 32, count: 1});
  kept.release(); recreated.release(); assert.equal(owner.diagnostics.allocatedBytes, 0);
});

test('cached layout variants do not pin superseded uploads against the history byte budget', t => {
  const owner = histories(t, {maxBytes: 72}), source = geometry(), attribute = source.attributes.position;
  owner.capture(source, 0).release();
  for (let i = 1; i <= 12; i++) {
    source.setDrawRange(i, 1); attribute.array[0] = i; attribute.needsUpdate = true;
    const current = owner.capture(source, i);
    assert.equal(current.geometry.attributes.position.array[0], i); current.release();
    owner.capture(source, i, []).release();
    assert.equal(owner.diagnostics.allocatedBytes, 36, 'only the current upload remains without accepted older leases');
  }
  owner.dispose(); assert.equal(owner.diagnostics.allocatedBytes, 0);
});

test('disposing a geometry invalidates shared attributes that were unused by its own shader', t => {
  const owner = histories(t), a = geometry(), b = geometry();
  const color = new T.Float32BufferAttribute(Array(9).fill(.25), 3);
  a.setAttribute('color', color); b.setAttribute('color', color);
  const first = owner.capture(a, 1, ['position']), shared = owner.capture(b, 1, ['position', 'color']);
  color.array.fill(.75); a.dispose();
  const next = owner.capture(b, 2, ['position', 'color']);
  assert.deepEqual([...shared.geometry.attributes.color.array], Array(9).fill(.25));
  assert.deepEqual([...next.geometry.attributes.color.array], Array(9).fill(.75));
  first.release(); shared.release(); next.release();
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

test('shared refresh follows uploaded generations while freezing layouts and allocation boundaries', t => {
  const owner = histories(t), source = geometry(), attribute = source.attributes.position;
  const alias = new T.BufferGeometry().setAttribute('position', attribute);
  source.setDrawRange(1, 2); const first = owner.capture(source, 1);
  const empty = owner.capture(source, 1, []);
  attribute.array.fill(7); attribute.addUpdateRange(0, 1); attribute.needsUpdate = true;
  const uploaded = owner.capture(alias, 2);
  source.setDrawRange(0, 1); attribute.itemSize = 1; attribute.count = 9;
  const refreshed = first.refreshUploaded(true), reused = first.refreshUploaded(true);
  assert.equal(refreshed.geometry, reused.geometry, 'unchanged refreshes reuse private geometry identities');
  assert.equal(refreshed.geometry.attributes.position.itemSize, 3);
  assert.equal(refreshed.geometry.attributes.position.count, 3);
  assert.deepEqual(refreshed.geometry.drawRange, {start: 1, count: 2});
  assert.deepEqual([...refreshed.geometry.attributes.position.array], [7, 0, 0, 1, 0, 0, 0, 1, 0]);
  assert.deepEqual(first.geometry.attributes.position.array, new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
  attribute.itemSize = 3; attribute.count = 3; attribute.array.fill(9); alias.dispose();
  const replacement = owner.capture(alias, 3);
  assert.equal(refreshed.refreshUploaded(true), null, 'new allocations do not rewrite already recorded bindings');
  const rebound = refreshed.refreshUploaded();
  assert.deepEqual([...rebound.geometry.attributes.position.array], Array(9).fill(9));
  assert.deepEqual(rebound.geometry.drawRange, {start: 1, count: 2});
  owner.dispose();
  for (const packet of [first, empty, uploaded, refreshed, reused, replacement, rebound]) packet.release();
  assert.equal(owner.diagnostics.allocatedBytes, 0);
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

for (const Type of [Int8Array, Uint8Array, Int16Array, Uint16Array]) for (const components of [2, 3, 4])
for (const interleaved of [false, true])
test(`native ${Type.name} vec${components} ${interleaved ? 'interleaved' : 'separate'} ranges match pinned upload bytes`, t => {
  const owner = histories(t), source = new T.BufferGeometry(), device = geometryDevice(), records = new Map();
  const backend = {device, get(key) { if (!records.has(key)) records.set(key, {}); return records.get(key); }};
  const uploader = new SourceWebGPUAttributeUtils(backend), stride = interleaved ? 8 : components, offset = interleaved ? 4 : 0;
  const make = () => {
    const array = new Type(new ArrayBuffer((4 * stride + 8) * Type.BYTES_PER_ELEMENT), 4 * Type.BYTES_PER_ELEMENT, 4 * stride);
    for (let i = 0; i < array.length; i++) array[i] = i + 10;
    return interleaved ? new T.InterleavedBufferAttribute(new T.InterleavedBuffer(array, stride), components, offset, true)
      : new T.BufferAttribute(array, components, true);
  };
  const reference = make(), attribute = make(), referenceOwner = interleaved ? reference.data : reference;
  const inputOwner = interleaved ? attribute.data : attribute, original = reference.array, initial = [...original], packets = [];
  source.setAttribute('value', attribute);
  const padded = !interleaved && components * Type.BYTES_PER_ELEMENT % 4 !== 0;
  const firstRange = padded ? {start: 1, count: 0} : {start: offset, count: 4 / Type.BYTES_PER_ELEMENT};
  for (const target of [referenceOwner, inputOwner]) target.addUpdateRange(firstRange.start, firstRange.count);
  uploader.createAttribute(reference, 32);
  const layout = uploader.createShaderVertexBuffers({getAttributes: () => [reference]})[0];
  assert.equal(layout.arrayStride, interleaved ? stride * Type.BYTES_PER_ELEMENT : Math.ceil(components * Type.BYTES_PER_ELEMENT / 4) * 4);
  assert.equal(layout.attributes[0].offset, offset * Type.BYTES_PER_ELEMENT);
  const compare = renderId => {
    const packet = owner.capture(source, renderId, ['value']); packets.push(packet);
    const gpu = new Type(backend.get(referenceOwner).buffer.data), raw = [];
    if (interleaved) raw.push(...gpu.slice(0, attribute.array.length));
    else for (let vertex = 0; vertex < attribute.count; vertex++)
      raw.push(...gpu.slice(vertex * layout.arrayStride / Type.BYTES_PER_ELEMENT,
        vertex * layout.arrayStride / Type.BYTES_PER_ELEMENT + components));
    assert.deepEqual([...packet.geometry.attributes.value.array], raw);
    return packet;
  };
  const first = compare(1);
  assert.deepEqual(inputOwner.updateRanges, [firstRange], 'first upload keeps ranges, including empty interior ranges');
  for (const target of [referenceOwner, inputOwner]) { target.array.fill(70); target.needsUpdate = true; }
  uploader.updateAttribute(reference); compare(2);
  assert.deepEqual(inputOwner.updateRanges, []); assert.deepEqual(referenceOwner.updateRanges, []);
  assert.deepEqual([...first.geometry.attributes.value.array], initial, 'accepted older bytes remain immutable');
  const range = padded ? {start: components - 1, count: 2} : {start: offset, count: 4 / Type.BYTES_PER_ELEMENT};
  for (const target of [referenceOwner, inputOwner]) {
    target.array.fill(90); target.addUpdateRange(range.start, range.count); target.needsUpdate = true;
  }
  const held = compare(2);
  assert.deepEqual(inputOwner.updateRanges, [range], 'one upload per logical render retains later pending ranges');
  uploader.updateAttribute(reference); const updated = compare(3);
  assert.notEqual(updated.geometry, held.geometry);
  assert.deepEqual(inputOwner.updateRanges, []);
  if (padded) {
    for (const target of [referenceOwner, inputOwner]) {
      target.array.fill(110); target.addUpdateRange(components, 0); target.needsUpdate = true;
    }
    uploader.updateAttribute(reference);
    assert.deepEqual(compare(4).geometry.attributes.value.array, updated.geometry.attributes.value.array,
      'an empty range at a vertex boundary uploads no components');
  }
  assert.equal(reference.array, original); assert.equal(reference.itemSize, components, 'native padding stays private');
  for (const packet of packets) packet.release();
});

test('native unpadded normalized ranges preserve first-upload behavior and refuse source-invalid alignment', t => {
  const owner = histories(t), source = new T.BufferGeometry();
  const attribute = new T.BufferAttribute(new Uint8Array([10, 20, 30, 40]), 4, true);
  source.setAttribute('color', attribute); attribute.addUpdateRange(1, 1);
  const record = {}, uploader = new SourceWebGPUAttributeUtils({device: geometryDevice(), get: () => record});
  uploader.createAttribute(attribute, 32);
  const first = owner.capture(source, 1); assert.deepEqual(attribute.updateRanges, [{start: 1, count: 1}]);
  attribute.array.fill(90); attribute.needsUpdate = true;
  assert.throws(() => uploader.updateAttribute(attribute), 'source queue rejects the same unaligned write');
  assert.throws(() => owner.capture(source, 2), {code: 'THREE_NODE_BUNDLE_GEOMETRY_ATTRIBUTE'});
  assert.deepEqual(attribute.updateRanges, [{start: 1, count: 1}]);
  assert.deepEqual([...first.geometry.attributes.color.array], [10, 20, 30, 40]);
  attribute.clearUpdateRanges(); attribute.addUpdateRange(0, 4);
  const next = owner.capture(source, 2); assert.deepEqual([...next.geometry.attributes.color.array], [90, 90, 90, 90]);
  assert.deepEqual(attribute.updateRanges, []); first.release(); next.release();
});

for (const [Type, before, after] of [[Uint8Array, 3, 4], [Uint8Array, 4, 3], [Uint8Array, 2, 3], [Int16Array, 3, 2]])
test(`native ${Type.name} resident ${before}-to-${after} component padding changes require a new owner`, t => {
  const owner = histories(t), source = new T.BufferGeometry(), attribute = new T.BufferAttribute(new Type(12).fill(10), before, true);
  source.setAttribute('value', attribute); const first = owner.capture(source, 1);
  attribute.itemSize = after; attribute.count = 12 / after; attribute.array.fill(90);
  attribute.addUpdateRange(0, 4); attribute.needsUpdate = true;
  assert.throws(() => owner.capture(source, 2), {code: 'THREE_NODE_BUNDLE_GEOMETRY_CHANGED'});
  assert.deepEqual(attribute.updateRanges, [{start: 0, count: 4}]);
  assert.deepEqual([...first.geometry.attributes.value.array], Array(12).fill(10));
  source.setAttribute('value', new T.BufferAttribute(attribute.array, after, true));
  const replaced = owner.capture(source, 2);
  assert.deepEqual([...replaced.geometry.attributes.value.array], Array(12).fill(90));
  first.release(); replaced.release();
});

for (const Type of [Float32Array, Uint16Array])
test(`native ${Type.name} unpadded reinterpretation retains its legal raw layout`, t => {
  const owner = histories(t), source = new T.BufferGeometry(), normalized = Type === Uint16Array;
  const attribute = new T.BufferAttribute(new Type(12).fill(10), normalized ? 2 : 3, normalized);
  source.setAttribute('value', attribute); const first = owner.capture(source, 1);
  attribute.itemSize = 4; attribute.count = 3; attribute.array.fill(90); attribute.needsUpdate = true;
  const next = owner.capture(source, 2);
  assert.deepEqual([...next.geometry.attributes.value.array], Array(12).fill(90));
  assert.deepEqual([...first.geometry.attributes.value.array], Array(12).fill(10));
  first.release(); next.release();
});
